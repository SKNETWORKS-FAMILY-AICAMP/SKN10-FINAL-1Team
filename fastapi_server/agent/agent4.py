"""LangGraph graph for prediction_agent.

Handles CSV file analysis based on user queries.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, List, Annotated
import io
from dotenv import load_dotenv
import operator
import pandas as pd
import joblib
import base64

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import logging
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from sklearn.base import BaseEstimator, TransformerMixin # Ensure these are available for Preprocessor

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- State Definition ---
class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)
    user_query: Optional[str] = None
    csv_file_content: Optional[str] = None
    final_answer: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __post_init__(self):
        logger.info(f"AgentState __post_init__ called. Initial self.user_query: '{self.user_query}', self.messages: {self.messages}")
        if not self.user_query and self.messages:
            user_messages = [msg for msg in self.messages if isinstance(msg, HumanMessage)]
            if user_messages:
                self.user_query = user_messages[-1].content
                logger.info(f"AgentState __post_init__ set self.user_query to: '{self.user_query}'")
            else:
                logger.info("AgentState __post_init__: No HumanMessage found in self.messages.")
        else:
            logger.info("AgentState __post_init__: self.user_query already set or no messages.")

    def dict(self):
        result = super().dict()
        if self.final_answer and self.messages is not None:
            result["messages"] = self.messages + [AIMessage(content=self.final_answer)]
        return result


# --- Custom Class Definitions for Model Loading ---
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        self.binary_map = {'Yes': 1, 'No': 0}
        self.categorical_cols = []

    def fit(self, X, y=None):
        X = X.copy()
        X['totalcharges'] = pd.to_numeric(X['totalcharges'], errors='coerce')

        binary_cols = ['partner', 'dependents', 'phoneservice', 'paperlessbilling']
        for col in binary_cols:
            X[col] = X[col].map(self.binary_map)

        self.categorical_cols = [
            col for col in X.select_dtypes(include=['object', 'category']).columns
            if not pd.to_numeric(X[col], errors='coerce').notna().all()
        ]

        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        X['totalcharges'] = pd.to_numeric(X['totalcharges'], errors='coerce')
        X['totalcharges'] = X['totalcharges'].fillna(X['tenure'] * X['monthlycharges'])

        X['new_totalservices'] = (X[['phoneservice', 'internetservice', 'onlinesecurity',
                                     'onlinebackup', 'deviceprotection', 'techsupport',
                                     'streamingtv', 'streamingmovies']] == 'Yes').sum(axis=1)

        X['new_avg_charges'] = X['totalcharges'] / (X['tenure'] + 1e-5)
        X['new_increase'] = X['new_avg_charges'] / X['monthlycharges']
        X['new_avg_service_fee'] = X['monthlycharges'] / (X['new_totalservices'] + 1e-5)
        X['charge_increased'] = (X['monthlycharges'] > X['new_avg_charges']).astype(int)
        X['charge_growth_rate'] = (X['monthlycharges'] - X['new_avg_charges']) / (X['new_avg_charges'] + 1e-5)
        X['is_auto_payment'] = X['paymentmethod'].apply(lambda x: int('automatic' in str(x).lower() or 'bank' in str(x).lower()))

        binary_cols = ['partner', 'dependents', 'phoneservice', 'paperlessbilling']
        for col in binary_cols:
            X[col] = X[col].map(self.binary_map)

        for col in self.categorical_cols:
            X[col] = self.label_encoders[col].transform(X[col].astype(str))

        # Ensure the output is a NumPy array as expected by some sklearn models/pipelines
        # If X was already a DataFrame and processed, .values might not be needed
        # or could be X[self.categorical_cols + numeric_cols_if_any].values
        # For now, assuming X is processed to a state where .values is appropriate
        # This might need adjustment based on the exact structure of X after transforms
        if isinstance(X, pd.DataFrame):
            # Select only columns that were part of the fit or are known to be numeric
            # This is a simplification; a more robust approach would track all columns used in fit
            # For now, we assume the DataFrame X at this stage contains the correct columns for .values
            return X.values 
        return X # If X is not a DataFrame, return as is (e.g., already a NumPy array)


# --- ThresholdWrapper Class Definition for Model Loading ---
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import LabelEncoder # Added for Preprocessor

class ThresholdWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper for a classifier to adjust the prediction threshold.
    This class is required to load the pickled model.
    """
    def __init__(self, pipeline, threshold=0.5):
        self.pipeline = pipeline
        self.threshold = threshold

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        # Ensure classes_ attribute is set if the underlying pipeline/model has it
        if hasattr(self.pipeline, 'classes_'):
            self.classes_ = self.pipeline.classes_
        elif hasattr(self.pipeline.steps[-1][1], 'classes_'): # Common for scikit-learn Pipeline
            self.classes_ = self.pipeline.steps[-1][1].classes_
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return (proba[:, 1] >= self.threshold).astype(int)
        else:
            return proba.argmax(axis=1)


# --- Node Function for CSV Analysis ---

def predict_churn_with_pipeline(df: pd.DataFrame) -> str:
    # Preprocess the dataframe to match the training conditions
    df.columns = [col.lower() for col in df.columns]
    
    # Drop columns that were not used in training
    if 'unnamed: 0' in df.columns:
        df = df.drop(columns=['unnamed: 0'])
    if 'customerid' in df.columns:
        df = df.drop(columns=['customerid'])

    """
    Loads the churn prediction pipeline and makes predictions on the given DataFrame.
    The pipeline is expected to handle all necessary feature engineering.
    """
    try:
        # Construct the full path to the model file relative to this script's location
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'churn_pipeline_full.pkl')
        logger.info(f"Attempting to load model from: {model_path}")

        if not os.path.exists(model_path):
            error_msg = f"Model file not found at the specified path: {model_path}"
            logger.error(error_msg)
            return error_msg

        # HACK: Inject custom classes into the __main__ module
        # This is necessary because the model was pickled with these classes defined in the __main__ scope.
        import sys
        sys.modules['__main__'].ThresholdWrapper = ThresholdWrapper
        sys.modules['__main__'].Preprocessor = Preprocessor

        # Load the pipeline using joblib
        pipeline = joblib.load(model_path)
        logger.info("Churn prediction pipeline loaded successfully.")

        # Make predictions. The pipeline handles all preprocessing.
        predictions = pipeline.predict(df)
        
        # Get prediction probabilities (usually [P(Not Churn), P(Churn)])
        probabilities = pipeline.predict_proba(df)
        churn_probabilities = probabilities[:, 1] # Assuming the second column is P(Churn)

        # Generate a summary of predictions
        total_customers = len(predictions)
        churn_count = predictions.sum()
        not_churn_count = total_customers - churn_count
        churn_percentage = (churn_count / total_customers) * 100 if total_customers > 0 else 0

        result_summary = f"""
Churn Prediction Results:

Total Customers Analyzed: {total_customers}
Predicted to Churn: {churn_count} ({churn_percentage:.2f}%)
Predicted to Not Churn: {not_churn_count}

Individual Customer Churn Probabilities:
"""
        # Try to use an existing ID column if present, otherwise use index
        id_column_names = ['id', 'customerid', 'customer_id', 'user_id', 'userid']
        original_id_col = None
        for col_name in id_column_names:
            if col_name in df.columns:
                original_id_col = col_name
                break

        for i in range(total_customers):
            customer_identifier = df[original_id_col].iloc[i] if original_id_col else f"Customer {i+1}"
            result_summary += f"- {customer_identifier}: {churn_probabilities[i]:.4f} (Churn: {'Yes' if predictions[i] == 1 else 'No'})\n"

        logger.info(f"Prediction result: {result_summary}")
        return result_summary

    except Exception as e:
        error_msg = f"❌ An error occurred while using the .pkl model for prediction: {e}"
        logger.error(error_msg, exc_info=True)
        return error_msg


def csv_analysis_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    logger.info(f"--- CSV ANALYSIS NODE ---")
    logger.info(f"csv_analysis_node received state.user_query: '{state.user_query}'")
    logger.info(f"csv_analysis_node received state.csv_file_content (first 50 chars): '{state.csv_file_content[:50] if state.csv_file_content else None}'")
    
    user_query = state.user_query
    csv_content = state.csv_file_content

    if not csv_content:
        answer = "분석할 CSV 파일을 먼저 첨부해주세요."
        return {"final_answer": answer}

    if not user_query:
        answer = "CSV 파일에 대해 무엇이 궁금하신가요? 질문을 입력해주세요."
        return {"final_answer": answer}

    try:
        # The base64 decoding logic seems to be a workaround for a frontend issue.
        # It will be kept as is.
        lines = csv_content.strip().splitlines()
        if not lines:
            raise ValueError("CSV content is empty or could not be split into lines.")

        reconstructed_csv_content = csv_content
        try:
            potential_b64_header = lines[0]
            decoded_bytes = base64.b64decode(potential_b64_header)
            decoded_header_str = decoded_bytes.decode('utf-8').strip()
            if ',' in decoded_header_str:
                logger.info("Successfully decoded base64 header.")
                reconstructed_csv_content = decoded_header_str + ("\n" + "\n".join(lines[1:]) if len(lines) > 1 else "")
            else:
                logger.info("Decoded string does not look like a CSV header. Using original content.")
        except Exception:
            logger.info("First line is not base64 encoded. Processing as plain CSV.")

        csv_file_obj = io.StringIO(reconstructed_csv_content)
        df = pd.read_csv(csv_file_obj)

        logger.info(f"DataFrame created. Columns: {df.columns.tolist()}, Shape: {df.shape}")

        # --- LOGIC TO ROUTE BETWEEN PREDICTION AND GENERAL Q&A ---
        prediction_keywords = ["예측", "이탈", "churn", "predict"]
        if any(keyword in user_query.lower() for keyword in prediction_keywords):
            logger.info("User query contains prediction keywords. Routing to the .pkl pipeline.")
            final_answer = predict_churn_with_pipeline(df)
        else:
            logger.info("User query is for general Q&A. Routing to the pandas agent.")
            llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
            pandas_agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_executor_kwargs={"handle_parsing_errors": True},
                allow_dangerous_code=True
            )
            response = pandas_agent.invoke({"input": user_query})
            final_answer = response.get("output", "죄송합니다, 답변을 생성하지 못했습니다.")

        return {"final_answer": final_answer}

    except ValueError as ve:
        error_msg = f"❌ CSV 데이터 처리 오류: {ve}"
        logger.error(error_msg, exc_info=True)
        return {"final_answer": error_msg, "error_message": error_msg}
    except Exception as e:
        error_msg = f"❌ CSV 분석 중 예기치 않은 오류가 발생했습니다: {e}"
        logger.error(f"Unexpected error type: {type(e).__name__}", exc_info=True)
        return {"final_answer": error_msg, "error_message": error_msg}


# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("csv_analysis_node", csv_analysis_node)
workflow.set_entry_point("csv_analysis_node")
workflow.add_edge("csv_analysis_node", END)
app = workflow.compile()

# --- Main function to expose the graph ---
async def main():
    return app

if __name__ == "__main__":
    async def test_agent():
        graph = await main()
        
        # Test case 1: Analyze CSV file
        inputs = {"messages": [HumanMessage(content="CSV 파일에 대해 무엇이 궁금하신가요?")], "csv_file_content": "name,age\nJohn,25\nAlice,30"}
        async for event in graph.astream(inputs):
            if "csv_analysis_node" in event:
                print(event["csv_analysis_node"]["final_answer"])
        
    asyncio.run(test_agent())
