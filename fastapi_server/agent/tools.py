# f:\dev\SKN10-FINAL-1Team\swarm_agent\src\agent\tools.py
import json
import uuid
from typing import Dict, Any, List, Optional
import os
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import joblib
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import Tool
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from openai import OpenAI as OpenAIClient # For embeddings
from pinecone import Pinecone as PineconeClient # For vector search

# --- Analyst Chart Tool ---
def generate_chart_html(chart_type: str, data: dict, options: dict = None, chart_id: str = None) -> str:
    if chart_id is None:
        chart_id = f"chart-{uuid.uuid4().hex[:8]}"
    if options is None:
        options = {}
    data_json = json.dumps(data)
    options_json = json.dumps(options)
    html_template = f"""
    <div>
      <canvas id='{chart_id}'></canvas>
    </div>
    <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
    <script>
      const ctx_{chart_id} = document.getElementById('{chart_id}');
      if (ctx_{chart_id}) {{
        new Chart(ctx_{chart_id}, {{
          type: '{chart_type}',
          data: {data_json},
          options: {options_json}
        }});
      }} else {{
        console.error('Failed to find canvas element with ID: {chart_id}');
      }}
    </script>
    """
    return html_template

class ChartInputArgs(BaseModel):
    chart_type: str = Field(..., description="Type of chart (e.g., 'bar', 'line', 'pie').")
    data: Dict[str, Any] = Field(..., description="Data for the chart (labels, datasets). Example: {'labels': ['A', 'B'], 'datasets': [{'label': 'Series 1', 'data': [10, 20]}]}")
    options: Dict[str, Any] = Field(default={}, description="Chart.js options.")

def create_chart_tool_function(chart_type: str, data: Dict[str, Any], options: Dict[str, Any] = None) -> str:
    try:
        if not chart_type or not data:
            return "Error: 'chart_type' and 'data' are required."
        return generate_chart_html(chart_type=chart_type, data=data, options=options or {})
    except Exception as e:
        return f"Error during chart generation: {str(e)}"

analyst_chart_tool = Tool(
    name="ChartGenerator",
    func=create_chart_tool_function,
    description="Generates an HTML/JavaScript chart snippet from structured data. Input should include chart_type, data (with labels and datasets), and optional options.",
    args_schema=ChartInputArgs
)

# --- Customer Churn Prediction Tool ---
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

MODEL_PATH = Path(__file__).parent / "models" / "churn_pipeline_full.pkl"
_churn_model_cache = None

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        self.binary_map = {'Yes': 1, 'No': 0}
        self.categorical_cols = []

    def fit(self, X, y=None):
        X = X.copy()
        # Convert 'totalcharges' to numeric, coercing errors
        X['totalcharges'] = pd.to_numeric(X['totalcharges'], errors='coerce')

        # Identify binary columns and map them
        binary_cols = ['partner', 'dependents', 'phoneservice', 'paperlessbilling']
        for col in binary_cols:
            X[col] = X[col].map(self.binary_map)

        # Identify categorical columns for label encoding
        self.categorical_cols = [
            col for col in X.select_dtypes(include=['object', 'category']).columns
            if not pd.to_numeric(X[col], errors='coerce').notna().all()
        ]

        # Fit label encoders on categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Fit on string-converted column to handle mixed types
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        # Convert 'totalcharges' to numeric, coercing errors
        X['totalcharges'] = pd.to_numeric(X['totalcharges'], errors='coerce')
        # Impute missing 'totalcharges' using tenure and monthly charges
        X['totalcharges'] = X['totalcharges'].fillna(X['tenure'] * X['monthlycharges'])

        # --- Feature Engineering ---
        # Calculate the total number of additional services
        X['new_totalservices'] = (X[['phoneservice', 'internetservice', 'onlinesecurity',
                                     'onlinebackup', 'deviceprotection', 'techsupport',
                                     'streamingtv', 'streamingmovies']] == 'Yes').sum(axis=1)

        # Create new features based on charges and tenure
        X['new_avg_charges'] = X['totalcharges'] / (X['tenure'] + 1e-5)
        X['new_increase'] = X['new_avg_charges'] / (X['monthlycharges'] + 1e-5)
        X['new_avg_service_fee'] = X['monthlycharges'] / (X['new_totalservices'] + 1e-5)
        X['charge_increased'] = (X['monthlycharges'] > X['new_avg_charges']).astype(int)
        X['charge_growth_rate'] = (X['monthlycharges'] - X['new_avg_charges']) / (X['new_avg_charges'] + 1e-5)
        X['is_auto_payment'] = X['paymentmethod'].apply(lambda x: int('automatic' in str(x).lower() or 'bank' in str(x).lower()))

        # Apply binary mapping to the binary columns
        binary_cols = ['partner', 'dependents', 'phoneservice', 'paperlessbilling']
        for col in binary_cols:
            X[col] = X[col].map(self.binary_map)

        # Apply label encoding to the categorical columns
        for col in self.categorical_cols:
            X[col] = self.label_encoders[col].transform(X[col].astype(str))

        # The notebook returns X.values, which is a NumPy array.
        # This is crucial for the SMOTE part of the pipeline.
        return X.values

class ThresholdWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, pipeline, threshold=0.5):
        self.pipeline = pipeline
        self.threshold = threshold

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)
    
    def get_params(self, deep=True):
        params = self.pipeline.get_params(deep=deep) if hasattr(self.pipeline, 'get_params') else {}
        params['threshold'] = self.threshold
        return params

    def set_params(self, **params):
        if 'threshold' in params:
            self.threshold = params.pop('threshold')
        if hasattr(self.pipeline, 'set_params') and params:
            self.pipeline.set_params(**params)
        return self


def _load_churn_model():
    global _churn_model_cache
    if _churn_model_cache is not None:
        return _churn_model_cache

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Error: Churn prediction model file not found at {MODEL_PATH}")

    main_module = sys.modules['__main__']
    original_preprocessor = getattr(main_module, 'Preprocessor', None)
    original_thresholdwrapper = getattr(main_module, 'ThresholdWrapper', None)
    we_set_preprocessor = False
    we_set_thresholdwrapper = False

    try:
        if not (hasattr(main_module, 'Preprocessor') and main_module.Preprocessor == Preprocessor):
            if hasattr(main_module, 'Preprocessor'): 
                raise RuntimeError("Conflict: __main__.Preprocessor exists and is not the expected class for unpickling.")
            main_module.Preprocessor = Preprocessor
            we_set_preprocessor = True
        
        if not (hasattr(main_module, 'ThresholdWrapper') and main_module.ThresholdWrapper == ThresholdWrapper):
            if hasattr(main_module, 'ThresholdWrapper'): 
                raise RuntimeError("Conflict: __main__.ThresholdWrapper exists and is not the expected class for unpickling.")
            main_module.ThresholdWrapper = ThresholdWrapper
            we_set_thresholdwrapper = True

        loaded_pipeline = joblib.load(MODEL_PATH)
        _churn_model_cache = loaded_pipeline
        return _churn_model_cache
    except Exception as e:
        raise RuntimeError(f"Error loading churn model: {str(e)}")
    finally:
        if we_set_preprocessor:
            if original_preprocessor is not None:
                main_module.Preprocessor = original_preprocessor
            else:
                if hasattr(main_module, 'Preprocessor'): 
                    del main_module.Preprocessor
        
        if we_set_thresholdwrapper:
            if original_thresholdwrapper is not None:
                main_module.ThresholdWrapper = original_thresholdwrapper
            else:
                if hasattr(main_module, 'ThresholdWrapper'): 
                    del main_module.ThresholdWrapper

# --- Customer Churn Prediction & Analysis Tool ---

def _predict_churn_and_create_df(csv_data_string: str) -> pd.DataFrame:
    """
    Internal function to run churn prediction and return an enhanced DataFrame.
    This function is not a tool itself.
    """
    try:
        pipeline = _load_churn_model()
        data_io = StringIO(csv_data_string)
        df = pd.read_csv(data_io)

        if df.empty:
            raise ValueError("Input CSV data is empty.")

        # Store original df to join with later, and find CustomerID
        original_df = df.copy()
        id_col_found = None
        for col in df.columns:
            if col.lower().replace('_', '').replace(' ', '') == 'customerid':
                id_col_found = col
                break
        if id_col_found:
            original_df.rename(columns={id_col_found: 'CustomerID'}, inplace=True)
        else:
            # If no ID, create a temporary one for joining
            original_df['temp_id_for_join'] = range(len(original_df))


        # --- Data Normalization & Column Ordering (same as before) ---
        df.columns = [col.lower().replace('_', '') for col in df.columns]
        rename_map = {'internetserivce': 'internetservice'}
        df.rename(columns=rename_map, inplace=True)

        expected_cols_in_order = [
            'gender', 'seniorcitizen', 'partner', 'dependents', 'tenure',
            'phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity',
            'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv',
            'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod',
            'monthlycharges', 'totalcharges'
        ]

        df_for_prediction = pd.DataFrame()
        for col in expected_cols_in_order:
            if col in df.columns:
                df_for_prediction[col] = df[col]
            else:
                df_for_prediction[col] = np.nan
        df_for_prediction = df_for_prediction[expected_cols_in_order]

        # --- Prediction ---
        predictions = pipeline.predict(df_for_prediction)
        probabilities = pipeline.predict_proba(df_for_prediction)[:, 1]

        # --- Combine results with original data ---
        result_df = original_df
        result_df['ChurnPrediction'] = ["Yes" if p == 1 else "No" for p in predictions]
        result_df['ChurnProbability'] = probabilities
        
        if 'temp_id_for_join' in result_df.columns:
            result_df = result_df.drop(columns=['temp_id_for_join'])

        return result_df
    except (FileNotFoundError, RuntimeError, ValueError, pd.errors.EmptyDataError, KeyError, IndexError) as e:
        # Re-raise to be caught by the main tool function
        raise e


class ChurnAnalysisInputArgs(BaseModel):
    """Input schema for the CustomerChurnDataAnalyzer tool."""
    csv_data_string: str = Field(
        description="A string containing the full customer data in CSV format, including headers."
    )
    user_query: str = Field(
        description="The user's natural language question about the CSV data. This can be a general question or a specific one about churn predictions (e.g., 'Show me the top 5 customers most likely to churn')."
    )

def analyze_csv_with_churn_prediction(csv_data_string: str, user_query: str) -> str:
    """
    Performs churn prediction on the provided CSV data and then answers a user's question
    about the data (including the prediction results) using a powerful language model agent.
    """
    try:
        # Step 1: Run prediction and get the enhanced DataFrame
        predicted_df = _predict_churn_and_create_df(csv_data_string)

        # Step 2: Create a pandas DataFrame agent to answer the query
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        pandas_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=predicted_df,
            verbose=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
            allow_caching=False, # Disable caching to ensure fresh answers
        )

        # Step 3: Invoke the agent with the user's query
        # Use a structured prompt for better results
        prompt = f"""
        You are a data analyst. You are given a pandas DataFrame with customer data, which includes churn predictions.
        Your task is to answer the following user question based on the data.
        
        User Question: \"{user_query}\"
        
        Analyze the DataFrame and provide a clear, concise answer.
        If you are asked to show data, present it in a readable format.
        """
        response = pandas_agent.invoke({"input": prompt})

        if isinstance(response, dict) and 'output' in response:
            return response['output']
        return str(response)

    except (FileNotFoundError, RuntimeError, ValueError, pd.errors.EmptyDataError, KeyError, IndexError) as e:
        return f"Error during data processing or prediction: {str(e)}"
    except Exception as e:
        # This will catch errors from the agent execution as well
        return f"An unexpected error occurred during analysis: {str(e)}"

csv_churn_analyzer_tool = Tool(
    name="CustomerChurnDataAnalyzer",
    func=analyze_csv_with_churn_prediction,
    description=(
        "Use this tool when you need to answer questions about customer data from a CSV file. "
        "This tool first runs a churn prediction model on the data, adding 'ChurnPrediction' and 'ChurnProbability' columns, "
        "and then uses this enhanced data to answer the user's specific question. "
        "You MUST provide both the full CSV content as a string and the user's question."
    ),
    args_schema=ChurnAnalysisInputArgs
)

# --- Document Search Tools (Pinecone) ---
_openai_client_cache = None
_pinecone_client_cache = None
_pinecone_index_cache = None

def get_openai_client():
    global _openai_client_cache
    if _openai_client_cache is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        _openai_client_cache = OpenAIClient(api_key=api_key)
    return _openai_client_cache

def get_pinecone_index():
    global _pinecone_client_cache, _pinecone_index_cache
    if _pinecone_index_cache is None:
        api_key = os.getenv("PINECONE_API_KEY")
        env = os.getenv("PINECONE_ENV")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not all([api_key, env, index_name]):
            raise ValueError("Pinecone API key, environment, or index name not set.")
        if _pinecone_client_cache is None:
            _pinecone_client_cache = PineconeClient(api_key=api_key, environment=env)
        _pinecone_index_cache = _pinecone_client_cache.Index(index_name)
    return _pinecone_index_cache

class PineconeSearchArgs(BaseModel):
    query: str = Field(..., description="The search query string.")
    top_k: int = Field(default=3, description="Number of top results to return.")

def search_pinecone_documents(query: str, top_k: int = 3) -> str:
    try:
        openai_client = get_openai_client()
        pinecone_index = get_pinecone_index()
        
        embedding_response = openai_client.embeddings.create(input=[query], model="text-embedding-ada-002")
        query_embedding = embedding_response.data[0].embedding
        
        results = pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        
        output = []
        if results.matches:
            for match in results.matches:
                output.append(f"Document ID: {match.id}\nScore: {match.score:.4f}\nContent Snippet: {match.metadata.get('text_chunk', 'N/A')[:200]}...\nSource: {match.metadata.get('source', 'N/A')}")
            return "\n---\n".join(output)
        else:
            return "No relevant documents found."
    except Exception as e:
        print(f"Error during Pinecone search: {e}", file=sys.stderr)
        return f"Error searching documents: {e}"

search_pinecone_documents_tool = Tool(
    name="PineconeDocumentSearch",
    func=search_pinecone_documents,
    description="Searches for relevant documents in Pinecone based on a query. Returns document IDs, scores, snippets, and sources.",
    args_schema=PineconeSearchArgs
)

class DocumentContentArgs(BaseModel):
    document_id: str = Field(..., description="The ID of the document to retrieve.")

def get_document_content(document_id: str) -> str:
    try:
        pinecone_index = get_pinecone_index()
        fetched_vector = pinecone_index.fetch(ids=[document_id])
        if fetched_vector.vectors and document_id in fetched_vector.vectors:
            metadata = fetched_vector.vectors[document_id].metadata
            if metadata and 'text_chunk' in metadata:
                return f"Document ID: {document_id}\nSource: {metadata.get('source', 'N/A')}\n\nContent:\n{metadata['text_chunk']}"
            else:
                return f"Document ID {document_id} found, but it has no 'text_chunk' in metadata."
        else:
            return f"Document with ID '{document_id}' not found."
    except Exception as e:
        print(f"Error fetching document content: {e}", file=sys.stderr)
        return f"Error fetching document content for ID '{document_id}': {e}"

get_document_content_tool = Tool(
    name="PineconeDocumentContentRetriever",
    func=get_document_content,
    description="Retrieves the full text content of a specific document from Pinecone using its ID.",
    args_schema=DocumentContentArgs
)

# --- SQL Database Tools for Analyst ---
DB_URI = os.getenv("DB_URI")
sql_tools_for_analyst = []

if DB_URI:
    try:
        engine = create_engine(DB_URI)
        db = SQLDatabase(engine)
        llm_for_sql_toolkit = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
        sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm_for_sql_toolkit)
        sql_tools_for_analyst = sql_toolkit.get_tools()
        print("SQL Database tools initialized successfully for Analyst.", file=sys.stderr)
    except Exception as e:
        print(f"Error initializing SQL Database tools: {e}", file=sys.stderr)
        print("SQL tools will be unavailable for the Analyst Assistant.", file=sys.stderr)
        sql_tools_for_analyst = []
else:
    print("DB_URI not found. SQL tools will be unavailable.", file=sys.stderr)
    sql_tools_for_analyst = []

__all__ = [
    "analyst_chart_tool", 
    "csv_churn_analyzer_tool",
    "sql_tools_for_analyst", 
    "Preprocessor", 
    "ThresholdWrapper",
    "search_pinecone_documents_tool",
    "get_document_content_tool"
]
