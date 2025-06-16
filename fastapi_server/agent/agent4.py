"""LangGraph graph for prediction_agent.

Handles ML model predictions based on data from PostgreSQL.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, List, Annotated
import asyncio
import io # Keep for potential future use with other data sources, though not for DB->Pandas
from dotenv import load_dotenv
import operator
import pandas as pd
import psycopg2 # Added for PostgreSQL
import re # Added for parsing table name

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import joblib
import logging
from openai import OpenAI # Keep if LLM summary is still used
from langchain_openai import ChatOpenAI # Keep if LLM summary is still used

# Load environment variables from the project root .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'))

# Setup logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Database Credentials (from .env) ---
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# --- State Definition ---
class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)
    user_query: Optional[str] = None
    # csv_file_content: Optional[str] = None # REMOVED
    db_table_name_for_prediction: Optional[str] = None # ADDED: To store the table if explicitly set for prediction
    final_answer: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __post_init__(self):
        if not self.user_query and self.messages:
            user_messages = [msg for msg in self.messages if isinstance(msg, HumanMessage)]
            if user_messages:
                self.user_query = user_messages[-1].content

    def dict(self):
        result = super().dict()
        if self.final_answer and self.messages is not None:
            result["messages"] = self.messages + [AIMessage(content=self.final_answer)]
        return result

# --- Database Helper Functions ---
async def list_db_tables_async(conn):
    def _list_tables(conn_sync):
        with conn_sync.cursor() as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
            """)
            tables = [row[0] for row in cur.fetchall()]
            return tables
    return await asyncio.to_thread(_list_tables, conn)

async def fetch_data_from_table_async(conn, table_name: str) -> pd.DataFrame:
    def _fetch_data(conn_sync, query_sync):
        return pd.read_sql_query(query_sync, conn_sync)
    
    # Basic sanitization to prevent SQL injection for table name
    # In a production system, use a more robust whitelisting or parameterization if possible
    if not re.match(r"^[a-zA-Z0-9_]+$", table_name):
        raise ValueError(f"Invalid table name format: {table_name}")
    query = f"SELECT * FROM public.\"{table_name}\";" # Assuming tables are in public schema
    logger.info(f"Executing query: {query}")
    return await asyncio.to_thread(_fetch_data, conn, query)

# --- Node Function for Prediction ---
async def prediction_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- PREDICTION NODE (Telecom Churn Prediction via DB) ---")

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    MODEL_PATH = os.path.join(base_path, 'fastapi_server', 'models', 'churn_predictor_pipeline.pkl')
    CATEGORICAL_COLS_PATH = os.path.join(base_path, 'fastapi_server', 'models', 'categorical_cols.pkl')
    LABEL_ENCODERS_PATH = os.path.join(base_path, 'fastapi_server', 'models', 'label_encoders.pkl')

    EXPECTED_FEATURE_ORDER = [
        'seniorcitizen', 'partner', 'dependents', 'tenure', 'phoneservice',
        'multiplelines', 'onlinesecurity', 'onlinebackup', 'techsupport',
        'contract', 'paperlessbilling', 'paymentmethod', 'monthlycharges', 'totalcharges',
        'new_totalservices', 'new_avg_charges', 'new_increase', 'new_avg_service_fee',
        'charge_increased', 'charge_growth_rate', 'is_auto_payment',
        'expected_contract_months', 'contract_gap'
    ]
    CUSTOMER_ID_COL = 'customerid'
    PREDICTION_THRESHOLD = 0.312

    user_query_lower = state.user_query.lower() if state.user_query else ""
    db_conn = None
    input_df: Optional[pd.DataFrame] = None

    try:
        # Establish DB Connection
        if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD]):
            error_msg = "❌ 오류: 데이터베이스 연결 정보가 .env 파일에 올바르게 설정되지 않았습니다."
            logger.error(error_msg)
            return {"messages": state.messages + [AIMessage(content=error_msg)], "final_answer": error_msg, "error_message": error_msg}
        
        def _connect_db(): # Synchronous connect function for to_thread
            return psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        
        db_conn = await asyncio.to_thread(_connect_db)
        logger.info("Successfully connected to PostgreSQL database.")

        # --- Action Dispatch based on user_query ---
        
        # 1. Handle "list tables" or "테이블 목록" request
        if "list tables" in user_query_lower or "테이블 목록" in user_query_lower:
            try:
                # Ensure db_conn is passed to list_db_tables_async if it's needed
                # Assuming list_db_tables_async was updated to accept db_conn based on previous context
                table_list = await list_db_tables_async(db_conn) 
                if table_list:
                    # Format the table list for display
                    formatted_list = "\n".join([f"- {table}" for table in table_list])
                    response_message = f"사용 가능한 테이블 목록입니다:\n{formatted_list}\n\n어떤 테이블의 데이터로 예측을 진행하시겠습니까? '테이블 [테이블명]으로 예측' 형식으로 알려주세요."
                else:
                    response_message = "데이터베이스에 테이블이 없거나 접근할 수 없습니다."
            except Exception as e:
                logger.error(f"Error listing tables: {e}")
                response_message = f"테이블 목록을 가져오는 중 오류가 발생했습니다: {e}"
            
            final_answer = await self._get_final_answer(state.messages + [AIMessage(content=response_message)], "테이블 목록 조회")
            return {"messages": state.messages + [AIMessage(content=final_answer)], "final_answer": final_answer, "error_message": None}

        # User must select a table via 'list tables' command first.
        # We no longer parse table names from free-form input in this node.
        # Check if a table name for prediction is set from a previous selection

        if state.db_table_name_for_prediction: # This line will now be evaluated based on state from previous turns only
            logger.info(f"Attempting prediction using table: {state.db_table_name_for_prediction}")
            try:
                input_df = await fetch_data_from_table_async(db_conn, state.db_table_name_for_prediction)
                if input_df.empty:
                    error_msg = f"❌ 오류: 테이블 '{state.db_table_name_for_prediction}'에 데이터가 없거나 불러올 수 없습니다."
                    logger.error(error_msg)
                    return {"messages": state.messages + [AIMessage(content=error_msg)], "final_answer": error_msg, "error_message": error_msg}
                logger.info(f"Successfully fetched {len(input_df)} rows from table '{state.db_table_name_for_prediction}'.")
            except ValueError as ve:
                error_msg = f"❌ 오류: {ve}"
                logger.error(error_msg)
                return {"messages": state.messages + [AIMessage(content=error_msg)], "final_answer": error_msg, "error_message": error_msg}
            except Exception as db_e:
                error_msg = f"❌ 오류: 테이블 '{state.db_table_name_for_prediction}'에서 데이터 로딩 중 오류 발생: {db_e}"
                logger.error(error_msg)
                return {"messages": state.messages + [AIMessage(content=error_msg)], "final_answer": error_msg, "error_message": error_msg}
        else:
            # 3. If no specific action, prompt user
            clarification_message = "어떤 작업을 원하시나요? 다음 중 하나를 선택해주세요:\n1. 사용 가능한 '테이블 목록' 보기\n2. '테이블 [테이블명]으로 예측' 요청"
            logger.info("No specific action identified, prompting user for clarification.")
            return {"messages": state.messages + [AIMessage(content=clarification_message)], "final_answer": clarification_message, "error_message": None}

        # --- Proceed with Prediction if input_df is loaded ---
        if input_df is None:
            # This case should ideally be handled by the logic above, but as a safeguard:
            error_msg = "❌ 오류: 예측을 위한 입력 데이터가 준비되지 않았습니다. 테이블을 지정해주세요."
            logger.error(error_msg)
            return {"messages": state.messages + [AIMessage(content=error_msg)], "final_answer": error_msg, "error_message": error_msg}

        # 모델 및 전처리기 로드
        pipeline_final = await asyncio.to_thread(joblib.load, MODEL_PATH)
        CATEGORICAL_COLS = await asyncio.to_thread(joblib.load, CATEGORICAL_COLS_PATH)
        label_encoders = await asyncio.to_thread(joblib.load, LABEL_ENCODERS_PATH)

        if input_df.empty:
            final_answer = "❌ 오류: DataFrame이 비어있습니다. (데이터 로딩 후 확인)"
            return {"messages": state.messages + [AIMessage(content=final_answer)], "final_answer": final_answer, "error_message": final_answer}

        if CUSTOMER_ID_COL not in input_df.columns:
            # Try to find a case-insensitive match for customerid
            customer_id_col_actual = None
            for col in input_df.columns:
                if col.lower() == CUSTOMER_ID_COL.lower():
                    customer_id_col_actual = col
                    break
            if not customer_id_col_actual:
                final_answer = f"❌ 오류: 필수 컬럼인 '{CUSTOMER_ID_COL}' (또는 유사한 이름의 고객 ID 컬럼)이 테이블 데이터에 없습니다."
                return {"messages": state.messages + [AIMessage(content=final_answer)], "final_answer": final_answer, "error_message": final_answer}
            logger.info(f"Found customer ID column as '{customer_id_col_actual}'. Using it instead of '{CUSTOMER_ID_COL}'.")
            customer_ids = input_df[customer_id_col_actual]
            X_predict = input_df.drop(columns=[customer_id_col_actual], errors='ignore')
        else:
            customer_ids = input_df[CUSTOMER_ID_COL]
            X_predict = input_df.drop(columns=[CUSTOMER_ID_COL], errors='ignore')

        # Preprocessing (similar to original, adapt for potential case differences in columns from DB)
        # Convert all column names in X_predict to lowercase for consistent matching
        X_predict.columns = [col.lower() for col in X_predict.columns]
        
        # Ensure CATEGORICAL_COLS are also lowercase for matching
        CATEGORICAL_COLS_LOWER = [col.lower() for col in CATEGORICAL_COLS]
        label_encoders_lower = {k.lower(): v for k, v in label_encoders.items()}

        for col_lower in CATEGORICAL_COLS_LOWER:
            if col_lower in X_predict.columns and col_lower in label_encoders_lower:
                le = label_encoders_lower[col_lower]
                # Ensure data consistency before transform (e.g. convert to string if encoder expects strings)
                X_predict[col_lower] = X_predict[col_lower].astype(str)
                try:
                    X_predict[col_lower] = X_predict[col_lower].apply(lambda x: le.transform([x])[0] if pd.notna(x) and x in le.classes_ else -1)
                except Exception as le_err:
                    logger.warning(f"Label encoding error for column {col_lower} with value. Details: {le_err}. Assigning -1.")
                    X_predict[col_lower] = -1 # Fallback for problematic values
            elif col_lower in X_predict.columns: # Column exists but no encoder
                 logger.warning(f"Categorical column '{col_lower}' found in data but no label encoder available. It might be handled as is or dropped depending on model.")

        # Ensure all EXPECTED_FEATURE_ORDER (also lowercased) are present
        EXPECTED_FEATURE_ORDER_LOWER = [f.lower() for f in EXPECTED_FEATURE_ORDER]
        missing_cols = set(EXPECTED_FEATURE_ORDER_LOWER) - set(X_predict.columns)
        for col_lower in missing_cols:
            X_predict[col_lower] = 0 # Fill missing expected columns with 0
            logger.info(f"Added missing expected column '{col_lower}' with value 0.")

        # Reorder columns to match expected order, only include expected columns
        X_predict = X_predict[EXPECTED_FEATURE_ORDER_LOWER]

        predictions_proba = await asyncio.to_thread(pipeline_final.predict_proba, X_predict)
        predictions = (predictions_proba[:, 1] >= PREDICTION_THRESHOLD).astype(int)

        results_df = pd.DataFrame({
            CUSTOMER_ID_COL: customer_ids, # Use original customer_ids before X_predict column name changes
            'Churn Probability': predictions_proba[:, 1],
            'Churn Prediction': predictions
        })
        results_df['Churn Prediction'] = results_df['Churn Prediction'].map({1: 'Yes', 0: 'No'})

        # --- LLM Summary ---
        results_as_string = results_df.to_csv(index=False)
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        summary_prompt = f"""You are a helpful data analyst assistant. Present the following customer churn prediction results to a user in Korean.
Here are the results:
```csv
{results_as_string}
```
Your summary should:
1. Start with \"📊 고객 이탈 예측 분석 결과\".
2. Present the results in a Markdown table.
3. Summarize the overall trend (e.g., \"총 {len(results_df)}명 중 {results_df[results_df['Churn Prediction'] == 'Yes'].shape[0]}명 이탈 예측\").
4. Conclude with a friendly closing remark.
"""
        response = await llm.ainvoke(summary_prompt)
        final_answer = response.content

        logger.info("Prediction successful.")
        return {"messages": state.messages + [AIMessage(content=final_answer)], "final_answer": final_answer, "error_message": None}

    except Exception as e:
        error_msg = f"❌ 예측 중 오류 발생: {e}"
        logger.exception("Unhandled exception in prediction_node") # Log full traceback
        return {"messages": state.messages + [AIMessage(content=error_msg)], "final_answer": error_msg, "error_message": error_msg}
    finally:
        if db_conn:
            await asyncio.to_thread(db_conn.close)
            logger.info("PostgreSQL connection closed.")

# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("prediction_node", prediction_node)
workflow.set_entry_point("prediction_node")
workflow.add_edge("prediction_node", END)
app = workflow.compile()

# --- Main function to expose the graph ---
async def main():
    return app

if __name__ == "__main__":
    async def test_agent():
        graph = await main()
        
        # Test case 1: List tables
        # inputs1 = {"messages": [HumanMessage(content="테이블 목록 보여줘")]}
        # print("\n--- Testing: List Tables ---")
        # async for event in graph.astream(inputs1):
        #     if "prediction_node" in event:
        #         print(event["prediction_node"]["final_answer"])
        #     print("---")

        # Test case 2: Predict from a specific table (replace 'your_test_table' with an actual table name)
        # Ensure 'your_test_table' exists and has data compatible with the model.
        # Common tables might be 'customer_data_raw' or 'telecom_churn_data'
        test_table_name = "customer_data_raw" # <<-- REPLACE WITH YOUR ACTUAL TABLE NAME FOR TESTING
        inputs2 = {"messages": [HumanMessage(content=f"테이블 {test_table_name} 데이터로 예측해줘")]}
        print(f"\n--- Testing: Predict from table '{test_table_name}' ---")
        async for event in graph.astream(inputs2):
            if "prediction_node" in event:
                print(event["prediction_node"]["final_answer"])
            print("---")
        
        # Test case 3: Ambiguous query, should prompt for action
        # inputs3 = {"messages": [HumanMessage(content="이탈 예측 좀 해줘")]}
        # print("\n--- Testing: Ambiguous Query ---")
        # async for event in graph.astream(inputs3):
        #     if "prediction_node" in event:
        #         print(event["prediction_node"]["final_answer"])
        #     print("---")

    # Make sure your .env file has DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD set correctly.
    asyncio.run(test_agent())
