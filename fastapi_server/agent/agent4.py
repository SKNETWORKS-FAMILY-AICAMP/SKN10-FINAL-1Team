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
    csv_file_content: Optional[str] = None # REMOVED
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
    MODEL_PATH = os.path.join(base_path, 'fastapi_server', 'models', 'churn_pipeline_full.pkl')

    # EXPECTED_FEATURE_ORDER = [
    #     'seniorcitizen', 'partner', 'dependents', 'tenure', 'phoneservice',
    #     'multiplelines', 'onlinesecurity', 'onlinebackup', 'techsupport',
    #     'contract', 'paperlessbilling', 'paymentmethod', 'monthlycharges', 'totalcharges',
    #     'new_totalservices', 'new_avg_charges', 'new_increase', 'new_avg_service_fee',
    #     'charge_increased', 'charge_growth_rate', 'is_auto_payment',
    #     'expected_contract_months', 'contract_gap'
    # ]
    CUSTOMER_ID_COL = 'customerid'

    user_query_lower = state.user_query.lower() if state.user_query else ""
    db_conn = None

    try:
        # Establish DB Connection
        if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD]):
            error_msg = "❌ DB 연결 정보가 누락되었습니다."
            logger.error(error_msg)
            return state.dict() | {"final_answer": error_msg, "error_message": error_msg}
        
        def _connect_db(): # Synchronous connect function for to_thread
            return psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        
        db_conn = await asyncio.to_thread(_connect_db)
        logger.info("Successfully connected to PostgreSQL database.")

        # --- Action Dispatch based on user_query ---
        
        # 1. Handle "list tables" or "테이블 목록" request
        if "list tables" in user_query_lower or "테이블 목록" in user_query_lower:
            tables = await list_db_tables_async(db_conn)
            if not tables:
                msg = "📭 사용할 수 있는 테이블이 없습니다."
            else:
                formatted = "\n".join([f"- {t}" for t in tables])
                msg = f"📋 사용 가능한 테이블:\n{formatted}\n\n'테이블 [테이블명]으로 예측'이라고 입력해줘!"
            return state.dict() | {"final_answer": msg}  # 전체 교체해도 문제 없음

        # User must select a table via 'list tables' command first.
        # We no longer parse table names from free-form input in this node.
        # Check if a table name for prediction is set from a previous selection

        if not state.db_table_name_for_prediction:
            msg = "어떤 테이블로 예측할까요?\n- 테이블 목록 보기\n- '테이블 [테이블명]으로 예측' 형식으로 요청해줘."
            return state.dict() | {"final_answer": msg}  # 없음 안내 메시지

        df = await fetch_data_from_table_async(db_conn, state.db_table_name_for_prediction)
        if df.empty:
            error_msg = f"❌ '{state.db_table_name_for_prediction}' 테이블에 데이터가 없습니다."
            logger.error(error_msg)
            return state.dict() | {"final_answer": error_msg, "error_message": error_msg}

        model = await asyncio.to_thread(joblib.load, MODEL_PATH)

        if CUSTOMER_ID_COL.lower() not in [col.lower() for col in df.columns]:
            error_msg = f"❌ '{CUSTOMER_ID_COL}' 컬럼이 없습니다."
            logger.error(error_msg)
            return state.dict() | {"final_answer": error_msg, "error_message": error_msg}

        actual_cid_col = next(col for col in df.columns if col.lower() == CUSTOMER_ID_COL.lower())
        customer_ids = df[actual_cid_col]
        X = df.drop(columns=[actual_cid_col])

        predictions_proba = await asyncio.to_thread(model.predict_proba, X)
        predictions = await asyncio.to_thread(model.predict, X)

        result_df = pd.DataFrame({
            CUSTOMER_ID_COL: customer_ids,
            "Churn Probability": predictions_proba[:, 1],
            "Churn Prediction": ["Yes" if p else "No" for p in predictions]
        })

        summary_text = f"""📊 고객 이탈 예측 분석 결과:
```csv
{result_df.to_csv(index=False)}
```
총 {len(result_df)}명 중 {sum(result_df['Churn Prediction'] == 'Yes')}명이 이탈할 가능성이 있습니다.
감사합니다 :)"""

        return state.dict() | {"final_answer": summary_text}

    except Exception as e:
        error_msg = f"❌ 예측 중 오류: {e}"
        logger.exception(error_msg)
        return state.dict() | {"final_answer": error_msg, "error_message": error_msg}

    finally:
        if db_conn:
            await asyncio.to_thread(db_conn.close)

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
        inputs = {"messages": [HumanMessage(content="테이블 customer_data_raw 데이터로 예측해줘")]}  # 예시
        async for event in graph.astream(inputs):
            if "prediction_node" in event:
                print(event["prediction_node"]["final_answer"])
        
        # Test case 3: Ambiguous query, should prompt for action
        # inputs3 = {"messages": [HumanMessage(content="이탈 예측 좀 해줘")]}
        # print("\n--- Testing: Ambiguous Query ---")
        # async for event in graph.astream(inputs3):
        #     if "prediction_node" in event:
        #         print(event["prediction_node"]["final_answer"])
        #     print("---")

    # Make sure your .env file has DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD set correctly.
    asyncio.run(test_agent())
