"""LangGraph graph for analysis_agent with supervisor logic.

Handles DB queries and general questions based on node/edge routing.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, List, Annotated, Literal, TypedDict
import asyncio
import io
from dotenv import load_dotenv
import os
import operator # For adding to message history
import psycopg2
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage # For message types
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig # Added missing import
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# --- Configuration (Optional - can be used to pass API keys, model names, etc.) ---
class Configuration(TypedDict, total=False):
    openai_api_key: Optional[str]
    db_env_path: Optional[str] # Path to .env file for DB credentials

# --- State Definition ---
class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)
    user_query: Optional[str] = None
    csv_file_content: Optional[str] = None
    query_type: Optional[Literal["db_query", "category_predict_query", "general_query"]] = None
    sql_query: Optional[str] = None
    sql_result: Optional[Any] = None
    final_answer: Optional[str] = None
    error_message: Optional[str] = None
    visualization_output: Optional[str] = None
    sql_output_choice: Optional[Literal["summarize", "visualize"]] = None # Decision for SQL output processing

    class Config:
        arbitrary_types_allowed = True # For Annotated and operator.add with BaseMessage

# --- LLM and Prompts Setup ---
# Ensure OPENAI_API_KEY is set in your environment or passed via config
llm = ChatOpenAI(temperature=0, model="gpt-4o") # Or your preferred model

supervisor_prompt = PromptTemplate.from_template(
    "Analyze the user's question. Respond with a JSON object.\n"
    "The JSON object MUST contain a 'query_type' field set to one of 'db_query', 'category_predict_query', or 'general_query'.\n\n"
    "User Question: {user_query}\n\n"
    "JSON Response (must be a valid JSON object adhering to the Pydantic model `SupervisorDecision`):"
)

sql_generation_prompt = PromptTemplate.from_template(
    """Based on the following user question and the provided database schema information, convert the question into a SQL query in PostgreSQL syntax AND determine if the result should be summarized or visualized.
Accurately understand the user's intent and use the schema information to write a query with the correct tables and columns.
For text-based searches (e.g., using LIKE clauses on string columns), ensure the search is case-insensitive by using the ILIKE operator or by applying the LOWER() function to both the column and the search term, unless the user specifically requests a case-sensitive search.
If the information needed for the question is not in the schema or is ambiguous, please state that or use the most probable interpretation.

Database Schema Information:

Table Name: analytics_results
Columns:
  - id (uuid)
  - result_type (character varying)
  - s3_key (text)
  - meta (jsonb)
  - created_at (timestamp with time zone)
  - user_id (uuid)

Table Name: chat_messages
Columns:
  - id (uuid)
  - role (character varying)
  - content (text)
  - created_at (timestamp with time zone)
  - session_id (uuid)
  - metadata (text)

Table Name: chat_sessions
Columns:
  - id (uuid)
  - agent_type (character varying)
  - started_at (timestamp with time zone)
  - ended_at (timestamp with time zone)
  - user_id (uuid)
  - title (character varying)

Table Name: llm_calls
Columns:
  - id (uuid)
  - call_type (character varying)
  - prompt (text)
  - response (text)
  - tokens_used (integer)
  - latency_ms (integer)
  - created_at (timestamp with time zone)
  - user_id (uuid)
  - session_id (uuid)

Table Name: model_artifacts
Columns:
  - id (uuid)
  - artifact_type (character varying)
  - s3_key (text)
  - meta (jsonb)
  - created_at (timestamp with time zone)
  - user_id (uuid)


Table Name: organizations
Columns:
  - id (uuid)
  - name (character varying)
  - created_at (timestamp with time zone)

Table Name: summary_news_keywords
Columns:
  - id (uuid)
  - date (date)
  - keyword (text)
  - title (text)
  - summary (text)
  - url (text)

Table Name: telecom_customers
Columns:
  - customer_id (character varying)
  - gender (character varying)
  - senior_citizen (boolean)
  - partner (boolean)
  - dependents (boolean)
  - tenure (integer)
  - phone_service (boolean)
  - multiple_lines (character varying)
  - internet_service (character varying)
  - online_security (character varying)
  - online_backup (character varying)
  - device_protection (character varying)
  - tech_support (character varying)
  - streaming_tv (character varying)
  - streaming_movies (character varying)
  - contract (character varying)
  - paperless_billing (boolean)
  - payment_method (character varying)
  - monthly_charges (numeric)
  - total_charges (numeric)
  - churn (boolean)

Table Name: users
Columns:
  - password (character varying)
  - is_superuser (boolean)
  - id (uuid)
  - email (character varying)
  - name (character varying)
  - role (character varying)
  - created_at (timestamp with time zone)
  - last_login (timestamp with time zone)
  - is_active (boolean)
  - is_staff (boolean)
  - org_id (uuid)

Respond with a JSON object that strictly adheres to the Pydantic model `SQLGenerationOutput` shown below.
The `sql_query` field MUST contain ONLY the SQL query string, without any surrounding text, explanations, or markdown formatting like ```sql.
The `sql_output_choice` field MUST be either 'summarize' (if the user asks for a textual summary, explanation, or direct answer from data) or 'visualize' (if the user asks for a chart, graph, or visual representation). Prioritize 'visualize' if visualization is explicitly or implicitly requested. If unsure, default to 'summarize'.

```python
class SQLGenerationOutput(BaseModel):
    sql_query: str = Field(description="The generated SQL query. This field MUST contain ONLY the SQL query string, without any surrounding text, explanations, or markdown formatting like ```sql.")
    sql_output_choice: Literal["summarize", "visualize"] = Field(description="The type of output processing required for the SQL result: 'summarize' or 'visualize'.")
```

User Question: {user_query}

JSON Response (must be a valid JSON object conforming to SQLGenerationOutput):
    """
)

summarization_prompt = PromptTemplate.from_template(
    "Based on the following SQL query execution result, please summarize the answer to the user's original question naturally.\n\n"
    "say in Korean"
    "User Question: {user_query}\n"
    "SQL Query: {sql_query}\n"
    "SQL Result:\n{sql_result}\n\n"
    "Summary Answer:"
)

general_answer_prompt = PromptTemplate.from_template(
    "Please answer the following user question.\n\n"
    "User Question: {user_query}\n\n"
    "Answer:"
)

# --- Pydantic Models for Structured Output ---
class SupervisorDecision(BaseModel):
    query_type: str = Field(description="The type of the user's question (db_query, category_predict_query, or general_query)")

class SQLGenerationOutput(BaseModel):
    sql_query: str = Field(description="The generated SQL query. This field MUST contain ONLY the SQL query string, without any surrounding text, explanations, or markdown formatting like ```sql.")
    sql_output_choice: Literal["summarize", "visualize"] = Field(description="The type of output processing required for the SQL result: 'summarize' or 'visualize'.")

# --- Node Functions ---
async def supervisor_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- SUPERVISOR NODE (Debug v4 - Enhanced Message Parsing) ---")
    
    user_query_to_process: Optional[str] = None
    current_messages = state.messages # Keep a reference to the current messages

    print(f"Supervisor - Initial state.messages length: {len(current_messages) if current_messages else 0}")

    if current_messages:
        last_message_obj = current_messages[-1]
        print(f"Supervisor - Last message object: {last_message_obj}, type: {type(last_message_obj).__name__}")

        content_candidate: Optional[str] = None

        # Try to extract content based on common patterns
        if hasattr(last_message_obj, 'content') and isinstance(getattr(last_message_obj, 'content'), str):
            # Covers HumanMessage, AIMessage, and other BaseMessage with a .content attribute
            content_candidate = getattr(last_message_obj, 'content')
            print(f"Supervisor - Candidate from .content attribute: '{content_candidate[:100] if content_candidate else 'None'}...'")
        elif isinstance(last_message_obj, dict) and 'content' in last_message_obj and isinstance(last_message_obj['content'], str):
            # Covers cases where the message might be a dictionary (e.g., from serialization)
            content_candidate = last_message_obj['content']
            print(f"Supervisor - Candidate from dict['content']: '{content_candidate[:100] if content_candidate else 'None'}...'")
        elif isinstance(last_message_obj, str):
            # Covers cases where the last message itself is a plain string
            content_candidate = last_message_obj
            print(f"Supervisor - Candidate from last_message_obj being a string: '{content_candidate[:100] if content_candidate else 'None'}...'")
        
        # Ensure the extracted content is a non-empty string
        if content_candidate and content_candidate.strip():
            user_query_to_process = content_candidate.strip()
            print(f"Supervisor - Successfully extracted user query: '{user_query_to_process[:100]}...'")
        else:
            print(f"Supervisor - Content candidate was None, empty, or whitespace. Candidate: '{content_candidate}'")
    else:
        print("Supervisor - state.messages is empty.")

    if not user_query_to_process:
        error_msg = "Supervisor - No valid user query found in state.messages. Please provide input via the 'Messages' field with textual content."
        print(f"Supervisor - {error_msg}")
        return {
            "user_query": None,
            "query_type": "general_query", # Default to general_query if no input
            "error_message": error_msg,
            "messages": current_messages # Pass through existing messages
        }

    # If user_query_to_process is successfully set, proceed with LLM analysis
    print(f"Supervisor - Processing user query: '{user_query_to_process}' for LLM analysis.")
    
    supervisor_chain = supervisor_prompt | llm.with_structured_output(SupervisorDecision)
    
    try:
        # Ensure the user_query for the LLM is the one we processed
        parsed_output: SupervisorDecision = await supervisor_chain.ainvoke({"user_query": user_query_to_process}, config=config)
        query_type = parsed_output.query_type
        print(f"Supervisor Decision: query_type = {query_type}")

        updated_messages = current_messages + [AIMessage(content=f"Routing to {query_type} based on supervisor decision.")]
        return {
            "messages": updated_messages,
            "user_query": user_query_to_process,
            "query_type": query_type,
            "error_message": None # Clear any previous error messages
        }
    except Exception as e:
        error_msg = f"Supervisor - Error during LLM decision for query '{user_query_to_process[:50]}...': {e}"
        print(error_msg)
        return {
            "user_query": user_query_to_process, # Query was extracted, but LLM failed
            "query_type": "general_query", # Default to general if LLM fails
            "error_message": error_msg,
            "messages": current_messages # Pass through existing messages
        }

async def generate_sql_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- GENERATE SQL NODE ---")
    user_query = state.user_query # Get user_query from the state
    if not user_query:
        return {"error_message": "No user query found for SQL generation.", "sql_query": "", "sql_output_choice": None}

    print(f"Generating SQL and determining output choice for query: {user_query[:100]}...")
    # Use llm.with_structured_output for more robust parsing to the Pydantic model
    structured_llm_sql_gen = llm.with_structured_output(SQLGenerationOutput)
    # The runnable now takes the prompt and applies the structured LLM
    sql_generation_runnable = sql_generation_prompt | structured_llm_sql_gen

    try:
        # Pass the user_query to the runnable, which will format the prompt and call the structured LLM
        response_model = await sql_generation_runnable.ainvoke({"user_query": user_query}, config=config)
        # response_model should now be an instance of SQLGenerationOutput
        sql_query = response_model.sql_query.strip()
        sql_output_choice = response_model.sql_output_choice
        print(f"Generated SQL: {sql_query}")
        print(f"Determined SQL Output Choice: {sql_output_choice}")
        return {"sql_query": sql_query, "sql_output_choice": sql_output_choice, "error_message": None}
    except Exception as e:
        error_msg = f"Error generating SQL or determining output choice: {e}"
        print(error_msg)
        return {"error_message": error_msg, "sql_query": "", "sql_output_choice": None}

import asyncio # Required for asyncio.to_thread

# Helper function for synchronous DB operations, including dotenv loading
def _execute_sql_sync(sql_query: str, base_dir_analysis_env: str, base_dir_my_state_env: str) -> Dict[str, Any]:
    # Determine .env path and load environment variables
    dotenv_path_analysis = os.path.join(base_dir_analysis_env, '.env')
    dotenv_path_my_state = os.path.join(base_dir_my_state_env, '.env')
    specific_env_path = None

    if os.path.exists(dotenv_path_analysis):
        specific_env_path = dotenv_path_analysis
    elif os.path.exists(dotenv_path_my_state):
        specific_env_path = dotenv_path_my_state

    if specific_env_path:
        print(f"_execute_sql_sync: Loading .env from: {specific_env_path}")
        load_dotenv(dotenv_path=specific_env_path, override=True)
    else:
        print("_execute_sql_sync: No specific .env file found. Relying on system environment variables or a global .env.")
        load_dotenv(override=True) # Load global .env or system vars

    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")

    if not all([db_host, db_port, db_name, db_user, db_password]):
        error_msg = "_execute_sql_sync: Database connection details missing in environment variables."
        print(error_msg)
        return {"error_message": error_msg, "sql_result": ""}

    conn_string = f"host='{db_host}' port='{db_port}' dbname='{db_name}' user='{db_user}' password='{db_password}'"
    conn = None
    try:
        print(f"_execute_sql_sync: Connecting to DB with: {conn_string.replace(db_password, '****') if db_password else conn_string}")
        conn = psycopg2.connect(conn_string)
        print(f"_execute_sql_sync: Executing SQL: {sql_query}")
        df = pd.read_sql_query(sql_query, conn)
        sql_result_str = df.to_string()
        print(f"_execute_sql_sync: SQL Result (first 200 chars): {sql_result_str[:200]}")
        return {"sql_result": sql_result_str, "error_message": None}
    except (psycopg2.Error, pd.io.sql.DatabaseError) as e:
        error_msg = f"_execute_sql_sync: Database error: {e}"
        print(error_msg)
        return {"error_message": error_msg, "sql_result": ""}
    except ValueError as e: # Often from pandas if query is malformed for read_sql_query
        error_msg = f"_execute_sql_sync: SQL query validation error for pandas: {e}"
        print(error_msg)
        return {"error_message": error_msg, "sql_result": ""}
    except Exception as e:
        error_msg = f"_execute_sql_sync: An unexpected error occurred during SQL execution: {e}"
        print(error_msg)
        return {"error_message": error_msg, "sql_result": ""}
    finally:
        if conn:
            conn.close()
            print("_execute_sql_sync: DB 연결 종료.")

async def execute_sql_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- EXECUTE SQL NODE ---")
    sql_query = state.sql_query
    if not sql_query:
        return {"error_message": "No SQL query to execute.", "sql_result": ""}

    # Define base directories for .env file search relative to this file's location
    # analysis_agent/.env -> ../../.env
    base_dir_analysis_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # my_state_agent/.env -> ../../../my_state_agent
    base_dir_my_state_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'my_state_agent'))
    
    # Run the synchronous dotenv loading and database operations in a separate thread
    try:
        print(f"execute_sql_node: Calling asyncio.to_thread for SQL: {sql_query}")
        result_dict = await asyncio.to_thread(
            _execute_sql_sync, 
            sql_query, 
            base_dir_analysis_env, 
            base_dir_my_state_env
        )
        return result_dict
    except Exception as e: # Catch potential errors from asyncio.to_thread itself
        error_msg = f"execute_sql_node: Error running DB operations in thread: {e}"
        print(error_msg)
        return {"error_message": error_msg, "sql_result": ""}

async def create_visualization_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- CREATE VISUALIZATION NODE (Placeholder) ---")
    sql_result = state.sql_result
    if not sql_result:
        # If there's an error message from a previous node, pass it along.
        # Otherwise, set a specific error for this node.
        error_to_pass = state.error_message or "No SQL result to visualize."
        print(f"Create Visualization Node: Error - {error_to_pass}")
        return {"error_message": error_to_pass, "sql_result": sql_result} # Pass original sql_result for context

    # Placeholder: Simulate visualization creation
    # In a real scenario, this would generate a chart, table, or some visual representation.
    visualization_output = f"[Placeholder: Visualization for SQL result: {str(sql_result)[:100]}...]"
    print(f"Create Visualization Node: Generated - {visualization_output}")
    
    # For now, we'll just pass the original sql_result and a note about visualization.
    # If the next node needs specific visualization data, we'd add it to the state here.
    return {"sql_result": sql_result, "visualization_output": visualization_output, "error_message": None}

async def summarize_sql_result_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- SUMMARIZE SQL RESULT NODE ---")
    user_query = state.user_query # Get user_query from the state field set by supervisor
    sql_query = state.sql_query or ""
    sql_result = state.sql_result or ""

    if not user_query:
         return {"error_message": "No user query found for summarization.", "final_answer": ""}
    if not sql_result:
        return {"error_message": state.error_message or "No SQL result to summarize.", "final_answer": state.error_message or "SQL query execution failed or produced no result."}

    print(f"Summarizing for query: {user_query}, SQL: {sql_query[:100]}..., Result: {sql_result[:100]}...")
    summarization_chain = summarization_prompt | llm
    response = await summarization_chain.ainvoke(
        {"user_query": user_query, "sql_query": sql_query, "sql_result": sql_result},
        config=config
    )
    final_answer = response.content.strip()
    print(f"Summarized Answer: {final_answer}")
    return {"final_answer": final_answer}

async def general_question_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- GENERAL QUESTION NODE ---")
    if not state.messages or not isinstance(state.messages[-1], HumanMessage):
        return {"error_message": "No user query found for general question.", "final_answer": ""}
    user_query = state.messages[-1].content
    general_answer_chain = general_answer_prompt | llm
    response = await general_answer_chain.ainvoke({"user_query": user_query}, config=config)
    final_answer = response.content.strip()
    print(f"General Answer: {final_answer}")
    return {"final_answer": final_answer}

async def category_predict_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- CATEGORY PREDICT NODE (Telecom Churn Prediction) ---")
    # user_query is now handled by the logic below, or via state.csv_file_content

    # --- 모델 및 전처리 파일 경로 설정 ---
    base_path = os.path.join(os.path.dirname(__file__), '..', '..', '..') # Project root
    MODEL_PATH = os.path.join(base_path, 'churn_predictor_pipeline.pkl')
    CATEGORICAL_COLS_PATH = os.path.join(base_path, 'categorical_cols.pkl')
    LABEL_ENCODERS_PATH = os.path.join(base_path, 'label_encoders.pkl')
    
    # !!! 중요 !!! 사용자 직접 수정 필요 !!!
    # 아래 EXPECTED_FEATURE_ORDER는 실제 모델 학습에 사용된 **모든 특성들의 정확한 이름과 순서**로 반드시 수정해야 합니다.
    # (고객 ID 컬럼과 타겟 변수(이탈 여부)는 제외)
    # 이 순서는 모델 학습 시 사용된 데이터프레임의 X.columns 순서와 정확히 일치해야 합니다.
    # 순서가 맞지 않으면 모델이 특성을 잘못 해석하여 예측 결과가 완전히 틀릴 수 있습니다.
    EXPECTED_FEATURE_ORDER = [
        'seniorcitizen', 'partner', 'dependents', 'tenure', 'phoneservice',
        'multiplelines', 'onlinesecurity', 'onlinebackup',
        'techsupport',
        'contract', 'paperlessbilling', 'paymentmethod', 'monthlycharges',
        'totalcharges',
        # Derived features (순서는 학습 시와 동일해야 함)
        'new_totalservices', 'new_avg_charges', 'new_increase', 'new_avg_service_fee',
        'charge_increased', 'charge_growth_rate', 'is_auto_payment',
        'expected_contract_months', 'contract_gap'
    ] 
    CUSTOMER_ID_COL = 'customerid' # CSV 파일 내 고객 ID 컬럼명 (사용자 제공 sample_users_30.csv에 따름)
    PREDICTION_THRESHOLD = 0.312
    # --- 설정 끝 ---

    raw_data = None
    if state.csv_file_content:
        print("CSV 파일 내용을 직접 사용하여 데이터 로드 중...")
        try:
            raw_data = pd.read_csv(io.StringIO(state.csv_file_content))
        except Exception as e:
            error_msg = f"오류: 제공된 CSV 파일 내용을 파싱하는 중 문제가 발생했습니다: {e}"
            print(error_msg)
            return {"final_answer": error_msg, "error_message": error_msg}
    elif state.user_query and os.path.exists(state.user_query):
        print(f"CSV 파일 경로({state.user_query})를 사용하여 데이터 로드 중...")
        try:
            raw_data = await asyncio.to_thread(pd.read_csv, state.user_query)
        except Exception as e:
            error_msg = f"오류: CSV 파일({state.user_query})을 읽는 중 문제가 발생했습니다: {e}"
            print(error_msg)
            return {"final_answer": error_msg, "error_message": error_msg}
    else:
        error_msg = f"CSV 파일 경로가 제공되지 않았거나 파일 내용이 제공되지 않았습니다. 사용자 입력: '{state.user_query}', CSV 내용 제공 여부: {'제공됨' if state.csv_file_content else '제공안됨'}"
        print(error_msg)
        return {"final_answer": "오류: CSV 파일을 찾을 수 없거나 내용이 제공되지 않았습니다. 정확한 파일 경로를 입력하거나 파일 내용을 전달해주세요.", "error_message": error_msg}

    if raw_data is None or raw_data.empty:
        error_msg = "오류: CSV 데이터가 비어있거나 로드에 실패했습니다. (데이터 로드 후 확인)"
        print(error_msg)
        return {"final_answer": error_msg, "error_message": error_msg}

    try:
        # 1. 필수 파일 존재 확인
        if not os.path.exists(MODEL_PATH):
            error_msg = f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}"
            print(error_msg)
            return {"final_answer": "오류: 학습된 모델 파이프라인 파일을 찾을 수 없습니다.", "error_message": error_msg}
        if not os.path.exists(CATEGORICAL_COLS_PATH):
            error_msg = f"범주형 컬럼 목록 파일을 찾을 수 없습니다: {CATEGORICAL_COLS_PATH}"
            print(error_msg)
            return {"final_answer": "오류: 범주형 컬럼 정보 파일을 찾을 수 없습니다.", "error_message": error_msg}
        if not os.path.exists(LABEL_ENCODERS_PATH):
            error_msg = f"LabelEncoder 파일을 찾을 수 없습니다: {LABEL_ENCODERS_PATH}"
            print(error_msg)
            return {"final_answer": "오류: 학습된 LabelEncoder 파일을 찾을 수 없습니다.", "error_message": error_msg}

        # 2. 모델 및 전처리 객체 로드 (비동기 처리)
        print(f"모델 로드 시작: {MODEL_PATH}")
        pipeline_final = await asyncio.to_thread(joblib.load, MODEL_PATH)
        print(f"모델 로드 완료: {MODEL_PATH}")
        
        print(f"범주형 컬럼 목록 로드 시작: {CATEGORICAL_COLS_PATH}")
        CATEGORICAL_COLS = await asyncio.to_thread(joblib.load, CATEGORICAL_COLS_PATH)
        print(f"범주형 컬럼 목록 로드 완료. 목록: {CATEGORICAL_COLS}")
        
        print(f"LabelEncoder 로드 시작: {LABEL_ENCODERS_PATH}")
        loaded_label_encoders = await asyncio.to_thread(joblib.load, LABEL_ENCODERS_PATH)
        print("LabelEncoder 로드 완료.")

        # 3. CSV 데이터 사용 (이미 raw_data로 로드됨)
        input_df = raw_data.copy() # Use the already loaded raw_data
        print(f"로드된 CSV 데이터 사용. Shape: {input_df.shape}")
        
        if CUSTOMER_ID_COL not in input_df.columns:
            error_msg = f"'{CUSTOMER_ID_COL}' 컬럼이 CSV 파일에 없습니다."
            print(error_msg)
            return {"final_answer": f"오류: CSV 파일에 고객 ID 컬럼('{CUSTOMER_ID_COL}')이 없습니다.", "error_message": error_msg}
        
        customer_ids = input_df[CUSTOMER_ID_COL]
        X_predict = input_df.drop(columns=[CUSTOMER_ID_COL], errors='ignore')

        # 4. 데이터 전처리 (학습 과정과 동일하게 수행)
        # 4.1. 범주형 변수 인코딩 (미리 학습된 LabelEncoder 사용)
        for col in CATEGORICAL_COLS:
            if col in X_predict.columns:
                if col in loaded_label_encoders:
                    le = loaded_label_encoders[col]
                    print(f"Encoding column: {col} using loaded LabelEncoder.")
                    
                    # Convert current column to string type to ensure compatibility with LabelEncoder
                    current_column_str = X_predict[col].astype(str)
                    
                    # Prepare a list to store transformed values
                    transformed_values = []
                    
                    # Get known classes from the encoder to check against
                    known_classes = list(le.classes_)

                    for value in current_column_str:
                        if value in known_classes:
                            transformed_values.append(le.transform([value])[0])
                        else:
                            # Handle unseen labels: assign a placeholder (e.g., -1)
                            # Log a warning as this might impact prediction quality
                            print(f"Warning: Unseen label '{value}' in column '{col}'. Assigning -1 as placeholder.")
                            transformed_values.append(-1) 
                            
                    X_predict[col] = transformed_values
                else:
                    error_msg = f"오류: '{col}'에 대한 LabelEncoder를 찾을 수 없습니다. '{LABEL_ENCODERS_PATH}' 파일과 '{CATEGORICAL_COLS_PATH}' 파일의 내용을 확인하세요."
                    print(error_msg)
                    return {"final_answer": error_msg, "error_message": error_msg}
            else:
                # 필수 범주형 컬럼이 CSV에 없는 경우
                error_msg = f"필수 범주형 컬럼 '{col}'이(가) CSV 파일에 없습니다. ({CATEGORICAL_COLS_PATH} 목록 기준)"
                print(error_msg)
                return {"final_answer": f"오류: 예측에 필요한 '{col}' 컬럼이 CSV 파일에 없습니다.", "error_message": error_msg}
        
        # 4.2. 특성 순서 정렬 및 누락된 특성 처리
        # 모든 EXPECTED_FEATURE_ORDER 컬럼이 X_predict에 있도록 보장 (없으면 NaN으로 채움)
        # 이 작업은 사용자가 EXPECTED_FEATURE_ORDER를 정확히 제공했을 때 의미가 있습니다.
        for col in EXPECTED_FEATURE_ORDER:
            if col not in X_predict.columns:
                print(f"경고: 예측 데이터에 '{col}' 컬럼이 없어 NaN으로 채웁니다. (EXPECTED_FEATURE_ORDER 기준). 모델 성능에 영향을 줄 수 있습니다.")
                X_predict[col] = np.nan 
        
        # 최종적으로 모델이 기대하는 순서대로 컬럼을 정렬
        # 만약 EXPECTED_FEATURE_ORDER에 없는 컬럼이 X_predict에 있다면, 이 단계에서 제외됨.
        try:
            X_predict = X_predict[EXPECTED_FEATURE_ORDER]
        except KeyError as e:
            error_msg = f"오류: EXPECTED_FEATURE_ORDER에 정의된 컬럼 중 일부가 CSV에 없거나, 전처리 후에도 생성되지 않았습니다. 누락된 컬럼 가능성: {e}. EXPECTED_FEATURE_ORDER를 확인하세요."
            print(error_msg)
            return {"final_answer": error_msg, "error_message": error_msg}

        # 4.3. NumPy 배열로 변환 (파이프라인이 NumPy를 기대하는 경우)
        X_predict_np = X_predict.to_numpy()
        print(f"데이터 전처리 완료. 예측할 데이터 Shape: {X_predict_np.shape}")

        # 5. 이탈 확률 및 예측 수행
        y_proba = pipeline_final.predict_proba(X_predict_np)[:, 1]
        y_pred_binary = (y_proba >= PREDICTION_THRESHOLD).astype(int)

        # 6. 결과 생성
        results = []
        for cid, proba, pred in zip(customer_ids, y_proba, y_pred_binary):
            status = "이탈 예상" if pred == 1 else "유지 예상"
            results.append(f"고객 ID {cid}: 이탈 확률 {proba*100:.2f}%, 예측 결과: {status}")
        
        final_answer = "\n".join(results)
        if not final_answer:
            final_answer = "예측할 고객 데이터가 없습니다."
        
        print("예측 완료.")
        return {"final_answer": final_answer}

    except FileNotFoundError as e: # 구체적인 파일명을 포함하도록 수정
        error_msg = f"오류: 파일 관련 문제 - {e}. 경로를 확인하세요."
        print(error_msg)
        return {"final_answer": error_msg, "error_message": error_msg}
    except pd.errors.EmptyDataError:
        error_msg = f"오류: CSV 파일이 비어있습니다 - {user_query}"
        print(error_msg)
        return {"final_answer": error_msg, "error_message": error_msg}
    except KeyError as e: # 컬럼 관련 오류 (좀 더 일반적인 메시지로)
        error_msg = f"오류: 데이터 처리 중 컬럼 관련 문제가 발생했습니다. 누락/불일치 컬럼 가능성: {e}. CSV 파일, 컬럼 정의(EXPECTED_FEATURE_ORDER, CATEGORICAL_COLS)를 확인하세요."
        print(error_msg)
        return {"final_answer": error_msg, "error_message": error_msg}
    except Exception as e:
        error_msg = f"예측 중 알 수 없는 오류 발생: {e}"
        import traceback
        traceback.print_exc()
        return {"final_answer": f"오류: 고객 이탈 예측 중 문제가 발생했습니다. ({e})", "error_message": error_msg}

def route_sql_output(state: AgentState) -> Literal["create_visualization_node", "summarize_sql_result_node"]:
    choice = state.sql_output_choice
    # If execute_sql_node itself resulted in an error (indicated by error_message and no sql_result)
    # both visualization and summarization nodes have internal logic to handle this.
    # The choice made by the supervisor should still be respected if possible.
    if state.error_message and not state.sql_result:
        print(f"Error detected before routing SQL output: {state.error_message}. Proceeding with choice: {choice}")

    if choice == "visualize":
        print("Routing to create_visualization_node based on sql_output_choice.")
        return "create_visualization_node"
    elif choice == "summarize":
        print("Routing to summarize_sql_result_node based on sql_output_choice.")
        return "summarize_sql_result_node"
    else:
        # Fallback if sql_output_choice is somehow not set for a db_query path
        print(f"Warning: sql_output_choice is '{choice}'. Defaulting to summarize_sql_result_node.")
        return "summarize_sql_result_node"

def route_query(state: AgentState) -> Literal["generate_sql_node", "category_predict_node", "general_question_node"]:
    query_type = state.query_type
    print(f"Routing based on query_type: {query_type}")
    if query_type == "db_query":
        return "generate_sql_node"
    elif query_type == "category_predict_query":
        return "category_predict_node"
    elif query_type == "general_query":
        return "general_question_node"
    else:
        # This case should ideally not be reached if supervisor is strict
        print(f"Warning: Unknown query_type '{query_type}', defaulting to general_question_node.")
        return "general_question_node"

# --- Graph Definition ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("generate_sql_node", generate_sql_node)
workflow.add_node("execute_sql_node", execute_sql_node)
workflow.add_node("create_visualization_node", create_visualization_node)
workflow.add_node("summarize_sql_result_node", summarize_sql_result_node)
workflow.add_node("general_question_node", general_question_node)
workflow.add_node("category_predict_node", category_predict_node)

# Set entry point
workflow.set_entry_point("supervisor")

# Conditional edges from supervisor
workflow.add_conditional_edges(
    "supervisor",
    route_query,
    {
        "generate_sql_node": "generate_sql_node",
        "category_predict_node": "category_predict_node",
        "general_question_node": "general_question_node"
    }
)

# Edges for DB query flow (now common for 3 branches)
workflow.add_edge("generate_sql_node", "execute_sql_node")
# After execute_sql_node, route to either visualization or summarization
workflow.add_conditional_edges(
    "execute_sql_node",
    route_sql_output,
    {
        "create_visualization_node": "create_visualization_node",
        "summarize_sql_result_node": "summarize_sql_result_node"
    }
)
workflow.add_edge("create_visualization_node", END)
workflow.add_edge("summarize_sql_result_node", END)

# Edges from placeholder nodes to the SQL generation flow
workflow.add_edge("category_predict_node", END)

# Edge for general question
workflow.add_edge("general_question_node", END)

# Compile the graph
app = workflow.compile()
graph = app # For langgraph dev compatibility

# To make it runnable with langgraph dev, ensure it's assigned to 'graph'
# Example of how to run (for testing locally):
# async def main():
#     inputs = {"user_query": "지난 달 사용자 수는 몇 명인가요?"}
#     # For testing, you can invoke the graph like this:
#     # inputs = {"messages": [HumanMessage(content="우리 회사 직원들 중 가장 연봉이 높은 상위 3명은 누구인가요?")]}
#     # async for event in app.astream_events(inputs, version="v1"):
#     #     kind = event["event"]
#     #     if kind == "on_chat_model_stream":
#     #         content = event["data"]["chunk"].content
#     #         if content:
#     #             print(content, end="")
#     #     elif kind == "on_tool_end":
#     #         print(f"\nTool Output: {event['data']['output']}")
#     #     # print(f"\n--- Event: {kind} ---")
#     #     # print(event["data"])
# if __name__ == "__main__":
#     import asyncio
#     async def main_test():
#         app_test = await main()
#         inputs = {"input": "우리 회사 테이블 목록 좀 보여줘"}
#     # inputs = {"input": "오늘 날씨 어때?"}
#     async for event in app_test.astream_events(inputs, version="v1"):
#             kind = event["event"]
#             if kind == "on_chat_model_stream":
#                 content = event["data"]["chunk"].content
#                 if content:
#                     print(content, end="")
#                 print(f"\nTool Output: {event['data']['output']}")
#             elif kind == "on_chain_end" and event["name"] == "AgentGraph": # Check for the graph's end
#                 print("\n--- Final State ---")
#                 print(event["data"].get("output")) # Access final output from the event

#     asyncio.run(main_test())
