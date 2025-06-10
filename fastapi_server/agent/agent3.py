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
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage # Added SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder # Added ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig # Added missing import
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import logging # Added logging
from datetime import datetime
import re # Added import for regex
import json # Added for direct OpenAI call
from openai import OpenAI # Added for direct OpenAI call
from fastapi_server.agent.prompt import (
    supervisor_chat_prompt_agent3,
    sql_generation_chat_prompt_agent3,
    sql_result_summary_chat_prompt_agent3,
    general_chat_prompt_agent3
)

# Setup logger for agent3
logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
# Basic logging configuration if not configured elsewhere (e.g., in a main app setup)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
        
    def __post_init__(self):
        # Extract user query from messages if coming from supervisor
        if not self.user_query and self.messages:
            # Extract user input from the last human message
            user_messages = [msg for msg in self.messages if isinstance(msg, HumanMessage)]
            if user_messages:
                self.user_query = user_messages[-1].content
    
    def dict(self):
        """Return dict representation to ensure compatibility with supervisor state"""
        result = super().dict()
        # When returning to supervisor, ensure final answer is properly formatted as a new message
        if self.final_answer and self.messages is not None:
            result["messages"] = self.messages + [AIMessage(content=self.final_answer)]
        return result

# --- LLM and Prompts Setup ---
# Ensure OPENAI_API_KEY is set in your environment or passed via config
llm = ChatOpenAI(temperature=0, model="gpt-4o") # Or your preferred model

# --- Helper function for OpenAI API message format ---
def _lc_messages_to_openai_format(lc_messages: List[BaseMessage]) -> List[Dict[str, str]]:
    openai_messages = []
    for msg in lc_messages:
        if isinstance(msg, HumanMessage):
            openai_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            openai_messages.append({"role": "assistant", "content": msg.content})
        # SystemMessages from state.messages are less common here as the main system prompt is usually separate
        elif isinstance(msg, SystemMessage):
             openai_messages.append({"role": "system", "content": msg.content})
    return openai_messages

# --- Pydantic Models for Structured Output ---
class SupervisorDecision(BaseModel):
    query_type: str = Field(description="The type of the user's question (db_query, category_predict_query, or general_query)")

class SQLGenerationOutput(BaseModel):
    sql_query: str = Field(description="The generated SQL query. This field MUST contain ONLY the SQL query string, without any surrounding text, explanations, or markdown formatting like ```sql.")
    sql_output_choice: Literal["summarize", "visualize"] = Field(description="The type of output processing required for the SQL result: 'summarize' or 'visualize'.")

# --- Node Functions ---
async def supervisor_node(state: AgentState, config: Optional[RunnableConfig] = None):
    """Determines the type of query (db, category_predict, or general)."""
    logger.info("--- Entering supervisor_node ---")
    if not state.messages:
        logger.warning("Supervisor_node: No messages in state. Cannot determine query type.")
        # Potentially set a default or error state
        state.query_type = "general_query" # Fallback, or handle error appropriately
        state.error_message = "No input message found."
        return state
        
    logger.debug(f"Supervisor_node: Current messages: {state.messages}")
    
    # The user_query is still useful for logging or if other parts need it, 
    # but the prompt now relies on the full message history.
    if not state.user_query:
        user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
        if user_messages:
            state.user_query = user_messages[-1].content
        else:
            logger.warning("Supervisor_node: No HumanMessage found to extract user_query.")
            # Fallback if no human message, though MessagesPlaceholder handles history
            state.query_type = "general_query"
            state.error_message = "No human message found in history."
            return state

    logger.info(f"Supervisor_node: User query for routing: '{state.user_query}'")
    logger.info(f"Invoking supervisor_chat_prompt_agent3 with messages: {state.messages}")
    chain = supervisor_chat_prompt_agent3 | llm | JsonOutputParser(pydantic_object=SupervisorDecision) # Replaced with direct OpenAI call
    client = OpenAI()
    
    try:
        # Construct System Prompt for OpenAI API
        # supervisor_chat_prompt.messages[0] is the SystemMessage
        system_prompt_content = supervisor_chat_prompt_agent3.messages[0].content

        openai_api_messages = [{"role": "system", "content": system_prompt_content}]
        openai_api_messages.extend(_lc_messages_to_openai_format(state.messages))

        logger.debug(f"Supervisor_node: Sending to OpenAI API: {openai_api_messages}")

        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o", # Ensure this matches the intended model
            messages=openai_api_messages,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        raw_response_content = completion.choices[0].message.content
        logger.debug(f"Supervisor_node: Raw OpenAI response: {raw_response_content}")
        response_data = json.loads(raw_response_content)
        response = SupervisorDecision(**response_data)
        logger.info(f"Supervisor_node: LLM decision: {response.query_type}")
        state.query_type = response.query_type
        state.error_message = None # Clear previous errors
    except Exception as e:
        logger.error(f"Supervisor_node: Error invoking LLM or parsing output: {e}", exc_info=True)
        state.query_type = "general_query"  # Fallback on error
        state.final_answer = f"죄송합니다, 요청을 이해하는 중 오류가 발생했습니다: {e}"
        state.error_message = str(e)
    
    logger.info(f"--- Exiting supervisor_node with query_type: {state.query_type} ---")
    return state

async def generate_sql_node(state: AgentState, config: Optional[RunnableConfig] = None):
    logger.info("--- Entering generate_sql_node ---")
    if not state.messages:
        logger.error("generate_sql_node: No messages in state. Cannot generate SQL.")
        state.error_message = "No input message found for SQL generation."
        state.final_answer = "SQL 쿼리를 생성하기 위한 입력 메시지가 없습니다."
        return state
        
    logger.debug(f"generate_sql_node: User query from state: {state.user_query}") # user_query might be stale, messages is source of truth
    logger.debug(f"generate_sql_node: Full messages for prompt: {state.messages}")

    logger.info(f"Invoking sql_generation_chat_prompt_agent3 with messages: {state.messages}")
    chain = sql_generation_chat_prompt_agent3 | llm | JsonOutputParser(pydantic_object=SQLGenerationOutput) # Replaced with direct OpenAI call
    client = OpenAI()

    try:
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        # sql_generation_chat_prompt.messages[0] is the SystemMessage
        system_prompt_template = sql_generation_chat_prompt_agent3.messages[0].content
        system_prompt_content_sql = system_prompt_template.replace("{{current_date}}", current_date_str)

        openai_api_messages_sql = [{"role": "system", "content": system_prompt_content_sql}]
        openai_api_messages_sql.extend(_lc_messages_to_openai_format(state.messages))
        
        logger.debug(f"generate_sql_node: Sending to OpenAI API: {openai_api_messages_sql}")

        completion_sql = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o", # Ensure this matches the intended model
            messages=openai_api_messages_sql,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        raw_response_content_sql = completion_sql.choices[0].message.content
        logger.debug(f"generate_sql_node: Raw OpenAI response: {raw_response_content_sql}")
        response_data_sql = json.loads(raw_response_content_sql)
        response = SQLGenerationOutput(**response_data_sql)
        
        # Clean the SQL query to remove any prepended JSON-like structures
        raw_sql_query = response.sql_query
        logger.debug(f"Raw SQL query from LLM: {raw_sql_query}")
        
        # Regex to find the actual SQL query, robustly handling optional prepended JSON objects.
        # It looks for common SQL keywords after any number of {...} blocks.
        # This regex assumes SQL queries start with standard keywords like SELECT, INSERT, UPDATE, DELETE, WITH, CREATE, ALTER, DROP.
        # It captures from the SQL keyword to the end of the string.
        match = re.search(r'^(?:\{.*?\})*?(SELECT\s.*|INSERT\s.*|UPDATE\s.*|DELETE\s.*|WITH\s.*|CREATE\s.*|ALTER\s.*|DROP\s.*)$', raw_sql_query.strip(), re.IGNORECASE | re.DOTALL)
        
        if match:
            cleaned_sql_query = match.group(1).strip() # Get the captured SQL part
            if not cleaned_sql_query.endswith(';'):
                cleaned_sql_query += ';'
            logger.info(f"Cleaned SQL query: {cleaned_sql_query}")
            state.sql_query = cleaned_sql_query
        else:
            # If regex doesn't match, log a warning and use the raw query, 
            # or handle as an error if it's critical that it's clean.
            logger.warning(f"Could not extract clean SQL from: {raw_sql_query}. Using raw query.")
            state.sql_query = raw_sql_query # Fallback to raw query
            # Ensure it ends with a semicolon if it looks like SQL
            if state.sql_query and isinstance(state.sql_query, str) and not state.sql_query.strip().endswith(';') and any(keyword in state.sql_query.upper() for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
                 state.sql_query = state.sql_query.strip() + ';'

        state.sql_output_choice = response.sql_output_choice
        logger.info(f"Final state.sql_query: {state.sql_query}, Output choice: {state.sql_output_choice}")
        state.error_message = None # Clear previous errors
    except Exception as e:
        logger.error(f"Error generating SQL: {e}", exc_info=True)
        state.error_message = f"Error generating SQL: {e}"
        state.final_answer = f"죄송합니다, SQL 쿼리를 생성하는 중 오류가 발생했습니다: {e}"
        state.sql_query = None
    logger.info("--- Exiting generate_sql_node ---")
    return state

async def execute_sql_node(state: AgentState, config: Optional[RunnableConfig] = None):
    logger.info("--- Entering execute_sql_node ---")
    if not state.sql_query:
        logger.error("No SQL query to execute.")
        state.error_message = "No SQL query to execute."
        state.final_answer = "실행할 SQL 쿼리가 없습니다."
        # If there's no SQL query, we can't proceed to summarize or visualize.
        # We should indicate an error and perhaps end or route to a fallback.
        # For now, setting final_answer and error_message. The graph might need a dedicated error handling path.
        return state

    logger.info(f"Executing SQL: {state.sql_query}")
    
    # Determine the base directory for .env files
    # Assuming this script is in 'fastapi_server/agent/agent3.py'
    # Adjust if your structure is different
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir_analysis_env = os.path.join(current_script_dir, '..', '..', 'analysis_env') # Path to analysis_env folder
    base_dir_my_state_env = os.path.join(current_script_dir, '..', '..', 'my_state_env') # Path to my_state_env folder

    try:
        # Run the synchronous DB operation in a separate thread
        result_df = await asyncio.to_thread(
            _execute_sql_sync, 
            state.sql_query,
            base_dir_analysis_env,
            base_dir_my_state_env
        )
        
        if isinstance(result_df, pd.DataFrame):
            state.sql_result = result_df.to_dict(orient='records') # Convert DataFrame to list of dicts
            logger.info(f"SQL Result:\n{state.sql_result}")
        elif isinstance(result_df, dict) and 'sql_result' in result_df:
            # 결과가 딕셔너리 형태로 반환된 경우 (sql_result 키가 있으면 정상적인 결과로 간주)
            state.sql_result = result_df['sql_result']
            state.error_message = result_df.get('error_message', None)
            logger.info(f"SQL Result from dict: {state.sql_result}")
        else: # Error string from _execute_sql_sync
            state.sql_result = str(result_df) if result_df is not None else None
            state.error_message = f"Error executing SQL: {result_df}"
            state.final_answer = f"SQL 실행 중 오류: {result_df}"
            logger.error(f"Error executing SQL (returned as string): {result_df}")
        
        # SQL 결과가 존재하면 오류 메시지는 None으로 설정
        if state.sql_result and not state.error_message:
            state.error_message = None

    except Exception as e:
        logger.error(f"Exception executing SQL: {e}", exc_info=True)
        state.error_message = f"Exception executing SQL: {e}"
        state.sql_result = None
        state.final_answer = f"죄송합니다, SQL 쿼리를 실행하는 중 예외가 발생했습니다: {e}"
    logger.info("--- Exiting execute_sql_node ---")
    return state

async def summarize_sql_result_node(state: AgentState, config: Optional[RunnableConfig] = None):
    logger.info("--- Entering summarize_sql_result_node ---")
    
    # 디버깅을 위해 상태 출력
    logger.info(f"State before summarize_sql_result_node: error_message={state.error_message}, sql_result type={type(state.sql_result)}, sql_query={state.sql_query}")
    
    # 결과가 None인 경우
    if not state.sql_result:
        logger.warning("SQL result is None or empty.")
        if state.error_message:
            state.final_answer = f"SQL 실행 중 오류가 발생하여 결과를 요약할 수 없습니다: {state.error_message}"
        else:
            state.final_answer = "요약할 SQL 실행 결과가 없습니다."
        return state
    
    # SQL 결과가 문자열인지 확인하고, 문자열이 아니면 문자열로 변환
    if not isinstance(state.sql_result, str):
        logger.info(f"Converting SQL result from {type(state.sql_result)} to string")
        state.sql_result = str(state.sql_result)
    
    # SQL 결과가 비어있거나 너무 짧은지 확인
    if len(state.sql_result.strip()) < 5:
        logger.warning(f"SQL result is suspiciously short: '{state.sql_result}'")
        state.final_answer = "SQL 쿼리 결과가 비어있거나 처리할 수 없는 형식입니다."
        return state
        
    # 데이터프레임 출력 형식이 제대로 되었는지 확인
    if "count" in state.sql_result and any(c.isdigit() for c in state.sql_result):
        logger.info("SQL result contains 'count' and numbers, which looks like a valid result")
    
    logger.info(f"SQL Result for summary (processed): {state.sql_result[:200]}...")
    logger.debug(f"Summarizing SQL result for user query: {state.user_query}")
    logger.debug(f"SQL Query for summary: {state.sql_query}")

    try:
        # 상세 로깅 추가
        logger.info(f"Preparing to call LLM with SQL Query: {state.sql_query}")
        logger.info(f"SQL Result first 100 chars: {state.sql_result[:100]}")
        
        # 사용자 마지막 메시지 찾기
        user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
        last_user_message = user_messages[-1].content if user_messages else "SQL 쿼리 결과를 요약해주세요."
        logger.info(f"Last user message: {last_user_message[:100]}")
        
        # 체인 구성 및 호출
        # SQL 쿼리와 결과를 명시적으로 포함하는 사용자 메시지 생성
        explicit_sql_message = HumanMessage(
            content=f"SQL 쿼리: {state.sql_query}\n\nSQL 결과:\n{state.sql_result}\n\n이 데이터를 바탕으로 질문에 답변해주세요."
        )
        
        # 기존 메시지 복사 및 SQL 메시지 추가
        messages_with_sql = state.messages.copy()
        messages_with_sql.append(explicit_sql_message)
        
        # 체인 구성 및 호출
        chain = sql_result_summary_chat_prompt_agent3 | llm
        response = await chain.ainvoke(
            {
                "messages": messages_with_sql, 
                "sql_query": str(state.sql_query), 
                "sql_result": state.sql_result
            },
            config=config
        )
        
        # 응답 확인 및 처리
        state.final_answer = response.content
        if not state.final_answer or len(state.final_answer.strip()) < 10:
            logger.warning(f"LLM returned empty or very short response: '{state.final_answer}'")
            state.final_answer = f"SQL 쿼리 '{state.sql_query}'의 결과는 {state.sql_result}입니다."
            
        logger.info(f"Generated summary: {state.final_answer}")
        state.error_message = None # Clear previous errors
    except Exception as e:
        logger.error(f"Error summarizing SQL result: {e}", exc_info=True)
        state.error_message = f"Error summarizing SQL result: {e}"
        state.final_answer = f"죄송합니다, SQL 결과를 요약하는 중 오류가 발생했습니다: {e}"
    logger.info("--- Exiting summarize_sql_result_node ---")
    return state

async def general_question_node(state: AgentState, config: Optional[RunnableConfig] = None):
    logger.info("--- Entering general_question_node ---")
    if not state.messages:
        logger.error("general_question_node: No messages in state. Cannot generate answer.")
        state.error_message = "No input message found for general question."
        state.final_answer = "질문에 답변하기 위한 입력 메시지가 없습니다."
        return state

    logger.debug(f"Answering general question from user query (from state): {state.user_query}")
    logger.info(f"Invoking general_chat_prompt_agent3 with messages: {state.messages}")
    chain = general_chat_prompt_agent3 | llm
    try:
        # Pass the full message history to the chain
        response = await chain.ainvoke({"messages": state.messages}, config=config)
        state.final_answer = response.content
        logger.info(f"Generated general answer: {state.final_answer}")
        state.error_message = None # Clear previous errors
    except Exception as e:
        logger.error(f"Error answering general question: {e}", exc_info=True)
        state.error_message = f"Error answering general question: {e}"
        state.final_answer = f"죄송합니다, 일반 질문에 답변하는 중 오류가 발생했습니다: {e}"
    logger.info("--- Exiting general_question_node ---")
    return state

async def category_predict_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- CATEGORY PREDICT NODE (Telecom Churn Prediction with Csv File Content) ---")

    # --- 경로 설정 ---
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    MODEL_PATH = os.path.join(base_path, 'churn_predictor_pipeline.pkl')
    CATEGORICAL_COLS_PATH = os.path.join(base_path, 'categorical_cols.pkl')
    LABEL_ENCODERS_PATH = os.path.join(base_path, 'label_encoders.pkl')

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

    csv_data_str: Optional[str] = None

    # 1. state.csv_file_content (LangGraph Studio의 'Csv File Content' 필드) 확인
    if state.csv_file_content:
        print("INFO: Using CSV data from state.csv_file_content.")
        csv_data_str = state.csv_file_content
    # 2. state.user_query (Chat 또는 Messages 입력) 확인
    elif hasattr(state, 'user_query') and state.user_query:
        print(f"INFO: Attempting to use state.user_query for CSV data. Content (first 100 chars): '{state.user_query[:100]}...'")
        # 2a. state.user_query를 파일 경로로 시도
        if os.path.exists(state.user_query):
            try:
                print(f"INFO: state.user_query '{state.user_query}' is an existing path. Reading file.")
                def read_file_sync(path):
                    with open(path, 'r', encoding='utf-8') as f_sync:
                        return f_sync.read()
                csv_data_str = await asyncio.to_thread(read_file_sync, state.user_query)
                if not csv_data_str:
                    print(f"WARNING: File at '{state.user_query}' was empty.")
            except Exception as e:
                print(f"WARNING: Error reading file from state.user_query path '{state.user_query}': {e}. Will attempt to treat as raw content.")
        
        # 2b. state.user_query를 파일 경로로 읽지 못했거나, 경로가 아니었다면 원본 CSV 내용으로 간주
        if csv_data_str is None: # 파일 읽기 실패 또는 경로가 아니었음
            print("INFO: Treating state.user_query as raw CSV content.")
            csv_data_str = state.user_query # pd.read_csv가 이후에 파싱 시도

    # CSV 데이터를 어디에서도 찾지 못한 경우 오류 반환
    if csv_data_str is None:
        error_message_parts = ["❌ 오류: CSV 데이터를 찾을 수 없습니다."]
        checked_sources = ["'Csv File Content' 필드"]
        if hasattr(state, 'user_query'):
            checked_sources.append("'User Query' / 채팅 메시지 (파일 경로 또는 CSV 내용 직접 입력)")
        error_message_parts.append(f"확인한 입력 소스: {', '.join(checked_sources)}.")
        error_message_parts.append("Csv File Content 필드에 직접 CSV 내용을 붙여넣거나, 채팅으로 CSV 파일의 전체 경로 또는 CSV 내용 자체를 입력해주세요.")
        final_answer = "\n".join(error_message_parts)
        current_messages = state.messages # Get current messages
        updated_messages = current_messages + [AIMessage(content=final_answer)] if current_messages else [AIMessage(content=final_answer)]
        return {"messages": updated_messages, "final_answer": final_answer, "error_message": final_answer}

    print(f"INFO: CSV data obtained. Length: {len(csv_data_str)}. Preview (first 200 chars): {csv_data_str[:200]}...")

    try:
        # --- 모델과 전처리 객체 비동기 로드 ---
        pipeline_final = await asyncio.to_thread(joblib.load, MODEL_PATH)
        CATEGORICAL_COLS = await asyncio.to_thread(joblib.load, CATEGORICAL_COLS_PATH)
        label_encoders = await asyncio.to_thread(joblib.load, LABEL_ENCODERS_PATH)

        # --- CSV 문자열 → DataFrame 변환 ---
        if not csv_data_str: # 이중 확인, csv_data_str이 None이나 빈 문자열이면 에러 발생 방지
            final_answer = "❌ 오류: 내부 로직 오류 - CSV 데이터 문자열이 비어있습니다."
            current_messages = state.messages # Get current messages
            updated_messages = current_messages + [AIMessage(content=final_answer)] if current_messages else [AIMessage(content=final_answer)]
            return {"messages": updated_messages, "final_answer": final_answer, "error_message": final_answer}
        # --- BEGIN REVISED CSV DATA CLEANING LOGIC ---
        raw_lines = csv_data_str.strip().splitlines()
        cleaned_lines = []

        if not raw_lines:
            final_answer = "❌ 오류: CSV 데이터 문자열이 비어있습니다."
            current_messages = state.messages
            updated_messages = current_messages + [AIMessage(content=final_answer)] if current_messages else [AIMessage(content=final_answer)]
            return {"messages": updated_messages, "final_answer": final_answer, "error_message": final_answer}

        MIN_COMMAS_THRESHOLD = 1  # 쉼표가 이 개수 이상이면 '강력한' CSV 라인으로 간주

        # '강력한' CSV 라인들의 인덱스를 찾음
        strong_csv_indices = [i for i, line in enumerate(raw_lines) if line.count(',') >= MIN_COMMAS_THRESHOLD]

        if strong_csv_indices:
            # '강력한' 라인들이 존재하면, 이들의 범위를 핵심 CSV 블록으로 간주
            core_block_start_idx = strong_csv_indices[0]
            core_block_end_idx = strong_csv_indices[-1]

            if core_block_start_idx > 0:
                print(f"INFO: CSV 시작 전 {core_block_start_idx}개의 라인을 질문으로 간주하고 제거합니다. 첫번째 제거된 라인: '{raw_lines[0][:100]}...'", flush=True)
            if core_block_end_idx < len(raw_lines) - 1:
                print(f"INFO: CSV 종료 후 {len(raw_lines) - 1 - core_block_end_idx}개의 라인을 질문으로 간주하고 제거합니다. 마지막 제거된 라인: '{raw_lines[-1][:100]}...'", flush=True)

            # 핵심 CSV 블록 추출
            core_block_lines = raw_lines[core_block_start_idx : core_block_end_idx + 1]

            if not core_block_lines: # Should not happen if strong_csv_indices is not empty
                final_answer = "❌ 오류: CSV 핵심 블록 추출에 실패했습니다."
                current_messages = state.messages
                updated_messages = current_messages + [AIMessage(content=final_answer)] if current_messages else [AIMessage(content=final_answer)]
                return {"messages": updated_messages, "final_answer": final_answer, "error_message": final_answer}

            # 핵심 블록의 첫 줄을 헤더로 사용
            header_line = core_block_lines[0]
            cleaned_lines.append(header_line)
            commas_in_header = header_line.count(',') # 헤더는 MIN_COMMAS_THRESHOLD 이상일 것임

            # 핵심 블록 내의 데이터 라인 정제
            for i in range(1, len(core_block_lines)):
                line = core_block_lines[i]
                if line.count(',') >= MIN_COMMAS_THRESHOLD: # '강력한' 데이터 라인
                    cleaned_lines.append(line)
                elif commas_in_header >= MIN_COMMAS_THRESHOLD: # 헤더는 '강했으나', 현재 라인은 '약함' (0개의 쉼표)
                    if not line.strip(): # 의도적으로 비어있는 라인이면 유지
                        cleaned_lines.append(line)
                    else: # 내용이 있는 0쉼표 라인은 블록 내 질문으로 의심하여 필터링
                        print(f"INFO: CSV 블록 내에서 0개의 쉼표를 가진 비어있지 않은 라인을 필터링합니다: '{line[:100]}...'", flush=True)
                else: # 헤더도 '약했고' (이 경우는 strong_csv_indices 로직상 거의 없음) 현재 라인도 '약하면' 일단 포함
                    cleaned_lines.append(line)
        else:
            # '강력한' CSV 라인이 하나도 없음 (모든 라인의 쉼표 < MIN_COMMAS_THRESHOLD, 예: 모두 0개)
            # 이 경우 단일 열 CSV이거나 전체가 질문일 수 있음. 일단 모든 라인을 사용.
            print(f"INFO: 모든 라인의 쉼표 개수가 {MIN_COMMAS_THRESHOLD}개 미만입니다. 단일 열 CSV로 간주하거나 전체가 질문일 수 있습니다. 모든 라인을 사용합니다.", flush=True)
            cleaned_lines = raw_lines

        if not cleaned_lines:
            final_answer = "❌ 오류: CSV 데이터 정제 후 내용이 없습니다."
            current_messages = state.messages
            updated_messages = current_messages + [AIMessage(content=final_answer)] if current_messages else [AIMessage(content=final_answer)]
            return {"messages": updated_messages, "final_answer": final_answer, "error_message": final_answer}
        
        cleaned_csv_data_str = "\n".join(cleaned_lines)
        
        print(f"INFO: 최종 정제된 CSV 데이터 미리보기 (첫 200자): {cleaned_csv_data_str[:200]}...", flush=True)
        input_df = await asyncio.to_thread(pd.read_csv, io.StringIO(cleaned_csv_data_str))
        # --- END REVISED CSV DATA CLEANING LOGIC ---

        if CUSTOMER_ID_COL not in input_df.columns:
            final_answer = f"❌ 오류: '{CUSTOMER_ID_COL}' 컬럼이 CSV에 없습니다."
            current_messages = state.messages # Get current messages
            updated_messages = current_messages + [AIMessage(content=final_answer)] if current_messages else [AIMessage(content=final_answer)]
            return {"messages": updated_messages, "final_answer": final_answer, "error_message": final_answer}

        customer_ids = input_df[CUSTOMER_ID_COL]
        X_predict = input_df.drop(columns=[CUSTOMER_ID_COL], errors='ignore')

        # --- 범주형 컬럼 인코딩 ---
        for col in CATEGORICAL_COLS:
            if col in X_predict.columns:
                if col in label_encoders:
                    le = label_encoders[col]
                    X_predict[col] = X_predict[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    print(f"WARNING: Label encoder for column '{col}' not found. Skipping encoding.")
            else:
                print(f"WARNING: Categorical column '{col}' not found in input CSV. Skipping.")

        # --- 누락된 컬럼 처리 (모델이 기대하는 모든 컬럼이 있는지 확인) ---
        missing_cols = set(EXPECTED_FEATURE_ORDER) - set(X_predict.columns)
        for col in missing_cols:
            print(f"INFO: Adding missing column '{col}' with default value 0.")
            X_predict[col] = 0 # 또는 np.nan 등 적절한 기본값

        # --- 컬럼 순서 정렬 ---
        X_predict = X_predict[EXPECTED_FEATURE_ORDER]

        # --- 예측 수행 ---
        predictions_proba = await asyncio.to_thread(pipeline_final.predict_proba, X_predict)
        predictions = (predictions_proba[:, 1] >= PREDICTION_THRESHOLD).astype(int)

        # --- 결과 생성 ---
        results_df = pd.DataFrame({
            CUSTOMER_ID_COL: customer_ids,
            'Churn Probability': predictions_proba[:, 1],
            'Churn Prediction (Threshold 0.312)': predictions
        })
        results_df['Churn Prediction (Threshold 0.312)'] = results_df['Churn Prediction (Threshold 0.312)'].map({1: 'Yes', 0: 'No'})

        final_answer = "📊 고객 이탈 예측 결과:\n" + results_df.to_string(index=False)
        print(f"Prediction successful. Result preview: {final_answer[:200]}...")
        current_messages = state.messages # Get current messages
        updated_messages = current_messages + [AIMessage(content=final_answer)] if current_messages else [AIMessage(content=final_answer)]
        return {"messages": updated_messages, "final_answer": final_answer, "error_message": None}

    except pd.errors.EmptyDataError:
        error_msg = "❌ 오류: 입력된 CSV 데이터가 비어 있거나 잘못된 형식입니다. CSV 내용을 다시 확인해주세요."
        print(f"ERROR: {error_msg}")
        current_messages = state.messages # Get current messages
        updated_messages = current_messages + [AIMessage(content=error_msg)] if current_messages else [AIMessage(content=error_msg)]
        return {"messages": updated_messages, "final_answer": error_msg, "error_message": error_msg}
    except FileNotFoundError as e:
        error_msg = f"❌ 오류: 모델 또는 전처리 파일을 찾을 수 없습니다. 경로를 확인해주세요. ({e})"
        print(f"ERROR: {error_msg}")
        current_messages = state.messages # Get current messages
        updated_messages = current_messages + [AIMessage(content=error_msg)] if current_messages else [AIMessage(content=error_msg)]
        return {"messages": updated_messages, "final_answer": error_msg, "error_message": error_msg}
    except KeyError as e:
        error_msg = f"❌ 오류: CSV 데이터에 필요한 컬럼이 누락되었거나, 모델 학습 시 사용된 컬럼과 다릅니다. (오류 컬럼: {e}) CSV 파일을 확인해주세요."
        print(f"ERROR: {error_msg}")
        current_messages = state.messages # Get current messages
        updated_messages = current_messages + [AIMessage(content=error_msg)] if current_messages else [AIMessage(content=error_msg)]
        return {"messages": updated_messages, "final_answer": error_msg, "error_message": error_msg}
    except ValueError as e: # Often from pandas if query is malformed for read_sql_query
        error_msg = f"❌ 오류: 데이터 변환 중 값 오류가 발생했습니다. CSV 데이터 타입을 확인해주세요. (오류: {e})"
        print(f"ERROR: {error_msg}")
        current_messages = state.messages # Get current messages
        updated_messages = current_messages + [AIMessage(content=error_msg)] if current_messages else [AIMessage(content=error_msg)]
        return {"messages": updated_messages, "final_answer": error_msg, "error_message": error_msg}
    except Exception as e:
        error_msg = f"❌ 예측 중 알 수 없는 오류 발생: {e}"
        print(f"ERROR: {error_msg}")
        current_messages = state.messages # Get current messages
        updated_messages = current_messages + [AIMessage(content=error_msg)] if current_messages else [AIMessage(content=error_msg)]
        return {"messages": updated_messages, "final_answer": error_msg, "error_message": error_msg}

async def create_visualization_node(state: AgentState, config: Optional[RunnableConfig] = None):
    logger.info("--- Entered create_visualization_node ---")
    
    # Validate sql_result upfront
    if (
        not state.sql_result or 
        not isinstance(state.sql_result, list) or
        (state.sql_result and not all(isinstance(row, dict) for row in state.sql_result))
    ):
        logger.warning(f"SQL result is None, not a list of dicts, or empty. Type: {type(state.sql_result)}")
        state.error_message = "SQL result is None, not a list of dicts, or empty."
        state.final_answer = "시각화할 SQL 결과가 없거나 올바르지 않은 형식입니다."
        state.visualization_output = "```mermaid\ngraph TD\n  A[데이터 없음 또는 형식 오류]\n```"
        return state

    try:
        llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
        # OPENAI_API_KEY는 환경 변수에서 자동으로 로드됩니다.

        # SQL 결과를 JSON 문자열로 변환 (ensure_ascii=False로 한글 처리)
        # 프롬프트에 너무 긴 내용이 들어가지 않도록 길이 제한
        sql_result_str = json.dumps(state.sql_result, indent=2, ensure_ascii=False)
        MAX_PROMPT_SQL_RESULT_LEN = 8000 # LLM 프롬프트 토큰 제한 고려
        
        if len(sql_result_str) > MAX_PROMPT_SQL_RESULT_LEN:
            truncation_note = f"... (내용이 너무 길어 일부가 잘렸습니다. 원본 행 수: {len(state.sql_result)})"
            sql_result_str = sql_result_str[:MAX_PROMPT_SQL_RESULT_LEN - len(truncation_note) - 10] + "\n" + truncation_note
            logger.info(f"Mermaid 생성을 위한 SQL 결과가 프롬프트 제한을 초과하여 일부 잘라냈습니다. 원본 길이: {len(sql_result_str)}")

        prompt_text = f"""
        [사용자 질문]
    {state.user_query}
다음은 SQL 쿼리 결과입니다. 이 데이터를 시각적으로 효과적으로 표현할 수 있는 Mermaid 다이어그램(markdown 코드블록, 시작과 끝을 반드시 ```mermaid ... ```로 감쌀 것)과, 해당 시각화가 무엇을 의미하는지 간결한 한국어 설명(답변)을 함께 만들어주세요.

        - 데이터의 특성(집계, 분포, 관계, 순서 등)에 따라 다음 Mermaid 다이어그램 타입 중에서 가장 적합한 것을 자동으로 선택하여 생성해주세요. 각 타입의 예시와 간단한 설명은 다음과 같습니다. (실제 데이터에 맞게 내용을 채우고, 모든 텍스트는 한국어로 작성해야 합니다):

          - **Pie Chart (pie):** 데이터 항목들의 전체에 대한 비율을 원형으로 표현합니다. 각 조각의 크기가 해당 항목의 비율을 나타냅니다.
            예시:
            ```mermaid
            pie title 월별 지출 비율
                "식비" : 40
                "교통비" : 20
                "공과금" : 15
                "여가" : 25
            ```

          - **Line Chart (xyChart 사용):** 시간 경과나 순서에 따른 데이터 값의 변화 추세를 선으로 연결하여 보여줍니다. (Mermaid 최신 버전에서는 xyChart 사용 권장)
            예시:
            ```mermaid
            xychart-beta
                title "월별 웹사이트 방문자 수"
                x-axis "월" ["1월", "2월", "3월", "4월", "5월"]
                y-axis "방문자 수" 0 --> 1000
                line [300, 450, 600, 500, 750]
            ```

          - **Bar Chart (xyChart 사용):** 데이터 항목들의 값을 막대로 표현하여 비교합니다. `xyChart`를 사용하여 바 차트도 그릴 수 있습니다.
            예시:
            ```mermaid
            xychart-beta
                title "분기별 제품 판매량"
                x-axis "분기" ["1분기", "2분기", "3분기", "4분기"]
                y-axis "판매량 (단위: 개)" 0 --> 200
                bar [80, 120, 150, 100]
            ```

          - **Graph TD (Flowchart):** 작업, 프로세스, 시스템 구성 요소 간의 순서, 흐름, 관계를 노드와 화살표로 표현합니다.
            예시:
            ```mermaid
            graph TD
                A[데이터 입력] --> B(데이터 처리);
                B --> C{{조건 분기}};
                C -- 조건1 --> D[결과 A];
                C -- 조건2 --> E[결과 B];
            ```

          - **C4Context Diagram:** 소프트웨어 시스템과 그 주변 환경(사용자, 외부 시스템) 간의 상호작용을 높은 수준에서 개략적으로 보여주는 아키텍처 다이어그램입니다.
            예시:
            ```mermaid
            C4Context
              title 온라인 쇼핑몰 시스템 컨텍스트
              Enterprise_Boundary(b0, "회사 시스템 경계") {{
                Person(customer, "고객", "상품을 구매하는 사용자")
                System(shoppingMall, "온라인 쇼핑몰", "상품 검색, 주문, 결제 기능 제공")

                System_Ext(paymentGateway, "결제 시스템", "외부 PG사 연동")
                System_Ext(deliverySystem, "배송 시스템", "외부 배송 업체 연동")
              }}

              Rel(customer, shoppingMall, "상품 주문")
              Rel(shoppingMall, paymentGateway, "결제 요청")
              Rel(shoppingMall, deliverySystem, "배송 요청")
            ```

          - **Class Diagram:** 시스템을 구성하는 클래스들의 속성, 오퍼레이션(메서드) 및 클래스 간의 관계(상속, 연관, 의존 등)를 시각화합니다.
            예시:
            ```mermaid
            classDiagram
              주문 <|-- 일반주문
              주문 <|-- 특별주문
              주문 : +String 주문번호
              주문 : +Date 주문일자
              주문 : +calculateTotalPrice()
              class 일반주문{{
                  +String 배송지
                  +calculateShippingCost()
              }}
              class 고객{{
                  +String 이름
                  +List<주문> 주문내역
                  +placeOrder(주문)
              }}
              고객 "1" -- "*" 주문 : 주문한다
            ```

          - **Gantt Chart:** 프로젝트의 작업(태스크) 목록, 각 작업의 시작일, 종료일, 기간 등을 시간 축에 막대로 표시하여 일정 관리에 사용됩니다.
            예시:
            ```mermaid
            gantt
                dateFormat  YYYY-MM-DD
                title       프로젝트 A 진행 현황
                excludes    weekends

                section 기획 단계
                요구사항 분석    :task1, 2024-01-01, 7d
                화면 설계       :task2, after task1, 5d

                section 개발 단계
                기능 개발      :task3, after task2, 20d
                테스트         :task4, after task3, 10d
            ```
        - 마크다운 코드블록 외의 불필요한 설명은 절대 포함하지 마세요.
        - 노드·라벨 이름에 특수문자(", \\n 등)는 Mermaid 문법에 맞게 적절히 이스케이프하거나 제거하세요.
        - 모든 다이어그램 내 텍스트(설명, 노드, 라벨 등)는 반드시 한국어로 작성하세요.
        - 마지막에 "설명:" 이라는 제목 아래, 시각화가 보여주는 핵심 인사이트를 한두 문장으로 한국어로 설명하세요.

SQL 결과:
```json
{sql_result_str}
```

Mermaid 다이어그램과 설명:
"""
        messages = [HumanMessage(content=prompt_text)]
        
        logger.info(f"LLM에게 Mermaid 다이어그램 생성 요청 중... SQL 결과 (일부): {sql_result_str[:200]}...")
        
        response = await llm.ainvoke(messages)
        llm_output = response.content.strip()

        logger.info(f"LLM이 생성한 Mermaid 내용 (일부): {llm_output[:300]}...")

        # LLM 출력이 올바른 Mermaid 마크다운 블록인지 확인
        if llm_output.startswith("```mermaid") and llm_output.endswith("```"):
            state.visualization_output = llm_output
            state.final_answer = llm_output
            state.error_message = None
        else:
            logger.error(f"LLM이 유효한 Mermaid 마크다운 블록을 반환하지 않았습니다. 출력: {llm_output}")
            state.error_message = "LLM으로부터 유효한 Mermaid 다이어그램을 받지 못했습니다."
            state.final_answer = "Mermaid 다이어그램 생성에 실패했습니다. LLM이 올바른 형식을 반환하지 않았습니다."
            state.visualization_output = "```mermaid\ngraph TD\n  A[LLM 생성 오류: 잘못된 형식]\n```"
            
    except Exception as e:
        logger.error(f"LLM을 통해 Mermaid 다이어그램 생성 중 오류: {e}", exc_info=True)
        state.error_message = f"LLM Mermaid 다이어그램 생성 중 오류: {str(e)[:100]}"
        state.final_answer = f"LLM Mermaid 다이어그램 생성 중 오류가 발생했습니다: {str(e)[:100]}"
        state.visualization_output = "```mermaid\ngraph TD\n  A[오류 발생: 다이어그램 생성 실패]\n```"
        return state # 예외 발생 시에도 state 반환 보장

        
    logger.info(f"create_visualization_node state after processing: visualization_output length: {len(state.visualization_output or '')}, final_answer length: {len(state.final_answer or '')}")
    return state

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
        # Instead of converting to string, we return the DataFrame directly.
        # execute_sql_node will handle converting this DataFrame to a list of dicts.
        print(f"_execute_sql_sync: SQL query executed successfully. DataFrame shape: {df.shape}, Columns: {df.columns.tolist()}")
        return df  # Return the DataFrame object directly
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
