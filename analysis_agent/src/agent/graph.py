"""LangGraph graph for analysis_agent with supervisor logic.

Handles DB queries and general questions based on node/edge routing.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, List, Annotated, Literal, TypedDict
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

# --- Configuration (Optional - can be used to pass API keys, model names, etc.) ---
class Configuration(TypedDict, total=False):
    openai_api_key: Optional[str]
    db_env_path: Optional[str] # Path to .env file for DB credentials

# --- State Definition ---
class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)
    user_query: Optional[str] = None
    query_type: Optional[Literal["db_query", "general_query"]] = None
    sql_query: Optional[str] = None
    sql_result: Optional[Any] = None
    final_answer: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True # For Annotated and operator.add with BaseMessage

# --- LLM and Prompts Setup ---
# Ensure OPENAI_API_KEY is set in your environment or passed via config
llm = ChatOpenAI(temperature=0, model="gpt-4o") # Or your preferred model

supervisor_prompt = PromptTemplate.from_template(
    "사용자의 질문을 분석하여 이 질문이 데이터베이스 조회를 통해 답변해야 하는 질문인지, 아니면 일반적인 지식으로 답변할 수 있는 질문인지 결정해주세요. "
    "'query_type' 필드에 'db_query' 또는 'general_query' 중 하나를 포함하는 JSON 객체로 답변해주세요.\n\n"
    "사용자 질문: {user_query}\n\n"
    "JSON 응답:"
)

sql_generation_prompt = PromptTemplate.from_template(
    "다음 사용자 질문을 PostgreSQL 문법의 SQL 쿼리로 변환해주세요. "
    "사용자의 의도를 정확히 파악하고, 필요한 테이블과 컬럼을 추론하여 쿼리를 작성하세요. "
    "만약 테이블이나 컬럼 정보가 부족하다면, 가장 가능성이 높은 일반적인 이름을 사용하세요. "
    "예시: '지난 달 매출액은 얼마인가요?' -> 'SELECT SUM(amount) FROM sales WHERE sale_date >= date_trunc(\'month\', current_date - interval \'1 month\') AND sale_date < date_trunc(\'month\', current_date);'\n\n"
    "사용자 질문: {user_query}\n\n"
    "SQL 쿼리:"
)

summarization_prompt = PromptTemplate.from_template(
    "다음 SQL 쿼리 실행 결과를 바탕으로 사용자의 원래 질문에 대한 답변을 자연스럽게 요약해주세요.\n\n"
    "사용자 질문: {user_query}\n"
    "SQL 쿼리: {sql_query}\n"
    "SQL 결과:\n{sql_result}\n\n"
    "요약 답변:"
)

general_answer_prompt = PromptTemplate.from_template(
    "다음 사용자 질문에 대해 답변해주세요.\n\n"
    "사용자 질문: {user_query}\n\n"
    "답변:"
)

# --- Pydantic Models for Structured Output ---
class SupervisorDecision(BaseModel):
    query_type: str = Field(description="사용자 질문의 유형 (db_query 또는 general_query)") # Changed from Literal to str

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
        decision_result = await supervisor_chain.ainvoke({"user_query": user_query_to_process}, config=config)
        query_type = decision_result.query_type
        print(f"Supervisor - LLM decision: query_type='{query_type}' for query: '{user_query_to_process[:100]}...'")
        
        return {
            "user_query": user_query_to_process,
            "query_type": query_type,
            "messages": current_messages # Pass through existing messages
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
    user_query = state.user_query # Get user_query from the state field set by supervisor
    
    if not user_query:
        print("Error in generate_sql_node: state.user_query is None or empty.")
        return {"error_message": "No user query found for SQL generation (state.user_query is missing).", "sql_query": ""}
    print(f"Generating SQL for: {user_query}")
    sql_generation_chain = sql_generation_prompt | llm
    response = await sql_generation_chain.ainvoke({"user_query": user_query}, config=config)
    sql_query = response.content.strip()
    if sql_query.startswith("```sql"):
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    print(f"Generated SQL: {sql_query}")
    return {"sql_query": sql_query}

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

# --- Conditional Edges Logic ---
def should_route_to_db_or_general(state: AgentState) -> Literal["generate_sql", "general_question"]:
    query_type = state.query_type
    user_query_snippet = state.user_query[:50] + "..." if hasattr(state, 'user_query') and state.user_query and len(state.user_query) > 50 else (state.user_query if hasattr(state, 'user_query') and state.user_query else "[No user_query in state or empty]")
    print(f"--- ROUTING DECISION --- State Query Type: {query_type}, User Query Snippet: '{user_query_snippet}'")

    # Check if supervisor explicitly set an error and defaulted
    if hasattr(state, 'error_message') and state.error_message and "Supervisor LLM decision processing error" in state.error_message:
        print(f"Routing to general_question due to supervisor error: {state.error_message}")
        return "general_question"
    
    if query_type == "db_query":
        print("Routing to: generate_sql")
        return "generate_sql"
    
    # Fallback for general_query, None query_type, or other unexpected cases
    print(f"Routing to: general_question (Actual query_type: {query_type})")
    return "general_question"

# --- Graph Definition ---
workflow = StateGraph(AgentState, config_schema=Configuration)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("summarize_sql_result", summarize_sql_result_node)
workflow.add_node("general_question", general_question_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    should_route_to_db_or_general,
    {
        "generate_sql": "generate_sql",
        "general_question": "general_question",
    },
)

workflow.add_edge("generate_sql", "execute_sql")
workflow.add_edge("execute_sql", "summarize_sql_result")
workflow.add_edge("summarize_sql_result", END)
workflow.add_edge("general_question", END)

# Compile the graph
graph = workflow.compile(checkpointer=None) # Add checkpointer if persistence is needed

# To make it runnable with langgraph dev, ensure it's assigned to 'graph'
# Example of how to run (for testing locally):
# async def main():
#     inputs = {"user_query": "지난 달 사용자 수는 몇 명인가요?"}
#     # For testing, you can invoke the graph like this:
    # inputs = {"messages": [HumanMessage(content="우리 회사 직원들 중 가장 연봉이 높은 상위 3명은 누구인가요?")]}
    # async for event in app.astream_events(inputs, version="v1"):
    #     kind = event["event"]
    #     if kind == "on_chat_model_stream":
    #         content = event["data"]["chunk"].content
    #         if content:
    #             print(content, end="")
    #     elif kind == "on_tool_end":
    #         print(f"\nTool Output: {event['data']['output']}")
    #     # print(f"\n--- Event: {kind} ---")
    #     # print(event["data"])
# if __name__ == "__main__":
#     import asyncio
#     async def main_test():
#         app_test = await main()
#         inputs = {"input": "우리 회사 테이블 목록 좀 보여줘"}
    # # inputs = {"input": "오늘 날씨 어때?"}
    # async for event in app_test.astream_events(inputs, version="v1"):
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
