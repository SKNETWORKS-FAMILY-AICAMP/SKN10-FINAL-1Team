"""LangGraph graph for analysis_agent with supervisor logic.

Handles DB queries and general questions based on node/edge routing.
"""

from __future__ import annotations

import os
from typing import TypedDict, Optional, Dict, Any, Literal, List, Annotated
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
class AgentState(TypedDict):
    input: Optional[str] = None  # Raw input string from the user
    messages: Annotated[List[BaseMessage], operator.add] # Conversation history
    query_type: Optional[str] = None  # Changed from Literal to str
    sql_query: Optional[str] = None
    sql_result: Optional[str] = None
    final_answer: Optional[str] = None
    error_message: Optional[str] = None
    # Configuration can be added to state if needed per invocation
    # config: Optional[Configuration] = None 

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
    # reasoning: Optional[str] = Field(default=None, description="질문 유형 판단 근거 (선택 사항)")

# --- Node Functions ---
async def supervisor_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- SUPERVISOR NODE ---")
    
    new_input_str = state.get("input")
    messages_to_add_to_history = []
    user_query_for_llm = ""

    if new_input_str:
        # New raw input from the user for this turn
        human_message = HumanMessage(content=new_input_str)
        messages_to_add_to_history = [human_message]
        user_query_for_llm = new_input_str
    elif state.get("messages") and isinstance(state["messages"][-1], HumanMessage):
        # No new raw input, but there's existing history ending with a HumanMessage
        # This could happen if the graph is re-entered or in a multi-step sequence
        # where 'input' was cleared, and we are processing the last human message.
        user_query_for_llm = state["messages"][-1].content
    else:
        # This is the error condition: no new input string, and no valid HumanMessage in history.
        raise ValueError("Supervisor: No input string provided and no prior HumanMessage found in history to process.")

    # LLM call for supervision using user_query_for_llm
    structured_llm = llm.with_structured_output(schema=SupervisorDecision)
    # supervisor_chain = supervisor_prompt | llm | JsonOutputParser()
    # response = await supervisor_chain.ainvoke({"user_query": user_query}, config=config)
    response_model_instance = await structured_llm.ainvoke(supervisor_prompt.format(user_query=user_query_for_llm), config=config)
    print(f"Supervisor decision: {response_model_instance}")

    output_dict = {
        "query_type": response_model_instance.query_type,
        "input": None  # Clear the input field after processing
    }
    if messages_to_add_to_history:
        output_dict["messages"] = messages_to_add_to_history # This will be added to state['messages'] by operator.add
        
    return output_dict

async def generate_sql_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- GENERATE SQL NODE ---")
    if not state['messages'] or not isinstance(state['messages'][-1], HumanMessage):
        # This case should ideally be handled by the supervisor or routing logic
        # Or ensure that generate_sql_node is only called when appropriate
        return {"error_message": "No user query found for SQL generation.", "sql_query": ""}
    user_query = state['messages'][-1].content
    sql_generation_chain = sql_generation_prompt | llm
    response = await sql_generation_chain.ainvoke({"user_query": user_query}, config=config)
    sql_query = response.content.strip()
    print(f"Generated SQL: {sql_query}")
    return {"sql_query": sql_query}

async def execute_sql_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- EXECUTE SQL NODE ---")
    sql_query = state.get("sql_query")
    if not sql_query:
        return {"error_message": "SQL 쿼리가 생성되지 않았습니다.", "sql_result": ""}

    # Determine .env path
    effective_config = config.get("configurable", {}) if config else {}
    db_env_path_from_config = effective_config.get("db_env_path")
    
    dotenv_path_to_try = None

    if db_env_path_from_config and os.path.exists(db_env_path_from_config):
        dotenv_path_to_try = db_env_path_from_config
        print(f"설정에서 .env 경로 사용: {dotenv_path_to_try}")
    else:
        # 1. Try analysis_agent/.env (relative to this graph.py)
        # graph.py is in analysis_agent/src/agent/
        # analysis_agent/.env is at ../../.env from graph.py
        analysis_agent_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
        if os.path.exists(analysis_agent_env_path):
            dotenv_path_to_try = analysis_agent_env_path
            print(f"analysis_agent/.env 경로 사용: {dotenv_path_to_try}")
        else:
            # 2. Try my_state_agent/.env as a fallback (relative to this graph.py)
            # my_state_agent/.env is at ../../../my_state_agent/.env from graph.py
            my_state_agent_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'my_state_agent', '.env'))
            if os.path.exists(my_state_agent_env_path):
                dotenv_path_to_try = my_state_agent_env_path
                print(f"my_state_agent/.env 경로 사용: {dotenv_path_to_try}")

    if dotenv_path_to_try:
        load_dotenv(dotenv_path=dotenv_path_to_try)
        print(f".env 파일을 다음 경로에서 로드했습니다: {dotenv_path_to_try}")
    else:
        print(f"경고: .env 파일을 구성된 경로 또는 기본 경로들에서 찾을 수 없습니다. 시스템 환경 변수를 확인합니다.")
        # Fallback to system environment variables check, only error if they are also missing
        if not all([os.getenv("DB_HOST"), os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_PASSWORD")]):
            return {"error_message": "DB .env 파일을 찾을 수 없거나, 필수 DB 환경 변수가 시스템에도 설정되지 않았습니다.", "sql_result": ""}

    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        missing = [v for v, k in [("DB_HOST", DB_HOST), ("DB_NAME", DB_NAME), ("DB_USER", DB_USER), ("DB_PASSWORD", DB_PASSWORD)] if not k]
        return {"error_message": f"PostgreSQL 연결 정보 누락: {', '.join(missing)}", "sql_result": ""}

    conn_string = f"host='{DB_HOST}' port='{DB_PORT}' dbname='{DB_NAME}' user='{DB_USER}' password='{DB_PASSWORD}'"
    conn = None
    try:
        print(f"DB 연결 시도: {DB_NAME}@{DB_HOST}:{DB_PORT}")
        conn = psycopg2.connect(conn_string)
        print("DB 연결 성공.")
        print(f"실행할 쿼리: {sql_query}")
        df = pd.read_sql_query(sql_query, conn)
        return {"sql_result": df.to_string() if not df.empty else "결과 데이터가 없습니다."}
    except psycopg2.Error as e:
        print(f"DB 오류: {e}")
        return {"error_message": f"DB 오류: {e}", "sql_result": ""}
    except Exception as e:
        print(f"SQL 실행 중 오류: {e}")
        return {"error_message": f"SQL 실행 중 오류: {e}", "sql_result": ""}
    finally:
        if conn:
            conn.close()
            print("DB 연결 종료.")

async def summarize_sql_result_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- SUMMARIZE SQL RESULT NODE ---")
    user_query = state["messages"][-1].content
    sql_query = state.get("sql_query", "N/A")
    sql_result = state.get("sql_result", "N/A")
    error_message = state.get("error_message")

    if error_message:
        return {"final_answer": f"데이터 조회 중 오류가 발생했습니다: {error_message}"}
    if not sql_result or sql_result == "N/A":
         return {"final_answer": "데이터베이스에서 관련 정보를 찾지 못했습니다."}

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
    if not state['messages'] or not isinstance(state['messages'][-1], HumanMessage):
        # Similar to generate_sql_node, ensure this node is called appropriately
        return {"error_message": "No user query found for general question.", "final_answer": ""}
    user_query = state['messages'][-1].content
    general_answer_chain = general_answer_prompt | llm
    response = await general_answer_chain.ainvoke({"user_query": user_query}, config=config)
    final_answer = response.content.strip()
    print(f"General Answer: {final_answer}")
    return {"final_answer": final_answer}

# --- Conditional Edges Logic ---
def should_route_to_db_or_general(state: AgentState) -> Literal["generate_sql", "general_question"]:
    query_type = state.get("query_type")
    if query_type == "db_query":
        return "generate_sql"
    else: # general_query or fallback
        return "general_question"

# --- Graph Definition ---
workflow = StateGraph(AgentState)

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
# The config_schema=Configuration can be added if you intend to use runtime config for nodes
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
