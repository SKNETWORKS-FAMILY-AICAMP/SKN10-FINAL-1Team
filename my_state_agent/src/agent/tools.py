from typing import List, Dict, Any, Optional, Annotated
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState # For Annotated[WorkflowState, InjectedState]
from langgraph.types import Command
from src.agent.state import WorkflowState # Assuming state.py is in the same directory
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# 공통 도구 - 에이전트 전환 도구
@tool
def transfer_to_analysis_agent():
    """
    데이터 분석이 필요한 경우 데이터 분석 전문가에게 전환합니다.
    데이터 시각화, 통계 분석, 예측 모델링 등의 작업에 적합합니다.
    """
    return Command(
        goto="analytics_agent",
        update={"current_agent": "analytics_agent"},
        graph=Command.PARENT
    )

@tool
def transfer_to_document_agent():
    """
    문서 처리가 필요한 경우 문서 처리 전문가에게 전환합니다.
    문서 요약, 정보 추출, 질문 응답 등의 작업에 적합합니다.
    """
    return Command(
        goto="rag_agent", # Name used in supervisor
        update={"current_agent": "rag_agent"},
        graph=Command.PARENT
    )

@tool
def transfer_to_code_agent(): # Renamed from conversation for clarity with code_agent
    """
    코드 관련 작업이나 일반적인 대화가 필요한 경우 코드/대화 전문가에게 전환합니다.
    코드 작성, 디버깅, 일상적인 질문, 조언 등에 적합합니다.
    """
    return Command(
        goto="code_agent", # Name used in supervisor
        update={"current_agent": "code_agent"},
        graph=Command.PARENT
    )

def get_common_tools():
    """모든 에이전트가 사용할 수 있는 공통 도구"""
    return [
        transfer_to_analysis_agent,
        transfer_to_document_agent,
        transfer_to_code_agent
    ]

# 데이터 분석 도구
@tool
def analyze_data(
    data_description: str,  # Example: table name or query hint
    analysis_type: str,     # Example: "select_all", "list_tables", "custom_query"
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    데이터베이스에서 데이터를 조회하고 분석을 수행합니다.
    .env 파일은 'my_state_agent' 폴더에 있어야 합니다.
    Args:
        data_description: 분석할 데이터에 대한 설명 (예: 테이블 이름 또는 SQL 쿼리).
        analysis_type: 분석 유형 (예: 'list_tables', 'select_all', 'execute_query').
                       'execute_query'인 경우 data_description이 SQL 쿼리로 사용됩니다.
    """
    print(f"[Tool Call] analyze_data: desc='{data_description}', type='{analysis_type}'")

    # .env 파일 경로 설정 (tools.py 기준)
    # tools.py는 my_state_agent/src/agent/tools.py
    # .env는 my_state_agent/.env
    # 따라서 ../../.env
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')

    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f".env 파일을 다음 경로에서 로드했습니다: {os.path.abspath(dotenv_path)}")
    else:
        print(f"경고: .env 파일을 다음 경로에서 찾을 수 없습니다: {os.path.abspath(dotenv_path)}")
        # Fallback to environment variables if .env is not found or if variables are already set
        if not all([os.getenv("DB_HOST"), os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_PASSWORD")]):
            return "오류: .env 파일을 찾을 수 없거나, 필수 데이터베이스 환경 변수가 설정되지 않았습니다."

    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")

    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        missing = [var_name for var_name, var_val in [
            ("DB_HOST", DB_HOST), ("DB_NAME", DB_NAME),
            ("DB_USER", DB_USER), ("DB_PASSWORD", DB_PASSWORD)
        ] if not var_val]
        return f"오류: PostgreSQL 연결 정보가 누락되었습니다: {', '.join(missing)}. .env 파일 또는 환경 변수를 확인해주세요."

    conn_string = f"host='{DB_HOST}' port='{DB_PORT}' dbname='{DB_NAME}' user='{DB_USER}' password='{DB_PASSWORD}'"
    conn = None
    try:
        print(f"데이터베이스 연결 시도: {DB_NAME}@{DB_HOST}:{DB_PORT}")
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        print("데이터베이스 연결 성공.")

        query = ""
        if analysis_type == "list_tables":
            query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            df = pd.read_sql_query(query, conn)
            return f"데이터베이스의 테이블 목록:\n{df.to_string()}"
        elif analysis_type == "select_all" and data_description:
            # 간단한 테이블 이름 검증 (SQL 인젝션 방지를 위해 실제 사용 시에는 더 강력한 검증 필요)
            if not data_description.replace('_','').isalnum():
                return "오류: 테이블 이름이 유효하지 않습니다."
            query = f"SELECT * FROM {data_description};" # 테이블 이름은 data_description으로 가정
            df = pd.read_sql_query(query, conn)
            return f"'{data_description}' 테이블 데이터:\n{df.to_string()}"
        elif analysis_type == "execute_query" and data_description:
            query = data_description # data_description을 직접 SQL 쿼리로 사용
            # 주의: 이 방식은 SQL 인젝션에 취약할 수 있으므로, 실제 운영 환경에서는 쿼리를 안전하게 구성해야 합니다.
            # 예를 들어, 사용자 입력을 직접 쿼리에 삽입하는 대신 매개변수화된 쿼리를 사용해야 합니다.
            print(f"실행할 쿼리: {query}")
            df = pd.read_sql_query(query, conn)
            return f"쿼리 실행 결과:\n{df.to_string()}"
        else:
            return "오류: 유효한 analysis_type ('list_tables', 'select_all', 'execute_query')과 data_description을 제공해주세요."

    except psycopg2.Error as e:
        print(f"데이터베이스 오류: {e}")
        return f"데이터베이스 오류: {e}"
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        return f"분석 중 오류 발생: {e}"
    finally:
        if conn:
            conn.close()
            print("데이터베이스 연결 종료.")

@tool
def create_visualization(
    data_description: str,
    visualization_type: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    데이터 시각화를 생성합니다.
    Args:
        data_description: 시각화할 데이터에 대한 설명
        visualization_type: 시각화 유형 (bar, line, scatter, pie, heatmap 등)
    """
    print(f"[Tool Call] create_visualization: desc='{data_description}', type='{visualization_type}'")
    return f"{data_description}에 대한 {visualization_type} 시각화가 생성되었습니다: [이미지 URL 또는 설명]"

@tool
def predict_trend(
    data_description: str,
    time_horizon: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    데이터 기반 추세 예측을 수행합니다.
    Args:
        data_description: 예측할 데이터에 대한 설명
        time_horizon: 예측 기간 (예: '1 month', '1 year', '5 years')
    """
    print(f"[Tool Call] predict_trend: desc='{data_description}', horizon='{time_horizon}'")
    return f"{data_description}의 {time_horizon} 예측 결과: [예측 데이터 또는 설명]"

def data_analysis_tools():
    """데이터 분석 에이전트를 위한 도구 목록"""
    return [
        analyze_data,
        create_visualization,
        predict_trend
    ]

# 문서 처리 도구
@tool
def summarize_document(
    document_content: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig,
    max_length: Optional[int] = 500
) -> str:
    """
    문서 내용을 요약합니다.
    Args:
        document_content: 요약할 문서 내용
        max_length: 요약 최대 길이 (문자 수)
    """
    print(f"[Tool Call] summarize_document: content_len='{len(document_content)}', max_len='{max_length}'")
    return f"문서 요약 (첫 50자): {document_content[:50]}... [요약된 내용]"

@tool
def extract_information(
    document_content: str,
    info_type: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    문서에서 특정 유형의 정보를 추출합니다.
    Args:
        document_content: 정보를 추출할 문서 내용
        info_type: 추출할 정보 유형 (entities, dates, numbers, key_points 등)
    """
    print(f"[Tool Call] extract_information: content_len='{len(document_content)}', info_type='{info_type}'")
    return f"추출된 {info_type}: [추출된 정보 목록]"

@tool
def answer_document_question(
    document_content: str,
    question: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    문서 내용을 기반으로 질문에 답변합니다.
    Args:
        document_content: 질문의 근거가 되는 문서 내용
        question: 문서에 대한 질문
    """
    print(f"[Tool Call] answer_document_question: content_len='{len(document_content)}', question='{question}'")
    return f"질문 '{question}'에 대한 답변: [이용자 수가 점점 증가할 것 같아요]"

def document_processing_tools():
    """문서 처리 에이전트를 위한 도구 목록"""
    return [
        summarize_document,
        extract_information,
        answer_document_question
    ]

# 대화/코드 도구 (code_agent가 사용할 도구들)
@tool
def search_information(
    query: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    사용자 질문에 대한 정보를 검색합니다. (일반 검색 또는 코드 관련 검색)
    Args:
        query: 검색 쿼리
    """
    print(f"[Tool Call] search_information: query='{query}'")
    return f"'{query}'에 대한 검색 결과: [mcp로는 context7을 추천드립니다. context7은 최신 코드 문서들을 llm에 전달해주는 mcp에요.]"

@tool(return_direct=True)
def get_recommendations(
    category: str,
    preferences: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    사용자 선호도에 따라 추천을 제공합니다.
    Args:
        category: 추천 카테고리 (예: 'movies', 'books', 'restaurants', 'code_libraries')
        preferences: 사용자 선호도 설명
    """
    print(f"[Tool Call] get_recommendations: category='{category}', preferences='{preferences}'")
    return f"{category} 카테고리에서 '{preferences}'에 맞는 추천: [react]"

@tool
def track_conversation(
    current_agent_name: str,
    note: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    대화의 현재 상태나 중요한 정보를 기록합니다.
    Args:
        current_agent_name: 현재 활성화된 에이전트의 이름
        note: 기록할 내용
    """
    if not state.get('agent_history'):
        state['agent_history'] = []
    state['agent_history'].append({
        "agent": current_agent_name,
        "note": note
    })
    print(f"[Tool Call] track_conversation: agent='{current_agent_name}', note='{note}'")
    # This tool primarily updates state, so the return message is for confirmation.
    return f"대화 내용이 기록되었습니다: {note}"

def code_agent_tools(): # Renamed from conversation_tools
    """코드 및 일반 대화 에이전트를 위한 도구 목록"""
    return [
        search_information,
        get_recommendations,
        track_conversation
        # 여기에 실제 코드 작성/분석 도구를 추가해야 합니다.
    ]

async def get_mcp_tools(base_path: str) -> list:
    """
    Initializes the MultiServerMCPClient and retrieves tools from configured MCP servers.
    Args:
        base_path: The base directory of the project to resolve relative paths for local servers.
                   Example: If tools.py is in 'my_langraph_agent/src/agent', and math_server.py
                   is intended for 'my_langraph_agent/mcp_servers/math_server.py',
                   base_path should point to 'my_langraph_agent'.
    """
    # IMPORTANT: User needs to ensure math_server.py exists at this path
    # and the weather server is running if they want to use these specific examples.
    # The path to math_server.py should be relative to the 'base_path' passed to this function.
    # For example, if base_path is 'd:\dev\SKN10-FINAL-1Team\my_langraph_agent'
    # then math_server_path will be 'd:\dev\SKN10-FINAL-1Team\my_langraph_agent\mcp_servers\math_server.py'
    math_server_path = os.path.join(base_path, "mcp_servers", "math_server.py")
    
    print(f"DEBUG tools.py: Attempting to use math_server.py from: {math_server_path}")
    print("DEBUG tools.py: Please ensure 'math_server.py' exists at the specified path or update the path in your agent setup.")
    print("DEBUG tools.py: Also ensure any other MCP servers (e.g., weather server on port 8000) are running.")

    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [math_server_path], 
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp", # Standard MCP port for weather server example
                "transport": "streamable_http",
            }
            # Add other MCP server configurations here, for example:
            # "context7": {
            #     "url": "http://localhost:7777/mcp", # Example if context7 runs locally on port 7777
            #     "transport": "streamable_http",
            # }
        }
    )
    try:
        print("DEBUG tools.py: Attempting to connect to MCP servers and get tools...")
        mcp_tools_list = await client.get_tools()
        print(f"DEBUG tools.py: Successfully retrieved {len(mcp_tools_list)} MCP tools.")
        # for tool_instance in mcp_tools_list:
        #     print(f"DEBUG tools.py: MCP Tool: {tool_instance.name}, Description: {tool_instance.description}")
        return mcp_tools_list
    except Exception as e:
        print(f"ERROR tools.py: Failed to get MCP tools: {e}")
        print("ERROR tools.py: Please check your MCP server configurations and ensure servers are running.")
        print(f"ERROR tools.py: Specifically, check the path for math_server.py: {math_server_path} and if the weather server is up at http://localhost:8000/mcp.")
        return [] # Return empty list on failure
