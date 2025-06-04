from typing import List, Dict, Any, Optional, Annotated
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState # For Annotated[WorkflowState, InjectedState]
from langgraph.types import Command
from .state import WorkflowState
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import os

# 공통 도구 - 에이전트 전환 도구
@tool
def transfer_to_analysis_agent():
    """
    데이터 분석이 필요한 경우 데이터 분석 전문가에게 전환합니다.
    데이터 시각화, 통계 분석, 예측 모델링 등의 작업에 적합합니다.
    """
    from langgraph.graph.message import add_messages
    return Command(
        goto="analytics_agent",
        update={
            "current_agent": "analytics_agent",
            "messages": add_messages([])
        },
        graph=Command.PARENT
    )

@tool
def transfer_to_document_agent():
    """
    문서 처리가 필요한 경우 문서 처리 전문가에게 전환합니다.
    문서 요약, 정보 추출, 질문 응답 등의 작업에 적합합니다.
    """
    from langgraph.graph.message import add_messages
    return Command(
        goto="rag_agent", # Name used in supervisor
        update={
            "current_agent": "rag_agent",
            "messages": add_messages([])
        },
        graph=Command.PARENT
    )

@tool
def transfer_to_code_agent():
    """
    코드 관련 작업이나 일반적인 대화가 필요한 경우 코드/대화 전문가에게 전환합니다.
    코드 작성, 디버깅, 일상적인 질문, 조언 등에 적합합니다.
    """
    from langgraph.graph.message import add_messages
    return Command(
        goto="code_agent", # Name used in supervisor
        update={
            "current_agent": "code_agent",
            "messages": add_messages([])
        },
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
def analyze_data(data: str) -> str:
    """데이터를 분석합니다. 데이터는 JSON 형식 또는 CSV 형식의 문자열이어야 합니다."""
    return f"데이터 분석 결과: {data[:50]}... (분석 완료)"

@tool
def create_visualization(data: str, chart_type: str) -> str:
    """
    데이터를 시각화합니다.
    data: JSON 또는 CSV 형식의 데이터
    chart_type: 차트 유형 (bar, line, pie 등)
    """
    return f"{chart_type} 차트 생성 완료: {data[:30]}..."

data_analysis_tools = [
    analyze_data,
    create_visualization
]

# 문서 처리 도구
@tool
def search_documents(query: str) -> str:
    """
    문서 데이터베이스에서 특정 쿼리와 관련된 정보를 검색합니다.
    query: 검색할 키워드 또는 질문
    """
    return f"'{query}'에 대한 검색 결과: 관련 문서 3개 발견"

@tool
def summarize_document(document: str) -> str:
    """
    긴 문서를 요약합니다.
    document: 요약할 문서 내용
    """
    return f"문서 요약: {document[:30]}... (요약 완료)"

document_processing_tools = [
    search_documents,
    summarize_document
]

# 코드 관련 도구
@tool
def explain_code(code: str) -> str:
    """
    코드를 설명합니다.
    code: 설명할 코드 블록
    """
    return f"코드 설명: {code[:30]}... (설명 완료)"

@tool
def debug_code(code: str, error_message: str) -> str:
    """
    코드의 디버깅을 도와줍니다.
    code: 디버깅할 코드 블록
    error_message: 발생한 오류 메시지
    """
    return f"디버깅 결과: {error_message} - 원인 파악 및 수정 제안"

code_agent_tools = [
    explain_code,
    debug_code
]
