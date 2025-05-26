from typing import List, Dict, Any, Optional, Annotated
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState # For Annotated[WorkflowState, InjectedState]
from langgraph.types import Command
from src.agent.state import WorkflowState # Assuming state.py is in the same directory

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
    data_description: str,
    analysis_type: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    데이터 분석을 수행합니다.
    Args:
        data_description: 분석할 데이터에 대한 설명
        analysis_type: 분석 유형 (descriptive, correlation, regression, classification 등)
    """
    print(f"[Tool Call] analyze_data: desc='{data_description}', type='{analysis_type}'")
    return f"{data_description}에 대한 {analysis_type} 분석 결과입니다: [분석 내용 샘플]"

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
