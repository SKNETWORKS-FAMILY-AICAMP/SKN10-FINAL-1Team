from langchain_core.tools import tool
from langgraph.types import Command
from typing import List, Dict, Any, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState
from typing import Annotated
from .state import WorkflowState

# 공통 도구 - 에이전트 전환 도구
def get_common_tools():
    """모든 에이전트가 사용할 수 있는 공통 도구"""
    return [
        transfer_to_analysis_agent,
        transfer_to_document_agent,
        transfer_to_conversation_agent
    ]

@tool
def transfer_to_analysis_agent():
    """
    데이터 분석이 필요한 경우 데이터 분석 전문가에게 전환합니다.
    데이터 시각화, 통계 분석, 예측 모델링 등의 작업에 적합합니다.
    """
    return Command(
        goto="analysis_agent",
        update={"current_agent": "analysis_agent"},
        graph=Command.PARENT
    )

@tool
def transfer_to_document_agent():
    """
    문서 처리가 필요한 경우 문서 처리 전문가에게 전환합니다.
    문서 요약, 정보 추출, 질문 응답 등의 작업에 적합합니다.
    """
    return Command(
        goto="document_agent",
        update={"current_agent": "document_agent"},
        graph=Command.PARENT
    )

@tool
def transfer_to_conversation_agent():
    """
    일반적인 대화나 정보 제공이 필요한 경우 대화 전문가에게 전환합니다.
    일상적인 질문, 조언, 추천 등에 적합합니다.
    """
    return Command(
        goto="conversation_agent",
        update={"current_agent": "conversation_agent"},
        graph=Command.PARENT
    )

# 데이터 분석 도구
def data_analysis_tools():
    """데이터 분석 에이전트를 위한 도구 목록"""
    return [
        analyze_data,
        create_visualization,
        predict_trend
    ]

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
    # 실제 구현에서는 데이터 소스에 연결하여 분석 수행
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
    # 실제 구현에서는 데이터를 시각화하고 이미지 URL이나 설명 반환
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
    # 실제 구현에서는 예측 모델을 실행하고 결과 반환
    return f"{data_description}의 {time_horizon} 예측 결과: [예측 데이터 또는 설명]"

# 문서 처리 도구
def document_processing_tools():
    """문서 처리 에이전트를 위한 도구 목록"""
    return [
        summarize_document,
        extract_information,
        answer_document_question
    ]

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
    # 실제 구현에서는 문서 요약 로직 구현
    return f"문서 요약: [요약된 내용]"

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
    # 실제 구현에서는 문서에서 정보 추출 로직 구현
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
    # 실제 구현에서는 문서 기반 QA 로직 구현
    return f"질문 '{question}'에 대한 답변: [답변 내용]"

# 대화 도구
def conversation_tools():
    """대화 에이전트를 위한 도구 목록"""
    return [
        search_information,
        get_recommendations,
        track_conversation
    ]

@tool
def search_information(
    query: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    사용자 질문에 대한 정보를 검색합니다.
    
    Args:
        query: 검색 쿼리
    """
    # 실제 구현에서는 검색 API 호출 등 구현
    return f"'{query}'에 대한 검색 결과: [검색 결과]"

@tool
def get_recommendations(
    category: str,
    preferences: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    사용자 선호도에 기반한 추천을 제공합니다.
    
    Args:
        category: 추천 카테고리 (movies, books, restaurants 등)
        preferences: 사용자 선호도 설명
    """
    # 실제 구현에서는 추천 시스템 로직 구현
    return f"{preferences}에 기반한 {category} 추천: [추천 목록]"

@tool
def track_conversation(
    note: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    대화 내용을 추적하고 중요 정보를 기록합니다.
    
    Args:
        note: 기록할 대화 내용 메모
    """
    # 대화 이력에 메모 추가
    agent_history = state.get("agent_history", [])
    agent_history.append({
        "agent": state.get("current_agent", "unknown"),
        "note": note
    })
    
    return f"대화 내용이 기록되었습니다: {note}"
