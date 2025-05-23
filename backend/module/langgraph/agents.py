from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

# Django 통합을 위한 임포트
from django.conf import settings
from conversations.models import AgentType, ChatSession, ChatMessage, LlmCall

from .tools import (
    data_analysis_tools,
    document_processing_tools,
    conversation_tools,
    get_common_tools
)

def create_analytics_agent(model_name="gpt-4o-2024-08-06", temperature=0, session_id=None):
    """데이터 분석 전문 에이전트 생성 (AgentType.ANALYTICS와 일치)"""
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    
    # 시스템 메시지 정의
    system_message = """당신은 데이터 분석 전문가입니다. 
    사용자의 데이터 관련 질문을 이해하고 적절한 분석을 수행하여 통찰력 있는 결과를 제공합니다.
    데이터 시각화, 통계 분석, 예측 모델링에 전문성이 있습니다.
    다른 전문가의 도움이 필요한 경우 적절하게 전환하세요."""
    
    # 도구 준비
    tools = data_analysis_tools() + get_common_tools()
    
    # 에이전트 생성
    agent = create_react_agent(
        llm,
        tools,
        system_message
    )
    
    # Django 세션 연결 (선택적)
    if session_id:
        try:
            session = ChatSession.objects.get(id=session_id)
            # 시스템 메시지 저장
            ChatMessage.objects.create(
                session=session,
                role="system",
                content=system_message
            )
        except ChatSession.DoesNotExist:
            pass
    
    return agent


def create_rag_agent(model_name="gpt-4o-2024-08-06", temperature=0, session_id=None):
    """문서 처리 전문 에이전트 생성 (AgentType.RAG와 일치)"""
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    
    # 시스템 메시지 정의
    system_message = """당신은 문서 처리와 지식 검색 전문가입니다.
    문서 요약, 정보 추출, 질문 응답, 문서 변환과 같은 문서 관련 작업을 수행합니다.
    다양한 형식(PDF, DOC, TXT 등)의 문서를 처리할 수 있습니다.
    다른 전문가의 도움이 필요한 경우 적절하게 전환하세요."""
    
    # 도구 준비
    tools = document_processing_tools() + get_common_tools()
    
    # 에이전트 생성
    agent = create_react_agent(
        llm,
        tools,
        system_message
    )
    
    # Django 세션 연결 (선택적)
    if session_id:
        try:
            session = ChatSession.objects.get(id=session_id)
            # 시스템 메시지 저장
            ChatMessage.objects.create(
                session=session,
                role="system",
                content=system_message
            )
        except ChatSession.DoesNotExist:
            pass
    
    return agent


def create_code_agent(model_name="gpt-4o-2024-08-06", temperature=0, session_id=None):
    """코드 전문 에이전트 생성 (AgentType.CODE와 일치)"""
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    
    # 시스템 메시지 정의
    system_message = """당신은 코드 분석 및 개발 전문가입니다.
    코드 작성, 수정, 버그 수정, 코드 분석 등 프로그래밍 관련 작업을 전문적으로 수행합니다.
    여러 프로그래밍 언어와 프레임워크에 대한 지식을 갖추고 있습니다.
    특수한 문서 처리나 데이터 분석이 필요한 경우 적절한 전문가에게 전환하세요."""
    
    # 도구 준비 (대화 도구 대신 코드 관련 도구를 사용해야 하지만, 임시로 기존 도구 사용)
    tools = conversation_tools() + get_common_tools()
    
    # 에이전트 생성
    agent = create_react_agent(
        llm,
        tools,
        system_message
    )
    
    # Django 세션 연결 (선택적)
    if session_id:
        try:
            session = ChatSession.objects.get(id=session_id)
            # 시스템 메시지 저장
            ChatMessage.objects.create(
                session=session,
                role="system",
                content=system_message
            )
        except ChatSession.DoesNotExist:
            pass
    
    return agent
