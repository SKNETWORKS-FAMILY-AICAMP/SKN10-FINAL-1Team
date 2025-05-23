from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Dict, Callable, Any, List, Optional
import re
from .state import WorkflowState
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_supervisor

try:
    from conversations.models import ChatSession, ChatMessage, AgentType
except ImportError:
    # 테스트 환경에서는 모델을 사용할 수 없을 수 있음
    ChatSession = None
    ChatMessage = None
    AgentType = None

def create_router(model_name: str = "gpt-4o-2024-08-06", session_id: Optional[str] = None):
    """
    Supervisor 패턴을 활용한 에이전트 라우팅 함수 생성
    
    Args:
        model_name: 사용할 LLM 모델명
        session_id: Django 세션 ID (선택적)
    """
    # Supervisor 생성 준비
    def supervisor_router(state: WorkflowState) -> Dict[str, Any]:
        """
        Supervisor를 사용하여 메시지 내용을 분석하고 적절한 에이전트로 라우팅합니다.
        """
        # 이미 현재 에이전트가 지정되어 있으면 그대로 유지
        if state.get("current_agent"):
            return {"current_agent": state["current_agent"]}
            
        messages = state.get("messages", [])
        if not messages:
            # 메시지가 없으면 기본값으로 코드 에이전트 사용
            return {"current_agent": "code_agent"}
            
        # 마지막 메시지 확인 (일반적으로 사용자 메시지)
        last_message = messages[-1]
        if not isinstance(last_message, HumanMessage):
            # 마지막 메시지가 사용자 메시지가 아니면 기본값 사용
            return {"current_agent": "code_agent"}
        
        # Supervisor 사용 준비
        prompt = SystemMessage(content=(
            "당신은 다음 에이전트들을 관리하는 감독자입니다:\n"
            "1. analytics_agent: 데이터 분석, 차트 생성, 통계 처리, 비즈니스 인텔리전스를 담당\n"
            "2. rag_agent: 문서 관리, 지식베이스 검색, 정보 요약, PDF 처리를 담당\n"
            "3. code_agent: 코드 작성, 디버깅, 프로그래밍 도움, API 개발을 담당\n\n"
            "사용자의 요청을 분석하여 가장 적절한 에이전트 하나를 선택하세요. "
            "응답은 analytics_agent, rag_agent, code_agent 중 하나만 반환해야 합니다."
        ))
        
        # 모델 초기화
        llm = ChatOpenAI(model=model_name, temperature=0)
        
        # 라우팅 결정
        content = last_message.content
        response = llm.invoke([prompt, HumanMessage(content=f"사용자 요청: {content}\n가장 적절한 에이전트를 선택하세요.")])
        
        # LLM 응답에서 에이전트 추출
        agent_response = response.content.strip().lower()
        
        if "analytics" in agent_response:
            selected_agent = "analytics_agent"
        elif "rag" in agent_response:
            selected_agent = "rag_agent"
        elif "code" in agent_response:
            selected_agent = "code_agent"
        else:
            # 기본값: 코드 에이전트
            selected_agent = "code_agent"
        
        # Django DB 연동 (선택적)
        if session_id and ChatSession is not None and ChatMessage is not None:
            try:
                # 시스템 메시지 기록
                ChatMessage.objects.create(
                    session_id=session_id,
                    role="system",
                    content=f"라우터가 '{selected_agent}'를 선택했습니다."
                )
            except Exception as e:
                print(f"라우터 로그 저장 오류: {e}")
        
        return {"current_agent": selected_agent}
    
    return supervisor_router

def create_agent_conditional_edge_handlers() -> Dict[str, Callable]:
    """
    조건부 엣지 핸들러 생성
    
    Returns:
        조건부 엣지를 위한 함수 매핑
    """
    
    def route_to_analytics(state: WorkflowState) -> bool:
        return state.get("current_agent") == "analytics_agent"
        
    def route_to_rag(state: WorkflowState) -> bool:
        return state.get("current_agent") == "rag_agent"
        
    def route_to_code(state: WorkflowState) -> bool:
        return state.get("current_agent") == "code_agent"
    
    return {
        route_to_analytics: "analytics_agent",
        route_to_rag: "rag_agent",
        route_to_code: "code_agent"
    }

def create_result_processor():
    """각 에이전트의 결과를 처리하는 함수"""
    
    def process_result(state: WorkflowState) -> Dict[str, Any]:
        """
        에이전트 처리 결과를 기록하고 상태를 업데이트합니다.
        """
        # 완료된 작업 추적
        completed_tasks = state.get("completed_tasks", [])
        current_agent = state.get("current_agent", "unknown")
        
        if current_agent not in completed_tasks:
            completed_tasks.append(current_agent)
        
        # 에이전트 기록 추가
        agent_history = state.get("agent_history", [])
        messages = state.get("messages", [])
        
        # 마지막 AI 메시지가 있으면 해당 내용을 기록
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        if ai_messages:
            latest_ai_message = ai_messages[-1].content
            agent_history.append({
                "agent": current_agent,
                "response": latest_ai_message[:100] + "..." if len(latest_ai_message) > 100 else latest_ai_message
            })
        
        # 다음 상호작용을 위해 현재 에이전트 초기화
        # 에이전트 전환 도구를 통해 명시적으로 다른 에이전트로 전환하지 않는 한
        # 새 사용자 입력에 대해 라우터가 다시 실행됨
        return {
            "completed_tasks": completed_tasks,
            "agent_history": agent_history,
            "current_agent": None  # 라우터가 다음 메시지에 대해 다시 결정하도록 초기화
        }
    
    return process_result
