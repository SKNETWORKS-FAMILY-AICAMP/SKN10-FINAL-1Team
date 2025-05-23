from langgraph.graph import StateGraph, START, END
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from typing import Dict, Callable, Any, List, Optional
import re
import uuid

# Django 모델 임포트
try:
    from conversations.models import AgentType, ChatSession, ChatMessage, LlmCall
except ImportError:
    # 테스트 환경에서는 모델을 사용할 수 없을 수 있음
    ChatSession = None
    AgentType = None
    ChatMessage = None
    LlmCall = None

# 로컬 모듈 임포트
from .state import WorkflowState
from .agents import (
    create_analytics_agent,
    create_rag_agent,
    create_code_agent
)
from .router import (
    create_router,
    create_agent_conditional_edge_handlers,
    create_result_processor
)


def create_multi_agent_graph(model_name="gpt-4o-2024-08-06", memory=None, session_id: Optional[str]=None):
    """
    3가지 전문 에이전트로 분기되는 멀티에이전트 시스템 그래프를 생성합니다.
    
    Args:
        model_name: 사용할 LLM 모델명
        memory: 사용할 메모리 객체 (기본값: None)
        session_id: Django 세션 ID (선택적)
        
    Returns:
        컴파일된 멀티에이전트 그래프
    """
    # 메모리 초기화
    if memory is None:
        memory = ConversationBufferMemory(return_messages=True)
    
    # Django 세션 연결 (선택적)
    chat_session = None
    if session_id:
        try:
            chat_session = ChatSession.objects.get(id=session_id)
        except ChatSession.DoesNotExist:
            pass
    
    # 에이전트 생성
    analytics_agent = create_analytics_agent(model_name, session_id=session_id)
    rag_agent = create_rag_agent(model_name, session_id=session_id)
    code_agent = create_code_agent(model_name, session_id=session_id)
    
    # 라우터 및 결과 처리기 생성
    router = create_router(model_name=model_name, session_id=session_id)
    result_processor = create_result_processor()
    
    # 그래프 생성
    workflow = StateGraph(WorkflowState)
    
    # 노드 추가
    workflow.add_node("router", router)
    workflow.add_node("analytics_agent", analytics_agent)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("code_agent", code_agent)
    workflow.add_node("result_processor", result_processor)
    
    # 엣지 설정
    workflow.add_edge(START, "router")
    
    # 라우터에서 조건부 엣지 설정
    conditional_edges = create_agent_conditional_edge_handlers()
    workflow.add_conditional_edges("router", conditional_edges)
    
    # 에이전트에서 결과 처리기로 엣지 설정
    workflow.add_edge("analytics_agent", "result_processor")
    workflow.add_edge("rag_agent", "result_processor")
    workflow.add_edge("code_agent", "result_processor")
    
    # 추가 상호작용을 위해 결과 처리기에서 라우터로 루프백
    workflow.add_edge("result_processor", "router")
    
    # 그래프 컴파일
    return workflow.compile()


class MultiAgentExecutor:
    """멀티에이전트 실행기"""
    
    def __init__(self, model_name="gpt-4o-2024-08-06", session_id=None):
        """
        멀티에이전트 실행기를 초기화합니다.
        
        Args:
            model_name: 사용할 LLM 모델명
            session_id: Django 세션 ID (선택적)
        """
        self.model_name = model_name
        self.session_id = session_id
        self.memory = ConversationBufferMemory(return_messages=True)
        self.graph = create_multi_agent_graph(model_name, self.memory, session_id=session_id)
        self.state = {
            "messages": [],
            "metadata": {"session_id": session_id} if session_id else {},
            "completed_tasks": [],
            "agent_history": []
        }
        
        # Django 세션 연결 (선택적)
        self.chat_session = None
        if session_id and ChatSession is not None:
            try:
                self.chat_session = ChatSession.objects.get(id=session_id)
            except Exception:
                pass
    
    def process(self, user_input: str) -> Dict[str, Any]:
        """
        사용자 입력을 처리합니다.
        
        Args:
            user_input: 사용자 입력 텍스트
            
        Returns:
            처리 결과와 상태 정보가 포함된 디셔너리
        """
        # 사용자 입력을 메시지로 변환
        user_message = HumanMessage(content=user_input)
        
        # 현재 상태에 메시지 추가
        messages = self.state.get("messages", [])
        updated_messages = messages + [user_message]
        
        # Django DB에 사용자 메시지 저장 (선택적)
        if self.chat_session and ChatMessage is not None:
            try:
                ChatMessage.objects.create(
                    session=self.chat_session,
                    role='user',
                    content=user_input
                )
            except Exception as e:
                print(f"Django DB 저장 오류: {e}")
        
        # 상태 업데이트
        updated_state = {
            **self.state,
            "messages": updated_messages
        }
        
        # 그래프 실행
        result = self.graph.invoke(updated_state)
        
        # 결과 저장
        self.state = result
        
        # 에이전트 응답 추출
        response = self._get_last_ai_message(result.get("messages", []))
        current_agent = result.get("current_agent")
        
        return {
            "response": response,
            "agent": current_agent,
            "agent_history": result.get("agent_history", [])
        }
    
    async def aprocess(self, user_input: str) -> Dict[str, Any]:
        """
        사용자 입력을 비동기적으로 처리합니다.
        
        Args:
            user_input: 사용자 입력 텍스트
            
        Returns:
            처리 결과와 상태 정보가 포함된 디셔너리
        """
        # 사용자 입력을 메시지로 변환
        user_message = HumanMessage(content=user_input)
        
        # 현재 상태에 메시지 추가
        messages = self.state.get("messages", [])
        updated_messages = messages + [user_message]
        
        # Django DB에 사용자 메시지 비동기 저장 (선택적)
        if self.chat_session and ChatMessage is not None:
            try:
                from asgiref.sync import sync_to_async
                create_message = sync_to_async(ChatMessage.objects.create)
                await create_message(
                    session=self.chat_session,
                    role='user',
                    content=user_input
                )
            except Exception as e:
                print(f"Django DB 비동기 저장 오류: {e}")
        
        # 상태 업데이트
        updated_state = {
            **self.state,
            "messages": updated_messages
        }
        
        # 그래프 비동기 실행
        result = await self.graph.ainvoke(updated_state)
        
        # 결과 저장
        self.state = result
        
        # 에이전트 응답 추출
        response = self._get_last_ai_message(result.get("messages", []))
        current_agent = result.get("current_agent")
        
        return {
            "response": response,
            "agent": current_agent,
            "agent_history": result.get("agent_history", [])
        }
    
    def _get_last_ai_message(self, messages: List[Any]) -> str:
        """메시지 목록에서 마지막 AI 메시지의 내용을 반환합니다."""
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        if ai_messages:
            return ai_messages[-1].content
        return "응답을 생성하지 못했습니다."
    
    def reset(self):
        """대화 상태를 초기화합니다."""
        self.state = {
            "messages": [],
            "metadata": {"session_id": self.session_id} if self.session_id else {},
            "completed_tasks": [],
            "agent_history": []
        }
        
        # Django 세션 연결 상태도 초기화
        self.chat_session = None
        if self.session_id and ChatSession is not None:
            try:
                self.chat_session = ChatSession.objects.get(id=self.session_id)
            except Exception:
                pass


# 간편한 사용을 위한 함수
def create_multi_agent_executor(model_name="gpt-4o-2024-08-06", session_id=None) -> MultiAgentExecutor:
    """
    멀티에이전트 실행기를 생성합니다.
    
    Args:
        model_name: 사용할 LLM 모델명
        session_id: Django 세션 ID (선택적)
        
    Returns:
        멀티에이전트 실행기 인스턴스
    """
    return MultiAgentExecutor(model_name, session_id=session_id)


# 스트리밍 처리를 위한 비동기 버전
async def create_streaming_multi_agent_graph(model_name="gpt-4o-2024-08-06", session_id=None):
    """
    스트리밍을 지원하는 비동기 멀티에이전트 그래프를 생성합니다.
    
    Args:
        model_name: 사용할 LLM 모델명
        session_id: Django 세션 ID (선택적)
        
    Returns:
        스트리밍 처리 함수
    """
    graph = create_multi_agent_graph(model_name, session_id=session_id)
    
    async def process_with_streaming(state, stream_mode="updates"):
        """상태를 스트리밍 모드로 처리합니다."""
        # Django 세션 ID가 있으면 메타데이터에 추가
        if session_id and "metadata" in state:
            state["metadata"]["session_id"] = session_id
            
        async for chunk in graph.astream(state, stream_mode=stream_mode):
            yield chunk
    
    return process_with_streaming