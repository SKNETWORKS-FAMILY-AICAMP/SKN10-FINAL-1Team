from typing import List, Dict, Any, Optional
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict, NotRequired

class MessagesState(TypedDict):
    """모든 에이전트가 공유하는 메시지 상태"""
    messages: List[AnyMessage]
    metadata: NotRequired[Dict[str, Any]]

class WorkflowState(MessagesState):
    """작업 흐름 추적을 위한 확장된 상태"""
    current_agent: NotRequired[str]
    completed_tasks: NotRequired[List[str]]
    results: NotRequired[Dict[str, Any]]
    agent_history: NotRequired[List[Dict[str, Any]]]
