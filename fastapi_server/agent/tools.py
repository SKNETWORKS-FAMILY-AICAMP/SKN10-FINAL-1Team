from typing import List, Dict, Any, Optional, Annotated
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState # For Annotated[WorkflowState, InjectedState]
from langgraph.types import Command
from .state import WorkflowState
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import os

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
