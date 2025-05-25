from __future__ import annotations

import asyncio
import os
from typing import List, Dict, Any, Optional, Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI # Still needed if any direct use remains, or for type hinting if preferred
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent # InjectedState will be used in tools.py
from langgraph_supervisor import create_supervisor
# Command will be used in tools.py
# TypedDict, NotRequired will be used in state.py

# .env 파일 로드 (프로젝트 루트에 있는 .env 파일을 기준으로 경로 설정)
# 이 파일의 위치에 따라 경로 조정 필요 (my_langraph_agent/src/agent/.env 또는 my_langraph_agent/.env 등)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env') # Assumes .env is in my_langraph_agent directory
if not os.path.exists(dotenv_path):
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env') # Assumes .env is in SKN10-FINAL-1Team directory
load_dotenv(dotenv_path=dotenv_path)

# --- API Key Debug Print (Optional) ---
OPENAI_API_KEY_ENV = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY_ENV and len(OPENAI_API_KEY_ENV) > 10:
    print(f"DEBUG graph.py: OPENAI_API_KEY seems set (length: {len(OPENAI_API_KEY_ENV)}, first 5 chars: {OPENAI_API_KEY_ENV[:5]}) ")
else:
    print("DEBUG graph.py: OPENAI_API_KEY is NOT SET or is very short.")
# --- End API Key Debug Print ---

# 1. 상태 및 도구 임포트
from src.agent.state import MessagesState, WorkflowState # WorkflowState는 tools.py에서 사용되지만, graph.py에서도 직접 참조될 수 있으므로 유지
from src.agent.tools import get_common_tools, data_analysis_tools, document_processing_tools, code_agent_tools

# 3. 에이전트 및 슈퍼바이저 직접 선언

# 공통 LLM 설정
MODEL_IDENTIFIER = "openai:gpt-4o-2024-08-06" # init_chat_model 형식
LLM_TEMPERATURE = 0.7
LLM_STREAMING = True

# Analytics Agent Runnable
analytics_agent_system_prompt_string = """당신은 데이터 분석 전문가입니다.
데이터 시각화, 통계 분석, 예측 모델링과 같은 데이터 관련 작업을 수행합니다.
복잡한 데이터셋을 처리하고 실행 가능한 인사이트를 도출할 수 있습니다.
다른 전문가의 도움이 필요한 경우 적절하게 전환하세요."""
analytics_agent_tools = data_analysis_tools() + get_common_tools()
analytics_agent_llm = init_chat_model(
    MODEL_IDENTIFIER,
    temperature=LLM_TEMPERATURE,
    model_kwargs={"streaming": LLM_STREAMING}
)
analytics_agent_runnable = create_react_agent(
    model=analytics_agent_llm,
    tools=analytics_agent_tools,
    prompt=analytics_agent_system_prompt_string,
    name="analytics_agent"
)
print("Analytics Agent Runnable 정의 완료 (using init_chat_model)")

# RAG Agent Runnable
rag_agent_system_prompt_string = """당신은 문서 처리와 지식 검색 전문가입니다.
문서 요약, 정보 추출, 질문 응답, 문서 변환과 같은 문서 관련 작업을 수행합니다.
PDF, TXT, DOCX 등 다양한 형식의 문서를 처리할 수 있습니다.
필요한 정보를 정확하고 빠르게 찾아 제공합니다.
다른 전문가의 도움이 필요한 경우 적절하게 전환하세요."""
rag_agent_tools_list = document_processing_tools() + get_common_tools()
rag_agent_llm = init_chat_model(
    MODEL_IDENTIFIER,
    temperature=LLM_TEMPERATURE,
    model_kwargs={"streaming": LLM_STREAMING}
)
rag_agent_runnable = create_react_agent(
    model=rag_agent_llm,
    tools=rag_agent_tools_list,
    prompt=rag_agent_system_prompt_string,
    name="rag_agent"
)
print("RAG Agent Runnable 정의 완료 (using init_chat_model)")

# Code Agent Runnable
code_agent_system_prompt_string = """당신은 코드 분석 및 개발 전문가이자 일반적인 대화 상대입니다.
코드 작성, 수정, 버그 수정, 코드 분석 등 프로그래밍 관련 작업을 전문적으로 수행합니다.
또한, 일반적인 질문에 답하고 대화를 나눌 수 있습니다.
여러 프로그래밍 언어와 프레임워크에 대한 지식을 갖추고 있습니다.
특수한 문서 처리나 데이터 분석이 필요한 경우 적절한 전문가에게 전환하세요."""
code_agent_tools_list = code_agent_tools() + get_common_tools()
code_agent_llm = init_chat_model(
    MODEL_IDENTIFIER,
    temperature=LLM_TEMPERATURE,
    model_kwargs={"streaming": LLM_STREAMING}
)

# 모델에 get_recommendations 도구 강제 바인딩 (일시적으로 주석 처리하여 비활성화)
# code_agent_llm_forced = code_agent_llm.bind_tools(
#     tools=code_agent_tools_list,  # 전체 도구 목록 제공
#     tool_choice={"type": "function", "function": {"name": "get_recommendations"}}
# )

code_agent_runnable = create_react_agent(
    model=code_agent_llm,  # 원본 LLM 사용 (강제 바인딩 없음)
    tools=code_agent_tools_list,
    prompt=code_agent_system_prompt_string,
    name="code_agent"
)
print("Code Agent Runnable 정의 완료 (using init_chat_model, get_recommendations 강제 호출 설정됨)")

# Supervisor
# Supervisor LLM (can be same or different from agent LLMs)
supervisor_llm_instance = init_chat_model(
    MODEL_IDENTIFIER,
    temperature=LLM_TEMPERATURE,
    model_kwargs={"streaming": LLM_STREAMING}
)
print("Supervisor LLM 인스턴스 생성 완료 (using init_chat_model)")

agents_dict = {
    "analytics_agent": analytics_agent_runnable,
    "rag_agent": rag_agent_runnable,
    "code_agent": code_agent_runnable,
}

supervisor_system_prompt_text = (
    "You are a supervisor of multiple AI agents. "
    "You receive user requests and determine which agent is best suited to handle them. "
    "The agents are:\n"
    "- 'analytics_agent': Data analysis, visualization, statistics, prediction.\n"
    "- 'rag_agent': Document summarization, information extraction, document-based Q&A.\n"
    "- 'code_agent': Code writing, debugging, general questions, coding-related questions, advice, code snippets, programming concepts, etc.\n"
    "Please respond in Korean, and the agents will respond in Korean as well. "
    "If the request is unclear or requires additional information, you can ask the user a question. "
    "When all tasks are complete or the user is satisfied, you can end the conversation by using 'FINISH'. "
    "Agent switching should be done using the exact name of the agent (e.g. 'analytics_agent'). "
    "If the request is a greeting, a simple question that doesn't require a specialized agent, "
    "or if you can answer it directly, you can respond with 'FINISH'."
)

supervisor_graph = create_supervisor(
    agents=list(agents_dict.values()), # Pass the agent runnables
    model=supervisor_llm_instance,
    prompt=supervisor_system_prompt_text,
).compile(checkpointer=None)
print("Supervisor 그래프가 전역적으로 컴파일되었습니다.")
# 5. 메인 실행 로직
async def main():
    print("멀티 에이전트 시스템을 초기화합니다...")
    # 전역적으로 선언된 supervisor_graph를 사용합니다.
    # (이전 create_main_supervisor_graph 호출 및 관련 print문은 제거됨)

    # 대화 시작
    while True:
        user_input = input("사용자: ")
        if user_input.lower() in ["exit", "quit"]:
            print("대화를 종료합니다.")
            break

        # Supervisor는 메시지 목록을 입력으로 받습니다.
        # WorkflowState를 직접 사용하기보다는, supervisor가 내부적으로 메시지를 관리하도록 함.
        # `astream`은 일반적으로 최종 출력 스트림을 제공합니다.
        # `astream_events`는 더 상세한 중간 단계 이벤트를 제공합니다.
        # create_supervisor는 `astream_events`와 잘 작동합니다.
        
        # For `astream_events`, the input is a dictionary, often just `{"messages": [HumanMessage(content=user_input)]}`
        # if not using a checkpointer with `thread_id`.
        # If a checkpointer is used, a `configurable` dict with `thread_id` is needed.
        
        # Let's use astream_events for more detailed output, similar to common supervisor examples.
        # The input to the supervisor graph is typically a dictionary with a "messages" key.
        async for event in supervisor_graph.astream_events(
            {"messages": [HumanMessage(content=user_input)]},
            version="v2", # Use v2 for the latest event structure
            # config={"recursion_limit": 10} # Optional: set recursion limit
        ):
            kind = event["event"]
            
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="", flush=True)
            elif kind == "on_tool_start":
                print(f"\n--- Executing Tool: {event['name']} ---")
                print(f"    Args: {event['data'].get('input')}")
            elif kind == "on_tool_end":
                print(f"--- Tool End: {event['name']} ---")
                # print(f"    Output: {event['data'].get('output')}") # Can be verbose
                print(f"--- Tool Output (first 100 chars): {str(event['data'].get('output'))[:100]} ---")
            elif kind == "on_chain_end" or kind == "on_chat_model_end" or kind == "on_llm_end":
                # These can be noisy, print if useful for debugging
                # print(f"Event: {kind}, Name: {event.get('name')}")
                pass # Avoid too much noise
            # else:
                # print(f"Event: {kind}, Data: {event['data']}") # For debugging other events

        print("\n---------------------")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨.")