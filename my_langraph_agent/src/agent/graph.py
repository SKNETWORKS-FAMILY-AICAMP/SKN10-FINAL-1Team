

from __future__ import annotations
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
import asyncio

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

# --- PostgreSQL Checkpointer Imports ---
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
# --- End PostgreSQL Checkpointer Imports ---

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

# --- PostgreSQL Env Var Debug Print (Optional) ---
print(f"DEBUG graph.py: DB_HOST={os.getenv('DB_HOST')}, DB_NAME={os.getenv('DB_NAME')}")
# --- End PostgreSQL Env Var Debug Print ---

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
다른 전문가의 도움이 필요한 경우 적절하게 전환하세요.

# 도구 사용 시나리오:
1. analyze_data 도구 사용:
   - 사용자가 "월별 판매 데이터의 통계적 특성이 궁금해요"라고 물으면 → analyze_data(data_description="월별 판매 데이터", analysis_type="descriptive") 호출
   - 사용자가 "나이와 소득 간의 관계가 있을까요?"라고 물으면 → analyze_data(data_description="나이와 소득 데이터", analysis_type="correlation") 호출
   - 사용자가 "어떤 요인이 주택 가격에 영향을 미치나요?"라고 물으면 → analyze_data(data_description="주택 가격 및 특성 데이터", analysis_type="regression") 호출

2. create_visualization 도구 사용:
   - 사용자가 "지역별 매출을 시각화해 주세요"라고 요청하면 → create_visualization(data_description="지역별 매출 데이터", visualization_type="bar") 호출
   - 사용자가 "시간에 따른 주가 변동을 보여주세요"라고 요청하면 → create_visualization(data_description="주가 시계열 데이터", visualization_type="line") 호출
   - 사용자가 "제품 카테고리별 판매 비중을 시각화해주세요"라고 요청하면 → create_visualization(data_description="제품 카테고리별 판매 데이터", visualization_type="pie") 호출

3. predict_trend 도구 사용:
   - 사용자가 "향후 3개월 동안의 매출을 예측해 주세요"라고 요청하면 → predict_trend(data_description="매출 데이터", time_horizon="3 months") 호출
   - 사용자가 "내년에 사용자 수가 어떻게 변할까요?"라고 물으면 → predict_trend(data_description="사용자 수 데이터", time_horizon="1 year") 호출
   - 사용자가 "5년 후 시장 점유율 예상은 어떻게 되나요?"라고 물으면 → predict_trend(data_description="시장 점유율 데이터", time_horizon="5 years") 호출

사용자의 질문이나 요청에 적절한 도구를 사용하여 응답하세요. 적절한 도구가 없거나 다른 에이전트의 도움이 필요한 경우, 전환 도구를 사용하세요."""
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
다른 전문가의 도움이 필요한 경우 적절하게 전환하세요.

# 도구 사용 시나리오:
1. summarize_document 도구 사용:
   - 사용자가 "이 연구 보고서를 요약해 주세요"라고 요청하면 → summarize_document(document_content="[문서 내용]", max_length=500) 호출
   - 사용자가 "이 계약서의 핵심 내용만 간략하게 알려주세요"라고 요청하면 → summarize_document(document_content="[계약서 내용]", max_length=300) 호출
   - 사용자가 "너무 긴 이메일인데 짧게 요약해 줄래요?"라고 요청하면 → summarize_document(document_content="[이메일 내용]", max_length=200) 호출

2. extract_information 도구 사용:
   - 사용자가 "이 문서에서 모든 날짜를 추출해주세요"라고 요청하면 → extract_information(document_content="[문서 내용]", info_type="dates") 호출
   - 사용자가 "이 논문에서 중요한 개체명을 찾아주세요"라고 요청하면 → extract_information(document_content="[논문 내용]", info_type="entities") 호출
   - 사용자가 "이 보고서에서 핵심 요점만 뽑아주세요"라고 요청하면 → extract_information(document_content="[보고서 내용]", info_type="key_points") 호출

3. answer_document_question 도구 사용:
   - 사용자가 "이 논문에서 연구 방법론은 무엇인가요?"라고 물으면 → answer_document_question(document_content="[논문 내용]", question="연구 방법론은 무엇인가요?") 호출
   - 사용자가 "이 계약서에 위약금 조항이 있나요?"라고 물으면 → answer_document_question(document_content="[계약서 내용]", question="위약금 조항이 있나요?") 호출
   - 사용자가 "이 문서에 따르면 향후 사용자 수 예측은 어떻게 되나요?"라고 물으면 → answer_document_question(document_content="[문서 내용]", question="향후 사용자 수 예측은 어떻게 되나요?") 호출

사용자의 문서 관련 요청에 적절한 도구를 사용하여 응답하세요. 적절한 도구가 없거나 다른 에이전트의 도움이 필요한 경우, 전환 도구를 사용하세요."""
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
다른 전문가의 도움이 필요한 경우 적절하게 전환하세요.

# 도구 사용 시나리오:
1. search_information 도구 사용:
   - 사용자가 "파이썬에서 비동기 프로그래밍은 어떻게 하나요?"라고 물으면 → search_information(query="파이썬 비동기 프로그래밍 방법") 호출
   - 사용자가 "React와 Vue의 차이점이 뭔가요?"라고 물으면 → search_information(query="React와 Vue 프레임워크 차이점") 호출
   - 사용자가 "MCP란 무엇인가요?"라고 물으면 → search_information(query="MCP 의미와 활용") 호출
   - 사용자가 "LangGraph에서 에이전트를 어떻게 만드나요?"라고 물으면 → search_information(query="LangGraph 에이전트 생성 방법") 호출

2. get_recommendations 도구 사용:
   - 사용자가 "데이터 분석에 좋은 파이썬 라이브러리를 추천해주세요"라고 요청하면 → get_recommendations(category="code_libraries", preferences="파이썬 데이터 분석") 호출
   - 사용자가 "웹 개발을 배우고 싶은데 어떤 언어부터 시작하는 게 좋을까요?"라고 물으면 → get_recommendations(category="programming_languages", preferences="웹 개발 입문") 호출
   - 사용자가 "AI 개발에 필요한 기술 스택을 추천해주세요"라고 요청하면 → get_recommendations(category="tech_stack", preferences="AI 개발") 호출

3. track_conversation 도구 사용:
   - 사용자의 중요한 선호도를 기록할 때 → track_conversation(current_agent_name="code_agent", note="사용자는 파이썬 기반 데이터 분석에 관심이 있음") 호출
   - 진행 중인 작업을 추적할 때 → track_conversation(current_agent_name="code_agent", note="사용자는 웹 앱 개발 중으로 React 관련 정보 요청 중") 호출
   - 다른 에이전트로 전환하기 전에 상태를 기록할 때 → track_conversation(current_agent_name="code_agent", note="코드 문제 해결 후 데이터 분석으로 전환 필요") 호출

사용자의 코드 관련 질문이나 일반적인 질문에 적절한 도구를 사용하여 응답하세요. 코드 질문이 아닌 경우에도 general purpose 에이전트로서 일반적인 대화에 응답할 수 있습니다. 적절한 도구가 없거나 다른 에이전트의 도움이 필요한 경우, 전환 도구를 사용하세요."""
code_agent_tools_list = code_agent_tools() + get_common_tools()
code_agent_llm = init_chat_model(
    MODEL_IDENTIFIER,
    temperature=LLM_TEMPERATURE,
    model_kwargs={"streaming": LLM_STREAMING}
)
code_agent_runnable = create_react_agent(
    model=code_agent_llm,
    tools=code_agent_tools_list,
    prompt=code_agent_system_prompt_string,
    name="code_agent"
)
print("Code Agent Runnable 정의 완료 (using init_chat_model)")

# Supervisor
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

# PostgreSQL 연결 정보 (환경 변수에서 로드)
# .env 파일에 다음 변수들이 설정되어 있어야 합니다:
# DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SSLMODE (optional, e.g., 'prefer' or 'require')
DB_CONNECT_STRING = (
    f"postgresql://{os.environ.get('DB_USER')}:{os.environ.get('DB_PASSWORD')}"
    f"@{os.environ.get('DB_HOST')}:{os.environ.get('DB_PORT', '5432')}"
    f"/{os.environ.get('DB_NAME')}"
)
# SSL mode can be added later if needed
# if os.environ.get('DB_SSLMODE'):
#     DB_CONNECT_STRING += f"?sslmode={os.environ.get('DB_SSLMODE')}"

# supervisor_graph를 전역 변수로 선언 (main 함수 내에서 설정됨)
supervisor_graph = None

# 5. 메인 실행 로직
async def main():
    global supervisor_graph # 전역 supervisor_graph 사용 선언

    print("멀티 에이전트 시스템을 초기화합니다...")

    # PostgreSQL 연결 풀 및 checkpointer 설정
    async with AsyncConnectionPool(
        conninfo=DB_CONNECT_STRING,
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    ) as pool:
        memory_saver = AsyncPostgresSaver(pool)
        # 중요: 처음 DB 테이블을 생성할 때만 다음 줄의 주석을 해제하고 실행하세요.
        await memory_saver.setup()
        # print("PostgreSQL checkpointer 테이블 설정 완료 (필요한 경우).")

        supervisor_graph = create_supervisor(
            agents=list(agents_dict.values()), # Pass the agent runnables
            model=supervisor_llm_instance,
            prompt=supervisor_system_prompt_text,
        ).compile(checkpointer=memory_saver) # checkpointer 추가
        print("Supervisor 그래프가 PostgreSQL checkpointer와 함께 컴파일되었습니다.")

        # 대화 스레드 ID (고정 또는 동적 할당 가능)
        # 간단한 예시로 고정된 thread_id를 사용합니다.
        # 실제 애플리케이션에서는 사용자별 또는 세션별 고유 ID를 사용하는 것이 좋습니다.
        thread_id = "default_chat_thread_v2" # 이전 버전과 구분하기 위해 _v2 추가 가능
        print(f"현재 대화 스레드 ID: {thread_id}")


        # 이전 대화 불러오기 (선택 사항)
        # config = {"configurable": {"thread_id": thread_id}}
        # past_messages = await memory_saver.aget(config)
        # if past_messages:
        #     print("\n--- 이전 대화 내용 ---")
        #     for msg_type, content_list in past_messages.items():
        #         if msg_type == "messages": # messages 키 아래에 실제 메시지들이 있음
        #             for msg_data in content_list: # LangGraph 메시지 객체
        #                 if isinstance(msg_data, HumanMessage):
        #                     print(f"사용자: {msg_data.content}")
        #                 elif isinstance(msg_data, AIMessage):
        #                     print(f"AI: {msg_data.content}")
        #     print("--- 이전 대화 끝 ---\n")
        # else:
        #     print(f"'{thread_id}'에 대한 이전 대화 내용이 없습니다.")


        # 대화 시작
        while True:
            user_input = input("사용자: ")
            if user_input.lower() in ["exit", "quit"]:
                print("대화를 종료합니다.")
                break

            # Supervisor는 메시지 목록을 입력으로 받습니다.
            # Checkpointer를 사용하므로 `configurable`에 `thread_id`를 전달해야 합니다.
            config = {"configurable": {"thread_id": thread_id}}
            
            current_input_messages = [HumanMessage(content=user_input)]

            async for event in supervisor_graph.astream_events(
                {"messages": current_input_messages},
                config=config, # config 전달
                version="v2",
                # output_keys=None, # 모든 output key를 스트리밍 (기본값)
                # input_keys=None, # 모든 input key를 스트리밍 (기본값)
                # stream_mode="values" # "values" | "updates" | "debug" (기본값 "values")
            ):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        print(content, end="", flush=True)
                elif kind == "on_tool_start":
                    print(f"\n--- Executing Tool: {event['name']} ({event['tags']}) ---")
                    print(f"    Args: {event['data'].get('input')}")
                elif kind == "on_tool_end":
                    tool_output = str(event['data'].get('output'))
                    print(f"--- Tool End: {event['name']} ---")
                    if len(tool_output) > 200:
                        print(f"--- Tool Output (first 200 chars): {tool_output[:200]}... ---")
                    else:
                        print(f"--- Tool Output: {tool_output} ---")

                # 디버깅을 위해 추가적인 이벤트 로깅 (필요시 주석 해제)
                # elif kind in ["on_chain_start", "on_chain_end", "on_llm_start", "on_llm_end", "on_retriever_start", "on_retriever_end"]:
                #     print(f"\n--- Event: {kind} | Name: {event['name']} | Tags: {event['tags']} | ID: {event['run_id']} ---")
                #     if event['data'].get('input'):
                #         print(f"    Input: {str(event['data']['input'])[:150]}")
                #     if event['data'].get('output') and kind.endswith("_end"):
                #         print(f"    Output: {str(event['data']['output'])[:150]}")
                # elif kind not in ["on_chat_model_stream", "on_tool_start", "on_tool_end"]:
                #      print(f"Event: {kind}, Name: {event.get('name')}, Data: {event['data']}")


            print("\n---------------------")

if __name__ == "__main__":
    # Windows에서 asyncio 사용 시 SelectorEventLoopPolicy 설정 (Python 3.8+ 에서는 기본값일 수 있음)
    if os.name == 'nt' and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()