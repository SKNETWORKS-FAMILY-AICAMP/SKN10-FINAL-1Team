from __future__ import annotations

import asyncio
import os
import sys
import functools
from typing import Sequence, Annotated, TypedDict, List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState

from langgraph.graph.message import add_messages

# Add project root to sys.path to enable imports from the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



# Import tools and state
from src.agent.tools import get_common_tools, data_analysis_tools, document_processing_tools, code_agent_tools
from src.agent.state import MessagesState

# Import LLM and agent creation utilities
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# --- Load .env --- #
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env') 
if not os.path.exists(dotenv_path):
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

OPENAI_API_KEY_ENV = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY_ENV and len(OPENAI_API_KEY_ENV) > 10:
    print(f"DEBUG state_graph_supervisor.py: OPENAI_API_KEY seems set (length: {len(OPENAI_API_KEY_ENV)}, first 5 chars: {OPENAI_API_KEY_ENV[:5]}) ")
else:
    print("DEBUG state_graph_supervisor.py: OPENAI_API_KEY is NOT SET or is very short.")
# --- End Load .env --- #

# --- Constants and LLM/Agent Setup --- #
MODEL_IDENTIFIER = "openai:gpt-4o-2024-08-06"
LLM_TEMPERATURE = 0.7
LLM_STREAMING = True

# Initialize LLMs for each agent
# Function to initialize LLM 
def get_llm(temperature=LLM_TEMPERATURE, streaming=LLM_STREAMING, callbacks=None):
    if callbacks is None:
        callbacks = []
    
    # Fall back to standard initialization
    return init_chat_model(
        MODEL_IDENTIFIER,
        temperature=temperature,
        streaming=streaming,
        model_provider="openai",
        callbacks=callbacks
    )

# Analytics Agent
analytics_agent_llm = init_chat_model(
    MODEL_IDENTIFIER,
    temperature=LLM_TEMPERATURE,
    model_kwargs={"streaming": LLM_STREAMING}
)
analytics_agent_tools_list = data_analysis_tools() 
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

3. predict_trend 도구 사용:
   - 사용자가 "향후 3개월 동안의 매출을 예측해 주세요"라고 요청하면 → predict_trend(data_description="매출 데이터", time_horizon="3 months") 호출"""

# RAG Agent
rag_agent_llm = init_chat_model(
    MODEL_IDENTIFIER,
    temperature=LLM_TEMPERATURE,
    model_kwargs={"streaming": LLM_STREAMING}
)
rag_agent_tools_list = document_processing_tools()
rag_agent_system_prompt_string = """당신은 문서 처리와 지식 검색 전문가입니다.
문서 요약, 정보 추출, 질문 응답, 문서 변환과 같은 문서 관련 작업을 수행합니다.
PDF, TXT, DOCX 등 다양한 형식의 문서를 처리할 수 있습니다.
필요한 정보를 정확하고 빠르게 찾아 제공합니다.

# 도구 사용 시나리오:
1. summarize_document 도구 사용:
   - 사용자가 "너무 긴 이메일인데 짧게 요약해 줄래요?"라고 요청하면 → summarize_document(document_content="[이메일 내용]", max_length=200) 호출

2. extract_information 도구 사용:
   - 사용자가 "이 문서에서 모든 날짜를 추출해주세요"라고 요청하면 → extract_information(document_content="[문서 내용]", info_type="dates") 호출

3. answer_document_question 도구 사용:
   - 사용자가 "이 문서에 따르면 향후 사용자 수 예측은 어떻게 되나요?"라고 물으면 → answer_document_question(document_content="[문서 내용]", question="향후 사용자 수 예측은 어떻게 되나요?") 호출"""

# Code Agent
code_agent_llm = init_chat_model(
    MODEL_IDENTIFIER,
    temperature=LLM_TEMPERATURE,
    model_kwargs={"streaming": LLM_STREAMING}
)
code_agent_tools_list = code_agent_tools()
code_agent_system_prompt_string = """당신은 코딩 전문가이자 숙련된 개발자입니다.
코드 작성, 디버깅, 일반적인 프로그래밍 질문에 대한 답변, 기술적 조언 제공 등 다양한 개발 관련 작업을 수행합니다.
여러 프로그래밍 언어와 프레임워크에 대한 지식을 갖추고 있습니다.
다른 전문가의 도움이 필요한 경우 적절하게 전환하세요.

# 도구 사용 시나리오:
1. search_information 도구 사용:
   - 사용자가 "파이썬에서 비동기 프로그래밍은 어떻게 하나요?"라고 물으면 → search_information(query="파이썬 비동기 프로그래밍 방법") 호출

2. get_recommendations 도구 사용:
   - 사용자가 "AI 개발에 필요한 기술 스택을 추천해주세요"라고 요청하면 → get_recommendations(category="tech_stack", preferences="AI 개발") 호출

3. track_conversation 도구 사용:
   - 사용자의 중요한 선호도를 기록할 때 → track_conversation(current_agent_name="code_agent", note="사용자는 파이썬 기반 데이터 분석에 관심이 있음") 호출"""

# Supervisor LLM - lower temperature for more deterministic routing
supervisor_llm_instance = init_chat_model(
    MODEL_IDENTIFIER,
    temperature=LLM_TEMPERATURE,
    model_kwargs={"streaming": LLM_STREAMING}
)
supervisor_system_prompt_text = (
    "You are a supervisor of multiple AI agents. "
    "You receive user requests and determine which agent is best suited to handle them. "
    "The agents are:\n"
    "- 'analytics_agent': Data analysis, visualization, statistics, prediction.\n"
    "- 'rag_agent': Document summarization, information extraction, document-based Q&A.\n"
    "- 'code_agent': Code writing, debugging, general questions, coding-related questions, advice, code snippets, programming concepts, etc.\n"
    "Please respond in Korean, and the agents will respond in Korean as well. "
    "If the request is unclear or requires additional information, you can ask the user a question. "
    "When you decide to finish the conversation (using 'FINISH'), you MUST first provide a complete summary of all information the agents have provided, "
    "including all outputs, code examples, and analysis results. DO NOT just say 'task completed' or refer to examples without showing them. "
    "Your final response should contain ALL the information the user needs, exactly as provided by the agents. "
    "Agent switching should be done using the exact name of the agent (e.g. 'analytics_agent'). "
    "If the request is a greeting, a simple question that doesn't require a specialized agent, "
    "you can answer it directly and respond with 'FINISH'."
)

# Initialize the agent runnables
analytics_agent_runnable = create_react_agent(
    model=analytics_agent_llm,
    tools=analytics_agent_tools_list,
    prompt=analytics_agent_system_prompt_string,
    name="analytics_agent"
)
print("Analytics Agent Runnable created successfully")

rag_agent_runnable = create_react_agent(
    model=rag_agent_llm,
    tools=rag_agent_tools_list,
    prompt=rag_agent_system_prompt_string,
    name="rag_agent"
)
print("RAG Agent Runnable created successfully")

code_agent_runnable = create_react_agent(
    model=code_agent_llm,
    tools=code_agent_tools_list,
    prompt=code_agent_system_prompt_string,
    name="code_agent"
)
print("Code Agent Runnable created successfully")
# --- End Constants and LLM/Agent Setup --- #


# 1. Define Supervisor State
class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_node: str

# 2. Define Nodes
async def supervisor_router_node(state: SupervisorState, config: RunnableConfig):
    print("--- SUPERVISOR ROUTER ---")
    
    # Extract user messages and agent messages separately
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    agent_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage) and msg.content and not msg.content.lower().startswith("supervisor:")]
    
    # Prepare the complete prompt with system instructions, user input, and agent responses
    prompt_messages = [SystemMessage(content=supervisor_system_prompt_text)] + list(state["messages"])
    
    # For finishing, add an extra instruction to include agent outputs
    if len(agent_messages) > 0:
        agent_outputs_summary = "\n\n지금까지 에이전트들이 제공한 정보:\n"
        for i, msg in enumerate(agent_messages):
            agent_type = "알 수 없는 에이전트"
            if "analytics_agent" in str(msg.tags) if hasattr(msg, 'tags') else "":
                agent_type = "분석 에이전트"
            elif "rag_agent" in str(msg.tags) if hasattr(msg, 'tags') else "":
                agent_type = "문서 에이전트"
            elif "code_agent" in str(msg.tags) if hasattr(msg, 'tags') else "":
                agent_type = "코드 에이전트"
            
            agent_outputs_summary += f"\n--- {agent_type} 응답 {i+1} ---\n{msg.content}\n"
        
        # Add agent outputs as a system message to ensure it's included in the summary
        prompt_messages.append(SystemMessage(content=f"{agent_outputs_summary}\n\n위의 모든 에이전트 정보를 바탕으로 최종 응답을 생성하세요. 'FINISH'로 대화를 마칠 경우, 반드시 위 정보를 모두 포함한 완전한 요약을 제공하세요."))
    
    # Get response from the supervisor LLM
    response_ai_message = await supervisor_llm_instance.ainvoke(prompt_messages, config=config)
    
    # Determine next node based on response
    next_agent_name = "FINISH" # Default to FINISH
    response_content = response_ai_message.content.lower() # Case-insensitive matching

    # More robust parsing for agent names
    if "analytics_agent" in response_content:
        next_agent_name = "analytics_agent"
    elif "rag_agent" in response_content:
        next_agent_name = "rag_agent"
    elif "code_agent" in response_content:
        next_agent_name = "code_agent"
    elif "finish" in response_content: # Explicitly check for FINISH
        next_agent_name = "FINISH"
        
        # For FINISH, ensure the response includes agent information by prepending "Supervisor Summary:" 
        # and make sure it hasn't been truncated by checking length
        if len(response_ai_message.content) < 100 and len(agent_messages) > 0:
            # If response is too short and we have agent messages, force a more comprehensive summary
            response_ai_message = AIMessage(content=f"슈퍼바이저 요약:\n\n{response_ai_message.content}\n\n---에이전트 제공 정보 요약---\n{agent_outputs_summary}")
    
    print(f"Supervisor decision: {next_agent_name}, Full response: {response_ai_message.content[:200]}...")
    return {
        "messages": [response_ai_message],
        "next_node": next_agent_name
    }

async def agent_node_wrapper(state: SupervisorState, agent_runnable, agent_name: str, config: RunnableConfig):
    print(f"--- AGENT: {agent_name} ---")
    initial_messages_count = len(state["messages"])
    
    # Agents created by create_react_agent expect input like {"messages": ...} 
    # and output {"messages": updated_message_list}
    agent_result = await agent_runnable.ainvoke({"messages": state["messages"]}, config=config)
    
    updated_messages_from_agent = agent_result["messages"]
    
    # Extract only the new messages added by the agent to avoid duplication by add_messages
    new_messages = updated_messages_from_agent[initial_messages_count:]
    
    return {"messages": new_messages}


# 3. Define Conditional Edges Logic
def determine_next_node(state: SupervisorState):
    print(f"--- Determining Next Node based on Supervisor's decision: {state['next_node']} ---")
    return state["next_node"]

# 4. Construct the StateGraph
workflow = StateGraph(SupervisorState)

workflow.add_node("supervisor_router", supervisor_router_node)
workflow.add_node("analytics_agent", analytics_agent_runnable)
workflow.add_node("rag_agent", rag_agent_runnable)
workflow.add_node("code_agent", code_agent_runnable)

workflow.set_entry_point("supervisor_router")

workflow.add_conditional_edges(
    "supervisor_router",
    determine_next_node,
    {
        "analytics_agent": "analytics_agent",
        "rag_agent": "rag_agent",
        "code_agent": "code_agent",
        "FINISH": END
    }
)

workflow.add_edge("analytics_agent", "supervisor_router")
workflow.add_edge("rag_agent", "supervisor_router")
workflow.add_edge("code_agent", "supervisor_router")

# 5. Compile the graph
# Checkpointer can be added here if persistence is needed
# from langgraph.checkpoint.sqlite import SqliteSaver
# memory = SqliteSaver.from_conn_string(":memory:")
# app = workflow.compile(checkpointer=memory)
app = workflow.compile()
print("StateGraph-based supervisor compiled successfully.")

# 6. Main Execution Logic
async def main():
    print("Initializing StateGraph-based multi-agent system...")

    # Implement MCP tool fetching if needed, similar to what was done in my_langraph_agent/src/agent/graph.py
    try:
        # Initialize any special tools like MCP tools that need async initialization
        # Similar to how the my_langraph_agent initializes MCP tools
        print("Checking for MCP tools...")
        # This would be where you'd fetch MCP tools like in the original implementation
        # Example: code_agent_tools_list.extend(await get_mcp_tools())
    except Exception as e:
        print(f"Warning: Could not initialize MCP tools: {e}")

    # Conversation loop with thread tracking for checkpointing
    thread_id_counter = 0
    while True:
        user_input = input("사용자: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting conversation.")
            break

        thread_id_counter += 1
        current_thread_id = f"thread_{thread_id_counter}"
        
        # For a checkpointer-enabled graph, add thread_id to enable conversation persistence
        inputs = {"messages": [HumanMessage(content=user_input)]}
        config = {"recursion_limit": 25}  # Default is 25, can adjust
        
        # When using a checkpointer, add thread_id to enable conversation persistence
        config["configurable"] = {"thread_id": current_thread_id}

        print(f"\n--- Invoking graph for input: '{user_input[:50]}...' ---")
        
        # Use astream_events for detailed streaming feedback
        async for event in app.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]
            
            if kind == "on_chat_model_stream":
                # Stream chunks from the model
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="", flush=True)
            elif kind == "on_tool_start":
                # Show when a tool starts executing
                print(f"\n--- Executing Tool: {event['name']} ---")
                print(f"    Args: {event['data'].get('input')}")
            elif kind == "on_tool_end":
                # Show tool execution results (truncated)
                tool_output = str(event['data'].get('output'))
                print(f"--- Tool End: {event['name']} --- Output (first 100 chars): {tool_output[:100]} ---")
            elif kind == "on_chain_start":
                # Show when a new agent takes over
                if "agent" in event.get("tags", []):
                    agent_name = next((tag for tag in event.get("tags", []) if tag in ["analytics_agent", "rag_agent", "code_agent", "supervisor_router"]), "unknown")
                    print(f"\n=== Agent {agent_name} is processing the request ===\n")
            # Skip verbose events to reduce noise
            # elif kind == "on_chain_end" or kind == "on_llm_end":
            #    pass

        print("\n---------------------")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nUser interrupted the process.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
