"""
LangGraph Swarm with 4 ReAct Agents.
- PlannerAgent: Specializes in planning tasks.
- ResearcherAgent: Specializes in finding information.
- CoderAgent: Specializes in writing simple code snippets.
- CommunicatorAgent: Specializes in user interaction and summarizing.
"""

from __future__ import annotations

import os
from typing import Annotated

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm

# 0. 환경 변수에서 API 키 로드 (실제 사용 시 .env 파일 등에 설정)
# 예: os.environ["OPENAI_API_KEY"] = "your_api_key_here"
# 이 코드는 API 키가 환경 변수에 설정되어 있다고 가정합니다.

# 1. LLM 초기화
model = ChatOpenAI(model="gpt-4o", temperature=0) # 또는 gpt-3.5-turbo

# 2. 각 에이전트를 위한 간단한 도구 정의
@tool
def plan_task_tool(task_description: str) -> str:
    """Generates a simple plan for the given task description."""
    return f"Plan for '{task_description}': 1. Understand requirements. 2. Break down into steps. 3. Execute steps. 4. Review."

@tool
def search_info_tool(query: str) -> str:
    """Simulates searching for information based on a query."""
    if "LangGraph Swarm" in query:
        return "LangGraph Swarm allows multiple specialized agents to collaborate by dynamically handing off tasks."
    return f"Search results for '{query}': Information not found in this simple tool."

@tool
def write_code_tool(code_request: str) -> str:
    """Simulates writing a simple code snippet for the given request."""
    return f"```python\n# Code for: {code_request}\nprint('Hello from CoderAgent!')\n```"

@tool
def summarize_text_tool(text_to_summarize: str) -> str:
    """Provides a very simple summary of the given text."""
    return f"Summary: {text_to_summarize[:50]}..."

# 3. 에이전트 이름 정의
PLANNER_AGENT = "PlannerAgent"
RESEARCHER_AGENT = "ResearcherAgent"
CODER_AGENT = "CoderAgent"
COMMUNICATOR_AGENT = "CommunicatorAgent"

agent_names = [PLANNER_AGENT, RESEARCHER_AGENT, CODER_AGENT, COMMUNICATOR_AGENT]

# 4. 각 에이전트 생성
# Handoff tools
handoff_tools_planner = [create_handoff_tool(agent_name=name) for name in agent_names if name != PLANNER_AGENT]
handoff_tools_researcher = [create_handoff_tool(agent_name=name) for name in agent_names if name != RESEARCHER_AGENT]
handoff_tools_coder = [create_handoff_tool(agent_name=name) for name in agent_names if name != CODER_AGENT]
handoff_tools_communicator = [create_handoff_tool(agent_name=name) for name in agent_names if name != COMMUNICATOR_AGENT]


planner_agent = create_react_agent(
    model,
    tools=[plan_task_tool] + handoff_tools_planner,
    prompt=f"You are {PLANNER_AGENT}. Your role is to create plans for user requests. You can delegate execution to other agents if needed.",
    name=PLANNER_AGENT,
)

researcher_agent = create_react_agent(
    model,
    tools=[search_info_tool] + handoff_tools_researcher,
    prompt=f"You are {RESEARCHER_AGENT}. You are skilled at finding information. If you need to plan or code, handoff to the appropriate agent.",
    name=RESEARCHER_AGENT,
)

coder_agent = create_react_agent(
    model,
    tools=[write_code_tool] + handoff_tools_coder,
    prompt=f"You are {CODER_AGENT}. You write simple Python code snippets. For research or complex planning, handoff to other agents.",
    name=CODER_AGENT,
)

communicator_agent = create_react_agent(
    model,
    tools=[summarize_text_tool] + handoff_tools_communicator,
    prompt=f"You are {COMMUNICATOR_AGENT}. You are responsible for communicating with the user, clarifying requests, and summarizing results from other agents. You can handoff tasks to specialized agents.",
    name=COMMUNICATOR_AGENT,
)

all_agents = [planner_agent, researcher_agent, coder_agent, communicator_agent]

# 5. 스웜 생성
# CommunicatorAgent를 기본 활성 에이전트로 설정하여 사용자 요청을 먼저 받도록 합니다.
workflow = create_swarm(
    all_agents,
    default_active_agent=COMMUNICATOR_AGENT
)

# Compile the graph
app = workflow.compile()

# 7. 예제 실행 코드
if __name__ == "__main__":
    # 대화 ID 설정 (동일 ID로 호출하면 대화가 이어짐)
    conversation_id = "my-swarm-conversation-1"
    config = {"configurable": {"thread_id": conversation_id}}

    print("Starting Swarm Conversation (type 'exit' to quit)")
    print(f"Default active agent: {COMMUNICATOR_AGENT}")
    print("-" * 30)

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Exiting conversation.")
            break

        if not user_input.strip():
            continue

        # 스웜 호출
        try:
            response = app.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
            )
            # 응답에서 마지막 AI 메시지 추출 및 출력
            # create_swarm과 create_react_agent는 일반적으로 {'messages': [AIMessage(...)]} 형태의 딕셔너리를 반환합니다.
            if response and "messages" in response and response["messages"]:
                ai_message = response["messages"][-1]
                if hasattr(ai_message, 'content'):
                    print(f"AI: {ai_message.content}")
                else: # 일부 구버전 Langchain/LangGraph에서는 다를 수 있음
                    print(f"AI: {ai_message}")
            else:
                print("AI: (No message content found in response)")
            print("-" * 30)

        except Exception as e:
            print(f"Error invoking swarm: {e}")
            break




