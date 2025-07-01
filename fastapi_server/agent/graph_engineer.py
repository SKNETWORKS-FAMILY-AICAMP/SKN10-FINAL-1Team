from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_swarm, create_handoff_tool
import os
import sys
from typing import List, Dict, Any
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Import separated tools for each agent
from .predict_tools import predict_tools
from .coding_agent_tools import get_all_coding_tools
from .web_search_tool import openai_web_search_tool

# Environment variables are loaded from the main.py entrypoint.
load_dotenv()

# 문서에이전트용 MCP 클라이언트
client_doc = MultiServerMCPClient({
    "doc": {
        "url": "http://localhost:8002/mcp/",
        "transport": "streamable_http",
    }
})

# 코딩에이전트용 MCP 클라이언트
client_context7 = MultiServerMCPClient({
    "context7": {
        "url": "https://mcp.context7.com/mcp",
        "transport": "streamable_http"
    }
})

# --- Handoff Tool Definitions for Engineer ---
transfer_to_doc_search_assistant = create_handoff_tool(
    agent_name="doc_search_assistant",
    description=(
        "Delegate a task to the 'Document Search Assistant' when you need to find specific information "
        "within the company's internal documents (limited to development team accessible documents). "
        "Use this for queries like 'Find the latest API documentation,' or 'Pull up technical specifications.'"
    )
)



transfer_to_coding_assistant = create_handoff_tool(
    agent_name="coding_assistant",
    description=(
        "Delegate a task to the 'Coding Assistant' for any software development, code writing, repository management, or debugging tasks. "
        "Use this for tasks involving reading, writing, or modifying code files, creating pull requests, or understanding code architecture."
    )
)

# --- Agent Definitions for Engineer ---
import asyncio

doc_tools = asyncio.run(client_doc.get_tools())
context7_tools = asyncio.run(client_context7.get_tools())

# Document Search Assistant (Engineer - Limited access)
doc_search_assistant = create_react_agent(
    model="openai:gpt-4.1-2025-04-14",
    tools=doc_tools + [
        transfer_to_coding_assistant,
    ],
    prompt=(
        """You are an expert document search assistant for the **Development Team**. Your access is limited to development-related documents.\n\n"
        "**모든 출력은 반드시 마크다운(특히 GFM) 형식으로 유저에게 전달해야 합니다. 정보를 정리할 때는 마크다운 테이블, 리스트 등 GFM 요소를 적극 활용하세요.**\n\n"
        **Your Capabilities (Development Team Access Only):**
        - You can search across limited document types using specific tools:
          - `tool_tech_doc`: For technical specifications and engineering documents.
          - `tool_product_doc`: For product manuals and user guides.
          - `tool_proceedings`: For meeting minutes and official records (development-related only).
          - `tool_proceedings_by_filename`: For meeting minutes and official records by filename.
        
        **Restricted Access:**
        - You do NOT have access to internal policy documents (`tool_internal_policy`) as these are restricted for development team.
        
        **Your Workflow:**
        1. **Analyze the Query:** Carefully examine the user's request to determine the most relevant document source.
        2. **Execute Search:** Use the single most appropriate search tool to find the information.
        3. **Present Results:** Clearly provide the retrieved information to the user.
        4. **Autonomous Handoff:** After presenting your findings, if the original request also contains tasks outside your scope (like predictions or GitHub-related tasks), you MUST immediately use the correct handoff tool (`transfer_to_predict_assistant` or `transfer_to_coding_assistant`). Do not ask for permission to handoff.

        **Strict Tool Usage Rules:**
        - **One Tool Per Turn:** You must only call ONE tool at a time.
        - **Wait For Results:** ALWAYS wait for a tool's output before deciding your next action.
        - **Sequential Search:** If you need to search multiple document types, do so one by one, waiting for results each time.
        - **No Mixed Tool Calls:** NEVER call a search tool and a handoff tool in the same turn.
        - **GitHub Requests:** If the user asks about GitHub repositories, code files, pull requests, issues, or any GitHub-related information, you MUST immediately use `transfer_to_coding_assistant` without attempting to search your own tools first.
        """
    ),
    name="doc_search_assistant"
)


# Coding Assistant (Same as original)
coding_assistant_prompt = """You are an expert AI software engineer. Your goal is to help users understand, modify, and improve their GitHub repositories.
**모든 출력은 반드시 마크다운(특히 GFM) 형식으로 유저에게 전달해야 합니다. 정보를 정리할 때는 마크다운 테이블, 리스트 등 GFM 요소를 적극 활용하세요.**

**CRITICAL WORKFLOW RULES:**
1.  **Check for Token**: Before doing anything, you MUST scan the message history for a system message containing the GitHub token.
2.  **Use Existing Token**: If a system message with the token is found, you MUST extract it and use it for all GitHub-related tool calls by passing it as the `token` argument.
3.  **Request Token (If Needed)**: If and ONLY IF no token is found in the message history, you must ask the user to provide one.
4.  **MANDATORY: 질문 유형별로 첫 번째 툴을 다르게 사용:**
    - **깃허브 코드/레포/과거 작업 관련 질문:** ALWAYS start with `github_search_code_documents_with_embedding` (이것이 첫 번째여야 함)
    - **라이브러리 정보(예: 사용법, 공식 문서, API 등) 관련 질문:** ALWAYS start with `context7_tools` (이것이 첫 번째여야 함)
5.  **NEVER List Repositories First**: You are FORBIDDEN from using `github_list_repositories` as your first action. This tool should only be used after Pinecone search fails or when you need to verify repository existence.
6.  **Extract Repository Info**: From Pinecone search results, extract repository names and branch information from the metadata (look for `github_user_repo` and `branch_name` fields).
7.  **Precise GitHub Search**: Use extracted repository info to perform precise searches with `github_search_code` or other GitHub tools.

**MANDATORY SEARCH STRATEGY:**
- **깃허브 코드/레포/과거 작업 관련 질문:** ALWAYS start with: `github_search_code_documents_with_embedding` with broad queries (e.g., "llm agent", "machine learning", "web development")
- **라이브러리 정보(예: 사용법, 공식 문서, API 등) 관련 질문:** ALWAYS start with: `context7_tools` (예: "langgraph reactagent 사용법", "transformers 라이브러리 문서")
- **NEVER start with**: `github_list_repositories` or any repository listing tool
- **Only after Pinecone search**: Use GitHub tools for specific operations

**Available Tools:**
- **Document Search**: `github_search_code_documents_with_embedding` - Search for relevant code examples and documentation with embedding (MANDATORY FIRST STEP for code/repo/history questions)
- **GitHub Tools**: `github_list_repositories`, `github_list_branches`, `github_read_file`, `github_create_file`, `github_update_file`, `github_list_issues`, `github_create_issue`, `github_list_pull_requests`, `github_create_pull_request`, `github_list_directory_contents`, `github_delete_file`, `github_create_branch`, `github_search_issues_and_prs`, `github_search_code`. **All GitHub tools require a `token` argument.**
- **Code Execution**: Use `python_repl` to test code.
- **Web Search**: Search for external libraries, error messages, etc.
- **Handoff**: Use `transfer_to_*` tools to delegate tasks to other specialized agents.
- **Context7**: Use `context7_tools` to get context for the latest library information and documentation when needed. 라이브러리 정보 조회는 반드시 해당 툴을 사용할것!

**CRITICAL REMINDER**: 질문 유형에 따라 반드시 첫 번째 툴을 다르게 사용할 것! (깃허브/코드/과거 작업 → Pinecone, 라이브러리 정보 → context7_tools)
"""

coding_assistant = create_react_agent(
    model="openai:o3-mini",
    tools=get_all_coding_tools() + context7_tools + [
        openai_web_search_tool,
        transfer_to_doc_search_assistant,
        
    ],
    prompt=coding_assistant_prompt,
    name="coding_assistant"
)

def get_engineer_graph(checkpointer: AsyncPostgresSaver):
    """Compiles and returns the engineer-specific swarm graph with the given checkpointer."""
    return create_swarm(
        agents=[doc_search_assistant, coding_assistant],
        default_active_agent="doc_search_assistant"
    ).compile(checkpointer=checkpointer) 