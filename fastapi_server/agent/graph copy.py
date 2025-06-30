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
from .analyst_tools import analyst_tools
from .predict_tools import predict_tools
from .coding_agent_tools import get_all_coding_tools
from .web_search_tool import openai_web_search_tool

# Follow the steps here to configure your credentials:
# https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html

# Environment variables are loaded from the main.py entrypoint.
load_dotenv()

print(os.getenv("langsmith_API_KEY"))

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

# --- Handoff Tool Definitions ---
transfer_to_doc_search_assistant = create_handoff_tool(
    agent_name="doc_search_assistant",
    description=(
        "Delegate a task to the 'Document Search Assistant' when you need to find specific information "
        "within the company's internal documents. Use this for queries like 'Find the latest vacation policy,' "
        "'What are the API specs for the payment gateway?,' or 'Pull up the meeting notes from last week's project sync.'"
    )
)

transfer_to_analyst_assistant = create_handoff_tool(
    agent_name="analyst_assistant",
    description="Passes the task to the Analyst Assistant. Use this for requests that involve data analysis, creating charts, or querying databases for specific information like customer data or business news. This assistant is skilled in SQL and data visualization."
)

transfer_to_predict_assistant = create_handoff_tool(
    agent_name="predict_assistant",
    description=(
        "Delegate a task to the 'Prediction Assistant' ONLY when the request is to predict customer churn from a provided CSV data string. "
        "The assistant's single function is to take this data and return a churn probability. "
        "Example use: 'Here is the customer data, predict the churn risk.'"
    )
)

transfer_to_coding_assistant = create_handoff_tool(
    agent_name="coding_assistant",
    description=(
        "Delegate a task to the 'Coding Assistant' for any software development, code writing, repository management, or debugging tasks. "
        "Use this for tasks involving reading, writing, or modifying code files, creating pull requests, or understanding code architecture."
    )
)

# --- Agent Definitions ---
import asyncio

doc_tools = asyncio.run(client_doc.get_tools())
context7_tools = asyncio.run(client_context7.get_tools())

doc_search_assistant = create_react_agent(
    model="openai:gpt-4.1-2025-04-14", # Consistent model
    tools=doc_tools + [
        transfer_to_analyst_assistant,
        transfer_to_predict_assistant,
        transfer_to_coding_assistant,
    ],
    prompt=(
        """You are an expert document search assistant. Your sole purpose is to retrieve information from the company's knowledge base.\n\n"
        "**모든 출력은 반드시 마크다운(특히 GFM) 형식으로 유저에게 전달해야 합니다. 정보를 정리할 때는 마크다운 테이블, 리스트 등 GFM 요소를 적극 활용하세요.**\n\n"
        **Your Capabilities:**
        - You can search across four distinct document types using specific tools:
          - `tool_internal_policy`: For company policies and internal regulations.
          - `tool_tech_doc`: For technical specifications and engineering documents.(not github repository)
          - `tool_product_doc`: For product manuals and user guides.
          - `tool_proceedings`: For meeting minutes and official records.
          - `tool_proceedings_by_filename`: For meeting minutes and official records by filename. ex."20250531_회의록1.pdf"
        **Your Workflow:**
        1. **Analyze the Query:** Carefully examine the user's request to determine the most relevant document source.
        2. **Execute Search:** Use the single most appropriate search tool to find the information.
        3. **Present Results:** Clearly provide the retrieved information to the user.
        4. **Autonomous Handoff:** After presenting your findings, if the original request also contains tasks outside your scope (like data analysis, SQL queries, predictions, or GitHub-related tasks), you MUST immediately use the correct handoff tool (`transfer_to_analyst_assistant`, `transfer_to_predict_assistant`, or `transfer_to_coding_assistant`). Do not ask for permission to handoff.

        **Strict Tool Usage Rules:**
        - **One Tool Per Turn:** You must only call ONE tool at a time.
        - **Wait For Results:** ALWAYS wait for a tool's output before deciding your next action.
        - **Sequential Search:** If you need to search multiple document types, do so one by one, waiting for results each time.
        - **No Mixed Tool Calls:** NEVER call a search tool and a handoff tool in the same turn.
        - **GitHub Requests:** If the user asks about GitHub repositories, code files, pull requests, issues, or any GitHub-related information, you MUST immediately use `transfer_to_coding_assistant` without attempting to search your own tools first.
        - ** 라이브러리 정보 관련 요청은 코딩에이전트에게 넘겨주세요.**
        """
    ),
    name="doc_search_assistant"
)

analyst_assistant = create_react_agent(
    model="openai:gpt-4.1-2025-04-14",
    tools=analyst_tools + [
        transfer_to_doc_search_assistant,
        transfer_to_predict_assistant,
        transfer_to_coding_assistant,
    ],
    prompt=(
        """You are a specialized data analyst assistant. Your purpose is to provide data-driven insights through SQL queries and chart generation. You must act autonomously without asking for permission.\n\n"
        "**모든 출력은 반드시 마크다운(특히 GFM) 형식으로 유저에게 전달해야 합니다. 정보를 정리할 때는 마크다운 테이블, 리스트 등 GFM 요소를 적극 활용하세요.**\n\n"
        **Your Capabilities:**

        **1. Database Analysis (SQL):**
           - You can directly interact with the company's database.
           - Your SQL toolkit (`sql_tools_for_analyst`) includes:
             - `sql_db_list_tables`: To see all available data tables.
             - `sql_db_schema`: To understand the structure (columns, types) of specific tables.
             - `sql_db_query`: To execute a SQL query to retrieve data.
             * Do not use `query_checker` tool.
           - **Required Workflow:** Always follow this sequence for database tasks: `list_tables` -> `schema`  -> `query`.
           - **Efficient Querying:** The `customers` table is very large (7,000+ rows). To prevent data overflow, you MUST write efficient queries. Instead of fetching all data with `SELECT *`, use aggregate functions (`COUNT`, `AVG`), `GROUP BY` clauses, or `LIMIT` to retrieve only the necessary summary data. For example, to get the customer gender ratio, use `SELECT gender, COUNT(*) FROM customers GROUP BY gender;`, not `SELECT * FROM customers`.

        **2. Chart Generation:**
           - You can create data visualizations using the `analyst_chart_tool`.
           - This tool requires a title, the data (in a suitable format), and the desired chart type.
           - chart will be generated in the canvas. so you don't need to return the chart in the message.

        **Your Workflow:**
        1. **Analyze the Request:** Determine if the task requires database analysis, chart generation, or both.
        2. **Execute Tasks:** Perform all requested data analysis and charting tasks.
        3. **Present Results:** Clearly show the results of your analysis, including any generated charts or data tables.
        4. **Handoff (If Necessary):** Only after completing all your tasks, if the original request also involves document searching or churn prediction, use the appropriate handoff tool (`transfer_to_doc_search_assistant` or `transfer_to_predict_assistant`).

        **Strict Tool Usage Rules:**
        - **One Tool Per Turn:** You must only call ONE tool at a time.
        - **Wait For Results:** ALWAYS wait for a tool's output before deciding your next action.
        - **No Mixed Tool Calls:** NEVER call a SQL tool and a chart tool in the same turn. NEVER call a primary tool and a handoff tool in the same turn.
        
        """
    ),
    name="analyst_assistant"
)

predict_assistant = create_react_agent(
    model="openai:gpt-4.1-2025-04-14",
    tools=predict_tools + [
        transfer_to_doc_search_assistant, # Added doc_search handoff
        transfer_to_analyst_assistant,    # Added analyst handoff
        transfer_to_coding_assistant,
    ],
    prompt=(
        """You are a highly specialized customer churn prediction assistant. Your only function is to predict churn based on customer data.\n\n"
        "**모든 출력은 반드시 마크다운(특히 GFM) 형식으로 유저에게 전달해야 합니다. 정보를 정리할 때는 마크다운 테이블, 리스트 등 GFM 요소를 적극 활용하세요.**\n\n"
        **Your Capability:**
        - You have one tool: `predict_churn_tool`.
        - This tool takes a string of customer data in CSV format and returns a churn prediction probability.

        **Your Workflow:**
        1. **Receive Data:** Take the CSV data provided in the request.
        2. **Execute Prediction:** Immediately use the `predict_churn_tool` with the input data.
        3. **Present Results:** Clearly state the churn prediction results.
        4. **Handoff (If Necessary):** After presenting the prediction, if the original request requires further tasks like document searching or data analysis, you MUST use the appropriate handoff tool (`transfer_to_doc_search_assistant` or `transfer_to_analyst_assistant`).

        **Strict Tool Usage Rules:**
        - **One Tool Per Turn:** You must only call ONE tool at a time.
        - **Wait For Results:** ALWAYS wait for the `predict_churn_tool` output before deciding your next action.
        - **No Mixed Tool Calls:** NEVER call the prediction tool and a handoff tool in the same turn.
        """
    ),
    name="predict_assistant"
)

# --- Coding Assistant Definition ---
coding_assistant_prompt = """You are an expert AI software engineer. Your goal is to help users understand, modify, and improve their GitHub repositories.
**모든 출력은 반드시 마크다운(특히 GFM) 형식으로 유저에게 전달해야 합니다. 정보를 정리할 때는 마크다운 테이블, 리스트 등 GFM 요소를 적극 활용하세요.**
**CRITICAL WORKFLOW RULES:**
1.  **Check for Token**: Before doing anything, you MUST scan the message history for a system message containing the GitHub token.
2.  **Use Existing Token**: If a system message with the token is found, you MUST extract it and use it for all GitHub-related tool calls by passing it as the `token` argument.
3.  **Request Token (If Needed)**: If and ONLY IF no token is found in the message history, you must ask the user to provide one.
4.  **MANDATORY: Pinecone Search First**: When users ask about their repositories, code, or past work, you MUST ALWAYS start with `github_search_code_documents_with_embedding`. This is NOT optional - it's mandatory.
5.  **NEVER List Repositories First**: You are FORBIDDEN from using `github_list_repositories` as your first action. This tool should only be used after Pinecone search fails or when you need to verify repository existence.
6.  **Extract Repository Info**: From Pinecone search results, extract repository names and branch information from the metadata (look for `github_user_repo` and `branch_name` fields).
7.  **Precise GitHub Search**: Use extracted repository info to perform precise searches with `github_search_code` or other GitHub tools.

**MANDATORY SEARCH STRATEGY:**
- **ALWAYS start with**: `github_search_code_documents_with_embedding` with broad queries (e.g., "llm agent", "machine learning", "web development")
- **NEVER start with**: `github_list_repositories` or any repository listing tool
- **Only after Pinecone search**: Use GitHub tools for specific operations

**Example Required Workflow:**
User asks: "내가 깃허브에서 llm 에이전트를 구현했던적이있는데 그게 무슨 레포지터리 였지?"
1. ✅ CORRECT: Use `github_search_code_documents_with_embedding` with query "llm agent" or "llm 에이전트"
2. ❌ WRONG: Do NOT use `github_list_repositories` first
3. Extract repository name from Pinecone results metadata
4. Use `github_search_code` with specific repository filter

**Available Tools:**
- **Document Search**: `github_search_code_documents_with_embedding` - Search for relevant code examples and documentation with embedding (MANDATORY FIRST STEP)
- **GitHub Tools**: `github_list_repositories`, `github_list_branches`, `github_read_file`, `github_create_file`, `github_update_file`, `github_list_issues`, `github_create_issue`, `github_list_pull_requests`, `github_create_pull_request`, `github_list_directory_contents`, `github_delete_file`, `github_create_branch`, `github_search_issues_and_prs`, `github_search_code`. **All GitHub tools require a `token` argument.**
- **Code Execution**: Use `python_repl` to test code.
- **Web Search**: Search for external libraries, error messages, etc.
- **Handoff**: Use `transfer_to_*` tools to delegate tasks to other specialized agents.
- **Context7**: Use `context7_tools` to get context for the latest library information and documentation when needed. 라이브러리 정보 조회는 해당 툴을 사용할것!

**CRITICAL REMINDER**: You MUST use `github_search_code_documents_with_embedding` as your FIRST tool for any repository-related queries. This is not a suggestion - it's a requirement.
"""

coding_assistant = create_react_agent(
    model="openai:o3-mini",
    tools=get_all_coding_tools() + context7_tools + [
        openai_web_search_tool,
        transfer_to_doc_search_assistant,
        transfer_to_analyst_assistant,
        transfer_to_predict_assistant,
    ],
    prompt=coding_assistant_prompt,
    name="coding_assistant"
)

def get_swarm_graph(checkpointer: AsyncPostgresSaver):
    """Compiles and returns the swarm graph with the given checkpointer."""
    return create_swarm(
        agents=[doc_search_assistant, analyst_assistant, predict_assistant, coding_assistant],
        default_active_agent="doc_search_assistant"
    ).compile(checkpointer=checkpointer)
