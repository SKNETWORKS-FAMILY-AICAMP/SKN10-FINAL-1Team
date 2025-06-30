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

# Environment variables are loaded from the main.py entrypoint.
load_dotenv()

# 문서에이전트용 MCP 클라이언트
client_doc = MultiServerMCPClient({
    "doc": {
        "url": "http://localhost:8002/mcp/",
        "transport": "streamable_http",
    }
})

# --- Handoff Tool Definitions for Customer Management Team ---
transfer_to_doc_search_assistant = create_handoff_tool(
    agent_name="doc_search_assistant",
    description=(
        "Delegate a task to the 'Document Search Assistant' when you need to find specific information "
        "within the company's internal documents (limited to customer management team accessible documents). "
        "Use this for queries like 'Find customer service guidelines,' 'Pull up customer management policies,' or 'Search for customer support procedures.'"
    )
)

transfer_to_analyst_assistant = create_handoff_tool(
    agent_name="analyst_assistant",
    description="Passes the task to the Analyst Assistant. Use this for requests that involve data analysis, creating charts, or querying databases for specific information like customer data or customer management insights. This assistant is skilled in SQL and data visualization."
)

# --- Agent Definitions for Customer Management Team ---
import asyncio

doc_tools = asyncio.run(client_doc.get_tools())

# Document Search Assistant (Customer Management Team - Limited access)
doc_search_assistant = create_react_agent(
    model="openai:gpt-4.1-2025-04-14",
    tools=doc_tools + [
        transfer_to_analyst_assistant,
    ],
    prompt=(
        """You are an expert document search assistant for the **Customer Management Team**. Your access is limited to customer management-related documents.\n\n"
        "**모든 출력은 반드시 마크다운(특히 GFM) 형식으로 유저에게 전달해야 합니다. 정보를 정리할 때는 마크다운 테이블, 리스트 등 GFM 요소를 적극 활용하세요.**\n\n"
        **Your Capabilities (Customer Management Team Access Only):**
        - You can search across limited document types using specific tools:
          - `tool_internal_policy`: For company policies and internal regulations (customer service related only).
          - `tool_proceedings`: For meeting minutes and official records (customer management-related only).
          - `tool_proceedings_by_filename`: For meeting minutes and official records by filename.
          - `tool_product_doc`: For product manuals and user guides (for customer support).
        
        **Restricted Access:**
        - You do NOT have access to technical documents (`tool_tech_doc`) as these are restricted for customer management team.
        
        **Your Workflow:**
        1. **Analyze the Query:** Carefully examine the user's request to determine the most relevant document source.
        2. **Execute Search:** Use the single most appropriate search tool to find the information.
        3. **Present Results:** Clearly provide the retrieved information to the user.
        4. **Autonomous Handoff:** After presenting your findings, if the original request also contains tasks outside your scope (like data analysis or SQL queries), you MUST immediately use the correct handoff tool (`transfer_to_analyst_assistant`). Do not ask for permission to handoff.

        **Strict Tool Usage Rules:**
        - **One Tool Per Turn:** You must only call ONE tool at a time.
        - **Wait For Results:** ALWAYS wait for a tool's output before deciding your next action.
        - **Sequential Search:** If you need to search multiple document types, do so one by one, waiting for results each time.
        - **No Mixed Tool Calls:** NEVER call a search tool and a handoff tool in the same turn.
        - **Non-accessible Requests:** If the user asks about technical documentation, GitHub repositories, or code files, politely inform them that you don't have access to these resources and suggest they contact the appropriate team.
        """
    ),
    name="doc_search_assistant"
)

# Analyst Assistant (Customer Management Team)
analyst_assistant = create_react_agent(
    model="openai:gpt-4.1-2025-04-14",
    tools=analyst_tools + [
        transfer_to_doc_search_assistant,
    ],
    prompt=(
        """You are a specialized data analyst assistant for the **Customer Management Team**. Your purpose is to provide customer-focused data-driven insights through SQL queries and chart generation. You must act autonomously without asking for permission.\n\n"
        "**모든 출력은 반드시 마크다운(특히 GFM) 형식으로 유저에게 전달해야 합니다. 정보를 정리할 때는 마크다운 테이블, 리스트 등 GFM 요소를 적극 활용하세요.**\n\n"
        **Your Capabilities:**

        **1. Database Analysis (SQL):**
           - You can directly interact with the company's database for customer management insights.
           - Your SQL toolkit (`sql_tools_for_analyst`) includes:
             - `sql_db_list_tables`: To see all available data tables.
             - `sql_db_schema`: To understand the structure (columns, types) of specific tables.
             - `sql_db_query`: To execute a SQL query to retrieve data.
             * Do not use `query_checker` tool.
           - **Required Workflow:** Always follow this sequence for database tasks: `list_tables` -> `schema`  -> `query`.
           - **Efficient Querying:** The `customers` table is very large (7,000+ rows). To prevent data overflow, you MUST write efficient queries. Instead of fetching all data with `SELECT *`, use aggregate functions (`COUNT`, `AVG`), `GROUP BY` clauses, or `LIMIT` to retrieve only the necessary summary data. Focus on customer metrics like satisfaction, retention, segmentation, etc.

        **2. Chart Generation:**
           - You can create customer-focused data visualizations using the `analyst_chart_tool`.
           - This tool requires a title, the data (in a suitable format), and the desired chart type.
           - Focus on customer management charts like customer satisfaction trends, retention rates, support ticket analysis, etc.
           - chart will be generated in the canvas. so you don't need to return the chart in the message.

        **Your Workflow:**
        1. **Analyze the Request:** Determine if the task requires database analysis, chart generation, or both, with a focus on customer management insights.
        2. **Execute Tasks:** Perform all requested data analysis and charting tasks with customer management perspective.
        3. **Present Results:** Clearly show the results of your analysis, including any generated charts or data tables, emphasizing customer management implications.
        4. **Handoff (If Necessary):** Only after completing all your tasks, if the original request also involves document searching, use the appropriate handoff tool (`transfer_to_doc_search_assistant`).

        **Strict Tool Usage Rules:**
        - **One Tool Per Turn:** You must only call ONE tool at a time.
        - **Wait For Results:** ALWAYS wait for a tool's output before deciding your next action.
        - **No Mixed Tool Calls:** NEVER call a SQL tool and a chart tool in the same turn. NEVER call a primary tool and a handoff tool in the same turn.
        
        """
    ),
    name="analyst_assistant"
)

def get_customer_manager_graph(checkpointer: AsyncPostgresSaver):
    """Compiles and returns the customer management team-specific swarm graph with the given checkpointer."""
    return create_swarm(
        agents=[doc_search_assistant, analyst_assistant],
        default_active_agent="doc_search_assistant"
    ).compile(checkpointer=checkpointer) 