from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_swarm, create_handoff_tool
import os
import sys
from typing import List, Dict, Any # Added Type for Pinecone tools
from pydantic import BaseModel, Field # For tool input schema
from openai import OpenAI
from pinecone import Pinecone as PineconeClient # Renamed to avoid conflict
from langchain_core.tools import Tool
from .tools import analyst_chart_tool, predict_churn_tool, sql_tools_for_analyst # Import chart, churn, and SQL tools
from .coding_agent_tools import get_all_coding_tools # Import coding tools
from .web_search_tool import openai_web_search_tool # Import new web search tool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv

# Environment variables are loaded from the main.py entrypoint.
load_dotenv()
print(os.getenv("langsmith_API_KEY"))
# --- Pinecone/OpenAI Client Initialization ---
def init_clients():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Environment variable OPENAI_API_KEY is not set.")
    openai_client = OpenAI(api_key=openai_api_key)

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV")
    if not pinecone_api_key or not pinecone_env:
        raise ValueError("Environment variables PINECONE_API_KEY or PINECONE_ENV are missing.")
    
    pc = PineconeClient(api_key=pinecone_api_key)
    index_name = os.getenv("PINECONE_INDEX_NAME", "dense-index")

    existing_indexes = [idx_spec['name'] for idx_spec in pc.list_indexes()]
    if index_name not in existing_indexes:
        raise ValueError(f"Index '{index_name}' does not exist in Pinecone. Current indexes: {existing_indexes}")

    pinecone_index = pc.Index(index_name)
    print(f"Successfully connected to Pinecone index '{index_name}'.", file=sys.stderr)
    return openai_client, pinecone_index

OPENAI_CLIENT, PINECONE_INDEX = None, None
try:
    OPENAI_CLIENT, PINECONE_INDEX = init_clients()
except ValueError as e:
    print(f"Error initializing clients: {e}", file=sys.stderr)
    # Allow graph to load but tools will fail if clients are needed.

# --- Embedding and Context Building Functions (for Pinecone tools) ---
def embed_query(text: str) -> List[float]:
    if not OPENAI_CLIENT:
        raise ValueError("OpenAI client not initialized for embedding.")
    resp = OPENAI_CLIENT.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp.data[0].embedding

def build_context_from_matches(matches: List[Dict[str, Any]]) -> str:
    contexts = []
    if not matches:
        return ""
    for m in matches:
        metadata = m.get("metadata", {})
        chunk_text = metadata.get("text", "")
        filename = metadata.get("original_filename", "Unknown")
        
        if chunk_text:
            context_entry = f"Source File: {filename}\nContent:\n{chunk_text}"
            contexts.append(context_entry)
    return "\n\n---\n\n".join(contexts)

# --- Pydantic Model for Search Tool Inputs ---
class SearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=3, description="Number of documents to retrieve")

# --- Pinecone Search Tool Functions ---
def _run_pinecone_search(query: str, namespace: str, top_k: int = 3) -> str:
    if not OPENAI_CLIENT or not PINECONE_INDEX:
        return "Error: OpenAI or Pinecone client not initialized."
    if not namespace:
        return "Error: Namespace not specified for Pinecone search."
    try:
        query_vector = embed_query(query)
        index_stats = PINECONE_INDEX.describe_index_stats()
        if namespace not in index_stats.namespaces or \
           index_stats.namespaces[namespace].vector_count == 0:
            return f"Namespace '{namespace}' not found in Pinecone or is empty."

        res = PINECONE_INDEX.query(
            vector=query_vector,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True
        )
        matches = res.get("matches", [])
        if not matches:
            return f"No relevant information found in namespace '{namespace}' for query: '{query}'."
        
        context = build_context_from_matches(matches)
        return context if context else "Could not extract context from search results."
    except Exception as e:
        return f"Error during Pinecone search in namespace '{namespace}': {e}"

def internal_policy_search(query: str, top_k: int = 3) -> str:
    """Searches internal company policies and HR documents (e.g., vacation policy, benefits, code of conduct)."""
    return _run_pinecone_search(query, namespace="internal_policy", top_k=top_k)

def tech_doc_search(query: str, top_k: int = 3) -> str:
    """Searches technical documents, development guides, and API specifications."""
    return _run_pinecone_search(query, namespace="technical_document", top_k=top_k)

def product_doc_search(query: str, top_k: int = 3) -> str:
    """Searches product manuals, feature descriptions, and user guides."""
    return _run_pinecone_search(query, namespace="product_document", top_k=top_k)

def proceedings_search(query: str, top_k: int = 3) -> str:
    """Searches meeting minutes, decisions, and work instructions."""
    return _run_pinecone_search(query, namespace="proceedings", top_k=top_k)

# --- LangChain Tool Objects for Pinecone Search ---
tool_internal_policy = Tool(
    name="InternalPolicySearch",
    func=internal_policy_search,
    description="Searches internal company policies and HR documents. Use for queries about vacation, benefits, code of conduct, etc.",
    args_schema=SearchInput
)

tool_tech_doc = Tool(
    name="TechnicalDocumentSearch",
    func=tech_doc_search,
    description="Searches technical documents, development guides, and API specifications. Use for technical questions, API usage, etc.",
    args_schema=SearchInput
)

tool_product_doc = Tool(
    name="ProductDocumentSearch",
    func=product_doc_search,
    description="Searches product manuals, feature descriptions, and user guides. Use for questions about product features or how to use a product.",
    args_schema=SearchInput
)

tool_proceedings = Tool(
    name="ProceedingsSearch",
    func=proceedings_search,
    description="Searches meeting minutes, decisions, and work instructions. Use for finding past decisions or discussion summaries.",
    args_schema=SearchInput
)

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
doc_search_assistant = create_react_agent(
    model="openai:gpt-4.1-2025-04-14", # Consistent model
    tools=[
        tool_internal_policy,
        tool_tech_doc,
        tool_product_doc,
        tool_proceedings,
        transfer_to_analyst_assistant,
        transfer_to_predict_assistant,
        transfer_to_coding_assistant,
    ],
    prompt=(
        """You are an expert document search assistant. Your sole purpose is to retrieve information from the company's knowledge base.

        **Your Capabilities:**
        - You can search across four distinct document types using specific tools:
          - `tool_internal_policy`: For company policies and internal regulations.
          - `tool_tech_doc`: For technical specifications and engineering documents.
          - `tool_product_doc`: For product manuals and user guides.
          - `tool_proceedings`: For meeting minutes and official records.

        **Your Workflow:**
        1. **Analyze the Query:** Carefully examine the user's request to determine the most relevant document source.
        2. **Execute Search:** Use the single most appropriate search tool to find the information.
        3. **Present Results:** Clearly provide the retrieved information to the user.
        4. **Autonomous Handoff:** After presenting your findings, if the original request also contains tasks outside your scope (like data analysis, SQL queries, or predictions), you MUST immediately use the correct handoff tool (`transfer_to_analyst_assistant` or `transfer_to_predict_assistant`). Do not ask for permission to handoff.

        **Strict Tool Usage Rules:**
        - **One Tool Per Turn:** You must only call ONE tool at a time.
        - **Wait For Results:** ALWAYS wait for a tool's output before deciding your next action.
        - **Sequential Search:** If you need to search multiple document types, do so one by one, waiting for results each time.
        - **No Mixed Tool Calls:** NEVER call a search tool and a handoff tool in the same turn.
        """
    ),
    name="doc_search_assistant"
)

analyst_assistant = create_react_agent(
    model="openai:gpt-4.1-2025-04-14",
    tools=[
        analyst_chart_tool,
        *sql_tools_for_analyst, # Unpack all SQL tools
        transfer_to_doc_search_assistant,
        transfer_to_predict_assistant,
        transfer_to_coding_assistant,
    ],
    prompt=(
        """You are a specialized data analyst assistant. Your purpose is to provide data-driven insights through SQL queries and chart generation. You must act autonomously without asking for permission.

        **Your Capabilities:**

        **1. Database Analysis (SQL):**
           - You can directly interact with the company's database.
           - Your SQL toolkit (`sql_tools_for_analyst`) includes:
             - `sql_db_list_tables`: To see all available data tables.
             - `sql_db_schema`: To understand the structure (columns, types) of specific tables.
             - `sql_db_query`: To execute a SQL query to retrieve data.
             - `sql_db_query_checker`: To validate the syntax of a SQL query before execution.
           - **Required Workflow:** Always follow this sequence for database tasks: `list_tables` -> `schema` -> `query_checker` -> `query`.
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
    tools=[
        predict_churn_tool,
        transfer_to_doc_search_assistant, # Added doc_search handoff
        transfer_to_analyst_assistant,    # Added analyst handoff
        transfer_to_coding_assistant,
    ],
    prompt=(
        """You are a highly specialized customer churn prediction assistant. Your only function is to predict churn based on customer data.

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

**Your Workflow:**
1.  **Check for Token**: Before doing anything, you MUST scan the message history for a system message containing the GitHub token.
2.  **Use Existing Token**: If a system message with the token is found, you MUST extract it and use it for all GitHub-related tool calls (e.g., `github_list_repositories`) by passing it as the `token` argument. Do not ask the user for it if it's already in a system message.
3.  **Request Token (If Needed)**: If and ONLY IF no token is found in the message history, you must ask the user to provide one. Example: "To perform this task, I need a GitHub Personal Access Token. Could you please provide one?"
4.  **Execute and Report**: Use the tools to perform the user's request and report the results clearly.

**Available Tools:**
- **GitHub Tools**: `github_list_repositories`, `github_list_branches`, `github_read_file`, `github_create_file`, `github_update_file`. **All GitHub tools require a `token` argument.**
- **Code Execution**: Use `python_repl` to test code.
- **Web Search**: Search for external libraries, error messages, etc.
- **Handoff**: Use `transfer_to_*` tools to delegate tasks to other specialized agents.
"""

coding_assistant = create_react_agent(
    model="openai:gpt-4.1",
    tools=get_all_coding_tools() + [
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
