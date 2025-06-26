"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict, List, Optional, Annotated

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langgraph.graph import StateGraph, END, add_messages

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI
import os
import sys
from typing import List, Dict, Any # Added Type for Pinecone tools
from pydantic import BaseModel, Field # For tool input schema
from openai import OpenAI
from pinecone import Pinecone as PineconeClient # Renamed to avoid conflict
from langchain_core.tools import Tool
from .tools import analyze_csv_with_churn_prediction, analyst_chart_tool, csv_churn_analyzer_tool, sql_tools_for_analyst # Import chart, churn, and SQL tools

import logging
from langgraph.checkpoint.memory import MemorySaver


# Environment variables are loaded from the main.py entrypoint.

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

# --- Agent Definitions ---
# Note: Handoff tools are removed as the supervisor will handle delegation.

# Define a shared LLM for all agents to ensure consistency and prevent errors.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

doc_search_assistant = create_react_agent(
    model=llm, # Using a consistent, powerful model
    tools=[
        tool_internal_policy,
        tool_tech_doc,
        tool_product_doc,
        tool_proceedings,
    ],
    prompt=(
        """You are an expert document search assistant. Your sole purpose is to retrieve information from the company's knowledge base.

        **Your Capabilities:**
        - You can search across four types of documents:
          1. **Internal Policies:** For HR, company rules, and administrative guidelines.
          2. **Technical Docs:** For API specifications, development guides, and engineering standards.
          3. **Product Docs:** For user manuals, feature descriptions, and customer-facing guides.
          4. **Meeting Proceedings:** For summaries of past meetings, decisions, and action items.

        **Your Workflow:**
        1. **Analyze the Request:** Determine which document type is most likely to contain the answer.
        2. **Execute Search:** Use the appropriate search tool (e.g., `InternalPolicySearch`).
        3. **Present Results:** Clearly provide the information found.
        
        **Strict Tool Usage Rules:**
        - **One Tool Per Turn:** You must only call ONE tool at a time.
        - **Wait For Results:** ALWAYS wait for a tool's output before deciding your next action.
        - **Sequential Search:** If you need to search multiple document types, do so one by one, waiting for results each time.
        """
    ),
    name="doc_search_assistant"
)

analyst_assistant = create_react_agent(
    model=llm,
    tools=[
        analyst_chart_tool,
        *sql_tools_for_analyst, # Unpack all SQL tools
    ],
    prompt=(
        """You are a specialized data analyst assistant. Your purpose is to provide data-driven insights through SQL queries and chart generation.

        **Your Capabilities:**

        **1. Database Analysis (SQL):**
           - You can directly interact with the company's database.
           - Your SQL toolkit (`sql_tools_for_analyst`) includes:
             - `sql_db_list_tables`: To see all available data tables.
             - `sql_db_schema`: To understand the structure (columns, types) of specific tables.
             - `sql_db_query`: To execute a SQL query to retrieve data.
             - `sql_db_query_checker`: To validate the syntax of a SQL query before execution.
           - **Required Workflow:** Always follow this sequence for database tasks: `list_tables` -> `schema` -> `query_checker` -> `query`.

        **2. Chart Generation:**
           - You can create data visualizations using the `analyst_chart_tool`.
           - This tool requires a title, the data (in a suitable format), and the desired chart type.

        **Your Workflow:**
        1. **Analyze the Request:** Determine if the task requires database analysis, chart generation, or both.
        2. **Execute Tasks:** Perform all requested data analysis and charting tasks.
        3. **Present Results:** Clearly show the results of your analysis, including any generated charts or data tables.

        **Strict Tool Usage Rules:**
        - **One Tool Per Turn:** You must only call ONE tool at a time.
        - **Wait For Results:** ALWAYS wait for a tool's output before deciding your next action.
        - **No Mixed Tool Calls:** NEVER call a SQL tool and a chart tool in the same turn.
        """
    ),
    name="analyst_assistant"
)

# --- Prediction Assistant Definition (Uncompiled Workflow) ---
class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    csv_file_content: Optional[str] = None

def csv_intake_node(state: AgentState):
    """If CSV content is present, analyze it directly. Otherwise, return an error."""
    print("---[Graph] In csv_intake_node---")
    csv_content = state.get("csv_file_content")
    
    if csv_content:
        print("[Graph] CSV content found, calling analysis tool directly.")
        query = state["messages"][-1].content
        tool_input = {"query": query, "csv_file_content": csv_content}
        result = analyze_csv_with_churn_prediction.invoke(tool_input)
        message = AIMessage(content=result, name="CustomerChurnDataAnalyzer")
        return {"messages": [message]}
    else:
        print("[Graph] No CSV content, returning error message.")
        message = AIMessage(content="Error: Prediction requires a CSV file, but none was provided.")
        return {"messages": [message]}

# Define the graph structure for the prediction assistant
predict_workflow = StateGraph(AgentState)
predict_workflow.add_node("csv_intake", csv_intake_node)
predict_workflow.set_entry_point("csv_intake")
predict_workflow.set_finish_point("csv_intake")


# --- Graph Compilation & Caching ---

_graph_cache = {}
logger = logging.getLogger(__name__)

def get_checkpoint_saver():
    """
    Initializes and returns the checkpointer.
    Falls back to in-memory if DB_URI is not set or langgraph-postgres is not installed.
    """
    db_uri = os.getenv("DB_URI")
    if db_uri:
        try:
            from langgraph_postgres import AsyncPostgresSaver
            logger.info("DB_URI found, using AsyncPostgresSaver for persistence.")
            return AsyncPostgresSaver.from_conn_string(db_uri)
        except ImportError:
            logger.error("DB_URI is set, but 'langgraph-postgres' is not installed. Please run 'pip install langgraph-postgres'. Falling back to MemorySaver.")
            return MemorySaver()
    else:
        logger.warning("DB_URI not found. Using in-memory checkpointer. Conversation history will not be saved.")
        return MemorySaver()

def get_graph(agent_name: str):
    """
    Retrieves a compiled LangGraph agent from a cache or compiles it if not present.
    """
    if agent_name in _graph_cache:
        print(f"Returning cached graph for: {agent_name}")
        return _graph_cache[agent_name]

    print(f"Compiling graph for: {agent_name}")
    checkpointer = get_checkpoint_saver()

    if agent_name == "prediction":
        # Compile the stateful prediction graph with its checkpointer
        graph = predict_workflow.compile(checkpointer=checkpointer)
        _graph_cache[agent_name] = graph
        return graph

    elif agent_name == "doc_search":
        _graph_cache[agent_name] = doc_search_assistant
        return doc_search_assistant

    elif agent_name == "analyst":
        _graph_cache[agent_name] = analyst_assistant
        return analyst_assistant

    else:  # Default case for the supervisor
        # The supervisor needs a compiled instance of the prediction agent to delegate to.
        predict_assistant = predict_workflow.compile(checkpointer=checkpointer)

        doc_search_assistant.name = "doc_search_assistant"
        analyst_assistant.name = "analyst_assistant"
        predict_assistant.name = "predict_assistant"

        # Define the system prompt for the supervisor
        system_prompt = (
            "You are a supervisor managing a team of expert assistants. "
            "Based on the user's request, you must delegate the task to the appropriate assistant. "
            "Your team consists of:\n"
            "- **doc_search_assistant**: Searches internal company documents.\n"
            "- **analyst_assistant**: Performs data analysis, SQL queries, and creates charts.\n"
            "- **predict_assistant**: Predicts customer churn from data.\n\n"
            "Route the user's request to the correct assistant to handle the task. "
            "If a task requires multiple assistants, route them sequentially. "
            "Only the user can mark the task as complete. Continue processing until the user is satisfied."
        )
        
        # The system prompt is used to guide the supervisor in routing tasks.
        # We pass it in as a `messages_modifier`.
        supervisor_graph = create_supervisor(
            model=llm,
            agents=[doc_search_assistant, analyst_assistant, predict_assistant],
        )
        graph = supervisor_graph.compile(checkpointer=checkpointer)
        _graph_cache[agent_name] = graph
        return graph
