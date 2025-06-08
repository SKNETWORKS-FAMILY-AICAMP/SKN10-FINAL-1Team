"""LangGraph graph for analysis_agent with supervisor logic.

Handles DB queries and general questions based on node/edge routing.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, List, Annotated, Literal, TypedDict
import asyncio
import io
from dotenv import load_dotenv
import os
import operator # For adding to message history
import psycopg2
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage # For message types
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig # Added missing import
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import json
import io

# --- Configuration (Optional - can be used to pass API keys, model names, etc.) ---
class Configuration(TypedDict, total=False):
    openai_api_key: Optional[str]
    db_env_path: Optional[str] # Path to .env file for DB credentials

# --- State Definition ---
class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)
    user_query: Optional[str] = None
    csv_file_content: Optional[str] = None
    query_type: Optional[Literal["db_query", "category_predict_query"]] = None
    sql_query: Optional[str] = None
    sql_result: Optional[Any] = None
    final_answer: Optional[str] = None
    error_message: Optional[str] = None
    visualization_output: Optional[str] = None
    sql_output_choice: Optional[Literal["summarize", "visualize"]] = None # Decision for SQL output processing

    class Config:
        arbitrary_types_allowed = True # For Annotated and operator.add with BaseMessage

# --- LLM and Prompts Setup ---
# Ensure OPENAI_API_KEY is set in your environment or passed via config
llm = ChatOpenAI(temperature=0, model="gpt-4o") # Or your preferred model

supervisor_prompt = PromptTemplate.from_template(
    "Analyze the user's question. Respond with a JSON object.\n"
    "The JSON object MUST contain a 'query_type' field set to one of 'db_query' or 'category_predict_query'.\n\n"
    "User Question: {user_query}\n\n"
    "JSON Response (must be a valid JSON object adhering to the Pydantic model `SupervisorDecision`):"
)

sql_generation_prompt = PromptTemplate.from_template(
    """Based on the following user question and the provided database schema information, convert the question into a SQL query in PostgreSQL syntax AND determine if the result should be summarized or visualized.
Accurately understand the user's intent and use the schema information to write a query with the correct tables and columns.
For text-based searches (e.g., using LIKE clauses on string columns), ensure the search is case-insensitive by using the ILIKE operator or by applying the LOWER() function to both the column and the search term, unless the user specifically requests a case-sensitive search.
If the information needed for the question is not in the schema or is ambiguous, please state that or use the most probable interpretation.

Database Schema Information:

Table Name: analytics_results
Columns:
  - id (uuid)
  - result_type (character varying)
  - s3_key (text)
  - meta (jsonb)
  - created_at (timestamp with time zone)
  - user_id (uuid)

Table Name: chat_messages
Columns:
  - id (uuid)
  - role (character varying)
  - content (text)
  - created_at (timestamp with time zone)
  - session_id (uuid)
  - metadata (text)

Table Name: chat_sessions
Columns:
  - id (uuid)
  - agent_type (character varying)
  - started_at (timestamp with time zone)
  - ended_at (timestamp with time zone)
  - user_id (uuid)
  - title (character varying)

Table Name: llm_calls
Columns:
  - id (uuid)
  - call_type (character varying)
  - prompt (text)
  - response (text)
  - tokens_used (integer)
  - latency_ms (integer)
  - created_at (timestamp with time zone)
  - user_id (uuid)
  - session_id (uuid)

Table Name: model_artifacts
Columns:
  - id (uuid)
  - artifact_type (character varying)
  - s3_key (text)
  - meta (jsonb)
  - created_at (timestamp with time zone)
  - user_id (uuid)


Table Name: organizations
Columns:
  - id (uuid)
  - name (character varying)
  - created_at (timestamp with time zone)

Table Name: summary_news_keywords
Columns:
  - id (uuid)
  - date (date)
  - keyword (text)
  - title (text)
  - summary (text)
  - url (text)

Table Name: telecom_customers
Columns:
  - customer_id (character varying)
  - gender (character varying)
  - senior_citizen (boolean)
  - partner (boolean)
  - dependents (boolean)
  - tenure (integer)
  - phone_service (boolean)
  - multiple_lines (character varying)
  - internet_service (character varying)
  - online_security (character varying)
  - online_backup (character varying)
  - device_protection (character varying)
  - tech_support (character varying)
  - streaming_tv (character varying)
  - streaming_movies (character varying)
  - contract (character varying)
  - paperless_billing (boolean)
  - payment_method (character varying)
  - monthly_charges (numeric)
  - total_charges (numeric)
  - churn (boolean)

Table Name: users
Columns:
  - password (character varying)
  - is_superuser (boolean)
  - id (uuid)
  - email (character varying)
  - name (character varying)
  - role (character varying)
  - created_at (timestamp with time zone)
  - last_login (timestamp with time zone)
  - is_active (boolean)
  - is_staff (boolean)
  - org_id (uuid)

Respond with a JSON object that strictly adheres to the Pydantic model `SQLGenerationOutput` shown below.
The `sql_query` field MUST contain ONLY the SQL query string, without any surrounding text, explanations, or markdown formatting like ```sql.
The `sql_output_choice` field MUST be either 'summarize' (if the user asks for a textual summary, explanation, or direct answer from data) or 'visualize' (if the user asks for a chart, graph, or visual representation). Prioritize 'visualize' if visualization is explicitly or implicitly requested. If unsure, default to 'summarize'.

```python
class SQLGenerationOutput(BaseModel):
    sql_query: str = Field(description="The generated SQL query. This field MUST contain ONLY the SQL query string, without any surrounding text, explanations, or markdown formatting like ```sql.")
    sql_output_choice: Literal["summarize", "visualize"] = Field(description="The type of output processing required for the SQL result: 'summarize' or 'visualize'.")
```

User Question: {user_query}

JSON Response (must be a valid JSON object conforming to SQLGenerationOutput):
    """
)

summarization_prompt = PromptTemplate.from_template(
    "Based on the following SQL query execution result, please summarize the answer to the user's original question naturally.\n\n"
    "say in Korean"
    "User Question: {user_query}\n"
    "SQL Query: {sql_query}\n"
    "SQL Result:\n{sql_result}\n\n"
    "Summary Answer:"
)

# --- Pydantic Models for Structured Output ---
class SupervisorDecision(BaseModel):
    query_type: str = Field(description="The type of the user's question (db_query or category_predict_query)")

class SQLGenerationOutput(BaseModel):
    sql_query: str = Field(description="The generated SQL query. This field MUST contain ONLY the SQL query string, without any surrounding text, explanations, or markdown formatting like ```sql.")
    sql_output_choice: Literal["summarize", "visualize"] = Field(description="The type of output processing required for the SQL result: 'summarize' or 'visualize'.")

# --- Node Functions ---
async def supervisor_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- SUPERVISOR NODE (Debug v4 - Enhanced Message Parsing) ---")
    
    user_query_to_process: Optional[str] = None
    current_messages = state.messages # Keep a reference to the current messages

    print(f"Supervisor - Initial state.messages length: {len(current_messages) if current_messages else 0}")

    if current_messages:
        last_message_obj = current_messages[-1]
        print(f"Supervisor - Last message object: {last_message_obj}, type: {type(last_message_obj).__name__}")

        content_candidate: Optional[str] = None

        # Try to extract content based on common patterns
        if hasattr(last_message_obj, 'content') and isinstance(getattr(last_message_obj, 'content'), str):
            # Covers HumanMessage, AIMessage, and other BaseMessage with a .content attribute
            content_candidate = getattr(last_message_obj, 'content')
            print(f"Supervisor - Candidate from .content attribute: '{content_candidate[:100] if content_candidate else 'None'}...'")
        elif isinstance(last_message_obj, dict) and 'content' in last_message_obj and isinstance(last_message_obj['content'], str):
            # Covers cases where the message might be a dictionary (e.g., from serialization)
            content_candidate = last_message_obj['content']
            print(f"Supervisor - Candidate from dict['content']: '{content_candidate[:100] if content_candidate else 'None'}...'")
        elif isinstance(last_message_obj, str):
            # Covers cases where the last message itself is a plain string
            content_candidate = last_message_obj
            print(f"Supervisor - Candidate from last_message_obj being a string: '{content_candidate[:100] if content_candidate else 'None'}...'")
        
        # Ensure the extracted content is a non-empty string
        if content_candidate and content_candidate.strip():
            user_query_to_process = content_candidate.strip()
            print(f"Supervisor - Successfully extracted user query: '{user_query_to_process[:100]}...'")
        else:
            print(f"Supervisor - Content candidate was None, empty, or whitespace. Candidate: '{content_candidate}'")
    else:
        print("Supervisor - state.messages is empty.")

    if not user_query_to_process:
        error_msg = "Supervisor - No valid user query found in state.messages. Please provide input via the 'Messages' field with textual content."
        print(f"Supervisor - {error_msg}")
        return {
            "user_query": None,
            "query_type": "category_predict_query", # Default to category_predict_query if no input
            "error_message": error_msg,
            "messages": current_messages # Pass through existing messages
        }

    # If user_query_to_process is successfully set, decide routing strategy
    print(f"Supervisor - User query for routing: '{user_query_to_process}'")

    if "예측" in user_query_to_process:
        print(f"Supervisor - '예측' keyword found. Proceeding with LLM analysis for query type.")
        supervisor_chain = supervisor_prompt | llm.with_structured_output(SupervisorDecision)
        try:
            parsed_output: SupervisorDecision = await supervisor_chain.ainvoke({"user_query": user_query_to_process}, config=config)
            query_type = parsed_output.query_type
            print(f"Supervisor LLM Decision: query_type = {query_type}")
            updated_messages = current_messages + [AIMessage(content=f"Routing to {query_type} based on supervisor LLM decision.")]
            return {
                "messages": updated_messages,
                "user_query": user_query_to_process,
                "query_type": query_type,
                "error_message": None
            }
        except Exception as e:
            error_msg = f"Supervisor - Error during LLM decision for '예측' query '{user_query_to_process[:50]}...': {e}"
            print(error_msg)
            # Fallback if LLM fails for a "예측" query.
            # Defaulting to category_predict_query as per original logic for LLM failure.
            return {
                "user_query": user_query_to_process,
                "query_type": "category_predict_query", 
                "error_message": error_msg,
                "messages": current_messages
            }
    else:
        print(f"Supervisor - '예측' keyword NOT found. Routing directly to generate_sql_node.")
        query_type = "db_query"
        updated_messages = current_messages + [AIMessage(content=f"Routing to {query_type} (SQL generation) as '예측' was not in query.")]
        return {
            "messages": updated_messages,
            "user_query": user_query_to_process,
            "query_type": query_type,
            "error_message": None
        }

async def generate_sql_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- GENERATE SQL NODE ---")
    user_query = state.user_query # Get user_query from the state
    if not user_query:
        return {"error_message": "No user query found for SQL generation.", "sql_query": "", "sql_output_choice": None}

    print(f"Generating SQL and determining output choice for query: {user_query[:100]}...")
    # Use llm.with_structured_output for more robust parsing to the Pydantic model
    structured_llm_sql_gen = llm.with_structured_output(SQLGenerationOutput)
    # The runnable now takes the prompt and applies the structured LLM
    sql_generation_runnable = sql_generation_prompt | structured_llm_sql_gen

    try:
        # Pass the user_query to the runnable, which will format the prompt and call the structured LLM
        response_model = await sql_generation_runnable.ainvoke({"user_query": user_query}, config=config)
        # response_model should now be an instance of SQLGenerationOutput
        sql_query = response_model.sql_query.strip()
        sql_output_choice = response_model.sql_output_choice
        print(f"Generated SQL: {sql_query}")
        print(f"Determined SQL Output Choice: {sql_output_choice}")
        return {"sql_query": sql_query, "sql_output_choice": sql_output_choice, "error_message": None}
    except Exception as e:
        error_msg = f"Error generating SQL or determining output choice: {e}"
        print(error_msg)
        return {"error_message": error_msg, "sql_query": "", "sql_output_choice": None}

import asyncio # Required for asyncio.to_thread

# Helper function for synchronous DB operations, including dotenv loading
def _execute_sql_sync(sql_query: str, base_dir_analysis_env: str, base_dir_my_state_env: str) -> Dict[str, Any]:
    # Determine .env path and load environment variables
    dotenv_path_analysis = os.path.join(base_dir_analysis_env, '.env')
    dotenv_path_my_state = os.path.join(base_dir_my_state_env, '.env')
    specific_env_path = None

    if os.path.exists(dotenv_path_analysis):
        specific_env_path = dotenv_path_analysis
    elif os.path.exists(dotenv_path_my_state):
        specific_env_path = dotenv_path_my_state

    if specific_env_path:
        print(f"_execute_sql_sync: Loading .env from: {specific_env_path}")
        load_dotenv(dotenv_path=specific_env_path, override=True)
    else:
        print("_execute_sql_sync: No specific .env file found. Relying on system environment variables or a global .env.")
        load_dotenv(override=True) # Load global .env or system vars

    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")

    if not all([db_host, db_port, db_name, db_user, db_password]):
        error_msg = "_execute_sql_sync: Database connection details missing in environment variables."
        print(error_msg)
        return {"error_message": error_msg, "sql_result": ""}

    conn_string = f"host='{db_host}' port='{db_port}' dbname='{db_name}' user='{db_user}' password='{db_password}'"
    conn = None
    try:
        print(f"_execute_sql_sync: Connecting to DB with: {conn_string.replace(db_password, '****') if db_password else conn_string}")
        conn = psycopg2.connect(conn_string)
        print(f"_execute_sql_sync: Executing SQL: {sql_query}")
        df = pd.read_sql_query(sql_query, conn)
        sql_result_str = df.to_string()
        print(f"_execute_sql_sync: SQL Result (first 200 chars): {sql_result_str[:200]}")
        return {"sql_result": sql_result_str, "error_message": None}
    except (psycopg2.Error, pd.io.sql.DatabaseError) as e:
        error_msg = f"_execute_sql_sync: Database error: {e}"
        print(error_msg)
        return {"error_message": error_msg, "sql_result": ""}
    except ValueError as e: # Often from pandas if query is malformed for read_sql_query
        error_msg = f"_execute_sql_sync: SQL query validation error for pandas: {e}"
        print(error_msg)
        return {"error_message": error_msg, "sql_result": ""}
    except Exception as e:
        error_msg = f"_execute_sql_sync: An unexpected error occurred during SQL execution: {e}"
        print(error_msg)
        return {"error_message": error_msg, "sql_result": ""}
    finally:
        if conn:
            conn.close()
            print("_execute_sql_sync: DB 연결 종료.")

async def execute_sql_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- EXECUTE SQL NODE ---")
    sql_query = state.sql_query
    if not sql_query:
        return {"error_message": "No SQL query to execute.", "sql_result": ""}

    # Define base directories for .env file search relative to this file's location
    # analysis_agent/.env -> ../../.env
    base_dir_analysis_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # my_state_agent/.env -> ../../../my_state_agent
    base_dir_my_state_env = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'my_state_agent'))
    
    # Run the synchronous dotenv loading and database operations in a separate thread
    try:
        print(f"execute_sql_node: Calling asyncio.to_thread for SQL: {sql_query}")
        result_dict = await asyncio.to_thread(
            _execute_sql_sync, 
            sql_query, 
            base_dir_analysis_env, 
            base_dir_my_state_env
        )
        return result_dict
    except Exception as e: # Catch potential errors from asyncio.to_thread itself
        error_msg = f"execute_sql_node: Error running DB operations in thread: {e}"
        print(error_msg)
        return {"error_message": error_msg, "sql_result": ""}

async def create_visualization_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- CREATE VISUALIZATION NODE (Placeholder) ---")
    sql_result = state.sql_result
    if not sql_result:
        # If there's an error message from a previous node, pass it along.
        # Otherwise, set a specific error for this node.
        error_to_pass = state.error_message or "No SQL result to visualize."
        print(f"Create Visualization Node: Error - {error_to_pass}")
        return {"error_message": error_to_pass, "sql_result": sql_result} # Pass original sql_result for context

    # --- Extract user query for visualization hint ---
    user_query = ""
    # Attempt to get the latest user message from state.get("messages")
    # AgentState is expected to be dict-like or have a 'messages' attribute.
    messages = state.messages
    if messages and isinstance(messages, list) and len(messages) > 0:
        last_message = messages[-1]
        # Assuming last_message has a 'content' attribute (e.g., HumanMessage)
        if hasattr(last_message, 'content'):
            user_query = str(last_message.content)
        elif isinstance(last_message, str): # If messages are just strings
            user_query = last_message

    print(f"Create Visualization Node: User query for context: '{user_query[:200]}...'" if user_query else "Create Visualization Node: No user query found for context.")

    # Simple keyword-based hint extraction (Korean keywords)
    visualization_hint = "auto" # Default
    query_lower = user_query.lower()

    if any(kw in query_lower for kw in ["시간", "추세", "흐름", "시계열"]):
        visualization_hint = "timeseries"
    elif any(kw in query_lower for kw in ["카테고리", "그룹", "항목", "비교"]):
        # Check if it's not a pie chart request
        if not any(kw_pie in query_lower for kw_pie in ["원형", "파이", "비율", "점유율"]):
             visualization_hint = "barchart"
    elif any(kw in query_lower for kw in ["관계", "상관"]):
        visualization_hint = "scatterplot"
    elif any(kw in query_lower for kw in ["분포", "히스토그램"]):
        visualization_hint = "histogram"
    
    # Pie chart has specific keywords and can override '비교' if '비율' etc. are present
    if any(kw in query_lower for kw in ["원형", "파이", "비율", "점유율"]):
        visualization_hint = "piechart"

    print(f"Create Visualization Node: Determined visualization hint: '{visualization_hint}'")

    # Prepare string literals for safe embedding in the generated Python code
    # repr(json.dumps(sql_result)) creates a string like "'[\"key\": \"value\"]'"
    # which is a valid Python string literal representing the JSON string.
    safe_sql_data_json_literal = repr(json.dumps(sql_result))
    safe_visualization_hint_literal = repr(str(visualization_hint)) # Ensure hint is string, then get its literal form

    # Generate Python code for visualization, incorporating the hint
    python_code_template = """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # For numeric type checking and NaN handling
import json # For json.loads in generated code
import io # Potentially for pd.read_json(io.StringIO(...)) if used later

# Data obtained from the SQL query, as a Python string literal containing JSON
sql_data_json_string = __SQL_DATA_JSON_LITERAL__
# Hint from user query analysis, as a Python string literal
visualization_hint_from_query = __VISUALIZATION_HINT_LITERAL__

def generate_visualization(data_json_string, hint='auto'):
    print(f"--- Executing Generated Visualization Code (Hint: {{hint}}) ---")
    
    # Attempt to load data from JSON string
    data = []
    if data_json_string:
        try:
            # It's safer to load the JSON string into a Python object first
            loaded_data = json.loads(data_json_string)
            # Then create a DataFrame. This handles various JSON structures.
            if isinstance(loaded_data, list) and all(isinstance(item, dict) for item in loaded_data):
                data = loaded_data # Looks like a list of records, good for DataFrame
            elif isinstance(loaded_data, dict) and 'data' in loaded_data and 'columns' in loaded_data:
                # Handles cases like {'columns': ['col1', 'col2'], 'data': [[val1, val2], ...]}
                df_from_dict = pd.DataFrame(loaded_data['data'], columns=loaded_data['columns'])
                data = df_from_dict.to_dict(orient='records') # Convert back to list of dicts for consistency if needed
            else: # Fallback for other structures, or if it's already a list of dicts
                data = loaded_data if isinstance(loaded_data, list) else [loaded_data]

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON data: {{e}}")
            print(f"JSON string was: {{data_json_string[:500]}}...") # Print a snippet of the problematic string
            return
        except Exception as e_df: # Catch other potential pandas errors
            print(f"Error creating DataFrame from JSON: {{e_df}}")
            return
    
    if not data:
        print("No data loaded from JSON string to visualize.")
        return

    try:
        df = pd.DataFrame(data) # Now data should be a list of dictionaries
        if df.empty:
            print("\nDataFrame is empty. No data to visualize.")
            if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                 print(f"Columns: {{list(data[0].keys())}}")
            elif hasattr(data, 'columns'):
                 print(f"Columns: {{list(data.columns)}}")
            return

        print("\n--- Data Preview (First 5 rows) ---")
        print(df.head())
        print("\n--- Descriptive Statistics (Numeric) ---")
        print(df.describe(include=np.number))
        print("\n--- Descriptive Statistics (Categorical) ---")
        print(df.describe(include=['object', 'category']))

        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols_initial = df.select_dtypes(include=['datetime', 'timedelta']).columns
        potential_dt_cols = [col for col in df.columns if df[col].dtype == 'object']

        for col in list(datetime_cols_initial) + potential_dt_cols:
            if col in df.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        converted_col = pd.to_datetime(df[col], errors='coerce')
                        if not converted_col.isnull().all(): 
                            df[col] = converted_col
                except Exception:
                    pass 
        datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'timedelta64[ns]']).columns

        plotted = False

        # --- 1. Attempt visualization based on HINT --- 
        if hint != 'auto':
            print(f"\n--- Attempting visualization based on hint: {{hint}} ---")
            if hint == "timeseries":
                if len(datetime_cols) > 0 and len(numeric_cols) > 0:
                    time_col, val_col = datetime_cols[0], numeric_cols[0]
                    try:
                        plt.figure(figsize=(12, 6))
                        df_sorted_time = df.dropna(subset=[time_col, val_col]).sort_values(by=time_col)
                        if not df_sorted_time.empty:
                            plt.plot(df_sorted_time[time_col], df_sorted_time[val_col], marker='o', linestyle='-')
                            plt.title(f'Time Series Plot: {{val_col}} over {{time_col}} (Hinted)')
                            plt.xlabel(time_col); plt.ylabel(val_col); plt.grid(True); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.show()
                            print(f"Displayed time series plot (hinted): '{{val_col}}' vs '{{time_col}}'.")
                            plotted = True
                        else: print(f"No valid data for hinted time series: {{time_col}}, {{val_col}}.")
                    except Exception as e: print(f"Hinted time series plot failed: {{e}}")
                else: print(f"Hinted timeseries needs datetime & numeric cols. Found: D={{len(datetime_cols)}}, N={{len(numeric_cols)}})")
            
            elif hint == "barchart":
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    cat_col, num_col = categorical_cols[0], numeric_cols[0]
                    unique_cats = df[cat_col].nunique()
                    if 0 < unique_cats <= 25:
                        try:
                            if pd.api.types.is_numeric_dtype(df[num_col]):
                                plt.figure(figsize=(12, 7))
                                df.groupby(cat_col)[num_col].mean().plot(kind='bar', edgecolor='k')
                                plt.title(f'Bar Chart: Mean {{num_col}} by {{cat_col}} (Hinted)')
                                plt.xlabel(cat_col); plt.ylabel(f'Mean {{num_col}}'); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.show()
                                print(f"Displayed bar chart (hinted): Mean '{{num_col}}' by '{{cat_col}}'.")
                                plotted = True
                            else: print(f"Hinted bar chart: '{{num_col}}' not numeric for mean.")
                        except Exception as e: print(f"Hinted bar chart failed: {{e}}")
                    elif unique_cats > 0: print(f"Hinted bar chart: '{{cat_col}}' has {{unique_cats}} unique values (max 25). Skipping.")
                else: print(f"Hinted barchart needs categorical & numeric cols. Found: C={{len(categorical_cols)}}, N={{len(numeric_cols)}})")

            elif hint == "scatterplot":
                if len(numeric_cols) >= 2:
                    x_scat, y_scat = numeric_cols[0], numeric_cols[1]
                    try:
                        plt.figure(figsize=(10, 6))
                        plt.scatter(df[x_scat], df[y_scat], alpha=0.7)
                        plt.title(f'Scatter Plot: {{y_scat}} vs {{x_scat}} (Hinted)')
                        plt.xlabel(x_scat); plt.ylabel(y_scat); plt.grid(True); plt.tight_layout(); plt.show()
                        print(f"Displayed scatter plot (hinted): '{{y_scat}}' vs '{{x_scat}}'.")
                        plotted = True
                    except Exception as e: print(f"Hinted scatter plot failed: {{e}}")
                else: print(f"Hinted scatterplot needs >=2 numeric cols. Found: {{len(numeric_cols)}})")

            elif hint == "histogram":
                if len(numeric_cols) > 0:
                    hist_c = numeric_cols[0]
                    try:
                        plt.figure(figsize=(10, 6))
                        df[hist_c].plot(kind='hist', bins=15, edgecolor='k')
                        plt.title(f'Histogram of {{hist_c}} (Hinted)'); plt.xlabel(hist_c); plt.ylabel('Frequency'); plt.tight_layout(); plt.show()
                        print(f"Displayed histogram (hinted) of '{{hist_c}}'.")
                        plotted = True
                    except Exception as e: print(f"Hinted histogram failed: {{e}}")
                else: print(f"Hinted histogram needs numeric col. Found: {{len(numeric_cols)}})")
            
            elif hint == "piechart":
                if len(categorical_cols) > 0:
                    pie_cat_col = categorical_cols[0]
                    unique_pie_cats = df[pie_cat_col].nunique()
                    if 0 < unique_pie_cats <= 10:
                        try:
                            plt.figure(figsize=(8, 8))
                            pie_data_source = df[pie_cat_col].value_counts()
                            title_suffix = f"Distribution of {{pie_cat_col}} (Hinted)"
                            # If a numeric col is available and suitable, sum it by category
                            if len(numeric_cols) > 0:
                                pie_num_col = numeric_cols[0]
                                # Check if numeric column is appropriate for sum (e.g. not an ID like column)
                                if df[pie_num_col].nunique() > 1 and df[pie_num_col].sum() != 0 and not all(df[pie_num_col].apply(lambda x: isinstance(x, int) and x > 10000)) : # Heuristic
                                    pie_data_source = df.groupby(pie_cat_col)[pie_num_col].sum()
                                    title_suffix = f"Sum of {{pie_num_col}} by {{pie_cat_col}} (Hinted)"
                            
                            plt.pie(pie_data_source, labels=pie_data_source.index, autopct='%1.1f%%', startangle=90, counterclock=False)
                            plt.title(title_suffix); plt.axis('equal'); plt.show()
                            print(f"Displayed pie chart (hinted) for '{{pie_cat_col}}'.")
                            plotted = True
                        except Exception as e: print(f"Hinted pie chart failed: {{e}}")
                    elif unique_pie_cats > 0: print(f"Hinted pie chart: '{{pie_cat_col}}' has {{unique_pie_cats}} unique values (max 10). Skipping.")
                else: print(f"Hinted piechart needs categorical col. Found: {{len(categorical_cols)}})")
            else:
                print(f"Unknown or unsupported visualization hint: {{hint}}. Proceeding to automatic detection.")

        # --- 2. Fallback to AUTOMATIC detection if hint failed or was 'auto' ---
        if not plotted:
            print("\n--- Hint-based visualization not performed or failed. Attempting automatic generic visualization ---")
            # Auto Time series plot
            if not plotted and len(datetime_cols) > 0 and len(numeric_cols) > 0:
                time_col, val_col = datetime_cols[0], numeric_cols[0]
                try:
                    plt.figure(figsize=(12, 6))
                    df_sorted_time = df.dropna(subset=[time_col, val_col]).sort_values(by=time_col)
                    if not df_sorted_time.empty:
                        plt.plot(df_sorted_time[time_col], df_sorted_time[val_col], marker='o', linestyle='-')
                        plt.title(f'Time Series Plot of {{val_col}} over {{time_col}} (Auto)')
                        plt.xlabel(time_col); plt.ylabel(val_col); plt.grid(True); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.show()
                        print(f"Displayed auto time series: '{{val_col}}' vs '{{time_col}}'.")
                        plotted = True
                    else: print(f"No valid data for auto time series: {{time_col}}, {{val_col}}.")
                except Exception as e: print(f"Auto time series failed: {{e}}")
            
            # Auto Bar chart
            if not plotted and len(categorical_cols) > 0 and len(numeric_cols) > 0:
                cat_col, num_col = categorical_cols[0], numeric_cols[0]
                unique_cats = df[cat_col].nunique()
                if 0 < unique_cats <= 25:
                    try:
                        if pd.api.types.is_numeric_dtype(df[num_col]):
                            plt.figure(figsize=(12, 7))
                            df.groupby(cat_col)[num_col].mean().plot(kind='bar', edgecolor='k')
                            plt.title(f'Bar Chart: Mean {{num_col}} by {{cat_col}} (Auto)')
                            plt.xlabel(cat_col); plt.ylabel(f'Mean {{num_col}}'); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.show()
                            print(f"Displayed auto bar chart: Mean '{{num_col}}' by '{{cat_col}}'.")
                            plotted = True
                        else: print(f"Auto bar chart: '{{num_col}}' not numeric for mean.")
                    except Exception as e: print(f"Auto bar chart failed: {{e}}")
                elif unique_cats > 0: print(f"Auto bar chart: '{{cat_col}}' has {{unique_cats}} unique values (max 25). Skipping.")

            # Auto Scatter plot
            if not plotted and len(numeric_cols) >= 2:
                x_scat, y_scat = numeric_cols[0], numeric_cols[1]
                try:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(df[x_scat], df[y_scat], alpha=0.7)
                    plt.title(f'Scatter Plot: {{y_scat}} vs {{x_scat}} (Auto)')
                    plt.xlabel(x_scat); plt.ylabel(y_scat); plt.grid(True); plt.tight_layout(); plt.show()
                    print(f"Displayed auto scatter plot: '{{y_scat}}' vs '{{x_scat}}'.")
                    plotted = True
                except Exception as e: print(f"Auto scatter plot failed: {{e}}")

            # Auto Histogram
            if not plotted and len(numeric_cols) > 0:
                hist_c = numeric_cols[0]
                try:
                    plt.figure(figsize=(10, 6))
                    df[hist_c].plot(kind='hist', bins=15, edgecolor='k')
                    plt.title(f'Histogram of {{hist_c}} (Auto)'); plt.xlabel(hist_c); plt.ylabel('Frequency'); plt.tight_layout(); plt.show()
                    print(f"Displayed auto histogram of '{{hist_c}}'.")
                    plotted = True
                except Exception as e: print(f"Auto histogram failed: {{e}}")
        
        if not plotted:
            print("\nCould not automatically determine or apply a suitable plot type.")
            print("Please examine the DataFrame and descriptive statistics. You can use 'sql_data' or 'df' with Matplotlib/Seaborn/Plotly for custom visualizations.")

    except ImportError as e_import:
        print(f"Import error: {{e_import}}. Ensure pandas, matplotlib, numpy are installed.")
    except Exception as e:
        print(f"An error occurred during visualization: {{e}}")
        if isinstance(data, (list, dict)) and len(str(data)) < 1000: print(data)
        else: print(f"Data type {{type(data)}} may be too large to print.")

if __name__ == '__main__':
    # Example data for direct script execution (testing)
    example_sql_data = [{'product_category': 'Electronics', 'sales_date': '2023-01-15', 'revenue': 1200, 'units_sold': 10},
                        {'product_category': 'Books', 'sales_date': '2023-01-16', 'revenue': 150, 'units_sold': 12},
                        {'product_category': 'Electronics', 'sales_date': '2023-01-17', 'revenue': 800, 'units_sold': 6},
                        {'product_category': 'Home Goods', 'sales_date': '2023-01-18', 'revenue': 450, 'units_sold': 20}]
    
    # Test with different hints
    # generate_visualization(example_sql_data, hint='timeseries') # Needs sales_date as datetime
    # generate_visualization(example_sql_data, hint='barchart')
    # generate_visualization(example_sql_data, hint='piechart') 
    # generate_visualization(example_sql_data, hint='histogram') # for revenue
    # generate_visualization(example_sql_data, hint='scatterplot') # revenue vs units_sold
    
    # Default execution with hint from global (if set by main node)
    if 'sql_data' in globals() and sql_data is not None:
        generate_visualization(sql_data, hint=visualization_hint_from_query if 'visualization_hint_from_query' in globals() else 'auto')
    else:
        print("sql_data not found. Running with example data and 'auto' hint.")
        generate_visualization(example_sql_data, hint='auto')
"""
    python_code_for_visualization = python_code_template.replace("__SQL_DATA_JSON_LITERAL__", safe_sql_data_json_literal)
    python_code_for_visualization = python_code_for_visualization.replace("__VISUALIZATION_HINT_LITERAL__", safe_visualization_hint_literal)
    print(f"Create Visualization Node: Generated Python code for visualization (Hint: {visualization_hint}).")
    return {"sql_result": sql_result, "visualization_output": python_code_for_visualization, "error_message": None}

async def summarize_sql_result_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- SUMMARIZE SQL RESULT NODE ---")
    user_query = state.user_query # Get user_query from the state field set by supervisor
    sql_query = state.sql_query or ""
    sql_result = state.sql_result or ""
    current_messages = state.messages # Get current messages

    if not user_query:
         # Add error message to messages as well
        error_msg = "No user query found for summarization."
        updated_messages = current_messages + [AIMessage(content=error_msg)]
        return {"messages": updated_messages, "error_message": error_msg, "final_answer": ""}
    if not sql_result:
        error_msg = state.error_message or "No SQL result to summarize."
        final_response_msg = state.error_message or "SQL query execution failed or produced no result."
        updated_messages = current_messages + [AIMessage(content=final_response_msg)]
        return {"messages": updated_messages, "error_message": error_msg, "final_answer": final_response_msg}

    print(f"Summarizing for query: {user_query}, SQL: {sql_query[:100]}..., Result: {sql_result[:100]}...")
    summarization_chain = summarization_prompt | llm
    response = await summarization_chain.ainvoke(
        {"user_query": user_query, "sql_query": sql_query, "sql_result": sql_result},
        config=config
    )
    final_answer = response.content.strip()
    print(f"Summarized Answer: {final_answer}")
    
    # Add the final_answer to the messages list as an AIMessage
    updated_messages = current_messages + [AIMessage(content=final_answer)]
    
    return {"messages": updated_messages, "final_answer": final_answer}


async def category_predict_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- CATEGORY PREDICT NODE (Telecom Churn Prediction with Csv File Content) ---")

    # --- 경로 설정 ---
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    MODEL_PATH = os.path.join(base_path, 'churn_predictor_pipeline.pkl')
    CATEGORICAL_COLS_PATH = os.path.join(base_path, 'categorical_cols.pkl')
    LABEL_ENCODERS_PATH = os.path.join(base_path, 'label_encoders.pkl')

    EXPECTED_FEATURE_ORDER = [
        'seniorcitizen', 'partner', 'dependents', 'tenure', 'phoneservice',
        'multiplelines', 'onlinesecurity', 'onlinebackup', 'techsupport',
        'contract', 'paperlessbilling', 'paymentmethod', 'monthlycharges', 'totalcharges',
        'new_totalservices', 'new_avg_charges', 'new_increase', 'new_avg_service_fee',
        'charge_increased', 'charge_growth_rate', 'is_auto_payment',
        'expected_contract_months', 'contract_gap'
    ]
    CUSTOMER_ID_COL = 'customerid'
    PREDICTION_THRESHOLD = 0.312

    csv_data_str: Optional[str] = None

    # 1. state.csv_file_content (LangGraph Studio의 'Csv File Content' 필드) 확인
    if state.csv_file_content:
        print("INFO: Using CSV data from state.csv_file_content.")
        csv_data_str = state.csv_file_content
    # 2. state.user_query (Chat 또는 Messages 입력) 확인
    elif hasattr(state, 'user_query') and state.user_query:
        print(f"INFO: Attempting to use state.user_query for CSV data. Content (first 100 chars): '{state.user_query[:100]}...'")
        # 2a. state.user_query를 파일 경로로 시도
        if os.path.exists(state.user_query):
            try:
                print(f"INFO: state.user_query '{state.user_query}' is an existing path. Reading file.")
                def read_file_sync(path):
                    with open(path, 'r', encoding='utf-8') as f_sync:
                        return f_sync.read()
                csv_data_str = await asyncio.to_thread(read_file_sync, state.user_query)
                if not csv_data_str:
                    print(f"WARNING: File at '{state.user_query}' was empty.")
            except Exception as e:
                print(f"WARNING: Error reading file from state.user_query path '{state.user_query}': {e}. Will attempt to treat as raw content.")
        
        # 2b. state.user_query를 파일 경로로 읽지 못했거나, 경로가 아니었다면 원본 CSV 내용으로 간주
        if csv_data_str is None: # 파일 읽기 실패 또는 경로가 아니었음
            print("INFO: Treating state.user_query as raw CSV content.")
            csv_data_str = state.user_query # pd.read_csv가 이후에 파싱 시도

    # CSV 데이터를 어디에서도 찾지 못한 경우 오류 반환
    if csv_data_str is None:
        error_message_parts = ["❌ 오류: CSV 데이터를 찾을 수 없습니다."]
        checked_sources = ["'Csv File Content' 필드"]
        if hasattr(state, 'user_query'):
            checked_sources.append("'User Query' / 채팅 메시지 (파일 경로 또는 CSV 내용 직접 입력)")
        error_message_parts.append(f"확인한 입력 소스: {', '.join(checked_sources)}.")
        error_message_parts.append("Csv File Content 필드에 직접 CSV 내용을 붙여넣거나, 채팅으로 CSV 파일의 전체 경로 또는 CSV 내용 자체를 입력해주세요.")
        final_answer = "\n".join(error_message_parts)
        current_messages = state.messages # Get current messages
        updated_messages = current_messages + [AIMessage(content=final_answer)] if current_messages else [AIMessage(content=final_answer)]
        return {"messages": updated_messages, "final_answer": final_answer, "error_message": final_answer}

    print(f"INFO: CSV data obtained. Length: {len(csv_data_str)}. Preview (first 200 chars): {csv_data_str[:200]}...")

    try:
        # --- 모델과 전처리 객체 비동기 로드 ---
        pipeline_final = await asyncio.to_thread(joblib.load, MODEL_PATH)
        CATEGORICAL_COLS = await asyncio.to_thread(joblib.load, CATEGORICAL_COLS_PATH)
        label_encoders = await asyncio.to_thread(joblib.load, LABEL_ENCODERS_PATH)

        # --- CSV 문자열 → DataFrame 변환 ---
        if not csv_data_str: # 이중 확인, csv_data_str이 None이나 빈 문자열이면 에러 발생 방지
            final_answer = "❌ 오류: 내부 로직 오류 - CSV 데이터 문자열이 비어있습니다."
            current_messages = state.messages # Get current messages
            updated_messages = current_messages + [AIMessage(content=final_answer)] if current_messages else [AIMessage(content=final_answer)]
            return {"messages": updated_messages, "final_answer": final_answer, "error_message": final_answer}
        # --- BEGIN REVISED CSV DATA CLEANING LOGIC ---
        raw_lines = csv_data_str.strip().splitlines()
        cleaned_lines = []

        if not raw_lines:
            final_answer = "❌ 오류: CSV 데이터 문자열이 비어있습니다."
            current_messages = state.messages
            updated_messages = current_messages + [AIMessage(content=final_answer)] if current_messages else [AIMessage(content=final_answer)]
            return {"messages": updated_messages, "final_answer": final_answer, "error_message": final_answer}

        MIN_COMMAS_THRESHOLD = 1  # 쉼표가 이 개수 이상이면 '강력한' CSV 라인으로 간주

        # '강력한' CSV 라인들의 인덱스를 찾음
        strong_csv_indices = [i for i, line in enumerate(raw_lines) if line.count(',') >= MIN_COMMAS_THRESHOLD]

        if strong_csv_indices:
            # '강력한' 라인들이 존재하면, 이들의 범위를 핵심 CSV 블록으로 간주
            core_block_start_idx = strong_csv_indices[0]
            core_block_end_idx = strong_csv_indices[-1]

            if core_block_start_idx > 0:
                print(f"INFO: CSV 시작 전 {core_block_start_idx}개의 라인을 질문으로 간주하고 제거합니다. 첫번째 제거된 라인: '{raw_lines[0][:100]}...'", flush=True)
            if core_block_end_idx < len(raw_lines) - 1:
                print(f"INFO: CSV 종료 후 {len(raw_lines) - 1 - core_block_end_idx}개의 라인을 질문으로 간주하고 제거합니다. 마지막 제거된 라인: '{raw_lines[-1][:100]}...'", flush=True)

            # 핵심 CSV 블록 추출
            core_block_lines = raw_lines[core_block_start_idx : core_block_end_idx + 1]

            if not core_block_lines: # Should not happen if strong_csv_indices is not empty
                final_answer = "❌ 오류: CSV 핵심 블록 추출에 실패했습니다."
                current_messages = state.messages
                updated_messages = current_messages + [AIMessage(content=final_answer)] if current_messages else [AIMessage(content=final_answer)]
                return {"messages": updated_messages, "final_answer": final_answer, "error_message": final_answer}

            # 핵심 블록의 첫 줄을 헤더로 사용
            header_line = core_block_lines[0]
            cleaned_lines.append(header_line)
            commas_in_header = header_line.count(',') # 헤더는 MIN_COMMAS_THRESHOLD 이상일 것임

            # 핵심 블록 내의 데이터 라인 정제
            for i in range(1, len(core_block_lines)):
                line = core_block_lines[i]
                if line.count(',') >= MIN_COMMAS_THRESHOLD: # '강력한' 데이터 라인
                    cleaned_lines.append(line)
                elif commas_in_header >= MIN_COMMAS_THRESHOLD: # 헤더는 '강력'했으나, 현재 라인은 '약함' (0개의 쉼표)
                    if not line.strip(): # 의도적으로 비어있는 라인이면 유지
                        cleaned_lines.append(line)
                    else: # 내용이 있는 0쉼표 라인은 블록 내 질문으로 의심하여 필터링
                        print(f"INFO: CSV 블록 내에서 0개의 쉼표를 가진 비어있지 않은 라인을 필터링합니다: '{line[:100]}...'", flush=True)
                else: # 헤더도 '약했고' (이 경우는 strong_csv_indices 로직상 거의 없음) 현재 라인도 '약하면' 일단 포함
                    cleaned_lines.append(line)
        else:
            # '강력한' CSV 라인이 하나도 없음 (모든 라인의 쉼표 < MIN_COMMAS_THRESHOLD, 예: 모두 0개)
            # 이 경우 단일 열 CSV이거나 전체가 질문일 수 있음. 일단 모든 라인을 사용.
            print(f"INFO: 모든 라인의 쉼표 개수가 {MIN_COMMAS_THRESHOLD}개 미만입니다. 단일 열 CSV로 간주하거나 전체가 질문일 수 있습니다. 모든 라인을 사용합니다.", flush=True)
            cleaned_lines = raw_lines

        if not cleaned_lines:
            final_answer = "❌ 오류: CSV 데이터 정제 후 내용이 없습니다."
            current_messages = state.messages
            updated_messages = current_messages + [AIMessage(content=final_answer)] if current_messages else [AIMessage(content=final_answer)]
            return {"messages": updated_messages, "final_answer": final_answer, "error_message": final_answer}
        
        cleaned_csv_data_str = "\n".join(cleaned_lines)
        
        print(f"INFO: 최종 정제된 CSV 데이터 미리보기 (첫 200자): {cleaned_csv_data_str[:200]}...", flush=True)
        input_df = await asyncio.to_thread(pd.read_csv, io.StringIO(cleaned_csv_data_str))
        # --- END REVISED CSV DATA CLEANING LOGIC ---

        if CUSTOMER_ID_COL not in input_df.columns:
            final_answer = f"❌ 오류: '{CUSTOMER_ID_COL}' 컬럼이 CSV에 없습니다."
            current_messages = state.messages # Get current messages
            updated_messages = current_messages + [AIMessage(content=final_answer)] if current_messages else [AIMessage(content=final_answer)]
            return {"messages": updated_messages, "final_answer": final_answer, "error_message": final_answer}

        customer_ids = input_df[CUSTOMER_ID_COL]
        X_predict = input_df.drop(columns=[CUSTOMER_ID_COL], errors='ignore')

        # --- 범주형 컬럼 인코딩 ---
        for col in CATEGORICAL_COLS:
            if col in X_predict.columns:
                if col in label_encoders:
                    le = label_encoders[col]
                    X_predict[col] = X_predict[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    print(f"WARNING: Label encoder for column '{col}' not found. Skipping encoding.")
            else:
                print(f"WARNING: Categorical column '{col}' not found in input CSV. Skipping.")

        # --- 누락된 컬럼 처리 (모델이 기대하는 모든 컬럼이 있는지 확인) ---
        missing_cols = set(EXPECTED_FEATURE_ORDER) - set(X_predict.columns)
        for col in missing_cols:
            print(f"INFO: Adding missing column '{col}' with default value 0.")
            X_predict[col] = 0 # 또는 np.nan 등 적절한 기본값

        # --- 컬럼 순서 정렬 ---
        X_predict = X_predict[EXPECTED_FEATURE_ORDER]

        # --- 예측 수행 ---
        predictions_proba = await asyncio.to_thread(pipeline_final.predict_proba, X_predict)
        predictions = (predictions_proba[:, 1] >= PREDICTION_THRESHOLD).astype(int)

        # --- 결과 생성 ---
        results_df = pd.DataFrame({
            CUSTOMER_ID_COL: customer_ids,
            'Churn Probability': predictions_proba[:, 1],
            'Churn Prediction (Threshold 0.312)': predictions
        })
        results_df['Churn Prediction (Threshold 0.312)'] = results_df['Churn Prediction (Threshold 0.312)'].map({1: 'Yes', 0: 'No'})

        final_answer = "📊 고객 이탈 예측 결과:\n" + results_df.to_string(index=False)
        print(f"Prediction successful. Result preview: {final_answer[:200]}...")
        current_messages = state.messages # Get current messages
        updated_messages = current_messages + [AIMessage(content=final_answer)] if current_messages else [AIMessage(content=final_answer)]
        return {"messages": updated_messages, "final_answer": final_answer, "error_message": None}

    except pd.errors.EmptyDataError:
        error_msg = "❌ 오류: 입력된 CSV 데이터가 비어 있거나 잘못된 형식입니다. CSV 내용을 다시 확인해주세요."
        print(f"ERROR: {error_msg}")
        current_messages = state.messages # Get current messages
        updated_messages = current_messages + [AIMessage(content=error_msg)] if current_messages else [AIMessage(content=error_msg)]
        return {"messages": updated_messages, "final_answer": error_msg, "error_message": error_msg}
    except FileNotFoundError as e:
        error_msg = f"❌ 오류: 모델 또는 전처리 파일을 찾을 수 없습니다. 경로를 확인해주세요. ({e})"
        print(f"ERROR: {error_msg}")
        current_messages = state.messages # Get current messages
        updated_messages = current_messages + [AIMessage(content=error_msg)] if current_messages else [AIMessage(content=error_msg)]
        return {"messages": updated_messages, "final_answer": error_msg, "error_message": error_msg}
    except KeyError as e:
        error_msg = f"❌ 오류: CSV 데이터에 필요한 컬럼이 누락되었거나, 모델 학습 시 사용된 컬럼과 다릅니다. (오류 컬럼: {e}) CSV 파일을 확인해주세요."
        print(f"ERROR: {error_msg}")
        current_messages = state.messages # Get current messages
        updated_messages = current_messages + [AIMessage(content=error_msg)] if current_messages else [AIMessage(content=error_msg)]
        return {"messages": updated_messages, "final_answer": error_msg, "error_message": error_msg}
    except ValueError as e:
        error_msg = f"❌ 오류: 데이터 변환 중 값 오류가 발생했습니다. CSV 데이터 타입을 확인해주세요. (오류: {e})"
        print(f"ERROR: {error_msg}")
        current_messages = state.messages # Get current messages
        updated_messages = current_messages + [AIMessage(content=error_msg)] if current_messages else [AIMessage(content=error_msg)]
        return {"messages": updated_messages, "final_answer": error_msg, "error_message": error_msg}
    except Exception as e:
        error_msg = f"❌ 예측 중 알 수 없는 오류 발생: {e}"
        print(f"ERROR: {error_msg}")
        current_messages = state.messages # Get current messages
        updated_messages = current_messages + [AIMessage(content=error_msg)] if current_messages else [AIMessage(content=error_msg)]
        return {"messages": updated_messages, "final_answer": error_msg, "error_message": error_msg}

def route_sql_output(state: AgentState) -> Literal["create_visualization_node", "summarize_sql_result_node"]:
    choice = state.sql_output_choice
    # If execute_sql_node itself resulted in an error (indicated by error_message and no sql_result)
    # both visualization and summarization nodes have internal logic to handle this.
    # The choice made by the supervisor should still be respected if possible.
    if state.error_message and not state.sql_result:
        print(f"Error detected before routing SQL output: {state.error_message}. Proceeding with choice: {choice}")

    if choice == "visualize":
        print("Routing to create_visualization_node based on sql_output_choice.")
        return "create_visualization_node"
    elif choice == "summarize":
        print("Routing to summarize_sql_result_node based on sql_output_choice.")
        return "summarize_sql_result_node"
    else:
        # Fallback if sql_output_choice is somehow not set for a db_query path
        print(f"Warning: sql_output_choice is '{choice}'. Defaulting to summarize_sql_result_node.")
        return "summarize_sql_result_node"

def route_query(state: AgentState) -> Literal["generate_sql_node", "category_predict_node"]:
    query_type = state.query_type
    print(f"Routing based on query_type: {query_type}")
    if query_type == "db_query":
        return "generate_sql_node"
    elif query_type == "category_predict_query":
        return "category_predict_node"
    else:
        # This case should ideally not be reached if supervisor is strict and only returns valid types
        # Consider raising an error or defaulting to a safe node if necessary.
        print(f"ERROR: Unknown query_type '{query_type}' received in route_query. This should not happen.")
        # As a fallback, returning one of the valid nodes, though this indicates an issue upstream.
        # Depending on requirements, this could raise an exception.
        # For now, let's default to category_predict_node if something unexpected happens, though supervisor should prevent this.
        return "category_predict_node" # Or raise ValueError(f"Invalid query_type: {query_type}")

# --- Graph Definition ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("generate_sql_node", generate_sql_node)
workflow.add_node("execute_sql_node", execute_sql_node)
workflow.add_node("create_visualization_node", create_visualization_node)
workflow.add_node("summarize_sql_result_node", summarize_sql_result_node)
workflow.add_node("category_predict_node", category_predict_node)

# Set entry point
workflow.set_entry_point("supervisor")

# Conditional edges from supervisor
workflow.add_conditional_edges(
    "supervisor",
    route_query,
    {
        "generate_sql_node": "generate_sql_node",
        "category_predict_node": "category_predict_node"
    }
)

# Edges for DB query flow (now common for 3 branches)
workflow.add_edge("generate_sql_node", "execute_sql_node")
# After execute_sql_node, route to either visualization or summarization
workflow.add_conditional_edges(
    "execute_sql_node",
    route_sql_output,
    {
        "create_visualization_node": "create_visualization_node",
        "summarize_sql_result_node": "summarize_sql_result_node"
    }
)
workflow.add_edge("create_visualization_node", END)
workflow.add_edge("summarize_sql_result_node", END)

# Edges from placeholder nodes to the SQL generation flow
workflow.add_edge("category_predict_node", END)

# Compile the graph
app = workflow.compile()
graph = app # For langgraph dev compatibility

# To make it runnable with langgraph dev, ensure it's assigned to 'graph'
# Example of how to run (for testing locally):
# async def main():
#     inputs = {"user_query": "지난 달 사용자 수는 몇 명인가요?"}
#     # For testing, you can invoke the graph like this:
#     # inputs = {"messages": [HumanMessage(content="우리 회사 직원들 중 가장 연봉이 높은 상위 3명은 누구인가요?")]}
#     # async for event in app.astream_events(inputs, version="v1"):
#     #     kind = event["event"]
#     #     if kind == "on_chat_model_stream":
#     #         content = event["data"]["chunk"].content
#     #         if content:
#     #             print(content, end="")
#     #     elif kind == "on_tool_end":
#     #         print(f"\nTool Output: {event['data']['output']}")
#     #     # print(f"\n--- Event: {kind} ---")
#     #     # print(event["data"])
# if __name__ == "__main__":
#     import asyncio
#     async def main_test():
#         app_test = await main()
#         inputs = {"input": "우리 회사 테이블 목록 좀 보여줘"}
#     # inputs = {"input": "오늘 날씨 어때?"}
#     async for event in app_test.astream_events(inputs, version="v1"):
#             kind = event["event"]
#             if kind == "on_chat_model_stream":
#                 content = event["data"]["chunk"].content
#                 if content:
#                     print(content, end="")
#                 print(f"\nTool Output: {event['data']['output']}")
#             elif kind == "on_chain_end" and event["name"] == "AgentGraph": # Check for the graph's end
#                 print("\n--- Final State ---")
#                 print(event["data"].get("output")) # Access final output from the event

#     asyncio.run(main_test())
