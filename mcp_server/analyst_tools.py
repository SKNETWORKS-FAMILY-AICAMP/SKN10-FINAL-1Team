import json
import uuid
from typing import Dict, Any, List, Optional
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
from pydantic import BaseModel, Field

# SQL 관련 imports (선택적)
try:
    from langchain_community.utilities.sql_database import SQLDatabase
    from sqlalchemy import create_engine
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    from langchain_openai import ChatOpenAI
    SQL_AVAILABLE = True
except ImportError:
    SQL_AVAILABLE = False
    print("SQL Database tools not available. Install langchain-community and sqlalchemy for SQL support.", file=sys.stderr)

# --- Analyst Chart Tool ---
def generate_chart_html(title: str, chart_type: str, data: dict, options: Optional[Dict[str, Any]] = None) -> str:
    """Generates chart data as a JSON string containing HTML for the canvas and JS for the script."""
    chart_id = f"chart-{uuid.uuid4().hex[:8]}"
    
    chart_options = {
        'responsive': True,
        'plugins': {
            'title': {
                'display': True,
                'text': title
            }
        }
    }
    if options:
        chart_options.update(options)

    data_json = json.dumps(data)
    options_json = json.dumps(chart_options)
    
    canvas_html = f"<div><canvas id='{chart_id}'></canvas></div>"
    
    chart_id_js_safe = chart_id.replace('-', '_')
    script_js = f"""
      const ctx_{chart_id_js_safe} = document.getElementById('{chart_id}');
      if (ctx_{chart_id_js_safe}) {{
        new Chart(ctx_{chart_id_js_safe}, {{
          type: '{chart_type}',
          data: {data_json},
          options: {options_json}
        }});
      }} else {{
        console.error('Failed to find canvas element with ID: {chart_id}');
      }}
    """
    
    output = {
        "canvas_html": canvas_html,
        "script_js": script_js
    }
    return json.dumps(output)

# --- SQL Database Tools for Analyst (if available) ---
def get_sql_tools():
    """SQL 도구들을 반환합니다. SQL이 사용 불가능하면 빈 리스트를 반환합니다."""
    if not SQL_AVAILABLE:
        return []
    
    DB_URI = os.getenv("DB_URI")
    if not DB_URI:
        print("DB_URI not found. SQL tools will be unavailable.", file=sys.stderr)
        return []
    
    try:
        engine = create_engine(DB_URI)
        db = SQLDatabase(engine)
        llm_for_sql_toolkit = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
        sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm_for_sql_toolkit)
        sql_tools = sql_toolkit.get_tools()
        print("SQL Database tools initialized successfully for Analyst.", file=sys.stderr)
        return sql_tools
    except Exception as e:
        print(f"Error initializing SQL Database tools: {e}", file=sys.stderr)
        print("SQL tools will be unavailable for the Analyst Assistant.", file=sys.stderr)
        return [] 