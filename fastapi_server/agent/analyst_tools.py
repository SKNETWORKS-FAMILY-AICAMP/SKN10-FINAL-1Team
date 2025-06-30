import json
import uuid
from typing import Dict, Any, List, Optional
import os
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

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

class ChartInputArgs(BaseModel):
    title: str = Field(..., description="The title for the chart.")
    chart_type: str = Field(..., description="Type of chart (e.g., 'bar', 'line', 'pie').")
    data: Dict[str, Any] = Field(..., description="Data for the chart, following Chart.js structure (labels, datasets).")
    options: Optional[Dict[str, Any]] = Field(None, description="Optional Chart.js options to override defaults.")

analyst_chart_tool = StructuredTool.from_function(
    func=generate_chart_html,
    name="ChartGenerator",
    description="Generates the necessary HTML and JavaScript for a chart. Returns a JSON string with 'canvas_html' and 'script_js' keys.",
    args_schema=ChartInputArgs,
)

# --- SQL Database Tools for Analyst ---
DB_URI = os.getenv("DB_URI")
sql_tools_for_analyst = []

if DB_URI:
    try:
        engine = create_engine(DB_URI)
        db = SQLDatabase(engine)
        llm_for_sql_toolkit = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
        sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm_for_sql_toolkit)
        sql_tools_for_analyst = sql_toolkit.get_tools()
        print("SQL Database tools initialized successfully for Analyst.", file=sys.stderr)
    except Exception as e:
        print(f"Error initializing SQL Database tools: {e}", file=sys.stderr)
        print("SQL tools will be unavailable for the Analyst Assistant.", file=sys.stderr)
        sql_tools_for_analyst = []
else:
    print("DB_URI not found. SQL tools will be unavailable.", file=sys.stderr)
    sql_tools_for_analyst = []

# --- Export all analyst tools ---
analyst_tools = [analyst_chart_tool] + sql_tools_for_analyst

__all__ = [
    "analyst_chart_tool",
    "sql_tools_for_analyst", 
    "analyst_tools"
] 