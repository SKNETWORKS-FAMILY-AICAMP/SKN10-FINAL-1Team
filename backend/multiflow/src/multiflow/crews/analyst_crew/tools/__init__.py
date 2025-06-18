# src/analyst_crew/tools/__init__.py
from crewai_tools import NL2SQLTool
import os

DB_URI = os.getenv("DB_URI") or "sqlite:///memory"   # 개발용 기본값
nl2sql = NL2SQLTool(db_uri=DB_URI)

__all__ = ["nl2sql"]      # (선택) 외부 export 목록
