# mcp_server/server.py
import os
from dotenv import load_dotenv
load_dotenv()
from fastmcp import FastMCP

# 각 도구 모듈 import
import doc_search_tools
import predict_tools


# FastMCP 서버 인스턴스 생성
mcp = FastMCP("SKN10 MCP Server")

# --- Document Search Tools 등록 (Pinecone 검색) ---
@mcp.tool
def search_internal_policy(query: str, top_k: int = 3) -> str:
    """내부 정책 및 HR 문서를 검색합니다 (휴가 정책, 복지, 행동 강령 등)."""
    return doc_search_tools.internal_policy_search(query, top_k)

@mcp.tool
def search_technical_docs(query: str, top_k: int = 3) -> str:
    """기술 문서, 개발 가이드, API 명세서를 검색합니다."""
    return doc_search_tools.tech_doc_search(query, top_k)

@mcp.tool
def search_product_docs(query: str, top_k: int = 3) -> str:
    """제품 매뉴얼, 기능 설명, 사용자 가이드를 검색합니다."""
    return doc_search_tools.product_doc_search(query, top_k)

@mcp.tool
def search_proceedings(query: str, top_k: int = 3) -> str:
    """회의록, 결정 사항, 업무 지시사항을 검색합니다."""
    return doc_search_tools.proceedings_search(query, top_k)


# --- 파일명 기반 회의록 검색 툴 등록 ---
@mcp.tool
def search_proceedings_by_filename(filename: str, top_k: int = 3) -> str:
    """파일명으로 Pinecone proceedings namespace에서 회의록을 검색합니다."""
    return doc_search_tools.proceedings_text_with_filename(filename, top_k)


# --- Prediction Tools 등록 (머신러닝 예측) ---
@mcp.tool
def predict_churn(csv_data_string: str) -> str:
    """고객 이탈 예측을 수행합니다. CSV 형식의 고객 데이터를 입력하세요."""
    return predict_tools.predict_customer_churn(csv_data_string)



if __name__ == "__main__":
    print("🚀 SKN10 MCP Server 시작 중...")
    print("📍 사용 가능한 도구들:")
    print("  - Document Search: 내부 정책, 기술 문서, 제품 문서, 회의록 검색")
    print("  - Analyst Tools: 차트 생성")
    print("  - Prediction: 고객 이탈 예측")
    print(f"🌐 서버가 http://0.0.0.0:8000 에서 실행됩니다.")
    
    # HTTP 방식으로 MCP 서버 실행
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000) 