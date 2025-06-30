#!/usr/bin/env python3
"""
SKN10 MCP Server 문서/예측 에이전트 테스트 클라이언트

문서 검색(search_internal_policy)과 고객 이탈 예측(predict_churn)만 테스트합니다.
"""

import asyncio
import json
from fastmcp import Client

async def test_document_search():
    """문서 검색 도구를 테스트합니다."""
    print("\n🔍 문서 검색 도구 테스트:")
    try:
        async with Client("http://localhost:8000/mcp") as client:
            result = await client.call_tool("search_internal_policy", {
                "query": "휴가 정책",
                "top_k": 2
            })
            if isinstance(result, list):
                for i, item in enumerate(result):
                    text = getattr(item, "text", str(item))
                    print(f"  ✅ 내부 정책 검색 결과 {i+1}: {text[:150]}...")
            else:
                text = getattr(result, "text", str(result))
                print(f"  ✅ 내부 정책 검색 결과: {text[:150]}...")
    except Exception as e:
        print(f"  ❌ 문서 검색 테스트 실패: {e}")

async def test_predict_churn():
    """고객 이탈 예측 도구를 테스트합니다."""
    print("\n🤖 고객 이탈 예측 도구 테스트:")
    try:
        async with Client("http://localhost:8000/mcp") as client:
            dummy_csv = """customerid,gender,seniorcitizen,partner,dependents,tenure\nCUST001,Male,0,Yes,No,12\nCUST002,Female,1,No,Yes,24"""
            result = await client.call_tool("predict_churn", {
                "csv_data_string": dummy_csv
            })
            if isinstance(result, list):
                for i, item in enumerate(result):
                    text = getattr(item, "text", str(item))
                    print(f"  ✅ 예측 결과 {i+1}: {text[:100]}...")
            else:
                text = getattr(result, "text", str(result))
                print(f"  ✅ 예측 결과: {text[:100]}...")
    except Exception as e:
        print(f"  ❌ 고객 이탈 예측 테스트 실패: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 SKN10 MCP Server 문서/예측 에이전트 테스트 클라이언트")
    print("=" * 60)
    
    asyncio.run(test_document_search())
    asyncio.run(test_predict_churn())
    
    print("\n" + "=" * 60)
    print("🎉 테스트 완료!")
    print("=" * 60) 