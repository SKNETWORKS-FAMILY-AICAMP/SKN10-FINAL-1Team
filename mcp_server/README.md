# SKN10 MCP Server 🚀

[FastMCP](https://github.com/jlowin/fastmcp)를 사용하여 구축된 통합 MCP(Model Context Protocol) 서버입니다.

## 📋 기능

### 🔍 Document Search Tools (Pinecone 검색)
- **내부 정책 검색**: 휴가 정책, 복지, 행동 강령 등
- **기술 문서 검색**: 개발 가이드, API 명세서 등
- **제품 문서 검색**: 매뉴얼, 기능 설명, 사용자 가이드
- **회의록 검색**: 결정 사항, 업무 지시사항 등

### 📊 Analyst Tools (데이터 분석)
- **차트 생성**: Chart.js를 사용한 동적 차트 생성
- **SQL 도구**: 데이터베이스 쿼리 및 분석 (선택적)

### 🤖 Prediction Tools (머신러닝)
- **고객 이탈 예측**: CSV 데이터 기반 고객 이탈 예측

### 💻 GitHub Coding Tools
- **리포지토리 관리**: 목록 조회, 파일 읽기/쓰기
- **이슈 관리**: 이슈 목록 조회 및 관리
- **Python 실행**: 코드 실행 및 결과 반환
- **코드 검색**: Pinecone 기반 코드 문서 검색

## 🚀 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 다음 환경 변수를 설정하세요:

```env
# OpenAI API (문서 검색용)
OPENAI_API_KEY=your_openai_api_key

# Pinecone 설정 (문서 검색용)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
PINECONE_INDEX_NAME1=dense-index

# 데이터베이스 (SQL 도구용, 선택적)
DB_URI=postgresql://user:password@host:port/database

# GitHub (코딩 도구용, 각 호출시 토큰 전달)
# GITHUB_TOKEN은 각 도구 호출시 매개변수로 전달
```

### 3. 서버 실행

#### 모듈 방식으로 실행
```bash
python -m mcp_server.server
```

#### 직접 실행
```bash
cd mcp_server
python server.py
```

서버가 `http://0.0.0.0:8000`에서 실행됩니다.

## 🛠️ 사용법

### FastMCP 클라이언트로 연결
```python
from fastmcp import Client

async def main():
    # HTTP 연결
    async with Client("http://localhost:8000/mcp") as client:
        tools = await client.list_tools()
        print(f"사용 가능한 도구: {tools}")
        
        # 예시: 내부 정책 검색
        result = await client.call_tool("search_internal_policy", {
            "query": "휴가 정책",
            "top_k": 3
        })
        print(f"검색 결과: {result.text}")
```

### 도구별 사용 예시

#### 1. 문서 검색
```python
# 내부 정책 검색
await client.call_tool("search_internal_policy", {
    "query": "휴가 정책",
    "top_k": 3
})

# 기술 문서 검색
await client.call_tool("search_technical_docs", {
    "query": "API 사용법",
    "top_k": 5
})
```

#### 2. 차트 생성
```python
await client.call_tool("generate_chart", {
    "title": "월별 매출",
    "chart_type": "bar",
    "data": {
        "labels": ["1월", "2월", "3월"],
        "datasets": [{
            "label": "매출",
            "data": [100, 150, 200],
            "backgroundColor": "rgba(54, 162, 235, 0.2)"
        }]
    }
})
```

#### 3. 고객 이탈 예측
```python
csv_data = """
customerid,gender,seniorcitizen,partner,dependents,tenure
CUST001,Male,0,Yes,No,12
CUST002,Female,1,No,Yes,24
"""

await client.call_tool("predict_churn", {
    "csv_data_string": csv_data
})
```

#### 4. GitHub 도구
```python
# 리포지토리 목록
await client.call_tool("github_list_repositories", {
    "token": "your_github_token",
    "username": "your_username"
})

# 파일 읽기
await client.call_tool("github_read_file", {
    "token": "your_github_token",
    "repo_full_name": "owner/repo",
    "file_path": "README.md"
})
```

#### 5. Python 코드 실행
```python
await client.call_tool("execute_python", {
    "code": "print('Hello, MCP World!')"
})
```

## 🔧 설정

### Transport 방식 변경
서버는 기본적으로 HTTP transport를 사용하지만, 다른 방식으로도 실행할 수 있습니다:

```python
# STDIO (기본값)
mcp.run(transport="stdio")

# SSE
mcp.run(transport="sse", host="0.0.0.0", port=8000)

# HTTP (현재 설정)
mcp.run(transport="http", host="0.0.0.0", port=8000)
```

### 포트 변경
```python
mcp.run(transport="http", host="0.0.0.0", port=9000)
```

## 📦 파일 구조
```
mcp_server/
├── __init__.py
├── server.py              # 메인 서버 파일
├── doc_search_tools.py    # Pinecone 문서 검색 도구
├── analyst_tools.py       # 데이터 분석 도구
├── predict_tools.py       # 머신러닝 예측 도구
├── coding_tools.py        # GitHub 및 코딩 도구
├── requirements.txt       # 의존성 목록
└── README.md             # 이 파일
```

## 🎯 주의사항

1. **환경 변수**: 모든 필요한 API 키와 설정을 `.env` 파일에 올바르게 설정하세요.
2. **의존성**: 일부 도구는 선택적 의존성을 가집니다. 필요에 따라 추가 설치하세요.
3. **보안**: GitHub 토큰, API 키 등은 안전하게 관리하세요.
4. **모델 파일**: 예측 도구는 `models/churn_pipeline_full.pkl` 파일이 필요합니다.

## 🤝 기여

이 프로젝트는 [FastMCP](https://github.com/jlowin/fastmcp) 프레임워크를 기반으로 구축되었습니다. 