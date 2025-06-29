# Pinecone 검색 도구 사용법

## 개요

이 문서는 `coding_agent_tools.py`에 구현된 Pinecone 검색 도구의 사용법을 설명합니다.

## 구현된 도구

### 1. `github_search_code_documents`
- **기능**: 기본 Pinecone 검색 (임시 벡터 사용)
- **네임스페이스**: 사용자 ID
- **필터링**: 리포지토리 경로, 브랜치명
- **반환**: 검색 결과 목록 (ID, 점수, 메타데이터)

### 2. `github_search_code_documents_with_embedding`
- **기능**: OpenAI 임베딩을 사용한 정확한 벡터 검색
- **네임스페이스**: 사용자 ID
- **필터링**: 리포지토리 경로, 브랜치명
- **반환**: 검색 결과 목록 (ID, 점수, 메타데이터)

## 환경 변수 설정

다음 환경 변수들이 필요합니다:

```bash
# Pinecone 설정
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_INDEX_HOST=your_index_host

# OpenAI 설정 (임베딩 검색용)
OPENAI_API_KEY=your_openai_api_key
```

## 사용 예시

### 기본 검색
```python
from agent.coding_agent_tools import _search_tutorials

# 모든 튜토리얼에서 검색
results = _search_tutorials(
    user_id="be87b5f4-1096-46fd-9992-5703b9ef1229",
    query="자동차 등록 현황",
    top_k=5
)

# 특정 리포지토리에서만 검색
results = _search_tutorials(
    user_id="be87b5f4-1096-46fd-9992-5703b9ef1229",
    query="자동차 등록 현황",
    repo_path="SKNETWORKS-FAMILY-AICAMP/SKN10-1st-4Team",
    top_k=5
)

# 특정 브랜치에서만 검색
results = _search_tutorials(
    user_id="be87b5f4-1096-46fd-9992-5703b9ef1229",
    query="자동차 등록 현황",
    branch="main",
    top_k=5
)
```

### 임베딩 검색 (권장)
```python
from agent.coding_agent_tools import _search_tutorials_with_embedding

# 정확한 벡터 검색
results = _search_tutorials_with_embedding(
    user_id="be87b5f4-1096-46fd-9992-5703b9ef1229",
    query="자동차 등록 현황",
    top_k=5
)

# 필터링과 함께 사용
results = _search_tutorials_with_embedding(
    user_id="be87b5f4-1096-46fd-9992-5703b9ef1229",
    query="자동차 등록 현황",
    repo_path="SKNETWORKS-FAMILY-AICAMP/SKN10-1st-4Team",
    branch="main",
    top_k=5
)
```

## 반환 데이터 구조

```python
[
    {
        "id": "02_지역별_자동차_등록_현황_페이지.md_6b1e4f68ea14483aac9032d5e87e791e",
        "score": 0.85,  # 유사도 점수 (0-1)
        "metadata": {
            "original_filename": "02_지역별_자동차_등록_현황_페이지.md",
            "github_user_repo": "SKNETWORKS-FAMILY-AICAMP/SKN10-1st-4Team",
            "branch_name": "main",
            "original_document_id": "SKN10-1st-4Team_b3f4afc18f7944348dfbf24f3b4e6d3d",
            "text": "# Chapter 2: 지역별 자동차 등록 현황 페이지\n\n이전 [제 1 장: 시각화 라이브러리 사용]...",
            # 기타 메타데이터...
        }
    },
    # ... 더 많은 결과
]
```

## 필터링 옵션

### 리포지토리 경로 필터링
```python
repo_path="SKNETWORKS-FAMILY-AICAMP/SKN10-1st-4Team"  # 특정 리포지토리만 검색
```

### 브랜치 필터링
```python
branch="main"  # 특정 브랜치만 검색
```

### 복합 필터링
```python
# 특정 리포지토리의 특정 브랜치에서만 검색
results = _search_tutorials_with_embedding(
    user_id="be87b5f4-1096-46fd-9992-5703b9ef1229",
    query="자동차 등록 현황",
    repo_path="SKNETWORKS-FAMILY-AICAMP/SKN10-1st-4Team",
    branch="main",
    top_k=5
)
```

## 에러 처리

### 일반적인 에러
- **환경 변수 누락**: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_INDEX_HOST` 확인
- **OpenAI 라이브러리 누락**: `pip install openai` 실행
- **네트워크 오류**: Pinecone 서비스 연결 상태 확인

### 에러 메시지 예시
```
RuntimeError: PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.
RuntimeError: OpenAI 라이브러리가 설치되지 않았습니다. 'pip install openai'를 실행하세요.
RuntimeError: Pinecone 검색 중 오류 발생: [구체적인 오류 내용]
```

## 테스트

테스트 스크립트를 실행하여 도구가 정상적으로 작동하는지 확인할 수 있습니다:

```bash
cd fastapi_server
python test_pinecone_search.py
```

## 주의사항

1. **네임스페이스**: 사용자 ID가 네임스페이스로 사용되므로, 각 사용자의 튜토리얼이 분리됩니다.
2. **벡터 차원**: 기본 검색은 1536차원 임시 벡터를 사용합니다. 정확한 검색을 위해서는 임베딩 검색을 사용하세요.
3. **API 제한**: Pinecone과 OpenAI API 사용량 제한을 확인하세요.
4. **메타데이터**: 검색 결과의 메타데이터는 튜토리얼 업로드 시 설정된 내용에 따라 달라집니다.
5. **필터링 필드**: 실제 Pinecone 데이터에서는 `github_user_repo`와 `branch_name` 필드를 사용합니다.

## LangGraph 에이전트에서 사용

이 도구들은 LangGraph 에이전트에서 다음과 같이 사용할 수 있습니다:

```python
from agent.coding_agent_tools import get_all_coding_tools

# 모든 도구 가져오기 (Pinecone 검색 도구 포함)
tools = get_all_coding_tools()

# 에이전트에 도구 할당
agent = create_agent(tools=tools)
```

에이전트는 자동으로 `github_search_code_documents`와 `github_search_code_documents_with_embedding` 도구를 인식하고 사용할 수 있습니다.

## 실제 데이터 예시

Pinecone에 저장된 실제 데이터 구조:

```json
{
  "namespace": "be87b5f4-1096-46fd-9992-5703b9ef1229",
  "id": "02_지역별_자동차_등록_현황_페이지.md_6b1e4f68ea14483aac9032d5e87e791e",
  "metadata": {
    "branch_name": "main",
    "github_user_repo": "SKNETWORKS-FAMILY-AICAMP/SKN10-1st-4Team",
    "original_document_id": "SKN10-1st-4Team_b3f4afc18f7944348dfbf24f3b4e6d3d",
    "original_filename": "02_지역별_자동차_등록_현황_페이지.md",
    "text": "# Chapter 2: 지역별 자동차 등록 현황 페이지\n\n..."
  },
  "values": [0.00952531863, -0.0268489849, ...]  // 1536차원 벡터
}
``` 