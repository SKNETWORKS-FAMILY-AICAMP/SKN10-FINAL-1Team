# LangGraph 에이전트 정량적 평가 툴 가이드

이 가이드는 `fastapi_server/agent/graph.py`에 정의된 LangGraph 에이전트들을 LangSmith를 이용하여 정량적으로 평가하는 방법을 설명합니다.

## 📋 개요

### 평가 대상 에이전트
- **📄 doc_search_assistant**: 문서 검색 전문 에이전트
- **📊 analyst_assistant**: 데이터 분석 및 차트 생성 에이전트  
- **🔮 predict_assistant**: 고객 이탈 예측 에이전트
- **💻 coding_assistant**: 코딩 및 GitHub 관련 에이전트

### 평가 지표
- **관련성 (Relevance)**: 응답이 질문과 얼마나 관련이 있는지
- **완성도 (Completeness)**: 답변이 질문에 대해 완전한 정보를 제공하는지
- **에이전트 협업 (Agent Cooperation)**: 에이전트 간 핸드오프가 적절히 이루어지는지

## 🛠️ 설치 및 설정

### 1. 의존성 설치
```bash
# fastapi_server 디렉토리에서 실행
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일에 다음 환경 변수들을 설정하세요:

```bash
# LangSmith 설정
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=langgraph-agent-evaluation

# 데이터베이스 설정 
DB_URI=postgresql://user:password@host:port/database

# OpenAI API 설정 (평가 모델용)
OPENAI_API_KEY=your_openai_api_key_here

# 기타 필요한 환경 변수들
MCP_SERVER_URL=your_mcp_server_url
```

### 3. LangSmith 계정 설정
1. [LangSmith](https://smith.langchain.com)에 가입하세요
2. API 키를 발급받으세요
3. 프로젝트를 생성하세요

## 🚀 사용법

### 전체 에이전트 평가
```bash
python run_evaluation.py --all
```

### 특정 에이전트만 평가
```bash
# 문서 검색 에이전트만 평가
python run_evaluation.py --agent doc_search

# 분석 에이전트만 평가  
python run_evaluation.py --agent analyst

# 예측 에이전트만 평가
python run_evaluation.py --agent predict

# 코딩 에이전트만 평가
python run_evaluation.py --agent coding
```

### 고급 옵션
```bash
# 동시 실행 수 조정 (기본값: 2)
python run_evaluation.py --all --concurrency 4

# 결과 파일명 지정
python run_evaluation.py --all --output my_evaluation_results

# 빠른 평가 모드 (테스트 케이스 축소)
python run_evaluation.py --all --quick
```

## 📊 평가 과정

### 1. 데이터셋 생성
각 에이전트별로 다음과 같은 테스트 케이스가 자동 생성됩니다:

#### 문서 검색 에이전트
- 회사 정책 조회
- 회의록 검색
- 기술 문서 검색
- 제품 매뉴얼 검색

#### 분석 에이전트
- 데이터베이스 쿼리
- 차트 생성
- 통계 분석
- 뉴스 데이터 분석

#### 예측 에이전트  
- 고객 이탈 예측
- CSV 데이터 처리

#### 코딩 에이전트
- 라이브러리 사용법 질문
- 코딩 패턴 질문
- GitHub 관련 질문

### 2. 평가 실행
- LangSmith의 `aevaluate` 함수를 사용하여 비동기로 평가 실행
- GPT-4o를 판정 모델로 사용하여 응답 품질 평가
- 각 테스트 케이스에 대해 관련성, 완성도 등을 0-1 점수로 평가

### 3. 결과 분석
- 에이전트별 평균 점수 계산
- 강점과 약점 식별
- 개선 사항 제안

## 📁 결과 파일

평가 완료 후 다음 파일들이 생성됩니다:

### `evaluation_results_YYYYMMDD_HHMMSS.json`
```json
{
  "doc_search": {
    "timestamp": "20240115_143022",
    "status": "completed",
    "details": "평가 결과 상세 정보"
  },
  "analyst": {
    "timestamp": "20240115_143022", 
    "status": "completed",
    "details": "평가 결과 상세 정보"
  }
}
```

### LangSmith 대시보드
- 각 에이전트별 상세한 평가 로그
- 실행 추적 정보
- 성능 지표 시각화

## 🔧 커스터마이징

### 테스트 케이스 수정
`evaluation_config.py`의 `AGENT_TEST_CASES`를 수정하여 테스트 케이스를 변경할 수 있습니다:

```python
AGENT_TEST_CASES = {
    "doc_search": {
        "questions": [
            "새로운 테스트 질문 추가",
            # ... 기존 질문들
        ],
        "expected_keywords": [
            ["새로운", "키워드", "추가"],
            # ... 기존 키워드들  
        ]
    }
}
```

### 평가자 수정
`evaluation_tool.py`의 평가 함수들을 수정하여 평가 기준을 변경할 수 있습니다:

```python
async def custom_evaluator(self, outputs: Dict, reference_outputs: Dict) -> bool:
    """커스텀 평가 로직 구현"""
    # 여기에 새로운 평가 로직 추가
    pass
```

### 평가 지표 추가
새로운 평가 지표를 추가하려면:

1. `evaluation_config.py`의 `EVALUATION_METRICS`에 새 지표 정의
2. `evaluation_tool.py`에 해당 평가 함수 구현
3. `aevaluate` 호출 시 새 평가자 추가

## 🐛 문제 해결

### 자주 발생하는 오류

#### 1. 환경 변수 오류
```
ValueError: LANGSMITH_API_KEY 환경변수가 설정되지 않았습니다.
```
**해결**: `.env` 파일에 `LANGSMITH_API_KEY` 설정

#### 2. 데이터베이스 연결 오류
```
psycopg2.OperationalError: connection failed
```
**해결**: `DB_URI` 환경변수 확인, 데이터베이스 서버 상태 확인

#### 3. OpenAI API 오류
```
openai.AuthenticationError: Invalid API key
```
**해결**: `OPENAI_API_KEY` 환경변수 확인

### 디버깅
평가 중 오류가 발생하면:
1. 로그 확인
2. 환경 변수 재확인
3. 네트워크 연결 상태 확인
4. LangSmith 대시보드에서 세부 로그 확인

## 📈 성능 최적화

### 동시 실행 수 조정
```bash
# CPU 코어 수에 맞춰 조정 (권장: 2-4)
python run_evaluation.py --all --concurrency 4
```

### 평가 시간 단축
```bash
# 빠른 평가 모드 사용
python run_evaluation.py --all --quick
```

### 에이전트별 개별 평가
특정 에이전트만 평가하여 시간 절약:
```bash
python run_evaluation.py --agent doc_search
```

## 📚 추가 자료

- [LangSmith 공식 문서](https://docs.smith.langchain.com/)
- [LangGraph 평가 가이드](https://langchain-ai.github.io/langgraph/tutorials/#evaluation)
- [LangChain 평가 프레임워크](https://python.langchain.com/docs/guides/evaluation/)

## 🤝 기여

평가 툴 개선을 위한 제안이나 버그 리포트는 언제든 환영합니다!

1. 이슈 생성
2. 기능 제안
3. 코드 개선 PR 