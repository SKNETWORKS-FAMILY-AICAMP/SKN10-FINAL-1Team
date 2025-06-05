# LangGraph Agent FastAPI Server

이 프로젝트는 LangGraph 기반 멀티 에이전트 시스템을 FastAPI 웹 서버로 서빙하는 어플리케이션입니다.

## 프로젝트 구조

```
fastapi_server/
├── __init__.py          # 패키지 초기화 파일
├── main.py              # FastAPI 애플리케이션 메인 파일
├── agent_service.py     # LangGraph 에이전트와의 인터페이스
├── models.py            # API 요청/응답 모델
├── requirements.txt     # 필요한 의존성
├── start_server.py      # 서버 실행 스크립트
└── README.md            # 이 문서
```

## 설치 및 실행 방법

### 1. 가상 환경 설정

항상 가상 환경을 사용하는 것이 좋습니다. 가상 환경을 생성하고 활성화하세요:

```bash
# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화 (Windows)
venv\Scripts\activate

# 가상 환경 활성화 (Linux/Mac)
source venv/bin/activate
```

### 2. 의존성 설치

필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

프로젝트 루트 디렉토리에 `.env` 파일이 있는지 확인하세요. 다음과 같은 환경 변수가 필요합니다:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. 서버 실행

다음 명령을 사용하여 서버를 시작합니다:

```bash
# 방법 1: start_server.py 스크립트 사용
python start_server.py

# 방법 2: uvicorn 직접 사용
uvicorn fastapi_server.main:app --host 0.0.0.0 --port 8001 --reload
```

서버가 성공적으로 시작되면 http://localhost:8001 에서 접근할 수 있습니다.

## API 엔드포인트

### 1. 상태 확인

- **URL**: GET /
- **응답**: 서버 상태 정보

### 2. 채팅 메시지 전송

- **URL**: POST /api/chat
- **요청 본문**:
  ```json
  {
    "message": "사용자 메시지",
    "thread_id": "선택적_대화_ID"
  }
  ```
- **응답**: 에이전트 응답

### 3. WebSocket 스트리밍 채팅

- **URL**: WebSocket /api/chat/ws/{thread_id}
- **사용법**: 
  - 연결 후, JSON 형식의 메시지 전송: `{"message": "사용자 메시지"}`
  - 서버는 다양한 이벤트 타입을 포함한 JSON 응답을 스트리밍합니다.
  - 이벤트 타입: `token`, `agent_change`, `tool_start`, `tool_end`, `done`, `error`

## 예제 사용 코드

### HTTP API 사용 예제 (Python)

```python
import requests

response = requests.post(
    "http://localhost:8001/api/chat",
    json={"message": "지역별 매출을 분석해줘"}
)
print(response.json())
```

### WebSocket 스트리밍 예제 (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8001/api/chat/ws/my-thread-1');

ws.onopen = () => {
  console.log('Connected to server');
  ws.send(JSON.stringify({
    message: '데이터 시각화를 도와줘'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
  
  if (data.type === 'token') {
    // 토큰 처리 (점진적으로 UI에 텍스트 추가)
    process.stdout.write(data.content);
  } else if (data.type === 'agent_change') {
    console.log(`Agent changed to: ${data.agent}`);
  } else if (data.type === 'done') {
    console.log('\nResponse complete!');
  }
};
```

## 문제 해결

1. **ImportError**: 필요한 모듈을 찾지 못하는 경우, 가상 환경이 활성화되어 있고 모든 의존성이 설치되어 있는지 확인하세요.

2. **API 키 오류**: OPENAI_API_KEY가 올바르게 설정되어 있는지 확인하세요.

3. **포트 충돌**: 8001 포트가 이미 사용 중인 경우, `main.py`에서 포트 번호를 변경하세요.
