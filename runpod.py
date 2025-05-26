import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 이름 설정 (환경 변수 RUNPOD_MODEL_NAME 또는 기본값 사용)
MODEL_NAME = os.getenv("RUNPOD_MODEL_NAME", "unsloth/Qwen3-32B-bnb-4bit")

# API 키 및 Base URL 설정 (환경 변수에서 가져오거나, 없을 경우 기존 하드코딩된 값 사용 - 보안상 .env 사용 권장)
# .env 파일에 다음을 추가하세요:
# RUNPOD_API_KEY="your_runpod_api_key"
# RUNPOD_BASE_URL="your_runpod_base_url"

OPENAI_API_KEY = os.getenv("RUNPOD_API_KEY", "")
OPENAI_API_BASE = os.getenv("RUNPOD_BASE_URL", "")

# ChatOpenAI 클라이언트 초기화
chat = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    temperature=0.7,  # 예시 코드 값
    max_tokens=8192,  # 예시 코드 값
    top_p=0.8,        # 예시 코드 값
    presence_penalty=1.5, # 예시 코드 값
    model_kwargs={
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False}, # 요청하신 파라미터 반영
        }
    },
)

# 메시지 생성 (예시 코드에 맞춰 수정)
messages = [
    HumanMessage(content="Give me a short introduction to large language models."),
]

# 모델 호출
response = chat.invoke(messages)

# 응답 출력
print(response.content)