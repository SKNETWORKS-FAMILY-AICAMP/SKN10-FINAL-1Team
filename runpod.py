import os
import time  # 시간 측정을 위해 추가
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

def get_runpod_chat_model():
    # .env 파일에서 환경 변수 로드 (함수 호출 시점에 다시 로드될 수 있으나, 전역 로드도 유지)
    # load_dotenv() # 함수 내에서 호출하거나, 호출하는 쪽에서 이미 로드되었다고 가정

    MODEL_NAME = os.getenv("RUNPOD_MODEL_NAME", "unsloth/Qwen3-32B-bnb-4bit")
    OPENAI_API_KEY = os.getenv("RUNPOD_API_KEY", "")
    OPENAI_API_BASE = os.getenv("RUNPOD_BASE_URL", "")

    chat = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        temperature=0.7,
        max_tokens=8192,
        top_p=0.8,
        presence_penalty=1.5,
        model_kwargs={
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        },
    )
    return chat

# ChatOpenAI 클라이언트 초기화 (테스트용)
if __name__ == "__main__":
    chat = get_runpod_chat_model()

    # 메시지 생성 (테스트용)
    messages = [
        HumanMessage(content="세종대왕이 뭘 했어?"),
    ]

    # 시간 측정 시작
    start_time = time.time()

    # 모델 호출
    response = chat.invoke(messages)

    # 시간 측정 종료
    end_time = time.time()

    # 응답 출력
    print(f"응답 내용: {response.content}")

    # TPS 계산
    duration = end_time - start_time
    # 간단하게 글자 수를 토큰 수로 가정 (정확한 토큰 수는 모델별 토크나이저 필요)
    num_tokens = len(response.content) 
    tps = num_tokens / duration if duration > 0 else 0

    print(f"\n--- 성능 측정 ---")
    print(f"호출 시간: {duration:.4f} 초")
    print(f"응답 토큰 수 (글자 수 기준): {num_tokens} 개")
    print(f"초당 토큰 (TPS): {tps:.2f} tokens/sec")
