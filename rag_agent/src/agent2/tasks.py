------------- tasks.py -------------\n# (원본에서 변경 없음 — 동기 함수로 이미 작성되어 있으므로 그대로 사용)\n# 파일 전문은 생략했으나 embed_query / retrieve_best_namespace / build_context\n# / generate_answer 네 함수가 그대로여야 합니다.\n
# rag_agent/src/agent/tasks.py

import os
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv

# ─── tasks.py가 import 되는 순간 .env를 읽어오도록 설정 ───
HERE         = Path(__file__).resolve().parent       # → rag_agent/src/agent
PROJECT_ROOT = HERE.parent.parent.parent.parent       # → SKN10-FINAL_1TEAM (최상위)
DOTENV_PATH  = PROJECT_ROOT / ".env"                  # → SKN10-FINAL_1TEAM/.env
load_dotenv(dotenv_path=DOTENV_PATH)

# ─── 이제부터 os.getenv()를 호출하면 최상위 .env 값이 로드됨 ───
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENV")  # 혹은 PINECONE_ENVIRONMENT 라고 했다면 그 이름으로

from openai import OpenAI
from pinecone import Pinecone

INDEX_NAME = "dense-index"  # 실제 사용 중인 Pinecone 인덱스 이름

# 1) OpenAI 및 Pinecone 클라이언트 전역 초기화
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)


def embed_query(question: str) -> list:
    """
    LangGraph 노드용 함수: 질문을 임베딩 벡터로 변환합니다.
    """
    resp = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=question
    )
    # 최신 버전에서는 resp.data[0].embedding 형태로 내보냄
    return resp.data[0].embedding


def retrieve_best_namespace(query_vector: list, top_k: int = 5) -> tuple:
    """
    LangGraph 노드용 함수: 벡터(query_vector)를 받아
    Pinecone에서 가장 매칭도가 높은 네임스페이스명과 matches 리스트를 반환합니다.
    """
    stats = index.describe_index_stats()
    available_namespaces = list(stats.namespaces.keys())
    if not available_namespaces:
        raise ValueError("인덱스에 네임스페이스가 없습니다.")

    best_ns = None
    best_score = -1.0
    best_matches = None

    for ns in available_namespaces:
        if stats.namespaces[ns]["vector_count"] == 0:
            continue
        res = index.query(
            vector=query_vector,
            namespace=ns,
            top_k=top_k,
            include_metadata=True
        )
        if not res.matches:
            continue
        top_score = res.matches[0].score
        if top_score > best_score:
            best_score = top_score
            best_ns = ns
            best_matches = res.matches

    if best_ns is None:
        raise ValueError("매칭되는 네임스페이스를 찾을 수 없습니다.")
    return best_ns, best_matches


def build_context(matches: list) -> str:
    """
    LangGraph 노드용 함수: Pinecone 검색 결과(matches)에서
    metadata['text']를 모아 하나의 큰 문자열(Context)로 만듭니다.
    """
    contexts = []
    for m in matches:
        chunk_text = m.metadata.get("text", "")
        if chunk_text:
            contexts.append(chunk_text)
    return "\n---\n".join(contexts)


def generate_answer(question: str, context: str) -> str:
    """
    LangGraph 노드용 함수: 질문(question)과 조립된 context를
    ChatCompletion API에 보내서 답변을 생성합니다.
    """
    prompt = f"""아래 Context를 참고하여 질문에 답변해주세요.

Context:
{context}

질문:
{question}
"""
    resp = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 친절한 어시스턴트입니다."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.0,
        max_tokens=512
    )
    return resp.choices[0].message.content.strip()
