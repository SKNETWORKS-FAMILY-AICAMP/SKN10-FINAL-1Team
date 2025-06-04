import os
from uuid import uuid4

# --------------------------------------------------
# 최신 Pinecone SDK 방식으로 가져오기
# --------------------------------------------------
from pinecone import Pinecone, ServerlessSpec

from dotenv import load_dotenv
load_dotenv()  # .env 파일에서 환경 변수 로드
# --------------------------------------------------
# 최신 OpenAI 클라이언트 (>=1.0.0) 방식으로 가져오기
# --------------------------------------------------
from openai import OpenAI

# --------------------------------------------------
# 1) Pinecone 및 OpenAI 클라이언트 초기화
# --------------------------------------------------
def init_clients():
    # 1-1) OpenAI 클라이언트 생성
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("⚠️ 환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")
    # OpenAI 인스턴스를 만듭니다.
    openai_client = OpenAI(api_key=openai_api_key)
    print("✅ OpenAI 클라이언트 생성 완료")

    # 1-2) Pinecone 인스턴스 생성
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env     = os.getenv("PINECONE_ENVIRONMENT")   # 예: "us-east1-gcp" 또는 "us-west1-gcp" 등
    if not pinecone_api_key or not pinecone_env:
        raise ValueError("⚠️ 환경변수 PINECONE_API_KEY 또는 PINECONE_ENV가 누락되었습니다.")

    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    print("✅ Pinecone 클라이언트 생성 완료")

    # 1-3) 인덱스 존재 여부 확인
    index_name = "dense-index"  # 실제 사용 중인 인덱스 이름으로 교체하세요
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        raise ValueError(f"⚠️ 인덱스 '{index_name}'가 Pinecone에 존재하지 않습니다. 현재 인덱스 목록: {existing_indexes}")

    # 1-4) 해당 인덱스 객체 가져오기
    index = pc.Index(index_name)
    print(f"✅ Pinecone 인덱스 '{index_name}' 연결 완료 (Namespaces: {len(index.describe_index_stats().namespaces)})")

    return openai_client, index


# --------------------------------------------------
# 2) 질문 문장을 임베딩 벡터로 변환
# --------------------------------------------------
def embed_query(openai_client: OpenAI, text: str) -> list:
    """
    최신 OpenAI 클라이언트에서는 resp.data[0].embedding 으로 벡터에 접근해야 합니다.
    """
    resp = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp.data[0].embedding


# --------------------------------------------------
# 3) 여러 네임스페이스 중 “가장 높은 유사도”를 준 네임스페이스와 매칭 결과 반환
# --------------------------------------------------
def retrieve_best_namespace(index, query_vector: list, top_k: int = 5):
    """
    1) index.describe_index_stats()를 통해 모든 네임스페이스 목록을 얻는다.
    2) 각 네임스페이스별로 query_vector를 index.query()로 검색하고,
       matches[0].score 를 비교해서 “최고 유사도”를 찾는다.
    3) 가장 높은 유사도를 준 네임스페이스(best_ns)와 해당 네임스페이스의 전체 매칭 결과(best_matches)를 반환.
    """
    stats = index.describe_index_stats()
    available_namespaces = list(stats.namespaces.keys())
    if not available_namespaces:
        raise ValueError("⚠️ 인덱스에 네임스페이스가 없습니다.")

    best_ns = None
    best_score = -1.0
    best_matches = None

    for ns in available_namespaces:
        count = stats.namespaces[ns]["vector_count"]
        if count == 0:
            # 비어 있는 네임스페이스 건너뛰기
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
        raise ValueError("⚠️ 어떤 네임스페이스에서도 매칭 결과를 찾을 수 없습니다.")
    
    print(f"🔍 선택된 네임스페이스: '{best_ns}' (최고 유사도: {best_score:.4f})")
    return best_ns, best_matches


# --------------------------------------------------
# 4) 검색된 매칭 결과에서 실제 텍스트(메타데이터)를 꺼내 Context 로 결합
# --------------------------------------------------
def build_context_from_matches(matches):
    """
    res.matches 리스트 안의 각 item.metadata 에 들어 있는 텍스트 필드를 추출합니다.
    업로드 시 metadata 키가 "text"였다고 가정했습니다.
    """
    contexts = []
    for m in matches:
        chunk_text = m.metadata.get("text", "")
        if chunk_text:
            contexts.append(chunk_text)

    return "\n---\n".join(contexts)


# --------------------------------------------------
# 5) LLM ChatCompletion 호출하여 답변 생성
# --------------------------------------------------
def generate_answer_with_context(openai_client: OpenAI, question: str, context: str) -> str:
    """
    최신 OpenAI 클라이언트에서는 client.chat.completions.create(...) 형태를 씁니다.
    """
    prompt = f"""아래 Context를 참고해서 질문에 답변해주세요.

Context:
{context}

질문:
{question}
"""
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 친절한 어시스턴트입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1024
    )
    # resp.choices[0].message.content 으로 답변 추출
    return resp.choices[0].message.content.strip()


# --------------------------------------------------
# 6) 메인 루프
# --------------------------------------------------
def main():
    # 6-1) 클라이언트 초기화
    openai_client, index = init_clients()

    while True:
        question = input("🔍 질문을 입력하세요 (‘exit’ 입력 시 종료): ").strip()
        if question.lower() in ("exit", "quit"):
            print("👋 프로그램을 종료합니다.")
            break
        if not question:
            continue

        # 6-2) 질문 → 임베딩 벡터
        try:
            q_vec = embed_query(openai_client, question)
        except Exception as e:
            print(f"⚠️ 임베딩 생성 오류: {e}")
            continue

        # 6-3) 여러 네임스페이스 중 가장 유사도가 높은 네임스페이스와 match 불러오기
        try:
            chosen_ns, matches = retrieve_best_namespace(index, q_vec, top_k=5)
        except ValueError as e:
            print(str(e))
            continue
        except Exception as e:
            print(f"⚠️ 검색 중 오류 발생: {e}")
            continue

        # 6-4) matches에서 context 텍스트 결합
        context = build_context_from_matches(matches)
        if not context:
            print("⚠️ 선택된 네임스페이스에서도 메타데이터 내 텍스트를 읽어오지 못했습니다.")
            continue

        # 6-5) LLM 호출하여 답변 생성
        try:
            answer = generate_answer_with_context(openai_client, question, context)
        except Exception as e:
            print(f"⚠️ LLM 답변 생성 중 오류: {e}")
            continue

        # 6-6) 결과 출력
        print("\n🏆 최종 답변:\n" + answer + "\n")


if __name__ == "__main__":
    main()
