import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

load_dotenv()
#안녕
def init_clients():
    """
    OpenAI 및 Pinecone 클라이언트를 초기화하고
    Pinecone 연결 상태를 출력합니다.
    """
    load_dotenv()
    # OpenAI 클라이언트
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Pinecone 클라이언트
    try:
        pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # 연결 확인
        idxs = pc.list_indexes().names()
        print(f"✅ Pinecone 연결 성공: {len(idxs)}개의 인덱스 확인됨")
    except Exception as e:
        print(f"❌ Pinecone 연결 실패: {e}")
        raise
    index = pc.Index("dense-index")  # or your index name
    return openai_client, index


def embed_query(openai_client, text: str):
    """
    OpenAI text-embedding-3-large 모델을 사용해 텍스트를 임베딩 벡터로 변환
    """
    resp = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp.data[0].embedding


def retrieve_docs(index, query_vector: list, top_k: int = 5):
    """
    Pinecone에서 쿼리 벡터 기반으로 상위 k개 문서를 검색
    검색된 문서 수와 각 문서의 유사도 점수를 함께 반환
    """
    res = index.query(
        vector=query_vector,
        namespace="product_document",  # your namespace
        top_k=top_k,
        include_metadata=True
    )
    matches = res.matches
    print(f"🔍 유사 문서 검색 개수: {len(matches)}개")
    for idx, m in enumerate(matches, 1):
        print(f"  문서 {idx} 유사도 점수: {m.score:.4f}")
    return matches


def generate_answer(openai_client, query: str, contexts: list):
    # system prompt 작성
    system_prompt = "You are a knowledgeable assistant. Use the provided context to answer the question."
    # context 합치기
    context_block = "\n\n---\n\n".join(contexts)
    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # 또는 사용 가능한 LLM
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=512
    )
    return resp.choices[0].message.content.strip()


def main():
    openai_client, index = init_clients()
    query = input("🔍 질문을 입력하세요: ")
    # 텍스트 임베딩
    q_vec = embed_query(openai_client, query)

    # 문서 검색 및 점수 출력
    matches = retrieve_docs(index, q_vec, top_k=5)
    if not matches:
        print("⚠️ 인덱스에서 문서를 찾지 못했습니다.")
        return

    # 검색된 메타데이터 중 텍스트만 추출
    contexts = [m.metadata.get("chunk_text") or m.metadata.get("text") or "" for m in matches]

    # 최고 유사도 점수 출력
    best_score = max(m.score for m in matches)
    print(f"🏆 최고 유사도 점수: {best_score:.4f}\n")

    # 답변 생성
    answer = generate_answer(openai_client, query, contexts)
    print("📝 생성된 답변:\n")
    print(answer)

if __name__ == "__main__":
    main()
