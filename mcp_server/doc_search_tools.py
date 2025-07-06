import os
from dotenv import load_dotenv
load_dotenv()
import sys
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from pinecone import Pinecone as PineconeClient

# --- Pinecone/OpenAI Client Initialization ---
def init_clients():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Environment variable OPENAI_API_KEY is not set.")
    openai_client = OpenAI(api_key=openai_api_key)

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV")
    if not pinecone_api_key or not pinecone_env:
        raise ValueError("Environment variables PINECONE_API_KEY or PINECONE_ENV are missing.")
    
    pc = PineconeClient(api_key=pinecone_api_key)
    # host 기반으로 인덱스 생성
    index_host = os.getenv("PINECONE_INDEX_HOST")
    if not index_host:
        raise ValueError("Environment variable PINECONE_INDEX_HOST is not set. Please set your Pinecone index host URL.")
    pinecone_index = pc.Index(host=index_host)
    print(f"Successfully connected to Pinecone index host '{index_host}'.", file=sys.stderr)
    return openai_client, pinecone_index

OPENAI_CLIENT, PINECONE_INDEX = None, None
try:
    OPENAI_CLIENT, PINECONE_INDEX = init_clients()
except ValueError as e:
    print(f"Error initializing clients: {e}", file=sys.stderr)

# --- Embedding and Context Building Functions ---
def embed_query(text: str) -> List[float]:
    if not OPENAI_CLIENT:
        raise ValueError("OpenAI client not initialized for embedding.")
    resp = OPENAI_CLIENT.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp.data[0].embedding

def build_context_from_matches(matches: List[Dict[str, Any]]) -> str:
    contexts = []
    if not matches:
        return ""
    for m in matches:
        metadata = m.get("metadata", {})
        chunk_text = metadata.get("text", "")
        filename = metadata.get("original_filename", "Unknown")
        
        if chunk_text:
            context_entry = f"Source File: {filename}\nContent:\n{chunk_text}"
            contexts.append(context_entry)
    return "\n\n---\n\n".join(contexts)

# --- RAG Fusion Functions ---
def create_similar_questions(query: str) -> List[str]:
    """원본 질문으로부터 유사한 질문 4개를 생성합니다."""
    if not OPENAI_CLIENT:
        print("Warning: OpenAI client not initialized, returning original query only.", file=sys.stderr)
        return [query]
    
    create_similar_questions_prompt = """
    당신은 질문 확장 전문가입니다. 주어진 질문을 바탕으로 유사하지만 다른 관점에서 접근하는 질문 4개를 생성해주세요.
    
    원본 질문: {user_input}
    
    다음 조건을 만족하는 4개의 유사 질문을 생성해주세요:
    1. 원본 질문과 같은 주제를 다룸
    2. 다른 키워드나 표현을 사용
    3. 약간 다른 관점에서 접근
    4. 각 질문은 한 줄로 작성
    
    4개의 질문만 생성하고, 번호나 다른 설명은 포함하지 마세요.
    """
    
    try:
        formatted_prompt = create_similar_questions_prompt.format(user_input=query)
        resp = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": formatted_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        similar_questions = resp.choices[0].message.content.strip().split("\n")
        # 빈 줄 제거 및 정리
        similar_questions = [q.strip() for q in similar_questions if q.strip()]
        
        # 최대 4개까지만 사용
        return similar_questions[:4]
        
    except Exception as e:
        print(f"Error generating similar questions: {e}", file=sys.stderr)
        return [query]

def calculate_rrf_scores(question_to_doc: Dict[int, List[str]]) -> Dict[str, float]:
    """RRF(Reciprocal Rank Fusion) 점수를 계산합니다."""
    document_score = {}
    
    for question_idx, doc_ids in question_to_doc.items():
        for rank, doc_id in enumerate(doc_ids):
            if doc_id not in document_score:
                document_score[doc_id] = float(1 / (60 + 1 + rank))
            else:
                document_score[doc_id] += float(1 / (60 + 1 + rank))
    
    return document_score

def get_document_ids_from_matches(matches: List[Dict[str, Any]]) -> List[str]:
    """검색 결과에서 문서 ID를 추출합니다."""
    return [m.get("id", "") for m in matches if m.get("id")]

def combine_text_from_fetch(fetch_res: Dict[str, Any], final_doc_ids: List[str]) -> str:
    """Fetch 결과에서 텍스트를 결합합니다."""
    contexts = []
    
    for doc_id in final_doc_ids:
        vec_info = fetch_res.get("vectors", {}).get(doc_id, {})
        metadata = vec_info.get("metadata", {})
        text = metadata.get("text", "")
        filename = metadata.get("original_filename", "Unknown")
        
        if text:
            context_entry = f"Source File: {filename}\nContent:\n{text}"
            contexts.append(context_entry)
    
    return "\n\n---\n\n".join(contexts)

# --- Enhanced Pinecone Search with RAG Fusion ---
def _run_rag_fusion_search(query: str, namespace: str, top_k: int = 3) -> str:
    """RAG Fusion 방식으로 Pinecone 검색을 수행합니다."""
    if not OPENAI_CLIENT or not PINECONE_INDEX:
        return "Error: OpenAI or Pinecone client not initialized."
    if not namespace:
        return "Error: Namespace not specified for Pinecone search."
    
    try:
        # 1. 네임스페이스 존재 확인
        index_stats = PINECONE_INDEX.describe_index_stats()
        if namespace not in index_stats.namespaces or \
           index_stats.namespaces[namespace].vector_count == 0:
            return f"Namespace '{namespace}' not found in Pinecone or is empty."

        # 2. 유사 질문 생성 (원본 질문 포함)
        similar_questions = create_similar_questions(query)
        similar_questions.append(query)  # 원본 질문도 포함
        
        print(f"Generated {len(similar_questions)} questions for RAG Fusion", file=sys.stderr)
        
        # 3. 각 질문에 대해 검색 수행
        question_to_doc = {}
        
        for i, question in enumerate(similar_questions):
            if not question.strip():
                continue
                
            query_vector = embed_query(question.strip())
            
            res = PINECONE_INDEX.query(
                vector=query_vector,
                namespace=namespace,
                top_k=4,  # 각 질문당 4개 문서 검색
                include_metadata=True
            )
            
            matches = res.get("matches", [])
            if matches:
                question_to_doc[i] = get_document_ids_from_matches(matches)
                print(f"Question {i+1}: Found {len(matches)} matches", file=sys.stderr)
        
        if not question_to_doc:
            return f"No relevant information found in namespace '{namespace}' for query: '{query}'."
        
        # 4. RRF 점수 계산
        document_scores = calculate_rrf_scores(question_to_doc)
        
        # 5. 상위 top_k개 문서 선택
        top_documents = sorted(
            document_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        final_doc_ids = [doc_id for doc_id, _ in top_documents]
        
        if not final_doc_ids:
            return f"No relevant documents found after RRF scoring for query: '{query}'."
        
        # 6. 최종 문서들 fetch
        fetch_res = PINECONE_INDEX.fetch(
            ids=final_doc_ids,
            namespace=namespace
        )
        
        # 7. 컨텍스트 생성
        context = combine_text_from_fetch(fetch_res, final_doc_ids)
        
        if not context:
            return "Could not extract context from search results."
        
        print(f"RAG Fusion completed: {len(final_doc_ids)} documents, context length: {len(context)}", file=sys.stderr)
        return context
        
    except Exception as e:
        print(f"Error during RAG Fusion search in namespace '{namespace}': {e}", file=sys.stderr)
        return f"Error during RAG Fusion search in namespace '{namespace}': {e}"

# --- Legacy Simple Search (for backward compatibility) ---
def _run_pinecone_search(query: str, namespace: str, top_k: int = 3) -> str:
    """기존 단순 검색 방식 (하위 호환성을 위해 유지)"""
    if not OPENAI_CLIENT or not PINECONE_INDEX:
        return "Error: OpenAI or Pinecone client not initialized."
    if not namespace:
        return "Error: Namespace not specified for Pinecone search."
    try:
        query_vector = embed_query(query)
        index_stats = PINECONE_INDEX.describe_index_stats()
        if namespace not in index_stats.namespaces or \
           index_stats.namespaces[namespace].vector_count == 0:
            return f"Namespace '{namespace}' not found in Pinecone or is empty."

        res = PINECONE_INDEX.query(
            vector=query_vector,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True
        )
        matches = res.get("matches", [])
        if not matches:
            return f"No relevant information found in namespace '{namespace}' for query: '{query}'."
        
        context = build_context_from_matches(matches)
        return context if context else "Could not extract context from search results."
    except Exception as e:
        return f"Error during Pinecone search in namespace '{namespace}': {e}"

# FastMCP 호환 함수들 (RAG Fusion 적용)
def internal_policy_search(query: str, top_k: int = 3) -> str:
    """Searches internal company policies and HR documents using RAG Fusion (e.g., vacation policy, benefits, code of conduct)."""
    return _run_rag_fusion_search(query, namespace="internal_policy", top_k=top_k)

def tech_doc_search(query: str, top_k: int = 3) -> str:
    """Searches technical documents, development guides, and API specifications using RAG Fusion."""
    return _run_rag_fusion_search(query, namespace="technical_document", top_k=top_k)

def product_doc_search(query: str, top_k: int = 3) -> str:
    """Searches product manuals, feature descriptions, and user guides using RAG Fusion."""
    return _run_rag_fusion_search(query, namespace="product_document", top_k=top_k)

def proceedings_search(query: str, top_k: int = 3) -> str:
    """Searches meeting minutes, decisions, and work instructions using RAG Fusion."""
    return _run_rag_fusion_search(query, namespace="proceedings", top_k=top_k)

def proceedings_text_with_filename(filename: str, top_k: int = 3) -> str:
    """파일명으로 Pinecone proceedings namespace에서 회의록 검색 (벡터+파이썬 필터, 3072차원)"""
    if not PINECONE_INDEX:
        return "Error: Pinecone client not initialized."
    namespace = "proceedings"
    try:
        # 전체 벡터 중 top_k*10개를 받아서 filename으로 필터링 (3072차원 zero vector)
        res = PINECONE_INDEX.query(
            namespace=namespace,
            vector=[0.0]*3072,  # 3072차원 zero vector
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        matches = res.get("matches", [])
        filtered = [m for m in matches if m.get("metadata", {}).get("original_filename") == filename]
        if not filtered:
            return f"No relevant information found in namespace '{namespace}' for filename: '{filename}'."
        results = []
        for m in filtered[:top_k]:
            meta = m.get("metadata", {})
            fname = meta.get("original_filename", "Unknown")
            text = meta.get("text", "")
            results.append(f"Source File: {fname}\nContent:\n{text}")
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Error during filename search in namespace '{namespace}': {e}"

# 기존 단순 검색 함수들 (하위 호환성을 위해 유지)
def internal_policy_search_simple(query: str, top_k: int = 3) -> str:
    """Searches internal company policies using simple search (legacy)."""
    return _run_pinecone_search(query, namespace="internal_policy", top_k=top_k)

def tech_doc_search_simple(query: str, top_k: int = 3) -> str:
    """Searches technical documents using simple search (legacy)."""
    return _run_pinecone_search(query, namespace="technical_document", top_k=top_k)

def product_doc_search_simple(query: str, top_k: int = 3) -> str:
    """Searches product documents using simple search (legacy)."""
    return _run_pinecone_search(query, namespace="product_document", top_k=top_k)

def proceedings_search_simple(query: str, top_k: int = 3) -> str:
    """Searches proceedings using simple search (legacy)."""
    return _run_pinecone_search(query, namespace="proceedings", top_k=top_k) 