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

# --- Pinecone Search Tool Functions ---
def _run_pinecone_search(query: str, namespace: str, top_k: int = 3) -> str:
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

# FastMCP 호환 함수들
def internal_policy_search(query: str, top_k: int = 3) -> str:
    """Searches internal company policies and HR documents (e.g., vacation policy, benefits, code of conduct)."""
    return _run_pinecone_search(query, namespace="internal_policy", top_k=top_k)

def tech_doc_search(query: str, top_k: int = 3) -> str:
    """Searches technical documents, development guides, and API specifications."""
    return _run_pinecone_search(query, namespace="technical_document", top_k=top_k)

def product_doc_search(query: str, top_k: int = 3) -> str:
    """Searches product manuals, feature descriptions, and user guides."""
    return _run_pinecone_search(query, namespace="product_document", top_k=top_k)

def proceedings_search(query: str, top_k: int = 3) -> str:
    """Searches meeting minutes, decisions, and work instructions."""
    return _run_pinecone_search(query, namespace="proceedings", top_k=top_k)

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