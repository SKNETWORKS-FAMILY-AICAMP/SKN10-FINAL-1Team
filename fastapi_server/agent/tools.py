# f:\dev\SKN10-FINAL-1Team\swarm_agent\src\agent\tools.py
import os
import sys
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import Tool
from openai import OpenAI as OpenAIClient # For embeddings
from pinecone import Pinecone as PineconeClient # For vector search

# --- Document Search Tools (Pinecone) ---
_openai_client_cache = None
_pinecone_client_cache = None
_pinecone_index_cache = None

def get_openai_client():
    global _openai_client_cache
    if _openai_client_cache is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        _openai_client_cache = OpenAIClient(api_key=api_key)
    return _openai_client_cache

def get_pinecone_index():
    global _pinecone_client_cache, _pinecone_index_cache
    if _pinecone_index_cache is None:
        api_key = os.getenv("PINECONE_API_KEY")
        env = os.getenv("PINECONE_ENV")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not all([api_key, env, index_name]):
            raise ValueError("Pinecone API key, environment, or index name not set.")
        if _pinecone_client_cache is None:
            _pinecone_client_cache = PineconeClient(api_key=api_key, environment=env)
        _pinecone_index_cache = _pinecone_client_cache.Index(index_name)
    return _pinecone_index_cache

class PineconeSearchArgs(BaseModel):
    query: str = Field(..., description="The search query string.")
    top_k: int = Field(default=3, description="Number of top results to return.")

def search_pinecone_documents(query: str, top_k: int = 3) -> str:
    try:
        openai_client = get_openai_client()
        pinecone_index = get_pinecone_index()
        
        embedding_response = openai_client.embeddings.create(input=[query], model="text-embedding-ada-002")
        query_embedding = embedding_response.data[0].embedding
        
        results = pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        
        output = []
        if results.matches:
            for match in results.matches:
                output.append(f"Document ID: {match.id}\nScore: {match.score:.4f}\nContent Snippet: {match.metadata.get('text_chunk', 'N/A')[:200]}...\nSource: {match.metadata.get('source', 'N/A')}")
            return "\n---\n".join(output)
        else:
            return "No relevant documents found."
    except Exception as e:
        print(f"Error during Pinecone search: {e}", file=sys.stderr)
        return f"Error searching documents: {e}"

search_pinecone_documents_tool = Tool(
    name="PineconeDocumentSearch",
    func=search_pinecone_documents,
    description="Searches for relevant documents in Pinecone based on a query. Returns document IDs, scores, snippets, and sources.",
    args_schema=PineconeSearchArgs
)

class DocumentContentArgs(BaseModel):
    document_id: str = Field(..., description="The ID of the document to retrieve.")

def get_document_content(document_id: str) -> str:
    try:
        pinecone_index = get_pinecone_index()
        fetched_vector = pinecone_index.fetch(ids=[document_id])
        if fetched_vector.vectors and document_id in fetched_vector.vectors:
            metadata = fetched_vector.vectors[document_id].metadata
            if metadata and 'text_chunk' in metadata:
                return f"Document ID: {document_id}\nSource: {metadata.get('source', 'N/A')}\n\nContent:\n{metadata['text_chunk']}"
            else:
                return f"Document ID {document_id} found, but it has no 'text_chunk' in metadata."
        else:
            return f"Document with ID '{document_id}' not found."
    except Exception as e:
        print(f"Error fetching document content: {e}", file=sys.stderr)
        return f"Error fetching document content for ID '{document_id}': {e}"

get_document_content_tool = Tool(
    name="PineconeDocumentContentRetriever",
    func=get_document_content,
    description="Retrieves the full text content of a specific document from Pinecone using its ID.",
    args_schema=DocumentContentArgs
)

__all__ = [
    "search_pinecone_documents_tool",
    "get_document_content_tool"
]
