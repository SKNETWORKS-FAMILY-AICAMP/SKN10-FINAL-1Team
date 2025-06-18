import os
import sys # Added for stderr printing and exit
print("--- pinecone_tools.py loading ---", file=sys.stderr) # Added for debugging
from typing import Type, List, Dict, Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone as PineconeClient # Renamed to avoid conflict with pinecone module

# Load environment variables from .env file
load_dotenv()

# --- Client Initialization ---
def init_clients():
    # OpenAI Client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("⚠️ 환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")
    openai_client = OpenAI(api_key=openai_api_key)

    # Pinecone Client
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV")
    if not pinecone_api_key or not pinecone_env:
        raise ValueError("⚠️ 환경변수 PINECONE_API_KEY 또는 PINECONE_ENV가 누락되었습니다.")
    
    pc = PineconeClient(api_key=pinecone_api_key)
    index_name = os.getenv("PINECONE_INDEX_NAME", "dense-index") # Default to "dense-index" if not set

    existing_indexes = [idx_spec['name'] for idx_spec in pc.list_indexes()]
    if index_name not in existing_indexes:
        raise ValueError(f"⚠️ 인덱스 '{index_name}'가 Pinecone에 존재하지 않습니다. 현재 인덱스 목록: {existing_indexes}")

    pinecone_index = pc.Index(index_name)
    # print(f"✅ Pinecone 인덱스 '{index_name}' 연결 완료 (Namespaces: {len(pinecone_index.describe_index_stats().namespaces)}) ")
    return openai_client, pinecone_index

# Initialize clients globally or handle as needed
# For simplicity in tools, we'll initialize them here. 
# In a larger app, consider passing them or using a shared context.
try:
    OPENAI_CLIENT, PINECONE_INDEX = init_clients()
except Exception as e: # Changed to catch any exception
    print(f"FATAL: Client initialization error in pinecone_tools.py: {e}", file=sys.stderr)
    # Set to None so tools can check and fail gracefully or skip operations
    OPENAI_CLIENT, PINECONE_INDEX = None, None
    sys.exit(1) # Exit to make sure the error is noticed

# --- Embedding Function ---
def embed_query(openai_client: OpenAI, text: str) -> List[float]:
    if not openai_client:
        raise ValueError("OpenAI client not initialized.")
    resp = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp.data[0].embedding

# --- Context Building Function ---
def build_context_from_matches(matches: List[Dict[str, Any]]) -> str:
    contexts = []
    if not matches:
        return ""
    for m in matches:
        metadata = m.get("metadata", {})
        chunk_text = metadata.get("text", "")
        filename = metadata.get("original_filename", "Unknown") # Provide a default
        
        if chunk_text:
            # Format the context to include both filename and content
            context_entry = f"Source File: {filename}\nContent:\n{chunk_text}"
            contexts.append(context_entry)
            
    # Join entries with a clear separator
    return "\n\n---\n\n".join(contexts)


class SearchInput(BaseModel):
    query: str = Field(..., description="검색어")
    top_k: int = Field(default=3, description="가져올 문서 수")

class _BaseNamespaceTool(BaseTool):
    args_schema: Type[BaseModel] = SearchInput
    namespace: str = "" # To be defined by subclasses

    def _run(self, query: str, top_k: int = 3) -> str:
        if not OPENAI_CLIENT or not PINECONE_INDEX:
            return "⚠️ 클라이언트가 초기화되지 않아 검색을 수행할 수 없습니다."

        if not self.namespace:
            return "⚠️ 검색할 네임스페이스가 지정되지 않았습니다."

        try:
            # print(f"\n📄 RAG 실행: Namespace='{self.namespace}', Query='{query}', Top_k={top_k}")
            query_vector = embed_query(OPENAI_CLIENT, query)
            # print(f"   - 질문 임베딩 완료 (벡터 크기: {len(query_vector)})")

            index_stats = PINECONE_INDEX.describe_index_stats()
            if self.namespace not in index_stats.namespaces or \
               index_stats.namespaces[self.namespace].vector_count == 0:
                message = f"'{self.namespace}' 네임스페이스를 Pinecone에서 찾을 수 없거나, 해당 네임스페이스에 데이터가 없습니다."
                # print(f"   - 경고: {message}")
                return message

            res = PINECONE_INDEX.query(
                vector=query_vector,
                namespace=self.namespace,
                top_k=top_k,
                include_metadata=True
            )
            matches = res.get("matches", [])
            # print(f"   - Pinecone 검색 완료: {len(matches)}개 결과 수신")

            if not matches:
                message = f"'{self.namespace}' 네임스페이스에서 '{query}' 질문과 관련된 정보를 찾지 못했습니다."
                # print(f"   - 정보 없음: {message}")
                return message
            
            context = build_context_from_matches(matches)
            if not context:
                message = "검색된 정보에서 답변을 생성할 컨텍스트를 추출하지 못했습니다."
                # print(f"   - 컨텍스트 구축 실패: {message}")
                return message
            
            # print(f"   - 컨텍스트 구축 완료 (길이: {len(context)})")
            return context
        
        except ValueError as ve:
            # print(f"   - 값 오류: {ve}")
            return f"검색 중 오류 발생: {ve}"
        except Exception as e:
            # print(f"   - 일반 오류: {e}")
            return f"알 수 없는 오류로 검색에 실패했습니다: {e}"

# --- Specific Search Tools ---
class InternalPolicySearchTool(_BaseNamespaceTool):
    name: str = "Internal Policy Search"
    description: str = "사내 정책 및 HR 관련 문서를 검색합니다. (예: 휴가 규정, 복지, 행동 강령)"
    namespace: str = "internal_policy"

class TechDocSearchTool(_BaseNamespaceTool):
    name: str = "Technical Document Search"
    description: str = "기술 문서, 개발 가이드, API 명세 등을 검색합니다."
    namespace: str = "technical_document" # Ensure this matches your Pinecone namespace

class ProductDocSearchTool(_BaseNamespaceTool):
    name: str = "Product Document Search"
    description: str = "제품 설명서, 기능 소개, 사용자 매뉴얼 등을 검색합니다."
    namespace: str = "product_document" # Ensure this matches your Pinecone namespace

class ProceedingsSearchTool(_BaseNamespaceTool):
    name: str = "Proceedings Search"
    description: str = "회의록, 결정 사항, 업무 지시 등을 검색합니다."
    namespace: str = "proceedings" # Ensure this matches your Pinecone namespace

# Example usage (for testing purposes, can be removed)
if __name__ == '__main__':
    # Ensure you have a .env file with OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
    # And that your Pinecone index/namespaces are populated.
    
    # Test one of the tools
    if OPENAI_CLIENT and PINECONE_INDEX:
        # test_tool = ProductDocSearchTool()
        # test_query = "최신 스마트폰 기능은 뭐가 있나요?"
        
        # test_tool = InternalPolicySearchTool()
        # test_query = "연차 사용 규정이 어떻게 되나요?"

        test_tool = TechDocSearchTool()
        test_query = "파이썬 비동기 프로그래밍 가이드라인 알려줘"
        
        print(f"Testing tool: {test_tool.name} for namespace: {test_tool.namespace}")
        print(f"Query: {test_query}")
        results = test_tool._run(query=test_query, top_k=2)
        print("\n--- Results ---")
        print(results)
    else:
        print("Clients not initialized, skipping test.")
