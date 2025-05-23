import os
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# 환경 변수 로드
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class HybridSearch:
    """
    하이브리드 검색 클래스 - Dense와 Sparse 임베딩을 모두 활용한 검색 기능 제공
    
    Attributes:
        index_name (str): Pinecone 인덱스 이름
        runpod_api_key (str): RunPod API 키 (환경 변수: RUNPOD_API_KEY)
        runpod_endpoint_id (str): RunPod 엔드포인트 ID (환경 변수: RUNPOD_ENDPOINT_ID)
        openai_api_key (str): OpenAI API 키 (환경 변수: OPENAI_API_KEY)
        pinecone_api_key (str): Pinecone API 키 (환경 변수: PINECONE_API_KEY)
    """
    
    def __init__(
        self, 
        index_name: str = None,
        runpod_api_key: str = None,
        runpod_endpoint_id: str = None,
        openai_api_key: str = None,
        pinecone_api_key: str = None,
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """
        하이브리드 검색 클래스 초기화
        
        Args:
            index_name (str, optional): Pinecone 인덱스 이름 (기본값: 환경 변수에서 PINECONE_INDEX 또는 'hybrid-search-index')
            runpod_api_key (str, optional): RunPod API 키 (기본값: 환경 변수에서 RUNPOD_API_KEY)
            runpod_endpoint_id (str, optional): RunPod 엔드포인트 ID (기본값: 환경 변수에서 RUNPOD_ENDPOINT_ID)
            openai_api_key (str, optional): OpenAI API 키 (기본값: 환경 변수에서 OPENAI_API_KEY)
            pinecone_api_key (str, optional): Pinecone API 키 (기본값: 환경 변수에서 PINECONE_API_KEY)
            cloud (str): 클라우드 제공업체 (aws, gcp, azure)
            region (str): 클라우드 리전
        """
        # 환경 변수에서 기본값 로드
        self.index_name = index_name or os.getenv("PINECONE_INDEX", "hybrid-search-index")
        self.cloud = cloud or os.getenv("PINECONE_CLOUD", "aws")
        self.region = region or os.getenv("PINECONE_REGION", "us-east-1")
        
        # API 키 설정 (인수 우선, 없으면 환경 변수에서 로드)
        self.runpod_api_key = runpod_api_key or os.getenv("RUNPOD_API_KEY")
        self.runpod_endpoint_id = runpod_endpoint_id or os.getenv("RUNPOD_ENDPOINT_ID")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        
        # API 키 검증
        if not self.runpod_api_key:
            raise ValueError("RunPod API 키가 필요합니다. 환경 변수 RUNPOD_API_KEY를 설정하거나 생성자에 직접 전달하세요.")
        if not self.runpod_endpoint_id:
            raise ValueError("RunPod 엔드포인트 ID가 필요합니다. 환경 변수 RUNPOD_ENDPOINT_ID를 설정하거나 생성자에 직접 전달하세요.")
        if not self.openai_api_key:
            raise ValueError("OpenAI API 키가 필요합니다. 환경 변수 OPENAI_API_KEY를 설정하거나 생성자에 직접 전달하세요.")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API 키가 필요합니다. 환경 변수 PINECONE_API_KEY를 설정하거나 생성자에 직접 전달하세요.")
        
        # RunPod API URL 및 헤더 설정
        self.runpod_url = f"https://api.runpod.ai/v2/{self.runpod_endpoint_id}/runsync"
        self.runpod_headers = {
            "Authorization": f"Bearer {self.runpod_api_key}",
            "Content-Type": "application/json"
        }
        
        # 클라이언트 초기화
        self._initialize_clients()
    
    def _initialize_clients(self):
        """API 클라이언트 초기화"""
        try:
            # OpenAI 클라이언트 초기화
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            
            # Pinecone 클라이언트 초기화
            self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
            print("API 클라이언트 초기화 완료")
        except Exception as e:
            print(f"API 클라이언트 초기화 중 오류 발생: {e}")
            raise
    
    def get_sparse_embeddings(self, texts: List[str], model: str = "dabitbol/bge-m3-sparse-elastic") -> List[Dict[str, List[float]]]:
        """
        텍스트 목록에 대한 sparse 임베딩을 생성합니다. (어휘 검색용)
        
        Args:
            texts (list): 임베딩할 텍스트 목록
            model (str): 사용할 모델 이름
            
        Returns:
            list: sparse 임베딩 목록 (indices와 values 포함)
        """
        payload = {
            "input": {
                "model": model,
                "input": texts,
                "task": "feature-extraction"
            }
        }
        
        try:
            print(f"RunPod API 요청 전송 중: {self.runpod_url}")
            response = requests.post(self.runpod_url, headers=self.runpod_headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # API 응답 구조 로깅 (일부만 출력)
            print(f"RunPod API 응답 수신: {json.dumps(result, ensure_ascii=False)[:500]}...")
            
            # 응답 구조 확인
            if "output" not in result:
                print(f"API 응답에 'output' 키가 없습니다: {result}")
                return []
            
            # output이 딕셔너리이고 data 키가 있는 경우
            if isinstance(result["output"], dict) and "data" in result["output"]:
                data = result["output"]["data"]
                
                # data가 리스트인 경우 (임베딩 목록)
                if isinstance(data, list):
                    sparse_embeddings = []
                    for item in data:
                        # OpenAI 형식의 임베딩을 Pinecone sparse 형식으로 변환
                        if isinstance(item, dict) and "embedding" in item:
                            # dense 임베딩을 sparse로 변환 (상위 값만 유지)
                            embedding = item["embedding"]
                            # 임베딩 값의 절대값 기준 상위 1%만 유지 (sparse 표현 위해)
                            indices = []
                            values = []
                            
                            # 상위 값 선택 (상위 5%만 유지)
                            if len(embedding) > 0:
                                # 임베딩 값과 인덱스 쌍 생성
                                pairs = [(abs(val), i, val) for i, val in enumerate(embedding)]
                                # 절대값 기준 내림차순 정렬
                                pairs.sort(reverse=True)
                                # 상위 5% 선택 (최소 20개는 유지)
                                top_k = max(20, int(len(embedding) * 0.05))
                                top_pairs = pairs[:top_k]
                                
                                # indices와 values 추출
                                for _, idx, val in top_pairs:
                                    indices.append(idx)
                                    values.append(val)
                            
                            sparse_embeddings.append({
                                "indices": indices,
                                "values": values
                            })
                    
                    print(f"생성된 sparse 임베딩: {len(sparse_embeddings)}개")
                    return sparse_embeddings
                else:
                    print(f"data가 리스트가 아닙니다: {data}")
            else:
                print(f"output이 유효한 형식이 아닙니다: {result['output']}")
        except Exception as e:
            print(f"Sparse 임베딩 생성 중 오류 발생: {e}")
        
        return []
    
    def get_dense_embeddings(self, texts: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
        """
        텍스트 목록에 대한 dense 임베딩을 생성합니다. (의미 검색용)
        
        Args:
            texts (list): 임베딩할 텍스트 목록
            model (str): 사용할 OpenAI 모델 이름
            
        Returns:
            list: dense 임베딩 목록
        """
        try:
            response = self.openai_client.embeddings.create(
                input=texts,
                model=model
            )
            
            # 임베딩 추출
            embeddings = [record.embedding for record in response.data]
            print(f"Dense 임베딩 생성 완료: {len(embeddings)}개")
            
            return embeddings
        except Exception as e:
            print(f"Dense 임베딩 생성 중 오류 발생: {e}")
            return []
    
    def setup_pinecone_index(self, force_recreate=False):
        """
        Pinecone 하이브리드 인덱스를 설정합니다.
        단일 인덱스에서 dense와 sparse 벡터를 모두 지원합니다.
        
        Args:
            force_recreate (bool): True인 경우 기존 인덱스를 삭제하고 새로 생성
        
        Returns:
            Pinecone Index 객체
        """
        # 서버리스 스펙 정의 (AWS)
        spec = ServerlessSpec(cloud=self.cloud, region=self.region)
        
        # 하이브리드 인덱스 이름
        hybrid_index_name = self.index_name
        
        # 인덱스 리스트 확인
        indexes = self.pinecone_client.list_indexes().names()
        
        # 인덱스가 이미 존재하는 경우
        if hybrid_index_name in indexes:
            if force_recreate:
                # 강제 재생성 옵션이 켜져 있을 때만 삭제
                print(f"기존 인덱스 '{hybrid_index_name}' 삭제 중...")
                self.pinecone_client.delete_index(hybrid_index_name)
                print(f"인덱스 '{hybrid_index_name}' 삭제 완료")
                
                # 인덱스 생성
                print(f"하이브리드 인덱스 '{hybrid_index_name}' 생성 중...")
                self.pinecone_client.create_index(
                    name=hybrid_index_name,
                    dimension=3072,  # text-embedding-3-large 모델의 차원 수
                    metric="dotproduct",  # sparse 벡터를 지원하기 위해 dotproduct 사용
                    spec=spec
                )
                print(f"하이브리드 인덱스 생성 완료")
            else:
                print(f"기존 인덱스 '{hybrid_index_name}'에 연결 중...")
        else:
            # 인덱스가 존재하지 않는 경우 새로 생성
            print(f"하이브리드 인덱스 '{hybrid_index_name}' 생성 중...")
            self.pinecone_client.create_index(
                name=hybrid_index_name,
                dimension=3072,  # text-embedding-3-large 모델의 차원 수
                metric="dotproduct",  # sparse 벡터를 지원하기 위해 dotproduct 사용
                spec=spec
            )
            print(f"하이브리드 인덱스 생성 완료")
        
        # 인덱스 연결
        hybrid_index = self.pinecone_client.Index(hybrid_index_name)
        return hybrid_index
    
    def index_documents(self, texts: List[str], ids: List[str] = None):
        """
        문서를 Pinecone 인덱스에 저장합니다.
        
        Args:
            texts (list): 인덱싱할 텍스트 목록
            ids (list): 텍스트 ID 목록 (없으면 자동 생성)
        """
        # ID가 없으면 자동 생성
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        # ID와 텍스트 수가 일치하는지 확인
        if len(ids) != len(texts):
            raise ValueError("문서 ID와 텍스트 수가 일치해야 합니다.")
        
        print(f"\n=== {len(texts)}개 문서 인덱싱 시작 ===")
        
        # Dense 임베딩 생성 (의미적 검색)
        dense_embeddings = self.get_dense_embeddings(texts)
        if not dense_embeddings:
            raise ValueError("Dense 임베딩 생성 실패")
        
        # Sparse 임베딩 생성 (어휘적 검색)
        sparse_embeddings = self.get_sparse_embeddings(texts)
        has_sparse = len(sparse_embeddings) > 0
        
        if has_sparse:
            for sparse_emb in sparse_embeddings:
                print(f"Sparse 임베딩 생성 성공: indices={len(sparse_emb['indices'])}, values={len(sparse_emb['values'])}")
        else:
            print("Sparse 임베딩 생성 실패, Dense 임베딩만 사용합니다.")
        
        try:
            # Pinecone 하이브리드 인덱스 설정 (인덱싱 시에는 force_recreate=True로 설정하여 깨끗한 상태에서 시작)
            hybrid_index = self.setup_pinecone_index(force_recreate=True)
            
            # 메타데이터 준비
            metadata = [{"text": text} for text in texts]
            
            # 벡터 업서트 준비
            vectors_to_upsert = []
            
            for i, (doc_id, dense_emb) in enumerate(zip(ids, dense_embeddings)):
                # 기본 벡터 정보 (ID, 임베딩, 메타데이터)
                vector_data = {
                    "id": doc_id,
                    "values": dense_emb,
                    "metadata": metadata[i]
                }
                
                # Sparse 임베딩이 있으면 추가
                if has_sparse and i < len(sparse_embeddings):
                    vector_data["sparse_values"] = sparse_embeddings[i]
                
                vectors_to_upsert.append(vector_data)
            
            # 벡터 업서트 (최대 100개씩 배치로 처리)
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i+batch_size]
                hybrid_index.upsert(vectors=batch)
            
            print(f"{len(texts)}개 문서 인덱싱 완료")
            
        except Exception as e:
            print(f"문서 인덱싱 중 오류 발생: {e}")
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5):
        """
        하이브리드 검색을 수행합니다.
        
        Args:
            query (str): 검색 쿼리
            top_k (int): 반환할 결과 수
            alpha (float): dense와 sparse 결과의 가중치 (0.0: sparse만, 1.0: dense만)
            
        Returns:
            list: 검색 결과 목록
        """
        # Dense 임베딩 생성
        dense_embeddings = self.get_dense_embeddings([query])
        if not dense_embeddings:
            print("Dense 임베딩 생성 실패")
            return []
        
        dense_embedding = dense_embeddings[0]
        
        # Sparse 임베딩 생성
        sparse_embeddings = self.get_sparse_embeddings([query])
        has_sparse = len(sparse_embeddings) > 0
        
        if has_sparse:
            sparse_embedding = sparse_embeddings[0]
            print(f"Sparse 임베딩 생성 성공: indices={len(sparse_embedding['indices'])}, values={len(sparse_embedding['values'])}")
        else:
            print("Sparse 임베딩 생성 실패, Dense 검색만 진행합니다.")
        
        # Pinecone 하이브리드 인덱스 설정 (검색 시에는 force_recreate=False로 설정하여 기존 인덱스 유지)
        hybrid_index = self.setup_pinecone_index(force_recreate=False)
        
        try:
            # 하이브리드 검색 수행 (단일 쿼리로 Dense와 Sparse 벡터 모두 사용)
            if has_sparse:
                # Dense와 Sparse 벡터 모두 사용하는 하이브리드 검색
                results = hybrid_index.query(
                    vector=dense_embedding,
                    sparse_vector=sparse_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    alpha=alpha  # alpha 매개변수로 가중치 조정 (Pinecone API에서 지원)
                )
            else:
                # Dense 벡터만 사용하는 검색
                results = hybrid_index.query(
                    vector=dense_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
            
            return results.get('matches', [])
        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            return []


# 사용 예시
def example_usage():
    """
    HybridSearch 클래스 사용 예시
    """
    # 설정 정보
    runpod_api_key = os.getenv("RUNPOD_API_KEY")
    runpod_endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    # HybridSearch 인스턴스 생성
    search = HybridSearch(
        index_name="hybrid-search-index",
        runpod_api_key=runpod_api_key,
        runpod_endpoint_id=runpod_endpoint_id,
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key
    )
    
    # 예제 문서
    documents = [
        "인공지능은 인간의 학습능력과 추론능력, 지각능력을 인공적으로 구현한 컴퓨터 시스템입니다.",
        "머신러닝은 컴퓨터가 데이터로부터 패턴을 학습하여 의사결정을 내리는 기술입니다.",
        "딥러닝은 인공 신경망을 기반으로 한 머신러닝의 한 분야입니다.",
        "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술입니다.",
        "컴퓨터 비전은 컴퓨터가 이미지나 비디오를 이해하는 기술입니다."
    ]
    
    # 문서 인덱싱
    custom_ids = [f"doc_{i}" for i in range(len(documents))]
    search.index_documents(documents, custom_ids)
    
    # 검색 수행
    query = "인공지능과 사람의 언어 이해"
    results = search.hybrid_search(query, top_k=3, alpha=0.7)
    
    # 결과 출력
    print(f"\n검색 쿼리: {query}")
    if results:
        for i, result in enumerate(results):
            print(f"{i+1}. 점수: {result['score']:.4f}, ID: {result['id']}, 텍스트: {result['metadata']['text']}")
    else:
        print("검색 결과가 없습니다.")


if __name__ == "__main__":
    example_usage()
