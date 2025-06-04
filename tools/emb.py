import requests
import json
import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
# RunPod API 설정 (어휘 검색을 위한 sparse 임베딩)
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.environ.get("EMB_ENDPOINT_ID", "")
RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

# RunPod 요청 헤더
runpod_headers = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

# OpenAI API 설정 (의미 검색을 위한 dense 임베딩)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Pinecone API 설정
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

def get_sparse_embeddings(texts: List[str], model: str = "dabitbol/bge-m3-sparse-elastic") -> List[Dict[str, List[float]]]:
    """
    텍스트 목록에 대한 sparse 임베딩을 생성합니다. (어휘 검색용)
    
    Args:
        texts (list): 임베딩할 텍스트 목록
        model (str): 사용할 모델 이름
        
    Returns:
        list: sparse 임베딩 목록 (indices와 values 포함)
    """
    # RunPod API가 설정되어 있는지 확인
    if not RUNPOD_API_KEY or not ENDPOINT_ID:
        print("RunPod API 키 또는 엔드포인트 ID가 설정되지 않았습니다. 대체 방법으로 sparse 임베딩을 생성합니다.")
        return _generate_fallback_sparse_embeddings(texts)
        
    payload = {
        "input": {
            "model": model,
            "input": texts,
            "task": "feature-extraction"
        }
    }
    
    try:
        print(f"RunPod API 요청 전송 중: {RUNPOD_URL}")
        response = requests.post(RUNPOD_URL, headers=runpod_headers, json=payload, timeout=30)  # 타임아웃 설정
        response.raise_for_status()
        result = response.json()
        
        # API 응답 구조 로깅 (일부만 출력)
        print(f"RunPod API 응답 수신: {json.dumps(result, ensure_ascii=False)[:500]}...")
        
        # 응답 구조 확인
        if "output" not in result:
            print(f"API 응답에 'output' 키가 없습니다: {result}")
            print("대체 방법으로 sparse 임베딩 생성 시도...")
            return _generate_fallback_sparse_embeddings(texts)
        
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
                
                if not sparse_embeddings:
                    print("유효한 sparse 임베딩을 생성할 수 없습니다. 대체 방법으로 시도합니다.")
                    return _generate_fallback_sparse_embeddings(texts)
                
                print(f"RunPod API로 생성된 sparse 임베딩: {len(sparse_embeddings)}개")
                return sparse_embeddings
            else:
                print(f"data가 리스트가 아닙니다: {type(data)}")
        else:
            print(f"output 구조 확인: {type(result['output'])}")
        
        print("RunPod API로 sparse 임베딩 생성 실패, 대체 방법으로 시도합니다.")
        return _generate_fallback_sparse_embeddings(texts)
    except Exception as e:
        print(f"Sparse 임베딩 생성 중 오류 발생: {e}")
        # 더 자세한 디버그 정보
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"응답 내용: {response.text[:500]}")
        print("대체 방법으로 sparse 임베딩 생성 시도...")
        return _generate_fallback_sparse_embeddings(texts)


def _generate_fallback_sparse_embeddings(texts: List[str]) -> List[Dict[str, List[float]]]:
    """
    RunPod API 사용이 불가능할 때 텍스트 기반 빈도 분석을 통해 대체 sparse 임베딩을 생성합니다.
    
    Args:
        texts (List[str]): 임베딩할 텍스트 목록
        
    Returns:
        List[Dict[str, List[float]]]: Pinecone 형식의 sparse 벡터 목록
    """
    import re
    from collections import Counter
    
    sparse_embeddings = []
    word_to_index = {}  # 전역 단어-인덱스 매핑
    current_index = 0
    
    print("텍스트 기반 빈도 분석으로 대체 sparse 임베딩 생성 중...")
    
    # 모든 텍스트에서 단어 추출 및 인덱스 할당
    all_words = set()
    for text in texts:
        # 소문자 변환 및 토큰화 (단순 공백 기준)
        words = re.findall(r'\w+', text.lower())
        all_words.update(words)
    
    # 단어-인덱스 매핑 생성
    for word in all_words:
        if word not in word_to_index:
            word_to_index[word] = current_index
            current_index += 1
    
    # 각 텍스트에 대한 sparse 벡터 생성
    for text in texts:
        words = re.findall(r'\w+', text.lower())
        word_counts = Counter(words)
        
        # Pinecone 형식의 sparse 벡터 생성
        indices = []
        values = []
        
        for word, count in word_counts.items():
            if word in word_to_index:
                indices.append(word_to_index[word])
                # TF 값 (단어 빈도)
                values.append(float(count) / len(words))  # 정규화된 빈도 값
        
        # 인덱스 기준으로 정렬
        paired = sorted(zip(indices, values), key=lambda x: x[0])
        if paired:
            indices, values = zip(*paired)
            indices, values = list(indices), list(values)
        else:
            indices, values = [], []
        
        sparse_embeddings.append({
            "indices": indices,
            "values": values
        })
    
    print(f"대체 방법으로 생성된 sparse 임베딩: {len(sparse_embeddings)}개 (단어 사전 크기: {len(word_to_index)}개)")
    return sparse_embeddings

def get_dense_embeddings(texts: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
    """
    텍스트 목록에 대한 dense 임베딩을 생성합니다. (의미 검색용)
    
    Args:
        texts (list): 임베딩할 텍스트 목록
        model (str): 사용할 OpenAI 모델 이름
        
    Returns:
        list: dense 임베딩 목록
    """
    try:
        response = openai_client.embeddings.create(
            input=texts,
            model=model
        )
        # 임베딩 추출
        return [record.embedding for record in response.data]
    except Exception as e:
        print(f"Dense 임베딩 생성 중 오류 발생: {e}")
        return []

def setup_pinecone_index(
    index_name: str = "dense-index", 
    dimension: int = 3072, 
    metric: str = "cosine", 
    cloud: str = "aws", 
    region: str = "us-east-1", 
    force_recreate: bool = False
):
    """
    Pinecone Dense 인덱스를 설정합니다.
    
    Args:
        index_name (str): 생성하거나 연결할 인덱스의 이름
        dimension (int): Dense 벡터의 차원 수
        metric (str): 사용할 거리 측정 기준
        cloud (str): 사용할 클라우드 제공자
        region (str): 사용할 클라우드 지역
        force_recreate (bool): True인 경우 기존 인덱스를 삭제하고 새로 생성
    
    Returns:
        Pinecone Index 객체
    """
    spec = ServerlessSpec(cloud=cloud, region=region)
    
    existing_indexes = pinecone_client.list_indexes().names()
    
    if index_name in existing_indexes:
        if force_recreate:
            print(f"기존 인덱스 '{index_name}' 삭제 중...")
            pinecone_client.delete_index(index_name)
            print(f"인덱스 '{index_name}' 삭제 완료. 새 인덱스 생성 중...")
            pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=spec
            )
            print(f"Dense 인덱스 '{index_name}' 생성 완료.")
        else:
            print(f"기존 인덱스 '{index_name}'에 연결합니다.")
    else:
        print(f"Dense 인덱스 '{index_name}' 생성 중...")
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            vector_type="dense",  # 공식 문서에 따른 하이브리드 설정
            spec=spec
        )
        print(f"하이브리드 인덱스 '{index_name}' 생성 완료.")
    
    # 인덱스 객체 반환
    index = pinecone_client.Index(index_name)
    return index

def index_documents(texts: List[str], ids: List[str] = None) -> None:
    """
    문서를 Pinecone 인덱스에 저장합니다.
    
    Args:
        texts (list): 인덱싱할 텍스트 목록
        ids (list): 텍스트 ID 목록 (없으면 자동 생성)
    """
    if ids is None:
        ids = [f"doc{i}" for i in range(len(texts))]
    
    print(f"\n=== {len(texts)}개 문서 인덱싱 시작 ===")
    
    # Dense 임베딩 생성
    dense_embeddings = get_dense_embeddings(texts)
    if not dense_embeddings:
        print("Dense 임베딩 생성 실패")
        return
    
    print(f"Dense 임베딩 생성 완료: {len(dense_embeddings)}개")
    
    try:
        # Pinecone Dense 인덱스 설정 (인덱싱 시에는 force_recreate=True로 설정하여 깨끗한 상태에서 시작)
        dense_index = setup_pinecone_index(force_recreate=True)
        
        # 메타데이터 준비
        metadata = [{"text": text} for text in texts]
        
        vectors_to_upsert = []
        for i, (id_val, dense_vec, meta) in enumerate(zip(ids, dense_embeddings, metadata)):
            vectors_to_upsert.append((id_val, dense_vec, meta))
        
        if vectors_to_upsert:
            dense_index.upsert(vectors=vectors_to_upsert)
        
        print(f"Dense 인덱스 업서트 완료: {len(ids)}개 벡터")
        print(f"=== {len(texts)}개 문서 인덱싱 완료 ===\n")
    except Exception as e:
        print(f"인덱싱 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def dense_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Dense 벡터 검색을 수행합니다.
    
    Args:
        query (str): 검색 쿼리
        top_k (int): 반환할 결과 수
        
    Returns:
        list: 검색 결과 목록
    """
    # Dense 임베딩 생성
    dense_embeddings = get_dense_embeddings([query])
    if not dense_embeddings:
        print("Dense 임베딩 생성 실패, 검색을 진행할 수 없습니다.")
        return []
    
    dense_embedding = dense_embeddings[0]
    
    # Pinecone Dense 인덱스 설정 (검색 시에는 force_recreate=False로 설정하여 기존 인덱스 유지)
    dense_index = setup_pinecone_index(force_recreate=False)
    
    try:
        # Dense 벡터만 사용하는 검색
        results = dense_index.query(
            vector=dense_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results.get('matches', [])
    except Exception as e:
        print(f"검색 중 오류 발생: {e}")
        return []

# 예제 사용법
if __name__ == "__main__":
    # 예제 문서
    documents = [
        "인공지능은 인간의 학습능력과 추론능력, 지각능력을 인공적으로 구현한 컴퓨터 시스템입니다.",
        "머신러닝은 컴퓨터가 데이터로부터 패턴을 학습하여 의사결정을 내리는 기술입니다.",
        "딥러닝은 인공 신경망을 기반으로 한 머신러닝의 한 분야입니다.",
        "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술입니다.",
        "컴퓨터 비전은 컴퓨터가 이미지나 비디오를 이해하는 기술입니다."
    ]
    
    try:
        # 문서 인덱싱 - 모든 문서를 한 번에 인덱싱하고 명시적인 ID 부여
        print("\n=== 문서 인덱싱 ===")
        custom_ids = [f"doc_{i}" for i in range(len(documents))]
        index_documents(documents, custom_ids)
        
        # Dense 검색 테스트
        print("\n=== Dense 검색 테스트 ===")
        query = "인공지능과 사람의 언어 이해"
        results = dense_search(query, top_k=3)
        
        print(f"\n검색 쿼리: {query}")
        if results:
            for i, result in enumerate(results):
                print(f"{i+1}. 점수: {result['score']:.4f}, ID: {result['id']}, 텍스트: {result['metadata']['text']}")
        else:
            print("검색 결과가 없습니다.")
        
        # 추가 쿼리 테스트
        other_queries = [
            "컴퓨터가 데이터에서 패턴을 찾는 기술",
            "신경망 기반 학습 방식",
            "컴퓨터 이미지 처리 기술"
        ]
        
        for test_query in other_queries:
            print(f"\n=== 추가 검색 테스트: '{test_query}' ===")
            test_results = dense_search(test_query, top_k=2)
            
            if test_results:
                for i, result in enumerate(test_results):
                    print(f"{i+1}. 점수: {result['score']:.4f}, ID: {result['id']}, 텍스트: {result['metadata']['text']}")
            else:
                print("검색 결과가 없습니다.")
        
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()