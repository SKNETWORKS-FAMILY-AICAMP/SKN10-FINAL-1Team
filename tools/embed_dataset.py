import os
import sys
import re
import hashlib
from typing import List, Dict, Tuple
import PyPDF2
from bs4 import BeautifulSoup, Comment
import glob
from tqdm import tqdm
import unicodedata

# Import functions from emb.py
from emb import setup_pinecone_index, get_dense_embeddings, get_sparse_embeddings, index_documents

# 올바른 Pinecone 클라이언트 버전을 사용하고 있는지 확인
import pkg_resources
installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
print(f"Pinecone 클라이언트 버전: {installed_packages.get('pinecone', '설치되지 않음')}")


# Define paths
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
NAMESPACES = ['Internal Policy', 'Product document', 'Technical document']

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    PDF 파일에서 텍스트를 추출하고 전처리합니다.
    
    Args:
        pdf_path (str): PDF 파일 경로
        
    Returns:
        str: 추출 및 전처리된 텍스트
    """
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + " "
        
        # 텍스트 전처리
        # 1. 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 2. 불필요한 줄바꿈 제거
        text = re.sub(r'\n+', ' ', text)
        
        # 3. 페이지 번호와 같은 패턴 제거 (예: [Page 1], Page 1 등)
        text = re.sub(r'\[?Page\s+\d+\]?', '', text)
        
        # 4. 제어 문자 제거
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # 5. 불필요한 특수 기호 정리 (예: -, *, =로 구성된 구분선)
        text = re.sub(r'[\-\*\=]{3,}', '', text)
        
        return text.strip()
    except Exception as e:
        print(f"PDF 처리 중 오류 발생 ({pdf_path}): {e}")
        return ""

def extract_text_from_html(html_path: str) -> str:
    """
    HTML 파일에서 텍스트를 추출하고 전처리합니다.
    
    Args:
        html_path (str): HTML 파일 경로
        
    Returns:
        str: 추출 및 전처리된 텍스트
    """
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 0. Django 템플릿 태그 제거 (BeautifulSoup 파싱 전)
        content = re.sub(r'{%.*?%}', '', content, flags=re.DOTALL)  # {% ... %} 제거
        content = re.sub(r'{{.*?}}', '', content, flags=re.DOTALL) # {{ ... }} 제거
        content = re.sub(r'{#.*?#}', '', content, flags=re.DOTALL) # {# ... #} 제거 (Django 주석)

        soup = BeautifulSoup(content, 'html.parser')
        
        # 1. 불필요한 태그 및 내용 제거
        for tag in soup(['script', 'style', 'meta', 'link', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()
        
        # 2. 주석 제거
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # 3. 개행 처리를 위해 특정 태그를 개행으로 변환
        for tag in soup.find_all(['br', 'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            tag.append("\n")
        
        # 4. 텍스트 추출
        text = soup.get_text(separator=' ', strip=True)
        
        # 5. 텍스트 전처리
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 연속된 줄바꿈 제거
        text = re.sub(r'\n+', '\n', text)
        
        # HTML 문자 엔티티 정리 (예: &nbsp;, &amp; 등)
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        
        # 제어 문자 제거
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text.strip()
    except Exception as e:
        print(f"HTML 처리 중 오류 발생 ({html_path}): {e}")
        return ""

def get_files_by_namespace() -> Dict[str, List[str]]:
    """
    네임스페이스(폴더)별로 파일 경로를 수집합니다.
    
    Returns:
        Dict[str, List[str]]: 네임스페이스별 파일 경로 목록
    """
    files_by_namespace = {}
    
    for namespace in NAMESPACES:
        namespace_path = os.path.join(DATASET_PATH, namespace)
        pdf_files = glob.glob(os.path.join(namespace_path, '**', '*.pdf'), recursive=True)
        html_files = glob.glob(os.path.join(namespace_path, '**', '*.html'), recursive=True)
        
        files_by_namespace[namespace] = pdf_files + html_files
        print(f"{namespace}: {len(files_by_namespace[namespace])} 파일 찾음")
    
    return files_by_namespace

def sanitize_id(id_str: str) -> str:
    """
    ID를 Pinecone 호환 형식으로 변환합니다 (ASCII 문자만 허용).
    
    Args:
        id_str (str): 원본 ID 문자열
        
    Returns:
        str: ASCII 문자만 포함하는 ID
    """
    # 영숫자(알파벳과 숫자) 및 일부 특수문자만 유지
    # 안전하게 파일 이름의 해시 값으로 대체
    hash_obj = hashlib.md5(id_str.encode('utf-8'))
    hash_str = hash_obj.hexdigest()
    
    # 네임스페이스와 해시를 합쳐서 고유 ID 생성
    # 네임스페이스는 영문으로 변환 (ASCII 호환)
    namespace_map = {
        "Internal Policy": "internal_policy",
        "Product document": "product_doc", 
        "Technical document": "tech_doc"
    }
    
    # 네임스페이스 추출 (ID에서 첫 부분)
    if "_" in id_str:
        orig_namespace = id_str.split("_")[0]
        namespace_prefix = namespace_map.get(orig_namespace, "doc")
    else:
        namespace_prefix = "doc"
    
    # 해시 앞에 네임스페이스 접두사 추가 (분류 용이하게)
    return f"{namespace_prefix}_{hash_str}"

def preprocess_text(text: str) -> str:
    """
    텍스트에 대한 일반적인 전처리를 수행합니다.
    
    Args:
        text (str): 처리할 원본 텍스트
        
    Returns:
        str: 전처리된 텍스트
    """
    if not text or len(text) < 10:  # 너무 짧은 텍스트는 건너뜀
        return text
        
    # 1. 유니코드 정규화 (NFC 형식으로 변환)
    text = unicodedata.normalize('NFC', text)
    
    # 2. 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 3. 중복 문장 부호 제거 (예: !!!, ???, ...)
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    # 4. 특수 문자 제거 (필요한 경우 적절히 조정)
    # text = re.sub(r'[^\w\s\.,;:!?\(\)\-\'\"\[\]]+', '', text)
    
    # 5. 문장 사이에 적절한 공백 추가
    text = re.sub(r'([.!?])([^\s])', r'\1 \2', text)
    
    return text.strip()

def process_namespace(namespace: str, file_paths: List[str]):
    """
    네임스페이스(폴더)별로 파일을 처리하고 Pinecone에 임베딩합니다.
    
    Args:
        namespace (str): 네임스페이스 이름
        file_paths (List[str]): 처리할 파일 경로 목록
    """
    print(f"\n=== {namespace} 네임스페이스 처리 시작 ({len(file_paths)} 파일) ===")
    
    texts = []
    ids = []
    original_filenames = []
    
    # 각 파일의 텍스트 추출
    for file_path in tqdm(file_paths, desc=f"{namespace} 파일 처리"):
        file_name = os.path.basename(file_path)
        # 원본 파일 ID 생성 (나중에 로깅용)
        original_file_id = f"{namespace}_{file_name.replace('.', '_')}"
        # Pinecone 호환 ID 생성
        file_id = sanitize_id(original_file_id)
        
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.html'):
            text = extract_text_from_html(file_path)
        else:
            print(f"지원하지 않는 파일 형식: {file_path}")
            continue
                
        # 추출된 텍스트에 대한 추가 전처리 수행
        text = preprocess_text(text)
        
        if text:
            texts.append(text)
            ids.append(file_id)
            original_filenames.append(file_name)
    
    if not texts:
        print(f"{namespace} 네임스페이스에서 처리할 텍스트가 없습니다.")
        return
    
    print(f"{namespace} 네임스페이스: {len(texts)}개 문서 추출 완료, 임베딩 시작...")
    
    # ID 매핑 정보를 로깅 (나중에 참조할 수 있도록)
    id_map_file = os.path.join(os.path.dirname(__file__), f"{namespace.lower().replace(' ', '_')}_id_map.txt")
    with open(id_map_file, 'w', encoding='utf-8') as f:
        for i, (file_id, original_name) in enumerate(zip(ids, original_filenames)):
            f.write(f"{file_id}\t{original_name}\n")
    
    print(f"ID 매핑 저장 완료: {id_map_file}")
    
    # Dense 인덱스 설정 (첫 네임스페이스만 force_recreate=True, 나머지는 False)
    is_first = namespace == NAMESPACES[0]
    # emb.py의 setup_pinecone_index 기본값을 사용 (index_name="dense-index", metric="cosine")
    dense_index = setup_pinecone_index(force_recreate=is_first) 
    
    # 배치 처리 (메모리 효율성 위해)
    batch_size = 5
    for i in range(0, len(texts), batch_size):
        batch_end = min(i + batch_size, len(texts))
        batch_texts = texts[i:batch_end]
        batch_ids = ids[i:batch_end]
        
        # Dense 임베딩 생성
        dense_embeddings = get_dense_embeddings(batch_texts)
        if not dense_embeddings:
            print(f"배치 {i//batch_size+1} Dense 임베딩 생성 실패")
            continue
        

        
        # 메타데이터 준비 (텍스트와 네임스페이스 포함)
        metadata = [
            {
                "text": text, 
                "namespace": namespace,
                "original_filename": original_filenames[i+j] if i+j < len(original_filenames) else ""
            } 
            for j, text in enumerate(batch_texts)
        ]
        
        # 네임스페이스 변환 (공백 제거 및 소문자로 변환)
        pinecone_namespace = namespace.lower().replace(' ', '_')
        
        records_in_batch = []
        for j, doc_id in enumerate(batch_ids):
            dense_vec = dense_embeddings[j]
            meta = metadata[j]
            record = {
                "id": doc_id,
                "values": dense_vec,
                "metadata": meta
            }
            


            records_in_batch.append(record)
        
        if records_in_batch:
            try:
                print(f"배치 업서트 시도: {len(records_in_batch)}개 레코드, 네임스페이스: {pinecone_namespace}")
                dense_index.upsert(
                    vectors=records_in_batch,
                    namespace=pinecone_namespace
                )
                print(f"{namespace} 배치 {i//batch_size+1}/{(len(texts)-1)//batch_size+1} 업서트 성공 ({len(records_in_batch)}개 레코드)")
            except Exception as e:
                print(f"배치 {i//batch_size+1} 업서트 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"{namespace} 배치 {i//batch_size+1}/{(len(texts)-1)//batch_size+1}: 업서트할 레코드가 없습니다.")
    
    print(f"=== {namespace} 네임스페이스 처리 완료 ===\n")

def main():
    """
    메인 함수: 모든 네임스페이스의 파일을 처리하고 임베딩합니다.
    """
    print("=== 데이터셋 임베딩 시작 ===")
    
    # 파일 수집
    files_by_namespace = get_files_by_namespace()
    
    # 각 네임스페이스 처리
    for namespace, file_paths in files_by_namespace.items():
        process_namespace(namespace, file_paths)
    
    print("=== 데이터셋 임베딩 완료 ===")

if __name__ == "__main__":
    main()
