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
import re

# Import functions from emb.py
from emb import setup_pinecone_index, get_dense_embeddings, get_sparse_embeddings, index_documents

# 올바른 Pinecone 클라이언트 버전을 사용하고 있는지 확인
import pkg_resources
installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
print(f"Pinecone 클라이언트 버전: {installed_packages.get('pinecone', '설치되지 않음')}")


# Define paths
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
NAMESPACES = ['Internal Policy', 'Product document', 'Technical document', 'Proceedings']

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
        "Technical document": "tech_doc",
        "Proceedings": "proceedings"
    }
    
    # 네임스페이스 추출 (ID에서 첫 부분)
    if "_" in id_str:
        orig_namespace = id_str.split("_")[0]
        namespace_prefix = namespace_map.get(orig_namespace, "doc")
    else:
        namespace_prefix = "doc"
    
    # 해시 앞에 네임스페이스 접두사 추가 (분류 용이하게)
    return f"{namespace_prefix}_{hash_str}"

def chunk_text(text: str, max_chars: int = 7000, overlap: int = 200) -> List[str]:
    """
    텍스트를 지정된 최대 문자 수와 오버랩을 사용하여 청크로 나눕니다.
    OpenAI text-embedding-3-large 모델의 최대 토큰은 8192입니다.
    1 토큰 ~= 4자로 가정하고, max_chars는 토큰 한계보다 충분히 작게 설정합니다.
    (예: 7000자 ~= 1750 토큰)

    Args:
        text (str): 청킹할 원본 텍스트
        max_chars (int): 각 청크의 최대 문자 수
        overlap (int): 청크 간의 문자 오버랩 수

    Returns:
        List[str]: 텍스트 청크 목록
    """
    if not text or not isinstance(text, str):
        return []

    chunks = []
    start_index = 0
    text_len = len(text)

    while start_index < text_len:
        end_index = min(start_index + max_chars, text_len)
        chunk = text[start_index:end_index]
        chunks.append(chunk)

        if end_index == text_len:
            break  # 텍스트의 끝에 도달

        # 다음 청크의 시작 위치를 오버랩을 고려하여 설정
        start_index += (max_chars - overlap)
        
        # 만약 오버랩으로 인해 start_index가 end_index를 넘어가면 무한 루프 방지
        if start_index >= end_index:
            # 이 경우는 max_chars가 overlap보다 작거나 같을 때 발생 가능성이 있지만,
            # 일반적인 사용 (max_chars > overlap)에서는 드묾.
            # 남은 텍스트가 매우 짧을 때도 발생 가능.
            # 마지막 청크를 이미 추가했으므로 중단.
            if end_index == text_len and chunks[-1] == text[start_index- (max_chars - overlap):]:
                 break
            # 혹은 남은 부분을 새 청크로 추가하고 종료
            remaining_text = text[start_index:]
            if remaining_text.strip():
                chunks.append(remaining_text)
            break
            
    # 비어 있거나 공백만 있는 청크 제거
    return [c for c in chunks if c and c.strip()]

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

    date_prefixes = []
    if namespace == "Proceedings":
        for original_file_name_for_date in original_filenames:
            match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", original_file_name_for_date)
            if match:
                year, month, day = match.groups()
                date_prefixes.append(f"{year}년 {month}월 {day}일 회의록: ")
            else:
                date_prefixes.append("") # No date found, empty prefix
    else:
        date_prefixes = [""] * len(original_filenames)

    # Dense 인덱스 설정 (첫 네임스페이스만 force_recreate=True, 나머지는 False)
    is_first = namespace == NAMESPACES[0]
    dense_index = setup_pinecone_index(force_recreate=is_first)

    all_chunk_texts: List[str] = []
    all_chunk_ids: List[str] = []
    all_chunk_metadata: List[Dict] = []

    print(f"문서 내용 청킹 시작 ({len(texts)}개 문서)...")
    for doc_idx, doc_text in enumerate(tqdm(texts, desc=f"Chunking {namespace}")):
        doc_id = ids[doc_idx]
        original_file = original_filenames[doc_idx]
        date_prefix = date_prefixes[doc_idx]
        
        # 텍스트 청킹 (최대 7000자, 약 1750 토큰, 오버랩 200자)
        # OpenAI text-embedding-3-large 모델의 최대 토큰은 8192
        # 1 토큰 ~= 4자로 가정하고 안전 마진을 둠
        doc_chunks = chunk_text(doc_text, max_chars=7000, overlap=200) 
        
        for chunk_idx, chunk_content in enumerate(doc_chunks):
            # 청크 ID는 원래 문서 ID에 청크 번호를 추가하여 생성
            # sanitize_id는 이미 md5 해시된 doc_id에 대해 불필요할 수 있으나, 일관성 유지
            chunk_id = sanitize_id(f"{doc_id}_chunk_{chunk_idx}")
            
            prefixed_chunk_content = date_prefix + chunk_content
            
            all_chunk_texts.append(prefixed_chunk_content)
            all_chunk_ids.append(chunk_id)
            all_chunk_metadata.append({
                "text": prefixed_chunk_content,
                "original_document_id": doc_id,
                "original_filename": original_file,
                "namespace": namespace,
                "chunk_index": chunk_idx
            })

    if not all_chunk_texts:
        print(f"{namespace} 네임스페이스에서 처리할 청크가 없습니다.")
        return

    print(f"{namespace} 네임스페이스: {len(all_chunk_texts)}개 청크 생성 완료, 임베딩 및 업서트 시작...")

    # 배치 처리 (메모리 효율성 위해)
    batch_size = 32 # OpenAI API 및 Pinecone 권장 사항에 따라 배치 크기 조정 가능
    for i in range(0, len(all_chunk_texts), batch_size):
        batch_end = min(i + batch_size, len(all_chunk_texts))
        current_batch_texts = all_chunk_texts[i:batch_end]
        current_batch_ids = all_chunk_ids[i:batch_end]
        current_batch_metadata_list = all_chunk_metadata[i:batch_end]
        
        # Dense 임베딩 생성
        # get_dense_embeddings 함수는 내부적으로 OpenAI API의 배치 제한을 처리할 수 있음
        dense_embeddings = get_dense_embeddings(current_batch_texts)
        if not dense_embeddings or len(dense_embeddings) != len(current_batch_texts):
            print(f"배치 {i//batch_size+1} Dense 임베딩 생성 실패 또는 개수 불일치. 건너<0xEB><0x84><0x88>니다.")
            # 실패한 텍스트 로깅 (선택 사항)
            # for k, txt in enumerate(current_batch_texts):
            #     if k >= len(dense_embeddings):
            #         print(f"  실패 텍스트 (ID: {current_batch_ids[k]}): {txt[:100]}...")
            continue
        
        # 네임스페이스 변환 (공백 제거 및 소문자로 변환)
        pinecone_namespace = namespace.lower().replace(' ', '_')
        
        records_to_upsert = []
        for j, chunk_id_val in enumerate(current_batch_ids):
            # 메타데이터는 이미 all_chunk_metadata에서 준비됨
            meta = current_batch_metadata_list[j]
            record = {
                "id": chunk_id_val,
                "values": dense_embeddings[j],
                "metadata": meta
            }
            records_to_upsert.append(record)
        
        if records_to_upsert:
            try:
                # print(f"배치 업서트 시도: {len(records_to_upsert)}개 레코드, 네임스페이스: {pinecone_namespace}")
                dense_index.upsert(
                    vectors=records_to_upsert,
                    namespace=pinecone_namespace
                )
                # print(f"{namespace} 배치 {i//batch_size+1}/{(len(all_chunk_texts)-1)//batch_size+1} 업서트 성공 ({len(records_to_upsert)}개 레코드)")
            except Exception as e:
                print(f"배치 {i//batch_size+1} 업서트 중 오류 발생: {e}")
                # import traceback
                # traceback.print_exc()
        # else:
            # print(f"{namespace} 배치 {i//batch_size+1}/{(len(all_chunk_texts)-1)//batch_size+1}: 업서트할 레코드가 없습니다.")
    
    print(f"=== {namespace} 네임스페이스 처리 완료 ({len(all_chunk_texts)} 청크 업서트 시도) ===\n")

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
