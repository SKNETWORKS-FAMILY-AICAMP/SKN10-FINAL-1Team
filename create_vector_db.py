from bs4 import BeautifulSoup
import re
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from chromadb import Client
from dotenv import load_dotenv
"""
1. Loader로 파일 로드(html은 BeautifulSoup로 전처리하므로 필요 x)
2. 전처리 후 Document() 객체 생성 (page_content, metadata)
3. RecursiveCharacterTextSplitter로 해당 Document 객체를 split
4. Document 객체들을 리스트에 extend시킴.
5. 해당 Document 리스트를 기준으로 Chroma DB 생성
"""



# BeautifulSoup로 해당 경로의 html 파일을 읽고 전처리해주는 함수
def load_and_clean_data(file_path) :
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        text = soup.get_text()
        # {% ... %} 같은 Django 템플릿 태그 제거
        text = re.sub(r"{%.*?%}", "", text)
        return text

# 지정된 디렉토리의 모든 html를 읽고 전처리하여 딕셔너리를 담은 리스트를 반환
def load_all_document(directory):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, # 청크 크기
    chunk_overlap=50, # 청크 간의 중복되는 문자 수
    length_function=len,
    is_separator_regex=False,
    )
    # os.listdir() : 지정된 디렉토리의 파일과 디렉토리 목록을 반환
    for filename in os.listdir(directory):
        # filename.endswith(".html") : html 파일로 끝나는지 확인
        if filename.endswith(".html"):
            file_path = os.path.join(directory, filename)
            content = load_and_clean_data(file_path)
            doc = Document(page_content=content, metadata={"source": file_path})
            split_doc = text_splitter.split_documents([doc])  # 리스트로 감싸야 함
            documents.extend(split_doc)
    return documents

# Chroma DB 생성 (루트 디렉토리에)
def create_chroma_db(documents) :
    # 해당 Document 리스트를 기준으로 Chroma Db에 저장
    # 주의점 : 같은 문서를 두 번 저장하면, 덮어쓰기가 아니라 추가가 된다!
    db = Chroma.from_documents(
        documents, OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")), persist_directory="./chroma_db", collection_name="my_db"
    )
    # 모든 문서 ID를 가져와 Chunk의 개수 확인(원본 문서 개수 x, chunk의 갯수 O)
    docs = db.get()
    print(f"문서 개수: {len(docs['ids'])}")
    print("-----Chroma Db에 저장되었습니다-----")
    print(docs)

def main() :
    load_dotenv()
    ai_computing_document = load_all_document("templates/product/ai_computing")
    db_document = load_all_document("templates/product/db")
    network = load_all_document("templates/product/network")
    all_docs = ai_computing_document + db_document + network
    create_chroma_db(all_docs)

main()
