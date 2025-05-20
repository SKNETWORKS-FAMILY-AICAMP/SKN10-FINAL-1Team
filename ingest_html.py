import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # ✅ 최신 경로

load_dotenv()  # 🔐 .env에서 키 불러오기

html_folder = "./product/"
docs = []

for filename in os.listdir(html_folder):
    if not filename.endswith(".html"):
        continue
    with open(os.path.join(html_folder, filename), 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        text = soup.get_text(separator="\n").strip()
        source_url = f"http://localhost:8800/product/{filename}"
        docs.append(Document(page_content=text, metadata={"source": source_url}))

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embedding = OpenAIEmbeddings()  # 🔑 .env에서 자동 인식
vectorstore = Chroma.from_documents(split_docs, embedding, persist_directory="./chroma_store")
vectorstore.persist()

print("✅ HTML 문서 벡터화 완료!")
