import chainlit as cl
from dotenv import load_dotenv
import os

# ✅ 최신 권장 import
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

TEMPLATE = """
당신은 AI SERV의 가이드 문서를 기반으로 질문에 답하는 도우미입니다.

문서 내용:
{context}

사용자 질문:
{question}

문서 내용을 바탕으로 핵심만 간단하게 설명해주고,
자세한 내용은 관련 문서 링크로 안내해주세요.
"""

prompt = PromptTemplate(
    template=TEMPLATE,
    input_variables=["context", "question"]  # ⬅️ context 추가 중요!!!
)

def build_chain():
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory="./chroma_store",
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa

@cl.on_chat_start
def start_chat():
    cl.user_session.set("qa_chain", build_chain())
    cl.Message(content="안녕하세요! AI SERV 가이드 챗봇입니다. 궁금한 걸 물어보세요 🤖").send()

@cl.on_message
async def respond(message: cl.Message):
    qa_chain = cl.user_session.get("qa_chain")
    result = qa_chain(message.content)

    answer = result["result"]
    sources = result.get("source_documents", [])
    if sources:
        links = "\n\n📎 관련 문서 링크:\n"
        # 중복 제거된 링크 목록 만들기
        unique_links = {}
        for doc in sources:
            source = doc.metadata['source']
            title = source.split("/")[-1]
            unique_links[source] = title  # 중복 자동 제거됨

        # 출력
        if unique_links:
            links = "\n\n📎 관련 문서 링크:\n"
            for url, title in unique_links.items():
                links += f"- [{title}]({url})\n"
            answer += links

    await cl.Message(content=answer).send()
