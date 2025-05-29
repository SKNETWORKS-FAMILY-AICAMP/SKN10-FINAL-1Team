from django.shortcuts import render, redirect
from .models import *
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

"""
1. get 방식으로 파라미터 입력받음
2. 파라미터가 있다면 해당 세션 선택, 없다면 세션들중에 가장 최근의 세션 선택
3. 세션 자체가 아무것도 없다면 새 세션을 생성 (세션 title은 "새 세션")
4. 해당 세션으로 사용자 입력을 메시지로 넣고, 챗봇 응답도 메세지로 넣음.

"""



# Create your views here.
def index(request) :
    session_id = request.GET.get('session_id') # GET 파라미터가 없다면, request.GET.get()은 기본적으로 None을 반환
    if session_id is None : 
        if ChatSession.objects.exists() : # 남은 세션이 1개라도 있다면 가장 최근의 세션을 선택
            session_id = ChatSession.objects.order_by('-started_at').first().id
        else : # 남은 세션이 없을때, 새 세션을 생성하여 선택
            new_session = ChatSession.objects.create()
            session_id = new_session.id
    

    if request.method == 'POST' :
        # 유저 메시지 저장
        
        user_msg = request.POST.get('input-box').strip()
        user_chat = ChatMessage(is_human=True, message=user_msg)
        user_chat.save()
        
        # 1. 기존에 만든 chroma_db 경로
        persist_directory = "./chroma_db"
        # 2. 임베딩 모델 (기존과 동일해야 함)
        embedding_model = OpenAIEmbeddings()

        # 3. 저장된 Chroma DB 불러오기
        vectordb = Chroma(
            embedding_function=embedding_model,
            persist_directory=persist_directory,
            collection_name="my_db"  # 기존에 만든 컬렉션 이름
        )

        # 5. RAG용 Chain 구성
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        llm = ChatOpenAI(temperature=0)  # OpenAI API 호출 (GPT-4 등)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # 6. LangChain + RAG에 user_msg
        result = qa_chain(user_msg)
        chatbot_chat = ChatMessage(is_human=False, message=result['result'])
        chatbot_chat.save()

    content = {"messages" : ChatMessage.objects.all().order_by('created_at')}
    return render(request, "code_seeker/index.html", content)

def delete_message(request) : 
    ChatMessage.objects.all().delete()
    return redirect("homepage")