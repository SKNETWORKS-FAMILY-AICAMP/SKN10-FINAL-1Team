from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypedDict, Dict, Sequence, Union, Optional, Any
import asyncio

from asgiref.sync import sync_to_async
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from fastapi_server.agent.prompt import (
    document_type_system_prompt_agent2,
    proceedings_summary_prompt_agent2,
    internal_policy_summary_prompt_template_agent2,
    product_document_summary_prompt_template_agent2,
    technical_document_summary_prompt_template_agent2,
    unknown_document_type_prompt_agent2,
    rag_answer_generation_prompt_agent2,
    rag_system_message_agent2,
    create_similar_questions_agent2
)
load_dotenv()

def init_clients():
    # 1-1) OpenAI 클라이언트 생성
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("⚠️ 환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")
    # OpenAI 인스턴스를 만듭니다.
    openai_client = OpenAI(api_key=openai_api_key)
    print("✅ OpenAI 클라이언트 생성 완료")

    # 1-2) Pinecone 인스턴스 생성
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env     = os.getenv("PINECONE_ENVIRONMENT")   # 예: "us-east1-gcp" 또는 "us-west1-gcp" 등
    if not pinecone_api_key or not pinecone_env:
        raise ValueError("⚠️ 환경변수 PINECONE_API_KEY 또는 PINECONE_ENVIRONMENT가 누락되었습니다.")

    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
    print("✅ Pinecone 클라이언트 생성 완료")

    # 1-3) 인덱스 존재 여부 확인
    index_name = "dense-index"  # 실제 사용 중인 인덱스 이름으로 교체하세요
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        raise ValueError(f"⚠️ 인덱스 '{index_name}'가 Pinecone에 존재하지 않습니다. 현재 인덱스 목록: {existing_indexes}")

    # 1-4) 해당 인덱스 객체 가져오기
    index = pc.Index(index_name)
    print(f"✅ Pinecone 인덱스 '{index_name}' 연결 완료 (Namespaces: {len(index.describe_index_stats().namespaces)})")

    return openai_client, index


# --------------------------------------------------
# 2) 질문 문장을 임베딩 벡터로 변환
# --------------------------------------------------
def embed_query(openai_client: OpenAI, text: str) -> list:
    """
    최신 OpenAI 클라이언트에서는 resp.data[0].embedding 으로 벡터에 접근해야 합니다.
    """
    resp = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp.data[0].embedding

# --------------------------------------------------
# 3) 검색된 매칭 결과에서 실제 텍스트(메타데이터)를 꺼내 Context 로 결합
# --------------------------------------------------
def build_context_from_matches(matches):
    """
    res.matches 리스트 안의 각 item.metadata 에 들어 있는 텍스트 필드를 추출합니다.
    업로드 시 metadata 키가 "text"였다고 가정했습니다.
    """
    contexts = []
    for m in matches:
        chunk_text = m.metadata.get("text", "")
        if chunk_text:
            contexts.append(chunk_text)

    return "\n---\n".join(contexts)

# --------------------------------------------------
# 4) fetch_res.vectors에서 metadata중에 text만 추출해서 join
# --------------------------------------------------
def combine_text(fetch_res, final_doc_ids) :
    texts = []
    for doc_id in final_doc_ids:
        vec_info = fetch_res.vectors.get(doc_id)
        text = vec_info["metadata"].get("text", "")
        if text:
            texts.append(text)

    # 4) 최종 context 조립
    context = "\n---\n".join(texts)
    return context

# --------------------------------------------------
# 5) 검색된 매칭 결과에서 문서 ID 추출
# --------------------------------------------------
def get_document_id(matches) :
    ids = []
    for m in matches:
        ids.append(m.id)
        print(m.score,end=" ") # 디버깅용 점수 출력 (pinecone은 기본적으로 점수 내림차순으로 matches가 정렬됨)
    print() 
    return ids


# --------------------------------------------------
# 6) 유사 질문 생성 함수 (4개의 유사 질문 생성)
# --------------------------------------------------
def create_similar_questions(message) :
    client = OpenAI()
    formatted_prompt = create_similar_questions_agent2.format(user_input=message)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": formatted_prompt}
        ],
        temperature=0,
        max_tokens=100
    )
    
    similar_questions = resp.choices[0].message.content.strip().split("\n")
    print(similar_questions)
    print(f"생성된 유사 질문의 자료형 : {type(similar_questions)}")
    return similar_questions

@dataclass
class State:
    # Compatible with both direct user_input and messages-based interface
    user_input: str = ""
    document_type: str = ""
    result: str = ""
    messages: Sequence[BaseMessage] = None
    
    def __post_init__(self):
        # If initialized from supervisor with messages but no user_input, extract user_input
        if not self.user_input and self.messages:
            # Extract user input from the last human message
            user_messages = [msg for msg in self.messages if isinstance(msg, HumanMessage)]
            if user_messages:
                self.user_input = user_messages[-1].content
    
    def dict(self):
        """Return dict representation with messages if present"""
        result = {
            "result": self.result,
            "document_type": self.document_type,
            "user_input": self.user_input
        }
        # If this was called with messages, return updated messages too
        if self.messages is not None:
            result["messages"] = self.messages + [AIMessage(content=self.result)] if self.result else self.messages
        return result  # 챗봇 결과


def choose_document_type(message):
    """
    OpenAI 클라이언트를 사용하여 문서 타입을 분류합니다. 리턴 데이터 형식은 기존과 동일하게 유지합니다.
    """
    client = OpenAI()
    formatted_prompt = document_type_system_prompt_agent2.format(user_input=message)
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": formatted_prompt}
        ],
        temperature=0,
        max_tokens=100
    )
    
    classified_type = resp.choices[0].message.content.strip()
    print(f"문서 타입 분류 결과: {classified_type}")
    return classified_type

def choose_node(state: State):
    # Extract the user input from the state
    user_input = state.user_input

    # Choose document type
    document_type = choose_document_type(user_input)
    
    # Update state with document type
    state.document_type = document_type
    
    # Print document type for debugging
    # print(f"Document Type: {document_type}")
    
    return state.dict()

def choose_one(state: State) -> str:
    choice = state.document_type
    # Use logging instead of print to avoid output being captured in response
    # print(f"(choice_one) Choice: {choice}")
    # This must return the string key for conditional edge routing
    if choice in ["internal_policy", "product_document", "technical_document", "proceedings"]:
        return choice
    else:
        return "product_document"  # Default fallback

def execute_rag(state: State):
    # 1) openai_client와 index 초기화
    openai_client, pinecone_index = init_clients()
    namespace_to_search = state.document_type
    index_stats = pinecone_index.describe_index_stats()

    # 2) RAG Fusion을 위한 dictionary 초기화
    # question_to_doc : 질문번호와 해당 질문에 대한 문서 ID 매핑
    # document_score : 문서 id와 해당 문서의 RRF 점수 매핑
    question_to_doc = dict.fromkeys([0,1,2,3,4])
    document_score = dict()

    # 3) 질문을 유사 질문 4개로 확장하고, 원본 질문도 포함
    # similar_questions : 유사 질문 4개 + 원본 질문 1개
    similar_questions = create_similar_questions(state.user_input)
    similar_questions.append(state.user_input) 
    
    # 4) RAG Fusion을 위한 질문-문서 매핑
    # 각 질문마다 embedding 벡터를 생성하고 Pinecone에서 유사도 검색을 수행하여 최대 문서 4개를 찾는다.
    # 각 질문에 대해 문서 ID를 매핑함.
    for i , question in enumerate(similar_questions) :
        query_vector = embed_query(openai_client, question.strip())
        if not namespace_to_search or namespace_to_search == "unknown":
            message = f"문서 타입이 '{namespace_to_search}'(으)로 분류되어 Pinecone 검색을 수행하지 않습니다."
            # 이 코드는 execute_rag_node가 'unknown' 타입으로 호출될 경우를 대비한 방어 코드입니다.
            state.result = "적절한 문서 저장소를 찾을 수 없어 검색을 수행할 수 없습니다."
            return state.dict()

        if namespace_to_search not in index_stats.namespaces or \
            index_stats.namespaces[namespace_to_search].vector_count == 0:
            message = f"'{namespace_to_search}' 네임스페이스를 Pinecone에서 찾을 수 없거나, 해당 네임스페이스에 데이터가 없습니다. Pinecone 대시보드에서 네임스페이스 이름과 데이터 존재 여부를 확인해주세요."
            state.result = message
            return state.dict()

        res = pinecone_index.query(
            vector=query_vector,
            namespace=namespace_to_search,
            top_k=4, # 검색할 문서 수
            include_metadata=True
        )

        matches = res.matches
        print(f"질문 {i+1} : '{question}'")
        print(f"   - Pinecone 검색 완료: {len(matches)}개 결과 수신")

        if not matches:
            message = f"'{namespace_to_search}' 네임스페이스에서 '{question}' 질문과 관련된 정보를 찾지 못했습니다."
            # print(f"   - 정보 없음: {message}")
            state.result = message
            return state.dict()
        question_to_doc[i] = get_document_id(matches)

    print(question_to_doc) # 디버깅용 출력 (질문1~5에 대한 문서 id 매핑)
    print()

    # 5) RRF 점수 계산
    for key in question_to_doc :
        for i, doc_id in enumerate(question_to_doc[key]) :
            if doc_id not in document_score :
                document_score[doc_id] = float(1/(60+1+i))
            else : 
                document_score[doc_id] += float(1/(60+1+i))
    print(f"문서 점수 : {document_score}") # 디버깅용 출력 (문서 id와 점수 매핑)

    # 6) RRF 점수가 높은 상위 2개 슬라이싱
    top2 = sorted(
        document_score.items(),
        key=lambda x: x[1],
        reverse=True
    )[:2]
    final_doc_ids = [doc_id for doc_id, _ in top2]
    print(final_doc_ids) # 디버깅용 출력 (최종 문서 ID 리스트)

    # 7) 최종 문서 ID를 사용하여 Pinecone에서 fetch
    fetch_res = pinecone_index.fetch(
        ids=final_doc_ids,
        namespace=namespace_to_search
    )

    # 8) 최종 context 조립
    context = combine_text(fetch_res, final_doc_ids)

    state.result = context
    print(f"(길이: {len(context)})")  
    print(f"컨텍스트 요약 50자 : {context}\n")
    return state.dict()
    

def summarize_node(state: State):
    text = state.result
    document_type = state.document_type
    user_input = state.user_input

    if state.document_type == "proceedings":
        system_message = proceedings_summary_prompt_agent2
    elif state.document_type == "internal_policy":
        system_message = internal_policy_summary_prompt_template_agent2.format(user_input=user_input)
    elif state.document_type == "product_document":
        system_message = product_document_summary_prompt_template_agent2.format(user_input=user_input)
    elif state.document_type == "technical_document":
        system_message = technical_document_summary_prompt_template_agent2.format(user_input=user_input)
    else: # unknown or fallback
        system_message = unknown_document_type_prompt_agent2
    
    system_message = SystemMessagePromptTemplate.from_template(system_message)
    human_message = HumanMessagePromptTemplate.from_template("{text}")

    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    # 2. LLM 생성
    llm = ChatOpenAI(model="gpt-4o")

    # 3. Prompt와 LLM 결합
    chatbot = chat_prompt | llm

    # 4. 실행
    response = chatbot.invoke({"text": text})
    result = response.content
    # print(result)  # 디버깅용 출력 제거
    state.result = result
    
    # 문서 타입이 최종 결과에 포함되지 않도록, document_type을 제외한 상태만 반환
    result_state = state.dict()
    if "document_type" in result_state:
        # document_type 값이 최종 출력에 포함되지 않도록 제거
        del result_state["document_type"]
    
    return result_state

# 비동기 노드 래퍼 함수들 정의
async def async_choose_node(state: State) :
    return await sync_to_async(choose_node)(state)

async def async_execute_rag(state: State) :
    return await sync_to_async(execute_rag)(state)

async def async_summarize_node(state: State) :
    return await sync_to_async(summarize_node)(state)

async def async_choose_one(state: State) :
    return await sync_to_async(choose_one)(state)

# Define the graph with async nodes
graph = (
    StateGraph(State)
    # (1) choose_node 분기 노드 등록 (outputs에 리턴 키 명시)
    .add_node("choose_node", async_choose_node)
    # (2) RAG 실행 노드들 등록
    .add_node("product_node", async_execute_rag)
    .add_node("proceedings_node", async_execute_rag)
    .add_node("hr_policy_node", async_execute_rag)
    .add_node("technical_document_node", async_execute_rag)
    # (3) summarize_node 등록 (최종 노드)
    .add_node("summarize_node", async_summarize_node)
    # (4) START → 분기 노드(choose_node) → (next_node 값에 따라) 분기
    .add_edge(START,"choose_node")
    .add_conditional_edges(
        "choose_node",
        async_choose_one,
        {
            "product_document": "product_node",
            "proceedings": "proceedings_node",
            "internal_policy": "hr_policy_node",
            "technical_document": "technical_document_node"
        }
    )
    # (5) 각 RAG 노드 → summarize_node 연결
    .add_edge("product_node", "summarize_node")
    .add_edge("proceedings_node", "summarize_node")
    .add_edge("hr_policy_node", "summarize_node")
    .add_edge("technical_document_node", "summarize_node")
    .add_edge("summarize_node", END)
    # (6) 최종 컴파일
    .compile(name="New Graph")
)