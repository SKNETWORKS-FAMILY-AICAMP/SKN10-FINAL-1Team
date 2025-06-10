from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypedDict, Dict, Sequence, Union, Optional, Any
import asyncio

from asgiref.sync import sync_to_async
from langchain_core.prompts import PromptTemplate
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
    rag_system_message_agent2
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
# 3) 여러 네임스페이스 중 “가장 높은 유사도”를 준 네임스페이스와 매칭 결과 반환
# --------------------------------------------------
def retrieve_best_namespace(index, query_vector: list, top_k: int = 5):
    """
    1) index.describe_index_stats()를 통해 모든 네임스페이스 목록을 얻는다.
    2) 각 네임스페이스별로 query_vector를 index.query()로 검색하고,
       matches[0].score 를 비교해서 “최고 유사도”를 찾는다.
    3) 가장 높은 유사도를 준 네임스페이스(best_ns)와 해당 네임스페이스의 전체 매칭 결과(best_matches)를 반환.
    """
    stats = index.describe_index_stats()
    available_namespaces = list(stats.namespaces.keys())
    if not available_namespaces:
        raise ValueError("⚠️ 인덱스에 네임스페이스가 없습니다.")

    best_ns = None
    best_score = -1.0
    best_matches = None

    for ns in available_namespaces:
        count = stats.namespaces[ns]["vector_count"]
        if count == 0:
            # 비어 있는 네임스페이스 건너뛰기
            continue

        res = index.query(
            vector=query_vector,
            namespace=ns,
            top_k=top_k,
            include_metadata=True
        )
        if not res.matches:
            continue

        top_score = res.matches[0].score
        if top_score > best_score:
            best_score = top_score
            best_ns = ns
            best_matches = res.matches

    if best_ns is None:
        raise ValueError("⚠️ 어떤 네임스페이스에서도 매칭 결과를 찾을 수 없습니다.")
    
    print(f"🔍 선택된 네임스페이스: '{best_ns}' (최고 유사도: {best_score:.4f})")
    return best_ns, best_matches


# --------------------------------------------------
# 4) 검색된 매칭 결과에서 실제 텍스트(메타데이터)를 꺼내 Context 로 결합
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
# 5) LLM ChatCompletion 호출하여 답변 생성
# --------------------------------------------------
def generate_answer_with_context(openai_client: OpenAI, question: str, context: str) -> str:
    """
    최신 OpenAI 클라이언트에서는 client.chat.completions.create(...) 형태를 씁니다.
    """
    formatted_prompt = rag_answer_generation_prompt_agent2.format(context=context, question=question)
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": rag_system_message_agent2},
            {"role": "user", "content": formatted_prompt}
        ],
        temperature=0.0,
        max_tokens=1024
    )
    # resp.choices[0].message.content 으로 답변 추출
    return resp.choices[0].message.content.strip()



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
    Langchain LLM을 사용하여 문서 타입을 분류합니다. 리턴 데이터 형식은 기존과 동일하게 유지합니다.
    """
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo') # 또는 다른 모델
    prompt = PromptTemplate.from_template(document_type_system_prompt_agent2)
    chain = prompt | llm
    
    response = chain.ainvoke({"input": message})
    classified_type = response.content.strip()
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
    # print(f"\n📄 RAG 노드 실행: 문서 타입 = '{state.document_type}', 질문 = '{state.user_input}'")
    openai_client, pinecone_index = init_clients()
    # print("   - 클라이언트 초기화 완료")

    query_vector = embed_query(openai_client, state.user_input)
    # print(f"   - 질문 임베딩 완료 (벡터 크기: {len(query_vector)})")

    namespace_to_search = state.document_type
    if not namespace_to_search or namespace_to_search == "unknown":
        message = f"문서 타입이 '{namespace_to_search}'(으)로 분류되어 Pinecone 검색을 수행하지 않습니다."
        # print(f"   - 정보: {message}")
        # 'unknown'일 경우, unknown_handler_node에서 이미 메시지를 설정했을 수 있으므로, 여기서는 덮어쓰지 않거나
        # 혹은 여기서 다른 메시지를 설정할 수 있습니다. 여기서는 검색 불가 메시지만 남깁니다.
        # 실제로는 'unknown' 타입은 이 노드로 오지 않고 unknown_handler_node로 가야 합니다.
        # 이 코드는 execute_rag_node가 'unknown' 타입으로 호출될 경우를 대비한 방어 코드입니다.
        state.result = "적절한 문서 저장소를 찾을 수 없어 검색을 수행할 수 없습니다."
        return state.dict()

    # print(f"   - Pinecone 네임스페이스 '{namespace_to_search}'에서 검색 시작...")
    index_stats = pinecone_index.describe_index_stats()
    if namespace_to_search not in index_stats.namespaces or \
        index_stats.namespaces[namespace_to_search].vector_count == 0:
        message = f"'{namespace_to_search}' 네임스페이스를 Pinecone에서 찾을 수 없거나, 해당 네임스페이스에 데이터가 없습니다. Pinecone 대시보드에서 네임스페이스 이름과 데이터 존재 여부를 확인해주세요."
        # print(f"   - 경고: {message}")
        state.result = message
        return state.dict()

    res = pinecone_index.query(
        vector=query_vector,
        namespace=namespace_to_search,
        top_k=5, # 검색할 문서 수
        include_metadata=True
    )
    matches = res.matches
    # print(f"   - Pinecone 검색 완료: {len(matches)}개 결과 수신")

    if not matches:
        message = f"'{namespace_to_search}' 네임스페이스에서 '{state.user_input}' 질문과 관련된 정보를 찾지 못했습니다."
        # print(f"   - 정보 없음: {message}")
        state.result = message
        return state.dict()
    
    context = build_context_from_matches(matches)
    if not context:
        message = "검색된 정보에서 답변을 생성할 컨텍스트를 추출하지 못했습니다."
        print(f"   - 컨텍스트 구축 실패: {message}")
        state.result = message
        return state.dict()
    print(f"   - 컨텍스트 구축 완료 (길이: {len(context)})")

    state.result = context
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
async def async_choose_node(state: State):
    return await sync_to_async(choose_node)(state)

async def async_execute_rag(state: State):
    return await sync_to_async(execute_rag)(state)

async def async_summarize_node(state: State):
    return await sync_to_async(summarize_node)(state)

async def async_choose_one(state: State):
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