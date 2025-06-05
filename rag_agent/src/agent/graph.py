from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypedDict, Dict # Dict 추가

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
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
        raise ValueError("⚠️ 환경변수 PINECONE_API_KEY 또는 PINECONE_ENV가 누락되었습니다.")

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
    prompt = f"""아래 Context를 참고해서 질문에 답변해주세요.

    Context:
    {context}

    질문:
    {question}
    """
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 친절한 어시스턴트입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1024
    )
    # resp.choices[0].message.content 으로 답변 추출
    return resp.choices[0].message.content.strip()



@dataclass
class State:
    # 초기값을 설정해 주지 않으면 인자가 required가 된다.
    user_input : str # 사용자 입력
    document_type: str = ""   # 문서 형태
    result: str = ""  # 챗봇 결과


def choose_document_type(message):
    prompt = PromptTemplate(
        input_variables=["input"],
        template="The user's question is: '{input}'. Analyze the question and categorize it into one of these document types: product, hr_policy, technical_document, proceedings. " \
        "Respond with EXACTLY ONE of these four values: 'product_document' for Product-related questions, 'proceedings' for Meeting Proceedings, 'internal_policy' for HR Policy documents, or 'technical_document' for Technical Documentation. " \
        "You must ONLY respond with one of these four exact values, without any additional text or explanation."
    )

    # LLM 설정 
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # prompt + llm
    chain = prompt | llm

    # 4. 실행
    response = chain.invoke({"input": message})
    print(response.content)
    return response

def choose_node(state: State) -> str:
    user_input = state.user_input
    document_type = choose_document_type(user_input).content.strip().lower()
    keywords = ["internal_policy", "product_document", "technical_document", "proceedings"]

    if document_type not in keywords : 
        state.document_type = "unknown"
    else : 
        state.document_type = document_type

    return {
        "document_type": state.document_type,
        "result": None
    }

def choose_one(state: State) -> str:
    choice = state.document_type
    if choice in ["internal_policy", "product_document", "technical_document", "proceedings"]:
        return choice
    else:
        return "unknown"

def execute_rag(state: State) -> str :
    print(f"\n📄 RAG 노드 실행: 문서 타입 = '{state.document_type}', 질문 = '{state.user_input}'")
    openai_client, pinecone_index = init_clients()
    print("   - 클라이언트 초기화 완료")

    query_vector = embed_query(openai_client, state.user_input)
    print(f"   - 질문 임베딩 완료 (벡터 크기: {len(query_vector)})")

    namespace_to_search = state.document_type
    if not namespace_to_search or namespace_to_search == "unknown":
        message = f"문서 타입이 '{namespace_to_search}'(으)로 분류되어 Pinecone 검색을 수행하지 않습니다."
        print(f"   - 정보: {message}")
        # 'unknown'일 경우, unknown_handler_node에서 이미 메시지를 설정했을 수 있으므로, 여기서는 덮어쓰지 않거나
        # 혹은 여기서 다른 메시지를 설정할 수 있습니다. 여기서는 검색 불가 메시지만 남깁니다.
        # 실제로는 'unknown' 타입은 이 노드로 오지 않고 unknown_handler_node로 가야 합니다.
        # 이 코드는 execute_rag_node가 'unknown' 타입으로 호출될 경우를 대비한 방어 코드입니다.
        return {"result": "적절한 문서 저장소를 찾을 수 없어 검색을 수행할 수 없습니다."}


    print(f"   - Pinecone 네임스페이스 '{namespace_to_search}'에서 검색 시작...")
    index_stats = pinecone_index.describe_index_stats()
    if namespace_to_search not in index_stats.namespaces or \
        index_stats.namespaces[namespace_to_search].vector_count == 0:
        message = f"'{namespace_to_search}' 네임스페이스를 Pinecone에서 찾을 수 없거나, 해당 네임스페이스에 데이터가 없습니다. Pinecone 대시보드에서 네임스페이스 이름과 데이터 존재 여부를 확인해주세요."
        print(f"   - 경고: {message}")
        return {"result": message}

    res = pinecone_index.query(
        vector=query_vector,
        namespace=namespace_to_search,
        top_k=5, # 검색할 문서 수
        include_metadata=True
    )
    matches = res.matches
    print(f"   - Pinecone 검색 완료: {len(matches)}개 결과 수신")

    if not matches:
        message = f"'{namespace_to_search}' 네임스페이스에서 '{state.user_input}' 질문과 관련된 정보를 찾지 못했습니다."
        print(f"   - 정보 없음: {message}")
        return {"result": message}
    
    context = build_context_from_matches(matches)
    if not context:
        message = "검색된 정보에서 답변을 생성할 컨텍스트를 추출하지 못했습니다."
        print(f"   - 컨텍스트 구축 실패: {message}")
        return {"result": message}
    print(f"   - 컨텍스트 구축 완료 (길이: {len(context)})")

    return {"result": context}
    

def summarize_node(state: State) -> str:
    text = state.result
    document_type = state.document_type
    user_input = state.user_input
    order = ""

    if document_type == "proceedings" : 
        order = (
            "너는 회의록 어시스턴트야. 내가 회의록 텍스트를 주면, "
            "첫번째로 회의 주제 / 목적, "
            "두번째로 핵심 논의 사항, "
            "세번째로 결정된 사항, "
            "네번째로 보류되거나 추가 논의 항목을 "
            "요약해서 한국어로 알려줘."
        )
    elif document_type == "internal_policy" :
        order = (
            f"너는 사내 문서 어시스턴트야. 내가 사내 문서 텍스트를 주면, "
            f"이 중에서 \"{user_input}\"에 해당하는 대답을 "
            f"최대 150자로 요약해서 한국어로 알려줘."
        )
    elif document_type == "product_document" : 
        order = (
            f"너는 제품 어시스턴트야. 내가 제품 관련 텍스트를 주면, "
            f"이 중에서 \"{user_input}\"에 해당하는 대답을 "
            f"최대 150자로 요약해서 한국어로 알려줘."
        )
    elif document_type == "technical_document" : 
        order = (
            f"너는 기술 문서 어시스턴트야. 내가 기술 문서 텍스트를 주면, "
            f"이 중에서 \"{user_input}\"에 해당하는 대답을 "
            f"최대 150자로 요약해서 한국어로 알려줘."
        )
    else : 
        order = (
            f"너는 텍스트가 뭐가 오던 상관없이 이말만 하면 돼. '원하시는 문서 유형을 알아내지 못했습니다...죄송합니다.'"
        )
    system_message = SystemMessagePromptTemplate.from_template(order)
    human_message = HumanMessagePromptTemplate.from_template("{text}")

    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    # 2. LLM 생성
    llm = ChatOpenAI(model="gpt-4o-mini")

    # 3. Prompt와 LLM 결합
    chatbot = chat_prompt | llm

    # 4. 실행
    response = chatbot.invoke({"text": text})
    result = response.content
    print(result)
    state.result = result

    return {
        "result" : state.result
    }

# Define the graph
graph = (
    StateGraph(State)
    # (1) choose_node 분기 노드 등록 (outputs에 리턴 키 명시)
    .add_node("choose_node", choose_node)
    # (2) RAG 실행 노드들 등록
    .add_node("product_node", execute_rag)
    .add_node("proceedings_node", execute_rag)
    .add_node("hr_policy_node", execute_rag)
    .add_node("technical_document_node", execute_rag)
    # (3) summarize_node 등록 (최종 노드)
    .add_node("summarize_node", summarize_node)
    # (4) START → 분기 노드(choose_node) → (next_node 값에 따라) 분기
    .add_edge(START,"choose_node")
    .add_conditional_edges(
        "choose_node",
        choose_one,
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