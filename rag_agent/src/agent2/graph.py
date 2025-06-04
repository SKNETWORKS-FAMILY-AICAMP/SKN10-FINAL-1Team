# graph.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypedDict, Dict # Dict 추가

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

# rag.py에서 RAG 파이프라인 함수들 가져오기
# rag.py 파일이 graph.py와 동일한 디렉토리 또는 Python 경로에 있어야 합니다.
from rag import (
    init_clients,
    embed_query,
    build_context_from_matches,
    generate_answer_with_context
)

# .env 파일 경로 지정 및 로드
# graph.py 파일 위치: SKN10-FINAL-1Team/rag_agent/src/agent/
# .env 파일 위치: SKN10-FINAL-1Team/
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f".env 파일 로드 완료: {dotenv_path}")
else:
    print(f"경고: .env 파일을 찾을 수 없습니다. 경로: {dotenv_path}")
    # 필수 환경 변수가 없다면 여기서 중단하거나, 기본값을 사용하도록 처리할 수 있습니다.
    # 여기서는 아래 main 블록에서 환경 변수 존재 여부를 다시 한번 확인합니다.

class Configuration(TypedDict):
    """Configurable parameters for the agent."""
    my_configurable_param: str # 예시 파라미터, 실제 사용하지 않으면 제거 가능

@dataclass
class State:
    user_input: str
    document_type: str = ""  # LLM이 분류한 문서 타입 (Pinecone 네임스페이스와 일치해야 함)
    result: str = ""         # 최종 RAG 결과 또는 에러 메시지


def choose_document_type_llm(message: str):
    """LLM을 호출하여 사용자 질문에 가장 적합한 문서 유형을 결정합니다."""
    # Pinecone 네임스페이스 이름과 일치하도록 키워드 및 설명 수정
    # 'minutes_document'는 회의록을 위한 예시 네임스페이스입니다.
    # 실제 Pinecone 네임스페이스 상황에 맞게 키워드와 설명을 조정해야 합니다.
    prompt_template_str = (
        "'{input}'은(는) 사용자 질문입니다. "
        "이 질문이 다음 문서 유형 중 어떤 것에 가장 적합한지 판단해주세요: "
        "internal_policy (사내 규정 및 정책 관련), "
        "product_document (제품 설명 및 사양 관련), "
        "technical_document (기술 문서, 문제 해결, 매뉴얼 관련), "
        "minutes_document (회의록 요약 및 검색 관련), "
        "unknown (위 어떤 유형에도 명확히 속하지 않는 일반 질문 또는 분류 불가능한 질문). "
        "당신의 답변은 반드시 다음 키워드 중 정확히 하나여야 합니다: "
        "'internal_policy', 'product_document', 'technical_document', 'minutes_document', 'unknown'. "
        "다른 어떤 추가 텍스트도 포함하지 마십시오."
    )
    prompt = PromptTemplate(
        input_variables=["input"],
        template=prompt_template_str
    )
    # OPENAI_API_KEY 환경 변수가 설정되어 있어야 합니다.
    llm = ChatOpenAI(model="gpt-3.5-turbo") # 또는 선호하는 다른 모델
    chain = prompt | llm
    response = chain.invoke({"input": message})
    print(f"LLM 원본 문서 타입 분류 응답: {response.content}")
    return response

def classify_document_node(state: State) -> Dict[str, str]:
    """LLM을 사용하여 문서 유형을 분류하고, 파싱하여 state.document_type을 업데이트할 딕셔너리를 반환합니다."""
    user_input = state.user_input
    raw_llm_response = choose_document_type_llm(user_input).content.strip().lower()

    # 실제 Pinecone 네임스페이스 및 'unknown' 포함 (minutes_document는 예시)
    # ** 중요: 이 keywords 리스트는 실제 Pinecone 네임스페이스와 일치해야 합니다. **
    # ** 'minutes_document'는 예시이며, 실제 네임스페이스가 없다면 제거하거나 수정해야 합니다. **
    keywords = ["internal_policy", "product_document", "technical_document", "minutes_document", "unknown"]
    parsed_document_type = ""

    for keyword in keywords:
        if f"'{keyword}'" in raw_llm_response or keyword == raw_llm_response:
            parsed_document_type = keyword
            break
    
    if not parsed_document_type: # 키워드가 단순히 포함된 경우
        for keyword in keywords:
            if keyword in raw_llm_response:
                parsed_document_type = keyword
                print(f"단순 포함으로 '{keyword}' 키워드 파싱됨.")
                break
    
    if not parsed_document_type:
        print(f"경고: LLM 응답에서 유효한 문서 타입 키워드를 파싱하지 못했습니다. 원본: '{raw_llm_response}'. 'unknown'으로 처리합니다.")
        parsed_document_type = "unknown"

    print(f"최종 파싱된 문서 타입: '{parsed_document_type}'")
    return {"document_type": parsed_document_type}

def route_after_classification(state: State) -> str:
    """state.document_type에 따라 다음 노드로 라우팅합니다."""
    doc_type = state.document_type
    print(f"라우팅 함수 진입 - state.document_type: '{doc_type}'")

    # 실제 Pinecone 네임스페이스와 'unknown'에 대한 라우팅
    # ** 중요: 이 조건문들은 위 keywords 리스트 및 실제 네임스페이스와 일치해야 합니다. **
    if doc_type == "internal_policy":
        return "internal_policy"
    elif doc_type == "product_document":
        return "product_document"
    elif doc_type == "technical_document":
        return "technical_document"
    elif doc_type == "minutes_document": # 회의록 네임스페이스 예시
        return "minutes_document"
    elif doc_type == "unknown":
        return "unknown"
    else:
        error_message = f"분류 노드로부터 예기치 않은 문서 타입('{doc_type}')을 받았습니다. 'unknown'으로 처리합니다."
        print(error_message)
        # 이 경우, classify_document_node에서 이미 'unknown'으로 처리되었을 가능성이 높음
        return "unknown"


def execute_rag_node(state: State) -> Dict[str, str]:
    """공통 RAG 파이프라인을 실행하고 결과를 반환합니다."""
    print(f"\n📄 RAG 노드 실행: 문서 타입 = '{state.document_type}', 질문 = '{state.user_input}'")
    try:
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

        # rag.py의 generate_answer_with_context 함수에서 max_tokens 조절 가능
        answer = generate_answer_with_context(openai_client, state.user_input, context)
        print(f"   - LLM 답변 생성 완료 (일부): '{answer[:100]}...'") # 전체 답변 대신 일부만 로깅
        return {"result": answer}

    except ValueError as ve:
        print(f"RAG 파이프라인 값 에러: {ve}")
        return {"result": f"정보 처리 중 오류 발생: {ve}"}
    except Exception as e:
        print(f"RAG 파이프라인 예외 발생: {e} (타입: {type(e).__name__})")
        import traceback
        traceback.print_exc() # 디버깅을 위해 전체 traceback 출력
        return {"result": f"예기치 않은 오류가 발생했습니다: {e}"}

def unknown_handler_node(state: State) -> Dict[str, str]:
    """분류할 수 없는 질문이나 'unknown' 타입으로 지정된 경우 호출되는 노드입니다."""
    print(f"\n❓ 'unknown' 핸들러 노드 실행: 질문 = '{state.user_input}'")
    message = "죄송합니다, 질문의 의도를 정확히 파악하거나 질문에 해당하는 적절한 문서를 찾을 수 없었습니다. 다른 방식으로 질문해주시겠어요?"
    return {"result": message}


# 그래프 정의
workflow = StateGraph(State, config_schema=Configuration)

# 노드 추가
workflow.add_node("classify_document_node", classify_document_node)
workflow.add_node("unknown_handler_node", unknown_handler_node)

# RAG 노드: Pinecone 네임스페이스와 일치하는 이름으로 노드 등록 (가독성 향상)
workflow.add_node("internal_policy_rag", execute_rag_node)
workflow.add_node("product_document_rag", execute_rag_node)
workflow.add_node("technical_document_rag", execute_rag_node)
workflow.add_node("minutes_document_rag", execute_rag_node) # 회의록 네임스페이스용 RAG 노드 (예시)


# 그래프 흐름 정의
workflow.add_edge(START, "classify_document_node")

workflow.add_conditional_edges(
    "classify_document_node",
    route_after_classification,
    {
        # route_after_classification에서 반환된 키와 매핑될 노드
        "internal_policy": "internal_policy_rag",
        "product_document": "product_document_rag",
        "technical_document": "technical_document_rag",
        "minutes_document": "minutes_document_rag", # 회의록 네임스페이스 예시
        "unknown": "unknown_handler_node"
    }
)

# 각 RAG 처리 노드 및 unknown 핸들러 노드 실행 후, 그래프를 종료
workflow.add_edge("internal_policy_rag", END)
workflow.add_edge("product_document_rag", END)
workflow.add_edge("technical_document_rag", END)
workflow.add_edge("minutes_document_rag", END) # 회의록 네임스페이스 예시
workflow.add_edge("unknown_handler_node", END)

graph = workflow.compile(name="Interactive RAG Agent Graph")


# --- 로컬 테스트를 위한 실행 코드 ---
if __name__ == "__main__":
    if not all(os.getenv(key) for key in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]):
        print("⚠️  필수 환경 변수(OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT)가 설정되지 않았습니다. .env 파일을 확인하세요.")
    else:
        print("✅ 환경 변수가 로드되었습니다. RAG 에이전트가 준비되었습니다.")
        print("질문을 입력하세요. 종료하려면 'exit' 또는 'quit'를 입력하세요.")

        while True:
            user_question = input("\n당신의 질문: ")
            if user_question.lower() in ["exit", "quit"]:
                print("프로그램을 종료합니다.")
                break
            if not user_question.strip():
                print("질문이 입력되지 않았습니다. 다시 시도해주세요.")
                continue

            inputs = {"user_input": user_question}
            
            print(f"\n--- 질문 처리 중: \"{user_question}\" ---")
            try:
                final_state = graph.invoke(inputs)
                print(f"\n🤖 에이전트 답변:")
                print(final_state.get('result', '결과를 가져올 수 없습니다.'))
                # print(f"   (최종 상태 Debug: document_type='{final_state.get('document_type')}', user_input='{final_state.get('user_input')}')") # 디버깅용

            except Exception as e:
                print(f"   처리 중 오류 발생: {e}")
                # import traceback # 자세한 오류를 보려면 주석 해제
                # traceback.print_exc()
            print("--- 질문 처리 완료 ---")