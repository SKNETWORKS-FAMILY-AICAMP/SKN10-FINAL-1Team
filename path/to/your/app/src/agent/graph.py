"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

load_dotenv()
class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


@dataclass
class State:
    # 초기값을 설정해 주지 않으면 인자가 required가 된다.
    user_input : str # 사용자 입력
    document_type: str = ""   # 문서 형태
    result: str = ""  # 챗봇 결과


def choose_document_type(message):
    prompt = PromptTemplate(
        input_variables=["input"],
        template="'{input}'가 사용자 질문이야. 너가 판단해서 해당 질문이 Product/사내 문서/기술 문서/회의록 중에 어떤 질문에 해당하는지 하나를 선택하고," \
        "Product관련 질문이면 'product', 회의록 관련 질문이면 'proceedings', 사내 문서 관련 질문이면 'hr_policy', 기술 문서 관련 질문이면 'technical_document' 라고 답해줘."
    )

    # LLM 설정 
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # prompt + llm
    chain = prompt | llm

    # 4. 실행
    response = chain.invoke({"input": message})
    print(response.content)
    return response

def choose_document(state: State) -> str:
    user_input = state.user_input
    document_type = choose_document_type(user_input).content.strip().lower()
    state.document_type = document_type

    if document_type == "product" :
        return "product"
    elif document_type == "proceedings" :
        return "proceedings"
    elif document_type == "hr_policy" :
        return "hr_policy"
    elif document_type == "technical_document" :
        return "technical_document"
    else : 
        raise ValueError(f"Invalid document type returned: {document_type}")

def product_node(state: State) -> str:
    pass

def proceedings_node(state: State) -> str:
    pass

def hr_policy_node(state: State) -> str:
    pass

def technical_document_node(state: State) -> str:
    pass


# Define the graph
graph = (
    StateGraph(State, config_schema=Configuration)
    .add_node("product_node", product_node)
    .add_node("proceedings", proceedings_node)
    .add_node("hr_policy", hr_policy_node)
    .add_node("technical_document", technical_document_node)
    .add_conditional_edges(
        START,  # 이전 노드 이름
        choose_document,  # 실행될 함수
        {  # 분기 조건
            "product": "product_node",
            "proceedings": "proceedings",
            "hr_policy": "hr_policy",
            "technical_document": "technical_document"
        }
    )
    .compile(name="New Graph")
)
