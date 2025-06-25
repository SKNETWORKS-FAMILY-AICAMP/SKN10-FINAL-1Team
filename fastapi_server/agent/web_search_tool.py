from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from openai import OpenAI
import os

# OpenAI 클라이언트를 초기화합니다.
# 환경 변수에서 OPENAI_API_KEY를 자동으로 사용합니다.
client = OpenAI()

class WebSearchInput(BaseModel):
    query: str = Field(..., description="웹 검색을 위한 검색어입니다.")

def _perform_web_search(query: str) -> str:
    """
    OpenAI 클라이언트의 내장 웹 검색 기능을 사용하여 웹 검색을 수행합니다.
    """
    try:
        # 사용자가 제공한 코드 스니펫을 사용하여 응답을 생성합니다.
        response = client.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search_preview"}],
            input=query
        )
        # response 객체에 output_text 속성이 있다고 가정합니다.
        # 실제 속성 이름은 OpenAI 라이브러리 버전에 따라 다를 수 있습니다.
        return response.output_text
    except Exception as e:
        return f"웹 검색 중 오류가 발생했습니다: {e}"

# LangChain 도구를 생성합니다.
openai_web_search_tool = Tool(
    name="openai_web_search",
    func=_perform_web_search,
    description="최신 이벤트에 대한 질문에 답하거나 인터넷의 정보에 접근하기 위해 OpenAI의 기능을 사용하여 웹 검색을 수행합니다.",
    args_schema=WebSearchInput,
)
