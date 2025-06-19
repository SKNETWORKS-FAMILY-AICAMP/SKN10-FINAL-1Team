"""LangGraph graph for prediction_agent.

Handles CSV file analysis based on user queries.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, List, Annotated
import io
from dotenv import load_dotenv
import operator
import pandas as pd

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import logging
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- State Definition ---
class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)
    user_query: Optional[str] = None
    csv_file_content: Optional[str] = None
    final_answer: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __post_init__(self):
        if not self.user_query and self.messages:
            user_messages = [msg for msg in self.messages if isinstance(msg, HumanMessage)]
            if user_messages:
                self.user_query = user_messages[-1].content

    def dict(self):
        result = super().dict()
        if self.final_answer and self.messages is not None:
            result["messages"] = self.messages + [AIMessage(content=self.final_answer)]
        return result

# --- Node Function for CSV Analysis ---
def csv_analysis_node(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    print("--- CSV ANALYSIS NODE ---")
    
    user_query = state.get('user_query')
    csv_content = state.get('csv_file_content')

    if not csv_content:
        answer = "분석할 CSV 파일을 먼저 첨부해주세요."
        return {"final_answer": answer}

    if not user_query:
        answer = "CSV 파일에 대해 무엇이 궁금하신가요? 질문을 입력해주세요."
        return {"final_answer": answer}

    try:
        # Create a DataFrame from the CSV string content
        csv_file = io.StringIO(csv_content)
        df = pd.read_csv(csv_file)

        # Initialize the language model
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

        # Create a pandas dataframe agent
        pandas_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
            allow_dangerous_code=True
        )

        # Invoke the agent with the user's query
        response = pandas_agent.invoke({"input": user_query})
        
        final_answer = response.get("output", "죄송합니다, 답변을 생성하지 못했습니다.")

        return {"final_answer": final_answer}

    except Exception as e:
        error_msg = f"❌ CSV 분석 중 오류가 발생했습니다: {e}"
        logger.exception(error_msg)
        return {"final_answer": error_msg, "error_message": error_msg}


# --- Graph Definition ---
workflow = StateGraph(AgentState)
workflow.add_node("csv_analysis_node", csv_analysis_node)
workflow.set_entry_point("csv_analysis_node")
workflow.add_edge("csv_analysis_node", END)
app = workflow.compile()

# --- Main function to expose the graph ---
async def main():
    return app

if __name__ == "__main__":
    async def test_agent():
        graph = await main()
        
        # Test case 1: Analyze CSV file
        inputs = {"messages": [HumanMessage(content="CSV 파일에 대해 무엇이 궁금하신가요?")], "csv_file_content": "name,age\nJohn,25\nAlice,30"}
        async for event in graph.astream(inputs):
            if "csv_analysis_node" in event:
                print(event["csv_analysis_node"]["final_answer"])
        
    asyncio.run(test_agent())
