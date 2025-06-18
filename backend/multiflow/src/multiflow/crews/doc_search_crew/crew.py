from pathlib import Path # Add this import
from crewai import Crew, Process, Task, Agent
from crewai.project import CrewBase, agent, task, crew
from .tools.pinecone_tools import (
    InternalPolicySearchTool, TechDocSearchTool, ProductDocSearchTool, ProceedingsSearchTool
)
@CrewBase
class DocSearchCrew:
    def __init__(self, chat_history=None):
        self.chat_history = chat_history
    """사내 문서 검색 전용 Crew"""

    _crew_dir = Path(__file__).parent # Get the directory of the current file (crew.py)
    agents_config = str(_crew_dir / "config" / "agents.yaml")
    tasks_config  = str(_crew_dir / "config" / "tasks.yaml")

    @agent
    def doc_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config["doc_retriever"],  # type: ignore[index]
            verbose=True,
            memory=self.chat_history, # Pass chat history to the agent
            tools=[
                InternalPolicySearchTool(),
                TechDocSearchTool(),
                ProductDocSearchTool(),
                ProceedingsSearchTool(),
            ]
        )

    @task
    def search_task(self) -> Task:
        return Task(config=self.tasks_config["search_task"])  # type: ignore[index]

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.doc_retriever()],
            tasks=[self.search_task()],
            process=Process.sequential,   # 순차 실행
            verbose=True
            # memory=True # Memory is now handled at the agent level
        )
