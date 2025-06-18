from crewai import Agent, Task, Crew, Process            # Agent‧Task 직접 import
from crewai.project import CrewBase, agent, task, crew
from .tools import nl2sql                        # NL2SQLTool, CodeInterpreterTool 래퍼
from .tools.visualization_tools import ChartGenerationTool
from pathlib import Path # Add this import
                                                        # (tools/__init__.py에서 객체 생성)
from crewai import Agent, Task, Crew
from crewai.process import Process
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput
import json




@CrewBase
class AnalystCrew:
    def __init__(self, chat_history=None):
        self.chat_history = chat_history
    """데이터 분석·시각화 전담 crew"""

    # --- Updated config paths ---
    _config_path = Path(__file__).parent / "config"
    agents_config = str(_config_path / "agents.yaml")
    tasks_config  = str(_config_path / "tasks.yaml")
    # --- End of updated config paths ---


    def _create_visualization_check_agent(self) -> Agent:
        return Agent(
            role="Visualization Requirement Analyst",
            goal=(
                "Analyze the user's query to determine if a visual representation (e.g., chart, graph, plot) "
                "is explicitly or implicitly requested. Respond with only 'true' if visualization is needed, or 'false' otherwise."
            ),
            backstory=(
                "You are an expert in understanding user intent for data presentation. "
                "Your sole purpose is to decide if a visual is needed based on the query."
            ),
            verbose=True,
            allow_delegation=False
        )

    def require_viz(self, output: TaskOutput) -> bool:
        """Checks if the original user query, extracted from db_query_task's JSON output, requests visualization."""
        original_query = ""
        try:
            db_query_output_str = output.raw_output
            if not db_query_output_str:
                print("Warning: db_query_task output is empty for require_viz. Defaulting to False.")
                return False

            parsed_db_output = json.loads(db_query_output_str)
            original_query = parsed_db_output.get("original_query")

            if not original_query:
                print(f"Warning: 'original_query' not found in db_query_task output: {db_query_output_str}. Defaulting to False.")
                return False
            
            original_query = str(original_query) # Ensure it's a string

        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON from db_query_task output: '{db_query_output_str}'. Defaulting to False.")
            # Attempt to use the raw output as a fallback if it's a simple string and might be the query itself
            # This is a heuristic and might not always be correct.
            if isinstance(db_query_output_str, str) and len(db_query_output_str) < 200: # Arbitrary length to avoid huge SQL results
                 print(f"Info: Attempting to use raw output as query: '{db_query_output_str}'")
                 original_query = db_query_output_str
            else:
                return False # If JSON parsing fails and raw output is not suitable, assume no viz.
        except Exception as e:
            print(f"Error processing db_query_task output in require_viz: {e}. Defaulting to False.")
            return False

        if not original_query.strip():
            print("Warning: Extracted original query is empty for require_viz. Defaulting to False.")
            return False

        viz_check_agent = self._create_visualization_check_agent()
        
        viz_check_task = Task(
            description=f"User query: '{original_query}'. Does this query require a visual representation like a chart or graph? Answer with only the word 'true' or 'false'.",
            expected_output="A single word: 'true' or 'false'.",
            agent=viz_check_agent
        )

        viz_check_crew = Crew(
            agents=[viz_check_agent],
            tasks=[viz_check_task],
            process=Process.sequential,
            verbose=0 # Keep this less verbose to avoid cluttering main logs
        )

        try:
            print(f"Running visualization check for query: '{original_query}'")
            result = viz_check_crew.kickoff(inputs={'query': original_query})
            print(f"Visualization check result: '{result}'")
            if isinstance(result, str):
                return result.strip().lower() == 'true'
            else:
                print(f"Warning: Visualization check returned non-string result: {result}. Defaulting to False.")
                return False
        except Exception as e:
            print(f"Error during visualization check: {e}. Defaulting to False.")
            return False

    # ── agents ────────────────────────────────────────────────────────────────


    @agent
    def db_analyst(self) -> Agent:
        """DB 조회·통계 산출"""
        return Agent(
            config=self.agents_config["db_analyst"],
            tools=[nl2sql],                              # 자연어→SQL 실행 툴
            memory=self.chat_history, # Pass chat history to the agent
            max_iter=8,
            verbose=True
        )

    @agent
    def viz_analyst(self) -> Agent:
        """시각화 담당"""
        chart_tool = ChartGenerationTool()
        return Agent(
            config=self.agents_config["viz_analyst"],
            tools=[chart_tool],                             # 파이썬 코드 실행(그래프 생성), Chart.js 생성
            memory=self.chat_history, # Pass chat history to the agent
            max_iter=6,
            verbose=True
        )

    # ── task ─────────────────────────────────────────────────────────────────
    @task   
    def db_query(self) -> Task:
        """자연어를 SQL로 변환·실행하는 단계"""
        return Task(config=self.tasks_config["db_query"])

    def data_viz(self) -> Task:
        """쿼리 결과를 시각화하는 단계"""
        return ConditionalTask(
            config=self.tasks_config["data_viz"],
            condition=self.require_viz
        )



    # ── crew ─────────────────────────────────────────────────────────────────
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.db_analyst(), self.viz_analyst()],
            tasks=[self.db_query(), self.data_viz()],
            process=Process.sequential,                # 매니저 기반 구조 :contentReference[oaicite:2]{index=2}
            planning=True                                # Planner 사용 :contentReference[oaicite:3]{index=3}
        )
