from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task # Updated imports
from .tools.prediction_tools import ChurnPredictionTool

# Note: CREW_DIR is not needed when using CrewBase as paths are relative
# from pathlib import Path
# CREW_DIR = Path(__file__).parent.resolve()

@CrewBase
class PredictCrew():
    """PredictCrew uses CrewBase to define agents, tasks, and the crew for churn prediction."""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, chat_history=None):
        self.chat_history = chat_history
        # CrewBase will handle loading of agents and tasks from YAML.
        # We need to instantiate tools if they are not managed by CrewBase or need specific setup.
        self.churn_tool = ChurnPredictionTool()

    @agent
    def churn_predictor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['churn_predictor_agent'], # type: ignore[index]
            tools=[self.churn_tool], # Pass the instantiated tool
            memory=self.chat_history, # Pass chat history to the agent
            verbose=True,
            allow_delegation=False
        )

    @task
    def predict_churn_task(self) -> Task:
        return Task(
            config=self.tasks_config['predict_churn_task'], # type: ignore[index]
            agent=self.churn_predictor_agent()
            # verbose=True # Verbosity can be set at crew level or agent level
        )

    @crew
    def crew(self) -> Crew:
        """Creates the churn prediction crew"""
        return Crew(
            agents=self.agents, # Uses agents defined with @agent decorator
            tasks=self.tasks,    # Uses tasks defined with @task decorator
            process=Process.sequential,
            verbose=True
        )

# Example usage (optional, for direct testing of this crew)
if __name__ == "__main__":
    print("Initializing PredictCrew (CrewBase version) for a test run...")
    
    # Sample CSV data string (replace with actual data for testing)
    # Ensure the columns match expected_features in ChurnPredictionTool
    sample_csv_data = (
        "customer_id,gender,senior_citizen,partner,dependents,tenure,phone_service,multiple_lines,internet_service,online_security,online_backup,device_protection,tech_support,streaming_tv,streaming_movies,contract,paperless_billing,payment_method,monthly_charges,total_charges\n"
        "CUST001,Male,0,Yes,No,12,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,50.00,600.00\n"
        "CUST002,Female,1,No,No,1,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Mailed check,70.15,70.15"
    )
    
    inputs = {'csv_data': sample_csv_data}
    print(f"\nRunning PredictCrew with sample CSV data:\n{sample_csv_data}\n")
    
    try:
        # Instantiate the crew (CrewBase handles the setup)
        predict_crew_instance = PredictCrew()
        # Kick off the crew
        prediction_result = predict_crew_instance.crew().kickoff(inputs=inputs)
        
        print("\nPrediction Result:")
        print(prediction_result)
    except Exception as e:
        print(f"An error occurred during the test run: {e}")
        import traceback
        traceback.print_exc()
