#!/usr/bin/env python
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

from random import randint

from pydantic import BaseModel, Field

import uuid # For generating session IDs
from crewai import Agent, Task, Crew,LLM # New imports
from crewai.flow import Flow, listen, router, start,or_
from crewai.process import Process # New import for Crew process
from datetime import datetime # Add this
from multiflow.crews.doc_search_crew.crew import DocSearchCrew # Add this for the imported crew
from multiflow.crews.analyst_crew.crew import AnalystCrew # Assuming this is the class name
from multiflow.crews.predict_crew.predict_crew import PredictCrew # For churn prediction
from multiflow.crews.predict_crew.tools.prediction_tools import ThresholdWrapper # Add this to solve the joblib loading issue
from multiflow.crews.predict_crew.tools.prediction_tools import Preprocessor # Added for Preprocessor deserialization
import traceback # Added for detailed exception logging
import json # For parsing execution plan
from multiflow.services.chat_history_service import DjangoChatHistory # Import our chat history service

# New State for Intent Analysis
class IntentState(BaseModel):
    query: str = ""
    intent: str = "" # Initial broad intent, e.g., 'document_search', 'data_analysis', 'churn_prediction', 'db_query_and_predict'
    execution_plan: list = Field(default_factory=list) # To store the sequence of crews and their inputs
    current_step_index: int = 0 # To track progress through the execution_plan
    step_outputs: dict = Field(default_factory=dict) # To store outputs from each step, keyed by output_id
    retrieved_customer_data: str = "" # For storing data fetched by AnalystCrew (may be deprecated later)
    session_id: uuid.UUID = None # To store the current chat session ID for history

llm = LLM(
    model="gpt-4o",
    stream=True
    
)
# New Flow for Intent Analysis
class IntentAnalysisFlow(Flow[IntentState]):

    @start()
    def get_user_query(self):
        print("Intent Analysis Flow: Starting")
        # For now, let's use a sample query. 
        # To test churn prediction, you'd change sample_query to a CSV string.
        # Example for testing churn (ensure columns match ChurnPredictionTool.expected_features):
        # sample_query for churn prediction with direct CSV
        # sample_query = (
        #     "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
        #     "CUST001,Male,0,Yes,No,12,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,50.00,600.00\n"
        #     "CUST002,Female,1,No,No,1,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Mailed check,70.15,70.15"
        # )
        # sample_query = "고객의 성비를 시각화해줘."
        # sample_query = "What is the meaning of life?"
        # sample_query = "Analyze the sales data and show me a trend chart."
        sample_query = "총지출이 가장높은 고객의 모든 정보로 churn 예측해줘."
        print(f"Analyzing query: \"{sample_query}\"")
        self.state.query = sample_query

        # Initialize session_id for chat history
        # In a real application, this ID would likely be passed in or retrieved from an existing session.
        # For this example, we generate a new one. Ensure a ChatSession with this ID exists in Django.
        if not self.state.session_id:
            self.state.session_id = uuid.uuid4()
            print(f"Initialized session_id for chat history: {self.state.session_id}")
            # TODO: Ensure a ChatSession object with this ID (and associated user) exists in the database.
            # This might involve a call like:
            # from conversations.models import ChatSession, User # Assuming User model is accessible
            # from asgiref.sync import sync_to_async
            # try:
            #     user = await sync_to_async(User.objects.first)() # Get a default user or actual logged-in user
            #     if user:
            #         await sync_to_async(ChatSession.objects.get_or_create)(
            #             id=self.state.session_id, 
            #             defaults={'user': user, 'title': f"Chat Session {self.state.session_id}"}
            #         )
            #     else:
            #         print("WARNING: No user found to associate with ChatSession.")
            # except Exception as e:
            #     print(f"WARNING: Could not create ChatSession for {self.state.session_id}: {e}")

    @listen(get_user_query)
    def analyze_intent(self):
        print("Analyzing user intent...")

        intent_analyzer_agent = Agent(
            role="User Intent Analyzer",
            goal="Accurately classify the user's query into 'document_search', 'data_analysis', 'churn_prediction', or 'db_query_and_predict'.",
            backstory=(
                "You are an AI assistant responsible for understanding what the user wants to do. "
                "Your primary function is to categorize requests to route them to the correct specialized team."
            ),
            verbose=True,
            allow_delegation=False,
            # llm=... # You can specify an LLM here if needed, otherwise it uses the default
        )

        intent_analysis_task = Task(
            description=(
                "Analyze the following user query: '{query}'. "
                "Determine if the user's primary goal is to perform a 'document_search' "
                "(e.g., find information in documents, look up facts, ask questions about a knowledge base), "
                "'data_analysis' (e.g., analyze datasets, create charts, get insights from structured data, perform calculations on data), "
                "'churn_prediction' (e.g., predict customer churn using provided data, 'will this customer leave?', 'run churn model on this data'), "
                "or 'db_query_and_predict' (e.g., 'fetch customer data and predict churn', 'get user details from DB then run prediction'). "
                "Respond with ONLY 'document_search', 'data_analysis', 'churn_prediction', or 'db_query_and_predict'."
            ),
            expected_output="A string, either 'document_search', 'data_analysis', or 'churn_prediction'.",
            agent=intent_analyzer_agent,
        )

        crew = Crew(
            agents=[intent_analyzer_agent],
            tasks=[intent_analysis_task],
            process=Process.sequential, # Explicitly define the process
            verbose=True
        )

        result = crew.kickoff(inputs={'query': self.state.query})
        print(f"Intent analysis result: {result}")
        # Ensure result is a string before stripping
        if isinstance(result, str):
            self.state.intent = result.strip().lower()
        elif hasattr(result, 'raw') and isinstance(result.raw, str): # Common for CrewAI task outputs
             self.state.intent = result.raw.strip().lower()
        else:
            self.state.intent = str(result).strip().lower() # Fallback

    @listen(analyze_intent) # This new method listens to the completion of analyze_intent
    def create_execution_plan(self):
        print("Creating execution plan...")

        orchestrator_agent = Agent(
            role="Crew Orchestrator",
            goal=(
                "Based on the user's query and initial intent, create a step-by-step execution plan. "
                "This plan will define which specialized crew(s) to call, in what order, "
                "and what inputs each crew needs. Ensure inputs can be sourced from the original query "
                "or outputs of previous steps."
            ),
            backstory=(
                "You are a master planner for a team of AI assistant crews. You analyze user requests "
                "and break them down into manageable tasks for specialized crews. Your plans must be clear, "
                "logical, and ensure seamless data flow between crew operations. "
                "'DocSearchCrew' is specialized in finding information within internal company documents (e.g., policies, product docs, technical articles, meeting minutes). "
                "'AnalystCrew' handles querying structured data like customer information or news using natural language to SQL (NL2SQL) and can also generate visualizations of this data. "
                "Available crews for the 'crew' field in the plan are: 'DocSearchCrew', 'AnalystCrew', 'PredictCrew'. "
                "Use these exact names. "
                "The plan should be a JSON list of dictionaries, where each dictionary represents a step. "
                "Each step must have 'crew' (string name), 'inputs' (dict), and 'output_id' (string for referencing output). "
                "For inputs, use placeholders like '{original_query}' for the initial user query, "
                "or '{name_of_previous_output_id}' to use the output from a previous step."
            ),
            verbose=True,
            allow_delegation=False,
            llm=llm,
            # llm=... # Specify if needed, otherwise uses default from environment
        )

        # Task for the orchestrator agent to create the plan
        plan_creation_task_definition = Task(
            description=(
                "User query: '{query}'.\n"
                "Initial intent (if available): '{intent}'.\n"
                "Create an execution plan as a JSON list of steps. Each step is a dictionary with 'crew', 'inputs', 'output_id'.\n"
                "Available crew names for the 'crew' field: 'DocSearchCrew', 'AnalystCrew', 'PredictCrew'.\n"
                "\nExample for a single 'DocSearchCrew' task (e.g., user asks 'Find the latest policy on remote work'):\n"
                "[\n"
                "  {{\n"
                "    \"crew\": \"DocSearchCrew\",\n"
                "    \"inputs\": {{ \"query\": \"{{original_query}}\" }},\n"
                "    \"output_id\": \"found_policy_document\"\n"
                "  }}\n"
                "]\n"
                "\nExample for 'AnalystCrew' to fetch and visualize data (e.g., 'Show me a chart of sales per region'):\n"
                "[\n"
                "  {{\n"
                "    \"crew\": \"AnalystCrew\",\n"
                "    \"inputs\": {{ \"query\": \"{{original_query}}\" }},\n"
                "    \"output_id\": \"sales_visualization\"\n"
                "  }}\n"
                "]\n"
                "\nExample for 'DocSearchCrew' then 'AnalystCrew' (e.g., 'Find product specs and summarize them'):\n"
                "[\n"
                "  {{\n"
                "    \"crew\": \"DocSearchCrew\",\n"
                "    \"inputs\": {{ \"query\": \"Find specifications for Product X\" }},\n"
                "    \"output_id\": \"product_x_specs\"\n"
                "  }},\n"
                "  {{\n"
                "    \"crew\": \"AnalystCrew\",\n"
                "    \"inputs\": {{ \"query\": \"Summarize these specifications: {{product_x_specs}}\" }},\n"
                "    \"output_id\": \"specs_summary\"\n"
                "  }}\n"
                "]\n"
                "\nFor a task requiring database query then churn prediction (e.g., initial intent 'db_query_and_predict', user query 'Get customer details for ID 123 and predict churn'):\n"
                "The plan should first use 'AnalystCrew' to fetch data. Its 'inputs' should be like {{'query': 'Retrieve customer data for ID 123 suitable for churn prediction'}}. Its 'output_id' could be 'customer_data_csv'.\n"
                "Then, 'PredictCrew' takes 'inputs' like {{'csv_data': '{{customer_data_csv}}'}} and 'output_id' like 'churn_forecast'.\n"
                "Ensure all placeholders like {{original_query}} or {{previous_output_id}} are correctly used.\n"
            ),
            expected_output=(
                "A JSON string representing the list of execution steps. "
                "Each step must contain 'crew', 'inputs', and 'output_id'."
            ),
            agent=orchestrator_agent
        )

        # Format the description with actual query and intent values for this specific execution
        # This ensures the agent has the necessary context for its current task execution.
        formatted_description = plan_creation_task_definition.description.format(
            query=self.state.query, 
            intent=self.state.intent if self.state.intent else "Not specified"
        )
        
        # Create a new Task instance with the dynamically formatted description for execution
        # This is often how context is injected if not using a Crew's input mechanism directly.
        executable_task = Task(
            description=formatted_description,
            expected_output=plan_creation_task_definition.expected_output,
            agent=orchestrator_agent
        )

        # Create a temporary crew to execute the planning task
        planning_crew = Crew(
            agents=[orchestrator_agent],
            tasks=[executable_task],
            process=Process.sequential,
            verbose=True
        )

        plan_json_str_output = planning_crew.kickoff()
        
        print(f"Orchestrator Agent proposed plan (raw output): {plan_json_str_output}")
        
        processed_plan_str = ""
        if hasattr(plan_json_str_output, 'raw_output') and isinstance(plan_json_str_output.raw_output, str):
            processed_plan_str = plan_json_str_output.raw_output
        elif isinstance(plan_json_str_output, str):
            processed_plan_str = plan_json_str_output
        else:
            processed_plan_str = str(plan_json_str_output) # Fallback for other TaskOutput types
            print(f"Warning: Plan output was not a string or raw_output string, converted to: {processed_plan_str}")

        try:
            self.state.execution_plan = json.loads(processed_plan_str)
            if not isinstance(self.state.execution_plan, list):
                print(f"Execution plan is not a list, but: {type(self.state.execution_plan)}. Plan: {self.state.execution_plan}")
                raise ValueError("Execution plan is not a list after parsing.")
            for i, step in enumerate(self.state.execution_plan):
                if not isinstance(step, dict):
                    print(f"Execution plan step {i} is not a dictionary: {step}")
                    raise ValueError(f"Execution plan step {i} is not a dictionary.")
                if not all(k in step for k in ("crew", "inputs", "output_id")):
                    print(f"Execution plan step {i} ('{step.get('crew', 'UnknownCrew')}') is missing required keys (crew, inputs, output_id): {step}")
                    raise ValueError(f"Execution plan step {i} is missing required keys (crew, inputs, output_id).")
            print(f"Successfully parsed and validated execution plan: {self.state.execution_plan}")
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError parsing execution plan: {e}. Plan string was: '{processed_plan_str}'")
            self.state.execution_plan = [] 
        except ValueError as e:
            print(f"ValueError validating execution plan structure: {e}. Parsed plan (if any): {getattr(self.state, 'execution_plan', 'Not Set')}")
            self.state.execution_plan = [] 
        except Exception as e:
            print(f"Unexpected error parsing/validating execution plan: {type(e).__name__} - {e}. Plan string was: '{processed_plan_str}'")
            self.state.execution_plan = []

        self.state.current_step_index = 0
        self.state.step_outputs = {}



    @router(create_execution_plan) # Listens to plan creation and each step completion
    def route_first_step(self):
        print(f"Routing first step. Current index: {self.state.current_step_index}, Plan length: {len(self.state.execution_plan)}")
        if not self.state.execution_plan:
            print("Execution plan is empty. Finalizing flow.")
            return "finish_plan" # Or an error handling state

        if self.state.current_step_index < len(self.state.execution_plan):
            print(f"Next step in plan: {self.state.execution_plan[self.state.current_step_index]['crew']}")
            return "run_plan"
        else:
            print("Execution plan complete. Finalizing flow.")
            return "finish_plan"

    @listen("run_plan")
    def run_planned_crew(self):
        current_step_info = self.state.execution_plan[self.state.current_step_index]
        crew_name = current_step_info['crew']
        inputs_template = current_step_info['inputs']
        output_id = current_step_info['output_id']

        print(f"Executing step {self.state.current_step_index + 1}: Crew '{crew_name}' with output ID '{output_id}'")

        # Resolve inputs
        resolved_inputs = {}
        for key, value_template in inputs_template.items():
            if isinstance(value_template, str):
                if value_template == "{original_query}":
                    resolved_inputs[key] = self.state.query
                elif value_template.startswith("{") and value_template.endswith("}"):
                    placeholder = value_template[1:-1]
                    if placeholder in self.state.step_outputs:
                        resolved_inputs[key] = self.state.step_outputs[placeholder]
                    else:
                        print(f"Error: Placeholder '{placeholder}' not found in step_outputs. Available: {list(self.state.step_outputs.keys())}")
                        resolved_inputs[key] = None # Or handle error appropriately
                else:
                    resolved_inputs[key] = value_template # Literal value
            else:
                resolved_inputs[key] = value_template # Non-string literal value
        
        print(f"  Resolved inputs for {crew_name}: {resolved_inputs}")

        # Instantiate chat history service for the current session
        chat_history_service = None
        if self.state.session_id:
            # Ensure the DjangoChatHistory is instantiated correctly. 
            # It requires the session_id for its operations.
            chat_history_service = DjangoChatHistory(session_id=self.state.session_id)
            print(f"  Instantiated DjangoChatHistory for session: {self.state.session_id}")
        else:
            print("  Warning: session_id not found in state. Chat history will not be enabled for this crew.")

        crew_instance = None
        try:
            if crew_name == "DocSearchCrew":
                crew_instance = DocSearchCrew(chat_history=chat_history_service)
                print(f"  DocSearchCrew instantiated {'with' if chat_history_service else 'without'} chat history.")
            elif crew_name == "AnalystCrew":
                crew_instance = AnalystCrew(chat_history=chat_history_service)
                print(f"  AnalystCrew instantiated {'with' if chat_history_service else 'without'} chat history.")
            elif crew_name == "PredictCrew":
                crew_instance = PredictCrew(chat_history=chat_history_service)
                print(f"  PredictCrew instantiated {'with' if chat_history_service else 'without'} chat history.")
            else:
                print(f"Error: Unknown crew name '{crew_name}' in execution plan.")
                self.state.step_outputs[output_id] = f"Error: Unknown crew name '{crew_name}'"
                self.state.current_step_index += 1
                return # Skip to next routing decision

            if crew_instance:
                result = crew_instance.crew().kickoff(inputs=resolved_inputs)
                print(f"  {crew_name} Result (raw): {str(result)[:250]}...")
                
                if hasattr(result, 'raw') and result.raw is not None:
                    self.state.step_outputs[output_id] = result.raw
                elif hasattr(result, 'result') and result.result is not None:
                    self.state.step_outputs[output_id] = result.result
                elif isinstance(result, str):
                    self.state.step_outputs[output_id] = result
                elif result is not None: 
                    self.state.step_outputs[output_id] = str(result)
                else:
                    self.state.step_outputs[output_id] = "No output produced or output was None."
                print(f"  Stored output for '{output_id}': {str(self.state.step_outputs[output_id])[:100]}...")

        except Exception as e:
            print(f"Error running {crew_name}: {type(e).__name__} - {e!r}")
            print("Traceback:")
            traceback.print_exc()
            self.state.step_outputs[output_id] = f"Error during {crew_name} execution: {type(e).__name__} - {e!r}"
        
        self.state.current_step_index += 1

    @listen("finish_plan")
    def finish(self):
        print("IntentAnalysisFlow: Execution complete.")
        if self.state.step_outputs:
            print("Final outputs from all steps:")
            for step_id, output in self.state.step_outputs.items():
                print(f"  Output ID '{step_id}': {str(output)[:200]}...") # Truncate long outputs
        else:
            print("No outputs were generated by the execution plan, or plan was empty.")

    @router(run_planned_crew) # Listens to plan creation and each step completion
    def route_next_step(self):
        print(f"Routing next step. Current index: {self.state.current_step_index}, Plan length: {len(self.state.execution_plan)}")
        if not self.state.execution_plan:
            print("Execution plan is empty. Finalizing flow.")
            return "finish_plan" # Or an error handling state

        if self.state.current_step_index < len(self.state.execution_plan):
            print(f"Next step in plan: {self.state.execution_plan[self.state.current_step_index]['crew']}")
            return "run_plan"
        else:
            print("Execution plan complete. Finalizing flow.")
            return "finish_plan"

# Function to run this flow
def run_intent_analysis():
    intent_flow = IntentAnalysisFlow()
    intent_flow.plot("intent_analysis_flow.png") 
    # Plotting might be less relevant now or needs to be adapted if the flow structure is very dynamic
    
    try:
        intent_flow.kickoff()
    except Exception as e:
        print(f"An error occurred during flow kickoff or execution: {type(e).__name__} - {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("--- multiflow/main.py execution started ---")
    run_intent_analysis()
    print("--- multiflow/main.py execution finished ---")
