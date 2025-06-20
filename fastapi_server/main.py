import os
import sys
import logging
from typing import List, Dict, Any, Optional

import uvicorn
from dotenv import load_dotenv
import logging # Ensure logging is imported
import json # Import json module
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState # Added for WebSocket state checking
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re # Import re for regex operations
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

# --- Custom Class Definitions for Model Loading ---
# These classes must be defined in the main script's namespace for joblib to find them.
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        self.binary_map = {'Yes': 1, 'No': 0}
        self.categorical_cols = []

    def fit(self, X, y=None):
        X = X.copy()
        X['totalcharges'] = pd.to_numeric(X['totalcharges'], errors='coerce')

        binary_cols = ['partner', 'dependents', 'phoneservice', 'paperlessbilling']
        for col in binary_cols:
            X[col] = X[col].map(self.binary_map)

        self.categorical_cols = [
            col for col in X.select_dtypes(include=['object', 'category']).columns
            if not pd.to_numeric(X[col], errors='coerce').notna().all()
        ]

        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        X['totalcharges'] = pd.to_numeric(X['totalcharges'], errors='coerce')
        X['totalcharges'] = X['totalcharges'].fillna(X['tenure'] * X['monthlycharges'])

        X['new_totalservices'] = (X[['phoneservice', 'internetservice', 'onlinesecurity',
                                     'onlinebackup', 'deviceprotection', 'techsupport',
                                     'streamingtv', 'streamingmovies']] == 'Yes').sum(axis=1)

        X['new_avg_charges'] = X['totalcharges'] / (X['tenure'] + 1e-5)
        X['new_increase'] = X['new_avg_charges'] / X['monthlycharges']
        X['new_avg_service_fee'] = X['monthlycharges'] / (X['new_totalservices'] + 1e-5)
        X['charge_increased'] = (X['monthlycharges'] > X['new_avg_charges']).astype(int)
        X['charge_growth_rate'] = (X['monthlycharges'] - X['new_avg_charges']) / (X['new_avg_charges'] + 1e-5)
        X['is_auto_payment'] = X['paymentmethod'].apply(lambda x: int('automatic' in str(x).lower() or 'bank' in str(x).lower()))

        binary_cols = ['partner', 'dependents', 'phoneservice', 'paperlessbilling']
        for col in binary_cols:
            X[col] = X[col].map(self.binary_map)

        for col in self.categorical_cols:
            X[col] = self.label_encoders[col].transform(X[col].astype(str))

        return X.values

class ThresholdWrapper:
    def __init__(self, pipeline, threshold=0.5):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict(self, X):
        probas = self.pipeline.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


# --- Project Root and Environment Loading ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

dotenv_path = os.path.join(project_root, '.env')
if not os.path.exists(dotenv_path):
    dotenv_path = os.path.join(project_root, '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

# --- Agent Imports ---
from fastapi_server.agent.agent3 import app as analysis_app
from fastapi_server.agent.agent4 import app as prediction_app
from fastapi_server.agent.agent_supervisor import supervisor_app # Import the new supervisor
from fastapi_server.app.ml_models import run_customer_ml_model

# --- Pydantic Models for API Requests ---
class AnalysisRequest(BaseModel):
    messages: List[Dict[str, Any]]

class PredictionRequest(BaseModel):
    user_query: Optional[str] = None
    csv_file_content: Optional[str] = None

# --- FastAPI Application Setup ---
app = FastAPI(
    title="SKN10-FINAL-1Team Multi-Agent API",
    description="Provides endpoints for dedicated analysis and prediction agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.get("/", tags=["Health Check"])
async def root():
    """Health check endpoint to confirm the server is running."""
    return {"status": "online", "message": "Multi-Agent API is running"}

@app.get("/api/tables", tags=["Data"])
async def get_tables_via_agent():
    """
    Provides a list of SQL tables by querying the analysis_agent.
    This acts as a proxy for frontends expecting a direct /api/tables endpoint.
    """
    try:
        user_message_for_agent3 = "List all tables in the database. Please respond with only a JSON array of table names, for example: [\"table1\", \"table2\"]"
        messages_for_agent: List[BaseMessage] = [HumanMessage(content=user_message_for_agent3)]
        input_data_for_agent = {"messages": messages_for_agent}
        
        agent_response = await analysis_app.ainvoke(input_data_for_agent)
        logging.info(f"/api/tables: Raw agent_response from agent3: {agent_response}")
        final_answer_from_agent = agent_response.get("final_answer")
        logging.info(f"/api/tables: Extracted final_answer_from_agent: {final_answer_from_agent}")
        
        tables = []
        if isinstance(final_answer_from_agent, str):
            try:
                # Attempt to find a JSON array within the string
                json_array_match = re.search(r'\[(.*?)\]', final_answer_from_agent)
                if json_array_match:
                    json_array_str = json_array_match.group(0) # Get the full match (e.g., "[\"tbl1\", \"tbl2\"]")
                    parsed_json = json.loads(json_array_str)
                    if isinstance(parsed_json, list):
                        tables = [str(table_name) for table_name in parsed_json]
                    else:
                        logging.warning(f"/api/tables: Extracted array string, but not a list after parsing: {json_array_str}")
                else:
                    logging.warning(f"/api/tables: No JSON array found in agent response string: {final_answer_from_agent}")
                    # Fallback for simple comma-separated string if no JSON array detected
                    if ',' in final_answer_from_agent and not final_answer_from_agent.strip().startswith('[') and not final_answer_from_agent.strip().endswith(']'):
                         tables = [name.strip() for name in final_answer_from_agent.split(',')]

            except json.JSONDecodeError:
                logging.error(f"/api/tables: JSONDecodeError while parsing agent response: {final_answer_from_agent}", exc_info=True)
            except Exception as e:
                logging.error(f"/api/tables: Unexpected error during string parsing: {e}", exc_info=True)

        elif isinstance(final_answer_from_agent, list):
            tables = [str(table_name) for table_name in final_answer_from_agent]
        else:
            if final_answer_from_agent is not None:
                logging.warning(f"/api/tables: Agent returned an unexpected type: {type(final_answer_from_agent)}, value: {final_answer_from_agent}")

        return {"tables": tables}
        
    except Exception as e:
        logging.error(f"Error in /api/tables endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error while fetching tables: {str(e)}")

@app.post("/analysis_agent", tags=["Agents"])
async def invoke_analysis_agent(request: AnalysisRequest):
    """Invokes the analysis agent (agent3) for DB queries and general questions."""
    try:
        messages: List[BaseMessage] = []
        for msg in request.messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        
        input_data = {"messages": messages}
        result = await analysis_app.ainvoke(input_data)
        
        return {"response": result.get("final_answer", "No answer generated.")}
    except Exception as e:
        logging.error(f"Error in analysis agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/prediction_agent", tags=["Agents"])
async def invoke_prediction_agent(request: PredictionRequest):
    """Invokes the prediction agent (agent4) for ML model predictions from CSV data."""
    if not request.user_query and not request.csv_file_content:
        raise HTTPException(status_code=400, detail="Either 'user_query' or 'csv_file_content' must be provided.")

    try:
        input_data = {
            "user_query": request.user_query,
            "csv_file_content": request.csv_file_content
        }
        result = await prediction_app.ainvoke(input_data)

        return {"response": result.get("final_answer", "No prediction generated.")}
    except Exception as e:
        logging.error(f"Error in prediction agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# --- WebSocket Endpoint for Supervisor Agent ---
@app.websocket("/ws/supervisor")
async def websocket_supervisor(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection accepted for supervisor.")
    try:
        while True:
            try:
                data = await websocket.receive_text()
                payload = json.loads(data)
                user_query = payload.get("user_query")
                csv_file_content = payload.get("csv_file_content")

                if not user_query and not csv_file_content:
                    continue

                # 1. Run supervisor to decide the route
                supervisor_input = {"user_query": user_query, "csv_file_content": csv_file_content}
                supervisor_result = await supervisor_app.ainvoke(supervisor_input)
                route = supervisor_result.get("route")

                # 2. Based on route, invoke the correct agent
                if route == "prediction":
                    logging.info("Supervisor routing to: prediction_agent")
                    # Inform the client that we are switching agents
                    await websocket.send_json({"event_type": "agent_change", "data": {"agent": "prediction"}})
                    
                    # Prepare input for the prediction agent
                    prediction_input = {
                        "messages": [HumanMessage(content=user_query)] if user_query else [],
                        "user_query": user_query,
                        "csv_file_content": csv_file_content
                    }
                    
                    # Stream the prediction agent's response back to the client
                    response_sent = False
                    async for chunk in prediction_app.astream(prediction_input):
                        if isinstance(chunk, dict) and not response_sent:
                            final_answer = chunk.get("csv_analysis_node", {}).get("final_answer")
                            error_message = chunk.get("csv_analysis_node", {}).get("error_message")

                            if error_message:
                                await websocket.send_json({"event_type": "error", "data": {"error": error_message}})
                                response_sent = True
                            elif final_answer:
                                await websocket.send_json({"event_type": "token", "data": {"token": final_answer}})
                                response_sent = True
                        if response_sent:
                            break # Stop after getting the first valid response

                elif route == "conversation":
                    # Placeholder for a general conversational agent
                    logging.info("Supervisor routing to: conversation (placeholder)")
                    await websocket.send_json({"event_type": "token", "data": {"token": "This is a placeholder for the conversational agent."}})

                else:
                    await websocket.send_json({"event_type": "error", "data": {"error": f"Unknown route '{route}' decided by supervisor."}})

                # 3. Send done signal
                await websocket.send_json({"event_type": "done"})

            except json.JSONDecodeError:
                logging.warning("Invalid JSON received on supervisor WebSocket.")
                await websocket.send_json({"event_type": "error", "data": {"error": "Invalid JSON format."}})
            except Exception as e:
                logging.error(f"Error in supervisor WebSocket: {e}", exc_info=True)
                await websocket.send_json({"event_type": "error", "data": {"error": str(e)}})

    except WebSocketDisconnect:
        logging.info("Client disconnected from supervisor WebSocket.")
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()


# --- WebSocket Endpoint for Prediction Agent (Direct Connection) ---
@app.websocket("/ws/prediction_agent")
async def websocket_prediction_agent(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection accepted for prediction_agent.")
    try:
        while True:
            data = await websocket.receive_text()
            logging.debug(f"Received data on WebSocket: {data}")
            
            try:
                payload = json.loads(data)
                user_query = payload.get("user_query")
                csv_file_content = payload.get("csv_file_content")
                
                if not user_query and not csv_file_content:
                    logging.warning("WebSocket received empty query and no file.")
                    continue

                input_data_for_agent = {
                    "messages": [HumanMessage(content=user_query)] if user_query else [],
                    "user_query": user_query,
                    "csv_file_content": csv_file_content
                }
                
                logging.info(f"Invoking prediction_app (LangGraph) with query: '{user_query}' and file size: {len(csv_file_content) if csv_file_content else 0}")
                
                response_sent = False
                async for current_state in prediction_app.astream(
                    input_data_for_agent,
                    config={"configurable": {"thread_id": "prediction-thread"}}
                ):
                    logging.debug(f"Agent stream: current_state: {current_state}")
                    if isinstance(current_state, dict) and not response_sent:
                        final_answer = None
                        error_message = None
                        
                        if "csv_analysis_node" in current_state:
                            node_output = current_state.get("csv_analysis_node")
                            if isinstance(node_output, dict):
                                final_answer = node_output.get("final_answer")
                                error_message = node_output.get("error_message")

                        if error_message:
                            logging.error(f"Agent error from stream: {error_message}")
                            await websocket.send_json({"event_type": "error", "data": {"error": error_message}})
                            response_sent = True
                        elif final_answer:
                            logging.info(f"Sending final_answer from stream: {final_answer}")
                            await websocket.send_json({"event_type": "token", "data": {"token": final_answer}})
                            response_sent = True

                    if response_sent:
                        break
                
                logging.info("Sending 'done' signal to client.")
                await websocket.send_json({"event_type": "done"})

            except json.JSONDecodeError:
                logging.warning("Invalid JSON received on WebSocket.")
                await websocket.send_json({"event_type": "error", "data": {"error": "Invalid JSON format received."}})
                await websocket.send_json({"event_type": "done"})
            except Exception as e:
                logging.error(f"Error processing WebSocket message: {e}", exc_info=True)
                await websocket.send_json({"event_type": "error", "data": {"error": f"An internal server error occurred: {str(e)}"}})
                await websocket.send_json({"event_type": "done"})
                
    except WebSocketDisconnect:
        logging.info("Client disconnected from prediction_agent WebSocket.")
    except Exception as e:
        logging.error(f"Unhandled error in prediction_agent WebSocket main loop: {e}", exc_info=True)
        if websocket.client_state != WebSocketState.DISCONNECTED:
             try:
                 await websocket.send_json({"event_type": "error", "data": {"error": "A critical server error occurred, disconnecting."}})
                 await websocket.close(code=1011)
             except Exception as close_e:
                 logging.error(f"Error trying to inform client or close WebSocket: {close_e}")
    finally:
        logging.info("Ensuring prediction_agent WebSocket connection is closed after loop/exception.")
        if websocket and websocket.client_state != WebSocketState.DISCONNECTED:
             try:
                 await websocket.close()
             except Exception as final_close_e:
                 logging.error(f"Error during final WebSocket close in finally block: {final_close_e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
