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


# --- WebSocket Endpoint for Prediction Agent ---
@app.websocket("/ws/prediction_agent")
async def websocket_prediction_agent(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection accepted for prediction_agent.")
    try:
        while True:
            data = await websocket.receive_text()
            logging.debug(f"Received data on WebSocket: {data}")
            
            try:
                request_data = json.loads(data)
                user_query = request_data.get("user_query")

                if user_query is None:
                    logging.warning("user_query not found in WebSocket message.")
                    await websocket.send_text(json.dumps({"type": "error", "content": "user_query not found in message"}))
                    continue

                # Construct input for agent4, using HumanMessage as seen in agent4.py's test_agent
                input_for_agent = {"messages": [HumanMessage(content=user_query)]}
                logging.info(f"Invoking prediction_app with input: {input_for_agent}")

                async for event in prediction_app.astream_events(input_for_agent, version="v2"):
                    event_type = event["event"]
                    event_name = event.get("name")
                    event_data = event.get("data", {})
                    
                    logging.debug(f"Agent event: type='{event_type}', name='{event_name}', data='{event_data}'")

                    if event_type == "on_chat_model_stream":
                        chunk = event_data.get("chunk")
                        if hasattr(chunk, 'content') and chunk.content:
                            await websocket.send_text(json.dumps({"type": "stream", "content": chunk.content}))
                    
                    elif event_type == "on_chain_end" and event_name == "LangGraph": # Assuming 'LangGraph' is the name for the top-level graph events
                        output_state = event_data.get("output", {})
                        if isinstance(output_state, dict):
                            final_answer = output_state.get("final_answer")
                            error_message = output_state.get("error_message")

                            if error_message:
                                logging.error(f"Agent error in final state: {error_message}")
                                await websocket.send_text(json.dumps({"type": "error", "content": error_message}))
                            elif final_answer:
                                logging.info(f"Sending final_answer from LangGraph end: {final_answer}")
                                await websocket.send_text(json.dumps({"type": "final_answer", "content": final_answer}))
                            else:
                                logging.info("LangGraph finished, no specific final_answer or error_message in output state.")
                                await websocket.send_text(json.dumps({"type": "info", "content": "Processing complete."}))
                        else:
                            logging.warning(f"Unexpected output format at LangGraph end: {output_state}")
                            await websocket.send_text(json.dumps({"type": "info", "content": "Processing complete with unexpected output format."}))
                        
                        await websocket.send_text(json.dumps({"type": "stream_end"})) # Signal end of messages for this request

                    elif event_type.endswith("_error"): # Catch specific error events like on_chain_error, on_tool_error etc.
                        error_detail = str(event_data.get("error", event_data)) # data could be the error itself or a dict containing it
                        logging.error(f"Agent error event ({event_type}): {error_detail}")
                        await websocket.send_text(json.dumps({"type": "error", "content": f"Agent error ({event_type}): {error_detail}"}))
                        await websocket.send_text(json.dumps({"type": "stream_end"})) # End stream on error

            except json.JSONDecodeError:
                logging.warning("Invalid JSON received on WebSocket.")
                await websocket.send_text(json.dumps({"type": "error", "content": "Invalid JSON received"}))
            except Exception as e: # Catch errors during message processing for one client request
                logging.error(f"Error processing WebSocket message: {e}", exc_info=True)
                await websocket.send_text(json.dumps({"type": "error", "content": f"An internal server error occurred: {str(e)}"}))
                await websocket.send_text(json.dumps({"type": "stream_end"})) # Ensure client knows this stream is done
                
    except WebSocketDisconnect:
        logging.info("Client disconnected from prediction_agent WebSocket.")
    except Exception as e: # Catch errors in the main WebSocket accept/while loop
        logging.error(f"Unhandled error in prediction_agent WebSocket main loop: {e}", exc_info=True)
        if websocket.client_state != WebSocketState.DISCONNECTED:
             try:
                 # Try to inform client if possible
                 await websocket.send_text(json.dumps({"type": "error", "content": "A critical server error occurred, disconnecting."}))
                 await websocket.close(code=1011) # Internal Error code for WebSocket
             except Exception as close_e: # If sending/closing also fails
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
