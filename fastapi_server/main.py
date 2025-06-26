import logging
import asyncio
import sys
import os
import json
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import Runnable
from langchain_core.messages import messages_to_dict, HumanMessage, AIMessage
from pydantic import BaseModel

# Add project root to Python path to allow sibling imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi_server.agent.graph import get_graph

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models for API ---    
class Message(BaseModel):
    role: str
    content: str

class Input(BaseModel):
    messages: list[Message]
    csv_file_content: str | None = None

class Config(BaseModel):
    configurable: dict

class InvokeRequest(BaseModel):
    input: Input
    config: Config

# --- FastAPI App Initialization ---
if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(title="LangGraph-FastAPI-Postgres", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper for Server-Sent Events (SSE) ---
def dump_to_sse(data: dict) -> str:
    """
    Formats a dictionary to an SSE-compliant string.
    It finds a 'messages' list anywhere in the nested dictionary,
    serializes it, and sends it in a standardized top-level format.
    """
    
    def find_and_serialize_messages(d):
        """Recursively search for a 'messages' list and serialize it."""
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "messages" and isinstance(v, list):
                    serialised_messages = []
                    for msg in v:
                        role = "unknown"
                        if isinstance(msg, HumanMessage):
                            role = "user"
                        elif isinstance(msg, AIMessage):
                            role = "assistant"
                        else:
                            role = getattr(msg, 'type', 'unknown')
                        
                        content = getattr(msg, 'content', '')
                        serialised_messages.append({"role": role, "content": str(content)})
                    return serialised_messages
                
                if isinstance(v, dict):
                    found = find_and_serialize_messages(v)
                    if found is not None:
                        return found
        return None

    serialised_list = find_and_serialize_messages(data)
    
    # If messages were found, wrap them in the standardized structure the frontend expects.
    if serialised_list:
        output_data = {"messages": serialised_list}
        try:
            json_data = json.dumps(output_data, ensure_ascii=False)
            return f"data: {json_data}\n\n"
        except (TypeError, json.JSONDecodeError) as e:
            print(f"Error serializing data to JSON: {e}", file=sys.stderr)
            return ""

    # If no 'messages' key is found, don't send an empty event.
    return ""

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "FastAPI server is running"}

@app.post("/{agent_name}/invoke")
async def stream_agent(
    request: Request,
    agent_name: str,
    body: InvokeRequest,
) -> StreamingResponse:


    logger.info(f"---[ FastAPI Server START: /{agent_name}/invoke ]---")

    config_dict = body.config.model_dump()
    thread_id = config_dict.get("configurable", {}).get("thread_id", "N/A")
    logger.info(f"[FastAPI] Using thread_id/session_id: {thread_id}")

    graph = get_graph(agent_name)
    logger.info(f"[FastAPI] Returning pre-compiled graph for: {agent_name}")

    input_dict = body.input.model_dump()
    if body.input.csv_file_content:
        csv_size = len(body.input.csv_file_content)
        logger.info(f"[FastAPI] Added CSV content (size: {csv_size} bytes) to graph input.")

    async def event_stream():
        final_payload = None
        try:
            async for chunk in graph.astream(
                input_dict, config={"configurable": {"thread_id": thread_id}}
            ):
                logger.info(f"RAW Graph chunk: {chunk}")
                # Create a new dictionary to avoid mutating the stream's internal state.
                # Filter out the large CSV content to avoid sending it to the client.
                chunk_to_send = {
                    key: value
                    for key, value in chunk.items()
                    if key != "csv_file_content"
                }

                # The dump_to_sse function will handle filtering and formatting.
                # It returns an empty string for chunks without messages, so we can yield it directly.
                yield dump_to_sse(chunk_to_send)
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            error_payload = {"event": "error", "data": str(e)}
            yield dump_to_sse(error_payload)
        finally:
            logger.info(f"---[ FastAPI Server END: /{agent_name}/invoke ]---")
            if final_payload:
                logger.info(f"Last payload: {final_payload}")

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# --- Main Entry Point for direct execution ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
