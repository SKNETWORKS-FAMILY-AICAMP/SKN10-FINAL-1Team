from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi.responses import StreamingResponse
import json
from dotenv import load_dotenv
import subprocess

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

import os
import sys
import asyncio

# Windows-specific: Set the asyncio event loop policy for psycopg
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from contextlib import asynccontextmanager
from collections import ChainMap
from agent.graph import get_graph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

app = FastAPI(
    title="LangGraph Swarm Agent Server",
    description="A server for running a LangGraph swarm agent.",
    version="1.0.0"
)

# --- Security Middleware ---
INTERNAL_SECRET = os.getenv("FASTAPI_INTERNAL_SECRET")

# @app.middleware("http")
# async def verify_internal_secret(request: Request, call_next):
#     # Allow access to docs and openapi.json without the secret header for convenience
#     if request.url.path in ["/docs", "/openapi.json", "/redoc"]:
#         return await call_next(request)

#     # For all other paths, require the secret header
#     secret_header = request.headers.get("X-Internal-Secret")
#     if not INTERNAL_SECRET or secret_header != INTERNAL_SECRET:
#         raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing internal secret key")

#     response = await call_next(request)
#     return response

# --- Pydantic Models for API ---
class UserInput(BaseModel):
    content: str

class RequestBody(BaseModel):
    messages: List[UserInput]
    csv_file_content: Optional[str] = None

class InvocationRequest(BaseModel):
    input: RequestBody
    config: Dict[str, Any] = {}

# --- API Endpoints ---
def make_serializable(data):
    """
    Recursively converts objects into a JSON-serializable format.
    Handles objects with a .dict() method (like Pydantic models) by calling it.
    Also handles ChainMap objects.
    """
    if isinstance(data, ChainMap):
        return dict(data)
    if isinstance(data, (list, tuple)):
        return type(data)(make_serializable(item) for item in data)
    if isinstance(data, dict):
        return {key: make_serializable(value) for key, value in data.items()}
    if hasattr(data, 'dict') and callable(getattr(data, 'dict')):
        return data.dict()
    return data


@app.post("/invoke")
async def invoke_agent(invocation_request: InvocationRequest):
    """
    Invokes the agent swarm with a user request and streams both message and debug events.
    """
    messages = [("user", msg.content) for msg in invocation_request.input.messages]
    config = invocation_request.config.copy()

    # Check for CSV data and add it to the config's metadata
    csv_content = invocation_request.input.csv_file_content
    if csv_content:
        if "metadata" not in config:
            config["metadata"] = {}
        config["metadata"]["csv_file_content"] = csv_content

    async def event_stream():
        # Create the checkpointer and graph for each request to isolate lifecycles.
        async with AsyncPostgresSaver.from_conn_string(os.environ["DB_URI"]) as checkpointer:
            graph = get_graph(checkpointer)
            async for chunk in graph.astream(
                {"messages": messages},
                config=config,
                stream_mode=["messages"]
            ):
                try:
                    serializable_chunk = make_serializable(chunk)
                    yield f"data: {json.dumps(serializable_chunk, ensure_ascii=False)}\n\n"
                except Exception as e:
                    print(f"---[FASTAPI-SERVER-ERROR] Failed to serialize chunk: {e}")
                    error_payload = {
                        "event": "error",
                        "data": {"message": "An unexpected error occurred during stream serialization."}
                    }
                    yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/redeploy-webhook", summary="Redeploy Webhook", description="Triggers a redeployment of the pod.")
async def redeploy(request: Request):
    # Secure the endpoint with the same internal secret
    secret_header = request.headers.get("X-Internal-Secret")
    if not INTERNAL_SECRET or secret_header != INTERNAL_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing internal secret key")

    try:
        # Run the redeploy script in the background without blocking
        subprocess.Popen(["/bin/sh", "./redeploy.sh"])
        return {"message": "Redeployment process initiated."}
    except Exception as e:
        print(f"---[FASTAPI-SERVER-ERROR] Failed to start redeploy script: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate redeployment.")


@app.get("/", summary="Root endpoint", description="Provides a welcome message.")
def read_root():
    return {"message": "Welcome to the LangGraph Agent API"}

# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


