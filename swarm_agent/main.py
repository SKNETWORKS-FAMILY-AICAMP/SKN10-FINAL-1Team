import asyncio
import json
from typing import AsyncIterator, Dict, Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint

from agent.graph import graph

# --- FastAPI App Initialization ---
app = FastAPI(
    title="LangGraph Swarm Agent Server",
    version="1.0",
    description="A server for running the LangGraph swarm agent with real-time tool updates.",
)

# --- CORS Middleware --- 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"]
,    allow_headers=["*"],
)

# --- Custom Streaming Endpoint with Tool Update Logic ---

async def stream_events(
    thread_id: str, 
    input_data: Dict[str, Any]
) -> AsyncIterator[str]:
    """
    A custom endpoint to stream agent events, processing tool calls in real-time.
    This function implements the logic described in the user's memories to provide
    real-time updates on tool usage to the frontend.
    """
    current_tool_state = {"assistant": {"messages": []}}
    active_tool_calls: Dict[str, Dict[str, Any]] = {}

    config = RunnableConfig(
        configurable={"thread_id": thread_id},
        stream_mode="messages",
        output_keys=["messages"],
    )

    try:
        async for event in graph.astream(input=input_data, config=config):
            # Forward the original LangGraph event
            yield f"data: {json.dumps(event)}\n\n"

            # --- Real-time Tool Update Processing ---
            # The logic below is based on the user's provided memories.
            if isinstance(event, list) and len(event) > 1 and event[0] == "messages":
                message = event[1]
                msg_type = message.get("type")

                if msg_type == "ai" and message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        if tool_call["id"] not in active_tool_calls:
                            active_tool_calls[tool_call["id"]] = tool_call
                            current_tool_state["assistant"]["messages"].append(message)
                            yield f'data: {json.dumps({"event": "tool_update", "data": current_tool_state})}\n\n'
                
                elif msg_type == "ai" and message.get("tool_call_chunks"):
                    for chunk in message["tool_call_chunks"]:
                        if chunk["id"] in active_tool_calls:
                            active_tool_calls[chunk["id"]]["args"] += chunk["args"]
                            # Find and update the message in the state
                            for msg in current_tool_state["assistant"]["messages"]:
                                if msg.get("tool_calls") and msg["tool_calls"][0]["id"] == chunk["id"]:
                                    msg["tool_calls"][0]["args"] = active_tool_calls[chunk["id"]]["args"]
                                    break
                            yield f'data: {json.dumps({"event": "tool_update", "data": current_tool_state})}\n\n'

                elif msg_type == "tool":
                    current_tool_state["assistant"]["messages"].append(message)
                    yield f'data: {json.dumps({"event": "tool_update", "data": current_tool_state})}\n\n'

    except Exception as e:
        error_message = {"event": "error", "data": str(e)}
        yield f"data: {json.dumps(error_message)}\n\n"

@app.post("/agent/stream")
async def agent_stream_endpoint(request: Dict[str, Any]):
    """
    Endpoint to invoke the agent and get a real-time stream of events,
    including custom 'tool_update' events.
    """
    try:
        thread_id = request["thread_id"]
        input_data = request["input"]
        return StreamingResponse(stream_events(thread_id, input_data), media_type="text/event-stream")
    except KeyError as e:
        return {"error": f"Missing required field: {e}"}, 400

# --- Standard LangGraph Routes for Compatibility ---
# This adds the default LangGraph endpoints at /agent
# You can use these for standard interactions or debugging.
from langgraph.server import add_routes
add_routes(app, graph, path="/agent",
    input_schema=Any,
    output_schema=Any,
    config_schema=Checkpoint
)

# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
