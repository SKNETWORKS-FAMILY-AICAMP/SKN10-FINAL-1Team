
from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
import os
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import json
import uuid
from datetime import datetime
import logging

# Add project root to sys.path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import agent module
from fastapi_server.agent_service import AgentService, get_agent_service
from fastapi_server.models import ChatRequest, ChatResponse, ChatMessage

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if not os.path.exists(dotenv_path):
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Check OpenAI API Key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY or len(OPENAI_API_KEY) < 10:
    raise ValueError("OPENAI_API_KEY is not set or invalid. Please check your .env file.")

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph Agent API",
    description="API for interacting with a multi-agent system built with LangGraph",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active websocket connections and their associated thread IDs
active_connections = {}

@app.on_event("startup")
async def startup_event():
    """Initialize agent service on startup."""
    logger.info("Starting up FastAPI server and initializing LangGraph agent...")
    # Agent service is lazily initialized using dependency


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down FastAPI server...")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "online", "message": "LangGraph Agent API is running"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    agent_service: AgentService = Depends(get_agent_service)
):
    """
    Process a chat message through the LangGraph agent system.
    """
    try:
        # Generate a thread ID if not provided
        thread_id = request.thread_id or f"thread_{uuid.uuid4()}"
        
        # Process the request through the agent service
        response = await agent_service.process_message(
            request.message, 
            thread_id=thread_id
        )
        
        return ChatResponse(
            thread_id=thread_id,
            message_id=str(uuid.uuid4()),
            content=response["content"],
            created_at=datetime.now().isoformat(),
            role="assistant",
            agent=response.get("agent", "system")
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent processing error: {str(e)}")


@app.websocket("/api/chat/ws/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str, agent_service: AgentService = Depends(get_agent_service)):
    """
    WebSocket endpoint for streaming chat interactions with the LangGraph agent.
    """
    await websocket.accept()
    
    # Register this connection
    connection_id = str(uuid.uuid4())
    active_connections[connection_id] = {"websocket": websocket, "thread_id": thread_id}
    
    try:
        logger.info(f"WebSocket connection established: {connection_id} for thread {thread_id}")
        
        while True:
            # Receive and parse the message
            data = await websocket.receive_text()
            try:
                message_data = json.loads(data)
                user_message = message_data.get("message", "")
                
                if not user_message:
                    await websocket.send_json({
                        "type": "error",
                        "content": "Invalid message format. Expected 'message' field."
                    })
                    continue
                
                # Process through agent with streaming
                await agent_service.process_message_stream(
                    user_message,
                    thread_id=thread_id,
                    websocket=websocket
                )
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "content": "Invalid JSON format"
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "content": f"Server error: {str(e)}"
            })
        except:
            pass
    finally:
        # Clean up on disconnect
        if connection_id in active_connections:
            del active_connections[connection_id]
        logger.info(f"WebSocket connection closed: {connection_id}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
