from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ChatMessage(BaseModel):
    """Model for chat messages."""
    role: str = Field(..., description="Role of the message sender (user, assistant, system, tool)")
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(None, description="Name of the tool or agent (if applicable)")
    created_at: Optional[str] = Field(None, description="ISO timestamp of when the message was created")


class ChatRequest(BaseModel):
    """Model for chat API requests."""
    message: str = Field(..., description="User message content")
    thread_id: Optional[str] = Field(None, description="Thread identifier for conversation tracking")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the request")


class ChatResponse(BaseModel):
    """Model for chat API responses."""
    thread_id: str = Field(..., description="Thread identifier for conversation tracking")
    message_id: str = Field(..., description="Unique identifier for this message")
    content: str = Field(..., description="Assistant response content")
    created_at: str = Field(..., description="ISO timestamp of when the response was created")
    role: str = Field("assistant", description="Role of the message sender")
    agent: str = Field("system", description="Agent that processed the request")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the response")


class StreamEvent(BaseModel):
    """Model for streaming events."""
    type: str = Field(..., description="Event type (e.g., 'token', 'agent_change', 'tool_start', 'tool_end', 'done')")
    content: Optional[str] = Field(None, description="Content of the event (e.g., text token)")
    agent: Optional[str] = Field(None, description="Agent name, if applicable")
    tool: Optional[Dict[str, Any]] = Field(None, description="Tool information, if applicable")
