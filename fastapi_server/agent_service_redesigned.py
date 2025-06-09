import asyncio
import os
import sys
import json
import re  # Added for regex
import logging
import httpx
from typing import Dict, Any, Optional, List, AsyncGenerator, Pattern  # Added Pattern
from fastapi import WebSocket
from contextlib import asynccontextmanager

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.graph import app as agent_app  # Assuming this path is correct

# Logger setup
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class AgentService:
    """
    Service class to interact with the LangGraph agent, with enhanced
    event handling and filtering for cleaner output.
    """
    def __init__(self):
        """Initialize the agent service."""
        self.agent_app = agent_app
        self.thread_counter = 0
        self.postgres_uri = os.environ.get('POSTGRES_CONNECTION_STRING', '')
        self.denied_streaming_nodes = {"choose_node"}  # Add actual node names to silence
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Compiled regex patterns for filtering internal data
        self.internal_data_patterns: List[Pattern] = [
            re.compile(r'^\s*\{\s*"query_type":.*?\}\s*$', re.DOTALL),
            re.compile(r'^\s*\{\s*"sql_query":.*?\}\s*$', re.DOTALL),
            re.compile(r'^\s*\{\s*"sql_output_choice":.*?\}\s*$', re.DOTALL),
            re.compile(r'^\s*\{\s*"db_query":.*?\}\s*$', re.DOTALL),
        ]
        logger.info("AgentService initialized with structured event handling and filtering.")

    async def _send_websocket_json(self, websocket: WebSocket, data: Dict[str, Any]):
        """Safely send JSON data over WebSocket."""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}", exc_info=True)

    def _is_internal_data(self, content: str) -> bool:
        """Check if the content matches any internal data patterns."""
        if not isinstance(content, str):
            return False
        stripped_content = content.strip()
        for pattern in self.internal_data_patterns:
            if pattern.fullmatch(stripped_content):
                logger.debug(f"Filtered internal data by pattern '{pattern.pattern}': {stripped_content[:100]}...")
                return True
        if len(stripped_content) < 10 and stripped_content in ["{}", "[]", "null", "true", "false"]:
             logger.debug(f"Filtered short/artifact data: {stripped_content}")
             return True
        return False

    async def _handle_chat_model_stream(
        self,
        event: Dict[str, Any],
        websocket: WebSocket,
        current_agent: str,
        content_accumulator: List[str]
    ) -> None:
        event_node_name = event.get("name", "")
        if event_node_name in self.denied_streaming_nodes:
            logger.debug(f"Chat model stream from denied node '{event_node_name}'. Suppressing WebSocket send and accumulation.")
            return

        """Handles 'on_chat_model_stream' events."""
        try:
            chunk_data = event.get("data", {}).get("chunk", {})
            content = ""
            if hasattr(chunk_data, 'content'):
                content = chunk_data.content
            elif isinstance(chunk_data, dict) and 'content' in chunk_data:
                content = chunk_data['content']
            elif isinstance(chunk_data, str):
                content = chunk_data
            else:
                content = str(chunk_data)

            if content and not self._is_internal_data(content):
                content_accumulator.append(content)
                await self._send_websocket_json(websocket, {
                    "event_type": "token",
                    "data": {"token": content, "agent": current_agent}
                })
            elif content:
                logger.debug(f"Skipping internal data chunk: {content[:100]}...")

        except Exception as e:
            logger.warning(f"Error processing stream token: {str(e)} in event: {event}", exc_info=True)

    async def _handle_chain_start(
        self,
        event: Dict[str, Any],
        websocket: WebSocket,
        current_agent_ref: List[str]
    ) -> None:
        """Handles 'on_chain_start' events to detect agent changes."""
        tags = event.get("tags", [])
        known_agents = {"analytics_agent", "rag_agent", "code_agent", "supervisor"}
        
        new_agent = None
        for tag in tags:
            if tag in known_agents:
                new_agent = tag
                break
        
        if new_agent and new_agent != current_agent_ref[0]:
            logger.info(f"Agent changed from '{current_agent_ref[0]}' to '{new_agent}'")
            current_agent_ref[0] = new_agent
            await self._send_websocket_json(websocket, {
                "event_type": "agent_change",
                "data": {"agent": current_agent_ref[0]}
            })

    async def fetch_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Fetch chat history from PostgreSQL database."""
        try:
            if not self.postgres_uri:
                logger.warning("POSTGRES_CONNECTION_STRING is not set.")
                return []
            
            query = "SELECT * FROM chat_messages WHERE session_id = $1 ORDER BY created_at ASC"
            nextjs_url = os.environ.get('NEXTJS_API_URL', 'http://localhost:3000')
            api_token = os.environ.get('API_TOKEN', 'your-default-api-token')

            response = await self.http_client.post(
                f"{nextjs_url}/api/db",
                json={"sql": query, "params": [session_id]},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_token}"
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get('rows', [])
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching chat history: {e.response.status_code} - {e.response.text}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Error fetching chat history: {str(e)}", exc_info=True)
            return []

    def convert_to_langchain_messages(self, db_messages: List[Dict[str, Any]]) -> List[Any]:
        """Convert database messages to LangChain messages."""
        messages = []
        messages.append(SystemMessage(content="You are a helpful assistant. Please provide answers in Korean."))
        
        for msg in db_messages:
            content = msg.get("content", "")
            role = msg.get("role", "user").lower()
            
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role in ["assistant", "ai"]:
                messages.append(AIMessage(content=content))
        return messages

    async def process_message(self, message: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a message (non-streaming)."""
        if not thread_id:
            self.thread_counter += 1
            thread_id = f"thread_nonstream_{self.thread_counter}"
            
        logger.info(f"Processing non-streaming message in thread {thread_id}")
        
        previous_messages = await self.fetch_chat_history(thread_id)
        langchain_messages = self.convert_to_langchain_messages(previous_messages)
        langchain_messages.append(HumanMessage(content=message))
        
        inputs = {"messages": langchain_messages}
        config = {"recursion_limit": 25, "configurable": {"thread_id": thread_id}}
        
        final_content = "Error: No response generated."
        active_agent = "supervisor"
        
        try:
            result = await self.agent_app.ainvoke(inputs, config=config)
            response_messages = result.get("messages", [])
            
            ai_responses = [m for m in response_messages if isinstance(m, AIMessage)]
            if ai_responses:
                raw_final_content = ai_responses[-1].content
                if not self._is_internal_data(raw_final_content):
                    final_content = raw_final_content
                else:
                    final_content = "[내용이 필터링되었습니다]"
                
                for m in reversed(response_messages):
                    if hasattr(m, 'name') and m.name in ["analytics_agent", "rag_agent", "code_agent"]:
                        active_agent = m.name
                        break
            
        except Exception as e:
            logger.error(f"Error in non-streaming process_message: {str(e)}", exc_info=True)
            final_content = f"Error: {str(e)}"
        
        return {"content": final_content, "agent": active_agent, "thread_id": thread_id}

    async def process_message_stream(
        self, 
        message: str, 
        thread_id: str,
        websocket: WebSocket
    ) -> None:
        """Process a message and stream results via WebSocket."""
        logger.info(f"Processing stream for message in thread {thread_id}")
        
        previous_messages = await self.fetch_chat_history(thread_id)
        langchain_messages = self.convert_to_langchain_messages(previous_messages)
        langchain_messages.append(HumanMessage(content=message))
        
        inputs = {"messages": langchain_messages}
        config = {"recursion_limit": 25, "configurable": {"thread_id": thread_id}}
        
        current_agent_ref = ["supervisor"]
        content_accumulator: List[str] = []

        event_action_map = {
            "on_chat_model_stream": lambda event: self._handle_chat_model_stream(event, websocket, current_agent_ref[0], content_accumulator),
            "on_chain_start": lambda event: self._handle_chain_start(event, websocket, current_agent_ref),
        }

        try:
            async for event in self.agent_app.astream_events(inputs, config=config, version="v2"):
                event_kind = event["event"]
                if handler := event_action_map.get(event_kind):
                    await handler(event)

            final_clean_content = "".join(content_accumulator)
            logger.info(f"Streaming finished. Final accumulated content length: {len(final_clean_content)}")
            await self._send_websocket_json(websocket, {
                "event_type": "done",
                "data": {"content": final_clean_content, "agent": current_agent_ref[0]}
            })
            
        except Exception as e:
            logger.error(f"Error during message streaming: {str(e)}", exc_info=True)
            await self._send_websocket_json(websocket, {
                "event_type": "error",
                "data": {"error": f"Streaming Error: {str(e)}"}
            })

# Singleton pattern for AgentService
_agent_service_instance: Optional[AgentService] = None

async def get_agent_service() -> AgentService:
    """
    Dependency to get the AgentService instance.
    Uses lazy initialization.
    """
    global _agent_service_instance
    if _agent_service_instance is None:
        _agent_service_instance = AgentService()
    return _agent_service_instance
