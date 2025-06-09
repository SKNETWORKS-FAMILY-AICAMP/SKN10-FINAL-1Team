import asyncio
import os
import sys
import json
import logging
import httpx
from typing import Dict, Any, Optional, List, AsyncGenerator
from fastapi import WebSocket
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import LangGraph agent from the local agent module
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent.graph import app as agent_app


class AgentService:
    """
    Service class to interact with the LangGraph agent.
    """
    def __init__(self):
        """Initialize the agent service."""
        self.agent_app = agent_app
        self.thread_counter = 0
        self.active_conversations = {}
        self.postgres_uri = os.environ.get('POSTGRES_CONNECTION_STRING', '')
        self.http_client = httpx.AsyncClient(timeout=30.0)
        logger.info("AgentService initialized")
        
    async def fetch_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Fetch chat history from PostgreSQL database for the given session ID.
        
        Args:
            session_id: The session ID to fetch messages for
            
        Returns:
            List of message objects
        """
        try:
            if not self.postgres_uri:
                logger.warning("POSTGRES_CONNECTION_STRING is not set, cannot fetch chat history")
                return []
                
            # Create a request to the Next.js API route for querying the database
            query = """SELECT * FROM chat_messages 
                       WHERE session_id = $1 
                       ORDER BY created_at ASC"""
            
            # Docker 환경에 맞게 frontend 서비스 URL 설정
            nextjs_url = os.environ.get('NEXTJS_API_URL', 'http://localhost:3000')
            
            # JWT 토큰 환경변수에서 가져오기 (실제 환경에서는 인증 서비스에서 발급받아야 함)
            api_token = os.environ.get('API_TOKEN', 'default-dev-token')
            
            # Use httpx to make async request to the Next.js API route with parameterized query
            response = await self.http_client.post(
                f"{nextjs_url}/api/db",  # Configurable frontend URL
                json={"sql": query, "params": [session_id]},  # 필드 이름 'query' -> 'sql'로 변경
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_token}"
                }
            )
            
            # Check response status
            if response.status_code != 200:
                logger.error(f"Error fetching chat history: {response.text}")
                return []
                
            result = response.json()
            # Next.js API returns rows in a nested object
            if 'rows' in result:
                return result['rows']
            return result
            
        except Exception as e:
            logger.error(f"Error fetching chat history: {str(e)}", exc_info=True)
            return []
            
    def convert_to_langchain_messages(self, db_messages: List[Dict[str, Any]]) -> List[Any]:
        """
        Convert database message format to LangChain message format.
        
        Args:
            db_messages: List of messages from database
            
        Returns:
            List of LangChain messages
        """
        messages = []
        
        # Add a system message to provide context
        messages.append(SystemMessage(content="You are a helpful assistant that answers questions based on previous conversations."))
        
        # Convert database messages to LangChain messages
        for msg in db_messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant" or role == "ai":
                messages.append(AIMessage(content=content))
        
        return messages

    async def process_message(self, message: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a message through the LangGraph agent.
        
        Args:
            message: The user message to process
            thread_id: Optional thread ID for conversation tracking
            
        Returns:
            Dict containing the agent response
        """
        if not thread_id:
            self.thread_counter += 1
            thread_id = f"thread_{self.thread_counter}"
            
        logger.info(f"Processing message in thread {thread_id}")
        
        # Fetch previous messages for this thread/session
        previous_messages = await self.fetch_chat_history(thread_id)
        langchain_messages = self.convert_to_langchain_messages(previous_messages)
        
        # Add the current message
        langchain_messages.append(HumanMessage(content=message))
        
        # Prepare input for the agent with full message history
        inputs = {"messages": langchain_messages}
        
        # Configure the agent run
        config = {
            "recursion_limit": 25,
            "configurable": {"thread_id": thread_id}
        }
        
        # Capture the final content and active agent
        final_content = ""
        active_agent = "supervisor"
        
        try:
            # Process through the LangGraph agent
            result = await self.agent_app.ainvoke(inputs, config=config)
            
            # Extract messages from result
            messages = result.get("messages", [])
            
            # Get the final assistant message
            assistant_messages = [m for m in messages if m.type == "ai"]
            
            if assistant_messages:
                final_content = assistant_messages[-1].content
                
                # Try to determine which agent was active last
                for m in reversed(messages):
                    if hasattr(m, 'name') and m.name in ["analytics_agent", "rag_agent", "code_agent"]:
                        active_agent = m.name
                        break
            else:
                final_content = "No response generated."
        
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            final_content = f"Error processing your request: {str(e)}"
        
        return {
            "content": final_content,
            "agent": active_agent,
            "thread_id": thread_id
        }
        
    async def process_message_stream(
        self, 
        message: str, 
        thread_id: str,
        websocket: WebSocket
    ) -> None:
        """
        Process a message through the LangGraph agent and stream the results via WebSocket.
        
        Args:
            message: The user message to process
            thread_id: Thread ID for conversation tracking
            websocket: WebSocket connection to stream results to
        """
        logger.info(f"Processing message stream in thread {thread_id}")
        
        # Fetch previous messages for this thread/session
        previous_messages = await self.fetch_chat_history(thread_id)
        langchain_messages = self.convert_to_langchain_messages(previous_messages)
        
        # Add the current message
        langchain_messages.append(HumanMessage(content=message))
        
        # Prepare input for the agent with full message history
        inputs = {"messages": langchain_messages}
        
        # Configure the agent run
        config = {
            "recursion_limit": 25,
            "configurable": {"thread_id": thread_id}
        }
        
        current_agent = "supervisor"
        content_buffer = ""
        
        try:
            # Use astream_events to get detailed streaming feedback
            async for event in self.agent_app.astream_events(inputs, config=config, version="v2"):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    # Stream token from the model - safely handle different structures
                    try:
                        chunk = event.get("data", {}).get("chunk", {})
                        # Handle different ways content might be accessed
                        if hasattr(chunk, 'content'):
                            content = chunk.content
                        elif isinstance(chunk, dict) and 'content' in chunk:
                            content = chunk['content']
                        else:
                            content = str(chunk)
                            
                        if content:
                            # 내부 처리 데이터 필터링 (JSON, SQL 쿼리 등의 내부 상태 정보)
                            # {"query_type":...} 또는 {"sql_query":...} 형식의 JSON 필터링
                            if (content.startswith('{"query_type":') or 
                                content.startswith('{"sql_query":') or
                                '{"sql_query":' in content or
                                '{"sql_output_choice":' in content):
                                logger.debug(f"Filtering internal state data: {content}")
                                continue
                                
                            # 최종 응답에 추가
                            content_buffer += content
                            await websocket.send_json({
                                "event_type": "token",
                                "data": {
                                    "token": content,
                                    "agent": current_agent
                                }
                            })
                    except Exception as e:
                        logger.warning(f"Error processing stream token: {str(e)}")
                        # Continue processing even if one token fails
                
                # elif kind == "on_tool_start":
                #     # Tool execution start
                #     tool_name = event.get("name", "unknown_tool")
                #     tool_args = event.get("data", {}).get("input", {})
                #     
                #     await websocket.send_json({
                #         "event_type": "tool_start",
                #         "data": {
                #             "tool": {
                #                 "name": tool_name,
                #                 "args": tool_args
                #             },
                #             "agent": current_agent
                #         }
                #     })
                
                # elif kind == "on_tool_end":
                #     # Tool execution end
                #     tool_name = event.get("name", "unknown_tool")
                #     tool_output = event.get("data", {}).get("output", "")
                #     
                #     # SQL 관련 내부 데이터 필터링
                #     filtered_output = tool_output
                #     if isinstance(filtered_output, str):
                #         # JSON 데이터 구조가 있는지 확인
                #         if ('{"query_type":' in filtered_output or 
                #             '{"sql_query":' in filtered_output or 
                #             '{"sql_output_choice":' in filtered_output):
                #             logger.info(f"Filtering internal SQL data from tool output: {tool_name}")
                #             filtered_output = "[내부 처리 중...]"  # 사용자에게는 단순한 처리 중 메시지만 표시
                #     
                #     await websocket.send_json({
                #         "event_type": "tool_end",
                #         "data": {
                #             "tool": {
                #                 "name": tool_name,
                #                 "output": str(filtered_output)[:200] + ("..." if len(str(filtered_output)) > 200 else "")
                #             },
                #             "agent": current_agent
                #         }
                #     })
                
                elif kind == "on_chain_start":
                    # Agent change detection
                    if "agent" in event.get("tags", []):
                        for tag in event.get("tags", []):
                            if tag in ["analytics_agent", "rag_agent", "code_agent", "supervisor_router"]:
                                new_agent = tag
                                if new_agent != current_agent:
                                    current_agent = new_agent
                                    await websocket.send_json({
                                        "event_type": "agent_change",
                                        "data": {
                                            "agent": current_agent
                                        }
                                    })
                                break
            
            # Send the final completion event
            await websocket.send_json({
                "event_type": "done",
                "data": {
                    "content": content_buffer,
                    "agent": current_agent
                }
            })
            
        except Exception as e:
            logger.error(f"Error in streaming process: {str(e)}", exc_info=True)
            await websocket.send_json({
                "event_type": "error",
                "data": {
                    "error": f"Error: {str(e)}"
                }
            })


# Singleton pattern for AgentService
_agent_service_instance = None

async def get_agent_service() -> AgentService:
    """
    Dependency to get the AgentService instance.
    Uses lazy initialization to create the service only when needed.
    """
    global _agent_service_instance
    if _agent_service_instance is None:
        _agent_service_instance = AgentService()
    return _agent_service_instance
