from __future__ import annotations

import asyncio
import os
import sys
import functools
import json
import logging
import textwrap
from typing import Sequence, Annotated, TypedDict, List, Dict, Any, Optional, Callable

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.graph.message import add_messages
import re
import json

# 상대 임포트
from .tools import code_agent_tools
from .state import MessagesState
from .agent2 import graph as rag_agent_graph
from .agent3 import graph as analytics_agent_graph
from .prompt import (
    SUPERVISOR_SYSTEM_MESSAGE_GRAPH,
    CODE_SYSTEM_MESSAGE_GRAPH
)

# Import LLM and agent creation utilities
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# --- Load .env --- #
load_dotenv()

OPENAI_API_KEY_ENV = os.environ.get('OPENAI_API_KEY')
if OPENAI_API_KEY_ENV and len(OPENAI_API_KEY_ENV) > 10:
    print(f"DEBUG graph.py: OPENAI_API_KEY seems set (length: {len(OPENAI_API_KEY_ENV)}, first 5 chars: {OPENAI_API_KEY_ENV[:5]}) ")
else:
    print("DEBUG graph.py: OPENAI_API_KEY is NOT SET or is very short.")
# --- End Load .env --- #

# --- Constants and LLM/Agent Setup --- #
MODEL_NAME = "gpt-4o-2024-05-13"  # Current OpenAI model to use
LLM_TEMPERATURE = 0.7
LLM_STREAMING = True

# Initialize LLMs for each agent
def get_llm(temperature=LLM_TEMPERATURE, streaming=LLM_STREAMING, callbacks=None):
    """Get the LLM with configurable parameters"""
    if callbacks is None:
        callbacks = []
    
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        streaming=streaming,
        callbacks=callbacks
    )


# System messages for specific agents are defined in their respective modules (agent2.py, agent3.py)
# or imported directly for agents created in this file (e.g., code_agent from prompt.py).

# --- Agent Creation Functions --- #
# Supervisor agent is now implemented directly as supervisor_router_node

# The analytics_agent is now imported directly from agent3.py as analytics_agent_graph
# The rag_agent is now imported directly from agent2.py as rag_agent_graph

def create_code_agent(tools=None):
    """Create the code/conversation agent"""
    code_llm = get_llm()
    code_tools = code_agent_tools
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", CODE_SYSTEM_MESSAGE_GRAPH),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    return create_react_agent(
        model=code_llm,
        tools=code_tools,
        prompt=prompt,
        name="code_agent"
    )

# --- SupervisorState Definition --- #
class SupervisorState(TypedDict):
    """State for the supervisor graph"""
    # Use the imported add_messages function from langgraph
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_node: str

# --- State Management --- #
# We use the imported add_messages from langgraph for basic state management

# Add a custom function to process message state and clean all messages before they reach the frontend
def process_messages_for_frontend(state: SupervisorState):
    """Process the messages state to clean all routing directives and internal communication
    
    This should be called before returning final results to the frontend API.
    """
    # Apply final message filtering to remove all routing directives and internal metadata
    return {"messages": filter_final_output(state["messages"])}

# --- Supervisor Node --- #
async def supervisor_router_node(state: SupervisorState, config: RunnableConfig):
    """Process user input and route to appropriate agent"""
    print("--- SUPERVISOR ROUTER ---")
    
    # Extract user messages and agent messages separately
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    agent_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage) and msg.content and not msg.content.lower().startswith("supervisor:")]
    
    # Prepare the complete prompt with system instructions and user input
    prompt_messages = [SystemMessage(content=SUPERVISOR_SYSTEM_MESSAGE_GRAPH)] + list(state["messages"])
    
    # For finishing, add an extra instruction to include agent outputs
    if len(agent_messages) > 0:
        agent_outputs_summary = "\n\n지금까지 에이전트들이 제공한 정보:\n"
        for i, msg in enumerate(agent_messages):
            agent_type = "알 수 없는 에이전트"
            if hasattr(msg, 'tags'):
                tags = str(msg.tags) if msg.tags else ""
                if "analytics_agent" in tags:
                    agent_type = "분석 에이전트"
                elif "rag_agent" in tags:
                    agent_type = "문서 에이전트"
                elif "code_agent" in tags:
                    agent_type = "코드 에이전트"
            
            agent_outputs_summary += f"\n--- {agent_type} 응답 {i+1} ---\n{msg.content}\n"
        
        # Add agent outputs as a system message to ensure it's included in the summary
        prompt_messages.append(SystemMessage(content=f"{agent_outputs_summary}\n\n위의 모든 에이전트 정보를 바탕으로 최종 응답을 생성하세요. 'FINISH'로 대화를 마칠 경우, 반드시 위 정보를 모두 포함한 완전한 요약을 제공하세요."))
    
    # Get response from the supervisor LLM
    supervisor_llm = get_llm(temperature=LLM_TEMPERATURE, streaming=LLM_STREAMING)
    response_ai_message = await supervisor_llm.ainvoke(prompt_messages, config=config)
    
    # Determine next node based on response
    next_agent_name = "FINISH" # Default to FINISH
    response_content = response_ai_message.content # Preserve case for matching exact phrases
    response_content_lower = response_content.lower() # For case-insensitive fallback matching

    # First, try to match the exact routing directives from the system prompt
    if "Transfer to DATA ANALYTICS AGENT" in response_content:
        next_agent_name = "analytics_agent"
        print("Matched exact directive for analytics_agent")
    elif "Transfer to DOCUMENT RAG AGENT" in response_content:
        next_agent_name = "rag_agent"
        print("Matched exact directive for rag_agent")
    elif "Transfer to CODE/CONVERSATION AGENT" in response_content:
        next_agent_name = "code_agent"
        print("Matched exact directive for code_agent")
    elif "FINISH" in response_content and not any(term in response_content_lower for term in ["transfer to", "전달", "전환", "넘기"]):
        next_agent_name = "FINISH"
        print("Matched exact directive for FINISH")
    # Fallback to our more flexible pattern matching if exact matches fail
    elif any(term in response_content_lower for term in ["analytics_agent", "analytics specialist", "data analytics", "analytics expert", "분석 에이전트"]):
        next_agent_name = "analytics_agent"
        print("Matched pattern for analytics_agent")
    elif any(term in response_content_lower for term in ["rag_agent", "document rag", "rag specialist", "document agent", "문서 에이전트", "문서 처리"]):
        next_agent_name = "rag_agent"
        print("Matched pattern for rag_agent")
    elif any(term in response_content_lower for term in ["code_agent", "code specialist", "conversation agent", "code expert", "코드 에이전트"]):
        next_agent_name = "code_agent"
        print("Matched pattern for code_agent")
    
    # Print the decision making process
    print(f"Response content snippet: '{response_content[:100]}...'")
    print(f"Final routing decision: {next_agent_name}")
    
    # Clean the message by removing routing directives
    cleaned_content = clean_message_content(response_content)
    
    # For FINISH, ensure the response includes agent information
    if next_agent_name == "FINISH" and len(cleaned_content) < 100 and len(agent_messages) > 0:
        # If we're finishing but have a short response, add agent summary
        response_ai_message = AIMessage(content=f"슈퍼바이저 요약:\n\n{cleaned_content}\n\n---에이전트 제공 정보 요약---\n{agent_outputs_summary}")
    else:
        # Otherwise use the cleaned content
        response_ai_message = AIMessage(content=cleaned_content)
    
    print(f"Supervisor decision: {next_agent_name}, Response: {response_ai_message.content[:100]}...")
    return {
        "messages": [response_ai_message],
        "next_node": next_agent_name
    }

# --- Message Cleaning --- #

def is_routing_message(content):
    """Check if the message is a routing message or a system message that shouldn't be shown to the user.
    
    Args:
        content: The message content to check
        
    Returns:
        bool: True if the message is a routing message, False otherwise
    """
    # Check for common routing/system message patterns
    routing_indicators = [
        'db_query',
        'category_predict_query', 
        'general_query',
        'Routing to',
        'ROUTING:',
        'INTERNAL:',
        'Transfer to'
    ]
    
    # If the content is not a string, it's likely not a routing message
    if not isinstance(content, str):
        return False
    
    # Check if the content contains any routing indicators
    return any(indicator in content for indicator in routing_indicators)
def clean_message_content(content: str) -> str:
    """Clean routing directives from messages for frontend display.
    
    Args:
        content: Original message content with possible routing directives
        
    Returns:
        Cleaned message without routing directives
    """
    # Start with the original content
    cleaned = content
    
    # 1. First pass: Remove exact routing directive phrases with stronger matching
    # These patterns cover both bold and non-bold variations with flexible spacing
    routing_directives = [
        # Bold variations
        r'\*\*\s*Transfer\s+to\s+DATA\s+ANALYTICS\s+AGENT\s*\*\*',
        r'\*\*\s*Transfer\s+to\s+DOCUMENT\s+RAG\s+AGENT\s*\*\*',
        r'\*\*\s*Transfer\s+to\s+CODE\s*/\s*CONVERSATION\s+AGENT\s*\*\*',
        r'\*\*\s*FINISH\s*\*\*',
        
        # Non-bold variations with flexible spacing
        r'Transfer\s+to\s+DATA\s+ANALYTICS\s+AGENT',
        r'Transfer\s+to\s+DOCUMENT\s+RAG\s+AGENT', 
        r'Transfer\s+to\s+CODE\s*/\s*CONVERSATION\s+AGENT',
        r'FINISH',
        
        # Korean variations
        r'.*전달.*AGENT.*',
        r'.*전환.*AGENT.*',
        
        # Lines starting with agent names or transfers
        r'^\s*DATA\s+ANALYTICS\s+AGENT[:\s]*',
        r'^\s*DOCUMENT\s+RAG\s+AGENT[:\s]*',
        r'^\s*CODE\s*/\s*CONVERSATION\s+AGENT[:\s]*',
        r'^\s*SUPERVISOR[:\s]*'
    ]
    
    # Apply all patterns
    for pattern in routing_directives:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # 2. Second pass: Remove entire lines that are likely routing related
    lines = cleaned.split('\n')
    filtered_lines = []
    
    # These patterns catch lines that are likely just routing instructions
    line_patterns = [
        r'.*document\s+rag\s+agent.*',
        r'.*analytics\s+agent.*', 
        r'.*code\s+agent.*',
        r'.*transfer.*agent.*',
        r'.*\*\*agent\*\*.*',
        r'.*전달.*전문가.*',  # Korean variants
        r'.*다음\s+에이전트로\s+전환.*',
        r'^\s*final\s+decision.*$',
        r'^\s*routing\s+to.*$',
        r'^\s*handing\s+off.*$'
    ]
    
    for line in lines:
        line_lower = line.lower()
        # Skip lines that are just about routing
        should_skip = False
        
        for pattern in line_patterns:
            if re.search(pattern, line_lower):
                # Only filter out short lines that are likely just directions
                # Longer matching lines might contain relevant content along with an agent mention
                if len(line.strip()) < 80:
                    should_skip = True
                    break
        
        if not should_skip:
            filtered_lines.append(line)
    
    cleaned = '\n'.join(filtered_lines)
    
    # 3. Clean up the result
    # Remove multiple consecutive blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # Remove any punctuation or whitespace at the beginning that might be artifacts
    cleaned = re.sub(r'^[\s,.;:\-]+', '', cleaned)
    
    # Ensure it's not empty
    cleaned = cleaned.strip()
    if not cleaned:
        cleaned = "I'm reviewing your request."
    
    # Add debug output to monitor cleaning results
    print(f"CLEANING OUTPUT - Original length: {len(content)}, Cleaned length: {len(cleaned)}")
    
    return cleaned

def filter_final_output(messages):
    """Filter the final output messages for frontend display.
    
    This function should be called just before returning messages to the frontend.
    It removes all internal routing directives and metadata.
    
    Args:
        messages: List of messages from the LangGraph agents
        
    Returns:
        Cleaned list of messages suitable for frontend display
    """
    filtered_messages = []
    
    # Process each message to remove routing directives
    for msg in messages:
        if isinstance(msg, AIMessage):
            # Deep cleaning for AI messages which may contain routing directives
            cleaned_content = clean_message_content(msg.content)
            
            # Remove any mentions of agent names at the beginning of paragraphs
            cleaned_content = re.sub(r'^\*\*.*AGENT\*\*:', '', cleaned_content)
            cleaned_content = re.sub(r'^---.*---$', '', cleaned_content, flags=re.MULTILINE)
            
            # Remove any internal conversation history summaries
            cleaned_content = re.sub(r'---이전 대화 요약---.*?---', '', cleaned_content, flags=re.DOTALL)
            
            # Create new message with cleaned content
            # Only use content attribute to avoid potential missing attributes like tags
            if cleaned_content.strip():
                new_msg = AIMessage(content=cleaned_content.strip())
                filtered_messages.append(new_msg)
        else:
            # Keep human messages unchanged
            filtered_messages.append(msg)
    
    return filtered_messages

# --- Conditional Edge Logic --- #
def determine_next_node(state: SupervisorState):
    """Determine which node to route to next based on supervisor decision"""
    print(f"--- Determining Next Node based on Supervisor's decision: {state['next_node']} ---")
    return state["next_node"]  # This should be set by supervisor_router_node

# --- Agent Node Wrapper --- #
async def agent_node_wrapper(state: SupervisorState, agent_runnable, agent_name: str, config: RunnableConfig):
    """Wrapper for agent nodes to handle message state correctly"""
    print(f"--- AGENT: {agent_name} ---")
    initial_messages_count = len(state["messages"])
    
    # Agents created by create_react_agent expect input like {"messages": ...} 
    # and output {"messages": updated_message_list}
    agent_result = await agent_runnable.ainvoke({"messages": state["messages"]}, config=config)
    
    updated_messages_from_agent = agent_result["messages"]
    
    # Extract only the new messages added by the agent to avoid duplication by add_messages
    new_messages = updated_messages_from_agent[initial_messages_count:]
    
    # Clean the agent's messages to remove routing directives
    cleaned_messages = []
    for msg in new_messages:
        if isinstance(msg, AIMessage):
            # Clean the message content to remove routing directives
            cleaned_content = clean_message_content(msg.content)
            # Create a new AIMessage with the cleaned content - only include attributes that exist in AIMessage
            cleaned_msg = AIMessage(
                content=cleaned_content,
                additional_kwargs=msg.additional_kwargs if hasattr(msg, 'additional_kwargs') else None
            )
            cleaned_messages.append(cleaned_msg)
        else:
            # Keep non-AI messages as is
            cleaned_messages.append(msg)
    
    # Print a sample of cleaned content for debugging
    if cleaned_messages and isinstance(cleaned_messages[0], AIMessage):
        print(f"Agent cleaned message sample: '{cleaned_messages[0].content[:100]}...'")
    
    return {"messages": cleaned_messages}

# --- Graph Construction --- #
# Define adapters for imported agent graphs to handle message-based state format
async def rag_agent_adapter(state: SupervisorState, config: RunnableConfig):
    """Adapter for rag_agent_graph to handle message-based state"""
    print("--- AGENT: rag_agent ---")
    
    # Extract query from the most recent human message
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    user_query = user_messages[-1].content if user_messages else ""
    
    # Prepare input for rag_agent_graph which expects a State with user_input
    agent_input = {"user_input": user_query, "messages": state["messages"]}
    
    # Invoke the RAG agent graph
    agent_result = await rag_agent_graph.ainvoke(agent_input, config=config)
    
    # Extract answer from agent result
    if agent_result and "result" in agent_result:
        answer = agent_result["result"]
        # Create an AIMessage with the answer
        response_message = AIMessage(content=answer)
        return {"messages": [response_message]}
    else:
        # Fallback message if no result is found
        return {"messages": [AIMessage(content="I couldn't find relevant information for your query.")]}

async def analytics_agent_adapter(state: SupervisorState, config: RunnableConfig):
    """Adapter for analytics_agent_graph to handle message-based state"""
    print("--- AGENT: analytics_agent ---")
    
    # Analytics agent already expects messages-based state
    agent_result = await analytics_agent_graph.ainvoke({"messages": state["messages"]}, config=config)
    
    # Extract only new messages from the result
    if agent_result and "messages" in agent_result:
        initial_messages_count = len(state["messages"])
        new_messages = agent_result["messages"][initial_messages_count:]
        
        # Debug the new messages content
        print(f"Analytics agent produced {len(new_messages)} new messages")
        for i, msg in enumerate(new_messages):
            print(f"Message {i}: {msg.type} - {msg.content[:100]}..." if len(msg.content) > 100 else msg.content)
        
        # Ensure there's at least one non-routing message for the user
        has_user_facing_content = any(not is_routing_message(msg.content) for msg in new_messages if isinstance(msg, AIMessage))
        
        if not has_user_facing_content and new_messages:
            # Extract the final answer from agent_result if available
            final_answer = agent_result.get("final_answer", None)
            if final_answer:
                new_messages.append(AIMessage(content=final_answer))
            else:
                # If no final_answer and all messages are routing-only, add a generic response
                new_messages.append(AIMessage(content="I've analyzed your request but found no specific answer to display."))
        
        return {"messages": new_messages}
    else:
        # Fallback message if no result is found
        return {"messages": [AIMessage(content="I couldn't perform the requested analytics operation.")]}

def build_graph():
    """Build the main supervisor graph with all agent nodes"""
    # Create the code agent node (still using the original implementation)
    code_agent_runnable = create_code_agent()
    
    # Create state graph with SupervisorState
    workflow = StateGraph(SupervisorState)
    
    # Add supervisor router node (asynchronous function node)
    workflow.add_node("supervisor_router", supervisor_router_node)
    
    # Add analytics_agent node - using adapter for the compiled graph from agent3.py
    workflow.add_node(
        "analytics_agent", 
        analytics_agent_adapter
    )
    
    # Add rag_agent node - using adapter for the compiled graph from agent2.py
    workflow.add_node(
        "rag_agent", 
        rag_agent_adapter
    )
    
    # Add code agent node with wrapper function (keeping original implementation for this one)
    workflow.add_node(
        "code_agent", 
        functools.partial(agent_node_wrapper, agent_runnable=code_agent_runnable, agent_name="code_agent")
    )
    
    # Add a final processing node to clean all messages before they reach the frontend
    workflow.add_node("final_processor", process_messages_for_frontend)
    
    # Set the entry point - this is the first node called when the graph executes
    workflow.set_entry_point("supervisor_router")
    
    # Add conditional edges from supervisor to agent nodes and END state
    workflow.add_conditional_edges(
        "supervisor_router",
        determine_next_node,
        {
            "analytics_agent": "analytics_agent",
            "rag_agent": "rag_agent",
            "code_agent": "code_agent",
            "FINISH": "final_processor"  # Route to final processing instead of END
        }
    )
    
    # Add edges from each agent back to the supervisor router
    workflow.add_edge("analytics_agent", END)
    workflow.add_edge("rag_agent", END)
    workflow.add_edge("code_agent", END)
    
    # Add edge from final processor to END
    workflow.add_edge("final_processor", END)
    
    return workflow.compile()

# Create the compiled graph application
app = build_graph()
print("StateGraph-based supervisor compiled successfully.")