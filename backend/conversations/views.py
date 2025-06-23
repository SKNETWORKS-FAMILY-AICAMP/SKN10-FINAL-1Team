import json
import json
import uuid
import httpx
import os
import asyncio
from openai import AsyncOpenAI
from django.http import StreamingHttpResponse, Http404, JsonResponse
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .models import ChatSession, ChatMessage, AgentType

@login_required
def chatbot_view(request, session_id=None):
    user = request.user
    if session_id:
        session = get_object_or_404(ChatSession, id=session_id, user=user)
        initial_messages = ChatMessage.objects.filter(session_id=session_id).order_by('created_at')
        
        for message in initial_messages:
            message.processed_tool_events = []
            if message.role == 'assistant' and isinstance(message.tool_data, dict):
                tool_events = {}
                for agent_state in message.tool_data.values():
                    if isinstance(agent_state, dict) and 'messages' in agent_state:
                        # First pass: collect tool calls
                        for msg in agent_state.get('messages', []):
                            if isinstance(msg, dict) and msg.get('type') == 'ai' and msg.get('tool_calls'):
                                for tc in msg.get('tool_calls', []):
                                    if isinstance(tc, dict) and 'id' in tc:
                                        # Pretty print args for template
                                        tc['args_pretty'] = json.dumps(tc.get('args', {}), indent=2)
                                        tool_events[tc['id']] = {'call': tc, 'output': 'Pending...'}
                        
                        # Second pass: collect tool outputs
                        for msg in agent_state.get('messages', []):
                            if isinstance(msg, dict) and msg.get('type') == 'tool' and 'tool_call_id' in msg:
                                tool_call_id = msg['tool_call_id']
                                if tool_call_id in tool_events:
                                    output_content = msg.get('content', '')
                                    try:
                                        # Pretty-print JSON if possible
                                        parsed_json = json.loads(output_content)
                                        tool_events[tool_call_id]['output'] = json.dumps(parsed_json, indent=2)
                                    except (json.JSONDecodeError, TypeError):
                                        tool_events[tool_call_id]['output'] = output_content
                
                if tool_events:
                    message.processed_tool_events = list(tool_events.values())

        active_session_id = session_id
    else:
        initial_messages = []
        active_session_id = None
    
    sessions = ChatSession.objects.filter(user=user).order_by('-started_at')

    return render(request, 'conversations/chatbot.html', {
        'sessions': sessions,
        'initial_messages': initial_messages,
        'active_session_id': active_session_id,
    })


@login_required
@require_POST
async def session_create_view(request):
    user = request.user
    # 기본 에이전트 타입으로 새 세션 생성
    session = await ChatSession.objects.acreate(user=user, agent_type=AgentType.DEFAULT)
    return JsonResponse({'session_id': str(session.id)})


@require_POST
@csrf_exempt
async def chat_stream(request, session_id):
    try:
        data = json.loads(request.body)
        message_content = data.get('message')
        user = request.user

        if not message_content:
            return JsonResponse({"error": "Message content is empty."}, status=400)

        session = await ChatSession.objects.aget(id=session_id, user=user)
        # Save user message first
        await ChatMessage.objects.acreate(session=session, role='user', content=message_content)
        
        # Check if this is the first message to generate a title
        is_first_message = await ChatMessage.objects.filter(session=session).acount() == 1

        thread_id = str(session.thread_id) if session.thread_id else str(uuid.uuid4())
        if not session.thread_id:
            session.thread_id = uuid.UUID(thread_id)
            await session.asave()

        fastapi_url = os.environ.get("FASTAPI_232SERVER_URL", "http://127.0.0.1:8001")

        async def event_stream():
            # Stream title first if it's the first message
            if is_first_message:
                async for title_event in _stream_and_save_title(session, message_content):
                    yield title_event

            # Then, proceed with the main agent response streaming
            final_ai_content = ""
            final_tool_data = None
            current_tool_state = {"assistant": {"messages": []}}
            active_tool_calls = {}
            known_tool_call_ids = set()

            try:
                internal_secret = os.getenv("FASTAPI_INTERNAL_SECRET")
                headers = {"X-Internal-Secret": internal_secret} if internal_secret else {}
                async with httpx.AsyncClient() as client:
                    payload = {
                        "input": {"messages": [{"role": "user", "content": message_content}]},
                        "config": {
                            "configurable": {"thread_id": thread_id},
                            "stream_mode": ["messages", "updates"]
                        }
                    }
                    async with client.stream("POST", f"{fastapi_url}/invoke", json=payload, headers=headers, timeout=300.0) as response:
                        response.raise_for_status()
                        tool_use_started_sent = False # Flag to send start event only once
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    raw_data = line[len("data: "):].strip()
                                    if not raw_data:
                                        continue
                                    chunk = json.loads(raw_data)
                                    event_type, payload = chunk[0], chunk[1]

                                    if event_type == "messages":
                                        message_chunk = payload[0]
                                        metadata = payload[1]

                                        # Send a signal to show the tool UI as soon as tool usage is detected
                                        if not tool_use_started_sent and message_chunk.get("tool_call_chunks"):
                                            yield f"data: {json.dumps({'event': 'tool_use_started'})}\n\n"
                                            tool_use_started_sent = True

                                        # Stream pure text content
                                        if (message_chunk.get("type") == "AIMessageChunk" and not message_chunk.get("tool_call_chunks")):
                                            content = message_chunk.get("content", "")
                                            if content:
                                                final_ai_content += content
                                                sse_payload = {"event": "message_chunk", "data": content}
                                                yield f"data: {json.dumps(sse_payload)}\n\n"

                                        # Handle tool-related messages for real-time UI updates
                                        if message_chunk.get("type") == "AIMessageChunk":
                                            # 1. Accumulate tool call argument chunks using 'index'
                                            tool_call_chunks = message_chunk.get("tool_call_chunks", [])
                                            if tool_call_chunks:
                                                for tc_chunk in tool_call_chunks:
                                                    chunk_index = tc_chunk.get("index")
                                                    if chunk_index is not None:
                                                        if chunk_index not in active_tool_calls:
                                                            # First chunk for this index, contains name and id
                                                            active_tool_calls[chunk_index] = {
                                                                "name": tc_chunk.get("name"), 
                                                                "args": tc_chunk.get("args", ""), 
                                                                "id": tc_chunk.get("id")
                                                            }
                                                        else:
                                                            # Subsequent chunks, append args
                                                            active_tool_calls[chunk_index]["args"] += tc_chunk.get("args", "")
                                            
                                            # 2. Check for tool call end signal, then yield update with complete args
                                            if message_chunk.get("response_metadata", {}).get("finish_reason") == "tool_calls":
                                                if active_tool_calls:
                                                    # Sort tool calls by index to ensure correct order for multiple calls
                                                    sorted_indices = sorted(active_tool_calls.keys())
                                                    complete_tool_calls = [active_tool_calls[i] for i in sorted_indices]
                                                    
                                                    # Store the IDs for later lookup when tool results arrive
                                                    for tc in complete_tool_calls:
                                                        if tc.get("id"):
                                                            known_tool_call_ids.add(tc["id"])
                                                    
                                                    # Find if an 'ai' message already exists to append to
                                                    existing_ai_message = None
                                                    for msg in reversed(current_tool_state["assistant"]["messages"]):
                                                        if msg.get("type") == "ai":
                                                            existing_ai_message = msg
                                                            break
                                                    
                                                    if existing_ai_message:
                                                        # Append new tool calls to the existing message's tool_calls list
                                                        if "tool_calls" not in existing_ai_message:
                                                            existing_ai_message["tool_calls"] = []
                                                        existing_ai_message["tool_calls"].extend(complete_tool_calls)
                                                    else:
                                                        # Or create a new ai message if it's the first tool call
                                                        ai_message = {"type": "ai", "tool_calls": complete_tool_calls}
                                                        current_tool_state["assistant"]["messages"].append(ai_message)
                                                    
                                                    final_tool_data = current_tool_state.copy()
                                                    yield f'data: {json.dumps({"event": "tool_update", "data": current_tool_state})}\n\n'
                                                    active_tool_calls.clear()

                                        # 3. Handle tool output
                                        elif message_chunk.get("type") == "tool":
                                            tool_call_id = message_chunk.get("tool_call_id")
                                            # Check if the result corresponds to a known tool call
                                            if tool_call_id in known_tool_call_ids:
                                                tool_message = {
                                                    "type": "tool",
                                                    "tool_call_id": tool_call_id,
                                                    "content": message_chunk.get("content", ""),
                                                    "name": message_chunk.get("name")
                                                }
                                                current_tool_state["assistant"]["messages"].append(tool_message)
                                                final_tool_data = current_tool_state.copy()
                                                yield f'data: {json.dumps({"event": "tool_update", "data": current_tool_state})}\n\n'

                                    elif event_type == "updates":
                                        # The 'updates' event contains the final state, which can be used for saving.
                                        final_tool_data = payload

                                except (json.JSONDecodeError, IndexError) as e:
                                    print(f"---[STREAM PARSE ERROR] {e} on line: {raw_data}")
                                    continue
                
                if final_ai_content or final_tool_data:
                    # Try to parse args from string to JSON for saving
                    if final_tool_data and "assistant" in final_tool_data:
                        for msg in final_tool_data["assistant"].get("messages", []):
                            if msg.get("type") == "ai" and "tool_calls" in msg:
                                for call in msg.get("tool_calls", []):
                                    try:
                                        call["args"] = json.loads(call["args"])
                                        print(call["args"])
                                    except (json.JSONDecodeError, TypeError):
                                        pass # Keep as string if not valid JSON

                    await ChatMessage.objects.acreate(
                        session=session,
                        role='assistant',
                        content=final_ai_content.strip(),
                        tool_data=final_tool_data
                    )

            except httpx.RequestError as e:
                print(f"---[HTTPX STREAM ERROR] An error occurred: {e}")
                yield f"data: {json.dumps({'event': 'error', 'data': 'Could not connect to the AI service.'})}\n\n"
            except Exception as e:
                import traceback
                print(f"---[STREAM ERROR] An error occurred in event_stream: {e}")
                traceback.print_exc()
                yield f"data: {json.dumps({'event': 'error', 'data': 'An internal error occurred.'})}\n\n"

        return StreamingHttpResponse(event_stream(), content_type='text/event-stream')

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)
    except ChatSession.DoesNotExist:
        raise Http404("Chat session not found")
    except Exception as e:
        return JsonResponse({"error": f"An unexpected error occurred: {str(e)}"}, status=500)


async def _stream_and_save_title(session, user_message):
    """
    Generates a title based on the user's first message using OpenAI's streaming API,
    yields SSE events for real-time UI updates, and saves the final title to the database.
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    final_title = ""
    try:
        prompt = f"""Create a very short, concise title in Korean (5 words maximum, 20 characters total) for a conversation starting with this message. Do not use any special characters like quotes or brackets. Just provide the raw text title.\n\nUser Message: \"{user_message}\"\n"""        
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.5,
        )
        # Signal the start of title streaming
        yield f"data: {json.dumps({'event': 'title_start'})}\n\n"
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                final_title += content
                yield f"data: {json.dumps({'event': 'title_chunk', 'data': content})}\n\n"
        
        final_title = final_title.strip().replace('"', '')
        if final_title:
            session.title = final_title
            await session.asave(update_fields=['title'])

        # Signal the end of title streaming with the final title
        yield f"data: {json.dumps({'event': 'title_end', 'data': final_title})}\n\n"

    except Exception as e:
        print(f"---[TITLE GEN ERROR] Could not generate title: {e}")
        # Silently fail, maybe yield an end event with a default title
        yield f"data: {json.dumps({'event': 'title_end', 'data': 'New Chat'})}\n\n"


