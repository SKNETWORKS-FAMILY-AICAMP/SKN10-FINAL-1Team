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
from django.views.decorators.http import require_http_methods, require_POST
from .models import ChatSession, ChatMessage, AgentType

@login_required
def chatbot_view(request, session_id=None):
    user = request.user
    if session_id:
        session = get_object_or_404(ChatSession, id=session_id, user=user)
        initial_messages = ChatMessage.objects.filter(session_id=session_id).order_by('created_at')
        
        for message in initial_messages:
            message.tool_calls = []
            if message.role == 'assistant' and isinstance(message.tool_data, dict):
                tool_events = {}
                # This loop finds all tool calls and their arguments
                for agent_state in message.tool_data.values():
                    if isinstance(agent_state, dict) and 'messages' in agent_state:
                        for msg in agent_state.get('messages', []):
                            if isinstance(msg, dict) and msg.get('type') == 'ai' and msg.get('tool_calls'):
                                for tc in msg.get('tool_calls', []):
                                    if isinstance(tc, dict) and 'id' in tc:
                                        try:
                                            args_pretty = json.dumps(tc.get('args', {}), indent=2, ensure_ascii=False)
                                        except (TypeError, json.JSONDecodeError):
                                            args_pretty = str(tc.get('args', {}))
                                        tc['args_pretty'] = args_pretty
                                        tool_events[tc['id']] = {'call': tc, 'output': 'Pending...'}
                
                # This loop finds the output for each tool call
                for agent_state in message.tool_data.values():
                    if isinstance(agent_state, dict) and 'messages' in agent_state:
                        for msg in agent_state.get('messages', []):
                            if isinstance(msg, dict) and msg.get('type') == 'tool' and 'tool_call_id' in msg:
                                tool_call_id = msg['tool_call_id']
                                if tool_call_id in tool_events:
                                    tool_events[tool_call_id]['output'] = msg.get('content', '')

                # This formats the data for the template
                if tool_events:
                    processed_calls = []
                    for event in tool_events.values():
                        call_data = event.get('call', {})
                        # Combine the 'call' dict with the 'output' key
                        call_data['output'] = event.get('output', '')
                        processed_calls.append(call_data)
                    message.tool_calls = processed_calls

        active_session_id = session_id
    else:
        initial_messages = []
        active_session_id = None
    
    sessions = ChatSession.objects.filter(user=user, deleted_check=False).order_by('-started_at')

    return render(request, 'conversations/chatbot.html', {
        'sessions': sessions,
        'initial_messages': initial_messages,
        'active_session_id': active_session_id,
    })


@login_required
def session_list_view(request):
    """
    API view to list all non-deleted chat sessions for the logged-in user.
    """
    if request.method == 'GET':
        user = request.user
        sessions = ChatSession.objects.filter(user=user, deleted_check=False).order_by('-started_at')
        
        sessions_data = [
            {
                'id': str(session.id),
                'title': session.title,
                'started_at': session.started_at.isoformat()
            }
            for session in sessions
        ]
        
        return JsonResponse(sessions_data, safe=False)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


@login_required
def message_list_view(request, session_id):
    """
    API view to list all messages for a specific chat session, including processed tool data.
    """
    if request.method == 'GET':
        user = request.user
        # Ensure the session exists and belongs to the logged-in user to prevent unauthorized access.
        get_object_or_404(ChatSession, id=session_id, user=user)
        
        messages_query = ChatMessage.objects.filter(session_id=session_id).order_by('created_at')
        
        messages_data = []
        for message in messages_query:
            tool_calls_data = []
            if message.role == 'assistant' and isinstance(message.tool_data, dict):
                # This logic is adapted from chatbot_view to process tool data for the API response.
                tool_events = {}
                # First pass: find all tool calls and their arguments.
                for agent_state in message.tool_data.values():
                    if isinstance(agent_state, dict) and 'messages' in agent_state:
                        for msg in agent_state.get('messages', []):
                            if isinstance(msg, dict) and msg.get('type') == 'ai' and msg.get('tool_calls'):
                                for tc in msg.get('tool_calls', []):
                                    if isinstance(tc, dict) and 'id' in tc:
                                        tool_events[tc['id']] = {'call': tc, 'output': 'Pending...'}
                
                # Second pass: find the output for each tool call.
                for agent_state in message.tool_data.values():
                    if isinstance(agent_state, dict) and 'messages' in agent_state:
                        for msg in agent_state.get('messages', []):
                            if isinstance(msg, dict) and msg.get('type') == 'tool' and 'tool_call_id' in msg:
                                tool_call_id = msg['tool_call_id']
                                if tool_call_id in tool_events:
                                    tool_events[tool_call_id]['output'] = msg.get('content', '')

                # Third pass: format the data for the frontend.
                if tool_events:
                    for event in tool_events.values():
                        call_data = event.get('call', {})
                        call_data['output'] = event.get('output', '')
                        tool_calls_data.append({
                            'name': call_data.get('name'),
                            'args': call_data.get('args'),
                            'output': call_data.get('output')
                        })

            messages_data.append({
                'id': str(message.id),
                'role': message.role,
                'content': message.content,
                'createdAt': message.created_at.isoformat(),
                'tool_calls': tool_calls_data,
            })
            
        return JsonResponse(messages_data, safe=False)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


@login_required
@require_POST
async def session_create_view(request):
    try:
        data = json.loads(request.body)
        title = data.get('title', '새로운 채팅')
    except json.JSONDecodeError:
        title = '새로운 채팅'

    user = request.user
    session = await ChatSession.objects.acreate(user=user, title=title)
    
    session_data = {
        'id': str(session.id),
        'title': session.title,
        'started_at': session.started_at.isoformat()
    }
    return JsonResponse(session_data, status=201)


@require_POST
@csrf_exempt
async def chat_stream(request, session_id):
    try:
        user_message = None
        if 'application/json' in request.content_type:
            data = json.loads(request.body)
            user_message = data.get("message")
        else:  # Assumes multipart/form-data
            user_message = request.POST.get("message")
            # Files are in request.FILES but are not yet processed.
            # This change is to prevent the 400 error.
        user = request.user

        if not user_message:
            return JsonResponse({"error": "Message content is empty."}, status=400)

        session = await ChatSession.objects.aget(id=session_id, user=user)
        # Save user message first
        await ChatMessage.objects.acreate(session=session, role='user', content=user_message)
        
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
                async for title_event in _stream_and_save_title(session, user_message):
                    yield title_event

            # Then, proceed with the main agent response streaming
            final_ai_content = ""
            final_tool_data = None
            current_tool_state = {"assistant": {"messages": []}}
            active_tool_calls = {}  # Use index as key for simplicity
            known_tool_call_ids = set()

            try:
                internal_secret = os.getenv("FASTAPI_INTERNAL_SECRET")
                headers = {"X-Internal-Secret": internal_secret} if internal_secret else {}
                async with httpx.AsyncClient() as client:
                    github_token = request.user.github_token if hasattr(request.user, 'github_token') else None
                    payload = {
                        "input": {
                            "messages": [
                                {"role": "user", "content": user_message}
                            ]
                        },
                        "config": {
                            "configurable": {
                                "thread_id": thread_id,
                                "user_id": str(request.user.id),
                                "github_token": github_token,
                            }
                        },
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
                                    
                                    # JSON 파싱 전에 기본적인 유효성 검사
                                    if not raw_data.startswith('[') and not raw_data.startswith('{'):
                                        print(f"---[INVALID JSON] Skipping invalid data: {raw_data[:100]}...")
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
                                    # JSON 파싱 오류 시에도 스트리밍을 계속 진행
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

                    # 최종 도구 상태를 한 번 더 전송하여 확실히 완료 상태 전달
                    if final_tool_data:
                        print(f"---[FINAL UPDATE] Sending final tool state")
                        try:
                            # JSON 직렬화 전에 유효성 검사
                            final_data = {"event": "tool_update", "data": final_tool_data}
                            json_str = json.dumps(final_data, ensure_ascii=False)
                            yield f'data: {json_str}\n\n'
                            print(f"---[FINAL UPDATE] Successfully sent final tool state")
                        except Exception as e:
                            print(f"---[FINAL UPDATE ERROR] Failed to send final tool state: {e}")
                    
                    # 스트리밍 완료 이벤트 전송
                    try:
                        yield f"data: {json.dumps({'event': 'stream_end'})}\n\n"
                        print(f"---[STREAM END] Streaming completed successfully")
                    except Exception as e:
                        print(f"---[STREAM END ERROR] Failed to send stream_end: {e}")

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


@login_required
@require_http_methods(["DELETE"])
async def session_delete_view(request, session_id):
    try:
        session = await ChatSession.objects.aget(id=session_id, user=request.user)
        session.deleted_check = True
        await session.asave(update_fields=['deleted_check'])
        return JsonResponse({'status': 'success', 'message': 'Session marked as deleted.'})
    except ChatSession.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Session not found.'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

