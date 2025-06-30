import json
import copy
import uuid
import httpx
import os
import asyncio
import sys
import base64
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
    
    sessions = ChatSession.objects.filter(user=user, deleted_check=False).order_by('-started_at')

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
        user = request.user
        message_content = ''
        file_name = None
        file_content = None
        agent = 'default'  # Default agent

        # --- DEBUG ---
        print(f"\n---[ chat_stream START ]---")
        print(f"Request Content-Type: {request.content_type}")
        # --- END DEBUG ---

        # Handle multipart/form-data for file uploads, or application/json for standard messages
        if request.content_type.startswith('multipart/form-data'):
            form_data = request.POST
            message_content = form_data.get('message', '')
            agent = form_data.get('agent', 'default')
            file_obj = request.FILES.get('file')
            if file_obj:
                file_name = file_obj.name
                try:
                    file_content = file_obj.read().decode('utf-8')
                except UnicodeDecodeError:
                    file_content = "Error: Could not decode file content. Please ensure it is UTF-8 encoded."
        
        elif request.content_type.startswith('application/json'):
            raw_body = request.body.decode('utf-8')
            print(f"Raw JSON Body: {raw_body}")
            data = json.loads(raw_body)
            
            message_content = data.get('message', '')
            agent = data.get('agent', 'default')
            file_name = data.get('file_name')
            # Check for 'csv_file_content' and fall back to 'file_content'
            csv_file_content = data.get('csv_file_content') or data.get('file_content')

            # --- DEBUG ---
            print(f"Agent: {agent}")
            print(f"Message: {message_content[:100]}...") # Log first 100 chars
            print(f"File Name: {file_name}")
            print(f"File Content Received: {'Yes' if csv_file_content else 'No'}")
            # --- END DEBUG ---

            # If CSV content is present, automatically set agent to 'prediction'
            if csv_file_content:
                agent = 'prediction'
                file_content = csv_file_content # Use the CSV content
                print(f"CSV content detected. Agent override to: {agent}")

        else:
             return JsonResponse({"error": f"Unsupported content type: {request.content_type}"}, status=415)

        # --- DEBUG ---
        print(f"Agent: {agent}")
        print(f"Message: {message_content[:100] if message_content else ''}...") # Log first 100 chars
        print(f"File Name: {file_name}")
        print(f"File Content Received: {'Yes' if file_content else 'No'}")
        # --- END DEBUG ---

        if not message_content and not file_content:
            return JsonResponse({"error": "Message or file content is empty."}, status=400)

        session = await ChatSession.objects.aget(id=session_id, user=user)
        # Save user message first
        await ChatMessage.objects.acreate(
            session=session,
            role='user',
            content=message_content,
            file_name=file_name,
            file_content=file_content
        )
        
        # Check if this is the first message to generate a title
        is_first_message = not await ChatMessage.objects.filter(session=session).aexists() == 1

        thread_id = str(session.thread_id) if session.thread_id else str(uuid.uuid4())
        if not session.thread_id:
            session.thread_id = uuid.UUID(thread_id)
            await session.asave()

        # For local development, directly target the local FastAPI server.
    # For production, change this back to: os.getenv("FASTAPI_SERVER_URL", "http://127.0.0.1:8001")
        fastapi_url = "http://127.0.0.1:8001"

        async def event_stream():
            # Stream title first if it's the first message
            if is_first_message:
                async for title_event in _stream_and_save_title(session, message_content):
                    yield title_event

            # Then, proceed with the main agent response streaming
            final_ai_content = ""
            final_tool_data = None
            current_tool_state = {"assistant": {"messages": []}}
            try:
                fastapi_base_url = "http://127.0.0.1:8001"
                internal_secret = os.getenv("FASTAPI_INTERNAL_SECRET")
                headers = {"X-Internal-Secret": internal_secret} if internal_secret else {}

                async with httpx.AsyncClient() as client:
                    # Prepare payload for FastAPI
                    payload_input = {"messages": [{"role": "user", "content": message_content}]}
                    if file_content:
                        try:
                            decoded_content = base64.b64decode(file_content).decode('utf-8')
                            payload_input["csv_file_content"] = decoded_content
                        except Exception as e:
                            print(f"Error decoding base64 content: {e}")
                            payload_input["csv_file_content"] = "ERROR: Invalid file content"
                        if not message_content:
                            payload_input["messages"] = [{"content": f"첨부된 파일 '{file_name}'을 분석해줘."}]

                    payload = {
                        "input": payload_input,
                        "config": {"configurable": {"thread_id": str(uuid.uuid4())}} # Use a new thread_id for each turn to avoid context pollution
                    }
                    
                    request_url = f"{fastapi_base_url}/prediction/invoke" if agent == 'prediction' else f"{fastapi_base_url}/default/invoke"
                    
                    print(f"---[ Calling FastAPI Server ]---URL: {request_url}")

                    final_content_to_save = ""
                    
                    async with client.stream("POST", request_url, json=payload, headers=headers, timeout=300.0) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            print(f"---[RAW FASTAPI LINE]---> {line}")
                            if not line.startswith("data:"):
                                continue
                            
                            json_str = line[len("data:"):].strip()
                            if not json_str:
                                continue

                            try:
                                data = json.loads(json_str)
                                # FastAPI server is not streaming, but sending the final response in a single JSON.
                                # We need to parse this JSON and extract the assistant's message.
                                if "messages" in data and isinstance(data["messages"], list):
                                    for message in data["messages"]:
                                        if isinstance(message, dict) and message.get("role") == "assistant":
                                            content_to_stream = message.get("content")
                                            if content_to_stream:
                                                final_content_to_save += content_to_stream
                                                # Send the full content as a single chunk to the frontend
                                                sse_payload = {"event": "message_chunk", "data": content_to_stream}
                                                yield f"data: {json.dumps(sse_payload)}\n\n"
                                                break # Found the assistant message, no need to check others

                            except json.JSONDecodeError:
                                print(f"---[STREAM WARNING] Failed to decode JSON: {json_str}")
                                continue
                    
                    yield f"data: {json.dumps({'event': 'stream_end'})}\n\n"

                    if final_content_to_save:
                        await ChatMessage.objects.acreate(
                            session=session,
                            role='assistant',
                            content=final_content_to_save.strip()
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


@login_required
@require_POST
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
