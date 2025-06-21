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

        fastapi_url = os.environ.get("FASTAPI_SERVER_URL", "http://127.0.0.1:8001")

        async def event_stream():
            # Stream title first if it's the first message
            if is_first_message:
                async for title_event in _stream_and_save_title(session, message_content):
                    yield title_event

            # Then, proceed with the main agent response streaming
            final_ai_content = ""
            final_tool_data = None
            
            try:
                internal_secret = os.getenv("FASTAPI_INTERNAL_SECRET")
                headers = {"X-Internal-Secret": internal_secret} if internal_secret else {}
                async with httpx.AsyncClient() as client:
                    payload = {
                        "input": {"messages": [{"role": "user", "content": message_content}]},
                        "config": {"configurable": {"thread_id": thread_id}}
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

                                        # Stream pure text content only if it's from the 'agent' node
                                        if (metadata.get("langgraph_node") == "agent" and
                                                message_chunk.get("type") == "AIMessageChunk" and
                                                not message_chunk.get("tool_call_chunks")):
                                            content = message_chunk.get("content", "")
                                            if content:
                                                final_ai_content += content
                                                sse_payload = {"event": "message_chunk", "data": content}
                                                yield f"data: {json.dumps(sse_payload)}\n\n"

                                    elif event_type == "updates":
                                        final_tool_data = payload
                                        yield f"data: {json.dumps({'event': 'tool_update', 'data': payload})}\n\n"

                                except (json.JSONDecodeError, IndexError):
                                    continue
                
                if final_ai_content:
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


