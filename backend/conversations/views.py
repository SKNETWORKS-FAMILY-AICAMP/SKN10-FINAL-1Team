import json
import uuid
from django.http import StreamingHttpResponse, Http404, JsonResponse
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .models import ChatSession, ChatMessage, AgentType
from .services import client
from django.views.decorators.http import require_POST

@login_required
def chatbot_view(request, session_id=None):
    user = request.user
    sessions = ChatSession.objects.filter(user=user).order_by('-started_at')

    if session_id:
        active_session = get_object_or_404(ChatSession, id=session_id, user=user)
        initial_messages = active_session.messages.all()
    else:
        active_session = sessions.first()
        if active_session:
            initial_messages = active_session.messages.all()
        else:
            initial_messages = []

    context = {
        'sessions': sessions,
        'active_session_id': active_session.id if active_session else None,
        'initial_messages': initial_messages,
    }
    return render(request, 'conversations/chatbot.html', context)


@login_required
@require_POST
async def session_create_view(request):
    user = request.user
    # 기본 에이전트 타입으로 새 세션 생성
    session = await ChatSession.objects.acreate(user=user, agent_type=AgentType.DEFAULT)
    return JsonResponse({'session_id': str(session.id)})


async def chat_stream(request, session_id):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message_content = data.get('message')

            user = request.user

            if not message_content:
                return StreamingHttpResponse("data: {\"error\": \"Message content is empty.\"}\n\n", content_type='text/event-stream', status=400)

            session = await ChatSession.objects.aget(id=session_id, user=user)

            await ChatMessage.objects.acreate(session=session, role='user', content=message_content)

            thread_id = str(session.thread_id) if session.thread_id else None

            async def event_stream():
                nonlocal thread_id
                try:
                    final_message_list = None

                    async for chunk in client.runs.stream(
                        thread_id=thread_id,
                        assistant_id="fe096781-5601-53d2-b2f6-0d3403f7e9ca", # TODO: 실제 Assistant ID로 교체
                        input={"messages": [{"role": "user", "content": message_content}]},
                        stream_mode=["messages", "debug"]
                    ):
                        payload = {"event": chunk.event, "data": chunk.data}
                        yield f"data: {json.dumps(payload)}\n\n"

                        if chunk.event == 'metadata' and not thread_id:
                            new_thread_id = chunk.data.get('thread_id')
                            if new_thread_id:
                                thread_id = new_thread_id
                                session.thread_id = uuid.UUID(thread_id)
                                await session.asave()

                        if chunk.event == 'debug' and chunk.data.get('type') == 'task_result':
                            # This event contains the final state of the messages after a task
                            result = chunk.data.get('payload', {}).get('result', [])
                            if result and isinstance(result, list) and isinstance(result[0], list) and result[0][0] == 'messages':
                                final_message_list = result[0][1]

                    # After the stream, process the final message list captured from the last task_result event
                    if final_message_list:
                        final_content = None
                        tool_data_for_db = []

                        # Find the last AI message for the final content
                        for msg in reversed(final_message_list):
                            if msg.get('type') == 'ai' and msg.get('content'):
                                final_content = msg.get('content')
                                break
                        
                        # Collect all tool-related messages (calls and results)
                        tool_data_for_db = [msg for msg in final_message_list if msg.get('type') == 'tool' or (msg.get('type') == 'ai' and msg.get('tool_calls'))]

                        if final_content:
                            await ChatMessage.objects.acreate(
                                session=session,
                                role='assistant',
                                content=final_content,
                                tool_data=tool_data_for_db if tool_data_for_db else None
                            )

                except Exception as e:
                    import traceback
                    print(f"---[STREAM ERROR] An error occurred in event_stream: {e}")
                    traceback.print_exc()
                    yield f"data: {json.dumps({'error': 'An internal error occurred.'})}\n\n"

            return StreamingHttpResponse(event_stream(), content_type='text/event-stream')

        except json.JSONDecodeError:
            return StreamingHttpResponse("data: {\"error\": \"Invalid JSON.\"}\n\n", content_type='text/event-stream', status=400)
        except ChatSession.DoesNotExist:
            raise Http404("Chat session not found")
        except Exception as e:
            return StreamingHttpResponse(f"data: {json.dumps({'error': 'An unexpected error occurred: ' + str(e)})}\n\n", content_type='text/event-stream', status=500)

    return StreamingHttpResponse("data: {\"error\": \"Invalid request method.\"}\n\n", content_type='text/event-stream', status=405)


