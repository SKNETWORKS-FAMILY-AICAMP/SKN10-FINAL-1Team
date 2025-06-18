from django.utils import timezone
from django.shortcuts import get_object_or_404, render # Added render
from rest_framework import status, viewsets
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.decorators import login_required # Added for view protection
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ChatSession, ChatMessage
from .serializers import ChatSessionSerializer, ChatMessageSerializer


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def get_chat_messages(request, session_pk):
    """
    API view to get all messages for a specific chat session or post a new message.
    """
    session = get_object_or_404(ChatSession, pk=session_pk, user=request.user)

    if request.method == 'GET':
        messages = ChatMessage.objects.filter(session=session).order_by('created_at')
        serializer = ChatMessageSerializer(messages, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        content = request.data.get('content')
        if not content:
            return Response({'error': 'Content is required'}, status=status.HTTP_400_BAD_REQUEST)

        # 1. Save user's message
        user_message = ChatMessage.objects.create(
            session=session,
            role='user',
            content=content
        )

        # 2. Create and save assistant's response (simple echo for now)
        # In a real scenario, this would involve calling an AI model
        assistant_response_content = f"I received your message: \"{content}\". I am a simple echo bot for now."
        assistant_message = ChatMessage.objects.create(
            session=session,
            role='assistant',
            content=assistant_response_content
        )

        # 3. Serialize both new messages and return them
        serializer = ChatMessageSerializer([user_message, assistant_message], many=True)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


@login_required
def chatbot_view(request):
    """Renders the chatbot page and displays a list of chat sessions with initial messages."""
    sessions = ChatSession.objects.filter(user=request.user).order_by('-started_at')
    
    initial_messages = []
    active_session_id = None
    if sessions.exists():
        active_session = sessions.first()
        active_session_id = active_session.id
        initial_messages = ChatMessage.objects.filter(session=active_session).order_by('created_at')

    context = {
        'sessions': sessions,
        'initial_messages': initial_messages,
        'active_session_id': active_session_id,
    }
    return render(request, 'conversations/chatbot.html', context)
