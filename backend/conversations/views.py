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


class ChatSessionViewSet(APIView):
    """채팅 세션을 관리하는 API 뷰셋"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """현재 사용자의 모든 채팅 세션 조회"""
        sessions = ChatSession.objects.filter(user=request.user).order_by('-started_at')
        serializer = ChatSessionSerializer(sessions, many=True)
        return Response(serializer.data)

    def post(self, request):
        """새 채팅 세션 생성"""
        # 요청 데이터에 사용자 ID 추가
        data = request.data.copy()
        data['user'] = request.user.id
        
        serializer = ChatSessionSerializer(data=data)
        if serializer.is_valid():
            session = serializer.save()
            
            # 시스템 메시지 생성 (선택적)
            welcome_message = f"Welcome to your new {session.agent_type} session. How can I help you today?"
            ChatMessage.objects.create(
                session=session,
                role="system",
                content=welcome_message
            )
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ChatSessionDetailView(APIView):
    """특정 채팅 세션을 관리하는 API 뷰"""
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        """특정 채팅 세션 조회"""
        session = get_object_or_404(ChatSession, pk=pk, user=request.user)
        serializer = ChatSessionSerializer(session)
        return Response(serializer.data)

    def patch(self, request, pk):
        """채팅 세션 업데이트 (주로 종료 시간)"""
        session = get_object_or_404(ChatSession, pk=pk, user=request.user)
        serializer = ChatSessionSerializer(session, data=request.data, partial=True)
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ChatMessageView(APIView):
    """채팅 메시지를 관리하는 API 뷰"""
    permission_classes = [IsAuthenticated]

    def get(self, request, session_pk):
        """특정 세션의 모든 메시지 조회"""
        # 해당 세션이 현재 사용자의 것인지 확인
        session = get_object_or_404(ChatSession, pk=session_pk, user=request.user)
        messages = ChatMessage.objects.filter(session=session).order_by('created_at')
        serializer = ChatMessageSerializer(messages, many=True)
        return Response(serializer.data)

    def post(self, request, session_pk):
        """새 메시지 생성 및 AI 응답 생성"""
        # 해당 세션이 현재 사용자의 것인지 확인
        session = get_object_or_404(ChatSession, pk=session_pk, user=request.user)
        
        # 사용자 메시지 생성
        data = request.data.copy()
        data['session'] = session.pk
        data['role'] = 'user'
        
        serializer = ChatMessageSerializer(data=data)
        if serializer.is_valid():
            user_message = serializer.save()
            
            # 여기서 AI 응답을 생성하는 로직을 추가할 수 있습니다
            # 지금은 간단한 예시 응답을 생성합니다
            ai_response_content = f"This is a mock response to your message: {user_message.content}"
            
            # AI 응답 메시지 생성
            ai_message = ChatMessage.objects.create(
                session=session,
                role="assistant",
                content=ai_response_content
            )
            
            # 사용자 메시지와 AI 응답을 모두 포함하여 반환
            both_messages = ChatMessage.objects.filter(id__in=[user_message.id, ai_message.id])
            messages_serializer = ChatMessageSerializer(both_messages, many=True)
            return Response(messages_serializer.data, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@login_required
def chatbot_view(request):
    """Renders the chatbot page."""
    return render(request, 'conversations/chatbot.html')
