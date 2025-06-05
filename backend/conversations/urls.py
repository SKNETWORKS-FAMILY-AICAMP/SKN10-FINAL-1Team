from django.urls import path, re_path
from . import views

urlpatterns = [
    # 채팅 세션 관련 URL
    re_path(r'^sessions/?$', views.ChatSessionViewSet.as_view(), name='chat_sessions'),
    re_path(r'^sessions/(?P<pk>[0-9a-f-]+)/?$', views.ChatSessionDetailView.as_view(), name='chat_session_detail'),
    
    # 채팅 메시지 관련 URL
    re_path(r'^sessions/(?P<session_pk>[0-9a-f-]+)/messages/?$', views.ChatMessageView.as_view(), name='chat_messages'),
]
