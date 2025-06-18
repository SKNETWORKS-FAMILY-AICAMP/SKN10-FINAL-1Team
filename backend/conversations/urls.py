from django.urls import path, re_path, include # Added include
from . import views

app_name = 'conversations'  # Define the application namespace

# Define API URL patterns separately for clarity
api_urlpatterns = [
    # 채팅 세션 관련 URL
    re_path(r'^sessions/?$', views.ChatSessionViewSet.as_view(), name='api-chat-sessions'), # Renamed
    re_path(r'^sessions/(?P<pk>[0-9a-f-]+)/?$', views.ChatSessionDetailView.as_view(), name='api-chat-session-detail'), # Renamed
    
    # 채팅 메시지 관련 URL
    re_path(r'^sessions/(?P<session_pk>[0-9a-f-]+)/messages/?$', views.ChatMessageView.as_view(), name='api-chat-messages'), # Renamed
]

urlpatterns = [
    # Web page route for chatbot
    path('chatbot/', views.chatbot_view, name='chatbot'), # New view for chatbot.html

    # Include API routes under 'api/' prefix
    path('api/', include(api_urlpatterns)),
]
