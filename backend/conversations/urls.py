from django.urls import path, include
from . import views

app_name = 'conversations'

# API routes are grouped under 'api/'
api_urlpatterns = [
    path('chat/stream/<uuid:session_id>/', views.chat_stream, name='chat_stream'),
    path('session/create/', views.session_create_view, name='session_create'),
]

urlpatterns = [
    # Web page routes
    path('chat/', views.chatbot_view, name='chatbot'),
    path('chat/<uuid:session_id>/', views.chatbot_view, name='chatbot_session'),

    # Include API routes
    path('api/', include((api_urlpatterns, 'api'), namespace='api')),
]
