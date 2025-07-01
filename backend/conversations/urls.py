from django.urls import path
from .views import (
    session_create_view,
    session_delete_view,
    session_list_view,
    message_list_view,
    chat_stream
)

app_name = 'conversations'

urlpatterns = [
    path('sessions/', session_list_view, name='session-list'),
    path('sessions/create/', session_create_view, name='session-create'),
    path('sessions/<uuid:session_id>/delete/', session_delete_view, name='session-delete'),
    path('sessions/<uuid:session_id>/messages/', message_list_view, name='message-list'),
    path('sessions/<uuid:session_id>/invoke/', chat_stream, name='chat-stream'),
]
