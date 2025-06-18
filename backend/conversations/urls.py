from django.urls import path, re_path, include # Added include
from . import views

app_name = 'conversations'  # Define the application namespace

# Define API URL patterns separately for clarity
api_urlpatterns = [
    path('sessions/<uuid:session_pk>/messages/', views.get_chat_messages, name='api-get-chat-messages'),
]

urlpatterns = [
    # Web page route for chatbot
    path('chatbot/', views.chatbot_view, name='chatbot'), # New view for chatbot.html

    # Include API routes under 'api/' prefix
    path('api/', include(api_urlpatterns)),
]
