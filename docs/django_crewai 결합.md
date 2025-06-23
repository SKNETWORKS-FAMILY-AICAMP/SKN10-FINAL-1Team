CrewAI 스트리밍과 Django 연동 방법
1. CrewAI 스트리밍 설정
먼저 CrewAI에서 스트리밍을 활성화해야 합니다:

from crewai import LLM

# 스트리밍이 활성화된 LLM 생성
llm = LLM(
    model="openai/gpt-4o",
    stream=True  # 스트리밍 활성화
)

2. 이벤트 리스너를 통한 실시간 스트리밍
CrewAI의 이벤트 시스템을 활용하여 스트리밍 청크를 실시간으로 처리할 수 있습니다:

from crewai.utilities.events import LLMStreamChunkEvent
from crewai.utilities.events.base_event_listener import BaseEventListener
import asyncio
import json

class DjangoChatListener(BaseEventListener):
    def __init__(self, websocket_manager=None):
        super().__init__()
        self.websocket_manager = websocket_manager
        
    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(LLMStreamChunkEvent)
        def on_llm_stream_chunk(self, event: LLMStreamChunkEvent):
            # 각 청크를 Django 웹소켓으로 전송
            if self.websocket_manager:
                chunk_data = {
                    'type': 'stream_chunk',
                    'content': event.chunk,
                    'timestamp': event.timestamp.isoformat()
                }
                # Django Channels를 통해 웹소켓으로 전송
                asyncio.create_task(
                    self.websocket_manager.send_to_group(
                        'chat_room', 
                        json.dumps(chunk_data)
                    )
                )

# 리스너 인스턴스 생성
chat_listener = DjangoChatListener()

3. Django Channels를 활용한 웹소켓 설정
Django에서 실시간 통신을 위해 Django Channels를 사용합니다:

# consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from crewai import Agent, Task, Crew
from your_app.crewai_integration import DjangoChatListener

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = 'chat_room'
        self.room_group_name = f'chat_{self.room_name}'
        
        # 그룹에 참가
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        # 그룹에서 나가기
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        
        # CrewAI 에이전트 실행 (비동기)
        await self.run_crewai_agent(message)

    async def run_crewai_agent(self, user_message):
        # 웹소켓 매니저를 리스너에 전달
        chat_listener = DjangoChatListener(websocket_manager=self)
        
        # 스트리밍이 활성화된 LLM으로 에이전트 생성
        llm = LLM(model="openai/gpt-4o", stream=True)
        
        agent = Agent(
            role='챗봇 어시스턴트',
            goal='사용자의 질문에 도움이 되는 답변 제공',
            backstory='친근하고 도움이 되는 AI 어시스턴트입니다.',
            llm=llm,
            verbose=True
        )
        
        task = Task(
            description=f"사용자 질문에 답변: {user_message}",
            expected_output="도움이 되는 상세한 답변",
            agent=agent
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )
        
        # 크루 실행 (스트리밍 응답이 이벤트로 전송됨)
        result = crew.kickoff()
        
        # 최종 결과 전송
        await self.send_to_group('chat_room', json.dumps({
            'type': 'final_result',
            'content': str(result),
            'timestamp': datetime.now().isoformat()
        }))

    async def send_to_group(self, group_name, message):
        await self.channel_layer.group_send(
            f'chat_{group_name}',
            {
                'type': 'chat_message',
                'message': message
            }
        )

    async def chat_message(self, event):
        message = event['message']
        await self.send(text_data=message)

4. Django 설정
# settings.py
INSTALLED_APPS = [
    # ... 기존 앱들
    'channels',
    'your_chat_app',
]

ASGI_APPLICATION = 'your_project.asgi.application'

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379)],
        },
    },
}

# asgi.py
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from your_chat_app.routing import websocket_urlpatterns

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            websocket_urlpatterns
        )
    ),
})

5. 프론트엔드 JavaScript
// chat.js
const chatSocket = new WebSocket(
    'ws://' + window.location.host + '/ws/chat/'
);

chatSocket.onmessage = function(e) {
    const data = JSON.parse(e.data);
    
    if (data.type === 'stream_chunk') {
        // 스트리밍 청크를 실시간으로 화면에 추가
        appendStreamChunk(data.content);
    } else if (data.type === 'final_result') {
        // 최종 결과 처리
        finalizeChatResponse();
    }
};

function appendStreamChunk(chunk) {
    const chatMessages = document.querySelector('#chat-messages');
    const currentMessage = document.querySelector('.streaming-message');
    
    if (!currentMessage) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'streaming-message';
        chatMessages.appendChild(messageDiv);
    }
    
    // 청크를 기존 메시지에 추가
    document.querySelector('.streaming-message').innerHTML += chunk;
    
    // 스크롤을 맨 아래로
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function sendMessage() {
    const messageInput = document.querySelector('#message-input');
    const message = messageInput.value;
    
    chatSocket.send(JSON.stringify({
        'message': message
    }));
    
    messageInput.value = '';
}

6. HTML 템플릿
<!-- chat.html -->
<!DOCTYPE html>
<html>
<head>
    <title>CrewAI 챗봇</title>
    <style>
        #chat-messages {
            height: 400px;
            overflow-y: scroll;
            border: 1px solid #ccc;
            padding: 10px;
            margin-bottom: 10px;
        }
        .streaming-message {
            background-color: #f0f0f0;
            padding: 5px;
            margin: 5px 0;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div id="chat-messages"></div>
    <input type="text" id="message-input" placeholder="메시지를 입력하세요...">
    <button onclick="sendMessage()">전송</button>
    
    <script src="{% static 'js/chat.js' %}"></script>
</body>
</html>

주요 특징
실시간 스트리밍: CrewAI의 LLMStreamChunkEvent를 통해 응답을 실시간으로 스트리밍
웹소켓 연동: Django Channels를 통한 실시간 양방향 통신
이벤트 기반: CrewAI의 이벤트 시스템을 활용한 깔끔한 연동
확장 가능: 다양한 CrewAI 이벤트를 추가로 처리 가능