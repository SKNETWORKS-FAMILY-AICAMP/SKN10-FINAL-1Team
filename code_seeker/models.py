from django.db import models
import uuid

class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255, blank=True)  # 예: "상담 세션", "GPT와의 대화"
    created_at = models.DateTimeField(auto_now_add=True)

class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, related_name='messages', on_delete=models.CASCADE)
    is_human = models.BooleanField()
    message = models.TextField() # 메시지 내용
    timestamp = models.DateTimeField(auto_now_add=True)  # 대화 시간