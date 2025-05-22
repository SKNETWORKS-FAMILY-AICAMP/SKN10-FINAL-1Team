from django.db import models
import uuid
class ChatMessage(models.Model):
    chat_id = models.UUIDField(default=uuid.uuid4, editable=False)  # 세션 구분용 고유 ID
    is_human = models.BooleanField()
    message = models.TextField() # 메시지 내용
    timestamp = models.DateTimeField(auto_now_add=True)  # 대화 시간