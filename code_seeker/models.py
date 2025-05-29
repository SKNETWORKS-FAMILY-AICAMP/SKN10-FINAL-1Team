from django.db import models
import uuid

class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    title = models.CharField(max_length=60, default="새 세션")


class ChatMessage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=20) # system | human
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)



class LlmCall(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="llm_calls")
    provider = models.CharField(max_length=50)
    model = models.CharField(max_length=100)
    prompt_tokens = models.PositiveIntegerField()
    completion_tokens = models.PositiveIntegerField()
    cost_usd = models.DecimalField(max_digits=10, decimal_places=4)
    latency_ms = models.PositiveIntegerField(null=True, blank=True)
    called_at = models.DateTimeField(auto_now_add=True)


