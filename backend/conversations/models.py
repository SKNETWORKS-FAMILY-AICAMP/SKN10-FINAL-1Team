"""conversations/models.py  –  채팅·LLM 호출"""

import uuid
from django.db import models
from accounts.models import User


class AgentType(models.TextChoices):
    CODE = "code", "Code"
    RAG = "rag", "RAG"
    ANALYTICS = "analytics", "Analytics"


class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="chat_sessions")
    agent_type = models.CharField(max_length=20, choices=AgentType.choices)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "chat_sessions"


class ChatMessage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=20)  # user | assistant | system
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "chat_messages"
        ordering = ["created_at"]


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

    class Meta:
        db_table = "llm_calls"
        indexes = [models.Index(fields=["called_at"], name="idx_llm_called_at")]
