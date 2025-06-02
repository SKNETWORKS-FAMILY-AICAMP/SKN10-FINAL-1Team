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
    title = models.CharField(max_length=60, default="새 세션")

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


class CheckpointBlob(models.Model):
    thread_id = models.TextField(primary_key=True) # Django가 식별자로 사용하도록 primary_key=True 설정
    checkpoint_ns = models.TextField(default='')
    channel = models.TextField()
    version = models.TextField()
    type = models.TextField()
    blob = models.BinaryField(null=True, blank=True)

    class Meta:
        db_table = 'checkpoint_blobs'
        # 실제 DB의 복합 고유성은 unique_together로 정의
        unique_together = (('thread_id', 'checkpoint_ns', 'channel', 'version'),)
        managed = False

    def __str__(self):
        return f"{self.thread_id} - {self.checkpoint_ns} - {self.channel} - {self.version}"

class CheckpointMigration(models.Model):
    v = models.IntegerField(primary_key=True)

    class Meta:
        db_table = 'checkpoint_migrations'
        managed = False

    def __str__(self):
        return str(self.v)

class CheckpointWrite(models.Model):
    thread_id = models.TextField(primary_key=True) # Django가 식별자로 사용하도록 primary_key=True 설정
    checkpoint_ns = models.TextField(default='')
    checkpoint_id = models.TextField()
    task_id = models.TextField()
    idx = models.IntegerField()
    channel = models.TextField()
    type = models.TextField(null=True, blank=True)
    blob = models.BinaryField()

    class Meta:
        db_table = 'checkpoint_writes'
        unique_together = (('thread_id', 'checkpoint_ns', 'checkpoint_id', 'task_id', 'idx'),)
        managed = False

    def __str__(self):
        return f"{self.thread_id} - {self.checkpoint_id} - {self.task_id} - {self.idx}"

class Checkpoint(models.Model):
    thread_id = models.TextField(primary_key=True) # Django가 식별자로 사용하도록 primary_key=True 설정
    checkpoint_ns = models.TextField(default='')
    checkpoint_id = models.TextField()
    parent_checkpoint_id = models.TextField(null=True, blank=True)
    type = models.TextField(null=True, blank=True)
    checkpoint = models.JSONField()
    metadata = models.JSONField(default=dict)

    class Meta:
        db_table = 'checkpoints'
        unique_together = (('thread_id', 'checkpoint_ns', 'checkpoint_id'),)
        managed = False

    def __str__(self):
        return f"{self.thread_id} - {self.checkpoint_ns} - {self.checkpoint_id}"