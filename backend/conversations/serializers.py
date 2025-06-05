from rest_framework import serializers
from .models import ChatSession, ChatMessage, LlmCall, AgentType


class ChatMessageSerializer(serializers.ModelSerializer):
    """ChatMessage 모델 직렬화를 위한 serializer"""
    
    class Meta:
        model = ChatMessage
        fields = ['id', 'session', 'role', 'content', 'created_at']
        read_only_fields = ['id', 'created_at']


class LlmCallSerializer(serializers.ModelSerializer):
    """LlmCall 모델 직렬화를 위한 serializer"""
    
    class Meta:
        model = LlmCall
        fields = ['id', 'session', 'provider', 'model', 'prompt_tokens', 
                 'completion_tokens', 'cost_usd', 'latency_ms', 'called_at']
        read_only_fields = ['id', 'called_at']


class ChatSessionSerializer(serializers.ModelSerializer):
    """ChatSession 모델 직렬화를 위한 serializer"""
    messages = ChatMessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = ChatSession
        fields = ['id', 'user', 'agent_type', 'started_at', 'ended_at', 'title', 'messages']
        read_only_fields = ['id', 'started_at']

    def validate_agent_type(self, value):
        """agent_type 필드 유효성 검사"""
        if value not in [choice[0] for choice in AgentType.choices]:
            raise serializers.ValidationError(f"유효하지 않은 에이전트 유형입니다. 유효한 값: {AgentType.choices}")
        return value
