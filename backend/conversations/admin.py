from django.contrib import admin

from django.utils.html import format_html
from .models import ChatSession, ChatMessage, LlmCall
import json # JSONField 내용을 파싱하거나 요약할 때 사용 가능


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ('created_at',)
    fields = ('role', 'content', 'created_at')
    ordering = ('created_at',)


class LlmCallInline(admin.TabularInline):
    model = LlmCall
    extra = 0
    readonly_fields = ('called_at',)
    fields = ('provider', 'model', 'prompt_tokens', 'completion_tokens', 'cost_usd', 'latency_ms', 'called_at')
    ordering = ('-called_at',)


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'agent_type', 'started_at', 'ended_at', 'duration', 'message_count')
    list_filter = ('agent_type', 'started_at')
    search_fields = ('user__email', 'user__name')
    readonly_fields = ('started_at', 'ended_at')
    list_select_related = ('user',)
    inlines = [ChatMessageInline, LlmCallInline]
    
    def duration(self, obj):
        if obj.ended_at:
            duration = obj.ended_at - obj.started_at
            return f"{duration.seconds // 60}m {duration.seconds % 60}s"
        return "Ongoing"
    duration.short_description = 'Duration'
    
    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = 'Messages'


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('truncated_content', 'role', 'session', 'created_at')
    list_filter = ('role', 'created_at')
    search_fields = ('content', 'session__user__email')
    readonly_fields = ('created_at',)
    list_select_related = ('session__user',)
    
    def truncated_content(self, obj):
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content
    truncated_content.short_description = 'Content'


@admin.register(LlmCall)
class LlmCallAdmin(admin.ModelAdmin):
    list_display = ('id', 'session', 'model', 'cost_usd', 'latency_ms', 'called_at')
    list_filter = ('provider', 'model', 'called_at')
    search_fields = ('session__user__email', 'model')
    readonly_fields = ('called_at',)
    list_select_related = ('session__user',)
    
    def has_add_permission(self, request):
        return False  # Prevent manual addition of LLM calls


# 모든 LangGraph 테이블에 적용할 읽기 전용 Admin 클래스
class ReadOnlyAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        # 추가 기능 비활성화
        return False
    def has_change_permission(self, request, obj=None):
        # 변경 기능 비활성화
        return False
    def has_delete_permission(self, request, obj=None):
        # 삭제 기능 비활성화
        return False
    
    # list_display의 항목들이 변경 페이지로 연결되는 링크가 되지 않도록 설정
    # 이렇게 하면 list_display의 각 항목이 링크로 표시되지 않습니다.
    def get_list_display_links(self, request, list_display):
        return None



