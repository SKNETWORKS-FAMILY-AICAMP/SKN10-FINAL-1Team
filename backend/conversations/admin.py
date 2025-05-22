from django.contrib import admin
from django.utils.html import format_html
from .models import ChatSession, ChatMessage, LlmCall


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
