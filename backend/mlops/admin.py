from django.contrib import admin
from django.utils.html import format_html
from .models import AnalyticsResult, ModelArtifact


@admin.register(AnalyticsResult)
class AnalyticsResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'result_type', 'created_at', 's3_key_preview')
    list_filter = ('result_type', 'created_at')
    search_fields = ('user__email', 's3_key', 'meta')
    readonly_fields = ('created_at', 'meta_prettified')
    list_select_related = ('user',)
    
    def s3_key_preview(self, obj):
        return obj.s3_key[:50] + '...' if len(obj.s3_key) > 50 else obj.s3_key
    s3_key_preview.short_description = 'S3 Key'
    
    def meta_prettified(self, obj):
        import json
        from pygments import highlight
        from pygments.lexers import JsonLexer
        from pygments.formatters import HtmlFormatter
        from django.utils.safestring import mark_safe
        
        if not obj.meta:
            return ""
            
        response = json.dumps(obj.meta, indent=2, ensure_ascii=False)
        response = response[:5000]  # Limit the size to prevent performance issues
        
        # Truncate and add ellipsis if necessary
        if len(response) > 5000:
            response = response[:5000] + '... (truncated)'
            
        # Format the JSON
        formatter = HtmlFormatter(style='colorful')
        response = highlight(response, JsonLexer(), formatter)
        style = "<style>" + formatter.get_style_defs() + "</style><br>"
        return mark_safe(style + response)
    
    meta_prettified.short_description = 'Metadata'


@admin.register(ModelArtifact)
class ModelArtifactAdmin(admin.ModelAdmin):
    list_display = ('name', 'version', 'stage', 'created_by', 'created_at', 's3_key_preview')
    list_filter = ('stage', 'created_at')
    search_fields = ('name', 'version', 's3_key')
    readonly_fields = ('created_at', 'metrics_prettified')
    list_select_related = ('created_by',)
    
    def s3_key_preview(self, obj):
        return obj.s3_key[:50] + '...' if len(obj.s3_key) > 50 else obj.s3_key
    s3_key_preview.short_description = 'S3 Key'
    
    def metrics_prettified(self, obj):
        if not obj.metrics:
            return ""
            
        import json
        from pygments import highlight
        from pygments.lexers import JsonLexer
        from pygments.formatters import HtmlFormatter
        from django.utils.safestring import mark_safe
        
        response = json.dumps(obj.metrics, indent=2, ensure_ascii=False)
        response = response[:5000]  # Limit the size to prevent performance issues
        
        # Truncate and add ellipsis if necessary
        if len(response) > 5000:
            response = response[:5000] + '... (truncated)'
            
        # Format the JSON
        formatter = HtmlFormatter(style='colorful')
        response = highlight(response, JsonLexer(), formatter)
        style = "<style>" + formatter.get_style_defs() + "</style><br>"
        return mark_safe(style + response)
    
    metrics_prettified.short_description = 'Metrics'
