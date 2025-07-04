from django.contrib import admin
from django.utils.html import format_html
from .models import Document, GitRepository, CodeFile, EmbedChunk, TelecomCustomers, SummaryNewsKeywords


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'doc_type', 'org', 'version', 'created_at')
    list_filter = ('doc_type', 'org')
    search_fields = ('title', 's3_key')
    list_select_related = ('org', 'uploaded_by')
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'


@admin.register(GitRepository)
class GitRepositoryAdmin(admin.ModelAdmin):
    list_display = ('repo_url', 'org', 'default_branch', 'fetched_at')
    list_filter = ('org',)
    search_fields = ('repo_url',)
    readonly_fields = ('fetched_at',)


@admin.register(CodeFile)
class CodeFileAdmin(admin.ModelAdmin):
    list_display = ('file_path', 'repo', 'language', 'loc')
    list_filter = ('repo', 'language')
    search_fields = ('file_path', 'latest_commit')
    list_select_related = ('repo',)
    readonly_fields = ('id',)


@admin.register(EmbedChunk)
class EmbedChunkAdmin(admin.ModelAdmin):
    list_display = ('id', 'chunk_index', 'document', 'file', 'hash_short')
    list_filter = ('document', 'file')
    search_fields = ('hash', 'pinecone_id')
    readonly_fields = ('id',)
    list_select_related = ('document', 'file')
    
    def hash_short(self, obj):
        return f"{obj.hash[:10]}..." if obj.hash else ""
    hash_short.short_description = 'Hash'





@admin.register(TelecomCustomers)
class TelecomCustomersAdmin(admin.ModelAdmin) :
    list_display = ('id', 'customer_id', 'gender', 'partner', 'dependents' ,'churn')
    list_filter = ('dependents','churn', 'gender')
    search_fields = ('id','customer_id')
    readonly_fields = ('id', 'customer_id')


@admin.register(SummaryNewsKeywords)
class SummaryNewsKeywordsAdmin(admin.ModelAdmin):
    list_display = ('date', 'keyword_display', 'title_short', 'url_short')
    list_filter = ('date', 'keyword')
    search_fields = ('title', 'summary', 'keyword')
    list_select_related = ()
    date_hierarchy = 'date'
    ordering = ('-date', 'keyword')
    
    def keyword_display(self, obj):
        return obj.keyword
    keyword_display.short_description = 'Keyword'
    
    def title_short(self, obj):
        return f"{obj.title[:50]}..." if len(obj.title) > 50 else obj.title
    title_short.short_description = 'Title'
    
    def url_short(self, obj):
        return format_html('<a href="{}" target="_blank">Link</a>', obj.url)
    url_short.short_description = 'URL'
