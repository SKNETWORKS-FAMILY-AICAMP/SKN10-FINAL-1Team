from django.contrib import admin
from django.utils.html import format_html
from .models import Document, GitRepository, CodeFile, EmbedChunk


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
