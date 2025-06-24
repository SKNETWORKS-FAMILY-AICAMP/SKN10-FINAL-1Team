from django.urls import path, include
from . import views

app_name = 'knowledge'  # Define the application namespace

# Define API URL patterns separately for clarity
# These are based on the Django-Next.js 연결 종합 가이드 memory
# and assume corresponding views will be created in views.py
api_urlpatterns = [
    # Document related URLs
    path('documents/', views.document_list_create_view, name='api-document-list-create'), # For GET (list) and POST (create)
    path('documents/<uuid:pk>/', views.document_detail_view, name='api-document-detail'), # For GET (retrieve), PUT/PATCH (update), DELETE
    path('documents/<uuid:pk>/summary/', views.document_summary_view, name='api-document-summary'), # For GET (summary)
    
    # Repository related URLs (assuming similar patterns to documents if needed)
    # path('repositories/', views.repository_list_create_view, name='api-repository-list-create'),
    # path('repositories/<uuid:pk>/', views.repository_detail_view, name='api-repository-detail'),
    # path('repositories/<uuid:pk>/files/', views.repository_files_view, name='api-repository-files'),
    
    # File related URLs (assuming similar patterns if needed)
    # path('files/<uuid:pk>/content/', views.file_content_view, name='api-file-content'),

    # Search URL (from Pinecone integration example)
    path('search/', views.search_documents_view, name='api-search-documents'), # For POST (search)
]

urlpatterns = [
    # Web page route for dashboard
    path('dashboard/', views.dashboard_view, {"screen_type" : "home"}, name='dashboard'), # New view for dashboard.html
    path('dashboard/<str:screen_type>', views.dashboard_view, name='dashboard'), # New view for dashboard.html
    path('dashboard/create_index/', views.create_index, name='create_index'),
    path('dashboard/delete_index/', views.delete_index, name='delete_index'),
    path('dashboard/delete_user/', views.delete_user, name='delete_user'),
    path('dashboard/index_detail/<str:index_name>', views.index_detail, name='index_detail'),
    path('dashboard/session_detail/<uuid:session_id>', views.session_detail, name='session_detail'),

    # Include API routes under 'api/' prefix
    path('api/', include(api_urlpatterns)),
]
