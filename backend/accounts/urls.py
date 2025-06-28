from django.urls import path, re_path, include # Added include
from rest_framework_simplejwt.views import TokenRefreshView
from . import views

app_name = 'accounts'  # Define the application namespace

# Define API URL patterns separately for clarity
api_urlpatterns = [
    # 유연한 URL 패턴 사용 - 슬래시 유무 상관없이 처리
    re_path(r'^login/?$', views.login_view, name='api-login'), # Renamed for clarity
    re_path(r'^logout/?$', views.logout_view, name='api-logout'), # Renamed for clarity
    re_path(r'^me/?$', views.user_detail, name='api-user-detail'), # Renamed for clarity
    re_path(r'^profile/?$', views.update_profile, name='api-update-profile'), # Renamed for clarity
    re_path(r'^token/refresh/?$', TokenRefreshView.as_view(), name='api-token-refresh'), # Renamed for clarity
]

urlpatterns = [
    # User-facing authentication pages
    path('login_page/', views.login_page_view, name='login_page'),
    path('logout_page/', views.logout_page_view, name='logout_page'),

    # Web page routes
    path('profile/', views.profile_view, name='profile'), # New view for profile.html
    path('settings/', views.settings_view, name='settings'), # New view for setting.html
    path('delete_github_token/', views.delete_github_token, name='delete_github_token'),
    path('github/repositories/', views.list_github_repositories, name='list_github_repositories'),
    path('github/connect/', views.github_connect_view, name='github_connect'),
    path('github/disconnect/', views.github_disconnect_view, name='github_disconnect'),
    path('github/add_by_url/', views.add_repository_by_url, name='add_repository_by_url'),
    path('scan-selected-repositories/', views.scan_selected_repositories_view, name='scan_selected_repositories'),
    path('list-branches-for-url/', views.list_branches_for_url_view, name='list_branches_for_url'),
    path('github/list_branches/', views.ajax_list_repository_branches, name='ajax_list_repository_branches'),
    path('list-public-repository-branches/', views.list_public_repository_branches, name='list_public_repository_branches'), # New URL for public repo branches

    # Include API routes under 'api/' prefix
    path('api/', include(api_urlpatterns)),
]