from django.urls import path, re_path
from rest_framework_simplejwt.views import TokenRefreshView
from . import views

urlpatterns = [
    # 유연한 URL 패턴 사용 - 슬래시 유무 상관없이 처리
    re_path(r'^login/?$', views.login_view, name='login'),
    re_path(r'^logout/?$', views.logout_view, name='logout'),
    re_path(r'^me/?$', views.user_detail, name='user-detail'),
    re_path(r'^profile/?$', views.update_profile, name='update-profile'),
    re_path(r'^token/refresh/?$', TokenRefreshView.as_view(), name='token-refresh'),
]
