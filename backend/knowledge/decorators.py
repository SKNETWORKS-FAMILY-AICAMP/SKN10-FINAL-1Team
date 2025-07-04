from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages
from rest_framework import permissions

def admin_required(view_func):
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            # 로그인되지 않은 사용자는 로그인 페이지로 리디렉션
            return redirect('accounts:login')

        user = request.user
        # 조직 이름이 'administrator'이거나 역할이 'admin'인지 확인
        is_admin_by_org = user.org and user.org.name.lower() == 'administrator'
        is_admin_by_role = user.role == 'admin'

        if is_admin_by_org or is_admin_by_role or user.is_superuser:
            # 권한이 있으면 원래 뷰 함수 실행
            return view_func(request, *args, **kwargs)
        else:
            # 권한이 없으면 에러 메시지와 함께 이전 페이지 또는 홈페이지로 리디렉션
            messages.error(request, '🚫 이 페이지에 접근할 권한이 없습니다.')
            # 'HTTP_REFERER'를 사용하여 이전 페이지로 리디렉션, 없으면 홈페이지로
            referer = request.META.get('HTTP_REFERER', '/')
            return redirect(referer)
    return _wrapped_view 

class IsAdminUser(permissions.BasePermission):
    """
    Allows access only to admin users (by role or organization).
    """
    message = '🚫 이 API에 접근할 권한이 없습니다.'

    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        
        user = request.user
        is_admin_by_org = user.org and user.org.name.lower() == 'administrator'
        is_admin_by_role = user.role == 'admin'
        
        return is_admin_by_org or is_admin_by_role or user.is_superuser 