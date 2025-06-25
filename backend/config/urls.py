"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
    path('', TemplateView.as_view(template_name="home.html"), name='home'),
    path('_header.html', TemplateView.as_view(template_name="_header.html"), name='header_partial'), # Serve _header.html for JS fetch
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include, re_path
from django.http import JsonResponse
from django.shortcuts import render # Added for home_view
from django.views.generic import TemplateView # Added to serve _header.html
from django_nextjs.views import nextjs_page

# Existing health_check view
def health_check(request):
    return JsonResponse({'status': 'ok'})

# View for rendering home.html
def home_view(request):
    return render(request, 'home.html')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view, name='home'),  # Root URL for home.html
    path('_header.html', TemplateView.as_view(template_name="_header.html"), name='header_partial'), # Serve _header.html for JS fetch

    # Application URLs: each app's urls.py will define specific web and API routes.
    # For example, accounts.urls might define 'profile/' for web and 'api/login/' for API.
    # This will result in paths like '/accounts/profile/' and '/accounts/api/login/'.
    path('accounts/', include('accounts.urls')),  # Ensuring no hidden characters

    # Catch-all for the Next.js app must come before other apps that might catch similar patterns
    path('chatbot/', nextjs_page(stream=True), name='chatbot-root'),
    path('chatbot/<path:path>', nextjs_page(stream=True), name='chatbot-path'),

    path('conversations/', include('conversations.urls')), # Assuming app name is 'conversations'
    path('knowledge/', include('knowledge.urls')),     # Added for knowledge app

    # Health check URL (kept original re_path for flexibility if needed)
    re_path(r'^api/health-check/?$', health_check, name='health-check'),
]
