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

api_urlpatterns = [
    path('health-check/', health_check, name='health-check'),
    # path('accounts/', include('accounts.urls')), # Moved to main urlpatterns
    path('conversations/', include('conversations.urls')),
    path('knowledge/', include('knowledge.urls')),
]

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(api_urlpatterns)),
    path('accounts/', include('accounts.urls')), # Added accounts urls here
    
    # Django template views
    path('', home_view, name='home'),  # Django home template
    path('_header.html', TemplateView.as_view(template_name="_header.html"), name='header_partial'),
    
    # Define other Django template routes before the catch-all
    # e.g., path('django-pages/example/', example_view, name='example'),
    
    # Next.js specific routes (if you want certain paths to always go to Next.js)
    path('nextjs/', nextjs_page(stream=True), name='nextjs_root'),
    
    # Catch-all for remaining Next.js pages (keep this last)
    re_path(r'^(?!api/|admin/|accounts/|_header\.html|nextjs/).*$', nextjs_page(stream=True)),
]
