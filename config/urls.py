from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include("code_seeker.urls")),
    path('user/', include("user.urls")),
]
