from django.contrib import admin
from django.urls import path, include
from .views import *

urlpatterns = [
    path('', index, name='homepage'),
    path('delete_message/', delete_message, name="delete_message")
]