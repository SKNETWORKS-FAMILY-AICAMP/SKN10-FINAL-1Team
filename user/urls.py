from django.contrib import admin
from django.urls import path, include
from .views import *

urlpatterns = [
    path('login/', user_login, name='login'),
    path('login_process/', user_login_process, name='login_process'),
    path('logout/', user_logout, name='logout'),
]