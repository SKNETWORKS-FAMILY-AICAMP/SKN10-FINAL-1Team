from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from dotenv import load_dotenv
from django.contrib.auth import logout
from django.contrib.auth import login
import requests
import os

# Create your views here.
def user_login(request) :
    load_dotenv()
    client_id = os.getenv("GITHUB_CLIENT_ID")
    # redirect_uri : 로그인을 처리하는 url
    redirect_uri = 'http://localhost:8000/user/login_process'
    github_auth_url = f'https://github.com/login/oauth/authorize?client_id={client_id}&redirect_uri={redirect_uri}&scope=read:user'
    return redirect(github_auth_url)

def user_login_process(request) :
    code = request.GET.get('code')
    if not code:
        return render(request, "user/login_fail.html")

    # 1. access token 요청
    token_url = 'https://github.com/login/oauth/access_token'
    data = {
        'client_id': os.getenv("GITHUB_CLIENT_ID"),
        'client_secret': os.getenv("GITHUB_CLIENT_SECRET"),
        'code': code,
    }
    headers = {'Accept': 'application/json'}
    res = requests.post(token_url, data=data, headers=headers)
    access_token = res.json().get('access_token')

    if not access_token:
        return render(request, "user/login_fail.html")

    # 2. user_info = 계정 정보
    user_info = requests.get(
        'https://api.github.com/user',
        headers={'Authorization': f'token {access_token}'}
    ).json()

    print(user_info)
    # 3. user_info중에서 username와 프로필 사진을 받아옴.
    username = user_info.get("login")
    image = user_info.get("avatar_url")
    if not username:
        return render(request, "user/login_fail.html")

    # 4. Django 유저로 연결 (없으면 생성)
    user, _ = User.objects.get_or_create(username=username)
    login(request, user)  # 여기서 Django 세션 로그인 발생
    return render(request, "code_seeker/index.html", {"username": username, "image" : image})

def user_logout(request):
    logout(request)  # 세션에서 사용자 로그아웃
    request.session.flush()  # 세션 데이터 초기화
    return redirect('homepage')  # 로그아웃 후 로그인 페이지로 이동