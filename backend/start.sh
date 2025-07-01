#!/bin/bash

echo "🚀 Django 애플리케이션 시작 중..."

# 환경변수 확인
echo "🔧 환경변수 확인:"
echo "DEBUG: $DEBUG"
echo "SECRET_KEY 설정: $(if [ -n "$SECRET_KEY" ]; then echo "✅ 설정됨"; else echo "❌ 설정되지 않음"; fi)"

# 데이터베이스 마이그레이션
echo "📊 데이터베이스 마이그레이션 실행 중..."
python manage.py migrate --noinput

# 정적 파일 수집
echo "📁 정적 파일 수집 중..."
python manage.py collectstatic --noinput

# Django 서버 시작
echo "🌐 Django 서버 시작 중..."
exec uvicorn config.asgi:application \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info 