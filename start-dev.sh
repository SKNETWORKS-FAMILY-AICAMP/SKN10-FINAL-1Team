#!/bin/bash

echo "🚀 Django + Next.js 개발환경 시작..."

# 환경변수 파일 확인
if [ ! -f .env ]; then
    echo "⚠️  .env 파일이 없습니다. env.example을 복사합니다..."
    cp env.example .env
    echo "📝 .env 파일을 수정하여 필요한 환경변수를 설정하세요."
fi

# Docker Compose로 개발환경 시작
echo "🐳 Docker 컨테이너를 빌드하고 시작합니다..."
docker-compose up --build

echo "✅ 배포 완료!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "👑 Django Admin: http://localhost:8000/admin" 