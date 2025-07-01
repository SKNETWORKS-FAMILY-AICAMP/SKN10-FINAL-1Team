#!/bin/bash

echo "🚀 Django + Next.js 프로덕션 배포 시작..."

# 환경변수 파일 확인
if [ ! -f .env ]; then
    echo "⚠️  .env 파일이 없습니다. env.example을 복사합니다..."
    cp env.example .env
    echo "📝 .env 파일을 수정하여 필요한 환경변수를 설정하세요."
    echo "⚠️  프로덕션 배포 전에 .env 파일의 보안 설정을 확인하세요!"
fi

# 프로덕션용 Docker Compose로 시작
echo "🐳 프로덕션 Docker 컨테이너를 빌드하고 시작합니다..."
docker-compose -f docker-compose.prod.yml up --build -d

echo "✅ 프로덕션 배포 완료!"
echo "🌐 웹사이트: http://localhost"
echo "🔧 API: http://localhost/api"
echo "👑 Django Admin: http://localhost/admin"

# 상태 확인
echo ""
echo "📊 컨테이너 상태:"
docker-compose -f docker-compose.prod.yml ps 