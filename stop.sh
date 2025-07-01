#!/bin/bash

echo "🛑 Docker 컨테이너를 중지합니다..."

# 개발용 컨테이너 중지
if [ -f docker-compose.yml ]; then
    echo "📦 개발용 컨테이너 중지 중..."
    docker-compose down
fi

# 프로덕션용 컨테이너 중지
if [ -f docker-compose.prod.yml ]; then
    echo "📦 프로덕션용 컨테이너 중지 중..."
    docker-compose -f docker-compose.prod.yml down
fi

echo "✅ 모든 컨테이너가 중지되었습니다."

# 옵션: 볼륨도 삭제
read -p "❓ 볼륨(데이터)도 삭제하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  볼륨을 삭제합니다..."
    docker-compose down -v 2>/dev/null || true
    docker-compose -f docker-compose.prod.yml down -v 2>/dev/null || true
    echo "✅ 볼륨이 삭제되었습니다."
fi 