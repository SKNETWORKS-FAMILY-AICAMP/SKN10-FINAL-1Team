# Django + Next.js Docker Compose 배포 가이드

이 프로젝트는 Django backend와 Next.js frontend를 Docker Compose로 배포할 수 있습니다.

## 프로젝트 구조

```
SKN10-FINAL-1Team/
├── backend/           # Django 애플리케이션
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/          # Next.js 애플리케이션
│   ├── Dockerfile
│   └── package.json
├── nginx/             # Nginx 설정
│   └── nginx.conf
├── docker-compose.yml      # 개발용
├── docker-compose.prod.yml # 프로덕션용
└── env.example        # 환경변수 예제
```

## 사전 준비사항

1. **Docker 및 Docker Compose 설치**
   - Docker Desktop (Windows/Mac)
   - Docker Engine + Docker Compose (Linux)

2. **환경변수 설정**
   ```bash
   # env.example을 참고하여 .env 파일 생성
   cp env.example .env
   # 필요한 환경변수 값들을 설정
   ```

## 배포 방법

### 1. 개발용 배포 (Development)

개발용 배포는 볼륨 마운트를 통해 실시간 코드 변경사항을 반영합니다.

```bash
# 컨테이너 빌드 및 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d --build
```

**접속 URL:**
- Frontend (Next.js): http://localhost:3000
- Backend API (Django): http://localhost:8000
- Django Admin: http://localhost:8000/admin

### 2. 프로덕션용 배포 (Production)

프로덕션용 배포는 Nginx 리버스 프록시를 포함합니다.

```bash
# 프로덕션용 컨테이너 빌드 및 실행
docker-compose -f docker-compose.prod.yml up --build

# 백그라운드 실행
docker-compose -f docker-compose.prod.yml up -d --build
```

**접속 URL:**
- 모든 서비스: http://localhost (Nginx를 통한 라우팅)
- Frontend: http://localhost/
- Backend API: http://localhost/api/
- Django Admin: http://localhost/admin/

### 3. 개별 서비스 관리

```bash
# 특정 서비스만 재시작
docker-compose restart backend
docker-compose restart frontend

# 특정 서비스 로그 확인
docker-compose logs backend
docker-compose logs frontend

# 컨테이너 내부 접속
docker-compose exec backend bash
docker-compose exec frontend sh
```

## 주요 설정

### Django Backend (Port 8000)
- **Dockerfile**: Python 3.11 기반
- **의존성**: requirements.txt
- **설정**: config.settings
- **서버**: Gunicorn

### Next.js Frontend (Port 3000)
- **Dockerfile**: Node.js 18 Alpine 기반
- **의존성**: package.json
- **빌드**: npm run build
- **서버**: npm start

### Nginx (Port 80)
- **역할**: 리버스 프록시
- **라우팅**:
  - `/` → Next.js Frontend
  - `/api/` → Django Backend
  - `/admin/` → Django Admin
  - `/static/`, `/media/` → Django Static Files

## 환경변수 설정

`env.example` 파일을 참고하여 다음 환경변수들을 설정하세요:

```bash
# Django 설정
DEBUG=1
SECRET_KEY=your-secret-key-here
DJANGO_SETTINGS_MODULE=config.settings

# Django-Next.js 통합 (중요!)
NEXTJS_SERVER_URL=http://nextjs_frontend:3000

# Frontend API URL (브라우저에서 Django 접근)
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000

# API Keys
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
LANGGRAPH_API_KEY=your-langgraph-api-key

# AWS 설정
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_STORAGE_BUCKET_NAME=your-s3-bucket
```

### 중요한 환경변수 설명

- **`NEXTJS_SERVER_URL`**: Django에서 Next.js 서버에 접근할 때 사용하는 URL
  - Docker Compose: `http://nextjs_frontend:3000` (내부 네트워크)
  - 로컬 개발: `http://127.0.0.1:3000`

- **`NEXT_PUBLIC_BACKEND_URL`**: 브라우저에서 Django API에 접근할 때 사용하는 URL
  - Docker Compose: `http://localhost:8000` (호스트 포트)
  - 로컬 개발: `http://127.0.0.1:8000`

## 데이터베이스 마이그레이션

```bash
# Django 마이그레이션 실행
docker-compose exec backend python manage.py migrate

# 슈퍼유저 생성
docker-compose exec backend python manage.py createsuperuser

# 정적 파일 수집 (이미 Dockerfile에서 실행됨)
docker-compose exec backend python manage.py collectstatic --noinput
```

## 트러블슈팅

### 1. 포트 충돌
```bash
# 사용 중인 포트 확인
netstat -tulpn | grep :3000
netstat -tulpn | grep :8000

# 컨테이너 중지 후 재시작
docker-compose down
docker-compose up --build
```

### 2. 볼륨 문제
```bash
# 볼륨 정리
docker-compose down -v
docker volume prune

# 이미지 재빌드
docker-compose build --no-cache
```

### 3. 네트워크 문제
```bash
# 네트워크 상태 확인
docker network ls
docker-compose exec backend ping frontend
docker-compose exec frontend ping backend
```

### 4. 로그 확인
```bash
# 전체 로그
docker-compose logs

# 실시간 로그
docker-compose logs -f

# 특정 서비스 로그
docker-compose logs backend
docker-compose logs frontend
```

## 배포 중단

```bash
# 컨테이너 중지
docker-compose down

# 컨테이너 및 볼륨 삭제
docker-compose down -v

# 이미지까지 삭제
docker-compose down --rmi all -v
```

## 주의사항

1. **보안**: 프로덕션 환경에서는 SECRET_KEY, API 키 등을 안전하게 관리하세요.
2. **데이터베이스**: 외부 PostgreSQL 사용 시 DATABASE_URL을 설정하세요.
3. **CORS**: django-nextjs 라이브러리 설정에 맞게 CORS 설정을 확인하세요.
4. **SSL**: 프로덕션에서는 HTTPS 설정을 추가하세요. 