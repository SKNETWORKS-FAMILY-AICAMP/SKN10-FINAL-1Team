# 🚀 Django Elastic Beanstalk 배포 가이드

이 가이드는 Django 프로젝트를 AWS Elastic Beanstalk에 배포하는 방법을 설명합니다.

## 📋 사전 준비사항

1. **AWS 계정** - AWS 콘솔 접근 권한
2. **Python 환경** - 로컬에서 스크립트 실행을 위해
3. **데이터베이스** (선택사항) - PostgreSQL 서버 (없으면 SQLite 사용)

## 🎯 1단계: 배포 파일 준비

### 1.1 배포용 ZIP 파일 생성

```bash
cd backend
python deploy_to_eb.py
```

이 스크립트는 `django_eb_deployment.zip` 파일을 생성합니다.

### 1.2 생성된 파일 확인

- `django_eb_deployment.zip` - 배포용 파일
- `.ebextensions/` - EB 설정 폴더
- `application.py` - WSGI 진입점
- `requirements.txt` - Python 의존성

## 🌐 2단계: AWS Elastic Beanstalk 설정

### 2.1 Elastic Beanstalk 콘솔 접속

1. [AWS 콘솔](https://aws.amazon.com/console/)에 로그인
2. **Elastic Beanstalk** 서비스 검색 및 선택
3. 원하는 리전 선택 (예: 아시아 태평양 서울 - ap-northeast-2)

### 2.2 새 애플리케이션 생성

1. **"애플리케이션 생성"** 클릭
2. 애플리케이션 정보 입력:
   - **애플리케이션 이름**: `django-chatbot-app`
   - **설명**: `Django 채팅봇 애플리케이션`

### 2.3 환경 생성

1. **"환경 생성"** 클릭
2. 환경 계층 선택: **"웹 서버 환경"**
3. 환경 정보 입력:
   - **환경 이름**: `django-chatbot-prod`
   - **도메인**: 자동 생성 또는 원하는 이름

### 2.4 플랫폼 설정

1. **플랫폼**: `Python`
2. **플랫폼 브랜치**: `Python 3.11 running on 64bit Amazon Linux 2`
3. **플랫폼 버전**: 최신 버전 선택

### 2.5 애플리케이션 코드 업로드

1. **"코드 업로드"** 선택
2. **"로컬 파일"** 선택
3. 생성된 `django_eb_deployment.zip` 파일 업로드
4. **버전 레이블**: `v1.0.0` (또는 원하는 버전)

## ⚙️ 3단계: 환경변수 설정

### 3.1 Configuration 설정

1. 환경 생성 후 **"Configuration"** 탭 클릭
2. **"Software"** 섹션에서 **"Edit"** 클릭

### 3.2 환경 속성 추가

**필수 설정:**
```
SECRET_KEY = your-django-secret-key-here
DEBUG = False
```

**데이터베이스 설정 (PostgreSQL 사용 시):**
```
DB_NAME = mydatabase
DB_USER = myuser
DB_PASSWORD = hinton1234
DB_HOST = 35.170.244.126
DB_PORT = 5432
```

**API 키 설정:**
```
OPENAI_API_KEY = your-openai-api-key
PINECONE_API_KEY = your-pinecone-api-key
PINECONE_ENVIRONMENT = your-pinecone-environment
LANGGRAPH_API_URL = http://127.0.0.1:2024
LANGGRAPH_API_KEY = your-langgraph-api-key
```

### 3.3 설정 적용

1. **"Apply"** 클릭
2. 환경 업데이트 대기 (약 2-5분)

## 🗃️ 4단계: 데이터베이스 설정 (선택사항)

### 4.1 PostgreSQL 사용 시

환경변수에 데이터베이스 정보를 설정하면 자동으로 PostgreSQL 연결됩니다.

### 4.2 SQLite 사용 시

환경변수에 DB 정보가 없으면 자동으로 SQLite를 사용합니다.
- 별도 설정 불필요
- 파일 기반 데이터베이스

## 🚀 5단계: 배포 및 확인

### 5.1 배포 진행

1. **"Create environment"** 클릭
2. 배포 진행 상황 모니터링 (약 5-10분)
3. 환경 상태가 **"Ok"** (녹색)가 될 때까지 대기

### 5.2 애플리케이션 접속

1. 환경 대시보드에서 **URL** 확인
2. 브라우저에서 URL 접속
3. Django 홈페이지 확인

### 5.3 접속 가능한 경로

- `/` - Django 메인 홈페이지
- `/admin/` - Django 관리자 페이지
- `/api/` - Django REST API
- `/accounts/` - 사용자 관리

## 🔧 6단계: 추가 설정

### 6.1 관리자 계정 생성

EB CLI 또는 SSH를 통해 접속 후:

```bash
python manage.py createsuperuser
```

### 6.2 정적 파일 설정

정적 파일은 자동으로 `/static/` 경로에서 서빙됩니다.

### 6.3 로그 확인

1. EB 콘솔에서 **"Logs"** 탭
2. **"Request Logs"** → **"Full Logs"** 다운로드

## 🎛️ 7단계: 업데이트 및 유지보수

### 7.1 코드 업데이트

1. 로컬에서 코드 수정
2. `python deploy_to_eb.py` 실행
3. EB 콘솔에서 **"Upload and deploy"**
4. 새 ZIP 파일 업로드

### 7.2 환경변수 수정

1. **Configuration** → **Software** → **Edit**
2. 환경 속성 수정
3. **Apply** 클릭

### 7.3 스케일링 설정

1. **Configuration** → **Capacity**
2. 인스턴스 타입 및 수량 조정

## 🚨 문제해결

### 배포 실패 시

1. **로그 확인**: Logs 탭에서 에러 메시지 확인
2. **환경변수 확인**: 필수 환경변수 설정 여부
3. **의존성 확인**: requirements.txt 파일 내용

### 500 에러 발생 시

1. `DEBUG=True`로 임시 설정하여 에러 확인
2. 데이터베이스 연결 상태 확인
3. Static 파일 경로 확인

### 데이터베이스 마이그레이션 오류

1. EB SSH 접속
2. 수동으로 마이그레이션 실행:
   ```bash
   python manage.py migrate
   ```

## 💰 비용 관리

### 예상 비용 (월 기준)

- **t3.micro (1 인스턴스)**: 약 $8-15
- **Load Balancer**: 약 $20
- **기타 (트래픽, 스토리지)**: 약 $5-10

### 비용 절약 팁

1. **개발/테스트**: t3.micro 사용
2. **운영 중단 시**: 환경 종료
3. **모니터링**: CloudWatch로 리소스 사용량 확인

## 📞 지원 및 문의

- **AWS 문서**: [Elastic Beanstalk Python 가이드](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-django.html)
- **Django 문서**: [Django 배포 가이드](https://docs.djangoproject.com/en/stable/howto/deployment/)

---

**배포 성공을 기원합니다! 🎉** 