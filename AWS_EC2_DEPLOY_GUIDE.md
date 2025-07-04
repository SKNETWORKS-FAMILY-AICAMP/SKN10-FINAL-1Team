# AWS EC2 + Docker Compose 배포 가이드

이 가이드는 Django backend + Next.js frontend를 AWS EC2에서 Docker Compose로 배포하는 방법을 설명합니다.

## 1. EC2 인스턴스 설정

### 1.1 EC2 인스턴스 생성
1. **AMI 선택**: Ubuntu Server 22.04 LTS
2. **인스턴스 타입**: t3.medium 이상 권장 (메모리 4GB+)
3. **키 페어**: 새로 생성하거나 기존 키 페어 사용
4. **보안 그룹 설정**:
   ```
   Type        Protocol    Port Range    Source
   SSH         TCP         22           My IP
   HTTP        TCP         80           0.0.0.0/0
   HTTPS       TCP         443          0.0.0.0/0
   Custom TCP  TCP         8000         0.0.0.0/0 (개발용, 나중에 제거)
   Custom TCP  TCP         3000         0.0.0.0/0 (개발용, 나중에 제거)
   ```
5. **스토리지**: 20GB 이상 (gp3 권장)

### 1.2 Elastic IP 할당 (선택사항)
- 고정 IP가 필요한 경우 Elastic IP를 할당하고 인스턴스에 연결

## 2. 서버 환경 구성

### 2.1 EC2 접속 및 기본 설정
```bash
# EC2에 SSH 접속
ssh -i "your-key.pem" ubuntu@your-ec2-public-ip

# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
sudo apt install -y curl wget git unzip
```

### 2.2 Docker 설치
```bash
# Docker 설치 스크립트 실행
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 현재 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER

# Docker Compose 설치
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 재로그인 또는 그룹 변경 적용
newgrp docker

# 설치 확인
docker --version
docker-compose --version
```

## 3. 프로젝트 배포

### 3.1 프로젝트 업로드
```bash
# Git을 통한 배포 (권장)
git clone https://github.com/your-username/SKN10-FINAL-1Team.git
cd SKN10-FINAL-1Team

# 또는 파일 압축해서 업로드
# scp -i "your-key.pem" -r ./SKN10-FINAL-1Team ubuntu@your-ec2-ip:~/
```

### 3.2 환경변수 설정
```bash
# .env 파일 생성
cp env.example .env
nano .env
```

**EC2용 환경변수 설정**:
```bash
# Django Settings
SECRET_KEY=your-very-secure-secret-key-here
DEBUG=0
DJANGO_SETTINGS_MODULE=config.settings

# Database (SQLite 사용 시 생략)
# DATABASE_URL=postgresql://user:password@db:5432/dbname

# Next.js Frontend URL (for django-nextjs integration)
NEXTJS_SERVER_URL=http://nextjs_frontend:3000

# Frontend API URL (EC2 Public IP로 변경)
NEXT_PUBLIC_BACKEND_URL=http://YOUR_EC2_PUBLIC_IP

# API Keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-pinecone-index
LANGGRAPH_API_KEY=your-langgraph-api-key

# AWS Settings
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_STORAGE_BUCKET_NAME=your-s3-bucket
AWS_S3_REGION_NAME=us-east-1

# GitHub Settings
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
```

### 3.3 프로덕션 배포 실행
```bash
# 실행 권한 부여
chmod +x start-prod.sh stop.sh

# 프로덕션 배포 실행
./start-prod.sh

# 또는 직접 실행
docker-compose -f docker-compose.prod.yml up --build -d
```

## 4. 도메인 및 SSL 설정 (선택사항)

### 4.1 도메인 연결
1. 도메인의 A 레코드를 EC2 Public IP로 설정
2. CNAME 레코드 설정 (www.yourdomain.com → yourdomain.com)

### 4.2 Let's Encrypt SSL 인증서 설정
```bash
# Certbot 설치
sudo apt install -y certbot python3-certbot-nginx

# SSL 인증서 발급 (도메인이 있는 경우)
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# 자동 갱신 테스트
sudo certbot renew --dry-run
```

## 5. 모니터링 및 관리

### 5.1 상태 확인
```bash
# 컨테이너 상태 확인
docker-compose -f docker-compose.prod.yml ps

# 로그 확인
docker-compose -f docker-compose.prod.yml logs

# 실시간 로그
docker-compose -f docker-compose.prod.yml logs -f

# 특정 서비스 로그
docker-compose -f docker-compose.prod.yml logs backend
docker-compose -f docker-compose.prod.yml logs frontend
```

### 5.2 서비스 관리
```bash
# 서비스 재시작
docker-compose -f docker-compose.prod.yml restart

# 특정 서비스만 재시작
docker-compose -f docker-compose.prod.yml restart backend

# 서비스 중지
docker-compose -f docker-compose.prod.yml stop

# 서비스 완전 종료 및 컨테이너 제거
docker-compose -f docker-compose.prod.yml down

# 볼륨까지 함께 제거 (데이터 삭제됨!)
docker-compose -f docker-compose.prod.yml down -v
```

### 5.3 업데이트 배포
```bash
# 코드 업데이트
git pull origin main

# 컨테이너 재빌드 및 재시작
docker-compose -f docker-compose.prod.yml up --build -d

# 또는 무중단 업데이트
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

## 6. 백업 및 복구

### 6.1 데이터 백업
```bash
# Django 데이터베이스 백업
docker-compose -f docker-compose.prod.yml exec backend python manage.py dumpdata > backup.json

# 미디어 파일 백업
sudo tar -czf media_backup.tar.gz -C /var/lib/docker/volumes/ backend_media

# 전체 프로젝트 백업
sudo tar -czf project_backup.tar.gz ~/SKN10-FINAL-1Team
```

### 6.2 정기 백업 설정
```bash
# crontab 편집
crontab -e

# 매일 새벽 2시에 백업 (예시)
0 2 * * * cd ~/SKN10-FINAL-1Team && docker-compose -f docker-compose.prod.yml exec -T backend python manage.py dumpdata > backup_$(date +\%Y\%m\%d).json
```

## 7. 트러블슈팅

### 7.1 일반적인 문제들
```bash
# 1. 포트 충돌 확인
sudo netstat -tulpn | grep :80
sudo netstat -tulpn | grep :443

# 2. Docker 상태 확인
sudo systemctl status docker

# 3. 메모리 사용량 확인
free -h
docker stats

# 4. 디스크 공간 확인
df -h
docker system df

# 5. Docker 로그 확인
sudo journalctl -u docker.service
```

### 7.2 성능 최적화
```bash
# 1. 사용하지 않는 Docker 리소스 정리
docker system prune -a

# 2. 로그 크기 제한 (docker-compose.prod.yml에 추가)
# logging:
#   driver: "json-file"
#   options:
#     max-size: "10m"
#     max-file: "3"

# 3. 메모리 스왑 비활성화 (성능 향상)
sudo swapoff -a
```

## 8. 보안 설정

### 8.1 방화벽 설정
```bash
# UFW 방화벽 활성화
sudo ufw enable

# 필수 포트만 개방
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443

# 개발용 포트는 프로덕션에서 차단
sudo ufw deny 8000
sudo ufw deny 3000

# 상태 확인
sudo ufw status
```

### 8.2 보안 권장사항
1. **환경변수 보안**: `.env` 파일 권한을 600으로 설정
2. **정기 업데이트**: OS 및 Docker 정기 업데이트
3. **로그 모니터링**: 비정상적인 접근 로그 모니터링
4. **백업**: 정기적인 데이터 백업

## 9. 접속 확인

배포 완료 후 다음 URL로 접속 확인:
- **메인 사이트**: `http://YOUR_EC2_PUBLIC_IP`
- **API**: `http://YOUR_EC2_PUBLIC_IP/api`
- **Django Admin**: `http://YOUR_EC2_PUBLIC_IP/admin`

SSL 설정 시:
- **메인 사이트**: `https://yourdomain.com`
- **API**: `https://yourdomain.com/api`
- **Django Admin**: `https://yourdomain.com/admin`

## 10. 비용 최적화

### 10.1 인스턴스 최적화
- **인스턴스 유형**: 실제 사용량에 맞게 조정
- **예약 인스턴스**: 장기 사용 시 비용 절감
- **스팟 인스턴스**: 개발/테스트 환경에서 활용

### 10.2 스토리지 최적화
- **EBS 볼륨 타입**: gp3 사용으로 비용 절감
- **스냅샷 관리**: 불필요한 스냅샷 정리
- **S3 연동**: 미디어 파일을 S3로 이전

---

이 가이드를 따라하면 AWS EC2에서 Django + Next.js 애플리케이션을 성공적으로 배포할 수 있습니다. 