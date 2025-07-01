# AWS EC2 + Docker Compose 배포 가이드 (Windows 환경)

이 가이드는 **Windows 환경**에서 Django backend + Next.js frontend를 AWS EC2에서 Docker Compose로 배포하는 방법을 설명합니다.

## 1. Windows 환경 사전 준비

### 1.1 필수 도구 설치
1. **Git for Windows**: https://git-scm.com/download/win
2. **VSCode** (권장): https://code.visualstudio.com/
3. **Windows Terminal** (권장): Microsoft Store에서 설치
4. **WinSCP** (파일 전송용): https://winscp.net/eng/download.php

### 1.2 SSH 클라이언트 선택
**옵션 1: Windows PowerShell/Terminal (권장)**
- Windows 10/11에 기본 내장된 OpenSSH 클라이언트 사용

**옵션 2: PuTTY**
- https://www.putty.org/ 에서 다운로드
- PuTTYgen으로 키 변환 필요

## 2. EC2 인스턴스 설정

### 2.1 EC2 인스턴스 생성
1. **AMI 선택**: Ubuntu Server 22.04 LTS
2. **인스턴스 타입**: t3.medium 이상 권장 (메모리 4GB+)
3. **키 페어**: 
   - 새로 생성: `your-key.pem` 파일 다운로드
   - Windows에서 사용하려면 권한 설정 필요
4. **보안 그룹 설정**:
   ```
   Type        Protocol    Port Range    Source
   SSH         TCP         22           My IP
   HTTP        TCP         80           0.0.0.0/0
   HTTPS       TCP         443          0.0.0.0/0
   Custom TCP  TCP         8000         0.0.0.0/0 (임시, 테스트용)
   Custom TCP  TCP         3000         0.0.0.0/0 (임시, 테스트용)
   ```
5. **스토리지**: 20GB 이상 (gp3 권장)

### 2.2 Windows에서 SSH 키 권한 설정
```powershell
# PowerShell에서 실행 (관리자 권한)
# 키 파일 위치로 이동
cd C:\Users\YourUsername\Downloads

# 키 파일 권한 설정
icacls your-key.pem /inheritance:r
icacls your-key.pem /grant:r "%USERNAME%":"(R)"
```

## 3. 프로젝트 업로드 방법

### 3.1 방법 1: Git 사용 (권장)
```powershell
# 1. GitHub에 프로젝트 푸시 (로컬에서)
git add .
git commit -m "Deploy to EC2"
git push origin main

# 2. EC2에서 클론 (SSH 접속 후)
git clone https://github.com/your-username/SKN10-FINAL-1Team.git
```

### 3.2 방법 2: WinSCP 사용
1. **WinSCP 실행**
2. **연결 설정**:
   - 호스트명: EC2 Public IP
   - 사용자명: `ubuntu`
   - 개인키 파일: `your-key.pem` 선택
3. **파일 업로드**: 로컬 프로젝트 폴더를 EC2 홈 디렉토리로 업로드

### 3.3 방법 3: SCP 명령어 (PowerShell)
```powershell
# PowerShell에서 실행
scp -i "your-key.pem" -r .\SKN10-FINAL-1Team ubuntu@YOUR_EC2_PUBLIC_IP:~/
```

## 4. EC2 서버 설정

### 4.1 EC2 접속
```powershell
# PowerShell/Windows Terminal에서 실행
ssh -i "your-key.pem" ubuntu@YOUR_EC2_PUBLIC_IP
```

### 4.2 서버 환경 구성
```bash
# EC2에 SSH 접속 후 실행
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
sudo apt install -y curl wget git unzip

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

## 5. 환경변수 설정

### 5.1 .env 파일 생성 (EC2에서)
```bash
cd SKN10-FINAL-1Team
cp env.example .env
nano .env
```

### 5.2 Windows용 환경변수 템플릿
```bash
# Django Settings
SECRET_KEY=your-very-secure-secret-key-here-make-it-long-and-random
DEBUG=0
DJANGO_SETTINGS_MODULE=config.settings

# Database (SQLite 사용 시 생략)
# DATABASE_URL=postgresql://user:password@db:5432/dbname

# Next.js Frontend URL (for django-nextjs integration)
NEXTJS_SERVER_URL=http://nextjs_frontend:3000

# Frontend API URL - EC2 Public IP로 변경!
NEXT_PUBLIC_BACKEND_URL=http://YOUR_EC2_PUBLIC_IP

# API Keys - 실제 값으로 변경하세요
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX_NAME=your-pinecone-index-name
LANGGRAPH_API_KEY=your-langgraph-api-key-here

# AWS Settings
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_STORAGE_BUCKET_NAME=your-s3-bucket-name
AWS_S3_REGION_NAME=ap-northeast-2

# GitHub Settings
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
```

## 6. 배포 실행

### 6.1 프로덕션 배포
```bash
# EC2에서 실행
cd SKN10-FINAL-1Team

# 실행 권한 부여
chmod +x start-prod.sh stop.sh

# 프로덕션 배포 실행
./start-prod.sh

# 또는 직접 실행
docker-compose -f docker-compose.prod.yml up --build -d
```

### 6.2 배포 상태 확인
```bash
# 컨테이너 상태 확인
docker-compose -f docker-compose.prod.yml ps

# 로그 확인
docker-compose -f docker-compose.prod.yml logs

# 실시간 로그 모니터링
docker-compose -f docker-compose.prod.yml logs -f
```

## 7. Windows용 관리 스크립트

### 7.1 Windows PowerShell 스크립트 생성
다음 내용을 `manage-ec2.ps1` 파일로 저장:

```powershell
# EC2 관리용 PowerShell 스크립트
param(
    [Parameter(Mandatory=$true)]
    [string]$EC2_IP,
    
    [Parameter(Mandatory=$true)]
    [string]$KeyPath,
    
    [Parameter(Mandatory=$true)]
    [ValidateSet("deploy", "status", "logs", "restart", "stop", "connect")]
    [string]$Action
)

$SSH_CMD = "ssh -i `"$KeyPath`" ubuntu@$EC2_IP"

switch ($Action) {
    "deploy" {
        Write-Host "🚀 프로덕션 배포 시작..." -ForegroundColor Green
        & cmd /c "$SSH_CMD 'cd SKN10-FINAL-1Team && ./start-prod.sh'"
    }
    "status" {
        Write-Host "📊 컨테이너 상태 확인..." -ForegroundColor Blue
        & cmd /c "$SSH_CMD 'cd SKN10-FINAL-1Team && docker-compose -f docker-compose.prod.yml ps'"
    }
    "logs" {
        Write-Host "📋 로그 확인..." -ForegroundColor Yellow
        & cmd /c "$SSH_CMD 'cd SKN10-FINAL-1Team && docker-compose -f docker-compose.prod.yml logs'"
    }
    "restart" {
        Write-Host "🔄 서비스 재시작..." -ForegroundColor Cyan
        & cmd /c "$SSH_CMD 'cd SKN10-FINAL-1Team && docker-compose -f docker-compose.prod.yml restart'"
    }
    "stop" {
        Write-Host "🛑 서비스 중지..." -ForegroundColor Red
        & cmd /c "$SSH_CMD 'cd SKN10-FINAL-1Team && docker-compose -f docker-compose.prod.yml down'"
    }
    "connect" {
        Write-Host "🔗 EC2에 연결..." -ForegroundColor Magenta
        & cmd /c "$SSH_CMD"
    }
}
```

### 7.2 PowerShell 스크립트 사용법
```powershell
# PowerShell에서 실행
# 배포
.\manage-ec2.ps1 -EC2_IP "YOUR_EC2_IP" -KeyPath "your-key.pem" -Action deploy

# 상태 확인
.\manage-ec2.ps1 -EC2_IP "YOUR_EC2_IP" -KeyPath "your-key.pem" -Action status

# 로그 확인
.\manage-ec2.ps1 -EC2_IP "YOUR_EC2_IP" -KeyPath "your-key.pem" -Action logs

# EC2 접속
.\manage-ec2.ps1 -EC2_IP "YOUR_EC2_IP" -KeyPath "your-key.pem" -Action connect
```

## 8. 배포 후 확인사항

### 8.1 웹사이트 접속 테스트
브라우저에서 다음 URL 접속:
- **메인 사이트**: `http://YOUR_EC2_PUBLIC_IP`
- **API 상태**: `http://YOUR_EC2_PUBLIC_IP/api`
- **Django Admin**: `http://YOUR_EC2_PUBLIC_IP/admin`

### 8.2 문제 해결 명령어
```bash
# EC2 SSH 접속 후 실행
cd SKN10-FINAL-1Team

# 컨테이너 재시작
docker-compose -f docker-compose.prod.yml restart

# 로그 확인
docker-compose -f docker-compose.prod.yml logs backend
docker-compose -f docker-compose.prod.yml logs frontend
docker-compose -f docker-compose.prod.yml logs nginx

# 메모리 사용량 확인
docker stats

# 디스크 공간 확인
df -h
```

## 9. Windows에서 로컬 개발 연동

### 9.1 VSCode Remote SSH 확장 설정
1. **VSCode 확장 설치**: Remote - SSH
2. **SSH 설정**:
   - `Ctrl+Shift+P` → "Remote-SSH: Connect to Host"
   - `ubuntu@YOUR_EC2_IP` 입력
   - SSH 키 경로 설정

### 9.2 Windows에서 파일 동기화
```powershell
# 파일 변경 후 EC2로 업로드
scp -i "your-key.pem" -r .\your-changed-files ubuntu@YOUR_EC2_IP:~/SKN10-FINAL-1Team/

# 또는 WinSCP 사용하여 GUI로 동기화
```

## 10. 도메인 및 HTTPS 설정

### 10.1 도메인 연결 (Route 53 또는 외부 DNS)
1. **A 레코드**: `yourdomain.com` → `YOUR_EC2_PUBLIC_IP`
2. **CNAME 레코드**: `www.yourdomain.com` → `yourdomain.com`

### 10.2 Let's Encrypt SSL 인증서 (EC2에서)
```bash
# Certbot 설치
sudo apt install -y certbot python3-certbot-nginx

# SSL 인증서 발급
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# 자동 갱신 설정
sudo crontab -e
# 다음 라인 추가: 0 2 * * * certbot renew --quiet
```

## 11. 백업 및 모니터링

### 11.1 Windows에서 자동 백업 스크립트
```powershell
# backup-ec2.ps1
param([string]$EC2_IP, [string]$KeyPath)

$DATE = Get-Date -Format "yyyyMMdd"
$BACKUP_DIR = "C:\EC2_Backups\$DATE"

# 백업 디렉토리 생성
New-Item -ItemType Directory -Force -Path $BACKUP_DIR

# 데이터베이스 백업
& cmd /c "ssh -i `"$KeyPath`" ubuntu@$EC2_IP 'cd SKN10-FINAL-1Team && docker-compose -f docker-compose.prod.yml exec -T backend python manage.py dumpdata' > $BACKUP_DIR\database_backup.json"

Write-Host "백업 완료: $BACKUP_DIR" -ForegroundColor Green
```

### 11.2 모니터링 설정
```bash
# EC2에서 실행 - 시스템 모니터링
watch -n 5 'docker stats --no-stream'

# 디스크 사용량 모니터링
watch -n 10 'df -h'
```

## 12. 보안 강화

### 12.1 EC2 보안 설정
```bash
# 방화벽 활성화
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443

# 개발용 포트 차단 (프로덕션에서)
sudo ufw deny 8000
sudo ufw deny 3000

# 실패한 로그인 시도 모니터링
sudo tail -f /var/log/auth.log
```

### 12.2 환경변수 보안
```bash
# .env 파일 권한 설정
chmod 600 .env

# Git에서 .env 파일 제외 확인
echo ".env" >> .gitignore
```

---

## 빠른 시작 체크리스트

✅ **사전 준비**
- [ ] AWS 계정 및 EC2 인스턴스 생성
- [ ] SSH 키 다운로드 및 권한 설정
- [ ] Git, VSCode, WinSCP 설치

✅ **배포 단계**
- [ ] EC2에 Docker 및 Docker Compose 설치
- [ ] 프로젝트 파일 업로드 (Git 또는 WinSCP)
- [ ] .env 파일 설정 (실제 API 키 입력)
- [ ] `./start-prod.sh` 실행
- [ ] 웹사이트 접속 확인

✅ **운영 관리**
- [ ] PowerShell 관리 스크립트 설정
- [ ] 도메인 및 SSL 인증서 설정
- [ ] 백업 시스템 구축
- [ ] 모니터링 시스템 설정

이 가이드를 따라하면 Windows 환경에서 AWS EC2에 성공적으로 배포할 수 있습니다! 🚀 