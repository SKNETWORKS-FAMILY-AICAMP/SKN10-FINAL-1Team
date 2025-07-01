# Windows용 AWS EC2 배포 빠른 시작 가이드 🚀

이 폴더는 **Windows 환경**에서 AWS EC2에 Django + Next.js 애플리케이션을 손쉽게 배포할 수 있는 PowerShell 스크립트들을 제공합니다.

## 📁 포함된 스크립트

| 스크립트 파일 | 용도 | 설명 |
|-------------|------|------|
| `setup-ec2.ps1` | 초기 설정 | EC2 인스턴스에 Docker, Git 등 필수 도구 설치 |
| `manage-ec2.ps1` | 배포 관리 | 배포, 상태확인, 로그, 재시작 등 관리 작업 |
| `backup-ec2.ps1` | 백업 | 데이터베이스 및 설정 파일 백업 |

## 🚀 5분 빠른 시작

### 1단계: AWS EC2 인스턴스 생성
1. **AWS 콘솔**에서 EC2 인스턴스 생성
2. **Ubuntu Server 22.04 LTS** 선택
3. **t3.medium** 이상 선택 (메모리 4GB+)
4. **키 페어** 생성 및 다운로드 (.pem 파일)
5. **보안 그룹** 설정:
   ```
   SSH (22) - My IP
   HTTP (80) - Anywhere
   HTTPS (443) - Anywhere
   Custom TCP (8000) - Anywhere (임시)
   Custom TCP (3000) - Anywhere (임시)
   ```

### 2단계: PowerShell에서 초기 설정
```powershell
# PowerShell을 관리자 권한으로 실행
cd windows-scripts

# EC2 초기 환경 설정 (Docker, Git 등 설치)
.\setup-ec2.ps1 -EC2_IP "YOUR_EC2_PUBLIC_IP" -KeyPath "C:\path\to\your-key.pem"
```

### 3단계: 프로젝트 업로드
**방법 1: Git 사용 (권장)**
```powershell
# 로컬에서 GitHub에 푸시
git add .
git commit -m "Deploy to EC2"
git push origin main

# EC2에서 클론 (SSH 접속 후)
ssh -i "your-key.pem" ubuntu@YOUR_EC2_IP
git clone https://github.com/your-username/SKN10-FINAL-1Team.git
exit
```

**방법 2: WinSCP 사용**
1. WinSCP 다운로드 및 설치
2. 호스트: EC2 Public IP, 사용자: ubuntu, 키파일: .pem
3. 프로젝트 폴더 업로드

### 4단계: 환경변수 설정
```powershell
# EC2에 SSH 접속
.\manage-ec2.ps1 -EC2_IP "YOUR_EC2_IP" -KeyPath "your-key.pem" -Action connect

# EC2에서 환경변수 설정
cd SKN10-FINAL-1Team
cp env.example .env
nano .env
# API 키들을 실제 값으로 변경
# NEXT_PUBLIC_BACKEND_URL=http://YOUR_EC2_PUBLIC_IP 로 변경
exit
```

### 5단계: 프로덕션 배포
```powershell
# 배포 실행
.\manage-ec2.ps1 -EC2_IP "YOUR_EC2_IP" -KeyPath "your-key.pem" -Action deploy
```

### 6단계: 웹사이트 확인
브라우저에서 접속:
- **웹사이트**: `http://YOUR_EC2_PUBLIC_IP`
- **API**: `http://YOUR_EC2_PUBLIC_IP/api`
- **Django Admin**: `http://YOUR_EC2_PUBLIC_IP/admin`

## 🛠️ 일상 관리 명령어

### 상태 확인
```powershell
.\manage-ec2.ps1 -EC2_IP "YOUR_IP" -KeyPath "your-key.pem" -Action status
```

### 로그 확인
```powershell
.\manage-ec2.ps1 -EC2_IP "YOUR_IP" -KeyPath "your-key.pem" -Action logs
```

### 서비스 재시작
```powershell
.\manage-ec2.ps1 -EC2_IP "YOUR_IP" -KeyPath "your-key.pem" -Action restart
```

### 백업 생성
```powershell
.\backup-ec2.ps1 -EC2_IP "YOUR_IP" -KeyPath "your-key.pem"
```

### 코드 업데이트 및 재배포
```powershell
.\manage-ec2.ps1 -EC2_IP "YOUR_IP" -KeyPath "your-key.pem" -Action update
```

### SSH 직접 접속
```powershell
.\manage-ec2.ps1 -EC2_IP "YOUR_IP" -KeyPath "your-key.pem" -Action connect
```

## 🔧 고급 설정

### 도메인 연결 및 HTTPS 설정
```bash
# EC2에 SSH 접속 후
sudo apt install -y certbot python3-certbot-nginx

# SSL 인증서 발급
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# 자동 갱신 설정
sudo crontab -e
# 추가: 0 2 * * * certbot renew --quiet
```

### 정기 백업 설정 (Windows 작업 스케줄러)
1. **작업 스케줄러** 실행
2. **기본 작업 만들기**
3. **프로그램**: `PowerShell.exe`
4. **인수**: `-ExecutionPolicy Bypass -File "C:\path\to\backup-ec2.ps1" -EC2_IP "YOUR_IP" -KeyPath "C:\path\to\your-key.pem"`
5. **트리거**: 매일 새벽 2시

## 🔒 보안 권장사항

### 1. SSH 키 보안
```powershell
# 키 파일 권한 설정 (setup-ec2.ps1에서 자동 실행됨)
icacls your-key.pem /inheritance:r
icacls your-key.pem /grant:r "%USERNAME%":"(R)"
```

### 2. 환경변수 보안
- `.env` 파일에 실제 API 키 입력
- GitHub에 `.env` 파일 업로드 금지
- 정기적으로 API 키 교체

### 3. 방화벽 보안
```bash
# 프로덕션에서 개발용 포트 차단
sudo ufw deny 8000
sudo ufw deny 3000
```

## 💰 비용 최적화 팁

### 1. 인스턴스 스케줄링
```powershell
# EC2 중지 (요금 절약)
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# EC2 시작
aws ec2 start-instances --instance-ids i-1234567890abcdef0
```

### 2. 스팟 인스턴스 활용
- 개발/테스트 환경에서 스팟 인스턴스 사용으로 비용 70% 절약

### 3. 모니터링 설정
- CloudWatch로 CPU 사용률 모니터링
- 사용량이 낮을 때 인스턴스 타입 다운그레이드

## 🐛 문제 해결

### SSH 연결 실패
```powershell
# 1. 키 파일 권한 확인
icacls your-key.pem

# 2. 연결 테스트
ssh -v -i "your-key.pem" ubuntu@YOUR_EC2_IP
```

### Docker 권한 오류
```bash
# EC2에서 실행
sudo usermod -aG docker $USER
newgrp docker
```

### 포트 접근 불가
```bash
# 방화벽 상태 확인
sudo ufw status

# 포트 허용
sudo ufw allow 80
sudo ufw allow 443
```

### 메모리 부족
```bash
# 메모리 사용량 확인
free -h
docker stats

# 불필요한 컨테이너 정리
docker system prune -a
```

## 📞 지원 및 문의

문제가 발생하거나 추가 기능이 필요한 경우:

1. **로그 확인**: `.\manage-ec2.ps1 -Action logs`
2. **시스템 상태**: `.\manage-ec2.ps1 -Action status`
3. **백업 생성**: `.\backup-ec2.ps1` (문제 발생 전 백업)

---

## 📋 체크리스트

배포 전 확인사항:
- [ ] AWS 계정 및 결제 설정
- [ ] EC2 인스턴스 생성 및 실행
- [ ] SSH 키 다운로드 및 권한 설정
- [ ] 보안 그룹 포트 설정
- [ ] Git 저장소 준비
- [ ] API 키 준비 (OpenAI, Pinecone 등)

배포 후 확인사항:
- [ ] 웹사이트 접속 확인
- [ ] Django Admin 접속 확인
- [ ] API 엔드포인트 테스트
- [ ] 로그 확인 및 오류 없음
- [ ] 백업 시스템 테스트

---

🎉 **축하합니다!** Windows 환경에서 AWS EC2에 성공적으로 배포하셨습니다! 