# AWS EC2 초기 설정 자동화 PowerShell 스크립트
# 사용법: .\setup-ec2.ps1 -EC2_IP "your-ip" -KeyPath "your-key.pem"

param(
    [Parameter(Mandatory=$true, HelpMessage="EC2 인스턴스의 Public IP 주소")]
    [string]$EC2_IP,
    
    [Parameter(Mandatory=$true, HelpMessage="SSH 키 파일 경로 (.pem 파일)")]
    [string]$KeyPath,
    
    [Parameter(Mandatory=$false, HelpMessage="GitHub 저장소 URL")]
    [string]$GitRepo = "https://github.com/your-username/SKN10-FINAL-1Team.git"
)

Write-Host "🚀 AWS EC2 초기 설정 자동화 스크립트" -ForegroundColor Cyan
Write-Host "📍 대상 서버: $EC2_IP" -ForegroundColor White
Write-Host "🔑 키 파일: $KeyPath" -ForegroundColor White
Write-Host "📦 Git 저장소: $GitRepo" -ForegroundColor White
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

# Step 1: SSH 키 권한 설정
Write-Host "🔐 SSH 키 파일 권한을 설정합니다..." -ForegroundColor Yellow
try {
    # 키 파일이 존재하는지 확인
    if (!(Test-Path $KeyPath)) {
        throw "SSH 키 파일을 찾을 수 없습니다: $KeyPath"
    }

    # 키 파일 권한 설정
    icacls $KeyPath /inheritance:r 2>$null
    icacls $KeyPath /grant:r "$env:USERNAME":"(R)" 2>$null
    Write-Host "✅ SSH 키 권한 설정 완료" -ForegroundColor Green
} catch {
    Write-Host "⚠️ SSH 키 권한 설정 경고: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "💡 관리자 권한으로 PowerShell을 실행해보세요." -ForegroundColor Yellow
}

# Step 2: EC2 연결 테스트
Write-Host "🌐 EC2 인스턴스 연결을 테스트합니다..." -ForegroundColor Yellow
$sshCmd = "ssh -i `"$KeyPath`" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$EC2_IP"
$testCommand = "$sshCmd 'echo `"Connection successful`"'"

try {
    $result = Invoke-Expression "cmd /c `"$testCommand`"" 2>$null
    if ($result -match "Connection successful") {
        Write-Host "✅ EC2 연결 성공" -ForegroundColor Green
    } else {
        throw "연결 테스트 실패"
    }
} catch {
    Write-Host "❌ EC2 연결 실패: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "💡 확인사항:" -ForegroundColor Yellow
    Write-Host "   - EC2 인스턴스가 실행 중인지 확인" -ForegroundColor Gray
    Write-Host "   - 보안 그룹에서 SSH (포트 22) 허용 확인" -ForegroundColor Gray
    Write-Host "   - IP 주소가 정확한지 확인" -ForegroundColor Gray
    exit 1
}

# Step 3: 시스템 업데이트
Write-Host "📦 시스템 패키지를 업데이트합니다..." -ForegroundColor Yellow
$updateCommand = "$sshCmd 'sudo apt update && sudo apt upgrade -y'"
try {
    Invoke-Expression "cmd /c `"$updateCommand`""
    Write-Host "✅ 시스템 업데이트 완료" -ForegroundColor Green
} catch {
    Write-Host "⚠️ 시스템 업데이트 중 경고가 있었습니다." -ForegroundColor Yellow
}

# Step 4: 필수 패키지 설치
Write-Host "🛠️ 필수 패키지를 설치합니다..." -ForegroundColor Yellow
$installPackagesCommand = "$sshCmd 'sudo apt install -y curl wget git unzip nano htop'"
try {
    Invoke-Expression "cmd /c `"$installPackagesCommand`""
    Write-Host "✅ 필수 패키지 설치 완료" -ForegroundColor Green
} catch {
    Write-Host "❌ 패키지 설치 실패" -ForegroundColor Red
    exit 1
}

# Step 5: Docker 설치
Write-Host "🐳 Docker를 설치합니다..." -ForegroundColor Yellow
$dockerInstallCommands = @(
    "curl -fsSL https://get.docker.com -o get-docker.sh",
    "sudo sh get-docker.sh",
    "sudo usermod -aG docker `$USER",
    "rm get-docker.sh"
)

foreach ($cmd in $dockerInstallCommands) {
    $dockerCommand = "$sshCmd '$cmd'"
    try {
        Invoke-Expression "cmd /c `"$dockerCommand`""
    } catch {
        Write-Host "⚠️ Docker 설치 단계에서 경고: $cmd" -ForegroundColor Yellow
    }
}
Write-Host "✅ Docker 설치 완료" -ForegroundColor Green

# Step 6: Docker Compose 설치
Write-Host "🐙 Docker Compose를 설치합니다..." -ForegroundColor Yellow
$composeInstallCommand = "$sshCmd 'sudo curl -L `"https://github.com/docker/compose/releases/latest/download/docker-compose-`$(uname -s)-`$(uname -m)`" -o /usr/local/bin/docker-compose && sudo chmod +x /usr/local/bin/docker-compose'"
try {
    Invoke-Expression "cmd /c `"$composeInstallCommand`""
    Write-Host "✅ Docker Compose 설치 완료" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker Compose 설치 실패" -ForegroundColor Red
    exit 1
}

# Step 7: 설치 확인
Write-Host "🔍 설치된 도구들의 버전을 확인합니다..." -ForegroundColor Yellow
$versionCheckCommand = "$sshCmd 'echo `"=== Docker 버전 ===`" && docker --version && echo `"`" && echo `"=== Docker Compose 버전 ===`" && docker-compose --version && echo `"`" && echo `"=== Git 버전 ===`" && git --version'"
try {
    Invoke-Expression "cmd /c `"$versionCheckCommand`""
    Write-Host "✅ 버전 확인 완료" -ForegroundColor Green
} catch {
    Write-Host "⚠️ 일부 도구의 버전 확인에 실패했습니다." -ForegroundColor Yellow
}

# Step 8: 프로젝트 클론 (선택사항)
if ($GitRepo -ne "https://github.com/your-username/SKN10-FINAL-1Team.git") {
    Write-Host "📥 프로젝트를 클론합니다..." -ForegroundColor Yellow
    $cloneCommand = "$sshCmd 'git clone $GitRepo'"
    try {
        Invoke-Expression "cmd /c `"$cloneCommand`""
        Write-Host "✅ 프로젝트 클론 완료" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ 프로젝트 클론 실패. 수동으로 업로드하세요." -ForegroundColor Yellow
    }
} else {
    Write-Host "📝 Git 저장소 URL이 기본값입니다. 실제 저장소 URL로 변경 후 다시 실행하세요." -ForegroundColor Yellow
}

# Step 9: 방화벽 설정
Write-Host "🔥 기본 방화벽을 설정합니다..." -ForegroundColor Yellow
$firewallCommands = @(
    "sudo ufw --force enable",
    "sudo ufw allow ssh",
    "sudo ufw allow 80",
    "sudo ufw allow 443",
    "sudo ufw allow 8000",
    "sudo ufw allow 3000"
)

foreach ($cmd in $firewallCommands) {
    $firewallCommand = "$sshCmd '$cmd'"
    try {
        Invoke-Expression "cmd /c `"$firewallCommand`""
    } catch {
        Write-Host "⚠️ 방화벽 설정 경고: $cmd" -ForegroundColor Yellow
    }
}
Write-Host "✅ 방화벽 설정 완료" -ForegroundColor Green

# Step 10: Docker 그룹 적용을 위한 재로그인 안내
Write-Host "🔄 Docker 권한 적용을 위해 재로그인이 필요합니다..." -ForegroundColor Yellow
$reloginCommand = "$sshCmd 'newgrp docker'"
try {
    Invoke-Expression "cmd /c `"$reloginCommand`""
    Write-Host "✅ Docker 그룹 권한 적용 완료" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Docker 그룹 권한 적용 경고" -ForegroundColor Yellow
}

# 설정 완료 및 다음 단계 안내
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "🎉 EC2 초기 설정이 완료되었습니다!" -ForegroundColor Green
Write-Host "" -ForegroundColor White

Write-Host "📋 다음 단계:" -ForegroundColor Cyan
Write-Host "1. 프로젝트 파일 업로드 (Git 또는 WinSCP 사용)" -ForegroundColor White
Write-Host "2. .env 파일 설정 (API 키 및 환경변수)" -ForegroundColor White
Write-Host "3. 프로덕션 배포 실행:" -ForegroundColor White
Write-Host "   .\manage-ec2.ps1 -EC2_IP $EC2_IP -KeyPath `"$KeyPath`" -Action deploy" -ForegroundColor Gray

Write-Host "" -ForegroundColor White
Write-Host "🔧 유용한 명령어:" -ForegroundColor Cyan
Write-Host "• 상태 확인: .\manage-ec2.ps1 -EC2_IP $EC2_IP -KeyPath `"$KeyPath`" -Action status" -ForegroundColor Gray
Write-Host "• 로그 확인: .\manage-ec2.ps1 -EC2_IP $EC2_IP -KeyPath `"$KeyPath`" -Action logs" -ForegroundColor Gray
Write-Host "• 백업 실행: .\backup-ec2.ps1 -EC2_IP $EC2_IP -KeyPath `"$KeyPath`"" -ForegroundColor Gray
Write-Host "• SSH 접속: .\manage-ec2.ps1 -EC2_IP $EC2_IP -KeyPath `"$KeyPath`" -Action connect" -ForegroundColor Gray

Write-Host "" -ForegroundColor White
Write-Host "🌐 배포 후 접속 URL:" -ForegroundColor Cyan
Write-Host "• 웹사이트: http://$EC2_IP" -ForegroundColor Gray
Write-Host "• API: http://$EC2_IP/api" -ForegroundColor Gray
Write-Host "• Django Admin: http://$EC2_IP/admin" -ForegroundColor Gray

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "✨ 설정 스크립트 실행 완료!" -ForegroundColor Green 