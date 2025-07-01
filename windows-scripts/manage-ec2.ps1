# AWS EC2 관리용 PowerShell 스크립트
# 사용법: .\manage-ec2.ps1 -EC2_IP "your-ip" -KeyPath "your-key.pem" -Action "deploy"

param(
    [Parameter(Mandatory=$true, HelpMessage="EC2 인스턴스의 Public IP 주소")]
    [string]$EC2_IP,
    
    [Parameter(Mandatory=$true, HelpMessage="SSH 키 파일 경로 (.pem 파일)")]
    [string]$KeyPath,
    
    [Parameter(Mandatory=$true, HelpMessage="실행할 액션")]
    [ValidateSet("deploy", "status", "logs", "restart", "stop", "connect", "backup", "update")]
    [string]$Action
)

# 색상 출력을 위한 함수
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

# SSH 명령 기본 템플릿
$SSH_CMD = "ssh -i `"$KeyPath`" ubuntu@$EC2_IP"
$PROJECT_DIR = "SKN10-FINAL-1Team"

Write-Host "🌟 AWS EC2 관리 스크립트" -ForegroundColor Cyan
Write-Host "📍 대상 서버: $EC2_IP" -ForegroundColor White
Write-Host "🔑 키 파일: $KeyPath" -ForegroundColor White
Write-Host "⚡ 액션: $Action" -ForegroundColor White
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

switch ($Action) {
    "deploy" {
        Write-Host "🚀 프로덕션 배포를 시작합니다..." -ForegroundColor Green
        Write-Host "📦 Docker 컨테이너를 빌드하고 시작합니다..." -ForegroundColor Yellow
        
        $deployCommand = "$SSH_CMD 'cd $PROJECT_DIR && chmod +x start-prod.sh && ./start-prod.sh'"
        Invoke-Expression "cmd /c `"$deployCommand`""
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ 배포가 성공적으로 완료되었습니다!" -ForegroundColor Green
            Write-Host "🌐 웹사이트: http://$EC2_IP" -ForegroundColor Cyan
            Write-Host "🔧 API: http://$EC2_IP/api" -ForegroundColor Cyan
            Write-Host "👑 Admin: http://$EC2_IP/admin" -ForegroundColor Cyan
        } else {
            Write-Host "❌ 배포 중 오류가 발생했습니다." -ForegroundColor Red
        }
    }
    
    "status" {
        Write-Host "📊 컨테이너 상태를 확인합니다..." -ForegroundColor Blue
        $statusCommand = "$SSH_CMD 'cd $PROJECT_DIR && docker-compose -f docker-compose.prod.yml ps'"
        Invoke-Expression "cmd /c `"$statusCommand`""
        
        Write-Host "💾 시스템 리소스 사용량:" -ForegroundColor Blue
        $resourceCommand = "$SSH_CMD 'echo `"=== CPU & 메모리 사용량 ===`" && free -h && echo `"`" && echo `"=== 디스크 사용량 ===`" && df -h'"
        Invoke-Expression "cmd /c `"$resourceCommand`""
    }
    
    "logs" {
        Write-Host "📋 컨테이너 로그를 확인합니다..." -ForegroundColor Yellow
        Write-Host "💡 Ctrl+C로 로그 모니터링을 중단할 수 있습니다." -ForegroundColor DarkYellow
        
        $logsCommand = "$SSH_CMD 'cd $PROJECT_DIR && docker-compose -f docker-compose.prod.yml logs --tail=50 -f'"
        Invoke-Expression "cmd /c `"$logsCommand`""
    }
    
    "restart" {
        Write-Host "🔄 서비스를 재시작합니다..." -ForegroundColor Cyan
        
        $restartCommand = "$SSH_CMD 'cd $PROJECT_DIR && docker-compose -f docker-compose.prod.yml restart'"
        Invoke-Expression "cmd /c `"$restartCommand`""
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ 서비스가 성공적으로 재시작되었습니다!" -ForegroundColor Green
        } else {
            Write-Host "❌ 재시작 중 오류가 발생했습니다." -ForegroundColor Red
        }
    }
    
    "stop" {
        Write-Host "🛑 서비스를 중지합니다..." -ForegroundColor Red
        Write-Host "⚠️  이 작업은 웹사이트를 완전히 중단시킵니다." -ForegroundColor Yellow
        
        $confirmation = Read-Host "정말로 서비스를 중지하시겠습니까? (y/N)"
        if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
            $stopCommand = "$SSH_CMD 'cd $PROJECT_DIR && docker-compose -f docker-compose.prod.yml down'"
            Invoke-Expression "cmd /c `"$stopCommand`""
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ 서비스가 성공적으로 중지되었습니다." -ForegroundColor Green
            } else {
                Write-Host "❌ 중지 중 오류가 발생했습니다." -ForegroundColor Red
            }
        } else {
            Write-Host "❌ 작업이 취소되었습니다." -ForegroundColor Yellow
        }
    }
    
    "backup" {
        Write-Host "💾 데이터베이스 백업을 생성합니다..." -ForegroundColor Magenta
        
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backupDir = "EC2_Backups\$timestamp"
        
        # 로컬 백업 디렉토리 생성
        if (!(Test-Path $backupDir)) {
            New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
        }
        
        # 데이터베이스 백업
        $backupCommand = "$SSH_CMD 'cd $PROJECT_DIR && docker-compose -f docker-compose.prod.yml exec -T backend python manage.py dumpdata'"
        $backupOutput = Invoke-Expression "cmd /c `"$backupCommand`""
        $backupOutput | Out-File -FilePath "$backupDir\database_backup.json" -Encoding UTF8
        
        Write-Host "✅ 백업이 완료되었습니다!" -ForegroundColor Green
        Write-Host "📂 백업 위치: $backupDir" -ForegroundColor Cyan
    }
    
    "update" {
        Write-Host "🔄 코드 업데이트 및 재배포를 시작합니다..." -ForegroundColor Green
        
        # Git pull
        Write-Host "📥 최신 코드를 가져옵니다..." -ForegroundColor Yellow
        $pullCommand = "$SSH_CMD 'cd $PROJECT_DIR && git pull origin main'"
        Invoke-Expression "cmd /c `"$pullCommand`""
        
        # 재배포
        Write-Host "🚀 컨테이너를 재빌드합니다..." -ForegroundColor Yellow
        $updateCommand = "$SSH_CMD 'cd $PROJECT_DIR && docker-compose -f docker-compose.prod.yml up --build -d'"
        Invoke-Expression "cmd /c `"$updateCommand`""
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ 업데이트가 성공적으로 완료되었습니다!" -ForegroundColor Green
        } else {
            Write-Host "❌ 업데이트 중 오류가 발생했습니다." -ForegroundColor Red
        }
    }
    
    "connect" {
        Write-Host "🔗 EC2 인스턴스에 SSH로 연결합니다..." -ForegroundColor Magenta
        Write-Host "💡 연결 후 'exit'를 입력하여 나갈 수 있습니다." -ForegroundColor DarkYellow
        
        Invoke-Expression "cmd /c `"$SSH_CMD`""
    }
}

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "✨ 작업이 완료되었습니다!" -ForegroundColor Green 