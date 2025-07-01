# AWS EC2 백업용 PowerShell 스크립트
# 사용법: .\backup-ec2.ps1 -EC2_IP "your-ip" -KeyPath "your-key.pem"

param(
    [Parameter(Mandatory=$true, HelpMessage="EC2 인스턴스의 Public IP 주소")]
    [string]$EC2_IP,
    
    [Parameter(Mandatory=$true, HelpMessage="SSH 키 파일 경로 (.pem 파일)")]
    [string]$KeyPath,
    
    [Parameter(Mandatory=$false, HelpMessage="백업 저장 경로")]
    [string]$BackupPath = "C:\EC2_Backups"
)

# 현재 날짜/시간으로 백업 폴더명 생성
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = Join-Path $BackupPath $timestamp
$projectDir = "SKN10-FINAL-1Team"

Write-Host "💾 AWS EC2 백업 스크립트" -ForegroundColor Cyan
Write-Host "📍 대상 서버: $EC2_IP" -ForegroundColor White
Write-Host "🔑 키 파일: $KeyPath" -ForegroundColor White
Write-Host "📂 백업 경로: $backupDir" -ForegroundColor White
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray

try {
    # 백업 디렉토리 생성
    Write-Host "📁 백업 디렉토리를 생성합니다..." -ForegroundColor Yellow
    if (!(Test-Path $backupDir)) {
        New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
        Write-Host "✅ 디렉토리 생성 완료: $backupDir" -ForegroundColor Green
    }

    $sshCmd = "ssh -i `"$KeyPath`" ubuntu@$EC2_IP"

    # 1. 데이터베이스 백업
    Write-Host "🗄️ Django 데이터베이스를 백업합니다..." -ForegroundColor Yellow
    $dbBackupCommand = "$sshCmd 'cd $projectDir && docker-compose -f docker-compose.prod.yml exec -T backend python manage.py dumpdata --indent 2'"
    $dbBackupFile = Join-Path $backupDir "database_backup.json"
    
    Invoke-Expression "cmd /c `"$dbBackupCommand`"" | Out-File -FilePath $dbBackupFile -Encoding UTF8
    
    if (Test-Path $dbBackupFile) {
        $fileSize = (Get-Item $dbBackupFile).Length
        Write-Host "✅ 데이터베이스 백업 완료: $([math]::Round($fileSize/1KB, 2)) KB" -ForegroundColor Green
    } else {
        throw "데이터베이스 백업 파일 생성 실패"
    }

    # 2. 환경변수 파일 백업
    Write-Host "⚙️ 환경변수 파일을 백업합니다..." -ForegroundColor Yellow
    $envBackupCommand = "$sshCmd 'cd $projectDir && cat .env'"
    $envBackupFile = Join-Path $backupDir "env_backup.txt"
    
    try {
        Invoke-Expression "cmd /c `"$envBackupCommand`"" | Out-File -FilePath $envBackupFile -Encoding UTF8
        Write-Host "✅ 환경변수 백업 완료" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ 환경변수 파일 백업 실패 (파일이 없을 수 있음)" -ForegroundColor Yellow
    }

    # 3. Docker Compose 설정 백업
    Write-Host "🐳 Docker 설정 파일을 백업합니다..." -ForegroundColor Yellow
    $dockerComposeBackupCommand = "$sshCmd 'cd $projectDir && cat docker-compose.prod.yml'"
    $dockerComposeBackupFile = Join-Path $backupDir "docker-compose.prod.yml"
    
    Invoke-Expression "cmd /c `"$dockerComposeBackupCommand`"" | Out-File -FilePath $dockerComposeBackupFile -Encoding UTF8
    Write-Host "✅ Docker Compose 설정 백업 완료" -ForegroundColor Green

    # 4. Nginx 설정 백업
    Write-Host "🌐 Nginx 설정을 백업합니다..." -ForegroundColor Yellow
    $nginxBackupCommand = "$sshCmd 'cd $projectDir && cat nginx/nginx.conf'"
    $nginxBackupFile = Join-Path $backupDir "nginx.conf"
    
    try {
        Invoke-Expression "cmd /c `"$nginxBackupCommand`"" | Out-File -FilePath $nginxBackupFile -Encoding UTF8
        Write-Host "✅ Nginx 설정 백업 완료" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Nginx 설정 백업 실패" -ForegroundColor Yellow
    }

    # 5. 시스템 정보 수집
    Write-Host "🔍 시스템 정보를 수집합니다..." -ForegroundColor Yellow
    $systemInfoCommand = "$sshCmd 'echo `"=== 시스템 정보 ===`" && uname -a && echo `"`" && echo `"=== 메모리 사용량 ===`" && free -h && echo `"`" && echo `"=== 디스크 사용량 ===`" && df -h && echo `"`" && echo `"=== Docker 버전 ===`" && docker --version && docker-compose --version && echo `"`" && echo `"=== 실행 중인 컨테이너 ===`" && docker ps'"
    $systemInfoFile = Join-Path $backupDir "system_info.txt"
    
    Invoke-Expression "cmd /c `"$systemInfoCommand`"" | Out-File -FilePath $systemInfoFile -Encoding UTF8
    Write-Host "✅ 시스템 정보 수집 완료" -ForegroundColor Green

    # 6. 백업 완료 로그 생성
    $backupLog = @"
AWS EC2 백업 완료 보고서
==============================

백업 일시: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
백업 대상: $EC2_IP
백업 경로: $backupDir

백업된 파일:
- database_backup.json (Django 데이터베이스)
- env_backup.txt (환경변수)
- docker-compose.prod.yml (Docker Compose 설정)
- nginx.conf (Nginx 설정)
- system_info.txt (시스템 정보)

백업 상태: 성공 ✅
"@

    $backupLogFile = Join-Path $backupDir "backup_log.txt"
    $backupLog | Out-File -FilePath $backupLogFile -Encoding UTF8

    # 백업 완료 알림
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
    Write-Host "🎉 백업이 성공적으로 완료되었습니다!" -ForegroundColor Green
    Write-Host "📂 백업 위치: $backupDir" -ForegroundColor Cyan
    Write-Host "📋 백업 내용:" -ForegroundColor White
    
    Get-ChildItem -Path $backupDir | ForEach-Object {
        $size = if ($_.Length -lt 1KB) { "$($_.Length) bytes" } 
                elseif ($_.Length -lt 1MB) { "$([math]::Round($_.Length/1KB, 2)) KB" }
                else { "$([math]::Round($_.Length/1MB, 2)) MB" }
        Write-Host "   📄 $($_.Name) ($size)" -ForegroundColor Gray
    }

    # Windows 탐색기에서 백업 폴더 열기 옵션
    $openFolder = Read-Host "`n📁 백업 폴더를 열어보시겠습니까? (y/N)"
    if ($openFolder -eq 'y' -or $openFolder -eq 'Y') {
        Invoke-Item $backupDir
    }

} catch {
    Write-Host "❌ 백업 중 오류가 발생했습니다: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
Write-Host "✨ 백업 스크립트 실행 완료!" -ForegroundColor Green 