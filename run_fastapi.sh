#!/bin/bash

# FastAPI 프로젝트가 있는 실제 경로를 지정합니다.
# 스크립트가 있는 위치를 기준으로 상대 경로를 사용하거나 절대 경로를 지정합니다.
FASTAPI_APP_DIR="$(dirname "$0")/fastapi_server_ver2"

# 로그 파일 경로
LOG_FILE="$FASTAPI_APP_DIR/fastapi_restart.log"
# PID 파일 경로 (재시작 시 기존 프로세스 종료용)
PID_FILE="$FASTAPI_APP_DIR/fastapi.pid"

echo "$(date) - Starting FastAPI application restart process for $FASTAPI_APP_DIR" >> $LOG_FILE

# 기존 FastAPI 프로세스가 있다면 종료
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null; then
        echo "$(date) - Killing existing FastAPI process with PID $OLD_PID" >> $LOG_FILE
        kill -9 $OLD_PID # 강제 종료 (SIGKILL)
        sleep 2 # 종료될 때까지 잠시 대기
    fi
    rm -f "$PID_FILE"
fi

# 특정 FastAPI 프로젝트 폴더로 이동하여 최신 코드 가져오기 (!!! dev 브랜치 지정 !!!)
echo "$(date) - Moving to $FASTAPI_APP_DIR and pulling latest code from Git (dev branch)..." >> $LOG_FILE
cd "$FASTAPI_APP_DIR"
git pull origin develop >> $LOG_FILE 2>&1

if [ $? -ne 0 ]; then
    echo "$(date) - Git pull failed in $FASTAPI_APP_DIR. Aborting restart." >> $LOG_FILE
    exit 1
fi

# 필요한 의존성 설치 (선택 사항, 필요시 주석 해제)
# echo "$(date) - Installing Python dependencies..." >> $LOG_FILE
# pip install -r requirements.txt >> $LOG_FILE 2>&1

# FastAPI 애플리케이션 백그라운드에서 시작 (Uvicorn 사용 예시)
# 'main:app'은 FastAPI 앱의 main.py 파일에 'app' 객체가 있다고 가정합니다.
echo "$(date) - Starting Uvicorn server for $FASTAPI_APP_DIR..." >> $LOG_FILE
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info & echo $! > $PID_FILE >> $LOG_FILE 2>&1

echo "$(date) - FastAPI application started. PID: $(cat $PID_FILE)" >> $LOG_FILE