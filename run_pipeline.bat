@echo off
setlocal

REM This script orchestrates the S3 upload, download, and LLM analysis pipeline.
REM It replaces the Python-based orchestrator to avoid process deadlocks on Windows.

REM --- Configuration ---
set "FILE_TO_UPLOAD=test_data.csv"
set "BUCKET_NAME=hinton-csv-upload"
set "TEMP_DOWNLOAD_FILE=temp_downloaded_%RANDOM%.csv"
set "ANALYSIS_RESULT_FILE=analysis_result.txt"

echo.
echo [PIPELINE] Starting...
echo.

REM --- Step 1: Upload file to S3 ---
echo [PIPELINE] Step 1: Uploading %FILE_TO_UPLOAD% to S3...
for /f "delims=" %%i in ('python upload_script.py --file %FILE_TO_UPLOAD% --bucket %BUCKET_NAME%') do set "OBJECT_KEY=%%i"

if not defined OBJECT_KEY (
    echo [PIPELINE] ERROR: Failed to get object key from upload script.
    exit /b 1
)
echo [PIPELINE] Upload successful. Object Key: %OBJECT_KEY%
echo.


REM --- Step 2: Download file from S3 ---
echo [PIPELINE] Step 2: Downloading %OBJECT_KEY% from S3...
python download_script.py --bucket %BUCKET_NAME% --key "%OBJECT_KEY%" --output %TEMP_DOWNLOAD_FILE%
if %errorlevel% neq 0 (
    echo [PIPELINE] ERROR: Download script failed.
    exit /b 1
)
if not exist %TEMP_DOWNLOAD_FILE% (
    echo [PIPELINE] ERROR: Download script ran but output file not found.
    exit /b 1
)
echo [PIPELINE] Download successful. File saved to: %TEMP_DOWNLOAD_FILE%
echo.


REM --- Step 3: Run LLM Analysis ---
echo [PIPELINE] Step 3: Running LLM analysis on %TEMP_DOWNLOAD_FILE%...
python llm_script.py --file %TEMP_DOWNLOAD_FILE% > %ANALYSIS_RESULT_FILE% 2>&1
if %errorlevel% neq 0 (
    echo [PIPELINE] ERROR: LLM analysis script failed.
    if exist %TEMP_DOWNLOAD_FILE% del %TEMP_DOWNLOAD_FILE%
    exit /b 1
)
echo [PIPELINE] LLM analysis successful. Result saved to %ANALYSIS_RESULT_FILE%
echo.

REM --- Step 4: Display Result ---
echo [PIPELINE] Step 4: Displaying analysis result...
echo --------------------------------------------------
type %ANALYSIS_RESULT_FILE%
echo --------------------------------------------------
echo.

REM --- Cleanup ---
if exist %TEMP_DOWNLOAD_FILE% (
    del %TEMP_DOWNLOAD_FILE%
    echo [PIPELINE] Cleaned up temporary file: %TEMP_DOWNLOAD_FILE%
)

endlocal
echo.
echo [PIPELINE] Finished successfully.
