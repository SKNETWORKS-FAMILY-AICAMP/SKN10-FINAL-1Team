# S3 파일 처리 및 LLM 분석 워크플로우

이 문서는 CSV 파일이 사용자로부터 업로드되어 S3에 저장되고, FastAPI 서버에서 해당 데이터를 분석하여 사용자에게 답변을 제공하는 전체 과정을 설명합니다.

## 전체 흐름도

```
1. 사용자 (브라우저)
     |
     |--- CSV 파일 업로드 ---> 2. Django 서버 (`backend`)
     |                           |
     |                           |--- S3에 파일 저장 & Key 반환 ---> 3. AWS S3
     |                           |
     |<-- s3_object_key ---|
     |
     |--- 분석 요청 (질문 + s3_object_key) ---> 4. FastAPI 서버 (`fastapi_server`)
                                                 |
                                                 |--- S3에서 직접 데이터 로드 (pandas)
                                                 |--- LLM을 이용한 분석 코드 생성
                                                 |--- 코드 실행 및 결과 반환
                                                 |
     |<--- 분석 결과 (스트리밍) ---|

```

---

## 1. CSV 파일 업로드 (Django `backend`)

-   **사용자 인터페이스**: 사용자는 `chatbot.html` 페이지에서 'Upload CSV' 버튼을 클릭하여 로컬 파일을 선택합니다.
-   **파일 전송**: `chatbot.html`의 `handleFileUpload` JavaScript 함수가 선택된 파일을 `FormData`에 담아 Django 서버의 `/conversations/upload_csv` 엔드포인트로 비동기 POST 요청을 보냅니다.
-   **S3 저장**: Django 서버의 `views.py`에 구현된 `upload_csv` 뷰가 요청을 처리합니다.
    -   `boto3` 라이브러리를 사용하여 수신된 파일을 AWS S3 버킷에 직접 업로드합니다.
    -   업로드 시 사용된 파일명을 `s3_object_key`로 정의합니다.
-   **키 반환**: 파일 저장이 완료되면, Django 서버는 `s3_object_key`를 JSON 형식으로 프론트엔드에 반환합니다.
-   **키 저장**: 프론트엔드는 반환된 `s3_object_key`를 브라우저의 `sessionStorage`에 저장하여 현재 세션 동안 파일의 위치를 기억합니다.

## 2. 데이터 분석 요청 (FastAPI `fastapi_server`)

-   **질문 전송**: 사용자가 CSV 파일과 관련된 질문을 입력하고 전송하면, `handleChatSubmission` JavaScript 함수가 실행됩니다.
-   **요청 구성**: 이 함수는 사용자 질문과 `sessionStorage`에 저장된 `s3_object_key`를 함께 FastAPI 서버의 `/chat/stream` 엔드포인트로 전송합니다.

## 3. S3 데이터 직접 분석 (Code Interpreter 패턴)

-   **요청 수신**: FastAPI 서버는 질문과 `s3_object_key`를 수신하고, 이를 LangGraph 에이전트로 전달합니다.
-   **도구 선택**: 에이전트는 질문의 의도를 파악하고, CSV 분석에 가장 적합한 `analyze_csv_data` 도구를 선택합니다.
-   **S3 직접 접근 (No Download!)**:
    -   `fastapi_server/agent/tools.py`의 `analyze_csv_data` 함수가 실행됩니다.
    -   **핵심 변경 사항**: 이 함수는 더 이상 별도의 스크립트를 통해 파일을 로컬에 다운로드하지 않습니다.
    -   대신, `pandas.read_csv("s3://<bucket-name>/<s3_object_key>")` 코드를 사용하여 S3에 저장된 CSV 파일을 메모리로 직접 스트리밍하여 DataFrame으로 읽어 들입니다. (`s3fs` 라이브러리 필요)
-   **LLM을 이용한 코드 생성 및 실행**:
    1.  **스키마 전송**: 생성된 DataFrame의 기본 정보(컬럼명, 데이터 타입, 상위 5개 행)를 추출합니다.
    2.  **코드 생성**: 이 스키마 정보와 사용자 질문을 LLM(GPT-4o)에 전달하여, 데이터 분석을 수행할 수 있는 Python 코드를 생성하도록 요청합니다.
    3.  **코드 실행**: 생성된 Python 코드를 서버 환경에서 `exec()`를 통해 안전하게 실행하여 분석 결과를 얻습니다.
    4.  **최종 답변 생성**: 실행 결과(예: 숫자, 표, 그래프 데이터 등)를 다시 LLM에 전달하여 사용자가 이해하기 쉬운 자연스러운 한국어 문장으로 최종 답변을 생성합니다.
-   **결과 스트리밍**: 생성된 최종 답변은 FastAPI 서버를 통해 사용자 브라우저로 실시간 스트리밍되어 챗봇 UI에 표시됩니다.

## 주요 개선 사항

-   **성능 향상**: S3 파일을 로컬 디스크에 다운로드하는 중간 단계를 제거하고 메모리에서 직접 처리함으로써, I/O 병목 현상을 해소하고 분석 속도를 크게 개선했습니다.
-   **프로세스 단순화**: 불필요한 `download_script.py` 파일을 제거하고 관련 로직을 `tools.py`로 통합하여 코드베이스를 단순화하고 유지보수성을 높였습니다.
