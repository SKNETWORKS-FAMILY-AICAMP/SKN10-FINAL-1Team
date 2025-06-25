# S3-LLM 자동화 파이프라인 트러블슈팅 및 최종 가이드

## 1. 개요

이 문서는 로컬 CSV 파일을 AWS S3에 업로드하고, 다시 다운로드받아 OpenAI LLM으로 분석하는 자동화 파이프라인을 구축하는 과정에서 발생한 심각한 오류들과 그 해결 과정을 상세히 기록합니다.

최종 목표는 이 파이프라인을 Django와 같은 웹 프레임워크에 안정적으로 통합하는 것이었습니다.

---

## 2. 발견된 오류 및 해결 과정

### 문제 1: 원인 불명의 프로세스 멈춤 (Deadlock)

-   **증상**: 파이썬 스크립트 실행 시, S3 업로드 또는 다운로드 단계 이후 아무런 오류 메시지 없이 프로세스가 무한정 멈추는 현상이 발생했습니다.
-   **근본 원인**: **Windows 환경**에서 부모 파이썬 프로세스가 `multiprocessing` 또는 `subprocess`를 통해 자식 파이썬 프로세스를 관리할 때, 자식 프로세스가 `boto3`나 `openai`와 같은 네트워크 라이브러리를 사용하면 **프로세스 간 표준 입출력(I/O) 파이프(Pipe)가 막히는 교착 상태(Deadlock)**가 발생하는 것으로 진단되었습니다. 이는 Windows의 프로세스 처리 방식과 관련된 미묘하고 복잡한 문제입니다.
-   **해결 과정**:
    1.  **(1차 시도 - 실패)**: `multiprocessing`을 사용해 각 작업을 별도의 프로세스로 분리했으나, 동일한 교착 상태 문제로 실패했습니다.
    2.  **(2차 시도 - 진단 성공)**: 파이썬을 완전히 배제하고, 각 스크립트를 OS가 직접 실행하도록 하는 **`run_pipeline.bat` 배치 파일**을 작성했습니다. 이 방법으로 파이프라인이 성공적으로 동작하는 것을 확인하며, **문제의 원인이 파이썬의 프로세스 관리 방식에 있음을 확신**하게 되었습니다. 이 단계는 매우 중요한 진단 과정이었습니다.
    3.  **(3차 시도 - 최종 해결)**: `.bat` 파일의 핵심 원리, 즉 **'불필요한 파이프를 만들지 않는 완벽한 프로세스 격리'**를 파이썬 코드로 구현했습니다. `pipeline_orchestrator.py`에서 `subprocess.run()`을 사용하되, **아무런 출력을 반환할 필요가 없는 `download_script.py`를 호출할 때 `capture_output=True` 인자를 제거**했습니다. 이 조치로 교착 상태의 마지막 원인을 제거하여 문제를 최종적으로 해결했습니다.

### 문제 2: LLM 분석 실패 (숨겨진 오류들)

교착 상태 문제를 해결하자, 이전에는 보이지 않았던 LLM 분석 단계의 오류들이 드러났습니다.

-   **증상**: `analysis_result.txt` 파일이 비어 있거나, 오류 메시지가 담겨 있었습니다.
-   **원인 1: `openai` 라이브러리 버전 비호환**: 설치된 `openai` 라이브러리는 v1.x 최신 버전이었으나, `llm_script.py`의 코드는 v0.x 구버전 문법을 사용하고 있었습니다.
    -   **해결**: `llm_script.py`의 코드를 `client = openai.OpenAI(...)` 및 `client.chat.completions.create(...)`와 같은 최신 v1.x 문법으로 전면 수정했습니다.
-   **원인 2: `.env` 환경 변수 로드 실패**: `llm_script.py`가 `OPENAI_API_KEY`를 제대로 불러오지 못했습니다.
    -   **해결**: 스크립트가 실행되는 위치에 관계없이 안정적으로 프로젝트 루트의 `.env` 파일을 찾을 수 있도록, `load_dotenv()` 함수를 아무 인자 없이 호출하는 가장 표준적인 방식으로 코드를 수정했습니다.

---

## 3. 최종 아키텍처 및 실행 방법

### 최종 구성 요소

-   `upload_script.py`: 단일 파일 업로드 및 S3 키 출력.
-   `download_script.py`: S3 키를 받아 파일 다운로드.
-   `llm_script.py`: 로컬 파일을 받아 LLM 분석 결과 출력.
-   **`pipeline_orchestrator.py`**: **(핵심 실행 파일)** 위 세 스크립트를 안정적인 `subprocess` 호출로 지휘(Orchestrate)하는 최종 파이썬 지휘자 스크립트입니다.

### 실행 방법

**이 자동화 파이프라인을 실행하는 유일하고 올바른 방법은 `pipeline_orchestrator.py`를 실행하는 것입니다.**

1.  **파일 준비**: 분석할 CSV 파일을 프로젝트 폴더 (`c:\dev\SKN10-FINAL_1Team\SKN10-FINAL-1Team`) 안에 위치시킵니다.
2.  **환경 변수 설정**: 프로젝트 루트에 `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `OPENAI_API_KEY`가 포함된 `.env` 파일을 생성합니다.
3.  **터미널에서 실행**:
    ```shell
    python pipeline_orchestrator.py
    ```
4.  **분석 파일 변경**: 다른 파일을 분석하려면, `pipeline_orchestrator.py` 파일 맨 아래 `if __name__ == '__main__':` 블록 안에 있는 `local_csv_file` 변수의 값을 원하는 파일명으로 변경하면 됩니다.

### Django 연동 가이드

완성된 `run_analysis_pipeline` 함수는 Django 프로젝트 어디서든 안전하게 호출할 수 있습니다.

```python
# Django의 views.py 등에서 사용하는 예시

from .pipeline_orchestrator import run_analysis_pipeline

def process_csv_view(request):
    # 사용자가 업로드한 파일을 특정 경로에 저장했다고 가정
    uploaded_file_path = 'path/to/your/uploaded_file.csv'

    # 파이프라인 실행
    analysis_result, error = run_analysis_pipeline(uploaded_file_path)

    if error:
        # 파이프라인 실패 시 에러 처리
        print(f"Pipeline failed: {error}")
        # ... 에러 페이지를 렌더링하거나 JSON 응답 반환
    else:
        # 파이프라인 성공 시 결과 처리
        print("Analysis successful!")
        # ... 결과 페이지를 렌더링하거나 JSON 응답 반환
```
