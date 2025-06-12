from typing import List, Dict, Any, Optional, Annotated
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState # For Annotated[WorkflowState, InjectedState]
from langgraph.types import Command
from src.agent.state import WorkflowState # Assuming state.py is in the same directory
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
import os
from dotenv import load_dotenv
import boto3
from botocore.errorfactory import ClientError
import logging
import tempfile
from github import Github
from git import Repo as GitRepo
from botocore.exceptions import NoCredentialsError

# --- Load .env --- #
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- AWS S3 헬퍼 함수 ---
def _get_s3_client():
    """S3 클라이언트 객체를 반환합니다.
    AWS 자격 증명은 환경 변수(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION),
    IAM 역할 등을 통해 Boto3가 자동으로 로드하도록 설정되어 있어야 합니다.
    """
    try:
        s3 = boto3.client('s3')
        # 간단한 연결 테스트 (선택 사항)
        s3.list_buckets() # 올바른 자격증명이 없으면 여기서 에러 발생
        logging.info("S3 클라이언트 생성 성공.")
        return s3
    except NoCredentialsError:
        logging.error("AWS 자격 증명을 찾을 수 없습니다. 환경 변수 또는 IAM 역할을 설정하세요.")
        raise
    except ClientError as e:
        logging.error(f"S3 클라이언트 생성 중 오류 발생: {e}")
        raise

def _upload_directory_to_s3(local_directory: str, bucket_name: str, s3_prefix: str):
    """
    로컬 디렉토리를 S3에 업로드하되, 코드 관련 파일만 필터링하여 업로드합니다.
    디렉토리 구조는 유지합니다.
    
    Args:
        local_directory (str): 업로드할 로컬 디렉토리 경로
        bucket_name (str): S3 버킷 이름
        s3_prefix (str): S3 내 저장될 경로 접두사
    """
    # 코드 관련 확장자 목록
    code_extensions = [
        '.py', '.ipynb',  # Python
        '.html', '.htm', '.css', '.js', '.jsx', '.ts', '.tsx',  # Web
        '.md', '.markdown', '.rst',  # Documentation
        '.java', '.kt', '.scala',  # JVM
        '.c', '.cpp', '.h', '.hpp',  # C/C++
        '.cs',  # C#
        '.go',  # Go
        '.rb',  # Ruby
        '.php',  # PHP
        '.swift',  # Swift
        '.rs',  # Rust
        '.sh', '.bash',  # Shell
        '.sql',  # SQL
        '.json', '.yml', '.yaml', '.xml', '.toml',  # Config
        '.txt',  # Text
    ]
    
    # 확장자 없는 특수 파일 이름 목록
    special_filenames = [
        '.gitignore', '.dockerignore',  # Git/Docker
        'Dockerfile', 'docker-compose.yml',  # Docker
        'requirements.txt', 'Pipfile', 'pyproject.toml',  # Python deps
        'package.json', 'package-lock.json', 'yarn.lock',  # JS deps
        'Gemfile', 'Gemfile.lock',  # Ruby deps
        'build.gradle', 'pom.xml',  # Java/Kotlin deps
        'Makefile', 'CMakeLists.txt',  # Build files
        'LICENSE', 'README'  # Common project files
    ]
    
    # S3 클라이언트 생성
    s3 = _get_s3_client()
    
    # 파일 수 카운트
    total_files = 0
    uploaded_files = 0
    skipped_files = 0
    
    for root, _, files in os.walk(local_directory):
        for filename in files:
            # 파일 경로 및 확장자 처리
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            
            # 확장자 확인 (확장자가 없는 파일도 이름을 검사)
            _, file_extension = os.path.splitext(filename)
            total_files += 1
            
            # 파일이 코드 관련 확장자를 가지거나 특수 파일명과 일치하는지 확인
            is_code_file = file_extension.lower() in code_extensions
            is_special_file = filename in special_filenames
            
            # .git 디렉토리 내 파일은 제외 
            if '.git' in relative_path:
                skipped_files += 1
                continue
                
            if is_code_file or is_special_file:
                try:
                    logging.info(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
                    s3.upload_file(local_path, bucket_name, s3_key)
                    uploaded_files += 1
                except ClientError as e:
                    logging.warning(f"Failed to upload {local_path} to {s3_key}: {e}")
                    # 부분적 실패 시 어떻게 처리할지 결정 (예: 계속 진행)
                except FileNotFoundError:
                    logging.error(f"업로드할 로컬 파일 없음 ({local_path})")
            else:
                skipped_files += 1
                
    logging.info(f"업로드 완료. 총 {total_files}개 파일 중 {uploaded_files}개 업로드됨, {skipped_files}개 건너뜀")




# 대화/코드 도구 (code_agent가 사용할 도구들)
@tool
def search_information(
    query: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    사용자 질문에 대한 정보를 검색합니다. (일반 검색 또는 코드 관련 검색)
    Args:
        query: 검색 쿼리
    """
    print(f"[Tool Call] search_information: query='{query}'")
    return f"'{query}'에 대한 검색 결과: [mcp로는 context7을 추천드립니다. context7은 최신 코드 문서들을 llm에 전달해주는 mcp에요.]"

@tool(return_direct=True)
def get_recommendations(
    category: str,
    preferences: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    사용자 선호도에 따라 추천을 제공합니다.
    Args:
        category: 추천 카테고리 (예: 'movies', 'books', 'restaurants', 'code_libraries')
        preferences: 사용자 선호도 설명
    """
    print(f"[Tool Call] get_recommendations: category='{category}', preferences='{preferences}'")
    return f"{category} 카테고리에서 '{preferences}'에 맞는 추천: [react]"

@tool
def track_conversation(
    current_agent_name: str,
    note: str,
    state: Annotated[WorkflowState, InjectedState],
    config: RunnableConfig
) -> str:
    """
    대화의 현재 상태나 중요한 정보를 기록합니다.
    Args:
        current_agent_name: 현재 활성화된 에이전트의 이름
        note: 기록할 내용
    """
    if not state.get('agent_history'):
        state['agent_history'] = []
    state['agent_history'].append({
        "agent": current_agent_name,
        "note": note
    })
    print(f"[Tool Call] track_conversation: agent='{current_agent_name}', note='{note}'")
    # This tool primarily updates state, so the return message is for confirmation.
    return f"대화 내용이 기록되었습니다: {note}"

def code_agent_tools(): # Renamed from conversation_tools
    """코드 및 일반 대화 에이전트를 위한 도구 목록"""
    return [
        search_information,
        get_recommendations,
        track_conversation
        # 여기에 실제 코드 작성/분석 도구를 추가해야 합니다.
    ]

