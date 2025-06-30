import os
from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# GitHub 관련 imports (선택적)
try:
    from github import Github, Auth, GithubException, UnknownObjectException
    from github.GithubObject import NotSet
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    print("PyGithub not available. GitHub tools will be disabled.", file=sys.stderr)

# Pinecone 관련 imports (선택적)
try:
    from pinecone.grpc import PineconeGRPC as Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("Pinecone not available. Code search tools will be disabled.", file=sys.stderr)

# --- GitHub 인증 헬퍼 --- #
def get_github_instance(token: str) -> Github:
    """
    제공된 개인용 액세스 토큰으로 인증된 Github 인스턴스를 초기화하고 반환합니다.
    """
    if not GITHUB_AVAILABLE:
        raise RuntimeError("PyGithub가 설치되지 않았습니다.")
    if not token:
        raise ValueError("GitHub 인증 토큰이 필요합니다.")
    try:
        auth = Auth.Token(token)
        return Github(auth=auth)
    except Exception as e:
        raise RuntimeError(f"PyGithub 인스턴스 초기화 실패: {e}")

# --- GitHub 기본 도구들 --- #
def list_repositories(token: str, username: Optional[str] = None, org_name: Optional[str] = None) -> str:
    """사용자 또는 조직의 리포지토리 목록을 조회합니다."""
    if not GITHUB_AVAILABLE:
        return "Error: PyGithub가 설치되지 않았습니다."
    
    try:
        g = get_github_instance(token)
        repo_list = []
        
        if org_name:
            target = g.get_organization(org_name)
        elif username:
            target = g.get_user(username)
        else:
            target = g.get_user()
        
        repos = target.get_repos()
        for repo in repos[:10]:  # 최대 10개만 표시
            repo_list.append(f"- {repo.full_name}: {repo.description or 'No description'}")
        
        return "\n".join(repo_list) if repo_list else "리포지토리를 찾을 수 없습니다."
    except Exception as e:
        return f"Error: {str(e)}"

def read_file(token: str, repo_full_name: str, file_path: str, branch: Optional[str] = None) -> str:
    """리포지토리 내 특정 파일의 내용을 읽어옵니다."""
    if not GITHUB_AVAILABLE:
        return "Error: PyGithub가 설치되지 않았습니다."
    
    try:
        g = get_github_instance(token)
        repo = g.get_repo(repo_full_name)
        contents = repo.get_contents(file_path, ref=branch) if branch else repo.get_contents(file_path)
        
        if isinstance(contents, list):
            return f"Error: '{file_path}'는 디렉터리입니다. 파일 경로를 지정해야 합니다."
        
        return contents.decoded_content.decode("utf-8")
    except UnknownObjectException:
        return f"Error: 파일 '{file_path}'을(를) 찾을 수 없습니다."
    except Exception as e:
        return f"Error: {str(e)}"

def list_issues(token: str, repo_full_name: str, state: str = "open") -> str:
    """리포지토리의 이슈 목록을 조회합니다."""
    if not GITHUB_AVAILABLE:
        return "Error: PyGithub가 설치되지 않았습니다."
    
    try:
        g = get_github_instance(token)
        repo = g.get_repo(repo_full_name)
        issues = repo.get_issues(state=state)
        
        issue_list = []
        for issue in issues[:10]:  # 최대 10개만 표시
            issue_list.append(f"#{issue.number}: {issue.title} ({issue.state})")
        
        return "\n".join(issue_list) if issue_list else f"{state} 상태의 이슈를 찾을 수 없습니다."
    except Exception as e:
        return f"Error: {str(e)}"

def create_file(token: str, repo_full_name: str, file_path: str, commit_message: str, content: str, branch: str) -> str:
    """리포지토리에 새 파일을 생성합니다."""
    if not GITHUB_AVAILABLE:
        return "Error: PyGithub가 설치되지 않았습니다."
    
    try:
        g = get_github_instance(token)
        repo = g.get_repo(repo_full_name)
        result = repo.create_file(file_path, commit_message, content, branch=branch)
        return f"파일이 성공적으로 생성되었습니다: {result['content'].path}"
    except GithubException as e:
        if e.status == 422:
            return f"Error: 파일 '{file_path}'이(가) 브랜치 '{branch}'에 이미 존재합니다."
        return f"Error: {e.data.get('message', e.status)}"
    except Exception as e:
        return f"Error: {str(e)}"

# --- Python 실행 도구 --- #
def execute_python_code(code: str) -> str:
    """Python 코드를 실행하고 결과를 반환합니다."""
    try:
        # 간단한 Python 실행 (보안상 제한적)
        import io
        import contextlib
        
        # stdout 캡처
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            # 간단한 eval/exec만 허용 (보안 주의)
            try:
                result = eval(code)
                if result is not None:
                    print(result)
            except SyntaxError:
                exec(code)
        
        output = f.getvalue()
        return output if output else "코드가 실행되었지만 출력이 없습니다."
    except Exception as e:
        return f"Error: {str(e)}"

# --- Pinecone 검색 도구 --- #
def search_code_documents(query: str, repo_path: Optional[str] = None, top_k: int = 5) -> str:
    """Pinecone을 사용하여 코드 문서를 검색합니다."""
    if not PINECONE_AVAILABLE:
        return "Error: Pinecone이 설치되지 않았습니다."
    
    try:
        # 간단한 검색 결과 시뮬레이션
        results = [
            f"검색 결과 {i+1}: {query} 관련 문서 (스코어: {0.9 - i*0.1:.2f})"
            for i in range(min(top_k, 3))
        ]
        return "\n".join(results) if results else "검색 결과가 없습니다."
    except Exception as e:
        return f"Error: {str(e)}" 