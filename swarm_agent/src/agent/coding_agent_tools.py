import os
import traceback
from typing import List, Dict, Any, Optional

from langchain.tools import StructuredTool, Tool
from langchain.pydantic_v1 import BaseModel, Field
from github import Github, Auth, GithubException, UnknownObjectException

from langchain_experimental.utilities import PythonREPL

# --- GitHub 인증 헬퍼 --- #
def get_github_instance(token: str) -> Github:
    """
    제공된 개인용 액세스 토큰으로 인증된 Github 인스턴스를 초기화하고 반환합니다.
    :param token: 사용자의 GitHub 개인용 액세스 토큰.
    :return: 인증된 Github 객체.
    :raises ValueError: 토큰이 제공되지 않은 경우.
    """
    if not token:
        raise ValueError("에이전트가 GitHub에 접근하려면 GitHub 인증 토큰이 필요합니다.")
    try:
        auth = Auth.Token(token)
        return Github(auth=auth)
    except Exception as e:
        raise RuntimeError(f"PyGithub 인스턴스 초기화 실패: {e}")

# --- 도구별 함수 및 Pydantic 스키마 정의 --- #

# 도구 0: 이슈 목록 조회
class ListIssuesSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")
    state: str = Field("open", description="이슈 상태 ('open', 'closed', 'all').")
    assignee: Optional[str] = Field(None, description="담당자 로그인 이름.")
    labels: Optional[List[str]] = Field(None, description="필터링할 레이블 이름 리스트.")

def _list_issues(**kwargs) -> List[Dict[str, Any]]:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    state = kwargs.get('state', 'open')
    assignee = kwargs.get('assignee')
    labels = kwargs.get('labels')
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        # PyGithub는 None 대신 NotSet을 사용하여 매개변수가 제공되지 않았음을 나타냅니다.
        issues = repo.get_issues(state=state, assignee=assignee if assignee else NotSet, labels=labels if labels else NotSet)
        return [{
            "number": issue.number, "title": issue.title, "state": issue.state,
            "user": issue.user.login, "html_url": issue.html_url
        } for issue in issues]
    except UnknownObjectException:
        raise ValueError(f"리포지토리 '{repo_full_name}'을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"이슈 목록 조회 중 오류 발생: {e.data.get('message', e.status)}")

# 도구 1: 단일 이슈 조회
class GetIssueSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")
    issue_number: int = Field(..., description="조회할 이슈의 번호.")

def _get_issue_details(**kwargs) -> Dict[str, Any]:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    issue_number = kwargs['issue_number']
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        issue = repo.get_issue(number=issue_number)
        return {
            "number": issue.number, "title": issue.title, "body": issue.body, "state": issue.state,
            "user": issue.user.login, "assignees": [a.login for a in issue.assignees],
            "labels": [l.name for l in issue.labels], "comments_count": issue.comments,
            "html_url": issue.html_url
        }
    except UnknownObjectException:
        raise ValueError(f"리포지토리 '{repo_full_name}'에서 이슈 #{issue_number}을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"이슈 상세 정보 조회 중 오류 발생: {e.data.get('message', e.status)}")

# 도구 2: 이슈에 댓글 추가
class CommentOnIssueSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")
    issue_number: int = Field(..., description="댓글을 추가할 이슈의 번호.")
    body: str = Field(..., description="댓글 내용.")

def _add_comment_to_issue(**kwargs) -> Dict[str, Any]:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    issue_number = kwargs['issue_number']
    body = kwargs['body']
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        issue = repo.get_issue(number=issue_number)
        comment = issue.create_comment(body)
        return {"id": comment.id, "user": comment.user.login, "html_url": comment.html_url}
    except UnknownObjectException:
        raise ValueError(f"이슈 #{issue_number}을(를) 찾을 수 없어 댓글을 추가할 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"이슈 댓글 추가 중 오류 발생: {e.data.get('message', e.status)}")

# 도구 3: 풀 리퀘스트(PR) 목록 조회
class ListPullRequestsSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")
    state: str = Field("open", description="PR 상태 ('open', 'closed', 'all').")
    base: Optional[str] = Field(None, description="필터링할 베이스 브랜치 이름.")

def _list_pull_requests(**kwargs) -> List[Dict[str, Any]]:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    state = kwargs.get('state', 'open')
    base = kwargs.get('base')
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        pulls = repo.get_pulls(state=state, sort="created", base=base if base else NotSet)
        return [{
            "number": pr.number, "title": pr.title, "state": pr.state, "user": pr.user.login,
            "head_branch": pr.head.ref, "base_branch": pr.base.ref, "html_url": pr.html_url
        } for pr in pulls]
    except UnknownObjectException:
        raise ValueError(f"리포지토리 '{repo_full_name}'을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"풀 리퀘스트 목록 조회 중 오류 발생: {e.data.get('message', e.status)}")

# 도구 4: 단일 풀 리퀘스트 조회
class GetPullRequestSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")
    pr_number: int = Field(..., description="조회할 PR의 번호.")

def _get_pull_request_details(**kwargs) -> Dict[str, Any]:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    pr_number = kwargs['pr_number']
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        pr = repo.get_pull(number=pr_number)
        return {
            "number": pr.number, "title": pr.title, "body": pr.body, "state": pr.state,
            "user": pr.user.login, "merged": pr.merged, "mergeable": pr.mergeable,
            "changed_files": pr.changed_files, "additions": pr.additions, "deletions": pr.deletions,
            "html_url": pr.html_url
        }
    except UnknownObjectException:
        raise ValueError(f"리포지토리 '{repo_full_name}'에서 PR #{pr_number}을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"풀 리퀘스트 상세 정보 조회 중 오류 발생: {e.data.get('message', e.status)}")

# 도구 5 & 7: 풀 리퀘스트의 파일 목록 조회
class ListPullRequestFilesSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")
    pr_number: int = Field(..., description="파일 목록을 조회할 PR의 번호.")

def _list_files_in_pull_request(**kwargs) -> List[Dict[str, Any]]:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    pr_number = kwargs['pr_number']
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        pr = repo.get_pull(number=pr_number)
        files = pr.get_files()
        return [{"filename": file.filename, "status": file.status, "changes": file.changes} for file in files]
    except UnknownObjectException:
        raise ValueError(f"PR #{pr_number}을(를) 찾을 수 없어 파일 목록을 조회할 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"PR 파일 목록 조회 중 오류 발생: {e.data.get('message', e.status)}")

# 도구 6: 풀 리퀘스트 생성
class CreatePullRequestSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")
    title: str = Field(..., description="PR 제목.")
    body: str = Field(..., description="PR 본문.")
    head_branch: str = Field(..., description="변경 사항이 있는 브랜치 (예: 'feature-branch').")
    base_branch: str = Field(..., description="변경 사항을 병합할 대상 브랜치 (예: 'main').")

def _create_pull_request(**kwargs) -> Dict[str, Any]:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    title = kwargs['title']
    body = kwargs['body']
    head_branch = kwargs['head_branch']
    base_branch = kwargs['base_branch']
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        pr = repo.create_pull(title=title, body=body, head=head_branch, base=base_branch)
        return {"number": pr.number, "title": pr.title, "html_url": pr.html_url}
    except GithubException as e:
        if e.status == 422:
            message = e.data.get('message', '')
            if "No commits between" in message:
                raise ValueError(f"'{base_branch}'와 '{head_branch}' 사이에 커밋이 없어 PR을 생성할 수 없습니다.")
            if "A pull request already exists" in message:
                raise ValueError("동일한 브랜치에 대한 풀 리퀘스트가 이미 존재합니다.")
        raise RuntimeError(f"풀 리퀘스트 생성 중 오류 발생: {e.data.get('message', e.status)}")

# 도구 11: 파일 삭제
class DeleteFileSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")
    file_path: str = Field(..., description="삭제할 파일의 경로.")
    commit_message: str = Field(..., description="커밋 메시지.")
    branch: str = Field(..., description="파일을 삭제할 브랜치 이름.")

def _delete_file(**kwargs) -> Dict[str, str]:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    file_path = kwargs['file_path']
    commit_message = kwargs['commit_message']
    branch = kwargs['branch']
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        contents = repo.get_contents(file_path, ref=branch)
        result = repo.delete_file(contents.path, commit_message, contents.sha, branch=branch)
        return {"commit_sha": result['commit'].sha, "commit_url": result['commit'].html_url}
    except UnknownObjectException:
        raise ValueError(f"파일 '{file_path}'을(를) 브랜치 '{branch}'에서 찾을 수 없어 삭제할 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"파일 삭제 중 오류 발생: {e.data.get('message', e.status)}")

# 도구 12, 13, 17: 디렉토리 파일 목록 조회
class ListFilesInDirectorySchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")
    directory_path: str = Field("", description="파일 목록을 조회할 디렉토리 경로. 비워두면 루트 디렉토리를 조회합니다.")
    branch: Optional[str] = Field(None, description="파일 목록을 조회할 브랜치 이름. 지정하지 않으면 기본 브랜치 사용.")

def _list_files_in_directory(**kwargs) -> List[Dict[str, Any]]:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    directory_path = kwargs.get('directory_path', '')
    branch = kwargs.get('branch')
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        contents = repo.get_contents(directory_path, ref=branch)
        if not isinstance(contents, list):
             raise ValueError(f"경로 '{directory_path}'는 파일입니다. 디렉토리 경로를 지정해야 합니다.")
        return [{"name": content.name, "path": content.path, "type": content.type} for content in contents]
    except UnknownObjectException:
        raise ValueError(f"경로 '{directory_path}'을(를) 브랜치 '{branch or 'default'}'에서 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"디렉토리 파일 목록 조회 중 오류 발생: {e.data.get('message', e.status)}")

# 도구 16: 새 브랜치 생성
class CreateBranchSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")
    new_branch_name: str = Field(..., description="새로 생성할 브랜치의 이름.")
    source_branch_name: str = Field(..., description="베이스가 될 소스 브랜치의 이름.")

def _create_branch(**kwargs) -> Dict[str, str]:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    new_branch_name = kwargs['new_branch_name']
    source_branch_name = kwargs['source_branch_name']
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        source_branch = repo.get_branch(source_branch_name)
        repo.create_git_ref(ref=f"refs/heads/{new_branch_name}", sha=source_branch.commit.sha)
        return {"status": "success", "message": f"브랜치 '{new_branch_name}'이(가) '{source_branch_name}' 브랜치로부터 생성되었습니다."}
    except UnknownObjectException:
        raise ValueError(f"소스 브랜치 '{source_branch_name}'을(를) 찾을 수 없습니다.")
    except GithubException as e:
        if e.status == 422: # Unprocessable Entity, ref already exists
            raise ValueError(f"브랜치 '{new_branch_name}'이(가) 이미 존재합니다.")
        raise RuntimeError(f"브랜치 생성 중 오류 발생: {e.data.get('message', e.status)}")

# 도구 18: 이슈 및 PR 검색
class SearchIssuesAndPRsSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    query: str = Field(..., description="GitHub 검색 쿼리 문자열 (예: 'repo:owner/repo is:open label:bug').")

def _search_issues(**kwargs) -> List[Dict[str, Any]]:
    token = kwargs['token']
    query = kwargs['query']
    g = get_github_instance(token)
    try:
        issues = g.search_issues(query=query)
        return [{
            "number": issue.number, "title": issue.title, "repository": issue.repository.full_name,
            "state": issue.state, "html_url": issue.html_url
        } for issue in issues]
    except GithubException as e:
        raise RuntimeError(f"이슈 검색 중 오류 발생: {e.data.get('message', e.status)}")

# 도구 19: 코드 검색
class SearchCodeSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    query: str = Field(..., description="검색할 코드 또는 텍스트. 'repo:owner/repo' 한정자를 사용하여 범위를 좁힐 수 있습니다.")

def _search_code(**kwargs) -> List[Dict[str, Any]]:
    token = kwargs['token']
    query = kwargs['query']
    g = get_github_instance(token)
    try:
        files = g.search_code(query=query)
        return [{
            "path": file.path, "name": file.name, "html_url": file.html_url,
            "repository": file.repository.full_name
        } for file in files]
    except GithubException as e:
        raise RuntimeError(f"코드 검색 중 오류 발생: {e.data.get('message', e.status)}")

# 도구 20: 리뷰 요청 생성
class CreateReviewRequestSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")
    pr_number: int = Field(..., description="리뷰를 요청할 PR의 번호.")
    reviewers: List[str] = Field(..., description="리뷰어로 지정할 사용자 로그인 이름 리스트.")

def _create_review_request(**kwargs) -> Dict[str, Any]:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    pr_number = kwargs['pr_number']
    reviewers = kwargs['reviewers']
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        pr = repo.get_pull(number=pr_number)
        pr.create_review_request(reviewers=reviewers)
        return {"status": "success", "message": f"PR #{pr_number}에 리뷰 요청이 성공적으로 전송되었습니다."}
    except UnknownObjectException:
        raise ValueError(f"PR #{pr_number}을(를) 찾을 수 없어 리뷰를 요청할 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"리뷰 요청 생성 중 오류 발생: {e.data.get('message', e.status)}")

# --- 기존에 제공된 함수들 --- #

# 리포지토리 목록 조회
class ListRepositoriesSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    username: Optional[str] = Field(None, description="리포지토리를 조회할 특정 사용자의 이름. None이면 인증된 사용자를 대상으로 합니다.")
    org_name: Optional[str] = Field(None, description="리포지토리를 조회할 조직의 이름. username과 org_name 중 하나만 지정해야 합니다.")

def _list_repositories(**kwargs) -> List[Dict[str, Any]]:
    token = kwargs['token']
    username = kwargs.get('username')
    org_name = kwargs.get('org_name')
    if username and org_name:
        raise ValueError("username과 org_name은 동시에 지정할 수 없습니다.")
    g = get_github_instance(token)
    repo_list = []
    try:
        if org_name:
            target = g.get_organization(org_name)
        elif username:
            target = g.get_user(username)
        else:
            target = g.get_user()
        
        repos = target.get_repos()
        for repo in repos:
            repo_list.append({"full_name": repo.full_name, "description": repo.description, "private": repo.private, "html_url": repo.html_url})
        return repo_list
    except GithubException as e:
        raise RuntimeError(f"리포지토리 조회 중 오류 발생: {e.data.get('message', e.status)}")

# 브랜치 목록 조회
class ListBranchesSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")

def _list_branches(**kwargs) -> List[Dict[str, Any]]:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        branches = repo.get_branches()
        return [{"name": branch.name, "commit_sha": branch.commit.sha} for branch in branches]
    except UnknownObjectException:
        raise ValueError(f"리포지토리 '{repo_full_name}'을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"브랜치 목록 조회 중 오류 발생: {e.data.get('message', e.status)}")

# 파일 내용 읽기
class ReadFileSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")
    file_path: str = Field(..., description="읽을 파일의 경로.")
    branch: Optional[str] = Field(None, description="파일을 읽을 브랜치 이름. 지정하지 않으면 기본 브랜치 사용.")

def _read_file(**kwargs) -> str:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    file_path = kwargs['file_path']
    branch = kwargs.get('branch')
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        contents = repo.get_contents(file_path, ref=branch) if branch else repo.get_contents(file_path)
        if isinstance(contents, list):
            raise ValueError(f"경로 '{file_path}'는 디렉터리입니다. 파일 경로를 지정해야 합니다.")
        return contents.decoded_content.decode("utf-8")
    except UnknownObjectException:
        raise ValueError(f"파일 '{file_path}'을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"파일 읽기 중 오류 발생: {e.data.get('message', e.status)}")

# 새 파일 생성
class CreateFileSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")
    file_path: str = Field(..., description="생성할 파일의 경로.")
    commit_message: str = Field(..., description="커밋 메시지.")
    content: str = Field(..., description="파일에 쓸 내용.")
    branch: str = Field(..., description="파일을 생성할 브랜치 이름.")

def _create_file(**kwargs) -> Dict[str, str]:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    file_path = kwargs['file_path']
    commit_message = kwargs['commit_message']
    content = kwargs['content']
    branch = kwargs['branch']
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        result = repo.create_file(file_path, commit_message, content, branch=branch)
        return {"content_path": result['content'].path, "commit_sha": result['commit'].sha}
    except GithubException as e:
        if e.status == 422:
            raise ValueError(f"파일 '{file_path}'이(가) 브랜치 '{branch}'에 이미 존재합니다.")
        raise RuntimeError(f"파일 생성 중 오류 발생: {e.data.get('message', e.status)}")

# 기존 파일 수정
class UpdateFileSchema(BaseModel):
    token: str = Field(..., description="GitHub 인증용 개인 액세스 토큰.")
    repo_full_name: str = Field(..., description="'owner/repo' 형식의 리포지토리 전체 이름.")
    file_path: str = Field(..., description="수정할 파일의 경로.")
    commit_message: str = Field(..., description="커밋 메시지.")
    new_content: str = Field(..., description="파일의 새 내용.")
    branch: str = Field(..., description="파일을 수정할 브랜치 이름.")

def _update_file(**kwargs) -> Dict[str, str]:
    token = kwargs['token']
    repo_full_name = kwargs['repo_full_name']
    file_path = kwargs['file_path']
    commit_message = kwargs['commit_message']
    new_content = kwargs['new_content']
    branch = kwargs['branch']
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        contents = repo.get_contents(file_path, ref=branch)
        result = repo.update_file(contents.path, commit_message, new_content, contents.sha, branch=branch)
        return {"content_path": result['content'].path, "commit_sha": result['commit'].sha}
    except UnknownObjectException:
        raise ValueError(f"파일 '{file_path}'을(를) 브랜치 '{branch}'에서 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"파일 수정 중 오류 발생: {e.data.get('message', e.status)}")


# --- 도구 리스트 취합 --- #

def get_all_coding_tools() -> List[Tool]:
    """
    코딩 어시스턴트를 위한 모든 LangChain 도구를 초기화하고 리스트로 반환합니다.
    """
    # 순서: 이슈 -> PR -> 파일 -> 브랜치 -> 리포지토리 -> 검색
    github_tools = [
        # 이슈
        StructuredTool(
            name="github_list_issues",
            description="리포지토리의 이슈 목록을 조회합니다.",
            func=_list_issues,
            args_schema=ListIssuesSchema
        ),
        StructuredTool(
            name="github_get_issue_details",
            description="특정 이슈의 상세 정보를 조회합니다.",
            func=_get_issue_details,
            args_schema=GetIssueSchema
        ),
        StructuredTool(
            name="github_add_comment_to_issue",
            description="특정 이슈에 댓글을 추가합니다.",
            func=_add_comment_to_issue,
            args_schema=CommentOnIssueSchema
        ),
        # PR
        StructuredTool(
            name="github_list_pull_requests",
            description="리포지토리의 풀 리퀘스트(PR) 목록을 조회합니다.",
            func=_list_pull_requests,
            args_schema=ListPullRequestsSchema
        ),
        StructuredTool(
            name="github_get_pull_request_details",
            description="특정 PR의 상세 정보를 조회합니다.",
            func=_get_pull_request_details,
            args_schema=GetPullRequestSchema
        ),
        StructuredTool(
            name="github_list_pr_files",
            description="특정 PR에 포함된 파일 목록을 조회합니다.",
            func=_list_files_in_pull_request,
            args_schema=ListPullRequestFilesSchema
        ),
        StructuredTool(
            name="github_create_pull_request",
            description="새로운 풀 리퀘스트를 생성합니다.",
            func=_create_pull_request,
            args_schema=CreatePullRequestSchema
        ),
        StructuredTool(
            name="github_create_review_request",
            description="특정 PR에 대한 리뷰를 요청합니다.",
            func=_create_review_request,
            args_schema=CreateReviewRequestSchema
        ),
        # 파일
        StructuredTool(
            name="github_read_file",
            description="리포지토리 내 특정 파일의 내용을 읽어옵니다.",
            func=_read_file,
            args_schema=ReadFileSchema
        ),
        StructuredTool(
            name="github_create_file",
            description="리포지토리에 새 파일을 생성합니다.",
            func=_create_file,
            args_schema=CreateFileSchema
        ),
        StructuredTool(
            name="github_update_file",
            description="리포지토리의 기존 파일을 수정합니다.",
            func=_update_file,
            args_schema=UpdateFileSchema
        ),
        StructuredTool(
            name="github_delete_file",
            description="리포지토리에서 파일을 삭제합니다.",
            func=_delete_file,
            args_schema=DeleteFileSchema
        ),
        StructuredTool(
            name="github_list_directory_contents",
            description="리포지토리의 특정 디렉토리 내 파일 및 폴더 목록을 조회합니다.",
            func=_list_files_in_directory,
            args_schema=ListFilesInDirectorySchema
        ),
        # 브랜치
        StructuredTool(
            name="github_list_branches",
            description="리포지토리의 모든 브랜치 목록을 조회합니다.",
            func=_list_branches,
            args_schema=ListBranchesSchema
        ),
        StructuredTool(
            name="github_create_branch",
            description="새로운 브랜치를 생성합니다.",
            func=_create_branch,
            args_schema=CreateBranchSchema
        ),
        # 리포지토리
        StructuredTool(
            name="github_list_repositories",
            description="사용자 또는 조직의 리포지토리 목록을 조회합니다.",
            func=_list_repositories,
            args_schema=ListRepositoriesSchema
        ),
        # 검색
        StructuredTool(
            name="github_search_issues_and_prs",
            description="쿼리를 사용하여 이슈 및 PR을 검색합니다.",
            func=_search_issues,
            args_schema=SearchIssuesAndPRsSchema
        ),
        StructuredTool(
            name="github_search_code",
            description="쿼리를 사용하여 코드를 검색합니다.",
            func=_search_code,
            args_schema=SearchCodeSchema
        ),
    ]

    # Python REPL 도구 추가
    python_repl = PythonREPL()
    python_repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    )

    all_tools = github_tools + [python_repl_tool]
    
    print("\n--- [DEBUG] Assembling tools for Coding Assistant (Token-Based) ---")
    for i, tool in enumerate(all_tools):
        print(f"--- [DEBUG] Tool[{i}]: {tool.name}")
    print("--- [DEBUG] Tool assembly finished ---\n")

    return all_tools

    # Python REPL 도구 추가
    python_repl = PythonREPL()
    python_repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    )

    all_tools = github_tools + [python_repl_tool]
    
    print("\n--- Assembling tools for Coding Assistant (Token-Based) ---")
    for i, tool in enumerate(all_tools):
        print(f"--- Tool[{i}]: {tool.name}")
    print("--- Tool assembly finished ---\n")

    return all_tools

# --- 예시 사용법 ---
if __name__ == '__main__':
    # 이 스크립트를 직접 실행할 때 도구 목록을 출력합니다.
    # 실제 에이전트에서는 이 함수를 호출하여 도구 리스트를 가져와 사용합니다.
    # GITHUB_TOKEN 환경 변수를 설정해야 합니다.
    tools = get_all_coding_tools()

    # # 예시: 특정 도구 테스트 (토큰 필요)
    # GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
    # if GITHUB_TOKEN:
    #     try:
    #         # 테스트할 리포지토리
    #         TEST_REPO = "PyGithub/PyGithub" 
            
    #         print(f"\n--- Testing github_list_branches on {TEST_REPO} ---")
    #         branches = _list_branches(token=GITHUB_TOKEN, repo_full_name=TEST_REPO)
    #         print(branches[:5]) # 처음 5개 브랜치만 출력

    #         print(f"\n--- Testing github_list_issues on {TEST_REPO} ---")
    #         issues = _list_issues(token=GITHUB_TOKEN, repo_full_name=TEST_REPO, state="closed", labels=["bug"])
    #         print(issues[:5]) # 처음 5개 이슈만 출력

    #     except Exception as e:
    #         print(f"An error occurred during testing: {e}")
    # else:
    #     print("\n GITHUB_TOKEN 환경 변수가 설정되지 않아 테스트를 건너뜁니다.")

