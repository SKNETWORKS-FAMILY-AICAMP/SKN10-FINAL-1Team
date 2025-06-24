종합 구현 가이드: PyGithub을 활용한 GitHub 자동화 및 React 에이전트 도구 구축섹션 1: 기본 원칙: 인증 및 PyGithub 객체 모델이 섹션에서는 이어지는 모든 도구의 기반이 되는 핵심 개념을 정립합니다. 개발자가 개별 기능 구현에 앞서 견고한 정신 모델을 구축할 수 있도록 설계되었습니다.1.1. 표준 인증 패턴: 토큰 기반 접근모든 도구는 사용자의 GitHub 인증 토큰을 수락해야 한다는 핵심 요구사항이 있습니다. 이 섹션에서는 이를 위한 표준적이고 재사용 가능한 코드 블록을 상세히 설명합니다.가장 직접적이고 안전한 인증 방법은 개인용 액세스 토큰(Personal Access Token, PAT)을 사용하는 것입니다. PyGithub 라이브러리는 이를 위해 github.Auth.Token 클래스를 제공하며, 이는 사용자 이름과 비밀번호를 사용하는 기존 방식보다 월등한 보안성을 가집니다.1 사용자 이름/비밀번호 인증은 더 이상 사용되지 않으며 보안에 취약합니다.1 따라서 모든 도구 함수의 시작점은 토큰을 사용하여 Github의 메인 인스턴스를 초기화하는 것입니다.다음은 토큰으로 인증된 Github 인스턴스를 생성하는 표준 함수입니다.Pythonfrom github import Github, Auth, GithubException

def get_github_instance(token: str):
    """
    제공된 개인용 액세스 토큰으로 인증된 Github 인스턴스를 초기화하고 반환합니다.

    :param token: 사용자의 GitHub 개인용 액세스 토큰.
    :return: 인증된 Github 객체.
    :raises ValueError: 토큰이 제공되지 않은 경우.
    """
    if not token:
        raise ValueError("GitHub 인증 토큰이 필요합니다.")
    try:
        auth = Auth.Token(token)
        return Github(auth=auth)
    except Exception as e:
        # 라이브러리 초기화 중 발생할 수 있는 예기치 않은 오류 처리
        raise RuntimeError(f"PyGithub 인스턴스 초기화 실패: {e}")

이 접근 방식은 현대적인 GitHub API 연동의 표준이며, 4와 5에서도 토큰 우선 접근 방식을 권장합니다. GitHub Enterprise를 사용하는 경우, base_url 매개변수를 https://{hostname}/api/v3 형식으로 명시해야 합니다. 이 점을 누락하면 인증 실패로 이어질 수 있으며, 이는 개발자들이 흔히 겪는 문제점 중 하나입니다.3PyGithub의 인증 메커니즘은 "지연(lazy)" 방식으로 동작한다는 점을 이해하는 것이 매우 중요합니다. Github(auth=auth)를 호출하여 인스턴스를 생성하는 것만으로는 즉시 토큰의 유효성을 검증하지 않습니다. 실제 API 호출은 g.get_user().login과 같이 객체의 속성에 접근할 때 비로소 발생합니다.3 이러한 동작 방식은 디버깅 시 혼란을 야기할 수 있습니다. 예를 들어, 유효하지 않은 토큰으로 Github 객체를 생성해도 즉시 오류가 발생하지 않기 때문입니다.따라서 토큰의 유효성을 명시적으로 확인하려면, 인증된 사용자의 로그인 정보와 같은 간단한 "핑(ping)" 작업을 수행하는 것이 필수적입니다. 이는 도구 22: 인증된 사용자 정보 가져오기가 단순한 유틸리티를 넘어, 에이전트의 GitHub 연결 상태를 확인하는 근본적인 상태 점검(health check) 도구로 기능해야 함을 시사합니다. React 애플리케이션과 같은 클라이언트 측에서는 에이전트 초기화 시퀀스에 이와 같은 검증 함수를 호출하여 인증 성공 여부를 즉시 확인하고 사용자에게 명확한 피드백을 제공하는 것이 좋습니다.1.2. PyGithub 객체를 통한 GitHub API 탐색PyGithub 라이브러리는 GitHub API의 복잡성을 계층적인 객체 모델로 추상화하여 개발자가 보다 직관적으로 API를 사용할 수 있도록 돕습니다. 이 구조를 이해하는 것은 효율적인 도구 개발의 핵심입니다.객체 모델 계층 구조:Github (진입점): 인증 정보를 사용하여 초기화되는 최상위 객체입니다.4 모든 작업은 이 객체에서 시작됩니다.AuthenticatedUser 및 NamedUser: Github 객체에서 g.get_user()를 호출하면 현재 인증된 사용자를 나타내는 AuthenticatedUser 객체를 얻습니다. 특정 사용자를 지정하려면 g.get_user("username")을 사용하여 NamedUser 객체를 얻을 수 있습니다.6Organization 및 Repository: g.get_organization("org_name")을 통해 조직 객체를, g.get_repo("owner/repo_name")을 통해 특정 리포지토리 객체를 직접 얻을 수 있습니다.6PaginatedList: User 또는 Organization 객체에서 get_repos() 메서드를 호출하면, 여러 Repository 객체를 포함하는 PaginatedList를 반환합니다.7하위 리소스 객체: 대부분의 구체적인 작업(예: 이슈, 풀 리퀘스트, 파일, 브랜치 관리)은 Repository 객체의 메서드를 통해 이루어집니다. Repository 객체는 Issue, PullRequest, Branch, ContentFile 등 다양한 하위 리소스에 접근하는 관문 역할을 합니다.10이러한 구조에서 가장 중요한 점은 Repository 객체의 중심성입니다. 파일 생성(repo.create_file()) 11, 이슈 조회(repo.get_issues()) 12, 브랜치 목록 확인(repo.get_branches()) 13 등 대부분의 핵심 기능은 Repository 객체의 메서드로 제공됩니다. 따라서 많은 도구들의 핵심 로직은 다음과 같은 2단계 프로세스를 따릅니다.인증을 통해 Github 인스턴스를 얻습니다.작업 대상이 되는 Repository 객체를 획득합니다.해당 Repository 객체에서 필요한 메서드를 호출합니다.이 보고서에서는 각 도구의 설명을 이 일관된 패턴에 기반하여 구성할 것입니다. 이는 개발자의 학습 곡선을 완만하게 하고, 요청된 22개 도구를 넘어 새로운 기능을 구축할 때 일관된 정신 모델을 제공할 것입니다.섹션 2: 리포지토리 및 브랜치 관리 도구이 섹션은 모든 GitHub 연동 에이전트의 기본이 되는 상위 수준의 리포지토리 및 소스 제어 작업에 중점을 둡니다.2.1. 도구 1: 리포지토리 목록 조회기능: 인증된 사용자, 특정 사용자 또는 특정 조직에 속한 리포지토리 목록을 조회합니다.구현: PyGithub은 /user/repos, /users/{user}/repos, /orgs/{org}/repos와 같은 서로 다른 API 엔드포인트를 추상화하여 일관된 메서드를 제공합니다.14 이 도구 함수는 선택적 매개변수인 username과 org_name의 존재 여부에 따라 적절한 PyGithub 메서드를 호출하도록 설계됩니다.인증된 사용자: g.get_user().get_repos() 4특정 사용자: g.get_user("username").get_repos() 9조직: g.get_organization("org_name").get_repos() 7Pythonfrom typing import List, Dict, Any, Optional

def list_repositories(
    token: str,
    username: Optional[str] = None,
    org_name: Optional[str] = None
) -> List]:
    """
    사용자 또는 조직의 리포지토리 목록을 조회합니다.

    :param token: GitHub 인증 토큰.
    :param username: 리포지토리를 조회할 특정 사용자의 이름. None이면 인증된 사용자를 대상으로 합니다.
    :param org_name: 리포지토리를 조회할 조직의 이름. username과 org_name 중 하나만 지정해야 합니다.
    :return: 리포지토리 정보 딕셔너리의 리스트.
    :raises ValueError: username과 org_name이 동시에 제공된 경우.
    """
    if username and org_name:
        raise ValueError("username과 org_name은 동시에 지정할 수 없습니다.")

    g = get_github_instance(token)
    repo_list =

    try:
        if org_name:
            target = g.get_organization(org_name)
            repos = target.get_repos()
        elif username:
            target = g.get_user(username)
            repos = target.get_repos()
        else:
            target = g.get_user()
            repos = target.get_repos()

        for repo in repos:
            repo_list.append({
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description,
                "private": repo.private,
                "fork": repo.fork,
                "html_url": repo.html_url,
                "stargazers_count": repo.stargazers_count,
                "language": repo.language,
            })
    except GithubException as e:
        raise RuntimeError(f"리포지토리 조회 중 오류 발생: {e.data.get('message', e.status)}")
    
    return repo_list
이 함수는 React 프론트엔드에서 사용하기 용이하도록 name, full_name, description, private, fork, html_url 등 핵심 정보를 포함하는 딕셔너리 리스트를 반환합니다.2.2. 도구 2: 리포지토리 검색기능: GitHub 전체에서 특정 쿼리와 일치하는 리포지토리를 검색합니다. 단순 목록 조회보다 강력하며, GitHub의 고급 검색 구문을 활용할 수 있습니다.구현: g.search_repositories(query, sort, order) 메서드를 사용합니다.6query 매개변수는 검색의 핵심이며, 다양한 한정자(qualifier)를 조합하여 정교한 검색 조건을 만들 수 있습니다.15에서는 키워드를 프로그래밍 방식으로 조합하여 query = '+'.join(keywords) + '+in:readme+in:description'과 같은 검색어를 만드는 방법을 보여줍니다. 6에서는 언어(language:python)나 이슈 레이블(good-first-issues:>3)을 기준으로 검색하는 예시를 제공합니다.Pythondef search_repositories_by_query(
    token: str,
    query: str,
    sort: str = "best-match",
    order: str = "desc"
) -> Dict[str, Any]:
    """
    제공된 쿼리를 사용하여 GitHub 리포지토리를 검색합니다.

    :param token: GitHub 인증 토큰.
    :param query: GitHub 검색 쿼리 문자열 (예: 'language:python stars:>1000').
    :param sort: 정렬 기준 ('stars', 'forks', 'help-wanted-issues', 'updated').
    :param order: 정렬 순서 ('asc', 'desc').
    :return: 검색 결과 총 개수와 리포지토리 정보 리스트를 포함하는 딕셔너리.
    """
    g = get_github_instance(token)
    repo_list =

    try:
        result = g.search_repositories(query=query, sort=sort, order=order)
        
        for repo in result:
            repo_list.append({
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description,
                "html_url": repo.html_url,
                "stargazers_count": repo.stargazers_count,
                "language": repo.language,
            })
            
        return {"total_count": result.totalCount, "items": repo_list}

    except GithubException as e:
        raise RuntimeError(f"리포지토리 검색 중 오류 발생: {e.data.get('message', e.status)}")
search_repositories 메서드는 GitHub 검색 API의 강력한 기능을 활용하는 관문입니다. 그러나 검색 쿼리 구문 자체가 하나의 미니 언어와 같아서, 정확하게 구성하지 않으면 오류의 원인이 될 수 있습니다. 30에서는 한 사용자가 한정자를 딕셔너리(qualifiers={...})로 전달하려다 혼란을 겪는 사례가 나타납니다. 올바른 방법은 쿼리 문자열에 직접 포함시키거나, g.search_users('Keyword', location='San Francisco')와 같이 함수에 키워드 인자로 전달하는 것입니다. 따라서 개발자는 in:, language:, stars:, forks:와 같은 일반적인 한정자를 조합하는 방법을 숙지해야 합니다.2.3. 도구 3: 리포지토리 브랜치 목록 조회기능: 주어진 리포지토리의 모든 브랜치를 조회합니다.구현: repo.get_branches() 메서드를 사용합니다.13 이 메서드는 Branch 객체들을 담고 있는 PaginatedList를 반환합니다. 도구 함수는 이 리스트를 순회하며 각 브랜치의 이름, 최신 커밋 SHA 등의 정보를 추출합니다.Pythondef list_branches(token: str, repo_full_name: str) -> List]:
    """
    특정 리포지토리의 모든 브랜치 목록을 조회합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :return: 브랜치 정보 딕셔너리의 리스트.
    """
    g = get_github_instance(token)
    branch_list =

    try:
        repo = g.get_repo(repo_full_name)
        branches = repo.get_branches()
        for branch in branches:
            branch_list.append({
                "name": branch.name,
                "commit_sha": branch.commit.sha,
            })
    except UnknownObjectException:
        raise ValueError(f"리포지토리 '{repo_full_name}'을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"브랜치 목록 조회 중 오류 발생: {e.data.get('message', e.status)}")
        
    return branch_list
2.4. 도구 4: 브랜치 상세 정보 조회기능: 특정 단일 브랜치의 상세 정보를 조회합니다.구현: repo.get_branch(branch="branch_name") 메서드를 사용합니다.13get_branches()로 반환된 Branch 객체는 "완전히 채워지지 않은(not fully populated)" 상태라는 점을 인지하는 것이 매우 중요합니다.13 이는 API와 라이브러리의 성능 최적화 때문입니다. 목록 보기에서는 최소한의 데이터를 제공하고, 상세 보기에서 모든 정보를 제공하는 방식입니다. 보호 상태(protection status)와 같은 상세 정보에 접근하려면, get_branch()를 사용하여 특정 브랜치를 다시 조회해야 합니다. 이 사실을 모르는 개발자는 목록에서 가져온 브랜치 객체의 속성에 접근하려다 AttributeError 예외를 마주하게 될 것입니다.따라서 '브랜치 목록 조회'와 '브랜치 상세 정보 조회' 도구는 명확히 구분되어야 합니다. 포괄적인 정보, 특히 보호 규칙(branch.protected, branch.get_required_status_checks())이 필요할 때는 항상 get_branch()를 사용해야 합니다.13Pythonfrom github import UnknownObjectException

def get_branch_details(token: str, repo_full_name: str, branch_name: str) -> Dict[str, Any]:
    """
    특정 브랜치의 상세 정보를 조회합니다. (보호 규칙 포함)

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param branch_name: 조회할 브랜치의 이름.
    :return: 브랜치 상세 정보를 담은 딕셔너리.
    """
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        branch = repo.get_branch(branch=branch_name)
        
        protection = branch.get_protection() if branch.protected else None
        protection_details = {}
        if protection:
            reviews = protection.required_pull_request_reviews
            protection_details = {
                "enforce_admins": protection.enforce_admins,
                "required_pull_request_reviews": {
                    "dismiss_stale_reviews": reviews.dismiss_stale_reviews,
                    "require_code_owner_reviews": reviews.require_code_owner_reviews,
                    "required_approving_review_count": reviews.required_approving_review_count,
                } if reviews else None,
            }

        return {
            "name": branch.name,
            "commit_sha": branch.commit.sha,
            "protected": branch.protected,
            "protection_details": protection_details,
        }
    except UnknownObjectException:
        raise ValueError(f"리포지토리 '{repo_full_name}' 또는 브랜치 '{branch_name}'을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"브랜치 상세 정보 조회 중 오류 발생: {e.data.get('message', e.status)}")
2.5. 도구 5: 브랜치 삭제기능: 리포지토리에서 브랜치를 삭제합니다.구현: 직관적인 repo.delete_branch()나 branch.delete() 메서드는 존재하지 않습니다. 브랜치를 삭제하는 올바른 절차는 기저에 있는 Git 참조(ref)를 직접 조작하는 것입니다. 이는 GitHub 이슈18를 통해 발견된 중요한 사실입니다.Git에서 브랜치는 특정 커밋을 가리키는 포인터(참조)에 불과합니다. 브랜치를 삭제한다는 것은 이 참조를 삭제하는 것을 의미합니다. PyGithub 라이브러리는 이러한 Git 수준의 현실을 그대로 반영합니다. 따라서 브랜치를 삭제하려면 repo.get_git_ref(f"heads/{branch_name}")를 호출하여 해당 브랜치의 참조를 얻은 다음, 반환된 객체에서 .delete() 메서드를 호출해야 합니다.18이는 개발자가 쉽게 빠질 수 있는 함정입니다. 이 보고서는 18에서 제공된 해결 방법을 이 도구의 표준 구현으로 제시하며, Git의 내부 모델과 연결하여 왜 이런 방식으로 작동하는지 설명합니다. 이는 혼란의 소지를 귀중한 학습 기회로 전환시킵니다.Pythondef delete_branch(token: str, repo_full_name: str, branch_name: str) -> bool:
    """
    리포지토리에서 특정 브랜치를 삭제합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param branch_name: 삭제할 브랜치의 이름.
    :return: 삭제 성공 시 True.
    """
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        ref = repo.get_git_ref(f"heads/{branch_name}")
        ref.delete()
        return True
    except UnknownObjectException:
        # 브랜치가 존재하지 않는 경우, 이미 삭제된 것으로 간주하고 성공으로 처리할 수 있음
        # 또는 오류를 발생시켜 호출자에게 알릴 수 있음. 여기서는 오류를 발생시킴.
        raise ValueError(f"브랜치 '{branch_name}'을(를) 찾을 수 없어 삭제할 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"브랜치 삭제 중 오류 발생: {e.data.get('message', e.status)}")
섹션 3: 콘텐츠 및 코드 상호작용 도구이 섹션은 리포지토리 내 파일에 대한 기본적인 CRUD(생성, 읽기, 수정, 삭제) 작업을 다룹니다.3.1. 도구 6: 파일 내용 읽기기능: 리포지토리 내 특정 경로에 있는 파일의 내용을 가져옵니다.구현: repo.get_contents("path/to/file.txt")를 호출한 후, 반환된 ContentFile 객체의 decoded_content 속성을 UTF-8로 디코딩합니다.11get_contents는 파일 내용을 Base64로 인코딩하여 반환하기 때문에 디코딩 과정이 필수적입니다. 또한, 이 메서드는 경로가 디렉터리를 가리킬 경우 해당 디렉터리의 내용 목록을 반환하는 데도 사용될 수 있습니다.11Pythondef read_file(token: str, repo_full_name: str, file_path: str, branch: Optional[str] = None) -> str:
    """
    리포지토리 내 파일의 내용을 읽어옵니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param file_path: 읽을 파일의 경로.
    :param branch: 파일을 읽을 브랜치 이름. 지정하지 않으면 기본 브랜치 사용.
    :return: 파일의 내용 (문자열).
    """
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        # ref 매개변수를 사용하여 특정 브랜치 지정
        contents = repo.get_contents(file_path, ref=branch) if branch else repo.get_contents(file_path)
        
        if isinstance(contents, list):
            raise ValueError(f"경로 '{file_path}'는 디렉터리입니다. 파일 경로를 지정해야 합니다.")
            
        return contents.decoded_content.decode("utf-8")
    except UnknownObjectException:
        raise ValueError(f"파일 '{file_path}'을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"파일 읽기 중 오류 발생: {e.data.get('message', e.status)}")
3.2. 도구 7: 새 파일 생성기능: 리포지토리에 지정된 내용으로 새 파일을 생성합니다.구현: repo.create_file(path, commit_message, content, branch) 메서드를 사용합니다.11 이 메서드는 직관적이며, branch 매개변수는 파일이 올바른 위치에 생성되도록 보장하는 데 매우 중요합니다. 커밋 메시지 또한 필수 매개변수로, 좋은 Git 습관을 강제합니다.Pythondef create_file(
    token: str,
    repo_full_name: str,
    file_path: str,
    commit_message: str,
    content: str,
    branch: str
) -> Dict[str, str]:
    """
    리포지토리의 특정 브랜치에 새 파일을 생성합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param file_path: 생성할 파일의 경로.
    :param commit_message: 커밋 메시지.
    :param content: 파일에 쓸 내용.
    :param branch: 파일을 생성할 브랜치 이름.
    :return: 생성된 파일 정보와 커밋 정보를 담은 딕셔너리.
    """
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        result = repo.create_file(file_path, commit_message, content, branch=branch)
        return {
            "content_path": result['content'].path,
            "commit_sha": result['commit'].sha,
            "commit_url": result['commit'].html_url
        }
    except GithubException as e:
        # 422 Unprocessable Entity는 파일이 이미 존재할 때 발생할 수 있음
        if e.status == 422:
            raise ValueError(f"파일 '{file_path}'이(가) 브랜치 '{branch}'에 이미 존재합니다.")
        raise RuntimeError(f"파일 생성 중 오류 발생: {e.data.get('message', e.status)}")
3.3. 도구 8: 기존 파일 수정기능: 기존 파일의 내용을 수정합니다.구현: repo.update_file(path, commit_message, new_content, file_sha, branch) 메서드를 사용합니다.11파일을 수정할 때, 수정 대상 파일의 SHA 해시값이 필수적으로 요구된다는 점이 매우 중요합니다. 이는 GitHub API의 콘텐츠 수정에 대한 근본적인 설계 원칙입니다. 20와 11 모두 sha가 필수 매개변수임을 보여줍니다. 그 이유는 동시성 제어(concurrency control) 메커니즘 때문입니다. 이 메커니즘은 사용자가 파일을 마지막으로 읽은 이후 다른 누군가에 의해 파일이 변경되지 않았음을 보장합니다. 만약 제공한 SHA가 서버에 있는 파일의 현재 SHA와 일치하지 않으면, 요청은 실패하게 됩니다.따라서 이 도구의 구현은 반드시 2단계 프로세스를 거쳐야 합니다. 첫째, get_contents()를 호출하여 파일과 현재 SHA를 검색하고, 둘째, 그 SHA를 사용하여 update_file()을 호출합니다. 이 워크플로우와 그 배경에 있는 이유를 명확히 이해하면, 개발자는 예기치 않은 업데이트 실패를 방지할 수 있습니다.Pythondef update_file(
    token: str,
    repo_full_name: str,
    file_path: str,
    commit_message: str,
    new_content: str,
    branch: str
) -> Dict[str, str]:
    """
    리포지토리의 특정 브랜치에 있는 기존 파일을 수정합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param file_path: 수정할 파일의 경로.
    :param commit_message: 커밋 메시지.
    :param new_content: 파일의 새 내용.
    :param branch: 파일을 수정할 브랜치 이름.
    :return: 수정된 파일 정보와 커밋 정보를 담은 딕셔너리.
    """
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        # 1. 파일의 현재 SHA를 얻기 위해 먼저 파일을 가져옴
        contents = repo.get_contents(file_path, ref=branch)
        
        # 2. 얻은 SHA를 사용하여 파일 업데이트
        result = repo.update_file(
            contents.path,
            commit_message,
            new_content,
            contents.sha,
            branch=branch
        )
        return {
            "content_path": result['content'].path,
            "commit_sha": result['commit'].sha,
            "commit_url": result['commit'].html_url
        }
    except UnknownObjectException:
        raise ValueError(f"파일 '{file_path}'을(를) 브랜치 '{branch}'에서 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"파일 수정 중 오류 발생: {e.data.get('message', e.status)}")
3.4. 도구 9: 파일 삭제기능: 리포지토리에서 파일을 삭제합니다.구현: repo.delete_file(path, commit_message, file_sha, branch) 메서드를 사용합니다.11 파일 수정과 마찬가지로, 파일 삭제 역시 동시성 제어를 위해 파일의 SHA가 필요합니다. 워크플로우는 동일합니다. 먼저 콘텐츠 객체를 가져와 SHA를 확인한 다음, 삭제 메서드를 호출합니다.Pythondef delete_file(
    token: str,
    repo_full_name: str,
    file_path: str,
    commit_message: str,
    branch: str
) -> Dict[str, str]:
    """
    리포지토리의 특정 브랜치에서 파일을 삭제합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param file_path: 삭제할 파일의 경로.
    :param commit_message: 커밋 메시지.
    :param branch: 파일을 삭제할 브랜치 이름.
    :return: 삭제 커밋 정보를 담은 딕셔너리.
    """
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        # 1. 삭제할 파일의 SHA를 얻기 위해 먼저 파일을 가져옴
        contents = repo.get_contents(file_path, ref=branch)
        
        # 2. 얻은 SHA를 사용하여 파일 삭제
        result = repo.delete_file(
            contents.path,
            commit_message,
            contents.sha,
            branch=branch
        )
        return {
            "commit_sha": result['commit'].sha,
            "commit_url": result['commit'].html_url
        }
    except UnknownObjectException:
        raise ValueError(f"파일 '{file_path}'을(를) 브랜치 '{branch}'에서 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"파일 삭제 중 오류 발생: {e.data.get('message', e.status)}")
3.5. 도구 10: 코드 검색기능: GitHub 전체 또는 특정 리포지토리 내에서 코드를 검색합니다.구현: g.search_code(query) 메서드를 사용합니다.15query 매개변수는 다른 검색 엔드포인트와 동일한 강력한 구문을 따릅니다. repo:owner/repo와 같은 한정자를 사용하여 검색 범위를 좁힐 수 있습니다. 15의 함수는 검색 수행 전 API 속도 제한(g.get_rate_limit())을 확인하는 모범 사례를 보여줍니다. 이는 리소스 소모가 많은 작업을 수행할 때 에이전트의 안정성을 높이는 좋은 모델입니다.Pythondef search_code_in_repo(
    token: str,
    query: str,
    repo_full_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    GitHub에서 코드를 검색합니다. 특정 리포지토리로 범위를 제한할 수 있습니다.

    :param token: GitHub 인증 토큰.
    :param query: 검색할 코드 또는 텍스트.
    :param repo_full_name: 검색 범위를 제한할 'owner/repo' 형식의 리포지토리.
    :return: 검색 결과 총 개수와 파일 정보 리스트를 포함하는 딕셔너리.
    """
    g = get_github_instance(token)
    
    full_query = query
    if repo_full_name:
        full_query += f" repo:{repo_full_name}"
        
    file_list =
    try:
        result = g.search_code(query=full_query)
        
        for content_file in result:
            file_list.append({
                "path": content_file.path,
                "name": content_file.name,
                "html_url": content_file.html_url,
                "repository": content_file.repository.full_name,
            })
            
        return {"total_count": result.totalCount, "items": file_list}
    except GithubException as e:
        raise RuntimeError(f"코드 검색 중 오류 발생: {e.data.get('message', e.status)}")
섹션 4: 이슈 관리 워크플로우 도구이 섹션은 프로젝트 관리 및 버그 추적의 핵심 기능인 GitHub 이슈의 전체 생명주기를 관리하는 데 전념합니다.4.1. 도구 11: 리포지토리 이슈 목록 조회기능: 상태(open, closed, all), 레이블, 담당자 등 필터를 사용하여 이슈 목록을 조회합니다.구현: repo.get_issues(state='open', labels=[...]) 메서드를 사용합니다.11 이 메서드는 Issue 객체의 PaginatedList를 반환하므로, 도구는 페이지네이션을 올바르게 처리해야 합니다 (섹션 6.2 참조). 함수는 일반적인 필터 매개변수를 에이전트에 노출시켜야 합니다.Pythondef list_issues(
    token: str,
    repo_full_name: str,
    state: str = "open",
    assignee: Optional[str] = None,
    labels: Optional[List[str]] = None
) -> List]:
    """
    리포지토리의 이슈 목록을 필터링하여 조회합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param state: 이슈 상태 ('open', 'closed', 'all').
    :param assignee: 담당자 로그인 이름.
    :param labels: 필터링할 레이블 이름 리스트.
    :return: 이슈 정보 딕셔너리의 리스트.
    """
    g = get_github_instance(token)
    issue_list =
    try:
        repo = g.get_repo(repo_full_name)
        issues = repo.get_issues(state=state, assignee=assignee, labels=labels or)
        
        for issue in issues:
            issue_list.append({
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "user": issue.user.login,
                "assignees": [a.login for a in issue.assignees],
                "labels": [l.name for l in issue.labels],
                "html_url": issue.html_url,
                "created_at": issue.created_at.isoformat(),
            })
        return issue_list
    except UnknownObjectException:
        raise ValueError(f"리포지토리 '{repo_full_name}'을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"이슈 목록 조회 중 오류 발생: {e.data.get('message', e.status)}")
4.2. 도구 12: 단일 이슈 조회기능: 번호를 사용하여 특정 이슈를 조회합니다.구현: repo.get_issue(number=issue_number) 메서드를 사용합니다.12 이는 본문, 댓글, 레이블을 포함한 단일 이슈의 전체 세부 정보를 가져오는 데 필수적인 직접적인 함수입니다.Pythondef get_issue_details(token: str, repo_full_name: str, issue_number: int) -> Dict[str, Any]:
    """
    리포지토리에서 특정 번호의 이슈 상세 정보를 조회합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param issue_number: 조회할 이슈의 번호.
    :return: 이슈 상세 정보를 담은 딕셔너리.
    """
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        issue = repo.get_issue(number=issue_number)
        
        return {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "state": issue.state,
            "user": issue.user.login,
            "assignees": [a.login for a in issue.assignees],
            "labels": [l.name for l in issue.labels],
            "comments_count": issue.comments,
            "html_url": issue.html_url,
            "created_at": issue.created_at.isoformat(),
            "updated_at": issue.updated_at.isoformat(),
            "closed_at": issue.closed_at.isoformat() if issue.closed_at else None,
        }
    except UnknownObjectException:
        raise ValueError(f"이슈 #{issue_number}을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"이슈 상세 정보 조회 중 오류 발생: {e.data.get('message', e.status)}")
4.3. 도구 13: 새 이슈 생성기능: 제목, 본문, 그리고 선택적으로 담당자와 레이블을 포함하는 새 이슈를 생성합니다.구현: repo.create_issue(title, body, assignee, labels, milestone) 메서드를 사용합니다.10 이 메서드는 10과 22의 예제에서 잘 문서화되어 있습니다. 도구 함수는 이를 래핑하여 에이전트가 프로그래밍 방식으로 이슈를 생성할 수 있는 깔끔한 인터페이스를 제공합니다.Pythondef create_issue(
    token: str,
    repo_full_name: str,
    title: str,
    body: Optional[str] = None,
    assignees: Optional[List[str]] = None,
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    리포지토리에 새 이슈를 생성합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param title: 이슈 제목.
    :param body: 이슈 본문.
    :param assignees: 담당자로 지정할 사용자 로그인 이름 리스트.
    :param labels: 적용할 레이블 이름 리스트.
    :return: 생성된 이슈의 정보를 담은 딕셔너리.
    """
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        new_issue = repo.create_issue(
            title=title,
            body=body or "",
            assignees=assignees or,
            labels=labels or
        )
        return get_issue_details(token, repo_full_name, new_issue.number)
    except UnknownObjectException:
        raise ValueError(f"리포지토리 '{repo_full_name}'을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"이슈 생성 중 오류 발생: {e.data.get('message', e.status)}")
4.4. 도구 14: 이슈에 댓글 추가기능: 기존 이슈에 새 댓글을 게시합니다.구현: 먼저 Issue 객체를 가져온 다음, issue.create_comment(body)를 호출합니다.23 이 프로세스는 전형적인 "가져온 후 행동(get-then-act)" 패턴입니다. 댓글을 추가하려는 Issue 객체를 먼저 가지고 있어야 하며, 이 객체 자체에 create_comment 메서드가 있습니다.Pythondef add_comment_to_issue(
    token: str,
    repo_full_name: str,
    issue_number: int,
    body: str
) -> Dict[str, Any]:
    """
    특정 이슈에 댓글을 추가합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param issue_number: 댓글을 추가할 이슈의 번호.
    :param body: 댓글 내용.
    :return: 생성된 댓글의 정보를 담은 딕셔너리.
    """
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        issue = repo.get_issue(number=issue_number)
        comment = issue.create_comment(body)
        
        return {
            "id": comment.id,
            "user": comment.user.login,
            "body": comment.body,
            "created_at": comment.created_at.isoformat(),
            "html_url": comment.html_url,
        }
    except UnknownObjectException:
        raise ValueError(f"이슈 #{issue_number}을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"이슈 댓글 추가 중 오류 발생: {e.data.get('message', e.status)}")
4.5. 도구 15: 이슈 검색기능: 검색 API를 사용하여 여러 리포지토리에 걸쳐 이슈를 찾습니다.구현: g.search_issues(query) 메서드를 사용합니다.25 이는 search_repositories 및 search_code와 유사하며, 쿼리 구문이 핵심입니다. 25에서는 PyGithub이 URL 인코딩을 처리하는 방식 때문에 repo:myOrg/myPrivateRepo+is:closed와 같은 쿼리에 어려움을 겪는 사용자를 보여줍니다. 이는 쿼리 문자열을 신중하게 구성해야 함을 시사합니다. 이 도구는 repo:PyGithub/PyGithub is:open label:bug와 같은 복잡한 쿼리를 허용해야 합니다.Pythondef search_issues_by_query(
    token: str,
    query: str,
    sort: str = "best-match",
    order: str = "desc"
) -> Dict[str, Any]:
    """
    제공된 쿼리를 사용하여 GitHub 이슈를 검색합니다.

    :param token: GitHub 인증 토큰.
    :param query: GitHub 검색 쿼리 문자열 (예: 'repo:owner/repo is:open label:bug').
    :param sort: 정렬 기준 ('comments', 'created', 'updated').
    :param order: 정렬 순서 ('asc', 'desc').
    :return: 검색 결과 총 개수와 이슈 정보 리스트를 포함하는 딕셔너리.
    """
    g = get_github_instance(token)
    issue_list =
    try:
        result = g.search_issues(query=query, sort=sort, order=order)
        
        for issue in result:
            issue_list.append({
                "number": issue.number,
                "title": issue.title,
                "repository": issue.repository.full_name,
                "state": issue.state,
                "html_url": issue.html_url,
            })
            
        return {"total_count": result.totalCount, "items": issue_list}
    except GithubException as e:
        raise RuntimeError(f"이슈 검색 중 오류 발생: {e.data.get('message', e.status)}")
섹션 5: 풀 리퀘스트 자동화 도구이 섹션은 코드 리뷰 및 병합 프로세스를 관리하는 도구를 제공하며, 이슈와 풀 리퀘스트 간의 깊은 연관성을 활용합니다.GitHub API와 이를 기반으로 하는 PyGithub는 풀 리퀘스트(PR)를 이슈의 특수한 형태로 취급합니다. 이는 개발자에게 패러다임 전환적인 깨달음을 줍니다. 27는 이 점을 명시적으로 언급하며, 29에서는 repo.issue(pr.number)를 통해 PR에 해당하는 이슈를 얻고 issue.create_comment()를 호출하여 PR에 댓글을 다는 방법을 보여줍니다. PullRequest 객체 자체에 편의를 위해 create_issue_comment()라는 메서드가 있지만, get_issue_comments()도 가지고 있다는 사실은 이 둘의 근본적인 연결을 증명합니다.26 PR과 이슈는 리포지토리 내에서 번호 체계, 레이블, 담당자, 댓글을 공유합니다.이 개념을 이해하면, 이슈 레이블, 담당자, 댓글을 관리하는 도구들을 거의 그대로 풀 리퀘스트에 적용할 수 있어 에이전트 로직의 복잡성을 크게 줄일 수 있습니다.5.1. 도구 16: 풀 리퀘스트 목록 조회기능: 상태, 베이스 브랜치 등으로 필터링하여 리포지토리의 PR 목록을 조회합니다.구현: repo.get_pulls(state='open', sort='created', base='main') 메서드를 사용합니다.8PullRequest 객체의 PaginatedList를 반환합니다.Pythondef list_pull_requests(
    token: str,
    repo_full_name: str,
    state: str = "open",
    base: Optional[str] = None
) -> List]:
    """
    리포지토리의 풀 리퀘스트 목록을 필터링하여 조회합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param state: PR 상태 ('open', 'closed', 'all').
    :param base: 필터링할 베이스 브랜치 이름.
    :return: PR 정보 딕셔너리의 리스트.
    """
    g = get_github_instance(token)
    pr_list =
    try:
        repo = g.get_repo(repo_full_name)
        pulls = repo.get_pulls(state=state, sort="created", base=base)
        
        for pr in pulls:
            pr_list.append({
                "number": pr.number,
                "title": pr.title,
                "state": pr.state,
                "user": pr.user.login,
                "html_url": pr.html_url,
                "created_at": pr.created_at.isoformat(),
                "head_branch": pr.head.ref,
                "base_branch": pr.base.ref,
            })
        return pr_list
    except UnknownObjectException:
        raise ValueError(f"리포지토리 '{repo_full_name}'을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"풀 리퀘스트 목록 조회 중 오류 발생: {e.data.get('message', e.status)}")
5.2. 도구 17: 단일 풀 리퀘스트 조회기능: 번호를 사용하여 특정 PR을 조회합니다.구현: repo.get_pull(number=pr_number) 메서드를 사용합니다.8 병합 가능 상태 및 head/base 브랜치 정보를 포함한 상세 PR 데이터를 얻는 데 필수적입니다.Pythondef get_pull_request_details(token: str, repo_full_name: str, pr_number: int) -> Dict[str, Any]:
    """
    리포지토리에서 특정 번호의 풀 리퀘스트 상세 정보를 조회합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param pr_number: 조회할 PR의 번호.
    :return: PR 상세 정보를 담은 딕셔너리.
    """
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        pr = repo.get_pull(number=pr_number)
        
        return {
            "number": pr.number,
            "title": pr.title,
            "body": pr.body,
            "state": pr.state,
            "user": pr.user.login,
            "assignees": [a.login for a in pr.assignees],
            "labels": [l.name for l in pr.labels],
            "html_url": pr.html_url,
            "created_at": pr.created_at.isoformat(),
            "head_branch": pr.head.ref,
            "head_repo": pr.head.repo.full_name if pr.head.repo else None,
            "base_branch": pr.base.ref,
            "merged": pr.merged,
            "mergeable": pr.mergeable,
            "comments": pr.comments,
            "commits": pr.commits,
            "additions": pr.additions,
            "deletions": pr.deletions,
            "changed_files": pr.changed_files,
        }
    except UnknownObjectException:
        raise ValueError(f"풀 리퀘스트 #{pr_number}을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"풀 리퀘스트 상세 정보 조회 중 오류 발생: {e.data.get('message', e.status)}")
5.3. 도구 18: 풀 리퀘스트 생성기능: 두 브랜치 간에 새 PR을 생성합니다.구현: repo.create_pull(title, body, head, base) 메서드를 사용합니다.2828는 GitPython과 같은 순수 Git 라이브러리로는 GitHub PR을 생성할 수 없으며, PyGithub과 같은 GitHub API 라이브러리를 반드시 사용해야 한다는 점을 명확히 합니다. create_pull 메서드는 제목, 본문, 변경 사항이 있는 브랜치(head), 그리고 대상 브랜치(base)를 요구합니다.Pythondef create_pull_request(
    token: str,
    repo_full_name: str,
    title: str,
    body: str,
    head_branch: str,
    base_branch: str,
    draft: bool = False
) -> Dict[str, Any]:
    """
    리포지토리에 새 풀 리퀘스트를 생성합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param title: PR 제목.
    :param body: PR 본문.
    :param head_branch: 변경 사항이 있는 브랜치 (예: 'feature-branch').
    :param base_branch: 변경 사항을 병합할 대상 브랜치 (예: 'main').
    :param draft: 초안(draft) PR로 생성할지 여부.
    :return: 생성된 PR의 정보를 담은 딕셔너리.
    """
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        new_pr = repo.create_pull(
            title=title,
            body=body,
            head=head_branch,
            base=base_branch,
            draft=draft
        )
        return get_pull_request_details(token, repo_full_name, new_pr.number)
    except UnknownObjectException:
        raise ValueError(f"리포지토리 '{repo_full_name}'을(를) 찾을 수 없습니다.")
    except GithubException as e:
        # 422 오류는 종종 head/base 브랜치 문제 또는 변경 사항 없음으로 인해 발생
        if e.status == 422:
            message = e.data.get('message', '')
            if "No commits between" in message:
                raise ValueError(f"'{base_branch}'와 '{head_branch}' 사이에 커밋이 없습니다.")
            if "A pull request already exists" in message:
                raise ValueError("동일한 브랜치에 대한 풀 리퀘스트가 이미 존재합니다.")
        raise RuntimeError(f"풀 리퀘스트 생성 중 오류 발생: {e.data.get('message', e.status)}")
5.4. 도구 19: 풀 리퀘스트에 댓글 추가기능: PR에 일반적인 댓글을 게시합니다.구현: PullRequest 객체를 가져온 다음, pr.create_issue_comment(body)를 호출합니다.8 이는 PR-as-Issue 패러다임을 활용하는 대표적인 예입니다. PR 객체에서도 메서드 이름이 create_issue_comment인 것은 이 둘의 연관성을 다시 한번 강조합니다. 이는 특정 코드 라인에 대한 리뷰 댓글(create_review_comment)과는 다른 개념입니다.26Pythondef add_comment_to_pull_request(
    token: str,
    repo_full_name: str,
    pr_number: int,
    body: str
) -> Dict[str, Any]:
    """
    특정 풀 리퀘스트에 일반 댓글을 추가합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param pr_number: 댓글을 추가할 PR의 번호.
    :param body: 댓글 내용.
    :return: 생성된 댓글의 정보를 담은 딕셔너리.
    """
    g = get_github_instance(token)
    try:
        repo = g.get_repo(repo_full_name)
        pr = repo.get_pull(number=pr_number)
        comment = pr.create_issue_comment(body)
        
        return {
            "id": comment.id,
            "user": comment.user.login,
            "body": comment.body,
            "created_at": comment.created_at.isoformat(),
            "html_url": comment.html_url,
        }
    except UnknownObjectException:
        raise ValueError(f"풀 리퀘스트 #{pr_number}을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"풀 리퀘스트 댓글 추가 중 오류 발생: {e.data.get('message', e.status)}")
5.5. 도구 20: 풀 리퀘스트의 파일 목록 조회기능: PR에서 변경된 파일 목록을 가져옵니다.구현: PullRequest 객체를 가져온 다음, pr.get_files()를 호출합니다.26 이는 파일 이름, 상태(추가, 수정, 삭제), 패치 데이터 등의 세부 정보를 포함하는 File 객체의 PaginatedList를 반환합니다.Pythondef list_files_in_pull_request(token: str, repo_full_name: str, pr_number: int) -> List]:
    """
    특정 풀 리퀘스트에서 변경된 파일 목록을 조회합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param pr_number: 조회할 PR의 번호.
    :return: 변경된 파일 정보 딕셔너리의 리스트.
    """
    g = get_github_instance(token)
    file_list =
    try:
        repo = g.get_repo(repo_full_name)
        pr = repo.get_pull(number=pr_number)
        files = pr.get_files()
        
        for file in files:
            file_list.append({
                "filename": file.filename,
                "status": file.status, # 'added', 'removed', 'modified'
                "additions": file.additions,
                "deletions": file.deletions,
                "changes": file.changes,
                "patch": file.patch,
            })
        return file_list
    except UnknownObjectException:
        raise ValueError(f"풀 리퀘스트 #{pr_number}을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"풀 리퀘스트 파일 목록 조회 중 오류 발생: {e.data.get('message', e.status)}")
5.6. 도구 21: 풀 리퀘스트의 커밋 목록 조회기능: PR의 커밋 히스토리를 조회합니다.구현: PullRequest 객체를 가져온 다음, pr.get_commits()를 호출합니다.26 이는 PR 내의 변경 사항을 감사하는 데 유용한 Commit 객체의 PaginatedList를 반환합니다.Pythondef list_commits_in_pull_request(token: str, repo_full_name: str, pr_number: int) -> List]:
    """
    특정 풀 리퀘스트에 포함된 커밋 목록을 조회합니다.

    :param token: GitHub 인증 토큰.
    :param repo_full_name: 'owner/repo' 형식의 리포지토리 전체 이름.
    :param pr_number: 조회할 PR의 번호.
    :return: 커밋 정보 딕셔너리의 리스트.
    """
    g = get_github_instance(token)
    commit_list =
    try:
        repo = g.get_repo(repo_full_name)
        pr = repo.get_pull(number=pr_number)
        commits = pr.get_commits()
        
        for commit in commits:
            commit_list.append({
                "sha": commit.sha,
                "message": commit.commit.message,
                "author_name": commit.commit.author.name,
                "author_email": commit.commit.author.email,
                "date": commit.commit.author.date.isoformat(),
                "html_url": commit.html_url,
            })
        return commit_list
    except UnknownObjectException:
        raise ValueError(f"풀 리퀘스트 #{pr_number}을(를) 찾을 수 없습니다.")
    except GithubException as e:
        raise RuntimeError(f"풀 리퀘스트 커밋 목록 조회 중 오류 발생: {e.data.get('message', e.status)}")
섹션 6: 고급 개념 및 유틸리티 도구이 마지막 섹션은 안정적이고 프로덕션 수준의 에이전트를 구축하기 위한 중요한 맥락과 도구를 제공합니다.6.1. 도구 22: 인증된 사용자 정보 가져오기기능: 토큰을 검증하고 인증된 사용자의 프로필 정보를 검색하는 간단하지만 필수적인 유틸리티입니다.구현: g.get_user()를 사용합니다.4앞서 1.1절에서 논의한 바와 같이, 이 도구는 인증 토큰을 검증하는 주요 메커니즘입니다. 에이전트가 새 토큰을 받으면 가장 먼저 이 함수를 호출해야 합니다. 이 함수는 AuthenticatedUser 객체를 반환하며, 여기서 login, name, email 등의 정보를 추출할 수 있습니다.Pythondef get_authenticated_user_details(token: str) -> Dict[str, Any]:
    """
    제공된 토큰으로 인증된 사용자의 상세 정보를 가져옵니다.
    토큰 유효성 검증에 사용될 수 있습니다.

    :param token: GitHub 인증 토큰.
    :return: 사용자 정보를 담은 딕셔너리.
    """
    g = get_github_instance(token)
    try:
        user = g.get_user()
        #.login과 같은 속성에 접근하여 lazy-loading을 트리거하고 인증을 강제함
        login = user.login 
        
        return {
            "login": login,
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "avatar_url": user.avatar_url,
            "html_url": user.html_url,
            "company": user.company,
            "location": user.location,
        }
    except BadCredentialsException:
        raise ValueError("제공된 GitHub 토큰이 유효하지 않습니다.")
    except GithubException as e:
        raise RuntimeError(f"사용자 정보 조회 중 오류 발생: {e.data.get('message', e.status)}")
6.2. PaginatedList 마스터하기get_repos(), get_issues(), get_pulls()와 같은 메서드는 단순한 파이썬 리스트가 아닌 PaginatedList 객체를 반환합니다. 이 객체는 API에서 다음 페이지의 결과를 자동으로 가져오는 이터레이터(iterator)입니다. 초보 개발자는 이를 일반 리스트처럼 취급하려다(예: len(paginated_list)) 오해의 소지가 있거나 비효율적인 코드를 작성할 수 있습니다.31에서는 한 사용자가 PaginatedList 객체 자체에서 get_comments 메서드를 호출하는 실수를 범하는 사례를 보여줍니다. 올바른 패턴은 중첩 루프를 사용하는 것입니다: for issue in r.get_issues(): for comment in issue.get_comments():....안정적인 에이전트를 구축하려면 이 개념을 반드시 이해해야 합니다. PaginatedList를 소비하고 표준 파이썬 리스트로 변환하는 정석적인 코드 패턴은 다음과 같습니다.Python# PaginatedList를 일반 리스트로 변환하는 예시
paginated_list = repo.get_issues()
all_issues = list(paginated_list) # 간단히 list()로 변환 가능

# 또는, UI에 총 개수를 표시해야 할 경우
total_count = paginated_list.totalCount
print(f"총 {total_count}개의 이슈가 있습니다.")

# 첫 페이지만 가져오고 싶을 경우
first_page_issues = paginated_list.get_page(0)
totalCount 속성을 사용하면 모든 항목을 가져오지 않고도 총 개수를 알 수 있어 "257개 중 10개 표시"와 같은 UI 요소를 구현하는 데 매우 유용합니다.6.3. 안정적인 오류 및 예외 처리PyGithub은 API 오류를 파이썬 예외로 변환하여 개발자가 오류를 체계적으로 처리할 수 있도록 합니다. 안정적인 에이전트는 이러한 예외를 포착하고 처리하여 사용자에게 의미 있는 피드백을 제공해야 합니다.주요 예외는 다음과 같습니다:github.BadCredentialsException (401 Unauthorized): 유효하지 않은 토큰이 제공되었을 때 발생합니다.3github.UnknownObjectException (404 Not Found): 리포지토리, 이슈, 파일 등 요청한 리소스가 존재하지 않을 때 발생합니다.18github.GithubException: 권한 부족(403 Forbidden), 유효성 검사 실패(422 Unprocessable Entity) 등 기타 API 오류를 포괄하는 일반적인 예외입니다.25에이전트는 이러한 기술적인 오류를 React UI에서 사용자 친화적인 메시지로 변환해야 합니다. 예를 들어, 이슈 조회 도구가 UnknownObjectException을 받으면 "이 리포지토리에서 #{number} 이슈를 찾을 수 없습니다."라고 보고해야 합니다. BadCredentialsException이 발생하면 사용자에게 토큰을 확인하도록 안내해야 합니다. 다음 표는 이러한 변환을 위한 가이드라인을 제공합니다.예외 이름HTTP 상태 코드일반적인 원인권장 에이전트 조치BadCredentialsException401잘못되었거나 만료된 토큰"GitHub 토큰이 유효하지 않습니다. 토큰을 확인하고 다시 시도하세요."UnknownObjectException404존재하지 않는 리소스 요청 (예: repo, issue, user)"요청한 리소스를 찾을 수 없습니다. 이름이나 번호가 올바른지 확인하세요."GithubException (Forbidden)403리소스에 접근할 권한 부족"이 작업을 수행할 권한이 없습니다. 토큰의 권한(scope)을 확인하세요."GithubException (Unprocessable)422유효하지 않은 요청 데이터 (예: 이미 존재하는 파일 생성)"요청을 처리할 수 없습니다. 입력값이 올바른지 확인하세요 (예: 파일이 이미 존재할 수 있습니다)."RateLimitExceededException403API 요청 한도 초과"API 요청 한도를 초과했습니다. 잠시 후 다시 시도하세요."결론 및 권장 사항이 보고서는 PyGithub 라이브러리를 사용하여 GitHub 관리 작업을 자동화하는 22개의 강력한 도구를 구현하는 포괄적인 가이드를 제시했습니다. 각 도구는 React 에이전트와 같은 외부 시스템에 통합될 수 있도록 설계되었으며, 필수적인 토큰 기반 인증 패턴을 따릅니다.분석을 통해 몇 가지 핵심 원칙이 도출되었습니다. 첫째, PyGithub의 지연 인증(lazy authentication) 특성으로 인해, 토큰의 유효성을 즉시 확인하기 위해서는 get_authenticated_user()와 같은 명시적인 API 호출이 필수적입니다. 둘째, 라이브러리의 계층적 객체 모델, 특히 Repository 객체의 중심적인 역할을 이해하는 것이 효율적인 개발의 핵심입니다. 셋째, 파일 수정 시 sha 값 요구, 브랜치 삭제 시 ref 사용, 풀 리퀘스트와 이슈의 개념적 통합과 같은 PyGithub의 특정 설계 패턴을 숙지하면 일반적인 함정을 피할 수 있습니다.권장 사항:일관된 인증 및 오류 처리: 모든 도구 구현 시 이 보고서에서 제시한 표준 get_github_instance 함수와 예외 처리 패턴을 채택하여 코드의 일관성과 안정성을 확보해야 합니다.페이지네이션 처리: PaginatedList를 다루는 기능(예: 목록 조회)을 구현할 때는 totalCount 속성을 활용하여 UI에 총 개수를 표시하고, 필요한 경우에만 전체 목록을 가져와 성능을 최적화해야 합니다.사용자 피드백 강화: 에이전트는 PyGithub 예외를 포착하여 "권한 부족" 또는 "리소스를 찾을 수 없음"과 같이 사용자에게 명확하고 실행 가능한 피드백으로 변환해야 합니다.문서화 활용: 이 보고서는 22개 도구에 대한 심층적인 가이드를 제공하지만, PyGithub는 지속적으로 업데이트됩니다. 새로운 기능이나 변경 사항에 대해서는 공식 문서를 정기적으로 참조하는 것이 중요합니다.이 가이드에서 제공된 코드와 원칙을 따르면, 개발팀은 GitHub 워크플로우를 자동화하고, 생산성을 높이며, 정교한 React 기반 관리 도구를 성공적으로 구축할 수 있을 것입니다.부록: 도구 기능 요약표도구 #도구 이름설명주요 PyGithub 메서드1리포지토리 목록 조회사용자 또는 조직의 리포지토리 목록을 가져옵니다.user.get_repos(), org.get_repos()2리포지토리 검색쿼리를 사용하여 GitHub 전체에서 리포지토리를 검색합니다.g.search_repositories()3리포지토리 브랜치 목록 조회특정 리포지토리의 모든 브랜치 목록을 가져옵니다.repo.get_branches()4브랜치 상세 정보 조회특정 브랜치의 상세 정보(보호 규칙 포함)를 가져옵니다.repo.get_branch()5브랜치 삭제리포지토리에서 특정 브랜치를 삭제합니다.repo.get_git_ref().delete()6파일 내용 읽기리포지토리 내 특정 파일의 내용을 읽습니다.repo.get_contents()7새 파일 생성리포지토리에 새 파일을 생성합니다.repo.create_file()8기존 파일 수정리포지토리의 기존 파일을 수정합니다.repo.update_file()9파일 삭제리포지토리에서 파일을 삭제합니다.repo.delete_file()10코드 검색쿼리를 사용하여 코드를 검색합니다.g.search_code()11리포지토리 이슈 목록 조회필터를 사용하여 리포지토리의 이슈 목록을 가져옵니다.repo.get_issues()12단일 이슈 조회번호로 특정 이슈의 상세 정보를 가져옵니다.repo.get_issue()13새 이슈 생성리포지토리에 새 이슈를 생성합니다.repo.create_issue()14이슈에 댓글 추가기존 이슈에 댓글을 추가합니다.issue.create_comment()15이슈 검색쿼리를 사용하여 여러 리포지토리에서 이슈를 검색합니다.g.search_issues()16풀 리퀘스트 목록 조회필터를 사용하여 리포지토리의 PR 목록을 가져옵니다.repo.get_pulls()17단일 풀 리퀘스트 조회번호로 특정 PR의 상세 정보를 가져옵니다.repo.get_pull()18풀 리퀘스트 생성두 브랜치 간에 새 PR을 생성합니다.repo.create_pull()19풀 리퀘스트에 댓글 추가기존 PR에 일반 댓글을 추가합니다.pr.create_issue_comment()20풀 리퀘스트의 파일 목록 조회PR에서 변경된 파일 목록을 가져옵니다.pr.get_files()21풀 리퀘스트의 커밋 목록 조회PR에 포함된 커밋 목록을 가져옵니다.pr.get_commits()22인증된 사용자 정보 가져오기인증 토큰을 검증하고 사용자 프로필을 가져옵니다.g.get_user()