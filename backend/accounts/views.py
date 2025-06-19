from django.contrib.auth import authenticate
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenRefreshView
from .serializers import UserSerializer
from django.shortcuts import render, redirect # Added for template rendering and redirect
from django.contrib.auth.decorators import login_required # Added for view protection
from django.contrib.auth import login as auth_login, logout as auth_logout # For session auth
from django.http import JsonResponse # For AJAX responses
from django.views.decorators.http import require_POST # For restricting to POST requests
from django.views.decorators.csrf import csrf_exempt # For exempting CSRF protection
from django.views.decorators.http import require_GET # For restricting to GET requests
import requests
import re # For parsing GitHub URL
from urllib.parse import urlparse # For parsing URLs
import tempfile # For creating temporary files and directories
from django.contrib import messages
import json # For JSON parsing

# Adjust sys.path to include project root for main/flow imports
import sys
import os
# Assuming views.py is in backend/accounts/, project_root is two levels up.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Imports from project root for scanning logic
from main import DEFAULT_INCLUDE_PATTERNS, DEFAULT_EXCLUDE_PATTERNS
from flow import create_tutorial_flow

# API Login view (SimpleJWT)
@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    email = request.data.get('email')
    password = request.data.get('password')
    
    if not email or not password:
        return Response({'detail': 'Email and password are required'}, status=status.HTTP_400_BAD_REQUEST)
    
    # Debugging info - remove in production
    print(f"Login attempt with email: {email}")
    
    # Check if the user exists in the database
    from django.contrib.auth import get_user_model
    User = get_user_model()
    
    try:
        user_exists = User.objects.filter(email=email).exists()
        print(f"User exists in database: {user_exists}")
        
        if not user_exists:
            # Create a test user for debugging if it doesn't exist
            print("Creating test user for debugging...")
            from django.contrib.auth.hashers import make_password
            from .models import Organization
            
            # Get or create default organization
            default_org, _ = Organization.objects.get_or_create(name="Default Organization")
            
            # Create test user
            User.objects.create(
                email=email,
                password=make_password(password),  # Properly hash the password
                name="Test User",
                org=default_org,
                role="admin",
                is_active=True,
                is_staff=True
            )
            print(f"Test user created with email: {email}")
    except Exception as e:
        print(f"Error checking/creating user: {e}")
    
    # Django's authenticate expects the USERNAME_FIELD value in the 'username' parameter
    # Since our User model has USERNAME_FIELD = 'email', we pass email to username parameter
    user = authenticate(username=email, password=password)
    print(f"Authentication result: {'Success' if user else 'Failed'}")
    
    if user:
        refresh = RefreshToken.for_user(user)
        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'user': UserSerializer(user).data
        })
    
    # More detailed error for debugging
    return Response({'detail': 'Invalid credentials. Please check your email and password.'}, 
                    status=status.HTTP_401_UNAUTHORIZED)

@api_view(['POST'])
@permission_classes([AllowAny])
def logout_view(request):
    # JWT doesn't really need server-side logout, but we keep the endpoint for API consistency
    return Response({"detail": "API Logout successful"})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_detail(request):
    serializer = UserSerializer(request.user)
    return Response(serializer.data)

@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def update_profile(request):
    user = request.user
    serializer = UserSerializer(user, data=request.data, partial=True)
    
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@login_required
def profile_view(request):
    """Renders the user profile page."""
    return render(request, 'accounts/profile.html')

@login_required
def list_github_repositories(request):
    user = request.user
    if not hasattr(user, 'github_access_token') or not user.github_access_token:
        return JsonResponse({'error': 'GitHub token not found or not connected.'}, status=400)

    token = user.github_access_token
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
        'X-GitHub-Api-Version': '2022-11-28'  # Recommended by GitHub
    }
    
    repositories = []
    next_page_url = 'https://api.github.com/user/repos' # Initial API URL
    # Parameters for the first request. For subsequent requests, the URL from response.links will be used directly.
    initial_params = {'type': 'all', 'per_page': 100, 'page': 1}
    is_first_page = True

    try:
        while next_page_url:
            if is_first_page:
                response = requests.get(next_page_url, headers=headers, params=initial_params)
                is_first_page = False
            else:
                # For subsequent pages, next_page_url already contains all necessary query parameters
                response = requests.get(next_page_url, headers=headers)
            
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            data = response.json()
            
            for repo in data:
                repositories.append({
                    'id': repo['id'],
                    'name': repo['name'],
                    'full_name': repo['full_name'],
                    'html_url': repo['html_url'],  # This is the general URL, not branch specific
                    'url': repo['url'],  # API URL for this repo, often used as a base for other API calls
                    'private': repo['private'],
                    'owner_login': repo['owner']['login'],
                    'default_branch': repo['default_branch']  # Add default branch information
                })
            
            if 'next' in response.links:
                next_page_url = response.links['next']['url']
            else:
                next_page_url = None # No more pages

    except requests.exceptions.RequestException as e:
        error_message = f'Failed to fetch repositories from GitHub: {str(e)}'
        if e.response is not None:
            error_message += f" - Status: {e.response.status_code} - Body: {e.response.text[:250]}" # Log part of body
        # Consider logging 'error_message' to server logs for more detailed debugging
        print(f"GitHub API Error: {error_message}") # Basic logging to console
        return JsonResponse({'error': 'Failed to fetch repositories from GitHub. Please check token permissions or try again later.', 'details': str(e)}, status=500)
    except Exception as e:
        print(f"Unexpected Error in list_github_repositories: {str(e)}") # Basic logging
        return JsonResponse({'error': f'An unexpected error occurred: {str(e)}'}, status=500)

    return JsonResponse({'repositories': repositories})


@login_required
@require_GET # This view only handles GET requests
def ajax_list_repository_branches(request):
    owner = request.GET.get('owner')
    repo_name = request.GET.get('repo_name')
    user = request.user

    if not owner or not repo_name:
        return JsonResponse({'error': 'Owner and repository name are required.'}, status=400)

    if not hasattr(user, 'github_access_token') or not user.github_access_token:
        return JsonResponse({'error': 'GitHub token not found or not connected.'}, status=400)

    token = user.github_access_token
    api_url = f'https://api.github.com/repos/{owner}/{repo_name}/branches'
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
        'X-GitHub-Api-Version': '2022-11-28'
    }

    branches = []
    try:
        page = 1
        while True:
            response = requests.get(api_url, headers=headers, params={'per_page': 100, 'page': page})
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            data = response.json()
            if not data: # No more data, break the loop
                break
            for branch_info in data:
                branches.append(branch_info['name'])
            
            if len(data) < 100: # Last page if less than per_page items returned
                break
            page += 1

    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text if e.response else 'No response body'
        print(f"GitHub API HTTPError in ajax_list_repository_branches for {owner}/{repo_name}: {e.response.status_code} - {error_detail}")
        if e.response.status_code == 404:
            return JsonResponse({'error': f'Repository {owner}/{repo_name} not found or access denied.'}, status=404)
        return JsonResponse({'error': f'Failed to fetch branches from GitHub: {str(e)}'}, status=e.response.status_code if e.response else 500)
    except requests.exceptions.RequestException as e:
        print(f"GitHub API RequestException in ajax_list_repository_branches for {owner}/{repo_name}: {str(e)}")
        return JsonResponse({'error': f'Network error while fetching branches: {str(e)}'}, status=500)
    except Exception as e:
        print(f"Unexpected error in ajax_list_repository_branches for {owner}/{repo_name}: {str(e)}")
        return JsonResponse({'error': 'An unexpected error occurred while fetching branches.'}, status=500)

    return JsonResponse({'branches': branches})

@login_required
def delete_github_token(request):
    if request.method == 'POST':
        user = request.user
        if hasattr(user, 'github_access_token') and user.github_access_token:
            user.github_access_token = None
            user.save()
            messages.success(request, 'GitHub token has been successfully disconnected.')
        else:
            messages.info(request, 'No GitHub token found to disconnect.')
    else:
        messages.error(request, 'Invalid request method.')
    return redirect('accounts:settings')

@login_required
def settings_view(request):
    """Renders the user settings page and GitHub connection status."""
    github_connected = False
    github_username = None
    if request.user.is_authenticated and hasattr(request.user, 'github_access_token') and request.user.github_access_token:
        github_connected = True
        try:
            headers = {'Authorization': f'token {request.user.github_access_token}'}
            response = requests.get('https://api.github.com/user', headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            github_user_data = response.json()
            github_username = github_user_data.get('login')
        except requests.exceptions.RequestException as e:
            messages.error(request, f"GitHub 토큰으로 사용자 정보를 가져오는 데 실패했습니다. 토큰을 확인하거나 다시 연결해주세요.")
            github_username = None # Ensure username is None if fetch fails
            
    context = {
        'github_connected': github_connected,
        'github_username': github_username,
    }
    return render(request, 'accounts/setting.html', context)


def _parse_github_url(url):
    """Parses a GitHub URL to extract owner and repository name."""
    # This regex handles URLs with or without '.git' and trailing slashes
    match = re.match(r"https?://github\.com/([^/]+)/([^/.]+)", url)
    if match:
        # .git suffix is removed from the repo name if present
        return match.group(1), match.group(2).replace('.git', '')
    return None, None


@login_required
@require_POST
def list_branches_for_url_view(request):
    """
    Accepts a GitHub repository URL via POST and returns a list of its branches.
    Requires the user to be logged in and have a GitHub token.
    """
    try:
        data = json.loads(request.body)
        repo_url = data.get('repository_url')
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON.'}, status=400)

    if not repo_url:
        return JsonResponse({'status': 'error', 'message': 'Repository URL is required.'}, status=400)

    user = request.user
    github_token = getattr(user, 'github_access_token', None)

    if not github_token:
        return JsonResponse({'status': 'error', 'message': 'GitHub token not found. Please connect your account.'}, status=403)

    owner, repo_name = _parse_github_url(repo_url)

    if not owner or not repo_name:
        return JsonResponse({'status': 'error', 'message': 'Invalid GitHub repository URL format.'}, status=400)

    api_url = f'https://api.github.com/repos/{owner}/{repo_name}/branches'
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json',
        'X-GitHub-Api-Version': '2022-11-28'
    }

    branches = []
    try:
        page = 1
        while True:
            response = requests.get(api_url, headers=headers, params={'per_page': 100, 'page': page})
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            for branch_info in data:
                branches.append(branch_info['name'])
            if len(data) < 100:
                break
            page += 1
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response else 500
        if status_code == 404:
             message = 'Repository not found or access denied. Check the URL and your token permissions.'
        else:
             message = f'Failed to fetch branches from GitHub: {str(e)}'
        return JsonResponse({'status': 'error', 'message': message}, status=status_code)

    return JsonResponse({'status': 'success', 'branches': branches})


@login_required
@require_POST
@csrf_exempt # AJAX POST 요청이므로 CSRF 보호 예외 처리 (또는 CSRF 토큰을 올바르게 전송)
def scan_selected_repositories_view(request):
    try:
        data = json.loads(request.body.decode('utf-8')) # 요청 본문을 JSON으로 파싱
        scan_type = data.get('scan_type')
        user = request.user
        github_token = getattr(user, 'github_access_token', None)
        scan_results = []

        if scan_type == 'public_branches':
            public_repo_url_base = data.get('public_repo_url')
            branches_to_scan = data.get('branches', [])

            if not public_repo_url_base or not branches_to_scan:
                return JsonResponse({'error': 'Public repository URL and at least one branch are required for this scan type.'}, status=400)


            parsed_base_url = _parse_github_url(public_repo_url_base)
            if not parsed_base_url:
                 return JsonResponse({'error': 'Invalid base public repository URL format.'}, status=400)
            
            owner, repo_name_only = parsed_base_url

            for branch_name in branches_to_scan:
                repo_scan_url = f"https://github.com/{owner}/{repo_name_only}/tree/{branch_name}"
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        shared_public = {
                            "repo_url": repo_scan_url,
                            "project_name": f"{repo_name_only}-{branch_name}", # README 및 출력물 이름용
                            "github_token": None,  # 공개 리포지토리는 토큰 불필요
                            "output_dir": temp_dir,
                            "include_patterns": DEFAULT_INCLUDE_PATTERNS,
                            "exclude_patterns": DEFAULT_EXCLUDE_PATTERNS,
                            "max_file_size": 100000,
                            "language": "english",
                            "use_cache": True,
                            "max_abstraction_num": 10,
                            "files": [],
                            "abstractions": [],
                            "relationships": {},
                            "chapter_order": [],
                            "chapters": [],
                            "final_output_dir": None
                        }
                        
                        tutorial_flow = create_tutorial_flow()
                        tutorial_flow.run(shared_public) # 실제 스캔 실행
                        
                        final_output_path = shared_public.get("final_output_dir", "Not specified")
                        file_list_string = shared_public.get('abstractions', 'No file list generated.')

                        if isinstance(file_list_string, str) and len(file_list_string) > 500:
                            file_list_message = file_list_string[:500] + "... (list truncated)"
                        elif isinstance(file_list_string, str):
                            file_list_message = file_list_string.replace('\n', '<br>')
                        else:
                            file_list_message = "File list not available or not a string."

                        scan_results.append({
                            'repo_url': public_repo_url_base, # 스캔 요청된 기본 리포 URL
                            'branch_name': branch_name,       # 스캔된 브랜치 이름
                            'status': 'success',
                            'message': f'Scan for branch {branch_name} completed. Files found:<br>{file_list_message}',
                            'output_path': final_output_path
                        })
                    except Exception as e:
                        scan_results.append({
                            'repo_url': public_repo_url_base,
                            'branch_name': branch_name,
                            'status': 'error',
                            'message': f'Failed to scan branch {branch_name}: {str(e)}'
                        })
            
            return JsonResponse({'status': 'completed', 'results': scan_results})

        elif scan_type == 'user_repos':
            repo_urls = data.get('repo_urls', [])
            if not github_token:
                return JsonResponse({'error': 'GitHub token not found. Please connect your GitHub account.'}, status=403)
            if not repo_urls:
                return JsonResponse({'error': 'No repository URLs provided for scanning.'}, status=400)

            for repo_url in repo_urls: # repo_url은 'https://github.com/owner/repo/tree/branch' 형태
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        parsed_scan_url = urlparse(repo_url)
                        path_parts = parsed_scan_url.path.strip('/').split('/')
                        
                        repo_name_for_readme = "scanned-repo" # 기본값
                        if len(path_parts) >= 4 and path_parts[2] == 'tree': # /owner/repo/tree/branch
                            repo_name_for_readme = f"{path_parts[1]}-{path_parts[3]}" # repo-branch
                        elif len(path_parts) >= 2: # /owner/repo (fallback)
                            repo_name_for_readme = f"{path_parts[0]}-{path_parts[1]}" # owner-repo

                        shared = {
                            "repo_url": repo_url,
                            "project_name": repo_name_for_readme,
                            "github_token": github_token,
                            "output_dir": temp_dir,
                            "include_patterns": DEFAULT_INCLUDE_PATTERNS,
                            "exclude_patterns": DEFAULT_EXCLUDE_PATTERNS,
                            "max_file_size": 100000,
                            "language": "english",
                            "use_cache": True,
                            "max_abstraction_num": 10,
                            "files": [],
                            "abstractions": [],
                            "relationships": {},
                            "chapter_order": [],
                            "chapters": [],
                            "final_output_dir": None
                        }
                        
                        tutorial_flow = create_tutorial_flow()
                        tutorial_flow.run(shared) # 실제 스캔 실행
                        
                        final_output_path = shared.get("final_output_dir", "Not specified")
                        file_list_string = shared.get('abstractions', 'No file list generated.')

                        if isinstance(file_list_string, str) and len(file_list_string) > 500:
                            file_list_message = file_list_string[:500] + "... (list truncated)"
                        elif isinstance(file_list_string, str):
                            file_list_message = file_list_string.replace('\n', '<br>')
                        else:
                            file_list_message = "File list not available or not a string."

                        scan_results.append({
                            'repo_url': repo_url,
                            'status': 'success',
                            'message': f'Files found:<br>{file_list_message}',
                            'output_path': final_output_path 
                        })
                    except Exception as e:
                        scan_results.append({
                            'repo_url': repo_url,
                            'status': 'error',
                            'message': f'Error scanning repository {repo_url}: {str(e)}'
                        })
            
            return JsonResponse({'status': 'completed', 'results': scan_results})
        else:
            return JsonResponse({'error': f"Invalid scan_type: '{scan_type}'. Expected 'public_branches' or 'user_repos'."}, status=400)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body.'}, status=400)
    except Exception as e:
        print(f"Unexpected error in scan_selected_repositories_view: {str(e)}") # 개발 중 임시 로깅
        return JsonResponse({'error': f'An unexpected server error occurred.'}, status=500)


@login_required
def github_connect_view(request):
    if request.method == 'POST':
        pat = request.POST.get('github_pat')
        if not pat:
            messages.error(request, "GitHub Personal Access Token을 입력해주세요.")
            return redirect('accounts:settings') 

        try:
            headers = {'Authorization': f'token {pat}'}
            response = requests.get('https://api.github.com/user', headers=headers)
            response.raise_for_status() 
            
            request.user.github_access_token = pat
            request.user.save()
            messages.success(request, "GitHub 계정이 성공적으로 연결되었습니다.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                messages.error(request, "유효하지 않은 GitHub Personal Access Token입니다. 토큰과 권한(예: repo, user)을 확인해주세요.")
            else:
                messages.error(request, f"GitHub API 오류: {e.response.status_code} - {e.response.json().get('message', '알 수 없는 오류')}")
        except requests.exceptions.RequestException as e:
            messages.error(request, f"GitHub 연결 중 오류가 발생했습니다: {e}")
        
        return redirect('accounts:settings')

    return redirect('accounts:settings')

@login_required
def github_disconnect_view(request):
    if request.method == 'POST': 
        if request.user.is_authenticated and hasattr(request.user, 'github_access_token') and request.user.github_access_token:
            request.user.github_access_token = None
            request.user.save()
            messages.success(request, "GitHub 계정 연결이 해제되었습니다.")
        else:
            messages.info(request, "연결된 GitHub 계정이 없습니다.")
    else:
        messages.error(request, "잘못된 요청 방식입니다. POST 요청을 사용해주세요.")
    return redirect('accounts:settings')

# User-facing Login Page View (Session-based)
def login_page_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password) # Use email as username
        if user is not None:
            auth_login(request, user)
            # Determine redirect URL after successful login
            redirect_url = request.GET.get('next', '/') # Redirect to 'next' if present, else to home
            if not redirect_url or redirect_url.startswith('/accounts/login_page'): # Avoid redirecting back to login
                redirect_url = '/' # Default to home if next is login page or empty
            return JsonResponse({'success': True, 'redirect_url': redirect_url})
        else:
            return JsonResponse({'success': False, 'error': 'Invalid credentials. Please try again.'}, status=400)
    # For GET request, just render the login page
    return render(request, 'accounts/login.html')

# User-facing Logout Page View (Session-based)
@login_required
def logout_page_view(request):
    auth_logout(request)
    return redirect('home') # Redirect to home page after logout

def _parse_github_url(url):
    """Parses a GitHub URL and returns (owner, repo_name) or None if invalid."""
    # Regex for standard GitHub repo URLs (http, https, with/without .git)
    match = re.match(r"^https?://github\.com/([^/]+)/([^/.]+)(\.git)?/?$", url)
    if match:
        return match.group(1), match.group(2)
    return None

@require_GET
def list_public_repository_branches(request):
    repo_url = request.GET.get('repo_url', '').strip()
    if not repo_url:
        return JsonResponse({'error': 'Repository URL is required.'}, status=400)

    # _parse_github_url 함수는 이미 views.py 내에 정의되어 있다고 가정합니다.
    # 없다면, 이전 단계에서 사용된 _parse_github_url 함수를 참고하여 추가해야 합니다.
    # def _parse_github_url(url):
    #     match = re.match(r"^https?://github\.com/([^/]+)/([^/.]+)(\.git)?/?$", url)
    #     if match:
    #         return match.group(1), match.group(2)
    #     return None

    parsed_url = _parse_github_url(repo_url)
    if not parsed_url:
        return JsonResponse({'error': 'Invalid GitHub repository URL format. Example: https://github.com/owner/repo'}, status=400)
    
    owner, repo_name = parsed_url
    
    api_url = f'https://api.github.com/repos/{owner}/{repo_name}/branches'
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'X-GitHub-Api-Version': '2022-11-28'
    }
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status() # 4XX 또는 5XX 응답 코드에 대해 HTTPError 발생
        
        branches_data = response.json()
        # 프론트엔드에서 각 브랜치 객체가 'name' 키를 가질 것으로 예상
        branches = [{'name': branch['name']} for branch in branches_data] 
        
        return JsonResponse(branches, safe=False) # safe=False는 리스트를 직접 JsonResponse로 보낼 때 필요

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return JsonResponse({'error': f'Repository not found or is not public: {owner}/{repo_name}.'}, status=404)
        elif e.response.status_code == 403: # API 속도 제한 또는 접근 권한 문제
             try:
                error_details = e.response.json()
                message = error_details.get('message', 'Access forbidden. This might be due to API rate limits or other permission issues.')
             except ValueError: # 응답이 JSON이 아닐 경우
                message = 'Access forbidden. GitHub API rate limit likely exceeded or other permission issues.'
             return JsonResponse({'error': message}, status=403)
        else: # 기타 HTTP 오류
            return JsonResponse({'error': f'GitHub API error: {str(e)} (Status: {e.response.status_code})'}, status=e.response.status_code)
    except requests.exceptions.RequestException as e: # 네트워크 문제 등
        return JsonResponse({'error': f'Error connecting to GitHub: {str(e)}'}, status=500)
    except ValueError: # JSON 디코딩 오류
        return JsonResponse({'error': 'Invalid JSON response from GitHub API.'}, status=500)


@login_required
def add_repository_by_url(request):
    if request.method == 'POST':
        repo_url = request.POST.get('repository_url', '').strip()
        user = request.user

        if not repo_url:
            messages.error(request, 'Please enter a GitHub repository URL.')
            return redirect('accounts:settings')

        parsed_url = _parse_github_url(repo_url)
        if not parsed_url:
            messages.error(request, 'Invalid GitHub repository URL format. Please use a format like https://github.com/owner/repository-name.')
            return redirect('accounts:settings')

        owner, repo_name = parsed_url

        if not hasattr(user, 'github_access_token') or not user.github_access_token:
            messages.error(request, 'GitHub account not connected. Please connect your GitHub account first.')
            return redirect('accounts:settings')

        token = user.github_access_token
        api_url = f'https://api.github.com/repos/{owner}/{repo_name}'
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'X-GitHub-Api-Version': '2022-11-28'
        }

        try:
            response = requests.get(api_url, headers=headers)
            if response.status_code == 200:
                # Successfully accessed the repository info
                # Here, you would typically save this repository to the user's list of managed repos
                # For now, just a success message.
                # Example: UserManagedRepository.objects.create(user=user, url=repo_url, name=f"{owner}/{repo_name}")
                messages.success(request, f'Repository "{owner}/{repo_name}" seems accessible and is ready to be tracked. (Actual saving/scanning to be implemented)')
            elif response.status_code == 404:
                messages.error(request, f'Could not find repository "{owner}/{repo_name}". It might be a private repository you do not have access to, or the URL is incorrect.')
            elif response.status_code == 401 or response.status_code == 403:
                messages.error(request, f'Authentication failed for "{owner}/{repo_name}". Your GitHub token might be invalid or lacks permissions for this repository.')
            else:
                messages.error(request, f'Failed to access repository "{owner}/{repo_name}". GitHub API returned status {response.status_code}. Details: {response.json().get("message", "No details provided.")}')
        
        except requests.exceptions.RequestException as e:
            messages.error(request, f'An error occurred while trying to connect to GitHub: {e}')
        
        return redirect('accounts:settings')

    # If not POST, just redirect (or handle as an error)
    return redirect('accounts:settings')
