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
import requests
import re # For parsing GitHub URL
from django.contrib import messages

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
                    'html_url': repo['html_url'],
                    'private': repo['private'],
                    'owner_login': repo['owner']['login']
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
