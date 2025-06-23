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
import subprocess # For running background processes
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from django.conf import settings
import shutil
from git import Repo, GitCommandError

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
    """Parses a GitHub URL to extract owner and repository name, ignoring subpaths."""
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname != 'github.com':
            return None, None
        
        path_parts = parsed_url.path.strip('/').split('/')
        
        if len(path_parts) >= 2:
            owner = path_parts[0]
            repo_name = path_parts[1]
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]
            return owner, repo_name
            
    except Exception:
        return None, None
        
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
    print(f"DEBUG: User: {user.email}, GitHub Token available: {'YES' if github_token and github_token.strip() else 'NO'}")

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
@csrf_exempt
def scan_selected_repositories_view(request):
    print("\n!!! ENTERING scan_selected_repositories_view !!!")
    try:
        data = json.loads(request.body.decode('utf-8'))
        scan_type = data.get('scan_type')
        print(f"DEBUG: scan_type received: '{scan_type}'")
        user = request.user
        github_token = getattr(user, 'github_access_token', None)
        scan_results = []
        s3_client = None
        print("DEBUG: BEFORE S3 CLIENT INITIALIZATION BLOCK (Corrected)")
        
        # Check for necessary S3 settings using getattr for safety
        aws_access_key_id = getattr(settings, 'AWS_ACCESS_KEY_ID', None)
        aws_secret_access_key = getattr(settings, 'AWS_SECRET_ACCESS_KEY', None)
        s3_bucket_name_from_settings = getattr(settings, 'AWS_STORAGE_BUCKET_NAME', None) # Use the name defined in settings.py
        aws_region_name = getattr(settings, 'AWS_DEFAULT_REGION', None)

        print(f"DEBUG: S3 Config Check -> AWS_ACCESS_KEY_ID set: {bool(aws_access_key_id)}, AWS_SECRET_ACCESS_KEY set: {bool(aws_secret_access_key)}, S3_BUCKET_NAME set: {bool(s3_bucket_name_from_settings)}, AWS_DEFAULT_REGION set: {bool(aws_region_name)}")

        if aws_access_key_id and aws_secret_access_key and s3_bucket_name_from_settings and aws_region_name:
            print(f"DEBUG: Attempting to initialize S3 client with bucket: {s3_bucket_name_from_settings}, region: {aws_region_name}, access_key_id: {'*' * (len(aws_access_key_id) - 4) + aws_access_key_id[-4:] if aws_access_key_id else 'Not Set'}")
            try:
                s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region_name
                )
                print(f"DEBUG: S3 client initialized successfully. Target Bucket: {s3_bucket_name_from_settings}")
            except NoCredentialsError:
                print("DEBUG: S3 client initialization failed: No AWS credentials found by Boto3.")
                s3_client = None
            except PartialCredentialsError:
                print("DEBUG: S3 client initialization failed: Incomplete AWS credentials found by Boto3.")
                s3_client = None
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code')
                if error_code == 'InvalidClientTokenId' or error_code == 'SignatureDoesNotMatch':
                    print(f"DEBUG: S3 client initialization failed: Invalid AWS credentials (Access Key ID or Secret Access Key). Error: {e}")
                elif error_code == 'AccessDenied':
                     print(f"DEBUG: S3 client initialization failed: Access Denied. Check IAM permissions for user and bucket. Error: {e}")
                else:
                    print(f"DEBUG: S3 client initialization failed with ClientError: {e}")
                s3_client = None
            except Exception as e:
                print(f"DEBUG: An unexpected error occurred during S3 client initialization: {e}")
                s3_client = None
        else:
            missing_s3_settings = []
            if not aws_access_key_id: missing_s3_settings.append("AWS_ACCESS_KEY_ID")
            if not aws_secret_access_key: missing_s3_settings.append("AWS_SECRET_ACCESS_KEY")
            if not s3_bucket_name_from_settings: missing_s3_settings.append("S3_BUCKET_NAME")
            if not aws_region_name: missing_s3_settings.append("AWS_DEFAULT_REGION")
            print(f"DEBUG: S3 client NOT initialized. Missing S3 configurations in settings: {', '.join(missing_s3_settings)}")
            s3_client = None 
        
        print("DEBUG: AFTER S3 CLIENT INITIALIZATION BLOCK (Corrected)")

        output_base_dir = os.path.join(PROJECT_ROOT, 'scanned_tutorials')
        os.makedirs(output_base_dir, exist_ok=True)
        
        main_script_path = os.path.join(PROJECT_ROOT, 'main.py')

        if scan_type == 'user_repos':
            repo_urls = data.get('repo_urls', [])
            if not repo_urls:
                return JsonResponse({'status': 'error', 'message': 'Repository URLs are required.'}, status=400)
            if not github_token:
                return JsonResponse({'status': 'error', 'message': 'GitHub token is required.'}, status=403)

            for repo_url in repo_urls:
                try:
                    owner, repo_name = _parse_github_url(repo_url)
                    if not owner or not repo_name:
                        scan_results.append({'repo_url': repo_url, 'status': 'error', 'message': 'Invalid GitHub URL format.'})
                        continue

                    repo_output_dir = os.path.join(output_base_dir, f"{owner}_{repo_name}")
                    # --- S3 Upload Logic for Original Files (User Repos - Default Branch) ---
                    if s3_client and s3_bucket_name_from_settings:
                        current_branch_for_s3 = "default_branch" # Placeholder for S3 key, ideally detect actual default branch
                        print(f"DEBUG: S3 Upload - User Repo '{repo_url}': Preparing to clone default branch and upload original files for repo {owner}/{repo_name}.")
                        temp_clone_dir = tempfile.mkdtemp()
                        print(f"DEBUG: S3 Upload - User Repo '{repo_url}': Created temp directory for cloning: {temp_clone_dir}")
                        try:
                            # 1. Construct the correct base .git URL using owner and repo_name
                            correct_git_clone_url_base = f"https://github.com/{owner}/{repo_name}.git"

                            # 2. Parse branch name from the input repo_url (which might be a /tree/ URL)
                            parsed_branch_name_for_clone = None
                            if "/tree/" in repo_url:
                                try:
                                    parts = repo_url.split("/tree/")
                                    if len(parts) > 1:
                                        branch_and_maybe_path = parts[1]
                                        # Ensure branch name doesn't include .git if it was mistakenly appended to repo_url previously
                                        parsed_branch_name_for_clone = branch_and_maybe_path.split('/')[0].replace('.git', '')
                                        print(f"DEBUG: S3 Upload - Parsed branch '{parsed_branch_name_for_clone}' from repo_url '{repo_url}'")
                                except Exception as e_parse_branch:
                                    print(f"DEBUG: S3 Upload - Could not parse branch from repo_url '{repo_url}'. Error: {e_parse_branch}")
                            
                            # Update current_branch_for_s3 based on parsing result (this var is defined before this try block)
                            if parsed_branch_name_for_clone:
                                current_branch_for_s3 = parsed_branch_name_for_clone
                            else:
                                current_branch_for_s3 = "default_branch" # Fallback if no branch parsed or repo_url is base

                            # 3. Prepare clone URL with token (using correct_git_clone_url_base)
                            git_repo_url_for_clone_with_token = correct_git_clone_url_base
                            if github_token:
                                if correct_git_clone_url_base.startswith('https://github.com/'):
                                    stripped_url = correct_git_clone_url_base[len('https://'):]
                                    git_repo_url_for_clone_with_token = f"https://{github_token}@{stripped_url}"
                                    print(f"DEBUG: S3 Upload - User Repo '{repo_url}': Using token for repo clone. Effective clone URL base: {git_repo_url_for_clone_with_token}")
                                else:
                                    print(f"WARNING: S3 Upload - User Repo '{repo_url}': Unexpected base repo URL format for token insertion: {correct_git_clone_url_base}")

                            # 4. Perform clone
                            clone_kwargs = {'depth': 1}
                            if parsed_branch_name_for_clone:
                                clone_kwargs['branch'] = parsed_branch_name_for_clone
                            
                            print(f"DEBUG: S3 Upload - User Repo '{repo_url}': Cloning from '{git_repo_url_for_clone_with_token}' (branch: {clone_kwargs.get('branch', 'default')}) into '{temp_clone_dir}'")
                            Repo.clone_from(git_repo_url_for_clone_with_token, temp_clone_dir, **clone_kwargs)
                            print(f"DEBUG: S3 Upload - User Repo '{repo_url}': Successfully cloned branch '{current_branch_for_s3}' of {owner}/{repo_name}.")

                            user_identifier = str(user.id) if user.is_authenticated and hasattr(user, 'id') and user.id is not None else "unknown_user"

                            for root_dir_walk, dirs_walk, files_in_dir_walk in os.walk(temp_clone_dir):
                                if '.git' in dirs_walk:
                                    dirs_walk.remove('.git')
                                
                                for file_name_walk in files_in_dir_walk:
                                    local_file_path = os.path.join(root_dir_walk, file_name_walk)
                                    relative_file_path_str = ''
                                    if local_file_path.startswith(temp_clone_dir + os.sep):
                                        relative_file_path_str = local_file_path[len(temp_clone_dir + os.sep):]
                                    elif local_file_path == temp_clone_dir:
                                        relative_file_path_str = ''
                                    else:
                                        relative_file_path_str = local_file_path # Fallback
                                    
                                    s3_object_key = f"original_sources/{user_identifier}/{owner}/{repo_name}/{current_branch_for_s3}/{relative_file_path_str.replace(os.sep, '/')}"
                                    
                                    print(f"DEBUG: S3 Upload - User Repo '{repo_url}': Attempting to upload '{local_file_path}' to S3 bucket '{s3_bucket_name_from_settings}' as '{s3_object_key}'")
                                    try:
                                        s3_client.upload_file(local_file_path, s3_bucket_name_from_settings, s3_object_key)
                                        print(f"DEBUG: S3 Upload - User Repo '{repo_url}': Successfully uploaded '{s3_object_key}'")
                                    except ClientError as e_s3_upload:
                                        print(f"DEBUG: S3 Upload - User Repo '{repo_url}': FAILED to upload '{s3_object_key}'. ClientError: {e_s3_upload}")
                                    except Exception as e_s3_upload_generic:
                                        print(f"DEBUG: S3 Upload - User Repo '{repo_url}': FAILED to upload '{s3_object_key}'. Exception: {e_s3_upload_generic}")
                        
                        except GitCommandError as e_git:
                            print(f"DEBUG: S3 Upload - User Repo '{repo_url}': Git clone FAILED for {owner}/{repo_name}. Error: {e_git}")
                        except Exception as e_clone_general:
                            print(f"DEBUG: S3 Upload - User Repo '{repo_url}': An unexpected error occurred during S3 prep/cloning for {owner}/{repo_name}. Error: {e_clone_general}")
                        finally:
                            if os.path.exists(temp_clone_dir):
                                print(f"DEBUG: S3 Upload - User Repo '{repo_url}': Cleaning up temp directory: {temp_clone_dir} for {owner}/{repo_name}")
                                # shutil.rmtree(temp_clone_dir) # Keep temp dir for LLM processing
                                print(f"DEBUG: S3 Upload - User Repo '{repo_url}': Temp directory cleaned up for {owner}/{repo_name}.")
                    else:
                        print(f"DEBUG: S3 Upload - User Repo '{repo_url}': SKIPPED for {owner}/{repo_name} because S3 client or bucket name is not configured.")
                    # --- End of S3 Upload Logic for User Repos ---
                    command = [
                        sys.executable,
                        main_script_path,
                        '--repo', repo_url,
                        '--output', repo_output_dir,
                        '--token', github_token,
                        '--language', 'korean'
                    ]
                    # Add S3 parameters if the client was initialized
                    if s3_client and s3_bucket_name_from_settings:
                        command.extend([
                            '--s3-bucket', s3_bucket_name_from_settings,
                            '--s3-access-key', aws_access_key_id,
                            '--s3-secret-key', aws_secret_access_key,
                            '--s3-region', aws_region_name,
                            '--user-id', str(user.id) if user.is_authenticated else 'anonymous'
                        ])
                    
                    subprocess.Popen(command, cwd=PROJECT_ROOT)
                    
                    scan_results.append({'repo_url': repo_url, 'status': 'success', 'message': 'Scan initiated successfully.'})

                except Exception as e:
                    print(f"Error launching scan for {repo_url}: {e}")
                    scan_results.append({'repo_url': repo_url, 'status': 'error', 'message': f'Failed to start scan: {str(e)}'})
        
        elif scan_type == 'public_branches':
            print(f"DEBUG: Entered 'public_branches' block. Branches to scan: {{data.get('branches', [])}}, Repo base URL: {{data.get('public_repo_url')}}")
            public_repo_url_base = data.get('public_repo_url')
            branches_to_scan = data.get('branches', [])

            if not public_repo_url_base or not branches_to_scan:
                return JsonResponse({'status': 'error', 'message': 'Repository URL and branches are required.'}, status=400)

            owner, repo_name = _parse_github_url(public_repo_url_base)
            if not owner or not repo_name:
                return JsonResponse({'status': 'error', 'message': 'Invalid GitHub URL format.'}, status=400)

            for branch_name in branches_to_scan:
                print(f"DEBUG: Processing branch: '{branch_name}' in 'public_branches' loop for repo {owner}/{repo_name}.")
                repo_scan_url = f"https://github.com/{owner}/{repo_name}/tree/{branch_name}"
                try:
                    repo_output_dir = os.path.join(output_base_dir, f"{owner}_{repo_name}_{branch_name}")
                    # --- S3 Upload Logic for Original Files ---
                    if s3_client and s3_bucket_name_from_settings:
                        print(f"DEBUG: S3 Upload - Branch '{branch_name}': Preparing to clone and upload original files for repo {owner}/{repo_name}.")
                        temp_clone_dir = tempfile.mkdtemp()
                        print(f"DEBUG: S3 Upload - Branch '{branch_name}': Created temp directory for cloning: {temp_clone_dir}")
                        try:
                            git_repo_url_for_clone = f"https://github.com/{owner}/{repo_name}.git"
                            print(f"DEBUG: S3 Upload - Branch '{branch_name}': Cloning from '{git_repo_url_for_clone}' (branch: {branch_name}) into '{temp_clone_dir}'")
                            Repo.clone_from(git_repo_url_for_clone, temp_clone_dir, branch=branch_name, depth=1)
                            print(f"DEBUG: S3 Upload - Branch '{branch_name}': Successfully cloned {owner}/{repo_name} branch {branch_name}.")

                            user_identifier = str(user.id) if user.is_authenticated and hasattr(user, 'id') and user.id is not None else "public_scan_user"

                            for root_dir_walk, dirs_walk, files_in_dir_walk in os.walk(temp_clone_dir):
                                # Exclude .git directory from S3 upload
                                if '.git' in dirs_walk:
                                    dirs_walk.remove('.git') 
                                
                                for file_name_walk in files_in_dir_walk:
                                    local_file_path = os.path.join(root_dir_walk, file_name_walk)
                                    # Ensure relative_file_path is correctly stripping temp_clone_dir prefix
                                    relative_file_path_str = ''
                                    if local_file_path.startswith(temp_clone_dir + os.sep):
                                        relative_file_path_str = local_file_path[len(temp_clone_dir + os.sep):]
                                    elif local_file_path == temp_clone_dir: # Should not happen for files
                                        relative_file_path_str = '' 
                                    else:
                                        relative_file_path_str = local_file_path # Fallback
                                    
                                    s3_object_key = f"original_sources/{user_identifier}/{owner}/{repo_name}/{branch_name}/{relative_file_path_str.replace(os.sep, '/')}"
                                    
                                    print(f"DEBUG: S3 Upload - Branch '{branch_name}': Attempting to upload '{local_file_path}' to S3 bucket '{s3_bucket_name_from_settings}' as '{s3_object_key}'")
                                    try:
                                        s3_client.upload_file(local_file_path, s3_bucket_name_from_settings, s3_object_key)
                                        print(f"DEBUG: S3 Upload - Branch '{branch_name}': Successfully uploaded '{s3_object_key}'")
                                    except ClientError as e_s3_upload:
                                        print(f"DEBUG: S3 Upload - Branch '{branch_name}': FAILED to upload '{s3_object_key}'. ClientError: {e_s3_upload}")
                                    except Exception as e_s3_upload_generic:
                                        print(f"DEBUG: S3 Upload - Branch '{branch_name}': FAILED to upload '{s3_object_key}'. Exception: {e_s3_upload_generic}")
                        
                        except GitCommandError as e_git:
                            print(f"DEBUG: S3 Upload - Branch '{branch_name}': Git clone FAILED for {owner}/{repo_name} branch {branch_name}. Error: {e_git}")
                        except Exception as e_clone_general:
                            print(f"DEBUG: S3 Upload - Branch '{branch_name}': An unexpected error occurred during S3 prep/cloning for {owner}/{repo_name} branch {branch_name}. Error: {e_clone_general}")
                        finally:
                            if os.path.exists(temp_clone_dir):
                                print(f"DEBUG: S3 Upload - Branch '{branch_name}': Cleaning up temp directory: {temp_clone_dir} for {owner}/{repo_name} branch {branch_name}")
                                # shutil.rmtree(temp_clone_dir) # Keep temp dir for LLM processing
                                print(f"DEBUG: S3 Upload - Branch '{branch_name}': Temp directory cleaned up for {owner}/{repo_name} branch {branch_name}.")
                    else:
                        print(f"DEBUG: S3 Upload - Branch '{branch_name}': SKIPPED for {owner}/{repo_name} branch {branch_name} because S3 client or bucket name is not configured.")
                    # --- End of S3 Upload Logic ---
                    command = [
                        sys.executable,
                        main_script_path,
                        '--repo', repo_scan_url,
                        '--output', repo_output_dir,
                        '--language', 'korean'
                    ]
                    # Add S3 parameters if the client was initialized
                    if s3_client and s3_bucket_name_from_settings:
                        command.extend([
                            '--s3-bucket', s3_bucket_name_from_settings,
                            '--s3-access-key', aws_access_key_id,
                            '--s3-secret-key', aws_secret_access_key,
                            '--s3-region', aws_region_name,
                            '--user-id', str(user.id) if user.is_authenticated else 'anonymous'
                        ])
                    
                    subprocess.Popen(command, cwd=PROJECT_ROOT)
                    
                    scan_results.append({'repo_url': repo_scan_url, 'status': 'success', 'message': 'Scan initiated successfully.'})
                except Exception as e:
                    print(f"Error launching scan for {repo_scan_url}: {e}")
                    scan_results.append({'repo_url': repo_scan_url, 'status': 'error', 'message': f'Failed to start scan: {str(e)}'})

        else:
            return JsonResponse({'status': 'error', 'message': f'Invalid scan_type: {scan_type}'}, status=400)

        return JsonResponse({'status': 'completed', 'results': scan_results})

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON in request body.'}, status=400)
    except Exception as e:
        print(f"Unexpected error in scan_selected_repositories_view: {e}")
        return JsonResponse({'status': 'error', 'message': 'An unexpected server error occurred.'}, status=500)

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
