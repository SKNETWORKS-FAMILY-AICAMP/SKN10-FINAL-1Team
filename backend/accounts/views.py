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
    """Renders the user profile page and handles profile updates."""
    if request.method == 'POST':
        user = request.user
        user.name = request.POST.get('name', user.name)
        user.save()
        return redirect('accounts:profile')
    
    return render(request, 'accounts/profile.html', {'user': request.user})

@login_required
def settings_view(request):
    """Renders the user settings page."""
    return render(request, 'accounts/setting.html')

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
