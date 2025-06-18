from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse # For placeholder API views

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .utils import get_index_lists,get_sessions, get_users
# Create your views here.

def dashboard_view(request):
    section = request.GET.get('section', 'home')
    context = {"section" : section}
    if section == "home" :
        pass
    elif section == "db" :
        context['indexes'] = get_index_lists()
    elif section == "log" :
        context['sessions'] = get_sessions() 
    elif section == "user" :
        context['users'] = get_users()
    elif section == "settings" :
        pass
    else : 
        return JsonResponse({'error': 'Invalid section'}, status=400)
    return render(request, 'knowledge/dashboard.html', context)

# Placeholder API views to match knowledge/urls.py
# These should be properly implemented later.

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def document_list_create_view(request):
    if request.method == 'GET':
        # Placeholder for listing documents
        return Response({'message': 'API: List documents placeholder'}, status=status.HTTP_200_OK)
    elif request.method == 'POST':
        # Placeholder for creating a document
        return Response({'message': 'API: Create document placeholder'}, status=status.HTTP_201_CREATED)

@api_view(['GET', 'PUT', 'PATCH', 'DELETE'])
@permission_classes([IsAuthenticated])
def document_detail_view(request, pk):
    # Placeholder for document detail, update, delete
    return Response({'message': f'API: Document detail for {pk} placeholder (method: {request.method})'}, status=status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def document_summary_view(request, pk):
    # Placeholder for document summary
    return Response({'message': f'API: Document summary for {pk} placeholder'}, status=status.HTTP_200_OK)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def search_documents_view(request):
    # Placeholder for searching documents (as in Pinecone example)
    return Response({'message': 'API: Search documents placeholder'}, status=status.HTTP_200_OK)
