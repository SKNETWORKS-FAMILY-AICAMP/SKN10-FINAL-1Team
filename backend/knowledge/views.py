from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse # For placeholder API views
from django.urls import reverse
from django.contrib import messages
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .utils import get_namespaces, get_index_lists, get_sessions, get_users, get_postgre_db, get_all_table
from .utils import make_index, remove_index, generate_password, get_documents, get_5_sessions
from conversations.models import ChatSession, ChatMessage
from accounts.models import User, Organization
import csv, io

# Create your views here.

"""dashboard 출력"""
def dashboard_view(request, screen_type):
    context = {"screen_type" : screen_type}
    if screen_type == "home" :
        context['recent_sessions'] = get_5_sessions()
    elif screen_type == "db" :
        db = request.GET.get('db', 'postgre')
        context['db'] = db

        if db == "postgre" :
            context['postgre_db'] = get_postgre_db()
            context['tables'] = get_all_table()
        elif db == "pinecone" :
            context['indexes'] = get_index_lists()
        elif db == "s3" :
            pass

    elif screen_type == "log" :
        context['sessions'] = get_sessions(request) 
    elif screen_type == "user" :
        context['users'] = get_users(request) 
    else : 
        return JsonResponse({'error': 'Invalid section'}, status=400)
    return render(request, 'knowledge/dashboard.html', context)

"""Index 생성"""
def create_index(request):
    if request.method == 'POST':
        index_name = request.POST.get('name') # 인덱스명
        vector_type = request.POST.get('vector_type') # 벡터 타입
        metric = request.POST.get('metric') # 유사도 측정 방식
        dimension = request.POST.get('dimension') # 차원 수
        cloud = request.POST.get('cloud')  # 클라우드 제공자
        region = request.POST.get('region')  # 클라우드 지역
        
        # index_name이나 dimension을 입력하지 않았을때 예외처리
        if index_name == "" or dimension == '' :
            messages.error(request, '❌ 인덱스명과 차원 수는 필수 입력 항목입니다!')
        # vector_type이 dense인데 dimension이 0이하인 경우 예외처리
        elif vector_type == "dense" and int(dimension) <= 0 : 
            messages.error(request, '❌ dense 인덱스의 차원 수는 0보다 큰 정수여야 합니다!')
        # 정상적으로 입력했을때 
        else : 
            dimension = int(dimension)
            is_created = make_index(name=index_name, vector_type=vector_type, metric=metric, dimension=dimension, cloud=cloud, region=region)

            if is_created: 
                messages.success(request, '✅ 성공적으로 인덱스가 생성되었습니다!')
            else: 
                messages.error(request, '⚠️ 이미 존재하는 인덱스명입니다!')

        url = reverse('knowledge:dashboard', args=['db']) + '?db=pinecone'
        return redirect(url)
    else : 
        return JsonResponse({'error': '잘못된 접근입니다! POST형식의 응답을 받지 못했습니다.'}, status=405)

"""Index 삭제"""
def delete_index(request):
    """특정 index를 제거"""
    if request.method == 'POST':
        index_name = request.POST.get('name') # 인덱스 이름
        # 입력한 인덱스 이름이 비어있을 때 예외처리
        if index_name == "" :
            messages.error(request, '❌ 삭제할 인덱스명을 입력해주세요!')
        else : 
            is_deleted = remove_index(index_name)
            # 정상적으로 삭제되었을 때
            if is_deleted :
                messages.success(request, '✅ 성공적으로 해당 인덱스가 삭제되었습니다!')
            # 해당 인덱스가 존재하지 않아 삭제되지 않았을 때
            else: 
                messages.error(request, '⚠️ 입력하신 인덱스가 DB에 없습니다!')
        
        url = reverse('knowledge:dashboard', args=['db']) + '?db=pinecone'
        return redirect(url)
    else : 
        return JsonResponse({'error': '잘못된 접근입니다! POST형식의 응답을 받지 못했습니다.'}, status=405)

"""User 생성"""
def create_user(request) :
    if request.method == 'POST':
        email = request.POST.get('email', '').strip() # 이메일
        pw = request.POST.get('pw', '')  # 비밀번호           
        pw_confirm = request.POST.get('pw_confirm', '') # 비밀번호 확인
        username = request.POST.get('username', '').strip() # 유저명
        authority = request.POST.get('authority') # 권한
        department = request.POST.get('department') # 부서
        org = Organization.objects.get(name=department) # 임시 방편으로 Organization를 생성
        redirect_url = reverse('knowledge:dashboard', args=['user'])

        # 서버사이드 검증
        if not email or not pw or not pw_confirm or not username:
            messages.error(request, '❌ 모든 필드를 입력해 주세요.')
            return redirect(redirect_url)
        if pw != pw_confirm:
            messages.error(request, '❌ 비밀번호가 일치하지 않습니다.')
            return redirect(redirect_url)
        # 중복 이메일 체크
        if User.objects.filter(email=email).exists():
            messages.error(request, '⚠️ 이미 등록된 이메일입니다.')
            return redirect(redirect_url)

        # 실제 생성
        try:
            user = User.objects.create_user(email=email,password=pw,name=username,role=authority,org=org)
        except Exception as e:
            messages.error(request, e)
            return redirect(redirect_url)
        
        messages.success(request, '✅ 사용자 생성이 정상적으로 완료되었습니다!')
        return redirect(redirect_url)
    else : 
        return JsonResponse({'error': '잘못된 접근입니다! POST형식의 응답을 받지 못했습니다.'}, status=405)

"""User 다중 생성"""
def create_multi_user(request) :
    if request.method == 'POST':
        # 1) 파일 꺼내기
        uploaded = request.FILES.get('file')
        redirect_url = reverse('knowledge:dashboard', args=['user'])
        if not uploaded:
            messages.error(request, '파일이 없습니다!')
            return redirect(redirect_url)

        # 2) CSV 읽기 (csv 모듈 + StringIO 사용)
        raw = uploaded.read()  # 바이너리 데이터
        try:
            data = raw.decode('utf-8')          # 먼저 UTF-8 시도
        except UnicodeDecodeError:
            try:
                data = raw.decode('cp949')      # 실패하면 CP949로 재시도
            except UnicodeDecodeError:
                data = raw.decode('euc-kr', errors='ignore')  
                # (필요하면 errors 옵션 추가)
        reader = csv.DictReader(io.StringIO(data))
        rows = list(reader) 
        print("✅ 정상적으로 csv파일을 읽었습니다.")

        # 3) rows 순회하면서 회원 생성
        for row in rows:
            name = row['name'].strip()
            id = row['id'].strip()
            authority = row['authority'].strip()
            dept = row['department'].strip()
            org = Organization.objects.get(name=dept)  # 예시

            User.objects.create_user(
                email=f'{id}@example.com', # 이메일 생성 로직
                password=generate_password(),     # 비번 지정
                name=name,
                role=authority,
                org=org,
            )
        messages.success(request, f'✅ {len(rows)}명의 계정을 생성했어!')
        return redirect(redirect_url)
    else : 
        return JsonResponse({'error': '잘못된 접근입니다! POST형식의 응답을 받지 못했습니다.'}, status=405)

"""User 삭제"""
def delete_user(request):
    if request.method == 'POST':
        email = request.POST.get('email') # 이메일
        url = reverse('knowledge:dashboard', args=['user'])

        try : 
            target = User.objects.get(email=email)
            print(User.objects.all())
        except : 
            messages.error(request, '⚠️ 해당 이메일의 계정이 없습니다!')
            return redirect(url)
        
        target.delete() # 완전 삭제
        messages.success(request, f'✅ {email} 계정이 삭제되었습니다.')
        return redirect(url)
    else : 
        return JsonResponse({'error': '잘못된 접근입니다! POST형식의 응답을 받지 못했습니다.'}, status=405)

"""User 다중 삭제"""
def delete_multi_user(request) :
    if request.method == 'POST':
        # 1) 파일 꺼내기
        uploaded = request.FILES.get('file')
        redirect_url = reverse('knowledge:dashboard', args=['user'])
        if not uploaded:
            messages.error(request, '파일이 없습니다!')
            return redirect(redirect_url)

        # 2) CSV 읽기 (csv 모듈 + StringIO 사용)
        raw = uploaded.read()  # 바이너리 데이터
        try:
            data = raw.decode('utf-8')          # 먼저 UTF-8 시도
        except UnicodeDecodeError:
            try:
                data = raw.decode('cp949')      # 실패하면 CP949로 재시도
            except UnicodeDecodeError:
                data = raw.decode('euc-kr', errors='ignore')  
                # (필요하면 errors 옵션 추가)
        reader = csv.DictReader(io.StringIO(data))
        rows = list(reader) 
        print("✅ 정상적으로 csv파일을 읽었습니다.")

        # 3) rows 순회하면서 회원 생성
        for row in rows:
            email = row['email'].strip()
            user = User.objects.get(email=email)
            user.delete()


        messages.success(request, f'✅ {len(rows)}명의 계정을 삭제했어!')
        return redirect(redirect_url)
    else : 
        return JsonResponse({'error': '잘못된 접근입니다! POST형식의 응답을 받지 못했습니다.'}, status=405)


# Placeholder API views to match knowledge/urls.py
# These should be properly implemented later.

"""해당 index 상세화면"""
def index_detail(request, index_name) :
    namespaces = get_namespaces(index_name)
    return JsonResponse({'namespaces': namespaces})

"""해당 namespace 상세화면"""
def namespace_detail(request,index_name,namespace_name) :
    documents = get_documents(index_name,namespace_name)
    print("✅ 정상적으로 문서들을 받았습니다!")
    return render(request, 'knowledge/documents.html', {'documents': documents,'index_name': index_name,'namespace_name': namespace_name,})

"""해당 session 상세화면"""
def session_detail(request, session_id) :
    session = ChatSession.objects.get(id=session_id)
    related_messages = session.messages.all()
    messages = []
    for msg in related_messages:
        messages.append({
            "id" : msg.id,
            "created_at" : msg.created_at.strftime("%Y-%m-%d %H:%M"),
            "role" : msg.role,
            "content" : msg.content
        })
    print(messages)
    return JsonResponse({"messages" : messages})




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
