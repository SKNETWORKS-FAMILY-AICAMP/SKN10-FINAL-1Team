from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypedDict, Dict, Sequence, Union, Optional, Any
import asyncio
from django.contrib import messages

from asgiref.sync import sync_to_async
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from conversations.models import ChatSession, ChatMessage, LlmCall
from accounts.models import User
from django.conf import settings
from django.db import connection
from functools import lru_cache
from django.core.paginator import Paginator 
import secrets
import string
import boto3
                    
def get_postgre_db() : 
    """PostgreSQL 데이터베이스 연결 정보 가져오는 함수"""
    db = settings.DATABASES['default']
    print("----------------------------------------------------------------")
    print(db['ENGINE'], db['NAME'])
    result = {
        "engine": db['ENGINE'],
        "name": db['NAME'],
        "host": db['HOST'],
        "port": db['PORT'],
    }
    return result


def get_all_table():
    """PostgreSQL 데이터베이스 모든 테이블의 정보를 가져오는 함수"""
    result = []
    all_table = set(connection.introspection.table_names()) 
    default_table = {'auth_group','auth_permission','auth_group_permissions','users_groups', 'users_user_permissions'
                    ,'django_content_type','django_session','django_admin_log','django_migrations'}
    tables = all_table.difference(default_table)
    print(tables)

    with connection.cursor() as cursor:
        for table in tables :
            # 테이블 레코드 수 가져오기
            cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
            count = cursor.fetchone()[0]

            # 테이블 크기 가져오기
            sql = f"""
                SELECT COUNT(*) AS cnt, pg_size_pretty(
                    pg_total_relation_size('public.{table}')
                )                           AS size
                FROM "{table}";
            """
            cursor.execute(sql)
            _, size = cursor.fetchone()  # (12345, '890 MB') 이런 튜플이 돌아옴
            result.append ({
                "name" :table,
                "count" : count,
                "size" : size
            })
    return result

@lru_cache(maxsize=1)
def connect_pinecone() :
    """Pinecone 클라이언트를 반환하는 헬퍼 함수"""
    load_dotenv()
    # 1-1) OpenAI 클라이언트 생성
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")
    openai_client = OpenAI(api_key=openai_api_key)
    
    # 1-2) Pinecone 인스턴스 생성
    pinecone_api_key = os.getenv("PINECONE_API_KEY") # Pinecone API 키
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT") # Pinecone 환경
    if not pinecone_api_key or not pinecone_env:
        raise ValueError("PINECONE_API_KEY 또는 PINECONE_ENVIRONMENT 환경 변수가 설정되어 있지 않습니다.")
    
    return Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

def get_index_lists() :
    """Pinecone Index 정보들을 가져오는 함수"""
    pc = connect_pinecone()

    # 1-3) 인덱스 목록 정보 가져오기
    results = []
    for idx in pc.list_indexes():               # idx는 IndexModel 객체
        name = idx.name                         # 문자열
        dim  = idx.dimension                    # 벡터 차원

        # describe_index_stats() : '해당' 인덱스 통계 정보 가져오기
        # pc.Index(name) : 해당 인덱스 객체 생성
        index_stats = pc.Index(name).describe_index_stats()
        idx_meta = pc.describe_index(name=name)
        region = idx_meta.spec.serverless.region
        cloud  = idx_meta.spec.serverless.cloud
        total_count = sum(
            ns.vector_count
            for ns in index_stats.namespaces.values()
        )
        results.append({
        "name":         name,
        "dimension":    dim,
        "total_count":  total_count,
        "region" : region,
        "cloud" : cloud
        })
    return results

def bucket_list(request):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_S3_REGION_NAME,
    )
    # 모든 버킷 이름 가져오기
    buckets = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    print(buckets)
    return buckets

def make_index(name, cloud, region, metric, vector_type, dimension) : 
    """Pinecone 인덱스를 생성하는 함수"""
    pc = connect_pinecone()
    spec = ServerlessSpec(cloud  = cloud,region = region)

    # 1-3) 인덱스 생성
    if name not in pc.list_indexes().names():
        # ⚠️벡터 타입이 sparse인 경우 dimension 인자를 전달하면 오류생김!
        if vector_type == "sparse" :
            pc.create_index(
            spec=spec,
            name=name,
            metric=metric,
            vector_type=vector_type
            )
        # ⚠️ 벡터 타입이 dense인 경우 모든 인자 전달
        else : 
            pc.create_index(
                spec=spec,
                name=name,
                dimension=dimension,
                metric=metric,
                vector_type=vector_type
            )
        print(f"✅ Pinecone 인덱스 생성: {name}")
        return True
    else:
        print(f"ℹ️ Pinecone 인덱스 이미 존재: {name}")
        return False

def remove_index(name) : 
    """Pinecone 인덱스를 삭제하는 함수"""
    pc = connect_pinecone()

    # 1-3) 해당 인덱스가 Pinecone index에 있는지 확인 후 인덱스 삭제 
    if name not in pc.list_indexes().names():
        print(f"ℹ️ {name} 인덱스가 Pinecone DB에 없음! ")
        return False
    else : 
        pc.delete_index(name=name)
        print(f"✅ 해당 {name} 인덱스를 성공적으로 삭제함!")
        return True

def get_namespaces(index_name) :
    """Pinecone 해당 인덱스의 네임스페이스들을 가져오는 함수"""
    pc = connect_pinecone()

    # 1. 인덱스 존재 여부 확인
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes :
        raise ValueError(f"⚠️ 인덱스 '{index_name}'가 Pinecone에 존재하지 않습니다. 현재 인덱스 목록: {existing_indexes}")
    
    # 2. 인덱스 연결
    index = pc.Index(index_name)
    namespaces = index.describe_index_stats().namespaces
    print(f"✅ Pinecone 인덱스 '{index_name}' 연결 완료 (Namespaces: {len(namespaces)})")

    # 3. 네임스페이스 전처리
    flatten_namespaces = {
        (name if name != '' else 'unknown') : info['vector_count']
        for name, info in namespaces.items()
    }
    print(flatten_namespaces)
    return flatten_namespaces

def get_documents(index_name, namespace_name) :
    """Pinecone 해당 인덱스의 네임스페이스들을 가져오는 함수"""
    pc = connect_pinecone()

    # 1. 인덱스 존재 여부 확인
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes :
        raise ValueError(f"⚠️ 인덱스 '{index_name}'가 Pinecone에 존재하지 않습니다. 현재 인덱스 목록: {existing_indexes}")
    
    # 2. 인덱스 연결
    index = pc.Index(index_name)
    # 전체 통계 한 번만 조회
    desc = index.describe_index_stats()

    # 네임스페이스별 카운트
    count = desc['namespaces'].get(namespace_name, {}).get('vector_count', 0)

    # 전체 차원은 top-level 에 있음
    dimension = desc.get('dimension', 0)

    # 이제 dummy 벡터가 올바른 길이를 갖습니다
    dummy = [0.0] * dimension

    res = index.query(
        vector=dummy,
        namespace=namespace_name,
        top_k=count,
        include_values=False,
        include_metadata=True,
    )

    print(f"✅ Pinecone 인덱스 '{index_name}' Pinecone namespace '{namespace_name}' 문서 수 '{count}'개)")

    vectors = [
    {   
        'id': m['id'],
        'namespace': m.get('metadata', {}).get('namespace',''),
        'original_filename' : m.get('metadata', {}).get('original_filename', ''),
        'text' : m.get('metadata', {}).get('text', ''),    
    } for m in res.get('matches', [])
    ]
    print(vectors)
    return vectors

def generate_password(length=15):
    """랜덤으로 비밀번호를 만드는 함수"""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def get_sessions(request) :
    """세션 목록을 가져오는 함수""" 
    sessions = ChatSession.objects.order_by("-started_at")
    page = request.GET.get('page', '1')  # 페이지
    paginator = Paginator(sessions, 10)  # 페이지당 10개씩 보여주기
    selected_sessions = paginator.get_page(page) # 10개의 sessions
    return selected_sessions

def get_users(request) :
    """유저 목록을 가져오는 함수""" 
    users = User.objects.order_by("-created_at")
    page = request.GET.get('page', '1') # 페이지
    paginator = Paginator(users, 10)  # 페이지당 10개씩 보여주기
    selected_users = paginator.get_page(page) # 10개의 sessions
    return selected_users

def get_5_sessions() :
    """세션 목록을 가져오는 함수""" 
    sessions = ChatSession.objects.order_by('-started_at')[:5]
    return sessions