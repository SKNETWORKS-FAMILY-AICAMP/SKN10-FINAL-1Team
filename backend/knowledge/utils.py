from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypedDict, Dict, Sequence, Union, Optional, Any
import asyncio

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




def get_index_lists() :
    """Pinecone 인덱스 목록을 가져오는 함수"""
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
    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

    # 1-3) 인덱스 목록 정보 가져오기
    results = []
    print(pc.list_indexes())
    for idx in pc.list_indexes():               # idx는 IndexModel 객체
        name = idx.name                         # 문자열
        dim  = idx.dimension                    # 벡터 차원

        # describe_index_stats() : '해당' 인덱스 통계 정보 가져오기
        # pc.Index(name) : 해당 인덱스 객체 생성
        index_stats = pc.Index(name).describe_index_stats()
        total_count = sum(
            ns.vector_count
            for ns in index_stats.namespaces.values()
        )
        results.append({
        "name":         name,
        "dimension":    dim,
        "total_count":  total_count
        })

    return results

def get_sessions() :
    """세션 목록을 가져오는 함수""" 
    sessions = ChatSession.objects.all()
    return sessions

def get_users() :
    """유저 목록을 가져오는 함수""" 
    users = User.objects.all()
    return users
    

