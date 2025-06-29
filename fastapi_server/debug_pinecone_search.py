#!/usr/bin/env python3
"""
Pinecone 검색 디버깅 스크립트
"""

import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.coding_agent_tools import get_pinecone_index

def debug_pinecone_search():
    """Pinecone 검색 문제를 진단합니다."""
    
    print("🔍 Pinecone 검색 디버깅 시작")
    print("=" * 50)
    
    # 1. 환경 변수 확인
    print("\n1️⃣ 환경 변수 확인")
    required_env_vars = [
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME", 
        "PINECONE_INDEX_HOST"
    ]
    
    for var in required_env_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {value[:10]}..." if len(value) > 10 else f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: 설정되지 않음")
    
    # 2. Pinecone 인덱스 연결 테스트
    print("\n2️⃣ Pinecone 인덱스 연결 테스트")
    try:
        index = get_pinecone_index()
        print("✅ Pinecone 인덱스 연결 성공")
        
        # 3. 인덱스 통계 확인
        print("\n3️⃣ 인덱스 통계 확인")
        stats = index.describe_index_stats()
        print(f"✅ 인덱스 통계: {stats}")
        
        # 4. 네임스페이스 확인
        print("\n4️⃣ 네임스페이스 확인")
        namespaces = stats.namespaces
        if namespaces:
            print("📋 사용 가능한 네임스페이스:")
            for ns, info in namespaces.items():
                print(f"   - {ns}: {info.vector_count}개 벡터")
        else:
            print("⚠️  네임스페이스가 없습니다.")
        
        # 5. 특정 네임스페이스에서 검색 테스트
        print("\n5️⃣ 네임스페이스별 검색 테스트")
        test_namespaces = ["user", "be87b5f4-1096-46fd-9992-5703b9ef1229"]
        
        for ns in test_namespaces:
            print(f"\n🔍 네임스페이스 '{ns}'에서 검색 테스트:")
            try:
                # 간단한 쿼리로 테스트 (3072차원 벡터 사용)
                query_response = index.query(
                    namespace=ns,
                    vector=[0.0] * 3072,  # 올바른 차원 사용
                    top_k=5,
                    include_metadata=True,
                    include_values=False
                )
                
                if query_response.matches:
                    print(f"✅ '{ns}'에서 {len(query_response.matches)}개 결과 발견")
                    for i, match in enumerate(query_response.matches[:2], 1):
                        print(f"   {i}. ID: {match.id}")
                        if match.metadata:
                            print(f"      파일명: {match.metadata.get('original_filename', 'N/A')}")
                            print(f"      리포지토리: {match.metadata.get('github_user_repo', 'N/A')}")
                else:
                    print(f"⚠️  '{ns}'에서 결과 없음")
                    
            except Exception as e:
                print(f"❌ '{ns}' 검색 실패: {e}")
        
        # 6. 메타데이터 필드 확인
        print("\n6️⃣ 메타데이터 필드 확인")
        if namespaces:
            # 첫 번째 네임스페이스에서 샘플 데이터 확인
            first_ns = list(namespaces.keys())[0]
            try:
                query_response = index.query(
                    namespace=first_ns,
                    vector=[0.0] * 3072,  # 올바른 차원 사용
                    top_k=1,
                    include_metadata=True,
                    include_values=False
                )
                
                if query_response.matches:
                    metadata = query_response.matches[0].metadata
                    print(f"📋 네임스페이스 '{first_ns}'의 메타데이터 필드:")
                    for key, value in metadata.items():
                        if isinstance(value, str) and len(value) > 50:
                            print(f"   - {key}: {value[:50]}...")
                        else:
                            print(f"   - {key}: {value}")
                else:
                    print(f"⚠️  '{first_ns}'에서 샘플 데이터를 찾을 수 없습니다.")
                    
            except Exception as e:
                print(f"❌ 메타데이터 확인 실패: {e}")
        
    except Exception as e:
        print(f"❌ Pinecone 인덱스 연결 실패: {e}")
        return False
    
    print("\n✅ 디버깅 완료")
    return True

if __name__ == "__main__":
    debug_pinecone_search() 