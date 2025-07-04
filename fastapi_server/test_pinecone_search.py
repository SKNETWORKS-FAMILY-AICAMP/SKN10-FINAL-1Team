#!/usr/bin/env python3
"""
Pinecone 검색 도구 테스트 스크립트
"""

import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.coding_agent_tools import _search_tutorials, _search_tutorials_with_embedding

def test_pinecone_search():
    """Pinecone 검색 도구를 테스트합니다."""
    
    # 환경 변수 확인
    required_env_vars = [
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME", 
        "PINECONE_INDEX_HOST"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    if missing_vars:
        print(f"❌ 다음 환경 변수가 설정되지 않았습니다: {missing_vars}")
        return False
    
    print("✅ 환경 변수가 모두 설정되었습니다.")
    
    # 테스트 파라미터 (실제 데이터에 맞게 수정)
    test_user_id = "be87b5f4-1096-46fd-9992-5703b9ef1229"  # 실제 사용자 ID
    test_query = "자동차 등록 현황"
    test_repo = "SKNETWORKS-FAMILY-AICAMP/SKN10-1st-4Team"  # 실제 리포지토리
    test_branch = "main"  # 실제 브랜치
    
    try:
        print(f"\n🔍 기본 검색 테스트 (쿼리: '{test_query}')")
        results = _search_tutorials(
            user_id=test_user_id,
            query=test_query,
            top_k=3
        )
        print(f"✅ 검색 성공! {len(results)}개 결과 발견")
        for i, result in enumerate(results, 1):
            print(f"  {i}. ID: {result['id']}, Score: {result['score']:.4f}")
            if result.get('metadata'):
                metadata = result['metadata']
                print(f"     파일명: {metadata.get('original_filename', 'N/A')}")
                print(f"     리포지토리: {metadata.get('github_user_repo', 'N/A')}")
                print(f"     브랜치: {metadata.get('branch_name', 'N/A')}")
                print(f"     텍스트 미리보기: {metadata.get('text', 'N/A')[:100]}...")
    
    except Exception as e:
        print(f"❌ 기본 검색 실패: {e}")
        return False
    
    try:
        print(f"\n🔍 임베딩 검색 테스트 (쿼리: '{test_query}')")
        results = _search_tutorials_with_embedding(
            user_id=test_user_id,
            query=test_query,
            top_k=3
        )
        print(f"✅ 임베딩 검색 성공! {len(results)}개 결과 발견")
        for i, result in enumerate(results, 1):
            print(f"  {i}. ID: {result['id']}, Score: {result['score']:.4f}")
            if result.get('metadata'):
                metadata = result['metadata']
                print(f"     파일명: {metadata.get('original_filename', 'N/A')}")
                print(f"     리포지토리: {metadata.get('github_user_repo', 'N/A')}")
                print(f"     브랜치: {metadata.get('branch_name', 'N/A')}")
                print(f"     텍스트 미리보기: {metadata.get('text', 'N/A')[:100]}...")
    
    except Exception as e:
        print(f"❌ 임베딩 검색 실패: {e}")
        return False
    
    # 필터링 테스트
    try:
        print(f"\n🔍 필터링 검색 테스트 (repo_path: '{test_repo}', branch: '{test_branch}')")
        results = _search_tutorials(
            user_id=test_user_id,
            query=test_query,
            repo_path=test_repo,
            branch=test_branch,
            top_k=3
        )
        print(f"✅ 필터링 검색 성공! {len(results)}개 결과 발견")
        for i, result in enumerate(results, 1):
            print(f"  {i}. ID: {result['id']}, Score: {result['score']:.4f}")
            if result.get('metadata'):
                metadata = result['metadata']
                print(f"     파일명: {metadata.get('original_filename', 'N/A')}")
                print(f"     리포지토리: {metadata.get('github_user_repo', 'N/A')}")
                print(f"     브랜치: {metadata.get('branch_name', 'N/A')}")
                print(f"     텍스트 미리보기: {metadata.get('text', 'N/A')[:100]}...")
    
    except Exception as e:
        print(f"❌ 필터링 검색 실패: {e}")
        return False
    
    print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
    return True

if __name__ == "__main__":
    print("🚀 Pinecone 검색 도구 테스트 시작")
    print("=" * 50)
    
    success = test_pinecone_search()
    
    if success:
        print("\n🎉 테스트 성공!")
        sys.exit(0)
    else:
        print("\n💥 테스트 실패!")
        sys.exit(1) 