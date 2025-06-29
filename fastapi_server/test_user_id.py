#!/usr/bin/env python3
"""
사용자 ID 설정과 Pinecone 검색 도구 테스트 스크립트
"""

import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.coding_agent_tools import set_current_user_id, get_current_user_id, _search_tutorials

def test_user_id_setting():
    """사용자 ID 설정 테스트"""
    print("=== 사용자 ID 설정 테스트 ===")
    
    # 테스트용 사용자 ID
    test_user_id = "be87b5f4-1096-46fd-9992-5703b9ef1229"
    
    # 사용자 ID 설정
    set_current_user_id(test_user_id)
    print(f"설정된 사용자 ID: {test_user_id}")
    
    # 사용자 ID 확인
    current_user_id = get_current_user_id()
    print(f"현재 사용자 ID: {current_user_id}")
    
    if current_user_id == test_user_id:
        print("✅ 사용자 ID 설정 성공")
    else:
        print("❌ 사용자 ID 설정 실패")
    
    return current_user_id == test_user_id

def test_pinecone_search():
    """Pinecone 검색 테스트"""
    print("\n=== Pinecone 검색 테스트 ===")
    
    try:
        # 간단한 검색 쿼리로 테스트
        results = _search_tutorials(
            query="test query",
            top_k=3
        )
        
        print(f"검색 결과 수: {len(results)}")
        for i, result in enumerate(results):
            print(f"결과 {i+1}: {result}")
        
        print("✅ Pinecone 검색 성공")
        return True
        
    except Exception as e:
        print(f"❌ Pinecone 검색 실패: {e}")
        return False

def test_without_user_id():
    """사용자 ID 없이 검색 테스트 (오류 발생 확인)"""
    print("\n=== 사용자 ID 없이 검색 테스트 ===")
    
    # 사용자 ID 초기화
    set_current_user_id(None)
    
    try:
        results = _search_tutorials(
            query="test query",
            top_k=3
        )
        print("❌ 예상된 오류가 발생하지 않음")
        return False
        
    except RuntimeError as e:
        if "사용자 ID가 설정되지 않았습니다" in str(e):
            print("✅ 예상된 오류 발생: 사용자 ID가 설정되지 않음")
            return True
        else:
            print(f"❌ 예상과 다른 오류: {e}")
            return False
    except Exception as e:
        print(f"❌ 예상과 다른 오류: {e}")
        return False

if __name__ == "__main__":
    print("사용자 ID 및 Pinecone 검색 도구 테스트 시작\n")
    
    # 테스트 실행
    test1_passed = test_user_id_setting()
    test2_passed = test_pinecone_search()
    test3_passed = test_without_user_id()
    
    print(f"\n=== 테스트 결과 ===")
    print(f"사용자 ID 설정 테스트: {'✅ 통과' if test1_passed else '❌ 실패'}")
    print(f"Pinecone 검색 테스트: {'✅ 통과' if test2_passed else '❌ 실패'}")
    print(f"사용자 ID 없이 검색 테스트: {'✅ 통과' if test3_passed else '❌ 실패'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 모든 테스트 통과!")
    else:
        print("\n⚠️ 일부 테스트 실패") 