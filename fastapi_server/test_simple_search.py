#!/usr/bin/env python3
"""
간단한 Pinecone 검색 테스트
"""

import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.coding_agent_tools import _search_tutorials

def test_simple_search():
    """간단한 검색 테스트를 수행합니다."""
    
    print("🔍 간단한 Pinecone 검색 테스트")
    print("=" * 40)
    
    # 실제 사용자 ID와 쿼리 사용
    user_id = "be87b5f4-1096-46fd-9992-5703b9ef1229"
    query = "llm 멀티에이전트 구현 문서"
    
    print(f"사용자 ID: {user_id}")
    print(f"검색 쿼리: {query}")
    print()
    
    try:
        results = _search_tutorials(
            user_id=user_id,
            query=query,
            top_k=5
        )
        
        print(f"✅ 검색 성공! {len(results)}개 결과 발견")
        print()
        
        for i, result in enumerate(results, 1):
            print(f"결과 {i}:")
            print(f"  ID: {result['id']}")
            print(f"  점수: {result['score']:.4f}")
            
            if result.get('metadata'):
                metadata = result['metadata']
                print(f"  파일명: {metadata.get('original_filename', 'N/A')}")
                print(f"  리포지토리: {metadata.get('github_user_repo', 'N/A')}")
                print(f"  브랜치: {metadata.get('branch_name', 'N/A')}")
                
                # 텍스트 미리보기
                text = metadata.get('text', '')
                if text:
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"  텍스트 미리보기: {preview}")
            
            print()
    
    except Exception as e:
        print(f"❌ 검색 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_search() 