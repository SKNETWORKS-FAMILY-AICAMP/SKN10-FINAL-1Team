#!/usr/bin/env python3
"""
간단한 LangGraph 에이전트 평가 테스트
"""

import asyncio
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client

# 환경 변수 로드
load_dotenv("../.env")

def test_langsmith():
    """LangSmith 연결 테스트"""
    try:
        # API 키 확인
        api_key = os.getenv("LANGSMITH_API_KEY")
        if not api_key:
            print("❌ LANGSMITH_API_KEY 환경변수가 설정되지 않았습니다.")
            return False
        
        # LangSmith 클라이언트 생성 및 연결 테스트
        client = Client(api_key=api_key)
        
        # 간단한 연결 테스트
        try:
            # 프로젝트 목록 가져오기 (연결 테스트용)
            list(client.list_projects(limit=1))
            print("✅ LangSmith 연결 성공")
            return True
        except Exception as e:
            print(f"❌ LangSmith 연결 실패: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ LangSmith 모듈을 찾을 수 없습니다: {e}")
        return False
    except Exception as e:
        print(f"❌ LangSmith 테스트 중 오류 발생: {e}")
        return False

def test_openai():
    """OpenAI 연결 테스트"""
    try:
        from langchain_openai import ChatOpenAI
        
        # API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            return False
        
        # OpenAI 클라이언트 생성 및 연결 테스트
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key,
            max_tokens=10
        )
        
        # 간단한 응답 테스트
        response = llm.invoke("Hello")
        print("✅ OpenAI 연결 성공")
        return True
        
    except ImportError as e:
        print(f"❌ OpenAI 모듈을 찾을 수 없습니다: {e}")
        return False
    except Exception as e:
        print(f"❌ OpenAI 테스트 중 오류 발생: {e}")
        return False

def test_agent_import():
    """에이전트 모듈 import 테스트 (별도 프로세스)"""
    try:
        # 별도 프로세스에서 agent import 테스트
        test_code = '''# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from agent.graph import get_swarm_graph
    print("SUCCESS: 에이전트 모듈 import 성공")
    exit(0)
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)
'''
        
        # 임시 파일 생성 (UTF-8 인코딩 명시)
        temp_file = Path("temp_agent_test.py")
        temp_file.write_text(test_code, encoding='utf-8')
        
        try:
            # 별도 프로세스에서 실행
            result = subprocess.run([
                sys.executable, str(temp_file)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ 에이전트 모듈 import 성공")
                return True
            else:
                print(f"❌ 에이전트 모듈 import 실패: {result.stderr}")
                return False
                
        finally:
            # 임시 파일 정리
            if temp_file.exists():
                temp_file.unlink()
        
    except subprocess.TimeoutExpired:
        print("❌ 에이전트 모듈 import 테스트 시간 초과")
        return False
    except Exception as e:
        print(f"❌ 에이전트 모듈 테스트 중 오류 발생: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("🔍 환경 설정 테스트를 시작합니다...")
    print("=" * 60)
    
    # 현재 디렉토리 정보
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print(f"Python 경로: {sys.executable}")
    print()
    
    # 환경변수 확인
    print("📋 환경변수 확인:")
    env_vars = ["LANGSMITH_API_KEY", "OPENAI_API_KEY", "DB_URI", "PINECONE_API_KEY"]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # 보안을 위해 일부만 표시
            masked_value = f"{value[:8]}...{value[-8:]}" if len(value) > 16 else "***"
            print(f"  {var}: {masked_value}")
        else:
            print(f"  {var}: ❌ 설정되지 않음")
    print()
    
    # 각 컴포넌트 테스트
    results = {}
    
    print("🧪 개별 컴포넌트 테스트:")
    print("-" * 40)
    
    # LangSmith 테스트
    print("1. LangSmith 연결 테스트...")
    results['langsmith'] = test_langsmith()
    print()
    
    # OpenAI 테스트
    print("2. OpenAI 연결 테스트...")
    results['openai'] = test_openai()
    print()
    
    # 에이전트 모듈 테스트
    print("3. 에이전트 모듈 import 테스트...")
    results['agent_import'] = test_agent_import()
    print()
    
    # 결과 요약
    print("=" * 60)
    print("📋 테스트 결과 요약")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"{test_name:<14} : {status}")
    
    # 전체 결과
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 모든 테스트가 통과했습니다! 평가 툴을 실행할 수 있습니다.")
        print("\n다음 명령어로 평가를 실행하세요:")
        print("  python run_evaluation.py --all")
    else:
        print("\n⚠️ 일부 테스트가 실패했습니다. 환경 설정을 확인해주세요.")
        print("\n실패한 테스트:")
        for test_name, passed in results.items():
            if not passed:
                print(f"  - {test_name}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 