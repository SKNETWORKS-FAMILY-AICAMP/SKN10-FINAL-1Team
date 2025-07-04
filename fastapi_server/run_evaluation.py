#!/usr/bin/env python3
"""
LangGraph 에이전트 평가 실행 스크립트
간단한 명령어로 평가를 실행할 수 있습니다.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from evaluation_tool import LangGraphAgentEvaluator
from evaluation_config import get_evaluation_config

def parse_arguments():
    """명령행 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="LangGraph 에이전트 정량적 평가 툴",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run_evaluation.py --all                    # 모든 에이전트 평가
  python run_evaluation.py --agent doc_search       # 문서 검색 에이전트만 평가
  python run_evaluation.py --agent analyst          # 분석 에이전트만 평가
  python run_evaluation.py --quick                  # 빠른 평가 (테스트 케이스 축소)
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", 
        action="store_true",
        help="모든 에이전트를 평가합니다"
    )
    group.add_argument(
        "--agent",
        choices=["doc_search", "analyst", "predict", "coding"],
        help="특정 에이전트만 평가합니다"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="빠른 평가 모드 (테스트 케이스 축소)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="결과 파일 이름 (기본값: evaluation_results)"
    )
    
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="동시 실행 수 (기본값: 2)"
    )
    
    return parser.parse_args()

async def run_single_agent_evaluation(evaluator, agent_name):
    """단일 에이전트 평가를 실행합니다."""
    print(f"🎯 {agent_name} 에이전트 평가 시작...")
    
    if agent_name == "doc_search":
        result = await evaluator.evaluate_doc_search_agent()
    elif agent_name == "analyst":
        result = await evaluator.evaluate_analyst_agent()
    elif agent_name == "predict":
        result = await evaluator.evaluate_predict_agent()
    elif agent_name == "coding":
        result = await evaluator.evaluate_coding_agent()
    else:
        raise ValueError(f"알 수 없는 에이전트: {agent_name}")
    
    print(f"✅ {agent_name} 에이전트 평가 완료!")
    return {agent_name: result}

async def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    print("🚀 LangGraph 에이전트 평가 시작")
    print("=" * 60)
    
    try:
        # 설정 확인
        config = get_evaluation_config()
        print(f"📊 LangSmith 프로젝트: {config.langsmith_project}")
        print(f"🤖 평가 모델: {config.judge_model}")
        print(f"⚡ 동시 실행 수: {args.concurrency}")
        print("-" * 60)
        
        # 평가자 초기화
        evaluator = LangGraphAgentEvaluator()
        
        # 데이터셋 설정
        await evaluator.setup_datasets()
        
        results = {}
        
        if args.all:
            # 모든 에이전트 평가
            print("📋 모든 에이전트를 평가합니다...")
            results = await evaluator.run_comprehensive_evaluation()
        
        elif args.agent:
            # 특정 에이전트만 평가
            results = await run_single_agent_evaluation(evaluator, args.agent)
        
        # 결과 출력
        print("\n" + "=" * 60)
        print("📋 평가 결과 요약")
        print("=" * 60)
        
        for agent_name, result in results.items():
            if result:
                print(f"✅ {agent_name}_agent: 평가 완료")
            else:
                print(f"❌ {agent_name}_agent: 평가 실패")
        
        print(f"\n📄 자세한 결과는 생성된 JSON 파일을 확인하세요.")
        
    except Exception as e:
        print(f"❌ 평가 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Windows에서 이벤트 루프 정책 설정 (필요시)
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main()) 