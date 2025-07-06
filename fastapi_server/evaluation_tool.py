"""
LangGraph 에이전트 정량적 평가 툴
graph.py의 에이전트들을 langsmith를 이용하여 평가합니다.
"""

import os
import sys
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import uuid

# Windows 환경에서 asyncio 이벤트 루프 정책 설정
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 환경 변수 로드 (먼저 실행)
from dotenv import load_dotenv
load_dotenv()

# 환경 변수 확인 및 경고
required_env_vars = ["LANGSMITH_API_KEY", "OPENAI_API_KEY", "DB_URI"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    print(f"❌ 다음 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
    print("💡 .env 파일에 다음 변수들을 설정해주세요:")
    for var in missing_vars:
        print(f"   {var}=your_value_here")
    exit(1)

# 필수 환경 변수 기본값 설정 (실제 값이 없을 때만)
if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = "dummy-key"

# LangSmith 관련 imports
from langsmith import Client, aevaluate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# 로컬 그래프 imports
try:
    from agent.graph import get_swarm_graph
except ImportError as e:
    print(f"⚠️ 에이전트 모듈 가져오기 실패: {e}")
    print("환경 변수를 설정하고 다시 시도해주세요.")
    exit(1)

from evaluation_config import EvaluationConfig, AGENT_TEST_CASES, EVALUATION_METRICS, REPORT_TEMPLATE

@dataclass
class EvaluationResult:
    """평가 결과를 저장하는 데이터 클래스"""
    agent_name: str
    test_case: str
    score: float
    details: Dict[str, Any]
    timestamp: str

class LangGraphAgentEvaluator:
    """LangGraph 에이전트 평가 클래스"""
    
    def __init__(self):
        # 실제 API 키 확인
        langsmith_key = os.getenv("LANGSMITH_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        db_uri = os.getenv("DB_URI")
        
        if not langsmith_key or langsmith_key == "dummy-key":
            raise ValueError("❌ LANGSMITH_API_KEY 환경변수가 설정되지 않았습니다.")
        
        if not openai_key or openai_key == "dummy-key":
            raise ValueError("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            
        if not db_uri or db_uri == "postgresql://dummy:dummy@localhost:5432/dummy":
            raise ValueError("❌ DB_URI 환경변수가 설정되지 않았습니다.")
        
        self.ls_client = Client()
        self.judge_llm = ChatOpenAI(model="gpt-4o")
        self.results: List[EvaluationResult] = []
    
    async def setup_datasets(self):
        """평가용 데이터셋을 설정합니다."""
        print("📊 데이터셋 생성 시작...")
        
        # 기존 데이터셋 확인 및 정리 헬퍼 함수
        def get_or_create_dataset(name: str, description: str):
            """기존 데이터셋을 확인하고 없으면 생성합니다."""
            try:
                # 기존 데이터셋 검색
                existing_datasets = list(self.ls_client.list_datasets(dataset_name=name))
                
                if existing_datasets:
                    print(f"  ⚠️ 기존 데이터셋 '{name}' 발견 - 삭제 후 재생성")
                    # 기존 데이터셋 삭제
                    for dataset in existing_datasets:
                        self.ls_client.delete_dataset(dataset_id=dataset.id)
                        print(f"  🗑️ 기존 데이터셋 삭제: {dataset.id}")
                
                # 새 데이터셋 생성
                dataset = self.ls_client.create_dataset(
                    dataset_name=name,
                    description=description
                )
                print(f"  ✅ 새 데이터셋 생성: {name}")
                return dataset
                
            except Exception as e:
                print(f"  ❌ 데이터셋 처리 중 오류: {e}")
                raise
        
        # 1. 문서 검색 에이전트 데이터셋
        doc_search_questions = [
            "출장 정책에 대해 알려주세요",
            "신제품 개발 회의록을 찾아주세요",
            "AI Train 사용법을 알려주세요",
            "데이터베이스 기술 문서를 찾아주세요"
        ]
        
        doc_search_expected = [
            "출장 정책 관련 정보가 포함되어야 함",
            "신제품 개발 회의록 내용이 포함되어야 함",
            "AI Train 사용법 정보가 포함되어야 함", 
            "데이터베이스 기술 문서 정보가 포함되어야 함"
        ]
        
        try:
            # 데이터셋 생성 (기존 데이터셋 확인 및 정리)
            print("  📄 문서 검색 에이전트 데이터셋 생성 중...")
            self.doc_search_dataset = get_or_create_dataset(
                "doc_search_agent_eval",
                "문서 검색 에이전트 평가 데이터셋"
            )
            
            # 예제 데이터 추가
            examples = []
            for q, e in zip(doc_search_questions, doc_search_expected):
                examples.append({
                    "inputs": {"question": q},
                    "outputs": {"expected": e}
                })
            
            for i, example in enumerate(examples):
                print(f"    예제 {i+1}/{len(examples)} 추가 중...")
                self.ls_client.create_example(
                    dataset_id=self.doc_search_dataset.id,
                    inputs=example["inputs"],
                    outputs=example["outputs"]
                )
            print("  ✅ 문서 검색 데이터셋 생성 완료")
            
        except Exception as e:
            print(f"  ❌ 문서 검색 데이터셋 생성 실패: {e}")
            raise
        
        # 2. 분석 에이전트 데이터셋  
        analyst_questions = [
            "고객 테이블의 성별 분포를 차트로 보여주세요",
            "최근 뉴스 키워드 데이터를 분석해주세요",
            "고객 데이터에서 연령대별 분포를 조회해주세요",
            "고객 수를 카운트해주세요"
        ]
        
        analyst_expected = [
            "성별 분포 차트가 생성되어야 함",
            "뉴스 키워드 분석 결과가 포함되어야 함", 
            "연령대별 분포 정보가 포함되어야 함",
            "고객 수 카운트 결과가 포함되어야 함"
        ]
        
        print("  📊 분석 에이전트 데이터셋 생성 중...")
        self.analyst_dataset = get_or_create_dataset(
            "analyst_agent_eval",
            "분석 에이전트 평가 데이터셋"
        )
        
        examples = []
        for q, e in zip(analyst_questions, analyst_expected):
            examples.append({
                "inputs": {"question": q},
                "outputs": {"expected": e}
            })
        
        for i, example in enumerate(examples):
            print(f"    예제 {i+1}/{len(examples)} 추가 중...")
            self.ls_client.create_example(
                dataset_id=self.analyst_dataset.id,
                inputs=example["inputs"],
                outputs=example["outputs"]
            )
        print("  ✅ 분석 에이전트 데이터셋 생성 완료")
        
        # 3. 예측 에이전트 데이터셋
        predict_questions = [
            {
                "question": "다음 고객 데이터로 이탈 예측을 해주세요",
                "csv_data": "tenure,monthly_charges,total_charges,contract,payment_method\n12,29.85,358.2,Month-to-month,Electronic check"
            }
        ]
        
        predict_expected = [
            "이탈 확률이 포함되어야 함"
        ]
        
        print("  🔮 예측 에이전트 데이터셋 생성 중...")
        self.predict_dataset = get_or_create_dataset(
            "predict_agent_eval",
            "예측 에이전트 평가 데이터셋"
        )
        
        examples = []
        for q, e in zip(predict_questions, predict_expected):
            examples.append({
                "inputs": q,  # 이미 딕셔너리 형태
                "outputs": {"expected": e}
            })
        
        for i, example in enumerate(examples):
            print(f"    예제 {i+1}/{len(examples)} 추가 중...")
            self.ls_client.create_example(
                dataset_id=self.predict_dataset.id,
                inputs=example["inputs"],
                outputs=example["outputs"]
            )
        print("  ✅ 예측 에이전트 데이터셋 생성 완료")
        
        # 4. 코딩 에이전트 데이터셋
        coding_questions = [
            "Python에서 리스트 컴프리헨션 사용법을 알려주세요",
            "FastAPI로 간단한 REST API를 만드는 방법을 설명해주세요", 
            "LangGraph의 create_react_agent 사용법을 알려주세요",
            "GitHub API를 사용하는 방법을 설명해주세요"
        ]
        
        coding_expected = [
            "리스트 컴프리헨션 사용법이 포함되어야 함",
            "FastAPI REST API 생성 방법이 포함되어야 함",
            "create_react_agent 사용법이 포함되어야 함", 
            "GitHub API 사용법이 포함되어야 함"
        ]
        
        print("  💻 코딩 에이전트 데이터셋 생성 중...")
        self.coding_dataset = get_or_create_dataset(
            "coding_agent_eval",
            "코딩 에이전트 평가 데이터셋"
        )
        
        examples = []
        for q, e in zip(coding_questions, coding_expected):
            examples.append({
                "inputs": {"question": q},
                "outputs": {"expected": e}
            })
        
        for i, example in enumerate(examples):
            print(f"    예제 {i+1}/{len(examples)} 추가 중...")
            self.ls_client.create_example(
                dataset_id=self.coding_dataset.id,
                inputs=example["inputs"],
                outputs=example["outputs"]
            )
        print("  ✅ 코딩 에이전트 데이터셋 생성 완료")
        
        print("✅ 모든 데이터셋이 성공적으로 생성되었습니다.")
    
    def question_to_messages(self, inputs: Dict) -> Dict:
        """질문을 메시지 형태로 변환합니다."""
        question = inputs.get('question', '')
        csv_data = inputs.get('csv_data', '')
        
        if csv_data:
            content = f"{question}\n\nCSV 데이터:\n{csv_data}"
        else:
            content = question
        
        return {"messages": [("user", content)]}
    
    async def relevance_evaluator(self, outputs: Dict, reference_outputs: Dict) -> bool:
        """응답의 관련성을 평가합니다."""
        instructions = (
            "주어진 실제 답변이 기대되는 내용을 포함하고 있는지 평가하세요. "
            "실제 답변이 기대 내용과 관련된 정보를 포함하고 있으면 'RELEVANT', "
            "그렇지 않으면 'IRRELEVANT'로 응답하세요. 다른 내용은 포함하지 마세요."
        )
        
        # 그래프 출력에서 마지막 메시지 추출
        if "messages" in outputs and outputs["messages"]:
            actual_answer = outputs["messages"][-1].content if hasattr(outputs["messages"][-1], 'content') else str(outputs["messages"][-1])
        else:
            actual_answer = str(outputs)
            
        expected_answer = reference_outputs.get("expected", "")
        
        user_msg = (
            f"실제 답변: {actual_answer}\n\n"
            f"기대 내용: {expected_answer}"
        )
        
        response = await self.judge_llm.ainvoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_msg}
        ])
        
        return response.content.upper() == "RELEVANT"
    
    async def completeness_evaluator(self, outputs: Dict, reference_outputs: Dict) -> bool:
        """응답의 완성도를 평가합니다.""" 
        instructions = (
            "주어진 실제 답변이 질문에 대해 완전하고 유용한 정보를 제공하는지 평가하세요. "
            "답변이 충분히 상세하고 도움이 되면 'COMPLETE', "
            "그렇지 않으면 'INCOMPLETE'로 응답하세요."
        )
        
        if "messages" in outputs and outputs["messages"]:
            actual_answer = outputs["messages"][-1].content if hasattr(outputs["messages"][-1], 'content') else str(outputs["messages"][-1])
        else:
            actual_answer = str(outputs)
        
        response = await self.judge_llm.ainvoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"답변: {actual_answer}"}
        ])
        
        return response.content.upper() == "COMPLETE"
    
    async def agent_handoff_evaluator(self, outputs: Dict, reference_outputs: Dict) -> bool:
        """에이전트 핸드오프가 적절한지 평가합니다."""
        # 메시지 체인에서 에이전트 전환이 있었는지 확인
        if "messages" in outputs and len(outputs["messages"]) > 1:
            # 여러 메시지가 있으면 에이전트 간 협업이 있었다고 가정
            return True
        return False
    
    async def evaluate_doc_search_agent(self):
        """문서 검색 에이전트를 평가합니다."""
        print("📄 문서 검색 에이전트 평가 시작...")
        
        # 체크포인터 없이 swarm 그래프 사용 (평가 목적)
        try:
            from langgraph_swarm import create_swarm
            from agent.graph import doc_search_assistant, analyst_assistant, predict_assistant, coding_assistant
            
            # 체크포인터 없이 컴파일
            graph = create_swarm(
                agents=[doc_search_assistant, analyst_assistant, predict_assistant, coding_assistant],
                default_active_agent="doc_search_assistant"
            ).compile()  # 체크포인터 없이 컴파일
            
            target = self.question_to_messages | graph
            
        except Exception as e:
            print(f"⚠️ 체크포인터 없는 그래프 생성 실패, 단일 에이전트 사용: {e}")
            # 단일 에이전트 대체 방안
            from agent.graph import doc_search_assistant
            target = self.question_to_messages | doc_search_assistant
        
        experiment_results = await aevaluate(
            target,
            data=self.doc_search_dataset.name,
            evaluators=[self.relevance_evaluator, self.completeness_evaluator],
            max_concurrency=1,  # 동시성을 1로 줄여서 안정성 확보
            experiment_prefix="doc_search_agent",
        )
        
        print(f"✅ 문서 검색 에이전트 평가 완료: {experiment_results}")
        return experiment_results
    
    async def evaluate_analyst_agent(self):
        """분석 에이전트를 평가합니다."""
        print("📊 분석 에이전트 평가 시작...")
        
        # 체크포인터 없이 swarm 그래프 사용 (평가 목적)
        try:
            from langgraph_swarm import create_swarm
            from agent.graph import doc_search_assistant, analyst_assistant, predict_assistant, coding_assistant
            
            # 체크포인터 없이 컴파일
            graph = create_swarm(
                agents=[doc_search_assistant, analyst_assistant, predict_assistant, coding_assistant],
                default_active_agent="doc_search_assistant"
            ).compile()  # 체크포인터 없이 컴파일
            
            target = self.question_to_messages | graph
            
        except Exception as e:
            print(f"⚠️ 체크포인터 없는 그래프 생성 실패, 단일 에이전트 사용: {e}")
            # 단일 에이전트 대체 방안
            from agent.graph import analyst_assistant
            target = self.question_to_messages | analyst_assistant
        
        experiment_results = await aevaluate(
            target,
            data=self.analyst_dataset.name,
            evaluators=[self.relevance_evaluator, self.completeness_evaluator],
            max_concurrency=1,  # 동시성을 1로 줄여서 안정성 확보
            experiment_prefix="analyst_agent",
        )
        
        print(f"✅ 분석 에이전트 평가 완료: {experiment_results}")
        return experiment_results
    
    async def evaluate_predict_agent(self):
        """예측 에이전트를 평가합니다."""
        print("🔮 예측 에이전트 평가 시작...")
        
        # 체크포인터 없이 swarm 그래프 사용 (평가 목적)
        try:
            from langgraph_swarm import create_swarm
            from agent.graph import doc_search_assistant, analyst_assistant, predict_assistant, coding_assistant
            
            # 체크포인터 없이 컴파일
            graph = create_swarm(
                agents=[doc_search_assistant, analyst_assistant, predict_assistant, coding_assistant],
                default_active_agent="doc_search_assistant"
            ).compile()  # 체크포인터 없이 컴파일
            
            target = self.question_to_messages | graph
            
        except Exception as e:
            print(f"⚠️ 체크포인터 없는 그래프 생성 실패, 단일 에이전트 사용: {e}")
            # 단일 에이전트 대체 방안
            from agent.graph import predict_assistant
            target = self.question_to_messages | predict_assistant
        
        experiment_results = await aevaluate(
            target,
            data=self.predict_dataset.name,
            evaluators=[self.relevance_evaluator, self.completeness_evaluator],
            max_concurrency=1,  # 예측은 동시성을 낮춤
            experiment_prefix="predict_agent",
        )
        
        print(f"✅ 예측 에이전트 평가 완료: {experiment_results}")
        return experiment_results
    
    async def evaluate_coding_agent(self):
        """코딩 에이전트를 평가합니다."""
        print("💻 코딩 에이전트 평가 시작...")
        
        # 체크포인터 없이 swarm 그래프 사용 (평가 목적)
        try:
            from langgraph_swarm import create_swarm
            from agent.graph import doc_search_assistant, analyst_assistant, predict_assistant, coding_assistant
            
            # 체크포인터 없이 컴파일
            graph = create_swarm(
                agents=[doc_search_assistant, analyst_assistant, predict_assistant, coding_assistant],
                default_active_agent="doc_search_assistant"
            ).compile()  # 체크포인터 없이 컴파일
            
            target = self.question_to_messages | graph
            
        except Exception as e:
            print(f"⚠️ 체크포인터 없는 그래프 생성 실패, 단일 에이전트 사용: {e}")
            # 단일 에이전트 대체 방안
            from agent.graph import coding_assistant
            target = self.question_to_messages | coding_assistant
        
        experiment_results = await aevaluate(
            target,
            data=self.coding_dataset.name,
            evaluators=[self.relevance_evaluator, self.completeness_evaluator],
            max_concurrency=1,  # 동시성을 1로 줄여서 안정성 확보
            experiment_prefix="coding_agent",
        )
        
        print(f"✅ 코딩 에이전트 평가 완료: {experiment_results}")
        return experiment_results
    
    async def run_comprehensive_evaluation(self):
        """모든 에이전트에 대한 종합 평가를 실행합니다."""
        print("🚀 LangGraph 에이전트 종합 평가 시작...")
        print("=" * 60)
        
        try:
            # 데이터셋 설정
            await self.setup_datasets()
            
            # 각 에이전트 평가 실행
            results = {}
            
            # 문서 검색 에이전트 평가
            results["doc_search"] = await self.evaluate_doc_search_agent()
            
            # 분석 에이전트 평가  
            results["analyst"] = await self.evaluate_analyst_agent()
            
            # 예측 에이전트 평가
            results["predict"] = await self.evaluate_predict_agent()
            
            # 코딩 에이전트 평가
            results["coding"] = await self.evaluate_coding_agent()
            
            # 결과 요약
            print("\n" + "=" * 60)
            print("📋 평가 결과 요약")
            print("=" * 60)
            
            for agent_name, result in results.items():
                if result and hasattr(result, 'aggregate_score'):
                    print(f"{agent_name}_agent: {result.aggregate_score:.2f}")
                else:
                    print(f"{agent_name}_agent: 평가 완료")
            
            # 결과를 JSON 파일로 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"evaluation_results_{timestamp}.json"
            
            serializable_results = {}
            for agent_name, result in results.items():
                if result:
                    serializable_results[agent_name] = {
                        "timestamp": timestamp,
                        "status": "completed",
                        "details": str(result)
                    }
                
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            print(f"📄 평가 결과가 {result_file}에 저장되었습니다.")
            return results
            
        except Exception as e:
            print(f"❌ 평가 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

async def main():
    """메인 실행 함수"""
    evaluator = LangGraphAgentEvaluator()
    await evaluator.run_comprehensive_evaluation()

if __name__ == "__main__":
    asyncio.run(main()) 