"""
LangGraph 에이전트 정량적 평가 툴
graph.py의 에이전트들을 langsmith를 이용하여 평가합니다.
"""

import asyncio
import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# LangSmith 관련 imports
from langsmith import Client, aevaluate
from langchain.chat_models import init_chat_model

# 로컬 그래프 imports
from agent.graph import get_swarm_graph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

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
        self.ls_client = Client()
        self.judge_llm = init_chat_model("gpt-4o")
        self.results: List[EvaluationResult] = []
        
        # 데이터베이스 URI 확인
        if not os.getenv("DB_URI"):
            raise ValueError("DB_URI 환경변수가 설정되지 않았습니다.")
    
    async def setup_datasets(self):
        """평가용 데이터셋들을 생성합니다."""
        
        # 1. 문서 검색 에이전트 데이터셋
        doc_search_questions = [
            "회사의 출장 정책에 대해 알려주세요",
            "신제품 개발 관련 회의록을 찾아주세요", 
            "AI Train 서비스 사용법을 설명해주세요",
            "데이터베이스 관련 기술 문서를 찾아주세요"
        ]
        
        doc_search_expected = [
            "출장 정책 관련 정보가 포함되어야 함",
            "신제품 개발 회의록 내용이 포함되어야 함",
            "AI Train 사용법 정보가 포함되어야 함", 
            "데이터베이스 기술 문서 정보가 포함되어야 함"
        ]
        
        self.doc_search_dataset = self.ls_client.create_dataset(
            "doc_search_agent_eval",
            inputs=[{"question": q} for q in doc_search_questions],
            outputs=[{"expected": e} for e in doc_search_expected],
        )
        
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
        
        self.analyst_dataset = self.ls_client.create_dataset(
            "analyst_agent_eval", 
            inputs=[{"question": q} for q in analyst_questions],
            outputs=[{"expected": e} for e in analyst_expected],
        )
        
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
        
        self.predict_dataset = self.ls_client.create_dataset(
            "predict_agent_eval",
            inputs=predict_questions,
            outputs=[{"expected": e} for e in predict_expected],
        )
        
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
        
        self.coding_dataset = self.ls_client.create_dataset(
            "coding_agent_eval",
            inputs=[{"question": q} for q in coding_questions], 
            outputs=[{"expected": e} for e in coding_expected],
        )
        
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
        
        async with AsyncPostgresSaver.from_conn_string(os.environ["DB_URI"]) as checkpointer:
            graph = get_swarm_graph(checkpointer)
            
            # 문서 검색 에이전트가 기본 활성 에이전트이므로 직접 사용
            target = self.question_to_messages | graph
            
            experiment_results = await aevaluate(
                target,
                data=self.doc_search_dataset,
                evaluators=[self.relevance_evaluator, self.completeness_evaluator],
                max_concurrency=2,
                experiment_prefix="doc_search_agent",
            )
            
            print(f"✅ 문서 검색 에이전트 평가 완료: {experiment_results}")
            return experiment_results
    
    async def evaluate_analyst_agent(self):
        """분석 에이전트를 평가합니다."""
        print("📊 분석 에이전트 평가 시작...")
        
        async with AsyncPostgresSaver.from_conn_string(os.environ["DB_URI"]) as checkpointer:
            graph = get_swarm_graph(checkpointer)
            
            # 분석 관련 질문으로 시작하여 자동 핸드오프 유도
            target = self.question_to_messages | graph
            
            experiment_results = await aevaluate(
                target,
                data=self.analyst_dataset,
                evaluators=[self.relevance_evaluator, self.completeness_evaluator, self.agent_handoff_evaluator],
                max_concurrency=2,
                experiment_prefix="analyst_agent",
            )
            
            print(f"✅ 분석 에이전트 평가 완료: {experiment_results}")
            return experiment_results
    
    async def evaluate_predict_agent(self):
        """예측 에이전트를 평가합니다."""
        print("🔮 예측 에이전트 평가 시작...")
        
        async with AsyncPostgresSaver.from_conn_string(os.environ["DB_URI"]) as checkpointer:
            graph = get_swarm_graph(checkpointer)
            
            target = self.question_to_messages | graph
            
            experiment_results = await aevaluate(
                target,
                data=self.predict_dataset,
                evaluators=[self.relevance_evaluator, self.completeness_evaluator],
                max_concurrency=1,  # 예측은 동시성을 낮춤
                experiment_prefix="predict_agent",
            )
            
            print(f"✅ 예측 에이전트 평가 완료: {experiment_results}")
            return experiment_results
    
    async def evaluate_coding_agent(self):
        """코딩 에이전트를 평가합니다."""
        print("💻 코딩 에이전트 평가 시작...")
        
        async with AsyncPostgresSaver.from_conn_string(os.environ["DB_URI"]) as checkpointer:
            graph = get_swarm_graph(checkpointer)
            
            target = self.question_to_messages | graph
            
            experiment_results = await aevaluate(
                target,
                data=self.coding_dataset,
                evaluators=[self.relevance_evaluator, self.completeness_evaluator],
                max_concurrency=2,
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