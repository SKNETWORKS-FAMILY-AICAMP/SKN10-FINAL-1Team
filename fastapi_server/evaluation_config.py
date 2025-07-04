"""
LangGraph 에이전트 평가 설정 파일
평가 관련 설정과 데이터셋 구성을 관리합니다.
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class EvaluationConfig:
    """평가 설정 클래스"""
    
    # LangSmith 설정
    langsmith_api_key: str = ""
    langsmith_project: str = "langgraph-agent-evaluation"
    
    # 모델 설정
    judge_model: str = "gpt-4o"
    
    # 평가 설정
    max_concurrency: int = 2
    timeout_seconds: int = 300
    
    # 데이터베이스 설정
    db_uri: str = ""
    
    def __post_init__(self):
        """환경 변수에서 설정 값들을 로드합니다."""
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY", self.langsmith_api_key)
        self.db_uri = os.getenv("DB_URI", self.db_uri)
        
        # 필수 환경 변수 확인
        if not self.langsmith_api_key:
            raise ValueError("LANGSMITH_API_KEY 환경변수가 설정되지 않았습니다.")
        if not self.db_uri:
            raise ValueError("DB_URI 환경변수가 설정되지 않았습니다.")

# 각 에이전트별 테스트 케이스 정의
AGENT_TEST_CASES = {
    "doc_search": {
        "questions": [
            "회사의 출장 정책에 대해 알려주세요",
            "신제품 개발 관련 회의록을 찾아주세요", 
            "AI Train 서비스 사용법을 설명해주세요",
            "데이터베이스 관련 기술 문서를 찾아주세요",
            "복지 혜택 관련 정책을 조회해주세요"
        ],
        "expected_keywords": [
            ["출장", "정책", "규정"],
            ["신제품", "개발", "회의록"],
            ["AI Train", "사용법", "가이드"],
            ["데이터베이스", "기술", "문서"],
            ["복지", "혜택", "정책"]
        ]
    },
    
    "analyst": {
        "questions": [
            "고객 테이블의 성별 분포를 차트로 보여주세요",
            "최근 뉴스 키워드 데이터를 분석해주세요",
            "고객 데이터에서 연령대별 분포를 조회해주세요",
            "고객 수를 카운트해주세요",
            "월별 고객 증가 추이를 분석해주세요"
        ],
        "expected_outputs": [
            "성별 분포 차트",
            "뉴스 키워드 분석",
            "연령대별 분포",
            "고객 수 카운트",
            "월별 증가 추이"
        ]
    },
    
    "predict": {
        "test_data": [
            {
                "question": "다음 고객 데이터로 이탈 예측을 해주세요",
                "csv_data": "tenure,monthly_charges,total_charges,contract,payment_method\n12,29.85,358.2,Month-to-month,Electronic check"
            },
            {
                "question": "고객 이탈 가능성을 예측해주세요", 
                "csv_data": "tenure,monthly_charges,total_charges,contract,payment_method\n36,45.50,1638.0,Two year,Credit card"
            }
        ],
        "expected_format": ["이탈", "확률", "예측"]
    },
    
    "coding": {
        "questions": [
            "Python에서 리스트 컴프리헨션 사용법을 알려주세요",
            "FastAPI로 간단한 REST API를 만드는 방법을 설명해주세요", 
            "LangGraph의 create_react_agent 사용법을 알려주세요",
            "GitHub API를 사용하는 방법을 설명해주세요",
            "Django에서 모델 관계 설정하는 방법을 알려주세요"
        ],
        "expected_concepts": [
            ["리스트", "컴프리헨션", "Python"],
            ["FastAPI", "REST", "API"],
            ["LangGraph", "create_react_agent"],
            ["GitHub", "API", "사용법"],
            ["Django", "모델", "관계"]
        ]
    }
}

# 평가 지표 정의
EVALUATION_METRICS = {
    "relevance": {
        "description": "응답이 질문과 얼마나 관련성이 있는지 평가",
        "scale": "0-1 (0: 관련없음, 1: 매우 관련있음)"
    },
    
    "completeness": {
        "description": "응답이 질문에 대해 완전한 정보를 제공하는지 평가",
        "scale": "0-1 (0: 불완전, 1: 완전)"
    },
    
    "accuracy": {
        "description": "응답의 정확성 평가",
        "scale": "0-1 (0: 부정확, 1: 정확)"
    },
    
    "agent_cooperation": {
        "description": "에이전트 간 협업이 적절히 이루어졌는지 평가",
        "scale": "0-1 (0: 협업 없음, 1: 적절한 협업)"
    }
}

# 결과 리포트 템플릿
REPORT_TEMPLATE = """
# LangGraph 에이전트 평가 보고서

## 평가 개요
- 평가 일시: {timestamp}
- 평가 대상: graph.py의 4개 에이전트
- 총 테스트 케이스: {total_test_cases}개

## 에이전트별 성능

### 📄 문서 검색 에이전트 (doc_search_assistant)
- 평균 관련성 점수: {doc_relevance:.2f}
- 평균 완성도 점수: {doc_completeness:.2f}
- 주요 강점: {doc_strengths}
- 개선 사항: {doc_improvements}

### 📊 분석 에이전트 (analyst_assistant)  
- 평균 관련성 점수: {analyst_relevance:.2f}
- 평균 완성도 점수: {analyst_completeness:.2f}
- 에이전트 협업 점수: {analyst_cooperation:.2f}
- 주요 강점: {analyst_strengths}
- 개선 사항: {analyst_improvements}

### 🔮 예측 에이전트 (predict_assistant)
- 평균 관련성 점수: {predict_relevance:.2f}
- 평균 완성도 점수: {predict_completeness:.2f}
- 주요 강점: {predict_strengths}
- 개선 사항: {predict_improvements}

### 💻 코딩 에이전트 (coding_assistant)
- 평균 관련성 점수: {coding_relevance:.2f}
- 평균 완성도 점수: {coding_completeness:.2f}
- 주요 강점: {coding_strengths}
- 개선 사항: {coding_improvements}

## 전체 평가 요약
- 전체 평균 점수: {overall_score:.2f}
- 가장 성능이 좋은 에이전트: {best_agent}
- 가장 개선이 필요한 에이전트: {worst_agent}

## 권장 사항
{recommendations}
"""

def get_evaluation_config() -> EvaluationConfig:
    """평가 설정 객체를 반환합니다."""
    return EvaluationConfig()

def get_test_cases_for_agent(agent_name: str) -> Dict[str, Any]:
    """특정 에이전트의 테스트 케이스를 반환합니다."""
    return AGENT_TEST_CASES.get(agent_name, {})

def get_all_test_cases() -> Dict[str, Dict[str, Any]]:
    """모든 에이전트의 테스트 케이스를 반환합니다."""
    return AGENT_TEST_CASES 