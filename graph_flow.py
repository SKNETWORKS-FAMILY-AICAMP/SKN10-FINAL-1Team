from typing import Dict, List, Any, Tuple, Optional, TypedDict, Annotated, Union
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# 로컬 모듈 임포트
from rag_pipeline.vector_store import Neo4jVectorSearch
from rag_pipeline.llm import GeminiLLM

# 환경 변수 로드
load_dotenv()

# 상태 클래스 정의
class QueryState(TypedDict):
    messages: List[Dict[str, str]]  # 대화 메시지 목록
    query: str  # 현재 쿼리
    query_type: str  # 쿼리 타입: "vector", "graph", "hybrid"
    vector_results: List[Dict]  # 벡터 검색 결과
    graph_results: List[Dict]  # 그래프 검색 결과
    combined_results: List[Dict]  # 결합된 결과
    related_nodes: Dict[str, Any]  # 관련 노드 정보
    graph_context: Dict[str, Any]  # 그래프 컨텍스트
    final_answer: str  # 최종 답변
    citations: List[Dict]  # 인용 정보


class HybridGraphFlow:
    """
    하이브리드 그래프 기반 RAG 검색 파이프라인
    
    LangGraph를 사용하여 그래프, 벡터, 하이브리드 검색 기능을 결합한 고급 RAG 시스템 구현
    """
    
    def __init__(self):
        """HybridGraphFlow 초기화"""
        self.vector_search = Neo4jVectorSearch()
        self.llm = GeminiLLM()
        
        # LangGraph 워크플로우 구성
        self.workflow = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """LangGraph 워크플로우 구성"""
        
        # StateGraph 생성
        graph = StateGraph(QueryState)
        
        # 노드 추가
        graph.add_node("determine_query_type", self._determine_query_type)
        graph.add_node("vector_search", self._vector_search)
        graph.add_node("graph_search", self._graph_search)
        graph.add_node("hybrid_search", self._hybrid_search)
        graph.add_node("gather_related_nodes", self._gather_related_nodes)
        graph.add_node("extract_graph_context", self._extract_graph_context)
        graph.add_node("generate_response", self._generate_response)
        
        # 조건부 엣지 및 워크플로우 정의
        graph.add_conditional_edges(
            "determine_query_type",
            lambda state: state["query_type"],
            {
                "vector": "vector_search",
                "graph": "graph_search",
                "hybrid": "hybrid_search"
            }
        )
        
        # 순수 벡터 검색은 그래프 관련 노드 건너뜀기 (그래프 정보 참고하지 않음)
        graph.add_edge("vector_search", "generate_response")
        
        # 그래프/하이브리드 검색은 그래프 정보 추출 노드 포함
        graph.add_edge("graph_search", "gather_related_nodes")
        graph.add_edge("hybrid_search", "gather_related_nodes")
        graph.add_edge("gather_related_nodes", "extract_graph_context")
        graph.add_edge("extract_graph_context", "generate_response")
        
        graph.add_edge("generate_response", END)
        
        # 시작 노드 설정
        graph.set_entry_point("determine_query_type")
        
        # 컴파일 및 반환
        return graph.compile()
    
    def _determine_query_type(self, state: QueryState) -> QueryState:
        """
        사용자 쿼리 의도 분석하여 최적의 검색 전략 결정 (LLM 활용)
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        query = state["query"]
        
        # LLM을 사용하여 쿼리 유형 결정
        from rag_pipeline.llm import GeminiLLM
        
        llm = GeminiLLM()
        
        prompt = f"""
당신은 연구 논문 검색 시스템의, 쿼리 의도 분석을 담당하는 전문가입니다.
사용자의 쿼리를 분석하여 적절한 검색 전략을 결정해야 합니다.

다음 세 가지 검색 전략 중 하나를 선택하세요:
1. "vector": 의미적 유사성에 기반한 검색이 필요할 때 (예: "당뇨병 치료에 관한 최신 연구", "COVID-19 백신의 효과")
2. "graph": 논문/저자/키워드 간의 관계나 네트워크를 탐색할 때 (예: "Smith 교수와 공동 연구한 저자들", "면역학과 신경학 분야를 연결하는 연구")
3. "hybrid": 복합적인 정보 요구가 있을 때 (기본값, 벡터 검색과 그래프 탐색을 모두 활용)

사용자 쿼리: {query}

위 쿼리에 가장 적합한 검색 전략은 무엇인가요? "vector", "graph", "hybrid" 중 하나만 출력하세요.
"""
        
        try:
            response = llm.model.generate_content(prompt)
            result = response.text.strip().lower()
            
            # 유효한 응답인지 확인
            if result in ["vector", "graph", "hybrid"]:
                query_type = result
            else:
                # 유효하지 않은 응답인 경우 하이브리드 검색 사용
                query_type = "hybrid"
                print(f"LLM 응답이 유효하지 않음 ('{result}'), 기본값 'hybrid' 사용")
        except Exception as e:
            # LLM 호출 실패 시 하이브리드 검색 사용
            query_type = "hybrid"
            print(f"쿼리 유형 결정 중 오류 발생: {e}")
        
        return {
            **state,
            "query_type": query_type
        }
    
    def _vector_search(self, state: QueryState) -> QueryState:
        """
        벡터 검색 수행 (그래프 정보 사용하지 않음)
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        query = state["query"]
        
        # 벡터 검색 수행
        vector_results = self.vector_search.semantic_search(query, top_k=10)
        
        # 벡터 검색 모드에서는 각 문서에 검색 타입 표시
        for result in vector_results:
            result["search_type"] = "vector"
        
        # 그래프 관련 필드 초기화 - 벡터 검색에서는 그래프 정보 사용하지 않음
        return {
            **state,
            "vector_results": vector_results,
            "graph_results": [],  # 그래프 결과 빈 리스트로 초기화
            "combined_results": vector_results,  # 벡터 결과가 최종 결과
            "related_nodes": {},  # 관련 노드 정보 빈 사전으로 초기화
            "graph_context": {}   # 그래프 컨텍스트 빈 사전으로 초기화
        }
    
    def _graph_search(self, state: QueryState) -> QueryState:
        """
        그래프 검색 수행
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        query = state["query"]
        
        # LLM을 사용하여 Cypher 쿼리 생성 (실제로는 더 정교한 구현 필요)
        # 여기서는 간단한 예시 Cypher 쿼리 사용
        cypher_query = """
        MATCH (a:Article)-[:HAS_KEYWORD]->(k:Keyword)
        WHERE k.term CONTAINS '비만'
        RETURN a.pmid as pmid, a.title as title, a.abstract as abstract, 
               collect(k.term) as keywords
        LIMIT 10
        """
        
        try:
            results, _ = self.vector_search.db.cypher_query(cypher_query)
            
            # 결과 포맷팅
            graph_results = []
            for row in results:
                pmid, title, abstract, keywords = row
                graph_results.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "keywords": keywords,
                    "search_type": "graph"  # 이 결과가 그래프 검색에서 나온 것임을 표시
                })
            
            return {
                **state,
                "graph_results": graph_results,
                "combined_results": graph_results  # 그래프 검색 모드에서는 그래프 결과가 최종 결과
            }
        
        except Exception as e:
            print(f"그래프 검색 오류: {e}")
            return {
                **state,
                "graph_results": [],
                "combined_results": []
            }
    
    def _hybrid_search(self, state: QueryState) -> QueryState:
        """
        하이브리드 검색 수행 (벡터 + 그래프)
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        query = state["query"]
        
        # 1. 먼저 벡터 검색 수행
        vector_results = self.vector_search.semantic_search(query, top_k=5)
        
        # 벡터 검색 결과가 없으면 종료
        if not vector_results:
            return {
                **state,
                "vector_results": [],
                "graph_results": [],
                "combined_results": []
            }
        
        # 2. 벡터 검색 결과를 기반으로 그래프 검색 수행
        pmids = [doc.get("pmid") for doc in vector_results if doc.get("pmid")]
        
        if not pmids:
            return {
                **state,
                "vector_results": vector_results,
                "graph_results": [],
                "combined_results": vector_results
            }
        
        # 벡터 검색으로 찾은 문서들과 연결된 다른 문서 찾기
        try:
            connections = []
            for pmid in pmids[:3]:  # 상위 3개만 처리
                article_connections = self.vector_search.find_article_connections(pmid)
                connections.extend(article_connections)
            
            # 그래프 결과와 벡터 결과 결합
            combined_results = vector_results.copy()
            
            # 중복 제거하면서 그래프 결과 추가
            existing_pmids = set(pmids)
            for conn in connections:
                if conn.get("pmid") and conn.get("pmid") not in existing_pmids:
                    conn["search_type"] = "graph"  # 이 결과가 그래프 검색에서 나온 것임을 표시
                    combined_results.append(conn)
                    existing_pmids.add(conn.get("pmid"))
                    
                    # 최대 15개 결과로 제한
                    if len(combined_results) >= 15:
                        break
            
            return {
                **state,
                "vector_results": vector_results,
                "graph_results": connections,
                "combined_results": combined_results
            }
        
        except Exception as e:
            print(f"하이브리드 검색 오류: {e}")
            return {
                **state,
                "vector_results": vector_results,
                "graph_results": [],
                "combined_results": vector_results
            }
    
    def _gather_related_nodes(self, state: QueryState) -> QueryState:
        """
        검색 결과의 관련 노드 정보 수집
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        combined_results = state.get("combined_results", [])
        
        if not combined_results:
            return {**state, "related_nodes": {}}
        
        # 첫 번째 문서의 PMID로 관련 노드 정보 가져오기
        first_doc_pmid = combined_results[0].get("pmid")
        if not first_doc_pmid:
            return {**state, "related_nodes": {}}
        
        # 관련 노드 정보 가져오기
        related_nodes = self.vector_search.get_related_nodes(first_doc_pmid)
        
        return {
            **state,
            "related_nodes": related_nodes
        }
    
    def _extract_graph_context(self, state: QueryState) -> QueryState:
        """
        그래프 컨텍스트 정보 추출
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        combined_results = state.get("combined_results", [])
        
        if not combined_results:
            return {**state, "graph_context": {}}
        
        # 그래프 컨텍스트 정보 추출
        graph_context = self.vector_search.get_graph_context_for_response(combined_results)
        
        return {
            **state,
            "graph_context": graph_context
        }
    
    def _generate_response(self, state: QueryState) -> QueryState:
        """
        LLM을 사용하여 최종 응답 생성
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        query = state.get("query", "")
        combined_results = state.get("combined_results", [])
        related_nodes = state.get("related_nodes", {})
        graph_context = state.get("graph_context", {})
        messages = state.get("messages", [])
        
        # 챗 히스토리 구성
        chat_history = messages[:-1] if messages else []
        
        if not combined_results:
            final_answer = "검색 결과가 없습니다. 다른 질문을 시도해 주세요."
            return {
                **state,
                "final_answer": final_answer
            }
        
        # LLM 응답 생성
        final_answer = self.llm.generate_response(
            query=query,
            retrieved_docs=combined_results,
            related_info=related_nodes,
            graph_context=graph_context,
            chat_history=chat_history
        )
        
        # 인용 정보 추출
        citations = [
            {
                "pmid": doc.get("pmid"),
                "title": doc.get("title"),
                "search_type": doc.get("search_type", "vector")
            }
            for doc in combined_results[:5]  # 상위 5개 결과만 인용으로 사용
        ]
        
        # 메시지 업데이트
        new_messages = list(messages)
        new_messages.append({"role": "assistant", "content": final_answer})
        
        return {
            **state,
            "final_answer": final_answer,
            "citations": citations,
            "messages": new_messages
        }
    
    def query(self, user_query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        사용자 쿼리 처리
        
        Args:
            user_query: 사용자 쿼리
            chat_history: 이전 대화 기록
            
        Returns:
            응답 결과
        """
        # 대화 기록이 없으면 빈 리스트로 초기화
        if chat_history is None:
            chat_history = []
        
        # 사용자 메시지 추가
        messages = list(chat_history)
        messages.append({"role": "user", "content": user_query})
        
        # 초기 상태 설정
        initial_state = {
            "messages": messages,
            "query": user_query,
            "query_type": "",
            "vector_results": [],
            "graph_results": [],
            "combined_results": [],
            "related_nodes": {},
            "graph_context": {},
            "final_answer": "",
            "citations": []
        }
        
        # 워크플로우 실행
        result = self.workflow.invoke(initial_state)
        
        # 결과 반환
        return {
            "answer": result["final_answer"],
            "messages": result["messages"],
            "query_type": result["query_type"],
            "retrieved_docs": result["combined_results"],
            "related_info": result["related_nodes"],
            "graph_context": result["graph_context"],
            "citations": result["citations"]
        }


# 테스트용
if __name__ == "__main__":
    graph_flow = HybridGraphFlow()
    result = graph_flow.query("비만과 고혈압의 연관성에 대해 알려주세요")
    print(f"쿼리 타입: {result['query_type']}")
    print(f"응답: {result['answer']}")
    print(f"검색된 문서 수: {len(result['retrieved_docs'])}")
    if result['retrieved_docs']:
        print(f"첫 번째 문서: {result['retrieved_docs'][0]['title']}")
