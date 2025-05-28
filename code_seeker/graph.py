from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from .models import *
# graph 객체 생성(선언)
# simple_graph 상 데이터 전달에 사용할 객체 -> State 클래스
class State(TypedDict):
    input: str    # 사용자 input data
    result: str   # Node의 result data

def screen_reset(request) :
    selected_session = None
    if ChatSession.objects.exists() : 
        selected_session = ChatSession.objects.order_by('-created_at').first()
    return selected_session

def 