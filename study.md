#랭그래프에 관하여

랭그래프는 에이전트와 도구 및 다른 에이전트와의 연결관계를 시각화하고 이를 실행하는 프레임워크

랭그래프에서 중요한 개념
1. 에이전트(Agent): 에이전트는 사용자의 요청을 처리하고 결과를 반환하는 역할을 합니다.
2. 도구(Tool): 도구는 에이전트가 사용할 수 있는 기능을 나타냅니다.
3. 연결관계(Edge): 연결관계는 에이전트와 도구, 다른 에이전트와의 연결 관계를 나타냅니다.
4. create_supervisor: create_supervisor는 특정 목적의 에이전트 들에 대하여 routing을 할수있음.(분기점)
5. create_react_agent: create_react_agent는 에이전트를 생성하는 함수이며, 도구와 같이 선언하여 에이전트가 목적을 완수할때까지 반복하게 할수있음.
6. subgraph: subgraph는 에이전트와 도구, 연결관계를 등을 캡슐화 하여 하나의 에이전트로 취급할수있음


#랭그래프 스튜디오

랭그래프 스튜디오는 러닝그래프를 시각화하고 실행하는 프로그램

설치 방법
pip install --upgrade "langgraph-cli[inmem]"    

랭그래프 프로젝트 시작 방법
langgraph new path/to/your/app --template new-langgraph-project-python

.env 작성 필요

env에 필요한 내용들
OPENAI_API_KEY=
LANGSMITH_API_KEY=
PINCONE_API_KEY=

랭그래프 스튜디오 실행 방법
cd path/to/your/app
pip install -e .
langgraph dev

예시 랭그래프 프로젝트 : feature-lecture에 위치함





