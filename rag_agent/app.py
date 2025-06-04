------------- app.py -------------\nimport os\nfrom pathlib import Path\nfrom dotenv import load_dotenv\n\nHERE = Path(__file__).resolve().parent           # SKN10-FINAL_1TEAM/rag_agent\nPROJECT_ROOT = HERE.parent                       # SKN10-FINAL_1TEAM\nload_dotenv(PROJECT_ROOT / \".env\")\n\nfrom src.agent.graph import graph                # 컴파일된 StateGraph\n\nEXECUTOR = graph                                 # 가독성을 위해 별칭\n\nif __name__ == \"__main__\":\n    while True:\n        q = input(\"🔍 질문 (‘exit’ 입력 시 종료): \").strip()\n        if q.lower() in (\"exit\", \"quit\"): break\n        if not q: continue\n\n        try:\n            result = EXECUTOR.invoke(             # 0.4.x → invoke()\n                {\"question\": q},\n                config={\n                    \"configurable\": {\n                        \"top_k\": 5,\n                        # \"pinecone_index_name\": \"dense-index\",\n                        # \"openai_model\": \"gpt-3.5-turbo\",\n                    }\n                },\n            )\n            print(\"🏆 답변:\\n\" + result[\"answer\"] + \"\\n\")\n        except Exception as e:\n            print(\"⚠️  오류:\", e, \"\\n\")\n
# rag_agent/src/agent/app.py

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── 최상위 .env 파일을 한번에 읽어들입니다 ───
# 현재 파일(app.py)의 위치: SKN10‐FINAL_1TEAM/rag_agent/app.py
HERE         = Path(__file__).resolve().parent       # → SKN10‐FINAL_1TEAM/rag_agent
PROJECT_ROOT = HERE.parent                            # → SKN10‐FINAL_1TEAM
DOTENV_PATH  = PROJECT_ROOT / ".env"                  # → SKN10‐FINAL_1TEAM/.env
load_dotenv(dotenv_path=DOTENV_PATH)

# ─── 이제 graph (→ tasks) 모듈을 import 합니다 ───
from src.agent.graph import graph

def main():
    # (load_dotenv를 다시 호출해도 이미 한 번 읽었으므로 결과는 동일합니다.)
    load_dotenv(dotenv_path=DOTENV_PATH)

    executor = graph

    while True:
        question = input("🔍 질문을 입력하세요 (‘exit’ 입력 시 종료): ").strip()
        if question.lower() in ("exit", "quit"):
            print("👋 프로그램을 종료합니다.")
            break
        if not question:
            continue

        try:
            # StateGraph.invoke()를 쓸 수도 있지만, v0.4.7에서는 run()으로 충분합니다.
            result = executor.run(
                inputs={"question": question},
                config={"configurable": {
                    # 필요에 따라 Configurable 파라미터를 여기에 추가하세요.
                    # 예시:
                    # "pinecone_index_name": "dense-index",
                    # "top_k": 5,
                    # "openai_model": "gpt-3.5-turbo",
                }}
            )
            # 노드 ID가 "GenerateAnswer"이므로, 그 노드의 반환값에서 "answer"를 꺼냅니다.
            final_answer = result["GenerateAnswer"]["answer"]
            print("\n🏆 최종 답변:\n" + final_answer + "\n")

        except Exception as e:
            print(f"⚠️ 그래프 실행 중 오류: {e}")
            continue

if __name__ == "__main__":
    main()
