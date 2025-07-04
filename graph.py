from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "web_search_preview"}],
    input="https://github.com/intel/ipex-llm  에서 가장 최근 업데이트 내용이 뭐야"
)

print(response.output_text)