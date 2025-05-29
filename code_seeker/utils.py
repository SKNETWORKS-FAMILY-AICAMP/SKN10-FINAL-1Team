from openai import OpenAI
from dotenv import load_dotenv

def summarize_message(message):
    load_dotenv()
    client = OpenAI()
    prompt = f'"{message}" 이건 사용자가 입력한 챗봇에게 전하는 메시지야. 한줄 제목으로 요약해줘.'

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "당신은 사용자의 메시지를 짧고 간결한 제목으로 요약해주는 도우미입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    summary = response.choices[0].message.content.strip()
    return summary