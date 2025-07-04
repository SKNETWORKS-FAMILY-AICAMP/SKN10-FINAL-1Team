import asyncio
import httpx
import json
import time

# 1. 요청 정보 설정
URL = "https://trendofpill.com/Chatbot/ChatWithNuti/"
BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzUwNjY1NTk4LCJpYXQiOjE3NTA2NjQ5OTgsImp0aSI6Ijc2OGM2OWIwZTA2NzRhYjlhZWMwNWJjYTkxYmMwNzQwIiwidXNlcl9pZCI6MjR9.SXUksSeI67ZlThzsyqGrdXUloHgaZrjAxnnGszSrbsI"
COOKIES = {
    'messages': '.eJwliUsOgCAMRK9CuiZGkQbuAoRIxV8iLJD7a-NmZt4b5yDGq9US79zasmeQo1Qo4TjLU4vwncxMvieLJP7yfVk3y4exrPTEgMSpEn5G8U4q6wEkQAgvpjkgQg:1uTbwE:T62i_NLe0qCyqLHrtddQAiUmTkwEyM4FQY0iPwjGEKw',
    'csrftoken': '0cLbfEVYA2gP76iGN14KRbN9vmUQD1ie',
    'sessionid': '7i9jdx2lwr25sk4a8lf6whom06n8q3lb'
}
HEADERS = {
    'Accept': '*/*',
    'Authorization': f'Bearer {BEARER_TOKEN}',
    'Content-Type': 'application/json',
    'Origin': 'https://trendofpill.com',
    'Referer': 'https://trendofpill.com/Chatbot/ChatWithNuti/',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Whale/4.32.315.15 Safari/537.36',
    'X-Csrftoken': COOKIES['csrftoken']
}

async def send_single_request(client, i):
    """단일 비동기 요청을 보냅니다."""
    print(f"[시작] 요청 {i + 1}")
    
    payload = {
        "user_query": f"비타민e 먹고싶어",
        "chat_room_id": None  # 항상 새 채팅방으로 시작
    }

    try:
        response = await client.post(URL, headers=HEADERS, json=payload, cookies=COOKIES, timeout=40.0)
        response.raise_for_status()
        print(f"[성공] 요청 {i + 1} - 상태 코드: {response.status_code}")
        return {"status": "success", "request_num": i + 1, "response": response.json()}

    except httpx.HTTPStatusError as e:
        print(f"[HTTP 에러] 요청 {i + 1}: {e}")
        return {"status": "http_error", "request_num": i + 1, "error": str(e), "response_text": e.response.text}
    except Exception as e:
        print(f"[일반 에러] 요청 {i + 1}: {e}")
        return {"status": "error", "request_num": i + 1, "error": str(e)}

async def main():
    """100개의 요청을 비동기적으로 실행합니다."""
    start_time = time.time()
    print("--- 100개 비동기 요청 시작 ---")

    async with httpx.AsyncClient() as client:
        tasks = [send_single_request(client, i) for i in range(1)]
        results = await asyncio.gather(*tasks)

    end_time = time.time()

    # 결과 요약
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    
    print("\n--- [ 최종 결과 ] ---")
    print(f"총 실행 시간: {end_time - start_time:.2f}초")
    print(f"총 요청: 100")
    print(f"성공: {success_count}")
    print(f"실패: {error_count}")
    
    if error_count > 0:
        print("\n--- 실패한 요청 상세 ---")
        for r in results:
            if r['status'] != 'success':
                print(f"  - 요청 #{r['request_num']}: {r['error']}")
                if 'response_text' in r:
                    print(f"    응답: {r['response_text'][:100]}...")

if __name__ == "__main__":
    # Python 3.8+ 에서는 다음과 같이 실행합니다.
    # asyncio.run(main())
    # Windows에서 발생할 수 있는 ProactorEventLoop 관련 경고를 방지하기 위해
    # SelectorEventLoop를 명시적으로 사용할 수 있습니다.
    if asyncio.get_event_loop().is_running():
        # Jupyter/IPython 환경 등 이미 루프가 실행 중일 때
        loop = asyncio.get_event_loop()
        loop.create_task(main())
    else:
        asyncio.run(main())
