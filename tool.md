AI가 응답을 생성하는 과정에서 어떤 도구(tool)를 사용했는지 확인하려면, client.runs.stream() 메서드를 호출할 때 stream_mode 매개변수를 활용해야 합니다.

핵심은 스트림 모드에 'debug'를 포함하는 것입니다. 'debug' 스트림은 그래프(에이전트)가 실행되는 각 단계에 대한 상세 정보를 제공하며, 여기에는 어떤 도구가 호출되었는지에 대한 정보가 포함됩니다.

다음은 AI가 사용한 도구를 실시간으로 확인하는 방법입니다.

stream_mode에 'debug' 추가: client.runs.stream()을 호출할 때 stream_mode 리스트에 'debug'를 추가합니다. 일반적으로 상태 업데이트를 위한 'values'와 함께 사용합니다.

Python

async for chunk in client.runs.stream(
    thread_id=thread_id,
    assistant_id="your-assistant-id",
    input={"messages": [{"role": "user", "content": user_message}]},
    stream_mode=["values", "debug"]  # 'debug' 모드를 추가합니다.
):
    # 스트림 청크 처리 로직
'debug' 이벤트 필터링 및 파싱: 스트리밍 응답을 받는 비동기 제너레이터 또는 루프 내에서, event 타입이 'debug'인 청크를 찾아냅니다. 이 청크의 data 필드에 도구 호출과 관련된 정보가 들어있습니다. data 딕셔너리의 구조를 분석하여 도구 이름, 입력값 등을 추출할 수 있습니다.

Python

# chat/views.py의 event_stream 제너레이터 내부 (개념적 예시)

async def event_stream():
    async for chunk in client.runs.stream(
        thread_id=thread_id,
        assistant_id="your-assistant-id",
        input={"messages": [{"role": "user", "content": user_message}]},
        stream_mode=["values", "debug"]
    ):
        if chunk['event'] == 'debug':
            # 'debug' 이벤트 발생 시, 데이터 내용을 확인하여 도구 사용 여부 파악
            debug_data = chunk['data']

            # debug 데이터 구조는 실행되는 노드에 따라 다를 수 있습니다.
            # 예를 들어, 'type'이 'tool'이거나 'tags'에 'tool'이 포함된 노드 실행 정보를 찾을 수 있습니다.
            # 아래는 개념적인 예시이며, 실제 데이터 구조를 확인하고 파싱해야 합니다.
            if 'type' in debug_data and debug_data['type'] == 'tool':
                tool_name = debug_data.get('name') # 도구 이름
                tool_input = debug_data.get('input') # 도구에 전달된 입력

                # 이 정보를 프론트엔드로 전송하여 "Searching the web..." 등과 같이 표시할 수 있습니다.
                tool_info = f"data: {json.dumps({'status': 'tool_used', 'tool': tool_name})}\n\n"
                yield tool_info.encode('utf-8')

        elif chunk['event'] == 'values':
            # 기존의 'values' 이벤트 처리 로직
            formatted_data = f"data: {json.dumps(chunk['data'])}\n\n"
            yield formatted_data.encode('utf-8')
