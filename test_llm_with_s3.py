import os
import boto3
import pandas as pd
from dotenv import load_dotenv
import multiprocessing
from queue import Empty
import sys
from urllib.parse import urlparse
from botocore.config import Config

# .env 파일에서 환경 변수 로드
load_dotenv('backend/.env')

# S3에서 CSV 파일 다운로드 함수
def download_from_s3(s3_url):
    """
    S3 URL에서 CSV 파일을 다운로드하고 pandas DataFrame으로 반환합니다.
    """
    try:
        parsed_url = urlparse(s3_url)
        bucket_name = parsed_url.netloc.split('.')[0]
        object_key = parsed_url.path.lstrip('/')
        
        print(f"버킷 이름: {bucket_name}"); sys.stdout.flush()
        print(f"객체 키: {object_key}"); sys.stdout.flush()
        
        config = Config(
            connect_timeout=10,
            read_timeout=60,
            retries={'max_attempts': 2}
        )
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION', 'us-east-1'),
            config=config
        )
        
        print("S3에서 객체 가져오는 중..."); sys.stdout.flush()
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        print("S3에서 객체 가져오기 완료."); sys.stdout.flush()
        
        print("CSV를 DataFrame으로 읽는 중..."); sys.stdout.flush()
        df = pd.read_csv(response['Body'])
        print("CSV를 DataFrame으로 읽기 완료."); sys.stdout.flush()
        return df
    except Exception as e:
        print(f"S3에서 파일 다운로드 중 오류 발생: {str(e)}"); sys.stdout.flush()
        return None

def log_to_file(message):
    with open("worker_log_test.txt", "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

def llm_worker(df, result_queue):
    from openai import OpenAI
    log_to_file("LLM worker process started.")
    try:
        log_to_file("Initializing OpenAI client...")
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        log_to_file("OpenAI client initialized.")
        
        data_summary = f"""
        CSV 데이터 요약:
        - 행 수: {len(df)}
        - 열 수: {len(df.columns)}
        - 열 이름: {', '.join(df.columns)}
        
        데이터 샘플 (처음 5행):
        {df.head(5).to_string()}
        """
        log_to_file("Data summary created.")
        
        prompt = f"""
        다음 CSV 데이터를 분석하고 주요 인사이트를 제공해주세요:
        
        {data_summary}
        
        데이터에 대한 주요 인사이트와 패턴을 알려주세요.
        """
        log_to_file("Prompt created.")
        
        log_to_file("Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 데이터 분석 전문가입니다. CSV 데이터를 분석하고 유용한 인사이트를 제공합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        log_to_file("Received response from OpenAI API.")
        
        result = response.choices[0].message.content
        log_to_file("Extracted result from response.")
        
        result_queue.put(result)
        log_to_file("Put result into the queue.")
        
    except Exception as e:
        error_message = f"LLM 작업 중 오류 발생: {e}"
        log_to_file(error_message)
        result_queue.put(error_message)

if __name__ == "__main__":
    # 이전 로그 파일 삭제
    if os.path.exists("worker_log_test.txt"):
        os.remove("worker_log_test.txt")

    # 가장 최근에 업로드된 S3 URL 사용
    s3_url = "https://hinton-csv-upload.s3.amazonaws.com/f70c2eee-cf8a-4d30-b911-3460ef47fdfc-test_data.csv"
    print(f"테스트할 S3 URL: {s3_url}"); sys.stdout.flush()

    df = download_from_s3(s3_url)

    if df is not None:
        print("\nCSV 파일을 성공적으로 다운로드했습니다."); sys.stdout.flush()

        result_queue = multiprocessing.Queue()
        
        process = multiprocessing.Process(target=llm_worker, args=(df, result_queue))
        process.start()
        
        print("\nLLM 분석 결과를 기다립니다... (최대 60초 대기)"); sys.stdout.flush()
        try:
            # 큐에서 결과를 가져올 때까지 대기
            result = result_queue.get(timeout=60)
            print("\nLLM 분석 결과:"); sys.stdout.flush()
            print(result); sys.stdout.flush()
            
            # 분석 결과를 파일에 저장
            with open("analysis_result_test.txt", "w", encoding="utf-8") as f:
                f.write(result)
            print("\n분석 결과를 'analysis_result_test.txt' 파일에 저장했습니다."); sys.stdout.flush()
            
        except Empty:
            print("\nLLM 분석 시간이 초과되었습니다. 큐에서 결과를 가져오지 못했습니다."); sys.stdout.flush()
        finally:
            # 프로세스가 여전히 실행 중인지 확인하고 정리
            if process.is_alive():
                print("분석 프로세스 강제 종료 중..."); sys.stdout.flush()
                process.terminate()
            process.join(timeout=5) # 프로세스가 완전히 종료될 때까지 잠시 대기
            print(f"프로세스 최종 정리 완료. Exit code: {process.exitcode}"); sys.stdout.flush()
    else:
        print("CSV 파일 다운로드에 실패했습니다."); sys.stdout.flush()
