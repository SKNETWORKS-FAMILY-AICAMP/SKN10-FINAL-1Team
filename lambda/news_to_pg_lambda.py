# news_to_pg_lambda.py - AWS Lambda function to fetch news and store in PostgreSQL

import os
import re
import json
import uuid
import requests
import psycopg2
from datetime import datetime, timedelta
from collections import Counter

# Environment variables (AWS Lambda environment variables)
NAVER_API_KEY = os.environ.get('API_KEY')
if not NAVER_API_KEY:
    print('API_KEY environment variable not set, checking for API_KEY in .env file')

BASE_URL = 'https://api-v2.deepsearch.com/v1/articles'


# Database configuration
DB_HOST = os.environ.get('DB_HOST')
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_PORT = os.environ.get('DB_PORT', '5432')

# Default values for local testing
if not DB_HOST:
    print('Database environment variables not set, using default values for local testing')
    DB_HOST = 'localhost'
if not DB_NAME:
    DB_NAME = 'mydatabase'  # Match the one in Django settings
if not DB_USER:
    DB_USER = 'myuser'      # Match the one in Django settings
if not DB_PASSWORD:
    DB_PASSWORD = 'mypassword'  # For local testing only

print(f"Database connection: {DB_HOST}:{DB_PORT}/{DB_NAME} (user: {DB_USER})")

def get_top_keywords_from_articles(limit=10):
    """
    Extracts top keywords from recent articles about telecom companies.
    Code adapted from final_bot.py
    """
    date_to = datetime.today().date()
    date_from = date_to - timedelta(days=3)  # Increased to 3 days to get more results
    
    # 통신사 관련 키워드를 더 구체적으로 지정하여 관련성 높은 뉴스만 가져오기
    payload = {
        'keyword': '통신사 AND (서비스 OR 요금제 OR 고객 OR 네트워크 OR 5G OR 데이터 OR 스마트폰) OR (SKT AND 통신) OR (KT AND 통신) OR (LG U+ AND 통신)',
        'date_from': str(date_from),
        'date_to': str(date_to),
        'page_size': 100
    }
    
    # Setup headers with API key
    headers = {'Authorization': f'Bearer {NAVER_API_KEY}'}
    
    print(f"Fetching articles from {date_from} to {date_to}")
    res = requests.get(BASE_URL, headers=headers, params=payload, timeout=30)
    
    # Check for API errors
    if res.status_code != 200:
        print(f"API Error: {res.status_code} - {res.text}")
        return ["SKT", "KT", "LG U+", "통신사", "5G", "통신", "인터넷", "AI", "데이터", "요금제"]  # Fallback keywords
        
    articles = res.json().get('data', [])
    print(f"Found {len(articles)} articles for keyword extraction")

    today_kw = f'{datetime.today().day}일'
    # 불용어 목록 확장 - 뉴스에서 자주 나오지만 의미가 적은 단어들 추가
    stopwords = {
        '있다', '기자', '보도', '이날', '위해', '대한', '제', '중', '관련', '했다', '밝혔다',
        '이번', '또한', '지난', '통해', '있으며', '것으로', '등', '및', '에서', '하는', '하고',
        '으로', '되어', '됐다', '수', '것', '더', '같은', today_kw, '경우', '회사',
        '처음', '때문', '정도', '현재', '그러나', '위한', '하면', '이에', '위해서',
        '위해', '통해서', '위한', '지난해', '이후', '지속', '전납', '오늘', '계획'
    }

    counter = Counter()
    for art in articles:
        text = art.get('title', '') + ' ' + art.get('summary', '')
        clean = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9\s]', '', text)
        for w in clean.split():
            if len(w) > 1 and w not in stopwords:
                counter[w] += 1

    return [w for w, _ in counter.most_common(limit)]

def fetch_news_for_keywords(keywords, limit=10):
    """
    Fetches news articles for the given keywords.
    Code adapted from final_bot.py
    """
    date_to = datetime.today().date()
    date_from = date_to - timedelta(days=5)  # Increased to 5 days to get more results
    
    # 통신사 관련 키워드만 확보
    telecom_keywords = [
        'SKT', 'KT', 'LG U+', '통신사', '5G', '통신', '네트워크', '요금제', '데이터', '스마트폰',
        'SK텔레콤', 'LG유플러스', '인터넷', '무선', '사이브워즈', '웹', '앱', '프로모션',
        '핸드폰', '폰', '아이폰', '갤럭시', '삼성전자', '애플', '온라인', '리프팅', '서비스'
    ]
    
    # 입력된 키워드가 통신사 관련 키워드에 있는지 확인하고, 없으면 통신사 키워드로 대체
    filtered_keywords = []
    for kw in keywords:
        if kw in telecom_keywords or any(telecom_kw in kw for telecom_kw in telecom_keywords):
            filtered_keywords.append(kw)
    
    # 통신사 관련 키워드가 없으면 기본 통신사 키워드 사용
    if not filtered_keywords:
        filtered_keywords = telecom_keywords[:limit]
        print(f"No telecom-related keywords found, using default telecom keywords")
    else:
        print(f"Using filtered telecom keywords: {filtered_keywords}")
    
    # Setup headers with API key
    headers = {'Authorization': f'Bearer {NAVER_API_KEY}'}
    results = []

    for kw in filtered_keywords[:limit]:
        # 검색 쿼리를 개선하여 통신사 관련 뉴스만 가져오기
        payload = {
            # 키워드 + 통신사 관련 컨텍스트 추가
            'keyword': f'{kw} AND (통신 OR 스마트폰 OR 네트워크 OR 서비스 OR 요금제)',
            'date_from': str(date_from),
            'date_to': str(date_to),
            'page_size': 1,
            'sort_by': 'accuracy'  # 정확도 기준으로 정렬
        }
        print(f"Fetching news for keyword: {kw}")
        res = requests.get(BASE_URL, headers=headers, params=payload, timeout=30)
        
        # Check for API errors
        if res.status_code != 200:
            print(f"API Error for keyword {kw}: {res.status_code} - {res.text}")
            continue
            
        articles = res.json().get('data', [])
        print(f"Found {len(articles)} articles for keyword: {kw}")

        if articles:
            art = articles[0]
            results.append({
                'date': date_to,
                'keyword': kw,
                'title': art.get('title', '(제목 없음)'),
                'summary': art.get('summary', '(요약 없음)'),
                'url': art.get('content_url', '')
            })
        else:
            results.append({
                'date': date_to,
                'keyword': kw,
                'title': '관련 뉴스 없음',
                'summary': '',
                'url': ''
            })

    return results

def is_telecom_related(title, summary):
    """
    제목과 요약에서 통신사 관련 키워드가 있는지 확인하는 함수
    """
    # 통신사 관련 키워드 목록
    telecom_keywords = [
        'SKT', 'KT', 'LG U+', 'SK텔레콤', 'LG유플러스', '통신사', '통신', '네트워크', '5G', '요금제', '데이터',
        '스마트폰', '인터넷', '무선', '사이버워즈', '웹', '앱', '프로모션', '핸드폰', '폰', '아이폰', '갤럭시',
        '삼성전자', '애플', '온라인', '리프팅', '서비스', 'SK', 'LG', '통화'
    ]
    
    # 제목과 요약에서 통신사 관련 키워드가 있는지 확인
    combined_text = (title + ' ' + summary).lower()
    for keyword in telecom_keywords:
        if keyword.lower() in combined_text:
            return True
    return False

def insert_into_db(news_items):
    """
    Inserts news items into the PostgreSQL SummaryNewsKeywords table
    """
    # 통신사 관련 뉴스만 필터링
    filtered_items = []
    for item in news_items:
        if is_telecom_related(item['title'], item['summary']):
            filtered_items.append(item)
        else:
            print(f"Skipping non-telecom news: {item['title']}")
    
    if not filtered_items:
        print("No telecom-related news found")
        return False
        
    print(f"Found {len(filtered_items)} telecom-related news items out of {len(news_items)}")
    news_items = filtered_items
    
    conn = None
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        
        # Create a cursor
        cur = conn.cursor()
        
        # Django 의존성 없이 테이블 생성
        table_created = create_table_if_not_exists(cur, conn)
        
        # Insert each news item
        inserted_count = 0
        for item in news_items:
            # Generate a UUID for the id field
            id = str(uuid.uuid4())  # Convert UUID to string for psycopg2
            
            # Insert the data
            # Try to insert each item, but don't fail the entire batch on one error
            try:
                cur.execute(
                    '''
                    INSERT INTO summary_news_keywords (id, date, keyword, title, summary, url)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO NOTHING
                    ''',
                    (
                        id,
                        item['date'],
                        item['keyword'],
                        item['title'],
                        item['summary'],
                        item['url']
                    )
                )
                inserted_count += 1
                print(f"Inserted news item for keyword: {item['keyword']}")
            except Exception as e:
                print(f"Error inserting item for keyword {item['keyword']}: {e}")
                # Continue with other items instead of failing completely
                
        print(f"Successfully inserted {inserted_count} out of {len(news_items)} news items")
        
        # Commit the transaction
        conn.commit()
        
        # Close the cursor
        cur.close()
        
        return True
    except Exception as e:
        print(f'Database error: {e}')
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def create_table_if_not_exists(cur, conn):
    """
    Django 의존성 없이 SummaryNewsKeywords 테이블을 생성하는 함수
    """
    try:
        # 테이블 존재 여부 확인
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'summary_news_keywords'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            print("Table 'summary_news_keywords' does not exist, creating it...")
            # Django 모델과 동일한 구조로 테이블 생성
            cur.execute("""
                CREATE TABLE IF NOT EXISTS summary_news_keywords (
                    id UUID PRIMARY KEY,
                    date DATE NOT NULL,
                    keyword TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT,
                    url VARCHAR(500),
                    CONSTRAINT unique_url UNIQUE (url)
                );
            """)
            conn.commit()
            print("Table created successfully!")
            
            # 인덱스 생성 (성능 최적화)
            try:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_news_date ON summary_news_keywords(date);
                    CREATE INDEX IF NOT EXISTS idx_news_keyword ON summary_news_keywords(keyword);
                """)
                conn.commit()
                print("Indexes created successfully!")
            except Exception as e:
                print(f"Warning: Could not create indexes: {e}")
        return True
    except Exception as e:
        print(f"Error checking/creating table: {e}")
        return False

def lambda_handler(event, context):
    """
    AWS Lambda handler function
    """
    try:
        print("Starting Lambda execution...")
        # Get top keywords
        keywords = get_top_keywords_from_articles(limit=10)
        print(f"Top keywords: {', '.join(keywords)}")
        
        # Fetch news for those keywords
        news_items = fetch_news_for_keywords(keywords, limit=10)
        print(f"Fetched {len(news_items)} news items")
        
        # Insert into PostgreSQL
        success = insert_into_db(news_items)
        
        result = {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully processed news',
                'count': len(news_items),
                'keywords': keywords,
                'success': success
            })
        }
        print(f"Lambda execution completed successfully")
        return result
    except Exception as e:
        error_message = f'Error: {str(e)}'
        print(error_message)
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': error_message
            })
        }

# For local testing
if __name__ == '__main__':
    # .env 파일에서 환경 변수 로드

    test_news = [
        {
            'date': datetime.today().date(),
            'keyword': 'SKT',
            'title': '[테스트] SK텔레콤 5G 서비스 확대',
            'summary': 'SK텔레콤이 5G 서비스 영역을 전국으로 확대한다고 발표했습니다.',
            'url': 'https://example.com/news/1'
        },
        {
            'date': datetime.today().date(),
            'keyword': 'KT',
            'title': '[테스트] KT, AI 기반 서비스 출시',
            'summary': 'KT가 인공지능 기반 서비스를 새롭게 출시했습니다.',
            'url': 'https://example.com/news/2'
        },
        {
            'date': datetime.today().date(),
            'keyword': 'LG U+',
            'title': '[테스트] LG U+, 홈 IoT 서비스 확장',
            'summary': 'LG U+가 가정용 IoT 서비스를 확장하여 스마트홈 시장을 공략합니다.',
            'url': 'https://example.com/news/3'
        },
        {
            'date': datetime.today().date(),
            'keyword': '통신사',
            'title': '[테스트] 국내 통신사 데이터 요금제 개편',
            'summary': '국내 주요 통신사들이 데이터 무제한 요금제를 새롭게 출시할 예정입니다.',
            'url': 'https://example.com/news/4'
        }
    ]
    success = insert_into_db(test_news)
    print({'statusCode': 200, 'body': json.dumps({
        'message': 'Successfully processed test news',
        'count': len(test_news),
        'success': success
    })})
    
    # 실제 API 테스트가 필요한 경우 아래 코드 주석 해제
    if NAVER_API_KEY and NAVER_API_KEY != '':
        run_real_test = input("Do you want to run with real API? (y/n): ").strip().lower()
        if run_real_test == 'y':
            print("Running with real API...")
            result = lambda_handler(None, None)
            print(result)
