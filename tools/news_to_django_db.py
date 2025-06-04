#!/usr/bin/env python
# news_to_django_db.py - Fetch news and add to Django database directly

import os
import re
import sys
import uuid
import requests
from datetime import datetime, timedelta
from collections import Counter
from dotenv import load_dotenv

# Add the Django project directory to Python path
BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend')
sys.path.append(BACKEND_DIR)
print(f"Adding Django backend path: {BACKEND_DIR}")

# Load environment variables
load_dotenv()

# Setup environment for Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
print(f"Using Django settings module: {os.environ['DJANGO_SETTINGS_MODULE']}")
print(f"Python path: {sys.path}")

# Import Django and setup
import django
django.setup()

# Now we can import Django models
from knowledge.models import SummaryNewsKeywords

# API configuration
NAVER_API_KEY = os.environ.get('API_KEY')
BASE_URL = 'https://api-v2.deepsearch.com/v1/articles'
HEADERS = {'Authorization': f'Bearer {NAVER_API_KEY}'}

def get_top_keywords_from_articles(limit=10):
    """
    Extracts top keywords from recent articles about telecom companies.
    Code adapted from final_bot.py
    """
    date_to = datetime.today().date()
    date_from = date_to - timedelta(days=3)  # Increased to 3 days to get more results
    payload = {
        'keyword': '통신사 OR SKT OR KT OR LG OR LG U+',
        'date_from': str(date_from),
        'date_to': str(date_to),
        'page_size': 100
    }
    
    print(f"Fetching articles from {date_from} to {date_to}")
    
    try:
        res = requests.get(BASE_URL, headers=HEADERS, params=payload, timeout=30)
        
        # Check for API errors
        if res.status_code != 200:
            print(f"API Error: {res.status_code} - {res.text}")
            return ["SKT", "KT", "LG U+", "통신사", "5G", "통신", "인터넷", "AI", "데이터", "요금제"]  # Fallback keywords
            
        articles = res.json().get('data', [])
        print(f"Found {len(articles)} articles for keyword extraction")
    except Exception as e:
        print(f"Error fetching articles: {e}")
        return ["SKT", "KT", "LG U+", "통신사", "5G", "통신", "인터넷", "AI", "데이터", "요금제"]  # Fallback keywords

    today_kw = f'{datetime.today().day}일'
    stopwords = {
        '있다', '기자', '보도', '이날', '위해', '대한', '제', '중', '관련', '했다', '밝혔다',
        '이번', '또한', '지난', '통해', '있으며', '것으로', '등', '및', '에서', '하는', '하고',
        '으로', '되어', '됐다', '수', '것', '더', '같은', today_kw
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
    results = []

    for kw in keywords[:limit]:
        payload = {
            'keyword': kw,
            'date_from': str(date_from),
            'date_to': str(date_to),
            'page_size': 1
        }
        
        print(f"Fetching news for keyword: {kw}")
        
        try:
            res = requests.get(BASE_URL, headers=HEADERS, params=payload, timeout=30)
            
            # Check for API errors
            if res.status_code != 200:
                print(f"API Error for keyword {kw}: {res.status_code} - {res.text}")
                continue
                
            articles = res.json().get('data', [])
            print(f"Found {len(articles)} articles for keyword: {kw}")
        except Exception as e:
            print(f"Error fetching news for keyword {kw}: {e}")
            continue

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

def save_to_django_db(news_items):
    """
    Saves news items directly to Django database using ORM
    """
    created_count = 0
    for item in news_items:
        try:
            # Create a new SummaryNewsKeywords object and save it
            news = SummaryNewsKeywords(
                id=uuid.uuid4(),
                date=item['date'],
                keyword=item['keyword'],
                title=item['title'],
                summary=item['summary'],
                url=item['url']
            )
            news.save()
            created_count += 1
            print(f"Saved news item for keyword: {item['keyword']}")
        except Exception as e:
            print(f"Error saving item for keyword {item['keyword']}: {e}")
    
    print(f"Successfully saved {created_count} out of {len(news_items)} news items")
    return created_count

def main():
    """
    Main function to run the pipeline
    """
    # Confirm Django models are accessible
    try:
        print(f"Checking Django model access: {SummaryNewsKeywords.__name__}")
    except Exception as e:
        print(f"Error accessing Django models: {e}")
        return 0
        
    if NAVER_API_KEY:
        print(f"API Key loaded (length: {len(NAVER_API_KEY)}): {NAVER_API_KEY[:4]}...{NAVER_API_KEY[-4:]}")
        try:
            # Get top keywords
            keywords = get_top_keywords_from_articles(limit=10)
            print(f"Top keywords: {', '.join(keywords)}")
            
            # Fetch news for those keywords
            news_items = fetch_news_for_keywords(keywords, limit=10)
            
            # Save to Django database
            created_count = save_to_django_db(news_items)
            
            print(f"Process completed. Created {created_count} news items.")
            return created_count
        except Exception as e:
            print(f"Error in main process: {e}")
            return 0
    else:
        print("WARNING: No API key found in environment variables or .env file!")
        # Use test data
        print("⚠️ Running in test mode with sample data")
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
        created_count = save_to_django_db(test_news)
        print(f"Process completed. Created {created_count} test news items.")
        return created_count

# For AWS Lambda
def lambda_handler(event, context):
    """
    AWS Lambda handler function
    """
    try:
        count = main()
        return {
            'statusCode': 200,
            'body': {
                'message': 'Successfully processed news',
                'count': count
            }
        }
    except Exception as e:
        print(f"Lambda error: {e}")
        return {
            'statusCode': 500,
            'body': {
                'message': f'Error: {str(e)}'
            }
        }

# For local execution
if __name__ == '__main__':
    main()
