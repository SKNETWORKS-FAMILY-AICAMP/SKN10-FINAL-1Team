# final_bot.py

import os
import re
import requests
import discord
import pandas as pd
from datetime import datetime, timedelta, time as dtime
from collections import Counter
from discord.ext import commands, tasks
from dotenv import load_dotenv

# ─── 환경 변수 로드 ─────────────────────────────────────────
load_dotenv()
TOKEN         = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_ID    = int(os.getenv("DISCORD_CHANNEL_ID", 0))
NAVER_API_KEY = os.getenv("API_KEY")   # DeepSearch News v2 키
BASE_URL      = "https://api-v2.deepsearch.com/v1/articles"
HEADERS       = {"Authorization": f"Bearer {NAVER_API_KEY}"}

if not TOKEN or not NAVER_API_KEY:
    raise ValueError("❌ .env에 DISCORD_BOT_TOKEN 또는 API_KEY가 없습니다.")

# ─── 디스코드 봇 세팅 ────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ─── 1) 키워드 추출 함수 ──────────────────────────────────────
def get_top_keywords_from_articles(limit=10):
    date_to   = datetime.today().date()
    date_from = date_to - timedelta(days=1)
    payload = {
        "keyword":    "통신사 OR SKT OR KT OR LG OR LG U+",
        "date_from":  str(date_from),
        "date_to":    str(date_to),
        "page_size":  100
    }
    res      = requests.get(BASE_URL, headers=HEADERS, params=payload, timeout=30)
    articles = res.json().get("data", [])

    today_kw = f"{datetime.today().day}일"
    stopwords = {
        "있다","기자","보도","이날","위해","대한","제","중","관련","했다","밝혔다",
        "이번","또한","지난","통해","있으며","것으로","등","및","에서","하는","하고",
        "으로","되어","됐다","수","것","더","같은", today_kw
    }

    counter = Counter()
    for art in articles:
        text  = art.get("title","") + " " + art.get("summary","")
        clean = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", "", text)
        for w in clean.split():
            if len(w) > 1 and w not in stopwords:
                counter[w] += 1

    return [w for w,_ in counter.most_common(limit)]

# ─── 2) 뉴스 요약 수집 함수 ───────────────────────────────────
def fetch_news_for_keywords(keywords, limit=10):
    date_to   = datetime.today().date()
    date_from = date_to - timedelta(days=1)
    now       = datetime.now()
    date_str  = now.strftime("%Y-%m-%d")
    time_str  = now.strftime("%H:%M:%S")
    results   = []

    for kw in keywords[:limit]:
        payload = {
            "keyword":    kw,
            "date_from":  str(date_from),
            "date_to":    str(date_to),
            "page_size":  1
        }
        res      = requests.get(BASE_URL, headers=HEADERS, params=payload, timeout=30)
        articles = res.json().get("data", [])

        if articles:
            art = articles[0]
            results.append({
                "날짜":      date_str,
                "생성시간":   time_str,
                "키워드":     kw,
                "기사 제목":  art.get("title","(제목 없음)"),
                "요약":      art.get("summary","(요약 없음)"),
                "URL":       art.get("content_url","")
            })
        else:
            results.append({
                "날짜":      date_str,
                "생성시간":   time_str,
                "키워드":     kw,
                "기사 제목":  "관련 뉴스 없음",
                "요약":      "",
                "URL":       ""
            })

    return results

# ─── 3) 봇 준비 및 스케줄러 설정 ───────────────────────────────
@bot.event
async def on_ready():
    print(f"🤖 Bot 로그인 완료: {bot.user} (ID: {bot.user.id})")
    if not daily_pipeline.is_running():
        daily_pipeline.start()
        # 봇 시작 직후 즉시 파이프라인 실행
        await daily_pipeline()

@tasks.loop(time=dtime(hour=16, minute=0, second=0))  # UTC16:00 → KST01:00
async def daily_pipeline():
    key    = datetime.now().strftime("%Y%m%d")
    print(f"⏳ Pipeline 시작: {key}")

    kws     = get_top_keywords_from_articles(limit=10)
    summaries = fetch_news_for_keywords(kws, limit=10)

    df    = pd.DataFrame(summaries)
    fname = f"summary_news_keywords_{key}.csv"
    df.to_csv(fname, index=False, encoding="utf-8-sig")
    print(f"✅ 저장 완료: {fname}")

# ─── 4) 디스코드 커맨드 정의 ───────────────────────────────────
@bot.command(name="요약")
async def send_summary(ctx):
    key  = datetime.now().strftime("%Y%m%d")
    path = f"summary_news_keywords_{key}.csv"
    if os.path.isfile(path):
        await ctx.send(file=discord.File(path))
    else:
        await ctx.send(f"⚠️ 파일이 없습니다: `{path}`")

@bot.command(name="키워드")
async def send_keywords(ctx):
    key  = datetime.now().strftime("%Y%m%d")
    path = f"summary_news_keywords_{key}.csv"
    if not os.path.isfile(path):
        return await ctx.send("❌ 요약 파일이 아직 생성되지 않았습니다.")
    df   = pd.read_csv(path)
    if "키워드" not in df.columns:
        return await ctx.send("❌ '키워드' 컬럼을 찾을 수 없습니다.")
    kws  = df["키워드"].dropna().unique().tolist()[:10]
    await ctx.send("🔑 오늘의 상위 키워드:\n" + ", ".join(kws))

# ─── 5) 봇 실행 ───────────────────────────────────────────────
if __name__ == "__main__":
    bot.run(TOKEN)
