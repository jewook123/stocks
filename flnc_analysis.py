"""
FLNC (Fluence Energy) 종합 주식 분석
- 3일간 뉴스 수집 및 감성 분석 (Claude + Web Search)
- SEC 공시 체크
- ESS 관련주 동향 분석

실행: python3 flnc_analysis.py
"""

import os
import json
import time
import requests
from datetime import datetime

# ── 설정 ──────────────────────────────────────────────────────────────────────
TARGET_TICKER = "FLNC"
TARGET_COMPANY = "Fluence Energy"
DAYS_BACK = 3

ESS_TICKERS = [
    ("FLNC", "Fluence Energy"),
    ("STEM", "Stem Inc"),
    ("ENPH", "Enphase Energy"),
    ("SEDG", "SolarEdge"),
    ("AES",  "AES Corporation"),
    ("NEE",  "NextEra Energy"),
    ("BE",   "Bloom Energy"),
    ("PLUG", "Plug Power"),
    ("FCEL", "FuelCell Energy"),
    ("ARRY", "Array Technologies"),
]

TOKEN = open("/home/claude/.claude/remote/.session_ingress_token").read().strip()
API_URL = "https://api.anthropic.com/v1/messages"

HEADERS = {
    "content-type": "application/json",
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "web-search-2025-03-05",
    "Authorization": f"Bearer {TOKEN}",
}

SEARCH_TOOL = [{
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
}]


# ── 공통 API 호출 ──────────────────────────────────────────────────────────────
def call_claude(messages: list, tools: list = None, max_tokens: int = 3000) -> str:
    """Claude API를 호출하고 텍스트 응답을 반환합니다."""
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools

    for attempt in range(3):
        try:
            resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            # 텍스트 블록만 추출
            texts = [
                block["text"]
                for block in data.get("content", [])
                if block.get("type") == "text"
            ]
            return "\n".join(texts).strip()

        except requests.exceptions.ReadTimeout:
            if attempt < 2:
                time.sleep(5)
                continue
            raise
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 529 and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            raise

    return ""


def call_claude_with_search(prompt: str, max_tokens: int = 4000) -> str:
    """웹 검색을 포함한 Claude API 호출."""
    return call_claude(
        messages=[{"role": "user", "content": prompt}],
        tools=SEARCH_TOOL,
        max_tokens=max_tokens,
    )


# ── 1. 뉴스 수집 및 감성 분석 ──────────────────────────────────────────────────
def analyze_news_sentiment() -> dict:
    """최근 3일간 FLNC 뉴스를 검색하고 감성 분석합니다."""
    today = datetime.now().strftime("%Y년 %m월 %d일")
    prompt = f"""오늘은 {today}입니다.

웹 검색을 통해 최근 {DAYS_BACK}일간 FLNC (Fluence Energy) 관련 뉴스를 찾아주세요.
다음 검색어들을 활용해 충분히 검색해주세요:
- "FLNC Fluence Energy news"
- "Fluence Energy stock"
- "FLNC earnings analyst"

수집한 뉴스들을 분석하여 아래 JSON 형식으로 응답해주세요:

{{
  "news_items": [
    {{
      "date": "YYYY-MM-DD",
      "title": "뉴스 제목",
      "source": "출처",
      "url": "URL",
      "sentiment": "positive|negative|neutral",
      "summary": "한 줄 요약"
    }}
  ],
  "overall_sentiment": "Bullish|Bearish|Neutral",
  "sentiment_score": -10에서 10 사이 정수,
  "positive_count": 긍정 뉴스 수,
  "negative_count": 부정 뉴스 수,
  "neutral_count": 중립 뉴스 수,
  "key_themes": ["테마1", "테마2", "테마3"],
  "catalysts": ["상승 촉매1", "상승 촉매2"],
  "risk_factors": ["리스크1", "리스크2"],
  "investor_summary": "투자자 관점 종합 요약 (2-3문장)",
  "price_info": {{
    "current_price": "현재가 또는 최근가",
    "price_change": "변동폭",
    "price_change_pct": "변동률"
  }}
}}

JSON만 응답하세요 (코드블록 없이)."""

    raw = call_claude_with_search(prompt, max_tokens=4000)

    # JSON 추출 시도
    for line in [raw, raw.strip()]:
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    # { } 블록 추출
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

    # 파싱 실패 시 raw 텍스트 포함해서 반환
    return {
        "news_items": [],
        "overall_sentiment": "Unknown",
        "sentiment_score": 0,
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "key_themes": [],
        "catalysts": [],
        "risk_factors": [],
        "investor_summary": raw[:500],
        "price_info": {},
        "_raw": raw,
    }


# ── 2. SEC 공시 조회 ─────────────────────────────────────────────────────────
def check_sec_filings() -> dict:
    """SEC EDGAR에서 최근 FLNC 공시를 조회합니다."""
    today = datetime.now().strftime("%Y년 %m월 %d일")
    prompt = f"""오늘은 {today}입니다.

웹 검색을 통해 Fluence Energy (FLNC, CIK: 0001872812)의 최근 SEC 공시를 찾아주세요.
다음을 검색해주세요:
- "Fluence Energy SEC filing 2026"
- "FLNC 8-K 10-Q SEC EDGAR"
- site:sec.gov Fluence Energy

결과를 아래 JSON 형식으로 응답해주세요:

{{
  "filings": [
    {{
      "form_type": "8-K 또는 10-Q 등",
      "date": "YYYY-MM-DD",
      "description": "공시 내용 요약",
      "url": "SEC EDGAR URL",
      "significance": "high|medium|low",
      "notes": "투자자 관점 메모"
    }}
  ],
  "total_count": 전체 공시 수,
  "last_10q_date": "최근 10-Q 날짜",
  "last_8k_date": "최근 8-K 날짜",
  "summary": "공시 동향 요약",
  "notable_events": ["주목할 이벤트1", "주목할 이벤트2"]
}}

JSON만 응답하세요 (코드블록 없이)."""

    raw = call_claude_with_search(prompt, max_tokens=4000)

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start:end])
            # filings 리스트가 있으면 total_count 자동 보정
            if "filings" in parsed and isinstance(parsed["filings"], list):
                if not parsed.get("total_count"):
                    parsed["total_count"] = len(parsed["filings"])
            return parsed
        except json.JSONDecodeError:
            # JSON이 잘린 경우 부분 파싱 시도
            pass

    # 부분 파싱: filings 배열 추출 시도
    filings_start = raw.find('"filings"')
    if filings_start >= 0:
        arr_start = raw.find("[", filings_start)
        # 배열 끝을 찾기 (중첩 고려)
        depth, arr_end = 0, -1
        for i, ch in enumerate(raw[arr_start:], arr_start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    arr_end = i + 1
                    break
        if arr_end > arr_start:
            try:
                filings = json.loads(raw[arr_start:arr_end])
                return {
                    "filings": filings,
                    "total_count": len(filings),
                    "summary": "SEC EDGAR 조회 완료",
                    "_raw": raw,
                }
            except json.JSONDecodeError:
                pass

    return {
        "filings": [],
        "total_count": 0,
        "summary": raw[:800] if raw else "조회 실패",
        "_raw": raw,
    }


# ── 3. ESS 관련주 동향 ────────────────────────────────────────────────────────
def analyze_ess_stocks() -> dict:
    """ESS 관련주 최근 주가 동향을 분석합니다 (2번 호출로 분할)."""
    today = datetime.now().strftime("%Y년 %m월 %d일")

    # 1차 호출: 주요 5개 종목 (FLNC 포함)
    group1 = ESS_TICKERS[:5]
    # 2차 호출: 나머지 5개 종목
    group2 = ESS_TICKERS[5:]

    def stock_prompt(tickers_group):
        ticker_list = ", ".join(f"{t} ({n})" for t, n in tickers_group)
        return f"""오늘은 {today}입니다.

다음 ESS(Energy Storage Systems) 관련 주식들의 최근 주가를 검색해주세요:
{ticker_list}

각 종목을 검색하여 아래 JSON 형식으로 응답하세요:

{{
  "stocks": [
    {{
      "ticker": "종목코드",
      "company": "회사명",
      "current_price": "현재가",
      "change_1d": "1일 변동률",
      "change_5d": "5일 변동률",
      "change_1m": "1개월 변동률",
      "52w_high": "52주 최고",
      "52w_low": "52주 최저",
      "trend": "강한상승|상승|횡보|하락|강한하락",
      "notes": "주요 이슈 한 줄"
    }}
  ]
}}

JSON만 응답하세요."""

    all_stocks = []

    for group in [group1, group2]:
        try:
            raw = call_claude_with_search(stock_prompt(group), max_tokens=2500)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                all_stocks.extend(data.get("stocks", []))
        except Exception as e:
            print(f"  [경고] 주가 데이터 일부 조회 실패: {e}")

    # 섹터 종합 분석 (별도 호출)
    stock_summary = json.dumps(all_stocks, ensure_ascii=False)
    sector_prompt = f"""오늘은 {today}입니다.

다음은 ESS(에너지저장) 관련 주식 데이터입니다:
{stock_summary}

이를 바탕으로 아래 JSON을 작성해주세요. 최신 ESS 섹터 뉴스도 검색하세요:

{{
  "sector_trend": "상승세|하락세|혼조세|횡보",
  "sector_summary": "ESS 섹터 동향 요약 (3문장)",
  "top_performer": "최고 성과 종목 및 이유",
  "worst_performer": "최저 성과 종목 및 이유",
  "flnc_vs_sector": "FLNC의 섹터 대비 성과 비교",
  "investment_outlook": "ESS 섹터 투자 전망 (3-4문장)",
  "key_catalysts": ["섹터 상승 촉매1", "섹터 상승 촉매2"],
  "sector_risks": ["섹터 리스크1", "섹터 리스크2"]
}}

JSON만 응답하세요."""

    sector_data = {
        "sector_trend": "N/A",
        "sector_summary": "",
        "top_performer": "",
        "worst_performer": "",
        "flnc_vs_sector": "",
        "investment_outlook": "",
        "key_catalysts": [],
        "sector_risks": [],
    }
    try:
        raw = call_claude_with_search(sector_prompt, max_tokens=2000)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            sector_data = json.loads(raw[start:end])
    except Exception as e:
        print(f"  [경고] 섹터 분석 실패: {e}")

    return {"stocks": all_stocks, **sector_data}


# ── 출력 헬퍼 ──────────────────────────────────────────────────────────────────
def sep(char="=", width=72):
    print(char * width)

def section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")

def subsection(title):
    pad = 68 - len(title)
    print(f"\n  ── {title} {'─'*max(pad,2)}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
    sep("█")
    print(f"  FLNC (Fluence Energy) 종합 주식 분석 리포트")
    print(f"  분석 일시: {now}")
    sep("█")

    results = {}

    # ── 섹션 1: 뉴스 감성 분석 ────────────────────────────────────────────────
    section(f"1. 구글/야후 뉴스 감성 분석 — 최근 {DAYS_BACK}일")
    print("  Claude Web Search로 뉴스 수집 및 분석 중...")

    sentiment = analyze_news_sentiment()
    results["sentiment"] = sentiment

    # 가격 정보
    pi = sentiment.get("price_info", {})
    if pi:
        price_str = pi.get("current_price", "N/A")
        chg = pi.get("price_change", "")
        pct = pi.get("price_change_pct", "")
        print(f"\n  📈 FLNC 현재가: {price_str}  {chg} ({pct})")

    # 감성 스코어 바
    score = sentiment.get("sentiment_score", 0)
    overall = sentiment.get("overall_sentiment", "N/A")
    bar_len = min(abs(int(score)), 10)
    direction = "+" if score >= 0 else "-"
    bar = "█" * bar_len + "░" * (10 - bar_len)
    print(f"\n  감성 점수: {overall}  [{direction}{bar}]  ({score:+d}/10)")

    pos = sentiment.get("positive_count", 0)
    neg = sentiment.get("negative_count", 0)
    neu = sentiment.get("neutral_count", 0)
    print(f"  뉴스 분포: 긍정 {pos}건  부정 {neg}건  중립 {neu}건")

    if sentiment.get("investor_summary"):
        print(f"\n  [투자자 요약]\n  {sentiment['investor_summary']}")

    if sentiment.get("key_themes"):
        print(f"\n  [주요 테마]  {' / '.join(sentiment['key_themes'])}")

    if sentiment.get("catalysts"):
        print("\n  [상승 촉매]")
        for c in sentiment["catalysts"]:
            print(f"    + {c}")

    if sentiment.get("risk_factors"):
        print("\n  [리스크 요인]")
        for r in sentiment["risk_factors"]:
            print(f"    - {r}")

    news_items = sentiment.get("news_items", [])
    if news_items:
        subsection("수집된 뉴스 목록")
        for i, n in enumerate(news_items, 1):
            tag = {"positive": "✅", "negative": "❌", "neutral": "─ "}.get(n.get("sentiment"), "?")
            src = f"[{n.get('source','')}]" if n.get("source") else ""
            print(f"  {i:2d}. {tag} [{n.get('date','')}] {n.get('title','')} {src}")
            if n.get("summary"):
                print(f"       → {n['summary']}")
            if n.get("url"):
                print(f"       🔗 {n['url']}")

    # ── 섹션 2: SEC 공시 ──────────────────────────────────────────────────────
    section("2. SEC 공시 내역 조회")
    print("  SEC EDGAR 검색 중...")

    filings_data = check_sec_filings()
    results["filings"] = filings_data

    total = filings_data.get("total_count", 0)
    summary = filings_data.get("summary", "")
    print(f"\n  총 공시 건수: {total}건")

    if filings_data.get("last_10q_date"):
        print(f"  최근 10-Q:  {filings_data['last_10q_date']}")
    if filings_data.get("last_8k_date"):
        print(f"  최근 8-K:   {filings_data['last_8k_date']}")

    if summary:
        print(f"\n  [공시 동향 요약]\n  {summary}")

    notable = filings_data.get("notable_events", [])
    if notable:
        print("\n  [주목할 이벤트]")
        for e in notable:
            print(f"    📋 {e}")

    filings = filings_data.get("filings", [])
    if filings:
        subsection("공시 목록")
        print(f"  {'폼타입':<10} {'날짜':<12} {'중요도':<8} 내용")
        print(f"  {'─'*10} {'─'*12} {'─'*8} {'─'*35}")
        for f in filings:
            sig_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(f.get("significance", "low"), "⚪")
            print(f"  {f.get('form_type',''):<10} {f.get('date',''):<12} {sig_icon} {f.get('description','')[:40]}")
            if f.get("notes"):
                print(f"  {'':<10} {'':<12}   → {f['notes']}")
            if f.get("url"):
                print(f"  {'':<10} {'':<12}   🔗 {f['url']}")

    # ── 섹션 3: ESS 관련주 동향 ───────────────────────────────────────────────
    section("3. ESS 관련주 동향 분석")
    print("  ESS 섹터 주가 데이터 검색 중...")

    ess_data = analyze_ess_stocks()
    results["ess"] = ess_data

    sector_trend = ess_data.get("sector_trend", "N/A")
    sector_summary = ess_data.get("sector_summary", "")

    print(f"\n  ESS 섹터 동향: {sector_trend}")
    if ess_data.get("top_performer"):
        print(f"  최고 성과:    {ess_data['top_performer']}")
    if ess_data.get("worst_performer"):
        print(f"  최저 성과:    {ess_data['worst_performer']}")
    if ess_data.get("flnc_vs_sector"):
        print(f"  FLNC vs 섹터: {ess_data['flnc_vs_sector']}")

    stocks = ess_data.get("stocks", [])
    if stocks:
        subsection("ESS 관련주 주가 현황")
        hdr = f"  {'티커':<6} {'회사명':<22} {'현재가':>8} {'1일':>7} {'5일':>7} {'1개월':>8} {'동향':<10} 특이사항"
        print(hdr)
        print("  " + "─" * 90)
        for s in stocks:
            trend_icon = {
                "강한상승": "▲▲", "상승": "▲ ", "횡보": "─ ",
                "하락": "▼ ", "강한하락": "▼▼",
            }.get(s.get("trend", ""), "? ")
            ticker = s.get("ticker", "")
            star = "★" if ticker == "FLNC" else " "
            print(
                f"  {star}{ticker:<5} {s.get('company',''):<22} "
                f"{s.get('current_price','N/A'):>8} "
                f"{s.get('change_1d','N/A'):>7} "
                f"{s.get('change_5d','N/A'):>7} "
                f"{s.get('change_1m','N/A'):>8} "
                f"{trend_icon} {s.get('trend',''):<8} "
                f"{s.get('notes','')[:35]}"
            )

    if sector_summary:
        subsection("섹터 동향 상세")
        print(f"  {sector_summary}")

    if ess_data.get("investment_outlook"):
        subsection("투자 전망")
        print(f"  {ess_data['investment_outlook']}")

    if ess_data.get("key_catalysts"):
        print("\n  [섹터 상승 촉매]")
        for c in ess_data["key_catalysts"]:
            print(f"    + {c}")

    if ess_data.get("sector_risks"):
        print("\n  [섹터 리스크]")
        for r in ess_data["sector_risks"]:
            print(f"    - {r}")

    # ── 최종 요약 ──────────────────────────────────────────────────────────────
    section("★ 종합 요약")
    total_news = len(news_items)
    print(f"""
  📰 뉴스 감성:   {overall}  ({score:+d}/10)  |  총 {total_news}건 (긍정 {pos} / 부정 {neg} / 중립 {neu})
  📋 SEC 공시:    최근 {total}건
  📈 ESS 섹터:    {sector_trend}  |  추적 종목 {len(stocks)}개
    """)

    if sentiment.get("investor_summary"):
        print(f"  FLNC 투자 포인트:\n  {sentiment['investor_summary']}\n")

    # JSON 저장
    output_path = f"/home/user/stocks/flnc_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"generated_at": now, **results},
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    sep("█")
    print(f"  분석 결과 저장: {output_path}")
    sep("█")
    print()


if __name__ == "__main__":
    main()
