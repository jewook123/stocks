"""
FLNC (Fluence Energy) 종합 주식 분석
- 3일간 뉴스 수집 및 감성 분석 (Claude + Web Search)
- SEC 공시 체크
- ESS 관련주 동향 분석
- 어닝 콜 발언 및 CEO 톤 분석
- Short Interest / 옵션 Put/Call Ratio
- 기관 투자자 13F 포지션 변동
- 경쟁사 비교 분석 (FLNC vs BE vs STEM vs ENPH)
- 매크로 지표 (금리, 에너지 정책, 관세)

실행: python3 flnc_analysis.py [--telegram]
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

API_URL = "https://api.anthropic.com/v1/messages"


def _build_headers() -> dict:
    """ANTHROPIC_API_KEY 환경변수 또는 Claude Code 세션 토큰으로 헤더를 구성합니다."""
    base = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "web-search-2025-03-05",
    }
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        base["x-api-key"] = api_key
        return base
    token_file = "/home/claude/.claude/remote/.session_ingress_token"
    if os.path.exists(token_file):
        base["Authorization"] = f"Bearer {open(token_file).read().strip()}"
        return base
    raise EnvironmentError(
        "Anthropic 인증 정보를 찾을 수 없습니다. "
        "ANTHROPIC_API_KEY 환경변수를 설정하거나 Claude Code 환경에서 실행하세요."
    )


HEADERS = _build_headers()

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


# ── 4. 어닝 콜 발언 분석 ──────────────────────────────────────────────────────
def analyze_earnings_call() -> dict:
    """최근 어닝 콜 하이라이트와 CEO/CFO 톤을 분석합니다."""
    today = datetime.now().strftime("%Y년 %m월 %d일")
    prompt = f"""오늘은 {today}입니다.

Fluence Energy (FLNC) 최근 어닝 콜(실적 발표 컨퍼런스 콜)을 검색해주세요.
- "Fluence Energy earnings call transcript 2026"
- "FLNC Q2 2026 conference call"
- "Fluence Energy CEO CFO guidance"

아래 JSON으로 응답하세요:

{{
  "call_date": "어닝 콜 날짜",
  "quarter": "Q2 FY2026 등",
  "ceo_tone": "Confident|Cautious|Mixed",
  "tone_score": -5에서 5 사이 정수 (긍정=양수),
  "key_statements": [
    {{"speaker": "CEO/CFO", "quote": "핵심 발언", "implication": "투자자 시사점"}}
  ],
  "guidance_change": "raised|maintained|lowered|withdrawn",
  "guidance_details": "가이던스 구체 내용",
  "analyst_qa_highlights": ["애널리스트 Q&A 주요 내용1", "주요 내용2"],
  "management_credibility": "High|Medium|Low",
  "credibility_reason": "신뢰도 판단 근거",
  "red_flags": ["우려 신호1", "우려 신호2"],
  "positive_signals": ["긍정 신호1", "긍정 신호2"]
}}

JSON만 응답하세요."""

    raw = call_claude_with_search(prompt, max_tokens=3000)
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    return {"ceo_tone": "N/A", "tone_score": 0, "key_statements": [],
            "guidance_change": "N/A", "red_flags": [], "positive_signals": [], "_raw": raw[:300]}


# ── 5. Short Interest / 옵션 Put/Call Ratio ────────────────────────────────────
def analyze_short_sentiment() -> dict:
    """공매도 잔고와 옵션 시장 심리를 분석합니다."""
    today = datetime.now().strftime("%Y년 %m월 %d일")
    prompt = f"""오늘은 {today}입니다.

FLNC (Fluence Energy) 공매도 및 옵션 데이터를 검색해주세요:
- "FLNC short interest 2026"
- "FLNC put call ratio options"
- "Fluence Energy short squeeze"

아래 JSON으로 응답하세요:

{{
  "short_interest_pct": "유동주식 대비 공매도 비율 (%)",
  "short_interest_shares": "공매도 주식 수",
  "days_to_cover": "숏커버 소요일 (Days to Cover)",
  "short_change": "전월 대비 증감",
  "short_trend": "increasing|decreasing|stable",
  "short_squeeze_risk": "High|Medium|Low",
  "put_call_ratio": "Put/Call 비율",
  "options_sentiment": "Bullish|Bearish|Neutral",
  "iv_rank": "IV Rank (내재변동성 순위, 0-100)",
  "notable_options": "주목할 옵션 포지션 (대규모 콜/풋 등)",
  "insider_trading": [
    {{"date": "날짜", "person": "직책", "type": "buy|sell", "shares": "주식수", "value": "금액"}}
  ],
  "summary": "공매도/옵션 시장 심리 종합 요약"
}}

JSON만 응답하세요."""

    raw = call_claude_with_search(prompt, max_tokens=2500)
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    return {"short_interest_pct": "N/A", "put_call_ratio": "N/A",
            "options_sentiment": "N/A", "summary": raw[:300]}


# ── 6. 기관 투자자 13F 포지션 변동 ────────────────────────────────────────────
def analyze_institutional() -> dict:
    """13F 기관 투자자 포지션 변동을 분석합니다."""
    today = datetime.now().strftime("%Y년 %m월 %d일")
    prompt = f"""오늘은 {today}입니다.

FLNC (Fluence Energy) 기관 투자자 동향을 검색해주세요:
- "FLNC institutional ownership 13F 2026"
- "Fluence Energy hedge fund holdings"
- "FLNC institutional investors"

아래 JSON으로 응답하세요:

{{
  "institutional_ownership_pct": "기관 보유 비율 (%)",
  "total_institutional_shares": "기관 총 보유 주식",
  "top_holders": [
    {{"name": "기관명", "shares": "보유주식", "pct": "비율", "change": "전분기 대비 변동", "action": "increased|decreased|new|closed"}}
  ],
  "notable_changes": [
    {{"institution": "기관명", "action": "매수/매도/신규/청산", "shares": "주식수", "significance": "시사점"}}
  ],
  "smart_money_trend": "Accumulating|Distributing|Neutral",
  "hedge_fund_count": "보유 헤지펀드 수",
  "qia_status": "카타르 국부펀드 현황",
  "aes_status": "AES 지분 현황",
  "summary": "기관 투자자 동향 종합 (2문장)"
}}

JSON만 응답하세요."""

    raw = call_claude_with_search(prompt, max_tokens=3000)
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    return {"institutional_ownership_pct": "N/A", "smart_money_trend": "N/A",
            "top_holders": [], "notable_changes": [], "summary": raw[:300]}


# ── 7. 경쟁사 비교 분석 ────────────────────────────────────────────────────────
def analyze_competitors() -> dict:
    """FLNC vs 주요 경쟁사 실적·밸류에이션을 비교합니다."""
    today = datetime.now().strftime("%Y년 %m월 %d일")
    prompt = f"""오늘은 {today}입니다.

Fluence Energy (FLNC)와 주요 경쟁사를 비교 분석해주세요.
검색: "Fluence Energy vs Stem vs Bloom Energy comparison 2026"

비교 대상: FLNC, BE (Bloom Energy), STEM (Stem Inc), ENPH (Enphase Energy)

아래 JSON으로 응답하세요:

{{
  "comparison": [
    {{
      "ticker": "종목코드",
      "company": "회사명",
      "revenue_ttm": "최근 12개월 매출",
      "revenue_growth_yoy": "매출 성장률 YoY",
      "gross_margin": "총이익률 (%)",
      "ebitda_margin": "EBITDA 마진 (%)",
      "backlog": "수주잔고 (해당시)",
      "ev_revenue": "EV/Revenue 배수",
      "market_cap": "시가총액",
      "net_cash": "순현금 (부채 차감)",
      "profitability": "Profitable|Near-break-even|Loss-making",
      "competitive_edge": "핵심 경쟁 우위"
    }}
  ],
  "flnc_strengths": ["FLNC 강점1", "FLNC 강점2"],
  "flnc_weaknesses": ["FLNC 약점1", "FLNC 약점2"],
  "market_position": "FLNC의 시장 내 위치",
  "winner_by_category": {{
    "growth": "성장률 1위 종목",
    "margin": "마진 1위 종목",
    "valuation": "밸류에이션 가장 저렴한 종목",
    "momentum": "모멘텀 1위 종목"
  }},
  "summary": "경쟁 구도 종합 요약 (2문장)"
}}

JSON만 응답하세요."""

    raw = call_claude_with_search(prompt, max_tokens=3500)
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    return {"comparison": [], "flnc_strengths": [], "flnc_weaknesses": [],
            "market_position": "N/A", "summary": raw[:300]}


# ── 8. 매크로 지표 분석 ────────────────────────────────────────────────────────
def analyze_macro() -> dict:
    """금리, 에너지 정책, 관세 등 매크로 환경을 분석합니다."""
    today = datetime.now().strftime("%Y년 %m월 %d일")
    prompt = f"""오늘은 {today}입니다.

ESS/청정에너지 섹터에 영향을 미치는 매크로 지표를 검색해주세요:
- "Fed interest rate 2026 energy storage"
- "IRA Inflation Reduction Act energy storage 2026"
- "US tariff China battery ESS 2026"
- "10 year treasury yield 2026"

아래 JSON으로 응답하세요:

{{
  "fed_rate": "현재 연준 기준금리",
  "fed_outlook": "금리 방향성 (dovish|hawkish|neutral)",
  "rate_impact_on_flnc": "금리가 FLNC에 미치는 영향",
  "treasury_10y": "10년물 국채 수익률",
  "ira_status": "IRA 에너지 저장 관련 세액공제 현황",
  "ira_impact": "IRA가 FLNC 수익성에 미치는 영향",
  "tariff_status": "중국산 배터리/ESS 관세 현황",
  "tariff_impact": "관세가 FLNC 원가에 미치는 영향",
  "policy_tailwinds": ["정책 순풍1", "정책 순풍2"],
  "policy_headwinds": ["정책 역풍1", "정책 역풍2"],
  "energy_demand_outlook": "데이터센터/AI 전력 수요 전망",
  "macro_score": -5에서 5 사이 정수 (ESS 섹터에 유리=양수),
  "macro_summary": "매크로 환경 종합 평가 (2문장)"
}}

JSON만 응답하세요."""

    raw = call_claude_with_search(prompt, max_tokens=2500)
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    return {"fed_rate": "N/A", "macro_score": 0, "policy_tailwinds": [],
            "policy_headwinds": [], "macro_summary": raw[:300]}


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
def main() -> dict:
    now = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
    sep("█")
    print(f"  FLNC (Fluence Energy) 종합 주식 분석 리포트")
    print(f"  분석 일시: {now}")
    sep("█")

    results = {"generated_at": now}

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

    # ── 섹션 4: 어닝 콜 분석 ──────────────────────────────────────────────────
    section("4. 어닝 콜 발언 분석")
    print("  최근 어닝 콜 검색 중...")

    ec = analyze_earnings_call()
    results["earnings_call"] = ec

    tone_icon = {"Confident": "🟢", "Cautious": "🟡", "Mixed": "🟠"}.get(ec.get("ceo_tone", ""), "⚪")
    guide_icon = {"raised": "⬆️", "maintained": "➡️", "lowered": "⬇️", "withdrawn": "❌"}.get(ec.get("guidance_change", ""), "❓")
    print(f"\n  어닝 콜: {ec.get('quarter','N/A')}  ({ec.get('call_date','N/A')})")
    print(f"  경영진 톤: {tone_icon} {ec.get('ceo_tone','N/A')}  ({ec.get('tone_score',0):+d}/5)  |  가이던스: {guide_icon} {ec.get('guidance_change','N/A').upper()}")
    if ec.get("guidance_details"):
        print(f"  가이던스 내용: {ec['guidance_details']}")
    print(f"  경영진 신뢰도: {ec.get('management_credibility','N/A')}  — {ec.get('credibility_reason','')}")

    if ec.get("key_statements"):
        subsection("핵심 발언")
        for s in ec["key_statements"][:4]:
            print(f"  [{s.get('speaker','')}] \"{s.get('quote','')}\"")
            if s.get("implication"):
                print(f"    → {s['implication']}")

    if ec.get("analyst_qa_highlights"):
        subsection("Q&A 주요 내용")
        for q in ec["analyst_qa_highlights"][:3]:
            print(f"    • {q}")

    if ec.get("positive_signals"):
        print("\n  [긍정 신호]")
        for s in ec["positive_signals"][:3]:
            print(f"    ✅ {s}")
    if ec.get("red_flags"):
        print("\n  [레드 플래그]")
        for r in ec["red_flags"][:3]:
            print(f"    🚩 {r}")

    # ── 섹션 5: Short Interest / 옵션 ────────────────────────────────────────
    section("5. Short Interest / 옵션 Put/Call Ratio")
    print("  공매도·옵션 데이터 검색 중...")

    short_data = analyze_short_sentiment()
    results["short_sentiment"] = short_data

    sq_icon = {"High": "🔥", "Medium": "🟡", "Low": "🟢"}.get(short_data.get("short_squeeze_risk", ""), "⚪")
    opt_icon = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}.get(short_data.get("options_sentiment", ""), "⚪")
    print(f"\n  공매도 비율:   {short_data.get('short_interest_pct','N/A')}  ({short_data.get('short_change','N/A')})")
    print(f"  Days to Cover: {short_data.get('days_to_cover','N/A')}  |  추세: {short_data.get('short_trend','N/A')}")
    print(f"  숏스퀴즈 위험: {sq_icon} {short_data.get('short_squeeze_risk','N/A')}")
    print(f"  Put/Call:      {short_data.get('put_call_ratio','N/A')}  |  옵션 심리: {opt_icon} {short_data.get('options_sentiment','N/A')}")
    if short_data.get("iv_rank"):
        print(f"  IV Rank:       {short_data['iv_rank']}")
    if short_data.get("notable_options"):
        print(f"  주목 옵션:     {short_data['notable_options']}")
    if short_data.get("summary"):
        print(f"\n  [요약] {short_data['summary']}")

    insider = short_data.get("insider_trading", [])
    if insider:
        subsection("내부자 거래 (Form 4)")
        for t in insider[:5]:
            act = "매수 ✅" if t.get("type") == "buy" else "매도 ❌"
            print(f"    [{t.get('date','')}] {t.get('person','')}  {act}  {t.get('shares','')}주  ({t.get('value','')})")

    # ── 섹션 6: 기관 투자자 13F ───────────────────────────────────────────────
    section("6. 기관 투자자 13F 포지션 변동")
    print("  기관 투자자 데이터 검색 중...")

    inst_data = analyze_institutional()
    results["institutional"] = inst_data

    sm_icon = {"Accumulating": "🟢", "Distributing": "🔴", "Neutral": "🟡"}.get(inst_data.get("smart_money_trend", ""), "⚪")
    print(f"\n  기관 보유 비율: {inst_data.get('institutional_ownership_pct','N/A')}")
    print(f"  스마트머니 동향: {sm_icon} {inst_data.get('smart_money_trend','N/A')}")
    print(f"  보유 헤지펀드: {inst_data.get('hedge_fund_count','N/A')}")
    if inst_data.get("qia_status"):
        print(f"  카타르 국부펀드(QIA): {inst_data['qia_status']}")
    if inst_data.get("aes_status"):
        print(f"  AES 지분: {inst_data['aes_status']}")

    top_holders = inst_data.get("top_holders", [])
    if top_holders:
        subsection("주요 기관 보유 현황")
        print(f"  {'기관명':<30} {'보유비율':>8} {'전분기 대비':>12}  동향")
        print(f"  {'─'*30} {'─'*8} {'─'*12}  {'─'*8}")
        for h in top_holders[:8]:
            act_icon = {"increased": "▲", "decreased": "▼", "new": "★", "closed": "✕"}.get(h.get("action", ""), "─")
            print(f"  {h.get('name',''):<30} {h.get('pct',''):>8} {h.get('change',''):>12}  {act_icon}")

    if inst_data.get("notable_changes"):
        subsection("주목할 포지션 변동")
        for c in inst_data["notable_changes"][:4]:
            print(f"  • {c.get('institution','')}  {c.get('action','')}  {c.get('shares','')}")
            if c.get("significance"):
                print(f"    → {c['significance']}")

    if inst_data.get("summary"):
        print(f"\n  [요약] {inst_data['summary']}")

    # ── 섹션 7: 경쟁사 비교 ───────────────────────────────────────────────────
    section("7. 경쟁사 비교 분석 (FLNC vs BE vs STEM vs ENPH)")
    print("  경쟁사 데이터 검색 중...")

    comp_data = analyze_competitors()
    results["competitors"] = comp_data

    comparison = comp_data.get("comparison", [])
    if comparison:
        subsection("실적·밸류에이션 비교")
        print(f"  {'종목':<6} {'매출(TTM)':>10} {'성장률':>8} {'총이익률':>8} {'EV/Rev':>7} {'시가총액':>10}  수익성")
        print(f"  {'─'*6} {'─'*10} {'─'*8} {'─'*8} {'─'*7} {'─'*10}  {'─'*14}")
        for c in comparison:
            prof_icon = {"Profitable": "✅", "Near-break-even": "🟡", "Loss-making": "❌"}.get(c.get("profitability", ""), "⚪")
            star = "★" if c.get("ticker") == "FLNC" else " "
            print(
                f"  {star}{c.get('ticker',''):<5} {c.get('revenue_ttm','N/A'):>10} "
                f"{c.get('revenue_growth_yoy','N/A'):>8} {c.get('gross_margin','N/A'):>8} "
                f"{c.get('ev_revenue','N/A'):>7} {c.get('market_cap','N/A'):>10}  {prof_icon}"
            )

    wbc = comp_data.get("winner_by_category", {})
    if wbc:
        subsection("카테고리별 1위")
        for cat, winner in wbc.items():
            label = {"growth": "성장률", "margin": "마진", "valuation": "밸류에이션", "momentum": "모멘텀"}.get(cat, cat)
            print(f"    {label:<10}: {winner}")

    if comp_data.get("flnc_strengths"):
        print("\n  [FLNC 강점]")
        for s in comp_data["flnc_strengths"]:
            print(f"    ✅ {s}")
    if comp_data.get("flnc_weaknesses"):
        print("\n  [FLNC 약점]")
        for w in comp_data["flnc_weaknesses"]:
            print(f"    ⚠️  {w}")
    if comp_data.get("market_position"):
        print(f"\n  [시장 위치] {comp_data['market_position']}")

    # ── 섹션 8: 매크로 지표 ───────────────────────────────────────────────────
    section("8. 매크로 지표 (금리 / 정책 / 관세)")
    print("  매크로 데이터 검색 중...")

    macro = analyze_macro()
    results["macro"] = macro

    macro_score = macro.get("macro_score", 0)
    macro_bar = "█" * abs(macro_score) + "░" * (5 - abs(macro_score))
    macro_dir = "+" if macro_score >= 0 else "-"
    fed_icon = {"dovish": "🕊️ ", "hawkish": "🦅", "neutral": "➡️"}.get(macro.get("fed_outlook", ""), "❓")

    print(f"\n  매크로 점수: [{macro_dir}{macro_bar}] ({macro_score:+d}/5)  — ESS 섹터에 {'유리' if macro_score >= 0 else '불리'}")
    print(f"  연준 기준금리: {macro.get('fed_rate','N/A')}  |  방향: {fed_icon} {macro.get('fed_outlook','N/A')}")
    print(f"  10년물 금리:   {macro.get('treasury_10y','N/A')}")
    print(f"  금리 영향:     {macro.get('rate_impact_on_flnc','N/A')}")

    subsection("IRA 세액공제 현황")
    print(f"  {macro.get('ira_status','N/A')}")
    if macro.get("ira_impact"):
        print(f"  FLNC 영향: {macro['ira_impact']}")

    subsection("관세 환경")
    print(f"  {macro.get('tariff_status','N/A')}")
    if macro.get("tariff_impact"):
        print(f"  FLNC 영향: {macro['tariff_impact']}")

    if macro.get("policy_tailwinds"):
        print("\n  [정책 순풍]")
        for t in macro["policy_tailwinds"]:
            print(f"    🌬️  {t}")
    if macro.get("policy_headwinds"):
        print("\n  [정책 역풍]")
        for h in macro["policy_headwinds"]:
            print(f"    ⛔ {h}")
    if macro.get("energy_demand_outlook"):
        print(f"\n  [AI/데이터센터 전력 수요] {macro['energy_demand_outlook']}")
    if macro.get("macro_summary"):
        print(f"\n  [종합] {macro['macro_summary']}")

    # ── 최종 요약 ──────────────────────────────────────────────────────────────
    section("★ 종합 요약")
    total_news = len(news_items)
    macro_score = macro.get("macro_score", 0)
    print(f"""
  📰 뉴스 감성:   {overall}  ({score:+d}/10)  |  총 {total_news}건 (긍정 {pos} / 부정 {neg} / 중립 {neu})
  🎙️  어닝 콜 톤:  {ec.get('ceo_tone','N/A')}  ({ec.get('tone_score',0):+d}/5)  |  가이던스: {ec.get('guidance_change','N/A').upper()}
  📉 공매도:      {short_data.get('short_interest_pct','N/A')}  |  숏스퀴즈: {short_data.get('short_squeeze_risk','N/A')}  |  P/C Ratio: {short_data.get('put_call_ratio','N/A')}
  🏛️  스마트머니:  {inst_data.get('smart_money_trend','N/A')}  |  기관보유: {inst_data.get('institutional_ownership_pct','N/A')}
  📋 SEC 공시:    최근 {total}건
  📈 ESS 섹터:    {sector_trend}  |  추적 종목 {len(stocks)}개
  🌍 매크로:      {macro_score:+d}/5  |  연준: {macro.get('fed_rate','N/A')}  |  10Y: {macro.get('treasury_10y','N/A')}
    """)

    if sentiment.get("investor_summary"):
        print(f"  FLNC 투자 포인트:\n  {sentiment['investor_summary']}\n")

    # JSON 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join(os.path.dirname(__file__) or ".", f"flnc_report_{ts}.json")
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

    return results


# ── Telegram 전송 ──────────────────────────────────────────────────────────────
def _tg_escape(text: str) -> str:
    """MarkdownV2 특수문자 이스케이프."""
    for ch in r"\_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text


def send_telegram(results: dict):
    """분석 결과를 Telegram 봇으로 전송합니다."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        print("  [Telegram] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 미설정, 전송 생략")
        return

    tg_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def send(text: str):
        for attempt in range(3):
            r = requests.post(
                tg_url,
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=15,
            )
            if r.status_code == 200:
                return
            time.sleep(2 * (attempt + 1))
        print(f"  [Telegram] 전송 실패: {r.status_code} {r.text[:100]}")

    sentiment = results.get("sentiment", {})
    filings_data = results.get("filings", {})
    ess_data = results.get("ess", {})

    score = sentiment.get("sentiment_score", 0)
    overall = sentiment.get("overall_sentiment", "N/A")
    pi = sentiment.get("price_info", {})
    price_str = pi.get("current_price", "N/A")
    price_chg = pi.get("price_change_pct", "")

    sentiment_emoji = "🟢" if score >= 3 else "🔴" if score <= -3 else "🟡"
    now_str = results.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M"))

    # ── 메시지 1: 헤더 + 감성 요약 ──────────────────────────────────────────
    pos = sentiment.get("positive_count", 0)
    neg = sentiment.get("negative_count", 0)
    neu = sentiment.get("neutral_count", 0)

    catalysts = "\n".join(f"  ✅ {c}" for c in sentiment.get("catalysts", [])[:3])
    risks = "\n".join(f"  ⚠️ {r}" for r in sentiment.get("risk_factors", [])[:3])
    investor_summary = sentiment.get("investor_summary", "")[:400]

    msg1 = (
        f"📊 <b>FLNC (Fluence Energy) 주식 분석</b>\n"
        f"🗓 {now_str}\n"
        f"{'─'*30}\n\n"
        f"💹 <b>현재가:</b> {price_str}  {price_chg}\n"
        f"{sentiment_emoji} <b>감성:</b> {overall}  ({score:+d}/10)\n"
        f"📰 뉴스: 긍정 {pos}건 | 부정 {neg}건 | 중립 {neu}건\n\n"
        f"<b>🚀 상승 촉매</b>\n{catalysts}\n\n"
        f"<b>⚠️ 리스크</b>\n{risks}\n\n"
        f"<b>💬 투자 포인트</b>\n{investor_summary}"
    )
    send(msg1)
    time.sleep(0.5)

    # ── 메시지 2: 주요 뉴스 ──────────────────────────────────────────────────
    news_items = sentiment.get("news_items", [])
    if news_items:
        lines = ["📰 <b>주요 뉴스 (최근 3일)</b>\n"]
        for n in news_items[:8]:
            tag = {"positive": "✅", "negative": "❌", "neutral": "─"}.get(
                n.get("sentiment", ""), "•"
            )
            title = n.get("title", "")[:70]
            date = n.get("date", "")[:10]
            src = n.get("source", "")
            url = n.get("url", "")
            link = f'<a href="{url}">{src}</a>' if url else src
            lines.append(f"{tag} [{date}] {title}\n     {link}")
        send("\n".join(lines))
        time.sleep(0.5)

    # ── 메시지 3: SEC 공시 ────────────────────────────────────────────────────
    total_filings = filings_data.get("total_count", 0)
    filings = filings_data.get("filings", [])
    if total_filings or filings:
        lines = [f"📋 <b>SEC 공시 ({total_filings}건)</b>\n"]
        for f in filings[:6]:
            sig = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                f.get("significance", "low"), "⚪"
            )
            form = f.get("form_type", "")
            date = f.get("date", "")
            desc = f.get("description", "")[:60]
            url = f.get("url", "")
            link = f'  <a href="{url}">🔗 상세보기</a>' if url else ""
            lines.append(f"{sig} <b>{form}</b>  {date}\n  {desc}{link}")
        notable = filings_data.get("notable_events", [])
        if notable:
            lines.append("\n<b>📌 주목 이벤트</b>")
            for e in notable[:3]:
                lines.append(f"  • {e}")
        send("\n".join(lines))
        time.sleep(0.5)

    # ── 메시지 4: ESS 관련주 동향 ────────────────────────────────────────────
    stocks = ess_data.get("stocks", [])
    sector_trend = ess_data.get("sector_trend", "N/A")
    if stocks:
        trend_emoji = {
            "상승세": "📈", "하락세": "📉", "혼조세": "📊", "횡보": "➡️"
        }.get(sector_trend, "📊")
        lines = [f"{trend_emoji} <b>ESS 관련주 동향 — {sector_trend}</b>\n"]
        lines.append(f"{'티커':<6} {'현재가':>8} {'1일':>7} {'1개월':>8}  동향")
        lines.append("─" * 40)
        for s in stocks:
            trend_icon = {
                "강한상승": "▲▲", "상승": "▲ ", "횡보": "─ ",
                "하락": "▼ ", "강한하락": "▼▼",
            }.get(s.get("trend", ""), "?")
            star = "⭐" if s.get("ticker") == "FLNC" else "  "
            lines.append(
                f"{star}{s.get('ticker',''):<5} {s.get('current_price',''):>8} "
                f"{s.get('change_1d',''):>7} {s.get('change_1m',''):>8}  {trend_icon}"
            )
        outlook = ess_data.get("investment_outlook", "")[:300]
        if outlook:
            lines.append(f"\n<b>💡 투자 전망</b>\n{outlook}")
        send("<pre>" + "\n".join(lines) + "</pre>")
        time.sleep(0.5)

    # ── 메시지 5: 어닝 콜 + Short Interest ───────────────────────────────────
    ec = results.get("earnings_call", {})
    short_data = results.get("short_sentiment", {})
    tone_icon = {"Confident": "🟢", "Cautious": "🟡", "Mixed": "🟠"}.get(ec.get("ceo_tone", ""), "⚪")
    guide_icon = {"raised": "⬆️", "maintained": "➡️", "lowered": "⬇️"}.get(ec.get("guidance_change", ""), "❓")
    sq_icon = {"High": "🔥", "Medium": "🟡", "Low": "🟢"}.get(short_data.get("short_squeeze_risk", ""), "⚪")
    opt_icon = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}.get(short_data.get("options_sentiment", ""), "⚪")

    lines = [f"🎙️ <b>어닝 콜 분석 — {ec.get('quarter','N/A')}</b>\n"]
    lines.append(f"{tone_icon} 경영진 톤: <b>{ec.get('ceo_tone','N/A')}</b> ({ec.get('tone_score',0):+d}/5)")
    lines.append(f"{guide_icon} 가이던스: <b>{ec.get('guidance_change','N/A').upper()}</b>")
    if ec.get("guidance_details"):
        lines.append(f"  {ec['guidance_details'][:100]}")
    if ec.get("positive_signals"):
        lines.append("\n✅ <b>긍정 신호</b>")
        for s in ec["positive_signals"][:2]:
            lines.append(f"  • {s}")
    if ec.get("red_flags"):
        lines.append("\n🚩 <b>레드 플래그</b>")
        for r in ec["red_flags"][:2]:
            lines.append(f"  • {r}")

    lines.append(f"\n📉 <b>Short Interest / 옵션</b>")
    lines.append(f"공매도: {short_data.get('short_interest_pct','N/A')}  |  DTC: {short_data.get('days_to_cover','N/A')}일")
    lines.append(f"숏스퀴즈: {sq_icon} {short_data.get('short_squeeze_risk','N/A')}  |  추세: {short_data.get('short_trend','N/A')}")
    lines.append(f"Put/Call: {short_data.get('put_call_ratio','N/A')}  |  옵션심리: {opt_icon} {short_data.get('options_sentiment','N/A')}")
    if short_data.get("notable_options"):
        lines.append(f"주목 옵션: {short_data['notable_options'][:80]}")
    send("\n".join(lines))
    time.sleep(0.5)

    # ── 메시지 6: 기관 투자자 13F ────────────────────────────────────────────
    inst_data = results.get("institutional", {})
    sm_icon = {"Accumulating": "🟢", "Distributing": "🔴", "Neutral": "🟡"}.get(inst_data.get("smart_money_trend", ""), "⚪")
    lines = [f"🏛️ <b>기관 투자자 13F 현황</b>\n"]
    lines.append(f"기관 보유: <b>{inst_data.get('institutional_ownership_pct','N/A')}</b>  |  헤지펀드: {inst_data.get('hedge_fund_count','N/A')}")
    lines.append(f"스마트머니: {sm_icon} <b>{inst_data.get('smart_money_trend','N/A')}</b>")
    if inst_data.get("qia_status"):
        lines.append(f"🇶🇦 QIA: {inst_data['qia_status'][:80]}")
    if inst_data.get("aes_status"):
        lines.append(f"⚡ AES: {inst_data['aes_status'][:80]}")
    top_h = inst_data.get("top_holders", [])
    if top_h:
        lines.append("\n<b>주요 보유 기관</b>")
        for h in top_h[:5]:
            act = {"increased": "▲", "decreased": "▼", "new": "★", "closed": "✕"}.get(h.get("action", ""), "─")
            lines.append(f"  {act} {h.get('name','')[:25]}  {h.get('pct','')}")
    if inst_data.get("notable_changes"):
        lines.append("\n<b>주목할 변동</b>")
        for c in inst_data["notable_changes"][:3]:
            lines.append(f"  • {c.get('institution','')} {c.get('action','')} {c.get('shares','')}")
    if inst_data.get("summary"):
        lines.append(f"\n{inst_data['summary'][:200]}")
    send("\n".join(lines))
    time.sleep(0.5)

    # ── 메시지 7: 경쟁사 비교 ────────────────────────────────────────────────
    comp_data = results.get("competitors", {})
    comparison = comp_data.get("comparison", [])
    lines = ["⚔️ <b>경쟁사 비교 분석</b>\n"]
    if comparison:
        lines.append("<pre>")
        lines.append(f"{'종목':<6} {'매출성장':>8} {'총이익률':>8} {'EV/Rev':>7}  수익성")
        lines.append("─" * 42)
        for c in comparison:
            prof = {"Profitable": "흑자", "Near-break-even": "손익분기", "Loss-making": "적자"}.get(c.get("profitability", ""), "N/A")
            star = "⭐" if c.get("ticker") == "FLNC" else "  "
            lines.append(
                f"{star}{c.get('ticker',''):<5} {c.get('revenue_growth_yoy','N/A'):>8} "
                f"{c.get('gross_margin','N/A'):>8} {c.get('ev_revenue','N/A'):>7}  {prof}"
            )
        lines.append("</pre>")
    wbc = comp_data.get("winner_by_category", {})
    if wbc:
        lines.append("<b>카테고리별 1위</b>")
        for cat, winner in wbc.items():
            label = {"growth": "성장률", "margin": "마진", "valuation": "밸류", "momentum": "모멘텀"}.get(cat, cat)
            lines.append(f"  {label}: {winner}")
    if comp_data.get("market_position"):
        lines.append(f"\n{comp_data['market_position'][:150]}")
    send("\n".join(lines))
    time.sleep(0.5)

    # ── 메시지 8: 매크로 지표 ────────────────────────────────────────────────
    macro = results.get("macro", {})
    macro_score = macro.get("macro_score", 0)
    macro_bar = "█" * abs(macro_score) + "░" * (5 - abs(macro_score))
    macro_dir = "+" if macro_score >= 0 else "-"
    fed_icon = {"dovish": "🕊️", "hawkish": "🦅", "neutral": "➡️"}.get(macro.get("fed_outlook", ""), "❓")

    lines = [f"🌍 <b>매크로 지표</b>  [{macro_dir}{macro_bar}] ({macro_score:+d}/5)\n"]
    lines.append(f"🏦 연준금리: <b>{macro.get('fed_rate','N/A')}</b>  {fed_icon} {macro.get('fed_outlook','N/A')}")
    lines.append(f"📊 10년물:   {macro.get('treasury_10y','N/A')}")
    lines.append(f"💸 금리 영향: {macro.get('rate_impact_on_flnc','N/A')[:80]}")
    lines.append(f"\n📋 IRA: {macro.get('ira_status','N/A')[:100]}")
    lines.append(f"🚢 관세: {macro.get('tariff_status','N/A')[:100]}")
    if macro.get("policy_tailwinds"):
        lines.append("\n🌬️ <b>정책 순풍</b>")
        for t in macro["policy_tailwinds"][:2]:
            lines.append(f"  • {t}")
    if macro.get("policy_headwinds"):
        lines.append("\n⛔ <b>정책 역풍</b>")
        for h in macro["policy_headwinds"][:2]:
            lines.append(f"  • {h}")
    if macro.get("energy_demand_outlook"):
        lines.append(f"\n⚡ AI 전력 수요: {macro['energy_demand_outlook'][:120]}")
    if macro.get("macro_summary"):
        lines.append(f"\n{macro['macro_summary'][:200]}")
    send("\n".join(lines))

    print("  [Telegram] 전송 완료 (총 8개 메시지)")


if __name__ == "__main__":
    import sys
    send_to_telegram = "--telegram" in sys.argv
    results = main()
    if send_to_telegram and results:
        print("\n  Telegram 전송 중...")
        send_telegram(results)
