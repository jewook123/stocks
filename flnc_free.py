"""
FLNC (Fluence Energy) 무료 주식 분석
Claude API 없이 동작하는 버전

사용 라이브러리 (모두 무료, 토큰 불필요):
  - yfinance       : 주가, 재무, 뉴스, 옵션, 기관보유
  - FinBERT        : 금융 특화 감성 분석 (HuggingFace 로컬 실행)
  - vaderSentiment : 룰 기반 감성 분석 (FinBERT 실패 시 fallback)
  - SEC EDGAR API  : 공시 (완전 무료, 인증 불필요)
  - requests       : HTTP 요청
  - pandas         : 데이터 처리

실행: python3 flnc_free.py [--telegram]
"""

import os
import json
import time
import warnings
import requests
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── 설정 ──────────────────────────────────────────────────────────────────────
TARGET_TICKER = "FLNC"
FLNC_CIK      = "0001868941"          # SEC EDGAR CIK (Fluence Energy)
DAYS_BACK     = 14

EDGAR_HEADERS = {"User-Agent": "StockAnalysis jewook89@gmail.com"}

ESS_PEERS = [
    ("FLNC", "Fluence Energy"),
    ("BE",   "Bloom Energy"),
    ("STEM", "Stem Inc"),
    ("ENPH", "Enphase Energy"),
    ("SEDG", "SolarEdge"),
    ("AES",  "AES Corporation"),
    ("NEE",  "NextEra Energy"),
    ("PLUG", "Plug Power"),
    ("FCEL", "FuelCell Energy"),
    ("ARRY", "Array Technologies"),
]

MACRO_SYMBOLS = {
    "^TNX":     "미국 10년물 국채(%)",
    "^VIX":     "VIX 공포지수",
    "^GSPC":    "S&P 500",
    "DX-Y.NYB": "달러 인덱스",
    "CL=F":     "WTI 원유($/배럴)",
    "GC=F":     "금($/oz)",
}


# ── 감성 분석 엔진 (FinBERT → VADER 순서로 시도) ───────────────────────────────
_finbert_pipe = None
_vader_analyzer = None


def _get_finbert():
    global _finbert_pipe
    if _finbert_pipe is not None:
        return _finbert_pipe
    try:
        from transformers import pipeline
        print("  [FinBERT] 모델 로딩 중... (최초 실행 시 ~440MB 다운로드, 이후 캐시)")
        _finbert_pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=-1,          # CPU
            truncation=True,
            max_length=512,
        )
        print("  [FinBERT] 로딩 완료")
        return _finbert_pipe
    except Exception as e:
        print(f"  [FinBERT] 로딩 실패 ({e}) → VADER로 대체")
        _finbert_pipe = "unavailable"
        return None


def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def classify_sentiment(text: str) -> tuple[str, float]:
    """텍스트 감성 분류. (label, score) 반환. score: -1.0 ~ +1.0"""
    if not text or not text.strip():
        return "neutral", 0.0

    pipe = _get_finbert()
    if pipe and pipe != "unavailable":
        try:
            res = pipe(text[:512])[0]
            label = res["label"].lower()   # positive / negative / neutral
            conf  = res["score"]
            score_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            return label, round(score_map.get(label, 0.0) * conf, 3)
        except Exception:
            pass

    # VADER fallback
    vader = _get_vader()
    compound = vader.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return "positive", round(compound, 3)
    elif compound <= -0.05:
        return "negative", round(compound, 3)
    return "neutral", round(compound, 3)


def fmt_b(val) -> str:
    """숫자를 $B / $M 형태로 포맷"""
    if val is None:
        return "N/A"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if abs(val) >= 1e9:
        return f"${val/1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.1f}M"
    return f"${val:,.0f}"


def fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val)*100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


# ── 1. 뉴스 수집 + FinBERT 감성 분석 ─────────────────────────────────────────
def analyze_news() -> dict:
    """yfinance로 뉴스를 가져오고 FinBERT로 감성을 분류합니다."""
    import yfinance as yf

    stock   = yf.Ticker(TARGET_TICKER)
    raw     = stock.news or []
    cutoff  = (datetime.utcnow() - timedelta(days=DAYS_BACK)).timestamp()

    items = []
    for n in raw:
        content = n.get("content", {})
        title   = content.get("title") or n.get("title", "")
        pub_ts  = content.get("pubDate") or n.get("providerPublishTime", 0)
        summary = content.get("summary") or ""
        source  = content.get("provider", {}).get("displayName") or ""
        url     = content.get("canonicalUrl", {}).get("url") or ""

        if isinstance(pub_ts, str):
            try:
                pub_ts = datetime.fromisoformat(pub_ts.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pub_ts = 0

        if float(pub_ts) < cutoff:
            continue

        label, score = classify_sentiment(f"{title}. {summary}")
        items.append({
            "title":     title,
            "date":      datetime.utcfromtimestamp(float(pub_ts)).strftime("%Y-%m-%d"),
            "source":    source,
            "url":       url,
            "sentiment": label,
            "score":     score,
        })

    if not items:
        return {
            "news_items": [], "overall_sentiment": "N/A",
            "sentiment_score": 0, "positive_count": 0,
            "negative_count": 0, "neutral_count": 0,
        }

    scores  = [i["score"] for i in items]
    avg     = sum(scores) / len(scores)
    scaled  = round(avg * 10, 1)          # -10 ~ +10

    pos = sum(1 for i in items if i["sentiment"] == "positive")
    neg = sum(1 for i in items if i["sentiment"] == "negative")
    neu = len(items) - pos - neg

    overall = "Bullish" if avg >= 0.15 else "Bearish" if avg <= -0.15 else "Neutral"

    return {
        "news_items":       items,
        "overall_sentiment": overall,
        "sentiment_score":  scaled,
        "positive_count":   pos,
        "negative_count":   neg,
        "neutral_count":    neu,
        "engine":           "FinBERT" if _finbert_pipe and _finbert_pipe != "unavailable" else "VADER",
    }


# ── 2. 주가 데이터 ────────────────────────────────────────────────────────────
def fetch_price() -> dict:
    """yfinance로 현재가, 수익률, 52주 범위를 가져옵니다."""
    import yfinance as yf

    stock = yf.Ticker(TARGET_TICKER)
    h1m   = stock.history(period="1mo")
    h5d   = stock.history(period="5d")
    h1y   = stock.history(period="1y")
    info  = stock.info

    cur   = h1m["Close"].iloc[-1]  if not h1m.empty else None
    prev  = h1m["Close"].iloc[-2]  if len(h1m) >= 2 else None
    s5d   = h5d["Close"].iloc[0]   if not h5d.empty else None
    s1m   = h1m["Close"].iloc[0]   if not h1m.empty else None

    def chg(now, then):
        return round((now - then) / then * 100, 2) if now and then else None

    return {
        "current_price":  round(cur, 2)       if cur  else None,
        "change_1d_pct":  chg(cur, prev),
        "change_5d_pct":  chg(cur, s5d),
        "change_1m_pct":  chg(cur, s1m),
        "high_52w":       round(h1y["High"].max(), 2) if not h1y.empty else None,
        "low_52w":        round(h1y["Low"].min(),  2) if not h1y.empty else None,
        "volume":         int(h1m["Volume"].iloc[-1])   if not h1m.empty else None,
        "avg_volume_20d": int(h1m["Volume"].mean())     if not h1m.empty else None,
        "market_cap":     fmt_b(info.get("marketCap")),
        "beta":           round(info.get("beta", 0), 2) if info.get("beta") else "N/A",
    }


# ── 3. SEC 공시 (EDGAR API, 무료) ─────────────────────────────────────────────
def fetch_sec_filings(days: int = 60) -> dict:
    """SEC EDGAR 공개 API로 최근 공시를 가져옵니다. 토큰 불필요."""
    url = f"https://data.sec.gov/submissions/CIK{FLNC_CIK}.json"
    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"filings": [], "total_count": 0, "error": str(e)}

    recent  = data.get("filings", {}).get("recent", {})
    cutoff  = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    HIGH = {"8-K", "10-K", "10-Q"}
    MED  = {"S-3ASR", "424B7", "SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A"}

    filings = []
    for form, date, doc, acc, items in zip(
        recent.get("form",            []),
        recent.get("filingDate",      []),
        recent.get("primaryDocument", []),
        recent.get("accessionNumber", []),
        recent.get("items",           []),
    ):
        if date < cutoff:
            break
        acc_path = acc.replace("-", "")
        filings.append({
            "form_type":    form,
            "date":         date,
            "items":        items,
            "url":          f"https://www.sec.gov/Archives/edgar/data/{int(FLNC_CIK)}/{acc_path}/{doc}",
            "significance": "high" if form in HIGH else "medium" if form in MED else "low",
        })

    return {
        "filings":     filings,
        "total_count": len(filings),
        "last_10q":    next((f["date"] for f in filings if f["form_type"] == "10-Q"), None),
        "last_8k":     next((f["date"] for f in filings if f["form_type"] == "8-K"),  None),
    }


# ── 4. 재무 지표 ──────────────────────────────────────────────────────────────
def fetch_fundamentals() -> dict:
    """yfinance로 핵심 재무 지표를 가져옵니다."""
    import yfinance as yf

    stock = yf.Ticker(TARGET_TICKER)
    info  = stock.info

    # 분기 손익계산서
    rev_q = gp_q = ni_q = gm_q = fcf_q = None
    try:
        qf = stock.quarterly_financials
        if qf is not None and not qf.empty:
            col = qf.columns[0]
            rev_q = qf.loc["Total Revenue", col]  if "Total Revenue" in qf.index else None
            gp_q  = qf.loc["Gross Profit",  col]  if "Gross Profit"  in qf.index else None
            ni_q  = qf.loc["Net Income",    col]  if "Net Income"    in qf.index else None
            gm_q  = (gp_q / rev_q * 100) if rev_q and gp_q else None
    except Exception:
        pass

    try:
        cf = stock.quarterly_cashflow
        if cf is not None and not cf.empty:
            col   = cf.columns[0]
            fcf_q = cf.loc["Free Cash Flow", col] if "Free Cash Flow" in cf.index else None
    except Exception:
        pass

    pe = info.get("trailingPE")
    return {
        "market_cap":     fmt_b(info.get("marketCap")),
        "pe_ratio":       round(pe, 1) if pe else "N/A (적자)",
        "pb_ratio":       round(info.get("priceToBook", 0), 2) if info.get("priceToBook") else "N/A",
        "ps_ratio":       round(info.get("priceToSalesTrailing12Months", 0), 2) if info.get("priceToSalesTrailing12Months") else "N/A",
        "revenue_ttm":    fmt_b(info.get("totalRevenue")),
        "revenue_growth": fmt_pct(info.get("revenueGrowth")),
        "gross_margin":   fmt_pct(info.get("grossMargins")),
        "ebitda_margin":  fmt_pct(info.get("ebitdaMargins")),
        "total_cash":     fmt_b(info.get("totalCash")),
        "total_debt":     fmt_b(info.get("totalDebt")),
        "revenue_q":      fmt_b(rev_q),
        "gross_margin_q": f"{gm_q:.1f}%" if gm_q else "N/A",
        "net_income_q":   fmt_b(ni_q),
        "fcf_q":          fmt_b(fcf_q),
    }


# ── 5. 옵션 & 공매도 ──────────────────────────────────────────────────────────
def fetch_options_short() -> dict:
    """yfinance로 옵션 체인과 공매도 지표를 계산합니다."""
    import yfinance as yf

    stock = yf.Ticker(TARGET_TICKER)
    info  = stock.info

    short_pct   = info.get("shortPercentOfFloat")
    short_ratio = info.get("shortRatio")

    result = {
        "short_pct_float":   f"{short_pct*100:.1f}%" if short_pct else "N/A",
        "days_to_cover":     round(short_ratio, 1)   if short_ratio else "N/A",
        "short_trend":       "N/A",
        "short_squeeze_risk": (
            "High"   if short_pct and short_pct > 0.20 else
            "Medium" if short_pct and short_pct > 0.10 else
            "Low"
        ) if short_pct else "N/A",
        "put_call_ratio":    "N/A",
        "options_sentiment": "N/A",
        "calls_volume":      "N/A",
        "puts_volume":       "N/A",
    }

    # Put/Call Ratio — 가장 가까운 3개 만기 합산
    try:
        exps = stock.options
        if exps:
            total_calls = total_puts = 0
            for exp in exps[:3]:
                chain = stock.option_chain(exp)
                total_calls += chain.calls["volume"].fillna(0).sum()
                total_puts  += chain.puts["volume"].fillna(0).sum()
            if total_calls > 0:
                pc = round(total_puts / total_calls, 2)
                result["put_call_ratio"]    = pc
                result["calls_volume"]      = f"{int(total_calls):,}"
                result["puts_volume"]       = f"{int(total_puts):,}"
                result["options_sentiment"] = (
                    "Bearish" if pc > 1.2 else
                    "Bullish" if pc < 0.7 else
                    "Neutral"
                )
    except Exception:
        pass

    return result


# ── 6. 기관 투자자 & 내부자 ───────────────────────────────────────────────────
def fetch_institutional() -> dict:
    """yfinance로 기관 보유 현황과 내부자 거래를 가져옵니다."""
    import yfinance as yf

    stock  = yf.Ticker(TARGET_TICKER)
    result = {"top_holders": [], "insider_transactions": [], "institutional_pct": "N/A"}

    try:
        inst = stock.institutional_holders
        if inst is not None and not inst.empty:
            pct_col    = [c for c in inst.columns if "%" in str(c) or "pct" in str(c).lower() or "Pct" in str(c)]
            share_col  = [c for c in inst.columns if "share" in str(c).lower() or "Share" in str(c)]
            holder_col = [c for c in inst.columns if "holder" in str(c).lower() or "Holder" in str(c)]

            for _, row in inst.head(8).iterrows():
                holder = str(row[holder_col[0]])  if holder_col else str(row.iloc[0])
                shares = int(row[share_col[0]])   if share_col  and pd.notna(row[share_col[0]])  else 0
                pct    = float(row[pct_col[0]])   if pct_col    and pd.notna(row[pct_col[0]])    else 0

                result["top_holders"].append({
                    "name":   holder,
                    "shares": f"{shares:,}",
                    "pct":    f"{pct:.2f}%",
                })
            total_pct = inst[pct_col[0]].sum() if pct_col else 0
            result["institutional_pct"] = f"{total_pct:.1f}%"
    except Exception:
        pass

    try:
        ins = stock.insider_transactions
        if ins is not None and not ins.empty:
            for _, row in ins.head(5).iterrows():
                txn = str(row.get("Transaction", "") or row.get("transaction", "")).upper()
                result["insider_transactions"].append({
                    "date":   str(row.get("Start Date", row.get("startDate", "")))[:10],
                    "person": str(row.get("Insider Trading", row.get("insiderTrading", "")))[:30],
                    "type":   "buy" if "BUY" in txn else "sell",
                    "shares": f"{int(row.get('Shares', row.get('shares', 0)) or 0):,}",
                })
    except Exception:
        pass

    # 스마트머니 방향 추정 (내부자 매수/매도 비율)
    buys  = sum(1 for t in result["insider_transactions"] if t["type"] == "buy")
    sells = sum(1 for t in result["insider_transactions"] if t["type"] == "sell")
    result["smart_money_trend"] = (
        "Accumulating" if buys > sells else
        "Distributing" if sells > buys else
        "Neutral"
    )

    return result


# ── 7. 경쟁사 비교 ────────────────────────────────────────────────────────────
def fetch_competitors() -> list[dict]:
    """yfinance로 ESS 경쟁사들의 핵심 지표를 비교합니다."""
    import yfinance as yf

    results = []
    for sym, company in ESS_PEERS:
        try:
            stock = yf.Ticker(sym)
            info  = stock.info
            h1m   = stock.history(period="1mo")

            cur   = h1m["Close"].iloc[-1] if not h1m.empty else None
            start = h1m["Close"].iloc[0]  if not h1m.empty else None
            chg1m = round((cur - start) / start * 100, 1) if cur and start else None

            results.append({
                "ticker":         sym,
                "company":        company,
                "price":          f"${cur:.2f}"       if cur    else "N/A",
                "change_1m":      f"{chg1m:+.1f}%"    if chg1m is not None else "N/A",
                "market_cap":     fmt_b(info.get("marketCap")),
                "revenue_growth": fmt_pct(info.get("revenueGrowth")),
                "gross_margin":   fmt_pct(info.get("grossMargins")),
                "ps_ratio":       round(info.get("priceToSalesTrailing12Months", 0), 1) if info.get("priceToSalesTrailing12Months") else "N/A",
                "pe_ratio":       round(info.get("trailingPE", 0), 1) if info.get("trailingPE") else "적자",
                "short_pct":      fmt_pct(info.get("shortPercentOfFloat")),
                "beta":           round(info.get("beta", 0), 2) if info.get("beta") else "N/A",
            })
        except Exception as e:
            results.append({"ticker": sym, "company": company, "price": "오류",
                             "change_1m": "N/A", "market_cap": "N/A",
                             "revenue_growth": "N/A", "gross_margin": "N/A",
                             "ps_ratio": "N/A", "pe_ratio": "N/A",
                             "short_pct": "N/A", "beta": "N/A"})
        time.sleep(0.3)

    return results


# ── 8. 매크로 지표 ────────────────────────────────────────────────────────────
def fetch_macro() -> dict:
    """yfinance로 금리, VIX, 달러 등 매크로 지표를 가져옵니다."""
    import yfinance as yf

    indicators = {}
    for sym, name in MACRO_SYMBOLS.items():
        try:
            h5d = yf.Ticker(sym).history(period="5d")
            if not h5d.empty:
                cur  = round(h5d["Close"].iloc[-1], 2)
                prev = round(h5d["Close"].iloc[0],  2)
                chg  = round((cur - prev) / prev * 100, 2) if prev else 0
                indicators[sym] = {"name": name, "value": cur, "change_5d": chg}
        except Exception:
            indicators[sym] = {"name": name, "value": None, "change_5d": None}
        time.sleep(0.2)

    tnx = indicators.get("^TNX", {}).get("value")
    vix = indicators.get("^VIX", {}).get("value")

    # 금리 환경 평가
    if tnx:
        rate_env   = "고금리 (ESS 자금조달 부담)" if tnx > 4.5 else "중금리" if tnx > 4.0 else "저금리 (ESS 우호적)"
        rate_score = -2 if tnx > 4.5 else -1 if tnx > 4.0 else 1
    else:
        rate_env, rate_score = "N/A", 0

    # 시장 공포 평가
    if vix:
        fear_env   = "공포 구간 (VIX>30)" if vix > 30 else "불안 구간 (VIX>20)" if vix > 20 else "안정 구간"
        fear_score = -2 if vix > 30 else -1 if vix > 20 else 1
    else:
        fear_env, fear_score = "N/A", 0

    return {
        "indicators":        indicators,
        "rate_environment":  rate_env,
        "fear_environment":  fear_env,
        "macro_score":       rate_score + fear_score,
    }


# ── 출력 헬퍼 ──────────────────────────────────────────────────────────────────
def sep(char="=", w=72):  print(char * w)
def section(t):           print(f"\n{'='*72}\n  {t}\n{'='*72}")
def sub(t):               print(f"\n  ── {t} {'─'*(66-len(t))}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> dict:
    now = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
    sep("█")
    print(f"  FLNC (Fluence Energy) 무료 주식 분석 리포트")
    print(f"  분석 일시: {now}  |  엔진: yfinance + FinBERT + EDGAR API")
    sep("█")

    results = {"generated_at": now}

    # ── 1. 뉴스 감성 ──────────────────────────────────────────────────────────
    date_from = (datetime.now() - timedelta(days=DAYS_BACK)).strftime("%Y-%m-%d")
    date_to   = datetime.now().strftime("%Y-%m-%d")
    section(f"1. 뉴스 감성 분석 ({date_from} ~ {date_to}, 최근 {DAYS_BACK}일) — FinBERT / VADER")
    print("  뉴스 수집 및 감성 분류 중...")
    news_data = analyze_news()
    results["news"] = news_data

    score   = news_data["sentiment_score"]
    overall = news_data["overall_sentiment"]
    pos, neg, neu = news_data["positive_count"], news_data["negative_count"], news_data["neutral_count"]
    engine  = news_data.get("engine", "VADER")
    bar     = "█" * min(abs(int(score)), 10) + "░" * (10 - min(abs(int(score)), 10))
    direction = "+" if score >= 0 else "-"

    print(f"\n  감성 [{engine}]: {overall}  [{direction}{bar}]  ({score:+.1f}/10)")
    print(f"  뉴스 분포: 긍정 {pos}건  부정 {neg}건  중립 {neu}건")

    items = news_data.get("news_items", [])
    if items:
        sub("뉴스 목록")
        for i, n in enumerate(items, 1):
            tag = {"positive": "✅", "negative": "❌", "neutral": "─ "}.get(n["sentiment"], "?")
            print(f"  {i:2d}. {tag} [{n['date']}] {n['title'][:70]}  [{n['source']}]")
            print(f"       점수: {n['score']:+.3f}")

    # ── 2. 주가 데이터 ────────────────────────────────────────────────────────
    section("2. 주가 데이터 — yfinance")
    print("  주가 데이터 수집 중...")
    price = fetch_price()
    results["price"] = price

    p = price.get("current_price")
    print(f"\n  현재가:    ${p}"           if p else "\n  현재가:    N/A")
    print(f"  1일 수익률: {price['change_1d_pct']:+.2f}%" if price.get('change_1d_pct') is not None else "  1일 수익률: N/A")
    print(f"  5일 수익률: {price['change_5d_pct']:+.2f}%" if price.get('change_5d_pct') is not None else "  5일 수익률: N/A")
    print(f"  1개월 수익률:{price['change_1m_pct']:+.2f}%" if price.get('change_1m_pct') is not None else "  1개월 수익률: N/A")
    print(f"  52주 최고:  ${price.get('high_52w', 'N/A')}   52주 최저: ${price.get('low_52w', 'N/A')}")
    print(f"  시가총액:   {price.get('market_cap', 'N/A')}   Beta: {price.get('beta', 'N/A')}")
    print(f"  거래량:     {price['volume']:,}" if price.get("volume") else "  거래량:     N/A")

    # ── 3. SEC 공시 ───────────────────────────────────────────────────────────
    section("3. SEC 공시 — EDGAR API (무료)")
    print("  EDGAR 공시 조회 중...")
    sec = fetch_sec_filings()
    results["sec"] = sec

    print(f"\n  총 공시: {sec['total_count']}건  |  최근 10-Q: {sec.get('last_10q','N/A')}  |  최근 8-K: {sec.get('last_8k','N/A')}")

    if sec["filings"]:
        sub("공시 목록")
        print(f"  {'폼타입':<12} {'날짜':<12} {'중요도':<6} 세부")
        print(f"  {'─'*12} {'─'*12} {'─'*6} {'─'*30}")
        for f in sec["filings"][:12]:
            sig = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(f["significance"], "⚪")
            items_str = f"  Items: {f['items']}" if f.get("items") else ""
            print(f"  {f['form_type']:<12} {f['date']:<12} {sig}    {items_str}")

    # ── 4. 재무 지표 ──────────────────────────────────────────────────────────
    section("4. 재무 지표 — yfinance")
    print("  재무 데이터 수집 중...")
    fund = fetch_fundamentals()
    results["fundamentals"] = fund

    print(f"\n  {'시가총액':<14}: {fund['market_cap']}")
    print(f"  {'PER':<14}: {fund['pe_ratio']}")
    print(f"  {'PBR':<14}: {fund['pb_ratio']}")
    print(f"  {'PSR':<14}: {fund['ps_ratio']}")
    print(f"  {'매출(TTM)':<14}: {fund['revenue_ttm']}   성장률: {fund['revenue_growth']}")
    print(f"  {'총이익률':<14}: {fund['gross_margin']}")
    print(f"  {'EBITDA 마진':<14}: {fund['ebitda_margin']}")
    sub("최근 분기 실적")
    print(f"  {'매출':<14}: {fund['revenue_q']}   총이익률: {fund['gross_margin_q']}")
    print(f"  {'순이익':<14}: {fund['net_income_q']}   FCF: {fund['fcf_q']}")
    print(f"  {'현금':<14}: {fund['total_cash']}   부채: {fund['total_debt']}")

    # ── 5. 옵션 & 공매도 ──────────────────────────────────────────────────────
    section("5. Short Interest / 옵션 Put/Call — yfinance")
    print("  옵션 체인 수집 중...")
    opts = fetch_options_short()
    results["options_short"] = opts

    sq_icon  = {"High": "🔥", "Medium": "🟡", "Low": "🟢"}.get(opts.get("short_squeeze_risk", ""), "⚪")
    opt_icon = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}.get(opts.get("options_sentiment", ""), "⚪")
    print(f"\n  공매도 비율:   {opts['short_pct_float']}  |  Days to Cover: {opts['days_to_cover']}")
    print(f"  숏스퀴즈 위험: {sq_icon} {opts['short_squeeze_risk']}")
    print(f"  Put/Call:      {opts['put_call_ratio']}  |  옵션 심리: {opt_icon} {opts['options_sentiment']}")
    print(f"  콜 거래량:     {opts['calls_volume']}  |  풋 거래량: {opts['puts_volume']}")

    # ── 6. 기관 투자자 ────────────────────────────────────────────────────────
    section("6. 기관 투자자 & 내부자 — yfinance")
    print("  기관 보유 데이터 수집 중...")
    inst = fetch_institutional()
    results["institutional"] = inst

    sm_icon = {"Accumulating": "🟢", "Distributing": "🔴", "Neutral": "🟡"}.get(inst.get("smart_money_trend", ""), "⚪")
    print(f"\n  기관 보유 비율: {inst['institutional_pct']}")
    print(f"  스마트머니:     {sm_icon} {inst.get('smart_money_trend', 'N/A')}")

    if inst["top_holders"]:
        sub("주요 기관 보유 현황")
        print(f"  {'기관명':<40} {'보유비율':>8}  {'주식수':>15}")
        print(f"  {'─'*40} {'─'*8}  {'─'*15}")
        for h in inst["top_holders"]:
            print(f"  {h['name']:<40} {h['pct']:>8}  {h['shares']:>15}")

    if inst["insider_transactions"]:
        sub("내부자 거래 (Form 4)")
        for t in inst["insider_transactions"]:
            act = "매수 ✅" if t["type"] == "buy" else "매도 ❌"
            print(f"  [{t['date']}] {t['person']:<30} {act}  {t['shares']}주")

    # ── 7. 경쟁사 비교 ────────────────────────────────────────────────────────
    section("7. 경쟁사 비교 — yfinance")
    print("  경쟁사 데이터 수집 중...")
    comps = fetch_competitors()
    results["competitors"] = comps

    sub("ESS 관련주 비교")
    print(f"  {'티커':<6} {'현재가':>8} {'1개월':>7} {'시총':>9} {'매출성장':>8} {'총이익률':>8} {'P/S':>6} {'공매도':>7}  Beta")
    print("  " + "─" * 75)
    for c in comps:
        star = "★" if c["ticker"] == "FLNC" else " "
        print(
            f"  {star}{c['ticker']:<5} {c['price']:>8} {c['change_1m']:>7} "
            f"{c['market_cap']:>9} {c['revenue_growth']:>8} {c['gross_margin']:>8} "
            f"{str(c['ps_ratio']):>6} {c['short_pct']:>7}  {c['beta']}"
        )

    # ── 8. 매크로 지표 ────────────────────────────────────────────────────────
    section("8. 매크로 지표 — yfinance")
    print("  매크로 지표 수집 중...")
    macro = fetch_macro()
    results["macro"] = macro

    ms = macro["macro_score"]
    mb = "█" * abs(ms) + "░" * (5 - abs(ms))
    md = "+" if ms >= 0 else "-"
    print(f"\n  매크로 점수: [{md}{mb}]  ({ms:+d}/5)  — ESS에 {'유리' if ms >= 0 else '불리'}")
    print(f"  금리 환경:   {macro['rate_environment']}")
    print(f"  시장 심리:   {macro['fear_environment']}")

    sub("지표 현황")
    print(f"  {'지표':<22} {'현재값':>9} {'5일 변동':>9}")
    print(f"  {'─'*22} {'─'*9} {'─'*9}")
    for sym, d in macro["indicators"].items():
        val = f"{d['value']:.2f}" if d.get("value") else "N/A"
        chg = f"{d['change_5d']:+.2f}%" if d.get("change_5d") is not None else "N/A"
        print(f"  {d['name']:<22} {val:>9} {chg:>9}")

    # ── 종합 요약 ──────────────────────────────────────────────────────────────
    section("★ 종합 요약")
    p_price = price.get("current_price", "N/A")
    p_1d    = price.get("change_1d_pct")
    p_1m    = price.get("change_1m_pct")
    print(f"""
  💹 현재가:       ${p_price}  ({f'{p_1d:+.2f}%' if p_1d is not None else 'N/A'} 1일 / {f'{p_1m:+.2f}%' if p_1m is not None else 'N/A'} 1개월)
  📰 뉴스 감성:    {overall}  ({score:+.1f}/10)  [{engine}]  긍정 {pos} / 부정 {neg} / 중립 {neu}
  📋 SEC 공시:     최근 {sec['total_count']}건  (10-Q: {sec.get('last_10q','N/A')}  8-K: {sec.get('last_8k','N/A')})
  📊 재무:         매출 {fund['revenue_ttm']}  성장 {fund['revenue_growth']}  총이익률 {fund['gross_margin']}
  📉 공매도:       {opts['short_pct_float']}  DTC: {opts['days_to_cover']}일  숏스퀴즈: {opts['short_squeeze_risk']}
  🏛️  스마트머니:   {sm_icon} {inst.get('smart_money_trend','N/A')}  기관보유: {inst['institutional_pct']}
  🌍 매크로:       {ms:+d}/5  |  {macro['rate_environment']}
    """)

    # JSON 저장
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(os.path.dirname(__file__) or ".", f"flnc_free_report_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    sep("█")
    print(f"  분석 결과 저장: {path}")
    sep("█")
    print()
    return results


# ── Telegram 전송 ──────────────────────────────────────────────────────────────
def send_telegram(results: dict):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id   = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        print("  [Telegram] 환경변수 미설정, 전송 생략")
        return

    tg_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def send(text: str):
        for attempt in range(3):
            r = requests.post(tg_url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=15)
            if r.status_code == 200:
                return
            time.sleep(2 * (attempt + 1))
        print(f"  [Telegram] 실패: {r.status_code}")

    news  = results.get("news", {})
    price = results.get("price", {})
    sec   = results.get("sec", {})
    fund  = results.get("fundamentals", {})
    opts  = results.get("options_short", {})
    inst  = results.get("institutional", {})
    comps = results.get("competitors", [])
    macro = results.get("macro", {})

    # ── MSG 1: 헤더 + 가격 + 감성 ──────────────────────────────────────────
    p     = price.get("current_price", "N/A")
    d1    = price.get("change_1d_pct")
    d1m   = price.get("change_1m_pct")
    score = news.get("sentiment_score", 0)
    ovr   = news.get("overall_sentiment", "N/A")
    pos, neg, neu = news.get("positive_count",0), news.get("negative_count",0), news.get("neutral_count",0)
    engine = news.get("engine", "VADER")
    s_emoji = "🟢" if score >= 3 else "🔴" if score <= -3 else "🟡"
    now_str = results.get("generated_at", "")

    total_news = pos + neg + neu or 1
    pos_bars = round(pos / total_news * 10)
    neg_bars = round(neg / total_news * 10)
    neu_bars = 10 - pos_bars - neg_bars
    sentiment_bar = "🟢" * pos_bars + "🔴" * neg_bars + "⬜" * neu_bars

    if score >= 7:
        score_label = "강한 긍정 신호"
    elif score >= 3:
        score_label = "긍정 우세"
    elif score <= -7:
        score_label = "강한 부정 신호"
    elif score <= -3:
        score_label = "부정 우세"
    else:
        score_label = "중립 (뚜렷한 방향성 없음)"

    send(
        f"📊 <b>FLNC 무료 주식 분석</b>  <i>(yfinance+{engine})</i>\n"
        f"🗓 {now_str}\n{'─'*30}\n\n"
        f"💹 <b>${p}</b>  {f'{d1:+.2f}%' if d1 is not None else ''} (1일) / {f'{d1m:+.2f}%' if d1m is not None else ''} (1개월)\n"
        f"52주: ${price.get('low_52w','N/A')} ~ ${price.get('high_52w','N/A')}\n\n"
        f"{s_emoji} <b>감성: {ovr}</b>  점수 {score:+.1f}/10\n"
        f"{sentiment_bar}\n"
        f"✅ 긍정 {pos}건  ❌ 부정 {neg}건  ─ 중립 {neu}건\n"
        f"→ {score_label}\n\n"
        f"<b>재무 요약</b>\n"
        f"시총: {fund.get('market_cap','N/A')}  매출: {fund.get('revenue_ttm','N/A')}  성장: {fund.get('revenue_growth','N/A')}\n"
        f"총이익률: {fund.get('gross_margin','N/A')}  EBITDA: {fund.get('ebitda_margin','N/A')}\n"
        f"PER: {fund.get('pe_ratio','N/A')}  PBR: {fund.get('pb_ratio','N/A')}  PSR: {fund.get('ps_ratio','N/A')}"
    )
    time.sleep(0.5)

    # ── MSG 2: 뉴스 목록 ──────────────────────────────────────────────────
    items = news.get("news_items", [])
    if items:
        lines = [f"📰 <b>최근 뉴스 ({len(items)}건)</b>\n"]
        for n in items[:10]:
            tag = {"positive": "✅", "negative": "❌", "neutral": "─"}.get(n["sentiment"], "•")
            url = n.get("url", "")
            src = n.get("source", "")
            link = f'<a href="{url}">{src}</a>' if url else src
            sent_label = {"positive": "긍정", "negative": "부정", "neutral": "중립"}.get(n["sentiment"], "")
            lines.append(
                f"{tag} <b>[{sent_label} {n['score']:+.2f}]</b>  {n['date']}\n"
                f"    {n['title'][:70]}\n"
                f"    {link}"
            )
        send("\n".join(lines))
        time.sleep(0.5)

    # ── MSG 3: SEC 공시 ────────────────────────────────────────────────────
    SEC_FORM_DESC = {
        "8-K":      "중요 이벤트 (실적·계약·경영진 변경 등)",
        "10-K":     "연간 보고서",
        "10-Q":     "분기 보고서",
        "SC 13D":   "5%↑ 지분 취득 (적극적 의도)",
        "SC 13G":   "5%↑ 지분 취득 (수동적 의도)",
        "SC 13D/A": "지분 변동 수정 신고",
        "SC 13G/A": "지분 변동 수정 신고",
        "S-3ASR":   "증권 자동 등록 (유상증자 등)",
        "424B7":    "증권 발행 설명서",
        "DEF 14A":  "주주총회 위임장",
        "4":        "내부자 거래 보고",
        "S-8":      "임직원 주식보상 등록",
        "NT 10-Q":  "분기 보고서 제출 지연 신고",
        "NT 10-K":  "연간 보고서 제출 지연 신고",
    }
    if sec.get("total_count", 0) > 0:
        filings_list = sec.get("filings", [])
        lines = [f"📋 <b>SEC 공시 ({sec['total_count']}건)</b>\n"]
        for f in filings_list:
            sig = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(f["significance"], "⚪")
            desc = SEC_FORM_DESC.get(f["form_type"], "기타 공시")
            items_str = f"  항목: {f['items']}" if f.get("items") else ""
            lines.append(f"{sig} <b>{f['form_type']}</b>  {f['date']}  <i>{desc}</i>{items_str}")
        send("\n".join(lines))
        time.sleep(0.5)

    # ── MSG 4: 공매도 + 기관 ──────────────────────────────────────────────
    sq_risk  = opts.get("short_squeeze_risk", "N/A")
    sq_icon  = {"High": "🔥", "Medium": "🟡", "Low": "🟢"}.get(sq_risk, "⚪")
    opt_icon = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}.get(opts.get("options_sentiment",""), "⚪")
    sm_trend = inst.get("smart_money_trend", "N/A")
    sm_icon  = {"Accumulating": "🟢", "Distributing": "🔴", "Neutral": "🟡"}.get(sm_trend, "⚪")

    short_pct_val = opts.get("short_pct_float", "N/A")
    dtc_val       = opts.get("days_to_cover", "N/A")
    if sq_risk == "High":
        sq_explain = (f"  → 유동주식의 {short_pct_val}가 공매도 포지션\n"
                      f"     커버(환매)에 {dtc_val}일 소요 → 급등 시 강제 환매(숏스퀴즈) 압력 발생")
    elif sq_risk == "Medium":
        sq_explain = f"  → 유동주식의 {short_pct_val}가 공매도 (중간 수준, 모니터링 필요)"
    else:
        sq_explain = f"  → 공매도 비율 낮음 ({short_pct_val}), 숏스퀴즈 위험 제한적"

    sm_explain = {
        "Accumulating": "순매수 → 상승 베팅",
        "Distributing": "순매도 → 하락 대비 또는 차익실현",
        "Neutral":       "뚜렷한 방향성 없음",
    }.get(sm_trend, "")

    lines = ["📉 <b>공매도 / 옵션 / 기관</b>\n",
             f"공매도: {short_pct_val}  DTC(커버 소요일): {dtc_val}일",
             f"숏스퀴즈 위험: {sq_icon} <b>{sq_risk}</b>",
             sq_explain,
             f"Put/Call 비율: {opts.get('put_call_ratio','N/A')}  옵션심리: {opt_icon} {opts.get('options_sentiment','N/A')}",
             f"\n🏛️ <b>스마트머니</b>: {sm_icon} {sm_trend}  기관보유: {inst.get('institutional_pct','N/A')}",
             f"  ※ 스마트머니 = 기관·내부자 등 정보력 있는 투자자들의 매매 방향",
             f"  현재 동향: {sm_explain}" if sm_explain else ""]
    top_h = inst.get("top_holders", [])
    if top_h:
        lines.append("\n<b>주요 기관 보유</b>")
        for h in top_h[:4]:
            lines.append(f"  {h['name'][:28]}  {h['pct']}")
    ins_txn = inst.get("insider_transactions", [])
    if ins_txn:
        lines.append("\n<b>내부자 거래</b>")
        for t in ins_txn[:3]:
            lines.append(f"  {'매수✅' if t['type']=='buy' else '매도❌'}  {t['person'][:20]}  {t['shares']}주")
    send("\n".join(lines))
    time.sleep(0.5)

    # ── MSG 5: 경쟁사 비교 ────────────────────────────────────────────────
    if comps:
        lines = ["⚔️ <b>ESS 경쟁사 비교</b>\n<pre>"]
        lines.append(f"{'티커':<6} {'현재가':>8} {'1개월':>7} {'총이익률':>8} {'P/S':>5}  공매도")
        lines.append("─" * 46)
        for c in comps:
            star = "⭐" if c["ticker"] == "FLNC" else "  "
            lines.append(
                f"{star}{c['ticker']:<5} {c['price']:>8} {c['change_1m']:>7} "
                f"{c['gross_margin']:>8} {str(c['ps_ratio']):>5}  {c['short_pct']}"
            )
        lines.append("</pre>")
        send("\n".join(lines))
        time.sleep(0.5)

    # ── MSG 6: 매크로 ─────────────────────────────────────────────────────
    ms = macro.get("macro_score", 0)
    lines = [f"🌍 <b>매크로</b>  ({ms:+d}/5)\n<pre>"]
    lines.append(f"{'지표':<20} {'현재':>8} {'5일변동':>9}")
    lines.append("─" * 40)
    for sym, d in macro.get("indicators", {}).items():
        val = f"{d['value']:.2f}" if d.get("value") else "N/A"
        chg = f"{d['change_5d']:+.2f}%" if d.get("change_5d") is not None else "N/A"
        lines.append(f"{d['name']:<20} {val:>8} {chg:>9}")
    lines.append("</pre>")
    lines.append(f"\n금리환경: {macro.get('rate_environment','N/A')}")
    lines.append(f"시장심리: {macro.get('fear_environment','N/A')}")
    send("\n".join(lines))

    print("  [Telegram] 전송 완료 (총 6개 메시지)")


if __name__ == "__main__":
    import sys
    results = main()
    if "--telegram" in sys.argv and results:
        print("\n  Telegram 전송 중...")
        send_telegram(results)
