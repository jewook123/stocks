# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated daily stock analysis tool for **Fluence Energy (FLNC)** and the broader ESS (Energy Storage Systems) sector. It calls the Claude API with web search to gather and synthesize market intelligence, then outputs a JSON report and optionally sends summaries to Telegram.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis (console output + JSON report)
python3 flnc_analysis.py

# Run with Telegram notification
python3 flnc_analysis.py --telegram
```

Output file: `flnc_report_YYYYMMDD_HHMM.json`

## Environment Variables

```bash
ANTHROPIC_API_KEY       # Required: Claude API key
TELEGRAM_BOT_TOKEN      # Optional: for Telegram delivery
TELEGRAM_CHAT_ID        # Optional: Telegram chat/channel ID
```

Authentication falls back to a Claude Code session token at `/home/claude/.claude/remote/.session_ingress_token` if `ANTHROPIC_API_KEY` is not set.

## Architecture

All logic lives in a single file: **`flnc_analysis.py`** (~1,250 lines).

### 8 Analysis Modules

Each module follows the same pattern: build a prompt → call Claude API with web search → parse JSON from response → merge into master `results` dict.

| Function | Module | Key Output |
|---|---|---|
| `analyze_news_sentiment()` | News | Sentiment score (−10 to +10), themes, catalysts |
| `check_sec_filings()` | SEC EDGAR | Recent 8-K/10-Q filings, significance ratings |
| `analyze_ess_stocks()` | Sector | Price/change data for 10 ESS stocks (2 API calls) |
| `analyze_earnings_call()` | Earnings | CEO tone score (−5 to +5), guidance, red flags |
| `analyze_short_sentiment()` | Short Interest | Short %, days-to-cover, put/call ratio, insider trades |
| `analyze_institutional()` | 13F | Institutional ownership, smart-money trend |
| `analyze_competitors()` | Competitive | FLNC vs BE/STEM/ENPH on revenue, margin, valuation |
| `analyze_macro()` | Macro | Fed rate, IRA credits, tariffs, macro score (−5 to +5) |

### API Call Details

- **Endpoint**: `https://api.anthropic.com/v1/messages`
- **Model**: `claude-sonnet-4-6`
- **Tool**: `web_search_20250305` (beta), max 5 searches per call
- **Retry**: 3 attempts, 5s delay on `ReadTimeout` or HTTP 529

### JSON Parsing Resilience

Responses are parsed with three fallbacks: full `json.loads()` → extract `{...}` block → extract specific arrays → return default empty structure with raw text.

## Automation

GitHub Actions workflow (`.github/workflows/flnc_daily.yml`):
- **Schedule**: `0 0 * * 1-5` (9 AM KST, weekdays)
- **Python**: 3.11, Ubuntu latest, 15-minute timeout
- Reports saved as artifacts for 30 days

Required GitHub Secrets: `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
