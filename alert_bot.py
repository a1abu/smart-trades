"""
Long-only technical + AI council trade alert bot, with a small RAG memory
of past triggers.

Split into two modes, run as separate GitHub Actions jobs so the heavy
embedding dependency only installs on the rare cycle that actually
triggers, not on every 5-15 minute cron tick:

  python alert_bot.py check    - cheap: candles + confluence check only,
                                  prints a JSON list of triggered tickers
                                  to stdout for the workflow to capture
  python alert_bot.py analyze  - only runs if check found something:
                                  RAG retrieval, fundamentals, news, the
                                  AI council, the alert, and saving the
                                  new entry back to memory.json

Fails open at every stage past the technical check: a broken news call
or a dead council member still results in an alert, just a plainer one.
"""

import os
import sys
import json
import time
import requests
import re
from datetime import datetime, timezone, timedelta

# ── Config ──────────────────────────────────────────────────────────

FINNHUB_KEY = os.environ["FINNHUB_API_KEY"]
OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_KEY_2 = os.environ["OPENROUTER_API_KEY_2"]
OPENROUTER_KEY_3 = os.environ["OPENROUTER_API_KEY_3"]
OPENROUTER_KEY_4 = os.environ["OPENROUTER_API_KEY_4"]
# Fallback only, never a primary provider. Every seat's model/provider/
# weight config is untouched, this only fires when a seat's normal call
# has already failed and exhausted its own retries.
NVIDIA_KEY = os.environ["NVIDIA_API_KEY"]
NVIDIA_FALLBACK_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
TAVILY_KEY = os.environ["TAVILY_API_KEY"]
NTFY_TOPIC = os.environ["NTFY_TOPIC"]
ALPACA_KEY_ID = os.environ["ALPACA_KEY_ID"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]
FMP_KEY = os.environ["FMP_API_KEY"]

# "5", "15", "60", or "D". Set per-workflow via the RESOLUTION env var.
RESOLUTION = os.environ.get("RESOLUTION", "D")
ALPACA_TIMEFRAME = {"5": "5Min", "15": "15Min", "60": "1Hour", "D": "1Day"}[RESOLUTION]

# Halal-screened watchlist. Edit this yourself, nothing here gets
# auto-added. Long-only, no leverage, no options, no shorting.
TICKERS = ["AAPL", "AMD", "GOOG"]
CRYPTO_TICKERS = [
    {"symbol": "BTC/USD", "source": "alpaca", "display": "BTC"},
    {"symbol": "PAXGUSD", "source": "kraken", "display": "PAXG"},
]

EMA_LEN = 50
RSI_LEN = 14
RSI_FLOOR = 30
RSI_CEIL = 50
RSI_BEAR_FLOOR = 50
RSI_BEAR_CEIL = 70
VOL_LEN = 20
VOL_MULT = 1.5
WINDOW_BARS = 4

MEMORY_FILE = "memory-repo/memory.json"
RAG_TOP_K = 3

# How many days after an alert to check whether the verdict held up.
# Multiple horizons since a call can be right short-term and wrong
# long-term, or the reverse, one checkpoint conflates those.
HORIZONS = {"3d": 3, "7d": 7, "14d": 14, "30d": 30, "60d": 60, "90d": 90}

# Two independent teams, each Analyst and Reviewer gets the same base
# context and does its own live search, reasoning entirely on its own,
# no artificial role-splitting between them. Each team's Arbiter
# reconciles its own Analyst+Reviewer. Reuse only ever happens *across*
# teams, never within one, so each team's internal disagreement always
# comes from two genuinely different models. Groq was removed entirely
# after repeated rate-limit failures, everything runs on OpenRouter now.
TEAMS = [
    {
        "label": "Team 1",
        "analyst": {"name": "OpenRouter / Nemotron 3 Ultra", "provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free", "key_id": "primary"},
        "reviewer": {"name": "OpenRouter / Gemma 4 31B", "provider": "openrouter", "model": "google/gemma-4-31b-it:free", "key_id": "secondary"},
        "arbiter": {"name": "OpenRouter / auto-router", "provider": "openrouter", "model": "openrouter/free", "key_id": "tertiary"},
    },
    {
        "label": "Team 2",
        "analyst": {"name": "OpenRouter / Gemma 4 31B", "provider": "openrouter", "model": "google/gemma-4-31b-it:free", "key_id": "secondary"},
        "reviewer": {"name": "OpenRouter / Nemotron 3 Ultra", "provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free", "key_id": "primary"},
        "arbiter": {"name": "OpenRouter / auto-router", "provider": "openrouter", "model": "openrouter/free", "key_id": "quaternary"},
    },
]
# Reads both Arbiter rulings and verifies specific claims against live
# search, runs before the Chief Arbiter, not after, so its corrections
# can actually change the final verdict rather than just footnote it.
FACT_CHECKER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
FACT_CHECKER_KEY = "tertiary"
# Sees both Arbiter rulings and the fact-check report at the same time.
# Also used as Team 1's Analyst and Team 2's Reviewer, so it isn't fully
# independent of what it's judging, accepted trade-off for raw capability.
CHIEF_ARBITER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
CHIEF_ARBITER_KEY = "quaternary"


# ── Technical indicators ────────────────────────────────────────────

def ema(values, length):
    k = 2 / (length + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(values, length):
    """Wilder-smoothed RSI, matches Pine's ta.rsi()."""
    out = [None] * len(values)
    if len(values) <= length:
        return out
    gains = [max(values[i] - values[i - 1], 0) for i in range(1, len(values))]
    losses = [max(values[i - 1] - values[i], 0) for i in range(1, len(values))]
    avg_gain = sum(gains[:length]) / length
    avg_loss = sum(losses[:length]) / length
    rs = avg_gain / avg_loss if avg_loss else float("inf")
    out[length] = 100 - 100 / (1 + rs)
    for i in range(length, len(gains)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        rs = avg_gain / avg_loss if avg_loss else float("inf")
        out[i + 1] = 100 - 100 / (1 + rs)
    return out


def sma(values, length):
    out = [None] * len(values)
    for i in range(length - 1, len(values)):
        out[i] = sum(values[i - length + 1 : i + 1]) / length
    return out


def fetch_candles(symbol, lookback_days=None, limit=500):
    lookback_days = lookback_days if lookback_days is not None else (300 if RESOLUTION == "D" else 14)
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get(
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
        headers={"APCA-API-KEY-ID": ALPACA_KEY_ID, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
        params={"timeframe": ALPACA_TIMEFRAME, "start": start, "limit": limit, "feed": "iex", "adjustment": "raw"},
        timeout=15,
    )
    if r.status_code in (401, 403):
        print(f"{symbol}: {r.status_code} from Alpaca, check ALPACA_KEY_ID/ALPACA_SECRET_KEY. Raw response: {r.text}", file=sys.stderr)
        return None
    r.raise_for_status()
    bars = r.json().get("bars")
    if not bars:
        print(f"{symbol}: no bars returned for {ALPACA_TIMEFRAME}. Raw response: {r.text}", file=sys.stderr)
        return None
    return {"c": [b["c"] for b in bars], "v": [b["v"] for b in bars], "h": [b["h"] for b in bars], "l": [b["l"] for b in bars]}


def fetch_crypto_candles(symbol, lookback_days=14, limit=500):
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get(
        "https://data.alpaca.markets/v1beta3/crypto/us/bars",
        headers={"APCA-API-KEY-ID": ALPACA_KEY_ID, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
        params={"symbols": symbol, "timeframe": ALPACA_TIMEFRAME, "start": start, "limit": limit},
        timeout=15,
    )
    if r.status_code in (401, 403):
        print(f"{symbol}: {r.status_code} from Alpaca crypto, check ALPACA_KEY_ID/ALPACA_SECRET_KEY. Raw response: {r.text}", file=sys.stderr)
        return None
    r.raise_for_status()
    bars = r.json().get("bars", {}).get(symbol)
    if not bars:
        print(f"{symbol}: no crypto bars returned for {ALPACA_TIMEFRAME}. Raw response: {r.text}", file=sys.stderr)
        return None
    return {"c": [b["c"] for b in bars], "v": [b["v"] for b in bars], "h": [b["h"] for b in bars], "l": [b["l"] for b in bars]}


KRAKEN_INTERVAL_MAP = {"5": 5, "15": 15, "60": 60, "D": 1440}


def fetch_kraken_candles(pair, lookback_days=None):
    """For crypto tickers Alpaca doesn't carry, PAXG specifically.
    Kraken's public OHLC endpoint, genuinely free, no key at all. Same
    output shape as fetch_crypto_candles so it's a drop-in for anything
    that already consumes that dict. Real limitation, not a bug: this
    endpoint caps out around 720 bars per request regardless of how far
    back you ask, fine for live confluence checking, shallow for deep
    backtesting."""
    interval = KRAKEN_INTERVAL_MAP.get(RESOLUTION, 15)
    r = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": pair, "interval": interval},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        print(f"{pair}: Kraken returned an error: {data['error']}", file=sys.stderr)
        return None
    result = data.get("result", {})
    # Kraken echoes the pair back under its own internal name, which can
    # differ slightly from what was requested, so just take whichever
    # key isn't "last".
    rows = next((v for k, v in result.items() if k != "last"), None)
    if not rows:
        print(f"{pair}: no Kraken OHLC rows returned. Raw response: {r.text}", file=sys.stderr)
        return None
    # [time, open, high, low, close, vwap, volume, count]
    return {
        "c": [float(row[4]) for row in rows],
        "h": [float(row[2]) for row in rows],
        "l": [float(row[3]) for row in rows],
        "v": [float(row[6]) for row in rows],
    }


def fetch_crypto_by_source(ticker, lookback_days=14):
    """Dispatches to the right provider for a crypto ticker, since not
    every crypto asset is on Alpaca, PAXG specifically isn't."""
    if ticker["source"] == "kraken":
        return fetch_kraken_candles(ticker["symbol"])
    return fetch_crypto_candles(ticker["symbol"], lookback_days=lookback_days)


def check_confluence(candles):
    closes = candles["c"]
    volumes = candles["v"]
    if len(closes) < EMA_LEN + WINDOW_BARS + 1:  # +1 so there's a prior bar to compare against
        return False, {}

    ema50 = ema(closes, EMA_LEN)
    rsi14 = rsi(closes, RSI_LEN)
    vol_avg = sma(volumes, VOL_LEN)

    def price_cross_above(i):
        return closes[i - 1] <= ema50[i - 1] and closes[i] > ema50[i]

    def price_cross_below(i):
        return closes[i - 1] >= ema50[i - 1] and closes[i] < ema50[i]

    def rsi_recovering(i):
        if rsi14[i] is None or rsi14[i - 1] is None:
            return False
        return RSI_FLOOR < rsi14[i] < RSI_CEIL and rsi14[i] > rsi14[i - 1]

    def rsi_weakening(i):
        if rsi14[i] is None or rsi14[i - 1] is None:
            return False
        return RSI_BEAR_FLOOR < rsi14[i] < RSI_BEAR_CEIL and rsi14[i] < rsi14[i - 1]

    def vol_spike(i):
        return vol_avg[i] is not None and volumes[i] > vol_avg[i] * VOL_MULT

    def bars_since_at(idx, cond_fn):
        for back in range(WINDOW_BARS + 1):
            i = idx - back
            if i < 1:
                return None
            if cond_fn(i):
                return back
        return None

    def confluence_at(idx, price_fn, rsi_fn):
        """Was confluence true looking back WINDOW_BARS from this bar?"""
        return (
            bars_since_at(idx, price_fn) is not None
            and bars_since_at(idx, rsi_fn) is not None
            and bars_since_at(idx, vol_spike) is not None
        )

    last_idx = len(closes) - 1
    now_bull = confluence_at(last_idx, price_cross_above, rsi_recovering)
    prev_bull = confluence_at(last_idx - 1, price_cross_above, rsi_recovering)
    fresh_bull = now_bull and not prev_bull  # only fire on the rising edge, not every bar it holds

    now_bear = confluence_at(last_idx, price_cross_below, rsi_weakening)
    prev_bear = confluence_at(last_idx - 1, price_cross_below, rsi_weakening)
    fresh_bear = now_bear and not prev_bear

    # Both firing in the same window is a rare, contradictory edge case
    # (price whipsawing across the EMA within WINDOW_BARS). Bullish takes
    # precedence if it somehow happens, simple, documented tie-break
    # rather than trying to report two directions from one check.
    fresh_trigger = fresh_bull or fresh_bear
    direction = "bullish" if fresh_bull else ("bearish" if fresh_bear else None)

    snapshot = {
        "close": closes[-1],
        "ema50": round(ema50[-1], 2),
        "rsi14": round(rsi14[-1], 2) if rsi14[-1] is not None else None,
        "volume": volumes[-1],
        "vol_avg20": round(vol_avg[-1], 2) if vol_avg[-1] is not None else None,
    }

    if fresh_trigger:
        snapshot["trigger_direction"] = direction
        if direction == "bullish":
            price_ago = bars_since_at(last_idx, price_cross_above)
            rsi_ago = bars_since_at(last_idx, rsi_recovering)
            vol_ago = bars_since_at(last_idx, vol_spike)
            snapshot["confluence_explanation"] = (
                f"Bullish setup. Price crossed above the {EMA_LEN}-bar EMA {price_ago} bar(s) ago. "
                f"RSI recovered into the {RSI_FLOOR}-{RSI_CEIL} zone {rsi_ago} bar(s) ago, "
                f"currently at {round(rsi14[last_idx], 2) if rsi14[last_idx] is not None else 'n/a'}. "
                f"Volume spiked above {VOL_MULT}x its {VOL_LEN}-bar average {vol_ago} bar(s) ago, "
                f"currently {volumes[last_idx]} vs a {round(vol_avg[last_idx], 0) if vol_avg[last_idx] is not None else 'n/a'} average."
            )
        else:
            price_ago = bars_since_at(last_idx, price_cross_below)
            rsi_ago = bars_since_at(last_idx, rsi_weakening)
            vol_ago = bars_since_at(last_idx, vol_spike)
            snapshot["confluence_explanation"] = (
                f"Bearish setup. Price crossed below the {EMA_LEN}-bar EMA {price_ago} bar(s) ago. "
                f"RSI weakened through the {RSI_BEAR_FLOOR}-{RSI_BEAR_CEIL} zone {rsi_ago} bar(s) ago, "
                f"currently at {round(rsi14[last_idx], 2) if rsi14[last_idx] is not None else 'n/a'}. "
                f"Volume spiked above {VOL_MULT}x its {VOL_LEN}-bar average {vol_ago} bar(s) ago, "
                f"currently {volumes[last_idx]} vs a {round(vol_avg[last_idx], 0) if vol_avg[last_idx] is not None else 'n/a'} average."
            )

    return fresh_trigger, snapshot


LONG_TREND_LEN = 200
ATR_LEN = 14
LEVELS_LEN = 20  # bars to look back for support/resistance


def atr(highs, lows, closes, length):
    if len(closes) < length + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    return sum(trs[-length:]) / length


def compute_technical_context(fetch_symbol, is_crypto, source="alpaca"):
    """Real, computed support/resistance, volatility, and longer-term
    trend, not left for a model to guess at from a single close price.
    Only called in analyze(), on a real trigger, not during the cheap
    check phase, this is enrichment for the AI council, not part of the
    confluence gate itself, which stays untouched."""
    if is_crypto:
        candles = fetch_kraken_candles(fetch_symbol) if source == "kraken" else fetch_crypto_candles(fetch_symbol)
    else:
        candles = fetch_candles(fetch_symbol)
    if not candles or len(candles.get("c", [])) < LONG_TREND_LEN + 1:
        return {"note": "not enough price history for extended technical context (support/resistance, ATR, long-term trend)"}

    closes, highs, lows = candles["c"], candles["h"], candles["l"]
    resistance = max(highs[-LEVELS_LEN:])
    support = min(lows[-LEVELS_LEN:])
    atr14 = atr(highs, lows, closes, ATR_LEN)
    ema_long = ema(closes, LONG_TREND_LEN)[-1]

    return {
        "resistance": round(resistance, 2),
        "support": round(support, 2),
        f"atr{ATR_LEN}": round(atr14, 2) if atr14 is not None else None,
        f"ema{LONG_TREND_LEN}": round(ema_long, 2),
        "long_term_trend": f"above EMA{LONG_TREND_LEN}, longer-term uptrend context" if closes[-1] > ema_long else f"below EMA{LONG_TREND_LEN}, longer-term downtrend context",
    }


# ── Context gathering (only runs in analyze mode) ───────────────────

def fetch_finnhub_fundamentals(symbol):
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/metric",
            params={"symbol": symbol, "metric": "all", "token": FINNHUB_KEY},
            timeout=15,
        )
        r.raise_for_status()
        m = r.json().get("metric", {})
        return {
            "pe": m.get("peBasicExclExtraTTM"),
            "debt_to_equity": m.get("totalDebt/totalEquityQuarterly"),
            "52w_high": m.get("52WeekHigh"),
            "52w_low": m.get("52WeekLow"),
        }
    except Exception as e:
        return {"error": f"Finnhub: {e}"}


def fetch_fmp_ratios(symbol):
    """Deeper ratio coverage than Finnhub's basic metrics: profitability,
    liquidity, leverage, all trailing-twelve-month. Free tier, 250
    calls/day, well within what this system needs."""
    try:
        r = requests.get(
            f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}",
            params={"apikey": FMP_KEY},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return {"error": "FMP: no data returned"}
        m = data[0]
        return {
            "roe": m.get("returnOnEquityTTM"),
            "current_ratio": m.get("currentRatioTTM"),
            "quick_ratio": m.get("quickRatioTTM"),
            "net_margin": m.get("netProfitMarginTTM"),
            "gross_margin": m.get("grossProfitMarginTTM"),
        }
    except Exception as e:
        return {"error": f"FMP: {e}"}


def fetch_fundamentals(symbol, is_crypto=False):
    if is_crypto:
        return {"note": "Traditional fundamentals (P/E, debt/equity) don't apply to crypto assets."}
    finnhub_data = fetch_finnhub_fundamentals(symbol)
    fmp_data = fetch_fmp_ratios(symbol)
    combined = {}
    if "error" not in finnhub_data:
        combined.update(finnhub_data)
    else:
        combined["finnhub_error"] = finnhub_data["error"]
    if "error" not in fmp_data:
        combined.update(fmp_data)
    else:
        combined["fmp_error"] = fmp_data["error"]
    return combined


def fetch_tavily_news(symbol):
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {TAVILY_KEY}"},
            json={"query": symbol, "topic": "news", "days": 7, "max_results": 5, "search_depth": "advanced"},
            timeout=20,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return [{"title": x["title"], "url": x["url"], "content": x["content"][:400]} for x in results]
    except Exception as e:
        return [{"error": f"Tavily: {e}"}]


def fetch_finnhub_news(symbol):
    try:
        to_date = datetime.now(timezone.utc).date()
        from_date = to_date - timedelta(days=7)
        r = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": symbol, "from": from_date.isoformat(), "to": to_date.isoformat(), "token": FINNHUB_KEY},
            timeout=15,
        )
        r.raise_for_status()
        results = r.json()[:5]
        return [{"title": x["headline"], "url": x["url"], "content": x.get("summary", "")[:400]} for x in results]
    except Exception as e:
        return [{"error": f"Finnhub news: {e}"}]


def fetch_news(symbol, is_crypto=False):
    combined = fetch_tavily_news(symbol)
    if not is_crypto:
        combined += fetch_finnhub_news(symbol)
    return combined


# ── RAG memory ───────────────────────────────────────────────────────

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE) as f:
            return json.load(f)
    except Exception as e:
        print(f"memory.json failed to load ({e}), starting fresh", file=sys.stderr)
        return []


def save_memory(memory):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=2)
    except (IOError, OSError, TypeError) as e:
        print(f"Failed to save memory.json ({e}), this run's memory update is lost but the alert already went out", file=sys.stderr)


def situation_text(symbol, snapshot, fundamentals, news):
    titles = "; ".join(n.get("title", "") for n in news if "error" not in n)
    return f"{symbol}: close {snapshot.get('close')}, RSI {snapshot.get('rsi14')}, volume {snapshot.get('volume')} vs avg {snapshot.get('vol_avg20')}. Fundamentals: {json.dumps(fundamentals)}. News: {titles}"


def retrieve_similar(text, memory, top_k=RAG_TOP_K):
    if not memory:
        return []
    try:
        import numpy as np
        model = _get_model()
        query_emb = model.encode(text)
        scored = []
        for entry in memory:
            past_emb = np.array(entry["embedding"])
            denom = (np.linalg.norm(query_emb) * np.linalg.norm(past_emb)) or 1e-9
            sim = float(np.dot(query_emb, past_emb) / denom)
            scored.append((sim, entry))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]
    except Exception as e:
        print(f"RAG retrieval failed ({e}), proceeding without memory context", file=sys.stderr)
        return []


def add_to_memory(memory, symbol, text, verdict, close_at_alert, is_crypto, fetch_symbol=None, source="alpaca", trigger_direction=None):
    try:
        model = _get_model()
        emb = model.encode(text).tolist()
        memory.append({
            "symbol": symbol,
            "fetch_symbol": fetch_symbol or symbol,
            "source": source,
            "trigger_direction": trigger_direction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "embedding": emb,
            "verdict": (verdict or "")[:500],
            "close_at_alert": close_at_alert,
            "is_crypto": is_crypto,
            "outcomes": {h: {"checked": False} for h in HORIZONS},
        })
    except Exception as e:
        print(f"Failed to add {symbol} to memory ({e}), skipping this entry", file=sys.stderr)
    return memory


# ── AI council ───────────────────────────────────────────────────────

def magnitude_label(pct):
    if pct is None:
        return ""
    abs_pct = abs(pct)
    if abs_pct < 1:
        return "barely"
    elif abs_pct < 3:
        return "modestly"
    elif abs_pct < 7:
        return "solidly"
    else:
        return "sharply"


def _build_news_block(news):
    return "\n".join(
        f"- {n.get('title', '?')}: {n.get('content', '')}" for n in news if "error" not in n
    ) or "No recent news found."


def _swing_grade(outcome_data):
    """Backward compatible: entries graded after the multi-timeframe
    rework carry a grades dict, older ones carry a single correct field
    that was always the swing-trade call under the hood anyway."""
    if "grades" in outcome_data:
        return outcome_data["grades"].get("swing_trade")
    return outcome_data.get("correct")


def _build_past_block(similar_past):
    if not similar_past:
        return "No similar past situations in memory yet."
    past_lines = []
    for e in similar_past:
        direction_tag = f" [{e['trigger_direction']} setup]" if e.get("trigger_direction") else ""
        line = f"- {e['timestamp'][:10]}{direction_tag}: {e['text'][:200]} -> verdict was: {e['verdict'][:150]}"
        graded_parts = []
        for h, data in (e.get("outcomes") or {}).items():
            if not data.get("checked"):
                continue
            swing_correct = _swing_grade(data)
            result = "CORRECT" if swing_correct else "WRONG" if swing_correct is False else "ungraded"
            pct = data.get("pct_change")
            part = f"{h}: {result} ({magnitude_label(pct)} {pct:+.1f}%)" if pct is not None else f"{h}: {result}"
            if data.get("reflection"):
                part += f" [why: {data['reflection'][:150]}]"
            graded_parts.append(part)
        if graded_parts:
            line += " | GRADED OUTCOMES -> " + "; ".join(graded_parts)
        past_lines.append(line)
    return "\n".join(past_lines)


def fetch_seat_search(symbol, is_crypto):
    """Independent live search for one Analyst/Reviewer seat, separate
    from the shared base news block, so each seat isn't just reading
    the same pre-fetched text as every other seat. Query is broadened
    to surface both company-specific and broader market-moving results
    in one call, not a second API call, same Tavily usage as before.
    Free Tavily call, not OpenRouter's :online plugin, which charges
    per result."""
    query = f"{symbol} stock news market impact" if not is_crypto else f"{symbol} crypto news market impact"
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {TAVILY_KEY}"},
            json={"query": query, "topic": "news", "days": 3, "max_results": 5, "search_depth": "advanced"},
            timeout=20,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return "\n".join(f"- {x['title']}: {x['content'][:300]}" for x in results) or "No independent search results found."
    except Exception as e:
        return f"(independent search unavailable: {e})"


def fetch_macro_context():
    """Fetched once per cron cycle, not per ticker, since broad macro
    conditions don't change ticker to ticker. GDELT for broad
    geopolitical/macro event coverage, neither Tavily nor Finnhub are
    built for that."""
    parts = []

    try:
        r = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={
                "query": "stock market OR federal reserve OR inflation OR interest rates",
                "mode": "artlist", "maxrecords": 5, "format": "json", "sort": "datedesc",
            },
            timeout=15,
        )
        r.raise_for_status()
        for a in r.json().get("articles", [])[:5]:
            parts.append(f"- {a.get('title', '?')} ({a.get('domain', '?')})")
    except Exception as e:
        parts.append(f"(GDELT unavailable: {e})")

    return "\n".join(parts) if parts else "No macro context available."


def build_analyst_prompt(symbol, snapshot, fundamentals, news_block, past_block, own_search_block, macro_block, is_crypto):
    if is_crypto:
        weighting = "This is a crypto asset, traditional fundamentals like P/E or debt ratios don't apply. Weigh the technical picture and the news/macro backdrop together, genuinely together, neither one primary."
    else:
        weighting = "Weigh the technical picture and the fundamentals together, genuinely together. Neither is primary, neither is background."

    direction = snapshot.get("trigger_direction", "bullish")
    if direction == "bullish":
        direction_note = "A bullish technical signal fired, price crossing above its average with RSI recovering and volume confirming. That's only a trigger to look, not evidence of anything on its own."
    else:
        direction_note = "A bearish technical signal fired, price crossing below its average with RSI weakening and volume confirming. This system is long-only, it never shorts, but a bearish signal is still real, legitimate grounds for a SELL or HOLD call, don't discount it just because there's no short to act on. That's only a trigger to look, not evidence of anything on its own."

    return f"""You are one of two independent analysts evaluating {symbol} for a
long-only, halal-compliant trader. You're reasoning entirely on your
own, you have not seen and will not see anyone else's opinion on this.
{direction_note} {weighting}

Technical snapshot: {json.dumps(snapshot)}
(includes support/resistance from the last {LEVELS_LEN} bars, {ATR_LEN}-period
ATR for volatility context, and the {LONG_TREND_LEN}-period EMA for
longer-term trend, alongside the original close/EMA50/RSI/volume, when
present, these are real computed levels, not estimates, use them. The
"confluence_explanation" field states exactly why this alert fired,
which of the three conditions triggered and how many bars ago, use it
directly, don't guess at or restate this differently. This explains why
you're looking right now, it is not, on its own, more important than
the fundamentals below.)

Fundamentals: {json.dumps(fundamentals)}

Recent news, last 7 days:
{news_block}

Your own independent live search, just performed:
{own_search_block}

Broader macro backdrop right now, current rates/inflation/employment
data and general market-moving events, not specific to {symbol}:
{macro_block}

Similar past situations from memory. GRADED OUTCOMES reflect what
actually happened to the price, real evidence, weigh it seriously.
Anything ungraded is just this system's own unverified past opinion,
weigh that the lightest of everything given:
{past_block}

Rules:
- Every one of the four timeframe verdicts below must be grounded in
BOTH the technical picture AND the fundamentals (or news/macro for
crypto), not just one. A day-trade call reasoning purely on technicals
while ignoring the fundamentals is incomplete, so is a long-term call
that ignores the technical picture entirely. Don't let either side get
crowded out at any single horizon, that includes day-trade and
long-term specifically, not just the middle two.
- Consider both the company-specific (micro) picture and the broader
macro backdrop above, don't reason about {symbol} in isolation from the
environment it's trading in, but don't force a macro angle in either if
nothing above is actually relevant to it.
- Every claim must trace to a specific number or fact given above. If the
data doesn't support a claim, don't make it. If you can't find a
genuinely strong case on one side, say so plainly rather than inventing
one to fill the section.
- "Hold" is not a safe default. Only land on hold if the bull and bear
cases are genuinely close in strength, and say why. Don't pick it just to
avoid committing to a read. THAT DOES NOT MEAN that you force a "Buy" or "Sell". You base your verdict on facts.
- Skip disclaimers and hedge phrases that aren't backed by a specific
number from the data.
- A stock can be a good day-trade and a bad long-term hold at the same
time, or the reverse. Give each timeframe its own honest verdict, don't
default to repeating the same call four times unless the data genuinely
supports that for all four.

Structure your answer exactly like this, one line per label, plain text
after each colon, no markdown formatting:

BULL CASE: [2-3 sentences, must cite at least one technical fact AND one fundamental fact, not just one side]
BEAR CASE: [2-3 sentences, must cite at least one technical fact AND one fundamental fact, not just one side]
DAY-TRADE: [BUY, HOLD, or SELL, exactly one word, next few hours to one day]
SWING-TRADE: [BUY, HOLD, or SELL, exactly one word, next few days to about two weeks]
SHORT-TERM: [BUY, HOLD, or SELL, exactly one word, next few weeks to about two months]
LONG-TERM: [BUY, HOLD, or SELL, exactly one word, several months and beyond]
REASON: [1-2 sentences, combining the technical and fundamental fact that most drove the SWING-TRADE call specifically]"""


def build_arbiter_prompt(symbol, team_label, analyst_opinion, reviewer_opinion):
    return f"""You're the Arbiter for {team_label} evaluating {symbol}. Your Analyst
and Reviewer each independently researched this and reasoned about it on
their own, without seeing each other's work. Each gave a separate
verdict for four timeframes, day-trade, swing-trade, short-term, and
long-term, since a stock can genuinely be a good short-term trade and a
bad long-term hold at once. Reconcile their two takes into one ruling
per timeframe for your team, don't just average them, weigh which one's
argument is actually better supported by real evidence, and do this
independently for each timeframe, they don't all have to resolve the
same way.

If BOTH the Analyst's and Reviewer's takes below are error messages or
otherwise contain no real analysis, you have no basis for an opinion on
any timeframe. Don't invent one. Rule HOLD across all four and say
plainly that neither input came through, that's a legitimate, honest
ruling, not a failure to reconcile.

Analyst's take:
{analyst_opinion}

Reviewer's take:
{reviewer_opinion}

Structure your answer exactly like this, one line per label, plain text
after each colon, no markdown formatting:

AGREEMENT: [where they agree, and on what evidence, or "none, only one side responded" if that's the case]
DISAGREEMENT: [where they disagree, which timeframes if it's not all of them, and which side has stronger evidence, or "n/a" if only one side responded]
DAY-TRADE: [BUY, HOLD, or SELL, exactly one word]
SWING-TRADE: [BUY, HOLD, or SELL, exactly one word]
SHORT-TERM: [BUY, HOLD, or SELL, exactly one word]
LONG-TERM: [BUY, HOLD, or SELL, exactly one word]
REASON: [2-3 sentences, citing the specific evidence that decided the SWING-TRADE ruling specifically]"""


def build_factcheck_prompt(symbol, snapshot, fundamentals, team1_ruling, team2_ruling, search_block):
    return f"""Two independent teams reached rulings on {symbol}. Your job is
verification, not opinion: check the specific factual claims in both
rulings below.

You have two separate sources, use the right one per claim:

1. Technical and fundamental figures (RSI, EMA, support/resistance,
volume, P/E, ROE, and similar) can ONLY be checked against the ground
truth data below, not web search. These are computed locally and were
never published anywhere for a search to find, that's expected, not a
gap. A claim matching this data is CONFIRMED. A claim citing a
different number than what's actually here is WRONG. Don't mark these
UNVERIFIABLE just because search doesn't surface them.

Ground truth technical snapshot: {json.dumps(snapshot)}
Ground truth fundamentals: {json.dumps(fundamentals)}

2. Everything else, news events, announcements, macro claims, named
deals or partnerships, check against the live search results below.
UNVERIFIABLE belongs here specifically, when search genuinely doesn't
cover it.

Live search results:
{search_block}

Team 1 ruling:
{team1_ruling}

Team 2 ruling:
{team2_ruling}

For each specific, checkable claim (a number, a date, a named event),
give one block in exactly this format, plain text, no markdown:

CLAIM: [the specific claim, quoted or closely paraphrased]
STATUS: [CONFIRMED, WRONG, or UNVERIFIABLE]
CORRECTION: [if WRONG, what's actually true. If CONFIRMED or
UNVERIFIABLE, write "n/a"]

Use WRONG when a claim contradicts either the ground truth data or the
search results. Use UNVERIFIABLE only for external/news/macro claims
search genuinely doesn't cover, never for technical or fundamental
figures, those always resolve to CONFIRMED or WRONG since the ground
truth data is always available. If a claim matches a different company
or ticker than {symbol}, that's WRONG, not unverifiable, say so
explicitly.

Check at most the 5 most consequential claims, the ones the verdict
actually leans on. Keep each block short."""


def build_chief_arbiter_prompt(symbol, team1_ruling, team2_ruling, factcheck_report):
    return f"""You're the Chief Arbiter for {symbol}, a long-only, halal-compliant
trade alert. Two independent teams each reached their own ruling across
four timeframes, day-trade, swing-trade, short-term, and long-term. A
fact-checker independently verified specific claims from both rulings
against live search, marking each one CONFIRMED, WRONG, or UNVERIFIABLE.
You're seeing that report at the same time as the two rulings, not after.

Team 1 ruling:
{team1_ruling}

Team 2 ruling:
{team2_ruling}

Fact-check report:
{factcheck_report}

WRONG and UNVERIFIABLE are not the same thing, don't treat them alike.
A claim marked WRONG carries real weight against whatever team made it,
it's been actively contradicted by live search. A claim marked
UNVERIFIABLE just means the search didn't cover it, that's neutral,
not evidence against the team that made it. Don't punish a team for a
claim search simply couldn't check.

Don't just average the two team rulings. If a WRONG claim undercuts the
core basis of a team's case for a given timeframe, let that change your
verdict for that timeframe specifically, it doesn't have to change all
four the same way.

Structure your answer exactly like this, one line per label, plain text
after each colon, no markdown formatting. The four verdicts must come
first, in this order:

DAY-TRADE: [BUY, HOLD, or SELL, exactly one word]
SWING-TRADE: [BUY, HOLD, or SELL, exactly one word]
SHORT-TERM: [BUY, HOLD, or SELL, exactly one word]
LONG-TERM: [BUY, HOLD, or SELL, exactly one word]
REASON: [2-3 sentences, citing the specific evidence that tipped the SWING-TRADE call, and noting if another timeframe genuinely differs and why]
TEAMS: [where the two teams agreed or disagreed, and why]
FACT-CHECK: [what was CONFIRMED, WRONG, or UNVERIFIABLE, and whether it changed any of the four verdicts]

Style: this is the only text a person actually reads, on their phone,
right now. Write plainly within each label. Never use em dashes, use a
period or comma instead. Skip AI-cliché phrasing entirely, no "in
conclusion," no "it's worth noting," no "at the end of the day," no
throat-clearing before the point. Say the thing directly."""


def _sanity_check(content):
    """A real answer is more than a few characters. Catches malformed or
    truncated responses that return HTTP 200 but garbage content, so
    that can't silently poison what an Arbiter or Chief Arbiter reads."""
    if not content or len(content.strip()) < 20:
        raise ValueError(f"suspiciously short/empty response: {content!r}")
    return content


MAX_RETRY_WAIT = 20  # seconds, hard cap regardless of what Retry-After says


def call_openrouter(prompt, model, key_id="primary", max_retries=3):
    if key_id == "secondary":
        api_key = OPENROUTER_KEY_2
    elif key_id == "tertiary":
        api_key = OPENROUTER_KEY_3
    elif key_id == "quaternary":
        api_key = OPENROUTER_KEY_4
    else:
        api_key = OPENROUTER_KEY
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.15},
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"OpenRouter request failed ({e}), waiting {wait}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        retryable = r.status_code == 429 or r.status_code >= 500
        if retryable and attempt < max_retries - 1:
            raw_wait = int(r.headers.get("Retry-After", 5 * (attempt + 1)))
            if raw_wait > MAX_RETRY_WAIT:
                print(f"OpenRouter {r.status_code} wants {raw_wait}s, over the {MAX_RETRY_WAIT}s cap, giving up now instead of blocking", file=sys.stderr)
                r.raise_for_status()
            wait = min(raw_wait, MAX_RETRY_WAIT)
            print(f"OpenRouter {r.status_code}, waiting {wait}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return _sanity_check(r.json()["choices"][0]["message"]["content"])


def call_nvidia(prompt, model=NVIDIA_FALLBACK_MODEL, max_retries=3):
    """Fallback only, called from call_with_fallback when the normal
    OpenRouter call already failed. Never called directly as a seat's
    primary provider."""
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {NVIDIA_KEY}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.15},
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"NVIDIA fallback request failed ({e}), waiting {wait}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        retryable = r.status_code == 429 or r.status_code >= 500
        if retryable and attempt < max_retries - 1:
            raw_wait = int(r.headers.get("Retry-After", 5 * (attempt + 1)))
            if raw_wait > MAX_RETRY_WAIT:
                print(f"NVIDIA fallback {r.status_code} wants {raw_wait}s, over the {MAX_RETRY_WAIT}s cap, giving up now instead of blocking", file=sys.stderr)
                r.raise_for_status()
            wait = min(raw_wait, MAX_RETRY_WAIT)
            print(f"NVIDIA fallback {r.status_code}, waiting {wait}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return _sanity_check(r.json()["choices"][0]["message"]["content"])


ALL_OPENROUTER_KEYS = ("primary", "secondary", "tertiary", "quaternary")


def call_with_fallback(prompt, model, key_id="primary", role_name="seat"):
    """Every seat goes through this now, not call_openrouter directly.
    Tries the seat's own assigned key first, exactly as configured,
    unchanged. If that fails, tries the other three OpenRouter keys with
    the same model before reaching for NVIDIA, since a different key on
    a provider we already trust beats reaching for a known-flaky one
    right away. Only if all four OpenRouter keys fail does this try
    NVIDIA as the final fallback. If that also fails, the exception
    propagates up exactly as before, the existing fail-open handling at
    each call site is untouched."""
    try:
        return call_openrouter(prompt, model, key_id=key_id)
    except Exception as e:
        print(f"{role_name}: assigned key ({key_id}) failed ({e}), trying other OpenRouter keys", file=sys.stderr)

    for backup_key in ALL_OPENROUTER_KEYS:
        if backup_key == key_id:
            continue
        try:
            result = call_openrouter(prompt, model, key_id=backup_key)
            print(f"{role_name}: backup key ({backup_key}) SUCCEEDED", file=sys.stderr)
            return result
        except Exception as e:
            print(f"{role_name}: backup key ({backup_key}) also failed ({e})", file=sys.stderr)

    print(f"{role_name}: all four OpenRouter keys failed, trying NVIDIA as last resort", file=sys.stderr)
    try:
        result = call_nvidia(prompt)
        print(f"{role_name}: NVIDIA fallback SUCCEEDED", file=sys.stderr)
        return result
    except Exception as e2:
        print(f"{role_name}: NVIDIA fallback FAILED ({e2})", file=sys.stderr)
        raise


def _call_member(member, prompt, seen_models=None):
    key_id = member.get("key_id", "primary")
    dedup_key = (key_id, member["model"])
    if seen_models is not None:
        if dedup_key in seen_models:
            print(f"{member['name']}: already called on this key earlier this run, pausing 10s to avoid a same-model rate-limit collision", file=sys.stderr)
            time.sleep(10)
        seen_models.add(dedup_key)
    return call_with_fallback(prompt, member["model"], key_id=key_id, role_name=member["name"])


def run_council(symbol, snapshot, fundamentals, news, similar_past=None, is_crypto=False, macro_block=None):
    news_block = _build_news_block(news)
    past_block = _build_past_block(similar_past)
    macro_block = macro_block or "No macro context available."

    all_opinions = {}
    team_rulings = {}
    seen_models = set()

    for team in TEAMS:
        member_opinions = {}
        for role in ("analyst", "reviewer"):
            member = team[role]
            own_search = fetch_seat_search(symbol, is_crypto)
            prompt = build_analyst_prompt(symbol, snapshot, fundamentals, news_block, past_block, own_search, macro_block, is_crypto)
            try:
                text = _call_member(member, prompt, seen_models=seen_models)
            except Exception as e:
                text = f"(no response: {e})"
            member_opinions[role] = text
            all_opinions[f"{team['label']} {role.capitalize()} ({member['name']})"] = text

        arbiter = team["arbiter"]
        arb_prompt = build_arbiter_prompt(symbol, team["label"], member_opinions["analyst"], member_opinions["reviewer"])
        try:
            ruling = _call_member(arbiter, arb_prompt, seen_models=seen_models)
        except Exception as e:
            ruling = f"(arbiter failed: {e})"
        team_rulings[team["label"]] = ruling
        all_opinions[f"{team['label']} Arbiter ({arbiter['name']})"] = ruling

    responded_rulings = {k: v for k, v in team_rulings.items() if not v.startswith("(arbiter failed")}
    if not responded_rulings:
        return None, all_opinions

    t1 = team_rulings.get("Team 1", "(unavailable)")
    t2 = team_rulings.get("Team 2", "(unavailable)")

    try:
        fc_search = fetch_seat_search(symbol, is_crypto)
        fc_prompt = build_factcheck_prompt(symbol, snapshot, fundamentals, t1, t2, fc_search)
        fc_key = (FACT_CHECKER_KEY, FACT_CHECKER_MODEL)
        if fc_key in seen_models:
            print("Fact-checker model already called on this key earlier this run, pausing 10s", file=sys.stderr)
            time.sleep(10)
        seen_models.add(fc_key)
        factcheck_report = call_with_fallback(fc_prompt, FACT_CHECKER_MODEL, key_id=FACT_CHECKER_KEY, role_name="Fact-checker")
    except Exception as e:
        factcheck_report = f"(fact-check unavailable: {e})"
    all_opinions["Fact-checker (Nemotron 3 Ultra)"] = factcheck_report

    try:
        chief_prompt = build_chief_arbiter_prompt(symbol, t1, t2, factcheck_report)
        chief_key = (CHIEF_ARBITER_KEY, CHIEF_ARBITER_MODEL)
        if chief_key in seen_models:
            print("Chief Arbiter model already called on this key earlier this run, pausing 10s", file=sys.stderr)
            time.sleep(10)
        verdict = call_with_fallback(chief_prompt, CHIEF_ARBITER_MODEL, key_id=CHIEF_ARBITER_KEY, role_name="Chief Arbiter")
    except Exception as e:
        verdict = f"Chief Arbiter failed ({e}), team rulings:\n" + json.dumps(team_rulings, indent=2)

    return verdict, all_opinions


# ── Delivery ─────────────────────────────────────────────────────────

def send_alert(symbol, snapshot, verdict, halal_screened=True):
    title = f"{symbol} {ALPACA_TIMEFRAME} signal"
    if not halal_screened:
        title = f"[NOT HALAL-SCREENED] {title}"
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"[{ALPACA_TIMEFRAME} chart, checked {fetched_at}, Alpaca free tier is ~15min delayed]\n\n"
    if not halal_screened:
        header += f"WARNING: {symbol} is NOT on your halal-screened watchlist. This is a one-off forced check only, treat it as informational, not a vetted call.\n\n"
    body = header + (verdict if verdict else f"Confluence fired but the AI council was unreachable.\n{json.dumps(snapshot)}")
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=body.encode("utf-8"),
            headers={"Title": title},
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        print(f"{symbol}: ntfy delivery failed ({e}), verdict was computed but never reached your phone", file=sys.stderr)


# ── Modes ────────────────────────────────────────────────────────────

def is_market_hours():
    """13:00-21:00 UTC, Monday-Friday, covers US market hours with some
    padding either side. This is a hard gate inside the script itself,
    independent of whatever external scheduler fired the job, so a
    misconfigured or stray trigger can't cause off-hours checks."""
    now = datetime.now(timezone.utc)
    return now.weekday() < 5 and 13 <= now.hour < 21


def check_only():
    """Cheap phase: candles + confluence check only. Prints a single-line
    JSON list to stdout, this is what the workflow captures as a job
    output, so nothing else may print to stdout in this mode."""
    asset_class = os.environ.get("ASSET_CLASS", "stocks")
    is_crypto = asset_class == "crypto"
    force = os.environ.get("FORCE_TRIGGER") == "1"
    force_tickers = {
        s.strip().upper() for s in os.environ.get("FORCE_TICKERS", "").split(",") if s.strip()
    }

    if not is_crypto and not is_market_hours() and not force and not force_tickers:
        print("Outside market hours (13-21 UTC, Mon-Fri), skipping stock check regardless of what triggered this run", file=sys.stderr)
        print(json.dumps([]))
        return

    results = []

    if is_crypto:
        for ticker in CRYPTO_TICKERS:
            symbol, display = ticker["symbol"], ticker["display"]
            try:
                candles = fetch_crypto_by_source(ticker)
                if not candles:
                    print(f"{display}: no candle data, skipping", file=sys.stderr)
                    continue
                triggered, snapshot = check_confluence(candles)
                if force:
                    print(f"{display}: FORCE_TRIGGER set, forcing trigger", file=sys.stderr)
                    triggered = True
                elif display in force_tickers or symbol in force_tickers:
                    print(f"{display}: in FORCE_TICKERS, forcing trigger", file=sys.stderr)
                    triggered = True
                if triggered:
                    results.append({
                        "symbol": display,
                        "fetch_symbol": symbol,
                        "source": ticker["source"],
                        "snapshot": snapshot,
                        "is_crypto": True,
                        "halal_screened": True,
                    })
                else:
                    print(f"{display}: no confluence this cycle", file=sys.stderr)
            except Exception as e:
                print(f"{display}: unhandled error during check ({e}), skipping this ticker, others still proceed", file=sys.stderr)
    else:
        symbols = list(TICKERS)
        extra = [s for s in force_tickers if s not in symbols]
        if extra:
            print(f"Adding {extra} for this run only, not halal-screened, not added to TICKERS", file=sys.stderr)
        symbols += extra

        for symbol in symbols:
            try:
                candles = fetch_candles(symbol)
                if not candles:
                    print(f"{symbol}: no candle data, skipping", file=sys.stderr)
                    continue
                triggered, snapshot = check_confluence(candles)
                is_extra = symbol not in TICKERS
                if force:
                    print(f"{symbol}: FORCE_TRIGGER set, forcing trigger", file=sys.stderr)
                    triggered = True
                elif is_extra or symbol in force_tickers:
                    print(f"{symbol}: in FORCE_TICKERS, forcing trigger", file=sys.stderr)
                    triggered = True
                if triggered:
                    results.append({
                        "symbol": symbol,
                        "fetch_symbol": symbol,
                        "source": "alpaca",
                        "snapshot": snapshot,
                        "is_crypto": False,
                        "halal_screened": not is_extra,
                    })
                else:
                    print(f"{symbol}: no confluence this cycle", file=sys.stderr)
            except Exception as e:
                print(f"{symbol}: unhandled error during check ({e}), skipping this ticker, others still proceed", file=sys.stderr)

    print(json.dumps(results))


def analyze():
    """Heavy phase: only runs when check found something. RAG retrieval,
    fundamentals, news, the council, the alert, and saving memory."""
    triggered_list = json.loads(os.environ.get("TRIGGERED", "[]"))
    memory = load_memory()

    print("Fetching macro context (GDELT), once for this run")
    macro_block = fetch_macro_context()

    for i, item in enumerate(triggered_list):
        if i > 0:
            print("Pausing 20s between tickers to avoid bursting rate limits", file=sys.stderr)
            time.sleep(20)

        symbol = item["symbol"]
        fetch_symbol = item.get("fetch_symbol", symbol)
        source = item.get("source", "alpaca")
        snapshot = item["snapshot"]
        is_crypto = item["is_crypto"]
        halal_screened = item.get("halal_screened", True)  # default True for older/manual payloads

        print(f"{symbol}: confluence triggered, gathering context")
        tech_context = compute_technical_context(fetch_symbol, is_crypto, source=source)
        snapshot = {**snapshot, **tech_context}
        fundamentals = fetch_fundamentals(symbol, is_crypto=is_crypto)
        news = fetch_news(symbol, is_crypto=is_crypto)

        text = situation_text(symbol, snapshot, fundamentals, news)
        similar_past = retrieve_similar(text, memory)
        print(f"{symbol}: found {len(similar_past)} similar past situations in memory")

        verdict, opinions = run_council(symbol, snapshot, fundamentals, news, similar_past=similar_past, is_crypto=is_crypto, macro_block=macro_block)
        send_alert(symbol, snapshot, verdict, halal_screened=halal_screened)

        memory = add_to_memory(memory, symbol, text, verdict, snapshot.get("close"), is_crypto, fetch_symbol=fetch_symbol, source=source, trigger_direction=snapshot.get("trigger_direction"))
        print(f"{symbol}: alert sent")

    save_memory(memory)


def backtest():
    """One-off diagnostic, not part of the live pipeline: walks real
    historical bars through the exact same check_confluence() that runs
    live, bar by bar, and reports how often it would have actually
    fired. Set BACKTEST_SYMBOL and optionally BACKTEST_IS_CRYPTO=1 and
    BACKTEST_SOURCE=kraken for tickers not on Alpaca, PAXG specifically."""
    symbol = os.environ.get("BACKTEST_SYMBOL", "NVDA")
    is_crypto = os.environ.get("BACKTEST_IS_CRYPTO") == "1"
    source = os.environ.get("BACKTEST_SOURCE", "alpaca")

    if is_crypto:
        candles = fetch_kraken_candles(symbol) if source == "kraken" else fetch_crypto_candles(symbol, lookback_days=60, limit=5000)
    else:
        candles = fetch_candles(symbol, lookback_days=60, limit=5000)

    if not candles:
        print(f"{symbol}: couldn't fetch data for backtest")
        return

    closes = candles["c"]
    volumes = candles["v"]
    total_bars = len(closes)
    min_bars = EMA_LEN + WINDOW_BARS
    if total_bars < min_bars:
        print(f"{symbol}: only {total_bars} bars available, need at least {min_bars}, can't backtest meaningfully")
        return

    fires = []
    for i in range(min_bars, total_bars + 1):
        window = {"c": closes[:i], "v": volumes[:i]}
        triggered, snapshot = check_confluence(window)
        if triggered:
            fires.append({"bar_index": i, "close": snapshot["close"]})

    rate = len(fires) / (total_bars - min_bars + 1) * 100
    print(f"{symbol} at {ALPACA_TIMEFRAME} resolution: {total_bars} bars checked, {len(fires)} would have triggered ({rate:.2f}% of eligible bars)")
    if fires:
        print("Most recent trigger points:")
        for f in fires[-15:]:
            print(f"  bar {f['bar_index']} of {total_bars}: close {f['close']}")
    else:
        print("Zero triggers across the whole window, either the setup is genuinely rare at this resolution, or the thresholds need loosening.")


def stats():
    """On-demand report of how the AI council's verdicts have actually
    performed, aggregated from memory.json's already-graded outcomes.
    Not a backtest against price data like backtest() is, this is a
    backtest against the system's own real track record. Pure math over
    what's already stored, no API calls, nothing new to fetch."""
    memory = load_memory()
    if not memory:
        print("No entries in memory.json yet, nothing to report on.")
        return

    per_horizon = {h: {"correct": 0, "wrong": 0, "ungraded": 0, "not_due": 0} for h in HORIZONS}
    per_ticker = {}
    per_direction = {"BUY": {"correct": 0, "wrong": 0}, "SELL": {"correct": 0, "wrong": 0}, "HOLD": {"correct": 0, "wrong": 0}}
    timeframes = ("day_trade", "swing_trade", "short_term")
    per_timeframe = {tf: {"correct": 0, "wrong": 0} for tf in timeframes}

    for entry in memory:
        symbol = entry.get("symbol", "?")
        per_ticker.setdefault(symbol, {h: {"correct": 0, "wrong": 0} for h in HORIZONS})
        direction = parse_verdict_direction(entry.get("verdict"))
        outcomes = entry.get("outcomes", {})

        for h in HORIZONS:
            data = outcomes.get(h, {})
            if not data.get("checked"):
                per_horizon[h]["not_due"] += 1
                continue

            swing_correct = _swing_grade(data)
            if swing_correct is True:
                per_horizon[h]["correct"] += 1
                per_ticker[symbol][h]["correct"] += 1
                if direction in per_direction:
                    per_direction[direction]["correct"] += 1
            elif swing_correct is False:
                per_horizon[h]["wrong"] += 1
                per_ticker[symbol][h]["wrong"] += 1
                if direction in per_direction:
                    per_direction[direction]["wrong"] += 1
            else:
                per_horizon[h]["ungraded"] += 1

            grades = data.get("grades")
            if grades:
                for tf in timeframes:
                    if grades.get(tf) is True:
                        per_timeframe[tf]["correct"] += 1
                    elif grades.get(tf) is False:
                        per_timeframe[tf]["wrong"] += 1
            elif swing_correct is not None:
                # Old-schema entry, only ever had the one implicit
                # swing-trade-equivalent grade.
                if swing_correct is True:
                    per_timeframe["swing_trade"]["correct"] += 1
                else:
                    per_timeframe["swing_trade"]["wrong"] += 1

    def rate_str(correct, wrong):
        graded = correct + wrong
        return f"{correct}/{graded} correct ({correct / graded * 100:.1f}%)" if graded else "no graded data yet"

    print(f"Total entries in memory.json: {len(memory)}\n")

    print("Accuracy by verdict timeframe (day-trade/swing-trade/short-term):")
    for tf in timeframes:
        d = per_timeframe[tf]
        print(f"  {tf.replace('_', '-')}: {rate_str(d['correct'], d['wrong'])}")
    print("  (older entries from before the multi-timeframe format only ever fed swing-trade here)")

    print("\nWin rate by horizon (swing-trade call, the one graded against every horizon):")
    for h in HORIZONS:
        d = per_horizon[h]
        print(f"  {h}: {rate_str(d['correct'], d['wrong'])}, {d['ungraded']} ungraded, {d['not_due']} not due yet")

    print("\nWin rate by ticker (swing-trade call):")
    for symbol, horizons_data in per_ticker.items():
        print(f"  {symbol}:")
        for h in HORIZONS:
            d = horizons_data[h]
            if d["correct"] + d["wrong"] == 0:
                continue
            print(f"    {h}: {rate_str(d['correct'], d['wrong'])}")

    print("\nWin rate by verdict direction (swing-trade call, across all horizons):")
    for direction, d in per_direction.items():
        print(f"  {direction}: {rate_str(d['correct'], d['wrong'])}")


def parse_all_verdicts(verdict_text):
    """Extract all four timeframe calls from the current prompt format.
    For memory entries from before the multi-timeframe restructuring,
    just a single VERDICT: line, that call is treated as swing_trade
    only, matching what grading already did for these entries, the
    other three stay None, not graded, not faked. Long-term is still
    requested and shown in the notification, it's just never graded
    against price outcomes in postcheck(), since our longest horizon
    check is 90 days, shorter than what "long-term" itself claims to
    mean, that restriction lives there specifically, not here."""
    empty = {"day_trade": None, "swing_trade": None, "short_term": None, "long_term": None}
    if not verdict_text:
        return dict(empty)

    labels = {"day_trade": "DAY-TRADE", "swing_trade": "SWING-TRADE", "short_term": "SHORT-TERM", "long_term": "LONG-TERM"}
    result = dict(empty)
    for key, label in labels.items():
        match = re.search(rf"{label}:\s*(BUY|HOLD|SELL)\b", verdict_text, re.IGNORECASE)
        if match:
            result[key] = match.group(1).upper()

    if any(result.values()):
        missing = [k for k, v in result.items() if v is None]
        if missing:
            print(f"parse_all_verdicts: only {4 - len(missing)}/4 labels parsed, missing {missing}, likely format drift on this entry's verdict text, grading will have real gaps for those timeframes", file=sys.stderr)
        return result

    # None of the four labels matched at all, this is a pre-restructuring
    # entry. Fall back to the old single VERDICT: label, then to a
    # generic last-match scan, same fallback chain as before, just now
    # feeding into swing_trade specifically rather than a bare return.
    labeled_old = re.search(r"VERDICT:\s*(BUY|HOLD|SELL)\b", verdict_text, re.IGNORECASE)
    if labeled_old:
        result["swing_trade"] = labeled_old.group(1).upper()
        return result
    matches = re.findall(r"\b(BUY|HOLD|SELL)\b", verdict_text)
    if matches:
        result["swing_trade"] = matches[-1]
    return result


def parse_verdict_direction(verdict_text):
    """The swing-trade call specifically, still the single grading
    target most of the system reports on. Kept as its own function since
    most callers only need this one value, not all four."""
    return parse_all_verdicts(verdict_text)["swing_trade"]


def grade_verdict(direction, pct_change, buy_sell_threshold=1.5, hold_threshold=3.0):
    if direction == "BUY":
        return pct_change > buy_sell_threshold
    if direction == "SELL":
        return pct_change < -buy_sell_threshold
    if direction == "HOLD":
        return abs(pct_change) < hold_threshold
    return None


def build_batch_reflection_prompt(items):
    """items: list of (entry, label, pct_change, grades) tuples, all
    graded wrong on swing-trade. Batches multiple reflections into one
    call instead of one call per wrong verdict, so a big backlog doesn't
    fire dozens of individual calls back to back into the same rate
    limits."""
    blocks = []
    for i, (entry, label, pct_change, grades) in enumerate(items, 1):
        other_lines = "\n".join(
            f"  - {k.replace('_', '-')}: {'correct' if v else 'wrong' if v is False else 'not applicable'}"
            for k, v in grades.items() if k != "swing_trade"
        )
        blocks.append(f"""ITEM {i}: {entry['symbol']} verdict from {entry['timestamp'][:10]}, checked at the {label} mark
Original situation: {entry['text'][:400]}
Verdict given: {entry['verdict'][:300]}
Actual outcome: price moved {magnitude_label(pct_change)} ({pct_change:+.1f}%), the swing-trade call was wrong here
Other timeframes at this same point:
{other_lines}""")

    return f"""Below are {len(items)} separate trading verdicts, each one wrong on
its swing-trade call. For EACH item, in 2-3 sentences, identify what in
the original reasoning was likely mistaken or what information was
probably missing, specific to that item, referencing its own original
reasoning directly, don't just restate that it was wrong. A barely
wrong call and a sharply wrong call likely have different explanations,
weigh that per item, don't give every item the same generic answer.

{chr(10).join(blocks)}

Structure your answer as exactly {len(items)} blocks, one per item, in
this format, plain text, no markdown:

REFLECTION 1: [your 2-3 sentence reflection for item 1]
REFLECTION 2: [your 2-3 sentence reflection for item 2]
...and so on for every item, in order, matching the item numbers above."""


def parse_batch_reflections(response_text, count):
    """Extract up to `count` numbered reflections from a batch response.
    Returns a list the same length as count, None for any that didn't
    parse, so one bit of format drift doesn't lose the reflections that
    did come through cleanly."""
    results = []
    for i in range(1, count + 1):
        match = re.search(rf"REFLECTION {i}:\s*(.+?)(?=REFLECTION {i + 1}:|$)", response_text, re.DOTALL | re.IGNORECASE)
        results.append(match.group(1).strip() if match else None)
    return results


def postcheck():
    """Daily sweep: for every memory entry, check whichever horizons
    (3d/7d/14d/30d/60d/90d) have now come due and haven't been graded
    yet. Fetches each symbol's current price at most once per run, no
    matter how many entries or horizons need it, then applies that price
    across everything due. Wrong verdicts are collected during grading
    and reflected on afterward in small batches, not one call each, a
    big backlog shouldn't fire dozens of individual calls back to back
    into the same rate limits."""
    memory = load_memory()
    now = datetime.now(timezone.utc)
    updated = False
    price_cache = {}
    pending_reflections = []

    def current_price(symbol, fetch_symbol, is_crypto, source="alpaca"):
        if symbol not in price_cache:
            if is_crypto:
                candles = fetch_kraken_candles(fetch_symbol) if source == "kraken" else fetch_crypto_candles(fetch_symbol)
            else:
                candles = fetch_candles(fetch_symbol)
            price_cache[symbol] = candles["c"][-1] if candles else None
        return price_cache[symbol]

    for entry in memory:
        try:
            entry_time = datetime.fromisoformat(entry["timestamp"])
        except Exception:
            continue
        age_days = (now - entry_time).total_seconds() / 86400
        symbol = entry["symbol"]
        fetch_symbol = entry.get("fetch_symbol", symbol)
        source = entry.get("source", "alpaca")
        is_crypto = entry.get("is_crypto", False)
        original_price = entry.get("close_at_alert")
        outcomes = entry.setdefault("outcomes", {h: {"checked": False} for h in HORIZONS})

        for label, horizon_days in HORIZONS.items():
            slot = outcomes.setdefault(label, {"checked": False})
            if slot.get("checked") or age_days < horizon_days:
                continue

            if not original_price:
                slot.update({"checked": True, "grades": {k: None for k in ("day_trade", "swing_trade", "short_term", "long_term")}})
                updated = True
                continue

            price = current_price(symbol, fetch_symbol, is_crypto, source=source)
            if price is None:
                print(f"{symbol}: couldn't fetch current price this run, {label} check deferred", file=sys.stderr)
                continue

            pct_change = (price - original_price) / original_price * 100
            directions = parse_all_verdicts(entry.get("verdict"))
            grades = {k: (grade_verdict(v, pct_change) if v else None) for k, v in directions.items()}
            grades["long_term"] = None  # shown in the notification, never graded, our 90d ceiling can't test it

            slot["checked"] = True
            slot["pct_change"] = round(pct_change, 2)
            slot["grades"] = grades

            if grades.get("swing_trade") is False:
                pending_reflections.append((slot, entry, label, pct_change, grades))

            summary = ", ".join(f"{k}={'correct' if v else 'wrong' if v is False else 'n/a'}" for k, v in grades.items())
            print(f"{symbol}: {label} graded, {pct_change:+.1f}%, {summary}")
            updated = True

    REFLECTION_BATCH_SIZE = 5
    for i in range(0, len(pending_reflections), REFLECTION_BATCH_SIZE):
        if i > 0:
            time.sleep(5)
        batch = pending_reflections[i:i + REFLECTION_BATCH_SIZE]
        items = [(entry, label, pct_change, grades) for (slot, entry, label, pct_change, grades) in batch]
        try:
            prompt = build_batch_reflection_prompt(items)
            response = call_with_fallback(prompt, FACT_CHECKER_MODEL, role_name=f"Reflection batch ({len(batch)} items)")
            reflections = parse_batch_reflections(response, len(batch))
            for (slot, entry, label, pct_change, grades), reflection in zip(batch, reflections):
                if reflection:
                    slot["reflection"] = reflection[:500]
                else:
                    print(f"{entry['symbol']}: batch reflection didn't parse for {label}, grading without one", file=sys.stderr)
        except Exception as e:
            print(f"Reflection batch failed entirely ({e}), {len(batch)} item(s) left without a reflection this run", file=sys.stderr)

    if updated:
        save_memory(memory)
    else:
        print("nothing due for postcheck this run")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        check_only()
    elif mode == "analyze":
        analyze()
    elif mode == "postcheck":
        postcheck()
    elif mode == "backtest":
        backtest()
    elif mode == "stats":
        stats()
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)
