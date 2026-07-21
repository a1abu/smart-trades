"""
Long-only technical + AI council trade alert bot.

Runs on a GitHub Actions cron. For each ticker: pulls candles from Alpaca
(free tier, 15-min delayed) and checks for an EMA/RSI/volume confluence.
If it fires, pulls fundamentals from Finnhub + recent news from Tavily,
asks a 4-model AI council for independent takes, has a chairman model
reconcile them, and pushes the verdict via ntfy. Fails open at every
stage past the technical check: a broken news call or a dead council
member still results in an alert, just a plainer one.
"""

import os
import json
import requests
from datetime import datetime, timezone, timedelta

# ── Config ──────────────────────────────────────────────────────────

FINNHUB_KEY = os.environ["FINNHUB_API_KEY"]
GROQ_KEY = os.environ["GROQ_API_KEY"]
OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
TAVILY_KEY = os.environ["TAVILY_API_KEY"]
NTFY_TOPIC = os.environ["NTFY_TOPIC"]
ALPACA_KEY_ID = os.environ["ALPACA_KEY_ID"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

# "5", "15", "60", or "D". Set per-workflow via the RESOLUTION env var.
RESOLUTION = os.environ.get("RESOLUTION", "D")

# Candles now come from Alpaca (free tier, 15-min delayed), not Finnhub,
# whose free tier stopped serving historical/intraday candles. Finnhub
# is still used below for fundamentals, which free tier does cover.
ALPACA_TIMEFRAME = {"5": "5Min", "15": "15Min", "60": "1Hour", "D": "1Day"}[RESOLUTION]

# Halal-screened watchlist. Edit this yourself, nothing here gets
# auto-added. Long-only, no leverage, no options, no shorting.
TICKERS = ["NVDA", "AMD", "IAU"]

# Crypto uses a different Alpaca endpoint, symbol format, and schedule
# (24/7, not tied to US market hours). See ASSET_CLASS handling in main().
CRYPTO_TICKERS = ["BTC/USD"]

EMA_LEN = 50
RSI_LEN = 14
RSI_FLOOR = 30
RSI_CEIL = 50
VOL_LEN = 20
VOL_MULT = 1.5
WINDOW_BARS = 4

# Free-tagged OpenRouter model IDs drift over time. If a council member
# starts erroring, check openrouter.ai/models for the current :free slug.
OPENROUTER_MODELS = {
    "OpenRouter / DeepSeek R1": "deepseek/deepseek-r1:free",
    "OpenRouter / Qwen3": "qwen/qwen3-235b-a22b:free",
    "OpenRouter / Gemma 3": "google/gemma-3-27b-it:free",
}

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


def fetch_candles(symbol):
    lookback_days = 200 if RESOLUTION == "D" else 14  # intraday only needs ~14 days for 50+ bars
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get(
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
        headers={"APCA-API-KEY-ID": ALPACA_KEY_ID, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
        params={"timeframe": ALPACA_TIMEFRAME, "start": start, "limit": 500, "feed": "iex", "adjustment": "raw"},
        timeout=15,
    )
    if r.status_code in (401, 403):
        print(f"{symbol}: {r.status_code} from Alpaca, check ALPACA_KEY_ID/ALPACA_SECRET_KEY. Raw response: {r.text}")
        return None
    r.raise_for_status()
    bars = r.json().get("bars")
    if not bars:
        print(f"{symbol}: no bars returned for {ALPACA_TIMEFRAME}. Raw response: {r.text}")
        return None
    return {
        "c": [b["c"] for b in bars],
        "v": [b["v"] for b in bars],
    }


def fetch_crypto_candles(symbol):
    """Alpaca's crypto endpoint: different URL, no feed param (crypto data
    isn't licensed/delayed like equities), and bars come back nested under
    the symbol rather than as a flat list."""
    lookback_days = 14
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get(
        "https://data.alpaca.markets/v1beta3/crypto/us/bars",
        headers={"APCA-API-KEY-ID": ALPACA_KEY_ID, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
        params={"symbols": symbol, "timeframe": ALPACA_TIMEFRAME, "start": start, "limit": 500},
        timeout=15,
    )
    if r.status_code in (401, 403):
        print(f"{symbol}: {r.status_code} from Alpaca crypto, check ALPACA_KEY_ID/ALPACA_SECRET_KEY. Raw response: {r.text}")
        return None
    r.raise_for_status()
    bars = r.json().get("bars", {}).get(symbol)
    if not bars:
        print(f"{symbol}: no crypto bars returned for {ALPACA_TIMEFRAME}. Raw response: {r.text}")
        return None
    return {
        "c": [b["c"] for b in bars],
        "v": [b["v"] for b in bars],
    }


def check_confluence(candles):
    closes = candles["c"]
    volumes = candles["v"]
    if len(closes) < EMA_LEN + WINDOW_BARS:
        return False, {}

    ema50 = ema(closes, EMA_LEN)
    rsi14 = rsi(closes, RSI_LEN)
    vol_avg = sma(volumes, VOL_LEN)

    def bars_since(cond_fn):
        idx = len(closes) - 1
        for back in range(WINDOW_BARS + 1):
            i = idx - back
            if i < 1:
                break
            if cond_fn(i):
                return back
        return None

    def price_cross(i):
        return closes[i - 1] <= ema50[i - 1] and closes[i] > ema50[i]

    def rsi_recovering(i):
        if rsi14[i] is None or rsi14[i - 1] is None:
            return False
        return RSI_FLOOR < rsi14[i] < RSI_CEIL and rsi14[i] > rsi14[i - 1]

    def vol_spike(i):
        return vol_avg[i] is not None and volumes[i] > vol_avg[i] * VOL_MULT

    price_hit = bars_since(price_cross)
    rsi_hit = bars_since(rsi_recovering)
    vol_hit = bars_since(vol_spike)

    triggered = price_hit is not None and rsi_hit is not None and vol_hit is not None
    snapshot = {
        "close": closes[-1],
        "ema50": round(ema50[-1], 2),
        "rsi14": round(rsi14[-1], 2) if rsi14[-1] is not None else None,
        "volume": volumes[-1],
        "vol_avg20": round(vol_avg[-1], 2) if vol_avg[-1] is not None else None,
    }
    return triggered, snapshot


# ── Context gathering (only runs on a confirmed trigger) ────────────

def fetch_fundamentals(symbol, is_crypto=False):
    if is_crypto:
        return {"note": "Traditional fundamentals (P/E, debt/equity) don't apply to crypto assets."}
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
        return {"error": str(e)}


def fetch_news(symbol):
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {TAVILY_KEY}"},
            json={
                "query": symbol,
                "topic": "news",
                "days": 7,
                "max_results": 5,
                "search_depth": "advanced",
            },
            timeout=20,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return [{"title": x["title"], "url": x["url"], "content": x["content"][:400]} for x in results]
    except Exception as e:
        return [{"error": str(e)}]


# ── AI council ───────────────────────────────────────────────────────

def build_prompt(symbol, snapshot, fundamentals, news, is_crypto=False):
    news_block = "\n".join(
        f"- {n.get('title', '?')}: {n.get('content', '')}" for n in news if "error" not in n
    ) or "No recent news found."

    if is_crypto:
        weighting = """This is a crypto asset. Traditional fundamentals like P/E or debt
ratios don't apply, ignore the fundamentals field below beyond its note.
Weight recent news and market narrative most heavily instead."""
    else:
        weighting = "Your verdict should be driven primarily by the fundamentals and news below."

    return f"""You are one of several independent analysts evaluating {symbol} for a
long-only, halal-compliant trader. A technical signal fired, that's only a
trigger to look, not evidence of anything on its own. {weighting}

Technical snapshot (context only, weight this least): {json.dumps(snapshot)}

Fundamentals: {json.dumps(fundamentals)}

Recent news, last 7 days:
{news_block}

Do the following, in order:

1. State the strongest case FOR buying, citing specific numbers or facts
from the data above. If you can't find a genuinely strong case, say so
rather than inventing one.

2. State the strongest case AGAINST buying (for holding or selling),
citing specific numbers or facts from the data above.

3. Weigh the two cases against each other and give your verdict: buy,
hold, or sell.

Rules:
- Every claim must trace to a specific number or fact given above. If the
data doesn't support a claim, don't make it.
- "Hold" is not a safe default. Only land on hold if the bull and bear
cases are genuinely close in strength, and say why. Don't pick it just to
avoid committing to a read. THAT DOES NOT MEAN that you force a "Buy" or "Sell". You base your verdict on facts.
- Don't invent facts not present in the data above.
- Skip disclaimers and hedge phrases that aren't backed by a specific
number from the data.

Keep the whole response between 50 and 250 words."""


def call_groq(prompt, model="llama-3.3-70b-versatile"):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_openrouter(prompt, model):
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def run_council(symbol, snapshot, fundamentals, news, is_crypto=False):
    prompt = build_prompt(symbol, snapshot, fundamentals, news, is_crypto=is_crypto)

    opinions = {}
    try:
        opinions["Groq / Llama 3.3 70B"] = call_groq(prompt)
    except Exception as e:
        opinions["Groq / Llama 3.3 70B"] = f"(no response: {e})"

    for name, model in OPENROUTER_MODELS.items():
        try:
            opinions[name] = call_openrouter(prompt, model)
        except Exception as e:
            opinions[name] = f"(no response: {e})"

    responded = {k: v for k, v in opinions.items() if not v.startswith("(no response")}
    if not responded:
        return None, opinions

    chairman_prompt = f"""Four analysts each did a bull case / bear case / verdict on
{symbol}, independently and without seeing each other's work. Reconcile
them into one final call, don't just tally votes.

Weigh how strong each analyst's evidence actually was, not just what they
concluded. A verdict backed by a specific number beats a verdict backed by
vague reasoning, even if more analysts landed on the other side. A 3-1
split doesn't automatically mean the 3 are right if the 1 dissenter cited
a hard number the others ignored.

Give, in order:
1. Where the analysts genuinely agree, and on what evidence.
2. Where they disagree, and which side has the stronger evidence.
3. Final verdict as the first line, exactly one word: BUY, HOLD, or SELL.
4. 2-3 sentences explaining the verdict, citing the specific evidence that
tipped it, not just "most analysts agreed."

Don't let "hold" absorb a genuine disagreement, if the evidence actually
points a direction, say so even if the analysts split.

Make sure that the response is readable but still informative, NOT ONLY bullet points but not only pure text either.

{json.dumps(responded, indent=2)}"""
    try:
        verdict = call_groq(chairman_prompt)
    except Exception as e:
        verdict = f"Chairman failed ({e}), raw opinions:\n" + json.dumps(responded, indent=2)

    return verdict, opinions


# ── Delivery ─────────────────────────────────────────────────────────

def send_alert(symbol, snapshot, verdict):
    title = f"{symbol} signal"
    body = verdict if verdict else f"Confluence fired but the AI council was unreachable.\n{json.dumps(snapshot)}"
    requests.post(
        f"https://ntfy.sh/{NTFY_TOPIC}",
        data=body.encode("utf-8"),
        headers={"Title": title},
        timeout=15,
    )


# ── Main ─────────────────────────────────────────────────────────────

def main():
    force = os.environ.get("FORCE_TRIGGER") == "1"
    asset_class = os.environ.get("ASSET_CLASS", "stocks")
    is_crypto = asset_class == "crypto"
    symbols = CRYPTO_TICKERS if is_crypto else TICKERS

    for symbol in symbols:
        candles = fetch_crypto_candles(symbol) if is_crypto else fetch_candles(symbol)
        if not candles:
            print(f"{symbol}: no candle data, skipping")
            continue

        triggered, snapshot = check_confluence(candles)
        if force:
            print(f"{symbol}: FORCE_TRIGGER set, running full pipeline regardless of confluence result")
            triggered = True
        if not triggered:
            print(f"{symbol}: no confluence this cycle")
            continue

        print(f"{symbol}: confluence triggered, gathering context")
        fundamentals = fetch_fundamentals(symbol, is_crypto=is_crypto)
        news_query = symbol.replace("/USD", "") if is_crypto else symbol
        news = fetch_news(news_query)
        verdict, opinions = run_council(symbol, snapshot, fundamentals, news, is_crypto=is_crypto)
        send_alert(symbol, snapshot, verdict)
        print(f"{symbol}: alert sent")


if __name__ == "__main__":
    main()
