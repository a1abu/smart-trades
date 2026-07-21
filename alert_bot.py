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
ALPACA_TIMEFRAME = {"5": "5Min", "15": "15Min", "60": "1Hour", "D": "1Day"}[RESOLUTION]

# Halal-screened watchlist. Edit this yourself, nothing here gets
# auto-added. Long-only, no leverage, no options, no shorting.
TICKERS = ["NVDA", "AMD", "IAU"]
CRYPTO_TICKERS = ["BTC/USD"]

EMA_LEN = 50
RSI_LEN = 14
RSI_FLOOR = 30
RSI_CEIL = 50
VOL_LEN = 20
VOL_MULT = 1.5
WINDOW_BARS = 4

MEMORY_FILE = "memory.json"
RAG_TOP_K = 3

# How many days after an alert to check whether the verdict held up.
# Multiple horizons since a call can be right short-term and wrong
# long-term, or the reverse, one checkpoint conflates those.
HORIZONS = {"3d": 3, "7d": 7, "14d": 14, "30d": 30, "60d": 60, "90d": 90}

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
    lookback_days = 200 if RESOLUTION == "D" else 14
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get(
        f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
        headers={"APCA-API-KEY-ID": ALPACA_KEY_ID, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
        params={"timeframe": ALPACA_TIMEFRAME, "start": start, "limit": 500, "feed": "iex", "adjustment": "raw"},
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
    return {"c": [b["c"] for b in bars], "v": [b["v"] for b in bars]}


def fetch_crypto_candles(symbol):
    lookback_days = 14
    start = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get(
        "https://data.alpaca.markets/v1beta3/crypto/us/bars",
        headers={"APCA-API-KEY-ID": ALPACA_KEY_ID, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
        params={"symbols": symbol, "timeframe": ALPACA_TIMEFRAME, "start": start, "limit": 500},
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
    return {"c": [b["c"] for b in bars], "v": [b["v"] for b in bars]}


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


# ── Context gathering (only runs in analyze mode) ───────────────────

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


_CIK_CACHE = None
_SEC_HEADERS = {"User-Agent": "smart-trades-personal-bot contact@example.com"}


def _get_cik(symbol):
    global _CIK_CACHE
    if _CIK_CACHE is None:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=_SEC_HEADERS, timeout=15)
        r.raise_for_status()
        _CIK_CACHE = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in r.json().values()}
    return _CIK_CACHE.get(symbol)


def fetch_sec_filings(symbol):
    try:
        cik = _get_cik(symbol)
        if not cik:
            return []
        r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=_SEC_HEADERS, timeout=15)
        r.raise_for_status()
        recent = r.json().get("filings", {}).get("recent", {})
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).date()
        cik_int = cik.lstrip("0")
        out = []
        for form, date_str, acc, doc in zip(
            recent.get("form", []), recent.get("filingDate", []),
            recent.get("accessionNumber", []), recent.get("primaryDocument", []),
        ):
            if form != "8-K":
                continue
            if datetime.strptime(date_str, "%Y-%m-%d").date() < cutoff:
                continue
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc.replace('-', '')}/{doc}"
            out.append({"title": f"8-K filed {date_str}", "url": url, "content": "Material event disclosure, see filing."})
        return out
    except Exception as e:
        return [{"error": f"SEC EDGAR: {e}"}]


def fetch_news(symbol, is_crypto=False):
    combined = fetch_tavily_news(symbol)
    if not is_crypto:
        combined += fetch_finnhub_news(symbol)
        combined += fetch_sec_filings(symbol)
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
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


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


def add_to_memory(memory, symbol, text, verdict, close_at_alert, is_crypto):
    try:
        model = _get_model()
        emb = model.encode(text).tolist()
        memory.append({
            "symbol": symbol,
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

def build_prompt(symbol, snapshot, fundamentals, news, similar_past=None, is_crypto=False):
    news_block = "\n".join(
        f"- {n.get('title', '?')}: {n.get('content', '')}" for n in news if "error" not in n
    ) or "No recent news found."

    if similar_past:
        past_lines = []
        for e in similar_past:
            line = f"- {e['timestamp'][:10]}: {e['text'][:200]} -> verdict was: {e['verdict'][:150]}"
            graded_parts = []
            for h, data in (e.get("outcomes") or {}).items():
                if not data.get("checked"):
                    continue
                result = "CORRECT" if data.get("correct") else "WRONG" if data.get("correct") is False else "ungraded"
                part = f"{h}: {result} ({data.get('pct_change')}%)"
                if data.get("reflection"):
                    part += f" [why: {data['reflection'][:150]}]"
                graded_parts.append(part)
            if graded_parts:
                line += " | GRADED OUTCOMES -> " + "; ".join(graded_parts)
            past_lines.append(line)
        past_block = "\n".join(past_lines)
    else:
        past_block = "No similar past situations in memory yet."

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

Similar past situations from memory. GRADED OUTCOMES reflect what actually
happened to the price at specific timeframes, that's real evidence, weigh
it seriously, and note that a call can be graded differently at different
horizons (right at 3 days, wrong at 30, or the reverse). Anything without
a graded outcome yet is just this system's own past opinion, unverified,
weight that the lightest of everything given:
{past_block}

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
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_openrouter(prompt, model):
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def run_council(symbol, snapshot, fundamentals, news, similar_past=None, is_crypto=False):
    prompt = build_prompt(symbol, snapshot, fundamentals, news, similar_past=similar_past, is_crypto=is_crypto)

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


# ── Modes ────────────────────────────────────────────────────────────

def check_only():
    """Cheap phase: candles + confluence check only. Prints a single-line
    JSON list to stdout, this is what the workflow captures as a job
    output, so nothing else may print to stdout in this mode."""
    asset_class = os.environ.get("ASSET_CLASS", "stocks")
    is_crypto = asset_class == "crypto"
    symbols = CRYPTO_TICKERS if is_crypto else TICKERS
    force = os.environ.get("FORCE_TRIGGER") == "1"

    results = []
    for symbol in symbols:
        try:
            candles = fetch_crypto_candles(symbol) if is_crypto else fetch_candles(symbol)
            if not candles:
                print(f"{symbol}: no candle data, skipping", file=sys.stderr)
                continue
            triggered, snapshot = check_confluence(candles)
            if force:
                print(f"{symbol}: FORCE_TRIGGER set, forcing trigger", file=sys.stderr)
                triggered = True
            if triggered:
                results.append({"symbol": symbol, "snapshot": snapshot, "is_crypto": is_crypto})
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

    for item in triggered_list:
        symbol = item["symbol"]
        snapshot = item["snapshot"]
        is_crypto = item["is_crypto"]

        print(f"{symbol}: confluence triggered, gathering context")
        fundamentals = fetch_fundamentals(symbol, is_crypto=is_crypto)
        news_query = symbol.replace("/USD", "") if is_crypto else symbol
        news = fetch_news(news_query, is_crypto=is_crypto)

        text = situation_text(symbol, snapshot, fundamentals, news)
        similar_past = retrieve_similar(text, memory)
        print(f"{symbol}: found {len(similar_past)} similar past situations in memory")

        verdict, opinions = run_council(symbol, snapshot, fundamentals, news, similar_past=similar_past, is_crypto=is_crypto)
        send_alert(symbol, snapshot, verdict)

        memory = add_to_memory(memory, symbol, text, verdict, snapshot.get("close"), is_crypto)
        print(f"{symbol}: alert sent")

    save_memory(memory)


def parse_verdict_direction(verdict_text):
    if not verdict_text:
        return None
    head = verdict_text.upper()[:60]
    if "BUY" in head:
        return "BUY"
    if "SELL" in head:
        return "SELL"
    if "HOLD" in head:
        return "HOLD"
    return None


def grade_verdict(direction, pct_change, hold_threshold=3.0):
    if direction == "BUY":
        return pct_change > 0
    if direction == "SELL":
        return pct_change < 0
    if direction == "HOLD":
        return abs(pct_change) < hold_threshold
    return None


def build_reflection_prompt(entry, horizon_label, pct_change):
    return f"""A trading verdict was given on {entry['timestamp'][:10]} for {entry['symbol']}.

Original situation: {entry['text'][:600]}

Verdict given: {entry['verdict'][:400]}

Actual outcome at the {horizon_label} mark: price moved {pct_change:+.1f}%
since the alert, which means this verdict was wrong over that specific
horizon (it may still be graded differently at other horizons).

In 2-3 sentences, identify what in the original reasoning was likely
mistaken or what information was probably missing, specific to this
timeframe. Be specific, reference the original reasoning directly, don't
just restate that it was wrong."""


def postcheck():
    """Daily sweep: for every memory entry, check whichever horizons
    (3d/7d/14d/30d/60d/90d) have now come due and haven't been graded
    yet. Fetches each symbol's current price at most once per run, no
    matter how many entries or horizons need it, then applies that price
    across everything due. Only calls an LLM when a given horizon graded
    wrong, correct horizons get graded silently."""
    memory = load_memory()
    now = datetime.now(timezone.utc)
    updated = False
    price_cache = {}

    def current_price(symbol, is_crypto):
        if symbol not in price_cache:
            candles = fetch_crypto_candles(symbol) if is_crypto else fetch_candles(symbol)
            price_cache[symbol] = candles["c"][-1] if candles else None
        return price_cache[symbol]

    for entry in memory:
        try:
            entry_time = datetime.fromisoformat(entry["timestamp"])
        except Exception:
            continue
        age_days = (now - entry_time).total_seconds() / 86400
        symbol = entry["symbol"]
        is_crypto = entry.get("is_crypto", False)
        original_price = entry.get("close_at_alert")
        outcomes = entry.setdefault("outcomes", {h: {"checked": False} for h in HORIZONS})

        for label, horizon_days in HORIZONS.items():
            slot = outcomes.setdefault(label, {"checked": False})
            if slot.get("checked") or age_days < horizon_days:
                continue

            if not original_price:
                slot.update({"checked": True, "correct": None})
                updated = True
                continue

            price = current_price(symbol, is_crypto)
            if price is None:
                print(f"{symbol}: couldn't fetch current price this run, {label} check deferred", file=sys.stderr)
                continue

            pct_change = (price - original_price) / original_price * 100
            direction = parse_verdict_direction(entry.get("verdict"))
            correct = grade_verdict(direction, pct_change)

            slot["checked"] = True
            slot["pct_change"] = round(pct_change, 2)
            slot["correct"] = correct

            if correct is False:
                try:
                    slot["reflection"] = call_groq(build_reflection_prompt(entry, label, pct_change))[:500]
                except Exception as e:
                    print(f"{symbol}: reflection call failed ({e}) for {label}, grading without one", file=sys.stderr)

            print(f"{symbol}: {label} graded, {pct_change:+.1f}%, verdict was {'correct' if correct else 'wrong' if correct is False else 'ungraded'}")
            updated = True

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
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)
