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

# Two independent teams, each Analyst and Reviewer gets the same base
# context and does its own live search, reasoning entirely on its own,
# no artificial role-splitting between them. Each team's Arbiter
# reconciles its own Analyst+Reviewer. Reuse only ever happens *across*
# teams, never within one, so each team's internal disagreement always
# comes from two genuinely different models. No new API keys, everything
# below sits on Groq or OpenRouter, both already connected.
TEAMS = [
    {
        "label": "Team 1",
        "analyst": {"name": "OpenRouter / Nemotron 3 Ultra", "provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"},
        "reviewer": {"name": "OpenRouter / Qwen3", "provider": "openrouter", "model": "qwen/qwen3-235b-a22b-07-25:free"},
        "arbiter": {"name": "Groq / GPT-OSS-20B", "provider": "groq", "model": "openai/gpt-oss-20b"},
    },
    {
        "label": "Team 2",
        "analyst": {"name": "OpenRouter / Gemma 4 31B", "provider": "openrouter", "model": "google/gemma-4-31b-it:free"},
        "reviewer": {"name": "OpenRouter / Nemotron 3 Ultra", "provider": "openrouter", "model": "nvidia/nemotron-3-ultra-550b-a55b:free"},
        "arbiter": {"name": "OpenRouter / Qwen3", "provider": "openrouter", "model": "qwen/qwen3-235b-a22b-07-25:free"},
    },
]
# Reads both Arbiter rulings and verifies specific claims against live
# search, runs before the Chief Arbiter, not after, so its corrections
# can actually change the final verdict rather than just footnote it.
FACT_CHECKER_MODEL = "openai/gpt-oss-120b"
# Sees both Arbiter rulings and the fact-check report at the same time.
# Also used as Team 1's Analyst and Team 2's Reviewer, so it isn't fully
# independent of what it's judging, accepted trade-off for raw capability.
CHIEF_ARBITER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


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
    lookback_days = lookback_days if lookback_days is not None else (200 if RESOLUTION == "D" else 14)
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
    return {"c": [b["c"] for b in bars], "v": [b["v"] for b in bars]}


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
    return {"c": [b["c"] for b in bars], "v": [b["v"] for b in bars]}


def check_confluence(candles):
    closes = candles["c"]
    volumes = candles["v"]
    if len(closes) < EMA_LEN + WINDOW_BARS + 1:  # +1 so there's a prior bar to compare against
        return False, {}

    ema50 = ema(closes, EMA_LEN)
    rsi14 = rsi(closes, RSI_LEN)
    vol_avg = sma(volumes, VOL_LEN)

    def price_cross(i):
        return closes[i - 1] <= ema50[i - 1] and closes[i] > ema50[i]

    def rsi_recovering(i):
        if rsi14[i] is None or rsi14[i - 1] is None:
            return False
        return RSI_FLOOR < rsi14[i] < RSI_CEIL and rsi14[i] > rsi14[i - 1]

    def vol_spike(i):
        return vol_avg[i] is not None and volumes[i] > vol_avg[i] * VOL_MULT

    def confluence_at(idx):
        """Was confluence true looking back WINDOW_BARS from this bar?"""
        def bars_since(cond_fn):
            for back in range(WINDOW_BARS + 1):
                i = idx - back
                if i < 1:
                    return None
                if cond_fn(i):
                    return back
            return None

        return (
            bars_since(price_cross) is not None
            and bars_since(rsi_recovering) is not None
            and bars_since(vol_spike) is not None
        )

    last_idx = len(closes) - 1
    now_confluence = confluence_at(last_idx)
    prev_confluence = confluence_at(last_idx - 1)
    fresh_trigger = now_confluence and not prev_confluence  # only fire on the rising edge, not every bar it holds

    snapshot = {
        "close": closes[-1],
        "ema50": round(ema50[-1], 2),
        "rsi14": round(rsi14[-1], 2) if rsi14[-1] is not None else None,
        "volume": volumes[-1],
        "vol_avg20": round(vol_avg[-1], 2) if vol_avg[-1] is not None else None,
    }
    return fresh_trigger, snapshot


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
        try:
            r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=_SEC_HEADERS, timeout=15)
            r.raise_for_status()
            _CIK_CACHE = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in r.json().values()}
        except (requests.exceptions.RequestException, ValueError, KeyError) as e:
            print(f"Failed to fetch/parse SEC ticker list ({e}), SEC filings unavailable this run", file=sys.stderr)
            _CIK_CACHE = {}
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


def _build_past_block(similar_past):
    if not similar_past:
        return "No similar past situations in memory yet."
    past_lines = []
    for e in similar_past:
        line = f"- {e['timestamp'][:10]}: {e['text'][:200]} -> verdict was: {e['verdict'][:150]}"
        graded_parts = []
        for h, data in (e.get("outcomes") or {}).items():
            if not data.get("checked"):
                continue
            result = "CORRECT" if data.get("correct") else "WRONG" if data.get("correct") is False else "ungraded"
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
    the same pre-fetched text as every other seat. Free Tavily call,
    not OpenRouter's :online plugin, which charges per result."""
    query = symbol.replace("/USD", "") if is_crypto else symbol
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


def build_analyst_prompt(symbol, snapshot, fundamentals, news_block, past_block, own_search_block, is_crypto):
    if is_crypto:
        weighting = "This is a crypto asset, traditional fundamentals like P/E or debt ratios don't apply."
    else:
        weighting = "Your verdict should be driven primarily by the fundamentals and news below."

    return f"""You are one of two independent analysts evaluating {symbol} for a
long-only, halal-compliant trader. You're reasoning entirely on your
own, you have not seen and will not see anyone else's opinion on this.
A technical signal fired, that's only a trigger to look, not evidence of
anything on its own. {weighting}

Technical snapshot: {json.dumps(snapshot)}

Fundamentals: {json.dumps(fundamentals)}

Recent news, last 7 days:
{news_block}

Your own independent live search, just performed:
{own_search_block}

Similar past situations from memory. GRADED OUTCOMES reflect what
actually happened to the price, real evidence, weigh it seriously.
Anything ungraded is just this system's own unverified past opinion,
weigh that the lightest of everything given:
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


def build_arbiter_prompt(symbol, team_label, analyst_opinion, reviewer_opinion):
    return f"""You're the Arbiter for {team_label} evaluating {symbol}. Your Analyst
and Reviewer each independently researched this and reasoned about it on
their own, without seeing each other's work. Reconcile their two takes
into one ruling for your team, don't just average them, weigh which
one's argument is actually better supported by real evidence.

Analyst's take:
{analyst_opinion}

Reviewer's take:
{reviewer_opinion}

Give, in order:
1. Where they agree, and on what evidence.
2. Where they disagree, and which side has the stronger evidence.
3. Your team's ruling as the first line, exactly one word: BUY, HOLD, or SELL.
4. 2-3 sentences explaining why, citing the specific evidence that
decided it.

Keep it under 200 words."""


def build_factcheck_prompt(symbol, team1_ruling, team2_ruling):
    return f"""Two independent teams reached rulings on {symbol}. Your job is
verification, not opinion: check the specific factual claims in both
rulings below against live search results. Flag anything incorrect,
outdated, or unsupported, say specifically what's wrong and what's
actually true instead. If everything checks out, say so plainly, don't
invent a problem just to seem thorough.

Team 1 ruling:
{team1_ruling}

Team 2 ruling:
{team2_ruling}

Search for whatever you need to verify the specific claims made above.
Structure your response as a list of claims checked and their status.
Keep it under 200 words."""


def build_chief_arbiter_prompt(symbol, team1_ruling, team2_ruling, factcheck_report):
    return f"""You're the Chief Arbiter for {symbol}, a long-only, halal-compliant
trade alert. Two independent teams each reached their own ruling. A
fact-checker independently verified both rulings against live search and
reports what, if anything, was wrong, you're seeing that report at the
same time as the two rulings, not after.

Team 1 ruling:
{team1_ruling}

Team 2 ruling:
{team2_ruling}

Fact-check report:
{factcheck_report}

If the fact-checker flagged something as incorrect, that correction
carries real weight, it's grounded in live verification, not another
opinion. Don't just average the two team rulings, if the fact-check
changes what's actually true, let it change your verdict.

Give, in order:
1. Where the two teams agreed or disagreed, and why.
2. Whether the fact-check changes anything, and how.
3. Final verdict as the first line, exactly one word: BUY, HOLD, or SELL.
4. 2-3 sentences explaining the verdict, citing the specific evidence
that tipped it.

Make sure the response is readable but still informative, not only
bullet points but not only pure text either."""


def _sanity_check(content):
    """A real answer is more than a few characters. Catches malformed or
    truncated responses that return HTTP 200 but garbage content, so
    that can't silently poison what an Arbiter or Chief Arbiter reads."""
    if not content or len(content.strip()) < 20:
        raise ValueError(f"suspiciously short/empty response: {content!r}")
    return content


def call_groq(prompt, model="openai/gpt-oss-120b", reasoning_effort=None, use_search=False, max_retries=3):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.15}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if use_search:
        payload["tools"] = [{"type": "browser_search"}]
    for attempt in range(max_retries):
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            json=payload,
            timeout=45,
        )
        if r.status_code == 429 and attempt < max_retries - 1:
            wait = int(r.headers.get("Retry-After", 5 * (attempt + 1)))
            print(f"Groq 429, waiting {wait}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return _sanity_check(r.json()["choices"][0]["message"]["content"])


def call_openrouter(prompt, model, max_retries=3):
    for attempt in range(max_retries):
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.15},
            timeout=30,
        )
        if r.status_code == 429 and attempt < max_retries - 1:
            wait = int(r.headers.get("Retry-After", 5 * (attempt + 1)))
            print(f"OpenRouter 429, waiting {wait}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return _sanity_check(r.json()["choices"][0]["message"]["content"])


def _call_member(member, prompt, use_search=False):
    if member["provider"] == "groq":
        return call_groq(prompt, model=member["model"], use_search=use_search)
    return call_openrouter(prompt, member["model"])


def run_council(symbol, snapshot, fundamentals, news, similar_past=None, is_crypto=False):
    news_block = _build_news_block(news)
    past_block = _build_past_block(similar_past)

    all_opinions = {}
    team_rulings = {}

    for team in TEAMS:
        member_opinions = {}
        for role in ("analyst", "reviewer"):
            member = team[role]
            use_search = member["provider"] == "groq"
            own_search = "(use your search tool for anything very recent)" if use_search else fetch_seat_search(symbol, is_crypto)
            prompt = build_analyst_prompt(symbol, snapshot, fundamentals, news_block, past_block, own_search, is_crypto)
            try:
                text = _call_member(member, prompt, use_search=use_search)
            except Exception as e:
                text = f"(no response: {e})"
            member_opinions[role] = text
            all_opinions[f"{team['label']} {role.capitalize()} ({member['name']})"] = text

        arbiter = team["arbiter"]
        arb_prompt = build_arbiter_prompt(symbol, team["label"], member_opinions["analyst"], member_opinions["reviewer"])
        try:
            ruling = _call_member(arbiter, arb_prompt, use_search=False)
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
        fc_prompt = build_factcheck_prompt(symbol, t1, t2)
        factcheck_report = call_groq(fc_prompt, model=FACT_CHECKER_MODEL, use_search=True)
    except Exception as e:
        factcheck_report = f"(fact-check unavailable: {e})"
    all_opinions["Fact-checker (GPT-OSS-120B)"] = factcheck_report

    try:
        chief_prompt = build_chief_arbiter_prompt(symbol, t1, t2, factcheck_report)
        verdict = call_openrouter(chief_prompt, CHIEF_ARBITER_MODEL)
    except Exception as e:
        verdict = f"Chief Arbiter failed ({e}), team rulings:\n" + json.dumps(team_rulings, indent=2)

    return verdict, all_opinions


# ── Delivery ─────────────────────────────────────────────────────────

def send_alert(symbol, snapshot, verdict):
    title = f"{symbol} {ALPACA_TIMEFRAME} signal"
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"[{ALPACA_TIMEFRAME} chart, checked {fetched_at}, Alpaca free tier is ~15min delayed]\n\n"
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

    for i, item in enumerate(triggered_list):
        if i > 0:
            print("Pausing 20s between tickers to avoid bursting rate limits", file=sys.stderr)
            time.sleep(20)

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


def backtest():
    """One-off diagnostic, not part of the live pipeline: walks real
    historical Alpaca bars through the exact same check_confluence() that
    runs live, bar by bar, and reports how often it would have actually
    fired. Set BACKTEST_SYMBOL and optionally BACKTEST_IS_CRYPTO=1."""
    symbol = os.environ.get("BACKTEST_SYMBOL", "NVDA")
    is_crypto = os.environ.get("BACKTEST_IS_CRYPTO") == "1"

    if is_crypto:
        candles = fetch_crypto_candles(symbol, lookback_days=60, limit=5000)
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

Actual outcome at the {horizon_label} mark: price moved {magnitude_label(pct_change)} ({pct_change:+.1f}%)
since the alert, which means this verdict was wrong over that specific
horizon (it may still be graded differently at other horizons). A barely
wrong call and a sharply wrong call likely have different explanations,
weigh that in your answer, a small miss might just be reasonable
reasoning that landed on the wrong side of noise, a large one more
likely means something material was missed.

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
    elif mode == "backtest":
        backtest()
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)
