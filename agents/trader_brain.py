"""
trader_brain.py — AI Agent 1: The Trader Brain.

Scoring model (v3.1):
  Phase 1 — Market Context:   max 73 pts (news included) / 65 pts (news excluded)
  Phase 2 — SMC Confirmation: max 35 pts
  Grand total:               108 pts (news incl) / 100 pts (news excl)

  All thresholds are percentage-based and auto-computed from settings.py.
  Python validates scores after AI responds — AI cannot cheat.
"""

import json

from loguru import logger
from openai import AzureOpenAI, OpenAI

from config import settings
from core.models import (
    MarketDataPayload,
    Phase1Scores,
    Phase2Scores,
    TraderDecision,
)
from services.data_collector import format_data_for_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# LLM CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

def _get_llm_client() -> OpenAI | AzureOpenAI:
    if settings.LLM_PROVIDER == "azure":
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            raise ValueError("Azure: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set")
        return AzureOpenAI(
            api_key        = settings.AZURE_OPENAI_API_KEY,
            azure_endpoint = settings.AZURE_OPENAI_ENDPOINT,
            api_version    = settings.AZURE_OPENAI_API_VERSION,
        )
    return OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)


def _get_model_name() -> str:
    if settings.LLM_PROVIDER == "azure":
        return settings.AZURE_OPENAI_DEPLOYMENT
    return settings.LLM_MODEL


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — fully dynamic from settings.py
# ═══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt(
    max_sl_pips: dict[str, int],
    previous_decision: dict | None = None,
) -> str:

    # ── Threshold numbers ────────────────────────────────────────────────────
    p1_max_std    = settings.PHASE1_MAX_SCORE           # 73
    p1_min_std    = settings.get_phase1_min_required()   # 44
    p1_max_excl   = settings.get_effective_p1_max(True)  # 65
    p1_min_excl   = settings.get_phase1_min_required(True)  # 39
    p2_max        = settings.PHASE2_MAX_SCORE            # 35
    p2_min        = settings.get_phase2_min_required()   # 16
    total_max_std = settings.get_total_max_score()       # 108
    total_min_std = settings.get_total_min_required()    # 60 (55%)
    total_max_excl = settings.get_total_max_score(True)  # 100
    total_min_excl = settings.get_total_min_required(True)  # 55

    p1_pct  = int(settings.PHASE1_MIN_PCT * 100)    # 60
    p2_pct  = int(settings.PHASE2_MIN_PCT * 100)    # 45
    tot_pct = int(settings.TOTAL_MIN_SCORE_PCT * 100)  # 55

    # ── R:R tier text ────────────────────────────────────────────────────────
    rr_lines = []
    for tier in settings.RR_TIERS:
        lo = int(tier["min_pct"] * 100)
        hi = int(tier["max_pct"] * 100) if tier["max_pct"] < 1.01 else 100
        rr_lines.append(
            f"  Score {lo}%–{hi}% → minimum R:R 1:{tier['rr_ratio']}"
            f"  (NO maximum — take 1:4 or 1:5 if structure offers it)"
        )
    rr_desc = "\n".join(rr_lines)

    # ── SL budget ────────────────────────────────────────────────────────────
    sl_budget_desc = "\n".join(
        f"  {pair}: max {pips}pips SL at 0.01 lot"
        for pair, pips in max_sl_pips.items()
    )

    # ── Previous scan context ────────────────────────────────────────────────
    prev_context = ""
    if previous_decision:
        prev_total = previous_decision.get("total_score", 0)
        prev_pct   = settings.get_score_pct(prev_total) * 100 if prev_total else 0
        prev_context = f"""
═══ PREVIOUS SCAN — justify any large score changes ═══
Decision: {previous_decision.get('decision', 'N/A')} | Pair: {previous_decision.get('pair', 'N/A')}
P1: {previous_decision.get('phase1_total', '?')}/{p1_max_std} | P2: {previous_decision.get('phase2_total', '?')}/{p2_max} | Total: {prev_total}/{total_max_std} ({prev_pct:.0f}%)
⚠️ If scores changed significantly, explain WHAT in the market changed. Never inflate to hit TAKE.
"""

    return f"""You are an elite Smart Money / ICT forex and gold trader AI. Score every pair honestly.

═══════════════════════════════════════════════════════
PHASE 1 — MARKET CONTEXT
Max: {p1_max_std}pts (news included) / {p1_max_excl}pts (news excluded)
Min to pass: {p1_min_std}/{p1_max_std} with news | {p1_min_excl}/{p1_max_excl} without news ({p1_pct}%)
═══════════════════════════════════════════════════════

1. WEEKLY + 4H BIAS — EMA Stack Alignment   [{settings.P1_WEEKLY_4H_BIAS_POINTS}pts MAX — HIGHEST WEIGHT]
   Weekly EMA (20>50>200) stacked AND 4H stack fully aligned same direction = 17–20
   Weekly clear, 4H mostly aligned (converging but not stacked) = 12–16
   Weekly clear but 4H mixed/sideways = 7–11
   Weekly and 4H contradicting = 0–6

2. REGIME — ADX + DI Context   [{settings.P1_REGIME_POINTS}pts MAX]
   ADX > 30, DI gap > 10pts = 12–15 (strong trend)
   ADX 25–30, clear DI direction = 8–11
   ADX 20–25 transitional = 4–7
   ADX < 20 or DI flat/crossed = 0–3 (RANGING — penalise heavily)

3. SESSION QUALITY — Kill Zones   [{settings.P1_SESSION_POINTS}pts MAX]
   London–NY overlap (12:00–16:00 UTC), Tue–Thu = 10–12 (prime)
   London open (07:00–10:00) or NY open (13:00–16:00) = 7–9
   Mid-session single session = 4–6
   Monday open / Friday close = 2–3

4. 1H TREND — Supertrend + EMA   [{settings.P1_1H_TREND_POINTS}pts MAX]
   Supertrend direction matches higher TF AND 1H EMAs stacked = 8–10
   One confirms, other neutral = 5–7
   Mixed/sideways = 2–4
   Contradicts higher TF = 0–1

5. 15MIN TRIGGER — Closed Candle   [{settings.P1_15MIN_TRIGGER_POINTS}pts MAX]
   Strong engulfing/pinbar/rejection wick at key level, fully closed = 6–8
   Moderate reversal candle = 3–5
   No signal / indecision = 0–2

6. NEWS DISTANCE + IMPACT   [{settings.P1_NEWS_POINTS}pts MAX]
   ⚠️ If pair shows "NEWS COMPONENT EXCLUDED": set news=0, NOT in denominator, use reduced thresholds.
   Clean calendar (no high-impact ≤ 6h) = {settings.P1_NEWS_POINTS}
   High-impact 60–120min away = 5–6
   High-impact 30–60min away  = 2–4
   High-impact < 30min        = 0–1

═══════════════════════════════════════════════════════
PHASE 2 — SMC CONFIRMATION
Max: {p2_max}pts | Min to pass: {p2_min}pts ({p2_pct}%)
═══════════════════════════════════════════════════════

1. LIQUIDITY SWEEP + REVERSAL   [{settings.P2_LIQUIDITY_SWEEP_POINTS}pts MAX — MOST IMPORTANT]
   Sweep of obvious highs/lows within last 5 candles + strong rejection reversal = 11–14
   Sweep 6–15 candles ago, reversal structure still intact = 6–10
   Old sweep (>15 candles) or weak/ambiguous reversal = 2–5
   No sweep detected = 0–1

2. BOS / CHoCH CONFIRMATION   [{settings.P2_BOS_CHOCH_POINTS}pts MAX]
   Recent CHoCH (Change of Character) in trade direction = 8–11
   Clean recent BOS in trade direction = 5–7
   Old or weak BOS = 2–4
   No BOS/CHoCH = 0–1

3. ORDER BLOCK   [{settings.P2_ORDER_BLOCK_POINTS}pts MAX]
   Fresh unmitigated OB, price at/entering it now = 5–7
   Active OB within 15pips of price = 3–4
   OB exists but price is far = 1–2
   No relevant OB = 0

4. FAIR VALUE GAP   [{settings.P2_FVG_POINTS}pts MAX]
   Price actively filling unmitigated FVG = 2–3
   FVG nearby but not filling yet = 1
   No FVG near price = 0

═══════════════════════════════════════════════════════
SCORING DECISION
═══════════════════════════════════════════════════════

TAKE = ALL three true:
  ✅ Phase 1 ≥ effective P1 min shown in pair header
  ✅ Phase 2 ≥ {p2_min}/{p2_max} ({p2_pct}%)
  ✅ Total   ≥ effective total min shown in pair header
           (News incl: {total_min_std}/{total_max_std} | News excl: {total_min_excl}/{total_max_excl})

R:R required by % of effective total:
{rr_desc}

LEAVE = any gate fails, or no clean setup.

═══════════════════════════════════════════════════════
RISK CONTEXT
═══════════════════════════════════════════════════════
Pairs: {', '.join(settings.PAIRS)}
Risk: {settings.MAX_LOSS_PER_TRADE_PCT}%/trade | Daily hard stop: {settings.DAILY_LOSS_LIMIT_PCT}% | No trade count limit

SL budget at min lot (0.01):
{sl_budget_desc}

{prev_context}
═══════════════════════════════════════════════════════
OUTPUT — valid JSON ONLY, no other text
═══════════════════════════════════════════════════════

{{
  "decision": "TAKE" or "LEAVE",
  "pair": one of {list(settings.PAIRS)} or null,
  "direction": "BUY" or "SELL" or null,
  "news_excluded": true or false,
  "phase1_scores": {{
    "weekly_4h_bias": 0–{settings.P1_WEEKLY_4H_BIAS_POINTS},
    "regime":         0–{settings.P1_REGIME_POINTS},
    "session":        0–{settings.P1_SESSION_POINTS},
    "trend_1h":       0–{settings.P1_1H_TREND_POINTS},
    "trigger_15m":    0–{settings.P1_15MIN_TRIGGER_POINTS},
    "news":           0–{settings.P1_NEWS_POINTS} (0 if excluded)
  }},
  "phase1_total": sum of phase1_scores,
  "phase2_scores": {{
    "liquidity_sweep": 0–{settings.P2_LIQUIDITY_SWEEP_POINTS},
    "bos_choch":       0–{settings.P2_BOS_CHOCH_POINTS},
    "order_block":     0–{settings.P2_ORDER_BLOCK_POINTS},
    "fvg":             0–{settings.P2_FVG_POINTS}
  }},
  "phase2_total": sum of phase2_scores,
  "total_score":  phase1_total + phase2_total,
  "reasoning":    "detailed per-pair analysis with every score explained"
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(
    market_data: MarketDataPayload,
    previous_decision: dict | None = None,
) -> TraderDecision:
    """Call the AI and return a validated TraderDecision."""
    client = _get_llm_client()

    balance = market_data.account_balance or 40.0
    max_sl_pips = {
        pair: settings.compute_max_sl_pips(balance, pair)
        for pair in settings.PAIRS
    }

    system_prompt  = _build_system_prompt(max_sl_pips, previous_decision)
    market_context = format_data_for_prompt(market_data)

    user_message = (
        f"Analyze ALL pairs: {', '.join(settings.PAIRS)}.\n"
        "Each pair header shows its effective scoring thresholds.\n"
        "Pick the HIGHEST conviction pair if any qualifies, otherwise LEAVE.\n\n"
        f"{market_context}"
    )

    logger.info("🧠 Sending data to AI Trader Brain...")

    try:
        response = client.chat.completions.create(
            model       = _get_model_name(),
            temperature = settings.LLM_TEMPERATURE,
            **{settings.LLM_MAX_TOKENS_PARAM: settings.LLM_MAX_TOKENS},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )

        raw = response.choices[0].message.content.strip()
        logger.debug(f"AI raw response: {raw[:600]}...")

        json_str = _extract_json(raw)
        data     = json.loads(json_str)

        decision = TraderDecision(
            decision      = data.get("decision", "LEAVE").upper(),
            pair          = data.get("pair"),
            direction     = data.get("direction"),
            news_excluded = data.get("news_excluded", False),
            phase1_scores = Phase1Scores(
                weekly_4h_bias = data.get("phase1_scores", {}).get("weekly_4h_bias", 0),
                regime         = data.get("phase1_scores", {}).get("regime",         0),
                session        = data.get("phase1_scores", {}).get("session",        0),
                trend_1h       = data.get("phase1_scores", {}).get("trend_1h",       0),
                trigger_15m    = data.get("phase1_scores", {}).get("trigger_15m",    0),
                news           = data.get("phase1_scores", {}).get("news",           0),
            ),
            phase1_total  = data.get("phase1_total", 0),
            phase2_scores = Phase2Scores(
                liquidity_sweep = data.get("phase2_scores", {}).get("liquidity_sweep", 0),
                bos_choch       = data.get("phase2_scores", {}).get("bos_choch",       0),
                order_block     = data.get("phase2_scores", {}).get("order_block",     0),
                fvg             = data.get("phase2_scores", {}).get("fvg",             0),
            ),
            phase2_total  = data.get("phase2_total", 0),
            total_score   = data.get("total_score",  0),
            reasoning     = data.get("reasoning", ""),
        )

        # Authoritative news_excluded from market data (prevents AI gaming)
        if decision.pair and decision.pair in market_data.pairs:
            decision.news_excluded = not market_data.pairs[decision.pair].news_fetch_ok

        decision = _validate_scores(decision)
        decision = _validate_decision(decision, market_data)
        _log_decision(decision)
        return decision

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        return TraderDecision(decision="LEAVE", reasoning=f"AI JSON parse error: {e}")
    except Exception as e:
        logger.error(f"AI Trader Brain error: {e}")
        return TraderDecision(decision="LEAVE", reasoning=f"AI error: {e}")


def _extract_json(raw: str) -> str:
    import re
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text  = "\n".join(lines)
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION — Python enforces all thresholds; AI cannot override
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_scores(decision: TraderDecision) -> TraderDecision:
    """Cap sub-scores to maximums and recompute totals from sub-scores."""
    p1 = decision.phase1_scores
    p1.weekly_4h_bias = max(0, min(p1.weekly_4h_bias, settings.P1_WEEKLY_4H_BIAS_POINTS))
    p1.regime         = max(0, min(p1.regime,         settings.P1_REGIME_POINTS))
    p1.session        = max(0, min(p1.session,        settings.P1_SESSION_POINTS))
    p1.trend_1h       = max(0, min(p1.trend_1h,       settings.P1_1H_TREND_POINTS))
    p1.trigger_15m    = max(0, min(p1.trigger_15m,    settings.P1_15MIN_TRIGGER_POINTS))
    p1.news           = max(0, min(p1.news,           settings.P1_NEWS_POINTS))
    if decision.news_excluded:
        p1.news = 0

    p2 = decision.phase2_scores
    p2.liquidity_sweep = max(0, min(p2.liquidity_sweep, settings.P2_LIQUIDITY_SWEEP_POINTS))
    p2.bos_choch       = max(0, min(p2.bos_choch,       settings.P2_BOS_CHOCH_POINTS))
    p2.order_block     = max(0, min(p2.order_block,     settings.P2_ORDER_BLOCK_POINTS))
    p2.fvg             = max(0, min(p2.fvg,             settings.P2_FVG_POINTS))

    real_p1 = (p1.weekly_4h_bias + p1.regime + p1.session +
               p1.trend_1h + p1.trigger_15m + p1.news)
    real_p2 = p2.liquidity_sweep + p2.bos_choch + p2.order_block + p2.fvg

    if decision.phase1_total != real_p1:
        logger.warning(f"AI P1 total {decision.phase1_total} → corrected to {real_p1}")
        decision.phase1_total = real_p1
    if decision.phase2_total != real_p2:
        logger.warning(f"AI P2 total {decision.phase2_total} → corrected to {real_p2}")
        decision.phase2_total = real_p2

    real_total = real_p1 + real_p2
    if decision.total_score != real_total:
        logger.warning(f"AI total {decision.total_score} → corrected to {real_total}")
        decision.total_score = real_total

    return decision


def _validate_decision(
    decision: TraderDecision,
    market_data: MarketDataPayload | None = None,
) -> TraderDecision:
    """Enforce effective % thresholds. AI cannot override Python validation."""
    for legacy, new in {"EXECUTE": "TAKE", "SKIP": "LEAVE", "WATCH": "LEAVE"}.items():
        if decision.decision == legacy:
            decision.decision = new

    if decision.decision != "TAKE":
        return decision

    # Resolve news_excluded authoritatively
    if market_data and decision.pair and decision.pair in market_data.pairs:
        decision.news_excluded = not market_data.pairs[decision.pair].news_fetch_ok

    news_excl     = decision.news_excluded
    p1_eff_max    = settings.get_effective_p1_max(news_excl)
    p1_min        = settings.get_phase1_min_required(news_excl)
    p2_min        = settings.get_phase2_min_required()
    total_eff_max = settings.get_total_max_score(news_excl)
    total_min     = settings.get_total_min_required(news_excl)
    news_tag      = " [news excl]" if news_excl else ""

    def reject(reason: str) -> TraderDecision:
        logger.warning(f"AI said TAKE but {reason} — overriding to LEAVE")
        decision.decision  = "LEAVE"
        decision.reasoning += f"\n[OVERRIDE] {reason}. TAKE → LEAVE."
        return decision

    if decision.phase1_total < p1_min:
        return reject(f"P1 {decision.phase1_total} < min {p1_min}/{p1_eff_max} "
                      f"({int(settings.PHASE1_MIN_PCT*100)}%){news_tag}")

    if decision.phase2_total < p2_min:
        return reject(f"P2 {decision.phase2_total} < min {p2_min}/{settings.PHASE2_MAX_SCORE} "
                      f"({int(settings.PHASE2_MIN_PCT*100)}%)")

    if decision.total_score < total_min:
        score_pct = settings.get_score_pct(decision.total_score, news_excl)
        return reject(f"Total {decision.total_score}/{total_eff_max} ({score_pct*100:.0f}%) "
                      f"< min {total_min} ({int(settings.TOTAL_MIN_SCORE_PCT*100)}%){news_tag}")

    if not decision.pair or not decision.direction:
        return reject("Missing pair or direction for TAKE")

    return decision


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def _log_decision(decision: TraderDecision):
    emoji    = {"TAKE": "🟢", "LEAVE": "🔴"}.get(decision.decision, "⚪")
    news_tag = " [news excl]" if decision.news_excluded else ""

    p1_eff_max    = settings.get_effective_p1_max(decision.news_excluded)
    p1_min        = settings.get_phase1_min_required(decision.news_excluded)
    total_eff_max = settings.get_total_max_score(decision.news_excluded)
    total_min     = settings.get_total_min_required(decision.news_excluded)
    score_pct     = settings.get_score_pct(decision.total_score, decision.news_excluded)
    p2_min        = settings.get_phase2_min_required()

    logger.info(f"\n{'─' * 65}")
    logger.info(f"{emoji} DECISION: {decision.decision}{news_tag}")
    if decision.pair:
        logger.info(f"   Pair: {decision.pair} | Direction: {decision.direction}")

    p1 = decision.phase1_scores
    logger.info(
        f"   Phase 1: {decision.phase1_total}/{p1_eff_max}  "
        f"(min {p1_min} = {int(settings.PHASE1_MIN_PCT*100)}%{news_tag})"
    )
    logger.info(
        f"     Bias: {p1.weekly_4h_bias}/{settings.P1_WEEKLY_4H_BIAS_POINTS}  "
        f"Regime: {p1.regime}/{settings.P1_REGIME_POINTS}  "
        f"Session: {p1.session}/{settings.P1_SESSION_POINTS}  "
        f"1H: {p1.trend_1h}/{settings.P1_1H_TREND_POINTS}  "
        f"15m: {p1.trigger_15m}/{settings.P1_15MIN_TRIGGER_POINTS}  "
        f"News: {p1.news}/{settings.P1_NEWS_POINTS}"
    )

    p2 = decision.phase2_scores
    logger.info(
        f"   Phase 2: {decision.phase2_total}/{settings.PHASE2_MAX_SCORE}  "
        f"(min {p2_min} = {int(settings.PHASE2_MIN_PCT*100)}%)"
    )
    logger.info(
        f"     Sweep: {p2.liquidity_sweep}/{settings.P2_LIQUIDITY_SWEEP_POINTS}  "
        f"BOS/CHoCH: {p2.bos_choch}/{settings.P2_BOS_CHOCH_POINTS}  "
        f"OB: {p2.order_block}/{settings.P2_ORDER_BLOCK_POINTS}  "
        f"FVG: {p2.fvg}/{settings.P2_FVG_POINTS}"
    )
    logger.info(
        f"   Total: {decision.total_score}/{total_eff_max} ({score_pct*100:.0f}%)  "
        f"[min {total_min} = {int(settings.TOTAL_MIN_SCORE_PCT*100)}%]"
    )
    logger.info(f"   Reasoning: {decision.reasoning[:250]}...")
    logger.info(f"{'─' * 65}")
