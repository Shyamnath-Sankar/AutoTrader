"""
trader_brain.py — AI Agent 1: The Trader Brain.

Uses OpenAI-compatible API to analyze market data and produce
a scored trading decision (TAKE / LEAVE / SCHEDULE).

Architecture (per diagram):
  - Reads ALL data, scores EVERYTHING, uses judgment not rules
  - Outputs: decision + scores + direction + pair + reasoning
  - Does NOT output SL/TP — Risk Engine computes those from swing structure
  - SCHEDULE: AI sets exact wakeup time with reason (reads what's MISSING
    and how fast it is MOVING toward target)

The prompt is built DYNAMICALLY from settings.py — change the max scores,
sub-score allocations, or minimum thresholds there and the prompt adapts.
"""

import json

from loguru import logger
from openai import OpenAI

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

def _get_llm_client() -> OpenAI:
    """Create an OpenAI-compatible client from settings."""
    return OpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC SYSTEM PROMPT — BUILT FROM settings.py
# ═══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt(max_sl_pips: dict[str, int], previous_decision: dict | None = None) -> str:
    """
    Build the scoring system prompt dynamically from settings.py values.
    If you change PHASE1_MAX_SCORE, sub-score allocations, or minimums,
    this prompt updates automatically.

    Args:
        max_sl_pips: dict of pair → max affordable SL pips at min lot
        previous_decision: previous cycle's decision for context (anti-inflation)
    """
    total_max = settings.get_total_max_score()
    total_min_required = settings.get_total_min_required()
    total_min_pct_display = int(settings.TOTAL_MIN_SCORE_PCT * 100)

    # Build R:R tier description from percentage-based tiers
    rr_desc = ""
    for tier in settings.RR_TIERS:
        lo = int(tier['min_pct'] * 100)
        hi = int(tier['max_pct'] * 100) if tier['max_pct'] <= 1.0 else 100
        rr_desc += f"  - Score {lo}%–{hi}% of max ({int(tier['min_pct'] * total_max)}–{int(min(tier['max_pct'], 1.0) * total_max)} pts): minimum R:R 1:{tier['rr_ratio']:.0f}\n"

    # Build SL budget info per pair
    sl_budget_desc = "\n".join(
        f"  {pair}: max {pips} pips SL at minimum lot (0.01)"
        for pair, pips in max_sl_pips.items()
    )

    # Build previous decision context
    prev_context = ""
    if previous_decision:
        prev_total = previous_decision.get('total_score', 0)
        prev_pct = settings.get_score_pct(prev_total) * 100 if prev_total else 0
        prev_context = f"""
═══ PREVIOUS ANALYSIS (compare against — justify score changes) ═══
Previous decision: {previous_decision.get('decision', 'N/A')}
Previous pair: {previous_decision.get('pair', 'N/A')}
Previous Phase 1: {previous_decision.get('phase1_total', 'N/A')}/{settings.PHASE1_MAX_SCORE}
Previous Phase 2: {previous_decision.get('phase2_total', 'N/A')}/{settings.PHASE2_MAX_SCORE}
Previous total: {prev_total}/{total_max} ({prev_pct:.0f}%)
Time since: recent (within the last scan cycle)

⚠️ If your scores change significantly from the previous scan, you MUST explain
what changed in the market data to justify the score difference. Do NOT inflate
scores just to reach the TAKE threshold. The market must actually move.
"""

    prompt = f"""You are an elite Smart Money / ICT forex trader brain. You analyze raw market data and score trade setups using a two-phase confidence system. You use JUDGMENT, not rigid rules — you read all the data, assess everything holistically, and score with explicit reasoning.

═══ SCORING SYSTEM ═══

PHASE 1 — MARKET CONTEXT (max {settings.PHASE1_MAX_SCORE} points)
You score each sub-component and explain WHY:

1. REGIME (ADX + DI context): up to {settings.P1_REGIME_POINTS} pts
   - Is the market trending or ranging? Look at ADX value and +DI vs -DI
   - ADX > 25 with clear DI separation = strong trend = high score
   - ADX < 20 or DI lines crossed/flat = weak/no trend = low score
   - ADX 20-25 = transitional, be cautious = moderate score

2. SESSION QUALITY (kill zones): up to {settings.P1_SESSION_POINTS} pts
   - London-NY overlap (12:00-16:00 UTC) = highest quality = 7-8 pts
   - Single session London (07:00-12:00) or NY alone (16:00-21:00) = 5-6 pts
   - Off-hours or thin liquidity = 2-3 pts
   - Consider day of week: Tue-Thu = best, Mon/Fri = weaker

3. NEWS DISTANCE + IMPACT: up to {settings.P1_NEWS_POINTS} pts
   - No news within 2 hours = full points (7-8)
   - High-impact news 60-120 min away = moderate (4-5)
   - High-impact news 30-60 min away = reduce (2-3)
   - High-impact news < 30 min = minimum (0-1)

4. WEEKLY + 4H BIAS ALIGNMENT: up to {settings.P1_WEEKLY_4H_BIAS_POINTS} pts
   - Both weekly and 4H EMAs pointing same direction with price on correct side = 12-14
   - Mostly aligned but some divergence = 8-11
   - Conflicting signals between timeframes = 4-7
   - Completely misaligned = 0-3

5. 1H TREND CONFIRMATION: up to {settings.P1_1H_TREND_POINTS} pts
   - 1H confirms higher timeframe bias with RSI, MACD, EMA all aligned = 10-12
   - Mostly confirming but one indicator diverges = 6-9
   - Mixed signals on 1H = 3-5
   - 1H contradicts higher TF = 0-2

6. 15MIN TRIGGER (closed candle): up to {settings.P1_15MIN_TRIGGER_POINTS} pts
   - Clear entry signal: engulfing, rejection wick at key level = 7-8
   - Moderate signal: price reacting but not definitive = 4-6
   - Weak or no trigger on 15M = 0-3

PHASE 1 MINIMUM TO PROCEED: {settings.PHASE1_MIN_REQUIRED} / {settings.PHASE1_MAX_SCORE}

────────────────────────────────────────

PHASE 2 — SMC CONFIRMATION (max {settings.PHASE2_MAX_SCORE} points)
Only score this if Phase 1 meets minimum. Check ICT/SMC concepts:

⚠️ SCORE PHASE 2 HONESTLY — do NOT rubber-stamp maximum scores.
A typical good setup scores 12-16/20, NOT 20/20 every time.
Only give maximum points when evidence is OVERWHELMING and RECENT.

1. LIQUIDITY SWEEP + REVERSAL: up to {settings.P2_LIQUIDITY_SWEEP_POINTS} pts
   (Most important in ICT/SMC methodology)
   - Sweep happened within last 5 candles AND price reversed = 7-8 pts
   - Sweep happened within last 15 candles with reversal = 4-6 pts
   - Sweep happened but is old (>15 candles) or no clear reversal = 1-3 pts
   - No sweep detected = 0 pts
   - CHECK "candles_ago" field — stale sweeps score LOW

2. ORDER BLOCK — active + price at it: up to {settings.P2_ORDER_BLOCK_POINTS} pts
   - Fresh unmitigated OB with price CURRENTLY touching/inside it = 4-5 pts
   - Active OB within 10 pips of current price = 2-3 pts
   - OB exists but price is far from it = 0-1 pts

3. FAIR VALUE GAP — unmitigated: up to {settings.P2_FVG_POINTS} pts
   - Price actively filling into an unmitigated FVG right now = 3-4 pts
   - Unmitigated FVG nearby but price not at it yet = 1-2 pts
   - No active FVG near price = 0 pts

4. BOS / CHoCH CONFIRMATION: up to {settings.P2_BOS_CHOCH_POINTS} pts
   - Recent BOS/CHoCH in your trade direction = 2-3 pts
   - Structure exists but not clearly confirming direction = 1 pt
   - No clear structure confirmation = 0 pts

PHASE 2 MINIMUM TO PROCEED: {settings.PHASE2_MIN_REQUIRED} / {settings.PHASE2_MAX_SCORE}

────────────────────────────────────────

TOTAL SCORE = Phase 1 + Phase 2 (max {total_max} points)
MINIMUM TOTAL TO TAKE: {total_min_required}/{total_max} ({total_min_pct_display}% of max)

BOTH Phase 1 ≥ {settings.PHASE1_MIN_REQUIRED} AND Phase 2 ≥ {settings.PHASE2_MIN_REQUIRED} are required. Not just total.
The total score must also reach at least {total_min_pct_display}% of the maximum ({total_min_required} points).

R:R Requirements by score percentage:
{rr_desc}
Lower conviction (60-70% of max) demands higher R:R (1:3) for capital protection.
Higher conviction (>70% of max) allows standard R:R (1:2).

═══ DECISION RULES ═══

TAKE: Phase 1 ≥ {settings.PHASE1_MIN_REQUIRED}, Phase 2 ≥ {settings.PHASE2_MIN_REQUIRED}, Total ≥ {total_min_pct_display}% of max ({total_min_required} pts)
  → You MUST provide: pair, direction (BUY/SELL)
  → The Risk Engine will compute SL/TP from swing structure — you do NOT set SL/TP

SCHEDULE: Setup is building but not ready yet — something is CLOSE to triggering
  → You MUST provide: schedule_minutes (exact time to wake up), schedule_reason (what you're waiting for)
  → Be SPECIFIC and QUANTITATIVE about what you're waiting for:
     Examples:
       "RSI at 57, dropping 2pts/candle, need <50 → schedule 9min"
       "Liquidity pool at 1.0820 unswept, price 6 pips away, dropping → schedule 12min"
       "ADX at 19, rising, need 20 → 4H closes in 45min → schedule 46min"
       "Setup fully cold, 4H not aligned → schedule at next London open kill zone 07:00 UTC"
  → Think about HOW FAST the market is moving toward your trigger point

LEAVE: No viable setup, market is unfavorable, nothing building
  → Explain why; the bot will reschedule at default {settings.DEFAULT_SCAN_INTERVAL_MINUTES}min interval

═══ RISK CONTEXT ═══

Account balance: loaded live each cycle
Max loss per trade: {settings.MAX_LOSS_PER_TRADE_PCT}% of account balance
Daily loss limit: {settings.DAILY_LOSS_LIMIT_PCT}% of account balance
Max trades per day: {settings.MAX_TRADES_PER_DAY}
Spread: ~{settings.SPREAD_PIPS} pips typical
Pairs: {', '.join(settings.PAIRS)}

CRITICAL — SL BUDGET (at minimum lot 0.01):
{sl_budget_desc}
The Risk Engine will compute your SL from the nearest swing high/low.
If the natural SL exceeds these pip budgets, your trade WILL be rejected.
Only signal TAKE when the swing structure offers a tight enough SL.

═══ EVALUATE ALL PAIRS ═══

You MUST evaluate EACH pair independently. Do not anchor on one pair.
If multiple pairs show setups, pick the highest conviction one.
If no pair has a viable setup, say LEAVE or SCHEDULE.

{prev_context}
═══ RESPONSE FORMAT ═══

You MUST respond with ONLY valid JSON, no other text. Use this exact structure:

{{
  "decision": "TAKE" | "SCHEDULE" | "LEAVE",
  "pair": "EURUSD" | "GBPUSD" | null,
  "direction": "BUY" | "SELL" | null,
  "phase1_scores": {{
    "regime": <0-{settings.P1_REGIME_POINTS}>,
    "session": <0-{settings.P1_SESSION_POINTS}>,
    "news": <0-{settings.P1_NEWS_POINTS}>,
    "weekly_4h_bias": <0-{settings.P1_WEEKLY_4H_BIAS_POINTS}>,
    "trend_1h": <0-{settings.P1_1H_TREND_POINTS}>,
    "trigger_15m": <0-{settings.P1_15MIN_TRIGGER_POINTS}>
  }},
  "phase1_total": <sum>,
  "phase2_scores": {{
    "liquidity_sweep": <0-{settings.P2_LIQUIDITY_SWEEP_POINTS}>,
    "order_block": <0-{settings.P2_ORDER_BLOCK_POINTS}>,
    "fvg": <0-{settings.P2_FVG_POINTS}>,
    "bos_choch": <0-{settings.P2_BOS_CHOCH_POINTS}>
  }},
  "phase2_total": <sum>,
  "total_score": <phase1_total + phase2_total>,
  "reasoning": "<your detailed analysis — why each score, what you see in the data>",
  "schedule_minutes": <integer or null>,
  "schedule_reason": "<what you're waiting for and how fast it's moving there, if SCHEDULE>"
}}"""

    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(
    market_data: MarketDataPayload,
    previous_decision: dict | None = None,
) -> TraderDecision:
    """
    Send market data to the AI and get a scored trading decision.
    Returns a TraderDecision with decision, scores, and reasoning.

    Args:
        market_data: collected market data payload
        previous_decision: the last decision dict (for anti-inflation context)
    """
    client = _get_llm_client()

    # Compute max SL budget per pair so the AI knows its limits
    max_sl_pips = {}
    balance = market_data.account_balance or 40.0  # fallback
    for pair in settings.PAIRS:
        max_sl_pips[pair] = settings.compute_max_sl_pips(balance, pair)

    system_prompt = _build_system_prompt(max_sl_pips, previous_decision)
    market_context = format_data_for_prompt(market_data)

    user_message = f"""Analyze the following market data and score the setup.
Evaluate ALL pairs ({', '.join(settings.PAIRS)}) independently and pick the best setup if any.
If multiple pairs look good, choose the highest-conviction one.
If no pair qualifies, decide SCHEDULE (if something is building) or LEAVE (if nothing viable).

{market_context}"""

    logger.info("🧠 Sending data to AI Trader Brain...")

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        raw_content = response.choices[0].message.content.strip()
        logger.debug(f"AI raw response: {raw_content[:500]}...")

        # Parse JSON from response (handle markdown code blocks if present)
        json_str = raw_content
        if json_str.startswith("```"):
            # Strip markdown code block
            lines = json_str.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block or not line.strip().startswith("```"):
                    json_lines.append(line)
            json_str = "\n".join(json_lines)

        data = json.loads(json_str)

        # Build TraderDecision from parsed JSON
        decision = TraderDecision(
            decision=data.get("decision", "LEAVE").upper(),
            pair=data.get("pair"),
            direction=data.get("direction"),
            phase1_scores=Phase1Scores(
                regime=data.get("phase1_scores", {}).get("regime", 0),
                session=data.get("phase1_scores", {}).get("session", 0),
                news=data.get("phase1_scores", {}).get("news", 0),
                weekly_4h_bias=data.get("phase1_scores", {}).get("weekly_4h_bias", 0),
                trend_1h=data.get("phase1_scores", {}).get("trend_1h", 0),
                trigger_15m=data.get("phase1_scores", {}).get("trigger_15m", 0),
            ),
            phase1_total=data.get("phase1_total", 0),
            phase2_scores=Phase2Scores(
                liquidity_sweep=data.get("phase2_scores", {}).get("liquidity_sweep", 0),
                order_block=data.get("phase2_scores", {}).get("order_block", 0),
                fvg=data.get("phase2_scores", {}).get("fvg", 0),
                bos_choch=data.get("phase2_scores", {}).get("bos_choch", 0),
            ),
            phase2_total=data.get("phase2_total", 0),
            total_score=data.get("total_score", 0),
            reasoning=data.get("reasoning", ""),
            schedule_minutes=data.get("schedule_minutes"),
            schedule_reason=data.get("schedule_reason"),
        )

        # Validate and enforce score caps
        decision = _validate_scores(decision)

        # Validate decision logic
        decision = _validate_decision(decision)

        _log_decision(decision)
        return decision

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response as JSON: {e}")
        logger.error(f"Raw response: {raw_content[:1000]}")
        return TraderDecision(
            decision="LEAVE",
            reasoning=f"AI response was not valid JSON: {e}",
        )
    except Exception as e:
        logger.error(f"AI Trader Brain error: {e}")
        return TraderDecision(
            decision="LEAVE",
            reasoning=f"AI analysis failed: {e}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE VALIDATION — enforce caps, sub-score integrity
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_scores(decision: TraderDecision) -> TraderDecision:
    """
    Enforce sub-score caps and verify totals match sub-score sums.
    The AI sometimes reports scores exceeding maximums or wrong totals.
    """
    # Cap individual Phase 1 sub-scores
    p1 = decision.phase1_scores
    p1.regime = max(0, min(p1.regime, settings.P1_REGIME_POINTS))
    p1.session = max(0, min(p1.session, settings.P1_SESSION_POINTS))
    p1.news = max(0, min(p1.news, settings.P1_NEWS_POINTS))
    p1.weekly_4h_bias = max(0, min(p1.weekly_4h_bias, settings.P1_WEEKLY_4H_BIAS_POINTS))
    p1.trend_1h = max(0, min(p1.trend_1h, settings.P1_1H_TREND_POINTS))
    p1.trigger_15m = max(0, min(p1.trigger_15m, settings.P1_15MIN_TRIGGER_POINTS))

    # Cap individual Phase 2 sub-scores
    p2 = decision.phase2_scores
    p2.liquidity_sweep = max(0, min(p2.liquidity_sweep, settings.P2_LIQUIDITY_SWEEP_POINTS))
    p2.order_block = max(0, min(p2.order_block, settings.P2_ORDER_BLOCK_POINTS))
    p2.fvg = max(0, min(p2.fvg, settings.P2_FVG_POINTS))
    p2.bos_choch = max(0, min(p2.bos_choch, settings.P2_BOS_CHOCH_POINTS))

    # Recompute totals from actual sub-scores (don't trust AI arithmetic)
    real_p1 = p1.regime + p1.session + p1.news + p1.weekly_4h_bias + p1.trend_1h + p1.trigger_15m
    real_p2 = p2.liquidity_sweep + p2.order_block + p2.fvg + p2.bos_choch

    if decision.phase1_total != real_p1:
        logger.warning(f"AI Phase 1 total {decision.phase1_total} ≠ sum {real_p1} — correcting")
        decision.phase1_total = real_p1

    if decision.phase2_total != real_p2:
        logger.warning(f"AI Phase 2 total {decision.phase2_total} ≠ sum {real_p2} — correcting")
        decision.phase2_total = real_p2

    real_total = real_p1 + real_p2
    if decision.total_score != real_total:
        logger.warning(f"AI total {decision.total_score} ≠ sum {real_total} — correcting")
        decision.total_score = real_total

    return decision


# ═══════════════════════════════════════════════════════════════════════════════
# DECISION VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_decision(decision: TraderDecision) -> TraderDecision:
    """
    Validate the AI's decision against our minimum thresholds.
    The AI might say TAKE but if scores don't meet minimums, we override.
    """
    # Normalize legacy decision names → new names
    legacy_map = {"EXECUTE": "TAKE", "SKIP": "LEAVE", "WATCH": "SCHEDULE"}
    decision.decision = legacy_map.get(decision.decision, decision.decision)

    if decision.decision == "TAKE":
        # Check Phase 1 minimum
        if decision.phase1_total < settings.PHASE1_MIN_REQUIRED:
            logger.warning(
                f"AI said TAKE but Phase 1 ({decision.phase1_total}) "
                f"< minimum ({settings.PHASE1_MIN_REQUIRED}) — overriding to SCHEDULE"
            )
            decision.decision = "SCHEDULE"
            decision.reasoning += (
                f"\n[OVERRIDE] Phase 1 score {decision.phase1_total} below "
                f"minimum {settings.PHASE1_MIN_REQUIRED}. Changed TAKE → SCHEDULE."
            )
            if not decision.schedule_minutes:
                decision.schedule_minutes = settings.DEFAULT_SCAN_INTERVAL_MINUTES
            if not decision.schedule_reason:
                decision.schedule_reason = "Phase 1 below threshold — waiting for improvement"
            return decision

        # Check Phase 2 minimum
        if decision.phase2_total < settings.PHASE2_MIN_REQUIRED:
            logger.warning(
                f"AI said TAKE but Phase 2 ({decision.phase2_total}) "
                f"< minimum ({settings.PHASE2_MIN_REQUIRED}) — overriding to SCHEDULE"
            )
            decision.decision = "SCHEDULE"
            decision.reasoning += (
                f"\n[OVERRIDE] Phase 2 score {decision.phase2_total} below "
                f"minimum {settings.PHASE2_MIN_REQUIRED}. Changed TAKE → SCHEDULE."
            )
            if not decision.schedule_minutes:
                decision.schedule_minutes = settings.DEFAULT_SCAN_INTERVAL_MINUTES
            if not decision.schedule_reason:
                decision.schedule_reason = "Phase 2 (SMC) below threshold — waiting for confirmation"
            return decision

        # Check total minimum (percentage-based)
        total_min_required = settings.get_total_min_required()
        score_pct = settings.get_score_pct(decision.total_score)
        if decision.total_score < total_min_required:
            pct_display = int(score_pct * 100)
            min_pct_display = int(settings.TOTAL_MIN_SCORE_PCT * 100)
            logger.warning(
                f"AI said TAKE but total ({decision.total_score}/{settings.get_total_max_score()} = {pct_display}%) "
                f"< minimum ({total_min_required} = {min_pct_display}%) — overriding to SCHEDULE"
            )
            decision.decision = "SCHEDULE"
            decision.reasoning += (
                f"\n[OVERRIDE] Total score {decision.total_score} ({pct_display}%) below "
                f"minimum {total_min_required} ({min_pct_display}%). Changed TAKE → SCHEDULE."
            )
            if not decision.schedule_minutes:
                decision.schedule_minutes = settings.DEFAULT_SCAN_INTERVAL_MINUTES
            if not decision.schedule_reason:
                decision.schedule_reason = "Total score below percentage threshold — waiting for conditions to improve"
            return decision

        # Check required fields for TAKE
        if not decision.pair or not decision.direction:
            logger.warning("AI said TAKE but missing pair/direction — overriding to LEAVE")
            decision.decision = "LEAVE"
            decision.reasoning += "\n[OVERRIDE] Missing pair or direction for TAKE."
            return decision

    elif decision.decision == "SCHEDULE":
        # Ensure we have schedule time
        if not decision.schedule_minutes or decision.schedule_minutes <= 0:
            decision.schedule_minutes = settings.DEFAULT_SCAN_INTERVAL_MINUTES

        # Cap max schedule time
        if decision.schedule_minutes > settings.MAX_SCHEDULE_MINUTES:
            decision.schedule_minutes = settings.MAX_SCHEDULE_MINUTES

        if not decision.schedule_reason:
            decision.schedule_reason = "AI scheduled without explicit reason"

    return decision


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def _log_decision(decision: TraderDecision):
    """Log the AI decision with colored output."""
    emoji = {"TAKE": "🟢", "SCHEDULE": "🟡", "LEAVE": "🔴"}.get(decision.decision, "⚪")

    logger.info(f"\n{'─' * 60}")
    logger.info(f"{emoji} AI DECISION: {decision.decision}")
    if decision.pair:
        logger.info(f"   Pair: {decision.pair} | Direction: {decision.direction}")
    logger.info(f"   Phase 1: {decision.phase1_total}/{settings.PHASE1_MAX_SCORE} "
                f"(min {settings.PHASE1_MIN_REQUIRED})")
    logger.info(f"     Regime: {decision.phase1_scores.regime}/{settings.P1_REGIME_POINTS} | "
                f"Session: {decision.phase1_scores.session}/{settings.P1_SESSION_POINTS} | "
                f"News: {decision.phase1_scores.news}/{settings.P1_NEWS_POINTS}")
    logger.info(f"     Bias: {decision.phase1_scores.weekly_4h_bias}/{settings.P1_WEEKLY_4H_BIAS_POINTS} | "
                f"Trend: {decision.phase1_scores.trend_1h}/{settings.P1_1H_TREND_POINTS} | "
                f"Trigger: {decision.phase1_scores.trigger_15m}/{settings.P1_15MIN_TRIGGER_POINTS}")
    logger.info(f"   Phase 2: {decision.phase2_total}/{settings.PHASE2_MAX_SCORE} "
                f"(min {settings.PHASE2_MIN_REQUIRED})")
    logger.info(f"     Liquidity: {decision.phase2_scores.liquidity_sweep}/{settings.P2_LIQUIDITY_SWEEP_POINTS} | "
                f"OB: {decision.phase2_scores.order_block}/{settings.P2_ORDER_BLOCK_POINTS} | "
                f"FVG: {decision.phase2_scores.fvg}/{settings.P2_FVG_POINTS} | "
                f"BOS: {decision.phase2_scores.bos_choch}/{settings.P2_BOS_CHOCH_POINTS}")
    total_max = settings.get_total_max_score()
    score_pct = settings.get_score_pct(decision.total_score)
    total_min_required = settings.get_total_min_required()
    logger.info(f"   Total: {decision.total_score}/{total_max} "
                f"({score_pct * 100:.0f}%) "
                f"(min {total_min_required} = {int(settings.TOTAL_MIN_SCORE_PCT * 100)}%)")

    if decision.schedule_minutes:
        logger.info(f"   ⏰ Schedule in: {decision.schedule_minutes} min")
        if decision.schedule_reason:
            logger.info(f"   Reason: {decision.schedule_reason}")

    logger.info(f"   Reasoning: {decision.reasoning[:300]}...")
    logger.info(f"{'─' * 60}")
