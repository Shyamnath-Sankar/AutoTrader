"""
trader_brain.py — AI Agent 1: The Trader Brain.

Uses OpenAI-compatible API to analyze market data and produce
a scored trading decision (EXECUTE / WATCH / SKIP).

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

def _build_system_prompt(target_pair: str | None = None) -> str:
    """
    Build the scoring system prompt dynamically from settings.py values.
    If you change PHASE1_MAX_SCORE, sub-score allocations, or minimums,
    this prompt updates automatically.

    Args:
        target_pair: if set, AI analyzes only this pair (concurrent mode)
    """
    total_max = settings.PHASE1_MAX_SCORE + settings.PHASE2_MAX_SCORE

    # Build R:R tier description
    rr_desc = ""
    for tier in settings.RR_TIERS:
        rr_desc += f"  - Score {tier['min_score']}–{tier['max_score']}: minimum R:R 1:{tier['rr_ratio']:.0f}\n"

    # Pair instruction
    if target_pair:
        pair_instruction = f"You are analyzing ONLY {target_pair}. Focus entirely on this pair."
    else:
        pair_instruction = f"Evaluate ALL pairs ({', '.join(settings.PAIRS)}) and pick the best setup if any."

    prompt = f"""You are an elite Smart Money / ICT forex trader brain. You analyze raw market data and score trade setups using a two-phase confidence system. You use JUDGMENT, not rigid rules — you read all the data, assess everything holistically, and score with explicit reasoning.

═══ SCORING SYSTEM ═══

PHASE 1 — MARKET CONTEXT (max {settings.PHASE1_MAX_SCORE} points)
You score each sub-component and explain WHY:

1. REGIME (ADX + DI context): up to {settings.P1_REGIME_POINTS} pts
   - Is the market trending or ranging? Look at ADX value and +DI vs -DI
   - ADX > 25 with clear DI separation = strong trend = high score
   - ADX < 20 or DI lines crossed/flat = weak/no trend = low score

2. SESSION QUALITY (kill zones): up to {settings.P1_SESSION_POINTS} pts
   - London-NY overlap = highest quality
   - Single session (London or NY) = good
   - Off-hours or thin liquidity = low score
   - Consider day of week: Tue-Thu best, Mon/Fri weaker

3. NEWS DISTANCE + IMPACT: up to {settings.P1_NEWS_POINTS} pts
   - No news within 2 hours = full points
   - High-impact news 30-60 min away = reduce points
   - High-impact news < 30 min = minimum points
   - Consider how news might affect the pair's currencies

4. WEEKLY + 4H BIAS ALIGNMENT: up to {settings.P1_WEEKLY_4H_BIAS_POINTS} pts
   - Are weekly and 4H EMAs pointing same direction?
   - Is price above/below key EMAs consistently on both timeframes?
   - Strong alignment = high score, conflicting = low score

5. 1H TREND CONFIRMATION: up to {settings.P1_1H_TREND_POINTS} pts
   - Does the 1H chart confirm the higher timeframe bias?
   - RSI, MACD, EMA structure on 1H aligned with direction?
   - Momentum (MACD histogram) expanding or contracting?

6. 15MIN TRIGGER (closed candle): up to {settings.P1_15MIN_TRIGGER_POINTS} pts
   - Is there a valid entry signal on the 15min closed candle?
   - Price reacting to a level, engulfing pattern, rejection wick?
   - Entry precision on 15min timeframe

PHASE 1 MINIMUM TO PROCEED: {settings.PHASE1_MIN_REQUIRED} / {settings.PHASE1_MAX_SCORE}

────────────────────────────────────────

PHASE 2 — SMC CONFIRMATION (max {settings.PHASE2_MAX_SCORE} points)
Only score this if Phase 1 meets minimum. Check ICT/SMC concepts:

1. LIQUIDITY SWEEP + REVERSAL: up to {settings.P2_LIQUIDITY_SWEEP_POINTS} pts
   (Most important in ICT/SMC methodology)
   - Has a liquidity pool been swept (stop hunt)?
   - Did price reverse after sweeping liquidity?
   - Recent sweep = high score, no sweep = low score

2. ORDER BLOCK — active + price at it: up to {settings.P2_ORDER_BLOCK_POINTS} pts
   - Is there an active (unmitigated) order block near price?
   - Is price currently reacting to / sitting on an OB zone?
   - Fresh OB > old OB

3. FAIR VALUE GAP — unmitigated: up to {settings.P2_FVG_POINTS} pts
   - Is there an unmitigated FVG near current price?
   - Price filling into a FVG = potential reversal zone
   - Count and proximity of active FVGs

4. BOS / CHoCH CONFIRMATION: up to {settings.P2_BOS_CHOCH_POINTS} pts
   - Has there been a break of structure (BOS) confirming trend?
   - Or a change of character (CHoCH) signaling reversal?
   - Does the structure analysis support your trade direction?

PHASE 2 MINIMUM TO PROCEED: {settings.PHASE2_MIN_REQUIRED} / {settings.PHASE2_MAX_SCORE}

────────────────────────────────────────

TOTAL SCORE = Phase 1 + Phase 2 (max {total_max} points)
MINIMUM TOTAL TO EXECUTE: {settings.TOTAL_MIN_REQUIRED}

BOTH Phase 1 ≥ {settings.PHASE1_MIN_REQUIRED} AND Phase 2 ≥ {settings.PHASE2_MIN_REQUIRED} are required. Not just total.

R:R Requirements by score:
{rr_desc}
═══ DECISION RULES ═══

EXECUTE: Phase 1 ≥ {settings.PHASE1_MIN_REQUIRED}, Phase 2 ≥ {settings.PHASE2_MIN_REQUIRED}, Total ≥ {settings.TOTAL_MIN_REQUIRED}
  → You MUST provide: pair, direction (BUY/SELL), sl_pips, tp_pips, rr_ratio

WATCH: Setup is building but not ready yet — something is CLOSE to triggering
  → You MUST provide: next_check_minutes (how long to wait), next_check_reason (what you're waiting for)
  → Examples: "RSI at 57, dropping 2pts/candle, need <50 → check in 8min"
              "Price 6 pips from liquidity pool, moving toward it → check in 12min"
              "ADX at 19, rising, need 20 → 4H closes in 45min → check in 46min"

SKIP: No viable setup, market is unfavorable
  → Explain why; the bot will reschedule at default interval

═══ RISK CONTEXT ═══

Account balance: loaded live each cycle
Max loss per trade: 5% of account balance (dynamic)
Spread: {settings.SPREAD_PIPS} pips
Max trades per day: {settings.MAX_TRADES_PER_DAY}
Daily loss limit: 15% of account balance (dynamic)
{pair_instruction}

═══ RESPONSE FORMAT ═══

You MUST respond with ONLY valid JSON, no other text. Use this exact structure:

{{
  "decision": "EXECUTE" | "WATCH" | "SKIP",
  "pair": "{target_pair or 'XAUUSD" | "USDJPY" | "EURUSD" | "GBPUSD" | null'}",
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
  "sl_pips": <integer or null>,
  "tp_pips": <integer or null>,
  "rr_ratio": <float or null>,
  "reasoning": "<your detailed analysis — why each score, what you see in the data>",
  "next_check_minutes": <integer or null>,
  "next_check_reason": "<what you're waiting for, if WATCH>"
}}"""

    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE (ALL PAIRS — legacy mode)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(market_data: MarketDataPayload) -> TraderDecision:
    """
    Send market data to the AI and get a scored trading decision.
    Returns a TraderDecision with decision, scores, and reasoning.
    """
    client = _get_llm_client()
    system_prompt = _build_system_prompt()
    market_context = format_data_for_prompt(market_data)

    user_message = f"""Analyze the following market data and score the setup.
Evaluate ALL pairs ({', '.join(settings.PAIRS)}) and pick the best setup if any.
If multiple pairs look good, choose the highest-conviction one.

{market_context}"""

    logger.info("🧠 Sending data to AI Trader Brain...")

    return _call_llm(client, system_prompt, user_message)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE SINGLE PAIR (concurrent mode)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_pair(market_data: MarketDataPayload, pair: str) -> TraderDecision:
    """
    Analyze a SINGLE pair using the AI.
    Used by the concurrent per-pair analysis pipeline.
    """
    client = _get_llm_client()
    system_prompt = _build_system_prompt(target_pair=pair)
    market_context = format_data_for_prompt(market_data)

    user_message = f"""Analyze the following market data for {pair} and score the setup.
Focus ONLY on {pair}.

{market_context}"""

    logger.info(f"🧠 Sending {pair} data to AI Trader Brain...")

    decision = _call_llm(client, system_prompt, user_message)

    # Ensure pair is set correctly
    if decision.decision == "EXECUTE" and not decision.pair:
        decision.pair = pair

    return decision


# ═══════════════════════════════════════════════════════════════════════════════
# LLM CALL (shared logic)
# ═══════════════════════════════════════════════════════════════════════════════

def _call_llm(
    client: OpenAI,
    system_prompt: str,
    user_message: str,
) -> TraderDecision:
    """Call the LLM and parse the response into a TraderDecision."""
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
            decision=data.get("decision", "SKIP").upper(),
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
            sl_pips=data.get("sl_pips"),
            tp_pips=data.get("tp_pips"),
            rr_ratio=data.get("rr_ratio"),
            reasoning=data.get("reasoning", ""),
            next_check_minutes=data.get("next_check_minutes"),
            next_check_reason=data.get("next_check_reason"),
        )

        # Validate decision logic
        decision = _validate_decision(decision)

        _log_decision(decision)
        return decision

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response as JSON: {e}")
        logger.error(f"Raw response: {raw_content[:1000]}")
        return TraderDecision(
            decision="SKIP",
            reasoning=f"AI response was not valid JSON: {e}",
            next_check_minutes=settings.DEFAULT_SCAN_INTERVAL_MINUTES,
        )
    except Exception as e:
        logger.error(f"AI Trader Brain error: {e}")
        return TraderDecision(
            decision="SKIP",
            reasoning=f"AI analysis failed: {e}",
            next_check_minutes=settings.DEFAULT_SCAN_INTERVAL_MINUTES,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_decision(decision: TraderDecision) -> TraderDecision:
    """
    Validate the AI's decision against our minimum thresholds.
    The AI might say EXECUTE but if scores don't meet minimums, we override.
    """
    if decision.decision == "EXECUTE":
        # Check Phase 1 minimum
        if decision.phase1_total < settings.PHASE1_MIN_REQUIRED:
            logger.warning(
                f"AI said EXECUTE but Phase 1 ({decision.phase1_total}) "
                f"< minimum ({settings.PHASE1_MIN_REQUIRED}) — overriding to WATCH"
            )
            decision.decision = "WATCH"
            decision.reasoning += (
                f"\n[OVERRIDE] Phase 1 score {decision.phase1_total} below "
                f"minimum {settings.PHASE1_MIN_REQUIRED}. Changed EXECUTE → WATCH."
            )
            if not decision.next_check_minutes:
                decision.next_check_minutes = settings.DEFAULT_SCAN_INTERVAL_MINUTES
            return decision

        # Check Phase 2 minimum
        if decision.phase2_total < settings.PHASE2_MIN_REQUIRED:
            logger.warning(
                f"AI said EXECUTE but Phase 2 ({decision.phase2_total}) "
                f"< minimum ({settings.PHASE2_MIN_REQUIRED}) — overriding to WATCH"
            )
            decision.decision = "WATCH"
            decision.reasoning += (
                f"\n[OVERRIDE] Phase 2 score {decision.phase2_total} below "
                f"minimum {settings.PHASE2_MIN_REQUIRED}. Changed EXECUTE → WATCH."
            )
            if not decision.next_check_minutes:
                decision.next_check_minutes = settings.DEFAULT_SCAN_INTERVAL_MINUTES
            return decision

        # Check total minimum
        if decision.total_score < settings.TOTAL_MIN_REQUIRED:
            logger.warning(
                f"AI said EXECUTE but total ({decision.total_score}) "
                f"< minimum ({settings.TOTAL_MIN_REQUIRED}) — overriding to WATCH"
            )
            decision.decision = "WATCH"
            decision.reasoning += (
                f"\n[OVERRIDE] Total score {decision.total_score} below "
                f"minimum {settings.TOTAL_MIN_REQUIRED}. Changed EXECUTE → WATCH."
            )
            if not decision.next_check_minutes:
                decision.next_check_minutes = settings.DEFAULT_SCAN_INTERVAL_MINUTES
            return decision

        # Check required fields for EXECUTE
        if not decision.pair or not decision.direction:
            logger.warning("AI said EXECUTE but missing pair/direction — overriding to SKIP")
            decision.decision = "SKIP"
            decision.reasoning += "\n[OVERRIDE] Missing pair or direction for EXECUTE."
            return decision

        if not decision.sl_pips or decision.sl_pips <= 0:
            logger.warning("AI said EXECUTE but missing/invalid SL pips — overriding to SKIP")
            decision.decision = "SKIP"
            decision.reasoning += "\n[OVERRIDE] Missing or invalid SL pips for EXECUTE."
            return decision

    elif decision.decision == "WATCH":
        # Ensure we have a next_check time
        if not decision.next_check_minutes or decision.next_check_minutes <= 0:
            decision.next_check_minutes = settings.DEFAULT_SCAN_INTERVAL_MINUTES

        # Cap max schedule time
        if decision.next_check_minutes > settings.MAX_SCHEDULE_MINUTES:
            decision.next_check_minutes = settings.MAX_SCHEDULE_MINUTES

    return decision


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def _log_decision(decision: TraderDecision):
    """Log the AI decision with colored output."""
    emoji = {"EXECUTE": "🟢", "WATCH": "🟡", "SKIP": "🔴"}.get(decision.decision, "⚪")

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
    logger.info(f"   Total: {decision.total_score}/{settings.PHASE1_MAX_SCORE + settings.PHASE2_MAX_SCORE} "
                f"(min {settings.TOTAL_MIN_REQUIRED})")

    if decision.sl_pips:
        logger.info(f"   SL: {decision.sl_pips} pips | TP: {decision.tp_pips} pips | R:R 1:{decision.rr_ratio}")

    if decision.next_check_minutes:
        logger.info(f"   Next check in: {decision.next_check_minutes} min")
        if decision.next_check_reason:
            logger.info(f"   Reason: {decision.next_check_reason}")

    logger.info(f"   Reasoning: {decision.reasoning[:300]}...")
    logger.info(f"{'─' * 60}")
