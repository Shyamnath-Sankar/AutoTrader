"""
trader_brain.py — AI Agent 1: The Trader Brain.

Uses LangChain ChatOpenAI + PydanticOutputParser for robust structured output.
Analyzes market data and produces a scored trading decision (EXECUTE / WATCH / SKIP).

The prompt is built DYNAMICALLY from settings.py — change the max scores,
sub-score allocations, or minimum thresholds there and the prompt adapts.
"""

import re

from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from config import settings
from core.models import (
    MarketDataPayload,
    Phase1Scores,
    Phase2Scores,
    TraderDecision,
)
from services.data_collector import format_data_for_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# LLM CLIENT (LangChain)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_llm() -> ChatOpenAI:
    """Create a LangChain ChatOpenAI instance from settings."""
    return ChatOpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

# Pydantic output parser for the TraderDecision schema
_decision_parser = PydanticOutputParser(pydantic_object=TraderDecision)


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

You MUST respond with ONLY valid JSON — no markdown, no commentary, no preamble.
{{format_instructions}}"""

    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE SINGLE PAIR (primary mode — used by LangGraph)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_pair(market_data: MarketDataPayload, pair: str) -> TraderDecision:
    """
    Analyze a SINGLE pair using the AI via LangChain.
    Used by the LangGraph brain node.
    """
    llm = _get_llm()

    # Build format instructions from Pydantic parser
    format_instructions = _decision_parser.get_format_instructions()

    # Build system prompt with format instructions baked in
    system_prompt = _build_system_prompt(target_pair=pair)
    system_prompt = system_prompt.replace("{format_instructions}", format_instructions)

    market_context = format_data_for_prompt(market_data)

    user_message = (
        f"Analyze the following market data for {pair} and score the setup.\n"
        f"Focus ONLY on {pair}.\n\n{market_context}"
    )

    logger.info(f"🧠 Sending {pair} data to AI Trader Brain (LangChain)...")

    try:
        messages = [
            ("system", system_prompt),
            ("human", user_message),
        ]
        response = llm.invoke(messages)

        raw_content = response.content.strip()
        logger.debug(f"AI raw response: {raw_content[:500]}...")

        # Extract JSON robustly — handle reasoning blocks, markdown, chatty preambles
        decision = _parse_response(raw_content)

        # Ensure pair is set correctly
        if decision.decision == "EXECUTE" and not decision.pair:
            decision.pair = pair

        # Validate decision
        decision = _validate_decision(decision)

        _log_decision(decision)
        return decision

    except Exception as e:
        logger.error(f"AI Trader Brain error: {e}")
        return TraderDecision(
            decision="SKIP",
            reasoning=f"AI analysis failed: {e}",
            next_check_minutes=settings.DEFAULT_SCAN_INTERVAL_MINUTES,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE (ALL PAIRS — legacy mode)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(market_data: MarketDataPayload) -> TraderDecision:
    """
    Send market data to the AI and get a scored trading decision.
    Returns a TraderDecision with decision, scores, and reasoning.
    """
    llm = _get_llm()

    format_instructions = _decision_parser.get_format_instructions()
    system_prompt = _build_system_prompt()
    system_prompt = system_prompt.replace("{format_instructions}", format_instructions)

    market_context = format_data_for_prompt(market_data)

    user_message = (
        f"Analyze the following market data and score the setup.\n"
        f"Evaluate ALL pairs ({', '.join(settings.PAIRS)}) and pick the best setup if any.\n\n"
        f"{market_context}"
    )

    logger.info("🧠 Sending data to AI Trader Brain (LangChain)...")

    try:
        messages = [
            ("system", system_prompt),
            ("human", user_message),
        ]
        response = llm.invoke(messages)

        raw_content = response.content.strip()
        logger.debug(f"AI raw response: {raw_content[:500]}...")

        decision = _parse_response(raw_content)
        decision = _validate_decision(decision)
        _log_decision(decision)
        return decision

    except Exception as e:
        logger.error(f"AI Trader Brain error: {e}")
        return TraderDecision(
            decision="SKIP",
            reasoning=f"AI analysis failed: {e}",
            next_check_minutes=settings.DEFAULT_SCAN_INTERVAL_MINUTES,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE PARSING — robust JSON extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_response(raw_content: str) -> TraderDecision:
    """
    Parse the AI's raw text response into a TraderDecision.

    Handles reasoning models (e.g. Nemotron) that produce:
      - <think>...</think> blocks before the JSON
      - Long chain-of-thought preamble text
      - Markdown ```json ... ``` blocks
      - Truncated JSON (from max_tokens cutoff)
    """
    import json

    # Step 1: Strip <think>...</think> reasoning blocks (Nemotron, DeepSeek, etc.)
    cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    # Also handle unclosed <think> (truncated response)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()

    if not cleaned:
        # Entire response was reasoning — no JSON at all
        logger.error("AI response was entirely reasoning (no JSON found)")
        logger.error(f"Raw response: {raw_content[:500]}")
        return TraderDecision(
            decision="SKIP",
            reasoning="AI response contained only reasoning, no JSON output",
            next_check_minutes=settings.DEFAULT_SCAN_INTERVAL_MINUTES,
        )

    json_str = cleaned

    # Step 2: Try markdown code block
    block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if block_match:
        json_str = block_match.group(1)
    else:
        # Step 3: Find the first { ... last } (greedy)
        obj_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if obj_match:
            json_str = obj_match.group(1)

    # Step 4: Try to parse
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Step 5: Try to fix truncated JSON (missing closing braces)
        # Count unmatched braces and close them
        open_braces = json_str.count("{") - json_str.count("}")
        if open_braces > 0:
            fixed = json_str + "}" * open_braces
            try:
                data = json.loads(fixed)
                logger.warning(f"Fixed truncated JSON by adding {open_braces} closing brace(s)")
            except json.JSONDecodeError as e2:
                logger.error(f"Failed to parse AI response as JSON: {e2}")
                logger.error(f"Cleaned response: {cleaned[:500]}")
                return TraderDecision(
                    decision="SKIP",
                    reasoning=f"AI response was not valid JSON: {e2}",
                    next_check_minutes=settings.DEFAULT_SCAN_INTERVAL_MINUTES,
                )
        else:
            logger.error(f"Failed to parse AI response as JSON")
            logger.error(f"Cleaned response: {cleaned[:500]}")
            return TraderDecision(
                decision="SKIP",
                reasoning="AI response was not valid JSON",
                next_check_minutes=settings.DEFAULT_SCAN_INTERVAL_MINUTES,
            )

    # Build TraderDecision from parsed JSON
    return TraderDecision(
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
