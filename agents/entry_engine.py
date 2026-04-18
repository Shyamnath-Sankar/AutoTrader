"""
entry_engine.py — AI Agent 3: The Entry Engine.

Architecture (per entry-architecture.svg):
  - Receives: direction + pair + OHLCV (1H + 15M) + SMC structures + current price
  - AI analyzes the candle data + SMC levels to pick the BEST entry point
  - Outputs: EntryCandidate with exact entry price, SL, TP
  - Candidate types by priority:
      1. Sweep entry   — liquidity swept + reversal candle → enter at close
      2. OB entry      — unmitigated order block → enter at OB edge
      3. FVG entry     — unmitigated fair value gap → enter at 50% (equilibrium)
      4. Swing + round — swing H/L or round numbers (.x000/.x500)

The AI reads raw OHLCV candles + all SMC structures and decides WHERE to enter.
The Risk Engine then validates the budget and executes.
"""

import json

from loguru import logger
from openai import OpenAI

from config import settings
from core.models import (
    EntryCandidate,
    MarketDataPayload,
    TraderDecision,
)
from services.data_collector import format_ohlcv_for_entry


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — ENTRY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_entry_prompt(pair: str, direction: str, current_price: float, pip_size: float) -> str:
    """Build the system prompt for the Entry Engine AI."""

    search_radius = settings.ENTRY_SEARCH_RADIUS_PIPS
    sl_buffer = settings.SL_BUFFER_PIPS
    tp_min_rr = settings.TP_MIN_RR

    return f"""You are an elite ICT / Smart Money Concepts entry specialist. Your ONLY job is to find the BEST entry point for a confirmed {direction} trade on {pair}.

The Trader Brain has already confirmed:
- Direction: {direction}
- Pair: {pair}
- Current price: {current_price}
- Pip size: {pip_size}

═══ YOUR TASK ═══

Analyze the OHLCV candle data and SMC structures provided. Find the OPTIMAL entry price using ICT methodology.

═══ ENTRY TYPES (by priority — pick the BEST available) ═══

1. SWEEP ENTRY (priority 1 — most precise ICT entry)
   - A liquidity pool was RECENTLY swept (check liquidity data)
   - Price showed a reversal candle after the sweep
   - Enter at the CLOSE of the reversal candle
   - This is the highest-probability entry in ICT methodology

2. ORDER BLOCK ENTRY (priority 2)
   - An ACTIVE (unmitigated) order block exists near current price
   - For {direction}: enter at the {"OB top edge" if direction == "BUY" else "OB bottom edge"}
   - OB must not be mitigated

3. FVG ENTRY (priority 3)
   - An UNMITIGATED fair value gap exists near current price
   - Enter at the 50% level of the gap (equilibrium)
   - FVG midpoint = (Top + Bottom) / 2

4. SWING / ROUND NUMBER ENTRY (priority 4)
   - Previous swing high/low within {search_radius} pips
   - Round numbers (.x000, .x500)

═══ RULES ═══

1. Entry MUST be within {search_radius} pips of current price ({current_price})
2. For {direction}:
   {"- Entry should be AT or BELOW current price (buying at discount/support)" if direction == "BUY" else "- Entry should be AT or ABOVE current price (selling at premium/resistance)"}
3. SL placement:
   {"- SL BELOW the nearest swing low (or OB bottom) — add " + str(sl_buffer) + " pip buffer" if direction == "BUY" else "- SL ABOVE the nearest swing high (or OB top) — add " + str(sl_buffer) + " pip buffer"}
4. TP = next key structure level in trade direction, minimum {tp_min_rr}× SL distance
   {"- Look for: next swing high, next OB top, next FVG top, resistance, liquidity pool above" if direction == "BUY" else "- Look for: next swing low, next OB bottom, next FVG bottom, support, liquidity pool below"}
5. If MULTIPLE entry types exist at similar levels, note the confluence_count
6. Calculate all distances in PIPS (price_diff / {pip_size})

═══ CRITICAL: YOU MUST FIND AN ENTRY ═══

The Trader Brain has already confirmed this is a valid trade setup. Your job is NOT to second-guess the direction — it is to find the BEST price level to enter. You MUST provide an entry.

If no perfect SMC level exists, use the current price as entry with structure-based SL/TP.
DO NOT return null or skip — the trade has been confirmed. Find the entry.

═══ RESPONSE FORMAT ═══

Respond with ONLY valid JSON:

{{
  "entry_type": "sweep" | "ob" | "fvg" | "swing" | "market",
  "priority": 1-5,
  "entry_price": <exact price level>,
  "sl_price": <exact SL price>,
  "sl_pips": <integer>,
  "tp_price": <exact TP price>,
  "tp_pips": <integer>,
  "rr_ratio": <TP distance / SL distance as float>,
  "confluence_count": <1-4, how many structures overlap at this level>,
  "distance_from_current": <pips from current price, 0 if at market>,
  "reasoning": "<explain exactly which candles/structures you see and why this entry>"
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

def find_entry(
    decision: TraderDecision,
    market_data: MarketDataPayload,
    raw_ohlcv: dict,
    current_price: float,
) -> EntryCandidate | None:
    """
    Use the AI to find the best entry point for a confirmed trade.

    Args:
        decision: TraderDecision with pair, direction, scores
        market_data: full market data payload (for SMC context)
        raw_ohlcv: raw OHLCV DataFrames {pair: {tf: DataFrame}}
        current_price: live ask (BUY) or bid (SELL)

    Returns:
        EntryCandidate with all entry details, or None on failure
    """
    pair = decision.pair
    direction = decision.direction
    pip_size = settings.PAIR_PIP_SIZES.get(pair, settings.PIP_SIZE)

    logger.info(f"🎯 Entry Engine analyzing {direction} {pair} @ {current_price}...")

    # Build the prompt context
    system_prompt = _build_entry_prompt(pair, direction, current_price, pip_size)

    # Format OHLCV candle tables
    ohlcv_context = format_ohlcv_for_entry(pair, raw_ohlcv)

    # Format SMC structures from market data
    pair_data = market_data.pairs.get(pair)
    smc_context = _format_smc_for_entry(pair_data) if pair_data else "No SMC data available."

    user_message = f"""Find the best {direction} entry for {pair}.

Current Price: {current_price}
Direction: {direction}
Trader Brain Score: {decision.total_score}/{settings.get_total_max_score()} ({settings.get_score_pct(decision.total_score) * 100:.0f}%)

═══ OHLCV CANDLE DATA ═══
{ohlcv_context}

═══ SMC STRUCTURE DATA ═══
{smc_context}

Analyze the candles and structures above. Find the OPTIMAL entry price, SL, and TP."""

    try:
        client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )

        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            temperature=0.15,  # low for precise price levels
            max_tokens=1500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        raw_content = response.choices[0].message.content.strip()
        logger.debug(f"Entry Engine raw response: {raw_content[:500]}...")

        # Parse JSON (handle markdown code blocks)
        json_str = _extract_json(raw_content)
        data = json.loads(json_str)

        # Build EntryCandidate
        candidate = EntryCandidate(
            entry_type=data.get("entry_type", "market"),
            priority=data.get("priority", 5),
            entry_price=float(data.get("entry_price", current_price)),
            sl_price=float(data.get("sl_price", 0)),
            sl_pips=int(data.get("sl_pips", 0)),
            tp_price=float(data.get("tp_price", 0)),
            tp_pips=int(data.get("tp_pips", 0)),
            rr_ratio=float(data.get("rr_ratio", 0)),
            confluence_count=int(data.get("confluence_count", 1)),
            distance_from_current=float(data.get("distance_from_current", 0)),
            reasoning=data.get("reasoning", ""),
        )

        # Validate and fix the candidate
        candidate = _validate_candidate(candidate, direction, current_price, pip_size)

        _log_candidate(candidate, pair, direction)
        return candidate

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Entry Engine response as JSON: {e}")
        logger.error(f"Raw response: {raw_content[:1000]}")
        # Fallback: enter at market with basic structure
        return _fallback_market_entry(direction, current_price, pair, market_data, pip_size)

    except Exception as e:
        logger.error(f"Entry Engine error: {e}")
        return _fallback_market_entry(direction, current_price, pair, market_data, pip_size)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_json(raw: str) -> str:
    """Extract JSON from a response that may be wrapped in markdown code blocks or have thinking tags."""
    import re
    text = raw

    # Strip <think>...</think> blocks (reasoning models)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Strip markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(json_lines)

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    return text


def _format_smc_for_entry(pair_data) -> str:
    """Format SMC data in detail for the Entry Engine."""
    lines = []
    for tf_key, smc_d in pair_data.smc.items():
        lines.append(f"\n── {pair_data.pair} {tf_key.upper()} Smart Money Concepts ──")
        lines.append(f"  Swing High: {smc_d.latest_swing_high}  |  Swing Low: {smc_d.latest_swing_low}")
        lines.append(f"  FVG Bullish: {smc_d.fvg_bullish_count}  |  FVG Bearish: {smc_d.fvg_bearish_count}")
        if smc_d.fvg_nearest_level:
            lines.append(f"  Nearest FVG: {smc_d.fvg_nearest_type} @ {smc_d.fvg_nearest_level}")
        lines.append(f"  OB Bullish: {smc_d.ob_bullish_count}  |  OB Bearish: {smc_d.ob_bearish_count}")
        if smc_d.ob_nearest_top:
            lines.append(
                f"  Nearest OB: {smc_d.ob_nearest_type} "
                f"[{smc_d.ob_nearest_bottom} - {smc_d.ob_nearest_top}]"
            )
        if smc_d.last_bos_type:
            lines.append(f"  Last BOS: {smc_d.last_bos_type} @ {smc_d.last_bos_level}")
        if smc_d.last_choch_type:
            lines.append(f"  Last CHoCH: {smc_d.last_choch_type} @ {smc_d.last_choch_level}")
        lines.append(f"  Liquidity Swept: {smc_d.liquidity_swept}")
        if smc_d.liquidity_swept:
            recency = f" ({smc_d.liquidity_candles_ago} candles ago)" if smc_d.liquidity_candles_ago is not None else ""
            lines.append(
                f"  Sweep Type: {smc_d.liquidity_sweep_type} @ {smc_d.liquidity_level}{recency}"
            )
        if smc_d.current_retracement_pct:
            lines.append(f"  Current Retracement: {smc_d.current_retracement_pct:.1f}%")
        if smc_d.deepest_retracement_pct:
            lines.append(f"  Deepest Retracement: {smc_d.deepest_retracement_pct:.1f}%")

    return "\n".join(lines)


def _validate_candidate(
    candidate: EntryCandidate,
    direction: str,
    current_price: float,
    pip_size: float,
) -> EntryCandidate:
    """
    Validate and fix an entry candidate from the AI.
    Ensures SL is on the correct side, pips are reasonable, etc.
    """
    # Fix entry_price if 0 or missing
    if candidate.entry_price <= 0:
        candidate.entry_price = current_price

    # Ensure SL is on the correct side
    if direction == "BUY" and candidate.sl_price >= candidate.entry_price:
        candidate.sl_price = candidate.entry_price - (settings.MIN_SL_PIPS * pip_size)

    if direction == "SELL" and candidate.sl_price <= candidate.entry_price:
        candidate.sl_price = candidate.entry_price + (settings.MIN_SL_PIPS * pip_size)

    # Recompute SL pips from prices
    sl_distance = abs(candidate.entry_price - candidate.sl_price)
    candidate.sl_pips = max(settings.MIN_SL_PIPS, int(sl_distance / pip_size))

    # Clamp SL to bounds
    candidate.sl_pips = min(candidate.sl_pips, settings.MAX_SL_PIPS)

    # Recompute SL price from clamped pips
    if direction == "BUY":
        candidate.sl_price = round(candidate.entry_price - (candidate.sl_pips * pip_size), 5)
    else:
        candidate.sl_price = round(candidate.entry_price + (candidate.sl_pips * pip_size), 5)

    # Ensure TP is on the correct side
    if direction == "BUY" and candidate.tp_price <= candidate.entry_price:
        candidate.tp_price = round(
            candidate.entry_price + (candidate.sl_pips * settings.TP_MIN_RR * pip_size), 5
        )
    if direction == "SELL" and candidate.tp_price >= candidate.entry_price:
        candidate.tp_price = round(
            candidate.entry_price - (candidate.sl_pips * settings.TP_MIN_RR * pip_size), 5
        )

    # Recompute TP pips
    candidate.tp_pips = int(abs(candidate.tp_price - candidate.entry_price) / pip_size)

    # Ensure minimum R:R
    if candidate.sl_pips > 0:
        if candidate.tp_pips < candidate.sl_pips * settings.TP_MIN_RR:
            candidate.tp_pips = int(candidate.sl_pips * settings.TP_MIN_RR)
            if direction == "BUY":
                candidate.tp_price = round(candidate.entry_price + (candidate.tp_pips * pip_size), 5)
            else:
                candidate.tp_price = round(candidate.entry_price - (candidate.tp_pips * pip_size), 5)

    # Compute R:R ratio
    if candidate.sl_pips > 0:
        candidate.rr_ratio = round(candidate.tp_pips / candidate.sl_pips, 2)

    # Compute distance from current
    candidate.distance_from_current = round(abs(candidate.entry_price - current_price) / pip_size, 1)

    # Clamp priority
    candidate.priority = max(1, min(candidate.priority, 5))

    return candidate


def _fallback_market_entry(
    direction: str,
    current_price: float,
    pair: str,
    market_data: MarketDataPayload,
    pip_size: float,
) -> EntryCandidate:
    """
    Fallback: enter at market price using swing structure for SL/TP.
    Used when the AI fails to produce a valid response.
    """
    logger.warning("⚠️ Using fallback market entry (AI entry engine failed)")

    pair_data = market_data.pairs.get(pair)
    swing_low = None
    swing_high = None

    if pair_data:
        for tf in ["15m", "1h"]:
            smc_data = pair_data.smc.get(tf)
            if smc_data:
                if smc_data.latest_swing_low and swing_low is None:
                    swing_low = smc_data.latest_swing_low
                if smc_data.latest_swing_high and swing_high is None:
                    swing_high = smc_data.latest_swing_high

    # Compute SL from swings
    if direction == "BUY":
        if swing_low:
            sl_price = swing_low - (settings.SL_BUFFER_PIPS * pip_size)
        else:
            sl_price = current_price - (15 * pip_size)  # 15 pip default SL
        sl_pips = max(settings.MIN_SL_PIPS, int(abs(current_price - sl_price) / pip_size))
    else:
        if swing_high:
            sl_price = swing_high + (settings.SL_BUFFER_PIPS * pip_size)
        else:
            sl_price = current_price + (15 * pip_size)
        sl_pips = max(settings.MIN_SL_PIPS, int(abs(sl_price - current_price) / pip_size))

    sl_pips = min(sl_pips, settings.MAX_SL_PIPS)

    # Compute TP
    tp_pips = int(sl_pips * settings.TP_MIN_RR)

    if direction == "BUY":
        sl_price = round(current_price - (sl_pips * pip_size), 5)
        tp_price = round(current_price + (tp_pips * pip_size), 5)
    else:
        sl_price = round(current_price + (sl_pips * pip_size), 5)
        tp_price = round(current_price - (tp_pips * pip_size), 5)

    return EntryCandidate(
        entry_type="market",
        priority=5,
        entry_price=current_price,
        sl_price=sl_price,
        sl_pips=sl_pips,
        tp_price=tp_price,
        tp_pips=tp_pips,
        rr_ratio=round(tp_pips / sl_pips, 2) if sl_pips > 0 else 0,
        confluence_count=1,
        distance_from_current=0,
        reasoning="Fallback market entry — AI entry engine was unavailable. "
                  "Using swing structure for SL, R:R multiplier for TP.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def _log_candidate(candidate: EntryCandidate, pair: str, direction: str):
    """Log the selected entry candidate."""
    type_emoji = {
        "sweep": "🥇", "ob": "🥈", "fvg": "🥉",
        "swing": "4️⃣", "round": "4️⃣", "market": "⚡"
    }
    emoji = type_emoji.get(candidate.entry_type, "📍")

    logger.info(f"\n{'─' * 60}")
    logger.info(f"{emoji} ENTRY ENGINE: {candidate.entry_type.upper()} entry (priority {candidate.priority})")
    logger.info(f"   {direction} {pair} @ {candidate.entry_price}")
    logger.info(f"   SL: {candidate.sl_price} ({candidate.sl_pips} pips)")
    logger.info(f"   TP: {candidate.tp_price} ({candidate.tp_pips} pips)")
    logger.info(f"   R:R: 1:{candidate.rr_ratio:.1f}")
    logger.info(f"   Confluence: {candidate.confluence_count} structures")
    logger.info(f"   Distance: {candidate.distance_from_current} pips from current")
    logger.info(f"   Reason: {candidate.reasoning[:300]}")
    logger.info(f"{'─' * 60}")
