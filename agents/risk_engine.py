"""
risk_engine.py — AI Agent 2: Risk Engine.

Architecture (per diagram):
  Receives: scores + direction + user profile + live price + swing structure
  Calculates: natural SL from swing structure, lot size from budget
  Validates: SL fits max_loss, daily trade count, daily loss limit
  Output: APPROVED (lots, SL, TP) or REJECTED (reason)

The Risk Engine computes SL/TP — the Trader Brain does NOT set these.
SL is placed beyond the nearest swing high/low + buffer pips.
TP is computed from the R:R tier requirement.
"""

import json

from loguru import logger
from openai import OpenAI

from config import settings
from core.models import (
    MarketDataPayload,
    PriceData,
    RiskApproval,
    TraderDecision,
)
from services.mt5_client import MT5Client


# ═══════════════════════════════════════════════════════════════════════════════
# RISK LIMIT CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_risk_limits(balance: float) -> dict[str, float]:
    """
    Compute the actual USD risk limits from the live account balance.

    Args:
        balance: current MT5 account balance in USD

    Returns:
        dict with 'max_loss_per_trade_usd' and 'daily_loss_limit_usd'
    """
    max_loss_per_trade_usd = round(balance * (settings.MAX_LOSS_PER_TRADE_PCT / 100.0), 2)
    daily_loss_limit_usd = round(balance * (settings.DAILY_LOSS_LIMIT_PCT / 100.0), 2)

    return {
        "max_loss_per_trade_usd": max_loss_per_trade_usd,
        "daily_loss_limit_usd": daily_loss_limit_usd,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SL COMPUTATION FROM SWING STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sl_from_swings(
    entry_price: float,
    direction: str,
    pair: str,
    market_data: MarketDataPayload,
) -> int | None:
    """
    Compute SL in pips from the nearest swing high/low in SMC data.

    For BUY:  SL below the nearest swing low  - buffer
    For SELL: SL above the nearest swing high + buffer

    Returns SL in pips, or None if no swing data available.
    """
    pip_size = settings.PAIR_PIP_SIZES.get(pair, settings.PIP_SIZE)
    pair_data = market_data.pairs.get(pair)
    if not pair_data:
        return None

    # Try 15m swings first (tighter SL), then 1h (wider but safer)
    swing_low = None
    swing_high = None

    for tf in ["15m", "1h"]:
        smc_data = pair_data.smc.get(tf)
        if smc_data:
            if smc_data.latest_swing_low and swing_low is None:
                swing_low = smc_data.latest_swing_low
            if smc_data.latest_swing_high and swing_high is None:
                swing_high = smc_data.latest_swing_high

    if direction == "BUY":
        if swing_low is None:
            return None
        sl_distance = entry_price - swing_low
        sl_pips = int(sl_distance / pip_size) + settings.SL_BUFFER_PIPS
    elif direction == "SELL":
        if swing_high is None:
            return None
        sl_distance = swing_high - entry_price
        sl_pips = int(sl_distance / pip_size) + settings.SL_BUFFER_PIPS
    else:
        return None

    # Clamp to bounds
    sl_pips = max(settings.MIN_SL_PIPS, min(sl_pips, settings.MAX_SL_PIPS))
    return sl_pips


# ═══════════════════════════════════════════════════════════════════════════════
# LOT SIZE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_lot_size(
    max_loss_usd: float,
    sl_pips: int,
    pair: str = "EURUSD",
    live_spread_pips: float | None = None,
) -> float | None:
    """
    Calculate the lot size based on risk parameters.

    Args:
        max_loss_usd: max amount willing to lose per trade (already in USD)
        sl_pips:      stop loss distance in pips
        pair:         trading pair (for pip value lookup)
        live_spread_pips: live spread in pips (falls back to settings.SPREAD_PIPS)

    Returns:
        lot size (float) or None if SL is too wide for budget
    """
    pip_value_micro = settings.PAIR_PIP_VALUES_MICRO.get(pair, settings.PIP_VALUE_PER_MICRO_LOT)
    spread_pips = live_spread_pips if live_spread_pips is not None else settings.SPREAD_PIPS

    # Total pip exposure = SL + spread
    total_pip_risk = sl_pips + spread_pips

    # At minimum lot (0.01), what would we lose?
    min_lot_risk = total_pip_risk * pip_value_micro

    if min_lot_risk > max_loss_usd:
        return None  # Can't even afford minimum lot — REJECT

    # How many micro lots can we afford?
    # lot_size = max_loss_usd / (total_pip_risk * pip_value_per_lot)
    # pip_value_per_lot = pip_value_micro / MIN_LOT (= pip_value_micro * 100)
    pip_value_per_full_lot = pip_value_micro / settings.MIN_LOT
    lot_size = max_loss_usd / (total_pip_risk * pip_value_per_full_lot)
    lot_size = round(lot_size, 2)
    lot_size = max(settings.MIN_LOT, min(lot_size, settings.MAX_LOT))

    return lot_size


def pips_to_price(entry_price: float, pips: int, direction: str, side: str, pair: str = "EURUSD") -> float:
    """
    Convert pips to a price level.

    Args:
        entry_price: current entry price
        pips:        number of pips
        direction:   "BUY" or "SELL"
        side:        "sl" or "tp"
        pair:        trading pair (for pip size lookup)

    Returns:
        price level rounded to 5 decimal places
    """
    pip = settings.PAIR_PIP_SIZES.get(pair, settings.PIP_SIZE)

    if direction == "BUY":
        if side == "sl":
            return round(entry_price - (pips * pip), 5)
        else:  # tp
            return round(entry_price + (pips * pip), 5)
    else:  # SELL
        if side == "sl":
            return round(entry_price + (pips * pip), 5)
        else:  # tp
            return round(entry_price - (pips * pip), 5)


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC RISK CHECKS (before AI call)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_hard_risk_checks(
    decision: TraderDecision,
    balance: float,
    trades_today_count: int,
    daily_pnl: float,
    take_attempts_today: int,
    mt5: MT5Client,
) -> RiskApproval | None:
    """
    Run deterministic risk checks before doing any computation.
    Returns a REJECTED RiskApproval if any check fails, or None if all pass.
    """
    risk_limits = compute_risk_limits(balance)
    daily_limit_usd = risk_limits["daily_loss_limit_usd"]

    # 1. Daily trade limit (successful executions)
    if trades_today_count >= settings.MAX_TRADES_PER_DAY:
        return RiskApproval(
            approved=False,
            reason=f"Daily trade limit reached: {trades_today_count}/{settings.MAX_TRADES_PER_DAY}",
        )

    # 2. Daily TAKE attempt limit (prevents rejection loops)
    if take_attempts_today >= settings.MAX_TAKE_ATTEMPTS_PER_DAY:
        return RiskApproval(
            approved=False,
            reason=f"Daily TAKE attempt limit reached: {take_attempts_today}/{settings.MAX_TAKE_ATTEMPTS_PER_DAY}",
        )

    # 3. Daily loss limit (percentage-based)
    if daily_pnl <= -daily_limit_usd:
        return RiskApproval(
            approved=False,
            reason=(
                f"Daily loss limit reached: ${daily_pnl:.2f} "
                f"(limit: -${daily_limit_usd:.2f} = {settings.DAILY_LOSS_LIMIT_PCT}% of ${balance:.2f})"
            ),
        )

    # 4. Existing position on same pair
    if decision.pair and mt5.has_open_position(decision.pair):
        return RiskApproval(
            approved=False,
            reason=f"Already have an open position on {decision.pair}",
        )

    # All hard checks passed
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# AI RISK VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _build_risk_prompt(max_loss_usd: float, daily_limit_usd: float) -> str:
    """Build the system prompt for the Risk Engine AI."""
    return f"""You are a strict forex risk manager. Your job is to validate or reject a trade that has been proposed by a trader AI.

You receive:
- The proposed trade details (pair, direction, scores)
- Computed SL/TP prices (from swing structure)
- Computed lot size (from risk budget)
- Account information (balance, equity)
- Today's trade count and P&L

Your ONLY job is to:
1. Verify the SL placement makes structural sense (not in the middle of a range)
2. Verify the R:R ratio meets the minimum for the score tier
3. Confirm the lot size doesn't exceed the risk budget
4. Check that daily limits are respected

R:R Requirements by score tier:
{chr(10).join(f"  Score {t['min_score']}-{t['max_score']}: min R:R 1:{t['rr_ratio']:.0f}" for t in settings.RR_TIERS)}

Default minimum R:R: 1:{settings.MIN_RR_RATIO}

You MUST respond with ONLY valid JSON:
{{
  "approved": true | false,
  "reason": "<explanation>"
}}

Be conservative. When in doubt, REJECT. Capital preservation is paramount."""


def evaluate(
    decision: TraderDecision,
    mt5: MT5Client,
    market_data: MarketDataPayload,
    trades_today_count: int = 0,
    daily_pnl: float = 0.0,
    take_attempts_today: int = 0,
) -> RiskApproval:
    """
    Evaluate a trade decision through the risk engine.

    Architecture:
    1. Run deterministic hard checks first (fast, free)
    2. Compute SL from swing structure
    3. Compute lot size from budget
    4. Compute TP from R:R tier
    5. Call AI for final sanity validation
    6. Return APPROVED or REJECTED with details
    """
    logger.info("⚖️  Running Risk Engine...")

    # ── Fetch account balance ONCE ──
    account = mt5.get_account_info()
    balance = account.balance if account else 0

    if balance <= 0:
        return RiskApproval(
            approved=False,
            reason="Cannot determine account balance — MT5 might be down",
        )

    risk_limits = compute_risk_limits(balance)
    max_loss_usd = risk_limits["max_loss_per_trade_usd"]
    daily_limit_usd = risk_limits["daily_loss_limit_usd"]

    logger.info(
        f"Risk limits from ${balance:.2f} balance: "
        f"per-trade=${max_loss_usd:.2f} ({settings.MAX_LOSS_PER_TRADE_PCT}%), "
        f"daily=${daily_limit_usd:.2f} ({settings.DAILY_LOSS_LIMIT_PCT}%)"
    )

    # ── Hard checks (deterministic) ──
    hard_reject = _run_hard_risk_checks(
        decision, balance, trades_today_count, daily_pnl, take_attempts_today, mt5,
    )
    if hard_reject:
        logger.info(f"🚫 Risk REJECTED (hard check): {hard_reject.reason}")
        return hard_reject

    # ── Get live price ──
    pair = decision.pair
    price_data = mt5.get_price(pair) if pair else None
    if not price_data:
        return RiskApproval(
            approved=False,
            reason=f"Cannot get live price for {pair} — MT5 might be down",
        )

    # ── Check spread ──
    if price_data.spread_pips > settings.SPREAD_MAX_PIPS:
        return RiskApproval(
            approved=False,
            reason=f"Spread too high: {price_data.spread_pips:.1f} pips (max {settings.SPREAD_MAX_PIPS})",
        )

    # ── Compute entry price ──
    entry_price = price_data.ask if decision.direction == "BUY" else price_data.bid

    # ── Compute SL from swing structure ──
    sl_pips = compute_sl_from_swings(entry_price, decision.direction, pair, market_data)

    if sl_pips is None:
        # Fallback: no swing data → reject (fail-closed approach per architecture)
        return RiskApproval(
            approved=False,
            reason="No swing structure data to compute SL — cannot determine safe stop level",
        )

    logger.info(f"📐 SL computed from swing structure: {sl_pips} pips")

    # ── Compute lot size ──
    lots = calculate_lot_size(
        max_loss_usd=max_loss_usd,
        sl_pips=sl_pips,
        pair=pair,
        live_spread_pips=price_data.spread_pips,
    )

    if lots is None:
        max_affordable_sl = settings.compute_max_sl_pips(balance, pair)
        return RiskApproval(
            approved=False,
            reason=(
                f"SL {sl_pips} pips too wide for budget "
                f"(${max_loss_usd:.2f} = {settings.MAX_LOSS_PER_TRADE_PCT}% of ${balance:.2f}, "
                f"max affordable SL: {max_affordable_sl} pips)"
            ),
        )

    # ── Determine required R:R from score tier ──
    required_rr = settings.MIN_RR_RATIO
    for tier in settings.RR_TIERS:
        if tier["min_score"] <= decision.total_score <= tier["max_score"]:
            required_rr = tier["rr_ratio"]
            break

    # ── Compute TP from R:R requirement ──
    tp_pips = int(sl_pips * required_rr)
    actual_rr = tp_pips / sl_pips if sl_pips > 0 else 0

    # ── Compute price levels ──
    sl_price = pips_to_price(entry_price, sl_pips, decision.direction, "sl", pair)
    tp_price = pips_to_price(entry_price, tp_pips, decision.direction, "tp", pair)

    logger.info(
        f"📊 Trade params: {lots} lots | SL={sl_pips}pips ({sl_price}) | "
        f"TP={tp_pips}pips ({tp_price}) | R:R 1:{actual_rr:.1f} (required 1:{required_rr:.0f})"
    )

    # ── AI validation (lightweight sanity check) ──
    try:
        client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )

        pip_value_micro = settings.PAIR_PIP_VALUES_MICRO.get(pair, settings.PIP_VALUE_PER_MICRO_LOT)

        risk_context = f"""Proposed Trade:
  Pair: {pair}
  Direction: {decision.direction}
  Total Score: {decision.total_score}
  Phase 1: {decision.phase1_total}/{settings.PHASE1_MAX_SCORE}
  Phase 2: {decision.phase2_total}/{settings.PHASE2_MAX_SCORE}

Computed from swing structure:
  Entry: {entry_price} ({decision.direction})
  SL: {sl_pips} pips → {sl_price}
  TP: {tp_pips} pips → {tp_price}
  R:R: 1:{actual_rr:.1f} (required: 1:{required_rr:.0f})

Live Price:
  Bid: {price_data.bid} | Ask: {price_data.ask} | Spread: {price_data.spread_pips} pips

Account:
  Balance: ${balance:.2f}
  Lot Size: {lots}
  Max Loss Per Trade: ${max_loss_usd:.2f} ({settings.MAX_LOSS_PER_TRADE_PCT}% of balance)
  Risk at {lots} lots: ${lots * pip_value_micro / settings.MIN_LOT * sl_pips:.2f}

Daily Stats:
  Trades today: {trades_today_count}/{settings.MAX_TRADES_PER_DAY}
  Daily P&L: ${daily_pnl:.2f} (limit: -${daily_limit_usd:.2f})

Trader Brain Reasoning: {decision.reasoning[:500]}"""

        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            temperature=0.1,  # very low for risk decisions
            max_tokens=300,
            messages=[
                {"role": "system", "content": _build_risk_prompt(max_loss_usd, daily_limit_usd)},
                {"role": "user", "content": risk_context},
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Parse JSON
        json_str = raw
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_lines = [l for l in lines if not l.strip().startswith("```")]
            json_str = "\n".join(json_lines)

        data = json.loads(json_str)

        ai_approved = data.get("approved", False)
        ai_reason = data.get("reason", "")

        if not ai_approved:
            logger.info(f"🚫 Risk AI REJECTED: {ai_reason}")
            return RiskApproval(
                approved=False,
                reason=f"Risk AI rejected: {ai_reason}",
            )

        # AI approved — build full approval
        logger.info(f"✅ Risk APPROVED: {lots} lots | SL={sl_price} TP={tp_price} | R:R 1:{actual_rr:.1f}")

        return RiskApproval(
            approved=True,
            lots=lots,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            entry_price=entry_price,
            reason=ai_reason or f"Approved: {lots} lots, R:R 1:{actual_rr:.1f}",
            rr_ratio=actual_rr,
        )

    except Exception as e:
        logger.error(f"Risk Engine AI error: {e}")
        # Fallback: use deterministic approval (all hard checks already passed)
        logger.info("Falling back to deterministic risk approval...")

        if actual_rr >= required_rr:
            return RiskApproval(
                approved=True,
                lots=lots,
                sl_price=sl_price,
                tp_price=tp_price,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                entry_price=entry_price,
                reason=f"Deterministic approval (AI unavailable). {lots} lots, R:R=1:{actual_rr:.1f}",
                rr_ratio=actual_rr,
            )
        else:
            return RiskApproval(
                approved=False,
                reason=f"R:R {actual_rr:.1f} below required {required_rr} (AI unavailable, deterministic reject)",
            )
