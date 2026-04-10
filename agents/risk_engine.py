"""
risk_engine.py — AI Agent 2: Risk Engine.

Validates the Trader Brain's EXECUTE decision against strict risk rules.
Uses OpenAI-compatible API with a separate focused prompt.

Checks:
  - Lot size fits within MAX_LOSS_PER_TRADE
  - SL distance is achievable
  - R:R meets tier requirements
  - Daily trade count ≤ MAX_TRADES_PER_DAY
  - Daily P&L loss ≤ DAILY_LOSS_LIMIT_USD
  - No existing position on same pair
"""

import json

from loguru import logger
from openai import OpenAI

from config import settings
from core.models import PriceData, RiskApproval, TraderDecision
from services.mt5_client import MT5Client


# ═══════════════════════════════════════════════════════════════════════════════
# LOT SIZE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_lot_size(
    balance: float,
    max_loss_usd: float,
    sl_pips: int,
    spread_pips: float,
) -> float | None:
    """
    Calculate the lot size based on risk parameters.

    Args:
        balance:      current account balance
        max_loss_usd: max amount willing to lose per trade
        sl_pips:      stop loss distance in pips
        spread_pips:  broker spread in pips

    Returns:
        lot size (float) or None if SL is too wide for budget
    """
    pip_value_micro = settings.PIP_VALUE_PER_MICRO_LOT  # $0.10 per pip at 0.01 lot

    # Subtract spread cost from risk budget
    spread_cost = spread_pips * pip_value_micro
    net_risk = max_loss_usd - spread_cost

    if net_risk <= 0:
        return settings.MIN_LOT  # minimum lot

    # How many pips can we afford at 0.01 lot?
    max_sl_pips_at_micro = net_risk / pip_value_micro

    if sl_pips > max_sl_pips_at_micro:
        return None  # SL too wide for budget — SKIP trade

    # Scale up if budget allows
    lot_size = round(net_risk / (sl_pips * pip_value_micro * 10), 2)
    lot_size = max(settings.MIN_LOT, min(lot_size, settings.MAX_LOT))

    return lot_size


def pips_to_price(entry_price: float, pips: int, direction: str, side: str) -> float:
    """
    Convert pips to a price level.

    Args:
        entry_price: current entry price
        pips:        number of pips
        direction:   "BUY" or "SELL"
        side:        "sl" or "tp"

    Returns:
        price level rounded to 5 decimal places
    """
    pip = settings.PIP_SIZE  # 0.0001 for EUR/GBP pairs

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
    mt5: MT5Client,
    trades_today_count: int,
    daily_pnl: float,
) -> RiskApproval | None:
    """
    Run deterministic risk checks before calling the AI.
    Returns a REJECTED RiskApproval if any check fails, or None if all pass.
    """
    # 1. Daily trade limit
    if trades_today_count >= settings.MAX_TRADES_PER_DAY:
        return RiskApproval(
            approved=False,
            reason=f"Daily trade limit reached: {trades_today_count}/{settings.MAX_TRADES_PER_DAY}",
        )

    # 2. Daily loss limit
    if daily_pnl <= -settings.DAILY_LOSS_LIMIT_USD:
        return RiskApproval(
            approved=False,
            reason=f"Daily loss limit reached: ${daily_pnl:.2f} (limit: -${settings.DAILY_LOSS_LIMIT_USD})",
        )

    # 3. Existing position on same pair
    if decision.pair and mt5.has_open_position(decision.pair):
        return RiskApproval(
            approved=False,
            reason=f"Already have an open position on {decision.pair}",
        )

    # 4. SL pips validation
    if not decision.sl_pips or decision.sl_pips <= 0:
        return RiskApproval(
            approved=False,
            reason="Invalid or missing SL pips from Trader Brain",
        )

    # 5. Lot size calculation
    account = mt5.get_account_info()
    balance = account.balance if account else 0

    lots = calculate_lot_size(
        balance=balance,
        max_loss_usd=settings.MAX_LOSS_PER_TRADE,
        sl_pips=decision.sl_pips,
        spread_pips=settings.SPREAD_PIPS,
    )

    if lots is None:
        return RiskApproval(
            approved=False,
            reason=f"SL too wide ({decision.sl_pips} pips) for risk budget (${settings.MAX_LOSS_PER_TRADE})",
        )

    # 6. R:R ratio check
    if decision.rr_ratio and decision.rr_ratio < settings.MIN_RR_RATIO:
        return RiskApproval(
            approved=False,
            reason=f"R:R ratio {decision.rr_ratio} below minimum {settings.MIN_RR_RATIO}",
        )

    # All hard checks passed
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# AI RISK VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def _build_risk_prompt() -> str:
    """Build the system prompt for the Risk Engine AI."""
    return f"""You are a strict forex risk manager. Your job is to validate or reject a trade that has been proposed by a trader AI.

You receive:
- The proposed trade details (pair, direction, scores, SL/TP in pips)
- The current live price (bid/ask)
- Account information (balance, equity, open positions)
- Today's trade count and P&L

Your ONLY job is to:
1. Validate the lot size fits within the max loss budget (${settings.MAX_LOSS_PER_TRADE}/trade)
2. Calculate exact SL and TP price levels from the entry price + pips
3. Verify the R:R ratio meets the minimum requirement
4. Check that daily limits are respected

R:R Requirements by score tier:
{chr(10).join(f"  Score {t['min_score']}-{t['max_score']}: min R:R 1:{t['rr_ratio']:.0f}" for t in settings.RR_TIERS)}

Default minimum R:R: 1:{settings.MIN_RR_RATIO}

You MUST respond with ONLY valid JSON:
{{
  "approved": true | false,
  "lots": <float, e.g. 0.01>,
  "entry_price": <float>,
  "sl_price": <float>,
  "tp_price": <float>,
  "rr_ratio": <float>,
  "reason": "<explanation>"
}}

Be conservative. When in doubt, REJECT. Capital preservation is paramount."""


def evaluate(
    decision: TraderDecision,
    mt5: MT5Client,
    trades_today_count: int = 0,
    daily_pnl: float = 0.0,
) -> RiskApproval:
    """
    Evaluate a trade decision through the risk engine.

    1. Run deterministic hard checks first (fast, free)
    2. If hard checks pass, call AI for final validation
    3. Return APPROVED or REJECTED with details
    """
    logger.info("⚖️  Running Risk Engine...")

    # ── Hard checks (deterministic) ──
    hard_reject = _run_hard_risk_checks(decision, mt5, trades_today_count, daily_pnl)
    if hard_reject:
        logger.info(f"🚫 Risk REJECTED (hard check): {hard_reject.reason}")
        return hard_reject

    # ── Get live price ──
    price_data = mt5.get_price(decision.pair) if decision.pair else None
    if not price_data:
        return RiskApproval(
            approved=False,
            reason=f"Cannot get live price for {decision.pair} — MT5 might be down",
        )

    # ── Calculate entry, SL, TP prices ──
    entry_price = price_data.ask if decision.direction == "BUY" else price_data.bid
    sl_price = pips_to_price(entry_price, decision.sl_pips, decision.direction, "sl")
    tp_pips = decision.tp_pips or int(decision.sl_pips * settings.MIN_RR_RATIO)
    tp_price = pips_to_price(entry_price, tp_pips, decision.direction, "tp")

    # ── Calculate lot size ──
    account = mt5.get_account_info()
    balance = account.balance if account else 0

    lots = calculate_lot_size(
        balance=balance,
        max_loss_usd=settings.MAX_LOSS_PER_TRADE,
        sl_pips=decision.sl_pips,
        spread_pips=price_data.spread_pips,
    )

    if lots is None:
        return RiskApproval(
            approved=False,
            reason=f"SL {decision.sl_pips} pips too wide for budget after spread {price_data.spread_pips}",
        )

    # ── Determine required R:R from score tier ──
    required_rr = settings.MIN_RR_RATIO
    for tier in settings.RR_TIERS:
        if tier["min_score"] <= decision.total_score <= tier["max_score"]:
            required_rr = tier["rr_ratio"]
            break

    actual_rr = tp_pips / decision.sl_pips if decision.sl_pips > 0 else 0

    # ── AI validation ──
    try:
        client = OpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )

        risk_context = f"""Proposed Trade:
  Pair: {decision.pair}
  Direction: {decision.direction}
  Total Score: {decision.total_score}
  Phase 1: {decision.phase1_total}/{settings.PHASE1_MAX_SCORE}
  Phase 2: {decision.phase2_total}/{settings.PHASE2_MAX_SCORE}
  SL: {decision.sl_pips} pips | TP: {tp_pips} pips
  R:R: 1:{actual_rr:.1f} (required: 1:{required_rr:.0f})

Live Price:
  Bid: {price_data.bid} | Ask: {price_data.ask} | Spread: {price_data.spread_pips} pips
  Entry: {entry_price} ({decision.direction})
  SL Price: {sl_price}
  TP Price: {tp_price}

Account:
  Balance: ${balance:.2f}
  Calculated Lot Size: {lots}
  Max Loss: ${settings.MAX_LOSS_PER_TRADE}
  Risk per pip at {lots} lots: ${lots * settings.PIP_VALUE_PER_MICRO_LOT / settings.MIN_LOT:.4f}

Daily Stats:
  Trades today: {trades_today_count}/{settings.MAX_TRADES_PER_DAY}
  Daily P&L: ${daily_pnl:.2f} (limit: -${settings.DAILY_LOSS_LIMIT_USD})

Trader Brain Reasoning: {decision.reasoning[:500]}"""

        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            temperature=0.1,  # very low for risk decisions
            max_tokens=500,
            messages=[
                {"role": "system", "content": _build_risk_prompt()},
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

        approval = RiskApproval(
            approved=data.get("approved", False),
            lots=data.get("lots", lots),
            sl_price=data.get("sl_price", sl_price),
            tp_price=data.get("tp_price", tp_price),
            entry_price=data.get("entry_price", entry_price),
            reason=data.get("reason", ""),
            rr_ratio=data.get("rr_ratio", actual_rr),
        )

        # Safety override: never exceed our calculated lot size
        if approval.lots > lots:
            approval.lots = lots

        # Safety override: lot size bounds
        approval.lots = max(settings.MIN_LOT, min(approval.lots, settings.MAX_LOT))

        if approval.approved:
            logger.info(f"✅ Risk APPROVED: {approval.lots} lots | "
                        f"SL={approval.sl_price} TP={approval.tp_price} | "
                        f"R:R 1:{approval.rr_ratio:.1f}")
        else:
            logger.info(f"🚫 Risk REJECTED: {approval.reason}")

        return approval

    except Exception as e:
        logger.error(f"Risk Engine AI error: {e}")
        # Fallback: use deterministic calculation
        logger.info("Falling back to deterministic risk approval...")

        if actual_rr >= required_rr:
            return RiskApproval(
                approved=True,
                lots=lots,
                sl_price=sl_price,
                tp_price=tp_price,
                entry_price=entry_price,
                reason=f"Deterministic approval (AI unavailable). Lots={lots}, R:R=1:{actual_rr:.1f}",
                rr_ratio=actual_rr,
            )
        else:
            return RiskApproval(
                approved=False,
                reason=f"R:R {actual_rr:.1f} below required {required_rr} (AI unavailable, deterministic reject)",
            )
