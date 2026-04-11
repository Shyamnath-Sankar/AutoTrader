"""
risk_engine.py — AI Agent 2: Risk Engine.

Validates the Trader Brain's EXECUTE decision against strict risk rules.
Uses OpenAI-compatible API with a separate focused prompt.

Checks:
  - Lot size fits within MAX_LOSS_PER_TRADE (5% of balance)
  - SL distance is achievable
  - R:R meets tier requirements
  - Daily trade count ≤ MAX_TRADES_PER_DAY
  - Daily P&L loss ≤ DAILY_LOSS_LIMIT (15% of balance)
  - No existing position on same pair
"""

import json

from loguru import logger
from openai import OpenAI

from config import settings
from core.models import PriceData, RiskApproval, TraderDecision
from services.mt5_client import MT5Client


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC RISK CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_risk_limits(balance: float) -> dict:
    """
    Compute dynamic risk limits based on current account balance.

    Returns:
        dict with max_loss_per_trade and daily_loss_limit in USD
    """
    max_loss_per_trade = balance * settings.MAX_LOSS_PER_TRADE_PCT
    daily_loss_limit = balance * settings.DAILY_LOSS_LIMIT_PCT

    # Fallback to fixed values if balance is 0 or tiny
    if max_loss_per_trade < 0.01:
        max_loss_per_trade = settings.MAX_LOSS_PER_TRADE
    if daily_loss_limit < 0.01:
        daily_loss_limit = settings.DAILY_LOSS_LIMIT_USD

    return {
        "max_loss_per_trade": round(max_loss_per_trade, 2),
        "daily_loss_limit": round(daily_loss_limit, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOT SIZE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_lot_size(
    balance: float,
    max_loss_usd: float,
    sl_pips: int,
    spread_pips: float,
    symbol: str = "EURUSD",
) -> float | None:
    """
    Calculate the lot size based on risk parameters.
    Uses per-symbol pip value from SYMBOL_CONFIG.

    Args:
        balance:      current account balance
        max_loss_usd: max amount willing to lose per trade (5% of balance)
        sl_pips:      stop loss distance in pips
        spread_pips:  broker spread in pips
        symbol:       trading symbol for per-symbol pip config

    Returns:
        lot size (float) or None if SL is too wide for budget
    """
    sym_config = settings.get_symbol_config(symbol)
    pip_value_micro = sym_config["pip_value_micro"]
    min_lot = sym_config["min_lot"]
    max_lot = sym_config["max_lot"]

    # Subtract spread cost from risk budget
    spread_cost = spread_pips * pip_value_micro
    net_risk = max_loss_usd - spread_cost

    if net_risk <= 0:
        return min_lot  # minimum lot

    # How many pips can we afford at 0.01 lot?
    max_sl_pips_at_micro = net_risk / pip_value_micro

    if sl_pips > max_sl_pips_at_micro:
        return None  # SL too wide for budget — SKIP trade

    # Scale up if budget allows
    lot_size = round(net_risk / (sl_pips * pip_value_micro * 10), 2)
    lot_size = max(min_lot, min(lot_size, max_lot))

    return lot_size


def pips_to_price(
    entry_price: float,
    pips: int,
    direction: str,
    side: str,
    symbol: str = "EURUSD",
) -> float:
    """
    Convert pips to a price level using per-symbol pip size.

    Args:
        entry_price: current entry price
        pips:        number of pips
        direction:   "BUY" or "SELL"
        side:        "sl" or "tp"
        symbol:      trading symbol for pip_size lookup

    Returns:
        price level rounded appropriately
    """
    sym_config = settings.get_symbol_config(symbol)
    pip_size = sym_config["pip_size"]

    # Determine decimal places from pip size
    if pip_size >= 0.01:
        decimals = 2
    elif pip_size >= 0.001:
        decimals = 3
    else:
        decimals = 5

    if direction == "BUY":
        if side == "sl":
            return round(entry_price - (pips * pip_size), decimals)
        else:  # tp
            return round(entry_price + (pips * pip_size), decimals)
    else:  # SELL
        if side == "sl":
            return round(entry_price + (pips * pip_size), decimals)
        else:  # tp
            return round(entry_price - (pips * pip_size), decimals)


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC RISK CHECKS (before AI call)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_hard_risk_checks(
    decision: TraderDecision,
    mt5: MT5Client,
    trades_today_count: int,
    daily_pnl: float,
    balance: float,
) -> RiskApproval | None:
    """
    Run deterministic risk checks before calling the AI.
    Returns a REJECTED RiskApproval if any check fails, or None if all pass.
    """
    risk_limits = compute_risk_limits(balance)
    max_loss = risk_limits["max_loss_per_trade"]
    daily_limit = risk_limits["daily_loss_limit"]

    # 1. Daily trade limit
    if trades_today_count >= settings.MAX_TRADES_PER_DAY:
        return RiskApproval(
            approved=False,
            reason=f"Daily trade limit reached: {trades_today_count}/{settings.MAX_TRADES_PER_DAY}",
        )

    # 2. Daily loss limit (dynamic — 15% of balance)
    if daily_pnl <= -daily_limit:
        return RiskApproval(
            approved=False,
            reason=f"Daily loss limit reached: ${daily_pnl:.2f} (limit: -${daily_limit:.2f})",
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

    # 5. Lot size calculation (using per-symbol config)
    lots = calculate_lot_size(
        balance=balance,
        max_loss_usd=max_loss,
        sl_pips=decision.sl_pips,
        spread_pips=settings.SPREAD_PIPS,
        symbol=decision.pair or "EURUSD",
    )

    if lots is None:
        return RiskApproval(
            approved=False,
            reason=f"SL too wide ({decision.sl_pips} pips) for risk budget (${max_loss:.2f} = 5% of ${balance:.2f})",
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

def _build_risk_prompt(max_loss: float) -> str:
    """Build the system prompt for the Risk Engine AI."""
    return f"""You are a strict forex risk manager. Your job is to validate or reject a trade that has been proposed by a trader AI.

You receive:
- The proposed trade details (pair, direction, scores, SL/TP in pips)
- The current live price (bid/ask)
- Account information (balance, equity, open positions)
- Today's trade count and P&L

Your ONLY job is to:
1. Validate the lot size fits within the max loss budget (${max_loss:.2f}/trade = 5% of balance)
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
    balance: float = 0.0,
) -> RiskApproval:
    """
    Evaluate a trade decision through the risk engine.

    1. Run deterministic hard checks first (fast, free)
    2. If hard checks pass, call AI for final validation
    3. Return APPROVED or REJECTED with details

    Args:
        balance: current account balance (for dynamic 5% risk calc)
    """
    logger.info("⚖️  Running Risk Engine...")

    # Compute dynamic risk limits
    risk_limits = compute_risk_limits(balance)
    max_loss = risk_limits["max_loss_per_trade"]
    daily_limit = risk_limits["daily_loss_limit"]

    logger.info(
        f"   Risk limits: ${max_loss:.2f}/trade (5% of ${balance:.2f}) | "
        f"Daily cap: -${daily_limit:.2f}"
    )

    # ── Hard checks (deterministic) ──
    hard_reject = _run_hard_risk_checks(
        decision, mt5, trades_today_count, daily_pnl, balance
    )
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

    # ── Per-symbol config ──
    symbol = decision.pair or "EURUSD"
    sym_config = settings.get_symbol_config(symbol)

    # ── Calculate entry, SL, TP prices ──
    entry_price = price_data.ask if decision.direction == "BUY" else price_data.bid
    sl_price = pips_to_price(
        entry_price, decision.sl_pips, decision.direction, "sl", symbol
    )
    tp_pips = decision.tp_pips or int(decision.sl_pips * settings.MIN_RR_RATIO)
    tp_price = pips_to_price(
        entry_price, tp_pips, decision.direction, "tp", symbol
    )

    # ── Calculate lot size (per-symbol) ──
    lots = calculate_lot_size(
        balance=balance,
        max_loss_usd=max_loss,
        sl_pips=decision.sl_pips,
        spread_pips=price_data.spread_pips,
        symbol=symbol,
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

        pip_value_micro = sym_config["pip_value_micro"]

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
  Max Loss: ${max_loss:.2f} (5% of balance)
  Risk per pip at {lots} lots: ${lots * pip_value_micro / sym_config['min_lot']:.4f}

Daily Stats:
  Trades today: {trades_today_count}/{settings.MAX_TRADES_PER_DAY}
  Daily P&L: ${daily_pnl:.2f} (limit: -${daily_limit:.2f})

Trader Brain Reasoning: {decision.reasoning[:500]}"""

        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            temperature=0.1,  # very low for risk decisions
            max_tokens=500,
            messages=[
                {"role": "system", "content": _build_risk_prompt(max_loss)},
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

        # Safety override: lot size bounds (per-symbol)
        approval.lots = max(
            sym_config["min_lot"],
            min(approval.lots, sym_config["max_lot"]),
        )

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
