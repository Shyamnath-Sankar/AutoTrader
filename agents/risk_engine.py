"""
risk_engine.py — Risk Engine (fully deterministic, no LLM).

Architecture (per entry-architecture.svg):
  Receives: EntryCandidate from Entry Engine + account info + live price
  Validates: SL fits max_loss budget, R:R meets tier, daily limits
  Computes: lot size from budget, order type (market vs limit)
  Output: APPROVED (lots, SL, TP, order_type) or REJECTED (reason)

The Risk Engine is PURELY DETERMINISTIC — no AI calls.
All intelligence is in the Trader Brain (direction) and Entry Engine (levels).
Risk Engine only validates math and budget constraints.
"""

from loguru import logger

from config import settings
from core.models import (
    EntryCandidate,
    MarketDataPayload,
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
    """
    max_loss_per_trade_usd = round(balance * (settings.MAX_LOSS_PER_TRADE_PCT / 100.0), 2)
    daily_loss_limit_usd = round(balance * (settings.DAILY_LOSS_LIMIT_PCT / 100.0), 2)

    return {
        "max_loss_per_trade_usd": max_loss_per_trade_usd,
        "daily_loss_limit_usd": daily_loss_limit_usd,
    }


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

    Returns:
        lot size (float) or None if SL is too wide for budget
    """
    pip_value_micro = settings.PAIR_PIP_VALUES_MICRO.get(pair, settings.PIP_VALUE_PER_MICRO_LOT)
    spread_pips = live_spread_pips if live_spread_pips is not None else settings.SPREAD_PIPS

    total_pip_risk = sl_pips + spread_pips

    # At minimum lot (0.01), what would we lose?
    min_lot_risk = total_pip_risk * pip_value_micro

    if min_lot_risk > max_loss_usd:
        return None  # Can't even afford minimum lot — REJECT

    # How many micro lots can we afford?
    pip_value_per_full_lot = pip_value_micro / settings.MIN_LOT
    lot_size = max_loss_usd / (total_pip_risk * pip_value_per_full_lot)
    lot_size = round(lot_size, 2)
    lot_size = max(settings.MIN_LOT, min(lot_size, settings.MAX_LOT))

    return lot_size


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC RISK CHECKS
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
    Run deterministic risk checks before computation.
    Returns a REJECTED RiskApproval if any check fails, or None if all pass.
    """
    risk_limits = compute_risk_limits(balance)
    daily_limit_usd = risk_limits["daily_loss_limit_usd"]

    # 1. Daily trade limit
    if trades_today_count >= settings.MAX_TRADES_PER_DAY:
        return RiskApproval(
            approved=False,
            reason=f"Daily trade limit reached: {trades_today_count}/{settings.MAX_TRADES_PER_DAY}",
        )

    # 2. Daily TAKE attempt limit
    if take_attempts_today >= settings.MAX_TAKE_ATTEMPTS_PER_DAY:
        return RiskApproval(
            approved=False,
            reason=f"Daily TAKE attempt limit reached: {take_attempts_today}/{settings.MAX_TAKE_ATTEMPTS_PER_DAY}",
        )

    # 3. Daily loss limit
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

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATE — Main Risk Engine entry point
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(
    decision: TraderDecision,
    entry: EntryCandidate,
    mt5: MT5Client,
    market_data: MarketDataPayload,
    trades_today_count: int = 0,
    daily_pnl: float = 0.0,
    take_attempts_today: int = 0,
) -> RiskApproval:
    """
    Evaluate a trade using the Entry Engine's candidate — PURELY DETERMINISTIC.

    Pipeline:
    1. Run hard checks (daily limits, existing positions)
    2. Get live price and validate spread
    3. Validate SL fits budget → calculate lot size
    4. Check R:R meets tier requirement
    5. Determine order type (market vs limit)
    6. Return APPROVED or REJECTED
    """
    logger.info("⚖️  Running Risk Engine (deterministic)...")

    pair = decision.pair
    direction = decision.direction

    # ── Fetch account balance ──
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

    # ── Hard checks ──
    hard_reject = _run_hard_risk_checks(
        decision, balance, trades_today_count, daily_pnl, take_attempts_today, mt5,
    )
    if hard_reject:
        logger.info(f"🚫 Risk REJECTED (hard check): {hard_reject.reason}")
        return hard_reject

    # ── Get live price ──
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

    # ── Use the Entry Engine's values ──
    entry_price = entry.entry_price
    sl_pips = entry.sl_pips
    sl_price = entry.sl_price
    tp_price = entry.tp_price
    tp_pips = entry.tp_pips

    logger.info(
        f"📐 Entry Engine provided: {entry.entry_type} entry @ {entry_price} | "
        f"SL={sl_pips}pips | TP={tp_pips}pips | R:R 1:{entry.rr_ratio}"
    )

    # ── Validate SL pips ──
    if sl_pips < settings.MIN_SL_PIPS:
        return RiskApproval(
            approved=False,
            reason=f"SL too tight: {sl_pips} pips (minimum {settings.MIN_SL_PIPS})",
        )

    if sl_pips > settings.MAX_SL_PIPS:
        return RiskApproval(
            approved=False,
            reason=f"SL too wide: {sl_pips} pips (maximum {settings.MAX_SL_PIPS})",
        )

    # ── Calculate lot size ──
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

    # ── Check R:R meets tier requirement ──
    required_rr = settings.get_required_rr(decision.total_score)
    score_pct = settings.get_score_pct(decision.total_score)

    actual_rr = entry.rr_ratio
    if actual_rr < required_rr:
        logger.warning(
            f"R:R {actual_rr:.1f} below required {required_rr} for "
            f"score {decision.total_score} ({score_pct*100:.0f}%) — rejecting"
        )
        return RiskApproval(
            approved=False,
            reason=(
                f"R:R {actual_rr:.1f} below required 1:{required_rr:.0f} "
                f"(score {decision.total_score}/{settings.get_total_max_score()} = {score_pct*100:.0f}%)"
            ),
        )

    # ── Determine order type: market vs limit ──
    pip_size = settings.PAIR_PIP_SIZES.get(pair, settings.PIP_SIZE)
    current_price = price_data.ask if direction == "BUY" else price_data.bid
    distance_pips = abs(entry_price - current_price) / pip_size

    if distance_pips <= settings.MARKET_ORDER_THRESHOLD_PIPS:
        order_type = "market"
        actual_entry = current_price
    else:
        order_type = "limit"
        actual_entry = entry_price

    # ── APPROVED ──
    pip_value_micro = settings.PAIR_PIP_VALUES_MICRO.get(pair, settings.PIP_VALUE_PER_MICRO_LOT)
    risk_usd = lots * (pip_value_micro / settings.MIN_LOT) * sl_pips

    logger.info(
        f"✅ Risk APPROVED: {lots} lots | ${risk_usd:.2f} risk | "
        f"SL={sl_price} TP={tp_price} | "
        f"R:R 1:{actual_rr:.1f} | {order_type.upper()} order"
    )

    return RiskApproval(
        approved=True,
        lots=lots,
        sl_price=sl_price,
        tp_price=tp_price,
        sl_pips=sl_pips,
        tp_pips=tp_pips,
        entry_price=actual_entry,
        order_type=order_type,
        reason=f"Approved: {lots} lots, {entry.entry_type} entry, R:R 1:{actual_rr:.1f}, {order_type}",
        rr_ratio=actual_rr,
        entry_type=entry.entry_type,
        confluence_count=entry.confluence_count,
    )
