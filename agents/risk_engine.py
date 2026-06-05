"""
risk_engine.py — Risk Engine (fully deterministic, no LLM).

Changes from v3.0:
  - Removed: MAX_TRADES_PER_DAY check
  - Removed: MAX_TAKE_ATTEMPTS_PER_DAY check
  - Removed: spread check (spread cost already baked into lot sizing)
  - Removed: same-pair open position block (multiple trades on same pair allowed)
  - Fixed: uses effective thresholds (news_excluded) for R:R tier calculation
  - Fixed: symbol suffix applied via mt5_client._sym() (not here)
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
    """Compute USD risk limits from live account balance."""
    return {
        "max_loss_per_trade_usd": round(balance * (settings.MAX_LOSS_PER_TRADE_PCT / 100.0), 2),
        "daily_loss_limit_usd":   round(balance * (settings.DAILY_LOSS_LIMIT_PCT   / 100.0), 2),
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
    Calculate the lot size based on risk budget and SL in pips.
    Returns None if even the minimum lot exceeds the budget.
    """
    pip_value_micro = settings.PAIR_PIP_VALUES_MICRO.get(pair, settings.PIP_VALUE_PER_MICRO_LOT)
    spread_pips     = live_spread_pips if live_spread_pips is not None else settings.SPREAD_PIPS

    total_pip_risk = sl_pips + spread_pips
    min_lot_risk   = total_pip_risk * pip_value_micro

    if min_lot_risk > max_loss_usd:
        return None   # can't even afford minimum lot

    pip_value_per_full_lot = pip_value_micro / settings.MIN_LOT
    lot_size = max_loss_usd / (total_pip_risk * pip_value_per_full_lot)
    lot_size = round(lot_size, 2)
    lot_size = max(settings.MIN_LOT, min(lot_size, settings.MAX_LOT))
    return lot_size


# ═══════════════════════════════════════════════════════════════════════════════
# HARD RISK CHECKS (only daily loss limit remains)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_hard_risk_checks(
    balance: float,
    daily_pnl: float,
) -> RiskApproval | None:
    """
    Run the only remaining hard check: daily loss limit.
    Returns REJECTED RiskApproval if breach detected, None if all pass.
    """
    limits        = compute_risk_limits(balance)
    daily_limit   = limits["daily_loss_limit_usd"]

    if daily_pnl <= -daily_limit:
        return RiskApproval(
            approved=False,
            reason=(
                f"Daily loss limit reached: ${daily_pnl:.2f} "
                f"(limit: -${daily_limit:.2f} = {settings.DAILY_LOSS_LIMIT_PCT}% of ${balance:.2f})"
            ),
        )

    return None   # all checks passed


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATE — Main entry point
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
    1. Get live balance → compute risk limits
    2. Daily loss limit check (the only remaining hard gate)
    3. Validate SL pips → compute lot size
    4. Check R:R meets tier requirement
    5. Determine order type (market vs limit)
    6. Return APPROVED or REJECTED
    """
    logger.info("⚖️  Running Risk Engine (deterministic)...")

    pair      = decision.pair
    direction = decision.direction

    # ── Live balance ──────────────────────────────────────────────────────────
    account = mt5.get_account_info()
    balance = account.balance if account else 0

    if balance <= 0:
        return RiskApproval(
            approved=False,
            reason="Cannot determine account balance — MT5 might be down",
        )

    limits        = compute_risk_limits(balance)
    max_loss_usd  = limits["max_loss_per_trade_usd"]
    daily_lim_usd = limits["daily_loss_limit_usd"]

    logger.info(
        f"Risk limits from ${balance:.2f} balance: "
        f"per-trade=${max_loss_usd:.2f} ({settings.MAX_LOSS_PER_TRADE_PCT}%), "
        f"daily=${daily_lim_usd:.2f} ({settings.DAILY_LOSS_LIMIT_PCT}%)"
    )

    # ── Hard checks (daily loss only) ─────────────────────────────────────────
    hard_reject = _run_hard_risk_checks(balance, daily_pnl)
    if hard_reject:
        logger.info(f"🚫 Risk REJECTED (hard check): {hard_reject.reason}")
        return hard_reject

    # ── Get live price (for order type decision only) ─────────────────────────
    price_data = mt5.get_price(pair) if pair else None
    if not price_data:
        return RiskApproval(
            approved=False,
            reason=f"Cannot get live price for {pair} — MT5 HTTP server might be down",
        )

    # ── Entry Engine values ───────────────────────────────────────────────────
    entry_price = entry.entry_price
    sl_pips     = entry.sl_pips
    sl_price    = entry.sl_price
    tp_price    = entry.tp_price
    tp_pips     = entry.tp_pips

    logger.info(
        f"📐 Entry Engine: {entry.entry_type} entry @ {entry_price} | "
        f"SL={sl_pips}pips | TP={tp_pips}pips | R:R 1:{entry.rr_ratio}"
    )

    # ── Validate SL pips ──────────────────────────────────────────────────────
    if sl_pips < settings.MIN_SL_PIPS:
        return RiskApproval(
            approved=False,
            reason=f"SL too tight: {sl_pips} pips (min {settings.MIN_SL_PIPS})",
        )

    if sl_pips > settings.MAX_SL_PIPS:
        return RiskApproval(
            approved=False,
            reason=f"SL too wide: {sl_pips} pips (max {settings.MAX_SL_PIPS})",
        )

    # ── Calculate lot size (live spread used for cost accuracy) ──────────────
    lots = calculate_lot_size(
        max_loss_usd     = max_loss_usd,
        sl_pips          = sl_pips,
        pair             = pair,
        live_spread_pips = price_data.spread_pips,   # informational, not a gate
    )

    if lots is None:
        max_affordable = settings.compute_max_sl_pips(balance, pair)
        return RiskApproval(
            approved=False,
            reason=(
                f"SL {sl_pips} pips too wide for budget "
                f"(${max_loss_usd:.2f} = {settings.MAX_LOSS_PER_TRADE_PCT}% of ${balance:.2f}, "
                f"max affordable: {max_affordable} pips)"
            ),
        )

    # ── Check R:R meets tier requirement (using effective score %) ────────────
    news_excluded = decision.news_excluded
    required_rr   = settings.get_required_rr(decision.total_score, news_excluded)
    score_pct     = settings.get_score_pct(decision.total_score, news_excluded)
    actual_rr     = entry.rr_ratio

    if actual_rr < required_rr:
        return RiskApproval(
            approved=False,
            reason=(
                f"R:R {actual_rr:.1f} below required 1:{required_rr:.0f} "
                f"(score {decision.total_score}/{settings.get_total_max_score(news_excluded)} = {score_pct*100:.0f}%)"
            ),
        )

    # ── Order type: market vs limit ───────────────────────────────────────────
    pip_size      = settings.PAIR_PIP_SIZES.get(pair, settings.PIP_SIZE)
    current_price = price_data.ask if direction == "BUY" else price_data.bid
    distance_pips = abs(entry_price - current_price) / pip_size

    if distance_pips <= settings.MARKET_ORDER_THRESHOLD_PIPS:
        order_type   = "market"
        actual_entry = current_price
    else:
        order_type   = "limit"
        actual_entry = entry_price

    # ── APPROVED ──────────────────────────────────────────────────────────────
    pip_value_micro = settings.PAIR_PIP_VALUES_MICRO.get(pair, settings.PIP_VALUE_PER_MICRO_LOT)
    risk_usd        = lots * (pip_value_micro / settings.MIN_LOT) * sl_pips

    logger.info(
        f"✅ Risk APPROVED: {lots} lots | ${risk_usd:.2f} risk | "
        f"SL={sl_price} TP={tp_price} | R:R 1:{actual_rr:.1f} | {order_type.upper()} order"
    )

    return RiskApproval(
        approved        = True,
        lots            = lots,
        sl_price        = sl_price,
        tp_price        = tp_price,
        sl_pips         = sl_pips,
        tp_pips         = tp_pips,
        entry_price     = actual_entry,
        order_type      = order_type,
        reason          = f"Approved: {lots} lots, {entry.entry_type} entry, R:R 1:{actual_rr:.1f}, {order_type}",
        rr_ratio        = actual_rr,
        entry_type      = entry.entry_type,
        confluence_count= entry.confluence_count,
    )
