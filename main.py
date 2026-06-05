"""
main.py — Entry point & Orchestrator for the Smart Money Trading Bot.

Architecture:
  1. Hard gates (Python) — session, ADX, news (fail-open)
  2. Data collection   — MT5 Python lib (OHLCV), tradingview-ta (indicators), SMC
  3. AI Trader Brain   — scores ALL pairs, picks highest conviction → TAKE / LEAVE
  4. AI Entry Engine   — picks optimal entry: sweep → OB → FVG → swing
  5. Risk Engine       — validates budget, lot size, R:R (daily loss limit only)
  6. MT5 execution     — market or limit order (symbol suffix .m applied)
  7. Trade Monitor     — background TP1/TP2 management
  8. Trade Logger      — trades.json

Anti-overtrading protections:
  - Concurrency lock (no parallel analysis cycles)
  - Rejection cooldown with exponential escalation
  - Previous decision context (anti score inflation)
  - Daily loss limit (10% of balance) — the only hard stop
"""

import os
import sys

# Fix Windows console encoding for emoji/unicode output
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import threading
import time
import signal
from datetime import datetime

import pytz
from loguru import logger

# ── Setup logging before importing anything else ─────────────────────────────

from config import settings

os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

logger.remove()
logger.add(sys.stderr, level=settings.LOG_LEVEL, colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add(settings.LOG_FILE, level="DEBUG", rotation="10 MB", retention="30 days",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")

# ── Now import bot modules ───────────────────────────────────────────────────

from core.gates import run_all_gates
from services.data_collector import collect_all_data
from agents.trader_brain import analyze
from agents.entry_engine import find_entry
from agents.risk_engine import evaluate
from services.mt5_client import MT5Client
from services.trade_logger import TradeLogger
from services.trade_monitor import TradeMonitor
from core.scheduler import BotScheduler
import services.mt5_data as mt5_lib


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════════

mt5           = MT5Client()
trade_log     = TradeLogger()
bot_scheduler = BotScheduler()
trade_monitor = TradeMonitor(mt5)
shutdown_requested = False

# Concurrency lock — prevents overlapping analysis cycles
_analysis_lock = threading.Lock()


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    logger.info("\n🛑 Shutdown requested — finishing current cycle...")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ═══════════════════════════════════════════════════════════════════════════════
# COOLDOWN LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_cooldown_minutes(consecutive_rejections: int) -> int:
    """Exponential cooldown after consecutive rejections."""
    if consecutive_rejections <= 0:
        return settings.DEFAULT_SCAN_INTERVAL_MINUTES
    base       = settings.REJECTION_COOLDOWN_MINUTES
    escalation = settings.REJECTION_COOLDOWN_ESCALATION
    cooldown   = base * (escalation ** (consecutive_rejections - 1))
    return max(min(int(cooldown), settings.REJECTION_COOLDOWN_MAX_MINUTES),
               settings.DEFAULT_SCAN_INTERVAL_MINUTES)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis_cycle():
    """Protected entry point — acquires lock then runs the pipeline."""
    global shutdown_requested
    if shutdown_requested:
        return

    if not _analysis_lock.acquire(blocking=False):
        logger.warning("⚠️ Analysis cycle already running — skipping this trigger")
        return

    try:
        _run_analysis_cycle_locked()
    finally:
        _analysis_lock.release()


def _run_analysis_cycle_locked():
    """The actual analysis pipeline (runs within the concurrency lock)."""
    cycle_start = datetime.now(pytz.UTC)
    logger.info(f"\n{'═' * 60}")
    logger.info(f"🔄 ANALYSIS CYCLE START — {cycle_start.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info(f"   Mode: {settings.MODE} | Pairs: {', '.join(settings.PAIRS)}")
    logger.info(f"{'═' * 60}")

    # ── Step 1: Hard Gates ────────────────────────────────────────────────────
    logger.info("\n📋 Step 1: Running hard gates...")
    gates_passed, gate_results = run_all_gates(settings.PAIRS)

    if not gates_passed:
        failed_gate = next((g for g in gate_results if not g.passed), None)
        if failed_gate:
            logger.info(f"⏳ Scheduling retry in {failed_gate.skip_minutes} min")
            bot_scheduler.schedule_gate_retry(
                skip_minutes=failed_gate.skip_minutes,
                reason=failed_gate.reason,
            )
        else:
            bot_scheduler.schedule_default_scan()
        return

    # ── Step 2: Collect Market Data ──────────────────────────────────────────
    logger.info("\n📊 Step 2: Collecting market data...")
    market_data, raw_ohlcv = collect_all_data(mt5, trade_log)

    today_stats = trade_log.get_today_stats()
    logger.info(
        f"📈 Today: {today_stats['executed_count']} executed, "
        f"{today_stats['take_attempts']} TAKE attempts, "
        f"{today_stats['rejections_today']} rejected, "
        f"P&L: ${today_stats['total_pnl']:.2f}"
    )

    # ── Get previous decision for anti-inflation context ──────────────────────
    previous_decision = trade_log.get_last_decision()

    # ── Step 3: AI Trader Brain ───────────────────────────────────────────────
    logger.info("\n🧠 Step 3: AI Trader Brain analyzing all pairs...")
    decision = analyze(market_data, previous_decision=previous_decision)

    # ── Route decision ────────────────────────────────────────────────────────
    if decision.decision != "TAKE":
        logger.info("🔴 Decision: LEAVE — no viable setup")
        trade_log.log_decision(decision)
        bot_scheduler.schedule_default_scan()
        return

    logger.info("🟢 Decision: TAKE — running Entry Engine...")

    # ── Step 4: Get live price ────────────────────────────────────────────────
    price_data = mt5.get_price(decision.pair) if decision.pair else None
    if not price_data:
        logger.error(f"❌ Cannot get live price for {decision.pair}")
        trade_log.log_decision(decision)
        bot_scheduler.schedule_default_scan()
        return

    current_price = price_data.ask if decision.direction == "BUY" else price_data.bid

    # ── Step 5: AI Entry Engine ──────────────────────────────────────────────
    logger.info("\n🎯 Step 5: AI Entry Engine finding optimal entry...")
    entry = find_entry(
        decision      = decision,
        market_data   = market_data,
        raw_ohlcv     = raw_ohlcv,
        current_price = current_price,
    )

    if entry is None:
        logger.error("❌ Entry Engine failed to produce a candidate")
        trade_log.log_decision(decision)
        bot_scheduler.schedule_default_scan()
        return

    # ── Step 6: Risk Engine ──────────────────────────────────────────────────
    logger.info("\n⚖️ Step 6: Risk Engine validating...")
    risk_result = evaluate(
        decision           = decision,
        entry              = entry,
        mt5                = mt5,
        market_data        = market_data,
        trades_today_count = today_stats["executed_count"],
        daily_pnl          = today_stats["total_pnl"],
        take_attempts_today= today_stats["take_attempts"],
    )

    if not risk_result.approved:
        logger.info(f"🚫 Risk REJECTED: {risk_result.reason}")
        trade_log.log_decision(decision, risk=risk_result)

        consecutive = trade_log.get_consecutive_rejections(pair=decision.pair)
        cooldown    = _compute_cooldown_minutes(consecutive)
        logger.info(f"⏳ Rejection cooldown: {cooldown} min ({consecutive} consecutive)")
        bot_scheduler.schedule_cooldown(
            cooldown_minutes=cooldown,
            reason=f"TAKE rejected ({consecutive} consecutive): {risk_result.reason}",
        )
        return

    # ── Step 7: Execute on MT5 ───────────────────────────────────────────────
    logger.info("✅ Risk APPROVED — executing on MT5...")
    if settings.MODE == "demo":
        logger.info("🏷️  MODE: DEMO — trading on demo account")

    if risk_result.order_type == "limit":
        logger.info(f"📋 Placing LIMIT order @ {risk_result.entry_price}")
        order_result = mt5.place_limit_order(
            symbol      = decision.pair,
            direction   = decision.direction,
            volume      = risk_result.lots,
            price       = risk_result.entry_price,
            stop_loss   = risk_result.sl_price,
            take_profit = risk_result.tp_price,
        )
    else:
        logger.info("⚡ Placing MARKET order")
        order_result = mt5.place_market_order(
            symbol      = decision.pair,
            direction   = decision.direction,
            volume      = risk_result.lots,
            stop_loss   = risk_result.sl_price,
            take_profit = risk_result.tp_price,
        )

    if order_result.success:
        emoji = "📋" if risk_result.order_type == "limit" else "🎯"
        logger.info(
            f"{emoji} Trade EXECUTED! Ticket #{order_result.order} | "
            f"{decision.direction} {risk_result.lots} lots {decision.pair} | "
            f"Entry: {order_result.price or risk_result.entry_price} | "
            f"SL: {risk_result.sl_price} ({risk_result.sl_pips}pips) | "
            f"TP: {risk_result.tp_price} ({risk_result.tp_pips}pips) | "
            f"R:R 1:{risk_result.rr_ratio:.1f} | {risk_result.entry_type} ({risk_result.order_type})"
        )

        # ── Step 8: Register with Trade Monitor ──────────────────────────────
        if order_result.order and settings.MONITOR_ENABLED:
            tp1 = risk_result.sl_price + (risk_result.tp_price - risk_result.sl_price) * 0.5 \
                  if decision.direction == "BUY" \
                  else risk_result.sl_price - (risk_result.sl_price - risk_result.tp_price) * 0.5

            trade_monitor.track_position(
                ticket      = order_result.order,
                pair        = decision.pair,
                direction   = decision.direction,
                lots        = risk_result.lots,
                entry_price = order_result.price or risk_result.entry_price,
                sl_price    = risk_result.sl_price,
                tp1_price   = tp1,
                tp2_price   = risk_result.tp_price,
                order_type  = risk_result.order_type,
            )
        elif order_result.order and risk_result.order_type == "limit":
            trade_monitor.track_pending_order(order_result.order, decision.pair)
    else:
        logger.error(f"❌ Trade FAILED: {order_result.message} (code: {order_result.error_code})")

    # ── Step 9: Log everything ───────────────────────────────────────────────
    trade_log.log_decision(decision, risk=risk_result, order=order_result)
    bot_scheduler.schedule_default_scan()


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    """Print the bot startup banner."""
    total_max     = settings.get_total_max_score()
    total_min_req = settings.get_total_min_required()
    total_min_pct = int(settings.TOTAL_MIN_SCORE_PCT * 100)
    excl_max      = settings.get_total_max_score(True)
    excl_min      = settings.get_total_min_required(True)

    rr_parts = []
    for t in settings.RR_TIERS:
        lo = int(t['min_pct'] * 100)
        hi = int(t['max_pct'] * 100) if t['max_pct'] <= 1.0 else 100
        rr_parts.append(f"{lo}-{hi}%→1:{t['rr_ratio']:.0f}")
    rr_display = " · ".join(rr_parts)

    scoring_line   = (f"P1 max {settings.PHASE1_MAX_SCORE} | "
                      f"P2 max {settings.PHASE2_MAX_SCORE} | "
                      f"% thresholds")
    total_line     = (f"{total_min_req}/{total_max} ({total_min_pct}%) normal | "
                      f"{excl_min}/{excl_max} news-excluded")
    risk_line      = (f"{settings.MAX_LOSS_PER_TRADE_PCT}%/trade · "
                      f"{settings.DAILY_LOSS_LIMIT_PCT}%/day · no trade count limit")
    ohlcv_src      = "MT5 Python lib → yfinance fallback"

    if settings.LLM_PROVIDER == "azure":
        llm_display = f"{settings.AZURE_OPENAI_DEPLOYMENT} @ Azure AI Foundry"
    else:
        llm_display = f"{settings.LLM_MODEL} @ {settings.LLM_BASE_URL[:30]}"

    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║         SMART MONEY TRADING BOT v3.0 — ENTRY ENGINE        ║
║         ─────────────────────────────────────────           ║
║  AI-Powered · ICT/SMC · Precise Entry · Dual TP            ║
╠══════════════════════════════════════════════════════════════╣
║  Mode:       {settings.MODE.upper():<46}║
║  Pairs:      {', '.join(settings.PAIRS):<46}║
║  LLM:        {llm_display:<46}║
║  MT5 HTTP:   {settings.MT5_BASE_URL:<46}║
║  OHLCV src:  {ohlcv_src:<46}║
╠══════════════════════════════════════════════════════════════╣
║  Scoring:    {scoring_line:<46}║
║  Total:      {total_line:<46}║
║  R:R Tiers:  {rr_display:<46}║
║  Risk:       {risk_line:<46}║
║  News gate:  ENABLED (fail-open: unavailable → score excluded)║
║  SL/TP:      From swing structure + {settings.SL_BUFFER_PIPS}pip buffer              ║
║  Entry:      AI picks: sweep → OB → FVG → swing/round       ║
║  Orders:     Market (≤{settings.MARKET_ORDER_THRESHOLD_PIPS}pip) / Limit (>{settings.MARKET_ORDER_THRESHOLD_PIPS}pip)           ║
║  Monitor:    {'ENABLED — run monitor_server.py separately' if settings.MONITOR_ENABLED else 'DISABLED':<40}║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_prerequisites() -> bool:
    """Check that all required services are available before starting."""
    ok = True

    # ── LLM config ─────────────────────────────────────────────────────────
    if settings.LLM_PROVIDER == "azure":
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            logger.error("❌ Azure: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set!")
            ok = False
        elif not settings.AZURE_OPENAI_DEPLOYMENT:
            logger.error("❌ Azure: AZURE_OPENAI_DEPLOYMENT not set in .env")
            ok = False
        else:
            logger.info(f"✅ LLM: Azure AI Foundry — {settings.AZURE_OPENAI_DEPLOYMENT} v{settings.AZURE_OPENAI_API_VERSION}")
    else:
        if settings.LLM_API_KEY == "your-api-key-here":
            logger.error("❌ LLM_API_KEY not set in .env")
            ok = False
        else:
            logger.info(f"✅ LLM: {settings.LLM_MODEL} @ {settings.LLM_BASE_URL}")

    # ── MT5 Python lib (OHLCV) ──────────────────────────────────────────────
    if mt5_lib.initialize():
        logger.info("✅ MT5 Python lib connected — OHLCV from live terminal")
    else:
        logger.warning("⚠️  MT5 Python lib not available — OHLCV will use yfinance fallback")

    # ── MT5 HTTP (orders) ───────────────────────────────────────────────────
    if mt5.is_connected():
        account = mt5.get_account_info()
        if account:
            balance      = account.balance
            max_loss_usd = round(balance * (settings.MAX_LOSS_PER_TRADE_PCT / 100.0), 2)
            max_sl_eu    = settings.compute_max_sl_pips(balance, "EURUSD")
            max_sl_xau   = settings.compute_max_sl_pips(balance, "XAUUSD")
            logger.info(f"✅ MT5 HTTP connected — Balance: ${balance:.2f} | Leverage: 1:{account.leverage}")
            logger.info(f"   Risk: ${max_loss_usd:.2f}/trade ({settings.MAX_LOSS_PER_TRADE_PCT}%) | "
                        f"Max SL: EURUSD={max_sl_eu}p XAUUSD={max_sl_xau}p at 0.01 lot")
            if max_sl_eu < 10:
                logger.warning("⚠️  Very tight SL budget. Consider increasing balance or risk %.")
    else:
        logger.warning("⚠️  MT5 HTTP not connected — cannot execute trades")
        logger.warning(f"   Start the HTTP server: metatrader-http-server --port 8001")

    # ── Scoring consistency ──────────────────────────────────────────────────
    p1_sum = (settings.P1_REGIME_POINTS + settings.P1_SESSION_POINTS +
              settings.P1_NEWS_POINTS + settings.P1_WEEKLY_4H_BIAS_POINTS +
              settings.P1_1H_TREND_POINTS + settings.P1_15MIN_TRIGGER_POINTS)
    p2_sum = (settings.P2_LIQUIDITY_SWEEP_POINTS + settings.P2_ORDER_BLOCK_POINTS +
              settings.P2_FVG_POINTS + settings.P2_BOS_CHOCH_POINTS)

    if p1_sum != settings.PHASE1_MAX_SCORE:
        logger.warning(f"⚠️  Phase 1 sub-scores sum {p1_sum} ≠ PHASE1_MAX_SCORE {settings.PHASE1_MAX_SCORE}")
    else:
        logger.info(f"✅ Scoring: P1={p1_sum}pts | P2={p2_sum}pts | "
                    f"Total min {settings.get_total_min_required()}/{settings.get_total_max_score()} "
                    f"({int(settings.TOTAL_MIN_SCORE_PCT*100)}%)")

    logger.info(f"✅ XAUUSD pip: {settings.PAIR_PIP_SIZES.get('XAUUSD')} size | "
                f"${settings.PAIR_PIP_VALUES_MICRO.get('XAUUSD')}/pip at 0.01 lot")

    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point — starts the bot daemon."""
    print_banner()
    logger.info("🚀 Starting Smart Money Trading Bot v3.0...")

    if not check_prerequisites():
        logger.error("Prerequisites not met — fix the above and restart")
        sys.exit(1)

    # ── Start Trade Monitor background thread ─────────────────────────────────
    if settings.MONITOR_ENABLED:
        trade_monitor.start()
        logger.info("📡 Trade Monitor thread started (run monitor_server.py for standalone monitor)")

    # ── Start scheduler ────────────────────────────────────────────────────────
    bot_scheduler.start(analysis_callback=run_analysis_cycle)

    # ── Run first analysis immediately ─────────────────────────────────────────
    logger.info("\n🔄 Running initial analysis cycle...")
    run_analysis_cycle()

    logger.info("\n⏳ Bot running. Press Ctrl+C to stop.")
    try:
        while not shutdown_requested:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        logger.info("🛑 Shutting down...")
        bot_scheduler.stop()
        trade_monitor.stop()
        mt5_lib.shutdown()
        logger.info("👋 Smart Money Trading Bot stopped.")


if __name__ == "__main__":
    main()
