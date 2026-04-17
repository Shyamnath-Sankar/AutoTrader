"""
main.py — Entry point & Orchestrator for the Smart Money Trading Bot.

Architecture (per diagram):
  1. Hard gates (Python, not AI) — session, ADX, news
  2. Data collection layer — tradingview-ta, SMC library, news, MT5 account
  3. AI Trader Brain (Agent 1) — scores everything, outputs TAKE / LEAVE / SCHEDULE
  4. Risk Engine (Agent 2) — computes SL from swing structure, lot size, validates
  5. MT5 execution — place trade if approved
  6. Result logger — trades.json
  7. Scheduling intelligence — AI sets exact wakeup time, or cooldown after rejection

Anti-overtrading protections:
  - Concurrency lock (no parallel analysis cycles)
  - Rejection cooldown with exponential escalation
  - TAKE attempt limit per day
  - Previous decision context (anti score inflation)
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

logger.remove()  # remove default stderr handler
logger.add(sys.stderr, level=settings.LOG_LEVEL, colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add(settings.LOG_FILE, level="DEBUG", rotation="10 MB", retention="30 days",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")

# ── Now import bot modules ───────────────────────────────────────────────────

from core.gates import run_all_gates
from services.data_collector import collect_all_data
from agents.trader_brain import analyze
from agents.risk_engine import evaluate
from services.mt5_client import MT5Client
from services.trade_logger import TradeLogger
from core.scheduler import BotScheduler


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════════

mt5 = MT5Client()
trade_log = TradeLogger()
bot_scheduler = BotScheduler()
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
    """
    Compute cooldown minutes based on consecutive rejections.
    Uses exponential escalation: base * escalation^(rejections-1)
    """
    if consecutive_rejections <= 0:
        return settings.DEFAULT_SCAN_INTERVAL_MINUTES

    base = settings.REJECTION_COOLDOWN_MINUTES
    escalation = settings.REJECTION_COOLDOWN_ESCALATION
    cooldown = base * (escalation ** (consecutive_rejections - 1))
    cooldown = min(int(cooldown), settings.REJECTION_COOLDOWN_MAX_MINUTES)
    return max(cooldown, settings.DEFAULT_SCAN_INTERVAL_MINUTES)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis_cycle():
    """
    Run one complete analysis cycle:
      Gates → Data → Brain → Risk → Execute → Log

    Protected by a threading lock to prevent overlapping cycles.
    """
    global shutdown_requested
    if shutdown_requested:
        return

    # Concurrency guard — skip if another cycle is already running
    if not _analysis_lock.acquire(blocking=False):
        logger.warning("⚠️ Analysis cycle already running — skipping this trigger")
        return

    try:
        _run_analysis_cycle_locked()
    finally:
        _analysis_lock.release()


def _run_analysis_cycle_locked():
    """The actual analysis pipeline (runs within the lock)."""
    global shutdown_requested

    cycle_start = datetime.now(pytz.UTC)
    logger.info(f"\n{'═' * 60}")
    logger.info(f"🔄 ANALYSIS CYCLE START — {cycle_start.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info(f"   Mode: {settings.MODE} | Pairs: {', '.join(settings.PAIRS)}")
    logger.info(f"{'═' * 60}")

    # ── Step 1: Hard Gates ────────────────────────────────────────────────
    logger.info("\n📋 Step 1: Running hard gates...")
    gates_passed, gate_results = run_all_gates(settings.PAIRS)

    if not gates_passed:
        # Find the failed gate and schedule retry
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

    # ── Step 2: Collect Market Data ───────────────────────────────────────
    logger.info("\n📊 Step 2: Collecting market data...")
    market_data = collect_all_data(mt5, trade_log)

    # ── Get today's stats for context ─────────────────────────────────────
    today_stats = trade_log.get_today_stats()
    logger.info(
        f"📈 Today's stats: {today_stats['executed_count']} trades executed, "
        f"{today_stats['take_attempts']} TAKE attempts, "
        f"{today_stats['rejections_today']} rejections, "
        f"P&L: ${today_stats['total_pnl']:.2f}"
    )

    # ── Check TAKE attempt limit before calling AI ────────────────────────
    if today_stats["take_attempts"] >= settings.MAX_TAKE_ATTEMPTS_PER_DAY:
        logger.info(
            f"🚫 Daily TAKE attempt limit reached ({today_stats['take_attempts']}"
            f"/{settings.MAX_TAKE_ATTEMPTS_PER_DAY}) — scheduling default scan only"
        )
        bot_scheduler.schedule_default_scan()
        return

    # ── Get previous decision for anti-inflation context ──────────────────
    previous_decision = trade_log.get_last_decision()

    # ── Step 3: AI Trader Brain ───────────────────────────────────────────
    logger.info("\n🧠 Step 3: AI Trader Brain analyzing...")
    decision = analyze(market_data, previous_decision=previous_decision)

    # ── Step 4: Route Decision ────────────────────────────────────────────

    if decision.decision == "LEAVE":
        # ── LEAVE: Log and reschedule at default interval ──
        logger.info("🔴 Decision: LEAVE — no viable setup")
        trade_log.log_decision(decision)
        bot_scheduler.schedule_default_scan()
        return

    elif decision.decision == "SCHEDULE":
        # ── SCHEDULE: AI sets exact wakeup time ──
        schedule_min = decision.schedule_minutes or settings.DEFAULT_SCAN_INTERVAL_MINUTES
        logger.info(f"🟡 Decision: SCHEDULE — waking up in {schedule_min} min")
        if decision.schedule_reason:
            logger.info(f"   Reason: {decision.schedule_reason}")
        trade_log.log_decision(decision)
        bot_scheduler.schedule_analysis(
            minutes_from_now=schedule_min,
            reason=decision.schedule_reason or "AI scheduled wakeup",
        )
        return

    elif decision.decision == "TAKE":
        # ── TAKE: Run Risk Engine → MT5 ──
        logger.info("🟢 Decision: TAKE — running risk engine...")

        risk_result = evaluate(
            decision=decision,
            mt5=mt5,
            market_data=market_data,
            trades_today_count=today_stats["executed_count"],
            daily_pnl=today_stats["total_pnl"],
            take_attempts_today=today_stats["take_attempts"],
        )

        if not risk_result.approved:
            # ── REJECTED by risk engine ──
            logger.info(f"🚫 Risk REJECTED: {risk_result.reason}")
            trade_log.log_decision(decision, risk=risk_result)

            # Compute cooldown based on consecutive rejections
            consecutive = trade_log.get_consecutive_rejections(pair=decision.pair)
            cooldown = _compute_cooldown_minutes(consecutive)
            logger.info(
                f"⏳ Rejection cooldown: {cooldown} min "
                f"(consecutive rejections: {consecutive})"
            )
            bot_scheduler.schedule_cooldown(
                cooldown_minutes=cooldown,
                reason=f"TAKE rejected ({consecutive} consecutive): {risk_result.reason}",
            )
            return

        # ── APPROVED — Execute trade on MT5 ──
        logger.info("✅ Risk APPROVED — executing on MT5...")

        if settings.MODE == "demo":
            logger.info("🏷️  MODE: DEMO — placing trade on demo account")

        order_result = mt5.place_market_order(
            symbol=decision.pair,
            direction=decision.direction,
            volume=risk_result.lots,
            stop_loss=risk_result.sl_price,
            take_profit=risk_result.tp_price,
        )

        if order_result.success:
            logger.info(
                f"🎯 Trade EXECUTED! Ticket #{order_result.order} | "
                f"{decision.direction} {risk_result.lots} lots {decision.pair} | "
                f"Entry: {order_result.price} | SL: {risk_result.sl_price} ({risk_result.sl_pips}pips) | "
                f"TP: {risk_result.tp_price} ({risk_result.tp_pips}pips) | "
                f"R:R 1:{risk_result.rr_ratio:.1f}"
            )
        else:
            logger.error(f"❌ Trade FAILED: {order_result.message} (code: {order_result.error_code})")

        # Log everything
        trade_log.log_decision(decision, risk=risk_result, order=order_result)

        # Schedule next scan
        bot_scheduler.schedule_default_scan()
        return

    else:
        logger.warning(f"Unknown decision: {decision.decision}")
        bot_scheduler.schedule_default_scan()


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    """Print the bot startup banner."""
    total_max = settings.get_total_max_score()
    total_min_required = settings.get_total_min_required()
    total_min_pct = int(settings.TOTAL_MIN_SCORE_PCT * 100)

    # Build concise R:R tier string for the banner
    rr_parts = []
    for t in settings.RR_TIERS:
        lo = int(t['min_pct'] * 100)
        hi = int(t['max_pct'] * 100) if t['max_pct'] <= 1.0 else 100
        rr_parts.append(f"{lo}-{hi}%→1:{t['rr_ratio']:.0f}")
    rr_display = " · ".join(rr_parts)

    scoring_line = f"P1 max {settings.PHASE1_MAX_SCORE} (min {settings.PHASE1_MIN_REQUIRED}) + P2 max {settings.PHASE2_MAX_SCORE} (min {settings.PHASE2_MIN_REQUIRED})"
    total_line = f"{total_min_required}/{total_max} ({total_min_pct}%) to TAKE"
    risk_line = f"{settings.MAX_LOSS_PER_TRADE_PCT}%/trade · {settings.DAILY_LOSS_LIMIT_PCT}%/day · {settings.MAX_TRADES_PER_DAY} trades/day"

    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║              SMART MONEY TRADING BOT v2.0                   ║
║              ────────────────────────                       ║
║  AI-Powered · ICT/SMC · Two-Phase Scoring                   ║
╠══════════════════════════════════════════════════════════════╣
║  Mode:       {settings.MODE.upper():<46}║
║  Pairs:      {', '.join(settings.PAIRS):<46}║
║  LLM:        {f'{settings.LLM_MODEL} @ {settings.LLM_BASE_URL[:30]}':<46}║
║  MT5:        {settings.MT5_BASE_URL:<46}║
╠══════════════════════════════════════════════════════════════╣
║  Scoring:    {scoring_line:<46}║
║  Total:      {total_line:<46}║
║  R:R Tiers:  {rr_display:<46}║
║  Risk:       {risk_line:<46}║
║  Overtrading: max {settings.MAX_TAKE_ATTEMPTS_PER_DAY} TAKE attempts · {settings.REJECTION_COOLDOWN_MINUTES}min cooldown      ║
║  SL:         From swing structure + {settings.SL_BUFFER_PIPS}pip buffer             ║
║  Decisions:  TAKE / SCHEDULE / LEAVE                        ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_prerequisites() -> bool:
    """Check that all required services are available."""
    ok = True

    # Check API key
    if settings.LLM_API_KEY == "your-api-key-here":
        logger.error("❌ LLM_API_KEY not set! Set it in .env or settings.py")
        ok = False
    else:
        logger.info(f"✅ LLM API key configured (model: {settings.LLM_MODEL})")

    # Check MT5 connection
    if mt5.is_connected():
        account = mt5.get_account_info()
        if account:
            balance = account.balance
            max_sl_eurusd = settings.compute_max_sl_pips(balance, "EURUSD")
            max_loss_usd = round(balance * (settings.MAX_LOSS_PER_TRADE_PCT / 100.0), 2)

            logger.info(f"✅ MT5 connected — Balance: ${balance:.2f} | "
                         f"Leverage: 1:{account.leverage}")
            logger.info(f"   Risk budget: ${max_loss_usd:.2f}/trade "
                         f"({settings.MAX_LOSS_PER_TRADE_PCT}%) | "
                         f"Max SL: {max_sl_eurusd} pips at 0.01 lot")

            # Warn if budget is very tight
            if max_sl_eurusd < 10:
                logger.warning(
                    f"⚠️  Very tight SL budget ({max_sl_eurusd} pips). "
                    f"Consider increasing account balance or risk % for viable trading."
                )
    else:
        logger.warning("⚠️  MT5 not connected — bot will still run but cannot execute trades")
        logger.warning(f"   Start the MT5 HTTP server: metatrader-http-server --port 8001")

    # Check scoring config consistency
    p1_sum = (settings.P1_REGIME_POINTS + settings.P1_SESSION_POINTS +
              settings.P1_NEWS_POINTS + settings.P1_WEEKLY_4H_BIAS_POINTS +
              settings.P1_1H_TREND_POINTS + settings.P1_15MIN_TRIGGER_POINTS)
    p2_sum = (settings.P2_LIQUIDITY_SWEEP_POINTS + settings.P2_ORDER_BLOCK_POINTS +
              settings.P2_FVG_POINTS + settings.P2_BOS_CHOCH_POINTS)

    if p1_sum != settings.PHASE1_MAX_SCORE:
        logger.warning(f"⚠️  Phase 1 sub-scores sum to {p1_sum} but PHASE1_MAX_SCORE = {settings.PHASE1_MAX_SCORE}")
    else:
        logger.info(f"✅ Phase 1 scoring: {p1_sum} pts across 6 components")

    if p2_sum != settings.PHASE2_MAX_SCORE:
        logger.warning(f"⚠️  Phase 2 sub-scores sum to {p2_sum} but PHASE2_MAX_SCORE = {settings.PHASE2_MAX_SCORE}")
    else:
        logger.info(f"✅ Phase 2 scoring: {p2_sum} pts across 4 components")

    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point — starts the bot daemon."""
    print_banner()

    logger.info("🚀 Starting Smart Money Trading Bot v2.0...")

    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites not met — fix the issues above and restart")
        sys.exit(1)

    # Start scheduler
    bot_scheduler.start(analysis_callback=run_analysis_cycle)

    # Run first analysis immediately
    logger.info("\n🔄 Running initial analysis cycle...")
    run_analysis_cycle()

    # Keep the main thread alive
    logger.info("\n⏳ Bot is running. Press Ctrl+C to stop.")
    try:
        while not shutdown_requested:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        logger.info("🛑 Shutting down bot...")
        bot_scheduler.stop()
        logger.info("👋 Smart Money Trading Bot stopped. Goodbye!")


if __name__ == "__main__":
    main()
