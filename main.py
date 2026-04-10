"""
main.py — Entry point & Orchestrator for the Smart Money Trading Bot.

This is the main daemon that:
  1. Runs the hard gates
  2. Collects market data
  3. Calls the AI Trader Brain
  4. Routes decisions (EXECUTE → Risk → MT5, WATCH → Schedule, SKIP → Log)
  5. Loops forever on weekdays
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


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    logger.info("\n🛑 Shutdown requested — finishing current cycle...")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis_cycle():
    """
    Run one complete analysis cycle:
      Gates → Data → Brain → Risk → Execute → Log

    This is called by the scheduler or by the main loop.
    """
    global shutdown_requested
    if shutdown_requested:
        return

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

    # ── Step 3: AI Trader Brain ───────────────────────────────────────────
    logger.info("\n🧠 Step 3: AI Trader Brain analyzing...")
    decision = analyze(market_data)

    # ── Step 4: Route Decision ────────────────────────────────────────────

    if decision.decision == "SKIP":
        # ── SKIP: Log and reschedule ──
        logger.info("🔴 Decision: SKIP — no viable setup")
        trade_log.log_decision(decision)
        bot_scheduler.schedule_default_scan()
        return

    elif decision.decision == "WATCH":
        # ── WATCH: Schedule one-shot analysis ──
        logger.info(f"🟡 Decision: WATCH — scheduling in {decision.next_check_minutes} min")
        trade_log.log_decision(decision)
        bot_scheduler.schedule_analysis(
            minutes_from_now=decision.next_check_minutes or settings.DEFAULT_SCAN_INTERVAL_MINUTES,
            reason=decision.next_check_reason or "AI watching setup",
        )
        return

    elif decision.decision == "EXECUTE":
        # ── EXECUTE: Run Risk Engine → MT5 ──
        logger.info("🟢 Decision: EXECUTE — running risk engine...")

        # Get today's stats for risk checks
        today_stats = trade_log.get_today_stats()

        risk_result = evaluate(
            decision=decision,
            mt5=mt5,
            trades_today_count=today_stats["executed_count"],
            daily_pnl=today_stats["total_pnl"],
        )

        if not risk_result.approved:
            # ── REJECTED by risk engine ──
            logger.info(f"🚫 Risk REJECTED: {risk_result.reason}")
            trade_log.log_decision(decision, risk=risk_result)
            bot_scheduler.schedule_default_scan()
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
                f"Entry: {order_result.price} | SL: {risk_result.sl_price} | TP: {risk_result.tp_price}"
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
    banner = """
╔══════════════════════════════════════════════════════════════╗
║              SMART MONEY TRADING BOT                        ║
║              ──────────────────────                         ║
║  AI-Powered · ICT/SMC · Two-Phase Scoring                   ║
╠══════════════════════════════════════════════════════════════╣
║  Mode:       {mode:<15}                                     ║
║  Pairs:      {pairs:<45}║
║  LLM:        {llm:<45}║
║  MT5:        {mt5:<45}║
║  Scoring:    P1 max {p1_max} (min {p1_min}) + P2 max {p2_max} (min {p2_min})              ║
║  Total min:  {total_min}/80 to EXECUTE                              ║
║  Risk:       ${risk}/trade · ${daily_limit}/day · {max_trades} trades/day            ║
╚══════════════════════════════════════════════════════════════╝
""".format(
        mode=settings.MODE.upper(),
        pairs=", ".join(settings.PAIRS),
        llm=f"{settings.LLM_MODEL} @ {settings.LLM_BASE_URL[:35]}",
        mt5=settings.MT5_BASE_URL,
        p1_max=settings.PHASE1_MAX_SCORE,
        p1_min=settings.PHASE1_MIN_REQUIRED,
        p2_max=settings.PHASE2_MAX_SCORE,
        p2_min=settings.PHASE2_MIN_REQUIRED,
        total_min=settings.TOTAL_MIN_REQUIRED,
        risk=settings.MAX_LOSS_PER_TRADE,
        daily_limit=settings.DAILY_LOSS_LIMIT_USD,
        max_trades=settings.MAX_TRADES_PER_DAY,
    )
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
        logger.info(f"✅ MT5 connected — Balance: ${account.balance:.2f} | "
                     f"Leverage: 1:{account.leverage}")
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

    logger.info("🚀 Starting Smart Money Trading Bot...")

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
