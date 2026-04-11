"""
main.py — Entry point & Orchestrator for the Smart Money Trading Bot.

This is the main daemon that:
  1. Connects to MT5 natively (login/password/server)
  2. Fetches balance, computes dynamic 5% risk per trade
  3. Runs each pair through a LangGraph pipeline: Gates → Data → Brain → Risk → Execute
  4. If one pair has no trade, the others still run independently
  5. Routes decisions (EXECUTE → Risk → MT5, WATCH → Schedule, SKIP → Log)
  6. Loops forever on weekdays
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

from agents.risk_engine import compute_risk_limits
from services.mt5_client import MT5Client
from services.trade_logger import TradeLogger
from core.scheduler import BotScheduler
from graph.trading_graph import build_trading_graph, set_services


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════════

mt5 = MT5Client()
trade_log = TradeLogger()
bot_scheduler = BotScheduler()
shutdown_requested = False

# Build the LangGraph pipeline
set_services(mt5, trade_log)
trading_graph = build_trading_graph()


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    logger.info("\n🛑 Shutdown requested — finishing current cycle...")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS CYCLE (LANGGRAPH MULTI-PAIR)
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis_cycle():
    """
    Run one complete analysis cycle across ALL pairs via LangGraph:
      1. Fetch balance once
      2. For each pair, invoke the LangGraph pipeline
      3. Each pair runs: Gates → Data → Brain → Risk → Execute
      4. Log results and schedule next scan
    """
    global shutdown_requested
    if shutdown_requested:
        return

    cycle_start = datetime.now(pytz.UTC)
    logger.info(f"\n{'═' * 70}")
    logger.info(f"🔄 ANALYSIS CYCLE START — {cycle_start.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info(f"   Mode: {settings.MODE} | Pairs: {', '.join(settings.PAIRS)}")

    # ── Fetch balance once for all pairs ──
    account = mt5.get_account_info()
    if account:
        balance = account.balance
        risk_limits = compute_risk_limits(balance)
        logger.info(
            f"   Balance: ${balance:.2f} | "
            f"Risk/trade: ${risk_limits['max_loss_per_trade']:.2f} (5%) | "
            f"Daily cap: -${risk_limits['daily_loss_limit']:.2f} (15%)"
        )
    else:
        balance = 0.0
        logger.warning("   ⚠️ Could not fetch MT5 balance — using $0")

    # ── Get today's stats ──
    today_stats = trade_log.get_today_stats()
    logger.info(
        f"   Today: {today_stats['executed_count']}/{settings.MAX_TRADES_PER_DAY} trades | "
        f"P&L: ${today_stats['total_pnl']:.2f}"
    )
    logger.info(f"{'═' * 70}")

    # ── Check if we've hit daily limits (skip all pairs) ──
    if today_stats["executed_count"] >= settings.MAX_TRADES_PER_DAY:
        logger.info("🚫 Daily trade limit reached — skipping all pairs")
        bot_scheduler.schedule_default_scan()
        return

    daily_limit = compute_risk_limits(balance)["daily_loss_limit"] if balance > 0 else settings.DAILY_LOSS_LIMIT_USD
    if today_stats["total_pnl"] <= -daily_limit:
        logger.info(f"🚫 Daily loss limit reached (${today_stats['total_pnl']:.2f}) — skipping all pairs")
        bot_scheduler.schedule_default_scan()
        return

    # ── Run each pair through the LangGraph pipeline ──
    results = []

    for i, pair in enumerate(settings.PAIRS):
        if shutdown_requested:
            break
        if i > 0:
            time.sleep(2)  # stagger pairs to avoid TradingView rate limit

        logger.info(f"\n🔗 [{pair}] Invoking LangGraph pipeline...")

        try:
            # Build initial state for this pair
            initial_state = {
                "pair": pair,
                "balance": balance,
                "today_stats": today_stats,
                "gates_passed": False,
                "gate_reason": "",
                "market_data": None,
                "decision": None,
                "risk_result": None,
                "order_result": None,
                "status": "PENDING",
                "error": None,
            }

            # Invoke the compiled LangGraph
            final_state = trading_graph.invoke(initial_state)

            results.append({
                "pair": pair,
                "status": final_state.get("status", "ERROR"),
                "decision": final_state.get("decision"),
                "risk_result": final_state.get("risk_result"),
                "order_result": final_state.get("order_result"),
                "error": final_state.get("error"),
            })

        except Exception as e:
            logger.error(f"❌ [{pair}] LangGraph pipeline error: {e}")
            results.append({
                "pair": pair,
                "status": "ERROR",
                "error": str(e),
            })

    # ── Summarize results ──
    logger.info(f"\n{'─' * 70}")
    logger.info("📊 CYCLE SUMMARY:")

    executed = []
    watched = []
    skipped = []
    errors = []

    for r in results:
        status = r["status"]
        pair = r["pair"]

        if status == "EXECUTE":
            executed.append(r)
            logger.info(f"  🟢 {pair}: EXECUTED")
        elif status == "WATCH":
            watched.append(r)
            decision = r.get("decision")
            mins = decision.next_check_minutes if decision else "?"
            logger.info(f"  🟡 {pair}: WATCH (check in {mins} min)")
        elif status in ("SKIP", "GATE_FAILED"):
            skipped.append(r)
            reason = r.get("error", "")
            decision = r.get("decision")
            if decision:
                reason = decision.reasoning[:80]
            logger.info(f"  🔴 {pair}: {status} — {reason}")
        elif status == "REJECTED":
            skipped.append(r)
            risk = r.get("risk_result")
            reason = risk.reason if risk else "Unknown"
            logger.info(f"  🚫 {pair}: REJECTED — {reason}")
        else:
            errors.append(r)
            logger.info(f"  ❌ {pair}: ERROR — {r.get('error', 'Unknown')}")

    logger.info(f"\n  Executed: {len(executed)} | Watched: {len(watched)} | "
                f"Skipped: {len(skipped)} | Errors: {len(errors)}")
    logger.info(f"{'─' * 70}")

    # ── Schedule next scan ──
    # If any pair is in WATCH, use its next_check_minutes
    if watched:
        watch_minutes = [
            r["decision"].next_check_minutes
            for r in watched
            if r.get("decision") and r["decision"].next_check_minutes
        ]
        if watch_minutes:
            min_watch_minutes = min(watch_minutes)
            bot_scheduler.schedule_analysis(
                minutes_from_now=min_watch_minutes,
                reason=f"Watching {len(watched)} pair(s)",
            )
        else:
            bot_scheduler.schedule_default_scan()
    else:
        bot_scheduler.schedule_default_scan()


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    """Print the bot startup banner."""
    # Compute dynamic risk for display
    account = mt5.get_account_info()
    balance = account.balance if account else 0
    risk_limits = compute_risk_limits(balance)

    banner = """
╔══════════════════════════════════════════════════════════════╗
║              SMART MONEY TRADING BOT                        ║
║              ──────────────────────                         ║
║  AI-Powered · ICT/SMC · LangGraph Pipeline                   ║
╠══════════════════════════════════════════════════════════════╣
║  Mode:       {mode:<15}                                     ║
║  Pairs:      {pairs:<45}║
║  LLM:        {llm:<45}║
║  MT5:        {mt5_info:<45}║
║  Scoring:    P1 max {p1_max} (min {p1_min}) + P2 max {p2_max} (min {p2_min})              ║
║  Total min:  {total_min}/80 to EXECUTE                              ║
║  Risk:       {risk_pct}% of balance ({risk_usd})/trade             ║
║  Daily cap:  {daily_pct}% of balance ({daily_usd})/day             ║
║  Max trades: {max_trades}/day · LangGraph orchestration              ║
╚══════════════════════════════════════════════════════════════╝
""".format(
        mode=settings.MODE.upper(),
        pairs=", ".join(settings.PAIRS),
        llm=f"{settings.LLM_MODEL[:35]}",
        mt5_info=f"Login {settings.MT5_LOGIN} @ {settings.MT5_SERVER}",
        p1_max=settings.PHASE1_MAX_SCORE,
        p1_min=settings.PHASE1_MIN_REQUIRED,
        p2_max=settings.PHASE2_MAX_SCORE,
        p2_min=settings.PHASE2_MIN_REQUIRED,
        total_min=settings.TOTAL_MIN_REQUIRED,
        risk_pct=int(settings.MAX_LOSS_PER_TRADE_PCT * 100),
        risk_usd=f"${risk_limits['max_loss_per_trade']:.2f}",
        daily_pct=int(settings.DAILY_LOSS_LIMIT_PCT * 100),
        daily_usd=f"${risk_limits['daily_loss_limit']:.2f}",
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
        if account:
            balance = account.balance
            risk_limits = compute_risk_limits(balance)
            logger.info(
                f"✅ MT5 connected — Balance: ${balance:.2f} | "
                f"Leverage: 1:{account.leverage} | "
                f"Risk/trade: ${risk_limits['max_loss_per_trade']:.2f} (5%)"
            )
        else:
            logger.warning("⚠️  MT5 connected but cannot read account info")
    else:
        logger.warning("⚠️  MT5 not connected — check credentials and MT5 terminal")
        logger.warning(f"   Login: {settings.MT5_LOGIN} | Server: {settings.MT5_SERVER}")
        logger.warning(f"   Path: {settings.MT5_PATH}")

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

    # Check pairs have configs
    for pair in settings.PAIRS:
        sym_config = settings.get_symbol_config(pair)
        logger.info(
            f"✅ {pair}: pip_size={sym_config['pip_size']} | "
            f"pip_value={sym_config['pip_value_micro']} | "
            f"lots=[{sym_config['min_lot']}-{sym_config['max_lot']}] | "
            f"exchange={sym_config['exchange']}"
        )

    # Check LangGraph is ready
    logger.info("✅ LangGraph pipeline compiled (Gates → Data → Brain → Risk → Execute)")

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
        mt5.shutdown()
        logger.info("👋 Smart Money Trading Bot stopped. Goodbye!")


if __name__ == "__main__":
    main()
