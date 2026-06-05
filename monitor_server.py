"""
monitor_server.py — Standalone Trade Monitor Process.

Run this in a SEPARATE terminal alongside the main bot:
  python monitor_server.py

This process:
  1. Connects to MT5 (Python library for prices + HTTP for order actions)
  2. Loads all open trades from data/trades.json on startup
  3. Polls every MONITOR_POLL_SECONDS (default 30s) for TP1/TP2 hits
  4. At TP1: closes 50% of position + moves SL to breakeven
  5. At TP2: closes remaining position + logs result
  6. Cancels pending limit orders that have expired (> LIMIT_ORDER_EXPIRY_CANDLES × 15min)

This is fully independent of the main bot — it can run even if the main bot
crashes, and can be restarted without losing tracking state (reads trades.json).
"""

import os
import sys
import signal
import time
import json
from datetime import datetime

import pytz
from loguru import logger

# Fix Windows console encoding
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.makedirs("logs", exist_ok=True)

from config import settings

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<cyan>{time:HH:mm:ss}</cyan> | <level>{level: <8}</level> | [MONITOR] {message}")
logger.add("logs/monitor.log", level="DEBUG", rotation="10 MB", retention="30 days",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | [MONITOR] {message}")

from services.mt5_client import MT5Client
from services.trade_monitor import TradeMonitor, TrackedPosition, TrackedPendingOrder


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD OPEN TRADES FROM trades.json
# ═══════════════════════════════════════════════════════════════════════════════

def load_open_trades_from_log(monitor: TradeMonitor) -> int:
    """
    Read trades.json and register any open (executed, no result yet) trades
    for TP1/TP2 monitoring. This allows the monitor to resume after a restart.

    A trade is considered "open" if:
      - decision == TAKE
      - risk_approved == True
      - mt5_ticket is not None
      - result is None (no win/loss recorded yet)
    """
    filepath = settings.TRADES_FILE
    if not os.path.exists(filepath):
        logger.info(f"No trades.json found at {filepath} — starting fresh")
        return 0

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not read trades.json: {e}")
        return 0

    # Get currently open positions from MT5 to cross-reference tickets
    try:
        mt5_http = monitor.mt5
        open_positions = mt5_http.get_positions()
        open_tickets   = {p.get("ticket") for p in open_positions}
        open_orders    = mt5_http.get_pending_orders()
        pending_tickets = {o.get("ticket") for o in open_orders}
    except Exception as e:
        logger.warning(f"Could not fetch MT5 positions: {e} — loading all unresolved trades")
        open_tickets    = set()
        pending_tickets = set()

    loaded = 0
    for trade in records:
        # Only TAKE decisions that were executed and have no result yet
        if trade.get("decision") not in ("TAKE", "EXECUTE"):
            continue
        if trade.get("risk_approved") is not True:
            continue
        if trade.get("result") is not None:
            continue

        ticket = trade.get("mt5_ticket")
        if ticket is None:
            continue

        pair      = trade.get("pair", "")
        direction = trade.get("direction", "")
        lots      = trade.get("lots", settings.MIN_LOT)
        entry     = trade.get("entry_price", 0.0)
        sl        = trade.get("sl_price", 0.0)
        tp        = trade.get("tp_price", 0.0)
        order_t   = trade.get("order_type", "market")

        # Compute TP1 as midpoint between entry and TP2
        if direction == "BUY":
            tp1 = entry + (tp - entry) * 0.5
        else:
            tp1 = entry - (entry - tp) * 0.5

        if ticket in open_tickets:
            # It's a live open position
            monitor.track_position(
                ticket      = ticket,
                pair        = pair,
                direction   = direction,
                lots        = lots,
                entry_price = entry,
                sl_price    = sl,
                tp1_price   = tp1,
                tp2_price   = tp,
                order_type  = order_t,
            )
            loaded += 1
        elif ticket in pending_tickets:
            # It's a pending limit order
            monitor.track_pending_order(ticket=ticket, pair=pair)
            loaded += 1
        else:
            logger.debug(f"Trade #{ticket} {pair} not found in MT5 — skipping (may be closed already)")

    return loaded


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT BANNER
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    tp1_pct = int(settings.TP1_CLOSE_PCT * 100)
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║         SMART MONEY BOT — TRADE MONITOR SERVER              ║
║         ─────────────────────────────────────────           ║
║  Manages open positions post-execution:                     ║
║    1. TP1 hit → close {tp1_pct}% of position                       ║
║    2. TP1 hit → move SL to breakeven (entry price)          ║
║    3. TP2 hit → close remaining position                     ║
║    4. Pending limit → cancel after {settings.LIMIT_ORDER_EXPIRY_CANDLES}x15min candles       ║
╠══════════════════════════════════════════════════════════════╣
║  Poll interval: {settings.MONITOR_POLL_SECONDS}s                                       ║
║  MT5 HTTP:      {settings.MT5_BASE_URL:<44}║
║  Trades file:   {settings.TRADES_FILE:<44}║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print_banner()
    logger.info("🚀 Trade Monitor Server starting...")

    mt5 = MT5Client()

    # Check MT5 connectivity
    if not mt5.is_connected():
        logger.warning("⚠️  MT5 HTTP server not reachable — monitor will retry on each poll")
    else:
        account = mt5.get_account_info()
        if account:
            logger.info(f"✅ MT5 connected — Account: {account.login} | Balance: ${account.balance:.2f}")

    monitor = TradeMonitor(mt5)

    # Load open trades from trades.json on startup
    loaded = load_open_trades_from_log(monitor)
    if loaded > 0:
        logger.info(f"📂 Loaded {loaded} open trade(s) from trades.json for monitoring")
    else:
        logger.info("📂 No open trades to resume — waiting for new trades from main bot")

    # Graceful shutdown handling
    _running = [True]

    def signal_handler(signum, frame):
        logger.info("\n🛑 Monitor shutdown requested...")
        _running[0] = False

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"📡 Monitor active — polling every {settings.MONITOR_POLL_SECONDS}s | Ctrl+C to stop\n")

    # Main polling loop
    while _running[0]:
        try:
            # Check for new trades that arrived since last poll
            # (The main bot's TradeMonitor thread registers them; here we scan file for new ones)
            current_count = monitor.active_count
            new_loaded = load_open_trades_from_log(monitor)
            if new_loaded > current_count:
                logger.info(f"📂 Picked up {new_loaded - current_count} new trade(s) from log")

            # Monitor tick: check TP1/TP2 and pending order expiry
            monitor._check_positions()
            monitor._check_pending_orders()
            monitor._cleanup_inactive()

            # Show status every 5 minutes
            now_min = datetime.now(pytz.UTC).minute
            if now_min % 5 == 0:
                logger.info(
                    f"💓 Monitor heartbeat — "
                    f"tracking {monitor.active_count} position(s)"
                )

        except Exception as e:
            logger.error(f"Monitor loop error: {e}")

        time.sleep(settings.MONITOR_POLL_SECONDS)

    logger.info("📡 Trade Monitor Server stopped")


if __name__ == "__main__":
    main()
