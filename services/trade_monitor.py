"""
trade_monitor.py — Background Trade Monitor.

Architecture (per entry-architecture.svg):
  Watches open positions and manages them post-execution:
    1. TP1 hit → close 50% of position → move SL to breakeven
    2. TP2 hit → close remaining → log final result
    3. Limit order expiry → cancel unfilled pending orders after N candles

Runs as a background daemon thread, polling MT5 every MONITOR_POLL_SECONDS.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import pytz
from loguru import logger

from config import settings
from services.mt5_client import MT5Client


class TrackedPosition:
    """A position being monitored for TP1/TP2 management."""

    def __init__(
        self,
        ticket: int,
        pair: str,
        direction: str,
        lots: float,
        entry_price: float,
        sl_price: float,
        tp1_price: float,
        tp2_price: float,
        order_type: str = "market",
        created_at: datetime | None = None,
    ):
        self.ticket = ticket
        self.pair = pair
        self.direction = direction
        self.original_lots = lots
        self.entry_price = entry_price
        self.sl_price = sl_price
        self.tp1_price = tp1_price
        self.tp2_price = tp2_price
        self.order_type = order_type
        self.created_at = created_at or datetime.now(pytz.UTC)

        # State tracking
        self.tp1_hit = False
        self.tp1_closed = False
        self.sl_moved_to_be = False
        self.is_active = True


class TrackedPendingOrder:
    """A pending (limit) order being monitored for expiry."""

    def __init__(
        self,
        ticket: int,
        pair: str,
        created_at: datetime | None = None,
    ):
        self.ticket = ticket
        self.pair = pair
        self.created_at = created_at or datetime.now(pytz.UTC)
        self.expiry_candles = settings.LIMIT_ORDER_EXPIRY_CANDLES
        # 15min candles → expiry in minutes
        self.expiry_minutes = self.expiry_candles * 15
        self.is_active = True


class TradeMonitor:
    """Background daemon that manages open positions and pending orders."""

    def __init__(self, mt5: MT5Client):
        self.mt5 = mt5
        self._tracked_positions: dict[int, TrackedPosition] = {}
        self._tracked_pending: dict[int, TrackedPendingOrder] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ── Public API ───────────────────────────────────────────────────────────

    def track_position(
        self,
        ticket: int,
        pair: str,
        direction: str,
        lots: float,
        entry_price: float,
        sl_price: float,
        tp1_price: float,
        tp2_price: float,
        order_type: str = "market",
    ):
        """Register a new position for TP1/TP2 monitoring."""
        with self._lock:
            self._tracked_positions[ticket] = TrackedPosition(
                ticket=ticket,
                pair=pair,
                direction=direction,
                lots=lots,
                entry_price=entry_price,
                sl_price=sl_price,
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                order_type=order_type,
            )
        logger.info(
            f"📡 Trade Monitor: tracking position #{ticket} | "
            f"{direction} {lots} lots {pair} | TP1={tp1_price} TP2={tp2_price}"
        )

    def track_pending_order(self, ticket: int, pair: str):
        """Register a pending order for expiry monitoring."""
        with self._lock:
            self._tracked_pending[ticket] = TrackedPendingOrder(
                ticket=ticket,
                pair=pair,
            )
        logger.info(
            f"📡 Trade Monitor: tracking pending order #{ticket} | "
            f"{pair} | expires in {settings.LIMIT_ORDER_EXPIRY_CANDLES} candles"
        )

    def start(self):
        """Start the background monitoring thread."""
        if not settings.MONITOR_ENABLED:
            logger.info("Trade Monitor disabled in settings")
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="TradeMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"📡 Trade Monitor started (polling every {settings.MONITOR_POLL_SECONDS}s)")

    def stop(self):
        """Stop the monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("📡 Trade Monitor stopped")

    @property
    def active_count(self) -> int:
        """Number of actively tracked positions."""
        with self._lock:
            return sum(1 for p in self._tracked_positions.values() if p.is_active)

    # ── Background Loop ──────────────────────────────────────────────────────

    def _monitor_loop(self):
        """Main monitoring loop — runs in background thread."""
        while self._running:
            try:
                self._check_positions()
                self._check_pending_orders()
                self._cleanup_inactive()
            except Exception as e:
                logger.error(f"Trade Monitor error: {e}")

            time.sleep(settings.MONITOR_POLL_SECONDS)

    # ── Position Monitoring ──────────────────────────────────────────────────

    def _check_positions(self):
        """Check all tracked positions for TP1/TP2 hits."""
        with self._lock:
            positions_to_check = {
                k: v for k, v in self._tracked_positions.items() if v.is_active
            }

        if not positions_to_check:
            return

        # Get current open positions from MT5
        mt5_positions = self.mt5.get_positions()
        mt5_tickets = {p.get("ticket", 0) for p in mt5_positions}

        for ticket, tracked in positions_to_check.items():
            # Check if position still exists in MT5
            if ticket not in mt5_tickets:
                logger.info(f"📡 Position #{ticket} no longer open in MT5 — removing from tracking")
                with self._lock:
                    tracked.is_active = False
                continue

            # Get current price for the pair
            price_data = self.mt5.get_price(tracked.pair)
            if not price_data:
                continue

            current_price = price_data.bid if tracked.direction == "BUY" else price_data.ask

            # Check TP1
            if not tracked.tp1_hit:
                tp1_reached = self._check_tp_hit(
                    current_price, tracked.tp1_price, tracked.direction
                )
                if tp1_reached:
                    self._handle_tp1_hit(tracked)

            # Check TP2 (only after TP1 was handled)
            elif not tracked.tp1_closed:
                # Still waiting for TP1 close to process
                pass
            else:
                # TP1 handled, check TP2
                tp2_reached = self._check_tp_hit(
                    current_price, tracked.tp2_price, tracked.direction
                )
                if tp2_reached:
                    self._handle_tp2_hit(tracked)

    def _check_tp_hit(self, current_price: float, tp_price: float, direction: str) -> bool:
        """Check if price has reached or passed a TP level."""
        if direction == "BUY":
            return current_price >= tp_price
        else:
            return current_price <= tp_price

    def _handle_tp1_hit(self, tracked: TrackedPosition):
        """TP1 hit: close 50% of position, move SL to breakeven."""
        logger.info(f"🎯 TP1 HIT on position #{tracked.ticket}!")

        tracked.tp1_hit = True

        # Calculate 50% volume
        close_volume = round(tracked.original_lots * settings.TP1_CLOSE_PCT, 2)
        close_volume = max(settings.MIN_LOT, close_volume)

        # Close partial position
        result = self.mt5.close_partial(tracked.ticket, close_volume)
        if result.success:
            logger.info(
                f"✅ Closed {close_volume} lots (50%) of position #{tracked.ticket} at TP1"
            )
            tracked.tp1_closed = True

            # Move SL to breakeven (entry price)
            sl_moved = self.mt5.modify_sl(tracked.ticket, tracked.entry_price)
            if sl_moved:
                logger.info(
                    f"🔒 SL moved to breakeven ({tracked.entry_price}) on position #{tracked.ticket}"
                )
                tracked.sl_moved_to_be = True
            else:
                logger.warning(
                    f"⚠️ Failed to move SL to breakeven on position #{tracked.ticket}"
                )
        else:
            logger.error(
                f"❌ Failed to close partial position #{tracked.ticket}: {result.message}"
            )
            # Still mark TP1 as closed to avoid retry loops
            tracked.tp1_closed = True

    def _handle_tp2_hit(self, tracked: TrackedPosition):
        """TP2 hit: close remaining position, log result."""
        logger.info(f"🎯🎯 TP2 HIT on position #{tracked.ticket}!")

        # Close remaining position
        result = self.mt5.close_position(tracked.ticket)
        if result:
            logger.info(f"✅ Full close of position #{tracked.ticket} at TP2")
        else:
            logger.warning(f"⚠️ Close may have failed for #{tracked.ticket} — MT5 might have auto-closed")

        with self._lock:
            tracked.is_active = False

    # ── Pending Order Monitoring ─────────────────────────────────────────────

    def _check_pending_orders(self):
        """Check pending orders for expiry."""
        with self._lock:
            pending_to_check = {
                k: v for k, v in self._tracked_pending.items() if v.is_active
            }

        if not pending_to_check:
            return

        now = datetime.now(pytz.UTC)

        for ticket, tracked in pending_to_check.items():
            elapsed = (now - tracked.created_at).total_seconds() / 60
            if elapsed >= tracked.expiry_minutes:
                logger.info(
                    f"⏰ Pending order #{ticket} expired "
                    f"({elapsed:.0f}min > {tracked.expiry_minutes}min limit)"
                )
                cancelled = self.mt5.cancel_pending_order(ticket)
                if cancelled:
                    logger.info(f"✅ Cancelled expired pending order #{ticket}")
                else:
                    logger.warning(f"⚠️ Failed to cancel pending order #{ticket} — may already be filled")

                with self._lock:
                    tracked.is_active = False

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def _cleanup_inactive(self):
        """Remove inactive positions and pending orders from tracking."""
        with self._lock:
            inactive_pos = [k for k, v in self._tracked_positions.items() if not v.is_active]
            for k in inactive_pos:
                del self._tracked_positions[k]

            inactive_pending = [k for k, v in self._tracked_pending.items() if not v.is_active]
            for k in inactive_pending:
                del self._tracked_pending[k]
