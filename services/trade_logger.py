"""
trade_logger.py — Logs all trade decisions and results to data/trades.json.

Provides:
  - log_decision(): log any AI decision (TAKE, LEAVE)
  - log_execution(): log trade execution result from MT5
  - get_today_trades(): get today's trade history for AI context
  - get_today_stats(): get today's trade count + P&L + attempt count
"""

import json
import os
from datetime import datetime, date

import pytz
from loguru import logger

from config import settings
from core.models import OrderResult, RiskApproval, TradeRecord, TraderDecision


class TradeLogger:
    """Manages the trades.json log file."""

    def __init__(self, filepath: str | None = None):
        self.filepath = filepath or settings.TRADES_FILE
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create the data directory and trades.json if they don't exist."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _read_all(self) -> list[dict]:
        """Read all trade records from disk."""
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_all(self, records: list[dict]):
        """Write all trade records to disk."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    def _append(self, record: dict):
        """Append a single record to the log."""
        records = self._read_all()
        records.append(record)
        self._write_all(records)

    # ═══════════════════════════════════════════════════════════════════════
    # LOG A DECISION (any decision — TAKE, LEAVE)
    # ═══════════════════════════════════════════════════════════════════════

    def log_decision(
        self,
        decision: TraderDecision,
        risk: RiskApproval | None = None,
        order: OrderResult | None = None,
    ):
        """
        Log a complete trade decision cycle to trades.json.
        Called after the full pipeline runs (brain → risk → execute).
        """
        now = datetime.now(pytz.UTC).isoformat()

        record = TradeRecord(
            timestamp=now,
            pair=decision.pair or "",
            direction=decision.direction or "",
            decision=decision.decision,
            phase1_scores=decision.phase1_scores.model_dump(),
            phase1_total=decision.phase1_total,
            phase2_scores=decision.phase2_scores.model_dump(),
            phase2_total=decision.phase2_total,
            total_score=decision.total_score,
            reasoning=decision.reasoning,
        )

        # Add risk engine results
        if risk:
            record.risk_approved = risk.approved
            record.risk_reason = risk.reason
            record.lots = risk.lots
            record.entry_price = risk.entry_price
            record.sl_price = risk.sl_price
            record.tp_price = risk.tp_price
            record.sl_pips = risk.sl_pips
            record.tp_pips = risk.tp_pips
            record.rr_ratio = risk.rr_ratio

        # Add MT5 execution results
        if order:
            record.mt5_ticket = order.order
            record.mt5_message = order.message
            if order.success:
                record.entry_price = order.price or record.entry_price

        self._append(record.model_dump())
        logger.info(f"📝 Trade logged: {decision.decision} {decision.pair or ''} "
                     f"(score: {decision.total_score})")

    # ═══════════════════════════════════════════════════════════════════════
    # UPDATE TRADE RESULT (after trade closes)
    # ═══════════════════════════════════════════════════════════════════════

    def update_trade_result(
        self,
        mt5_ticket: int,
        result: str,  # "win" or "loss"
        pnl: float,
        pips_result: float,
    ):
        """Update an existing trade record with the final result."""
        records = self._read_all()
        for record in reversed(records):
            if record.get("mt5_ticket") == mt5_ticket:
                record["result"] = result
                record["pnl"] = pnl
                record["pips_result"] = pips_result
                self._write_all(records)
                logger.info(f"📝 Trade {mt5_ticket} updated: {result} | P&L: ${pnl:.2f}")
                return
        logger.warning(f"Trade ticket {mt5_ticket} not found in log")

    # ═══════════════════════════════════════════════════════════════════════
    # QUERIES
    # ═══════════════════════════════════════════════════════════════════════

    def get_today_trades(self) -> list[dict]:
        """Get all trade records from today (UTC)."""
        records = self._read_all()
        today = date.today().isoformat()
        return [
            r for r in records
            if r.get("timestamp", "").startswith(today)
        ]

    def get_today_stats(self) -> dict:
        """
        Get today's trade count, P&L, and attempt counts.

        Returns:
            dict with keys:
              - total_decisions: all logged decisions today
              - executed_count: risk-approved + successfully executed trades
              - take_attempts: total TAKE decisions (approved + rejected)
              - total_pnl: sum of P&L from executed trades
              - wins / losses / pending: execution result counts
              - rejections_today: TAKE decisions that were risk-rejected
              - last_rejection_pair: the pair of the most recent rejection
        """
        today_trades = self.get_today_trades()

        # TAKE attempts = all decisions where AI said TAKE (regardless of risk outcome)
        take_attempts = [
            t for t in today_trades
            if t.get("decision") in ("TAKE", "EXECUTE")  # support legacy name too
        ]

        # Successfully executed = TAKE + risk approved
        executed = [
            t for t in take_attempts
            if t.get("risk_approved") is True
        ]

        # Rejected = TAKE + risk rejected
        rejected = [
            t for t in take_attempts
            if t.get("risk_approved") is False
        ]

        total_pnl = sum(t.get("pnl", 0) or 0 for t in executed)

        last_rejection_pair = None
        if rejected:
            last_rejection_pair = rejected[-1].get("pair")

        return {
            "total_decisions": len(today_trades),
            "executed_count": len(executed),
            "take_attempts": len(take_attempts),
            "total_pnl": total_pnl,
            "wins": len([t for t in executed if t.get("result") == "win"]),
            "losses": len([t for t in executed if t.get("result") == "loss"]),
            "pending": len([t for t in executed if t.get("result") is None]),
            "rejections_today": len(rejected),
            "last_rejection_pair": last_rejection_pair,
        }

    def get_last_decision(self) -> dict | None:
        """Get the most recent decision for anti-inflation context."""
        records = self._read_all()
        if records:
            return records[-1]
        return None

    def get_recent_trades(self, count: int = 10) -> list[dict]:
        """Get the most recent N trade records."""
        records = self._read_all()
        return records[-count:] if len(records) >= count else records

    def get_pair_history(self, pair: str, count: int = 5) -> list[dict]:
        """Get recent trade history for a specific pair."""
        records = self._read_all()
        pair_trades = [r for r in records if r.get("pair") == pair]
        return pair_trades[-count:] if len(pair_trades) >= count else pair_trades

    def get_consecutive_rejections(self, pair: str | None = None) -> int:
        """
        Count consecutive TAKE rejections from the end of today's log.
        Used for cooldown escalation.
        """
        today_trades = self.get_today_trades()
        count = 0
        for t in reversed(today_trades):
            is_take = t.get("decision") in ("TAKE", "EXECUTE")
            is_rejected = t.get("risk_approved") is False
            matches_pair = (pair is None) or (t.get("pair") == pair)

            if is_take and is_rejected and matches_pair:
                count += 1
            else:
                break  # stop at first non-rejection
        return count
