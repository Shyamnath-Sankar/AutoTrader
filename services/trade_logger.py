"""
trade_logger.py — Logs all trade decisions and results to data/trades.json.

Provides:
  - log_decision(): log any AI decision (EXECUTE, WATCH, SKIP)
  - log_execution(): log trade execution result from MT5
  - get_today_trades(): get today's trade history for AI context
  - get_today_stats(): get today's trade count + P&L
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
    # LOG A DECISION (any decision — EXECUTE, WATCH, SKIP)
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
            sl_pips=decision.sl_pips or 0,
            tp_pips=decision.tp_pips or 0,
            rr_ratio=decision.rr_ratio or 0.0,
        )

        # Add risk engine results
        if risk:
            record.risk_approved = risk.approved
            record.risk_reason = risk.reason
            record.lots = risk.lots
            record.entry_price = risk.entry_price
            record.sl_price = risk.sl_price
            record.tp_price = risk.tp_price
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
        """Get today's trade count and total P&L."""
        today_trades = self.get_today_trades()
        executed = [t for t in today_trades if t.get("decision") == "EXECUTE" and t.get("risk_approved")]
        total_pnl = sum(t.get("pnl", 0) or 0 for t in executed)
        return {
            "total_decisions": len(today_trades),
            "executed_count": len(executed),
            "total_pnl": total_pnl,
            "wins": len([t for t in executed if t.get("result") == "win"]),
            "losses": len([t for t in executed if t.get("result") == "loss"]),
            "pending": len([t for t in executed if t.get("result") is None]),
        }

    def get_recent_trades(self, count: int = 10) -> list[dict]:
        """Get the most recent N trade records."""
        records = self._read_all()
        return records[-count:] if len(records) >= count else records

    def get_pair_history(self, pair: str, count: int = 5) -> list[dict]:
        """Get recent trade history for a specific pair."""
        records = self._read_all()
        pair_trades = [r for r in records if r.get("pair") == pair]
        return pair_trades[-count:] if len(pair_trades) >= count else pair_trades
