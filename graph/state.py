"""
state.py — LangGraph state definition for the trading pipeline.

The TradingState flows through every node in the graph,
accumulating data as it progresses: Gates → Data → Brain → Risk → Execute.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from core.models import MarketDataPayload, OrderResult, RiskApproval, TraderDecision


class TradingState(TypedDict, total=False):
    """State that flows through the LangGraph trading pipeline."""

    # ── Inputs (set before graph invocation) ──
    pair: str
    balance: float
    today_stats: dict[str, Any]

    # ── Gate results ──
    gates_passed: bool
    gate_reason: str

    # ── Market data ──
    market_data: Optional[MarketDataPayload]

    # ── AI decision (Trader Brain) ──
    decision: Optional[TraderDecision]

    # ── Risk engine result ──
    risk_result: Optional[RiskApproval]

    # ── Execution result ──
    order_result: Optional[OrderResult]

    # ── Final status ──
    status: str  # GATE_FAILED, SKIP, WATCH, EXECUTE, REJECTED, TRADE_FAILED, ERROR
    error: Optional[str]
