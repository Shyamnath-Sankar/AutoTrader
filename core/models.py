"""
models.py — Pydantic data models for the Smart Money Trading Bot.
Every structured data object the bot uses is defined here.

Architecture decisions:
  - TraderDecision uses TAKE / LEAVE (two-decision system)
  - Trader Brain provides scores + direction + pair; NO SL/TP (Risk Engine computes those)
  - Risk Engine computes SL from swing structure, TP from R:R, lot size from budget
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# Gate Results
# ═══════════════════════════════════════════════════════════════════════════════

class GateResult(BaseModel):
    """Result of a single hard gate check."""
    gate_name: str
    passed: bool
    skip_minutes: int = 0
    reason: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Market Data
# ═══════════════════════════════════════════════════════════════════════════════

class IndicatorSet(BaseModel):
    """Indicator values for a single timeframe."""
    timeframe: str
    close: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    rsi: Optional[float] = None
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_basis: Optional[float] = None
    adx: Optional[float] = None
    adx_plus_di: Optional[float] = None
    adx_minus_di: Optional[float] = None
    supertrend: Optional[float] = None


class SMCData(BaseModel):
    """Smart Money Concepts analysis for a pair."""
    pair: str
    timeframe: str
    # FVG
    fvg_bullish_count: int = 0
    fvg_bearish_count: int = 0
    fvg_nearest_level: Optional[float] = None
    fvg_nearest_type: Optional[str] = None  # "bullish" or "bearish"
    # Order Blocks
    ob_bullish_count: int = 0
    ob_bearish_count: int = 0
    ob_nearest_top: Optional[float] = None
    ob_nearest_bottom: Optional[float] = None
    ob_nearest_type: Optional[str] = None
    # BOS / CHoCH
    last_bos_type: Optional[str] = None     # "bullish" or "bearish"
    last_choch_type: Optional[str] = None
    last_bos_level: Optional[float] = None
    last_choch_level: Optional[float] = None
    # Liquidity
    liquidity_swept: bool = False
    liquidity_sweep_type: Optional[str] = None   # "bullish" or "bearish"
    liquidity_level: Optional[float] = None
    liquidity_candles_ago: Optional[int] = None   # how recent was the sweep
    # Swing H/L
    latest_swing_high: Optional[float] = None
    latest_swing_low: Optional[float] = None
    # Retracements
    current_retracement_pct: Optional[float] = None
    deepest_retracement_pct: Optional[float] = None


class NewsEvent(BaseModel):
    """An upcoming economic news event."""
    title: str
    impact: str          # "High", "Medium", "Low"
    currency: str
    time_utc: str
    minutes_away: int


class PairMarketData(BaseModel):
    """All collected market data for a single pair."""
    pair: str
    indicators: dict[str, IndicatorSet] = {}    # key = timeframe
    smc: dict[str, SMCData] = {}                # key = timeframe
    news_events: list[NewsEvent] = []


class MarketDataPayload(BaseModel):
    """Complete market data for all pairs, ready for AI consumption."""
    pairs: dict[str, PairMarketData] = {}
    account_balance: float = 0.0
    account_equity: float = 0.0
    daily_pnl: float = 0.0
    open_positions: list[dict[str, Any]] = []
    trades_today: list[dict[str, Any]] = []
    trades_today_count: int = 0
    timestamp: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# AI Scoring & Decision — Trader Brain (Agent 1)
#
# Decision types (per architecture diagram):
#   TAKE     → high-confidence setup, proceed to Risk Engine
#   LEAVE    → no viable setup, log and rescan at default interval
# ═══════════════════════════════════════════════════════════════════════════════

class Phase1Scores(BaseModel):
    """Sub-scores for Phase 1 — Market Context."""
    regime: int = 0
    session: int = 0
    news: int = 0
    weekly_4h_bias: int = 0
    trend_1h: int = 0
    trigger_15m: int = 0


class Phase2Scores(BaseModel):
    """Sub-scores for Phase 2 — SMC Confirmation."""
    liquidity_sweep: int = 0
    order_block: int = 0
    fvg: int = 0
    bos_choch: int = 0


class TraderDecision(BaseModel):
    """
    Output from AI Agent 1 — Trader Brain.

    The Trader Brain outputs ONLY scores, direction, pair, and reasoning.
    SL/TP/lots are computed by the Risk Engine from swing structure.
    """
    decision: str                    # "TAKE" or "LEAVE"
    pair: Optional[str] = None
    direction: Optional[str] = None  # "BUY", "SELL", or None
    phase1_scores: Phase1Scores = Field(default_factory=Phase1Scores)
    phase1_total: int = 0
    phase2_scores: Phase2Scores = Field(default_factory=Phase2Scores)
    phase2_total: int = 0
    total_score: int = 0
    reasoning: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Candidate — Output from AI Entry Analyzer
# ═══════════════════════════════════════════════════════════════════════════════

class EntryCandidate(BaseModel):
    """
    A single candidate entry level selected by the Entry Analyzer AI.

    The Entry Analyzer receives OHLCV + SMC structures + direction,
    and picks the optimal entry price with structure-based SL and TP.
    """
    entry_type: str = ""            # "sweep", "ob", "fvg", "swing", "round"
    priority: int = 4               # 1 (sweep/best) to 4 (swing/round)
    entry_price: float = 0.0
    sl_price: float = 0.0
    sl_pips: int = 0
    tp_price: float = 0.0
    tp_pips: int = 0
    rr_ratio: float = 0.0           # TP/SL ratio
    confluence_count: int = 1       # how many structures overlap
    distance_from_current: float = 0.0  # pips from current price
    reasoning: str = ""             # why this level is significant


# ═══════════════════════════════════════════════════════════════════════════════
# Risk Engine — Agent 2
#
# Risk Engine receives: scores + direction + swing structure + live price + account
# Risk Engine computes: natural SL from swing H/L, TP from R:R tier, lot size from budget
# ═══════════════════════════════════════════════════════════════════════════════

class RiskApproval(BaseModel):
    """
    Output from Risk Engine — deterministic validation of an entry candidate.
    """
    approved: bool
    lots: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    sl_pips: int = 0
    tp_pips: int = 0
    entry_price: float = 0.0
    order_type: str = "market"     # "market" or "limit"
    reason: str = ""
    rr_ratio: float = 0.0
    entry_type: str = ""           # sweep/ob/fvg/swing/round
    confluence_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# MT5 Data
# ═══════════════════════════════════════════════════════════════════════════════

class AccountInfo(BaseModel):
    """MT5 account information."""
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    profit: float = 0.0
    currency: str = "USD"
    leverage: int = 0
    login: int = 0
    server: str = ""


class PriceData(BaseModel):
    """Live bid/ask price from MT5."""
    symbol: str = ""
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    spread_pips: float = 0.0


class OrderResult(BaseModel):
    """Result from placing a market order via MT5."""
    success: bool = False
    order: Optional[int] = None
    message: str = ""
    symbol: str = ""
    type: str = ""
    volume: float = 0.0
    price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    error_code: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Record
# ═══════════════════════════════════════════════════════════════════════════════

class TradeRecord(BaseModel):
    """A complete trade log entry for trades.json."""
    timestamp: str
    pair: str
    direction: str
    decision: str                  # TAKE or LEAVE
    phase1_scores: dict = {}
    phase1_total: int = 0
    phase2_scores: dict = {}
    phase2_total: int = 0
    total_score: int = 0
    reasoning: str = ""
    risk_approved: Optional[bool] = None
    risk_reason: str = ""
    lots: float = 0.0
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    sl_pips: int = 0
    tp_pips: int = 0
    rr_ratio: float = 0.0
    entry_type: str = ""           # sweep/ob/fvg/swing/round
    order_type: str = "market"     # market or limit
    confluence_count: int = 0
    mt5_ticket: Optional[int] = None
    mt5_message: str = ""
    result: Optional[str] = None   # "win", "loss", or None (pending)
    pnl: Optional[float] = None
    pips_result: Optional[float] = None
