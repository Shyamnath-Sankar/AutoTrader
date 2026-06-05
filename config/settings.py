"""
settings.py — Central configuration for the Smart Money Trading Bot.
Every tunable variable lives here. No magic numbers anywhere else.

Scoring model:
  - All thresholds are PERCENTAGES (not hardcoded point values).
  - When news data is unavailable for a pair, the news component (P1_NEWS_POINTS)
    is EXCLUDED from the effective max score — no penalty, just a smaller denominator.
  - The 60% gate is then applied to the reduced effective max.
"""

import math
import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# SCORING CONFIG — percentage-based thresholds, not hardcoded point minimums
# Change these and the AI prompt adapts automatically.
# ═══════════════════════════════════════════════════════════════════════════════

# Phase 1: max 73pts with news, 65pts without
PHASE1_MAX_SCORE = 73
# Phase 2: max 35pts
PHASE2_MAX_SCORE = 35

# Percentage thresholds
PHASE1_MIN_PCT      = 0.60   # P1 must reach 60% of its effective max
PHASE2_MIN_PCT      = 0.45   # P2 must reach 45% of its effective max
TOTAL_MIN_SCORE_PCT = 0.55   # Total must reach 55% of effective total to TAKE

# Phase 1 sub-score allocations (must sum to PHASE1_MAX_SCORE = 73)
P1_WEEKLY_4H_BIAS_POINTS = 20   # Weekly + 4H EMA stack bias alignment
P1_REGIME_POINTS         = 15   # Regime — ADX + DI context
P1_SESSION_POINTS        = 12   # Session Quality — Kill Zones
P1_1H_TREND_POINTS       = 10   # 1H Trend Confirmation — Supertrend + EMA
P1_15MIN_TRIGGER_POINTS  =  8   # 15min Trigger — Closed Candle
P1_NEWS_POINTS           =  8   # News distance + impact (excluded when unavailable)

# Phase 2 sub-score allocations (must sum to PHASE2_MAX_SCORE = 35)
P2_LIQUIDITY_SWEEP_POINTS = 14  # Liquidity Sweep + reversal
P2_BOS_CHOCH_POINTS       = 11  # BOS / CHoCH confirmation
P2_ORDER_BLOCK_POINTS     =  7  # Active OB + price at it
P2_FVG_POINTS             =  3  # Unmitigated Fair Value Gap


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE HELPERS — all math is here, never duplicated elsewhere
# ═══════════════════════════════════════════════════════════════════════════════

def get_effective_p1_max(news_excluded: bool = False) -> int:
    """Phase 1 max points, minus news component when news is unavailable."""
    return PHASE1_MAX_SCORE - (P1_NEWS_POINTS if news_excluded else 0)


def get_phase1_min_required(news_excluded: bool = False) -> int:
    """Minimum Phase 1 score to proceed (60% of effective P1 max)."""
    return math.ceil(PHASE1_MIN_PCT * get_effective_p1_max(news_excluded))


def get_phase2_min_required() -> int:
    """Minimum Phase 2 score to proceed (35% of P2 max)."""
    return math.ceil(PHASE2_MIN_PCT * PHASE2_MAX_SCORE)


def get_total_max_score(news_excluded: bool = False) -> int:
    """Combined max score (P1 effective + P2)."""
    return get_effective_p1_max(news_excluded) + PHASE2_MAX_SCORE


def get_total_min_required(news_excluded: bool = False) -> int:
    """Minimum total score to TAKE (60% of effective total max)."""
    return math.ceil(TOTAL_MIN_SCORE_PCT * get_total_max_score(news_excluded))


def get_score_pct(total_score: int, news_excluded: bool = False) -> float:
    """Total score as a fraction of the effective max (0.0–1.0)."""
    mx = get_total_max_score(news_excluded)
    return total_score / mx if mx > 0 else 0.0


def get_required_rr(total_score: int, news_excluded: bool = False) -> float:
    """Minimum R:R ratio required for a given total score."""
    score_pct = get_score_pct(total_score, news_excluded)
    for tier in RR_TIERS:
        if tier["min_pct"] <= score_pct < tier["max_pct"]:
            return tier["rr_ratio"]
    return max(t["rr_ratio"] for t in RR_TIERS)


def compute_max_sl_pips(balance: float, pair: str = "EURUSD") -> int:
    """
    Maximum SL in pips affordable at minimum lot.
    Injected into the AI prompt so it doesn't suggest impossible SLs.
    """
    risk_budget = balance * (MAX_LOSS_PER_TRADE_PCT / 100.0)
    pip_value   = PAIR_PIP_VALUES_MICRO.get(pair, PIP_VALUE_PER_MICRO_LOT)
    spread_cost = SPREAD_PIPS * pip_value
    net_budget  = risk_budget - spread_cost
    if net_budget <= 0:
        return MIN_SL_PIPS
    return max(MIN_SL_PIPS, min(int(net_budget / pip_value), MAX_SL_PIPS))


# Backward-compat aliases used by banner / prerequisite checks
PHASE1_MIN_REQUIRED = get_phase1_min_required(False)
PHASE2_MIN_REQUIRED = get_phase2_min_required()


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING PAIRS
# ═══════════════════════════════════════════════════════════════════════════════

PAIRS    = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD"]
SCREENER = "forex"
EXCHANGE = "OANDA"

# JustMarkets appends ".m" to all demo account symbols
MT5_SYMBOL_SUFFIX = os.getenv("MT5_SYMBOL_SUFFIX", ".m")

# Per-pair TradingView screener/exchange overrides
PAIR_SCREENER_OVERRIDES = {
    "XAUUSD": "cfd",    # Gold uses CFD screener on TradingView
}
PAIR_EXCHANGE_OVERRIDES = {
    "XAUUSD": "OANDA",  # Gold via OANDA on TradingView
}

# yfinance symbol mapping (fallback when MT5 library unavailable)
YFINANCE_SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "XAUUSD": "GC=F",   # Gold futures front-month (yfinance fallback)
}

# Currency extraction (for news filtering)
PAIR_CURRENCIES = {
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "AUDUSD": ["AUD", "USD"],
    "USDCAD": ["USD", "CAD"],
    "NZDUSD": ["NZD", "USD"],
    "XAUUSD": ["USD"],          # Gold reacts to USD news (CPI / NFP / FOMC)
}

# Per-pair pip sizes
# XAUUSD: price quoted in USD/oz to 2 decimals (e.g. 3315.15).
#   1 pip = $0.10 price move → pip_size = 0.10
#   At 0.01 lot (100 oz × 0.01 = 1 oz): $0.10/pip × 1 oz = $0.10/pip
PAIR_PIP_SIZES = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "AUDUSD": 0.0001,
    "USDCAD": 0.0001,
    "NZDUSD": 0.0001,
    "XAUUSD": 0.10,     # $0.10 per pip (treat $0.10 move as 1 pip for sizing sanity)
}

# Per-pair pip values at 0.01 lot (micro lot)
# XAUUSD: 1 lot = 100 oz; 1 pip ($0.10) × 100 oz = $10/pip/lot → at 0.01 lot = $0.10/pip
PAIR_PIP_VALUES_MICRO = {
    "EURUSD": 0.10,     # $0.10/pip at 0.01 lot
    "GBPUSD": 0.10,
    "USDJPY": 0.067,
    "AUDUSD": 0.10,
    "USDCAD": 0.073,
    "NZDUSD": 0.10,
    "XAUUSD": 0.10,     # $0.10/pip at 0.01 lot (FIXED — was incorrectly 0.01)
}


# ═══════════════════════════════════════════════════════════════════════════════
# ACCOUNT / RISK
# ═══════════════════════════════════════════════════════════════════════════════

MAX_LOSS_PER_TRADE_PCT = 5.0    # % of balance to risk per trade
DAILY_LOSS_LIMIT_PCT   = 10.0   # % of balance as daily loss hard stop
SPREAD_PIPS            = 2.0    # typical spread (used in lot size math only)
MAX_TRADES_PER_DAY     = 999    # effectively unlimited — daily loss % is the only cap
MAX_TAKE_ATTEMPTS_PER_DAY = 999 # effectively unlimited

# R:R tiers by score % of effective total
# No upper cap on R:R — AI/Entry Engine can propose any R:R above the minimum
RR_TIERS = [
    {"min_pct": 0.55, "max_pct": 0.70, "rr_ratio": 2.5},   # 55-70% → min 1:2.5
    {"min_pct": 0.70, "max_pct": 1.01, "rr_ratio": 2.0},   # >70%   → min 1:2
]

# Default pip / lot values (4-digit forex pairs)
PIP_SIZE                 = 0.0001
PIP_VALUE_PER_MICRO_LOT  = 0.10
MIN_LOT                  = 0.01
MAX_LOT                  = 2.0

# SL computation from swing structure
SL_BUFFER_PIPS  =   3   # pips beyond swing H/L for SL placement
MIN_SL_PIPS     =   5   # minimum sensible SL
MAX_SL_PIPS     = 100   # absolute maximum SL (hard cap)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

ENTRY_SEARCH_RADIUS_PIPS    =  30  # search for levels within 30 pips of price
MARKET_ORDER_THRESHOLD_PIPS =   3  # enter at market if within 3 pips
LIMIT_ORDER_EXPIRY_CANDLES  =   8  # cancel limit after 8 × 15min = 2hrs
TP_MIN_RR                   = 2.0  # TP must be ≥ 2× SL distance


# ═══════════════════════════════════════════════════════════════════════════════
# OVERTRADING PROTECTION
# Daily trade / attempt limits are removed — daily loss % is the safety net.
# ═══════════════════════════════════════════════════════════════════════════════

REJECTION_COOLDOWN_MINUTES    =  30    # wait after a risk rejection
REJECTION_COOLDOWN_ESCALATION = 1.5   # multiply cooldown each consecutive rejection
REJECTION_COOLDOWN_MAX_MINUTES = 180  # max 3 hours cooldown cap
SAME_PAIR_COOLDOWN_MINUTES    =  60   # (kept for reference, not enforced)


# ═══════════════════════════════════════════════════════════════════════════════
# AI / LLM CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Standard OpenAI
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "your-api-key-here")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL    = os.getenv("LLM_MODEL",    "gpt-4o")

# Azure OpenAI / Azure AI Foundry
AZURE_OPENAI_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY",    "")
AZURE_OPENAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT",   "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

LLM_TEMPERATURE   = 0.2
LLM_MAX_TOKENS    = 4000

LLM_MAX_TOKENS_PARAM = os.getenv(
    "LLM_MAX_TOKENS_PARAM",
    "max_completion_tokens" if os.getenv("LLM_PROVIDER", "openai").lower() == "azure"
    else "max_tokens"
)


# ═══════════════════════════════════════════════════════════════════════════════
# MT5 — BOTH HTTP REST AND DIRECT PYTHON LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

# HTTP REST (metatrader-http-server) — used for order execution
MT5_BASE_URL = os.getenv("MT5_BASE_URL", "http://localhost:8001/api/v1")
MT5_TIMEOUT  = 10  # seconds

# Direct MT5 Python library credentials — used for OHLCV data
MT5_LOGIN    = int(os.getenv("MT5_LOGIN",    "0"))
MT5_PASSWORD =     os.getenv("MT5_PASSWORD", "")
MT5_SERVER   =     os.getenv("MT5_SERVER",   "")
MT5_PATH     =     os.getenv("MT5_PATH",     "")   # optional path to terminal64.exe

# Which data source to prefer for OHLCV candles
# "mt5lib"   → MT5 Python library (primary, Windows-only, real-time)
# "yfinance" → yfinance (fallback, cross-platform, may be slightly stale)
MT5_OHLCV_SOURCE = os.getenv("MT5_OHLCV_SOURCE", "mt5lib")


# ═══════════════════════════════════════════════════════════════════════════════
# HARD GATE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

ADX_MIN_THRESHOLD  = 20    # ADX 4H below this → skip
ADX_SKIP_MINUTES   = 15    # retry after 15min when ADX too low
NEWS_DANGER_MINUTES = 30   # block if high-impact news within this many minutes
NEWS_SKIP_MINUTES  = 15    # delay when news is pending
# Spread check is REMOVED — no longer enforced

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION TIMES (UTC hours, 24h format)
# ═══════════════════════════════════════════════════════════════════════════════

LONDON_OPEN  =  7
LONDON_CLOSE = 16
NY_OPEN      = 12
NY_CLOSE     = 21


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULING
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SCAN_INTERVAL_MINUTES = 15


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

MONITOR_ENABLED       = True
MONITOR_POLL_SECONDS  = 30      # check every 30 seconds
TP1_CLOSE_PCT         = 0.50    # close 50% of position at TP1
TP2_PRICE_OFFSET_PCT  = 0.0     # 0 = use the TP from risk engine directly


# ═══════════════════════════════════════════════════════════════════════════════
# MODE
# ═══════════════════════════════════════════════════════════════════════════════

MODE = os.getenv("BOT_MODE", "demo")   # "demo" or "live"


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

LOG_LEVEL   = "INFO"
LOG_FILE    = "logs/bot.log"
TRADES_FILE = "data/trades.json"


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS CALENDAR API
# ═══════════════════════════════════════════════════════════════════════════════

NEWS_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
