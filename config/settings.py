"""
settings.py — Central configuration for the Smart Money Trading Bot.
Every tunable variable lives here. No magic numbers anywhere else.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# SCORING CONFIG — AI scores up to these maxes, checks against these mins
# Change these and the AI prompt adapts automatically.
# ═══════════════════════════════════════════════════════════════════════════════

PHASE1_MAX_SCORE = 60           # total points available in Phase 1
PHASE2_MAX_SCORE = 20           # total points available in Phase 2
PHASE1_MIN_REQUIRED = 35        # minimum Phase 1 score to proceed
PHASE2_MIN_REQUIRED = 12        # minimum Phase 2 score to proceed
TOTAL_MIN_REQUIRED = 60         # minimum combined score to execute

# Phase 1 sub-score allocations (should sum to PHASE1_MAX_SCORE)
P1_REGIME_POINTS = 10           # ADX + DI context
P1_SESSION_POINTS = 8           # kill zone quality
P1_NEWS_POINTS = 8              # news distance + impact
P1_WEEKLY_4H_BIAS_POINTS = 14   # weekly + 4H bias alignment
P1_1H_TREND_POINTS = 12         # 1H trend confirmation
P1_15MIN_TRIGGER_POINTS = 8     # 15min trigger (closed candle)

# Phase 2 sub-score allocations (should sum to PHASE2_MAX_SCORE)
P2_LIQUIDITY_SWEEP_POINTS = 8   # liquidity sweep + reversal
P2_ORDER_BLOCK_POINTS = 5       # active OB + price at it
P2_FVG_POINTS = 4               # unmitigated fair value gap
P2_BOS_CHOCH_POINTS = 3         # BOS / CHoCH confirmation

# ═══════════════════════════════════════════════════════════════════════════════
# TRADING PAIRS
# ═══════════════════════════════════════════════════════════════════════════════

PAIRS = ["BTCUSD"]
SCREENER = "crypto"
EXCHANGE = "BINANCE"

# Per-symbol pip/lot configuration
# pip_size: smallest price increment counted as 1 pip
# pip_value_micro: dollar value per pip at 0.01 lot (micro lot)
# min_lot / max_lot: broker lot bounds
SYMBOL_CONFIG = {
    "BTCUSD": {
        "pip_size": 0.01,
        "pip_value_micro": 0.01,
        "min_lot": 0.01,
        "max_lot": 5.0,
        "screener": "crypto",
        "exchange": "BINANCE",
    },
    "XAUUSD": {
        "pip_size": 0.01,
        "pip_value_micro": 0.01,
        "min_lot": 0.01,
        "max_lot": 5.0,
        "screener": "cfd",
        "exchange": "TVC",
    },
    "USDJPY": {
        "pip_size": 0.01,
        "pip_value_micro": 0.07,
        "min_lot": 0.01,
        "max_lot": 2.0,
        "screener": "forex",
        "exchange": "OANDA",
    },
    "EURUSD": {
        "pip_size": 0.0001,
        "pip_value_micro": 0.10,
        "min_lot": 0.01,
        "max_lot": 2.0,
        "screener": "forex",
        "exchange": "OANDA",
    },
    "GBPUSD": {
        "pip_size": 0.0001,
        "pip_value_micro": 0.10,
        "min_lot": 0.01,
        "max_lot": 2.0,
        "screener": "forex",
        "exchange": "OANDA",
    },
}

# yfinance symbol mapping
YFINANCE_SYMBOLS = {
    "BTCUSD": "BTC-USD",
    "XAUUSD": "GC=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
}

# Currency extraction from pair (for news filtering)
PAIR_CURRENCIES = {
    "BTCUSD": ["BTC", "USD"],
    "XAUUSD": ["XAU", "USD"],
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "AUDUSD": ["AUD", "USD"],
    "USDCAD": ["USD", "CAD"],
    "NZDUSD": ["NZD", "USD"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# ACCOUNT / RISK
# ═══════════════════════════════════════════════════════════════════════════════

# Dynamic risk — 5% of account balance per trade (computed at runtime)
MAX_LOSS_PER_TRADE_PCT = 0.05   # 5% of balance
SPREAD_PIPS = 2.0               # broker typical spread (unified for all pairs)
LEVERAGE = 3000                 # 1:3000
MIN_RR_RATIO = 2.0              # minimum R:R safety floor
MAX_TRADES_PER_DAY = 2
DAILY_LOSS_LIMIT_PCT = 0.15     # 15% of balance daily loss cap

# Legacy fixed fallback (used when MT5 balance unavailable)
MAX_LOSS_PER_TRADE = 2.0
DAILY_LOSS_LIMIT_USD = 6.0

# R:R scaling by score
RR_TIERS = [
    {"min_score": 60, "max_score": 68, "rr_ratio": 3.0},
    {"min_score": 68, "max_score": 80, "rr_ratio": 2.0},
]

# Lot size bounds (global defaults — overridden by SYMBOL_CONFIG per pair)
PIP_SIZE = 0.0001               # default for forex pairs
PIP_VALUE_PER_MICRO_LOT = 0.10  # default for EURUSD/GBPUSD
MIN_LOT = 0.01
MAX_LOT = 2.0

# ═══════════════════════════════════════════════════════════════════════════════
# AI / LLM CONFIG (OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

LLM_API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
LLM_TEMPERATURE = 0.2          # low temp for consistent scoring
LLM_MAX_TOKENS = 2000          # enough for reasoning + JSON

# ═══════════════════════════════════════════════════════════════════════════════
# METATRADER 5 — NATIVE CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════

MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")

# ═══════════════════════════════════════════════════════════════════════════════
# HARD GATE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

ADX_MIN_THRESHOLD = 20          # ADX 4H below this → skip
ADX_SKIP_MINUTES = 30           # how long to skip when ADX too low
NEWS_DANGER_MINUTES = 30        # skip if high-impact news within this many minutes
NEWS_SKIP_MINUTES = 20          # delay when news is pending

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION TIMES (UTC hours, 24h format)
# ═══════════════════════════════════════════════════════════════════════════════

LONDON_OPEN = 7
LONDON_CLOSE = 16
NY_OPEN = 12
NY_CLOSE = 21

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULING
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SCAN_INTERVAL_MINUTES = 15   # fallback scan interval when no AI schedule
MAX_SCHEDULE_MINUTES = 240           # max time AI can schedule ahead (4 hours)

# ═══════════════════════════════════════════════════════════════════════════════
# MODE
# ═══════════════════════════════════════════════════════════════════════════════

MODE = os.getenv("BOT_MODE", "demo")   # "demo" or "live"

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

LOG_LEVEL = "INFO"
LOG_FILE = "logs/bot.log"
TRADES_FILE = "data/trades.json"

# ═══════════════════════════════════════════════════════════════════════════════
# NEWS CALENDAR API
# ═══════════════════════════════════════════════════════════════════════════════

NEWS_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — get symbol-specific config with fallback
# ═══════════════════════════════════════════════════════════════════════════════

def get_symbol_config(symbol: str) -> dict:
    """Get per-symbol config, falling back to global defaults."""
    return SYMBOL_CONFIG.get(symbol, {
        "pip_size": PIP_SIZE,
        "pip_value_micro": PIP_VALUE_PER_MICRO_LOT,
        "min_lot": MIN_LOT,
        "max_lot": MAX_LOT,
        "screener": SCREENER,
        "exchange": EXCHANGE,
    })
