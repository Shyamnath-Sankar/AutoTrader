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
PHASE1_MIN_REQUIRED = 32        # minimum Phase 1 score to proceed
PHASE2_MIN_REQUIRED = 10        # minimum Phase 2 score to proceed
TOTAL_MIN_SCORE_PCT = 0.60      # minimum total score as % of max to TAKE (60%)

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

PAIRS = ["EURUSD", "GBPUSD"]
SCREENER = "forex"
EXCHANGE = "OANDA"

# yfinance symbol mapping (EURUSD → EURUSD=X)
YFINANCE_SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
}

# Currency extraction from pair (for news filtering)
PAIR_CURRENCIES = {
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "AUDUSD": ["AUD", "USD"],
    "USDCAD": ["USD", "CAD"],
    "NZDUSD": ["NZD", "USD"],
}

# Per-pair pip sizes — critical for JPY pairs (0.01) vs others (0.0001)
PAIR_PIP_SIZES = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "AUDUSD": 0.0001,
    "USDCAD": 0.0001,
    "NZDUSD": 0.0001,
}

# Per-pair pip values at 0.01 lot (micro lot)
# EURUSD/GBPUSD: $0.10/pip, USDJPY: ~$0.067/pip (varies with rate)
PAIR_PIP_VALUES_MICRO = {
    "EURUSD": 0.10,
    "GBPUSD": 0.10,
    "USDJPY": 0.067,
    "AUDUSD": 0.10,
    "USDCAD": 0.073,
    "NZDUSD": 0.10,
}

# ═══════════════════════════════════════════════════════════════════════════════
# ACCOUNT / RISK
# ═══════════════════════════════════════════════════════════════════════════════

MAX_LOSS_PER_TRADE_PCT = 5.0    # max % of balance to risk per trade
DAILY_LOSS_LIMIT_PCT = 10.0     # max % of balance allowed as daily loss
SPREAD_PIPS = 2.0               # broker typical spread (used as fallback)
MAX_TRADES_PER_DAY = 2

# R:R scaling by score percentage (lower conviction → demand higher R:R)
# min_pct is inclusive, max_pct is exclusive (except the last tier's 1.01 catches 100%)
RR_TIERS = [
    {"min_pct": 0.60, "max_pct": 0.70, "rr_ratio": 3.0},
    {"min_pct": 0.70, "max_pct": 1.01, "rr_ratio": 2.0},
]

# Pip values (for lot size calculation) — defaults for EUR/GBP pairs
PIP_SIZE = 0.0001               # default for 4-digit pairs
PIP_VALUE_PER_MICRO_LOT = 0.10  # $0.10 per pip for 0.01 lot on EURUSD/GBPUSD
MIN_LOT = 0.01
MAX_LOT = 2.0

# SL computation from swing structure
SL_BUFFER_PIPS = 3              # pips beyond swing H/L for SL placement
MIN_SL_PIPS = 5                 # minimum sensible SL
MAX_SL_PIPS = 100               # absolute maximum SL (hard cap)

# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

ENTRY_SEARCH_RADIUS_PIPS = 30       # search for levels within 30 pips of price
MARKET_ORDER_THRESHOLD_PIPS = 3     # enter at market if within 3 pips
LIMIT_ORDER_EXPIRY_CANDLES = 8      # cancel limit order after 8 × 15min = 2hrs
TP_MIN_RR = 2.0                     # TP must be at least 2× SL distance

# ═══════════════════════════════════════════════════════════════════════════════
# OVERTRADING PROTECTION
# ═══════════════════════════════════════════════════════════════════════════════

MAX_TAKE_ATTEMPTS_PER_DAY = 5   # cap total TAKE attempts (rejected + approved)
REJECTION_COOLDOWN_MINUTES = 30         # wait after a TAKE gets rejected
REJECTION_COOLDOWN_ESCALATION = 1.5     # multiply cooldown each consecutive rejection
REJECTION_COOLDOWN_MAX_MINUTES = 180    # max cooldown cap (3 hours)
SAME_PAIR_COOLDOWN_MINUTES = 60         # wait before re-attempting same pair after rejection

# ═══════════════════════════════════════════════════════════════════════════════
# AI / LLM CONFIG (OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

LLM_API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = 0.2          # low temp for consistent scoring
LLM_MAX_TOKENS = 2000          # enough for reasoning + JSON

# ═══════════════════════════════════════════════════════════════════════════════
# MT5 HTTP REST API (metatrader-mcp-server)
# ═══════════════════════════════════════════════════════════════════════════════

MT5_BASE_URL = os.getenv("MT5_BASE_URL", "http://localhost:8001/api/v1")
MT5_TIMEOUT = 10                # seconds

# ═══════════════════════════════════════════════════════════════════════════════
# HARD GATE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

ADX_MIN_THRESHOLD = 20          # ADX 4H below this → skip
ADX_SKIP_MINUTES = 30           # how long to skip when ADX too low
NEWS_DANGER_MINUTES = 30        # skip if high-impact news within this many minutes
NEWS_SKIP_MINUTES = 20          # delay when news is pending
SPREAD_MAX_PIPS = 5.0           # reject if live spread exceeds this

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

DEFAULT_SCAN_INTERVAL_MINUTES = 15   # fallback scan interval

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
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_total_max_score() -> int:
    """Return the combined maximum score across both phases."""
    return PHASE1_MAX_SCORE + PHASE2_MAX_SCORE


def get_total_min_required() -> int:
    """
    Compute the minimum total score required to TAKE, derived from the
    percentage threshold and the current max score.

    Returns:
        integer threshold (ceiling of pct * max)
    """
    import math
    return math.ceil(TOTAL_MIN_SCORE_PCT * get_total_max_score())


def get_score_pct(total_score: int) -> float:
    """
    Compute the score as a fraction of the maximum possible score.

    Args:
        total_score: combined phase 1 + phase 2 score

    Returns:
        float in [0.0, 1.0]
    """
    max_score = get_total_max_score()
    if max_score <= 0:
        return 0.0
    return total_score / max_score


def get_required_rr(total_score: int) -> float:
    """
    Determine the minimum R:R ratio required for a given total score.

    Walks the RR_TIERS list (percentage-based) and returns the matching
    tier's rr_ratio. Falls back to the most conservative tier if no
    match is found (should not happen when TAKE threshold is enforced).

    Args:
        total_score: combined phase 1 + phase 2 score

    Returns:
        required risk:reward ratio (e.g. 3.0 means 1:3)
    """
    score_pct = get_score_pct(total_score)
    for tier in RR_TIERS:
        if tier["min_pct"] <= score_pct < tier["max_pct"]:
            return tier["rr_ratio"]
    # Fallback: return the most conservative (highest) R:R requirement
    return max(t["rr_ratio"] for t in RR_TIERS)


def compute_max_sl_pips(balance: float, pair: str = "EURUSD") -> int:
    """
    Compute the maximum SL in pips that the account can afford at minimum lot.
    This is injected into the AI prompt so it doesn't suggest impossible SLs.

    Formula: max_sl = (balance * risk_pct / 100 - spread_cost) / pip_value_micro
    """
    risk_budget = balance * (MAX_LOSS_PER_TRADE_PCT / 100.0)
    pip_value = PAIR_PIP_VALUES_MICRO.get(pair, PIP_VALUE_PER_MICRO_LOT)
    spread_cost = SPREAD_PIPS * pip_value
    net_budget = risk_budget - spread_cost
    if net_budget <= 0:
        return MIN_SL_PIPS
    max_sl = int(net_budget / pip_value)
    return max(MIN_SL_PIPS, min(max_sl, MAX_SL_PIPS))
