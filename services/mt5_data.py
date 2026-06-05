"""
mt5_data.py — MT5 Python Library OHLCV Data Client

Uses the official MetaTrader5 Python package to fetch candle data
directly from the running MT5 terminal on Windows.

Falls back gracefully if:
  - MetaTrader5 package is not installed
  - MT5 terminal is not running
  - Symbol not found (broker-specific suffix issues)

In those cases, data_collector.py falls back to yfinance.
"""

import os
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from config import settings

# ── Try importing the MT5 library ────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    _MT5_LIB_IMPORTED = True
except ImportError:
    mt5 = None  # type: ignore
    _MT5_LIB_IMPORTED = False
    logger.warning("MetaTrader5 package not installed — OHLCV will use yfinance. "
                   "Install with: pip install MetaTrader5")

# State flag — set after successful initialize()
_MT5_INITIALIZED = False

# MT5 timeframe mapping
_TF_MAP = None  # built lazily after import check


def _build_tf_map() -> dict:
    """Build the timeframe mapping after confirming mt5 is available."""
    if mt5 is None:
        return {}
    return {
        "15m": mt5.TIMEFRAME_M15,
        "1h":  mt5.TIMEFRAME_H1,
        "4h":  mt5.TIMEFRAME_H4,
        "1d":  mt5.TIMEFRAME_D1,
        "1w":  mt5.TIMEFRAME_W1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def initialize() -> bool:
    """
    Initialize the MT5 Python library connection.

    Tries credential-based login first (from .env MT5_LOGIN / MT5_PASSWORD /
    MT5_SERVER). If credentials are missing, attempts to connect to the
    already-logged-in terminal (requires terminal to be running).

    Returns True if connected, False otherwise.
    """
    global _MT5_INITIALIZED, _TF_MAP

    if not _MT5_LIB_IMPORTED or mt5 is None:
        return False

    # Build timeframe map on first call
    if _TF_MAP is None:
        _TF_MAP = _build_tf_map()

    if _MT5_INITIALIZED:
        return True

    try:
        login    = settings.MT5_LOGIN
        password = settings.MT5_PASSWORD
        server   = settings.MT5_SERVER
        path     = settings.MT5_PATH or None  # None = let MT5 auto-detect

        if login and password and server:
            logger.info(f"🔌 MT5 lib: connecting as {login} @ {server}...")
            kwargs = dict(login=login, password=password, server=server)
            if path:
                kwargs["path"] = path
            ok = mt5.initialize(**kwargs)
        else:
            logger.info("🔌 MT5 lib: connecting to already-running terminal...")
            ok = mt5.initialize(path=path) if path else mt5.initialize()

        if ok:
            info = mt5.account_info()
            if info:
                logger.info(f"✅ MT5 lib connected — Account: {info.login} | "
                            f"Balance: ${info.balance:.2f} | Server: {info.server}")
            else:
                logger.info("✅ MT5 lib connected (account info unavailable)")
            _MT5_INITIALIZED = True
            return True
        else:
            err = mt5.last_error()
            logger.warning(f"⚠️  MT5 lib init failed: {err} — OHLCV will use yfinance")
            return False

    except Exception as e:
        logger.warning(f"⚠️  MT5 lib exception during init: {e} — OHLCV will use yfinance")
        return False


def shutdown():
    """Gracefully shut down the MT5 connection."""
    global _MT5_INITIALIZED
    if _MT5_INITIALIZED and mt5 is not None:
        mt5.shutdown()
        _MT5_INITIALIZED = False
        logger.info("MT5 lib connection closed")


def is_connected() -> bool:
    """Return True if MT5 library is initialized and terminal is reachable."""
    return _MT5_INITIALIZED and mt5 is not None


# ═══════════════════════════════════════════════════════════════════════════════
# OHLCV DATA
# ═══════════════════════════════════════════════════════════════════════════════

def get_ohlcv(pair: str, timeframe: str, count: int = 150) -> pd.DataFrame | None:
    """
    Fetch OHLCV candle data from MT5 Python library.

    Args:
        pair:      pair name without suffix, e.g. "EURUSD"
        timeframe: "15m", "1h", "4h", "1d", "1w"
        count:     number of candles to fetch (from most recent)

    Returns:
        DataFrame with columns [open, high, low, close, volume] indexed by UTC datetime,
        or None if unavailable.
    """
    if not is_connected():
        if not initialize():
            return None

    global _TF_MAP
    if _TF_MAP is None:
        _TF_MAP = _build_tf_map()

    tf = _TF_MAP.get(timeframe)
    if tf is None:
        logger.warning(f"MT5 lib: unknown timeframe '{timeframe}'")
        return None

    # Append broker suffix (e.g. ".m" for JustMarkets demo)
    symbol = pair + settings.MT5_SYMBOL_SUFFIX

    try:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            # Try without suffix if first attempt fails
            if settings.MT5_SYMBOL_SUFFIX:
                rates = mt5.copy_rates_from_pos(pair, tf, 0, count)
                if rates is None or len(rates) == 0:
                    logger.warning(f"MT5 lib: no data for {symbol} or {pair} {timeframe}: {err}")
                    return None
                logger.debug(f"MT5 lib: fetched {pair} (no suffix) {timeframe}")
            else:
                logger.warning(f"MT5 lib: no data for {symbol} {timeframe}: {err}")
                return None

        df = pd.DataFrame(rates)

        # Convert time from UNIX timestamp to UTC datetime
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)

        # Standardise column names
        df.rename(columns={
            "open":        "open",
            "high":        "high",
            "low":         "low",
            "close":       "close",
            "tick_volume": "volume",
            "real_volume": "volume",
        }, inplace=True)

        # Keep only required columns
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].dropna()

        if df.empty:
            logger.warning(f"MT5 lib: empty DataFrame after processing {pair} {timeframe}")
            return None

        # Freshness check
        last_ts = df.index[-1]
        now = pd.Timestamp.now(tz="UTC")
        staleness_min = (now - last_ts).total_seconds() / 60
        if staleness_min > 60 and timeframe in ("15m", "1h"):
            logger.warning(f"⚠️  MT5 lib: {pair} {timeframe} data is {staleness_min:.0f}min stale")

        logger.debug(f"MT5 lib: {pair} {timeframe} → {len(df)} candles (last: {last_ts})")
        return df

    except Exception as e:
        logger.error(f"MT5 lib error fetching {pair} {timeframe}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ACCOUNT INFO (via Python lib — mirrors HTTP client API)
# ═══════════════════════════════════════════════════════════════════════════════

def get_account_balance() -> float | None:
    """Return current account balance from MT5 lib, or None."""
    if not is_connected():
        return None
    try:
        info = mt5.account_info()
        return info.balance if info else None
    except Exception:
        return None
