"""
data_collector.py — Collects all market data into a single payload for the AI.

Sources:
  1. tradingview-ta   → indicators per timeframe (RSI, EMA, MACD, BB, ADX, etc.)
  2. yfinance         → OHLCV candle arrays (input for SMC library)
  3. smartmoneyconcepts → FVG, OB, BOS/ChoCH, Liquidity, Swing H/L, Retracements
  4. News calendar    → upcoming events for relevant currencies
  5. MT5 client       → account balance, equity, open positions, daily P&L
  6. Trade logger     → today's trades for context
"""

from datetime import datetime, timedelta

import pandas as pd
import pytz
import yfinance as yf
from loguru import logger
from tradingview_ta import TA_Handler, Interval

from config import settings
from core.models import (
    IndicatorSet,
    MarketDataPayload,
    NewsEvent,
    PairMarketData,
    SMCData,
)
from services.mt5_client import MT5Client

# Try importing SMC library — it may fail on some systems
try:
    from smartmoneyconcepts.smc import smc
    SMC_AVAILABLE = True
except ImportError:
    logger.warning("smartmoneyconcepts not installed — SMC analysis will be skipped")
    SMC_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# TIMEFRAME MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

TV_TIMEFRAMES = {
    "weekly":  Interval.INTERVAL_1_WEEK,
    "4h":      Interval.INTERVAL_4_HOURS,
    "1h":      Interval.INTERVAL_1_HOUR,
    "15m":     Interval.INTERVAL_15_MINUTES,
}

# yfinance intervals + lookback periods for OHLCV candles
YF_TIMEFRAMES = {
    "1h":  {"interval": "1h",  "period": "1mo"},
    "15m": {"interval": "15m", "period": "5d"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TRADINGVIEW-TA INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_indicators(pair: str, timeframe_key: str, tv_interval) -> IndicatorSet | None:
    """Fetch indicator values from TradingView for a single pair + timeframe."""
    # Use per-symbol screener/exchange from SYMBOL_CONFIG
    sym_config = settings.get_symbol_config(pair)
    screener = sym_config.get("screener", settings.SCREENER)
    exchange = sym_config.get("exchange", settings.EXCHANGE)

    try:
        handler = TA_Handler(
            symbol=pair,
            screener=screener,
            exchange=exchange,
            interval=tv_interval,
        )
        analysis = handler.get_analysis()
        ind = analysis.indicators

        return IndicatorSet(
            timeframe=timeframe_key,
            close=ind.get("close"),
            open=ind.get("open"),
            high=ind.get("high"),
            low=ind.get("low"),
            volume=ind.get("volume"),
            rsi=ind.get("RSI"),
            ema20=ind.get("EMA20"),
            ema50=ind.get("EMA50"),
            ema200=ind.get("EMA200"),
            macd=ind.get("MACD.macd"),
            macd_signal=ind.get("MACD.signal"),
            macd_hist=ind.get("MACD.hist"),
            bb_upper=ind.get("BB.upper"),
            bb_lower=ind.get("BB.lower"),
            bb_basis=ind.get("BB.basis"),
            adx=ind.get("ADX"),
            adx_plus_di=ind.get("ADX+DI"),
            adx_minus_di=ind.get("ADX-DI"),
            supertrend=ind.get("Supertrend", None),
        )
    except Exception as e:
        logger.error(f"tradingview-ta error for {pair} {timeframe_key}: {e}")
        return None


def fetch_all_indicators(pair: str) -> dict[str, IndicatorSet]:
    """Fetch indicators for all configured timeframes."""
    result = {}
    for tf_key, tv_interval in TV_TIMEFRAMES.items():
        indicator_set = fetch_indicators(pair, tf_key, tv_interval)
        if indicator_set:
            result[tf_key] = indicator_set
        else:
            logger.warning(f"Missing indicators for {pair} {tf_key}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OHLCV CANDLES (yfinance — for SMC library)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(pair: str, interval: str, period: str) -> pd.DataFrame | None:
    """
    Fetch OHLCV candle data from yfinance.
    pair: "EURUSD" → converted to "EURUSD=X", "XAUUSD" → "GC=F"
    """
    yf_symbol = settings.YFINANCE_SYMBOLS.get(pair, f"{pair}=X")
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(interval=interval, period=period)
        if df.empty:
            logger.warning(f"yfinance returned empty data for {yf_symbol} {interval}")
            return None
        df.columns = [c.lower() for c in df.columns]
        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column '{col}' in yfinance data for {yf_symbol}")
                return None
        df = df[required].dropna()
        return df
    except Exception as e:
        logger.error(f"yfinance error for {yf_symbol} {interval}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SMC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_smc(pair: str, ohlcv_df: pd.DataFrame, timeframe: str) -> SMCData:
    """
    Run Smart Money Concepts analysis on OHLCV candle data.
    Uses the smartmoneyconcepts library.
    """
    smc_data = SMCData(pair=pair, timeframe=timeframe)

    if not SMC_AVAILABLE or ohlcv_df is None or ohlcv_df.empty:
        return smc_data

    try:
        # Swing highs and lows (needed by other functions)
        swing_df = smc.swing_highs_lows(ohlcv_df, swing_length=10)

        # --- Swing H/L ---
        swing_highs = swing_df[swing_df["HighLow"] == 1]
        swing_lows = swing_df[swing_df["HighLow"] == -1]
        if not swing_highs.empty:
            smc_data.latest_swing_high = float(swing_highs["Level"].iloc[-1])
        if not swing_lows.empty:
            smc_data.latest_swing_low = float(swing_lows["Level"].iloc[-1])

        # --- Fair Value Gaps ---
        try:
            fvg_df = smc.fvg(ohlcv_df, join_consecutive=False)
            if fvg_df is not None and not fvg_df.empty:
                # Active (unmitigated) FVGs
                active_fvg = fvg_df[fvg_df["MitigatedIndex"] == 0]
                bull_fvg = active_fvg[active_fvg["FVG"] == 1]
                bear_fvg = active_fvg[active_fvg["FVG"] == -1]
                smc_data.fvg_bullish_count = len(bull_fvg)
                smc_data.fvg_bearish_count = len(bear_fvg)

                # Nearest active FVG
                if not active_fvg.empty:
                    last_fvg = active_fvg.iloc[-1]
                    smc_data.fvg_nearest_type = "bullish" if last_fvg["FVG"] == 1 else "bearish"
                    smc_data.fvg_nearest_level = float((last_fvg["Top"] + last_fvg["Bottom"]) / 2)
        except Exception as e:
            logger.debug(f"FVG analysis error for {pair} {timeframe}: {e}")

        # --- Order Blocks ---
        try:
            ob_df = smc.ob(ohlcv_df, swing_df, close_mitigation=False)
            if ob_df is not None and not ob_df.empty:
                active_ob = ob_df[ob_df["MitigatedIndex"] == 0]
                bull_ob = active_ob[active_ob["OB"] == 1]
                bear_ob = active_ob[active_ob["OB"] == -1]
                smc_data.ob_bullish_count = len(bull_ob)
                smc_data.ob_bearish_count = len(bear_ob)

                if not active_ob.empty:
                    last_ob = active_ob.iloc[-1]
                    smc_data.ob_nearest_type = "bullish" if last_ob["OB"] == 1 else "bearish"
                    smc_data.ob_nearest_top = float(last_ob["Top"])
                    smc_data.ob_nearest_bottom = float(last_ob["Bottom"])
        except Exception as e:
            logger.debug(f"OB analysis error for {pair} {timeframe}: {e}")

        # --- BOS / CHoCH ---
        try:
            bos_df = smc.bos_choch(ohlcv_df, swing_df)
            if bos_df is not None and not bos_df.empty:
                # Last BOS
                bos_entries = bos_df[bos_df["BOS"].notna() & (bos_df["BOS"] != 0)]
                if not bos_entries.empty:
                    last_bos = bos_entries.iloc[-1]
                    smc_data.last_bos_type = "bullish" if last_bos["BOS"] == 1 else "bearish"
                    smc_data.last_bos_level = float(last_bos["Level"])

                # Last CHoCH
                choch_entries = bos_df[bos_df["CHOCH"].notna() & (bos_df["CHOCH"] != 0)]
                if not choch_entries.empty:
                    last_choch = choch_entries.iloc[-1]
                    smc_data.last_choch_type = "bullish" if last_choch["CHOCH"] == 1 else "bearish"
                    smc_data.last_choch_level = float(last_choch["Level"])
        except Exception as e:
            logger.debug(f"BOS/CHoCH analysis error for {pair} {timeframe}: {e}")

        # --- Liquidity ---
        try:
            liq_df = smc.liquidity(ohlcv_df, swing_df)
            if liq_df is not None and not liq_df.empty:
                swept = liq_df[liq_df["Swept"] > 0]
                if not swept.empty:
                    last_swept = swept.iloc[-1]
                    smc_data.liquidity_swept = True
                    smc_data.liquidity_sweep_type = "bullish" if last_swept["Liquidity"] == 1 else "bearish"
                    smc_data.liquidity_level = float(last_swept["Level"])
        except Exception as e:
            logger.debug(f"Liquidity analysis error for {pair} {timeframe}: {e}")

        # --- Retracements ---
        try:
            ret_df = smc.retracements(ohlcv_df, swing_df)
            if ret_df is not None and not ret_df.empty:
                last_ret = ret_df.iloc[-1]
                curr_ret = last_ret.get("CurrentRetracement%")
                deep_ret = last_ret.get("DeepestRetracement%")
                if curr_ret is not None and not pd.isna(curr_ret):
                    smc_data.current_retracement_pct = float(curr_ret)
                if deep_ret is not None and not pd.isna(deep_ret):
                    smc_data.deepest_retracement_pct = float(deep_ret)
        except Exception as e:
            logger.debug(f"Retracement analysis error for {pair} {timeframe}: {e}")

    except Exception as e:
        logger.error(f"SMC analysis failed for {pair} {timeframe}: {e}")

    return smc_data


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NEWS EVENTS (for context — gate already checked separately)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_news_events(pairs: list[str]) -> list[NewsEvent]:
    """Fetch upcoming news events for currencies in our pairs (for AI context)."""
    import requests as req

    currencies = set()
    for pair in pairs:
        for curr in settings.PAIR_CURRENCIES.get(pair, []):
            currencies.add(curr.upper())

    try:
        r = req.get(settings.NEWS_CALENDAR_URL, timeout=5)
        r.raise_for_status()
        events = r.json()
    except Exception:
        return []

    now_utc = datetime.now(pytz.UTC)
    cutoff = now_utc + timedelta(hours=6)  # look 6 hours ahead for context

    result = []
    for event in events:
        event_country = event.get("country", "").upper()
        if not any(curr in event_country for curr in currencies):
            continue

        impact = event.get("impact", "").capitalize()
        if impact not in ("High", "Medium"):
            continue

        try:
            event_date_str = event.get("date", "")
            if event_date_str:
                event_time = datetime.fromisoformat(
                    event_date_str.replace("Z", "+00:00")
                )
                if not event_time.tzinfo:
                    event_time = pytz.UTC.localize(event_time)

                if now_utc <= event_time <= cutoff:
                    minutes_away = int((event_time - now_utc).total_seconds() / 60)
                    result.append(NewsEvent(
                        title=event.get("title", "Unknown"),
                        impact=impact,
                        currency=event_country,
                        time_utc=event_time.isoformat(),
                        minutes_away=minutes_away,
                    ))
        except (ValueError, TypeError):
            continue

    return sorted(result, key=lambda x: x.minutes_away)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PER-PAIR DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def collect_pair_data(pair: str) -> PairMarketData:
    """
    Collect all market data for a single pair.
    Designed for concurrent execution via ThreadPoolExecutor.
    """
    logger.info(f"📊 Collecting data for {pair}...")
    pair_data = PairMarketData(pair=pair)

    # 1. TradingView indicators (per-symbol screener/exchange)
    pair_data.indicators = fetch_all_indicators(pair)

    # 2. SMC analysis (on 1H and 15min candles)
    for tf_key, yf_config in YF_TIMEFRAMES.items():
        ohlcv_df = fetch_ohlcv(pair, yf_config["interval"], yf_config["period"])
        if ohlcv_df is not None and len(ohlcv_df) >= 20:
            pair_data.smc[tf_key] = analyze_smc(pair, ohlcv_df, tf_key)
        else:
            pair_data.smc[tf_key] = SMCData(pair=pair, timeframe=tf_key)

    # 3. News events for this pair's currencies
    pair_data.news_events = fetch_news_events([pair])

    logger.info(f"✅ Data collection complete for {pair}")
    return pair_data


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COLLECT EVERYTHING (ALL PAIRS)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_all_data(mt5: MT5Client, trade_logger=None) -> MarketDataPayload:
    """
    Collect all market data for all configured pairs and assemble
    into a single MarketDataPayload for the AI.
    """
    payload = MarketDataPayload(
        timestamp=datetime.now(pytz.UTC).isoformat(),
    )

    # --- Account info from MT5 ---
    account = mt5.get_account_info()
    if account:
        payload.account_balance = account.balance
        payload.account_equity = account.equity
        payload.daily_pnl = account.profit
    else:
        logger.warning("Could not fetch MT5 account info — using defaults")

    # --- Open positions ---
    payload.open_positions = mt5.get_positions()

    # --- Today's trades from log ---
    if trade_logger:
        today_trades = trade_logger.get_today_trades()
        payload.trades_today = today_trades
        payload.trades_today_count = len(today_trades)

    # --- Per-pair data ---
    for pair in settings.PAIRS:
        pair_data = collect_pair_data(pair)
        payload.pairs[pair] = pair_data

    logger.info(f"✅ Data collection complete for {len(settings.PAIRS)} pairs")
    return payload


def collect_single_pair_data(
    pair: str,
    mt5: MT5Client,
    trade_logger=None,
) -> MarketDataPayload:
    """
    Collect market data for a SINGLE pair + account context.
    Used by the concurrent per-pair analysis pipeline.
    """
    payload = MarketDataPayload(
        timestamp=datetime.now(pytz.UTC).isoformat(),
    )

    # Account info
    account = mt5.get_account_info()
    if account:
        payload.account_balance = account.balance
        payload.account_equity = account.equity
        payload.daily_pnl = account.profit
    else:
        logger.warning("Could not fetch MT5 account info — using defaults")

    payload.open_positions = mt5.get_positions()

    if trade_logger:
        today_trades = trade_logger.get_today_trades()
        payload.trades_today = today_trades
        payload.trades_today_count = len(today_trades)

    # Only collect data for this one pair
    pair_data = collect_pair_data(pair)
    payload.pairs[pair] = pair_data

    return payload


def format_data_for_prompt(payload: MarketDataPayload) -> str:
    """
    Format the MarketDataPayload into a structured text block
    that the AI can read as context in the prompt.
    """
    lines = []
    lines.append("═══ MARKET DATA SNAPSHOT ═══")
    lines.append(f"Timestamp: {payload.timestamp}")
    lines.append(f"Account Balance: ${payload.account_balance:.2f}")
    lines.append(f"Account Equity: ${payload.account_equity:.2f}")
    lines.append(f"Daily P&L: ${payload.daily_pnl:.2f}")
    lines.append(f"Open Positions: {len(payload.open_positions)}")
    lines.append(f"Trades Today: {payload.trades_today_count}")

    if payload.open_positions:
        lines.append("\n── Open Positions ──")
        for pos in payload.open_positions:
            lines.append(
                f"  {pos.get('symbol', '?')} {pos.get('type', '?')} "
                f"{pos.get('volume', 0)} lots | P&L: ${pos.get('profit', 0):.2f}"
            )

    if payload.trades_today:
        lines.append("\n── Today's Trade History ──")
        for t in payload.trades_today[-5:]:  # last 5 trades
            lines.append(
                f"  {t.get('pair', '?')} {t.get('direction', '?')} | "
                f"Score: {t.get('total_score', '?')} | "
                f"Result: {t.get('result', 'pending')} | "
                f"P&L: ${t.get('pnl', 0):.2f}"
            )

    for pair, pair_data in payload.pairs.items():
        lines.append(f"\n{'═' * 50}")
        lines.append(f"═══ {pair} ═══")
        lines.append(f"{'═' * 50}")

        # Indicators per timeframe
        for tf_key, ind in pair_data.indicators.items():
            lines.append(f"\n── {pair} {tf_key.upper()} Indicators ──")
            lines.append(f"  Close: {ind.close}  |  High: {ind.high}  |  Low: {ind.low}")
            lines.append(f"  RSI: {ind.rsi:.1f}" if ind.rsi else "  RSI: N/A")
            lines.append(f"  EMA20: {ind.ema20}  |  EMA50: {ind.ema50}  |  EMA200: {ind.ema200}")
            lines.append(f"  MACD: {ind.macd}  |  Signal: {ind.macd_signal}  |  Hist: {ind.macd_hist}")
            lines.append(f"  BB Upper: {ind.bb_upper}  |  Basis: {ind.bb_basis}  |  Lower: {ind.bb_lower}")
            lines.append(f"  ADX: {ind.adx}  |  +DI: {ind.adx_plus_di}  |  -DI: {ind.adx_minus_di}")
            if ind.supertrend:
                lines.append(f"  Supertrend: {ind.supertrend}")

        # SMC data per timeframe
        for tf_key, smc_d in pair_data.smc.items():
            lines.append(f"\n── {pair} {tf_key.upper()} Smart Money Concepts ──")
            lines.append(f"  Swing High: {smc_d.latest_swing_high}  |  Swing Low: {smc_d.latest_swing_low}")
            lines.append(f"  FVG Bullish: {smc_d.fvg_bullish_count}  |  FVG Bearish: {smc_d.fvg_bearish_count}")
            if smc_d.fvg_nearest_level:
                lines.append(f"  Nearest FVG: {smc_d.fvg_nearest_type} @ {smc_d.fvg_nearest_level}")
            lines.append(f"  OB Bullish: {smc_d.ob_bullish_count}  |  OB Bearish: {smc_d.ob_bearish_count}")
            if smc_d.ob_nearest_top:
                lines.append(
                    f"  Nearest OB: {smc_d.ob_nearest_type} "
                    f"[{smc_d.ob_nearest_bottom} - {smc_d.ob_nearest_top}]"
                )
            if smc_d.last_bos_type:
                lines.append(f"  Last BOS: {smc_d.last_bos_type} @ {smc_d.last_bos_level}")
            if smc_d.last_choch_type:
                lines.append(f"  Last CHoCH: {smc_d.last_choch_type} @ {smc_d.last_choch_level}")
            lines.append(f"  Liquidity Swept: {smc_d.liquidity_swept}")
            if smc_d.liquidity_swept:
                lines.append(
                    f"  Sweep Type: {smc_d.liquidity_sweep_type} @ {smc_d.liquidity_level}"
                )
            if smc_d.current_retracement_pct:
                lines.append(f"  Current Retracement: {smc_d.current_retracement_pct:.1f}%")
            if smc_d.deepest_retracement_pct:
                lines.append(f"  Deepest Retracement: {smc_d.deepest_retracement_pct:.1f}%")

        # News
        if pair_data.news_events:
            lines.append(f"\n── {pair} Upcoming News ──")
            for event in pair_data.news_events[:5]:
                lines.append(
                    f"  [{event.impact}] {event.title} | "
                    f"{event.currency} | {event.minutes_away}min away"
                )

    return "\n".join(lines)
