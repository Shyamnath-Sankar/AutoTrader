"""
data_collector.py — Collects all market data into a single payload for the AI.

OHLCV source priority:
  1. MT5 Python library (services/mt5_data.py) — real-time, same machine
  2. yfinance — fallback if MT5 lib is unavailable or returns no data

Other sources:
  3. tradingview-ta   → indicators per timeframe (RSI, EMA, MACD, BB, ADX)
  4. News calendar    → upcoming events (ForexFactory API)
  5. MT5 HTTP client  → account balance, equity, open positions
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
from services import mt5_data as mt5lib          # MT5 Python library OHLCV
from services.mt5_client import MT5Client

# ── SMC library ───────────────────────────────────────────────────────────────
try:
    import io, sys as _sys
    _old_stdout = _sys.stdout
    _sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    from smartmoneyconcepts.smc import smc
    _sys.stdout = _old_stdout
    SMC_AVAILABLE = True
except Exception as e:
    try:
        _sys.stdout = _old_stdout
    except Exception:
        pass
    logger.warning(f"smartmoneyconcepts unavailable ({e}) — SMC analysis will be skipped")
    SMC_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# TIMEFRAME MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

TV_TIMEFRAMES = {
    "weekly": Interval.INTERVAL_1_WEEK,
    "4h":     Interval.INTERVAL_4_HOURS,
    "1h":     Interval.INTERVAL_1_HOUR,
    "15m":    Interval.INTERVAL_15_MINUTES,
}

# yfinance intervals + lookback periods (fallback)
YF_TIMEFRAMES = {
    "1h":  {"interval": "1h",  "period": "1mo"},
    "15m": {"interval": "15m", "period": "5d"},
}

# MT5 lib candle counts
MT5_CANDLE_COUNTS = {
    "1h":  120,   # 5 days of 1H candles
    "15m": 200,   # ~2 days of 15M candles
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TRADINGVIEW-TA INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_indicators(pair: str, timeframe_key: str, tv_interval) -> IndicatorSet | None:
    """Fetch indicator values from TradingView for a single pair + timeframe."""
    try:
        screener = settings.PAIR_SCREENER_OVERRIDES.get(pair, settings.SCREENER)
        exchange  = settings.PAIR_EXCHANGE_OVERRIDES.get(pair, settings.EXCHANGE)
        handler   = TA_Handler(symbol=pair, screener=screener, exchange=exchange, interval=tv_interval)
        analysis  = handler.get_analysis()
        ind       = analysis.indicators

        return IndicatorSet(
            timeframe     = timeframe_key,
            close         = ind.get("close"),
            open          = ind.get("open"),
            high          = ind.get("high"),
            low           = ind.get("low"),
            volume        = ind.get("volume"),
            rsi           = ind.get("RSI"),
            ema20         = ind.get("EMA20"),
            ema50         = ind.get("EMA50"),
            ema200        = ind.get("EMA200"),
            macd          = ind.get("MACD.macd"),
            macd_signal   = ind.get("MACD.signal"),
            macd_hist     = ind.get("MACD.hist"),
            bb_upper      = ind.get("BB.upper"),
            bb_lower      = ind.get("BB.lower"),
            bb_basis      = ind.get("BB.basis"),
            adx           = ind.get("ADX"),
            adx_plus_di   = ind.get("ADX+DI"),
            adx_minus_di  = ind.get("ADX-DI"),
            supertrend    = ind.get("Supertrend", None),
        )
    except Exception as e:
        logger.error(f"tradingview-ta error for {pair} {timeframe_key}: {e}")
        return None


def fetch_all_indicators(pair: str) -> dict[str, IndicatorSet]:
    """Fetch indicators for all configured timeframes."""
    result = {}
    for tf_key, tv_interval in TV_TIMEFRAMES.items():
        ind = fetch_indicators(pair, tf_key, tv_interval)
        if ind:
            result[tf_key] = ind
        else:
            logger.warning(f"Missing indicators for {pair} {tf_key}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OHLCV CANDLES — MT5 lib primary, yfinance fallback
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv_yfinance(pair: str, interval: str, period: str) -> pd.DataFrame | None:
    """Fallback: fetch OHLCV from yfinance."""
    yf_symbol = settings.YFINANCE_SYMBOLS.get(pair, f"{pair}=X")
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(interval=interval, period=period)
        if df.empty:
            logger.warning(f"yfinance returned empty data for {yf_symbol} {interval}")
            return None
        df.columns = [c.lower() for c in df.columns]
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column '{col}' in yfinance data for {yf_symbol}")
                return None
        df = df[required].dropna()
        if df.empty:
            return None
        # Freshness check
        now = pd.Timestamp.now(tz="UTC")
        try:
            last_ts = pd.Timestamp(df.index[-1])
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            staleness = (now - last_ts).total_seconds() / 60
            if staleness > 60 and interval in ("15m", "1h"):
                logger.warning(f"⚠️  yfinance {yf_symbol} {interval} is {staleness:.0f}min stale")
        except Exception:
            pass
        return df
    except Exception as e:
        logger.error(f"yfinance error for {yf_symbol} {interval}: {e}")
        return None


def fetch_ohlcv(pair: str, tf_key: str) -> pd.DataFrame | None:
    """
    Fetch OHLCV candles — tries MT5 Python library first, falls back to yfinance.

    Args:
        pair:   e.g. "EURUSD"
        tf_key: "15m" or "1h"
    """
    # ── Primary: MT5 Python library ─────────────────────────────────────────
    if settings.MT5_OHLCV_SOURCE == "mt5lib":
        count = MT5_CANDLE_COUNTS.get(tf_key, 150)
        df = mt5lib.get_ohlcv(pair, tf_key, count=count)
        if df is not None and len(df) >= 20:
            logger.debug(f"  [{pair} {tf_key}] MT5 lib: {len(df)} candles ✓")
            return df
        logger.warning(f"  [{pair} {tf_key}] MT5 lib returned no data — trying yfinance")

    # ── Fallback: yfinance ───────────────────────────────────────────────────
    yf_config = YF_TIMEFRAMES.get(tf_key)
    if yf_config is None:
        return None
    df = fetch_ohlcv_yfinance(pair, yf_config["interval"], yf_config["period"])
    if df is not None and len(df) >= 20:
        logger.debug(f"  [{pair} {tf_key}] yfinance: {len(df)} candles ✓")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SMC ANALYSIS — with recency tracking
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_smc(pair: str, ohlcv_df: pd.DataFrame, timeframe: str) -> SMCData:
    """Run Smart Money Concepts analysis on OHLCV candle data."""
    smc_data = SMCData(pair=pair, timeframe=timeframe)

    if not SMC_AVAILABLE or ohlcv_df is None or ohlcv_df.empty:
        return smc_data

    total_candles = len(ohlcv_df)

    try:
        swing_df = smc.swing_highs_lows(ohlcv_df, swing_length=10)

        # Swing H/L
        swing_highs = swing_df[swing_df["HighLow"] == 1]
        swing_lows  = swing_df[swing_df["HighLow"] == -1]
        if not swing_highs.empty:
            smc_data.latest_swing_high = float(swing_highs["Level"].iloc[-1])
        if not swing_lows.empty:
            smc_data.latest_swing_low  = float(swing_lows["Level"].iloc[-1])

        # FVG
        try:
            fvg_df = smc.fvg(ohlcv_df, join_consecutive=False)
            if fvg_df is not None and not fvg_df.empty:
                active_fvg = fvg_df[fvg_df["MitigatedIndex"] == 0]
                bull_fvg   = active_fvg[active_fvg["FVG"] ==  1]
                bear_fvg   = active_fvg[active_fvg["FVG"] == -1]
                smc_data.fvg_bullish_count = len(bull_fvg)
                smc_data.fvg_bearish_count = len(bear_fvg)
                if not active_fvg.empty:
                    last = active_fvg.iloc[-1]
                    smc_data.fvg_nearest_type  = "bullish" if last["FVG"] == 1 else "bearish"
                    smc_data.fvg_nearest_level = float((last["Top"] + last["Bottom"]) / 2)
        except Exception as e:
            logger.debug(f"FVG analysis error for {pair} {timeframe}: {e}")

        # Order Blocks
        try:
            ob_df = smc.ob(ohlcv_df, swing_df, close_mitigation=False)
            if ob_df is not None and not ob_df.empty:
                active_ob = ob_df[ob_df["MitigatedIndex"] == 0]
                bull_ob   = active_ob[active_ob["OB"] ==  1]
                bear_ob   = active_ob[active_ob["OB"] == -1]
                smc_data.ob_bullish_count = len(bull_ob)
                smc_data.ob_bearish_count = len(bear_ob)
                if not active_ob.empty:
                    last = active_ob.iloc[-1]
                    smc_data.ob_nearest_type   = "bullish" if last["OB"] == 1 else "bearish"
                    smc_data.ob_nearest_top    = float(last["Top"])
                    smc_data.ob_nearest_bottom = float(last["Bottom"])
        except Exception as e:
            logger.debug(f"OB analysis error for {pair} {timeframe}: {e}")

        # BOS / CHoCH
        try:
            bos_df = smc.bos_choch(ohlcv_df, swing_df)
            if bos_df is not None and not bos_df.empty:
                bos_entries   = bos_df[bos_df["BOS"].notna()   & (bos_df["BOS"]   != 0)]
                choch_entries = bos_df[bos_df["CHOCH"].notna() & (bos_df["CHOCH"] != 0)]
                if not bos_entries.empty:
                    last = bos_entries.iloc[-1]
                    smc_data.last_bos_type  = "bullish" if last["BOS"] == 1 else "bearish"
                    smc_data.last_bos_level = float(last["Level"])
                if not choch_entries.empty:
                    last = choch_entries.iloc[-1]
                    smc_data.last_choch_type  = "bullish" if last["CHOCH"] == 1 else "bearish"
                    smc_data.last_choch_level = float(last["Level"])
        except Exception as e:
            logger.debug(f"BOS/CHoCH analysis error for {pair} {timeframe}: {e}")

        # Liquidity (with recency)
        try:
            liq_df = smc.liquidity(ohlcv_df, swing_df)
            if liq_df is not None and not liq_df.empty:
                swept = liq_df[liq_df["Swept"] > 0]
                if not swept.empty:
                    last = swept.iloc[-1]
                    smc_data.liquidity_swept      = True
                    smc_data.liquidity_sweep_type = "bullish" if last["Liquidity"] == 1 else "bearish"
                    smc_data.liquidity_level      = float(last["Level"])
                    sweep_index = swept.index[-1]
                    try:
                        pos = ohlcv_df.index.get_loc(sweep_index)
                        smc_data.liquidity_candles_ago = total_candles - 1 - pos
                    except Exception:
                        smc_data.liquidity_candles_ago = None
        except Exception as e:
            logger.debug(f"Liquidity analysis error for {pair} {timeframe}: {e}")

        # Retracements
        try:
            ret_df = smc.retracements(ohlcv_df, swing_df)
            if ret_df is not None and not ret_df.empty:
                last = ret_df.iloc[-1]
                curr = last.get("CurrentRetracement%")
                deep = last.get("DeepestRetracement%")
                if curr is not None and not pd.isna(curr):
                    smc_data.current_retracement_pct = float(curr)
                if deep is not None and not pd.isna(deep):
                    smc_data.deepest_retracement_pct = float(deep)
        except Exception as e:
            logger.debug(f"Retracement analysis error for {pair} {timeframe}: {e}")

    except Exception as e:
        logger.error(f"SMC analysis failed for {pair} {timeframe}: {e}")

    return smc_data


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NEWS EVENTS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_news_events(pairs: list[str]) -> tuple[list[NewsEvent], bool]:
    """
    Fetch upcoming news events for currencies in the given pairs.

    Returns:
        (events, fetch_ok):
          - fetch_ok = False → calendar API failed → news component EXCLUDED from score
          - fetch_ok = True  → events list (may be empty = full news points)
    """
    import requests as req

    currencies = set()
    for pair in pairs:
        for curr in settings.PAIR_CURRENCIES.get(pair, []):
            currencies.add(curr.upper())

    try:
        r = req.get(settings.NEWS_CALENDAR_URL, timeout=5)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        logger.warning(f"News calendar fetch failed: {e} — news score will be EXCLUDED for affected pairs")
        return [], False   # fetch FAILED

    now_utc = datetime.now(pytz.UTC)
    cutoff  = now_utc + timedelta(hours=6)

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
                event_time = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
                if not event_time.tzinfo:
                    event_time = pytz.UTC.localize(event_time)
                if now_utc <= event_time <= cutoff:
                    minutes_away = int((event_time - now_utc).total_seconds() / 60)
                    result.append(NewsEvent(
                        title       = event.get("title", "Unknown"),
                        impact      = impact,
                        currency    = event_country,
                        time_utc    = event_time.isoformat(),
                        minutes_away= minutes_away,
                    ))
        except (ValueError, TypeError):
            continue

    return sorted(result, key=lambda x: x.minutes_away), True


# ═══════════════════════════════════════════════════════════════════════════════
# 5. COLLECT EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════════

def collect_all_data(mt5: MT5Client, trade_logger=None) -> tuple[MarketDataPayload, dict]:
    """
    Collect all market data for all configured pairs.
    Returns (MarketDataPayload, raw_ohlcv_dict).
    """
    # Ensure MT5 library is initialized for OHLCV
    if settings.MT5_OHLCV_SOURCE == "mt5lib":
        mt5lib.initialize()

    payload  = MarketDataPayload(timestamp=datetime.now(pytz.UTC).isoformat())
    raw_ohlcv = {}

    # Account info from MT5 HTTP client
    account = mt5.get_account_info()
    if account:
        payload.account_balance = account.balance
        payload.account_equity  = account.equity
        payload.daily_pnl       = account.profit
    else:
        logger.warning("Could not fetch MT5 account info — using defaults")

    payload.open_positions = mt5.get_positions()

    if trade_logger:
        today_trades = trade_logger.get_today_trades()
        payload.trades_today       = today_trades
        payload.trades_today_count = len(today_trades)

    for pair in settings.PAIRS:
        logger.info(f"📊 Collecting data for {pair}...")
        pair_data     = PairMarketData(pair=pair)
        raw_ohlcv[pair] = {}

        # 1. TradingView indicators
        pair_data.indicators = fetch_all_indicators(pair)

        # 2. OHLCV → SMC (MT5 lib primary, yfinance fallback)
        for tf_key in YF_TIMEFRAMES.keys():   # "1h", "15m"
            ohlcv_df = fetch_ohlcv(pair, tf_key)
            if ohlcv_df is not None and len(ohlcv_df) >= 20:
                raw_ohlcv[pair][tf_key] = ohlcv_df
                pair_data.smc[tf_key]   = analyze_smc(pair, ohlcv_df, tf_key)
            else:
                pair_data.smc[tf_key]   = SMCData(pair=pair, timeframe=tf_key)

        # 3. News events
        news_events, news_fetch_ok = fetch_news_events([pair])
        pair_data.news_events    = news_events
        pair_data.news_fetch_ok  = news_fetch_ok
        if not news_fetch_ok:
            logger.warning(f"  ⚠️  News fetch failed for {pair} — news component EXCLUDED from scoring")

        payload.pairs[pair] = pair_data

    logger.info(f"✅ Data collection complete for {len(settings.PAIRS)} pairs")
    return payload, raw_ohlcv


# ═══════════════════════════════════════════════════════════════════════════════
# FORMAT FOR AI PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

def format_data_for_prompt(payload: MarketDataPayload) -> str:
    """Format the MarketDataPayload into a structured text block for the AI."""
    lines = []
    lines.append("═══ MARKET DATA SNAPSHOT ═══")
    lines.append(f"Timestamp: {payload.timestamp}")
    lines.append(f"Account Balance: ${payload.account_balance:.2f}")
    lines.append(f"Account Equity:  ${payload.account_equity:.2f}")
    lines.append(f"Daily P&L:       ${payload.daily_pnl:.2f}")
    lines.append(f"Open Positions:  {len(payload.open_positions)}")
    lines.append(f"Trades Today:    {payload.trades_today_count}")

    if payload.open_positions:
        lines.append("\n── Open Positions ──")
        for pos in payload.open_positions:
            profit = pos.get("profit", 0) or 0
            lines.append(
                f"  {pos.get('symbol','?')} {pos.get('type','?')} "
                f"{pos.get('volume',0)} lots | P&L: ${profit:.2f}"
            )

    if payload.trades_today:
        lines.append("\n── Today's Trade History ──")
        for t in payload.trades_today[-5:]:
            pnl = t.get("pnl", 0) or 0
            lines.append(
                f"  {t.get('pair','?')} {t.get('direction','?')} | "
                f"Score: {t.get('total_score','?')} | "
                f"Decision: {t.get('decision','?')} | "
                f"Risk: {'✅' if t.get('risk_approved') else '❌' if t.get('risk_approved') is False else '—'} | "
                f"Result: {t.get('result','pending')} | P&L: ${pnl:.2f}"
            )

    for pair, pair_data in payload.pairs.items():
        news_excluded = not pair_data.news_fetch_ok

        # Compute effective thresholds for this pair
        p1_eff_max   = settings.get_effective_p1_max(news_excluded)
        p1_min_pts   = settings.get_phase1_min_required(news_excluded)
        p2_min_pts   = settings.get_phase2_min_required()
        total_eff_max = settings.get_total_max_score(news_excluded)
        total_min_pts = settings.get_total_min_required(news_excluded)

        lines.append(f"\n{'═' * 50}")
        lines.append(f"═══ {pair} ═══")
        if news_excluded:
            lines.append(f"  ⚠️  NEWS EXCLUDED — effective max: {total_eff_max}pts")
            lines.append(f"  Thresholds → P1 min {p1_min_pts}/{p1_eff_max} | P2 min {p2_min_pts}/20 | Total min {total_min_pts}/{total_eff_max}")
        else:
            lines.append(f"  Thresholds → P1 min {p1_min_pts}/60 | P2 min {p2_min_pts}/20 | Total min {total_min_pts}/80")
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
                lines.append(f"  Nearest OB: {smc_d.ob_nearest_type} [{smc_d.ob_nearest_bottom} – {smc_d.ob_nearest_top}]")
            if smc_d.last_bos_type:
                lines.append(f"  Last BOS: {smc_d.last_bos_type} @ {smc_d.last_bos_level}")
            if smc_d.last_choch_type:
                lines.append(f"  Last CHoCH: {smc_d.last_choch_type} @ {smc_d.last_choch_level}")
            lines.append(f"  Liquidity Swept: {smc_d.liquidity_swept}")
            if smc_d.liquidity_swept:
                recency = f" ({smc_d.liquidity_candles_ago} candles ago)" if smc_d.liquidity_candles_ago is not None else ""
                lines.append(f"  Sweep Type: {smc_d.liquidity_sweep_type} @ {smc_d.liquidity_level}{recency}")
            if smc_d.current_retracement_pct:
                lines.append(f"  Current Retracement: {smc_d.current_retracement_pct:.1f}%")
            if smc_d.deepest_retracement_pct:
                lines.append(f"  Deepest Retracement: {smc_d.deepest_retracement_pct:.1f}%")

        # News section — per-pair with effective threshold guidance
        lines.append(f"\n── {pair} News ──")
        if news_excluded:
            lines.append(
                f"  ⚠️  NEWS COMPONENT EXCLUDED: calendar fetch failed.\n"
                f"  Set news = 0 in your JSON for {pair}.\n"
                f"  The {settings.P1_NEWS_POINTS}pts news component is EXCLUDED from P1 max for this pair.\n"
                f"  Effective P1 max = {p1_eff_max}pts | P1 min = {p1_min_pts}pts | "
                f"Total min = {total_min_pts}/{total_eff_max}pts"
            )
        elif pair_data.news_events:
            lines.append(f"  Upcoming news (next 6h):")
            for event in pair_data.news_events[:5]:
                lines.append(
                    f"  [{event.impact}] {event.title} | {event.currency} | {event.minutes_away}min away"
                )
        else:
            lines.append(
                f"  ✅ No high/medium-impact news in next 6 hours — "
                f"score news at FULL {settings.P1_NEWS_POINTS} pts."
            )

    return "\n".join(lines)


def format_ohlcv_for_entry(pair: str, raw_ohlcv: dict, n_15m: int = 30, n_1h: int = 15) -> str:
    """Format raw OHLCV candle data as a readable table for the Entry Engine AI."""
    lines = []
    pair_ohlcv = raw_ohlcv.get(pair, {})

    for tf_key, label, n_candles in [("15m", "15M", n_15m), ("1h", "1H", n_1h)]:
        df = pair_ohlcv.get(tf_key)
        if df is None or df.empty:
            lines.append(f"\n── {pair} {label} Candles: NO DATA ──")
            continue

        recent = df.tail(n_candles)
        lines.append(f"\n── {pair} {label} Recent Candles (last {len(recent)}) ──")
        lines.append(f"  {'Time':<20} | {'Open':>10} | {'High':>10} | {'Low':>10} | {'Close':>10} | {'Volume':>10}")
        lines.append(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        for idx, row in recent.iterrows():
            ts = str(idx)
            if hasattr(idx, "strftime"):
                ts = idx.strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"  {ts:<20} | {row['open']:>10.5f} | {row['high']:>10.5f} | "
                f"{row['low']:>10.5f} | {row['close']:>10.5f} | {row.get('volume', 0):>10.0f}"
            )

    return "\n".join(lines)
