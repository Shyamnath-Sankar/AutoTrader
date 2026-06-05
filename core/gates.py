"""
gates.py — Hard gates that run BEFORE any AI analysis.
Pure Python, no AI. Deterministic pass/fail checks.

Gates:
  1. Session Gate — are we in an active trading session?
  2. ADX Gate    — is the market trending enough? (checks ALL pairs)
  3. News Gate   — any high-impact news too close? (fail-open: if calendar
                   unavailable, passes with a note that news will be excluded
                   from the AI score for affected pairs)
"""

from datetime import datetime, timedelta

import pytz
import requests
from loguru import logger
from tradingview_ta import TA_Handler, Interval

from config import settings
from core.models import GateResult


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SESSION GATE
# ═══════════════════════════════════════════════════════════════════════════════

def check_session_gate() -> GateResult:
    """Check if we are in London or New York session. Reject weekends."""
    now_utc = datetime.now(pytz.UTC)
    hour    = now_utc.hour
    weekday = now_utc.weekday()   # 0=Mon, 6=Sun

    if weekday >= 5:
        days_until_monday = 7 - weekday
        next_monday = now_utc.replace(
            hour=settings.LONDON_OPEN, minute=0, second=0, microsecond=0
        ) + timedelta(days=days_until_monday)
        skip_min = int((next_monday - now_utc).total_seconds() / 60)
        return GateResult(
            gate_name   = "session",
            passed      = False,
            skip_minutes= skip_min,
            reason      = f"Weekend — market closed until Monday {settings.LONDON_OPEN}:00 UTC",
        )

    in_london = settings.LONDON_OPEN <= hour < settings.LONDON_CLOSE
    in_ny     = settings.NY_OPEN     <= hour < settings.NY_CLOSE

    if not (in_london or in_ny):
        if hour < settings.LONDON_OPEN:
            next_open = now_utc.replace(hour=settings.LONDON_OPEN, minute=0, second=0, microsecond=0)
        elif hour >= settings.NY_CLOSE:
            next_open = (now_utc + timedelta(days=1)).replace(
                hour=settings.LONDON_OPEN, minute=0, second=0, microsecond=0
            )
        else:
            next_open = now_utc.replace(hour=settings.NY_OPEN, minute=0, second=0, microsecond=0)

        skip_min = max(1, int((next_open - now_utc).total_seconds() / 60))
        return GateResult(
            gate_name   = "session",
            passed      = False,
            skip_minutes= skip_min,
            reason      = f"Outside trading session (UTC {hour}:00). Next: {next_open.strftime('%H:%M')} UTC",
        )

    sessions = []
    if in_london:
        sessions.append("London")
    if in_ny:
        sessions.append("New York")

    return GateResult(
        gate_name = "session",
        passed    = True,
        reason    = f"Active session: {' + '.join(sessions)} (UTC {hour}:00)",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ADX GATE — checks ALL configured pairs
# ═══════════════════════════════════════════════════════════════════════════════

def check_adx_gate_single(pair: str) -> tuple[bool, float | None, str]:
    """Check ADX on 4H for a single pair. Returns (passed, adx_value, reason)."""
    try:
        handler  = TA_Handler(
            symbol   = pair,
            screener = settings.PAIR_SCREENER_OVERRIDES.get(pair, settings.SCREENER),
            exchange = settings.PAIR_EXCHANGE_OVERRIDES.get(pair, settings.EXCHANGE),
            interval = Interval.INTERVAL_4_HOURS,
        )
        analysis = handler.get_analysis()
        adx      = analysis.indicators.get("ADX")

        if adx is None:
            return False, None, f"ADX unavailable for {pair} 4H — blocking (fail-closed)"

        if adx < settings.ADX_MIN_THRESHOLD:
            return False, adx, f"{pair} ADX 4H = {adx:.1f} < {settings.ADX_MIN_THRESHOLD} (ranging)"

        return True, adx, f"{pair} ADX 4H = {adx:.1f} ≥ {settings.ADX_MIN_THRESHOLD} (trending)"

    except Exception as e:
        logger.error(f"ADX gate error for {pair}: {e}")
        return False, None, f"ADX gate error for {pair} ({e}) — blocking (fail-closed)"


def check_adx_gate(pairs: list[str] | None = None) -> GateResult:
    """
    Pass if ANY pair is trending (ADX ≥ threshold).
    Fail only if ALL pairs are ranging or errored.
    """
    if pairs is None:
        pairs = settings.PAIRS

    results    = []
    any_trending = False

    for pair in pairs:
        passed, adx_val, reason = check_adx_gate_single(pair)
        results.append(reason)
        if passed:
            any_trending = True
            logger.info(f"  ✅ {reason}")
        else:
            logger.info(f"  ⚠️  {reason}")

    if any_trending:
        return GateResult(
            gate_name = "adx",
            passed    = True,
            reason    = f"At least one pair trending: {'; '.join(results)}",
        )

    return GateResult(
        gate_name   = "adx",
        passed      = False,
        skip_minutes= settings.ADX_SKIP_MINUTES,
        reason      = f"No pairs trending: {'; '.join(results)}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NEWS GATE — fail-open when calendar unavailable
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_news_calendar() -> list[dict] | None:
    """Fetch the ForexFactory news calendar. Returns None on failure."""
    try:
        r = requests.get(settings.NEWS_CALENDAR_URL, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"News calendar fetch failed: {e}")
        return None


def check_news_gate(pairs: list[str] | None = None) -> GateResult:
    """
    Block if high-impact news is within NEWS_DANGER_MINUTES for any
    currency in our trading pairs.

    FAIL-OPEN: if the calendar is unavailable, we do NOT block trading.
    Instead we pass the gate — the news component will be EXCLUDED from
    the AI score for each affected pair (handled in data_collector + trader_brain).
    """
    if pairs is None:
        pairs = settings.PAIRS

    currencies = set()
    for pair in pairs:
        for curr in settings.PAIR_CURRENCIES.get(pair, []):
            currencies.add(curr.upper())

    events = _fetch_news_calendar()

    # Calendar unavailable → fail-open (let the AI handle with news excluded)
    if events is None:
        logger.warning(
            "⚠️  News gate: calendar unavailable — passing gate. "
            "News score will be EXCLUDED (not penalised) for each pair."
        )
        return GateResult(
            gate_name = "news",
            passed    = True,
            reason    = "News calendar unavailable — gate passes; news score excluded per-pair.",
        )

    if not events:
        return GateResult(
            gate_name = "news",
            passed    = True,
            reason    = "No events in calendar — safe to trade",
        )

    now_utc    = datetime.now(pytz.UTC)
    cutoff     = now_utc + timedelta(minutes=settings.NEWS_DANGER_MINUTES)
    dangerous  = []

    for event in events:
        event_country = event.get("country", "").upper()
        if not any(curr in event_country for curr in currencies):
            continue
        impact = event.get("impact", "").lower()
        if impact not in ("high", "red"):
            continue
        try:
            event_date_str = event.get("date", "")
            if event_date_str:
                event_time = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
                if not event_time.tzinfo:
                    event_time = pytz.UTC.localize(event_time)
                if now_utc <= event_time <= cutoff:
                    minutes_away = int((event_time - now_utc).total_seconds() / 60)
                    dangerous.append({
                        "title":       event.get("title", "Unknown"),
                        "currency":    event_country,
                        "minutes_away": minutes_away,
                    })
        except (ValueError, TypeError):
            continue

    if dangerous:
        event_list = ", ".join(f"{e['title']} ({e['minutes_away']}min)" for e in dangerous[:3])
        return GateResult(
            gate_name   = "news",
            passed      = False,
            skip_minutes= settings.NEWS_SKIP_MINUTES,
            reason      = f"High-impact news within {settings.NEWS_DANGER_MINUTES}min: {event_list}",
        )

    return GateResult(
        gate_name = "news",
        passed    = True,
        reason    = f"No high-impact news within {settings.NEWS_DANGER_MINUTES} minutes",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL GATES
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_gates(pairs: list[str] | None = None) -> tuple[bool, list[GateResult]]:
    """
    Run all three hard gates in sequence.
    Returns (all_passed, list_of_results).
    """
    if pairs is None:
        pairs = settings.PAIRS

    results = []

    # Gate 1: Session
    session = check_session_gate()
    results.append(session)
    if not session.passed:
        logger.info(f"🚫 Session gate FAILED: {session.reason}")
        return False, results
    logger.info(f"✅ Session gate passed: {session.reason}")

    # Gate 2: ADX (checks ALL pairs)
    adx = check_adx_gate(pairs)
    results.append(adx)
    if not adx.passed:
        logger.info(f"🚫 ADX gate FAILED: {adx.reason}")
        return False, results
    logger.info(f"✅ ADX gate passed")

    # Gate 3: News (fail-open — calendar unavailable passes the gate)
    news = check_news_gate(pairs)
    results.append(news)
    if not news.passed:
        logger.info(f"🚫 News gate FAILED: {news.reason}")
        return False, results
    logger.info(f"✅ News gate passed: {news.reason}")

    logger.info("✅ All gates passed")
    return True, results
