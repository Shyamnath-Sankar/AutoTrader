"""
gates.py — Hard gates that run BEFORE any AI analysis.
Pure Python, no AI. Deterministic pass/fail checks.

Gates:
  1. Session Gate — are we in an active trading session?
  2. ADX Gate    — is the market trending enough?
  3. News Gate   — any high-impact news too close?
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
    """
    Check if we are in an active trading session (London or New York).
    Weekends are also rejected.
    """
    now_utc = datetime.now(pytz.UTC)
    hour = now_utc.hour
    weekday = now_utc.weekday()  # 0=Mon, 6=Sun

    # Reject weekends
    if weekday >= 5:
        # Calculate minutes until Monday 07:00 UTC
        days_until_monday = 7 - weekday
        next_monday = now_utc.replace(
            hour=settings.LONDON_OPEN, minute=0, second=0, microsecond=0
        ) + timedelta(days=days_until_monday)
        skip_min = int((next_monday - now_utc).total_seconds() / 60)
        return GateResult(
            gate_name="session",
            passed=False,
            skip_minutes=skip_min,
            reason=f"Weekend — market closed until Monday {settings.LONDON_OPEN}:00 UTC",
        )

    # Check if within London or NY session
    in_london = settings.LONDON_OPEN <= hour < settings.LONDON_CLOSE
    in_ny = settings.NY_OPEN <= hour < settings.NY_CLOSE

    if not (in_london or in_ny):
        # Find next session open
        if hour < settings.LONDON_OPEN:
            # Before London open today
            next_open = now_utc.replace(
                hour=settings.LONDON_OPEN, minute=0, second=0, microsecond=0
            )
        elif hour >= settings.NY_CLOSE:
            # After NY close — next session is tomorrow London
            next_open = (now_utc + timedelta(days=1)).replace(
                hour=settings.LONDON_OPEN, minute=0, second=0, microsecond=0
            )
        else:
            # Between London close and NY open (shouldn't happen with overlap)
            next_open = now_utc.replace(
                hour=settings.NY_OPEN, minute=0, second=0, microsecond=0
            )

        skip_min = max(1, int((next_open - now_utc).total_seconds() / 60))
        return GateResult(
            gate_name="session",
            passed=False,
            skip_minutes=skip_min,
            reason=f"Outside trading session (UTC {hour}:00). Next session at {next_open.strftime('%H:%M')} UTC",
        )

    session_name = []
    if in_london:
        session_name.append("London")
    if in_ny:
        session_name.append("New York")
    session_str = " + ".join(session_name)

    return GateResult(
        gate_name="session",
        passed=True,
        reason=f"Active session: {session_str} (UTC {hour}:00)",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ADX GATE
# ═══════════════════════════════════════════════════════════════════════════════

def check_adx_gate(pair: str = "EURUSD") -> GateResult:
    """
    Check if ADX on 4H is above the minimum threshold.
    If ADX < ADX_MIN_THRESHOLD, market is not trending enough.
    """
    try:
        handler = TA_Handler(
            symbol=pair,
            screener=settings.SCREENER,
            exchange=settings.EXCHANGE,
            interval=Interval.INTERVAL_4_HOURS,
        )
        analysis = handler.get_analysis()
        adx = analysis.indicators.get("ADX")

        if adx is None:
            logger.warning(f"ADX value is None for {pair} 4H — passing gate by default")
            return GateResult(
                gate_name="adx",
                passed=True,
                reason=f"ADX unavailable for {pair} 4H — passing by default",
            )

        if adx < settings.ADX_MIN_THRESHOLD:
            return GateResult(
                gate_name="adx",
                passed=False,
                skip_minutes=settings.ADX_SKIP_MINUTES,
                reason=f"ADX 4H too low: {adx:.1f} < {settings.ADX_MIN_THRESHOLD} (market not trending)",
            )

        return GateResult(
            gate_name="adx",
            passed=True,
            reason=f"ADX 4H = {adx:.1f} ≥ {settings.ADX_MIN_THRESHOLD} — market trending",
        )

    except Exception as e:
        logger.error(f"ADX gate error for {pair}: {e}")
        # On error, pass the gate to avoid blocking the system
        return GateResult(
            gate_name="adx",
            passed=True,
            reason=f"ADX gate error ({e}) — passing by default",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NEWS GATE
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_news_calendar() -> list[dict]:
    """Fetch the ForexFactory news calendar for this week."""
    try:
        r = requests.get(settings.NEWS_CALENDAR_URL, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"News calendar fetch failed: {e} — treating as no news")
        return []


def check_news_gate(pairs: list[str] | None = None) -> GateResult:
    """
    Check if any high-impact news events are within NEWS_DANGER_MINUTES
    for the currencies in our trading pairs.
    """
    if pairs is None:
        pairs = settings.PAIRS

    # Collect all currencies we care about
    currencies = set()
    for pair in pairs:
        for curr in settings.PAIR_CURRENCIES.get(pair, []):
            currencies.add(curr.upper())

    events = _fetch_news_calendar()
    if not events:
        return GateResult(
            gate_name="news",
            passed=True,
            reason="No news calendar data available — passing",
        )

    now_utc = datetime.now(pytz.UTC)
    cutoff = now_utc + timedelta(minutes=settings.NEWS_DANGER_MINUTES)

    dangerous_events = []
    for event in events:
        # Match currency
        event_country = event.get("country", "").upper()
        if not any(curr in event_country for curr in currencies):
            continue

        # Only care about high-impact
        impact = event.get("impact", "").lower()
        if impact not in ("high", "red"):
            continue

        # Parse event time
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
                    dangerous_events.append({
                        "title": event.get("title", "Unknown"),
                        "impact": "High",
                        "currency": event_country,
                        "minutes_away": minutes_away,
                    })
        except (ValueError, TypeError):
            continue

    if dangerous_events:
        closest = dangerous_events[0]
        event_list = ", ".join(
            f"{e['title']} ({e['minutes_away']}min)" for e in dangerous_events[:3]
        )
        return GateResult(
            gate_name="news",
            passed=False,
            skip_minutes=settings.NEWS_SKIP_MINUTES,
            reason=f"High-impact news within {settings.NEWS_DANGER_MINUTES}min: {event_list}",
        )

    return GateResult(
        gate_name="news",
        passed=True,
        reason=f"No high-impact news within {settings.NEWS_DANGER_MINUTES} minutes",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL GATES
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_gates(pairs: list[str] | None = None) -> tuple[bool, list[GateResult]]:
    """
    Run all three hard gates in sequence.
    Returns (all_passed, list_of_results).
    If any gate fails, the first failure's skip_minutes is used.
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

    # Gate 2: ADX (check first pair as proxy)
    adx = check_adx_gate(pairs[0])
    results.append(adx)
    if not adx.passed:
        logger.info(f"🚫 ADX gate FAILED: {adx.reason}")
        return False, results

    # Gate 3: News
    news = check_news_gate(pairs)
    results.append(news)
    if not news.passed:
        logger.info(f"🚫 News gate FAILED: {news.reason}")
        return False, results

    logger.info("✅ All gates passed")
    return True, results
