"""
dryrun.py — One-shot pipeline test. Run before VPS deployment.
"""
import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from config import settings
from core.gates import run_all_gates
from services.data_collector import collect_all_data
from agents.trader_brain import analyze
from services.mt5_client import MT5Client
from services.trade_logger import TradeLogger
import services.mt5_data as mt5_lib

SEP = "=" * 62

def banner(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

banner("STEP 1 — Imports")
print("  All modules imported OK")

banner("STEP 2 — MT5 Python lib (OHLCV)")
ok = mt5_lib.initialize()
print(f"  Connected: {ok}")
if not ok:
    print("  Fallback: yfinance will be used for OHLCV")

banner("STEP 3 — MT5 HTTP (orders)")
mt5 = MT5Client()
connected = mt5.is_connected()
print(f"  Connected: {connected}")
if connected:
    acc = mt5.get_account_info()
    if acc:
        bal = round(acc.balance, 2)
        eq  = round(acc.equity, 2)
        print(f"  Login: {acc.login}  Balance: {bal}  Equity: {eq}")
        risk_usd = round(bal * settings.MAX_LOSS_PER_TRADE_PCT / 100, 2)
        print(f"  Max risk/trade: {risk_usd} ({settings.MAX_LOSS_PER_TRADE_PCT}%)")
else:
    print("  WARNING: MT5 HTTP not reachable — trades cannot be placed")

banner("STEP 4 — Hard Gates")
passed, results = run_all_gates(settings.PAIRS)
for g in results:
    tag = "PASS" if g.passed else "FAIL"
    print(f"  [{tag}] {g.gate_name}: {g.reason[:80]}")
print(f"\n  Gates passed: {passed}")

banner("STEP 5 — Scoring Config")
print(f"  Phase 1 max (news incl): {settings.PHASE1_MAX_SCORE}  min: {settings.get_phase1_min_required()} ({int(settings.PHASE1_MIN_PCT*100)}%)")
print(f"  Phase 1 max (news excl): {settings.get_effective_p1_max(True)}  min: {settings.get_phase1_min_required(True)} ({int(settings.PHASE1_MIN_PCT*100)}%)")
print(f"  Phase 2 max: {settings.PHASE2_MAX_SCORE}  min: {settings.get_phase2_min_required()} ({int(settings.PHASE2_MIN_PCT*100)}%)")
print(f"  Total (news incl): max {settings.get_total_max_score()}  min {settings.get_total_min_required()} ({int(settings.TOTAL_MIN_SCORE_PCT*100)}%)")
print(f"  Total (news excl): max {settings.get_total_max_score(True)}  min {settings.get_total_min_required(True)} ({int(settings.TOTAL_MIN_SCORE_PCT*100)}%)")
for t in settings.RR_TIERS:
    lo = int(t["min_pct"] * 100)
    hi = int(t["max_pct"] * 100) if t["max_pct"] < 1.01 else 100
    print(f"  R:R {lo}%-{hi}% -> min 1:{t['rr_ratio']}")

banner("STEP 6 — Data Collection (30-90s)")
trade_log = TradeLogger()
try:
    market_data, raw_ohlcv = collect_all_data(mt5, trade_log)
    bal = round(market_data.account_balance, 2)
    print(f"  Balance: {bal}  Open positions: {len(market_data.open_positions)}")
    for pair, pd in market_data.pairs.items():
        smc_tfs = list(pd.smc.keys())
        ind_tfs = list(pd.indicators.keys())
        n_events = len(pd.news_events)
        news_ok  = pd.news_fetch_ok
        def _candle_count(df):
            if df is None: return 0
            try: return len(df)
            except: return 0

        ohlcv_15m = _candle_count(raw_ohlcv.get(pair, {}).get("15m"))
        ohlcv_1h  = _candle_count(raw_ohlcv.get(pair, {}).get("1h"))
        print(
            f"  {pair}: "
            f"indicators={ind_tfs}  smc={smc_tfs}  "
            f"15m-candles={ohlcv_15m}  1h-candles={ohlcv_1h}  "
            f"news_ok={news_ok}  events={n_events}"
        )
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

banner("STEP 7 — AI Trader Brain")
try:
    decision = analyze(market_data)
    news_tag = " [news excluded]" if decision.news_excluded else ""
    eff_max  = settings.get_total_max_score(decision.news_excluded)
    score_pct = settings.get_score_pct(decision.total_score, decision.news_excluded)
    p1 = decision.phase1_scores
    p2 = decision.phase2_scores

    print(f"  Decision:  {decision.decision}{news_tag}")
    print(f"  Pair:      {decision.pair}   Direction: {decision.direction}")
    print()
    print(f"  Phase 1:  {decision.phase1_total}/{settings.get_effective_p1_max(decision.news_excluded)}")
    print(f"    Bias={p1.weekly_4h_bias}/20  Regime={p1.regime}/15  Session={p1.session}/12")
    print(f"    Trend={p1.trend_1h}/10  Trigger={p1.trigger_15m}/8  News={p1.news}/8")
    print()
    print(f"  Phase 2:  {decision.phase2_total}/{settings.PHASE2_MAX_SCORE}")
    print(f"    Sweep={p2.liquidity_sweep}/14  BOS={p2.bos_choch}/11  OB={p2.order_block}/7  FVG={p2.fvg}/3")
    print()
    print(f"  Total: {decision.total_score}/{eff_max}  ({score_pct*100:.1f}%)")
    print()
    print(f"  Reasoning: {decision.reasoning[:400]}")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print()
print(SEP)
print("  DRY RUN COMPLETE")
print(SEP)
if decision.decision == "TAKE":
    rr_req = settings.get_required_rr(decision.total_score, decision.news_excluded)
    print(f"  RESULT: TAKE {decision.direction} {decision.pair}")
    print(f"  Required min R:R: 1:{rr_req}  (no max cap)")
else:
    print("  RESULT: LEAVE — no qualifying setup at this moment")
print()
print("  Pipeline is working correctly. Safe to deploy on VPS.")
print(SEP)
