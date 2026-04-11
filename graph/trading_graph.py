"""
trading_graph.py — LangGraph StateGraph for the trading pipeline.

Defines the full pipeline as a graph:
  Gates → Data Collection → AI Brain → Risk Engine → MT5 Execution

Each node reads/writes to the shared TradingState.
Conditional edges handle branching (gate fail, skip, watch, reject).
"""

from loguru import logger

from config import settings
from core.gates import run_pair_gates
from services.data_collector import collect_single_pair_data
from agents.trader_brain import analyze_pair
from agents.risk_engine import evaluate
from services.trade_logger import TradeLogger
from services.mt5_client import MT5Client
from graph.state import TradingState

from langgraph.graph import StateGraph, END


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED SERVICES (injected at build time)
# ═══════════════════════════════════════════════════════════════════════════════

_mt5: MT5Client | None = None
_trade_log: TradeLogger | None = None


def set_services(mt5: MT5Client, trade_log: TradeLogger):
    """Inject shared services into the graph module (called once at startup)."""
    global _mt5, _trade_log
    _mt5 = mt5
    _trade_log = trade_log


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1: GATES
# ═══════════════════════════════════════════════════════════════════════════════

def gates_node(state: TradingState) -> dict:
    """Run per-pair hard gates (session, ADX, news)."""
    pair = state["pair"]
    logger.info(f"\n📋 [{pair}] Running gates...")

    try:
        gates_passed, gate_results = run_pair_gates(pair)

        if not gates_passed:
            failed_gate = next((g for g in gate_results if not g.passed), None)
            reason = failed_gate.reason if failed_gate else "Gate failed"
            logger.info(f"🚫 [{pair}] Gate failed: {reason}")
            return {
                "gates_passed": False,
                "gate_reason": reason,
                "status": "GATE_FAILED",
                "error": reason,
            }

        return {
            "gates_passed": True,
            "gate_reason": "All gates passed",
        }

    except Exception as e:
        logger.error(f"❌ [{pair}] Gates error: {e}")
        return {
            "gates_passed": False,
            "gate_reason": str(e),
            "status": "ERROR",
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2: COLLECT DATA
# ═══════════════════════════════════════════════════════════════════════════════

def collect_data_node(state: TradingState) -> dict:
    """Collect market data for the pair."""
    pair = state["pair"]
    logger.info(f"📊 [{pair}] Collecting market data...")

    try:
        market_data = collect_single_pair_data(pair, _mt5, _trade_log)
        return {"market_data": market_data}

    except Exception as e:
        logger.error(f"❌ [{pair}] Data collection error: {e}")
        return {
            "status": "ERROR",
            "error": f"Data collection failed: {e}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3: AI TRADER BRAIN
# ═══════════════════════════════════════════════════════════════════════════════

def brain_node(state: TradingState) -> dict:
    """Run the AI Trader Brain analysis via LangChain."""
    pair = state["pair"]
    market_data = state.get("market_data")

    if not market_data:
        return {
            "status": "ERROR",
            "error": "No market data available for brain",
        }

    logger.info(f"🧠 [{pair}] AI Trader Brain analyzing...")

    try:
        decision = analyze_pair(market_data, pair)

        if decision.decision == "SKIP":
            logger.info(f"🔴 [{pair}] Decision: SKIP — {decision.reasoning[:100]}")
            _trade_log.log_decision(decision)
            return {
                "decision": decision,
                "status": "SKIP",
            }

        if decision.decision == "WATCH":
            logger.info(
                f"🟡 [{pair}] Decision: WATCH — check in {decision.next_check_minutes} min"
            )
            _trade_log.log_decision(decision)
            return {
                "decision": decision,
                "status": "WATCH",
            }

        if decision.decision == "EXECUTE":
            logger.info(f"🟢 [{pair}] Decision: EXECUTE — running risk engine...")
            return {
                "decision": decision,
                "status": "PENDING_RISK",
            }

        # Unknown decision
        return {
            "decision": decision,
            "status": "SKIP",
        }

    except Exception as e:
        logger.error(f"❌ [{pair}] Brain error: {e}")
        return {
            "status": "ERROR",
            "error": f"Brain analysis failed: {e}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4: RISK ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def risk_node(state: TradingState) -> dict:
    """Run the Risk Engine to validate the EXECUTE decision."""
    pair = state["pair"]
    decision = state.get("decision")
    balance = state.get("balance", 0.0)
    today_stats = state.get("today_stats", {})

    if not decision:
        return {
            "status": "ERROR",
            "error": "No decision available for risk engine",
        }

    try:
        risk_result = evaluate(
            decision=decision,
            mt5=_mt5,
            trades_today_count=today_stats.get("executed_count", 0),
            daily_pnl=today_stats.get("total_pnl", 0.0),
            balance=balance,
        )

        if not risk_result.approved:
            logger.info(f"🚫 [{pair}] Risk REJECTED: {risk_result.reason}")
            _trade_log.log_decision(decision, risk=risk_result)
            return {
                "risk_result": risk_result,
                "status": "REJECTED",
            }

        logger.info(f"✅ [{pair}] Risk APPROVED — executing on MT5...")
        return {
            "risk_result": risk_result,
            "status": "PENDING_EXECUTE",
        }

    except Exception as e:
        logger.error(f"❌ [{pair}] Risk engine error: {e}")
        return {
            "status": "ERROR",
            "error": f"Risk engine failed: {e}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 5: EXECUTE ON MT5
# ═══════════════════════════════════════════════════════════════════════════════

def execute_node(state: TradingState) -> dict:
    """Execute the trade on MT5."""
    pair = state["pair"]
    decision = state.get("decision")
    risk_result = state.get("risk_result")

    if not decision or not risk_result:
        return {
            "status": "ERROR",
            "error": "Missing decision or risk_result for execution",
        }

    if settings.MODE == "demo":
        logger.info(f"🏷️  [{pair}] MODE: DEMO — placing trade on demo account")

    try:
        order_result = _mt5.place_market_order(
            symbol=decision.pair,
            direction=decision.direction,
            volume=risk_result.lots,
            stop_loss=risk_result.sl_price,
            take_profit=risk_result.tp_price,
        )

        if order_result.success:
            logger.info(
                f"🎯 [{pair}] Trade EXECUTED! Ticket #{order_result.order} | "
                f"{decision.direction} {risk_result.lots} lots {pair} | "
                f"Entry: {order_result.price} | SL: {risk_result.sl_price} | "
                f"TP: {risk_result.tp_price}"
            )
            _trade_log.log_decision(decision, risk=risk_result, order=order_result)
            return {
                "order_result": order_result,
                "status": "EXECUTE",
            }
        else:
            logger.error(
                f"❌ [{pair}] Trade FAILED: {order_result.message} "
                f"(code: {order_result.error_code})"
            )
            _trade_log.log_decision(decision, risk=risk_result, order=order_result)
            return {
                "order_result": order_result,
                "status": "TRADE_FAILED",
                "error": order_result.message,
            }

    except Exception as e:
        logger.error(f"❌ [{pair}] Execution error: {e}")
        return {
            "status": "ERROR",
            "error": f"MT5 execution failed: {e}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGES (routing logic)
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_gates(state: TradingState) -> str:
    """Route after gates: proceed to data collection or end."""
    if state.get("gates_passed"):
        return "collect_data"
    return END


def route_after_brain(state: TradingState) -> str:
    """Route after brain: proceed to risk engine or end."""
    status = state.get("status", "")
    if status == "PENDING_RISK":
        return "risk"
    return END


def route_after_risk(state: TradingState) -> str:
    """Route after risk: proceed to execution or end."""
    status = state.get("status", "")
    if status == "PENDING_EXECUTE":
        return "execute"
    return END


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD THE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def build_trading_graph() -> StateGraph:
    """
    Build and compile the LangGraph trading pipeline.

    Graph flow:
        START → gates → [pass?] → collect_data → brain → [EXECUTE?] → risk → [approved?] → execute → END
                         ↓ fail                          ↓ SKIP/WATCH            ↓ rejected
                         END                             END                     END

    Returns a compiled graph that can be invoked with `graph.invoke(initial_state)`.
    """
    graph = StateGraph(TradingState)

    # Add nodes
    graph.add_node("gates", gates_node)
    graph.add_node("collect_data", collect_data_node)
    graph.add_node("brain", brain_node)
    graph.add_node("risk", risk_node)
    graph.add_node("execute", execute_node)

    # Set entry point
    graph.set_entry_point("gates")

    # Add conditional edges
    graph.add_conditional_edges("gates", route_after_gates, {
        "collect_data": "collect_data",
        END: END,
    })
    graph.add_edge("collect_data", "brain")
    graph.add_conditional_edges("brain", route_after_brain, {
        "risk": "risk",
        END: END,
    })
    graph.add_conditional_edges("risk", route_after_risk, {
        "execute": "execute",
        END: END,
    })
    graph.add_edge("execute", END)

    return graph.compile()
