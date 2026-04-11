"""
mt5_client.py — Native MetaTrader 5 Python client.

Connects directly to MT5 terminal using login/password/server credentials.
No HTTP bridge needed — uses the official MetaTrader5 Python package.
"""

import time
from datetime import datetime, timedelta

import pytz
from loguru import logger

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not installed — MT5 features disabled")

from config import settings
from core.models import AccountInfo, PriceData, OrderResult


class MT5Client:
    """Native MetaTrader 5 client using the official Python library."""

    def __init__(self):
        self._connected = False
        self._connect()

    # ── connection ───────────────────────────────────────────────────────────

    def _connect(self) -> bool:
        """Initialize MT5 terminal and log in with credentials."""
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 Python package is not installed")
            return False

        # Initialize the MT5 terminal
        init_ok = mt5.initialize(
            path=settings.MT5_PATH,
            login=settings.MT5_LOGIN,
            password=settings.MT5_PASSWORD,
            server=settings.MT5_SERVER,
        )

        if not init_ok:
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            self._connected = False
            return False

        # Verify login
        account = mt5.account_info()
        if account is None:
            logger.error("MT5 account_info() returned None after initialization")
            self._connected = False
            return False

        self._connected = True
        logger.info(
            f"✅ MT5 connected — Login: {account.login} | "
            f"Server: {account.server} | "
            f"Balance: ${account.balance:.2f} | "
            f"Leverage: 1:{account.leverage}"
        )
        return True

    def reconnect(self) -> bool:
        """Attempt to reconnect to MT5."""
        logger.info("🔄 Reconnecting to MT5...")
        self.shutdown()
        time.sleep(2)
        return self._connect()

    def shutdown(self):
        """Gracefully shut down the MT5 connection."""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 connection closed")

    # ── account ──────────────────────────────────────────────────────────────

    def get_account_info(self) -> AccountInfo | None:
        """Get account balance, equity, margin, etc."""
        if not self._ensure_connected():
            return None
        try:
            info = mt5.account_info()
            if info is None:
                return None
            return AccountInfo(
                balance=info.balance,
                equity=info.equity,
                margin=info.margin,
                free_margin=info.margin_free,
                margin_level=info.margin_level if info.margin_level else 0.0,
                profit=info.profit,
                currency=info.currency,
                leverage=info.leverage,
                login=info.login,
                server=info.server,
            )
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return None

    # ── market data ──────────────────────────────────────────────────────────

    def get_price(self, symbol: str) -> PriceData | None:
        """Get live bid/ask/spread for a symbol."""
        if not self._ensure_connected():
            return None

        # Ensure the symbol is enabled in Market Watch
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"Failed to select symbol {symbol} in Market Watch")

        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"No tick data for {symbol}")
                return None

            # Get pip size for this symbol
            sym_config = settings.get_symbol_config(symbol)
            pip_size = sym_config["pip_size"]

            spread_raw = tick.ask - tick.bid
            spread_pips = spread_raw / pip_size if pip_size > 0 else 0

            return PriceData(
                symbol=symbol,
                bid=tick.bid,
                ask=tick.ask,
                spread=spread_raw,
                spread_pips=round(spread_pips, 1),
            )
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    def get_symbols(self) -> list | None:
        """List all available symbols."""
        if not self._ensure_connected():
            return None
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                return []
            return [s.name for s in symbols]
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return None

    def get_symbol_info(self, symbol: str) -> dict | None:
        """Get detailed symbol info (for dynamic pip/lot values)."""
        if not self._ensure_connected():
            return None
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None
            return {
                "name": info.name,
                "point": info.point,
                "digits": info.digits,
                "trade_tick_size": info.trade_tick_size,
                "trade_tick_value": info.trade_tick_value,
                "volume_min": info.volume_min,
                "volume_max": info.volume_max,
                "volume_step": info.volume_step,
                "spread": info.spread,
            }
        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None

    # ── orders ───────────────────────────────────────────────────────────────

    def place_market_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        stop_loss: float,
        take_profit: float,
    ) -> OrderResult:
        """
        Place a market BUY or SELL order.

        Args:
            symbol:      "EURUSD", "XAUUSD", etc.
            direction:   "BUY" or "SELL"
            volume:      lot size (e.g. 0.01)
            stop_loss:   price level (not pips)
            take_profit: price level (not pips)
        """
        if not self._ensure_connected():
            return OrderResult(success=False, message="MT5 not connected")

        # Ensure symbol is enabled
        if not mt5.symbol_select(symbol, True):
            return OrderResult(success=False, message=f"Symbol {symbol} not available")

        # Get current price for filling
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(success=False, message=f"No tick data for {symbol}")

        if direction.upper() == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        # Build the order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,  # max slippage in points
            "magic": 123456,  # magic number to identify bot orders
            "comment": "SmartMoney Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        logger.info(
            f"Placing {direction} {volume} lots {symbol} @ {price} | "
            f"SL={stop_loss} TP={take_profit}"
        )

        result = mt5.order_send(request)

        if result is None:
            error = mt5.last_error()
            return OrderResult(
                success=False,
                message=f"order_send failed: {error}",
            )

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(
                success=False,
                order=result.order if result.order else None,
                message=f"Order rejected: {result.comment}",
                symbol=symbol,
                type=direction,
                volume=volume,
                price=result.price if result.price else 0.0,
                stop_loss=stop_loss,
                take_profit=take_profit,
                error_code=result.retcode,
            )

        return OrderResult(
            success=True,
            order=result.order,
            message="Order filled",
            symbol=symbol,
            type=direction,
            volume=result.volume,
            price=result.price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    # ── positions ────────────────────────────────────────────────────────────

    def get_positions(self) -> list[dict]:
        """Get all currently open positions."""
        if not self._ensure_connected():
            return []
        try:
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                return []
            return [
                {
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == 0 else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "magic": pos.magic,
                    "comment": pos.comment,
                    "time": datetime.fromtimestamp(pos.time, tz=pytz.UTC).isoformat(),
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def close_position(self, ticket: int) -> dict | None:
        """Close a position by ticket number."""
        if not self._ensure_connected():
            return None
        try:
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                logger.error(f"Position {ticket} not found")
                return None

            pos = position[0]
            symbol = pos.symbol

            # Ensure symbol is enabled
            mt5.symbol_select(symbol, True)

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None

            # Close with opposite order
            if pos.type == 0:  # BUY position → SELL to close
                close_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:  # SELL position → BUY to close
                close_type = mt5.ORDER_TYPE_BUY
                price = tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": "SmartMoney Bot Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return {"success": True, "ticket": ticket, "close_price": result.price}
            else:
                err = result.comment if result else "Unknown"
                return {"success": False, "ticket": ticket, "error": err}

        except Exception as e:
            logger.error(f"Failed to close position {ticket}: {e}")
            return None

    # ── history ──────────────────────────────────────────────────────────────

    def get_trade_history(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict]:
        """Get completed trade history."""
        if not self._ensure_connected():
            return []
        try:
            if from_date:
                date_from = datetime.fromisoformat(from_date)
            else:
                date_from = datetime.now(pytz.UTC) - timedelta(days=1)

            if to_date:
                date_to = datetime.fromisoformat(to_date)
            else:
                date_to = datetime.now(pytz.UTC)

            deals = mt5.history_deals_get(date_from, date_to)
            if deals is None or len(deals) == 0:
                return []

            return [
                {
                    "ticket": deal.ticket,
                    "order": deal.order,
                    "symbol": deal.symbol,
                    "type": deal.type,
                    "volume": deal.volume,
                    "price": deal.price,
                    "profit": deal.profit,
                    "swap": deal.swap,
                    "commission": deal.commission,
                    "comment": deal.comment,
                    "time": datetime.fromtimestamp(
                        deal.time, tz=pytz.UTC
                    ).isoformat(),
                }
                for deal in deals
            ]
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []

    # ── utility ──────────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        """Check if the MT5 terminal is connected."""
        if not MT5_AVAILABLE or not self._connected:
            return False
        try:
            info = mt5.terminal_info()
            return info is not None
        except Exception:
            return False

    def has_open_position(self, symbol: str) -> bool:
        """Check if there is an open position on a given symbol."""
        positions = self.get_positions()
        return any(p.get("symbol", "").upper() == symbol.upper() for p in positions)

    def get_daily_pnl(self) -> float:
        """Calculate today's P&L from open positions."""
        positions = self.get_positions()
        return sum(p.get("profit", 0.0) for p in positions)

    def _ensure_connected(self) -> bool:
        """Ensure MT5 is connected, reconnect if needed."""
        if not MT5_AVAILABLE:
            return False
        if not self._connected:
            return self.reconnect()
        # Quick health check
        try:
            info = mt5.terminal_info()
            if info is None:
                logger.warning("MT5 terminal_info() returned None — reconnecting")
                return self.reconnect()
        except Exception:
            return self.reconnect()
        return True
