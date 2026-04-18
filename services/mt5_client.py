"""
mt5_client.py — HTTP REST wrapper for the metatrader-mcp-server.
All MT5 communication goes through http://localhost:8001/api/v1.
"""

import requests
from loguru import logger

from config import settings
from core.models import AccountInfo, PriceData, OrderResult


class MT5Client:
    """HTTP client for the MetaTrader 5 MCP REST API."""

    def __init__(self):
        self.base_url = settings.MT5_BASE_URL
        self.timeout = settings.MT5_TIMEOUT

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict | None = None) -> dict | list | None:
        """Send a GET request to the MT5 API."""
        url = f"{self.base_url}{endpoint}"
        try:
            r = requests.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            logger.error(f"MT5 API connection failed — is metatrader-http-server running on {self.base_url}?")
            return None
        except requests.exceptions.Timeout:
            logger.error(f"MT5 API timeout on GET {endpoint}")
            return None
        except Exception as e:
            logger.error(f"MT5 API error on GET {endpoint}: {e}")
            return None

    def _post(self, endpoint: str, payload: dict) -> dict | None:
        """Send a POST request to the MT5 API."""
        url = f"{self.base_url}{endpoint}"
        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            logger.error(f"MT5 API connection failed — is metatrader-http-server running on {self.base_url}?")
            return None
        except requests.exceptions.Timeout:
            logger.error(f"MT5 API timeout on POST {endpoint}")
            return None
        except Exception as e:
            logger.error(f"MT5 API error on POST {endpoint}: {e}")
            return None

    def _delete(self, endpoint: str) -> dict | None:
        """Send a DELETE request to the MT5 API."""
        url = f"{self.base_url}{endpoint}"
        try:
            r = requests.delete(url, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"MT5 API error on DELETE {endpoint}: {e}")
            return None

    # ── account ──────────────────────────────────────────────────────────────

    def get_account_info(self) -> AccountInfo | None:
        """Get account balance, equity, margin, etc."""
        data = self._get("/account/info")
        if data is None:
            return None
        try:
            return AccountInfo(**data)
        except Exception as e:
            logger.error(f"Failed to parse account info: {e}")
            return None

    # ── market data ──────────────────────────────────────────────────────────

    def get_price(self, symbol: str) -> PriceData | None:
        """Get live bid/ask/spread for a symbol."""
        data = self._get("/market/price", params={"symbol_name": symbol})
        if data is None:
            return None
        try:
            # Use per-pair pip size for spread calculation
            pip_size = settings.PAIR_PIP_SIZES.get(symbol, settings.PIP_SIZE)

            bid = data.get("bid", 0)
            ask = data.get("ask", 0)
            spread_raw = ask - bid
            spread_pips = spread_raw / pip_size if pip_size > 0 else 0
            return PriceData(
                symbol=data.get("symbol", symbol),
                bid=bid,
                ask=ask,
                spread=spread_raw,
                spread_pips=round(spread_pips, 1),
            )
        except Exception as e:
            logger.error(f"Failed to parse price data for {symbol}: {e}")
            return None

    def get_symbols(self) -> list | None:
        """List all available symbols."""
        return self._get("/market/symbols")

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
            symbol:      "EURUSD" or "GBPUSD"
            direction:   "BUY" or "SELL"
            volume:      lot size (e.g. 0.01)
            stop_loss:   price level (not pips)
            take_profit: price level (not pips)
        """
        payload = {
            "symbol": symbol,
            "volume": volume,
            "type": direction.upper(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        logger.info(f"Placing {direction} {volume} lots {symbol} | SL={stop_loss} TP={take_profit}")

        data = self._post("/order/market", payload)
        if data is None:
            return OrderResult(success=False, message="MT5 API unreachable")

        return OrderResult(
            success=data.get("success", False),
            order=data.get("order"),
            message=data.get("message", ""),
            symbol=data.get("symbol", symbol),
            type=data.get("type", direction),
            volume=data.get("volume", volume),
            price=data.get("price", 0.0),
            stop_loss=data.get("stop_loss", stop_loss),
            take_profit=data.get("take_profit", take_profit),
            error_code=data.get("error_code"),
        )

    # ── positions ────────────────────────────────────────────────────────────

    def get_positions(self) -> list[dict]:
        """Get all currently open positions."""
        data = self._get("/positions")
        if data is None:
            return []
        return data if isinstance(data, list) else []

    def close_position(self, ticket: int) -> dict | None:
        """Close a position by ticket number."""
        return self._delete(f"/positions/{ticket}")

    # ── history ──────────────────────────────────────────────────────────────

    def get_trade_history(self, from_date: str | None = None, to_date: str | None = None) -> list[dict]:
        """Get completed trade history."""
        params = {}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date
        data = self._get("/history/deals", params=params)
        if data is None:
            return []
        return data if isinstance(data, list) else []

    # ── utility ──────────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        """Check if the MT5 API is reachable."""
        info = self.get_account_info()
        return info is not None

    def has_open_position(self, symbol: str) -> bool:
        """Check if there is an open position on a given symbol."""
        positions = self.get_positions()
        return any(p.get("symbol", "").upper() == symbol.upper() for p in positions)

    def get_daily_pnl(self) -> float:
        """Calculate today's P&L from open positions."""
        positions = self.get_positions()
        return sum(p.get("profit", 0.0) for p in positions)

    # ── limit orders ─────────────────────────────────────────────────────────

    def place_limit_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        price: float,
        stop_loss: float,
        take_profit: float,
    ) -> OrderResult:
        """
        Place a limit (pending) BUY_LIMIT or SELL_LIMIT order.

        Args:
            symbol:      "EURUSD" or "GBPUSD"
            direction:   "BUY" or "SELL"
            volume:      lot size
            price:       limit entry price
            stop_loss:   SL price level
            take_profit: TP price level
        """
        order_type = "BUY_LIMIT" if direction.upper() == "BUY" else "SELL_LIMIT"
        payload = {
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

        logger.info(f"Placing {order_type} {volume} lots {symbol} @ {price} | SL={stop_loss} TP={take_profit}")

        data = self._post("/order/pending", payload)
        if data is None:
            # Fallback: try as market order if pending endpoint doesn't exist
            logger.warning("Pending order endpoint failed — falling back to market order")
            return self.place_market_order(symbol, direction, volume, stop_loss, take_profit)

        return OrderResult(
            success=data.get("success", False),
            order=data.get("order"),
            message=data.get("message", ""),
            symbol=data.get("symbol", symbol),
            type=data.get("type", order_type),
            volume=data.get("volume", volume),
            price=data.get("price", price),
            stop_loss=data.get("stop_loss", stop_loss),
            take_profit=data.get("take_profit", take_profit),
            error_code=data.get("error_code"),
        )

    # ── position management ──────────────────────────────────────────────────

    def close_partial(self, ticket: int, volume: float) -> OrderResult:
        """
        Close a partial volume of an open position.

        Args:
            ticket: position ticket number
            volume: volume to close (e.g., 0.01 to close half of 0.02)
        """
        payload = {"ticket": ticket, "volume": volume}
        logger.info(f"Closing {volume} lots of position #{ticket}")

        data = self._post(f"/positions/{ticket}/close-partial", payload)
        if data is None:
            # Fallback: try standard close
            logger.warning("Partial close endpoint failed — trying full close")
            result = self.close_position(ticket)
            return OrderResult(
                success=result is not None,
                order=ticket,
                message=str(result) if result else "Close failed",
            )

        return OrderResult(
            success=data.get("success", False),
            order=data.get("order", ticket),
            message=data.get("message", ""),
            volume=volume,
        )

    def modify_sl(self, ticket: int, new_sl: float) -> bool:
        """
        Modify the stop loss of an open position (e.g., move to breakeven).

        Args:
            ticket: position ticket number
            new_sl: new SL price level

        Returns:
            True if modification succeeded
        """
        payload = {"ticket": ticket, "stop_loss": new_sl}
        logger.info(f"Modifying SL of position #{ticket} → {new_sl}")

        data = self._post(f"/positions/{ticket}/modify", payload)
        if data is None:
            logger.warning(f"Failed to modify SL for position #{ticket}")
            return False

        return data.get("success", False)

    def modify_position(self, ticket: int, stop_loss: float = None, take_profit: float = None) -> bool:
        """
        Modify SL and/or TP of an open position.

        Args:
            ticket: position ticket number
            stop_loss: new SL price (or None to keep current)
            take_profit: new TP price (or None to keep current)

        Returns:
            True if modification succeeded
        """
        payload = {"ticket": ticket}
        if stop_loss is not None:
            payload["stop_loss"] = stop_loss
        if take_profit is not None:
            payload["take_profit"] = take_profit

        logger.info(f"Modifying position #{ticket} — SL={stop_loss}, TP={take_profit}")

        data = self._post(f"/positions/{ticket}/modify", payload)
        if data is None:
            logger.warning(f"Failed to modify position #{ticket}")
            return False

        return data.get("success", False)

    # ── pending order management ─────────────────────────────────────────────

    def get_pending_orders(self) -> list[dict]:
        """Get all pending (unfilled) orders."""
        data = self._get("/orders/pending")
        if data is None:
            return []
        return data if isinstance(data, list) else []

    def cancel_pending_order(self, ticket: int) -> bool:
        """Cancel a pending order by ticket."""
        logger.info(f"Cancelling pending order #{ticket}")
        data = self._delete(f"/orders/{ticket}")
        if data is None:
            return False
        return data.get("success", True)  # assume success if endpoint exists
