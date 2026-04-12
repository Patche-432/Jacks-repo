"""MT5Connection — thin MetaTrader5 wrapper used by ai_pro.py."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class MT5ConfigError(ValueError):
    """Raised when config is missing or contains invalid values."""


class MT5ConnectionError(RuntimeError):
    """Raised when the MT5 terminal is unreachable or login fails."""


class MT5NotConnectedError(RuntimeError):
    """Raised when a data method is called before connecting."""


# ── Common MT5 terminal paths (Windows) ──────────────────────────────────────
_CANDIDATE_PATHS: list[str] = [
    r"C:\Program Files\MetaTrader 5\terminal64.exe",
    r"C:\Program Files\MetaTrader 5\terminal.exe",
    r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
    r"C:\Program Files (x86)\MetaTrader 5\terminal.exe",
]


def _discover_terminal_paths(explicit: Optional[str] = None) -> list[str]:
    """Return candidate terminal exe paths, explicit first."""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)

    # Walk MetaQuotes user-data folder for broker-specific installs
    user_root = Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal"
    if user_root.exists():
        for child in user_root.iterdir():
            for exe in ("terminal64.exe", "terminal.exe"):
                p = child / exe
                if p.exists():
                    candidates.append(str(p))

    candidates.extend(p for p in _CANDIDATE_PATHS if Path(p).exists())

    # De-duplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for item in candidates:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


class MT5Connection:
    """
    Thin wrapper around the MetaTrader5 library.

    Supports:
    - Direct ``connect()`` / ``disconnect()`` calls
    - Context manager (``with MT5Connection(cfg) as conn:``)
    - Auto-discovery of the terminal executable
    - Thread-safe stop signal for poll loops
    """

    def __init__(self, cfg: dict) -> None:
        """
        Args:
            cfg: Dict with keys: login, password, server, path, timeout, portable.
                 Only ``login`` is mandatory for a broker login; the rest are optional.
        """
        try:
            import MetaTrader5  # noqa: F401 – presence check only
        except ImportError:
            raise ImportError(
                "MetaTrader5 package not found.\n"
                "  Install: pip install MetaTrader5"
            )

        if not isinstance(cfg, dict):
            raise MT5ConfigError(f"cfg must be a dict, got {type(cfg).__name__}.")

        self._cfg = cfg
        self._connected = False
        self._stop_event = threading.Event()

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialise the MT5 terminal and log in.  Returns True on success."""
        import MetaTrader5 as mt5

        log.info("MT5Connection: connecting …")
        base_kwargs = self._base_kwargs()
        if base_kwargs is None:
            return False

        # Try without an explicit path first, then with each candidate path
        attempts = [dict(base_kwargs)]  # attempt 1: no path override
        explicit_path = self._cfg.get("path")
        for path in _discover_terminal_paths(explicit_path):
            kw = dict(base_kwargs)
            kw["path"] = path
            attempts.append(kw)

        for kw in attempts:
            try:
                if mt5.initialize(**kw):
                    self._connected = True
                    self._log_account(mt5)
                    return True
                code, msg = mt5.last_error()
                log.debug("mt5.initialize() [%d] %s  kwargs=%s", code, msg, kw)
            except Exception as exc:
                log.debug("mt5.initialize() raised: %s  kwargs=%s", exc, kw)
            # Clean up between attempts
            try:
                mt5.shutdown()
            except Exception:
                pass

        code, msg = mt5.last_error()
        log.error("MT5 init failed after %d attempt(s) — last error [%d] %s",
                  len(attempts), code, msg)
        return False

    def disconnect(self) -> None:
        """Shut down the MT5 connection. Safe to call even if not connected."""
        if not self._connected:
            return
        import MetaTrader5 as mt5
        try:
            mt5.shutdown()
            log.info("MT5Connection: disconnected.")
        except Exception as exc:
            log.error("Error during mt5.shutdown(): %s", exc)
        finally:
            self._connected = False

    def stop(self) -> None:
        """Signal any waiting poll loop to exit (thread-safe)."""
        self._stop_event.set()

    def is_connected(self) -> bool:
        return self._connected

    # ── Data helpers ──────────────────────────────────────────────────────────

    def account_info(self):
        """Return MT5 account_info namedtuple or None."""
        self._require_connected()
        import MetaTrader5 as mt5
        try:
            result = mt5.account_info()
            if result is None:
                code, msg = mt5.last_error()
                log.error("account_info() None [%d] %s", code, msg)
            return result
        except Exception as exc:
            log.error("account_info() raised: %s", exc)
            return None

    def terminal_info(self):
        """Return MT5 terminal_info namedtuple or None."""
        self._require_connected()
        import MetaTrader5 as mt5
        try:
            result = mt5.terminal_info()
            if result is None:
                code, msg = mt5.last_error()
                log.error("terminal_info() None [%d] %s", code, msg)
            return result
        except Exception as exc:
            log.error("terminal_info() raised: %s", exc)
            return None

    def runtime_info(self) -> dict:
        """
        Return a serialisable snapshot of the connected terminal + account.
        Equivalent to the old ``_read_mt5_runtime_info()`` helper in ai_pro.py.
        """
        self._require_connected()
        import MetaTrader5 as mt5
        terminal = mt5.terminal_info()
        account  = mt5.account_info()
        symbols  = list(mt5.symbols_get() or [])
        visible  = [str(s.name) for s in symbols if getattr(s, "visible", False)]
        return {
            "connected":        True,
            "terminal_name":    getattr(terminal, "name",    None) if terminal else None,
            "terminal_company": getattr(terminal, "company", None) if terminal else None,
            "terminal_path":    getattr(terminal, "path",    None) if terminal else None,
            "login":            getattr(account,  "login",   None) if account else None,
            "server":           getattr(account,  "server",  None) if account else None,
            "account_name":     getattr(account,  "name",    None) if account else None,
            "currency":         getattr(account,  "currency",None) if account else None,
            "balance":          round(getattr(account, "balance", 0), 2) if account else 0,
            "equity":           round(getattr(account, "equity", 0), 2) if account else 0,
            "trade_allowed":    bool(getattr(account, "trade_allowed", False)) if account else False,
            "visible_symbols":  visible,
            "symbols_total":    len(symbols),
        }

    def place_test_trade(self, symbol: str = "EURUSD", volume: float = 0.001) -> dict:
        """
        Place a taste test market BUY trade at current price.
        Returns dict with ticket, price, status info.
        """
        self._require_connected()
        import MetaTrader5 as mt5

        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"ok": False, "error": f"Symbol {symbol} not found"}

            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"ok": False, "error": f"Cannot get price for {symbol}"}

            # Prepare order
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "deviation": 20,
                "magic": 234000,
                "comment": "Taste test trade 0.001",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Send order
            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return {
                    "ok": True,
                    "ticket": result.order,
                    "symbol": symbol,
                    "volume": volume,
                    "price": tick.ask,
                    "comment": "Test trade placed successfully"
                }
            else:
                code, msg = mt5.last_error()
                return {"ok": False, "error": f"Order failed [{result.retcode}]: {msg}"}
        except Exception as exc:
            log.error(f"Test trade error: {exc}")
            return {"ok": False, "error": str(exc)}

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "MT5Connection":
        if not self.connect():
            raise MT5ConnectionError(
                "MT5 failed to connect — check logs for details."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.disconnect()
        if exc_type and exc_type is not KeyboardInterrupt:
            log.error("Exception inside MT5Connection context: [%s] %s",
                      exc_type.__name__, exc_val)
        return False

    # ── Private ───────────────────────────────────────────────────────────────

    def _base_kwargs(self) -> Optional[dict]:
        """Build the kwargs for mt5.initialize(), excluding 'path'."""
        try:
            kw: dict = {
                "timeout":  int(self._cfg.get("timeout",  10_000)),
                "portable": bool(self._cfg.get("portable", False)),
            }
            if self._cfg.get("login"):
                kw["login"] = int(self._cfg["login"])
            if self._cfg.get("password"):
                kw["password"] = str(self._cfg["password"])
            if self._cfg.get("server"):
                kw["server"] = str(self._cfg["server"])
            return kw
        except (TypeError, ValueError) as exc:
            log.error("Bad value in MT5 config: %s", exc)
            return None

    def _log_account(self, mt5) -> None:
        """Log version and account summary immediately after connecting."""
        try:
            log.info("MT5 connected ✓  build=%s", mt5.version())
            acct = mt5.account_info()
            if acct:
                log.info("Account: %s | %s | %.2f %s",
                         acct.login, acct.server, acct.balance, acct.currency)
            else:
                log.warning("Connected but could not retrieve account info.")
        except Exception as exc:
            log.warning("Post-connect info error (non-fatal): %s", exc)

    def _require_connected(self) -> None:
        if not self._connected:
            raise MT5NotConnectedError(
                "Not connected. Call connect() or use the context manager first."
            )
