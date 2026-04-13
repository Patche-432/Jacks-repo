"""MT5Connection — thin MetaTrader5 wrapper used by ai_pro.py."""

from __future__ import annotations

import logging
import threading
import time
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
        self._last_heartbeat = 0
        self._heartbeat_interval = 30  # Check connection health every 30 seconds

        # Background monitor
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
        self._monitor_interval: int = 12  # seconds between health-checks

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
                    self._last_heartbeat = time.time()  # Initialize heartbeat timestamp
                    self._log_account(mt5)
                    # Verify account info to confirm successful connection
                    account = mt5.account_info()
                    if account is None:
                        log.warning("MT5 authenticated but account_info() failed")
                        self.disconnect()
                        continue
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
        # Stop the background monitor before closing the connection so it
        # doesn't race with mt5.shutdown().
        self.stop_monitor()

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
            self._last_heartbeat = 0  # Reset heartbeat on disconnect

    def stop(self) -> None:
        """Signal any waiting poll loop to exit (thread-safe)."""
        self._stop_event.set()

    def is_connected(self) -> bool:
        return self._connected

    def check_connection(self) -> bool:
        """
        Check if connection is still alive and update heartbeat.
        Detects stale connections with periodic health checks.
        Returns True if connected, False otherwise.
        """
        if not self._connected:
            return False
        
        current_time = time.time()
        
        # Skip heartbeat check if recently verified
        if current_time - self._last_heartbeat < self._heartbeat_interval:
            return True
        
        try:
            # Test connection with account info
            import MetaTrader5 as mt5
            account = mt5.account_info()
            if account is None:
                code, msg = mt5.last_error()
                log.warning("Connection health check failed [%d] %s", code, msg)
                self._connected = False
                return False
            
            # Connection is healthy, update heartbeat
            self._last_heartbeat = current_time
            return True
            
        except Exception as exc:
            log.error("Connection health check raised: %s", exc)
            self._connected = False
            return False

    def reconnect(self, max_attempts: int = 3) -> bool:
        """
        Attempt to reconnect to MT5 after a connection loss.
        Uses exponential backoff between attempts.
        
        Args:
            max_attempts: Maximum number of reconnection attempts (default 3)
        
        Returns:
            True if reconnected, False if all attempts failed
        """
        log.info("Attempting to reconnect to MT5 (max %d attempts)...", max_attempts)
        
        for attempt in range(1, max_attempts + 1):
            log.info("  Reconnect attempt %d/%d", attempt, max_attempts)
            
            # Check if connection was restored naturally
            if self.check_connection():
                log.info("Connection restored successfully")
                return True
            
            # Clean shutdown before retry
            self.disconnect()
            
            # Exponential backoff: 2s, 6s, 18s
            wait_time = 2 * (3 ** (attempt - 1))
            log.info("  Waiting %d seconds before next attempt...", wait_time)
            time.sleep(wait_time)
            
            # Try to reconnect
            try:
                if self.connect():
                    log.info("Reconnected successfully on attempt %d", attempt)
                    return True
            except Exception as exc:
                log.error("Reconnection attempt %d failed: %s", attempt, exc)
        
        log.error("All reconnection attempts failed ({} total)", max_attempts)
        return False

    # ── Background monitor ────────────────────────────────────────────────────

    def start_monitor(self, interval: int = 12) -> None:
        """
        Start a background daemon thread that periodically checks the
        connection and automatically calls ``reconnect()`` on failure.

        Safe to call multiple times — a new thread is only launched when
        none is already running.

        Args:
            interval: Seconds between health-checks (default 12, min 5).
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            log.debug("MT5 monitor already running — skipping start")
            return

        self._monitor_interval = max(5, int(interval))
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="mt5-connection-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        log.info("MT5 background monitor started (interval=%ds)", self._monitor_interval)

    def stop_monitor(self) -> None:
        """Signal the background monitor thread to stop and wait for it to exit."""
        self._monitor_stop.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=self._monitor_interval + 2)
            if self._monitor_thread.is_alive():
                log.warning("MT5 monitor thread did not stop within timeout")
        self._monitor_thread = None
        log.info("MT5 background monitor stopped")

    def _monitor_loop(self) -> None:
        """Internal loop executed by the background monitor thread."""
        log.debug("MT5 monitor loop started")
        while not self._monitor_stop.is_set():
            # Wait for the configured interval (or until stop is signalled)
            self._monitor_stop.wait(timeout=self._monitor_interval)
            if self._monitor_stop.is_set():
                break

            if not self._connected:
                # Not connected — skip (server-initiated disconnects are intentional)
                continue

            try:
                healthy = self.check_connection()
            except Exception as exc:
                log.error("MT5 monitor: health check raised %s", exc)
                healthy = False

            if not healthy:
                log.warning("MT5 monitor: connection lost — attempting reconnect…")
                try:
                    restored = self.reconnect()
                except Exception as exc:
                    log.error("MT5 monitor: reconnect raised %s", exc)
                    restored = False

                if restored:
                    log.info("MT5 monitor: connection restored successfully ✓")
                else:
                    log.error("MT5 monitor: reconnect failed — will retry in %ds",
                              self._monitor_interval)

        log.debug("MT5 monitor loop exited")

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

    def status(self) -> dict:
        """
        Return simplified connection status for dashboard sync.
        Used by server.py and web.py to update UI state.
        
        Returns:
            dict with keys: connected, account, error
        """
        if not self._connected:
            return {"connected": False, "account": None, "error": "Not connected"}
        
        try:
            account = self.account_info()
            if account is None:
                return {"connected": False, "account": None, "error": "account_info() returned None"}
            
            return {
                "connected": True,
                "account": {
                    "login": getattr(account, "login", None),
                    "server": getattr(account, "server", None),
                    "name": getattr(account, "name", None),
                    "currency": getattr(account, "currency", None),
                    "balance": round(getattr(account, "balance", 0), 2),
                    "equity": round(getattr(account, "equity", 0), 2),
                    "trade_allowed": bool(getattr(account, "trade_allowed", False)),
                },
                "error": None,
            }
        except Exception as exc:
            log.error("status() raised: %s", exc)
            return {"connected": False, "account": None, "error": str(exc)}

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

            # Check for success (retcode is 0-based, success codes are < 10030)
            if result.retcode in [mt5.TRADE_RETCODE_DONE, 10030]:  # 10030 = success with IOC
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
