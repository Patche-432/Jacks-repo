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
        self._last_heartbeat = 0.0
        self._heartbeat_interval = 30  # Check connection health every 30 seconds

        # Serialise ALL calls into the MetaTrader5 C extension. The MT5 Python
        # API is NOT thread-safe: concurrent calls from the Flask request
        # threads and the background monitor cause spurious failures that
        # previously caused the connection to "bounce" in and out.
        self._mt5_lock = threading.RLock()

        # Tolerance for transient failures so a single failed health-check
        # doesn't immediately flip the connection to "disconnected".
        self._consecutive_failures = 0
        self._failure_threshold = 3  # require N consecutive failures to disconnect

        # Background monitor
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
        self._monitor_interval: int = 12  # seconds between health-checks
        # Remember whether the monitor should be running so reconnect() can
        # restart it after a successful reconnect (previously the monitor died
        # permanently after the first auto-recovery).
        self._monitor_wanted = False

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

        last_code, last_msg = 0, ""
        with self._mt5_lock:
            for kw in attempts:
                initialised = False
                try:
                    initialised = bool(mt5.initialize(**kw))
                except Exception as exc:
                    log.debug("mt5.initialize() raised: %s  kwargs=%s", exc, kw)
                    initialised = False

                if initialised:
                    # Verify account info to confirm full login success.
                    try:
                        account = mt5.account_info()
                    except Exception as exc:
                        log.warning("account_info() raised after init: %s", exc)
                        account = None

                    if account is not None:
                        self._connected = True
                        self._last_heartbeat = time.time()
                        self._consecutive_failures = 0
                        self._log_account(mt5)
                        return True

                    # Authenticated but account missing — clean shutdown and retry
                    log.warning("MT5 authenticated but account_info() returned None; retrying")
                    try:
                        mt5.shutdown()
                    except Exception:
                        pass
                    continue

                # initialize() returned False — record the error and try next path
                try:
                    last_code, last_msg = mt5.last_error()
                except Exception:
                    last_code, last_msg = 0, ""
                log.debug("mt5.initialize() failed [%d] %s  kwargs=%s", last_code, last_msg, kw)
                # Ensure we're not leaving a half-initialized terminal behind
                try:
                    mt5.shutdown()
                except Exception:
                    pass

        log.error("MT5 init failed after %d attempt(s) — last error [%d] %s",
                  len(attempts), last_code, last_msg)
        self._connected = False
        return False

    def disconnect(self, stop_monitor: bool = True) -> None:
        """Shut down the MT5 connection. Safe to call even if not connected.

        Args:
            stop_monitor: If True (default), also stops the background monitor
                thread. Set to False for internal transitions (e.g. reconnect
                after a failed health-check) so the monitor survives the blip.
        """
        if stop_monitor:
            # Stop the background monitor before closing the connection so it
            # doesn't race with mt5.shutdown().
            self.stop_monitor()
            self._monitor_wanted = False

        if not self._connected:
            return
        import MetaTrader5 as mt5
        with self._mt5_lock:
            try:
                mt5.shutdown()
                log.info("MT5Connection: disconnected.")
            except Exception as exc:
                log.error("Error during mt5.shutdown(): %s", exc)
            finally:
                self._connected = False
                self._last_heartbeat = 0.0
                self._consecutive_failures = 0

    def stop(self) -> None:
        """Signal any waiting poll loop to exit (thread-safe)."""
        self._stop_event.set()

    def is_connected(self) -> bool:
        return self._connected

    def check_connection(self) -> bool:
        """
        Check if connection is still alive and update heartbeat.

        Uses a consecutive-failure threshold so a single transient glitch
        (network blip, MT5 terminal momentarily busy, symbol refresh, etc.)
        does NOT immediately mark the connection as dead. This is the main
        fix for the "connection bouncing in and out" symptom.

        Returns True if the connection is considered healthy.
        """
        if not self._connected:
            return False

        current_time = time.time()

        # Skip heartbeat check if recently verified
        if current_time - self._last_heartbeat < self._heartbeat_interval:
            return True

        import MetaTrader5 as mt5

        account = None
        err_code, err_msg = 0, ""
        try:
            with self._mt5_lock:
                account = mt5.account_info()
                if account is None:
                    try:
                        err_code, err_msg = mt5.last_error()
                    except Exception:
                        err_code, err_msg = 0, ""
        except Exception as exc:
            log.warning("Connection health check raised: %s", exc)
            account = None

        if account is not None:
            # Healthy: reset failure counter and update heartbeat
            self._last_heartbeat = current_time
            self._consecutive_failures = 0
            return True

        # Tolerate transient failures — only disconnect after N in a row
        self._consecutive_failures += 1
        log.warning(
            "Health check failed [%d] %s (%d/%d before reconnect)",
            err_code, err_msg,
            self._consecutive_failures, self._failure_threshold,
        )
        if self._consecutive_failures >= self._failure_threshold:
            log.error("Health check failed %d times in a row — marking disconnected",
                      self._consecutive_failures)
            self._connected = False
            return False

        # Still within tolerance — report as healthy to avoid UI flicker
        return True

    def reconnect(self, max_attempts: int = 3) -> bool:
        """
        Attempt to reconnect to MT5 after a connection loss.

        Uses a gentler backoff (2s, 4s, 8s) than before to keep dashboard
        flicker short. Always calls ``mt5.shutdown()`` between attempts and
        restores the background monitor if it was wanted.

        Args:
            max_attempts: Maximum number of reconnection attempts (default 3)

        Returns:
            True if reconnected, False if all attempts failed
        """
        log.info("Attempting to reconnect to MT5 (max %d attempts)…", max_attempts)

        # Reset failure counter so check_connection() starts fresh post-recovery
        self._consecutive_failures = 0

        for attempt in range(1, max_attempts + 1):
            log.info("  Reconnect attempt %d/%d", attempt, max_attempts)

            # Always shut down any half-initialised state before retrying, but
            # keep the monitor alive so we resume monitoring automatically.
            try:
                self.disconnect(stop_monitor=False)
            except Exception as exc:
                log.debug("disconnect() during reconnect raised: %s", exc)

            # Gentler backoff: 2s, 4s, 8s (was 2/6/18)
            wait_time = 2 * (2 ** (attempt - 1))
            log.info("  Waiting %d seconds before next attempt…", wait_time)
            # Honour monitor stop signal while waiting so shutdown is prompt
            if self._monitor_stop.wait(timeout=wait_time):
                log.info("Reconnect aborted — monitor stop requested")
                return False

            try:
                if self.connect():
                    log.info("Reconnected successfully on attempt %d", attempt)
                    # If the monitor thread exited while we were down, restart it
                    if self._monitor_wanted and not (
                        self._monitor_thread and self._monitor_thread.is_alive()
                    ):
                        self.start_monitor(self._monitor_interval)
                    return True
            except Exception as exc:
                log.error("Reconnection attempt %d raised: %s", attempt, exc)

        log.error("All reconnection attempts failed (%d total)", max_attempts)
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
        self._monitor_wanted = True
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
            with self._mt5_lock:
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
            with self._mt5_lock:
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
        with self._mt5_lock:
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
            with self._mt5_lock:
                # Get symbol info
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    return {"ok": False, "error": f"Symbol {symbol} not found"}

                # Ensure symbol is selected in Market Watch
                if not symbol_info.visible:
                    if not mt5.symbol_select(symbol, True):
                        return {"ok": False, "error": f"Cannot select symbol {symbol}"}

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

            if result is None:
                code, msg = mt5.last_error()
                return {"ok": False, "error": f"order_send returned None [{code}] {msg}"}

            # Check for success — TRADE_RETCODE_DONE (10009) is the canonical success code
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
