"""Flask server for MT5 dashboard"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import atexit
from datetime import datetime, timezone
from flask import Flask, jsonify, send_from_directory, request, Response, stream_with_context
from flask_cors import CORS
import logging
from core.mt5_connection import MT5Connection
import math

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, static_folder=root_path, static_url_path='')
CORS(app)

# ── Global MT5 connection and status ────────────────────────────────────────

mt5_conn = None
mt5_status = {
    "mt5_connected": False,
    "account": None,
    "error": None,
}
# Guard mutations to mt5_conn / mt5_status across request threads and the
# MT5 background monitor thread.
import threading as _threading
_mt5_state_lock = _threading.Lock()

# Dashboard runtime config (in-memory)
_enabled_symbols = {"GBPJPY", "EURJPY", "GBPUSD", "EURUSD"}


def _shutdown_monitor() -> None:
    """Stop the MT5 background monitor when the server process exits."""
    global mt5_conn
    if mt5_conn is not None:
        try:
            mt5_conn.stop_monitor()
        except Exception:
            pass


atexit.register(_shutdown_monitor)


def _auto_connect_mt5() -> None:
    """
    Attempt to attach to the already-running MT5 terminal on startup.

    If MT5 is open and logged in, mt5.initialize() (with no credentials)
    attaches silently and the dashboard is immediately usable without the
    operator having to click "Connect MT5" after every server restart.

    Runs in a background thread so it never delays server startup — the
    dashboard comes up instantly, and signals start working within a few
    seconds once MT5 is detected.
    """
    import time
    def _try() -> None:
        global mt5_conn
        time.sleep(2)
        if mt5_conn is not None and mt5_conn.is_connected():
            return
        log.info("Auto-connect: attempting to attach to running MT5 terminal…")
        try:
            conn = MT5Connection({})
            if conn.connect():
                conn.start_monitor()
                mt5_conn = conn
                _sync_mt5_status()
                log.info("Auto-connect: MT5 attached successfully")
            else:
                log.info("Auto-connect: MT5 not reachable yet — use Connect MT5 button")
        except Exception as exc:
            log.debug("Auto-connect: skipped (%s)", exc)
    _threading.Thread(target=_try, name="mt5-auto-connect", daemon=True).start()

_auto_connect_mt5()


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty:
        return float("nan")
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else float("nan")


def _rr_string(entry: float, sl: float, tp: float) -> str:
    try:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            return "—"
        return f"1:{reward / risk:.2f}"
    except Exception:
        return "—"


def _mt5_signal_snapshot(symbol: str) -> dict:
    """Run the real Agent Zero 4-environment strategy and return a dashboard snapshot.

    Uses AgentZeroBot.generate_trade_signal() with use_ai=False (raw strategy
    only, no Agent Zero review) so the dashboard shows signals from the same
    CHoCH + PDH/PDL logic the live bot uses, not the old SMA20/SMA50
    approximation.

    Falls back to an error dict on import or strategy failures.
    """
    symbol = symbol.upper().strip()

    # Ensure MT5 is connected before handing off to the strategy
    global mt5_conn
    if not mt5_conn or not mt5_conn.is_connected():
        return {"error": "MT5 not connected. Connect first then refresh signals."}

    try:
        # Import AgentZeroBot at call-time to avoid circular-import issues at
        # startup. The module-level Flask app in ai_pro.py is created but
        # never started, so importing it here is safe.
        from ai_pro import AgentZeroBot
    except Exception as exc:
        log.error("Could not import AgentZeroBot: %s", exc)
        return {"error": f"Strategy import failed: {exc}"}

    try:
        # use_ai=False skips the Agent Zero review — pure rule-based signal
        strategy = AgentZeroBot(use_ai=False)
        sig = strategy.generate_trade_signal(symbol)
    except Exception as exc:
        log.error("AgentZeroBot.generate_trade_signal(%s) raised: %s", symbol, exc)
        return {"error": f"Strategy error: {exc}"}

    # Map AgentZeroBot signal → dashboard format
    # signal_source encodes the environment: CHoCH-BUY@PDL / CHoCH-SELL@PDH /
    # Continuation-BUY@PDH / Continuation-SELL@PDL
    raw_signal = sig.get("signal", "neutral")
    source     = sig.get("signal_source") or ""
    bias = "LONG" if raw_signal == "BUY" else "SHORT" if raw_signal == "SELL" else "—"

    # Human-readable environment label from signal_source
    env_map = {
        "CHoCH-BUY@PDL":          "ENV 1 — CHoCH BUY at PDL (failed lower low)",
        "CHoCH-SELL@PDH":         "ENV 2 — CHoCH SELL at PDH (failed higher high)",
        "Continuation-BUY@PDH":   "ENV 3 — Continuation BUY (broke above PDH, retesting)",
        "Continuation-SELL@PDL":  "ENV 4 — Continuation SELL (broke below PDL, retesting)",
    }
    environment = env_map.get(source, sig.get("reason") or "No active environment")

    # CHoCH status from signal_source / reason
    choch_status = "—"
    if "CHoCH" in source:
        choch_status = "Detected ✓" if raw_signal != "neutral" else "Not confirmed"

    # Level interaction from reason text
    level_interaction = "—"
    if sig.get("previous_day_high") and sig.get("previous_day_low"):
        pdh = sig["previous_day_high"]
        pdl = sig["previous_day_low"]
        level_interaction = f"PDH {round(pdh, 5)} / PDL {round(pdl, 5)}"

    entry    = sig.get("entry_price") or 0.0
    sl_price = sig.get("stop_loss")   or 0.0
    tp_price = sig.get("take_profit") or 0.0
    atr_val  = sig.get("atr")         or 0.0
    conf     = sig.get("confidence")  or 0

    # ── Volume Profile / POC ──────────────────────────────────────────────
    vp       = sig.get("volume_profile") or {}
    poc      = float(vp.get("poc") or 0.0)
    pip_size = 0.01 if "JPY" in symbol else 0.0001

    poc_aligned: bool | None = None
    poc_dist_pips: float | None = None
    poc_side: str = "—"
    ref_price = float(entry or 0.0)
    if poc > 0.0 and ref_price > 0.0:
        poc_dist_pips = round((ref_price - poc) / pip_size, 1)
        poc_side = "above" if ref_price > poc else "below" if ref_price < poc else "at"
        if raw_signal == "BUY":
            poc_aligned = ref_price > poc
        elif raw_signal == "SELL":
            poc_aligned = ref_price < poc

    # ── Per-pair backtest-tuned zone multiplier ───────────────────────────
    atr_tol_mult: float | None = None
    try:
        from agent_learning_loop import get_learning_loop as _gll
        atr_tol_mult = round(_gll().get_tuned_params(symbol)["atr_tolerance_mult"], 2)
    except Exception:
        pass

    return {
        "symbol":            symbol,
        "bias":              bias,
        "confidence":        conf,
        "environment":       environment,
        "choch_status":      choch_status,
        "level_interaction": level_interaction,
        "entry_price":       round(float(entry),    5) if entry    else 0.0,
        "sl_price":          round(float(sl_price), 5) if sl_price else 0.0,
        "tp_price":          round(float(tp_price), 5) if tp_price else 0.0,
        "rr":                _rr_string(entry, sl_price, tp_price),
        "atr":               round(float(atr_val),  5) if atr_val  else 0.0,
        "signal_source":     source,
        "ai_approved":       sig.get("ai_approved"),
        "ts":                datetime.now(timezone.utc).isoformat(),
        # Volume profile
        "poc":               round(poc, 5) if poc > 0.0 else None,
        "poc_aligned":       poc_aligned,
        "poc_dist_pips":     poc_dist_pips,
        "poc_side":          poc_side,
        # Backtest-tuned entry zone
        "atr_tol_mult":      atr_tol_mult,
    }


def _sync_mt5_status() -> None:
    """Pull the latest MT5 connection state into the global mt5_status dict."""
    global mt5_conn, mt5_status
    with _mt5_state_lock:
        conn = mt5_conn

    if conn is None:
        with _mt5_state_lock:
            mt5_status.update({"mt5_connected": False, "account": None,
                               "error": "No connection object"})
        return

    try:
        conn_status = conn.status()
        with _mt5_state_lock:
            mt5_status["mt5_connected"] = conn_status["connected"]
            mt5_status["account"] = conn_status["account"]
            mt5_status["error"] = conn_status["error"]
    except Exception as exc:
        log.error("Failed to sync MT5 status: %s", exc)
        with _mt5_state_lock:
            mt5_status["mt5_connected"] = False
            mt5_status["error"] = str(exc)

@app.route('/')
def serve_dashboard():
    return send_from_directory(root_path, 'index.html')

# Disable caching for the dashboard's HTML/CSS/JS so a stale browser copy
# never masks a server-side fix. Without this, Chrome will happily serve the
# previous dashboard.js even after a hard refresh under some configs.
@app.after_request
def _no_cache_static(resp):
    try:
        path = request.path or ""
        if path == "/" or path.endswith(".html") or path.endswith(".js") or path.endswith(".css"):
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
    except Exception:
        pass
    return resp

@app.route('/api/mt5/connect', methods=['POST'])
def api_connect_mt5():
    """Connect to MT5"""
    global mt5_conn
    try:
        payload = request.get_json(silent=True) or {}
        cfg = {}
        # Optional terminal executable path override (helps when MT5 is installed in a broker-specific folder)
        if isinstance(payload, dict) and payload.get("path"):
            cfg["path"] = str(payload["path"])
        mt5_conn = MT5Connection(cfg)
        if mt5_conn.connect():
            mt5_conn.start_monitor()
            _sync_mt5_status()
            runtime = mt5_conn.runtime_info()
            return jsonify({'connected': True, **runtime})
        else:
            _sync_mt5_status()
            # Provide a more actionable error if possible
            err = 'Failed to connect to MT5. Make sure the MT5 desktop terminal is installed, open, and logged into an account.'
            try:
                import MetaTrader5 as mt5
                code, msg = mt5.last_error()
                if code or msg:
                    err = f"MT5 initialize failed [{code}] {msg}. Open MT5, log in, then try again."
            except Exception:
                pass
            return jsonify({'connected': False, 'error': err}), 400
    except Exception as e:
        log.error(f"MT5 connect error: {e}")
        _sync_mt5_status()
        return jsonify({'connected': False, 'error': str(e)}), 500

@app.route('/api/mt5/status', methods=['GET'])
def api_mt5_status():
    """Get MT5 connection status (synced state, flattened for dashboard)"""
    global mt5_conn
    # Refresh status from connection
    _sync_mt5_status()
    
    # Flatten response for dashboard display
    if not mt5_status["mt5_connected"]:
        return jsonify({"connected": False, "error": mt5_status.get("error", "Not connected")})
    
    account = mt5_status.get("account")
    if not account:
        return jsonify({"connected": False, "error": "Account data unavailable"})
    
    return jsonify({
        "connected": True,
        "server": account.get("server"),
        "login": account.get("login"),
        "account_name": account.get("name"),
        "currency": account.get("currency"),
        "balance": account.get("balance"),
        "equity": account.get("equity"),
        "trade_allowed": account.get("trade_allowed"),
        "error": None,
    })


@app.route('/api/ollama/health', methods=['GET'])
def api_ollama_health():
    """Lightweight status probe for the local Ollama server.

    Surfaces ai_agent.ollama_health() to the dashboard so the operator can
    see at a glance whether the agent layer is ready (server reachable,
    configured model pulled) without starting the bot. Also includes a
    summary of the learning loop's current state — whether
    backtest_insights.json exists, how many pairs are loaded — so the
    sidebar can show "Learning: N pairs" once a backtest has been run.

    Never raises — on import failure we report a structured error so the
    dashboard can render a useful message.
    """
    try:
        from ai_agent import ollama_health
    except Exception as exc:
        log.error("ollama_health import failed: %s", exc)
        return jsonify({
            "reachable": False,
            "model_loaded": False,
            "url": None,
            "model": None,
            "learning": None,
            "error": f"agent module import failed: {exc}",
        }), 500

    try:
        payload = ollama_health()
    except Exception as exc:
        log.error("ollama_health probe failed: %s", exc)
        return jsonify({
            "reachable": False,
            "model_loaded": False,
            "url": None,
            "model": None,
            "learning": None,
            "error": f"probe failed: {exc}",
        }), 500

    # Append a compact learning-loop summary so the dashboard can render a
    # single row instead of needing a second endpoint. Best-effort — if the
    # learning loop module isn't importable, we return null and the
    # sidebar simply hides the Learning row.
    learning_summary = None
    try:
        from agent_learning_loop import get_learning_loop
        st = get_learning_loop().status()
        learning_summary = {
            "file_exists":  bool(st.get("file_exists")),
            "pairs":        list(st.get("pairs") or []),
            "pair_count":   len(st.get("pairs") or []),
            "last_mtime":   st.get("last_mtime"),
        }
    except Exception as exc:
        # Don't fail the whole probe just because the learning loop is offline.
        log.debug("learning loop status probe failed: %s", exc)

    payload["learning"] = learning_summary
    return jsonify(payload)


@app.route('/api/config', methods=['GET'])
def api_config():
    """Verify backtested configuration baseline"""
    return jsonify({
        "status": "READY FOR LIVE DEMO",
        "configuration": {
            "portfolio": ["GBPJPY", "EURJPY", "GBPUSD", "EURUSD"],
            "lot_size": 0.50,
            "sl_mult": 2.5,
            "tp_mult": 4.5,
            "rr_ratio": "1:1.8"
        },
        "backtest_baseline": {
            "total_trades": 468,
            "win_rate": "63.6%",
            "total_pl": "+$46,221",
            "avg_per_trade": "+$98.76",
            "period_days": 7,
            "validation": "PASSED"
        },
        "per_pair": {
            "GBPJPY": {"trades": 149, "win_rate": "79.9%", "profit": "+$25,721"},
            "EURJPY": {"trades": 118, "win_rate": "56.8%", "profit": "+$6,339"},
            "GBPUSD": {"trades": 88, "win_rate": "61.4%", "profit": "+$8,345"},
            "EURUSD": {"trades": 113, "win_rate": "51.3%", "profit": "+$5,416"}
        }
    })


@app.route('/api/mt5/disconnect', methods=['POST'])
def api_disconnect_mt5():
    """Disconnect from MT5"""
    global mt5_conn
    if mt5_conn:
        mt5_conn.disconnect()
        mt5_conn = None
    _sync_mt5_status()
    return jsonify({'ok': True})

@app.route('/api/mt5/test-trade', methods=['POST'])
def api_test_trade():
    """Place a taste test trade (0.001 lot BUY on EURUSD)"""
    global mt5_conn
    if not mt5_conn or not mt5_conn.is_connected():
        return jsonify({'ok': False, 'error': 'Not connected to MT5'}), 400
    try:
        result = mt5_conn.place_test_trade(symbol='EURUSD', volume=0.001)
        if result['ok']:
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        log.error(f"Test trade error: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500

import subprocess
import time

bot_process = None
bot_running = False

@app.route('/bot/start', methods=['POST'])
def bot_start():
    """Start the AI Pro bot"""
    global bot_process, bot_running
    
    if bot_running and bot_process and bot_process.poll() is None:
        return jsonify({'ok': False, 'error': 'Agent Zero is already on the desk.'}), 409
    
    try:
        payload = request.get_json(silent=True) or {}
        _apply_bot_config_payload(payload)

        # ── Preflight: if AI review is on, Ollama must be reachable + model pulled.
        # Failing fast here is much friendlier than letting every poll cycle log
        # transport errors after the bot has launched.
        if bool(_bot_config.get('ai_review', True)):
            try:
                from ai_agent import ollama_health
                health = ollama_health()
            except Exception as exc:
                log.error("Ollama preflight import failed: %s", exc)
                return jsonify({
                    'ok': False,
                    'error': (
                        f"Agent layer (ai_agent) failed to import: {exc}. "
                        f"Run `python scripts/preflight.py` for details, or "
                        f"disable AI review in the sidebar to start strategy-only."
                    ),
                }), 500
            if not health.get('reachable'):
                return jsonify({
                    'ok': False,
                    'error': (
                        f"Ollama is not reachable at {health.get('url')}: "
                        f"{health.get('error') or 'unknown error'}. "
                        f"Run scripts/setup_ollama.ps1, or disable AI review "
                        f"in the sidebar to start strategy-only."
                    ),
                }), 503
            if not health.get('model_loaded'):
                return jsonify({
                    'ok': False,
                    'error': (
                        f"Ollama model '{health.get('model')}' is not pulled. "
                        f"Run: ollama pull {health.get('model')}"
                    ),
                }), 503

        ai_pro_path = os.path.join(root_path, 'ai_pro.py')
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.join(root_path, 'core') + os.pathsep + env.get('PYTHONPATH', '')
        env['BOT_AUTOSTART_SYMBOLS'] = ','.join(sorted(_enabled_symbols))
        env['BOT_AUTOSTART_VOLUME'] = str(_bot_config.get('volume', 0.50))
        env['BOT_AUTOSTART_POLL_SECS'] = str(_bot_config.get('poll_interval', 300))
        env['BOT_AUTOSTART_AUTO_TRADE'] = '1' if (
            bool(_bot_config.get('auto_trade', True)) and not bool(_bot_config.get('dry_run', False))
        ) else '0'
        env['BOT_AUTOSTART_USE_AI'] = '1' if bool(_bot_config.get('ai_review', True)) else '0'
        env['BOT_AUTOSTART_ATR_MULT'] = str(_bot_config.get('atr_mult', 1.5))
        env['BOT_AUTOSTART_SL_MULT'] = str(_bot_config.get('sl_mult', 2.5))
        env['BOT_AUTOSTART_TP_MULT'] = str(_bot_config.get('tp_mult', 4.5))
        bot_process = subprocess.Popen(
            [sys.executable, ai_pro_path, '--run'],
            cwd=root_path,
            env=env
        )
        # Give the process a moment to fail fast (missing deps, bad args, etc.)
        time.sleep(0.25)
        rc = bot_process.poll()
        if rc is not None:
            bot_running = False
            log.error("AI Pro bot failed to start (exit code: %s)", rc)
            return jsonify({'ok': False, 'error': f'Agent Zero could not wake up (bot exited with code {rc}). Check the bot console/log output.'}), 500

        bot_running = True
        log.info("AI Pro bot started (PID: %d)", bot_process.pid)
        return jsonify({
            'ok': True,
            'running': True,
            'pid': bot_process.pid,
            'config': {
                'symbols': sorted(_enabled_symbols),
                'volume': _bot_config.get('volume', 0.50),
                'poll_interval': _bot_config.get('poll_interval', 300),
                'dry_run': bool(_bot_config.get('dry_run', False)),
                'ai_review': bool(_bot_config.get('ai_review', True)),
                'auto_trade': bool(_bot_config.get('auto_trade', True)),
                'sl_mult': _bot_config.get('sl_mult', 2.5),
                'tp_mult': _bot_config.get('tp_mult', 4.5),
                'atr_mult': _bot_config.get('atr_mult', 1.5),
            },
        })
    except Exception as e:
        log.error(f"Failed to start bot: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/bot/stop', methods=['POST'])
def bot_stop():
    """Stop the AI Pro bot"""
    global bot_process, bot_running
    
    if bot_process and bot_process.poll() is None:
        bot_process.terminate()
        try:
            bot_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            bot_process.kill()
        log.info("AI Pro bot stopped")
    
    bot_running = False
    return jsonify({'ok': True, 'running': False})

@app.route('/bot/status', methods=['GET'])
def bot_status():
    """Get bot status with live market data for the Market Watch tab."""
    global bot_process, bot_running
    running = bot_process and bot_process.poll() is None
    if not running:
        bot_running = False

    # Sync MT5 connection state
    _sync_mt5_status()
    mt5_payload = {
        'connected': bool(mt5_status.get('mt5_connected')),
        'error': mt5_status.get('error'),
        'account': mt5_status.get('account'),
    }

    # ── Live open trades from MT5 ──────────────────────────────────────────
    open_trades = []
    if mt5_conn and mt5_conn.is_connected():
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get() or []
            for pos in positions:
                open_trades.append({
                    'symbol':      pos.symbol,
                    'direction':   'BUY' if pos.type == 0 else 'SELL',
                    'volume':      pos.volume,
                    'entry_price': round(pos.price_open, 5),
                    'stop_loss':   round(pos.sl, 5) if pos.sl else None,
                    'take_profit': round(pos.tp, 5) if pos.tp else None,
                    'profit':      round(pos.profit + pos.swap, 2),
                    'ticket':      pos.ticket,
                })
        except Exception as exc:
            log.debug("bot_status: positions fetch failed: %s", exc)

    # ── Last signal/bias/confidence from AI thought log ────────────────────
    market_bias  = None
    confidence   = None
    last_signal  = None
    try:
        recent = get_thoughts(limit=200)
        # Most recent execution → last_signal
        for t in reversed(recent):
            if t.get('source') == 'execution':
                sym = t.get('symbol', '')
                act = t.get('action', '').upper()
                if sym and act in ('BUY', 'SELL', 'buy', 'sell'):
                    last_signal = f"{sym} {act.upper()}"
                else:
                    # Parse from summary e.g. "Market order #xxx SELL @ ..."
                    import re as _re
                    m = _re.search(r'(BUY|SELL)', t.get('summary', ''), _re.IGNORECASE)
                    if m and sym:
                        last_signal = f"{sym} {m.group(1).upper()}"
                if last_signal:
                    break
        # Most recent strategy or ai_entry with a BUY/SELL action → bias
        for t in reversed(recent):
            src = t.get('source', '')
            if src in ('strategy', 'ai_pro_signal', 'ai_entry') and t.get('action') in ('buy', 'sell'):
                market_bias  = t['action'].upper()
                confidence   = t.get('confidence')
                break
    except Exception as exc:
        log.debug("bot_status: thought parse failed: %s", exc)

    # ── Session summary ────────────────────────────────────────────────────
    pairs_str = ', '.join(sorted(_enabled_symbols)) or 'GBPJPY, EURJPY, GBPUSD, EURUSD'
    if running:
        n_pos = len(open_trades)
        session_summary = (
            f"Portfolio: {pairs_str} | Lot: 0.50 | "
            f"Open: {n_pos} position{'s' if n_pos != 1 else ''} | Status: Running"
        )
    else:
        session_summary = f"Portfolio: {pairs_str} | Ready to start"

    return jsonify({
        'ok': True,
        'bot': {
            'running':          bool(running),
            'pid':              bot_process.pid if (running and bot_process) else None,
            'open_trades':      open_trades,
            'market_bias':      market_bias,
            'confidence':       confidence,
            'last_signal':      last_signal,
            'session_summary':  session_summary,
        },
        'mt5': mt5_payload,
    })


@app.route('/bot/signal/<symbol>', methods=['GET'])
def bot_signal(symbol: str):
    """Return a lightweight signal snapshot for the dashboard Signals tab."""
    try:
        if symbol.upper() not in _enabled_symbols:
            return jsonify({"error": f"{symbol.upper()} disabled"}), 404

        snap = _mt5_signal_snapshot(symbol)
        if snap.get("error"):
            # 503 (Service Unavailable) is semantically correct here: the
            # client's request is well-formed, but the upstream MT5 terminal
            # isn't connected yet. 400 (Bad Request) would falsely imply
            # the client sent something malformed, and turns the normal
            # "haven't connected yet" state into red console spam.
            return jsonify(snap), 503
        return jsonify(snap)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route('/bot/config/symbols', methods=['POST'])
def bot_config_symbols():
    """Enable/disable symbols from the Portfolio Watch cards."""
    try:
        payload = request.get_json(silent=True) or {}
        symbol = str(payload.get("symbol", "")).upper().strip()
        enabled = bool(payload.get("enabled"))
        if not symbol:
            return jsonify({"success": False, "message": "Missing symbol"}), 400

        if enabled:
            _enabled_symbols.add(symbol)
        else:
            _enabled_symbols.discard(symbol)

        return jsonify({"success": True, "enabled_symbols": sorted(_enabled_symbols)})
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500
from collections import deque
import threading
import json as _json

# ── Shared thought log ────────────────────────────────────────────────────
# The trading bot (ai_pro.py) runs as a separate subprocess, so we can't
# share a Python deque. Instead, both processes read/write a shared JSONL
# file. The server only reads it; the bot appends.
_THOUGHT_LOG_PATH = os.path.join(root_path, "ai_thoughts.jsonl")
_thoughts_lock = threading.Lock()
# In-memory cache of the most recent entries, rebuilt from disk on each read
# when the file has changed. Keeps reads fast for the 2s dashboard poll.
_thoughts_cache: deque = deque(maxlen=500)
_thoughts_cache_mtime: float = 0.0


def _reload_thoughts_from_file() -> None:
    """Refresh the in-memory cache from the shared JSONL file if it has changed."""
    global _thoughts_cache_mtime
    try:
        st = os.stat(_THOUGHT_LOG_PATH)
    except FileNotFoundError:
        # No bot has written yet — leave cache as-is (likely empty)
        return
    except Exception as exc:
        log.debug("thought log stat failed: %s", exc)
        return

    if st.st_mtime <= _thoughts_cache_mtime:
        return  # file hasn't changed

    entries: list = []
    try:
        with open(_THOUGHT_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(_json.loads(line))
                except Exception:
                    # Skip malformed lines (e.g. partial write in progress)
                    continue
    except Exception as exc:
        log.debug("thought log read failed: %s", exc)
        return

    with _thoughts_lock:
        _thoughts_cache.clear()
        _thoughts_cache.extend(entries[-_thoughts_cache.maxlen:])
        _thoughts_cache_mtime = st.st_mtime


def log_thought(source, symbol, stage, summary, detail=None, action=None, confidence=None):
    """Append a thought to the shared JSONL file (used if the server itself logs)."""
    try:
        from datetime import datetime, timezone
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "symbol": symbol,
            "stage": stage,
            "summary": summary,
            "detail": detail or "",
            "action": action or "",
            "confidence": round(confidence, 2) if confidence is not None else None,
        }
        line = _json.dumps(entry, ensure_ascii=False, default=str)
        with _thoughts_lock:
            with open(_THOUGHT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        log.debug("log_thought failed: %s", exc)


def get_thoughts(since_ts=None, limit=60):
    _reload_thoughts_from_file()
    with _thoughts_lock:
        items = list(_thoughts_cache)
    if since_ts:
        items = [t for t in items if str(t.get("ts", "")) > since_ts]
    return items[-limit:]


def clear_thoughts():
    """Clear the shared thought log (truncates the file)."""
    global _thoughts_cache_mtime
    try:
        with _thoughts_lock:
            # Truncate rather than delete so file handles remain valid
            with open(_THOUGHT_LOG_PATH, "w", encoding="utf-8") as f:
                f.truncate(0)
            _thoughts_cache.clear()
            _thoughts_cache_mtime = 0.0
    except Exception as exc:
        log.error("clear_thoughts failed: %s", exc)

@app.route('/bot/ai_thoughts', methods=['GET'])
def bot_ai_thoughts():
    try:
        since = request.args.get("since")
        limit = min(int(request.args.get("limit", 60)), 120)
        return jsonify({"ok": True, "thoughts": get_thoughts(since_ts=since, limit=limit)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "thoughts": []}), 500

@app.route('/bot/thoughts/clear', methods=['POST'])
def clear_thoughts_api():
    clear_thoughts()
    return jsonify({"ok": True})

def _empty_kpis() -> dict:
    return {
        'win_rate': 0.0, 'profit_factor': 0.0, 'total_trades': 0,
        'equity_return': 0.0, 'max_drawdown': 0.0, 'sharpe': 0.0,
        'avg_win_loss_ratio': 0.0, 'current_drawdown': 0.0,
        'recovery_factor': 0.0, 'sortino': 0.0,
        # Same shape as the populated path so the dashboard can read it
        # without null-checks. Both buckets default to a clean zero row.
        'direction_breakdown': {
            'BUY':  {'count': 0, 'wins': 0, 'losses': 0,
                     'win_rate': 0.0, 'total_pnl': 0.0},
            'SELL': {'count': 0, 'wins': 0, 'losses': 0,
                     'win_rate': 0.0, 'total_pnl': 0.0},
        },
        # Supporting numbers for the KPI subtitle lines.
        'wins_count': 0, 'losses_count': 0,
        'gross_profit': 0.0, 'gross_loss': 0.0,
        'total_pnl': 0.0,
        'max_drawdown_abs': 0.0, 'current_drawdown_abs': 0.0,
        'avg_win_abs': 0.0, 'avg_loss_abs': 0.0,
        'open_positions': 0, 'open_pnl': 0.0,
    }


@app.route('/bot/positions', methods=['GET'])
def bot_positions():
    """Return live open positions from MT5."""
    global mt5_conn
    if not mt5_conn or not mt5_conn.is_connected():
        return jsonify({'ok': True, 'positions': []})
    try:
        import MetaTrader5 as mt5
        positions = mt5.positions_get()
        if positions is None:
            return jsonify({'ok': True, 'positions': []})
        result = []
        for pos in positions:
            result.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == 0 else 'SELL',
                'volume': pos.volume,
                'open_price': round(pos.price_open, 5),
                'current_price': round(pos.price_current, 5),
                'sl': round(pos.sl, 5) if pos.sl else None,
                'tp': round(pos.tp, 5) if pos.tp else None,
                'profit': round(pos.profit, 2),
                'swap': round(pos.swap, 2),
                'open_time': datetime.fromtimestamp(pos.time, tz=timezone.utc).isoformat(),
                'comment': pos.comment,
                'magic': pos.magic,
            })
        return jsonify({'ok': True, 'positions': result})
    except Exception as exc:
        log.error("Error fetching positions: %s", exc)
        return jsonify({'ok': True, 'positions': [], 'error': 'Failed to retrieve positions'})


@app.route('/bot/history', methods=['GET'])
def bot_history():
    """Return recent closed-trade history from MT5 (last 30 days).

    Implementation note (May 2026 fix)
    ----------------------------------
    Earlier versions iterated `history_deals_get(...)` and emitted one
    row per closing deal (DEAL_ENTRY_OUT). That row reported:
      * `type`  from the closing deal — which is the *opposite* of the
                position's direction. Closing a BUY needs a SELL order,
                so every BUY trade was rendered as "SELL" in the UI.
      * `price` from the closing deal — which is the *exit* (SL/TP fill),
                not the entry.

    Both numbers correctly described "the order that closed the position",
    but the dashboard widget (and the operator) reads them as "the trade
    you took" — entry direction at entry price. So we now pair every
    closing deal (`entry ∈ {OUT, OUT_BY, INOUT}`) with the opening deal
    (`entry == IN`) sharing the same `position_id` and surface:
      * `type`        from the IN deal       (original direction)
      * `entry_price` from the IN deal       (actual fill at entry)
      * `exit_price`  from the OUT deal      (SL / TP / market close)
      * `time`        = exit_time            (when the trade closed)
      * `entry_time`  = IN deal's time
      * `profit/swap/commission` summed across all OUT deals for the
        position so a partial-close + final-close still reports total P&L.
    """
    global mt5_conn
    if not mt5_conn or not mt5_conn.is_connected():
        return jsonify({'ok': True, 'trades': []})
    try:
        import MetaTrader5 as mt5
        from datetime import timedelta
        from collections import defaultdict

        date_from = datetime.now(timezone.utc) - timedelta(days=30)
        date_to   = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(date_from, date_to)
        if deals is None:
            return jsonify({'ok': True, 'trades': []})

        # MT5 deal.entry codes — name them so the intent is obvious.
        DEAL_ENTRY_IN     = 0
        DEAL_ENTRY_OUT    = 1
        DEAL_ENTRY_INOUT  = 2   # netting close-and-reverse
        DEAL_ENTRY_OUT_BY = 3   # close by opposite position
        OUT_CODES = {DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT, DEAL_ENTRY_OUT_BY}

        # Group every deal by position_id so we can pair entries to exits.
        by_position: "dict[int, list]" = defaultdict(list)
        for d in deals:
            pid = int(getattr(d, "position_id", 0) or 0)
            if pid:
                by_position[pid].append(d)

        trades = []
        for pid, plist in by_position.items():
            ins  = [d for d in plist if d.entry == DEAL_ENTRY_IN]
            outs = [d for d in plist if d.entry in OUT_CODES]
            if not ins or not outs:
                # Position is still open (no OUT yet) or otherwise
                # malformed — skip; open positions are surfaced by
                # /bot/positions, not /bot/history.
                continue

            in_deal   = min(ins,  key=lambda d: d.time)   # earliest IN
            out_deal  = max(outs, key=lambda d: d.time)   # last OUT (final close)

            trades.append({
                'ticket':       int(pid),                    # broker position id
                'order':        int(getattr(out_deal, "order", 0) or 0),
                'symbol':       in_deal.symbol,
                # Direction comes from the OPENING deal — that's what
                # the operator placed. (deal.type: 0=BUY, 1=SELL.)
                'type':         'BUY' if in_deal.type == 0 else 'SELL',
                'volume':       float(in_deal.volume),
                # Entry price = where the trade actually filled.
                'price':        round(float(in_deal.price), 5),
                'entry_price':  round(float(in_deal.price), 5),
                'exit_price':   round(float(out_deal.price), 5),
                # Sum P&L across every OUT deal so partial closes are
                # included in the total.
                'profit':       round(sum(float(d.profit or 0)     for d in outs), 2),
                'swap':         round(sum(float(d.swap or 0)       for d in outs), 2),
                'commission':   round(sum(float(d.commission or 0) for d in outs), 2),
                # `time` is the close time so the dashboard's "recent
                # trades" sort keeps the most-recently-closed first.
                'time':         datetime.fromtimestamp(out_deal.time, tz=timezone.utc).isoformat(),
                'entry_time':   datetime.fromtimestamp(in_deal.time,  tz=timezone.utc).isoformat(),
                'comment':      in_deal.comment or out_deal.comment,
            })

        trades.sort(key=lambda x: x['time'], reverse=True)
        return jsonify({'ok': True, 'trades': trades[:100]})
    except Exception as exc:
        log.error("Error fetching history: %s", exc)
        return jsonify({'ok': True, 'trades': [], 'error': 'Failed to retrieve trade history'})


@app.route('/bot/performance', methods=['GET'])
def bot_performance():
    """Calculate performance KPIs and equity curve from MT5 history (last 90 days)."""
    global mt5_conn
    if not mt5_conn or not mt5_conn.is_connected():
        return jsonify({'ok': True, 'kpis': _empty_kpis(), 'equity_curve': []})
    try:
        import MetaTrader5 as mt5
        import statistics
        from datetime import timedelta
        date_from = datetime.now(timezone.utc) - timedelta(days=90)
        date_to = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(date_from, date_to)
        if deals is None or len(deals) == 0:
            return jsonify({'ok': True, 'kpis': _empty_kpis(), 'equity_curve': []})
        closed = [d for d in deals if d.entry == 1]
        if not closed:
            return jsonify({'ok': True, 'kpis': _empty_kpis(), 'equity_curve': []})

        # mt5.history_deals_get does NOT guarantee chronological order; sort
        # explicitly so the cumulative equity curve is monotonic-in-time.
        try:
            closed.sort(key=lambda d: getattr(d, 'time', 0))
        except Exception:
            pass

        profits = [d.profit + d.swap + d.commission for d in closed]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        total = len(profits)
        win_rate = (len(wins) / total * 100) if total > 0 else 0.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        # Equity curve
        cumulative, running = [], 0.0
        for p in profits:
            running += p
            cumulative.append(round(running, 2))

        # Determine starting equity from account balance
        account = mt5.account_info()
        initial_balance = ((account.balance - running) if account else 10000.0) or 10000.0
        equity_return = (running / initial_balance * 100) if initial_balance > 0 else 0.0

        # Max drawdown
        peak, max_dd = 0.0, 0.0
        for val in cumulative:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd
        base = initial_balance + max(cumulative) if cumulative else initial_balance
        max_dd_pct = (max_dd / base * 100) if base > 0 else 0.0
        current_val = cumulative[-1] if cumulative else 0.0
        current_base = initial_balance + (peak if peak > 0 else 0.0)
        current_dd_pct = ((peak - current_val) / current_base * 100) if (current_base > 0 and peak > current_val) else 0.0

        # Sharpe / Sortino (annualised, per-trade approximation)
        sharpe, sortino = 0.0, 0.0
        if len(profits) > 1:
            mean_p = statistics.mean(profits)
            std_p = statistics.stdev(profits)
            sharpe = round((mean_p / std_p * (252 ** 0.5)) if std_p > 0 else 0.0, 2)
            neg = [p for p in profits if p < 0]
            if len(neg) > 1:
                down_std = statistics.stdev(neg)
                sortino = round((mean_p / down_std * (252 ** 0.5)) if down_std > 0 else 0.0, 2)

        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (abs(sum(losses)) / len(losses)) if losses else 0.0
        avg_ratio = (avg_win / avg_loss) if avg_loss > 0 else (999.0 if avg_win > 0 else 0.0)
        # Recovery factor = net P&L / max drawdown. Keep the sign so a
        # net-losing strategy shows a negative (or zero) recovery, not a
        # misleading positive number.
        recovery = (running / max_dd) if max_dd > 0 else 0.0

        # ── Direction breakdown (BUY vs SELL) ─────────────────────────
        # Pair every closing deal with its opening deal (same logic the
        # /bot/history fix uses) so we can attribute each closed trade
        # to its ENTRY direction. Reading `closing_deal.type` directly
        # gives the inverse — closing a BUY emits a SELL order.
        from collections import defaultdict
        DEAL_ENTRY_IN     = 0
        DEAL_ENTRY_OUT    = 1
        DEAL_ENTRY_INOUT  = 2
        DEAL_ENTRY_OUT_BY = 3
        OUT_CODES = {DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT, DEAL_ENTRY_OUT_BY}

        by_position: "dict[int, list]" = defaultdict(list)
        for d in deals:
            pid = int(getattr(d, "position_id", 0) or 0)
            if pid:
                by_position[pid].append(d)

        dir_buckets = {
            "BUY":  {"count": 0, "wins": 0, "losses": 0, "total_pnl": 0.0},
            "SELL": {"count": 0, "wins": 0, "losses": 0, "total_pnl": 0.0},
        }
        for pid, plist in by_position.items():
            ins  = [d for d in plist if d.entry == DEAL_ENTRY_IN]
            outs = [d for d in plist if d.entry in OUT_CODES]
            if not ins or not outs:
                continue   # still open or malformed — skip
            in_deal = min(ins, key=lambda d: d.time)
            side = "BUY" if in_deal.type == 0 else "SELL"
            pnl  = sum(float(d.profit or 0)
                       + float(d.swap or 0)
                       + float(d.commission or 0) for d in outs)
            b = dir_buckets[side]
            b["count"]     += 1
            b["total_pnl"] += pnl
            if pnl > 0:
                b["wins"] += 1
            elif pnl < 0:
                b["losses"] += 1
            # zero P&L (rare — exact BE) is counted but not as win or loss

        # Round + add win-rate so the UI doesn't have to compute it.
        for side, b in dir_buckets.items():
            decided = b["wins"] + b["losses"]
            b["win_rate"]  = round((b["wins"] / decided * 100), 1) if decided else 0.0
            b["total_pnl"] = round(b["total_pnl"], 2)

        # ── Per-pair breakdown ─────────────────────────────────────────
        PAIRS_ORDER = ["EURUSD", "GBPUSD", "GBPJPY", "EURJPY"]
        pair_buckets: "dict[str, dict]" = {
            sym: {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0,
                  "buys": 0, "sells": 0}
            for sym in PAIRS_ORDER
        }
        for pid, plist in by_position.items():
            ins  = [d for d in plist if d.entry == DEAL_ENTRY_IN]
            outs = [d for d in plist if d.entry in OUT_CODES]
            if not ins or not outs:
                continue
            in_deal = min(ins, key=lambda d: d.time)
            sym = (getattr(in_deal, "symbol", "") or "").upper()
            if sym not in pair_buckets:
                continue
            pnl  = sum(float(d.profit or 0) + float(d.swap or 0)
                       + float(d.commission or 0) for d in outs)
            side = "buys" if in_deal.type == 0 else "sells"
            b = pair_buckets[sym]
            b["trades"]    += 1
            b["total_pnl"] += pnl
            b[side]        += 1
            if pnl > 0:
                b["wins"] += 1
            elif pnl < 0:
                b["losses"] += 1
        pair_stats = []
        for sym in PAIRS_ORDER:
            b = pair_buckets[sym]
            decided = b["wins"] + b["losses"]
            pair_stats.append({
                "symbol":    sym,
                "trades":    b["trades"],
                "wins":      b["wins"],
                "losses":    b["losses"],
                "win_rate":  round((b["wins"] / decided * 100), 1) if decided else 0.0,
                "total_pnl": round(b["total_pnl"], 2),
                "buys":      b["buys"],
                "sells":     b["sells"],
            })

        # Open-positions count — surfaced on the Total Trades card so the
        # operator can see "3 closed · 1 live" at a glance instead of just
        # the closed count.
        try:
            open_positions = list(mt5.positions_get() or [])
            open_count = len(open_positions)
            open_pnl   = float(sum(getattr(p, "profit", 0.0) or 0.0
                                    for p in open_positions))
        except Exception:
            open_count = 0
            open_pnl   = 0.0

        # Current floating drawdown in $: distance from the equity peak
        # to the current realised + floating equity. The percentage in
        # current_dd_pct above is computed against equity peak only;
        # we also surface the raw $ figure for the subtitle.
        current_dd_abs = max(0.0, peak - current_val)

        kpis = {
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'total_trades': total,
            'equity_return': round(equity_return, 2),
            'max_drawdown': round(max_dd_pct, 2),
            'sharpe': round(sharpe, 2),
            'avg_win_loss_ratio': round(avg_ratio, 2),
            'current_drawdown': round(current_dd_pct, 2),
            'recovery_factor': round(recovery, 2),
            'sortino': round(sortino, 2),
            # Per-direction wins/losses/P&L. Rendered as a "BUY vs SELL"
            # KPI card on the Performance page so the operator can see
            # at a glance whether the edge is symmetric or one-sided.
            'direction_breakdown': dir_buckets,
            # ── Supporting numbers for the KPI subtitle lines ────────
            # Each existing card now has a "WW/LL", "+$X / -$Y", etc.
            # row beneath the headline value. These fields feed those
            # rows. Kept on the same payload (vs. a second endpoint) so
            # the dashboard renders the whole Performance page from one
            # request — no flicker, no extra network round-trip.
            'wins_count':           len(wins),
            'losses_count':         len(losses),
            'gross_profit':         round(gross_profit, 2),
            'gross_loss':           round(gross_loss,   2),
            'total_pnl':            round(running,      2),     # net realised $
            'max_drawdown_abs':     round(max_dd,       2),     # peak-to-trough $
            'current_drawdown_abs': round(current_dd_abs, 2),   # peak-to-now $
            'avg_win_abs':          round(avg_win,      2),
            'avg_loss_abs':         round(avg_loss,     2),
            'open_positions':       int(open_count),
            'open_pnl':             round(open_pnl,     2),     # floating $
        }
        return jsonify({'ok': True, 'kpis': kpis, 'equity_curve': cumulative[-200:], 'pair_stats': pair_stats})
    except Exception as exc:
        log.error("Error calculating performance: %s", exc)
        return jsonify({'ok': True, 'kpis': _empty_kpis(), 'equity_curve': [], 'error': 'Failed to calculate performance'})


# In-memory config store (persists for the server lifetime)
_bot_config: dict = {}


def _apply_bot_config_payload(payload: dict) -> None:
    """Persist dashboard bot config and enabled symbols into process memory."""
    global _enabled_symbols, _bot_config
    if not isinstance(payload, dict):
        return

    if payload.get('symbols'):
        syms = [s.strip().upper() for s in str(payload['symbols']).split(',') if s.strip()]
        if syms:
            _enabled_symbols = set(syms)

    for key in ('volume', 'poll_interval', 'dry_run', 'ai_review', 'auto_trade',
                'sl_mult', 'tp_mult', 'atr_mult', 'pc_rr', 'be_buffer'):
        if key in payload:
            _bot_config[key] = payload[key]


@app.route('/bot/config', methods=['POST'])
def bot_config_apply():
    """Apply configuration from the dashboard sidebar."""
    try:
        payload = request.get_json(silent=True) or {}
        _apply_bot_config_payload(payload)
        return jsonify({
            'ok': True,
            'applied': {
                'symbols': sorted(_enabled_symbols),
                **{k: _bot_config.get(k) for k in ('volume', 'poll_interval', 'dry_run',
                                                     'ai_review', 'auto_trade', 'sl_mult',
                                                     'tp_mult', 'atr_mult', 'pc_rr', 'be_buffer')},
            },
        })
    except Exception as exc:
        log.error("Config apply error: %s", exc)
        return jsonify({'ok': False, 'error': 'Failed to apply configuration'}), 500


# ─── Agent (Orchestrator + pair bots) endpoints ────────────────────────────
#
# These three endpoints feed the Agents tab and are also referenced by the
# Signals tab and Zero Log header. They were missing from the server, which
# caused the dashboard to show "invalid response from /api/ollama/h…" — the
# frontend's fetch wrapper falls back to that string when /api/ollama/health
# fetches alongside these missing endpoints fail to parse JSON.
#
# All three are READ-only views of state already maintained by ai_agent.py:
#   /api/agent/matrix     → bot config + last verdicts + heatmap + memory
#   /api/agent/portfolio  → portfolio snapshot (risk, alignment, exposure)
#   /api/agent/directives → POST to update freeze/close/exposure/notes

PAIR_SYMBOLS = ("EURUSD", "GBPUSD", "GBPJPY", "EURJPY")

_latest_verdicts_by_symbol: dict = {}
_last_orch_confidence: float | None = None

# Path to the shared thought log written by the bot subprocess
_THOUGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai_thoughts.jsonl")


import json as _thoughts_json
import re  as _re


def _refresh_verdicts_from_thoughts() -> None:
    """Read ai_thoughts.jsonl and populate _latest_verdicts_by_symbol.

    The bot writes one thought entry per verdict cycle.  We scan the file
    for the most recent entry per pair that carries an action field.
    """
    global _latest_verdicts_by_symbol, _last_orch_confidence
    try:
        if not os.path.exists(_THOUGHTS_PATH):
            return
        latest: dict = {}
        with open(_THOUGHTS_PATH, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _thoughts_json.loads(line)
                except Exception:
                    continue
                sym    = (entry.get("symbol") or "").upper()
                action = (entry.get("action") or "").strip()
                if sym not in PAIR_SYMBOLS or not action:
                    continue
                ts = entry.get("ts") or ""
                if sym not in latest or ts > latest[sym].get("ts", ""):
                    latest[sym] = entry

        new_verdicts: dict = {}
        for sym, e in latest.items():
            new_verdicts[sym] = {
                "action":      e.get("action", "hold"),
                "confidence":  e.get("confidence"),
                "reason_code": _extract_reason_code_from_thought(e),
                "atr_profit":  _extract_atr_from_thought(e),
                "ts":          e.get("ts"),
                "summary":     (e.get("summary") or "")[:80],
            }
        _latest_verdicts_by_symbol = new_verdicts

        # Orch confidence = most recent entry from orchestrator source
        orch_conf = None
        orch_ts   = ""
        with open(_THOUGHTS_PATH, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _thoughts_json.loads(line)
                except Exception:
                    continue
                src = (entry.get("source") or "").lower()
                if ("orch" in src or "agent 0" in src) and entry.get("confidence") is not None:
                    ts = entry.get("ts") or ""
                    if ts > orch_ts:
                        orch_ts   = ts
                        orch_conf = entry.get("confidence")
        if orch_conf is not None:
            _last_orch_confidence = float(orch_conf)
    except Exception as exc:
        log.debug("_refresh_verdicts_from_thoughts failed: %s", exc)


def _extract_reason_code_from_thought(e: dict) -> str:
    """Pull ORC_* reason code from summary or detail text if present."""
    for key in ("summary", "detail"):
        m = _re.search(r'\bORC_\w+\b', e.get(key) or "")
        if m:
            return m.group(0)
    return ""


def _extract_atr_from_thought(e: dict) -> "float | None":
    """Pull atr_profit value from detail text, e.g. 'atr_p=1.23R'."""
    for key in ("detail", "summary"):
        m = _re.search(r'atr_p[=:]\s*([\d.]+)', e.get(key) or "")
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def _bot_config_for_symbol(sym: str) -> dict:
    """Read pair-bot tunables — live tuned params from backtest_insights.json
    merged with class-level constants so the dashboard shows what the bot
    will actually use on the next trade."""
    base: dict = {}
    try:
        from ai_agent import _PAIR_BOT_CLASSES
        cls = _PAIR_BOT_CLASSES.get(sym.upper())
        if cls is not None:
            base = {
                "min_atr_to_tighten": float(cls.MIN_ATR_PROFIT_TO_TIGHTEN),
                "trail_atr_mult":     float(cls.TRAIL_ATR_MULT),
                "close_on_trend":     bool(cls.CLOSE_ON_TREND_BROKEN),
                "close_on_structure": bool(cls.CLOSE_ON_STRUCTURE_BROKEN),
            }
    except Exception:
        pass

    # Overlay with live tuned params (from backtest_insights.json via learning loop)
    try:
        from agent_learning_loop import AgentLearningLoop, get_learning_loop
        ll = get_learning_loop() if callable(get_learning_loop) else AgentLearningLoop()
        tp = ll.get_tuned_params(sym.upper()) or {}
        if tp:
            base["sl_atr_mult"]      = round(float(tp.get("sl_atr_mult",      base.get("sl_atr_mult",      0))), 4)
            base["tp_atr_mult"]      = round(float(tp.get("tp_atr_mult",      base.get("tp_atr_mult",      0))), 4)
            base["trail_atr_mult"]   = round(float(tp.get("trail_atr_mult",   base.get("trail_atr_mult",   0))), 4)
            base["be_buffer_pips"]   = round(float(tp.get("be_buffer_pips",   base.get("be_buffer_pips",   0))), 2)
            base["partial_close_rr"] = round(float(tp.get("partial_close_rr", base.get("partial_close_rr",0))), 2)
            base["min_atr_to_tighten"] = round(float(tp.get("min_atr_to_tighten", base.get("min_atr_to_tighten", 0))), 4)
            base["tuned"] = True
    except Exception:
        base["tuned"] = False

    return base


@app.route('/api/agent/matrix', methods=['GET'])
def api_agent_matrix():
    """Snapshot of the multi-agent system: config, verdicts, heatmap, memory."""
    global bot_process, _latest_verdicts_by_symbol, _last_orch_confidence

    _refresh_verdicts_from_thoughts()   # pull latest actions from thought log

    running = bool(bot_process and bot_process.poll() is None)

    bots = {sym: _bot_config_for_symbol(sym) for sym in PAIR_SYMBOLS}

    # Heatmap + memory + directives come from the orchestrator singleton.
    heatmap = {"vetoes": {s: 0 for s in PAIR_SYMBOLS},
               "overrides": {s: 0 for s in PAIR_SYMBOLS}}
    memory_summary = "No prior decisions this session."
    directives_payload = {"freeze": [], "close": [], "max_exposure_pct": 0.0, "notes": ""}

    try:
        from ai_agent import get_orchestrator, get_directives
        orch = get_orchestrator()
        try:
            heatmap = orch.memory.heatmap_data()
        except Exception as exc:
            log.debug("heatmap fetch failed: %s", exc)
        try:
            memory_summary = orch.memory.summary_text()
        except Exception as exc:
            log.debug("memory summary failed: %s", exc)

        d = get_directives()
        directives_payload = {
            "freeze":           sorted(d.freeze_symbols),
            "close":            sorted(d.close_symbols),
            "max_exposure_pct": float(d.max_exposure_pct),
            "notes":            d.notes,
        }
    except Exception as exc:
        log.debug("orchestrator state fetch failed: %s", exc)

    # Mirror the latest Ollama health into the matrix so the frontend can
    # render the orchestrator's LLM-readiness in one shot.
    ollama_payload = {"reachable": False, "model_loaded": False,
                      "model": None, "latency_ms": None}
    try:
        from ai_agent import ollama_health
        ollama_payload = ollama_health()
    except Exception as exc:
        log.debug("ollama_health for matrix failed: %s", exc)

    return jsonify({
        "running":          running,
        "bots":             bots,
        "latest_verdicts":  _latest_verdicts_by_symbol,
        "orch_confidence":  _last_orch_confidence,
        "heatmap":          heatmap,
        "memory_summary":   memory_summary,
        "directives":       directives_payload,
        "ollama":           ollama_payload,
    })


@app.route('/api/agent/edge', methods=['GET'])
def api_agent_edge():
    """Orchestrator edge stats — how much value VETO/OVERRIDE interventions add."""
    try:
        from agent_memory import AgentMemory
        import sqlite3
        mem = AgentMemory()
        with mem._conn() as conn:
            rows = conn.execute("""
                SELECT
                    d.action_bot,
                    d.action_taken,
                    o.outcome,
                    o.profit_usd,
                    d.trend_intact,
                    d.structure_broken,
                    d.symbol
                FROM decisions d
                JOIN outcomes o ON o.decision_id = d.id
                WHERE o.outcome IN ('WIN','LOSS','BE')
            """).fetchall()

        total = approved = vetoed = overridden = 0
        approve_wins = approve_losses = 0
        approve_pnl  = 0.0
        veto_wins    = veto_losses    = 0
        veto_pnl     = 0.0
        override_wins= override_losses= 0
        override_pnl = 0.0
        pa_stats: dict = {}   # (trend_intact, structure_broken) -> {win,loss,pnl}

        for r in rows:
            total += 1
            bot    = r["action_bot"]
            taken  = r["action_taken"]
            out    = r["outcome"]
            pnl    = float(r["profit_usd"] or 0.0)
            changed = bot != taken

            if not changed:
                approved += 1
                if out == "WIN":   approve_wins   += 1; approve_pnl += pnl
                elif out == "LOSS":approve_losses += 1; approve_pnl += pnl
            elif taken == "hold" and bot in ("close","move_sl"):
                vetoed += 1
                if out == "WIN":   veto_wins   += 1; veto_pnl += pnl
                elif out == "LOSS":veto_losses += 1; veto_pnl += pnl
            elif taken == "close" and bot == "hold":
                overridden += 1
                if out == "WIN":   override_wins   += 1; override_pnl += pnl
                elif out == "LOSS":override_losses += 1; override_pnl += pnl

            # Price action conditional stats
            ti = r["trend_intact"]
            sb = r["structure_broken"]
            if ti is not None and sb is not None:
                key = (int(ti), int(sb))
                if key not in pa_stats:
                    pa_stats[key] = {"win": 0, "loss": 0, "pnl": 0.0}
                if out == "WIN":   pa_stats[key]["win"]  += 1; pa_stats[key]["pnl"] += pnl
                elif out == "LOSS":pa_stats[key]["loss"] += 1; pa_stats[key]["pnl"] += pnl

        def _wr(w, l): return round(w / (w+l) * 100, 1) if (w+l) > 0 else None

        pa_list = []
        labels = {(1,0):"trend=OK struct=OK", (1,1):"trend=OK struct=BRK",
                  (0,0):"trend=BRK struct=OK",(0,1):"trend=BRK struct=BRK"}
        for key, d in sorted(pa_stats.items()):
            n = d["win"] + d["loss"]
            pa_list.append({
                "state":    labels.get(key, str(key)),
                "wins":     d["win"],  "losses": d["loss"],
                "win_rate": _wr(d["win"], d["loss"]),
                "net_pnl":  round(d["pnl"], 2),
                "n":        n,
            })

        return jsonify({
            "ok": True,
            "total_labelled":   total,
            "approved":  {"n": approved,   "wins": approve_wins,  "losses": approve_losses,  "win_rate": _wr(approve_wins,  approve_losses),  "net_pnl": round(approve_pnl,  2)},
            "vetoed":    {"n": vetoed,     "wins": veto_wins,     "losses": veto_losses,     "win_rate": _wr(veto_wins,     veto_losses),     "net_pnl": round(veto_pnl,     2)},
            "overridden":{"n": overridden, "wins": override_wins, "losses": override_losses, "win_rate": _wr(override_wins, override_losses), "net_pnl": round(override_pnl, 2)},
            "price_action": pa_list,
        })
    except Exception as exc:
        log.error("api_agent_edge error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)})


@app.route('/api/agent/portfolio', methods=['GET'])
def api_agent_portfolio():
    """Live PortfolioSnapshot for the risk dial / alignment / exposure cards."""
    global mt5_conn

    payload = {
        "equity":             0.0,
        "balance":            0.0,
        "daily_pl":           0.0,
        "daily_loss_limit":   0.0,
        "daily_loss_breach":  False,
        "open_count":         0,
        "max_positions_total":0,
        "risk_budget_pct":    0.0,
        "alignment_score":    0.0,
        "correlation_load":   0.0,
        "pair_exposure":      {},
    }

    if not mt5_conn or not mt5_conn.is_connected():
        return jsonify(payload)

    try:
        import MetaTrader5 as mt5
        account = mt5.account_info()
        if account:
            payload["equity"]  = float(getattr(account, "equity", 0.0))
            payload["balance"] = float(getattr(account, "balance", 0.0))
            payload["daily_pl"] = round(payload["equity"] - payload["balance"], 2)

        positions = mt5.positions_get() or []
        payload["open_count"] = len(positions)

        # Per-pair exposure: lots, SL distance in pips, % equity at risk.
        pair_exp = {}
        sides = []
        for pos in positions:
            sym = pos.symbol.upper()
            digits = 3 if "JPY" in sym else 5
            pip = 10 ** -(digits - 1)
            sl_dist_pips = (abs(pos.price_open - pos.sl) / pip) if pos.sl else 0.0
            entry = pair_exp.setdefault(sym, {"lots": 0.0, "sl_pips": 0.0,
                                              "risk_pct": 0.0, "profit": 0.0})
            entry["lots"]    += float(pos.volume)
            entry["sl_pips"]  = max(entry["sl_pips"], sl_dist_pips)
            entry["profit"]  += float(pos.profit + pos.swap)
            sides.append("BUY" if pos.type == 0 else "SELL")

        payload["pair_exposure"] = pair_exp

        # Correlation load: fraction of positions on the dominant side.
        if sides:
            dom = max(sides.count("BUY"), sides.count("SELL"))
            payload["correlation_load"] = round(
                (dom - 1) / len(sides) if len(sides) > 1 else 0.0, 2)
    except Exception as exc:
        log.debug("portfolio snapshot failed: %s", exc)

    return jsonify(payload)


@app.route('/api/agent/directives', methods=['POST'])
def api_agent_directives():
    """Update GlobalDirectives from the dashboard's Directives form."""
    try:
        from ai_agent import GlobalDirectives, set_directives
    except Exception as exc:
        return jsonify({"ok": False, "error": f"agent module import failed: {exc}"}), 500

    try:
        data = request.get_json(silent=True) or {}
        freeze = {str(s).upper() for s in (data.get("freeze") or []) if s}
        close_ = {str(s).upper() for s in (data.get("close")  or []) if s}
        try:
            max_exp = float(data.get("max_exposure_pct") or 0)
        except Exception:
            max_exp = 0.0
        notes = str(data.get("notes") or "")[:300]

        d = GlobalDirectives(
            freeze_symbols=freeze,
            close_symbols=close_,
            max_exposure_pct=max(0.0, max_exp),
            notes=notes,
        )
        set_directives(d)
        return jsonify({"ok": True, "summary": d.summary()})
    except Exception as exc:
        log.error("directives update failed: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route('/api/agent/tuning', methods=['GET'])
def api_agent_tuning():
    """Per-pair self-tuning strategy params, keyed by symbol.

    Reads the live values the bot uses each poll cycle. When a backtest
    finishes, agent_learning_loop auto-reloads the new tunings, so this
    endpoint always reflects the currently-active per-pair strategy.

    Response shape:
        {
          "loaded":     bool,                     # any insights loaded?
          "file_exists": bool,
          "last_mtime": ISO timestamp | null,
          "pairs": {
            "EURUSD": {
              "params": {
                "atr_tolerance_mult": 1.5,
                "sl_atr_mult":        2.5,
                ...
              },
              "source":      "backtest" | "default",
              "backtest_at": ISO timestamp | null,
              "trade_count": int | null,
              "win_rate":    float | null
            },
            ...
          }
        }
    """
    payload = {
        "loaded":      False,
        "file_exists": False,
        "last_mtime":  None,
        "pairs":       {},
    }
    try:
        from agent_learning_loop import get_learning_loop
        loop = get_learning_loop()
        st = loop.status()
        payload["loaded"]      = bool(loop.has_insights())
        payload["file_exists"] = bool(st.get("file_exists"))
        payload["last_mtime"]  = st.get("last_mtime")
        payload["pairs"]       = loop.get_all_tuned_params()
    except Exception as exc:
        log.debug("agent tuning fetch failed: %s", exc)
        payload["error"] = str(exc)
    return jsonify(payload)


# ─── Backtest endpoint ─────────────────────────────────────────────────────
#
# POST /api/backtest/run {pairs?: [...], days?: int, lot_size?: float}
#
#   Runs odl.backtest.AgentZeroBacktester for each pair and returns an aggregate
#   summary plus per-pair breakdown — exactly what the BACKTEST tab renders.
#   Guarded by a module-level lock so two callers can't kick off parallel
#   backtests (each run fetches historical MT5 data and is heavyweight).

_BACKTEST_DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "EURJPY", "GBPJPY"]
_backtest_lock = _threading.Lock()


def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x) * 100.0:.1f}%"
    except Exception:
        return "0.0%"


def _fmt_money(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    sign = "+" if v >= 0 else "-"
    return f"{sign}${abs(v):,.2f}"


@app.route('/api/backtest/run', methods=['POST'])
def api_backtest_run():
    """Run a backtest across selected pairs and return aggregate + per-pair."""
    import time

    if not _backtest_lock.acquire(blocking=False):
        return jsonify({
            "ok": False,
            "error": "Another backtest is already running — please wait for it to finish."
        }), 409

    t0 = time.time()
    try:
        payload = request.get_json(silent=True) or {}
        pairs = payload.get("pairs") or _BACKTEST_DEFAULT_PAIRS
        pairs = [str(p).strip().upper() for p in pairs if str(p).strip()]
        if not pairs:
            pairs = list(_BACKTEST_DEFAULT_PAIRS)

        try:
            days = int(payload.get("days", 7))
        except Exception:
            days = 7
        days = max(1, min(days, 60))

        try:
            lot_size = float(payload.get("lot_size", 0.50))
        except Exception:
            lot_size = 0.50
        lot_size = max(0.01, min(lot_size, 100.0))

        # Import inside the handler so a server start without MT5 installed
        # still serves the other endpoints.
        try:
            from odl.backtest import AgentZeroBacktester
            from ai_pro import AgentZeroBot
        except Exception as exc:
            log.error("backtest import failed: %s", exc)
            return jsonify({
                "ok": False,
                "error": f"Backtester import failed: {exc}",
            }), 500

        per_pair = []
        period = ""
        total_trades = 0
        total_wins = 0
        total_losses = 0
        total_pnl = 0.0

        # One strategy instance shared across all pairs — MT5 init is expensive.
        strategy = AgentZeroBot(use_ai=False)
        for symbol in pairs:
            log.info("[BACKTEST] %s days=%d lot=%.2f", symbol, days, lot_size)
            try:
                # Pure strategy validation — whole position exits at first
                # SL/TP/timeout so we're measuring the raw signal edge.
                # (Trade-management overlay lives only in the live bot.)
                bt = AgentZeroBacktester(strategy=strategy, lot_size=lot_size)
                result = bt.run(symbol, days=days)
            except Exception as exc:
                log.exception("backtest failed for %s", symbol)
                per_pair.append({
                    "symbol": symbol,
                    "error": f"{exc}",
                    "trades": 0, "wins": 0, "losses": 0,
                    "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
                    "profit_factor": 0.0, "avg_rr": 0.0,
                })
                continue

            if not isinstance(result, dict) or result.get("error"):
                per_pair.append({
                    "symbol": symbol,
                    "error": (result or {}).get("error", "unknown backtester error"),
                    "trades": 0, "wins": 0, "losses": 0,
                    "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
                    "profit_factor": 0.0, "avg_rr": 0.0,
                })
                continue

            trades = int(result.get("total_trades", 0) or 0)
            wins   = int(result.get("wins", 0) or 0)
            losses = int(result.get("losses", 0) or 0)
            pnl    = float(result.get("total_pnl", 0.0) or 0.0)
            avg    = float(result.get("avg_pnl", 0.0) or 0.0)

            total_trades += trades
            total_wins   += wins
            total_losses += losses
            total_pnl    += pnl
            period = result.get("period", period)

            per_pair.append({
                "symbol":         symbol,
                "trades":         trades,
                "wins":           wins,
                "losses":         losses,
                "win_rate":       float(result.get("win_rate", 0.0) or 0.0),
                "total_pnl":      round(pnl, 2),
                "avg_pnl":        round(avg, 2),
                "profit_factor":  float(result.get("profit_factor", 0.0) or 0.0),
                "avg_rr":         float(result.get("avg_rr_achieved", 0.0) or 0.0),
                "avg_win_pips":   float(result.get("avg_win_pips", 0.0) or 0.0),
                "avg_loss_pips":  float(result.get("avg_loss_pips", 0.0) or 0.0),
                "win_rate_label": _fmt_pct(result.get("win_rate", 0.0) or 0.0),
                "pnl_label":      _fmt_money(pnl),
            })

        aggregate_win_rate = (total_wins / total_trades) if total_trades > 0 else 0.0
        aggregate_avg      = (total_pnl / total_trades) if total_trades > 0 else 0.0

        return jsonify({
            "ok":        True,
            "duration_s": round(time.time() - t0, 2),
            "pairs":     pairs,
            "days":      days,
            "lot_size":  lot_size,
            "period":    period,
            "aggregate": {
                "total_trades":  total_trades,
                "wins":          total_wins,
                "losses":        total_losses,
                "win_rate":      round(aggregate_win_rate, 4),
                "win_rate_label": _fmt_pct(aggregate_win_rate),
                "total_pnl":     round(total_pnl, 2),
                "total_pnl_label": _fmt_money(total_pnl),
                "avg_pnl":       round(aggregate_avg, 2),
                "avg_pnl_label": _fmt_money(aggregate_avg),
            },
            "per_pair":  per_pair,
        })
    except Exception as exc:
        log.exception("backtest run error")
        return jsonify({"ok": False, "error": f"{exc}"}), 500
    finally:
        try:
            strategy.shutdown()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        _backtest_lock.release()


# ─── Backtest streaming endpoint (SSE) ─────────────────────────────────────
#
# GET /api/backtest/stream?pairs=EURUSD,GBPUSD&days=7&lot_size=0.5
#
#   Returns text/event-stream. The client opens an EventSource, which only
#   supports GET + query params (that's why pairs comes in as a CSV).
#   A worker thread runs the backtest; progress events are pushed onto a
#   queue and flushed to the client as SSE messages. Final event is 'done'.

import queue as _queue
import json as _json

def _sse_pack(event: str, data) -> str:
    """Format a single SSE message. data is JSON-encoded."""
    payload = _json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


@app.route('/api/backtest/stream', methods=['GET'])
def api_backtest_stream():
    """Stream backtest progress + results as Server-Sent Events."""
    # Only one backtest at a time — reuse the same lock as the POST endpoint.
    if not _backtest_lock.acquire(blocking=False):
        def _busy():
            yield _sse_pack("error", {"error": "Another backtest is already running."})
        return Response(stream_with_context(_busy()),
                        mimetype='text/event-stream',
                        status=409)

    # Parse query params
    raw_pairs = (request.args.get("pairs") or "").strip()
    if raw_pairs:
        pairs = [p.strip().upper() for p in raw_pairs.split(",") if p.strip()]
    else:
        pairs = list(_BACKTEST_DEFAULT_PAIRS)
    if not pairs:
        pairs = list(_BACKTEST_DEFAULT_PAIRS)

    try:
        days = int(request.args.get("days", 7))
    except Exception:
        days = 7
    days = max(1, min(days, 60))

    try:
        lot_size = float(request.args.get("lot_size", 0.50))
    except Exception:
        lot_size = 0.50
    lot_size = max(0.01, min(lot_size, 100.0))

    # Shared queue: worker pushes events, generator yields them.
    q: "_queue.Queue[dict]" = _queue.Queue(maxsize=2000)
    _SENTINEL = {"__sentinel__": True}

    def _worker():
        import time
        t0 = time.time()
        try:
            try:
                from odl.backtest import AgentZeroBacktester
            except Exception as exc:
                q.put({"__event__": "error", "error": f"Backtester import failed: {exc}"})
                return

            def _push(ev: dict):
                try:
                    q.put(ev, timeout=2.0)
                except Exception:
                    pass

            _push({"__event__": "start",
                   "pairs": pairs, "days": days, "lot_size": lot_size})

            per_pair = []
            period = ""
            total_trades = 0
            total_wins = 0
            total_losses = 0
            total_pnl = 0.0

            # One strategy instance shared across all pairs so MT5 is only
            # initialised once. Shut it down in the finally block below.
            try:
                from ai_pro import AgentZeroBot as _AZB
                _strategy = _AZB(use_ai=False)
            except Exception as exc:
                _push({"__event__": "error", "error": f"Strategy init failed: {exc}"})
                return

            for symbol in pairs:
                log.info("[BACKTEST-SSE] %s days=%d lot=%.2f", symbol, days, lot_size)
                _push({"__event__": "pair_start", "symbol": symbol})

                try:
                    # Pure strategy validation — whole position exits at
                    # the first SL/TP/timeout so the numbers measure the
                    # raw signal itself, not the management overlay.
                    bt = AgentZeroBacktester(strategy=_strategy, lot_size=lot_size)
                    result = bt.run(
                        symbol,
                        days=days,
                        progress_cb=lambda p, s=symbol: _push({
                            "__event__": "progress",
                            "symbol": s,
                            **p,
                        }),
                    )
                except Exception as exc:
                    log.exception("backtest failed for %s", symbol)
                    per_pair.append({
                        "symbol": symbol,
                        "error": str(exc),
                        "total_trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "win_rate": 0.0,
                        "total_pnl": 0.0,
                    })
                    _push({"__event__": "pair_done",
                           "symbol": symbol, "error": str(exc)})
                    continue

                if "error" in result:
                    per_pair.append({
                        "symbol": symbol,
                        "error": result["error"],
                        "total_trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "win_rate": 0.0,
                        "total_pnl": 0.0,
                    })
                    _push({"__event__": "pair_done",
                           "symbol": symbol, "error": result["error"]})
                    continue

                # Accumulate totals
                pt = int(result.get("total_trades", 0) or 0)
                pw = int(result.get("wins", 0) or 0)
                pl = int(result.get("losses", 0) or 0)
                pp = float(result.get("total_pnl", 0.0) or 0.0)
                total_trades += pt
                total_wins += pw
                total_losses += pl
                total_pnl += pp
                if not period:
                    period = result.get("period", "")

                pair_entry = {
                    "symbol": symbol,
                    "total_trades": pt,
                    "wins": pw,
                    "losses": pl,
                    "win_rate": float(result.get("win_rate", 0.0) or 0.0),
                    "total_pnl": round(pp, 2),
                    "avg_pnl_per_trade": float(result.get("avg_pnl", 0.0) or 0.0),
                    "profit_factor": float(result.get("profit_factor", 0.0) or 0.0),
                    "avg_rr_achieved": float(result.get("avg_rr_achieved", 0.0) or 0.0),
                    "avg_win_pips": float(result.get("avg_win_pips", 0.0) or 0.0),
                    "avg_loss_pips": float(result.get("avg_loss_pips", 0.0) or 0.0),
                    "feature_importance": result.get("feature_importance"),
                    # Modelling assumptions used for this pair (entry fill,
                    # spread, intrabar policy, pip-USD model, etc).
                    "assumptions": result.get("assumptions"),
                    # Win/loss correlation breakdowns — plotted in the
                    # BACKTEST tab so the user can see which environment,
                    # hour, weekday, and confidence band correlated with
                    # winning vs losing trades.
                    "by_env":              result.get("by_env") or {},
                    "by_side":             result.get("by_side") or {},
                    "by_hour":             result.get("by_hour") or {},
                    "by_dow":              result.get("by_dow") or {},
                    # Calendar window this pair's trades came from. The
                    # dashboard renders it in the panel subtitles so the
                    # operator can see WHICH dates each "Mon"/"Tue" bar
                    # represents — a 4-Monday sample tells you very
                    # different things from a 26-Monday sample.
                    "date_range":          result.get("date_range") or None,
                    "confidence_buckets":  result.get("confidence_buckets") or {},
                    "pnl_distribution":    result.get("pnl_distribution") or {},
                    "component_correlations": result.get("component_correlations") or {},
                    "by_quality":          result.get("by_quality") or {},
                    "by_exit_reason":      result.get("by_exit_reason") or {},
                    # Max drawdown (peak-to-trough) for this pair, plus
                    # the trade-by-trade cum-P&L stream so the aggregate
                    # can be recomputed correctly across pairs.
                    "max_drawdown":        float(result.get("max_drawdown", 0.0) or 0.0),
                    "max_drawdown_pct":    float(result.get("max_drawdown_pct", 0.0) or 0.0),
                    "equity_curve":        result.get("equity_curve") or [],
                }
                per_pair.append(pair_entry)
                _push({"__event__": "pair_done", **pair_entry})

            # Aggregate ML across every decided trade we saw.
            aggregate_fi = None
            try:
                # Reuse the last backtester instance's compute — but it holds
                # its own trades list. Aggregate by reconstructing from
                # per_pair + self.trades would require passing a handle; the
                # simpler path is to collect fresh from bt.trades (the last
                # bt in the loop). We instead aggregate via a one-off.
                from odl.backtest import AgentZeroBacktester as _BT
                agg_bt = _BT(lot_size=lot_size)
                all_trades = []
                # Concatenate trades from each per-pair backtest via a second,
                # cheap feature-importance pass: gather raw trades from the
                # LAST bt instance only. (This is a known compromise — per-pair
                # FI is already included above; the aggregate FI is best-effort.)
                aggregate_fi = None
            except Exception:
                aggregate_fi = None

            aggregate_win_rate = (total_wins / total_trades) if total_trades else 0.0
            aggregate_avg = (total_pnl / total_trades) if total_trades else 0.0

            # ── Portfolio max drawdown ────────────────────────────
            # Summing per-pair MDDs overstates the damage because pair
            # drawdowns don't occur simultaneously. The honest way is to
            # merge every pair's trade P&L stream in chronological order
            # and run one equity curve on the combined flow.
            # equity_curve from analyze_results is List[float] (cumulative P&L),
            # not List[{"t":..., "pnl":...}].  Convert incremental deltas so
            # the running-drawdown loop below works with both formats.
            agg_stream: list[float] = []
            for p in per_pair:
                prev_pnl = 0.0
                for pt in (p.get("equity_curve") or []):
                    if isinstance(pt, dict):
                        agg_stream.append(float(pt.get("pnl", 0.0)))
                    else:
                        cur = float(pt or 0.0)
                        agg_stream.append(cur - prev_pnl)
                        prev_pnl = cur
            agg_running = 0.0
            agg_peak    = 0.0
            agg_mdd     = 0.0
            for pnl in agg_stream:
                agg_running += pnl
                if agg_running > agg_peak:
                    agg_peak = agg_running
                dd = agg_peak - agg_running
                if dd > agg_mdd:
                    agg_mdd = dd
            agg_mdd_pct = ((agg_mdd / agg_peak) * 100.0) if agg_peak > 1e-9 else 0.0

            _push({
                "__event__": "done",
                "ok": True,
                "duration_s": round(time.time() - t0, 2),
                "pairs": pairs,
                "days": days,
                "lot_size": lot_size,
                "period": period,
                "aggregate": {
                    "total_trades": total_trades,
                    "wins": total_wins,
                    "losses": total_losses,
                    "win_rate": aggregate_win_rate,
                    "win_rate_label": _fmt_pct(aggregate_win_rate),
                    "total_pnl": round(total_pnl, 2),
                    "total_pnl_label": _fmt_money(total_pnl),
                    "avg_pnl_per_trade": round(aggregate_avg, 2),
                    "avg_pnl_label": _fmt_money(aggregate_avg),
                    # MDD is reported as a non-negative magnitude; the
                    # label prefixes "-" so the card reads like a loss.
                    "max_drawdown":       round(agg_mdd, 2),
                    "max_drawdown_label": (f"-${agg_mdd:,.2f}"
                                           if agg_mdd > 0 else "$0.00"),
                    "max_drawdown_pct":   round(agg_mdd_pct, 2),
                },
                "per_pair": per_pair,
            })
        except Exception as exc:
            log.exception("backtest stream error")
            try:
                q.put({"__event__": "error", "error": f"{exc}"})
            except Exception:
                pass
        finally:
            try:
                _strategy.shutdown()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            try:
                q.put(_SENTINEL)
            except Exception:
                pass

    worker = _threading.Thread(target=_worker, name="backtest-sse", daemon=True)
    worker.start()

    def _generate():
        try:
            # Initial hello so proxies flush headers immediately
            yield _sse_pack("hello", {"ok": True})
            while True:
                try:
                    ev = q.get(timeout=30.0)
                except _queue.Empty:
                    # Keep connection alive through long CPU stretches
                    yield ": keepalive\n\n"
                    continue
                if ev is _SENTINEL or ev.get("__sentinel__"):
                    break
                name = ev.pop("__event__", "message")
                yield _sse_pack(name, ev)
        finally:
            _backtest_lock.release()

    return Response(
        stream_with_context(_generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache, no-transform',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )


# ── Agent Zero Memory API ─────────────────────────────────────────────────────
@app.route('/api/agent/memory')
def agent_memory_stats():
    """Return Agent Zero memory stats for the Agents tab."""
    try:
        from agent_memory import get_memory as _gam
        mem = _gam()
        stats = mem.stats()
        ctx   = mem.build_context()
        return jsonify({"ok": True, "stats": stats, "context_preview": ctx[:600]})
    except ImportError:
        return jsonify({"ok": False, "error": "agent_memory not available"}), 503
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


# ── Trade Memory API ─────────────────────────────────────────────────────────
@app.route('/api/backtest/memory')
def backtest_memory():
    """
    Return cumulative trade-memory stats from SQLite for all four pairs.
    Called by the dashboard's memory panel on the Backtest tab.
    Response shape:
        {
          "ok": true,
          "pairs": {
            "EURUSD": {n_runs, total_trades, wins, losses, win_rate,
                        total_pnl, earliest, latest, importances, tuned_params},
            ...
          }
        }
    """
    PAIRS = ["EURUSD", "GBPUSD", "EURJPY", "GBPJPY"]
    try:
        from odl.trade_memory import get_memory as _gm
        mem = _gm()
    except ImportError:
        return jsonify({"ok": False, "error": "trade_memory not available"}), 503

    result = {}
    for sym in PAIRS:
        try:
            stats = mem.get_cumulative_stats(sym)
            # JS reads total_wins/total_losses/earliest/latest;
            # get_cumulative_stats returns wins/losses/earliest_run/latest_run
            stats["total_wins"]   = stats.get("wins",         0)
            stats["total_losses"] = stats.get("losses",       0)
            stats["earliest"]     = stats.get("earliest_run")
            stats["latest"]       = stats.get("latest_run")
            # Also pull latest aggregated importances & tuned params
            agg = mem.aggregate_insights(sym)
            stats["importances"] = agg.get("importances") or {}
            stats["tuned_params"] = agg.get("tuned_params") or {}
            stats["n_runs"] = agg.get("n_runs", stats.get("n_runs", 0))
            result[sym] = stats
        except Exception as exc:
            result[sym] = {"symbol": sym, "error": str(exc), "n_runs": 0}

    return jsonify({"ok": True, "pairs": result})


if __name__ == '__main__':
    # Important: disable the auto-reloader so long-running bot subprocesses
    # started from HTTP requests are not interrupted by Flask restarts.
    # threaded=True is critical: /bot/signal/<symbol> instantiates AgentZeroBot and
    # runs generate_trade_signal(), which can take several seconds. Without
    # threading, four parallel signal fetches queue serially and BLOCK every
    # other endpoint (status, positions, ai_thoughts) — that's what made the
    # whole dashboard appear blank while the bot was running fine in the
    # background.
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False,
            threaded=True)
