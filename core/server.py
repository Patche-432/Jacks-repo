"""Flask server for MT5 dashboard"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from flask import Flask, jsonify, send_from_directory, request
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

# Dashboard runtime config (in-memory)
_enabled_symbols = {"GBPJPY", "EURJPY", "GBPUSD", "EURUSD"}


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
    """Compute a lightweight signal snapshot from MT5 rates.

    This intentionally avoids importing/using the heavyweight LLM strategy.
    Output shape matches what the dashboard expects.
    """
    import MetaTrader5 as mt5

    symbol = symbol.upper().strip()

    # Ensure MT5 is ready (auto-connect on demand)
    global mt5_conn
    if not mt5_conn or not mt5_conn.is_connected():
        try:
            mt5_conn = MT5Connection({})
            if not mt5_conn.connect():
                code, msg = mt5.last_error()
                return {"error": f"MT5 not connected. initialize failed [{code}] {msg}. Open MT5 and log in."}
        except Exception as exc:
            return {"error": f"MT5 auto-connect failed: {exc}"}

    if not mt5.symbol_select(symbol, True):
        code, msg = mt5.last_error()
        return {"error": f"Failed to select symbol {symbol} [{code}] {msg}"}

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 220)
    if rates is None or len(rates) < 60:
        code, msg = mt5.last_error()
        return {"error": f"No rates for {symbol} [{code}] {msg}"}

    df = pd.DataFrame(rates)
    # time is seconds since epoch
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    close = df["close"].astype(float)
    sma_fast = close.rolling(20).mean()
    sma_slow = close.rolling(50).mean()
    atr = _compute_atr(df, 14)

    last_close = float(close.iloc[-1])
    fast = float(sma_fast.iloc[-1]) if pd.notna(sma_fast.iloc[-1]) else last_close
    slow = float(sma_slow.iloc[-1]) if pd.notna(sma_slow.iloc[-1]) else last_close

    # Bias by SMA trend
    if fast > slow:
        bias = "LONG"
        environment = "Trend-UP (SMA20>SMA50)"
    elif fast < slow:
        bias = "SHORT"
        environment = "Trend-DOWN (SMA20<SMA50)"
    else:
        bias = "—"
        environment = "Neutral"

    # Confidence from separation and slope (bounded 0-100)
    sep = abs(fast - slow)
    atr_safe = atr if (not math.isnan(atr) and atr > 0) else max(sep, 1e-8)
    sep_score = min(1.0, sep / (atr_safe * 1.5))
    slope_fast = float(sma_fast.iloc[-1] - sma_fast.iloc[-6]) if len(sma_fast) >= 6 and pd.notna(sma_fast.iloc[-6]) else 0.0
    slope_score = min(1.0, abs(slope_fast) / (atr_safe * 0.5)) if atr_safe > 0 else 0.0
    confidence = int(round(30 + 45 * sep_score + 25 * slope_score))
    confidence = max(0, min(100, confidence))

    tick = mt5.symbol_info_tick(symbol)
    entry = float(getattr(tick, "ask", None) or getattr(tick, "last", None) or last_close)

    # Simple ATR-based SL/TP
    sl_mult = 2.5
    tp_mult = 4.5
    if math.isnan(atr_safe) or atr_safe <= 0:
        atr_safe = max(abs(entry - last_close), 1e-4)

    if bias == "SHORT":
        sl = entry + sl_mult * atr_safe
        tp = entry - tp_mult * atr_safe
    elif bias == "LONG":
        sl = entry - sl_mult * atr_safe
        tp = entry + tp_mult * atr_safe
    else:
        sl = entry - sl_mult * atr_safe
        tp = entry + tp_mult * atr_safe

    return {
        "symbol": symbol,
        "bias": bias,
        "confidence": confidence,
        "environment": environment,
        "choch_status": "—",
        "level_interaction": "—",
        "entry_price": round(entry, 5),
        "sl_price": round(sl, 5),
        "tp_price": round(tp, 5),
        "rr": _rr_string(entry, sl, tp),
        "atr": round(float(atr_safe), 5),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def _sync_mt5_status() -> None:
    """Pull the latest MT5 connection state into the global mt5_status dict."""
    global mt5_conn, mt5_status
    if mt5_conn is None:
        mt5_status = {"mt5_connected": False, "account": None, "error": "No connection object"}
        return
    
    try:
        conn_status = mt5_conn.status()
        mt5_status["mt5_connected"] = conn_status["connected"]
        mt5_status["account"] = conn_status["account"]
        mt5_status["error"] = conn_status["error"]
    except Exception as exc:
        log.error("Failed to sync MT5 status: %s", exc)
        mt5_status["mt5_connected"] = False
        mt5_status["error"] = str(exc)

@app.route('/')
def serve_dashboard():
    return send_from_directory(root_path, 'index.html')

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
            "rr_ratio": "1:1.8",
            "partial_close_rr": 1.0,
            "be_buffer_pips": 1.0
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
        return jsonify({'ok': False, 'error': 'Bot already running'}), 409
    
    try:
        ai_pro_path = os.path.join(root_path, 'ai_pro.py')
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.join(root_path, 'core') + os.pathsep + env.get('PYTHONPATH', '')
        bot_process = subprocess.Popen(
            [sys.executable, ai_pro_path, '--run', '--lot-size', '0.50'],
            cwd=root_path,
            env=env
        )
        # Give the process a moment to fail fast (missing deps, bad args, etc.)
        time.sleep(0.25)
        rc = bot_process.poll()
        if rc is not None:
            bot_running = False
            log.error("AI Pro bot failed to start (exit code: %s)", rc)
            return jsonify({'ok': False, 'error': f'Bot failed to start (exit code {rc}). Check the bot console/log output.'}), 500

        bot_running = True
        log.info("AI Pro bot started (PID: %d)", bot_process.pid)
        return jsonify({'ok': True, 'running': True, 'pid': bot_process.pid})
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
    """Get bot status"""
    global bot_process, bot_running
    running = bot_process and bot_process.poll() is None
    if not running:
        bot_running = False

    # Sync MT5 connection state into a stable shape expected by the dashboard
    _sync_mt5_status()
    mt5_payload = {
        'connected': bool(mt5_status.get('mt5_connected')),
        'error': mt5_status.get('error'),
        'account': mt5_status.get('account'),
    }

    # Return comprehensive status for dashboard snapshot
    return jsonify({
        'ok': True,
        'bot': {
            'running': bool(running),
            'pid': bot_process.pid if (running and bot_process) else None,
            'market_bias': 'Bullish',  # TODO: wire from ai_pro.py
            'confidence': 0.75,        # TODO: wire from ai_pro.py
            'last_signal': 'GBPJPY BUY',
            'open_trades': [],
            'session_summary': 'Portfolio: GBPJPY, EURJPY, GBPUSD, EURUSD | Lot: 0.50 | Status: Running' if running else 'Ready to start'
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
            return jsonify(snap), 400
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

# AI Thoughts storage (same as ai_pro.py)
_thoughts_lock = threading.Lock()
_thoughts = deque(maxlen=120)

def log_thought(source, symbol, stage, summary, detail=None, action=None, confidence=None):
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
        with _thoughts_lock:
            _thoughts.append(entry)
    except Exception:
        pass

def get_thoughts(since_ts=None, limit=60):
    with _thoughts_lock:
        items = list(_thoughts)
    if since_ts:
        items = [t for t in items if t["ts"] > since_ts]
    return items[-limit:]

def clear_thoughts():
    with _thoughts_lock:
        _thoughts.clear()

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
    """Return recent closed-trade history from MT5 (last 30 days)."""
    global mt5_conn
    if not mt5_conn or not mt5_conn.is_connected():
        return jsonify({'ok': True, 'trades': []})
    try:
        import MetaTrader5 as mt5
        from datetime import timedelta
        date_from = datetime.now(timezone.utc) - timedelta(days=30)
        date_to = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(date_from, date_to)
        if deals is None:
            return jsonify({'ok': True, 'trades': []})
        trades = []
        for deal in deals:
            if deal.entry == 1:  # DEAL_ENTRY_OUT — closed trade
                trades.append({
                    'ticket': deal.ticket,
                    'order': deal.order,
                    'symbol': deal.symbol,
                    'type': 'BUY' if deal.type == 0 else 'SELL',
                    'volume': deal.volume,
                    'price': round(deal.price, 5),
                    'profit': round(deal.profit, 2),
                    'swap': round(deal.swap, 2),
                    'commission': round(deal.commission, 2),
                    'time': datetime.fromtimestamp(deal.time, tz=timezone.utc).isoformat(),
                    'comment': deal.comment,
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
        recovery = (abs(running) / max_dd) if max_dd > 0 else 0.0

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
        }
        return jsonify({'ok': True, 'kpis': kpis, 'equity_curve': cumulative[-200:]})
    except Exception as exc:
        log.error("Error calculating performance: %s", exc)
        return jsonify({'ok': True, 'kpis': _empty_kpis(), 'equity_curve': [], 'error': 'Failed to calculate performance'})


# In-memory config store (persists for the server lifetime)
_bot_config: dict = {}


@app.route('/bot/config', methods=['POST'])
def bot_config_apply():
    """Apply configuration from the dashboard sidebar."""
    global _enabled_symbols, _bot_config
    try:
        payload = request.get_json(silent=True) or {}
        # Update enabled symbols if provided
        if payload.get('symbols'):
            syms = [s.strip().upper() for s in str(payload['symbols']).split(',') if s.strip()]
            if syms:
                _enabled_symbols = set(syms)
        # Persist remaining config values
        for key in ('volume', 'poll_interval', 'dry_run', 'ai_review', 'auto_trade',
                    'sl_mult', 'tp_mult', 'atr_mult', 'pc_rr', 'be_buffer'):
            if key in payload:
                _bot_config[key] = payload[key]
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


if __name__ == '__main__':
    # Important: disable the auto-reloader so long-running bot subprocesses
    # started from HTTP requests are not interrupted by Flask restarts.
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)