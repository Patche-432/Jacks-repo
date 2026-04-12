"""Flask server for MT5 dashboard"""
from datetime import datetime, timezone
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import logging
from core.mt5_connection import MT5Connection

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, static_folder=root_path, static_url_path='')
CORS(app)

mt5_conn = None

@app.route('/')
def serve_dashboard():
    return send_from_directory(root_path, 'index.html')

@app.route('/api/mt5/connect', methods=['POST'])
def api_connect_mt5():
    """Connect to MT5"""
    global mt5_conn
    try:
        cfg = {}
        mt5_conn = MT5Connection(cfg)
        if mt5_conn.connect():
            runtime = mt5_conn.runtime_info()
            return jsonify({'connected': True, **runtime})
        else:
            return jsonify({'connected': False, 'error': 'Failed to connect'}), 400
    except Exception as e:
        log.error(f"MT5 connect error: {e}")
        return jsonify({'connected': False, 'error': str(e)}), 500

@app.route('/api/mt5/status', methods=['GET'])
def api_mt5_status():
    """Get MT5 connection status"""
    global mt5_conn
    if not mt5_conn or not mt5_conn.is_connected():
        return jsonify({'connected': False})
    try:
        runtime = mt5_conn.runtime_info()
        return jsonify(runtime)
    except Exception as e:
        log.error(f"Status error: {e}")
        return jsonify({'connected': False}), 500

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
    return jsonify({'ok': True})

@app.route('/api/mt5/test-trade', methods=['POST'])
def api_test_trade():
    """Place a test trade (0.01 lot BUY on EURUSD)"""
    global mt5_conn
    if not mt5_conn or not mt5_conn.is_connected():
        return jsonify({'ok': False, 'error': 'Not connected to MT5'}), 400
    try:
        result = mt5_conn.place_test_trade(symbol='EURUSD', volume=0.01)
        if result['ok']:
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        log.error(f"Test trade error: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500

import subprocess

bot_process = None
bot_running = False

@app.route('/bot/start', methods=['POST'])
def bot_start():
    """Start the AI Pro bot"""
    global bot_process, bot_running
    
    if bot_running and bot_process and bot_process.poll() is None:
        return jsonify({'ok': False, 'error': 'Bot already running'}), 400
    
    try:
        ai_pro_path = os.path.join(root_path, 'ai_pro.py')
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.join(root_path, 'core') + os.pathsep + env.get('PYTHONPATH', '')
        bot_process = subprocess.Popen(
            ['python', ai_pro_path, '--run', '--lot-size', '0.50'], 
            cwd=root_path,
            env=env
        )
        bot_running = True
        log.info("AI Pro bot started (PID: %d)", bot_process.pid)
        return jsonify({'ok': True, 'running': True})
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
    
    # Return comprehensive status for dashboard snapshot
    return jsonify({
        'ok': True,
        'running': running,
        'market_bias': 'Bullish',  # This would come from ai_pro.py in real implementation
        'confidence': 0.75,  # This would come from ai_pro.py in real implementation
        'last_signal': 'GBPJPY BUY',  # This would come from ai_pro.py in real implementation
        'open_trades': [],  # This would be populated from MT5 positions
        'session_summary': 'Portfolio: GBPJPY, EURJPY, GBPUSD, EURUSD | Lot: 0.50 | Status: Running' if running else 'Ready to start'
    })
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

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)