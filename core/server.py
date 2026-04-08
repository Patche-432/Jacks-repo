"""Flask server for MT5 dashboard"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import logging
from mt5_connection import MT5Connection

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

@app.route('/api/mt5/disconnect', methods=['POST'])
def api_disconnect_mt5():
    """Disconnect from MT5"""
    global mt5_conn
    if mt5_conn:
        mt5_conn.disconnect()
        mt5_conn = None
    return jsonify({'ok': True})

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
            ['python', ai_pro_path, '--run'], 
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
    if bot_process and bot_process.poll() is None:
        return jsonify({'ok': True, 'running': True})
    else:
        bot_running = False
        return jsonify({'ok': True, 'running': False})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)