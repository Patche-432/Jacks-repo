"""Flask server for MT5 dashboard"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import logging
from core.mt5_connection import MT5Connection

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

mt5_conn = None

@app.route('/')
def serve_dashboard():
    """Serve the main dashboard"""
    return send_from_directory('.', 'index.html')

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

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)