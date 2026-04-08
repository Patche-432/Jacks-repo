"""Simple Flask server — serves the FORTIS dashboard and handles MT5 connections."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

from core.mt5_connection import MT5Connection

log = logging.getLogger(__name__)

# Repo root is one level above this file (core/)
ROOT = Path(__file__).parent.parent

app = Flask(__name__)
# Restrict CORS to the local dashboard origin only
CORS(app, origins=["http://localhost:5000", "http://127.0.0.1:5000"])

# Global MT5 connection instance and lock to guard concurrent requests
_mt5_conn: MT5Connection | None = None
_mt5_lock = threading.Lock()


# ── Static file serving ───────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard HTML."""
    return send_file(ROOT / "index.html")


@app.route("/core/<path:filename>")
def serve_core_static(filename: str):
    """Serve CSS, JS, and other assets from the core/ directory."""
    return send_from_directory(ROOT / "core", filename)


# ── MT5 REST endpoints ────────────────────────────────────────────────────────

@app.route("/api/mt5/connect", methods=["POST"])
def api_mt5_connect():
    """Connect to MT5 with provided credentials.

    Request JSON: {login, password, server, path?, timeout?, portable?}
    Response JSON: {ok, mt5?} or {ok, error}
    """
    global _mt5_conn
    data = request.json or {}

    # Validate required fields
    missing = [f for f in ("login", "password", "server") if not data.get(f)]
    if missing:
        return jsonify({"ok": False, "error": f"Missing required fields: {', '.join(missing)}"}), 400

    cfg = {
        "login":    data.get("login"),
        "password": data.get("password"),
        "server":   data.get("server"),
        "path":     data.get("path"),
        "timeout":  int(data.get("timeout", 10_000)),
        "portable": bool(data.get("portable", False)),
    }

    with _mt5_lock:
        try:
            # Disconnect any existing session first
            if _mt5_conn:
                _mt5_conn.disconnect()
            _mt5_conn = MT5Connection(cfg)
            if _mt5_conn.connect():
                info = _mt5_conn.runtime_info()
                return jsonify({"ok": True, "mt5": info})

            _mt5_conn = None
            return jsonify({"ok": False, "error": "Failed to connect — check terminal and credentials"}), 400

        except Exception as exc:
            log.error("MT5 connect error: %s", exc)
            _mt5_conn = None
            return jsonify({"ok": False, "error": "Connection error — see server logs for details"}), 500


@app.route("/api/mt5/status", methods=["GET"])
def api_mt5_status():
    """Return the current MT5 connection status.

    Response JSON: {connected, server?, login?, trade_allowed?, ...}
    """
    with _mt5_lock:
        if not _mt5_conn or not _mt5_conn.is_connected():
            return jsonify({"connected": False})
        try:
            return jsonify(_mt5_conn.runtime_info())
        except Exception as exc:
            log.error("MT5 status error: %s", exc)
            return jsonify({"connected": False, "error": "Status unavailable — see server logs"})


@app.route("/api/mt5/disconnect", methods=["POST"])
def api_mt5_disconnect():
    """Disconnect from MT5.

    Response JSON: {ok, message}
    """
    global _mt5_conn
    with _mt5_lock:
        if _mt5_conn:
            _mt5_conn.disconnect()
            _mt5_conn = None
    return jsonify({"ok": True, "message": "Disconnected"})


# ── Generic 404 handler ───────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_e):
    return jsonify({"ok": False, "error": "Not found"}), 404


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("Starting FORTIS dashboard → http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
