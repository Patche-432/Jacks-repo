"""
serve.py — Production launcher for the Fortis dashboard.

Replaces `python core/server.py` for production use. The development server
shipped with Flask is fine for local hacking but logs a warning ("This is a
development server. Do not use it in a production deployment.") and is
deliberately single-process / single-threaded by default. core_server.err.log
in this repo currently shows that warning — this script makes it go away.

Run from the repo root:
    python scripts/serve.py                 # auto-pick waitress / gunicorn
    python scripts/serve.py --host 0.0.0.0  # listen on all interfaces
    python scripts/serve.py --threads 16    # tune worker count

Defaults:
    host    = 127.0.0.1   (loopback only — same as the dev server)
    port    = 5000
    threads = 8           (per-connection threads in waitress)

The dashboard is single-user (one operator at a console), so 8 worker threads
is generous. Bump --threads only if you have multiple browsers polling the
backtest streaming endpoint at once.

Backend selection:
    Windows  → waitress (the standard Flask production option on Windows;
               pure-Python, no fork required, plays nicely with NSSM).
    Linux    → gunicorn if installed, else waitress. Both are fine; gunicorn
               is the more conventional Linux choice but waitress works there
               too and is already a runtime dependency.

This script never starts MT5 or the bot; it just serves the Flask app
(`core.server.app`). MT5 is connected via the dashboard's "Connect" button
exactly as before.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
from pathlib import Path

# Make the repo root importable so `from core.server import app` works
# regardless of where this is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Configure logging before importing the app so the app's own loggers
# inherit our format. core/server.py also calls logging.basicConfig — that
# call becomes a no-op once a handler is attached here, which is what we want.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("serve")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Production WSGI launcher for the Fortis dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default=os.environ.get("FORTIS_HOST", "127.0.0.1"),
                   help="Host/interface to bind. 127.0.0.1 = local only.")
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("FORTIS_PORT", "5000")),
                   help="TCP port to listen on.")
    p.add_argument("--threads", type=int,
                   default=int(os.environ.get("FORTIS_THREADS", "8")),
                   help="Number of worker threads (waitress) or workers (gunicorn).")
    p.add_argument("--backend",
                   choices=["auto", "waitress", "gunicorn"],
                   default=os.environ.get("FORTIS_WSGI", "auto"),
                   help="Force a specific WSGI server. 'auto' picks based on platform.")
    return p.parse_args()


def _import_app():
    """Import the Flask app, with a clean failure message if it errors."""
    try:
        from core.server import app  # noqa: WPS433 (deliberately late import)
    except Exception as exc:
        log.error("Failed to import core.server:app  →  %s", exc)
        log.error("Run `python scripts/preflight.py` to diagnose.")
        sys.exit(1)
    return app


def _select_backend(requested: str) -> str:
    """Resolve 'auto' to a concrete backend name, or honour an explicit choice."""
    if requested != "auto":
        return requested
    # On Windows, gunicorn isn't supported (no fork), so always pick waitress.
    if platform.system() == "Windows":
        return "waitress"
    # On Linux/macOS, prefer gunicorn if installed, fall back to waitress.
    try:
        import gunicorn  # noqa: F401
        return "gunicorn"
    except ImportError:
        return "waitress"


def _serve_waitress(app, host: str, port: int, threads: int) -> None:
    try:
        from waitress import serve
    except ImportError:
        log.error("waitress not installed. Run: pip install waitress")
        sys.exit(1)

    log.info("Starting waitress on http://%s:%d (threads=%d)", host, port, threads)
    log.info("Dashboard: http://%s:%d/  —  Ctrl+C to stop", host, port)
    # waitress handles SIGINT cleanly; no extra signal wrangling needed.
    # ident=None drops the "waitress" Server header — small surface-area win.
    serve(app, host=host, port=port, threads=threads, ident=None)


def _serve_gunicorn(app, host: str, port: int, workers: int) -> None:
    """Run gunicorn programmatically so this script behaves the same way as
    `python scripts/serve.py` regardless of which backend is chosen."""
    try:
        from gunicorn.app.base import BaseApplication
    except ImportError:
        log.error("gunicorn not installed. Run: pip install gunicorn  "
                  "(or use --backend waitress)")
        sys.exit(1)

    class _StandaloneApp(BaseApplication):
        def __init__(self, app, options):
            self._app = app
            self._options = options
            super().__init__()

        def load_config(self):
            for k, v in self._options.items():
                self.cfg.set(k, v)

        def load(self):
            return self._app

    options = {
        "bind":         f"{host}:{port}",
        "workers":      workers,
        "worker_class": "sync",
        # Long-running SSE backtests need a generous worker timeout.
        "timeout":      0,
        "loglevel":     "info",
        "accesslog":    "-",
    }
    log.info("Starting gunicorn on http://%s:%d (workers=%d)", host, port, workers)
    log.info("Dashboard: http://%s:%d/  —  Ctrl+C to stop", host, port)
    _StandaloneApp(app, options).run()


def main() -> int:
    args = _parse_args()
    app = _import_app()
    backend = _select_backend(args.backend)
    log.info("Selected WSGI backend: %s  (platform: %s)", backend, platform.system())

    if backend == "waitress":
        _serve_waitress(app, args.host, args.port, args.threads)
    else:
        _serve_gunicorn(app, args.host, args.port, args.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
