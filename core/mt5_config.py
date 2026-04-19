"""MT5 credentials management — plaintext JSON persistence.

NOTE: Credentials are stored as plaintext JSON. Do not commit the
credentials file (mt5_credentials.json) to version control.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)

# Credentials file path
_CREDS_FILE = Path(__file__).resolve().parent.parent / "mt5_credentials.json"


def load_credentials() -> Optional[Dict[str, Any]]:
    """Load MT5 credentials from disk. Returns dict with login/password/server/path or None."""
    if not _CREDS_FILE.exists():
        return None
    try:
        with open(_CREDS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as exc:
        log.error(f"Failed to load credentials: {exc}")
        return None


def save_credentials(login: str, password: str, server: str, path: str) -> bool:
    """Save MT5 credentials to disk as plaintext JSON. Returns True on success."""
    try:
        data = {
            "login": login,
            "password": password,
            "server": server,
            "path": path,
        }
        with open(_CREDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        log.info(f"Credentials saved to {_CREDS_FILE}")
        return True
    except Exception as exc:
        log.error(f"Failed to save credentials: {exc}")
        return False


def clear_credentials() -> bool:
    """Remove saved credentials file. Returns True on success."""
    try:
        if _CREDS_FILE.exists():
            _CREDS_FILE.unlink()
            log.info("Credentials cleared")
        return True
    except Exception as exc:
        log.error(f"Failed to clear credentials: {exc}")
        return False
