"""
MT5 Config Manager — auto-detects a running MT5 terminal and manages
encrypted credentials stored in core/mt5_credentials_config.json.

Key features
------------
- detect_mt5_process()      : find terminal64.exe / terminal.exe via psutil
- auto_detect_credentials() : connect to running terminal, read account info
- save_credentials()        : persist credentials (password Fernet-encrypted)
- load_credentials()        : read & decrypt stored credentials
- delete_credentials()      : wipe stored config + encryption key
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_CORE_DIR   = Path(__file__).resolve().parent
CONFIG_PATH = _CORE_DIR / "mt5_credentials_config.json"
_KEY_PATH   = _CORE_DIR / ".mt5_key"


# ── Encryption helpers ────────────────────────────────────────────────────────

def _get_or_create_key() -> bytes:
    """Return the Fernet key, creating and persisting one if it doesn't exist."""
    from cryptography.fernet import Fernet
    if _KEY_PATH.exists():
        return _KEY_PATH.read_bytes()
    key = Fernet.generate_key()
    _KEY_PATH.write_bytes(key)
    log.debug("MT5Config: new Fernet key written to %s", _KEY_PATH)
    return key


def encrypt_password(password: str) -> str:
    """Return a Fernet-encrypted, base64-encoded token for *password*."""
    from cryptography.fernet import Fernet
    return Fernet(_get_or_create_key()).encrypt(password.encode()).decode()


def decrypt_password(token: str) -> str:
    """Decrypt and return a previously encrypted password token."""
    from cryptography.fernet import Fernet
    return Fernet(_get_or_create_key()).decrypt(token.encode()).decode()


# ── Process detection ─────────────────────────────────────────────────────────

def detect_mt5_process() -> Optional[str]:
    """
    Search running processes for a MT5 terminal executable.

    Returns the full path to the terminal exe when found, otherwise *None*.
    Requires the optional ``psutil`` package; logs a warning if absent.
    """
    try:
        import psutil
    except ImportError:
        log.warning("MT5Config: psutil not installed — cannot auto-detect MT5 process")
        return None

    for proc in psutil.process_iter(["name", "exe"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if name in ("terminal64.exe", "terminal.exe"):
                return proc.info.get("exe")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return None


# ── Auto-detection ────────────────────────────────────────────────────────────

def auto_detect_credentials() -> Optional[dict]:
    """
    Try to connect to the already-running MT5 terminal and read account info.

    When MT5 is open and logged in the Python API can attach to it without
    supplying a password — the terminal handles authentication itself.

    Returns a credentials dict on success::

        {
            "login":    <int>,
            "password": "",          # intentionally empty; terminal is live
            "server":   "<broker>",
            "path":     "<exe path>",
            "timeout":  10000,
            "portable": False,
        }

    Returns *None* if MT5 is not running, not logged in, or the Python
    MetaTrader5 package is unavailable.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        log.warning("MT5Config: MetaTrader5 package not installed — skipping auto-detect")
        return None

    terminal_exe = detect_mt5_process()

    init_kwargs: dict = {}
    if terminal_exe:
        init_kwargs["path"] = terminal_exe

    try:
        if not mt5.initialize(**init_kwargs):
            code, msg = mt5.last_error()
            log.debug("MT5Config auto-detect: initialize() failed [%d] %s", code, msg)
            return None

        account  = mt5.account_info()
        terminal = mt5.terminal_info()

        if not account:
            log.debug("MT5Config auto-detect: connected but no account info (not logged in?)")
            mt5.shutdown()
            return None

        # Prefer the detected exe path; fall back to terminal_info path
        path = terminal_exe or getattr(terminal, "path", "") or ""

        result = {
            "login":    int(account.login),
            "password": "",        # terminal is running; password not required
            "server":   str(account.server),
            "path":     str(path),
            "timeout":  10_000,
            "portable": False,
        }
        log.info(
            "MT5Config: auto-detected MT5 login=%s server=%s",
            result["login"], result["server"],
        )
        return result

    except Exception as exc:
        log.warning("MT5Config: auto-detect raised: %s", exc)
        return None

    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


# ── Persistence ───────────────────────────────────────────────────────────────

def save_credentials(
    login,
    password: str,
    server: str,
    path: str,
    timeout: int = 10_000,
    portable: bool = False,
) -> None:
    """
    Persist MT5 credentials to ``core/mt5_credentials_config.json``.

    The password is Fernet-encrypted before writing; an empty string is stored
    as-is (terminals already running don't require a password).
    """
    encrypted_pw = encrypt_password(password) if password else ""
    data = {
        "login":    int(login) if login else None,
        "password": encrypted_pw,
        "server":   server   or "",
        "path":     path     or "",
        "timeout":  int(timeout),
        "portable": bool(portable),
    }
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log.info("MT5Config: credentials saved (login=%s)", login)


def load_credentials() -> Optional[dict]:
    """
    Load and decrypt credentials from ``core/mt5_credentials_config.json``.

    Returns *None* when the file does not exist or cannot be parsed.
    The returned dict always has a plain-text ``"password"`` key.
    """
    if not CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if data.get("password"):
            try:
                data["password"] = decrypt_password(data["password"])
            except Exception as exc:
                log.warning("MT5Config: could not decrypt stored password: %s", exc)
                data["password"] = ""
        return data
    except Exception as exc:
        log.warning("MT5Config: could not load credentials config: %s", exc)
        return None


def delete_credentials() -> None:
    """Remove the stored credentials file and encryption key."""
    removed = []
    for p in (CONFIG_PATH, _KEY_PATH):
        if p.exists():
            p.unlink()
            removed.append(p.name)
    if removed:
        log.info("MT5Config: removed %s", ", ".join(removed))
