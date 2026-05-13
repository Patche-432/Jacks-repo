"""
Pytest setup shared by every test in the suite.

Why this file exists
--------------------
The bot's production target is Windows + MetaTrader5, but CI runs on Linux
where the `MetaTrader5` PyPI package can't be installed. The strategy
module (`ai_pro`) does `import MetaTrader5 as mt5` at the top, which
would otherwise make every test that imports `ai_pro` fail to collect on
Linux.

We sidestep that by injecting a tiny stub into `sys.modules['MetaTrader5']`
BEFORE any test imports `ai_pro`. The stub exposes the symbolic constants
the strategy reads at module-import time (`TRADE_ACTION_DEAL`, retcodes,
filling-mode flags, etc.) and provides callable placeholders for the
functions the live bot uses. None of the placeholders contact a broker —
unit tests that exercise execution paths are expected to monkey-patch
the specific functions they need on a per-test basis.

If a CI runner has the real `MetaTrader5` package installed (Windows job),
this stub is a no-op: we only register it when the import would otherwise
fail.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _install_mt5_stub() -> None:
    """Register a placeholder MetaTrader5 module if the real one is absent."""
    try:
        import MetaTrader5  # noqa: F401 — presence check only
        return
    except ImportError:
        pass

    mt5 = types.ModuleType("MetaTrader5")

    # ── Constants the strategy reads at import time ──────────────────────
    # Values mirror the MetaTrader5 SDK (>=5.0.45). Anything missing here
    # gets a sensible default; tests that need the real value should
    # monkey-patch it explicitly.
    mt5.TRADE_ACTION_DEAL    = 1
    mt5.TRADE_ACTION_PENDING = 5
    mt5.TRADE_ACTION_SLTP    = 6
    mt5.TRADE_ACTION_MODIFY  = 7
    mt5.TRADE_ACTION_REMOVE  = 8
    mt5.TRADE_ACTION_CLOSE_BY = 10

    mt5.ORDER_TYPE_BUY        = 0
    mt5.ORDER_TYPE_SELL       = 1
    mt5.ORDER_TYPE_BUY_LIMIT  = 2
    mt5.ORDER_TYPE_SELL_LIMIT = 3
    mt5.ORDER_TYPE_BUY_STOP   = 4
    mt5.ORDER_TYPE_SELL_STOP  = 5

    mt5.ORDER_FILLING_FOK    = 0
    mt5.ORDER_FILLING_IOC    = 1
    mt5.ORDER_FILLING_RETURN = 2
    mt5.ORDER_FILLING_BOC    = 3

    mt5.ORDER_TIME_GTC        = 0
    mt5.ORDER_TIME_DAY        = 1
    mt5.ORDER_TIME_SPECIFIED  = 2

    mt5.POSITION_TYPE_BUY  = 0
    mt5.POSITION_TYPE_SELL = 1

    mt5.TRADE_RETCODE_DONE        = 10009
    mt5.TRADE_RETCODE_REQUOTE     = 10004
    mt5.TRADE_RETCODE_REJECT      = 10006
    mt5.TRADE_RETCODE_NO_MONEY    = 10019
    mt5.TRADE_RETCODE_INVALID_PRICE = 10015
    mt5.TRADE_RETCODE_INVALID_STOPS = 10016

    mt5.TIMEFRAME_M1  = 1
    mt5.TIMEFRAME_M5  = 5
    mt5.TIMEFRAME_M15 = 15
    mt5.TIMEFRAME_M30 = 30
    mt5.TIMEFRAME_H1  = 16385
    mt5.TIMEFRAME_H4  = 16388
    mt5.TIMEFRAME_D1  = 16408

    # ── Function placeholders ────────────────────────────────────────────
    # Tests that need real behaviour should monkey-patch these with
    # mock.patch('MetaTrader5.<fn>', ...).
    def _unimpl(*_args, **_kwargs):  # pragma: no cover — defensive
        raise RuntimeError(
            "MetaTrader5 stub function called without a test mock — "
            "patch the specific function you need."
        )

    for name in (
        "initialize", "shutdown", "last_error", "version",
        "account_info", "terminal_info",
        "symbol_info", "symbol_info_tick", "symbol_select", "symbols_get",
        "copy_rates_from_pos", "copy_rates_range",
        "positions_get", "history_deals_get", "history_orders_get",
        "order_send", "order_check",
    ):
        setattr(mt5, name, _unimpl)

    sys.modules["MetaTrader5"] = mt5


# Make the repo root importable so `from ai_pro import …` works regardless
# of where pytest was invoked from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_install_mt5_stub()
