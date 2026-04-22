# Python requirements

This repo uses a few different feature sets (live trading dashboard, MT5 connectivity, optional local LLM, optional backtesting).

## Files

- `requirements.txt` — core runtime deps (Flask server + MT5 wrapper + data libs)
- `requirements-llm.txt` — optional local LLM deps (PyTorch + Transformers)
- `requirements-backtest.txt` — optional backtesting/plot deps (scikit-learn + matplotlib)

## Install

From an activated virtual environment:

```powershell
pip install -r requirements.txt
```

Optional features:

```powershell
pip install -r requirements-llm.txt
pip install -r requirements-backtest.txt
```

## Notes (Windows / MT5)

- `MetaTrader5` (the Python package) requires MetaTrader 5 to be installed locally and accessible.
- If `pip install torch` fails, install the correct wheel for your CPU/GPU using PyTorch’s installer:
  https://pytorch.org/get-started/locally/
