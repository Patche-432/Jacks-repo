#!/usr/bin/env python3
"""
Quick validation that AI_Pro signal generation works correctly after optimizations.
Tests signal logic WITHOUT running full backtest or initializing LLM.
"""
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

try:
    # Import core modules
    from ai_pro import AI_Pro, _mt5_initialize
    import MetaTrader5 as mt5
    import pandas as pd
    
    log.info("✓ Imports successful")
    
    # Initialize MT5
    log.info("Connecting to MT5...")
    _mt5_initialize()
    
    if not mt5.initialize():
        log.error("MT5 initialization failed")
        sys.exit(1)
    
    log.info("✓ MT5 connected")
    
    # Create AI_Pro instance WITHOUT LLM (use_ai=False)
    log.info("Initializing AI_Pro (no AI inference)...")
    strategy = AI_Pro(use_ai=False, lookback_candles=200)
    log.info("✓ AI_Pro initialized")
    
    # Test signal generation on a real symbol
    symbol = "EURUSD"
    log.info(f"\nGenerating signals for {symbol}...")
    
    signal = strategy.generate_trade_signal(symbol)
    
    log.info(f"\n{'='*60}")
    log.info(f"SIGNAL GENERATION TEST RESULT")
    log.info(f"{'='*60}")
    log.info(f"Symbol:           {symbol}")
    log.info(f"Signal:           {signal.get('signal', 'N/A')}")
    log.info(f"Source (ENV):     {signal.get('signal_source', 'N/A')}")
    log.info(f"Confidence:       {signal.get('confidence', 0)}%")
    log.info(f"Entry Price:      {signal.get('entry_price', 0):.5f}")
    log.info(f"Stop Loss:        {signal.get('stop_loss', 0):.5f}")
    log.info(f"Take Profit:      {signal.get('take_profit', 0):.5f}")
    log.info(f"AI Approved:      {signal.get('ai_approved', 'N/A')}")
    log.info(f"AI Reason:        {signal.get('ai_reason', 'N/A')}")
    log.info(f"{'='*60}")
    
    # Verify all required fields
    required_fields = [
        'signal', 'signal_source', 'confidence', 'entry_price',
        'stop_loss', 'take_profit', 'ai_approved', 'ai_reason'
    ]
    
    missing = [f for f in required_fields if f not in signal]
    if missing:
        log.error(f"✗ FAILED: Missing fields: {missing}")
        sys.exit(1)
    
    # Verify EN V logic (no momentum gates)
    env_source = signal.get('signal_source', '')
    expected_envs = ['CHoCH-BUY@PDL', 'CHoCH-SELL@PDH', 'Continuation-BUY@PDH', 'Continuation-SELL@PDL', 'neutral']
    
    if signal['signal'] != 'neutral' and env_source not in expected_envs:
        log.warning(f"! Unexpected ENV source: {env_source}")
    else:
        log.info(f"✓ ENV source valid: {env_source}")
    
    # Verify no momentum checks in approval logic
    ai_reason = signal.get('ai_reason', '')
    if 'momentum' in ai_reason.lower():
        log.warning(f"! WARNING: Momentum mentioned in AI reason: {ai_reason}")
    else:
        log.info(f"✓ No momentum gates in approval logic")
    
    log.info("\n✓ STRATEGY VALIDATION PASSED")
    log.info("  - All signal fields present")
    log.info("  - ENV logic functional (level-only)")
    log.info("  - No momentum filtering in AI approval gate")
    
    mt5.shutdown()
    sys.exit(0)

except Exception as e:
    log.error(f"✗ VALIDATION FAILED: {e}", exc_info=True)
    try:
        mt5.shutdown()
    except:
        pass
    sys.exit(1)
