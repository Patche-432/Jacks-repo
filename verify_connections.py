#!/usr/bin/env python3
import urllib.request
import json
import sys

print("\n" + "="*70)
print("VERIFYING MT5 & API CONNECTIONS")
print("="*70 + "\n")

try:
    # Get bot status
    response = urllib.request.urlopen('http://localhost:5000/bot/status', timeout=5)
    data = json.loads(response.read())
    
    print("✓ API ENDPOINT: /bot/status (200 OK)\n")
    
    print("MT5 CONNECTION STATUS:")
    mt5 = data.get('mt5', {})
    if mt5.get('connected'):
        print(f"  ✓ Connected: {mt5.get('connected')}")
        print(f"    Server:    {mt5.get('server')}")
        print(f"    Account:   {mt5.get('account')}")
        print(f"    Balance:   {mt5.get('balance')} {mt5.get('currency')}")
        print(f"    Equity:    {mt5.get('equity')}")
        print(f"    Trade OK:  {mt5.get('trade_allowed')}")
    else:
        print(f"  ✗ NOT Connected - {mt5.get('error', 'unknown error')}")
    
    print("\nBOT STATUS:")
    print(f"  Running: {data.get('running')}")
    config = data.get('config', {})
    if config:
        print(f"  Symbols: {config.get('symbols')}")
        print(f"  Volume:  {config.get('volume')} lot")
        print(f"  Poll:    {config.get('poll_secs')}s")
    
    print("\nOPEN POSITIONS:")
    positions = data.get('open_positions', [])
    if positions:
        print(f"  Count: {len(positions)}")
        for p in positions:
            print(f"    - {p.get('symbol')}: {p.get('type')} {p.get('volume')} @ {p.get('price')}")
    else:
        print(f"  None (ready for trading)")
    
    print("\n" + "="*70)
    if mt5.get('connected') and mt5.get('trade_allowed'):
        print("✓✓✓ ALL CONNECTIONS OK - READY FOR TRADING ✓✓✓")
        sys.exit(0)
    else:
        print("⚠ CONNECTION ISSUES DETECTED")
        if data.get('bot_error'):
            print(f"Bot Error: {data['bot_error']}")
        sys.exit(1)
    print("="*70 + "\n")
    
except Exception as e:
    print(f"✗ ERROR: {e}")
    print(f"  Make sure ai_pro.py is running\n")
    sys.exit(1)
