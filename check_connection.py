#!/usr/bin/env python3
import urllib.request
import json
import sys

print("\n" + "="*60)
print("CHECKING MT5 & API CONNECTIONS")
print("="*60 + "\n")

try:
    response = urllib.request.urlopen('http://localhost:5000/bot/status', timeout=5)
    data = json.loads(response.read())
    
    print("✓ API ENDPOINT: http://localhost:5000/bot/status")
    print(f"  Status Code: 200 OK\n")
    
    print("MT5 CONNECTION:")
    print(f"  Connected:    {data['mt5'].get('connected')}")
    print(f"  Server:       {data['mt5'].get('server')}")
    print(f"  Account:      {data['mt5'].get('account')}")
    print(f"  Balance:      {data['mt5'].get('balance')} {data['mt5'].get('currency')}")
    print(f"  Equity:       {data['mt5'].get('equity')}")
    print(f"  Trade Allow:  {data['mt5'].get('trade_allowed')}\n")
    
    print("BOT STATUS:")
    print(f"  Running:      {data.get('running')}")
    print(f"  Bot Config:   {data.get('bot') is not None}\n")
    
    if data.get('bot_error'):
        print(f"⚠ Bot Error: {data['bot_error']}\n")
    
    print("OPEN POSITIONS:")
    positions = data.get('open_positions', [])
    print(f"  Count: {len(positions)}")
    if positions:
        for p in positions:
            print(f"    - {p.get('symbol')}: {p.get('type')} {p.get('volume')} @ {p.get('price')}")
    print()
    
    print("="*60)
    if data['mt5'].get('connected') and data['mt5'].get('trade_allowed'):
        print("✓ ALL CONNECTIONS OK - READY FOR TRADING")
    else:
        print("⚠ CONNECTIONS HAVE ISSUES")
    print("="*60 + "\n")
    
except Exception as e:
    print(f"✗ ERROR: {e}")
    print(f"  Make sure server is running at http://localhost:5000\n")
    sys.exit(1)
