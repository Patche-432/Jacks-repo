#!/usr/bin/env python3
import urllib.request
import json

try:
    response = urllib.request.urlopen('http://localhost:5000/bot/status', timeout=5)
    data = json.loads(response.read())
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f"ERROR: {e}")
