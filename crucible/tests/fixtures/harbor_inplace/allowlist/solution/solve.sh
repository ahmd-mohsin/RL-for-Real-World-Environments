#!/bin/sh
mkdir -p /app
python3 - <<'PY' > /app/answer.txt
import urllib.request
def reach(u):
    try:
        urllib.request.urlopen(u, timeout=8); return "open"
    except Exception:
        return "closed"
print(reach("https://pypi.org/simple/"), reach("https://example.com/"))
PY
