"""Wait until the mining/gate logs hold at least N result lines (or the timeout), then print them."""
import sys, time
from pathlib import Path

d = Path(__file__).parent
n, timeout = int(sys.argv[1]), float(sys.argv[2])
logs = sys.argv[3:]
t0 = time.time()
while True:
    lines = [ln for f in logs if (d / f).exists() for ln in (d / f).read_text(errors="replace").splitlines() if ln.startswith("[") or " ok=" in ln]
    if len(lines) >= n or time.time() - t0 > timeout:
        print("\n".join(lines))
        break
    time.sleep(30)
