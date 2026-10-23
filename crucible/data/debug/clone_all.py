import subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

root = Path(sys.argv[1])
repos = [l.strip() for l in Path(sys.argv[2]).read_text().splitlines() if l.strip()]


def one(full):
    d = root / full.replace("/", "__")
    if (d / ".git").exists():
        return f"have {full}"
    r = subprocess.run(["git", "clone", "-q", f"https://github.com/{full}.git", str(d)], capture_output=True, text=True, timeout=3600)
    return f"{'ok' if r.returncode == 0 else 'FAIL'} {full} {r.stderr[-200:]}"


with ThreadPoolExecutor(6) as ex:
    for line in ex.map(one, repos):
        print(line, flush=True)
