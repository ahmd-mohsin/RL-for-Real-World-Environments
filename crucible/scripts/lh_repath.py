"""Re-path stored episodes under the current path rules (``factory.repath``): no model call, no test run.

    uv run python scripts/lh_repath.py --roots data/targeting/debug data/targeting/debug-300 --store data/debug/gen/store \
        --out docs/research/swarm-pipeline/runs/repath_install.json [--allow-declared-reinstall]

Writes the summary and every flip (an attempt whose reward or path verdict changed), with the old and new evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crucible.factory.repath import scan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--store", required=True, help="the debug lane's window store (<store>/<split>/tasks/<id>/window.json)")
    p.add_argument("--out", required=True)
    p.add_argument("--allow-declared-reinstall", action="store_true", help="also excuse a bare reinstall of declared dependencies")
    a = p.parse_args()
    over = {"allow_declared_reinstall": True} if a.allow_declared_reinstall else None
    res = scan([Path(r) for r in a.roots], Path(a.store), rules_over=over)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "flips"}, indent=1))


if __name__ == "__main__":
    main()
