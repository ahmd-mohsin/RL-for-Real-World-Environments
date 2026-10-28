"""The seal guard (``factory.sealguard``): held-out content never reaches git by accident.

    seal_guard.py check --staged                 # the pre-commit hook (.githooks/pre-commit)
    seal_guard.py check --range origin/main..HEAD   # before a push, or in CI
    seal_guard.py check --tree origin/main --baseline   # the ratchet: nothing beyond the frozen baseline
    seal_guard.py check ... --i-know "<reason>"   # publish anyway, journaled
    seal_guard.py refresh [--data DIR]           # hash the held-out stores' hidden content into the fingerprints
    seal_guard.py baseline [--ref origin/main]   # freeze today's known exposures (paths and rules only)
    seal_guard.py install                        # git config core.hooksPath .githooks (asks nothing, prints what it did)

Exit status 1 on a refusal.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
sys.path.insert(0, str(HERE / "src"))

from crucible.factory import sealguard as sg

BASELINE = HERE / "docs/research/swarm-pipeline/runs/seal_baseline.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument("--staged", action="store_true")
    g.add_argument("--range")
    g.add_argument("--tree")
    c.add_argument("--baseline", action="store_true", help="tree mode: report only what the frozen baseline does not hold")
    c.add_argument("--i-know", dest="reason", default=None, help="let the findings through, journaled with this reason")
    r = sub.add_parser("refresh")
    r.add_argument("--data", type=Path, default=HERE / "data")
    b = sub.add_parser("baseline")
    b.add_argument("--ref", default="origin/main")
    sub.add_parser("install")
    a = ap.parse_args()
    seal = sg.Seal.load(REPO)

    if a.cmd == "check":
        if a.staged:
            found = sg.staged(REPO, seal)
        elif a.range:
            found = sg.commit_range(REPO, a.range, seal)
        else:
            found = sg.tree(REPO, a.tree, seal)
            if a.baseline:
                found = sg.ratchet(found, json.loads(BASELINE.read_text())["findings"] if BASELINE.is_file() else [])
        ok, msg = sg.decide(REPO, found, reason=a.reason)
        print(msg, file=sys.stderr if not ok else sys.stdout)
        sys.exit(0 if ok else 1)
    if a.cmd == "refresh":
        from crucible.factory import benchaudit as ba

        d = a.data
        from crucible.factory.lanes.windows import split_of

        # a held-out store can still hold windows of a repo the lead has since moved to training (carbon, 2026-10-23):
        # those are no longer held out, so they are not fingerprinted
        tasks = [t for t in ba.load_debug(d / "debug/gen/store/heldout") if split_of(t.repo) == "heldout"]
        key = None
        try:
            from crucible.bench.dynamic import load_key

            key = load_key()
        except Exception:  # noqa: BLE001 - no bench key on this machine: nothing to fingerprint
            key = None
        print(json.dumps(sg.refresh(REPO, tasks, key=key)))
        return
    if a.cmd == "baseline":
        found = sg.tree(REPO, a.ref, seal)
        rows = sorted({(f.rule, f.path) for f in found})
        BASELINE.write_text(json.dumps({"ref": a.ref, "note": "exposures public before the guard (paths and rules only; the audit has the "
                                        "evidence)", "findings": [{"rule": r_, "path": p} for r_, p in rows]}, indent=1) + "\n")
        print(f"{len(rows)} known exposures frozen in {BASELINE}")
        return
    if a.cmd == "install":
        subprocess.run(["git", "-C", str(REPO), "config", "core.hooksPath", ".githooks"], check=True)
        print("core.hooksPath = .githooks (pre-commit runs the seal guard); undo: git config --unset core.hooksPath")


if __name__ == "__main__":
    main()
