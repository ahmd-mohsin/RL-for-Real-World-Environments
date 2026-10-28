"""Benchmark contamination audit (``factory.benchaudit``): every held-out debug task and the kept benchmark
variants, against our public tree and every training source on this machine. No model call, no network.

    uv run python scripts/bench_audit.py [--data DIR] [--ref origin/main] \
        [--out docs/research/swarm-pipeline/runs/benchmark_contamination.json]

``--data`` is the ``crucible/data`` that holds the stores (a worktree has none of its own). The report holds counts,
paths and commit hashes only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from crucible.factory import benchaudit as ba
from crucible.factory.lanes.windows import HELD_OUT_REPOS

FIXING = HERE / "docs/research/swarm-pipeline/07-debug-lane/fixing/prompts"
# the kept benchmark tasks (07-debug-lane/fixing/README.md, 2026-10-24) and the one still in progress
BENCHMARK = ("db-go-urfave-cli-2b0adb14-d1", "db-node-h3-a6cd4654", "db-go-excelize-824ac2ce-d1", "db-go-excelize-65be17e3-d1",
             "db-node-unstorage-d7aee0b7-h2")


def held_id(i: str) -> bool:
    parts = i.split("-")
    return len(parts) >= 4 and any("-".join(parts[2:k]) in HELD_OUT_REPOS for k in range(3, len(parts)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--out", type=Path, default=HERE / "docs/research/swarm-pipeline/runs/benchmark_contamination.json")
    a = ap.parse_args()
    d = a.data
    debug = ba.load_debug(d / "debug/gen/store/heldout", [d / "debug/prompts"])
    variants = ba.variants(debug, FIXING, [b for b in BENCHMARK if b not in {t.id for t in debug}])
    held = debug + variants
    corpus = ba.training_corpus(d / "debug/gen/store/train", [d / "debug/prompts", FIXING], skip=held_id)
    grep = ba.git_grep(HERE.parent, a.ref)
    rep = ba.audit(held, grep, corpus, repo=HERE.parent, ref=a.ref, in_draw=lambda t: t.repo in HELD_OUT_REPOS)
    rep["benchmark"] = list(BENCHMARK)
    rep["corpus_docs"] = len(corpus)
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    for r in rep["tasks"]:
        mark = "*" if r["id"] in BENCHMARK else " "
        print(f"{mark} {r['id']:40} {r['verdict']:12} {'; '.join(r['why'])[:150]}")
    print(rep["counts"], f"-> {a.out}")


if __name__ == "__main__":
    main()
