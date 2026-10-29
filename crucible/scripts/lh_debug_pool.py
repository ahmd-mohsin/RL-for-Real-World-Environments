#!/usr/bin/env python
"""The debug lane's training pool: the tasks RL trains on, checked against their tracked manifest
(``docs/research/swarm-pipeline/07-debug-lane/training-pool/pool.json``) before anything is synced to a cluster.

The pool is its own debug root (default ``crucible/data/debug-pool``): training-split windows only, the coauthors'
tasks rebuilt there with ``lh_debug.py subset`` from their recorded recipes and gated there, with the clones they were
built from. ``delta_rl.py train --debug <pool> --targeting <pool>/targeting.json --tasks ...`` reads it as it is.

Commands (no Docker unless ``--images``; no model calls):

- ``check [--manifest M] [--debug D] [--images]``: every task the manifest calls ``ready`` is on disk as recorded: its
  candidate row (training split, every gate passed), its authored prompt (no lint finding, the bug report's sha256 the
  manifest's), its window (the start commit the manifest records, and the coauthor's recorded one when there is one),
  its gate record (all five gates, graded this start commit when the record names one) and its targeting record (``trainer.check_band`` for the
  student). ``--images``: the World image is present with the image id the manifest records. Exit 1 on any problem.
- ``targeting [--manifest M] [--out T]``: the targeting records of the ready tasks (``{model, band, keep, split,
  source}``, the schema ``lh_debug_target.py report`` writes; the band is the coauthors' recorded measurement, never a
  new one) to ``T`` (default ``<debug>/targeting.json``), what ``--targeting`` reads.
- ``relocate [--debug D]``: the lane's working clones borrow objects from ``src/`` through git alternates holding this
  machine's absolute path; each is rewritten relative to its objects dir (``benchrun.bundle``'s rule), so the pool works
  wherever it is copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MANIFEST = ROOT / "docs" / "research" / "swarm-pipeline" / "07-debug-lane" / "training-pool" / "pool.json"
GATES = ("S6", "S5", "S8", "S3", "S10")


def load(p: Path) -> Any:
    return json.loads(Path(p).read_text())


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ready(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {t: r for t, r in manifest["tasks"].items() if r.get("status") == "ready"}


def targeting_records(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The ready tasks' targeting records, in the trainer's schema, with where each band was measured."""
    model = str(manifest["student"]).split(":", 1)[0]
    return {t: {"model": model, "band": r["targeting"]["band"], "keep": r["targeting"]["keep"], "split": "train", "source": r["targeting"]["source"]}
            for t, r in sorted(ready(manifest).items())}


def image_id(image: str) -> str | None:
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, text=True, check=False)
    return r.stdout.strip() or None if r.returncode == 0 else None


def check_task(tid: str, row: Mapping[str, Any], debug: Path, cands: Mapping[str, Mapping[str, Any]], model: str, *, images: bool = False) -> list[str]:
    """Why ready task ``tid`` is not on disk as the manifest records it (empty: it is)."""
    from crucible.train.bridges.tinker.trainer import check_band

    bad: list[str] = []
    c = cands.get(tid)
    if c is None:
        return ["no candidate row (run lh_debug.py intent)"]
    if c.get("split") != "train":
        bad.append(f"split {c.get('split')!r}, not train")
    if not c.get("gates_ok"):
        bad.append("candidate row: not every gate passed")
    wf = debug / "gen" / "store" / "train" / "tasks" / tid / "window.json"
    if not wf.is_file():
        return [*bad, "no window in the training store"]
    w = load(wf)
    built = row.get("built") or {}
    if w.get("a") != built.get("window_a"):
        bad.append(f"start commit {str(w.get('a'))[:10]}, the manifest records {str(built.get('window_a'))[:10]}")
    rec_a = (row.get("recorded") or {}).get("window_a")
    if rec_a and w.get("a") != rec_a:
        bad.append(f"start commit {str(w.get('a'))[:10]} is not the coauthor's recorded {rec_a[:10]}")
    if ((w.get("world") or {}).get("image")) != built.get("image"):
        bad.append(f"World image {(w.get('world') or {}).get('image')}, the manifest records {built.get('image')}")
    af, pf = debug / "authored" / f"{tid}.json", debug / "prompts" / f"{tid}.json"
    if not af.is_file() or not pf.is_file():
        bad.append("no authored prompt or bug report")
    else:
        if load(af).get("findings"):
            bad.append(f"authored prompt has lint findings: {load(af)['findings'][:2]}")
        if sha256_file(pf) != row.get("prompt_sha256"):
            bad.append("the bug report is not the one the manifest records (sha256)")
    gf = debug / "gates" / f"{tid}.json"
    g = load(gf) if gf.is_file() else {}
    if not g.get("ok") or not set(GATES) <= set(g.get("passed") or []):
        bad.append(f"gates: ok={g.get('ok')} passed={g.get('passed')}")
    elif g.get("window_a") and g["window_a"] != w.get("a"):  # a record from before the field: ``lh_debug.gates_match``
        bad.append("the gate record graded another start commit")
    bad += check_band([tid], model, {tid: {"model": model, **row["targeting"]}})
    if images:
        got = image_id(str(built.get("image")))
        if got != built.get("image_id"):
            bad.append(f"World image id {got}, the manifest records {built.get('image_id')}")
    return bad


def check(manifest: Mapping[str, Any], debug: Path, *, images: bool = False) -> dict[str, list[str]]:
    cands = {r["id"]: r for r in load(debug / "candidates.json")} if (debug / "candidates.json").is_file() else {}
    model = str(manifest["student"]).split(":", 1)[0]
    return {t: check_task(t, r, debug, cands, model, images=images) for t, r in sorted(ready(manifest).items())}


def relocate(debug: Path) -> list[str]:
    """Rewrite each working clone's alternates that point into ``<debug>/src`` relative to its objects dir."""
    done = []
    for alt in sorted((debug / "gen" / "clones").glob("*/.git/objects/info/alternates")):
        lines = alt.read_text().splitlines()
        new = []
        for ln in lines:
            p = Path(ln.strip())
            if p.is_absolute() and (debug / "src") in p.parents:
                rel = p.relative_to(debug)  # src/<name>/.git/objects
                ln = "../../../../../" + str(rel)  # from gen/clones/<repo>/.git/objects
            new.append(ln)
        if new != lines:
            alt.write_text("\n".join(new) + "\n")
            done.append(str(alt.relative_to(debug)))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "targeting", "relocate"])
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--debug", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--images", action="store_true")
    a = ap.parse_args()
    m = load(Path(a.manifest)) if a.cmd != "relocate" else {}
    debug = Path(a.debug or (ROOT / "data" / "debug-pool"))
    if a.cmd == "targeting":
        out = Path(a.out) if a.out else debug / "targeting.json"
        recs = targeting_records(m)
        out.write_text(json.dumps(recs, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"records": len(recs), "out": str(out)}))
    elif a.cmd == "check":
        res = check(m, debug, images=a.images)
        for t, bad in res.items():
            print(f"{'ok  ' if not bad else 'FAIL'} {t}" + ("" if not bad else "\n     " + "\n     ".join(bad)))
        blocked = {t: r.get("why") for t, r in m["tasks"].items() if r.get("status") != "ready"}
        for t, why in sorted(blocked.items()):
            print(f"skip {t}: {m['tasks'][t].get('status')}: {why}")
        sys.exit(1 if any(res.values()) else 0)
    else:
        print(json.dumps({"relocated": relocate(debug)}, indent=1))


if __name__ == "__main__":
    main()
