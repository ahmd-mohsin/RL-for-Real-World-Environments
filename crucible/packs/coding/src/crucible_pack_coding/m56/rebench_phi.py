"""M5/M6 evaluation on SWE-rebench OpenHands trajectories (held-out, public; see ``corpora/swe_rebench.py``).

For each sampled issue: mine solution-first milestones from the gold patch (source files only; tests excluded);
for each attempt: replay its edits (M5 replay test; quarantined attempts are excluded and counted); evaluate every
milestone on the reconstructed state at 25/50/75/100% of the attempt's tool calls. One JSONL row per attempt is
cached (``data/m56/rebench_phi.jsonl``), so the M6 analyses (time-split type weights, Φ vs judge AUC) re-run cheaply.

``python -m crucible_pack_coding.m56.rebench_phi <n_instances> [seed]``
"""

from __future__ import annotations

import asyncio
import collections
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from crucible.workflow.filestate import FileState
from crucible.workflow.milestones.evaluate import FileStateView, evaluate
from crucible.workflow.milestones.mine import from_diff
from crucible.workflow.replay import EditReplayer
from crucible_pack_coding.corpora import swe_rebench as S

OUT = S.DATA.parent.parent.parent / "m56"
TEST = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$")
FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def mtype(mid: str) -> str:
    """created | deleted | located | removed | removed_call | added | added_call"""
    return mid.split("/")[1]


def sample(n: int, seed: int) -> list[str]:
    idx = S.index()
    by = collections.defaultdict(list)
    for r in idx:
        by[r["instance_id"]].append(r)
    mixed = sorted(i for i, rs in by.items() if 0 < sum(x["resolved"] for x in rs) < len(rs))
    return random.Random(seed).sample(mixed, n)


async def attempt(rec: dict[str, Any], meta: dict[str, Any], milestones: list) -> dict[str, Any]:
    base = S.raw_reader(meta["repo"], meta["base_commit"])
    root = S.workspace_root(rec)
    tr = S.to_trace(rec, meta)
    rp = EditReplayer(base, rec["model_patch"] or "", path_of=S.path_mapper(root), cwd=(root or "/").rstrip("/"), ignore=S.harness_file)
    rr = rp.replay(tr)
    calls = [e.seq for e in tr.events if e.type == "tool_call"]
    row: dict[str, Any] = {"trajectory_id": rec["trajectory_id"], "instance_id": meta["instance_id"], "repo": meta["repo"],
                           "created_at": str(meta["created_at"]), "resolved": rec["resolved"], "n_calls": len(calls),
                           "replay_strict": rr.ok, "replay_agent_state": bool(rr.meta.get("ok_agent_state")), "phi": {}}
    if not row["replay_agent_state"] or not milestones:
        return row
    states, _ = rp.states(tr)
    for f in FRACTIONS:
        cut = calls[min(len(calls) - 1, max(0, round(f * len(calls)) - 1))] if calls else 0
        upto = [fs for seq, fs in states if seq <= cut]
        fs = upto[-1] if upto else FileState(base=base)
        row["phi"][str(f)] = await evaluate(milestones, FileStateView(fs))
    return row


def run(n: int, seed: int = 0) -> Path:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"rebench_phi_n{n}_s{seed}.jsonl"
    done = {json.loads(x)["trajectory_id"] for x in out.read_text().splitlines()} if out.exists() else set()
    picks = sample(n, seed)
    inst = S.instances(["instance_id", "repo", "base_commit", "patch", "created_at"])
    idx = [r for r in S.index() if r["instance_id"] in set(picks) and r["trajectory_id"] not in done]
    mil: dict[str, list] = {}
    meta_out = OUT / f"rebench_milestones_n{n}_s{seed}.jsonl"
    with meta_out.open("a") as mf:
        for i in picks:
            m = inst[i]
            try:
                mil[i] = [c.milestone for c in from_diff(m["patch"], S.raw_reader(m["repo"], m["base_commit"]), include=lambda p: not TEST.search(p))]
            except Exception as e:  # noqa: BLE001 -- a gold patch that does not apply to the fetched base: no milestones
                mil[i] = []
                print("mining failed", i, str(e)[:120])
            mf.write(json.dumps({"instance_id": i, "milestones": [x.model_dump(mode="json") for x in mil[i]]}) + "\n")
    with out.open("a") as f:
        for k, rec in enumerate(S.load(idx)):
            iid = next(r["instance_id"] for r in idx if r["trajectory_id"] == rec["trajectory_id"])
            row = asyncio.run(attempt(rec, inst[iid], mil[iid]))
            row["milestone_types"] = {x.id: mtype(x.id) for x in mil[iid]}
            f.write(json.dumps(row) + "\n")
            if k % 50 == 0:
                print(f"{k}/{len(idx)} attempts, {time.time() - t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    print(run(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0))
