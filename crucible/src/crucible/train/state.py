"""The state plane: run-level checkpoint / resume (M12 checklist #1; article §1 "State Plane", §3.2 Ling partial rollout
"defined boundaries per round facilitate checkpointing and fault recovery").

A checkpoint is taken at the sample barrier (after a step's publish), the one point where the run is consistent:

- **trainer**: policy weights + optimizer moments (+ critic and its optimizer), update counters;
- **weight plane**: the published policy version (resume re-publishes the restored weights to the engines *at that
  version*, so every later response is stamped correctly);
- **control plane**: scheduler counters, per-source admitted counts, exhausted sources, the steps taken, the length
  history (length-bias monitor), pending eviction staleness; the source queues' remaining prompts; the rollout seed
  counter;
- **data plane**: every pool tag and group, and the values (``crucible.rollout/v1`` records) of finished, untrained
  rollouts — they train after resume exactly as they would have;
- **in-flight rollouts** (pending / running / grading at the barrier): restarted from scratch on resume (status
  pending, versions cleared) and counted in ``restarted_rollouts``; their partial generations are lost. (Resuming a
  rollout mid-episode needs the environment's own checkpoint — M10 ``runtime.checkpoint`` for Worlds — and is a
  rollout-function capability, not a run-level one.)
- **monitors and report**: alarm history, length window, counters.

Files: ``<dir>/step-<k>/state.pt`` written to a temp dir and renamed (atomic), plus ``manifest.json`` (sha256 of the
state file, step, version). Loading verifies the hash. The file is a pickle: load only checkpoints this run wrote.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from crucible.train.pool import LIVE, Group, Tag


def capture(rt: Any, step: int) -> dict:
    tr, sched, pool = rt.trainer, rt.sched, rt.pool
    trainer: dict[str, Any] = {"model": {k: v.detach().cpu().clone() for k, v in tr.model.state_dict().items()},
                               "opt": copy.deepcopy(tr.opt.state_dict()), "updates": tr.updates}
    if getattr(tr, "critic", None) is not None:
        trainer.update(critic={k: v.detach().cpu().clone() for k, v in tr.critic.state_dict().items()}, critic_opt=copy.deepcopy(tr.critic_opt.state_dict()),
                       critic_updates=tr.critic_updates)
    return {
        "schema": "crucible.run_checkpoint/v1",
        "step": step,
        "version": rt.version,
        "trainer": trainer,
        "scheduler": {"admitted": dict(sched.admitted), "exhausted": sorted(sched.exhausted), "steps": sched.steps, "lengths_seen": list(sched.lengths_seen),
                      "counters": dict(sched.counters), "ev_pending": list(sched._ev_pending)},
        "queues": {s: list(q) for s, q in rt._queues.items()},
        "seeds": rt._seeds,
        "pool": {"tags": {r: asdict(t) for r, t in pool.tags.items()}, "groups": {g: asdict(x) for g, x in pool.groups.items()},
                 "values": {r: v for r, v in pool.values.items() if pool.tags[r].status not in LIVE}, "next_index": pool._next_index, "rid": pool._rid},
        "monitor": copy.deepcopy(rt.monitor),
        "report": {k: getattr(rt.report, k) for k in ("aborted_calls", "resumed_calls", "reprefill_tokens", "publishes", "max_untrained", "masked_rollouts",
                                                       "late_rewards", "reward_timeouts", "credit_fallbacks", "failures", "restarted_rollouts")},
        "admission_blocked_s": rt.admission_blocked_s,
    }


def save(blob: dict, root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step-{blob['step']:06d}"
    tmp = Path(tempfile.mkdtemp(prefix=".ckpt-", dir=root))
    torch.save(blob, tmp / "state.pt")
    digest = hashlib.sha256((tmp / "state.pt").read_bytes()).hexdigest()
    (tmp / "manifest.json").write_text(json.dumps({"schema": blob["schema"], "step": blob["step"], "version": blob["version"], "sha256": digest}, indent=1))
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)  # atomic on one filesystem: a crash leaves the previous checkpoint intact
    return final


def latest(root: Path) -> Path | None:
    cands = sorted(p for p in Path(root).glob("step-*") if (p / "manifest.json").exists())
    return cands[-1] if cands else None


def load(path: Path) -> dict:
    path = Path(path)
    man = json.loads((path / "manifest.json").read_text())
    data = (path / "state.pt").read_bytes()
    if hashlib.sha256(data).hexdigest() != man["sha256"]:
        raise ValueError(f"checkpoint {path} is corrupt (sha256 mismatch)")
    return torch.load(path / "state.pt", weights_only=False)  # our own pickle (records, monitor): trusted by construction


def restore(rt: Any, blob: dict) -> int:
    """Install ``blob`` into a freshly constructed runtime (same configuration); returns the step to continue after."""
    if blob.get("schema") != "crucible.run_checkpoint/v1":
        raise ValueError(f"not a run checkpoint: {blob.get('schema')}")
    tr, sched, pool = rt.trainer, rt.sched, rt.pool
    t = blob["trainer"]
    tr.model.load_state_dict(t["model"])
    tr.opt.load_state_dict(t["opt"])
    tr.updates = t["updates"]
    if "critic" in t:
        tr.critic.load_state_dict(t["critic"])
        tr.critic_opt.load_state_dict(t["critic_opt"])
        tr.critic_updates = t["critic_updates"]
    s = blob["scheduler"]
    sched.admitted.update(s["admitted"])
    sched.exhausted = set(s["exhausted"])
    sched.steps, sched.lengths_seen, sched.counters, sched._ev_pending = s["steps"], list(s["lengths_seen"]), dict(s["counters"]), list(s["ev_pending"])
    for src, q in blob["queues"].items():
        rt._queues[src][:] = q
    rt._seeds = blob["seeds"]
    p = blob["pool"]
    pool.tags = {r: Tag(**d) for r, d in p["tags"].items()}
    pool.groups = {g: Group(**d) for g, d in p["groups"].items()}
    pool.values = dict(p["values"])
    pool._next_index, pool._rid = p["next_index"], p["rid"]
    restarted = 0
    for tag in pool.tags.values():
        if tag.status in LIVE:  # in flight at the barrier: restart from scratch
            restarted += tag.status != "pending"
            tag.status, tag.versions, tag.length, tag.reward, tag.done_at, tag.cuts = "pending", [], 0, None, None, 0
    for g in pool.groups.values():
        if g.status == "terminal" and any(pool.tags[r].status == "pending" for r in g.members):
            g.status = "open"
    pool.reindex()
    rt.monitor = blob["monitor"]
    for k, v in blob["report"].items():
        setattr(rt.report, k, v)
    rt.report.restarted_rollouts += restarted
    rt.admission_blocked_s = blob["admission_blocked_s"]
    rt.version = blob["version"]
    return int(blob["step"])
