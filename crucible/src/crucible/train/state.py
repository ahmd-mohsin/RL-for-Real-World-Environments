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
state file, step, version). A sharded trainer (``FSDPTrainer``) writes ``trainer/`` there instead of weights in
``state.pt``: every rank its own shards (``torch.distributed.checkpoint``), no gather. Loading verifies the hash. The file is a pickle: load only checkpoints this run wrote.

Distributed trainers (M12 audit H4 V3): through ``spmd.SPMDDriver`` every rank loads the trainer part of the checkpoint
(``SPMDDriver.load_state`` → ``load_trainer_state`` on each follower; the directory must be on storage every rank
reads, as a sharded save already requires). A replicated trainer used to load on the driver only, so the replicas
diverged after the first resumed step.

Version consistency (H2 V12): a checkpoint must be taken right after a publish (``capture`` refuses a trainer that
moved past the last published weights), so the restored weights are exactly the restored version's. Game seats (H2
V20): the per-seat records of finished game rollouts travel with the checkpoint.
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
    pub = getattr(rt, "publisher", None)
    at = getattr(pub, "published_updates", None)
    if at is not None and int(at) != int(tr.updates):
        # H2 V12: the engines serve the weights of update ``at`` as ``rt.version``; this trainer is at ``tr.updates``. Resume
        # would re-publish the newer weights under the old version, so one version would name two weight sets.
        raise ValueError(f"run checkpoint at step {step}: the trainer is at update {tr.updates} but version {rt.version} was published at "
                         f"update {at}; checkpoint right after a publish (checkpoint_every a multiple of publish_every)")
    if getattr(tr, "sharded", False):  # a sharded trainer: each rank writes its shards beside state.pt (``save``)
        trainer: dict[str, Any] = {"sharded": True, "updates": tr.updates}
    else:
        trainer = {"model": {k: v.detach().cpu().clone() for k, v in tr.model.state_dict().items()},
                   "opt": copy.deepcopy(tr.opt.state_dict()), "updates": tr.updates}
    if getattr(tr, "critic", None) is not None:
        trainer.update(critic={k: v.detach().cpu().clone() for k, v in tr.critic.state_dict().items()}, critic_opt=copy.deepcopy(tr.critic_opt.state_dict()),
                       critic_updates=tr.critic_updates)
    from crucible.train.identity import identity

    return {
        "schema": "crucible.run_checkpoint/v1",
        "step": step,
        "version": rt.version,
        "identity": identity(rt),  # the settings this state was trained with: checked on resume (R-1 step 4)
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
        # H2 V20: a finished game rollout's per-seat records (``Runtime._seats``); without them a resumed game trained
        # only seat 0 under the plain group id
        "seats": {r: list(v) for r, v in getattr(rt, "_seats", {}).items() if r in pool.tags and pool.tags[r].status not in LIVE},
    }


def save(blob: dict, root: Path, trainer: Any = None) -> Path:
    """``trainer``: required for a sharded trainer (``capture`` marked it): its collective ``save_sharded`` writes
    ``trainer/`` into the same temp dir, so the rename publishes both at once (``root`` must be shared by the ranks)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step-{blob['step']:06d}"
    tmp = Path(tempfile.mkdtemp(prefix=".ckpt-", dir=root))
    if blob.get("trainer", {}).get("sharded"):
        if trainer is None:
            raise ValueError("a sharded trainer's checkpoint needs save(blob, root, trainer=...)")
        trainer.save_sharded(str(tmp / "trainer"))
    torch.save(blob, tmp / "state.pt")
    digest = hashlib.sha256((tmp / "state.pt").read_bytes()).hexdigest()
    ident = (blob.get("identity") or {}).get("fingerprint")
    (tmp / "manifest.json").write_text(json.dumps({"schema": blob["schema"], "step": blob["step"], "version": blob["version"], "sha256": digest,
                                                   **({"identity": ident} if ident else {})}, indent=1))
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
    blob = torch.load(path / "state.pt", weights_only=False)  # our own pickle (records, monitor): trusted by construction
    blob["_path"] = str(path)
    return blob


def restore(rt: Any, blob: dict) -> int:
    """Install ``blob`` into a freshly constructed runtime; returns the step to continue after. The runtime's settings are
    checked against the checkpoint's first (``train.identity.check``): a refused change installs nothing."""
    from crucible.train.identity import check

    if blob.get("schema") != "crucible.run_checkpoint/v1":
        raise ValueError(f"not a run checkpoint: {blob.get('schema')}")
    event = check(rt, blob)
    tr, sched, pool = rt.trainer, rt.sched, rt.pool
    if callable(getattr(type(tr), "load_state", None)):  # ``spmd.SPMDDriver``: every rank loads (H4 V3)
        tr.load_state(blob["_path"], blob)
    else:
        apply_trainer_state(tr, blob["trainer"], blob["_path"])
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
    if hasattr(rt, "_seats"):
        rt._seats = {r: list(v) for r, v in blob.get("seats", {}).items()}
    rt.version = blob["version"]
    rt.report.identity_check = event
    return int(blob["step"])


def apply_trainer_state(tr: Any, t: dict, path: str | Path | None = None) -> None:
    """Install the trainer part of a run checkpoint (``blob["trainer"]``) into ``tr`` on this rank. A sharded trainer
    reads its shards from ``<path>/trainer`` (a collective: every rank calls it)."""
    if t.get("sharded"):
        if path is None:
            raise ValueError("a sharded trainer's checkpoint needs its directory")
        tr.load_sharded(str(Path(path) / "trainer"))
    else:
        tr.model.load_state_dict(t["model"])
        tr.opt.load_state_dict(t["opt"])
    tr.updates = t["updates"]
    if "critic" in t:
        tr.critic.load_state_dict(t["critic"])
        tr.critic_opt.load_state_dict(t["critic_opt"])
        tr.critic_updates = t["critic_updates"]
    configured_lr(tr)


def configured_lr(tr: Any) -> None:
    """The optimizers run at the configured learning rates: loading an optimizer's state also loads its param groups, so a
    resume with another ``lr`` (allowed by ``allow_change``) kept the checkpoint's while the config named the new one
    (R-1 step 4). No Crucible trainer schedules its learning rate, so the configured value is the current one."""
    for opt, cfg in ((getattr(tr, "opt", None), getattr(tr, "cfg", None)), (getattr(tr, "critic_opt", None), getattr(tr, "critic_cfg", None))):
        lr = getattr(cfg, "lr", None)
        if opt is not None and isinstance(lr, (int, float)) and not isinstance(lr, bool):
            for g in opt.param_groups:
                g["lr"] = float(lr)


def load_trainer_state(tr: Any, path: str | Path) -> None:
    """A follower's half of ``SPMDDriver.load_state``: read and verify the run checkpoint at ``path``, install its
    trainer part."""
    blob = load(Path(path))
    apply_trainer_state(tr, blob["trainer"], blob["_path"])
