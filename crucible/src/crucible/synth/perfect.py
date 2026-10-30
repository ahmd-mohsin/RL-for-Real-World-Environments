"""The perfect-task check (part2-paper-core §0): admission into the training pool requires P1–P4 together.

``perfect(task, pool, records, ...)`` → ``PerfectVerdict`` with one boolean per property and the reasons:
- **valid** (P1): the task's certificate status is ``certified`` (``provisional`` and ``rejected`` are not valid unless
  ``accept_provisional``); a task without a certificate is unmeasured, not valid;
- **learnable** (P2): a measured (or trusted-predictor) ``DifficultyRecord`` for this task version and the given student
  whose decision is ``in_band``; ``require_confident`` also demands the band test's posterior reached 1 − α;
- **on_intent** (P3): ``intent.P3Result.on_intent`` is True;
- **novel** (P4): ``novelty.novel`` against the pool's signatures (its key new, no near duplicate at the calibrated θ).
``admit`` is True only when all four are True; a property that could not be measured is False with an
``unmeasured`` reason. ``export_schemas`` gives the JSON schemas of the records the RSI loop reads (DESIGN item 19).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.cegis import CegisReport
from crucible.synth.intent import P3Result
from crucible.synth.learn import BandExample, BandVerdict, DifficultyRecord
from crucible.synth.novelty import DECISION_GATE, DEFAULT_THETA, DEFAULT_W, Signature, novel
from crucible.synth.task import Task


class PerfectVerdict(CrucibleModel):
    task_id: str
    version: int
    valid: bool
    learnable: bool
    on_intent: bool
    novel: bool
    admit: bool
    reasons: dict[str, list[str]] = Field(default_factory=dict)
    student: str | None = None


def perfect(task: Task, pool: Sequence[Signature], records: Sequence[DifficultyRecord], *, status: str | None, p3: P3Result | None,
            signature: Signature | None, student: str | None = None, accept_provisional: bool = False, require_confident: bool = False,
            w: float = DEFAULT_W, theta: float = DEFAULT_THETA, gate: float = DECISION_GATE) -> PerfectVerdict:
    reasons: dict[str, list[str]] = {"valid": [], "learnable": [], "on_intent": [], "novel": []}
    ok_status = {"certified", "admitted"} | ({"provisional"} if accept_provisional else set())
    valid = status in ok_status
    if not valid:
        reasons["valid"].append("unmeasured: no certificate" if status is None else f"certificate status {status}")
    recs = [r for r in records if r.key.task_id == task.id and r.key.task_version == task.version and (student is None or r.key.student == student)]
    rec = max(recs, key=lambda r: r.at) if recs else None
    learnable = bool(rec is not None and rec.decision == "in_band" and (rec.confident or not require_confident))
    if rec is None:
        reasons["learnable"].append("unmeasured: no difficulty record for this version and student")
    elif rec.decision != "in_band":
        reasons["learnable"].append(f"{rec.decision or 'undecided'} ({rec.source}; posterior error {rec.error})")
    elif require_confident and not rec.confident:
        reasons["learnable"].append(f"in band but not confident (posterior error {rec.error})")
    on = bool(p3 is not None and p3.on_intent)
    if p3 is None:
        reasons["on_intent"].append("unmeasured: no P3 result")
    elif p3.on_intent is None:
        reasons["on_intent"].append("unmeasured: " + "; ".join(p3.reasons))
    elif not p3.on_intent:
        reasons["on_intent"] += p3.reasons or ["off intent"]
    if signature is None:
        nv, reasons["novel"] = False, ["unmeasured: no signature"]
    else:
        n = novel(signature, pool, w=w, theta=theta, gate=gate)
        nv = n.novel
        reasons["novel"] += n.reasons
    return PerfectVerdict(task_id=task.id, version=task.version, valid=valid, learnable=learnable, on_intent=on, novel=nv,
                          admit=valid and learnable and on and nv, reasons={k: v for k, v in reasons.items() if v}, student=student)


def admit_pool(entries: Sequence[tuple[Task, str | None, P3Result | None, Signature | None]], records: Sequence[DifficultyRecord], *,
               student: str | None = None, **kw: Any) -> tuple[list[PerfectVerdict], dict[str, Any]]:
    """``perfect`` over candidates in order; each admitted task's signature joins the pool before the next is checked
    (the first of two duplicates is the novel one)."""
    pool: list[Signature] = []
    seen: list[Signature] = []  # every earlier candidate, admitted or not: novelty of the batch itself
    out = []
    novel_vs_seen = 0
    for task, status, p3r, sig in entries:
        v = perfect(task, pool, records, status=status, p3=p3r, signature=sig, student=student, **kw)
        out.append(v)
        if sig is not None:
            nk = {k: kw[k] for k in ("w", "theta", "gate") if k in kw}
            novel_vs_seen += int(novel(sig, seen, **nk).novel)
            seen.append(sig)
        if v.admit and sig is not None:
            pool.append(sig)
    n = len(out)
    return out, {"candidates": n, **{k: sum(1 for v in out if getattr(v, k)) for k in ("valid", "learnable", "on_intent", "novel", "admit")},
                 "novel_vs_earlier_candidates": novel_vs_seen}


class PoolItem(CrucibleModel):
    task: Task
    status: str  # the certificate's status: certified | provisional | rejected (the run report's outcome)
    run: str
    n: int
    cell: str | None = None  # the proposer's cell for this task (proposals.json)
    alt: list[int] | None = None  # [passes, n] of the certification's ALT solvers (non-student proxy rollouts)


def load_pool(domain_dir: str | Path, *, statuses: Sequence[str] = ("certified", "provisional"), runs: Sequence[str] | None = None) -> list[PoolItem]:
    """Every task of a domain's synthesis runs with its certificate status, proposer cell and ALT history (the
    training side only: a path under ``data/sealed`` is refused)."""
    root = Path(domain_dir)
    if "sealed" in root.resolve().parts:
        raise ValueError(f"{root} is under a sealed root: never a training pool")
    want = set(statuses)
    out: list[PoolItem] = []
    for run in sorted(p for p in root.iterdir() if (p / "report.json").exists()) if root.exists() else []:
        if runs is not None and run.name not in runs:
            continue
        rep = json.loads((run / "report.json").read_text())
        props = {}
        if (run / "proposals.json").exists():
            props = {int(p["n"]): p for p in json.loads((run / "proposals.json").read_text()) if "n" in p}
        for t in rep.get("tasks", []):
            st = {"admitted": "certified"}.get(t.get("status"), t.get("status"))
            if st not in want:
                continue
            p = run / "tasks" / f"t{int(t['n']):02d}" / "cegis.json"
            if not p.exists():
                continue
            raw = json.loads(p.read_text())
            c = CegisReport.model_validate(raw)
            if c.task is None:
                continue
            alt = ((raw.get("certificate") or {}).get("classes") or {}).get("ALT") or {}
            out.append(PoolItem(task=c.task, status=st, run=run.name, n=int(t["n"]), cell=(props.get(int(t["n"])) or {}).get("cell"),
                                alt=[int(alt.get("passes") or 0), int(alt["n"])] if alt.get("n") else None))
    return out


def export_schemas() -> dict[str, Any]:
    """JSON schemas of the derived records the RSI loop reads (DESIGN §7.7 item 19)."""
    from crucible.synth.frontier import Attribution, TargetRequest
    from crucible.synth.outcome import OutcomeRecord, TaskEvidence

    return {"BandVerdict": BandVerdict.model_json_schema(), "BandExample": BandExample.model_json_schema(),
            "Attribution": Attribution.model_json_schema(), "TargetRequest": TargetRequest.model_json_schema(),
            "PerfectVerdict": PerfectVerdict.model_json_schema(), "OutcomeRecord": OutcomeRecord.model_json_schema(),
            "TaskEvidence": TaskEvidence.model_json_schema()}
