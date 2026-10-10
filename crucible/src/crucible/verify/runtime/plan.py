"""Grade plans: gates made of checks, run in order, with explicit verdict semantics. Standard library only.

A **check** is a function ``fn(ctx) -> Verdict | bool | dict``. Its status is one of:

- ``pass`` / ``fail``: the candidate's result. Any ordinary exception raised while a check judges candidate output
  (malformed files, missing artifacts) is a **fail**: a candidate must not be able to turn a 0 into a masked grade by
  crashing the grader;
- ``infra``: an ``InfraError`` (the reference build failed, the node is busy, a cohort survived): the grade is masked,
  never scored, and ``reward.json`` says why;
- ``skip``: a gate it requires failed. A skipped check counts as not passed (reward 0), as in the seed graders.

A **gate** passes iff every check in it passes. Gates have a ``kind``: ``gate`` (a hard constraint: a failure zeroes
the reward), ``outcome`` (task success) or ``score`` (record-style, ``patterns.record``: the reward is the mean
normalized score, unbounded above 1 for a new record, instead of 1 for clearing a bar). With no outcome or score gates,
the outcome is "every gate passed" (coding tasks).

**Abstention** (``Plan.abstain`` + ``Plan.abstained``): when the agent gives up honestly, only hard gates are judged and
the reward is the policy's credit (full on a task constructed infeasible, 0 on a feasible one, a small constant on an
open problem); on an infeasible task, any claimed solution earns 0.

``Plan.run`` always produces a ``GradeReport``; ``write_outputs`` always writes ``reward.txt`` (never empty), a flat
numeric ``reward.json`` (what Harbor accepts), ``crucible_reward.json`` (the full multi-key document) and ``report.json``. An exception anywhere outside a check (setup) yields reward 0 with
``infra.status = masked`` and ``failure_kind = grader_exception`` (DESIGN §8.3 layer 7).

Tiers (DESIGN §10.2): each check is ``fast`` or ``full``; ``Plan.run(tier="fast")`` runs only fast checks.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.verify.runtime.trusted import InfraError

STATUSES = ("pass", "fail", "infra", "skip")


@dataclass
class Verdict:
    passed: bool
    detail: dict = field(default_factory=dict)
    infra: str | None = None  # set => status infra with this reason


@dataclass
class CheckResult:
    id: str
    gate: str
    status: str
    detail: dict = field(default_factory=dict)
    seconds: float = 0.0
    provenance: str | None = None
    tier: str = "full"


@dataclass
class Check:
    id: str
    gate: str
    fn: Callable[[Any], Any]
    tier: str = "full"  # "fast" checks also run in the fast (training) tier
    requires: tuple = ()  # gates that must have passed, else this check is skipped
    provenance: str | None = None  # the expert signal, incident or reference-differential it came from (A8)
    cost: str = "cheap"


@dataclass
class Gate:
    id: str
    kind: str = "gate"  # "gate" | "outcome"
    description: str = ""


@dataclass
class GradeReport:
    task: str
    tier: str
    checks: list[CheckResult]
    gates: dict[str, str]  # gate -> pass | fail | infra | skip
    reward: float
    outcome: float
    infra: dict
    trusted: bool
    elapsed_s: float
    meta: dict = field(default_factory=dict)

    def gate_bits(self) -> dict[str, int]:
        return {g: int(s == "pass") for g, s in self.gates.items()}

    def failing_gates(self) -> list[str]:
        return [g for g, s in self.gates.items() if s != "pass"]

    def reward_json(self) -> dict:
        """``crucible.core.schema.reward.RewardJson``-compatible. ``gates`` holds the hard constraints only; outcome and
        score gates go to ``components`` as ``check/<gate>`` (an abstention skips them without breaking a gate)."""
        kinds = self.meta.get("kinds", {})
        bits = self.gate_bits()
        comps = {f"check/{g}": float(v) for g, v in bits.items() if kinds.get(g, "gate") != "gate"}
        comps.update({f"time/{c.id}": round(c.seconds, 3) for c in self.checks if c.seconds})
        return {"reward": self.reward, "gates": {g: v for g, v in bits.items() if kinds.get(g, "gate") == "gate"}, "outcome": self.outcome,
                "infra": self.infra, "components": comps}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["failing_gates"] = self.failing_gates()
        return d


class Context:
    """What checks see: the trusted runner, the scratch root, task paths, and a shared scratch dict for artifacts
    produced by earlier checks (e.g. the build's install prefix)."""

    def __init__(self, trusted, *, task_dir: Path | None = None, candidate: Path | None = None, params: dict | None = None):
        self.trusted = trusted
        self.work = trusted.work
        self.task_dir = Path(task_dir) if task_dir else None
        self.candidate = Path(candidate) if candidate else None
        self.params = dict(params or {})
        self.artifacts: dict[str, Any] = {}
        self.revealed: dict[str, Any] = {}  # secrets (fresh seeds) revealed in the report only after the candidate ran


@dataclass
class Plan:
    task: str
    gates: list[Gate]
    checks: list[Check]
    setup: Callable[[Context], None] | None = None  # root-only preparation (references, owned copies) before checks
    abstain: Any = None  # patterns.record.AbstainPolicy: the task allows an honest give-up
    abstained: Callable[[Context], bool] | None = None  # did the agent give up (an ABSTAIN artifact / tool call)?

    def __post_init__(self) -> None:
        ids = [g.id for g in self.gates]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate gate ids")
        cids = [c.id for c in self.checks]
        if len(set(cids)) != len(cids):
            raise ValueError("duplicate check ids")
        for c in self.checks:
            if c.gate not in ids:
                raise ValueError(f"check {c.id} names unknown gate {c.gate}")
            if set(c.requires) - set(ids):
                raise ValueError(f"check {c.id} requires unknown gates {sorted(set(c.requires) - set(ids))}")
        empty = [g for g in ids if not any(c.gate == g for c in self.checks)]
        if empty:
            raise ValueError(f"gates with no checks: {empty}")

    def subset(self, tier: str) -> list[Check]:
        return [c for c in self.checks if tier == "full" or c.tier == "fast"]

    def run(self, ctx: Context, **kw: Any) -> GradeReport:
        """Synchronous entry (verifier images); checks may still be coroutine functions."""
        return asyncio.run(self.arun(ctx, **kw))

    async def arun(self, ctx: Context, *, tier: str = "full", stop_on_fail: bool = False, on_result: Callable | None = None) -> GradeReport:
        t0 = time.time()
        results: list[CheckResult] = []
        infra = {"status": "ok", "failure_kind": None, "reason": None}
        status: dict[str, str] = {}
        try:
            if self.setup:
                r0 = self.setup(ctx)
                if inspect.isawaitable(r0):
                    await r0
            gate_state: dict[str, list[str]] = {g.id: [] for g in self.gates}
            gave_up = bool(self.abstain and self.abstained and self.abstained(ctx))
            hard_ids = {g.id for g in self.gates if g.kind == "gate"}
            for c in self.subset(tier):
                if gave_up and c.gate not in hard_ids:  # an abstention is judged on the hard constraints only
                    r = CheckResult(c.id, c.gate, "skip", {"skipped": "the agent abstained"}, provenance=c.provenance, tier=c.tier)
                    results.append(r)
                    gate_state[c.gate].append(r.status)
                    continue
                failed_reqs = [g for g in c.requires if _gate_status(gate_state[g]) != "pass"]
                if failed_reqs or (stop_on_fail and any(_gate_status(v) in ("fail", "infra", "skip") for v in gate_state.values() if v)):
                    r = CheckResult(c.id, c.gate, "skip", {"skipped": f"requires {failed_reqs}" if failed_reqs else "an earlier gate failed"},
                                    provenance=c.provenance, tier=c.tier)
                else:
                    r = await _run_check(c, ctx)
                results.append(r)
                gate_state[c.gate].append(r.status)
                if on_result:
                    on_result(r)
            status = {g.id: _gate_status(gate_state[g.id]) if gate_state[g.id] else "skip" for g in self.gates}
            infra_checks = [r for r in results if r.status == "infra"]
            if infra_checks:
                infra = {"status": "masked", "failure_kind": "check_infra", "reason": f"{infra_checks[0].id}: {infra_checks[0].detail.get('infra', '')}"[:500]}
        except Exception as ex:  # noqa: BLE001 -- setup or the runner itself: never a candidate verdict
            infra = {"status": "masked", "failure_kind": "grader_exception", "reason": f"{type(ex).__name__}: {ex}"[:500]}
            results.append(CheckResult("_grader", "_grader", "infra", {"traceback": traceback.format_exc()[-3000:]}))
            status = {g.id: status.get(g.id, "skip") for g in self.gates}
        hard = [g.id for g in self.gates if g.kind == "gate"]
        outs = [g.id for g in self.gates if g.kind == "outcome"]
        scores = [g.id for g in self.gates if g.kind == "score"]
        if tier == "fast":  # gates with no fast check are not measured in the fast tier
            measured = {c.gate for c in self.subset("fast")}
            hard, outs, scores = [g for g in hard if g in measured], [g for g in outs if g in measured], [g for g in scores if g in measured]
            status = {g: s for g, s in status.items() if g in measured}
        all_hard = all(status.get(g) == "pass" for g in hard)
        gave_up = bool(self.abstain and self.abstained and self.abstained(ctx)) if infra["status"] == "ok" else False
        meta: dict = {"revealed": ctx.revealed, "kinds": {g.id: g.kind for g in self.gates}}
        if gave_up:
            outcome = float(self.abstain.feasibility == "infeasible")
            reward = self.abstain.credit() if all_hard else 0.0
            meta["abstained"] = {"feasibility": self.abstain.feasibility, "credit": reward}
        else:
            outcome = float(all(status.get(g) == "pass" for g in outs + scores)) if outs or scores else float(all_hard)
            if self.abstain and self.abstain.feasibility == "infeasible":
                outcome = 0.0  # the task cannot be done: any claimed solution is wrong
            term = 1.0
            if scores:  # record-style: reward grows with the normalized score instead of clearing a bar
                vals = [r.detail.get("normalized_reward", 0.0) for r in results if r.gate in scores and r.status == "pass"]
                term = sum(vals) / len(scores) if vals else 0.0
                meta["score"] = {r.id: {k: r.detail.get(k) for k in ("score", "normalized", "new_record")} for r in results if r.gate in scores}
            reward = term if infra["status"] == "ok" and all_hard and (not outs or all(status.get(g) == "pass" for g in outs)) and outcome > 0 else 0.0
            if not scores:
                reward = 1.0 if reward and outcome == 1.0 else 0.0
        trusted = bool(getattr(ctx.trusted, "is_root", False))
        return GradeReport(self.task, tier, results, status, reward, outcome if infra["status"] == "ok" else 0.0, infra, trusted,
                           round(time.time() - t0, 3), meta=meta)


def _gate_status(states: list[str]) -> str:
    if not states:
        return "pass"
    if "infra" in states:
        return "infra"
    if "fail" in states:
        return "fail"
    if "skip" in states:
        return "skip"
    return "pass"


async def _run_check(c: Check, ctx: Context) -> CheckResult:
    t0 = time.monotonic()
    try:
        v = c.fn(ctx)
        if inspect.isawaitable(v):
            v = await v
        if isinstance(v, Verdict):
            st = "infra" if v.infra else ("pass" if v.passed else "fail")
            detail = dict(v.detail)
            if v.infra:
                detail["infra"] = v.infra
        elif isinstance(v, dict):
            st, detail = ("pass" if v.get("pass") else "fail"), v
        else:
            st, detail = ("pass" if v else "fail"), {}
    except InfraError as ex:
        st, detail = "infra", {"infra": str(ex)[:2000]}
    except Exception as ex:  # noqa: BLE001 -- judging candidate output: a crash is the candidate's failure
        st, detail = "fail", {"error": f"{type(ex).__name__}: {ex}"[:2000], "traceback": traceback.format_exc()[-1500:]}
    return CheckResult(c.id, c.gate, st, detail, round(time.monotonic() - t0, 3), c.provenance, c.tier)


def flat_reward(rj: dict) -> dict:
    """Harbor's ``reward.json`` must be a flat dict of finite numbers (``harbor/verifier/verifier.py:104-121`` raises
    on anything else): ``reward``, ``outcome``, ``gate/<g>``, ``infra/masked``, ``progress``."""
    flat = {"reward": float(rj.get("reward", 0.0)), "outcome": float(rj.get("outcome", 0.0))}
    flat.update({f"gate/{g}": float(v) for g, v in (rj.get("gates") or {}).items()})
    flat.update({k: float(v) for k, v in (rj.get("components") or {}).items() if k.startswith("check/")})
    flat["infra/masked"] = float((rj.get("infra") or {}).get("status") == "masked")
    if rj.get("progress") is not None:
        flat["progress"] = float(rj["progress"])
    return flat


def prewrite_zero(out_dir: Path) -> None:
    """Before grading starts: an interrupted grader leaves reward 0, never a stale or agent-written reward."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("reward.json", "crucible_reward.json", "report.json"):
        (out_dir / name).unlink(missing_ok=True)
    (out_dir / "reward.txt").write_text("0\n")
    (out_dir / "reward.json").write_text(json.dumps({"reward": 0.0, "infra/masked": 1.0}) + "\n")


def write_outputs(report: GradeReport | None, out_dir: Path, *, error: str | None = None) -> dict:
    """Write ``reward.txt`` (never empty), a flat numeric ``reward.json`` (Harbor/NeMo-Gym readable), the full
    ``crucible_reward.json`` (``core.schema.reward.RewardJson``) and ``report.json``. With no report (the grader
    crashed before producing one) the reward is 0 and the grade is masked as ``grader_exception``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if report is None:
        rj = {"reward": 0.0, "gates": {}, "outcome": 0.0, "infra": {"status": "masked", "failure_kind": "grader_exception", "reason": (error or "")[:500]}}
        full = {"reward": 0.0, "error": error}
    else:
        rj = report.reward_json()
        full = report.to_dict()
    (out_dir / "crucible_reward.json").write_text(json.dumps(rj, indent=1, default=str) + "\n")
    (out_dir / "report.json").write_text(json.dumps(full, indent=1, default=str) + "\n")
    (out_dir / "reward.json").write_text(json.dumps(flat_reward(rj), indent=1) + "\n")
    (out_dir / "reward.txt").write_text(f"{rj['reward']:g}\n")
    return rj
