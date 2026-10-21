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

``Plan.run`` always produces a ``GradeReport``; ``write_outputs`` always writes ``crucible_reward.json`` (the full
multi-key document) and ``report.json``, and ``reward.txt`` plus a flat numeric ``reward.json`` (what Harbor accepts)
only for a real grade. An exception anywhere outside a check (setup) yields a masked grade with ``infra.status =
masked`` and ``failure_kind = grader_exception`` (DESIGN §8.3 layer 7), except a setup error the candidate's content
caused (``CandidateError``, or a copy error inside the candidate's tree), which fails every hard gate (M7 audit V16).

Tiers (DESIGN §10.2): each check is ``fast`` or ``full``; ``Plan.run(tier="fast")`` runs only fast checks.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import shutil
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.verify.runtime.trusted import CandidateError, InfraError

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


GATE_KINDS = ("gate", "outcome", "score")


@dataclass
class Gate:
    id: str
    # "gate" (a rule the instruction states: a failure zeroes the reward and the attempt's per-turn credit), "outcome"
    # (task success) or "score" (a graded bar). No default (M7 R6): a silent default was a silent credit decision (M6 R2)
    kind: str
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

    def gate_kinds(self) -> dict[str, str]:
        """Per gate: ``"gate"`` (a hard rule) or the declared kind of an outcome or score check (``meta["kinds"]``).
        Records carry this beside ``gate_bits()`` so per-turn credit can tell a broken rule from an unmet outcome (M6 R2).

        M7 R6: every gate of a graded report must be declared; an undeclared one raises, naming it, instead of being read
        as a hard rule (which zeroed an honest outcome's per-turn credit). A masked report is never credited, so it returns
        what it has, an undeclared gate read as a hard rule as before."""
        kinds = self.meta.get("kinds") or {}
        missing = [g for g in self.gates if g not in kinds]
        if missing and (self.infra or {}).get("status", "ok") == "ok":
            raise ValueError(f"{self.task}: gates {missing} declare no kind (meta['kinds']: 'gate', 'outcome' or 'score')")
        return {g: str(kinds.get(g, "gate")) for g in self.gates}

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
        bad = {g.id: g.kind for g in self.gates if g.kind not in GATE_KINDS}
        if bad:
            raise ValueError(f"gates with an unknown kind {bad}: each is one of {list(GATE_KINDS)}")
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
        fast = {c.gate for c in self.checks if c.tier == "fast"}
        # a hard gate the fast tier never measures lets the fast tier pay what the full tier rejects (M7 audit V17)
        self.fast_unmeasured_hard = sorted(g.id for g in self.gates if g.kind == "gate" and g.id not in fast) if fast else []

    def _gave_up(self, ctx: Context) -> bool:
        """Whether the agent abstained; an error reading the candidate's artifact means it did not (M7 audit V16: a
        directory named ABSTAIN raised and masked the grade)."""
        if not (self.abstain and self.abstained):
            return False
        try:
            return bool(self.abstained(ctx))
        except Exception:  # noqa: BLE001 - candidate-controlled content
            return False

    def _candidate_setup_fail(self, ctx: Context, tier: str, ex: Exception, t0: float) -> GradeReport:
        """M7 audit V16: candidate content that breaks ``setup`` (a FIFO in a tree the setup copies) masked the grade.
        It is the candidate's failure: every hard gate fails, reward 0, infra ok."""
        detail = {"error": f"setup could not use the candidate's content: {type(ex).__name__}: {ex}"[:1500]}
        results = [CheckResult(f"_setup:{g.id}", g.id, "fail", detail) for g in self.gates if g.kind == "gate"]
        status = {g.id: ("fail" if g.kind == "gate" else "skip") for g in self.gates}
        return GradeReport(self.task, tier, results, status, 0.0, 0.0, {"status": "ok", "failure_kind": None, "reason": None},
                           bool(getattr(ctx.trusted, "is_root", False)), round(time.time() - t0, 3),
                           meta={"revealed": ctx.revealed, "kinds": {g.id: g.kind for g in self.gates}, "setup_failed": True})

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
                try:
                    r0 = self.setup(ctx)
                    if inspect.isawaitable(r0):
                        await r0
                except (CandidateError, shutil.Error) as ex:
                    if not _candidate_caused(ex, ctx):
                        raise
                    return self._candidate_setup_fail(ctx, tier, ex, t0)
            gate_state: dict[str, list[str]] = {g.id: [] for g in self.gates}
            gave_up = self._gave_up(ctx)
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
            definite_fail = [g.id for g in self.gates if g.kind == "gate" and status.get(g.id) == "fail"]
            if infra_checks and not definite_fail:
                infra = {"status": "masked", "failure_kind": infra_checks[0].detail.get("failure_kind") or "check_infra",
                         "reason": f"{infra_checks[0].id}: {infra_checks[0].detail.get('infra', '')}"[:500]}
            elif infra_checks:
                # a hard gate failed for certain: the conjunction is 0 whatever went unmeasured (M7 audit V3: one infra
                # check turned a certain 0 into a mask, and the candidate can trigger some infra outcomes)
                infra["unmeasured_checks"] = [r.id for r in infra_checks][:20]
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
        gave_up = self._gave_up(ctx) if infra["status"] == "ok" else False
        meta: dict = {"revealed": ctx.revealed, "kinds": {g.id: g.kind for g in self.gates}}
        if gave_up:
            outcome = float(self.abstain.feasibility == "infeasible")
            reward = self.abstain.credit() if all_hard else 0.0
            meta["abstained"] = {"feasibility": self.abstain.feasibility, "credit": reward}
        else:
            base = all(status.get(g) == "pass" for g in outs + scores) if outs or scores else all_hard
            infeasible = bool(self.abstain and self.abstain.feasibility == "infeasible")  # any claimed solution is wrong
            # a score check may say whether its score is a success (``detail["success"]``): an imported environment's own
            # reward of 0 is measured (the gate passes) but is not task success (M11 audit G5: outcome was 1.0 for reward 0)
            success = all(r.detail.get("success", True) for r in results if r.gate in scores and r.status == "pass")
            outcome = 0.0 if infeasible else float(base and success)
            term = 1.0
            if scores:  # record-style: reward grows with the normalized score instead of clearing a bar
                per_gate = []  # averaged within each gate, then across gates (M7 audit V15: two checks in one gate paid 2.0)
                for g in scores:
                    vs = [float(r.detail.get("normalized_reward", 0.0)) for r in results if r.gate == g and r.status == "pass"]
                    per_gate.append(sum(vs) / max(1, len([r for r in results if r.gate == g])) if vs else 0.0)
                term = sum(per_gate) / len(scores)
                meta["score"] = {r.id: {k: r.detail.get(k) for k in ("score", "normalized", "new_record")} for r in results if r.gate in scores}
            reward = term if infra["status"] == "ok" and all_hard and (not outs or all(status.get(g) == "pass" for g in outs)) and base and not infeasible else 0.0
            if not scores:
                reward = 1.0 if reward and outcome == 1.0 else 0.0
        if tier == "fast" and self.fast_unmeasured_hard:
            meta["fast_unmeasured_hard_gates"] = self.fast_unmeasured_hard
        trusted = bool(getattr(ctx.trusted, "is_root", False))
        return GradeReport(self.task, tier, results, status, reward, outcome if infra["status"] == "ok" else 0.0, infra, trusted,
                           round(time.time() - t0, 3), meta=meta)


def _candidate_caused(ex: Exception, ctx: Context) -> bool:
    """A setup error the candidate caused: a ``CandidateError``, or a copy error whose every failing source is inside the
    candidate's tree (a FIFO, a socket, an unreadable file the candidate made)."""
    if isinstance(ex, CandidateError):
        return True
    if not isinstance(ex, shutil.Error) or ctx.candidate is None:
        return False
    rows = ex.args[0] if ex.args and isinstance(ex.args[0], list) else []
    root = str(Path(ctx.candidate).resolve())
    return bool(rows) and all(isinstance(r, tuple) and str(Path(str(r[0])).resolve()).startswith(root) for r in rows)


def _gate_status(states: list[str]) -> str:
    if not states:
        return "pass"
    if "fail" in states:  # a definite fail decides a gate; an unmeasured sibling cannot rescue it
        return "fail"
    if "infra" in states:
        return "infra"
    if "skip" in states:
        return "skip"
    return "pass"


def _container_lost(e: BaseException | None) -> BaseException | None:
    """The exception in ``e``'s cause chain that a lost container raised (``crucible.world.exec.EnvContainerLost``), matched
    by its failure kind: this runtime ships into verifier images without the rest of Crucible, so it imports nothing else."""
    seen: set[int] = set()
    while e is not None and id(e) not in seen:
        if getattr(e, "failure_kind", None) == "crucible:env_container_lost":
            return e
        seen.add(id(e))
        e = e.__cause__ or e.__context__
    return None


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
        if (lost := _container_lost(ex)) is not None:  # unless the host confirmed the container is gone (M1 audit R-11)
            st, detail = "infra", {"infra": str(lost)[:2000], "failure_kind": lost.failure_kind}
        else:
            st, detail = "fail", {"error": f"{type(ex).__name__}: {ex}"[:2000], "traceback": traceback.format_exc()[-1500:]}
    return CheckResult(c.id, c.gate, st, detail, round(time.monotonic() - t0, 3), c.provenance, c.tier)


REWARD_FILES = ("reward.json", "reward.txt", "mask.json", "crucible_reward.json", "report.json")
# reward files other runtimes read before ours, in a place the agent can write (M7 audit, sweep G5): rLLM's
# ShellScriptEvaluator reads /tmp/rllm/reward.json first, so an agent-written file there beat the verifier
FOREIGN_REWARD_FILES = ("/tmp/rllm/reward.json",)


def flat_reward(rj: dict) -> dict | None:
    """The canonical ``reward.json`` (DESIGN §4.3; research m11-m12/05 §2): a flat dict of finite numbers — Harbor raises
    on anything else (``harbor/verifier/verifier.py:104-121``) — with ``reward`` = the plan's gated reward, ``outcome``,
    ``gate/<g>``, ``check/<c>`` and ``progress``. Timing is not in it (it lives in ``report.json``), so the file is
    byte-identical whichever runtime graded. **A masked grade has no ``reward.json`` at all** (``None``): a ``0.0``
    there would be trained as a real 0, and without a ``reward`` key verifiers would even sum the remaining keys. The
    mask goes to ``mask.json`` and ``crucible_reward.json``. Not every runtime honours a missing file (M7 audit, sweep
    D4/G5): stock Harbor's ``Mean`` counts the trial as 0 and does not retry it (ship
    ``interop/harbor/metrics.MASKED_MEAN_PY``), and rLLM's ``ShellScriptEvaluator`` reads it as 0. Only runtimes that
    read ``mask.json`` or ``crucible_reward.json`` mask."""
    if (rj.get("infra") or {}).get("status") == "masked":
        return None
    flat = {"reward": float(rj["reward"]), "outcome": float(rj.get("outcome", 0.0))}
    flat.update({f"gate/{g}": float(v) for g, v in sorted((rj.get("gates") or {}).items())})
    flat.update({k: float(v) for k, v in sorted((rj.get("components") or {}).items()) if k.startswith("check/")})
    if rj.get("progress") is not None:
        flat["progress"] = float(rj["progress"])
    return flat


def prewrite_zero(out_dir: Path) -> None:
    """Before grading starts: remove every reward file, ours and ``FOREIGN_REWARD_FILES``, so an interrupted grader
    leaves *no* reward (an infrastructure failure where the runtime masks; ``flat_reward`` names those that read it as
    0), never a stale, agent-written or placeholder-0 reward. (Name kept for callers; it no longer writes a zero.)"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in REWARD_FILES:
        (out_dir / name).unlink(missing_ok=True)
    for f in FOREIGN_REWARD_FILES:
        try:
            Path(f).unlink(missing_ok=True)
        except OSError:  # not ours to remove (another user's file): the runtime that reads it is not in use
            pass


def write_reward_files(out_dir: Path, rj: dict, full: dict | None = None, *, extra: dict | None = None) -> dict:
    """The single producer of reward files, used by the in-container grader and the host grader alike:
    ``crucible_reward.json`` (the full document; ``reward`` null when masked), ``reward.json`` + ``reward.txt`` (exact:
    ``repr`` round-trips the float) only when not masked, ``mask.json`` when masked, ``report.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masked = (rj.get("infra") or {}).get("status") == "masked"
    if not masked:
        flat = flat_reward(rj) or {}
        if any(not math.isfinite(v) for v in flat.values()):  # strict JSON and Harbor refuse NaN (M7 audit V18)
            bad = sorted(k for k, v in flat.items() if not math.isfinite(v))
            rj = {**rj, "infra": {"status": "masked", "failure_kind": "verifier_error", "reason": f"non-finite reward values: {bad}"}}
            masked = True
    doc = {**rj, **(extra or {}), "reward": None if masked else float(rj["reward"])}
    (out_dir / "crucible_reward.json").write_text(json.dumps(doc, indent=1, default=str, sort_keys=True) + "\n")
    if full is not None:
        (out_dir / "report.json").write_text(json.dumps(full, indent=1, default=str) + "\n")
    if masked:
        for name in ("reward.json", "reward.txt"):
            (out_dir / name).unlink(missing_ok=True)
        inf = rj.get("infra") or {}
        (out_dir / "mask.json").write_text(json.dumps({"mask_sample": True, "failure_kind": inf.get("failure_kind"), "reason": inf.get("reason")},
                                                      sort_keys=True) + "\n")
    else:
        (out_dir / "mask.json").unlink(missing_ok=True)
        (out_dir / "reward.json").write_text(json.dumps(flat_reward(rj), indent=1, sort_keys=True) + "\n")
        (out_dir / "reward.txt").write_text(repr(float(rj["reward"])) + "\n")
    return doc


def write_outputs(report: GradeReport | None, out_dir: Path, *, error: str | None = None) -> dict:
    """In-container grader outputs (``write_reward_files``). With no report (the grader crashed before producing one) the
    grade is masked as ``grader_exception``: no reward file, never a 0."""
    if report is None:
        rj = {"reward": 0.0, "gates": {}, "outcome": 0.0, "infra": {"status": "masked", "failure_kind": "grader_exception", "reason": (error or "")[:500]}}
        full = {"reward": None, "error": error}
    else:
        rj = report.reward_json()
        full = report.to_dict()
    write_reward_files(out_dir, rj, full)
    return rj
