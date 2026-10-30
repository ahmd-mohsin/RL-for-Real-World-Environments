"""ONL: online certification (part2-paper-core §5, claim C6; M5). Every passing eval or training rollout is a trial on its
task's certificate; a flagged pass is a suspect; a confirmed one is an archived counterexample and quarantines the task.

**Inputs.** ``RolloutObs`` is one graded rollout, read from what the runners already write:
- ``from_episode_row``: an eval arm's ``episodes.jsonl`` row (``factory/tinker.py``: ``task``, ``rollout``, ``reward``,
  ``masked``, ``passed``, ``gates`` with the lane's ``path:*`` verdicts, ``milestones``);
- ``from_reward_row``: a training run's ``rewards.jsonl`` row (``train/bridges/tinker/trainer.py``: ``task``,
  ``episode``, ``reward``, ``step``, ``verifier_version``); a pass is reward 1;
- ``from_rollout_record``: a ``RolloutRecord`` (``rollout.json``: ``reward``, ``mask_sample``, ``reward_components``
  with ``hack/confirmed`` and ``hack_suspect``) with its World trace (``trace.json`` events) and the collected patch.
A masked rollout is never a trial; a failed one is not a trial of the reward's soundness either (it earned nothing).

**Monitors** (injected; a monitor maps a rollout to ``Flag``s and never clears one; a chain of thought may raise a flag,
never lower it):
- ``l1_rules``: the Level-1 hack rules of ``credit.hack_detector.rule_flags`` over the trace (test and verifier writes,
  exit spoofing, skipped tests, grader-tool shadowing, intercepted calls);
- ``gate_flags``: the lane's own measured path gates that failed (``path:protected``, ``path:plumbing``, ...);
- ``component_flags``: ``hack_suspect`` / ``hack/confirmed`` already on the record (the L2 judge's verdict).

**Confirmation** (injected ``Confirmer``: True confirmed, False cleared, None undecided and left a suspect):
- ``judged``: a pass the L2 judge confirmed (``hack/confirmed``) is confirmed; nothing else is decided;
- ``ReplayConfirmer``: executed, no model. The rollout's end state is replayed through the task's contract (the
  certificate's grader); then again with the flagged actions removed (``program_of`` builds both). The contract
  rejecting the replay (the deployed grader passed what the certified contract fails), or the replay passing only with
  the flagged actions, confirms the hack; a pass that holds without them clears the flag.

**Outputs** (``OnlineCertifier``): ``results(task_id)`` is the ``ClassResult`` ``ONL`` for the task's certificate (one
finding per passing rollout: clean, suspect or confirmed); ``with_online(cert, result)`` puts it in a certificate and
rejects the certificate on a confirmed counterexample; each confirmed one is a ``Counterexample`` (``cls="ONL"``, its
probe the replay program, ``expect="fail"``) added to the archive, and a ``QuarantineRecord`` for the task is appended to
``quarantine.jsonl`` (once per counterexample). The caller withdraws the task and sends it to repair.

``OnlineMonitor`` (M2's stub API: ``observe(task, run, verdict, flags)``) is kept for in-process callers.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import content_hash
from crucible.synth.contract import Verdict
from crucible.synth.falsify.agents import trace_program
from crucible.synth.falsify.base import ClassResult, Counterexample, FalsifyContext, Finding, Outcome, Probe
from crucible.synth.program import Lit, Program, Run, Step
from crucible.synth.task import Task

PASS_EPS = 1e-9
# a flag's category -> the defect a confirmed pass under it shows (D8: an affordance let the pass happen without the skill)
DEFECT_OF = {"test_tampering": "D8", "grader_tampering": "D8", "path": "D8", "answer_exposure": "D7", "network": "D7", "history": "D7", "judge": "D8"}


class Flag(CrucibleModel):
    rule: str
    category: str = "other"
    evidence: str = ""
    monitor: str = ""
    seq: int | None = None  # the trace event the flag points at


class RolloutObs(CrucibleModel):
    task_id: str
    rollout_id: str
    passed: bool | None = None
    reward: float | None = None
    masked: bool = False
    source: str = "eval"  # eval | train
    step: int | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)  # World trace events ({type, seq, data: {function, arguments, writes}})
    patch: str | None = None  # the collected end state as a diff (code tasks)
    answer: Any = None
    flags: list[Flag] = Field(default_factory=list)  # flags already on the row
    meta: dict[str, Any] = Field(default_factory=dict)


def _passed(reward: Any) -> bool | None:
    return None if reward is None else float(reward) >= 1.0 - PASS_EPS


def from_episode_row(row: Mapping[str, Any]) -> RolloutObs:
    """An eval arm's ``episodes.jsonl`` row (module doc)."""
    gates = row.get("gates") or {}
    flags = [Flag(rule=str(g), category="path", evidence=str(v), monitor="lane_gates") for g, v in sorted(gates.items()) if str(g).startswith("path:") and v == "fail"]
    return RolloutObs(task_id=str(row["task"]), rollout_id=str(row.get("rollout") or row.get("attempt")), passed=bool(row.get("passed")) if "passed" in row else _passed(row.get("reward")),
                      reward=row.get("reward"), masked=bool(row.get("masked")), source="eval", flags=flags,
                      meta={k: row.get(k) for k in ("attempt", "milestones", "stop", "final_phi") if k in row})


def from_reward_row(row: Mapping[str, Any]) -> RolloutObs:
    """A training run's ``rewards.jsonl`` row (module doc)."""
    return RolloutObs(task_id=str(row["task"]), rollout_id=str(row["episode"]), passed=_passed(row.get("reward")), reward=row.get("reward"),
                      masked=row.get("reward") is None, source="train", step=row.get("step"), meta={"verifier_version": row.get("verifier_version")})


def from_rollout_record(rec: Mapping[str, Any], trace: Mapping[str, Any] | None = None, *, patch: str | None = None) -> RolloutObs:
    """A ``RolloutRecord`` (``rollout.json``) with its World trace (``trace.json``) and the collected patch."""
    comps = rec.get("reward_components") or {}
    masked = bool(rec.get("mask_sample")) or rec.get("reward") is None
    flags = []
    if comps.get("hack/confirmed"):
        flags.append(Flag(rule="hack/confirmed", category="judge", monitor="reward_components"))
    if comps.get("hack_suspect") or (rec.get("progress") or {}).get("hack_suspect"):
        flags.append(Flag(rule="hack_suspect", category="path", monitor="reward_components"))
    return RolloutObs(task_id=str(rec.get("task")), rollout_id=str(rec.get("rollout_id")), passed=None if masked else _passed(rec.get("reward")),
                      reward=rec.get("reward"), masked=masked, trace=list((trace or {}).get("events") or []), patch=patch, flags=flags,
                      meta={"hack_confirmed": bool(comps.get("hack/confirmed"))})


# ---------------------------------------------------------------------------------------------------------- monitors

Monitor = Callable[[RolloutObs], list[Flag]]


def l1_rules(obs: RolloutObs) -> list[Flag]:
    """``credit.hack_detector.rule_flags`` over the rollout's trace events."""
    from crucible.credit.hack_detector import rule_flags

    events = [SimpleNamespace(type=e.get("type"), seq=e.get("seq", i), data=e.get("data") or {}) for i, e in enumerate(obs.trace)]
    return [Flag(rule=str(f["rule"]), category=str(f.get("category") or "other"), evidence=str(f.get("evidence") or ""), monitor="l1_rules", seq=f.get("seq"))
            for f in rule_flags(SimpleNamespace(events=events))]


def gate_flags(obs: RolloutObs) -> list[Flag]:
    return [f for f in obs.flags if f.monitor == "lane_gates"]


def component_flags(obs: RolloutObs) -> list[Flag]:
    return [f for f in obs.flags if f.monitor == "reward_components"]


DEFAULT_MONITORS: tuple[Monitor, ...] = (l1_rules, gate_flags, component_flags)


# ---------------------------------------------------------------------------------------------------------- confirmation

Confirmer = Callable[[Task, RolloutObs, Sequence[Flag]], Awaitable[tuple[bool | None, dict[str, Any]]]]


async def judged(task: Task, obs: RolloutObs, flags: Sequence[Flag]) -> tuple[bool | None, dict[str, Any]]:
    """The L2 judge's verdict when the record carries it; otherwise undecided."""
    return (True, {"by": "judge"}) if obs.meta.get("hack_confirmed") or any(f.rule == "hack/confirmed" for f in flags) else (None, {})


ProgramOf = Callable[[RolloutObs, Sequence[Flag]], tuple[Program, Program | None] | None]


def flagged_paths(flags: Sequence[Flag]) -> set[str]:
    out = set()
    for f in flags:
        ev = f.evidence.strip()
        if ev and "\n" not in ev and " " not in ev and ("/" in ev or "." in ev):
            out.add(ev.split("file:", 1)[-1])
    return out


def drop_files(patch: str, paths: set[str]) -> str:
    """A git diff without the file diffs that touch ``paths``."""
    chunks: list[list[str]] = []
    for ln in patch.splitlines(keepends=True):
        if ln.startswith("diff --git ") or not chunks:
            chunks.append([])
        chunks[-1].append(ln)
    keep = []
    for c in chunks:
        head = "".join(c[:6])
        if any(f" a/{p}" in head or f" b/{p}" in head or f"--- a/{p}" in head or f"+++ b/{p}" in head for p in paths):
            continue
        keep.append("".join(c))
    return "".join(keep)


def patch_programs(obs: RolloutObs, flags: Sequence[Flag]) -> tuple[Program, Program | None] | None:
    """The replay of a code rollout: its collected patch, and the patch without the files the flags name."""
    if not obs.patch:
        return None
    full = Program(steps=[Step(id="p0", tool="repo.apply_patch", args={"patch": Lit(v=obs.patch)})])
    paths = flagged_paths(flags)
    if not paths:
        return full, None
    rest = drop_files(obs.patch, paths)
    minus = Program(steps=[Step(id="p0", tool="repo.apply_patch", args={"patch": Lit(v=rest)})]) if rest.strip() else Program()
    return full, minus


def trace_programs(obs: RolloutObs, flags: Sequence[Flag]) -> tuple[Program, Program | None] | None:
    """The replay of a tool-call rollout: its calls as literal steps, and the calls without the flagged events."""
    calls = [e for e in obs.trace if e.get("type") == "tool_call"]
    if not calls:
        return None
    bad = {int(f.seq) for f in flags if f.seq is not None}

    def prog(skip: set[int]) -> Program:
        steps = [Step(id=f"o{i}", tool=str((e.get("data") or {}).get("function")), args={k: Lit(v=v) for k, v in sorted(((e.get("data") or {}).get("arguments") or {}).items())})
                 for i, e in enumerate(calls) if int(e.get("seq", i)) not in skip]
        return Program(steps=steps, answer=None if obs.answer is None else Lit(v=obs.answer))

    return prog(set()), (prog(bad) if bad else None)


class ReplayConfirmer:
    """Executed confirmation (module doc). ``ctx_of(task)`` gives the task's ``FalsifyContext`` (its World and exec
    registry); ``program_of`` turns a rollout into its replay and its replay without the flagged actions."""

    def __init__(self, ctx_of: Callable[[Task], FalsifyContext], program_of: ProgramOf = patch_programs):
        self.ctx_of, self.program_of = ctx_of, program_of

    async def __call__(self, task: Task, obs: RolloutObs, flags: Sequence[Flag]) -> tuple[bool | None, dict[str, Any]]:
        progs = self.program_of(obs, flags)
        if progs is None:
            return None, {"why": "no replayable end state"}
        full, minus = progs
        ctx = self.ctx_of(task)
        r = await ctx.run(task, full)
        v = ctx.grade(r, task.contract)
        if any(x.startswith("GRADER_ERROR") for x in v.reasons):
            return None, {"why": "replay unmeasured", "reasons": v.reasons}
        if not v.passed:
            return True, {"by": "replay", "why": "contract_rejects", "reasons": v.reasons, "program": full.model_dump(mode="json")}
        if minus is None:
            return None, {"why": "no flagged action to remove", "reasons": v.reasons}
        r2 = await ctx.run(task, minus)
        v2 = ctx.grade(r2, task.contract)
        if any(x.startswith("GRADER_ERROR") for x in v2.reasons):
            return None, {"why": "replay unmeasured", "reasons": v2.reasons}
        if not v2.passed:
            return True, {"by": "replay", "why": "pass_needs_flagged_actions", "reasons": v2.reasons, "program": full.model_dump(mode="json")}
        return False, {"by": "replay", "why": "pass_holds_without_flagged_actions"}


# ---------------------------------------------------------------------------------------------------------- the certifier

class QuarantineRecord(CrucibleModel):
    task_id: str
    task_version: int
    domain: str
    at: str
    reason: str
    counterexamples: list[str] = Field(default_factory=list)
    rollouts: list[str] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    source: str = "eval"
    step: int | None = None


class OnlineReport(CrucibleModel):
    seen: int = 0
    masked: int = 0
    failed: int = 0
    trials: int = 0
    flagged: int = 0
    confirmed: int = 0
    cleared: int = 0
    suspects: int = 0
    unknown_task: int = 0
    quarantined: list[str] = Field(default_factory=list)
    counterexamples: list[str] = Field(default_factory=list)


def defect_of(flags: Sequence[Flag]) -> str:
    for f in flags:
        if f.rule.startswith(("path:network", "path:history")):
            return "D7"
    return next((DEFECT_OF[f.category] for f in flags if f.category in DEFECT_OF), "D8")


class OnlineCertifier:
    """Consumes rollouts for known tasks (module doc). ``task_of(task_id)`` resolves a task (None: not certified here)."""

    def __init__(self, task_of: Callable[[str], Task | None], *, monitors: Sequence[Monitor] = DEFAULT_MONITORS, confirm: Confirmer | None = None,
                 archive: Any = None, quarantine_path: str | Path | None = None):
        self.task_of, self.monitors, self.confirm = task_of, list(monitors), confirm
        self.archive = archive
        self.quarantine_path = Path(quarantine_path) if quarantine_path is not None else None
        self.findings: dict[str, list[Finding]] = {}
        self.quarantines: list[QuarantineRecord] = []

    def flags_of(self, obs: RolloutObs) -> list[Flag]:
        out: list[Flag] = []
        seen: set[tuple[str, str]] = set()
        for m in self.monitors:
            for f in m(obs):
                k = (f.rule, f.evidence)
                if k not in seen:
                    seen.add(k)
                    out.append(f)
        return out

    async def consume(self, rows: Iterable[RolloutObs]) -> OnlineReport:
        rep = OnlineReport()
        for obs in rows:
            rep.seen += 1
            if obs.masked or obs.passed is None:
                rep.masked += 1
                continue
            if not obs.passed:
                rep.failed += 1
                continue
            task = self.task_of(obs.task_id)
            if task is None:
                rep.unknown_task += 1
                continue
            rep.trials += 1
            flags = self.flags_of(obs)
            label, defect, data = "clean", None, {"flags": [f.model_dump() for f in flags], "source": obs.source, "step": obs.step}
            program: Program | None = None
            if flags:
                rep.flagged += 1
                decided, ev = (None, {})
                if self.confirm is not None:
                    res = self.confirm(task, obs, flags)
                    decided, ev = await res if inspect.isawaitable(res) else res
                data["confirmation"] = {k: v for k, v in ev.items() if k != "program"}
                if isinstance(ev.get("program"), dict):
                    program = Program.model_validate(ev["program"])
                if decided is True:
                    label, defect = "confirmed", defect_of(flags)
                    rep.confirmed += 1
                elif decided is False:
                    rep.cleared += 1
                else:
                    label = "suspect"
                    rep.suspects += 1
            site = f"rollout:{obs.rollout_id}"
            p = Probe(cls="ONL", op="rollout", site=site, program=program, expect="fail", data={"task": obs.task_id})
            f = Finding(cls="ONL", op="rollout", site=site, label=label, defect=defect, probe=p,  # type: ignore[arg-type]
                        outcome=Outcome(passed=True, data=data))
            self.findings.setdefault(obs.task_id, []).append(f)
            if label == "confirmed":
                cx = Counterexample.of(task, f)
                rep.counterexamples.append(cx.id)
                if self.archive is not None:
                    self.archive.add(cx)
                q = QuarantineRecord(task_id=task.id, task_version=task.version, domain=task.domain, at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                     reason=f"ONL confirmed {defect}: {data.get('confirmation', {}).get('why', 'confirmed')}", counterexamples=[cx.id],
                                     rollouts=[obs.rollout_id], flags=flags, source=obs.source, step=obs.step)
                if self._quarantine(q):
                    rep.quarantined.append(task.id)
        return rep

    def _quarantine(self, q: QuarantineRecord) -> bool:
        key = content_hash({"task": q.task_id, "v": q.task_version, "cx": q.counterexamples}, 24)
        if any(content_hash({"task": x.task_id, "v": x.task_version, "cx": x.counterexamples}, 24) == key for x in self.quarantines):
            return False
        self.quarantines.append(q)
        if self.quarantine_path is not None:
            have = set()
            if self.quarantine_path.exists():
                for ln in self.quarantine_path.read_text().splitlines():
                    try:
                        d = json.loads(ln)
                        have.add(content_hash({"task": d["task_id"], "v": d["task_version"], "cx": d["counterexamples"]}, 24))
                    except (ValueError, KeyError):
                        continue
            if key not in have:
                self.quarantine_path.parent.mkdir(parents=True, exist_ok=True)
                with self.quarantine_path.open("a") as fh:
                    fh.write(json.dumps(q.model_dump(mode="json"), sort_keys=True) + "\n")
        return True

    def results(self, task_id: str) -> ClassResult:
        return ClassResult(cls="ONL", findings=list(self.findings.get(task_id, [])))


def with_online(cert: Any, result: ClassResult) -> Any:
    """``cert`` with the ONL class stat; a confirmed online counterexample rejects it (the task is quarantined)."""
    from crucible.synth.certificate import class_stat

    classes = {**cert.classes, "ONL": class_stat(result)}
    upd: dict[str, Any] = {"classes": classes}
    if result.c > 0:
        upd.update(status="rejected", admitted=False, reasons=[*cert.reasons, f"ONL:confirmed={result.c}"])
    elif result.suspects and not cert.policy.allow_open_suspects:
        upd.update(status="rejected", admitted=False, reasons=[*cert.reasons, f"ONL:suspects={len(result.suspects)}"])
    return cert.model_copy(update=upd)


# ---------------------------------------------------------------------------------------------------------- M2's API

class OnlineMonitor:
    """In-process form (M2): a graded synth ``Run`` and its flags; a flagged pass is a suspect handed to ``quarantine``."""

    name = "ONL"

    def __init__(self, quarantine: Callable[[Task, Finding], None] | None = None):
        self.quarantine = quarantine
        self.trials: dict[str, int] = {}

    def observe(self, task: Task, run: Run, verdict: Verdict, flags: Sequence[str] = ()) -> Finding | None:
        if not verdict.passed:
            return None
        self.trials[task.id] = self.trials.get(task.id, 0) + 1
        if not flags:
            return None
        p = Probe(cls="ONL", op="rollout", site=f"rollout:{self.trials[task.id]}", program=trace_program(run.calls, run.answer), expect="fail",
                  data={"flags": list(flags)})
        f = Finding(cls="ONL", op="rollout", site=p.site, label="suspect", defect=None, probe=p,
                    outcome=Outcome(passed=True, reasons=verdict.reasons, data={"flags": list(flags)}))
        if self.quarantine is not None:
            self.quarantine(task, f)
        return f


__all__ = ["DEFAULT_MONITORS", "Flag", "OnlineCertifier", "OnlineMonitor", "OnlineReport", "QuarantineRecord", "ReplayConfirmer", "RolloutObs",
           "component_flags", "defect_of", "drop_files", "from_episode_row", "from_reward_row", "from_rollout_record", "gate_flags", "judged", "l1_rules",
           "patch_programs", "trace_programs", "with_online"]
