"""Failure mining: audited student traces -> failure findings -> builder-intent requests (docs/research/swarm-pipeline/08).

The loop the factory runs on its student: run the student on candidate tasks, read its traces for where and why it
fails, and have the builder make tasks that target those failures. This module is the middle step. It adds no new trace
schema: a finding is the trace audit's own pieces (``traceaudit.EpisodeBundle``, ``Fact``, ``FirstWrongStep``,
``signature``; ``audit.AuditLabel`` when an auditor labelled the trace), arranged per failing episode.

- **Finding** (``FailureFinding``, ``finding``): one failing, unmasked episode. The task and window, the policy and its
  family, the milestones missed at the end, the milestones lost after being reached, the first wrong step
  (``traceaudit.first_wrong_step``: kind, step, turn, files), the cause from the closed taxonomy (an auditor label when one
  is given, else the facts' deterministic cause, else unlabelled), a **verbatim quote** from the rendered trace at the
  first wrong step (checked to be a substring of ``render_bundle``), the facts, the cluster ``signature``, a ``mode``
  (below) and the run's ``numerics`` (``factory.student``: an MLX run is ``dev``).
- **Mode** (``MODES``): what the student did wrong, from facts and the first wrong step only (no judge): ``no_edit``,
  ``claimed_done_without_progress`` (reply-before-action), ``lost_progress`` (Phi fell and never recovered),
  ``stalled`` (Phi stopped rising), ``never_progressed`` (edits, no milestone), ``harness_misuse``, ``world_fault``,
  ``budget``, ``env_flake``. The last four are not the student's skill: they are **routed** to their owner (harness,
  World, limits, infrastructure), never turned into a task intent (00 §7: hard for the wrong reason).
- **Builder-intent request** (``BuilderIntent``, ``builder_intents``): findings of one skill mode clustered (>=
  ``min_traces`` traces); it carries the behaviour a new task should test, the failure mode in the student's words (the
  quotes), the evidence (finding ids), the milestones and files where it happened, and the numerics. It is an
  ``IntentCard`` candidate (``intent_card``: ``status="candidate"``; 00 §8.3: a human approves every intent). Consumers:
  ``planner_intent()`` is the swarm generator's ``intent`` (``SwarmRunner(intent=...)``), ``cell(approved_by=...)`` is a
  root-curator ``Cell`` (the approved card plus the findings as observations).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.audit import AuditLabel
from crucible.factory.traceaudit import EpisodeBundle, Fact, FirstWrongStep, deterministic_cause, facts, first_wrong_step, render_bundle, signature
from crucible.factory.versioning import sha

# mode -> (the behaviour a task targeting it tests, the difficulty axes it varies); None: routed, never an intent
MODES: dict[str, tuple[str, tuple[str, ...]] | None] = {
    "lost_progress": ("Keeps earlier work working while extending it", ("dependent milestones", "shared code touched by later steps")),
    "stalled": ("Carries a multi-step change through to its last dependent step", ("milestone depth", "requires edges")),
    "never_progressed": ("Finds where a requested change belongs in an unfamiliar codebase and makes its first step pass",
                         ("files to locate", "indirection between the request and the code")),
    "no_edit": ("Moves from reading the code to a first concrete, tested edit", ("exploration needed before the first edit",)),
    "claimed_done_without_progress": ("Checks its own work against the request before it reports the work done",
                                      ("hidden checks beyond the obvious one", "a plausible but wrong first fix")),
    "harness_misuse": None,
    "world_fault": None,
    "budget": None,
    "env_flake": None,
}
ROUTE_OWNER = {"harness_misuse": "harness", "world_fault": "world", "budget": "limits", "env_flake": "infra"}
QUOTE_CHARS = 240


@dataclass
class FailureFinding:
    id: str
    trace: str
    task: str
    window: str
    policy: str
    family: str
    mode: str
    cause: str | None
    cause_source: str  # auditor | facts | unlabelled
    quote: str
    milestones_missed: list[str] = field(default_factory=list)
    milestones_lost: list[str] = field(default_factory=list)
    first_wrong_step: dict[str, Any] = field(default_factory=dict)
    facts: list[dict[str, Any]] = field(default_factory=list)
    signature: str = ""
    numerics: str = "final"
    phi_final: float | None = None
    stop: str | None = None

    @property
    def routed(self) -> str | None:
        """The owner a non-skill failure goes to (None: a skill failure, intent material)."""
        if self.cause not in (None, "skill_gap"):
            return {"world_bug": "world", "spec_defect": "spec", "unstated_format": "prompt", "verifier_defect": "verifier",
                    "budget": "limits", "env_flake": "infra"}.get(self.cause, "auditor")
        return ROUTE_OWNER.get(self.mode)

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "routed": self.routed}


def mode_of(b: EpisodeBundle, fs: Sequence[Fact], fws: FirstWrongStep) -> str:
    kinds = {f.kind for f in fs}
    if b.masked or "masked" in kinds:
        return "env_flake"
    if "world_fault" in kinds:
        return "world_fault"
    if "harness_misuse" in kinds:
        return "harness_misuse"
    if "reply_before_action" in kinds:
        return "claimed_done_without_progress"
    if "no_edit" in kinds:
        return "no_edit" if "limit_stop" not in kinds else "budget"
    if fws.kind == "lost":
        return "lost_progress"
    if fws.kind == "plateau":
        return "stalled" if "limit_stop" not in kinds else "budget"
    if "limit_stop" in kinds and fws.kind in ("flat", "divergence", "unmeasured"):
        return "budget"
    return "never_progressed"


def quote_at(b: EpisodeBundle, fws: FirstWrongStep, *, rendered: str | None = None) -> str:
    """A verbatim piece of the rendered trace at the first wrong step: the tool call there (or its result), else the
    final message, else the first tool call. Always a substring of ``render_bundle(b)``."""
    doc = rendered if rendered is not None else render_bundle(b)
    cands: list[str] = []
    steps = {st.i: st for st in b.steps}
    for i in [fws.step, *(fws.tool_calls or [])]:
        st = steps.get(i) if i else None
        if st is not None:
            cands.append(f"[tool {st.i}] {st.function} {json.dumps(st.arguments, default=str)}"[:QUOTE_CHARS])
            if st.result.strip():
                cands.append(st.result.strip()[:QUOTE_CHARS])
    if b.final_message.strip():
        cands.append(b.final_message.strip()[:QUOTE_CHARS])
    if b.steps:
        st = b.steps[0]
        cands.append(f"[tool {st.i}] {st.function} {json.dumps(st.arguments, default=str)}"[:QUOTE_CHARS])
    for c in cands:
        c = c.strip()
        if c and c in doc:
            return c
    return ""


def finding(b: EpisodeBundle, *, passing: Sequence[EpisodeBundle] = (), oracle_files: Sequence[str] = (), label: AuditLabel | None = None,
            numerics: str = "final") -> FailureFinding | None:
    """The finding of one episode; None when it passed (passes go to ``traceaudit.pass_audit``)."""
    if b.passed:
        return None
    fs = facts(b)
    fws = first_wrong_step(b, passing, oracle_files=oracle_files)
    det = deterministic_cause(fs)
    if label is not None and label.cause and label.status in ("labelled", "deterministic"):
        cause, src = label.cause, "auditor"
    elif det:
        cause, src = det, "facts"
    else:
        cause, src = None, "unlabelled"
    mode = mode_of(b, fs, fws)
    rendered = render_bundle(b)
    quote = (label.evidence.strip() if label is not None and label.evidence and label.evidence.strip() in rendered else "") or quote_at(b, fws, rendered=rendered)
    missed = sorted(m for m, ok in b.milestones.items() if not ok)
    return FailureFinding(id="ff-" + sha({"trace": b.id, "task": b.task})[:12], trace=b.id, task=b.task, window=b.window, policy=b.policy,
                          family=b.family, mode=mode, cause=cause, cause_source=src, quote=quote, milestones_missed=missed,
                          milestones_lost=list(fws.lost), first_wrong_step=asdict(fws), facts=[asdict(f) for f in fs],
                          signature=signature(b, fs, fws, cause), numerics=numerics, phi_final=b.phi_final, stop=b.stop)


def mine(bundles: Sequence[EpisodeBundle], *, labels: Mapping[str, AuditLabel] | None = None, oracle_files: Mapping[str, Sequence[str]] | None = None,
         numerics: Mapping[str, str] | str = "final") -> list[FailureFinding]:
    """Findings of every failing, unmasked-or-masked episode (masked ones become routed ``env_flake`` findings). Passing
    episodes of the same task are each failure's divergence reference. ``numerics``: one value, or per bundle id."""
    from crucible.factory.student import numerics_of

    by_task: dict[str, list[EpisodeBundle]] = {}
    for b in bundles:
        by_task.setdefault(b.task, []).append(b)
    out = []
    for b in bundles:
        passing = [p for p in by_task.get(b.task, []) if p.passed and p.id != b.id]
        num = str(numerics.get(b.id, "final")) if isinstance(numerics, Mapping) else numerics
        num = "dev" if num == "dev" else numerics_of(b.policy)  # the policy id's backend stamp (``...@mlx``) also says dev
        f = finding(b, passing=passing, oracle_files=(oracle_files or {}).get(b.window, ()), label=(labels or {}).get(b.id), numerics=num)
        if f is not None:
            out.append(f)
    return out


# ---- builder-intent requests ---------------------------------------------------------------------------------------------------
@dataclass
class BuilderIntent:
    id: str
    mode: str
    behaviour: str
    failure_mode: str
    difficulty_axes: list[str]
    evidence: list[str]  # finding ids
    traces: list[str]
    tasks: list[str]
    quotes: list[str]
    milestones: list[str]
    files: list[str]
    families: list[str]
    numerics: str  # dev when any finding came from a dev-only backend
    n: int
    status: str = "awaiting_human"

    def intent_card(self, *, surface: str = "python-repo") -> dict[str, Any]:
        """The ``cards.IntentCard`` candidate (validated): a human approves it before it leases a cell (00 §8.3)."""
        from crucible.factory.cards import IntentCard

        card = IntentCard(id=self.id, behaviour=self.behaviour, failure_mode=self.failure_mode, evidence=list(self.evidence),
                          difficulty_axes=list(self.difficulty_axes), coverage={"behaviour": self.behaviour, "surface": surface}, status="candidate")
        return card.model_dump(mode="json")

    def planner_intent(self) -> dict[str, Any]:
        """The swarm planner's ``intent`` (``SwarmRunner(intent=...)``): the behaviour, how the student failed it, and the
        quoted evidence, so the plan's milestones exercise exactly that."""
        return {"id": self.id, "behaviour": self.behaviour, "failure_mode": self.failure_mode, "difficulty_axes": self.difficulty_axes,
                "observed": [{"quote": q} for q in self.quotes[:5]], "numerics": self.numerics,
                "ask": (f"Plan milestones whose dependent steps exercise this: {self.behaviour}. The student failed it this way: "
                        f"{self.failure_mode}. Make the difficulty real work in the repository (dependent edits across files), never a "
                        "vaguer request or a stricter check.")}

    def cell(self, *, approved_by: str, surface: str = "python-repo", quota: int = 1) -> Any:
        """A root-curator ``Cell`` once a human approved the card; the findings ride as the cell's observations."""
        from crucible.factory.cards import IntentCard
        from crucible.factory.root import Cell

        card = IntentCard.model_validate({**self.intent_card(surface=surface), "status": "approved", "approved_by": approved_by})
        obs = [{"kind": "student_failure", "finding": fid, "trace": t, "quote": q} for fid, t, q in zip(self.evidence, self.traces, self.quotes, strict=False)]
        return Cell(behaviour=self.behaviour, surface=surface, intent=card.model_dump(mode="json"), quota=quota, observations=obs)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def builder_intents(findings: Sequence[FailureFinding], *, min_traces: int = 2, prefix: str = "bi") -> tuple[list[BuilderIntent], list[dict[str, Any]]]:
    """(builder-intent requests, routed findings). A skill mode with >= ``min_traces`` findings becomes one request;
    a routed finding (``FailureFinding.routed``) goes to its owner instead, never into an intent."""
    routed = [{"finding": f.id, "trace": f.trace, "owner": f.routed, "mode": f.mode, "cause": f.cause, "quote": f.quote} for f in findings if f.routed]
    by_mode: dict[str, list[FailureFinding]] = {}
    for f in findings:
        if not f.routed and MODES.get(f.mode):
            by_mode.setdefault(f.mode, []).append(f)
    out = []
    for mode, fs in sorted(by_mode.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(fs) < min_traces:
            continue
        behaviour, axes = MODES[mode]  # type: ignore[misc]
        quotes = [f.quote for f in fs if f.quote]
        where = sorted({m for f in fs for m in (f.milestones_lost or f.milestones_missed)})
        files = sorted({x for f in fs for x in (f.first_wrong_step.get("files") or [])})
        steps = [f.first_wrong_step.get("position") for f in fs if f.first_wrong_step.get("position") is not None]
        at = f" (first wrong step at {min(steps):.0%}-{max(steps):.0%} of the episode)" if steps else ""
        fm = (f"{len(fs)} failing episode(s) over {len({f.task for f in fs})} task(s): {mode.replace('_', ' ')}{at}; "
              f"milestones missed or lost: {', '.join(where[:8]) or 'none recorded'}")
        out.append(BuilderIntent(id=f"{prefix}-{mode.replace('_', '-')}-{sha([f.id for f in fs])[:8]}", mode=mode, behaviour=behaviour, failure_mode=fm,
                                 difficulty_axes=list(axes), evidence=[f.id for f in fs], traces=[f.trace for f in fs], tasks=sorted({f.task for f in fs}),
                                 quotes=quotes[:10], milestones=where, files=files[:20], families=sorted({f.family for f in fs if f.family}),
                                 numerics="dev" if any(f.numerics == "dev" for f in fs) else "final", n=len(fs)))
    return out, routed


def write_jsonl(path: Path | str, rows: Sequence[Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r.to_json() if hasattr(r, "to_json") else r, default=str) + "\n")
    return p
