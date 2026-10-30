"""The student-probe outcome classifier: which rollout outcomes the task did not intend (the RSI loop's probe side).

At task-creation time the student runs k = 4 → 8 probe rollouts on each new task (``learn.measure``); this module says
of each rollout whether its outcome is the one the task intends. Unintended outcomes drive the task's repair; intended
ones never do. Nothing here involves training.

| label | meaning |
|---|---|
| ``intended_pass`` | the derived values were derived (observed before they were written) and the intended evidence was read |
| ``intended_fail`` | a genuine skill gap: a skill-mode first wrong step on a task with executed evidence of fairness |
| ``unintended_pass`` | a shortcut, leak, vacuous check, frame hole, lax requirement or tool exploit let it pass |
| ``unintended_fail`` | hidden information, an over-strict check, an unsolvable or nondeterministic task, a grader error |
| ``uncertain`` | the executed evidence does not decide (and no panel decided): **never** treated as a task defect |

**The key rule.** A failure from a genuine skill gap must never be classed as a task defect, because the loop must never
ease a task toward the student's pass rate. So ``unintended_fail`` needs *task-level executed evidence* (the grader
erred; π* fails its own contract; REP shows nondeterminism; TAINT shows the failing pattern needs information the agent
cannot get; the only failing check is an unstated trace constraint; the answer equals π*'s under a loose comparison),
and anything else that is not a clean skill gap is ``uncertain``. Routed failures (the agent protocol broke, the call
budget ran out) are ``uncertain`` with their owner: they are neither the task's defect nor its skill.

**Evidence, executed first** (``TaskEvidence``, ``classify``):
- the grader's verdict and reason codes, re-checked by replaying the rollout's literal trace (a replay that disagrees is
  ``uncertain``: unreplayable);
- snapshot-replay attribution of the first wrong step (``frontier.attribute``);
- trace analysis against the oracle's taint map: every value the student wrote that a required pattern constrains must
  be sourced in the student's own earlier observations, the instruction, ``given`` or the tool catalogue
  (``falsify.taint.sourced``), and the collections π* reads as evidence should be read before the first write;
- the task's deterministic falsifier classes (REP, TAINT, NULL, RED, TWIN, EVID; MUT when asked): its confirmed
  defects are task-level evidence (D1/D2 lax, D5 hidden, D7 leak, D8 affordance, D9 insensitive, D11 nondeterministic);
- the certification's independent ALT solvers (another family, never a builder): their passes show the task solvable.
The calibrated cross-family **panel** (``falsify.agents.Panel``, λ from executed-truth anchors) decides the residue: an
attempt it judges valid that failed is ``unintended_fail`` (over-strict), one it judges invalid that passed is
``unintended_pass``; valid and passed is ``intended_pass``, invalid and failed ``intended_fail``; an abstention stays
``uncertain``.

``OutcomeRecord`` is the RSI agent's input (DESIGN §7.7 item 19; ``export_outcomes``): per rollout the label, the cause,
who decided, and the evidence. It carries reason codes and attribution, never a reward or pass rate.
``calibration_cases`` + ``confusion`` measure the classifier on cases with known labels (the planted defects).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import content_hash, norm_text
from crucible.synth.contract import answer_equal
from crucible.synth.falsify.base import BaseFalsifier, FalsifyContext, delta_of, run_classes
from crucible.synth.falsify.taint import contract_constants, sourced
from crucible.synth.learn import Rollout
from crucible.synth.novelty import observed_collections
from crucible.synth.program import Program, Run, _loose_eq, leaves, scalar_leaves, text_spans
from crucible.synth.task import Task

Label = Literal["intended_pass", "intended_fail", "unintended_pass", "unintended_fail", "uncertain"]
LABELS: tuple[Label, ...] = ("intended_pass", "intended_fail", "unintended_pass", "unintended_fail", "uncertain")
SKILL_KINDS = ("irreversible", "forbidden", "wrong_write", "omission", "answer", "no_action")
CRASH = re.compile(r"^(TypeError|KeyError|AttributeError|IndexError|ZeroDivisionError|NameError|RuntimeError|UndeclaredWrite|sqlite3\.\w+)\b")


class TaskEvidence(CrucibleModel):
    """Task-level executed evidence, computed once per task version."""

    task_id: str
    version: int
    oracle_passes: bool
    defects: dict[str, int] = Field(default_factory=dict)  # confirmed defect code -> count, over the classes run
    classes: dict[str, dict[str, Any]] = Field(default_factory=dict)  # class -> {applicable, n, c, defects}
    unsourced: list[Any] = Field(default_factory=list)  # TAINT D5 values (literals or contract constants nobody can source)
    leaked: list[Any] = Field(default_factory=list)  # TAINT D7: intent-card derive values the instruction states
    alt: list[int] | None = None  # [passes, n] of the independent ALT solvers
    oracle_entities: list[str] = Field(default_factory=list)
    oracle_write_tools: list[str] = Field(default_factory=list)
    oracle_read_collections: list[str] = Field(default_factory=list)


def default_classes(mut: bool = False) -> list[BaseFalsifier]:
    from crucible.synth.falsify import MUT, NULL, RED, REP, TAINT, TWIN
    from crucible.synth.intent import EVID

    out: list[BaseFalsifier] = [REP(), TAINT(), NULL(), RED(), TWIN(), EVID()]
    return [*out, MUT()] if mut else out


async def task_evidence(task: Task, ctx: FalsifyContext, *, alt: Sequence[int] | None = None, classes: Sequence[BaseFalsifier] | None = None,
                        mut: bool = False) -> TaskEvidence:
    o = await ctx.oracle(task)
    ok = ctx.grade(o, task.contract).passed
    res = await run_classes(task, list(classes) if classes is not None else default_classes(mut), ctx)
    defects: Counter[str] = Counter()
    cls: dict[str, dict[str, Any]] = {}
    unsourced: list[Any] = []
    leaked: list[Any] = []
    for r in res:
        ds = Counter(f.defect or "P3" for f in r.confirmed)
        cls[r.cls] = {"applicable": r.applicable, "n": r.n, "c": r.c, "defects": dict(ds)}
        defects.update(ds)
        if r.cls == "TAINT":
            for f in r.confirmed:
                v = f.probe.data.get("value")
                (leaked if f.defect == "D7" else unsourced).append(v)
    if not ok:
        defects["D10"] += 1
    reads: set[str] = set()
    for c in o.calls:
        if c.read_only and c.obs.value is not None:
            reads |= observed_collections(c.obs.value, o.start)
    ents = sorted({f"{a}.{c}.{k}" for a, c, k in {ch.addr.entity for ch in delta_of(o, task.contract)}})
    return TaskEvidence(task_id=task.id, version=task.version, oracle_passes=ok, defects=dict(defects), classes=cls, unsourced=unsourced, leaked=leaked,
                        alt=list(alt) if alt else None, oracle_entities=ents,
                        oracle_write_tools=sorted({c.tool for c in o.calls if not c.read_only}), oracle_read_collections=sorted(reads))


class OutcomeRecord(CrucibleModel):
    id: str = ""
    task_id: str
    version: int
    domain: str = ""
    student: str = ""
    checkpoint: str = "base"
    rollout: int
    passed: bool | None
    reasons: list[str] = Field(default_factory=list)
    label: Label
    cause: str
    decided_by: Literal["execution", "panel", "none"] = "execution"
    evidence: dict[str, Any] = Field(default_factory=dict)


def _loose_answer(a: Any, b: Any) -> bool:
    """Equal after the normalisations a reader would accept: a one-item list is its item, numbers compare as numbers,
    text by case and whitespace, lists as sets."""
    def norm(x: Any) -> Any:
        if isinstance(x, list | tuple) and len(x) == 1:
            x = x[0]
        if isinstance(x, list | tuple):
            return sorted(str(norm(y)) for y in x)
        if isinstance(x, str):
            s = x.strip().replace(",", "")
            try:
                return float(s)
            except ValueError:
                return norm_text(s)
        if isinstance(x, bool) or x is None:
            return x
        if isinstance(x, int | float):
            return float(x)
        return x

    return a is not None and b is not None and (answer_equal(a, b) or norm(a) == norm(b))


def _entity_rows(task: Task, oracle: Run, pattern_ids: Sequence[str]) -> list[dict[str, Any]]:
    """The start-state rows of the entities the given required patterns name (for relating a D5 value to a failure)."""
    out = []
    if oracle.start is None:
        return out
    for p in task.contract.required:
        if p.id not in pattern_ids:
            continue
        rows = oracle.start.collection(p.app, p.coll)
        key = p.key.strip("'\"")
        if key in rows:
            out.append(rows[key])
    return out


def _relates(v: Any, task: Task, oracle: Run, pattern_ids: Sequence[str]) -> bool:
    for p in task.contract.required:
        if p.id in pattern_ids and (_loose_eq(v, p.key.strip("'\"")) or any(_loose_eq(v, c) for _, owner, c in contract_constants(task.contract)
                                                                              if owner == p.id)):
            return True
    return any(any(_loose_eq(v, x) or (isinstance(x, str) and isinstance(v, str) and len(v) >= 3 and text_spans(v, x, min_len=3))
                   for x in row.values()) for row in _entity_rows(task, oracle, pattern_ids))


def _unsourced_writes(task: Task, r: Run, tools: Sequence[Any]) -> list[dict[str, Any]]:
    """Values the student wrote that a required pattern or answer check constrains and that nothing the student saw
    (its earlier observations, ι, ``given``, the catalogue) sources."""
    consts = [v for _, _, v in contract_constants(task.contract)]
    out = []
    for i, c in enumerate(r.calls):
        if c.read_only:
            continue
        seen = [x for prev in r.calls[:i] for _, x in leaves(prev.obs.value)]
        for loc, v in scalar_leaves(c.args, c.step):
            if not any(_loose_eq(v, k) for k in consts):
                continue
            if not sourced(v, task.instruction, task.intent.given, tools, seen):
                out.append({"call": i, "loc": loc, "value": v})
    return out


async def classify(task: Task, rollout: Rollout, ctx: FalsifyContext, tev: TaskEvidence, *, student: str = "", checkpoint: str = "base",
                   cell: str = "", panel: Any = None, budget_calls: int | None = None) -> OutcomeRecord:
    """One rollout's label (module docstring)."""
    from crucible.synth.falsify.agents import Attempt
    from crucible.synth.falsify.base import stated
    from crucible.synth.frontier import attribute

    base = {"task_id": task.id, "version": task.version, "domain": task.domain, "student": student, "checkpoint": checkpoint, "rollout": rollout.i,
            "passed": rollout.passed, "reasons": list(rollout.reasons)}
    ev: dict[str, Any] = {"task_defects": tev.defects, "alt": tev.alt, "oracle_passes": tev.oracle_passes}

    def rec(label: Label, cause: str, by: str = "execution") -> OutcomeRecord:
        r = OutcomeRecord(label=label, cause=cause, decided_by=by, evidence=ev, **base)  # type: ignore[arg-type]
        r.id = "oc-" + content_hash({"t": task.id, "v": task.version, "s": student, "c": checkpoint, "r": rollout.i}, 16)
        return r

    if rollout.passed is None:
        return rec("uncertain", "masked", "none")
    codes = {x.split(":", 1)[0] for x in rollout.reasons}
    if any(c.startswith("GRADER_ERROR") for c in codes):
        return rec("unintended_fail", "grader_error")
    if not tev.oracle_passes:
        return rec("unintended_fail" if not rollout.passed else "unintended_pass", "unsolvable: π* fails its own contract")
    if tev.defects.get("D11"):
        return rec("unintended_fail" if not rollout.passed else "unintended_pass", "nondeterministic: REP replays differ")
    prog = Program.model_validate(rollout.program) if rollout.program else Program(steps=[])
    r = await ctx.run(task, prog)
    v = ctx.grade(r.model_copy(update={"answer": rollout.answer}), task.contract)
    ev["replay_agrees"] = v.passed == rollout.passed
    if v.passed != rollout.passed:
        return rec("uncertain", "unreplayable: the replayed trace gets another verdict", "none")
    tools = await (await ctx.world(task)).tools()
    oracle = await ctx.oracle(task)
    writes = [c for c in r.calls if not c.read_only and c.obs.error is None]
    crashes = [{"call": i, "tool": c.tool, "error": (c.obs.error or "")[:160]} for i, c in enumerate(r.calls) if c.obs.error and CRASH.match(c.obs.error)]
    ev["world_crashes"] = crashes
    seen_cols: set[str] = set()
    first_write = next((i for i, c in enumerate(r.calls) if not c.read_only), len(r.calls))
    for c in r.calls[:first_write]:
        if c.read_only and c.obs.value is not None:
            seen_cols |= observed_collections(c.obs.value, r.start)
    skipped = sorted(set(tev.oracle_read_collections) - seen_cols)
    ev["evidence_skipped"] = skipped
    unsourced = _unsourced_writes(task, r, tools)
    ev["unsourced_writes"] = unsourced
    d = delta_of(r, task.contract)
    ents = {f"{a}.{c}.{k}" for a, c, k in {ch.addr.entity for ch in d}}
    o_ents = set(tev.oracle_entities)
    oins = {".".join(e.split(".")[:2]) for e in o_ents}
    outside = sorted(e for e in ents if e not in o_ents and not any(ch.kind == "insert" and f"{ch.addr.app}.{ch.addr.coll}" in oins
                                                                       and f"{ch.addr.app}.{ch.addr.coll}.{ch.addr.key}" == e for ch in d))
    missing_oracle = sorted(e for e in o_ents if e not in ents and ".".join(e.split(".")[:2]) not in {".".join(x.split(".")[:2]) for x in ents if x not in o_ents})
    ev["outside_entities"], ev["oracle_entities_missed"] = outside, missing_oracle
    other_tools = sorted({c.tool for c in writes} - set(tev.oracle_write_tools))
    ev["non_oracle_write_tools"] = other_tools

    async def residue(cause: str) -> OutcomeRecord:
        if panel is None:
            return rec("uncertain", cause, "none")
        verdict = await panel.judge(task, Attempt(trace=rollout.trace, answer=rollout.answer, reasons=list(rollout.reasons)), tag="outcome")
        ev["panel"] = {"decision": verdict.decision, "lam": verdict.lam, "votes": [x.model_dump(mode="json") for x in verdict.votes]}
        if verdict.decision == "abstain":
            return rec("uncertain", f"{cause}; panel abstained", "panel")
        valid = verdict.decision == "valid"
        if rollout.passed and valid and tev.leaked:
            # the panel judges the attempt against the instruction it was given; an instruction that gives the decision away
            # makes the shortcut a valid attempt (fixture D7: both members said valid at confidence ≥ 0.99), so a valid
            # verdict cannot clear a task-level leak
            return rec("uncertain", f"{cause}; panel: valid attempt, but the panel cannot judge a leak in the instruction", "panel")
        if rollout.passed:
            return rec("intended_pass" if valid else "unintended_pass", f"{cause}; panel: {'valid' if valid else 'invalid'} attempt", "panel")
        return rec("unintended_fail" if valid else "intended_fail", f"{cause}; panel: {'valid attempt the grader rejected' if valid else 'invalid attempt'}",
                   "panel")

    if rollout.passed:
        if not writes and tev.oracle_write_tools:
            return rec("unintended_pass", "vacuous: passed without a successful write while π* writes")
        if unsourced:
            return rec("unintended_pass", f"leak: wrote constrained value(s) {[u['value'] for u in unsourced][:3]} it never observed")
        if other_tools and not (set(tev.oracle_write_tools) & {c.tool for c in writes}):
            if tev.defects.get("D8"):
                return rec("unintended_pass", f"affordance: passed only through {other_tools} (D8 confirmed on the task)")
            return await residue(f"passed through write tools π* never uses: {other_tools}")
        if outside:
            if tev.defects.get("D2") or tev.defects.get("D9"):
                return rec("unintended_pass", f"frame hole: changed {outside[:3]} which π* never touches (D2/D9 confirmed)")
            return await residue(f"changed entities π* never touches: {outside[:3]}")
        if missing_oracle:
            if tev.defects.get("D1") or tev.defects.get("D4"):
                return rec("unintended_pass", f"lax requirement: skipped {missing_oracle[:3]} π* changes (D1/D4 confirmed)")
            return await residue(f"passed without π*'s changes to {missing_oracle[:3]}")
        if oracle.answer is not None and not _loose_answer(rollout.answer, oracle.answer):
            if tev.defects.get("D3"):
                return rec("unintended_pass", f"lax value: answer {rollout.answer!r} is not π*'s {oracle.answer!r} (D3 confirmed on the task)")
            return await residue(f"answer {rollout.answer!r} differs from π*'s {oracle.answer!r} yet passed")
        if skipped:
            # not decisive alone (pilot finding: π* reads the PO and receipts on a duplicate rejection, where they decide
            # nothing; TAINT's D7 also fires on "duplicates" as a verb): the residue, with the leak noted when TAINT saw one
            leak = f"; TAINT D7: the instruction states {tev.leaked[:3]}" if tev.leaked else ""
            return await residue(f"passed without reading {skipped}, which π* reads as evidence{leak}")
        return rec("intended_pass", "the derived values were observed before they were written, and the effects are π*'s")
    # ---- failures
    att = await attribute(task, rollout, ctx, cell=cell, student=student, checkpoint=checkpoint, budget_calls=budget_calls)
    ev["attribution"] = {"kind": att.kind, "mode": att.mode, "step": att.step, "patterns": att.patterns, "routed": att.routed,
                         "evidence_skipped": att.evidence_skipped}
    if att.routed:
        return rec("uncertain", f"routed:{att.routed} ({att.kind})", "none")
    if codes and codes <= {"TRACE"}:
        said = {m.id for m in stated(task.contract, task.instruction)}
        failed = [x.split(":", 1)[1] for x in rollout.reasons if x.startswith("TRACE:")]
        if failed and not any(f in said for f in failed):
            return rec("unintended_fail", f"over-strict: only unstated trace constraint(s) {failed} fail")
    if codes and codes <= {"ANSWER"} and _loose_answer(rollout.answer, oracle.answer):
        return rec("unintended_fail", f"over-strict: answer {rollout.answer!r} equals π*'s {oracle.answer!r} up to format")
    pats = att.patterns or [x.split(":", 1)[1] for x in rollout.reasons if x.startswith("MISSING_REQUIRED:")]
    failed_q = {x.split(":", 1)[1] for x in rollout.reasons if x.startswith("ANSWER:")}
    q_consts = [q.value for q in task.contract.answer if q.id in failed_q and q.value is not None]
    related = [u for u in tev.unsourced if _relates(u, task, oracle, pats) or any(_loose_eq(u, c) or u == c for c in q_consts)]
    if related:
        return rec("unintended_fail", f"hidden information: {related[:3]} (TAINT D5) is needed by failing pattern(s) {pats[:3]}")
    if tev.unsourced:
        return rec("uncertain", f"TAINT D5 on the task ({tev.unsourced[:3]}) but unrelated to the failing patterns", "none")
    if crashes:
        return await residue(f"the World crashed on {[c['tool'] for c in crashes][:3]}")
    if att.kind in SKILL_KINDS and "TAINT" in tev.classes and tev.classes["TAINT"]["applicable"]:
        return rec("intended_fail", f"skill_gap:{att.mode} ({att.kind} at call {att.step}); the task is solvable (π* passes) and communicable "
                                    f"(TAINT clean)" + (f"; independent ALT passed {tev.alt[0]}/{tev.alt[1]}" if tev.alt and tev.alt[0] else ""))
    return await residue(f"failure {att.kind} without decisive evidence")


def summary(records: Sequence[OutcomeRecord]) -> dict[str, Any]:
    return {"rollouts": len(records), "by_label": dict(Counter(r.label for r in records)), "by_decider": dict(Counter(r.decided_by for r in records)),
            "by_cause": dict(Counter(r.cause.split(":", 1)[0].split(" (")[0] for r in records)),
            "tasks_with_unintended": sorted({r.task_id for r in records if r.label.startswith("unintended")})}


def export_outcomes(records: Sequence[OutcomeRecord]) -> list[dict[str, Any]]:
    """The RSI agent's input (DESIGN §7.7 item 19): one record per rollout, evidence included, no reward."""
    return [r.model_dump(mode="json") for r in records]


def confusion(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """(expected, got) → the confusion matrix, accuracy on decided cases, coverage, and the count of skill gaps called
    task defects (must be 0)."""
    mat = {e: {g: 0 for g in LABELS} for e in LABELS}
    for e, g in pairs:
        mat[e][g] += 1
    decided = [(e, g) for e, g in pairs if g != "uncertain"]
    return {"n": len(pairs), "matrix": mat, "decided": len(decided), "coverage": round(len(decided) / len(pairs), 4) if pairs else None,
            "accuracy_decided": round(sum(e == g for e, g in decided) / len(decided), 4) if decided else None,
            "skill_gap_called_defect": sum(1 for e, g in pairs if e == "intended_fail" and g.startswith("unintended")),
            "defect_called_intended": sum(1 for e, g in pairs if e.startswith("unintended") and g.startswith("intended"))}
