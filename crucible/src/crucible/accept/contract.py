"""The Acceptance Contract, minimal subset for forged tasks (DESIGN §9, ARCHITECTURE M9; the M8 gate's prerequisite).

Implemented: **A1** oracle, **A2** floors, **A3** negatives and mutation score, **A4-lite** replay determinism,
**A5** leak, **A7** instruction <-> check coverage, **A8** grounding. Every check reports ``pass`` / ``fail`` /
``skip`` (a skip never counts as a pass); A4 (>= 40 draws), A6, A9-A13 report ``skip`` here and run in the full M9.
Everything is deterministic: no check is decided by an LLM (engineering rule 3).

A task is accepted only through an ``AcceptTask`` adapter (packs provide them: a World-graded toy task, a coding span
task graded on reconstructed file states). The report is bound to the task's content hash.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from crucible.forge.schema import CheckSpec, NegativeSpec, StatedProperty
from crucible.verify.runtime.plan import GradeReport
from crucible.world.leaks import leak_tokens

SKIPPED = {"A4": "full flake measurement (>= 40 draws) runs in M9", "A6": "isolation audit runs in M9 (M1 audits cover the world)",
           "A9": "rebuild fingerprints: M9", "A10": "frontier calibration: the forge Calibrator", "A11": "harness hygiene: M10",
           "A12": "tier agreement: M9", "A13": "grader attack suite: verify red-team (coding images)"}


class AcceptTask(Protocol):
    task_id: str
    task_hash: str
    instruction: str
    properties: list[StatedProperty]
    checks: list[CheckSpec]
    oracle: Any
    negatives: list[NegativeSpec]
    trap: str | None
    provenance: list[str]
    horizon: str

    async def grade(self, solution: Any, *, run: int = 0) -> tuple[GradeReport, str]:
        """Grade a solution from a fresh start; returns the report and the end-state fingerprint."""
        ...

    async def start_fingerprint(self) -> str: ...

    def agent_visible(self) -> dict[str, str]:
        """Everything the agent can read: instruction, documents, records, mail, workspace files."""
        ...

    def hidden_strings(self) -> list[str]:
        """Strings the agent must not be able to read: check literals, oracle-only values, canaries."""
        ...

    def solution_of(self, n: NegativeSpec) -> Any: ...

    def noop(self) -> Any: ...


@dataclass
class CheckOutcome:
    id: str
    status: str  # pass | fail | skip | error (a grade was masked: unmeasured, never a pass)
    detail: dict = field(default_factory=dict)


@dataclass
class AcceptReport:
    task_id: str
    task_hash: str
    checks: list[CheckOutcome]

    @property
    def ok(self) -> bool:
        return all(c.status == "pass" for c in self.checks if c.id not in SKIPPED)

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "task_hash": self.task_hash, "ok": self.ok, "checks": [asdict(c) for c in self.checks]}


async def a1_oracle(t: AcceptTask, *, grades: int = 3) -> CheckOutcome:
    rewards, fps, failing, reps = [], [], [], []
    for k in range(grades):
        rep, fp = await t.grade(t.oracle, run=k)
        reps.append(rep)
        rewards.append(rep.reward)
        fps.append(fp)
        failing.append(rep.failing_gates())
    start = await t.start_fingerprint()
    noop_rep, _ = await t.grade(t.noop(), run=100)
    outcome_checks = [c for c in t.checks if c.gate_kind in ("outcome", "score")]
    false_at_start = [c.gate for c in outcome_checks if noop_rep.gates.get(c.gate) != "pass"]
    reasons = []
    masked = [k for k, r in enumerate(reps) if (r.infra or {}).get("status", "ok") != "ok"]
    if masked or (noop_rep.infra or {}).get("status", "ok") != "ok":
        # an unmeasured grade proves nothing (M9 audit V2: a masked no-op passed the gold check)
        why = f"masked grades: oracle runs {masked}, no-op {(noop_rep.infra or {}).get('status', 'ok')}"
        return CheckOutcome("A1", "error", {"rewards": rewards, "reasons": [why]})
    if any(r != 1.0 for r in rewards):
        reasons.append(f"oracle rewards {rewards} (failing {failing})")
    if len(set(fps)) != 1:
        reasons.append("oracle replays end in different states")
    if fps and start == fps[0]:
        reasons.append("the oracle does not change the state (gold check)")
    if outcome_checks and not false_at_start:
        reasons.append("no outcome gate is false at the start state (gold check)")
    return CheckOutcome("A1", "fail" if reasons else "pass", {"rewards": rewards, "replay_identical": len(set(fps)) == 1, "reasons": reasons,
                                                            "outcome_gates_false_at_start": sorted(set(false_at_start))})


async def a2_floors(t: AcceptTask) -> CheckOutcome:
    reps = {"noop": (await t.grade(t.noop(), run=200))[0]}
    for n in (n for n in t.negatives if n.kind == "floor"):
        reps[n.id] = (await t.grade(t.solution_of(n), run=201))[0]
    rows = {k: r.reward for k, r in reps.items()}
    masked = sorted(k for k, r in reps.items() if (r.infra or {}).get("status", "ok") != "ok")
    if masked:  # a masked floor scored 0 is not a measured floor (M9 audit V2)
        return CheckOutcome("A2", "error", {"rewards": rows, "reasons": [f"masked floor grades: {masked}"]})
    bad = {k: v for k, v in rows.items() if v != 0.0}
    naive = [n.id for n in t.negatives if n.kind == "floor" and "naive" in n.id]
    reasons = ([f"floors score above 0: {bad}"] if bad else []) + ([] if naive else ["no naive floor"])
    return CheckOutcome("A2", "fail" if reasons else "pass", {"rewards": rows, "reasons": reasons})


async def a3_negatives(t: AcceptTask, *, mutation_min: float = 0.9) -> CheckOutcome:
    rows, reasons = {}, []
    gates_hit: set[str] = set()
    mutants = killed = 0
    gen = getattr(t, "generated_alternatives", None)
    for gid, sol in (gen() if callable(gen) else []):
        rep, _ = await t.grade(sol, run=301)
        ok = rep.reward == 1.0
        if not ok:
            reasons.append(f"generated alternative {gid} fails {rep.failing_gates()}: a check grades wording the instruction does not state")
        rows[gid] = {"kind": "valid_alternative", "generated": True, "expected": "pass", "reward": rep.reward, "failing": rep.failing_gates(), "ok": ok}
    for n in t.negatives:
        rep, _ = await t.grade(t.solution_of(n), run=300)
        failing = rep.failing_gates()
        if n.expected == "pass":
            ok = rep.reward == 1.0
            if not ok:
                reasons.append(f"valid alternative {n.id} fails {failing}: a check grades the oracle's internals, not the property")
        else:
            ok = rep.reward == 0.0 and rep.infra.get("status") == "ok" and (n.expected_gate is None or n.expected_gate in failing)
            if rep.reward == 0.0 and n.expected_gate:
                gates_hit.add(n.expected_gate)
            if not ok:
                reasons.append(f"negative {n.id} ({n.kind}) not killed at {n.expected_gate}: reward {rep.reward}, failing {failing}")
        if n.kind == "defect_mutant":
            mutants += 1
            killed += rep.reward == 0.0
        rows[n.id] = {"kind": n.kind, "expected_gate": n.expected_gate, "expected": n.expected, "reward": rep.reward, "failing": failing, "ok": ok}
    families = sorted({c.gate for c in t.checks})
    uncovered = [g for g in families if g not in gates_hit]
    if uncovered:
        reasons.append(f"check families with no negative failing them: {uncovered}")
    need = 4 if t.horizon in ("H3", "H4", "H5") else 1
    n_neg = sum(1 for n in t.negatives if n.expected == "fail")
    if n_neg < need:
        reasons.append(f"{n_neg} negatives, need >= {need} for {t.horizon}")
    if not any(n.kind == "valid_alternative" for n in t.negatives):
        reasons.append("no valid alternative (a different correct solution that must pass)")
    score = killed / mutants if mutants else None
    if mutants and score < mutation_min:
        reasons.append(f"mutation score {score:.2f} < {mutation_min}")
    return CheckOutcome("A3", "fail" if reasons else "pass", {"negatives": rows, "mutation_score": score, "families": families, "reasons": reasons})


async def a4_lite(t: AcceptTask, *, k: int = 3) -> CheckOutcome:
    rs = [await t.grade(t.oracle, run=400 + i) for i in range(k)]
    vals = {(r.reward, tuple(sorted(r.gates.items()))) for r, _ in rs}
    fps = {fp for _, fp in rs}
    ok = len(vals) == 1 and len(fps) == 1
    return CheckOutcome("A4-lite", "pass" if ok else "fail", {"replays": k, "distinct_verdicts": len(vals), "distinct_end_states": len(fps)})


def a5_leak(t: AcceptTask) -> CheckOutcome:
    visible = t.agent_visible()
    hits = []
    flat = {name: " ".join(text.split()) for name, text in visible.items()}
    for s in t.hidden_strings():
        if len(s) < 4:
            continue
        n = " ".join(s.split())
        for name, text in visible.items():
            if s in text or n in flat[name]:  # whitespace-normalized too: a re-wrapped paste is still the oracle
                hits.append({"string": s[:60], "in": name})
    tokens = leak_tokens(t.instruction)
    reasons = ([f"hidden material readable by the agent: {hits[:5]}"] if hits else []) + ([f"instruction leak tokens: {tokens}"] if tokens else [])
    return CheckOutcome("A5", "fail" if reasons else "pass", {"scanned": sorted(visible), "hits": hits[:20], "reasons": reasons})


def a7_coverage(t: AcceptTask) -> CheckOutcome:
    props = {p.id for p in t.properties}
    graded = {s for c in t.checks for s in c.states}
    unstated_checks = [c.id for c in t.checks if not c.states or not set(c.states) <= props]
    ungraded = sorted(props - graded)
    # the whole sentence, whitespace-normalized (M8 audit V8: the first 40 characters let a truncated property pass)
    flat = " ".join(t.instruction.split())
    missing_text = [p.id for p in t.properties if p.text.strip() and " ".join(p.text.split()) not in flat]
    reasons = []
    if unstated_checks:
        reasons.append(f"checks grading nothing the instruction states: {unstated_checks}")
    if ungraded:
        reasons.append(f"stated properties no check grades: {ungraded}")
    if missing_text:
        reasons.append(f"property sentences missing from the instruction: {missing_text}")
    return CheckOutcome("A7", "fail" if reasons else "pass", {"properties": sorted(props), "graded": sorted(graded), "reasons": reasons})


# A8's closed provenance grammar (M8 audit V11: any non-empty string, even "x", counted). Grounded: an expert signal
# (``sig_*``, resolved by M4), a policy or knowledge document (``kb:``), a recorded run or trace, a reference-differential
# check, or a registry defect (``defect:``, the M9 attack/audit ids). ``instruction:`` cites the stated property itself:
# not hidden knowledge (A7 covers it), so it is reported apart and left out of the share.
GROUNDED = re.compile(r"^(sig_\w+|kb:\S.*|reference_differential:\S.*|run:\S.*|trace:\S.*|defect:\S.*|M9 A1\d \S.*|A1\d audit:\s*\S.*)$")
STATED = "instruction:"


def a8_grounding(t: AcceptTask, resolve: Callable[[str], bool], *, min_share: float = 0.8) -> CheckOutcome:
    stated = [c for c in t.checks if (c.provenance or "").startswith(STATED)]
    hidden = [c for c in t.checks if c not in stated]
    grounded = [c for c in hidden if c.provenance and GROUNDED.match(c.provenance)]
    unknown = sorted({c.provenance for c in hidden if c.provenance and not GROUNDED.match(c.provenance)})
    share = len(grounded) / len(hidden) if hidden else 1.0
    dangling = sorted({c.provenance for c in grounded if c.provenance.startswith("sig_") and not resolve(c.provenance)})
    reasons = []
    if share < min_share:
        reasons.append(f"{share:.0%} of hidden checks are grounded (< {min_share:.0%})")
    if unknown:
        reasons.append(f"provenance that names no signal, document, run or reference: {unknown[:5]}")
    if dangling:
        reasons.append(f"provenance ids that do not resolve: {dangling}")
    with_prov = grounded
    if t.trap:
        sigs = [p for p in t.provenance if p.startswith("sig_")]
        if not sigs:
            reasons.append("a trap task must cite the signal it came from")
        elif not all(resolve(s) for s in sigs):
            reasons.append(f"trap signal ids do not resolve: {sigs}")
        trap_negs = [n for n in t.negatives if n.kind == "trap_reversion"]
        if not trap_negs:
            reasons.append("a trap task needs a trap-reversion negative")
        elif not any(n.provenance in sigs for n in trap_negs):
            reasons.append("no trap-reversion negative cites the trap's signal")
    return CheckOutcome("A8", "fail" if reasons else "pass", {"checks_with_provenance": f"{len(with_prov)}/{len(hidden)}", "share": round(share, 3),
                                                            "stated_checks": len(stated),
                                                            "trap_signals": [p for p in t.provenance if p.startswith("sig_")], "reasons": reasons})


async def accept(t: AcceptTask, *, resolve: Callable[[str], bool], include: Sequence[str] = ("A1", "A2", "A3", "A4-lite", "A5", "A7", "A8")) -> AcceptReport:
    out: list[CheckOutcome] = []
    for cid in include:
        if cid == "A1":
            out.append(await a1_oracle(t))
        elif cid == "A2":
            out.append(await a2_floors(t))
        elif cid == "A3":
            out.append(await a3_negatives(t))
        elif cid == "A4-lite":
            out.append(await a4_lite(t))
        elif cid == "A5":
            out.append(a5_leak(t))
        elif cid == "A7":
            out.append(a7_coverage(t))
        elif cid == "A8":
            out.append(a8_grounding(t, resolve))
    out += [CheckOutcome(k, "skip", {"reason": v}) for k, v in SKIPPED.items()]
    return AcceptReport(t.task_id, t.task_hash, out)


def summarize(reports: Sequence[AcceptReport]) -> dict[str, Any]:
    return {"accepted": sum(r.ok for r in reports), "total": len(reports),
            "failing": {r.task_id: [c.id for c in r.checks if c.status == "fail"] for r in reports if not r.ok}}
