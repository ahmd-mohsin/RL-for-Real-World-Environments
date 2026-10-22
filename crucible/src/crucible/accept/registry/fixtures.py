"""CPU power tests for per-task detectors (research m9-m10/01 §15.4 "power tests in CI"): each fixture runs the detector's
check on a clean task and on the same task with the historical defect re-injected, and returns both statuses. The
registry admits a detector only if it passes the clean one and fails the defective one.

``FakeTask`` is a programmable ``AcceptTask``: grades come from a function of the solution and the run, so a defect
(a nondeterministic reference, a weak negative, a perf bar without headroom, a no-op oracle) is one argument away.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.verify.runtime.plan import GradeReport


@dataclass
class Neg:
    id: str
    kind: str = "defect_mutant"
    expected_gate: str | None = "g"
    expected: str = "fail"
    solution: Any = None
    family: str | None = None
    provenance: str | None = None


@dataclass
class Prop:
    id: str
    text: str


@dataclass
class Chk:
    id: str
    gate: str
    states: list[str]
    provenance: str | None = "B.x"
    gate_kind: str = "gate"


def report(reward: float, gates: dict[str, str], *, infra: dict | None = None, elapsed: float = 0.1, meta: dict | None = None) -> GradeReport:
    """A fixture grade; its gates are hard rules unless ``meta["kinds"]`` says otherwise (M7 R6: a grade declares its kinds,
    and ``Chk``'s default ``gate_kind`` is "gate", so the A1 gold check sees the two agree)."""
    meta = dict(meta or {})
    meta["kinds"] = {g: "gate" for g in gates} | dict(meta.get("kinds") or {})
    return GradeReport(task="fake", tier="full", checks=[], gates=gates, reward=reward, outcome=reward, infra=infra or {"status": "ok"},
                       trusted=True, elapsed_s=elapsed, meta=meta)


@dataclass
class FakeTask:
    task_id: str = "fake/task@1"
    grade_fn: Callable[[Any, int], tuple[float, dict[str, str]]] = None  # (solution, run) -> (reward, gates)
    oracle: Any = "oracle"
    negatives: list[Neg] = field(default_factory=list)
    horizon: str = "H2"
    instruction: str = "Do the thing. The thing must be right."
    properties: list[Prop] = field(default_factory=lambda: [Prop("p1", "The thing must be right.")])
    checks: list[Chk] = field(default_factory=lambda: [Chk("c1", "g", ["p1"])])
    trap: str | None = None
    provenance: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)  # capability name -> value / callable

    def __post_init__(self):
        self.task_hash = hashlib.sha256(repr((self.task_id, sorted(self.extra), [(n.id, n.solution, n.kind) for n in self.negatives])).encode()).hexdigest()[:16]

    def grade_env(self) -> str:
        return f"{self.task_id}|{getattr(self.grade_fn, '__qualname__', '')}"

    def __getattr__(self, k: str):
        extra = self.__dict__.get("extra", {})
        if k in extra:
            v = extra[k]
            return v if callable(v) else (lambda: v)
        raise AttributeError(k)

    async def grade(self, solution: Any, *, run: int = 0):
        r, g = self.grade_fn(solution, run)
        return report(r, g), f"fp:{solution}:{r}"

    async def start_fingerprint(self) -> str:
        return "fp:start"

    def noop(self):
        return "noop"

    def solution_of(self, n: Neg):
        return n.solution or n.id

    def agent_visible(self) -> dict[str, str]:
        return {"instruction": self.instruction}

    def hidden_strings(self) -> list[str]:
        return list(self.extra.get("hidden", ["SECRET-ANSWER-1234"]))

    def key_inputs(self) -> dict:
        return {"overlay_hash": self.task_hash, "images": self.extra.get("images_ref", {})}


def killed_mutant() -> list:
    """One defect mutant every fixture grader scores 0 (an empty list would leave A3's mutation score unmeasured, which
    now fails A3: M9 audit V3)."""
    from crucible.accept.mutation.base import Mutant

    return [Mutant("drop_step/0", "drop_step", "mutant-drop-0", "omit the write")]


def good(sol: Any, run: int) -> tuple[float, dict[str, str]]:
    return (1.0, {"g": "pass"}) if sol in ("oracle", "alt") else (0.0, {"g": "fail"})


async def statuses(make: Callable[[bool], FakeTask], check: str, **kw: Any) -> dict[str, str]:
    from crucible.accept.run import accept_task

    out = {}
    for name, defective in (("clean", False), ("defective", True)):
        with tempfile.TemporaryDirectory() as d:
            rep = await accept_task(make(defective), only=[check], cache_root=Path(d), reg_version="power-test", **kw)
            out[name] = str(rep.results[check].status)
    return out


async def haar_reference_nondeterminism() -> dict[str, str]:
    """B.2: the reference's own output differs run to run (atomic compaction truncating at capacity 16384)."""
    import random

    def make(bad: bool) -> FakeTask:
        rng = random.Random(4)

        async def ref(i: int) -> dict:
            return {"haar_rects": 16384 if not bad else 16384 - rng.randint(1, 40), "ncv": "e3b0c4"}

        return FakeTask(grade_fn=good, extra={"reference_fingerprints": ref})

    return await statuses(make, "A4")


async def kinfu_bifurcation() -> dict[str, str]:
    """B.2: a fresh draw bifurcates about 5% of the time and fails a correct solution."""
    def make(bad: bool) -> FakeTask:
        async def fresh(sol, *, seed: int, gates=None):
            ok = not (bad and seed % 20 == 17)  # ~5% of draws
            return report(1.0 if ok else 0.0, {"g": "pass", "fresh": "pass" if ok else "fail"}), "fp"

        return FakeTask(grade_fn=good, extra={"fresh_gates": ["fresh"], "grade_fresh": fresh})

    return await statuses(make, "A4")


async def kinfu_headroom() -> dict[str, str]:
    """B.12: the v3 oracle's KinFu ratio 0.571 against a 0.65 bar: 12% headroom, below the required 15%."""
    def make(bad: bool) -> FakeTask:
        return FakeTask(grade_fn=good, extra={"perf_bars": {"kinfu": 0.65, "ncv": 0.66},
                                              "oracle_ratios": lambda rep: {"kinfu": 0.571 if bad else 0.50, "ncv": 0.40}})

    return await statuses(make, "A1")


async def fresh_cases_crash() -> dict[str, str]:
    """B.10: task03's fresh case generator draws unrunnable systems on ~18% of seeds and the grader crashes."""
    def make(bad: bool) -> FakeTask:
        async def fresh(sol, *, seed: int, gates=None):
            if bad and seed % 6 == 1:
                return report(0.0, {}, infra={"status": "masked", "failure_kind": "grader_exception", "reason": "box smaller than 2x cutoff"}), "fp"
            return report(1.0, {"g": "pass", "fresh": "pass"}), "fp"

        return FakeTask(grade_fn=good, extra={"fresh_gates": ["fresh"], "grade_fresh": fresh})

    return await statuses(make, "A1")


async def weak_negative() -> dict[str, str]:
    """B.4: the NCV out-of-bounds negative passed while padding was zero inside the allocation (the check never saw it)."""
    def make(bad: bool) -> FakeTask:
        def g(sol, run):
            if sol in ("oracle", "alt") or (bad and sol == "oob"):
                return 1.0, {"g": "pass"}
            return 0.0, {"g": "fail"}

        negs = [Neg("oob", family="bounds"), Neg("n2"), Neg("n3"), Neg("n4"), Neg("alt", kind="valid_alternative", expected="pass", expected_gate=None)]
        return FakeTask(grade_fn=g, negatives=negs, extra={"mutants": killed_mutant})

    return await statuses(make, "A3")


async def family_coverage() -> dict[str, str]:
    """B.13: the Adversary wrote no negative for one gate family (a gate nobody proved can fail)."""
    def make(bad: bool) -> FakeTask:
        def g(sol, run):
            if sol in ("oracle", "alt"):
                return 1.0, {"g": "pass", "policy": "pass"}
            return 0.0, {"g": "fail" if sol != "p" else "pass", "policy": "fail" if sol == "p" else "pass"}

        negs = [Neg("n1"), Neg("alt", kind="valid_alternative", expected="pass", expected_gate=None)] + ([] if bad else [Neg("p", expected_gate="policy")])
        checks = [Chk("c1", "g", ["p1"]), Chk("c2", "policy", ["p1"])]
        return FakeTask(grade_fn=g, negatives=negs, checks=checks, extra={"mutants": killed_mutant})

    return await statuses(make, "A3")


async def stale_image() -> dict[str, str]:
    """B.5: the verifier image tag resolves to a different image than the one the task was validated with."""
    pinned = "sha256:" + "a" * 64

    def make(bad: bool) -> FakeTask:
        return FakeTask(grade_fn=good, extra={"images_ref": {"verifier": "task04-verifier:v1"}, "pinned_images": {"verifier": pinned}})

    out = {}
    for name, bad in (("clean", False), ("defective", True)):
        out[name] = (await statuses(lambda b, bad=bad: make(bad), "A9", resolve_image=lambda ref, bad=bad: ("sha256:" + "b" * 64) if bad else pinned))["clean"]
    return out


async def sibling_overlap() -> dict[str, str]:
    """B.13: a span's start held the expert's port of a sibling fork (67-70% of the solution visible)."""
    return await statuses(lambda bad: FakeTask(grade_fn=good, extra={"sibling_overlap": 0.68 if bad else 0.05}), "A5")


async def oracle_noop() -> dict[str, str]:
    """HF survey D1: 450/500 oracles exit 0 without applying anything (heredoc indentation): the oracle does not change
    the state, and on a task where the start already passes, it 'passes'."""
    def make(bad: bool) -> FakeTask:
        t = FakeTask(grade_fn=(lambda s, r: (1.0, {"g": "pass"})) if bad else good)
        if bad:
            async def same(sol, *, run=0):
                return report(1.0, {"g": "pass"}), "fp:start"
            t.grade = same  # type: ignore[method-assign]
        return t

    return await statuses(make, "A1")


async def empty_ground_truth_floor() -> dict[str, str]:
    """HF survey D10: ~13% of rows have an empty ground truth, so a do-nothing policy scores 1 on them."""
    def make(bad: bool) -> FakeTask:
        return FakeTask(grade_fn=(lambda s, r: (1.0, {"g": "pass"}) if s in ("oracle", "noop") else (0.0, {"g": "fail"})) if bad else good,
                        negatives=[Neg("naive_floor", kind="floor")])

    return await statuses(make, "A2")


async def reference_alias() -> dict[str, str]:
    """HF survey D8: the reference implementation sits in the agent's workspace; aliasing it passes."""
    from crucible.accept.checks.extended import Attack

    def make(bad: bool) -> FakeTask:
        def g(sol, run):
            if sol in ("oracle",) or (bad and sol == "alias_reference"):
                return 1.0, {"g": "pass"}
            return 0.0, {"g": "fail"}

        async def took():
            return True

        return FakeTask(grade_fn=g, extra={"attacks": lambda: [Attack("alias_reference", "reference-alias", "alias_reference", "g", took)]})

    return await statuses(make, "A13")


async def harbor_taskconfig() -> dict[str, str]:
    """B.7: ``authors = ["..."]`` is not a valid Harbor TaskConfig; semantics-changing keys Harbor silently drops."""
    return await statuses(lambda bad: FakeTask(grade_fn=good, extra={"static_lint": ["task.toml: authors must be objects"] if bad else []}), "A9")
