"""Floors (05 §6.1, R6): solutions that do no work must score 0. Also the evidence primitives every verifier-validation
stage of M17 shares (``Graded``, ``grader``, ``Frac``, ``requirement_map``).

The five floors, each a *finite* 0 or the task goes back to the verifier writer:

| floor | World task (tool-call list) | answer / state task |
|---|---|---|
| ``no_op`` | ``task.noop()`` (``[]``) | ``task.noop()`` |
| ``empty_action`` | declare done with nothing; one write tool called with no arguments | ``empty`` given by the caller |
| ``constant_answer`` | declare done with a constant answer (``0``, ``yes``, ``no``, ``Done.``, ``N/A``) | ``constant_of(c)`` |
| ``original_state`` | the start state *is* the no-op's end state: recorded as the same grade | ``original_state`` given |
| ``copy_of_reference`` | declare done with the prompt copied back | ``reference_copy`` given (skip only when copying *is* the task) |

The authored floors of A2 (``task.negatives`` of kind ``floor``, the naive solution) are graded with them, so this stage
reuses the accept contract's floor set and adds what it lacks. The zero rule is A2's: a NaN, an out-of-range value or a
masked grade is not a 0 (a masked floor is *unmeasured*, and unmeasured never passes).
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

FLOOR_KINDS = ("no_op", "empty_action", "constant_answer", "original_state", "copy_of_reference")
CONSTANTS = ("0", "yes", "no", "Done.", "N/A")


# --- shared evidence primitives -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Graded:
    """One verdict of a verifier on one end state. ``reward`` None = masked (no verdict: never a pass, never a fail).
    ``checks``: check id -> passed (a skipped check counts as not passed, as in ``verify.runtime.plan``)."""

    reward: float | None
    checks: Mapping[str, bool] = field(default_factory=dict)
    fingerprint: str | None = None  # the end state's fingerprint (same fingerprint = same end state)
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def masked(self) -> bool:
        return self.reward is None

    @property
    def finite(self) -> bool:
        return isinstance(self.reward, (int, float)) and not isinstance(self.reward, bool) and math.isfinite(self.reward)

    def passed(self, at: float = 1.0) -> bool | None:
        return None if not self.finite else self.reward >= at  # type: ignore[operator]

    def failing(self) -> list[str]:
        return sorted(c for c, ok in self.checks.items() if not ok)

    def verdict_key(self) -> tuple:
        """What "the same verdict" means: reward and per-check outcome."""
        return (self.reward if self.finite else repr(self.reward), tuple(sorted(self.checks.items())))


def graded_from(x: Any) -> Graded:
    """A ``Graded`` from what graders return: ``Graded``; ``(GradeReport, fingerprint)`` (``AcceptTask.grade``); a
    ``GradeReport``; ``{"reward", "checks", "fingerprint"}``; a bare reward; ``None`` (masked)."""
    if isinstance(x, Graded):
        return x
    if x is None:
        return Graded(None)
    fp = None
    if isinstance(x, tuple) and len(x) == 2:
        x, fp = x
    if hasattr(x, "gates") and hasattr(x, "checks") and hasattr(x, "reward"):  # verify.runtime.plan.GradeReport
        masked = (getattr(x, "infra", None) or {}).get("status") == "masked"
        checks = {c.id: c.status == "pass" for c in x.checks}
        failing = sorted({str(t) for c in x.checks if c.status != "pass" for t in ((getattr(c, "detail", None) or {}).get("failing") or [])})
        return Graded(None if masked else x.reward, checks, fp, {"gates": dict(x.gates), **({"failing_tests": failing[:30]} if failing else {})})
    if isinstance(x, Mapping):
        return Graded(x.get("reward"), dict(x.get("checks") or {}), x.get("fingerprint", fp), dict(x.get("detail") or {}))
    if isinstance(x, bool):
        return Graded(1.0 if x else 0.0, {}, fp)
    if isinstance(x, (int, float)):
        return Graded(float(x), {}, fp)
    raise TypeError(f"cannot read a verdict from {type(x).__name__}")


GradeFn = Callable[[Any], Awaitable[Any]]


async def call(fn: Callable[..., Any], *a: Any, **kw: Any) -> Any:
    """Call a sync or async function."""
    out = fn(*a, **kw)
    return await out if inspect.isawaitable(out) else out


def grader(task: Any) -> Callable[[Any], Awaitable[Graded]]:
    """The verifier of an accept-contract task (``task.grade(solution) -> (GradeReport, fingerprint)``) as a ``Graded``
    function."""

    async def g(solution: Any) -> Graded:
        return graded_from(await task.grade(solution))

    return g


@dataclass(frozen=True)
class Frac:
    """A count with its denominator, and a Wilson interval (``factory.labels.wilson``)."""

    k: int
    n: int

    @property
    def rate(self) -> float | None:
        return self.k / self.n if self.n else None

    @property
    def interval(self) -> tuple[float, float]:
        from crucible.factory.labels import wilson

        return wilson(self.k, self.n)

    def pair(self) -> tuple[int, int]:
        return (self.k, self.n)

    def to_dict(self) -> dict:
        lo, hi = self.interval
        return {"k": self.k, "n": self.n, "rate": None if self.rate is None else round(self.rate, 4), "wilson": [round(lo, 4), round(hi, 4)]}


class JsonClient(Protocol):
    """The injectable model client of the LLM roles (``crucible.actors.llm.LLMClient.complete_json`` has this shape)."""

    async def complete_json(self, messages: list[dict[str, Any]], **kw: Any) -> dict[str, Any]: ...


def requirement_map(ledger: Any) -> dict[str, set[str]]:
    """check id -> the requirement ids it enforces (``RequirementLedger.checks``)."""
    return {c.id: set(c.requirement_ids) for c in getattr(ledger, "checks", [])}


def requirements_failed(g: Graded, ledger: Any) -> set[str]:
    """Requirements with at least one failing check in ``g``."""
    m = requirement_map(ledger)
    return {r for c in g.failing() for r in m.get(c, ())}


# --- floors ---------------------------------------------------------------------------------------------------------


@dataclass
class FloorSolution:
    kind: str  # one of FLOOR_KINDS, or "authored" (the accept contract's own floor negatives)
    id: str
    solution: Any
    note: str = ""
    same_as: str | None = None  # this floor is the same submission as another (original state of a tool-call task)
    not_applicable: str | None = None  # why this floor does not apply (copying is the task); never graded


@dataclass
class FloorConfig:
    constants: tuple[str, ...] = CONSTANTS
    exact_zero: bool = True  # A2: a floor's reward must be a finite 0.0 (not merely below the pass mark)


def is_actions(x: Any) -> bool:
    from crucible.accept.mutation.actions import is_actions as _is

    return _is(x) or x == []


def _first_write(oracle: Sequence[dict]) -> str | None:
    from crucible.forge.hacker import _is_read

    return next((a["function_name"] for a in oracle if a.get("function_name") and not _is_read(a)), None)


def floor_solutions(*, oracle: Any = None, prompt: str = "", noop: Any = (), authored: Sequence[tuple[str, Any]] = (),
                    empty: Any = None, constant_of: Callable[[str], Any] | None = None, original_state: Any = None,
                    reference_copy: Any = None, copy_is_task: bool = False, config: FloorConfig | None = None) -> list[FloorSolution]:
    """The floor set of one task. For a tool-call task (``oracle`` a list of actions) every floor is generated; for other
    kinds the caller gives ``empty``, ``constant_of``, ``original_state`` and ``reference_copy`` (a missing one is reported
    by ``run_floors`` as unmeasured, never skipped silently)."""
    cfg = config or FloorConfig()
    noop = list(noop) if isinstance(noop, tuple) else noop
    out = [FloorSolution("no_op", "no_op", noop)]
    out += [FloorSolution("authored", f"authored/{i}", s) for i, s in authored]
    actions = isinstance(oracle, list) and is_actions(oracle)
    if empty is not None:
        out.append(FloorSolution("empty_action", "empty_action", empty))
    elif actions:
        out.append(FloorSolution("empty_action", "empty_action/done", [{"kind": "done", "reason": ""}], "declare done, nothing else"))
        w = _first_write(oracle)
        if w:
            out.append(FloorSolution("empty_action", "empty_action/call", [{"kind": "tool_call", "tool_call_id": "floor0", "function_name": w, "arguments": {}}],
                                     f"{w} with no arguments"))
    for c in cfg.constants:
        if constant_of is not None:
            out.append(FloorSolution("constant_answer", f"constant_answer/{c}", constant_of(c)))
        elif actions:
            out.append(FloorSolution("constant_answer", f"constant_answer/{c}", [{"kind": "done", "reason": c, "answer": c}]))
    if original_state is not None:
        out.append(FloorSolution("original_state", "original_state", original_state))
    elif actions or noop == []:
        out.append(FloorSolution("original_state", "original_state", noop, "a tool-call task's start state is the no-op's end state", same_as="no_op"))
    if copy_is_task:
        out.append(FloorSolution("copy_of_reference", "copy_of_reference", None, not_applicable="copying the reference is the task"))
    else:
        if reference_copy is not None:
            out.append(FloorSolution("copy_of_reference", "copy_of_reference", reference_copy))
        elif actions and prompt:
            out.append(FloorSolution("copy_of_reference", "copy_of_reference/prompt", [{"kind": "done", "reason": "copied", "answer": prompt}],
                                     "the prompt copied back as the answer"))
    return out


def floors_for_task(task: Any, *, prompt: str | None = None, config: FloorConfig | None = None, **kw: Any) -> list[FloorSolution]:
    """Floors of an accept-contract task: ``task.noop()``, its authored floor negatives (A2's naive solution) and the
    generated ones."""
    authored = [(n.id, task.solution_of(n)) for n in getattr(task, "negatives", []) if getattr(n, "kind", "") == "floor"]
    return floor_solutions(oracle=getattr(task, "oracle", None), prompt=prompt if prompt is not None else getattr(task, "instruction", ""),
                           noop=task.noop(), authored=authored, config=config, **kw)


@dataclass
class FloorReport:
    rows: dict[str, dict]  # floor id -> {kind, reward, ok, ...}
    missing: list[str]  # floor kinds with no solution (unmeasured)

    @property
    def above_zero(self) -> list[str]:
        return [i for i, r in self.rows.items() if r["status"] == "above_zero"]

    @property
    def unmeasured(self) -> list[str]:
        return [i for i, r in self.rows.items() if r["status"] == "masked"] + [f"missing:{k}" for k in self.missing]

    @property
    def ok(self) -> bool:
        return not self.above_zero and not self.unmeasured

    def card(self) -> dict[str, float]:
        """``VerifierCard.floors``: the worst (highest) reward per floor kind; NaN where unmeasured."""
        out: dict[str, float] = {}
        for r in self.rows.values():
            if r["status"] == "not_applicable":
                continue
            v = r["reward"] if isinstance(r["reward"], (int, float)) and r["status"] != "masked" else float("nan")
            k = r["kind"]
            out[k] = v if k not in out or (math.isnan(v) or v > out[k]) else out[k]
        for k in self.missing:
            out[k] = float("nan")
        return out

    def reasons(self) -> list[str]:
        out = [f"floor {i} scores {self.rows[i]['reward']} (must be 0)" for i in self.above_zero]
        out += [f"floor {i} unmeasured" for i in self.unmeasured]
        return out


async def run_floors(floors: Sequence[FloorSolution], grade: Callable[[Any], Awaitable[Any]], *, config: FloorConfig | None = None,
                     require: Sequence[str] = FLOOR_KINDS) -> FloorReport:
    """Grade every floor. A floor passes only with a finite 0 (``exact_zero``) or a reward <= 0 otherwise."""
    cfg = config or FloorConfig()
    rows: dict[str, dict] = {}
    done: dict[str, Graded] = {}
    for f in floors:
        if f.not_applicable:
            rows[f.id] = {"kind": f.kind, "reward": None, "status": "not_applicable", "failing": [], "note": f.not_applicable}
            continue
        g = done[f.same_as] if f.same_as in done else graded_from(await call(grade, f.solution))
        done[f.id] = g
        if g.masked:
            st = "masked"
        elif not g.finite or (g.reward != 0.0 if cfg.exact_zero else g.reward > 0.0):  # type: ignore[operator]
            st = "above_zero"
        else:
            st = "zero"
        rows[f.id] = {"kind": f.kind, "reward": g.reward, "status": st, "failing": g.failing(), "note": f.note, **({"same_as": f.same_as} if f.same_as else {})}
    have = {f.kind for f in floors}
    return FloorReport(rows, [k for k in require if k not in have])
