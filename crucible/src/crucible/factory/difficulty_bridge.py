"""The controller side of the swarm's difficulty interface (swarm-pipeline/12 §3, 13 §5): ``factory.difficulty.control``
as a ``difficulty_hooks.Controller``.

- ``stats_from_targeting``: one S15 targeting row (``report.json`` / ``runs/s15_*``: passes, causes, modes, frontier
  results) plus its checkpoint-conditional row (``difficulty.conditional``) as the controller's inputs. A frontier result
  re-pathed under newer path rules (``factory.repath``: a stored attempt whose verdict flipped) overrides the stored one.
- ``choose_checkpoint``: the checkpoint start (C1) whose remaining milestones are expected in the band: the product of the
  measured ``p(m | prerequisites)`` over the milestones left, closest to 1/2 inside the band. Any unmeasured remaining
  milestone rules a level out (an unmeasured rate is never read as 0 or 1).
- ``ControlController.propose(task, stats)``: the controller's action (journaled whole, with every knob it ranked), then
  the first knob the reviser supports as one typed ``difficulty_hooks.KnobAction``. C1 counts as supported only when
  ``choose_checkpoint`` found a level; the chosen level rides in the action's ``params``.

Held-out tasks raise ``HeldOutRefused`` here (the controller's own check, V9).
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.credit.operators.branch_value import wilson
from crucible.factory.difficulty import Journal, checkpoint_levels, control
from crucible.factory.difficulty_hooks import KnobAction, from_control
from crucible.forge.learnability import BANDS, Band, PassRate


def _mid(m: Any) -> str:
    return str(m["id"] if isinstance(m, Mapping) else m.id)


def choose_checkpoint(conditional: Mapping[str, Any] | None, milestones: Sequence[Any], *, band: Band | None = None) -> dict[str, Any] | None:
    """``{level, pre, remaining, expected}`` of the checkpoint start expected closest to p = 1/2 inside ``band``, or None."""
    band = band or BANDS["train"]
    per = (conditional or {}).get("milestones") or {}
    best: dict[str, Any] | None = None
    for level, pre in enumerate(checkpoint_levels(milestones), 1):
        rest = [_mid(m) for m in milestones if _mid(m) not in pre]
        rates = [(per.get(m) or {}).get("p_given_prereqs") for m in rest]
        if not rest or any(r is None for r in rates):
            continue
        est = math.prod(float(r) for r in rates)
        if (band.lo is not None and est < band.lo) or (band.hi is not None and est > band.hi):
            continue
        if best is None or abs(est - 0.5) < abs(best["expected"] - 0.5):
            best = {"level": level, "pre": list(pre), "remaining": rest, "expected": round(est, 4)}
    return best


def k1_target(milestones: Sequence[Mapping[str, Any]], per_milestone: Mapping[str, float] | None) -> str | None:
    """K1's fault: the hardest (lowest measured rate) milestone outside every ``requires`` edge, leaving at least two."""
    if len(milestones) < 3:
        return None
    linked = {m["id"] for m in milestones if m.get("requires")} | {r for m in milestones for r in m.get("requires") or []}
    free = [m["id"] for m in milestones if m["id"] not in linked]
    if not free:
        return None
    per = per_milestone or {}
    return min(free, key=lambda m: (per.get(m, 1.0), [x["id"] for x in milestones].index(m)))


def rate_of(passes: int, n: int, *, masked: int = 0, model: str = "student", harness: str = "debug-300") -> PassRate:
    lo, hi = wilson(passes, n) if n else (0.0, 1.0)
    return PassRate(model, harness, n, passes, passes / n if n else 0.0, lo, hi, masked=masked)


def stats_from_targeting(row: Mapping[str, Any], conditional: Mapping[str, Any] | None = None, *, frontier_override: Mapping[str, bool] | None = None,
                         applied: Sequence[str] = (), milestones: Sequence[Any] = ()) -> dict[str, Any]:
    """The controller's inputs for one measured task. ``frontier_override``: model -> passed (the re-pathed verdict, which
    wins over the stored one); ``milestones``: the task's milestone records (ids and ``requires``) for C1."""
    front = {m: any(v) for m, v in (row.get("frontier") or {}).items()}
    for m, ok in (frontier_override or {}).items():
        front[m] = bool(ok) or front.get(m, False)
    n, passes = int(row.get("n") or 0), int(row.get("passes") or 0)
    return {"task": str(row.get("task")), "split": str(row.get("split") or "train"), "rate": rate_of(passes, n, masked=int(row.get("masked") or 0)),
            "causes": dict((row.get("cause") or {}).get("labels") or {}), "modes": dict(row.get("modes") or {}),
            # only milestones that HAVE prerequisites: a root milestone's p(m | nothing) is its plain rate, not a checkpoint signal
            "conditional_rates": {k: (v or {}).get("p_given_prereqs") for k, v in ((conditional or {}).get("milestones") or {}).items() if (v or {}).get("requires")},
            "conditional": dict(conditional or {}), "frontier": front, "frontier_passes": any(front.values()) if front else None,
            "applied": list(applied), "milestones": list(milestones), "mean_phi": row.get("mean_phi"), "per_milestone": dict(row.get("per_milestone") or {})}


@dataclass
class ControlController:
    """``difficulty_hooks.Controller`` over ``factory.difficulty.control``. ``supported``: the knobs the paired reviser
    can apply; ``journal``: the controller's action log (every proposal, also the ones that map to no knob)."""

    supported: Collection[str]
    journal: Journal | None = None
    band: Band | None = None
    last: dict[str, Any] = field(default_factory=dict)

    def propose(self, task: Any, stats: Mapping[str, Any]) -> KnobAction | None:
        tid = str(stats.get("task") or getattr(task, "task_id", task))
        a = control(tid, stats["rate"], split=str(stats.get("split") or "train"), band=self.band, causes=stats.get("causes"), modes=stats.get("modes"),
                    conditional_rates=stats.get("conditional_rates"), frontier_passes=stats.get("frontier_passes"), applied=stats.get("applied") or ())
        cp = choose_checkpoint(stats.get("conditional"), stats.get("milestones") or (), band=self.band) if stats.get("milestones") else None
        ok = {k for k in self.supported if k != "C1" or cp is not None}
        hook = from_control(a, supported=ok, params={"checkpoint": cp} if cp else None)
        if hook is not None and hook.knob != "C1" and hook.params.get("checkpoint"):
            hook = KnobAction(hook.direction, hook.knob, hook.reason, milestone=hook.milestone, evidence=hook.evidence)
        if hook is not None and hook.knob == "K1" and hook.milestone is None:
            target = k1_target(stats.get("milestones") or (), stats.get("per_milestone"))
            hook = KnobAction(hook.direction, hook.knob, hook.reason, milestone=target, evidence=hook.evidence) if target else None
        if hook is not None and hook.knob == "K6" and hook.milestone is None:  # the stuck requirement: the lowest measured rate
            per = stats.get("per_milestone") or {}
            stuck = min(per, key=lambda m: per[m]) if per else None
            hook = KnobAction(hook.direction, hook.knob, hook.reason, milestone=stuck, evidence=hook.evidence, params=hook.params)
        self.last = {"control": a.to_json(), "checkpoint": cp, "knob": hook.to_json() if hook else None,
                     "why_none": None if hook else (a.reason if a.action not in ("ease", "harden") else f"no supported knob among {list(a.knobs)}")}
        if self.journal is not None:
            self.journal.add(a, checkpoint=cp, chosen=hook.knob if hook else None, why_none=self.last["why_none"])
        return hook
