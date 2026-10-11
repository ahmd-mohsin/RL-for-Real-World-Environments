"""Acceptance checks for composite tasks (m13 G-3; audit 06 §4.4). They apply to tasks that expose ``composite`` (a
``forge.compose.Composite``: the component map); any other task is not applicable (``not_composite``).

- **A14 necessity**: for every component, the composite oracle *without that component's actions* must fail a gate the
  component owns. A component that can be dropped without failing its own gate is padding (only the last part is really
  graded), and the composite is rejected.
- **A15 complexity gain**: on the calibrated rung, ``p_composite <= min_i p_i - delta`` (``Policy.a15_delta``, 0.1),
  with the same frontier runs A10 uses. Otherwise the composite is "decomposable": no interaction, just a longer list. At
  the bench frontier (every component at p ≈ 0) the gain must show in the failure taxonomy instead: the composite fails a
  gate no component run fails. Pending until both sides have ``a10_min_runs`` scored runs.
- **A17 non-interference** (independent components only: ``Composite.dependent`` is False): each component's own oracle
  actions, run alone inside the composite, still pass every gate that component owns. Fusing must not break a part.
"""

from __future__ import annotations

from typing import Any

from crucible.accept.checks.base import AcceptContext, Out, register
from crucible.accept.model import Status


def _composite(t: Any):
    return getattr(t, "composite", None)


def _na() -> Out:
    return Out(Status.NA, {}, [], na_reason="not_composite")


@register("A14", 1, cost_class="cpu", depends_on=("A1",), detectors=("G-compose-padding",))
async def a14_necessity(t: Any, ctx: AcceptContext) -> Out:
    """Necessity: dropping any component's actions from the composite oracle fails a gate that component owns."""
    comp = _composite(t)
    if comp is None:
        return _na()
    reasons, rows = [], {}
    for c in comp.components:
        r, _ = await ctx.grade(t, f"A14:drop-{c.id}", comp.without(c.id), run=0)
        failing = set(r.failing_gates())
        owned_failing = sorted(failing & set(c.gates))
        rows[c.id] = {"source": c.source, "reward": r.reward, "owned_gates": c.gates, "owned_failing": owned_failing}
        if not owned_failing:
            reasons.append(f"component {c.id} ({c.source}) can be dropped without failing any gate it owns: padding")
    return Out.verdict(reasons, {"ablations": rows})


@register("A15", 1, cost_class="paid", depends_on=("A1",), detectors=("G-compose-decomposable",))
async def a15_complexity_gain(t: Any, ctx: AcceptContext) -> Out:
    """Complexity gain: the composite is measurably harder than its hardest component (or adds a new failure mode at the
    bench frontier)."""
    from crucible.forge.learnability import estimate

    comp = _composite(t)
    if comp is None:
        return _na()
    parts = list(getattr(t, "component_tasks", list)())
    runs = [r for r in (getattr(t, "calibration_runs", list)() or []) if not r.get("masked")]
    part_runs = [[r for r in (getattr(p, "calibration_runs", list)() or []) if not r.get("masked")] for p in parts]
    need_n = ctx.policy.a10_min_runs
    if len(parts) != len(comp.components) or len(runs) < need_n or any(len(pr) < need_n for pr in part_runs):
        return Out(Status.PENDING, {"composite_runs": len(runs), "component_runs": [len(pr) for pr in part_runs]},
                   [f"needs >= {need_n} scored frontier runs of the composite and of every component"])

    def rate(rs):
        return estimate([r.get("reward", 0) == 1.0 for r in rs], model="frontier", harness="mixed")

    rc, rp = rate(runs), [rate(pr) for pr in part_runs]
    delta = ctx.policy.a15_delta
    m: dict[str, Any] = {"composite": {"p": rc.p, "interval": [rc.lo, rc.hi], "n": rc.n},
                         "components": {c.id: {"p": r.p, "interval": [r.lo, r.hi], "n": r.n} for c, r in zip(comp.components, rp, strict=True)},
                         "delta": delta}
    hardest = min(r.p for r in rp)
    if hardest <= 0.05:  # bench frontier: compare failure taxonomies (namespaced gates)
        comp_fail = {g for r in runs for g in (r.get("failing") or [])}
        part_fail = {f"{c.id}.{g}" for c, pr in zip(comp.components, part_runs, strict=True) for r in pr for g in (r.get("failing") or [])}
        new = sorted(comp_fail - part_fail)
        m["new_failing_gates"] = new
        return Out.verdict([] if new else ["at the frontier the composite adds no failure mode its components lack: decomposable"], m)
    ok = rc.p <= hardest - delta
    return Out.verdict([] if ok else [f"composite p={rc.p:.2f} is not below the hardest component's {hardest:.2f} - {delta}: decomposable"], m)


@register("A17", 1, cost_class="cpu", depends_on=("A1",), detectors=("G-compose-interference",))
async def a17_non_interference(t: Any, ctx: AcceptContext) -> Out:
    """Non-interference: each independent component's own oracle keeps its verdict on its own gates inside the composite."""
    comp = _composite(t)
    if comp is None:
        return _na()
    if comp.dependent:
        return Out(Status.NA, {"operator": comp.operator}, [], na_reason="dependent_composite")
    reasons, rows = [], {}
    for c in comp.components:
        r, _ = await ctx.grade(t, f"A17:only-{c.id}", comp.component_oracle(c.id), run=0)
        broken = sorted(set(c.gates) & set(r.failing_gates()))
        rows[c.id] = {"source": c.source, "broken_owned_gates": broken}
        if broken:
            reasons.append(f"component {c.id} ({c.source}) no longer passes its own gates {broken} inside the composite: interference")
    return Out.verdict(reasons, {"components": rows})
