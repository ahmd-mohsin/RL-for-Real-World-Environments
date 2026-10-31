"""Denser Worlds for stored tasks (DESIGN §7.7 item 23; the lead's decision of 2026-10-30).

The re-certification (item 22) put 60 otherwise clean finance_ap tasks at *provisional*: their 30-invoice Worlds give
MUT about 60 distinct effects, and V5's 20% held-out share leaves about 50 shown, under the 59 that the rule of three
needs for U_MUT <= 0.05. The lead kept the held-out rule and asked for denser Worlds. This module re-instantiates a
stored task on a denser World of the same family and measures, per density level, what the certificate will see.

**A denser World must be a superset.** A task's instruction names entities (an invoice number, a vendor id, a PO), and its
contract pins keys. A new draw with more entities (finance_ap's ``invoices`` knob) re-draws everything, so the named
invoice would be another invoice with another decision. A ``Densifier`` maps a ``WorldRef`` and a level to a WorldRef of
the same family whose start state holds every entity of the original unchanged, plus new ones (finance_ap:
``synth_wire.finance_ap_denser``, the pack's ``draw(extra=)``: new invoices on new vendors only).

``densified(task, ref)`` is the next version of the task with only its World replaced: instruction, intent card, contract
and reference program are the same (a framework operation, no model call: never an edit of what the task asks).

**Checks** (``check_dense``; a failure is a finding, never a reason to edit the task):
- the reference program runs and passes its own contract on the denser World;
- its effect (the diff under the contract's normalizers) and its answer are the same as on the original World;
- every observation it made is the same, or only grew (a list holds every original element and more), never changed:
  a changed observation means the denser World could change what the program selects (e.g. "the first pending
  invoice"), and the task is reported as density-sensitive.

**The level** (``measure_level``, ``pick_level``): MUT on the shown split (``MUT(split="shown", holdout=h)``) and on the
held-out split, per level; the chosen level is the smallest whose shown split reaches ``target`` distinct effects
(59: Clopper–Pearson U <= 0.05 at c = 0). Measured on the real pool (runs/taskloop-1.md): MUT's mutant cap
(``FalsifyContext.max_mutants`` = 240) binds before density helps, because sibling swaps of read steps mostly share one
effect and fill the cap; the neighbourhood of a 1-invoice decision on a 30-60 extra-invoice World is about 400
mutants, so the dense certificates use ``max_mutants`` above it and record it (``cap_bound`` says when it binds).

Nothing here knows a World family: the densifier is injected (by import path in a ``spawn`` worker, like policy truths).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import cjson
from crucible.synth.certificate import class_stat
from crucible.synth.falsify.base import FalsifyContext, delta_of, falsify
from crucible.synth.falsify.mut import HOLDOUT, MUT
from crucible.synth.task import Task, WorldRef

TARGET_N = 59  # Clopper-Pearson upper bound <= 0.05 at c = 0 (the certificate's U_MUT bar)
Densifier = Callable[[WorldRef, int], "WorldRef | None"]


def densified(task: Task, ref: WorldRef) -> Task:
    """The next version of ``task`` on the World ``ref``; everything the task asks is unchanged."""
    new = task.next_version(world=ref)
    for k in ("instruction", "intent", "contract", "oracle", "actors", "domain", "id"):
        assert getattr(new, k) == getattr(task, k), k
    return new


def resolve(path: str) -> Densifier:
    mod, _, attr = path.partition(":")
    return getattr(importlib.import_module(mod), attr)


# ---------------------------------------------------------------------------------------------------------- checks

def grew(base: Any, dense: Any) -> bool:
    """``dense`` holds ``base`` with only more list elements (at any depth); scalars that differ next to a grown list
    (a total, a count, a page count) are part of the growth."""
    if cjson(base) == cjson(dense):
        return True
    if isinstance(base, list) and isinstance(dense, list):
        have = {cjson(x) for x in dense}
        return len(dense) >= len(base) and all(cjson(x) in have for x in base)
    if isinstance(base, dict) and isinstance(dense, dict):
        if set(base) - set(dense):
            return False
        lists = [k for k in base if isinstance(base[k], list)]
        for k, v in base.items():
            if isinstance(v, (list, dict)):
                if not grew(v, dense[k]):
                    return False
            elif cjson(v) != cjson(dense[k]) and not (lists and isinstance(v, (int, float)) and not isinstance(v, bool)):
                return False
        return True
    return False


def _masked(err: str | None) -> str | None:
    """An error message with filesystem paths masked (each World runs in its own work directory)."""
    return None if err is None else re.sub(r"(/[^\s'\"]+)+", "<path>", err)


class DenseCheck(CrucibleModel):
    ok: bool
    passes: bool = False
    same_effect: bool = False
    same_answer: bool = False
    observations: dict[str, str] = Field(default_factory=dict)  # step -> same | grew | changed | error
    problems: list[str] = Field(default_factory=list)

    @property
    def sensitive(self) -> bool:
        return any(v in ("changed", "error") for v in self.observations.values())


async def check_dense(base: Task, dense: Task, ctx: FalsifyContext) -> DenseCheck:
    """The checks of the module docstring (one context serves both Worlds: it keys Worlds by their ref)."""
    rb = await ctx.oracle(base)
    rd = await ctx.oracle(dense)
    out = DenseCheck(ok=False)
    if rd.error is not None:
        out.problems.append(f"the reference program fails on the denser World: {rd.error[:200]}")
    v = ctx.grade(rd, dense.contract)
    out.passes = bool(v.passed)
    if not v.passed:
        out.problems.append(f"the reference program fails its contract on the denser World: {list(v.reasons)[:6]}")
    eb = sorted([str(c.addr), c.kind, cjson(c.old), cjson(c.new)] for c in delta_of(rb, base.contract))
    ed = sorted([str(c.addr), c.kind, cjson(c.old), cjson(c.new)] for c in delta_of(rd, dense.contract))
    out.same_effect = eb == ed
    if not out.same_effect:
        out.problems.append("the reference program's effect differs on the denser World")
    out.same_answer = cjson(rb.answer) == cjson(rd.answer)
    if not out.same_answer:
        out.problems.append("the reference program's answer differs on the denser World")
    db = {c.step: c for c in rb.calls}
    for c in rd.calls:
        b = db.get(c.step)
        if b is None or not b.obs.ok or not c.obs.ok:  # an error is the same error when only the World's own paths differ (pilot)
            same = b is not None and b.obs.ok == c.obs.ok and _masked(b.obs.error) == _masked(c.obs.error) and cjson(b.obs.value) == cjson(c.obs.value)
            out.observations[c.step] = "same" if same else "error"
            continue
        out.observations[c.step] = "same" if cjson(b.obs.value) == cjson(c.obs.value) else ("grew" if grew(b.obs.value, c.obs.value) else "changed")
    ch = [k for k, x in out.observations.items() if x in ("changed", "error")]
    if ch:
        out.problems.append(f"observations changed on the denser World (density-sensitive): {ch}")
    out.ok = not out.problems
    return out


# ---------------------------------------------------------------------------------------------------------- levels

class DenseLevel(CrucibleModel):
    level: int
    max_mutants: int
    n_shown: int = 0
    c_shown: int = 0
    u_shown: float = 1.0
    suspects_shown: int = 0
    n_heldout: int = 0
    c_heldout: int = 0
    n_all: int = 0
    probes: int = 0
    cap_bound: bool = False  # the shown split hit max_mutants: the neighbourhood was sampled, not exhausted
    check: DenseCheck | None = None
    seconds: float = 0.0


async def measure_level(base: Task, dense: Task, ctx: FalsifyContext, *, level: int, holdout: float = HOLDOUT) -> DenseLevel:
    t0 = time.monotonic()
    chk = await check_dense(base, dense, ctx)
    out = DenseLevel(level=level, max_mutants=ctx.max_mutants, check=chk)
    if chk.passes:
        s = await falsify(dense, MUT(split="shown", holdout=holdout), ctx)
        h = await falsify(dense, MUT(split="heldout", holdout=holdout), ctx)
        ss, hs = class_stat(s), class_stat(h)
        out.n_shown, out.c_shown, out.u_shown, out.suspects_shown = ss.n, ss.c, round(ss.upper, 4), ss.suspects
        out.n_heldout, out.c_heldout = hs.n, hs.c
        out.probes = len(s.findings)
        out.cap_bound = len(s.findings) >= ctx.max_mutants
        out.n_all = ss.n + hs.n - 1  # Identity is in both splits
    out.seconds = round(time.monotonic() - t0, 1)
    return out


def pick_level(levels: Sequence[DenseLevel], target: int = TARGET_N) -> DenseLevel | None:
    """The smallest measured level whose checks pass and whose shown split reaches ``target`` distinct effects."""
    ok = [x for x in levels if x.check is not None and x.check.ok and x.n_shown >= target]
    return min(ok, key=lambda x: x.level) if ok else None


def densify_job(task_json: str, work: str, levels: Sequence[int], densifier: str, holdout: float, max_mutants: int, target: int = TARGET_N,
                make_world: str = "crucible.synth.recert:work_world") -> str:
    """Worker side (``spawn``): every level in ascending order until one reaches ``target`` (levels after it are not
    measured), plus level 0 (the original World) at the same cap as the counterfactual. JSON: levels, chosen, task."""

    async def go() -> str:
        task = Task.model_validate_json(task_json)
        mw = resolve(make_world)(work)
        ctx = FalsifyContext(mw, max_mutants=max_mutants)
        dz = resolve(densifier)
        rows: list[DenseLevel] = []
        chosen: DenseLevel | None = None
        dense_task: Task | None = None
        try:
            rows.append(await measure_level(task, task, ctx, level=0, holdout=holdout))
            for lv in sorted(levels):
                ref = dz(task.world, lv)
                if ref is None:
                    break
                d = densified(task, ref)
                row = await measure_level(task, d, ctx, level=lv, holdout=holdout)
                rows.append(row)
                if pick_level([row], target) is not None:
                    chosen, dense_task = row, d
                    break
        finally:
            await ctx.close()
        return json.dumps({"levels": [r.model_dump(mode="json") for r in rows], "chosen": chosen.level if chosen else None,
                           "dense_task": dense_task.model_dump(mode="json") if dense_task else None})

    return asyncio.run(go())


__all__ = ["TARGET_N", "DenseCheck", "DenseLevel", "Densifier", "check_dense", "densified", "densify_job", "grew", "measure_level", "pick_level", "resolve"]
