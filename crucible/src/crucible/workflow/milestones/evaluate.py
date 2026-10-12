"""Evaluating milestones on states: φ_i(s) ∈ [0, 1] for each milestone at each step boundary.

A milestone's check is a ``Query`` against one App plus an ``Expectation`` (``core/schema/milestone.py``). The same
milestone set runs against

- a live World, through its ``VerifierView`` (the policy never sees the result);
- a reconstructed state of a recorded trace, through ``FileStateView`` (fs ops ``read`` / ``exists`` / ``list`` /
  ``grep`` answered from a ``FileState``, with the fs adapter's exact result shapes);
- any object with ``async query(app, Query) -> QueryResult``.

``evaluate_series`` applies the cost policy of DESIGN §5.3.5: cheap milestones every step, medium ones at
checkpoints (build/test steps, or every k steps), expensive ones only at the end; a milestone not evaluated at a step
keeps its last value (``evaluated`` records which values are fresh). A check that cannot run (an error, not a
negative answer) yields ``None``, never 0: a failed measurement is not a zero.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from crucible.core.schema.milestone import Expectation, Milestone
from crucible.core.schema.state import Query, QueryResult
from crucible.core.textblock import block
from crucible.workflow.filestate import FileState


class StateView(Protocol):
    async def query(self, app: str, q: Query | dict[str, Any]) -> QueryResult: ...


def _count(value: Any) -> int | None:
    if isinstance(value, dict):
        for k in ("hits", "paths", "rows", "entries"):
            if isinstance(value.get(k), list):
                return len(value[k])
        if "count" in value:
            return int(value["count"])
        if "exists" in value:
            return int(bool(value["exists"]))
    if isinstance(value, list | tuple | str):
        return len(value)
    if isinstance(value, int | float | bool):
        return int(value)
    return None


def decode(v: Any) -> Any:
    """Undo the State Adapters' exact encodings (the sql adapter writes floats as ``{"f": float.hex()}`` so that
    fingerprints are exact): an expectation compares values, not encodings."""
    if isinstance(v, dict):
        if set(v) == {"f"} and isinstance(v["f"], str):
            try:
                return float.fromhex(v["f"])
            except ValueError:
                return v
        return {k: decode(x) for k, x in v.items()}
    if isinstance(v, list):
        return [decode(x) for x in v]
    return v


def phi(expect: Expectation, r: QueryResult) -> float | None:
    """φ for one query result; None when the check itself failed to run."""
    if r.error and not r.ok and r.exit_code is None:
        return None
    k = expect.kind
    if k == "ok":
        return 1.0 if r.ok and (r.exit_code in (None, 0)) else 0.0
    if not r.ok:
        return None
    v = decode(r.value)
    if k == "truthy":
        if isinstance(v, dict) and "exists" in v:
            return 1.0 if v["exists"] else 0.0
        return 1.0 if v else 0.0
    if k == "equals":
        return 1.0 if v == expect.value else 0.0
    if k == "regex":
        text = v if isinstance(v, str) else (v.get("text") if isinstance(v, dict) else r.stdout) or ""
        return 1.0 if re.search(str(expect.value), text or "") else 0.0
    if k in ("count_eq", "count_ge"):
        n = _count(v)
        if n is None:
            return None
        target = int(expect.value or 0)
        return 1.0 if (n == target if k == "count_eq" else n >= target) else 0.0
    if k == "between":
        try:
            x = float(v if not isinstance(v, dict) else v.get("value"))
        except (TypeError, ValueError):
            return None
        return 1.0 if (expect.lo is None or x >= expect.lo) and (expect.hi is None or x <= expect.hi) else 0.0
    if k == "score":
        try:
            x = float(v if not isinstance(v, dict) else v.get("score"))
        except (TypeError, ValueError):
            return None
        return min(1.0, max(0.0, x))
    raise ValueError(f"unknown expectation kind {k!r}")


@dataclass
class FileStateView:
    """fs-adapter queries answered from a reconstructed ``FileState`` (offline evaluation of recorded traces).
    ``listing``: optional full file list of the base snapshot (for globs over files the trace never touched)."""

    fs: FileState
    app: str = "repo"
    listing: Callable[[], Iterable[str]] | None = None

    def _paths(self) -> list[str]:
        known = {p for p, c in self.fs.files.items() if c is not None}
        if self.listing is not None:
            known |= {p for p in self.listing() if self.fs.files.get(p, "") is not None}
        return sorted(known)

    async def query(self, app: str, q: Query | dict[str, Any]) -> QueryResult:
        q = q if isinstance(q, Query) else Query.model_validate(q)
        a = q.args
        stale = getattr(self.fs, "stale", set())
        touched = [a["path"]] if "path" in a else list(a.get("paths") or ([] if not stale else self._paths()))
        if hit := [p for p in touched if p in stale]:
            # an unreplayable command wrote it: the rebuilt content is not the agent's, so the check is unmeasured
            return QueryResult(ok=False, error=f"stale after an unreplayable command: {hit[0]}")
        try:
            if q.op == "read":
                t = self.fs.read(a["path"])
                if t is None:
                    return QueryResult(ok=False, error=f"no such file {a['path']}")
                mb = int(a.get("max_bytes", 200_000))
                return QueryResult(ok=True, value={"text": t[:mb], "size": len(t.encode()), "truncated": len(t) > mb})
            if q.op == "exists":
                t = self.fs.read(a["path"])
                return QueryResult(ok=True, value={"exists": t is not None, "is_file": t is not None, "is_dir": False})
            if q.op == "list":
                pat = a.get("pattern")
                return QueryResult(ok=True, value=[p for p in self._paths() if not pat or fnmatch.fnmatch(p, pat)])
            if q.op == "grep":
                rx = re.compile(a["pattern"])
                g, max_hits = a.get("glob"), int(a.get("max_hits", 200))
                paths = a.get("paths") or [p for p in self._paths() if not g or fnmatch.fnmatch(p, g)]
                hits: list[list[Any]] = []
                for p in paths:
                    t = self.fs.read(p)
                    if t is None:
                        continue
                    for n, line in enumerate(t.split("\n"), 1):
                        if rx.search(line):
                            hits.append([p, n, line[:400]])
                            if len(hits) >= max_hits:
                                return QueryResult(ok=True, value={"hits": hits, "truncated": True})
                return QueryResult(ok=True, value={"hits": hits, "truncated": False})
            if q.op == "block_changed":
                now, base = self.fs.read(a["path"]), (self.fs.base(a["path"]) if self.fs.base else None)
                if now is None or base is None:
                    return QueryResult(ok=True, value={"changed": now != base, "found": False})
                b0, b1 = block(base, a["anchor"]), block(now, a["anchor"])
                return QueryResult(ok=True, value={"changed": b0 != b1, "found": b1 is not None})
        except (KeyError, re.error) as e:
            return QueryResult(ok=False, error=f"{q.op}: {e}")
        return QueryResult(ok=False, error=f"op {q.op!r} not available offline (needs a live World)")


@dataclass
class PhiSeries:
    """φ over a sequence of states: ``values[t][milestone_id]`` (None = could not be measured)."""

    milestones: list[str]
    steps: list[int]  # step index (or trace seq) of each state
    values: list[dict[str, float | None]] = field(default_factory=list)
    evaluated: list[set[str]] = field(default_factory=list)  # which milestones were freshly evaluated at each state

    def first_satisfied(self, mid: str) -> int | None:
        for t, row in enumerate(self.values):
            if (row.get(mid) or 0) >= 1.0:
                return t
        return None

    def column(self, mid: str) -> list[float | None]:
        return [row.get(mid) for row in self.values]


async def evaluate(milestones: Sequence[Milestone], view: StateView, only: set[str] | None = None) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for m in milestones:
        if only is not None and m.id not in only:
            continue
        try:
            r = await view.query(m.check.app, m.check.query)
        except Exception as e:  # noqa: BLE001 -- a check that crashes is an unmeasured value, never a zero
            r = QueryResult(ok=False, error=str(e)[:300])
        out[m.id] = phi(m.check.expect, r)
    return out


async def evaluate_series(milestones: Sequence[Milestone], states: Sequence[tuple[int, StateView]], *,
                          checkpoints: set[int] | None = None, every: int = 0, final_expensive: bool = True) -> PhiSeries:
    """Cost policy: cheap at every state; medium at ``checkpoints`` (state indices), every ``every`` states and at the
    last state; expensive only at the last state (if ``final_expensive``). Unevaluated values carry forward."""
    ids = [m.id for m in milestones]
    series = PhiSeries(milestones=ids, steps=[s for s, _ in states])
    last: dict[str, float | None] = dict.fromkeys(ids)
    n = len(states)
    for t, (_, view) in enumerate(states):
        final = t == n - 1
        due = {m.id for m in milestones if m.cost == "cheap"
               or (m.cost == "medium" and (final or (checkpoints and t in checkpoints) or (every and t % every == 0)))
               or (m.cost == "expensive" and final and final_expensive)}
        fresh = await evaluate([m for m in milestones if m.id in due], view)
        last.update(fresh)
        series.values.append(dict(last))
        series.evaluated.append(set(fresh))
    return series
