"""The stitch operator (m13 G-10; audit 06 §4.3): span tasks of one seed stitched back into the long-horizon task.

G-span (M8) cuts one expert end state into module tasks: each span's start pre-applies the oracle's changes to every
other module, and its oracle is its own module's changes. ``stitch`` inverts that, per file (``path -> diff``):

- **Owned files:** a span's oracle files. Two spans may own the same file only with the identical diff (a nested span,
  e.g. NCV inside People: recorded in ``nested``); a different diff is a conflict.
- **Consistency:** a span's start may pre-apply a file another span owns only with that span's oracle diff (otherwise the
  spans were cut from different end states).
- **Residual:** files every span's start pre-applies and no span owns (shared helpers no span teaches). They must agree
  across starts. ``residual="oracle"`` (default) makes them part of the stitched oracle, so the stitched task starts
  from the seed's base, as the original did; ``"start"`` pre-applies them (an easier variant, reported as such).
- **Uncovered:** a file only some starts pre-apply and no span owns: the stitched task could not provide it (error).
- **Gates:** the union of the spans' gates.

The stitched task reproduces the seed when its start and oracle together equal the seed's oracle file for file and its
gates equal the seed's: the same end state under the same grader gives the same verdicts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class Span:
    id: str
    start: dict[str, str]  # path -> diff (base -> span start)
    oracle: dict[str, str]  # path -> diff (span start -> span oracle)
    gates: list[str] = field(default_factory=list)


@dataclass
class Stitched:
    start: dict[str, str]
    oracle: dict[str, str]
    owners: dict[str, list[str]]
    residual: list[str]
    nested: list[tuple[str, str]]
    gates: list[str]
    residual_mode: str

    def files(self) -> dict[str, str]:
        return {**self.start, **self.oracle}


def stitch(spans: Sequence[Span], *, residual: str = "oracle", gate_order: Sequence[str] = ()) -> Stitched:
    if residual not in ("oracle", "start"):
        raise ValueError("residual must be 'oracle' or 'start'")
    if len(spans) < 2:
        raise ValueError("stitch needs at least two spans")
    owned: dict[str, str] = {}
    owners: dict[str, list[str]] = {}
    for s in spans:
        for p, d in s.oracle.items():
            if p in owned and owned[p] != d:
                raise ValueError(f"spans {owners[p]} and {s.id} change {p} differently: cut from different end states")
            owned[p] = d
            owners.setdefault(p, []).append(s.id)
    for s in spans:
        for p, d in s.start.items():
            if p in owned and owned[p] != d and s.id not in owners[p]:
                raise ValueError(f"{s.id}'s start pre-applies {p} differently from its owner {owners[p]}")
    nested = sorted((a.id, b.id) for a in spans for b in spans if a.id != b.id and a.oracle and set(a.oracle) <= set(b.oracle))
    in_every = set.intersection(*[set(s.start) for s in spans]) - set(owned)
    res: dict[str, str] = {}
    for p in sorted(in_every):
        diffs = {s.start[p] for s in spans}
        if len(diffs) > 1:
            raise ValueError(f"the spans' starts disagree on the residual file {p}")
        res[p] = diffs.pop()
    uncovered = sorted(set().union(*[set(s.start) for s in spans]) - set(owned) - in_every)
    if uncovered:
        raise ValueError(f"files only some starts pre-apply and no span owns (the stitched task cannot provide them): {uncovered[:10]}")
    gates: list[str] = []
    for g in [*gate_order, *[g for s in spans for g in s.gates]]:
        if g not in gates and any(g in s.gates for s in spans):
            gates.append(g)
    start = {} if residual == "oracle" else dict(res)
    oracle = {**owned, **(res if residual == "oracle" else {})}
    return Stitched(start, oracle, owners, sorted(res), nested, gates, residual)
