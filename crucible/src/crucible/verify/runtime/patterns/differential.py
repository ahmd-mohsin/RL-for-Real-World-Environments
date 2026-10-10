"""Differential execution, concurrency equivalence and fresh draws (DESIGN §8.1). Standard library only.

**Differential** (generalized from task03/04 ``ref`` / ``cand`` / ``gate_kinfu`` / ``gate_ncv``): for each case, the
*reference* (the original build, a reference world) runs as root into a 0700 directory, computed at grading time and
never shipped to the agent; the *candidate* runs unprivileged (fresh HOME/TMPDIR) into a candidate-writable directory
that is sealed before a trusted comparator reads it. References are cached per grade, so later gates (fresh, perf)
reuse them. Reference self-consistency (A4, Appendix B.2): ``self_check=k`` reruns a reference ``k`` times and fails
the grade as *infra* if the original is not deterministic on that case.

**Concurrency equivalence** (task04 ``gate_concurrency``; Appendix B.3): items run solo, then together for ``rounds``
rounds; every concurrent item must equal **its own solo run** bitwise (never the original's concurrent run: the
original itself may not be concurrency-safe).

**Fresh draws** (task03/04 ``gate_fresh``): workloads generated from a seed drawn at grading time with ``secrets``; the
seed is revealed in the report only after the candidate has run.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.verify.runtime.plan import Verdict
from crucible.verify.runtime.trusted import InfraError

# runner signatures:
#   reference(case_name, args, out_dir) -> None        (root; raise InfraError on failure)
#   candidate(case_name, args, out_dir) -> RunResult    (via Trusted.run(candidate=True))
#   compare(cand_dir, ref_dir) -> list[failure] | (list, tier)


@dataclass
class References:
    """Per-grade cache of reference outputs (root-only)."""

    root: Path
    run: Callable[[str, Any, Path], None]
    trusted: Any
    done: dict = field(default_factory=dict)

    def get(self, name: str, args: Any, *, self_check: int = 0, compare: Callable | None = None) -> Path:
        if name in self.done:
            return self.done[name]
        out = self.root / name
        self.trusted.private(out)
        self.run(name, args, out)
        for k in range(self_check):
            again = self.root / f"{name}.selfcheck{k}"
            self.trusted.private(again)
            self.run(name, args, again)
            diff = (compare or (lambda a, b: []))(again, out)
            diff = diff[0] if isinstance(diff, tuple) else diff
            if diff:
                raise InfraError(f"the reference is not self-consistent on {name}: {diff[:3]}")
        self.done[name] = out
        return out


def _cmp(compare: Callable, c: Path, r: Path) -> tuple[list, str | None]:
    res = compare(c, r)
    return (res[0], res[1]) if isinstance(res, tuple) else (res, None)


def differential(ctx, *, cases: dict[str, Any], refs: References, candidate: Callable[[str, Any, Path], Any],
                 compare: Callable, out_root: Path, success: Callable[[Any, Path], bool] | None = None, max_report: int = 12) -> Verdict:
    t = ctx.trusted
    detail, ok, tiers = {}, True, {}
    for name, args in cases.items():
        ref = refs.get(name, args)
        out = t.udir(Path(out_root) / name)
        rr = candidate(name, args, out)
        t.seal(out)
        produced = success(rr, out) if success else getattr(rr, "rc", 0) == 0
        if not produced:
            detail[name] = {"error": "run failed", "rc": getattr(rr, "rc", None), "log": _tail(getattr(rr, "log", ""), 15)}
            ok = False
            continue
        fails, tier = _cmp(compare, out, ref)
        if tier:
            tiers[name] = tier
        detail[name] = {"pass": not fails, "failures": fails[:max_report], **({"tier": tier} if tier else {})}
        ok &= not fails
    return Verdict(ok, {"cases": detail, **({"tiers": tiers} if tiers else {})})


def concurrency_equivalence(ctx, *, items: list[Any], solo: Callable[[int, Any, Path], Any], together: Callable[[list, Path], Any],
                            item_dir: Callable[[Path, int], Path], compare: Callable, out_root: Path, rounds: int = 2) -> Verdict:
    """``solo(k, item, out)`` runs one item alone; ``together(items, out)`` runs all at once, writing item k's
    artifacts to ``item_dir(out, k)``; ``compare`` is usually ``bitwise_tree``."""
    t = ctx.trusted
    solos = []
    for k, it in enumerate(items):
        d = t.udir(Path(out_root) / f"solo{k}")
        rr = solo(k, it, d)
        t.seal(d)
        if getattr(rr, "rc", 0) != 0:
            return Verdict(False, {"error": f"solo run {k} failed", "log": _tail(getattr(rr, "log", ""), 15)})
        solos.append(d)
    mism = []
    for rnd in range(rounds):
        d = t.udir(Path(out_root) / f"round{rnd}")
        rr = together(items, d)
        t.seal(d)
        if getattr(rr, "rc", 0) != 0:
            mism.append({"round": rnd, "error": "concurrent run failed", "log": _tail(getattr(rr, "log", ""), 15)})
            continue
        for k in range(len(items)):
            try:
                diff, _ = _cmp(compare, item_dir(d, k), solos[k])
            except Exception as ex:  # noqa: BLE001 -- malformed candidate output
                diff = [repr(ex)]
            if diff:
                mism.append({"round": rnd, "item": k, "differs": diff[:6]})
    return Verdict(not mism, {"mismatches": mism[:12], "items": len(items), "rounds": rounds})


def draw_seed(ctx, key: str = "fresh_seed", bits: int = 48) -> int:
    """A seed drawn at grading time; recorded in ``ctx.revealed`` (reported after the candidate ran)."""
    seed = secrets.randbits(bits)
    ctx.revealed[key] = seed
    return seed


def fresh(ctx, *, generate: Callable[[int], dict[str, Any]], refs: References, candidate: Callable, compare: Callable, out_root: Path,
          seed: int | None = None, success: Callable | None = None) -> Verdict:
    s = draw_seed(ctx) if seed is None else seed
    cases = {f"fresh/{k}": v for k, v in generate(s).items()}
    v = differential(ctx, cases=cases, refs=refs, candidate=candidate, compare=compare, out_root=out_root, success=success)
    v.detail["seed"] = s
    return v


def _tail(text: str, n: int = 40) -> str:
    return "\n".join((text or "").splitlines()[-n:])
