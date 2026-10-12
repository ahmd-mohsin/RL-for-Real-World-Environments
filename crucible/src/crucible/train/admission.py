"""Flaky-task admission filter for training sources (M12-infra D; art. §3.8, MiMo-V2.6: every task's reference solution
is rerun 8 times and any disagreement removes the task; the same rule as Crucible's A4 acceptance check, applied here
to whatever prompts a training source carries).

``screen(sources, rerun, k=8)`` reruns each prompt's reference ``k`` times (concurrently, bounded) and keeps only the
**stable** ones. ``rerun(source, prompt, i)`` returns the verdict of reference run ``i``: a bool, or any hashable full
verdict (reward with per-gate outcomes, as A4 compares), or ``None`` for an infrastructure failure. Verdicts:

- ``stable``: all ``k`` verdicts identical and passing (``passed(verdict)``; default: truthy / reward > 0) → kept;
- ``flaky``: the verdicts disagree → dropped (a flaky task trains noise: the same behaviour is paid and unpaid);
- ``broken``: identical and failing → dropped (the reference cannot solve it: the task or its verifier is wrong);
- ``infra``: more than ``max_infra`` runs failed on infrastructure → dropped and listed for a retry later.

``early_stop`` stops a prompt's reruns at the first disagreement. The report carries, per source, the counts and the
Clopper–Pearson 95% upper bound on the flake rate of the kept set (no disagreement observed in ``k`` runs bounds each
kept task's per-run flake probability; ``verify.calibrate`` supplies the bound).
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Hashable
from dataclasses import asdict, dataclass, field
from typing import Any

from crucible.verify.calibrate import clopper_pearson_upper

VERDICTS = ("stable", "flaky", "broken", "infra")


def _passed(v: Any) -> bool:
    """A reference run passes with a full reward (M9 audit V13: a partial 0.5 counted as a pass, A4 requires 1)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int | float):
        return v >= 1.0
    if isinstance(v, tuple) and v and isinstance(v[0], int | float):
        return v[0] >= 1.0  # (reward, gates...) as A4 builds them
    return bool(v)


def _frozen(v: Any) -> Hashable:
    """A hashable form of a verdict (a dict verdict raised TypeError outside the try and aborted the screen: V13)."""
    try:
        hash(v)
        return v
    except TypeError:
        return json.dumps(v, sort_keys=True, default=str)


@dataclass
class TaskScreen:
    source: str
    prompt: str
    verdict: str
    runs: int
    distinct: int
    infra: int


@dataclass
class ScreenReport:
    kept: dict[str, list[str]]
    screens: list[TaskScreen] = field(default_factory=list)
    k: int = 8

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for s in self.screens:
            out.setdefault(s.source, {v: 0 for v in VERDICTS})[s.verdict] += 1
        return out

    def dropped(self, verdict: str | None = None) -> list[TaskScreen]:
        return [s for s in self.screens if s.verdict != "stable" and (verdict is None or s.verdict == verdict)]

    def flake_upper_95(self) -> float:
        """Per-run flake probability bound for a kept task: 0 disagreements over the *fewest* verdicts any kept task had
        (Clopper–Pearson; M9 audit V13: a task with infra runs has fewer than ``k`` verdicts, the bound used ``k``)."""
        n = min((s.runs - s.infra for s in self.screens if s.verdict == "stable"), default=self.k)
        return clopper_pearson_upper(0, n)

    def to_dict(self) -> dict[str, Any]:
        return {"k": self.k, "counts": self.counts(), "flake_upper_95": self.flake_upper_95(), "kept": {s: len(p) for s, p in self.kept.items()},
                "dropped": [asdict(s) for s in self.dropped()]}


async def screen(sources: dict[str, list[str]], rerun: Callable[[str, str, int], Any] | dict[str, Callable], *, k: int = 8, concurrency: int = 64,
                 max_infra: int = 1, early_stop: bool = True, passed: Callable[[Any], bool] = _passed) -> ScreenReport:
    """``rerun``: one callable, or a per-source map (``{source: fn, "*": default}``); sync or async."""
    if k < 2:
        raise ValueError("k >= 2: one run cannot disagree with itself")
    sem = asyncio.Semaphore(concurrency)

    def fn_for(src: str) -> Callable:
        if callable(rerun):
            return rerun
        f = rerun.get(src, rerun.get("*"))
        if f is None:
            raise ValueError(f"no rerun function for source {src!r}")
        return f

    async def one(src: str, prompt: str, i: int) -> Any:
        async with sem:
            try:
                out = fn_for(src)(src, prompt, i)
                return await out if inspect.isawaitable(out) else out
            except Exception:  # noqa: BLE001 -- a crashing reference run is infrastructure, never a verdict
                return None

    async def task(src: str, prompt: str) -> TaskScreen:
        seen: list[Hashable] = []
        first: list[Any] = []  # the first verdict as returned (``seen`` holds hashable forms)
        infra = runs = 0
        if early_stop:  # sequential per prompt (prompts still run concurrently): stop at the first disagreement
            for i in range(k):
                v = await one(src, prompt, i)
                runs += 1
                if v is None:
                    infra += 1
                    if infra > max_infra:
                        break
                    continue
                first[:] = first or [v]
                seen.append(_frozen(v))
                if len(set(seen)) > 1:
                    break
        else:
            vs = await asyncio.gather(*(one(src, prompt, i) for i in range(k)))
            runs, infra, seen = k, sum(v is None for v in vs), [_frozen(v) for v in vs if v is not None]
            first = [next(v for v in vs if v is not None)] if seen else []
        distinct = len(set(seen))
        if infra > max_infra:
            verdict = "infra"
        elif distinct > 1:
            verdict = "flaky"
        elif not seen or not passed(first[0]):
            verdict = "broken"
        else:
            verdict = "stable"
        return TaskScreen(src, prompt, verdict, runs, distinct, infra)

    screens = await asyncio.gather(*(task(s, p) for s, ps in sources.items() for p in ps))
    kept = {s: [x.prompt for x in screens if x.source == s and x.verdict == "stable"] for s in sources}
    return ScreenReport(kept=kept, screens=list(screens), k=k)
