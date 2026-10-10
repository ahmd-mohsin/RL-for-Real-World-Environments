"""Performance bars against the original on the same machine (DESIGN §8.1, Appendix B.5). Standard library only.

Generalized from task03/04 ``gate_perf`` / ``perf_round``:

- rounds alternate original and candidate, so drift hits both; each metric keeps its **best of k** (minimum);
- ``ratio = candidate / original <= bar`` per metric;
- one **re-measurement** when bars fail, so a transient load cannot fail a correct solution (task03/04 behaviour);

and adds what the 2026-10-09 audit and Appendix B.5 asked for:

- a **load guard** before every round (1-minute load average and GPU utilization from ``nvidia-smi``, if present): a
  busy node waits, and if it never quiets the grade is **infra** (re-graded later), never a fail;
- a **round-spread check**: if a side's spread across rounds exceeds ``spread_max`` the measurement is contaminated;
  it is re-measured, and if still contaminated the result is infra, not a verdict;
- **outside timing**: ``round_fn`` may return the grader-measured wall time of each run (``RunResult.wall_s``); with
  ``wall_ratio_max`` the candidate's whole-process wall time must also beat that ratio, so timers forged inside the
  candidate's process cannot buy the bar alone (P26);
- an end-state **sanity hook** (task03's "the benchmark simulated the same physics"): computed results of timed runs
  must match the original's, else the gate fails (a fast wrong answer is not fast).

``round_fn(rnd) -> {"original": {metric: ms}, "candidate": {metric: ms}, "wall": {"original": s, "candidate": s}}``;
it raises ``CandidateError`` when the candidate's timed run fails or computes something different.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass

from crucible.verify.runtime.plan import Verdict
from crucible.verify.runtime.trusted import InfraError


class CandidateError(RuntimeError):
    """The candidate's timed run failed or computed a different result: a perf failure, not infra."""


@dataclass
class LoadGuard:
    max_load_per_core: float = 0.5  # 1-minute load average / cores
    max_gpu_util: int = 20  # percent, sampled before the round
    wait_s: float = 900.0
    poll_s: float = 15.0
    ignore_gpu: bool = False

    def sample(self) -> dict:
        s = {"load1": None, "cores": os.cpu_count() or 1, "gpu_util": None}
        try:
            s["load1"] = os.getloadavg()[0]
        except OSError:
            pass
        if not self.ignore_gpu and shutil.which("nvidia-smi"):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True,
                                     text=True, timeout=20, check=False).stdout
                vals = [int(x) for x in out.split() if x.strip().isdigit()]
                s["gpu_util"] = max(vals) if vals else None
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
        return s

    def quiet(self, s: dict) -> bool:
        busy_cpu = s["load1"] is not None and s["load1"] / s["cores"] > self.max_load_per_core
        busy_gpu = s["gpu_util"] is not None and s["gpu_util"] > self.max_gpu_util
        return not busy_cpu and not busy_gpu

    def wait(self) -> dict:
        t0 = time.monotonic()
        while True:
            s = self.sample()
            if self.quiet(s):
                return {**s, "waited_s": round(time.monotonic() - t0, 1)}
            if time.monotonic() - t0 > self.wait_s:
                raise InfraError(f"node never quiet for perf grading: {s}")
            time.sleep(self.poll_s)


@dataclass
class PerfBars:
    bars: dict  # metric -> max ratio (candidate / original)
    round_fn: Callable[[int], dict]
    rounds: int = 3
    remeasure: int = 1
    spread_max: float | None = 0.25  # (max - min) / min across rounds, per side and metric
    wall_ratio_max: float | None = None
    guard: LoadGuard | None = None

    def _attempt(self) -> dict:
        best = {"original": {}, "candidate": {}}
        rounds = {"original": {}, "candidate": {}}
        walls = {"original": [], "candidate": []}
        loads = []
        for rnd in range(self.rounds):
            if self.guard:
                loads.append(self.guard.wait())
            r = self.round_fn(rnd)
            for side in ("original", "candidate"):
                for m, v in r[side].items():
                    v = float(v)
                    best[side][m] = min(v, best[side].get(m, v))
                    rounds[side].setdefault(m, []).append(round(v, 4))
                if "wall" in r and r["wall"].get(side) is not None:
                    walls[side].append(float(r["wall"][side]))
        res, ok = {}, True
        for m, bar in self.bars.items():
            o, c = best["original"].get(m), best["candidate"].get(m)
            ratio = c / o if o and c is not None else float("inf")
            res[m] = {"original": round(o, 4) if o else None, "candidate": round(c, 4) if c is not None else None, "ratio": round(ratio, 3),
                      "max_ratio": bar, "pass": ratio <= bar}
            ok &= ratio <= bar
        spread = {}
        if self.spread_max is not None:
            for side, series in rounds.items():
                for m, vs in series.items():
                    if len(vs) > 1 and min(vs) > 0:
                        spread[f"{side}/{m}"] = round((max(vs) - min(vs)) / min(vs), 3)
        contaminated = sorted(k for k, v in spread.items() if v > self.spread_max) if self.spread_max is not None else []
        wall = None
        if self.wall_ratio_max is not None and walls["original"] and walls["candidate"]:
            wr = min(walls["candidate"]) / min(walls["original"])
            wall = {"ratio": round(wr, 3), "max_ratio": self.wall_ratio_max, "pass": wr <= self.wall_ratio_max,
                    "original_s": round(min(walls["original"]), 3), "candidate_s": round(min(walls["candidate"]), 3)}
            ok &= wall["pass"]
        return {"pass": ok, "metrics": res, "rounds": rounds, "spread": spread, "contaminated": contaminated, "wall": wall, "load": loads}

    def __call__(self, ctx=None) -> Verdict:
        attempts = []
        try:
            for _ in range(1 + self.remeasure):
                a = self._attempt()
                attempts.append(a)
                if a["pass"] and not a["contaminated"]:
                    break
        except CandidateError as ex:
            return Verdict(False, {"error": str(ex)[:1500], "attempts": attempts})
        last = attempts[-1]
        if not last["pass"] and last["contaminated"]:
            return Verdict(False, {"attempts": attempts}, infra=f"perf rounds contaminated: {last['contaminated'][:5]}")
        if last["pass"] and last["contaminated"]:
            # passing bars under noise: noise only slows runs down, but the original's best may have been the noisy one
            orig = [k for k in last["contaminated"] if k.startswith("original/")]
            if orig:
                return Verdict(False, {"attempts": attempts}, infra=f"original's perf rounds contaminated: {orig[:5]}")
        return Verdict(last["pass"], {**last, **({"first_attempts": attempts[:-1]} if len(attempts) > 1 else {})})
