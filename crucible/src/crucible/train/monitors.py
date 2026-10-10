"""Training monitors that catch silent failure (DESIGN §10.3 point 8; research m11-m12/06 checklist 24, 29, 31, 51–52).

``StepMonitor.record(step, ...)`` takes one optimizer step's facts and returns a flat metric dict (logged every step):

- staleness: ``staleness/{mean,max}``, ``trajectory_spans/{mean,max}``, ``on_policy_share``, evictions by reason;
- mismatch: whatever ``mismatch.correct`` measured (``rollout_probs_diff``, r_dis quantiles, masked / clipped shares);
- length: mean and p99 response length, truncation share, and the **length-bias KS distance** between this batch's
  lengths and the trailing steady-state window (DeepSeek / MiniMax: early batches skew short);
- learning signal: entropy, share of zero-variance groups, confirmed-hack rate, pass rate;
- timing: bubble time per phase (trainer idle, engine down, env wait), publish latency, re-prefill tokens.

``PassRateAlarm``: a jump of more than ``delta`` in the training pass rate within ``window`` steps raises an alarm; the
caller re-audits the batch's rollouts with the full verifier tier (a sudden jump is the classic signature of a hack:
LongCat's Lean4 pass rate fell from 97.9% to 27.9% once the verifier was fixed).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from crucible.train.sim import ks_distance


def _mean(x: Sequence[float]) -> float:
    return sum(x) / len(x) if x else 0.0


def _p(x: Sequence[float], q: float) -> float:
    if not x:
        return 0.0
    s = sorted(x)
    return float(s[min(len(s) - 1, int(q * len(s)))])


@dataclass
class PassRateAlarm:
    delta: float = 0.25
    window: int = 5
    history: deque = field(default_factory=lambda: deque(maxlen=64))

    def update(self, step: int, pass_rate: float) -> dict | None:
        self.history.append((step, pass_rate))
        recent = [p for s, p in self.history if s > step - self.window]
        older = [p for s, p in self.history if s <= step - self.window]
        if older and recent and max(recent) - _mean(older[-self.window:]) > self.delta:
            return {"step": step, "pass_rate": pass_rate, "baseline": round(_mean(older[-self.window:]), 4), "action": "full-tier re-audit of this batch"}
        return None


@dataclass
class StepMonitor:
    steady_window: int = 20
    alarm: PassRateAlarm = field(default_factory=PassRateAlarm)
    lengths: deque = field(default_factory=lambda: deque(maxlen=4096))
    alarms: list = field(default_factory=list)

    def record(self, step: int, *, staleness: Sequence[int] = (), spans: Sequence[int] = (), lengths: Sequence[int] = (), truncated: Sequence[bool] = (),
               rewards: Sequence[float] = (), group_rewards: Sequence[Sequence[float]] = (), entropy: float | None = None, hacks: int = 0,
               mismatch: dict | None = None, timing: dict | None = None, evicted: dict | None = None, worst: Sequence[int] = (),
               evicted_staleness: Sequence[int] = ()) -> dict:
        m: dict[str, float] = {}
        if staleness:
            m.update({"staleness/mean": _mean(staleness), "staleness/max": max(staleness), "on_policy_share": sum(1 for s in staleness if s == 0) / len(staleness)})
        if worst:
            m.update({"staleness_worst/mean": _mean(worst), "staleness_worst/max": max(worst)})
        if evicted_staleness:
            m.update({"evicted_staleness/mean": _mean(evicted_staleness), "evicted_staleness/max": max(evicted_staleness)})
        if spans:
            m.update({"trajectory_spans/mean": _mean(spans), "trajectory_spans/max": max(spans)})
        if lengths:
            steady = list(self.lengths)[-self.steady_window * len(lengths):]
            m.update({"length/mean": _mean(lengths), "length/p99": _p(lengths, 0.99), "length/ks_vs_steady": ks_distance(list(lengths), steady) if steady else 0.0})
            self.lengths.extend(lengths)
        if truncated:
            m["truncation_share"] = sum(truncated) / len(truncated)
        if rewards:
            pr = sum(1 for r in rewards if r >= 1.0) / len(rewards)
            m["pass_rate"] = pr
            a = self.alarm.update(step, pr)
            if a:
                self.alarms.append(a)
                m["alarm/pass_rate_jump"] = 1.0
        if group_rewards:
            m["zero_variance_groups"] = sum(1 for g in group_rewards if len(set(g)) <= 1) / len(group_rewards)
        if entropy is not None:
            m["entropy"] = entropy
        if rewards:
            m["confirmed_hack_rate"] = hacks / len(rewards)
        for k, v in (mismatch or {}).items():
            m[f"mismatch/{k}"] = v
        for k, v in (timing or {}).items():
            m[f"time/{k}"] = v
        for k, v in (evicted or {}).items():
            m[f"evicted/{k}"] = v
        return m
