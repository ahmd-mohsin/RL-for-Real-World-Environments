"""Quarantine of training tasks that teach nothing (a compute saver for any trainer's task scheduling).

Adapted from browser-use/rl-env-use (MIT; ``rl_env_use/climb.py`` 267-276: after two rejected rounds a failure audit sets
aside the train tasks that every variant failed on every rep). PATTERN, rewritten for RL: there the set-aside is
permanent and needs a proposer's "task fault" label; here it is measured, temporary and retried.

A group whose members all got the same outcome (every reward equal, and every final Φ equal or unmeasured) has no
signal: GRPO's advantage is zero for it and the trainer drops it (``zero_spread``), but its sampling cost was paid. A task
that produces such groups for ``q`` consecutive observations (each from a different optimizer step; default 2) is
**quarantined**: not sampled for the next ``r`` steps (default 3), then **retried once** (probation: exactly one group).
A probation group with spread releases the task (its strikes reset); a flat one quarantines it again; a probation group
lost to infrastructure (masked, never an observation) re-opens the probation. A task with Φ spread but no full pass is
never quarantined: it still teaches through step-wise credit. Masked groups are not observations.

Held-out tasks never train (V9): the quarantine refuses to be built over one (``split`` other than ``train``), as the
trainers already do. Every event (``quarantined``, ``probation``, ``released``, ``requarantined``, ``all_quarantined``)
is a row with its reason, for the run's ``quarantine.jsonl`` and the dashboards. ``to_json`` / ``from_json`` carry the
state across a resume. Imports the stdlib only, so every trainer (Tinker, the M12 runtime, a future vLLM trainer)
reuses it.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

EPS = 1e-6


@dataclass(frozen=True)
class QuarantineConfig:
    q: int = 2  # consecutive flat observations (distinct steps) before a task is quarantined; 0 disables the quarantine
    r: int = 3  # optimizer steps a quarantined task is not sampled

    def __post_init__(self) -> None:
        if self.q < 0 or self.r < 0:
            raise ValueError("quarantine q and r are >= 0")

    @property
    def enabled(self) -> bool:
        return self.q > 0


def flat(rewards: Sequence[float | None], phis: Sequence[float | None] = ()) -> bool:
    """No outcome spread: every reward equal and every measured final Φ equal (a group with Φ spread still teaches)."""
    rs = [float(x) for x in rewards if x is not None]
    ps = [float(x) for x in phis if x is not None]
    if len(rs) < 2:
        return False  # too few live members to call it flat (the trainer drops it for another reason)
    return statistics.pstdev(rs) <= EPS and (len(ps) < 2 or statistics.pstdev(ps) <= EPS)


@dataclass
class _Task:
    strikes: int = 0
    last_step: int | None = None  # the step of the last counted strike (one strike per step)
    until: int | None = None  # quarantined while step < until
    probation: bool = False  # past ``until``: one retry group allowed
    probe_out: bool = False  # the retry group is in flight
    times: int = 0  # how often it was quarantined
    since: int | None = None  # the step of the last quarantine: a group picked before it is not an observation of it (async R6)


@dataclass
class TaskQuarantine:
    tasks: list[str]
    config: QuarantineConfig = field(default_factory=QuarantineConfig)
    splits: Mapping[str, str] | None = None  # task -> split; anything but "train" is refused
    state: dict[str, _Task] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    # whether the last ``pick`` was a task's probation retry (async-RL audit R6, 2026-10-28): the caller tags the group it
    # launches with it and passes it back to ``observe`` / ``lost``, so only that group decides the probation
    last_probe: bool = False

    def __post_init__(self) -> None:
        held = sorted(t for t, s in (self.splits or {}).items() if str(s or "train") != "train")
        if held:
            raise ValueError(f"held-out tasks never train, so they are never quarantined either (V9): {held}")
        self.tasks = [str(t) for t in self.tasks]
        for t in self.tasks:
            self.state.setdefault(t, _Task())

    # ---- scheduling ----
    def blocked(self, task: str, step: int) -> bool:
        s = self.state.get(task)
        if s is None or not self.config.enabled:
            return False
        if s.until is not None and step < s.until:
            return True
        return s.probation and s.probe_out

    def _open_probations(self, step: int) -> None:
        for t, s in self.state.items():
            if s.until is not None and step >= s.until and not s.probation:
                s.probation, s.until = True, None
                self._event("probation", t, step, f"{self.config.r} steps served: one retry group")

    def pick(self, start: int, step: int) -> int:
        """The index (into ``tasks``) of the next task to sample, round-robin from ``start``, skipping blocked tasks. A
        task on probation gets its one retry group (marked in flight). When every task is blocked, the one released
        soonest is sampled (logged ``all_quarantined``): a run never stalls on its own quarantine."""
        n = len(self.tasks)
        self.last_probe = False
        if not self.config.enabled:
            return start % n
        self._open_probations(step)
        for j in range(n):
            i = (start + j) % n
            t = self.tasks[i]
            if not self.blocked(t, step):
                s = self.state[t]
                if s.probation:
                    s.probe_out = True
                    self.last_probe = True
                return i
        i = min(range(n), key=lambda x: (self.state[self.tasks[x]].until if self.state[self.tasks[x]].until is not None else step, x))
        self._event("all_quarantined", self.tasks[i], step, "every training task is quarantined: sampling the one released soonest")
        return i

    # ---- observations ----
    def observe(self, task: str, rewards: Sequence[float | None], phis: Sequence[float | None], step: int, *, picked: int | None = None,
                probe: bool | None = None) -> dict[str, Any] | None:
        """A landed, unmasked group of ``task`` at optimizer step ``step``; returns the event it caused, if any.

        Async-RL audit R6 (2026-10-28): under asynchronous sampling a group lands steps after it was picked. ``picked`` (the
        step its task was picked at) dates the observation: a group picked before the task's last quarantine measured the
        policy that quarantine already judged, and is not an observation (it used to release the task, or quarantine it
        again, in place of its retry group). ``probe`` (``last_probe`` after the pick that launched it): during a probation
        only the retry group decides; another group of the task is not an observation. Callers that pass neither keep the
        old behaviour (the landing step, and the first group to land decides)."""
        s = self.state.get(task)
        if s is None or not self.config.enabled:
            return None
        at = step if picked is None else int(picked)
        if picked is not None and s.since is not None and at < s.since:
            return None
        if s.probation and probe is False:
            return None
        is_flat = flat(rewards, phis)
        rs = [None if x is None else round(float(x), 6) for x in rewards]
        if s.probation:
            s.probation = s.probe_out = False
            if is_flat:
                return self._quarantine(task, step, f"retry group still flat (rewards {rs})", kind="requarantined")
            s.strikes, s.last_step = 0, None
            return self._event("released", task, step, f"retry group has outcome spread (rewards {rs})")
        if not is_flat:
            s.strikes, s.last_step = 0, None
            return None
        if s.last_step == at:
            return None  # one strike per step: two flat groups of one step are one observation
        s.strikes += 1
        s.last_step = at
        if s.strikes >= self.config.q:
            return self._quarantine(task, step, f"{s.strikes} consecutive flat groups (last rewards {rs}, no Φ spread)")
        return None

    def lost(self, task: str, *, probe: bool | None = None) -> None:
        """A group of ``task`` was masked (infrastructure) or cancelled: not an observation; a lost retry re-opens the
        probation (``probe`` False: another group of the task was lost, the retry is still out)."""
        s = self.state.get(task)
        if s is not None and s.probation and probe is not False:
            s.probe_out = False

    def _quarantine(self, task: str, step: int, why: str, *, kind: str = "quarantined") -> dict[str, Any]:
        s = self.state[task]
        s.until, s.strikes, s.last_step, s.times, s.since = step + self.config.r, 0, None, s.times + 1, step
        return self._event(kind, task, step, why, until=s.until)

    def _event(self, kind: str, task: str, step: int, why: str, **extra: Any) -> dict[str, Any]:
        e = {"event": kind, "task": task, "step": step, "reason": why, **extra}
        self.events.append(e)
        return e

    # ---- views and resume ----
    def active(self, step: int) -> list[str]:
        return [t for t in self.tasks if (s := self.state[t]).until is not None and step < s.until]

    def to_json(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "state": {t: asdict(s) for t, s in self.state.items()}}

    def load(self, doc: Mapping[str, Any] | None) -> TaskQuarantine:
        for t, s in ((doc or {}).get("state") or {}).items():
            if t in self.state:
                self.state[t] = _Task(**{k: v for k, v in dict(s).items() if k in _Task.__dataclass_fields__})
                self.state[t].probe_out = False  # a retry in flight at a crash is sampled again
        return self
