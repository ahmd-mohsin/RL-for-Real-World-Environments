"""Which training task to sample next: online difficulty filtering plus a reverse curriculum over checkpoint starts
(swarm-pipeline/12 §1, §4). Backend-neutral: the Tinker trainer uses it, and so can a DeltaAI (vLLM) trainer.

``DifficultySampler`` keeps a ``forge.difficulty.DifficultyTracker`` (p̂ per task, decayed as the policy moves) and the
existing ``train.quarantine.TaskQuarantine`` (which it does not duplicate: the quarantine still decides when a flat task
sits out and when it is retried). On top:

- **Filter.** A task whose p̂ interval is outside the band (default the train band [0.1, 0.7]) is not sampled, unless
  its Φ still spreads (it teaches through step credit) or ``explore`` picks it: with probability ``explore`` the next
  pick ignores the filter, so an out-of-band task is re-measured as the policy changes (Online Difficulty Filtering's
  filter is re-run every step for the same reason).
- **Weight.** Among the kept tasks a task is drawn with probability proportional to ``max(floor, signal)``, where
  ``signal = max(4p(1-p), 4 Var(Φ))`` (``TaskRate.signal``); an unmeasured task has signal 1, so every task is measured
  first. Draws use a seeded RNG whose state is saved with the run (a resume picks the same tasks).
- **Masked draws are never evidence** (the tracker drops them) and **held-out tasks are refused** at construction (V9).
- **Reverse curriculum.** A task built as a checkpoint start (``factory.difficulty.checkpoint_starts``: the start tree
  with the real fixes of the first L milestones applied) carries ``(parent, level)``; a higher level starts later (fewer
  milestones left). Per parent only the current level is sampled, starting at the highest. When the level's p̂ lower
  bound reaches ``promote`` (default 0.5, with at least ``min_n`` effective draws) the start moves one level earlier;
  when its upper bound falls below ``demote`` it moves back. Level 0 is the full task. Every move is an event row.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.forge.difficulty import TRAIN_BAND, DifficultyTracker, Draw
from crucible.forge.learnability import Band
from crucible.train.quarantine import TaskQuarantine


@dataclass(frozen=True)
class SamplerConfig:
    band_lo: float | None = TRAIN_BAND.lo
    band_hi: float | None = TRAIN_BAND.hi
    decay: float = 0.9  # per optimizer step, on the old evidence
    explore: float = 0.1  # share of picks that ignore the band filter (re-measure out-of-band tasks)
    floor: float = 0.05  # minimum weight of a kept task
    min_n: float = 4.0  # effective draws before the band filter or a curriculum move applies
    promote: float = 0.5
    demote: float = 0.1

    def __post_init__(self) -> None:
        if not (0.0 <= self.explore <= 1.0 and 0.0 < self.floor <= 1.0 and self.min_n >= 0):
            raise ValueError("explore in [0, 1], floor in (0, 1], min_n >= 0")
        if self.demote > self.promote:
            raise ValueError("demote must not exceed promote")

    @property
    def band(self) -> Band:
        return Band(lo=self.band_lo, hi=self.band_hi, on="p")


@dataclass
class DifficultySampler:
    tasks: list[str]
    config: SamplerConfig = field(default_factory=SamplerConfig)
    quarantine: TaskQuarantine | None = None
    splits: Mapping[str, str] | None = None
    levels: Mapping[str, tuple[str, int]] | None = None  # task -> (parent, level) for checkpoint starts; level 0 = full task
    requires: Mapping[str, Mapping[str, Sequence[str]]] | None = None
    seed: int = 0
    tracker: DifficultyTracker = field(init=False)
    events: list[dict[str, Any]] = field(default_factory=list)
    current: dict[str, int] = field(default_factory=dict)  # parent -> its current curriculum level

    def __post_init__(self) -> None:
        held = sorted(t for t, s in (self.splits or {}).items() if str(s or "train") != "train")
        if held:
            raise ValueError(f"held-out tasks never train, so they are never sampled (V9): {held}")
        self.tasks = [str(t) for t in self.tasks]
        if not self.tasks:
            raise ValueError("no tasks to sample")
        self.tracker = DifficultyTracker(decay=self.config.decay, requires={t: {m: list(r) for m, r in req.items()} for t, req in (self.requires or {}).items()})
        self.rng = random.Random(self.seed)
        for t, (parent, lv) in (self.levels or {}).items():
            if t in self.tasks:
                self.current[parent] = max(self.current.get(parent, 0), int(lv))

    # ---- what may be sampled ----
    def _curriculum_ok(self, t: str) -> bool:
        if not self.levels or t not in self.levels:
            return True
        parent, lv = self.levels[t]
        return int(lv) == self.current.get(parent, int(lv))

    def _kept(self, t: str) -> bool:
        r = self.tracker.rates.get(t)
        if r is None or r.full.n < self.config.min_n:
            return True
        inside = r.in_band(self.config.band)
        return bool(inside) or r.phi_var > 0.0  # Φ spread: still teaches through step credit

    def weight(self, t: str) -> float:
        r = self.tracker.rates.get(t)
        return max(self.config.floor, r.signal() if r is not None else 1.0)

    def pick(self, start: int, step: int) -> int:
        """The index (into ``tasks``) of the next task. Same contract as ``TaskQuarantine.pick``: never stalls."""
        n = len(self.tasks)
        q = self.quarantine
        if q is not None:
            q.last_probe = False
        if q is not None and q.config.enabled:
            q._open_probations(step)
        open_ = [i for i in range(n) if self._curriculum_ok(self.tasks[i]) and not (q is not None and q.blocked(self.tasks[i], step))]
        if not open_:
            i = q.pick(start, step) if q is not None else start % n  # the quarantine's own "all quarantined" rule
            self._event("all_blocked", self.tasks[i], step, "every task is blocked by the curriculum or the quarantine")
            return i
        explore = self.rng.random() < self.config.explore
        pool = open_ if explore else [i for i in open_ if self._kept(self.tasks[i])]
        if not pool:
            pool = open_
            self._event("all_out_of_band", self.tasks[open_[0]], step, "no task's interval overlaps the band: sampling by weight anyway")
        ws = [self.weight(self.tasks[i]) for i in pool]
        i = self.rng.choices(pool, weights=ws, k=1)[0]
        if q is not None and q.config.enabled:
            s = q.state[self.tasks[i]]
            if s.probation:
                s.probe_out = True
                q.last_probe = True
        return i

    # ---- observations ----
    def observe(self, task: str, draws: Sequence[Draw], step: int, *, rewards: Sequence[float | None] = (), phis: Sequence[float | None] = (),
                picked: int | None = None, probe: bool | None = None) -> None:
        """A landed group of ``task`` (masked members are ``Draw(passed=None)``). The quarantine sees the group's rewards
        and Φ exactly as before (``picked`` / ``probe``: ``TaskQuarantine.observe``); the tracker sees every draw; the
        curriculum may move."""
        if self.quarantine is not None and rewards:
            self.quarantine.observe(task, rewards, phis, step, picked=picked, probe=probe)
        r = self.tracker.observe(task, step, draws)
        if self.levels and task in self.levels and r.full.n >= self.config.min_n:
            parent, lv = self.levels[task]
            lo, hi = r.interval()
            lower = [lv2 for t2, (p2, lv2) in self.levels.items() if p2 == parent and lv2 < lv and t2 in self.tasks]
            higher = [lv2 for t2, (p2, lv2) in self.levels.items() if p2 == parent and lv2 > lv and t2 in self.tasks]
            if lo >= self.config.promote and lower:
                self.current[parent] = max(lower)
                self._event("start_earlier", task, step, f"p̂ lower bound {lo:.2f} >= {self.config.promote}: level {lv} -> {max(lower)}", parent=parent)
            elif hi < self.config.demote and higher:
                self.current[parent] = min(higher)
                self._event("start_later", task, step, f"p̂ upper bound {hi:.2f} < {self.config.demote}: level {lv} -> {min(higher)}", parent=parent)

    def lost(self, task: str, *, probe: bool | None = None) -> None:
        if self.quarantine is not None:
            self.quarantine.lost(task, probe=probe)

    def _event(self, kind: str, task: str, step: int, why: str, **extra: Any) -> dict[str, Any]:
        e = {"event": kind, "task": task, "step": step, "reason": why, **extra}
        self.events.append(e)
        return e

    # ---- views and resume ----
    def metrics(self) -> dict[str, float]:
        rows = [r for r in self.tracker.rates.values() if r.full.n > 0]
        inband = [r for r in rows if r.in_band(self.config.band)]
        return {"difficulty/measured": float(len(rows)), "difficulty/in_band": float(len(inband)),
                "difficulty/mean_signal": (sum(r.signal() for r in rows) / len(rows)) if rows else 0.0}

    def to_json(self) -> dict[str, Any]:
        return {"tracker": self.tracker.to_json(), "current": dict(self.current), "rng": list(self.rng.getstate()[1]), "rng_v": self.rng.getstate()[0],
                "rng_g": self.rng.getstate()[2]}

    def load(self, doc: Mapping[str, Any] | None) -> DifficultySampler:
        if not doc:
            return self
        self.tracker.load(doc.get("tracker"))
        self.current.update({k: int(v) for k, v in (doc.get("current") or {}).items()})
        if doc.get("rng") is not None:
            self.rng.setstate((doc["rng_v"], tuple(doc["rng"]), doc.get("rng_g")))
        return self


def requires_of(milestones: Sequence[Any]) -> dict[str, list[str]]:
    """{milestone: its direct prerequisites} from milestone objects (``.id``, ``.requires``) or mappings (the window lane's)."""
    out: dict[str, list[str]] = {}
    for m in milestones:
        mid = m.get("id") if isinstance(m, Mapping) else getattr(m, "id", None)
        req = m.get("requires") if isinstance(m, Mapping) else getattr(m, "requires", None)
        if mid is not None:
            out[str(mid)] = [str(r) for r in req or ()]
    return out


def draws_of(records: Sequence[Any]) -> list[Draw]:
    """``Draw`` per rollout record (``core.schema.rollout.RolloutRecord``): masked when ``mask_sample`` or no reward; passed
    when every check of the grade holds (``gates``; without gates, reward 1); Φ from ``progress.phi_final``; the final
    milestone vector from ``progress.milestones`` when the grader put it there."""
    out = []
    for r in records:
        prog = getattr(r, "progress", None) or {}
        phi = prog.get("phi_final") if isinstance(prog.get("phi_final"), int | float) else None
        if getattr(r, "mask_sample", False) or getattr(r, "reward", None) is None:
            out.append(Draw(None, phi))
            continue
        gates = getattr(r, "gates", None) or {}
        passed = all(bool(v) for v in gates.values()) if gates else float(r.reward) >= 1.0 - 1e-9
        ms = prog.get("milestones") if isinstance(prog.get("milestones"), Mapping) else None
        out.append(Draw(bool(passed), phi, str(getattr(r, "stop", None) or "") or None, ms))
    return out
