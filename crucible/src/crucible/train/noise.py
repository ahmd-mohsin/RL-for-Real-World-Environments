"""Noise-robust curriculum (M12-infra D; art. §3.8, LongCat: instruction and tool noise injected during RL, with the
intensity raised as the policy becomes robust to it).

``NoiseCurriculum`` owns one knob, the intensity ∈ [0, 1], and a control group: a deterministic ``clean_frac`` of
rollouts (by a hash of the rollout's key) runs clean. From the trained records it measures the clean / noisy reward
gap over a window and moves the intensity:

- gap ≤ ``raise_below``: the policy shrugs off the current noise → intensity + ``step``;
- gap ≥ ``lower_above``: the noise is drowning the signal → intensity − ``step``;
- otherwise held. Every change is in ``history`` (step, intensity, clean, noisy, gap).

It is a ``train.sinks`` sink (``RuntimeConfig.sinks=[curriculum]``): ``record("rollout", recs)`` reads each record's
``noise/intensity`` and ``noise/clean`` reward components, which the wrapped environments write. Two wrappers:

- ``token_env(factory)``: token environments (``reset(prompt, seed)`` / ``step(completion)``): observation noise
  (inserted distractor tokens, dropped tokens, substitutions; ``protect`` tokens are never touched) at the
  curriculum's intensity, and the intensity in the final ``Step.info`` (the runtime merges it into the record);
- ``world_noise(key, cfg)``: a ``world.noise.WorldNoise`` at the assigned intensity for one World episode (pass it to
  ``World(noise=...)``; record its level with ``components(key)``).
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from crucible.train.envs import Step


@dataclass
class TokenNoise:
    insert: float = 0.15  # per observation token: a distractor token is inserted after it
    drop: float = 0.0  # per observation token: dropped (semantics-changing: off by default)
    substitute: float = 0.0  # per observation token: replaced by a random token (semantics-changing: off by default)
    vocab: tuple[int, int] = (4, 31)  # inclusive range distractors are drawn from
    protect: frozenset[int] = field(default_factory=frozenset)  # tokens never dropped or substituted (control tokens)
    first_obs: bool = True  # perturb the first observation (the instruction) too


class NoiseCurriculum:
    def __init__(self, *, start: float = 0.0, step: float = 0.1, max_intensity: float = 1.0, clean_frac: float = 0.25, window: int = 64,
                 raise_below: float = 0.05, lower_above: float = 0.2, seed: Any = 0):
        if not 0.0 < clean_frac < 1.0:
            raise ValueError("clean_frac must be in (0, 1): the clean control group measures the gap")
        self.intensity, self.step_size, self.max = float(start), step, max_intensity
        self.clean_frac, self.window, self.raise_below, self.lower_above, self.seed = clean_frac, window, raise_below, lower_above, seed
        self.clean: deque[float] = deque(maxlen=window)
        self.noisy: deque[float] = deque(maxlen=window)
        self.history: list[dict[str, float]] = []
        self._assigned: dict[str, float] = {}
        self._noisy_level: deque[float] = deque(maxlen=window)

    # ---- assignment ---------------------------------------------------------------------------------------------
    def is_clean(self, key: str) -> bool:
        h = int(hashlib.sha256(f"{self.seed}|{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return h < self.clean_frac

    def assign(self, key: str) -> float:
        """The intensity for one rollout: 0 for the clean control group, else the current intensity (fixed at assignment)."""
        if key not in self._assigned:
            self._assigned[key] = 0.0 if self.is_clean(key) else self.intensity
            if len(self._assigned) > 100_000:  # bounded: the oldest assignments are long trained
                for k in list(self._assigned)[:50_000]:
                    del self._assigned[k]
        return self._assigned[key]

    def components(self, key: str) -> dict[str, float]:
        return {"noise/intensity": self.assign(key), "noise/clean": 1.0 if self.is_clean(key) else 0.0}

    # ---- feedback -------------------------------------------------------------------------------------------------
    def observe(self, reward: float | None, *, clean: bool, level: float) -> None:
        if reward is None:
            return
        if clean:
            self.clean.append(float(reward))
        elif abs(level - self.intensity) < 1e-9:  # assigned at the current level (a rollout from before a change is stale evidence)
            self.noisy.append(float(reward))
            self._noisy_level.append(level)
        if len(self.clean) >= self.window and len(self.noisy) >= self.window:
            c, n = sum(self.clean) / len(self.clean), sum(self.noisy) / len(self.noisy)
            gap, old = c - n, self.intensity
            if gap <= self.raise_below:
                self.intensity = min(self.max, self.intensity + self.step_size)
            elif gap >= self.lower_above:
                self.intensity = max(0.0, self.intensity - self.step_size)
            self.history.append({"intensity": self.intensity, "previous": old, "clean": c, "noisy": n, "gap": gap,
                                 "noisy_level": sum(self._noisy_level) / len(self._noisy_level)})
            self.clean.clear()
            self.noisy.clear()
            self._noisy_level.clear()

    def observe_records(self, records: Sequence[Any]) -> None:
        for r in records:
            rc = getattr(r, "reward_components", None) or {}
            if "noise/clean" in rc:
                self.observe(r.reward, clean=rc["noise/clean"] > 0, level=float(rc.get("noise/intensity", 0.0)))

    # ---- sink protocol (``RuntimeConfig.sinks``) ------------------------------------------------------------------
    def log(self, metrics: Mapping[str, float], step: int) -> None:
        return None

    def record(self, kind: str, data: Any, step: int) -> None:
        if kind == "rollout":
            self.observe_records(data if isinstance(data, list | tuple) else [data])

    def summary(self, data: Mapping[str, Any]) -> None:
        return None

    def finish(self, status: str) -> None:
        return None

    # ---- wrappers ---------------------------------------------------------------------------------------------------
    def token_env(self, factory: Callable[[], Any], spec: TokenNoise | None = None) -> Callable[[], Any]:
        spec = spec or TokenNoise()
        return lambda: NoisyTokenEnv(factory(), self, spec)

    def world_noise(self, key: str, cfg: Any = None):
        from crucible.world.noise import NoiseConfig, WorldNoise

        return WorldNoise(replace(cfg or NoiseConfig(), intensity=self.assign(key)), seed=key)


class NoisyTokenEnv:
    """A token environment seen through observation noise at the curriculum's assigned intensity."""

    def __init__(self, env: Any, cur: NoiseCurriculum, spec: TokenNoise):
        self.env, self.cur, self.spec = env, cur, spec
        self.level, self.key, self.rng, self.ops = 0.0, "", random.Random(0), 0

    def __getattr__(self, k: str) -> Any:
        return getattr(self.env, k)

    def _perturb(self, ids: list[int]) -> list[int]:
        s, p = self.spec, self.level
        if p == 0:
            return ids
        out: list[int] = []
        for t in ids:
            prot = t in s.protect
            r = self.rng.random()
            if not prot and r < s.drop * p:
                self.ops += 1
                continue
            if not prot and r < (s.drop + s.substitute) * p:
                out.append(self.rng.randint(*s.vocab))
                self.ops += 1
            else:
                out.append(t)
            if self.rng.random() < s.insert * p:
                out.append(self.rng.randint(*s.vocab))
                self.ops += 1
        return out

    async def reset(self, prompt: str, seed: int) -> list[int]:
        self.key = f"{prompt}|{seed}"
        self.level, self.rng, self.ops = self.cur.assign(self.key), random.Random(f"tokennoise|{self.key}"), 0
        obs = await self.env.reset(prompt, seed)
        return self._perturb(list(obs)) if self.spec.first_obs else list(obs)

    async def step(self, completion: list[int]) -> Step:
        st = await self.env.step(completion)
        obs = self._perturb(list(st.obs_ids))
        info = dict(st.info)
        if st.done:
            info["reward_components"] = {**info.get("reward_components", {}), **self.cur.components(self.key), "noise/ops": float(self.ops)}
        return Step(obs, st.done, st.reward, info)
