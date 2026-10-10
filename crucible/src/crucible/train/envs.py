"""Token-level environments for the M12 runtime (the trainer-facing env contract; Worlds plug in through the M10
harness + capture adapter, ``bridges``). An environment turns a model completion into the next observation tokens.

    reset(prompt, seed) -> obs_ids            first observation (the prompt)
    step(completion_ids) -> Step(obs_ids, done, reward, info)
    raise EnvInfraError on an infrastructure failure (the rollout is masked, never scored 0)

``CopyTask``: each turn the env shows a target token; the policy must answer it and stop. Reward = share of turns
answered correctly (outcome in [0, 1]). ``latency`` and ``long_tail`` make env steps take real (asyncio) time with a
heavy tail, so the runtime's overlap, aborts and staleness are exercised for real. ``infra_fail`` injects failures.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field


class EnvInfraError(RuntimeError):
    pass


@dataclass
class Step:
    obs_ids: list[int]
    done: bool
    reward: float | None = None
    info: dict = field(default_factory=dict)


class CopyTask:
    VOCAB, BOS, SEP, STOP = 32, 1, 2, 3  # tokens 4..31 are symbols

    def __init__(self, *, turns: int = 3, symbols: int = 8, latency_s: float = 0.0, long_tail: float = 0.0, tail_mult: float = 10.0, infra_fail: float = 0.0):
        self.turns, self.symbols, self.latency_s, self.long_tail, self.tail_mult, self.infra_fail = turns, symbols, latency_s, long_tail, tail_mult, infra_fail
        self.rng = random.Random(0)
        self.targets: list[int] = []
        self.k = 0
        self.correct = 0

    async def _wait(self) -> None:
        if self.latency_s:
            d = self.latency_s * (self.tail_mult if self.rng.random() < self.long_tail else 1.0)
            await asyncio.sleep(d)

    async def reset(self, prompt: str, seed: int) -> list[int]:
        self.rng = random.Random(f"{prompt}|{seed}")
        r = random.Random(prompt)  # the prompt fixes the targets: a group shares them
        self.targets = [r.randint(4, 3 + self.symbols) for _ in range(self.turns)]
        self.k, self.correct = 0, 0
        await self._wait()
        return [self.BOS, self.targets[0], self.SEP]

    async def step(self, completion: list[int]) -> Step:
        await self._wait()
        if self.infra_fail and self.rng.random() < self.infra_fail:
            raise EnvInfraError("sandbox lost")
        self.correct += int(bool(completion) and completion[0] == self.targets[self.k])  # the first answer token must be the target
        self.k += 1
        if self.k >= self.turns:
            return Step([], True, self.correct / self.turns)
        return Step([self.SEP, self.targets[self.k], self.SEP], False)
