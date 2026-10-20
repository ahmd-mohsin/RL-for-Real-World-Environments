"""An in-memory Tinker (``backend.Backend``) for tests and dry runs: no network, no key, no spend.

- **Sampling** is a ``responder(prompt_ids) -> (ids, logprobs) | ids``; ``ScriptedResponder`` replays the P6 scripted
  steps (``train.bridges.slime.episode.ScriptedSteps``: write, shell, final, chaos) as Hermes tool calls encoded by a
  codec, one script per episode, the episode found by token prefix (TITO guarantees a later call's prompt extends the
  earlier prompt + completion). ``cached_prompt_tokens`` reports the longest prefix this fake has already seen, the way a
  prefix cache would.
- **Training** computes the documented losses (Tinker docs, losses: ``importance_sampling`` -sum(ratio * A), ``ppo``
  -sum(min(ratio * A, clip(ratio, lo, hi) * A)), ``cispo`` -sum(sg(clip(ratio)) * logp * A)) with the learner's logprobs =
  the sampling logprobs + ``drift`` (0: on-policy, ratio 1), and keeps every call in ``calls``.
- **Checkpoints** are ``tinker://fake-<run>/{sampler_weights,weights}/<name>`` paths in a ``store`` dict that may be
  shared between instances (a resumed process); ``trainer(resume=)`` restores the optimizer-step count from it.
- ``fail`` queues exceptions raised by the next sample calls (an outage, a 400).
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.train.bridges.tinker.backend import FBOut, SampleOut, TokenDatum


@dataclass
class FakeTinker:
    responder: Callable[[list[int]], Any] | None = None
    store: dict[str, Any] = field(default_factory=dict)
    run: str = "run"
    drift: float = 0.0
    fail: list[BaseException] = field(default_factory=list)
    delay_s: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    async def sampler(self, model: str, path: str | None = None) -> _FakeSampler:
        if path is not None and path not in self.store.setdefault("sampler_paths", set()):
            raise KeyError(f"no sampler weights at {path}")
        self.calls.append({"op": "sampler", "model": model, "path": path})
        return _FakeSampler(self, model, path)

    async def trainer(self, model: str, *, rank: int, resume: str | None = None, seed: int | None = None) -> _FakeTrainer:
        st = {"step": 0}
        if resume is not None:
            if resume not in self.store.get("states", {}):
                raise KeyError(f"no training state at {resume}")
            st = dict(self.store["states"][resume])
        self.calls.append({"op": "trainer", "model": model, "rank": rank, "resume": resume})
        return _FakeTrainer(self, model, st)

    async def archive_url(self, path: str) -> str:
        if path not in self.store.get("sampler_paths", set()):
            raise KeyError(path)
        return f"https://fake.invalid/archive/{path.rsplit('/', 1)[-1]}.tar"


class _FakeSampler:
    def __init__(self, fake: FakeTinker, model: str, path: str | None):
        self.fake, self.model, self.path = fake, model, path

    async def sample(self, prompt_ids: Sequence[int], *, max_tokens: int, temperature: float, stop: Sequence[int] | None, seed: int | None) -> SampleOut:
        f = self.fake
        if f.delay_s:
            await asyncio.sleep(f.delay_s)
        if f.fail:
            raise f.fail.pop(0)
        if f.responder is None:
            raise RuntimeError("FakeTinker has no responder")
        out = f.responder(list(prompt_ids))
        if asyncio.iscoroutine(out):
            out = await out
        ids, lps = (out if isinstance(out, tuple) else (out, [-0.25] * len(out)))
        ids, lps = list(ids)[:max_tokens], list(lps)[:max_tokens]
        seen = f.store.setdefault("prefixes", [])
        cached = max((n for p, n in seen if list(prompt_ids[:n]) == p), default=0)
        seen.append((list(prompt_ids), len(prompt_ids)))
        del seen[:-256]
        stop_reason = "stop" if stop is not None and ids and ids[-1] in set(stop) else ("length" if len(ids) >= max_tokens else "stop")
        f.calls.append({"op": "sample", "path": self.path, "prompt": len(prompt_ids), "completion": len(ids), "temperature": temperature})
        return SampleOut(tokens=ids, logprobs=lps, stop_reason=stop_reason, cached_prompt_tokens=cached)


def fake_loss(d: TokenDatum, target_logprobs: Sequence[float], loss_fn: str, config: dict[str, float] | None = None) -> float:
    """The documented Tinker losses on one datum (sum over tokens)."""
    cfg = dict(config or {})
    lo, hi = cfg.get("clip_low_threshold", 0.8), cfg.get("clip_high_threshold", 1.2)
    total = 0.0
    for lp_t, lp_s, a in zip(target_logprobs, d.logprobs, d.advantages, strict=True):
        ratio = math.exp(lp_t - lp_s)
        if loss_fn == "importance_sampling":
            total -= ratio * a
        elif loss_fn == "ppo":
            total -= min(ratio * a, min(max(ratio, lo), hi) * a)
        elif loss_fn == "cispo":
            total -= min(max(ratio, lo), hi) * lp_t * a
        else:
            raise ValueError(f"fake Tinker has no loss {loss_fn!r}")
    return total


class _FakeTrainer:
    def __init__(self, fake: FakeTinker, model: str, state: dict[str, Any]):
        self.fake, self.model, self.state = fake, model, state

    async def forward_backward(self, datums: Sequence[TokenDatum], *, loss_fn: str, config: dict[str, float] | None) -> FBOut:
        tl = [[lp + self.fake.drift for lp in d.logprobs] for d in datums]
        loss = sum(fake_loss(d, t, loss_fn, config) for d, t in zip(datums, tl, strict=True))
        self.fake.calls.append({"op": "forward_backward", "loss_fn": loss_fn, "config": config, "datums": [d for d in datums], "loss": loss,
                                "tokens": sum(len(d.input_ids) for d in datums)})
        return FBOut(target_logprobs=tl, metrics={"loss:sum": loss})

    async def optim_step(self, *, learning_rate: float) -> dict[str, float]:
        self.state["step"] += 1
        self.fake.calls.append({"op": "optim_step", "lr": learning_rate, "step": self.state["step"]})
        return {"step": float(self.state["step"])}

    async def save_for_sampler(self, name: str, *, ttl_s: int | None) -> str:
        p = f"tinker://fake-{self.fake.run}/sampler_weights/{name}"
        self.fake.store.setdefault("sampler_paths", set()).add(p)
        self.fake.calls.append({"op": "save_for_sampler", "path": p})
        return p

    async def save_state(self, name: str, *, ttl_s: int | None) -> str:
        p = f"tinker://fake-{self.fake.run}/weights/{name}"
        self.fake.store.setdefault("states", {})[p] = dict(self.state)
        self.fake.calls.append({"op": "save_state", "path": p, "step": self.state["step"]})
        return p


class ScriptedResponder:
    """Replays scripted steps as completions: ``scripts`` is one step list per episode (cycled), each step rendered as the
    text a Hermes tool-calling model would sample (``codec.encode_text`` + its EOS)."""

    def __init__(self, scripts: Sequence[Sequence[tuple]], codec: Any, *, app: str = "repo", logprob: float = -0.25):
        self.scripts, self.codec, self.app, self.logprob = [list(s) for s in scripts], codec, app, logprob
        self._eps: list[tuple[list[int], Any]] = []  # (prompt + completion so far, cursor)
        self.started = 0

    def _encode(self, text: str) -> list[int]:
        enc = getattr(self.codec, "encode_text", None)
        return (list(enc(text)) if callable(enc) else list(self.codec.tok.encode(text, add_special_tokens=False))) + [self.codec.eos_ids[0]]

    def text_of(self, step: tuple) -> str:
        if step[0] == "write":
            return f"<tool_call>{json.dumps({'name': f'{self.app}__write_file', 'arguments': {'path': step[1], 'content': step[2]}})}</tool_call>"
        if step[0] == "shell":
            return f"<tool_call>{json.dumps({'name': f'{self.app}__shell', 'arguments': {'command': step[1]}})}</tool_call>"
        if step[0] == "final":
            return step[1]
        raise ValueError(f"unknown step {step[0]!r}")

    async def __call__(self, prompt: list[int]) -> tuple[list[int], list[float]]:
        from crucible.train.bridges.slime.episode import ScriptedSteps

        hit = [i for i, (ids, _) in enumerate(self._eps) if prompt[: len(ids)] == ids]
        if hit:
            i = max(hit, key=lambda j: len(self._eps[j][0]))
            cur = self._eps[i][1]
        else:
            cur = ScriptedSteps(self.scripts[self.started % len(self.scripts)])
            self.started += 1
            self._eps.append(([-1], cur))  # matches no prompt until this call records its completion
            i = len(self._eps) - 1
        while cur.i < len(cur.steps) and cur.steps[cur.i][0] == "chaos":
            out = cur.steps[cur.i][1]()
            if asyncio.iscoroutine(out):
                await out
            cur.i += 1
        if cur.i >= len(cur.steps):
            text = "done"
        else:
            text = self.text_of(cur.steps[cur.i])
            cur.i += 1
        ids = self._encode(text)
        self._eps[i] = (list(prompt) + ids, cur)
        return ids, [self.logprob] * len(ids)
