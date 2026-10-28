"""The DeltaAI plumbing smoke: real RL steps through ``PrimeRLBackend`` on a synthetic task, no World (swarm-pipeline/14 §5).

A word-reversal prompt, ``group`` samples per prompt from vLLM, a graded reward (``difflib`` ratio of the answer to the
reversed word, so groups have spread), Dr.GRPO advantages per prompt, one ``TokenDatum`` per sample in the Tinker bridge's
shape, then the bridge's step: batch -> prime-rl trainer -> adapter broadcast -> vLLM hot-load. Checked per step: the
trainer's step index, that the new adapter is served (a greedy probe under the new version, its logprob sum compared with
the previous version's), and that the checkpoint exists. A second call with ``resume`` (the trainer relaunched from that
checkpoint) continues at the next step index. Nothing here is a Crucible task: it proves the GPU path only.
"""

from __future__ import annotations

import difflib
import random
import statistics
from collections.abc import Callable, Sequence
from typing import Any

from crucible.train.bridges.tinker.backend import TokenDatum

WORDS = ("crucible", "verifier", "milestone", "gradient", "sandbox", "compaction", "potential", "rollout", "adapter", "benchmark")


def reward(word: str, answer: str) -> float:
    a = "".join(ch for ch in answer.strip().lower() if ch.isalpha())
    return difflib.SequenceMatcher(None, a, word[::-1]).ratio()


def datum(prompt: Sequence[int], completion: Sequence[int], logprobs: Sequence[float], adv: float, **meta: Any) -> TokenDatum:
    full = list(prompt) + list(completion)
    p = len(prompt)
    mask = [0] * (p - 1) + [1] * len(completion)
    return TokenDatum(input_ids=full[:-1], target_ids=full[1:], logprobs=[0.0] * (p - 1) + [float(x) for x in logprobs],
                      advantages=[0.0] * (p - 1) + [float(adv)] * len(completion), mask=mask, meta=dict(meta))


async def run(backend: Any, *, encode: Callable[[str], list[int]], decode: Callable[[Sequence[int]], str], model: str, steps: int, prompts: int = 4,
              group: int = 4, max_tokens: int = 32, learning_rate: float, loss: str = "importance_sampling", rank: int = 16,
              resume: str | None = None, sampler_path: str | None = None, seed: int = 0) -> dict[str, Any]:
    rng = random.Random(seed)
    trainer = await backend.trainer(model, rank=rank, resume=resume, seed=seed)
    sampler = await backend.sampler(model, sampler_path)
    probe = encode(f"Reverse the letters of the word '{WORDS[0]}'. Answer with the reversed word only.")
    prev = await sampler.sample(probe, max_tokens=max_tokens, temperature=0.0, stop=None, seed=seed)
    rows = []
    for s in range(steps):
        datums, rewards = [], []
        for i in range(prompts):
            w = rng.choice(WORDS)
            ids = encode(f"Reverse the letters of the word '{w}'. Answer with the reversed word only.")
            outs = [await sampler.sample(ids, max_tokens=max_tokens, temperature=1.0, stop=None, seed=seed + 1000 * s + 10 * i + g) for g in range(group)]
            rs = [reward(w, decode(o.tokens)) for o in outs]
            mean = statistics.fmean(rs)
            rewards.extend(rs)
            datums += [datum(ids, o.tokens, o.logprobs, r - mean, rollout=f"s{s}p{i}g{g}", segment=0) for g, (o, r) in enumerate(zip(outs, rs, strict=True))]
        fb = await trainer.forward_backward(datums, loss_fn=loss, config=None)
        await trainer.optim_step(learning_rate=learning_rate)
        path = await trainer.save_for_sampler(f"smoke-{s}", ttl_s=None)
        sampler = await backend.sampler(model, path)
        now = await sampler.sample(probe, max_tokens=max_tokens, temperature=0.0, stop=None, seed=seed)
        ckpt = await trainer.save_state(f"smoke-state-{s}", ttl_s=None)
        rows.append({"trainer_step": int(fb.metrics.get("primerl/step", -1)), "reward_mean": statistics.fmean(rewards), "datums": len(datums),
                     "trained_tokens": sum(d.n_train for d in datums), "adapter": path, "served_as": getattr(sampler, "name", None), "checkpoint": ckpt,
                     "probe_logprob_prev": sum(prev.logprobs), "probe_logprob_now": sum(now.logprobs), "probe_changed": now.tokens != prev.tokens
                     or abs(sum(now.logprobs) - sum(prev.logprobs)) > 1e-6, "metrics": fb.metrics})
        prev = now
    return {"steps": rows, "resumed_from": resume, "last_checkpoint": rows[-1]["checkpoint"] if rows else None,
            "last_adapter": rows[-1]["adapter"] if rows else None}
