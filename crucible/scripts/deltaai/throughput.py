"""Throughput of a running vLLM server for episode-shaped requests (doc 14-deltaai §5): ``--concurrency`` requests at
once, each a ``--prompt`` token prompt (random ids) and ``--max-tokens`` sampled, through ``VLLMBackend`` (the episodes'
own client). Prints one JSON line: aggregate sampled tokens/s, prefill tokens/s, request latency.

    python throughput.py --url http://127.0.0.1:8000 --model Qwen/Qwen3.6-35B-A3B --concurrency 16 --prompt 32000 --max-tokens 512
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

from crucible.train.bridges.vllm.client import VLLMBackend


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--prompt", type=int, default=32_000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    be = VLLMBackend(a.url)
    s = await be.sampler(a.model)
    rng = random.Random(0)
    lat, sampled, prefill = [], 0, 0
    t0 = time.monotonic()
    for _ in range(a.rounds):
        async def one() -> None:
            nonlocal sampled, prefill
            ids = [rng.randrange(1000, 60000) for _ in range(a.prompt)]
            t = time.monotonic()
            out = await s.sample(ids, max_tokens=a.max_tokens, temperature=1.0, stop=None, seed=rng.randrange(1 << 30))
            lat.append(time.monotonic() - t)
            sampled += len(out.tokens)
            prefill += a.prompt
        await asyncio.gather(*(one() for _ in range(a.concurrency)))
    dt = time.monotonic() - t0
    print(json.dumps({"concurrency": a.concurrency, "prompt_tokens": a.prompt, "max_tokens": a.max_tokens, "requests": len(lat), "wall_s": round(dt, 1),
                      "sampled_tok_s": round(sampled / dt, 1), "prefill_tok_s": round(prefill / dt, 1), "latency_p50_s": round(statistics.median(lat), 1),
                      "latency_max_s": round(max(lat), 1)}))
    await be.close()


if __name__ == "__main__":
    asyncio.run(main())
