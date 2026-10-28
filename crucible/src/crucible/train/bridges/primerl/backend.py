"""prime-rl's GPU trainer and vLLM server behind the Tinker bridge's ``Backend`` protocol (DeltaAI; swarm-pipeline/14).

The Tinker loop (``bridges/tinker/trainer.TinkerTrainer``) keeps everything Crucible owns: groups run by Crucible's own
runtime (native harness, checkpoint sidecar, path gates, honesty, compaction), the filters (verifier version, infra
groups dropped whole, staleness, zero spread), quarantine and the difficulty sampler, the advantages
(``datums.build_datums``: Dr.GRPO outcome, honesty in the advantage, M6 per-turn credit), resume and the dashboards. Only
the two remote calls change: sampling and the optimizer step go to processes prime-rl runs on the GPU nodes.

- **Sampling** (``sampler(model, path)``): vLLM's OpenAI ``/v1/completions`` on the inference server prime-rl launches
  (``inference.server.port``), token ids in (``prompt`` as a list of ids), token ids and logprobs out
  (``return_tokens_as_token_ids``, ``logprobs=0``), ``model`` = the LoRA adapter name of that weight version (or the base
  model before the first step). Errors go through the Tinker bridge's infra/task split (``backend.with_retries``).
  With routing replay or score centering on, sampling goes through prime-rl's ``/inference/v1/generate`` instead, which
  also returns the routed experts (``routing``) and, with ``logprobs = k``, the sampler's top-k head (``topk``,
  ``topk_server``); both are kept per call and stitched per training sequence. ``topk_cost`` measures the head's cost.
- **The optimizer step** (``trainer().forward_backward``): the batch's datums become prime-rl ``TrainingSample``s
  (``wire``), packed by prime-rl's own packer into ``<out>/batches/step_<s>/rank_<i>.bin`` (its filesystem rollout
  transport); the prime-rl trainer (FSDP2, expert / context parallel, LoRA; launched by ``deploy``) trains on it with
  Crucible's loss (``loss.crucible_loss``) and broadcasts the adapter to ``<out>/broadcasts/step_<s>``; the bridge answers
  its handshake (``.sender_ready`` -> ``.receiver_ready``; prime-rl ``transports/weights/base.py:17-26``), waits for
  ``.finished``, and hot-loads the adapter into vLLM under a per-version name (``/load_lora_adapter``), so groups still
  sampling from an older version keep their adapter (the staleness bound stays the trainer's). ``optim_step`` is a
  no-op (prime-rl's step already ran; the learning rate is the trainer's configuration and is checked equal).
- **Checkpoints**: ``save_state`` is prime-rl's own checkpoint of that step (``<out>/checkpoints/step_<s>``, weights +
  optimizer + scheduler; ``ckpt.interval = 1``); a resumed run relaunches the trainer with ``resume.step = s`` and the
  bridge continues at batch step s + 1. Step numbering (prime-rl v0.10.0, ``trainer/ckpt.Progress.step = 1``): a fresh
  trainer's first step is 1 (``FIRST_STEP``), and before it the trainer broadcasts its startup policy ``v{first - 1}`` to
  ``broadcasts/step_0``. Step s reads ``batches/step_<s>``, then writes ``broadcasts/step_<s>`` and
  ``checkpoints/step_<s>``. So Crucible's version v (after v optimizer steps) is prime-rl's step v. Smoke attempt 8
  (job 3473175) waited for ``step_-1``, on the earlier belief that the first step was 0, and timed out.

The process boundary is ``io`` (``FileIO``: the shared filesystem; a fake in the tests) and ``http`` (``HttpClient``:
httpx), so the protocol is tested without GPUs. Nothing here prices tokens: ``local_model`` is the student's row with zero
prices and the context the trainer was launched with (GPU hours are the cost, booked by Slurm).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from crucible.train.bridges.primerl import topk as topk_mod
from crucible.train.bridges.primerl import wire
from crucible.train.bridges.primerl.routing import RoutingStore, decode
from crucible.train.bridges.primerl.topk import TopKStore
from crucible.train.bridges.tinker.backend import FBOut, SampleOut, TokenDatum, with_retries
from crucible.train.bridges.tinker.models import Prices, TinkerModel, resolve

SENDER_READY, RECEIVER_READY, FINISHED = ".sender_ready", ".receiver_ready", ".finished"
log = logging.getLogger(__name__)

# the rows prime-rl's trainer logs per step (``trainer/rl/train.py``, v0.10.0): its tensor stats (``loss/mean``) and the
# optimizer's (``optim/grad_norm``); both are written after the step's checkpoint
TRAINER_METRIC_KEYS = ("loss/mean", "optim/grad_norm")
FIRST_STEP = 1  # a fresh prime-rl trainer's first step (``Progress.step``, v0.10.0); its startup broadcast is step_0
# The trainer scores every token under softmax(z / T) over the whole vocabulary, so the sampler must sample from exactly
# that: no top-k / top-p / min-p truncation and no penalties. Sent explicitly on every call, because vLLM's
# /v1/completions fills unset fields from the model's generation_config.json (Qwen3.6-35B-A3B ships top_k = 20,
# top_p = 0.95), and prime-rl's sampling-mask replay, the trainer's only way to score a truncated sampler, carries the
# score-centering heads here (``topk``). ``deploy.inference_toml`` also sets ``generation_config = "vllm"``.
NEUTRAL_SAMPLING = {"top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def local_model(name: str, *, context: int) -> TinkerModel:
    """The student's model row with zero prices and the trainer's ``seq_len`` as its context (module doc)."""
    return replace(resolve(name), prices=Prices(0.0, 0.0, 0.0, 0.0), list_prices=None, context=int(context))


class HttpFailure(Exception):
    """An HTTP error with the status the Tinker bridge's ``classify`` reads."""

    def __init__(self, status_code: int | None, text: str):
        super().__init__(f"HTTP {status_code}: {text[:300]}")
        self.status_code = status_code


class Http(Protocol):
    async def post(self, path: str, body: dict[str, Any], *, timeout_s: float) -> dict[str, Any]: ...


@dataclass
class HttpClient:
    """httpx over the vLLM server (module doc)."""

    base_url: str

    async def post(self, path: str, body: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s) as c:
            r = await c.post(path, json=body)
        if r.status_code >= 400:
            raise HttpFailure(r.status_code, r.text)
        return r.json() if r.content else {}


class IO(Protocol):
    async def send(self, step: int, rows: Sequence[dict[str, Any]]) -> None: ...

    async def wait_broadcast(self, step: int, *, timeout_s: float) -> Path: ...

    def metrics(self, step: int) -> dict[str, float]: ...

    def checkpoint(self, step: int) -> Path | None: ...


@dataclass
class FileIO:
    """The shared filesystem between the bridge and the prime-rl trainer (``output_dir`` of the trainer config)."""

    output_dir: Path
    ranks: int
    seq_len: int
    model: str
    poll_s: float = 1.0
    simple_pack: bool = False  # tests and dry runs only: prime-rl's packer otherwise
    send_timeout_s: float = 6 * 3600.0  # prime-rl's sender waits for every trainer rank's request with no bound of its own

    async def send(self, step: int, rows: Sequence[dict[str, Any]]) -> None:
        if self.simple_pack:
            d = self.output_dir / "batches" / f"step_{step}"
            d.mkdir(parents=True, exist_ok=True)
            for i, mbs in enumerate(wire.simple_pack(rows, ranks=self.ranks)):
                (d / f"rank_{i}.json").write_text(json.dumps(mbs))
            return
        from prime_rl.transports.batch.filesystem import FileSystemBatchSender  # type: ignore[import-not-found]

        grid = wire.pack(rows, seq_len=self.seq_len, ranks=self.ranks, model=self.model)
        # prime-rl's own sender: msgpack per rank, written to a temporary name and renamed (filesystem.py:17-35)
        try:
            await asyncio.wait_for(FileSystemBatchSender(self.output_dir, self.ranks, current_step=step).send(grid), self.send_timeout_s)
        except TimeoutError:
            raise TimeoutError(f"the prime-rl trainer requested no batch for step {step} in {self.send_timeout_s:.0f}s") from None

    async def wait_broadcast(self, step: int, *, timeout_s: float) -> Path:
        """Answer the trainer's offer of ``step`` and wait until its weights are on disk. Only an unanswered offer counts
        (``.sender_ready`` without ``.receiver_ready``): a relaunched trainer re-offers its resume step in a directory that
        still holds the previous trainer's answered markers until it deletes and recreates it (smoke attempt 11, job
        3474615: the bridge took the stale markers, and the new trainer waited for a receiver that had gone)."""
        d = self.output_dir / "broadcasts" / f"step_{step}"
        t0 = time.monotonic()
        while True:
            if (d / SENDER_READY).exists() and not (d / RECEIVER_READY).exists():
                try:
                    (d / RECEIVER_READY).touch()  # unblocks the trainer's broadcast
                    break
                except FileNotFoundError:  # the trainer was recreating the directory
                    pass
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"prime-rl trainer published no weights for step {step} in {timeout_s:.0f}s ({d})")
            await asyncio.sleep(self.poll_s)
        while not (d / FINISHED).exists():
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"prime-rl weights for step {step} not finished in {timeout_s:.0f}s ({d})")
            await asyncio.sleep(self.poll_s)
        return d

    def metrics(self, step: int) -> dict[str, float]:
        """The trainer's logged numbers for ``step`` from prime-rl's file monitor (best effort: {} when absent)."""
        out: dict[str, float] = {}
        root = self.output_dir / "monitors" / "file"
        for p in sorted(root.rglob("*.jsonl")) if root.exists() else []:
            for line in p.read_text(errors="replace").splitlines():  # best effort: a compressed episode chunk is skipped, never fatal
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("step") == step:
                    out.update({f"primerl/{k}": float(v) for k, v in row.items() if k != "step" and isinstance(v, int | float)})
        return out

    def checkpoint(self, step: int) -> Path | None:
        """The step's checkpoint once it is complete: prime-rl creates ``checkpoints/step_<s>`` before torch DCP saves into
        ``trainer/``, and DCP writes ``.metadata`` last. A directory alone may be half written, and a trainer stopped then
        (the smoke stops it after phase 1) would leave a checkpoint that cannot resume."""
        d = self.output_dir / "checkpoints" / f"step_{step}"
        return d if (d / "trainer" / ".metadata").exists() else None


@dataclass
class _Sampler:
    http: Http
    name: str
    path: str | None
    stats: dict[str, int]
    timeout_s: float = 3600.0
    routing: RoutingStore | None = None  # router replay: the generate endpoint, its routes kept per call (``routing``)
    topk: TopKStore | None = None  # score centering: the generate endpoint, the sampler's top-k head kept per call (``topk``)
    train_temperature: float | None = None  # the trainer's temperature: only calls sampled at it keep their heads

    async def sample(self, prompt_ids: Sequence[int], *, max_tokens: int, temperature: float, stop: Sequence[int] | None, seed: int | None) -> SampleOut:
        # vLLM reads a list of strings as a batch of text prompts and answers 200, so a mis-tokenised prompt would only
        # fail later, in the trainer's batch decode (smoke attempt 9: transformers 5 returned BatchEncoding keys)
        if not all(isinstance(t, int) and not isinstance(t, bool) for t in prompt_ids):
            raise TypeError(f"prompt token ids must be ints, got {list(prompt_ids)[:3]}")
        if self.routing is not None or self.topk is not None:
            return await self._generate(prompt_ids, max_tokens=max_tokens, temperature=temperature, stop=stop, seed=seed)
        body: dict[str, Any] = {"model": self.name, "prompt": list(prompt_ids), "max_tokens": int(max_tokens), "temperature": float(temperature),
                                "logprobs": 0, "return_tokens_as_token_ids": True, "skip_special_tokens": False, "stop_token_ids": list(stop or []),
                                **NEUTRAL_SAMPLING}
        if seed is not None:
            body["seed"] = int(seed)
        out = await with_retries("vllm completions", lambda: self.http.post("/v1/completions", body, timeout_s=self.timeout_s), stats=self.stats)
        return parse_completion(out)

    async def _generate(self, prompt_ids: Sequence[int], *, max_tokens: int, temperature: float, stop: Sequence[int] | None, seed: int | None) -> SampleOut:
        """vLLM's token-in/token-out ``/inference/v1/generate`` on prime-rl's server: the routed experts and (with
        ``topk``) the sampler's top-k head per token come back with the completion (``topk_server``). The prefix cache is
        salted per episode, so every position this episode feeds was forwarded by one of its own calls (``routing.stitch``).
        The heads need no salt: they are measured on each call's own decode steps, and vLLM's block hash includes the LoRA
        adapter, so no call reuses KV computed by another policy version. A call's heads are kept with its sampled-token
        logprobs (``topk.TopKStore`` tells equal calls apart by them), and only when it sampled at the training
        temperature (``train_temperature``)."""
        import numpy as np

        from crucible.train.bridges.tinker.sampler import _EPISODE

        k = self.topk.k if self.topk is not None else 0
        sp: dict[str, Any] = {"max_tokens": int(max_tokens), "temperature": float(temperature), "logprobs": k, "stop_token_ids": list(stop or []),
                              **NEUTRAL_SAMPLING}
        if seed is not None:
            sp["seed"] = int(seed)
        body: dict[str, Any] = {"model": self.name, "token_ids": list(prompt_ids), "sampling_params": sp}
        if (ep := _EPISODE.get()) is not None:
            body["cache_salt"] = f"crucible-{ep}"
        out = await with_retries("vllm generate", lambda: self.http.post("/inference/v1/generate", body, timeout_s=self.timeout_s), stats=self.stats)
        res, routes, heads = parse_generate(out, prompt_len=len(prompt_ids))
        full = list(prompt_ids) + list(res.tokens)
        if routes is not None and self.routing is not None:
            self.routing.put(full, routes)
        if heads is not None and self.topk is not None:
            if heads.ids.shape[0] != len(res.tokens):
                raise ValueError(f"vLLM generate returned {heads.ids.shape[0]} top-k rows for {len(res.tokens)} tokens")
            if self.train_temperature is None or abs(float(temperature) - self.train_temperature) <= 1e-9:
                self.topk.put(full, replace(heads, sampled=np.asarray(res.logprobs, dtype=np.float32)))
            else:  # vLLM's processed logprobs are softmax(z / T_call); the trainer scores at its own T (e.g. a greedy probe, T = 0)
                self.stats["sc_heads_other_temperature"] = self.stats.get("sc_heads_other_temperature", 0) + 1
        return res


async def topk_cost(http: Http, model: str, prompt_ids: Sequence[int], *, ks: Sequence[int] = (0, 8, 32), n: int = 8, max_tokens: int = 1024,
                    timeout_s: float = 600.0) -> list[dict[str, float]]:
    """What returning the top-k head costs on the running server (swarm-pipeline/14 §7.1): ``n`` concurrent generate calls
    per k (greedy-free, temperature 1, fixed seeds, ``ignore_eos`` so every call makes ``max_tokens``), their wall time,
    generated tokens per second and response bytes per token. k = 0 is the baseline."""
    import json as _json

    rows = []
    for k in ks:
        body = lambda i, k=k: {"model": model, "token_ids": list(prompt_ids), "cache_salt": f"crucible-topk-bench-{k}-{i}",
                               "sampling_params": {"max_tokens": int(max_tokens), "temperature": 1.0, "logprobs": int(k), "seed": i, "ignore_eos": True}}
        t0 = time.monotonic()
        outs = await asyncio.gather(*(http.post("/inference/v1/generate", body(i), timeout_s=timeout_s) for i in range(n)))
        wall = time.monotonic() - t0
        toks = sum(len((o.get("choices") or [{}])[0].get("token_ids") or []) for o in outs)
        size = sum(len(_json.dumps(o)) for o in outs)
        heads = sum(1 for o in outs if (o.get("choices") or [{}])[0].get("topk_ids"))
        rows.append({"k": float(k), "wall_s": wall, "tokens": float(toks), "tokens_per_s": toks / wall if wall > 0 else 0.0,
                     "response_bytes_per_token": size / max(toks, 1), "calls_with_heads": float(heads)})
    return rows


def parse_completion(out: dict[str, Any]) -> SampleOut:
    """vLLM's completion with ``return_tokens_as_token_ids``: tokens are ``"token_id:<n>"``."""
    ch = (out.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    toks = [int(str(t).rsplit(":", 1)[-1]) for t in lp.get("tokens") or []]
    lps = [float(x) if x is not None else 0.0 for x in lp.get("token_logprobs") or []]
    if len(toks) != len(lps):
        raise ValueError(f"vLLM returned {len(toks)} tokens and {len(lps)} logprobs")
    cached = ((out.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return SampleOut(tokens=toks, logprobs=lps, stop_reason="length" if ch.get("finish_reason") == "length" else "stop",
                     cached_prompt_tokens=int(cached) if isinstance(cached, int) else None)


def parse_generate(out: dict[str, Any], *, prompt_len: int = 0) -> tuple[SampleOut, Any, Any]:
    """prime-rl's generate response: ``token_ids``, the sampled logprobs packed as ``completion_logprobs`` (float32) or
    in ``logprobs.content``, ``routed_experts`` (``routing.decode``) and the top-k head ``topk_ids`` / ``topk_logprobs``
    (``topk.decode``; row i is completion token i, at position ``prompt_len + i``)."""
    import base64

    import numpy as np

    ch = (out.get("choices") or [{}])[0]
    toks = [int(t) for t in ch.get("token_ids") or []]
    cl = ch.get("completion_logprobs")
    if cl:
        lps = np.frombuffer(base64.b64decode(cl["data"]), dtype=np.dtype(cl["dtype"])).astype(float).tolist()
    else:
        lps = [float(x.get("logprob", 0.0)) for x in ((ch.get("logprobs") or {}).get("content") or [])]
    if len(toks) != len(lps):
        raise ValueError(f"vLLM generate returned {len(toks)} tokens and {len(lps)} logprobs")
    cached = ((out.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    so = SampleOut(tokens=toks, logprobs=lps, stop_reason="length" if ch.get("finish_reason") == "length" else "stop",
                   cached_prompt_tokens=int(cached) if isinstance(cached, int) else None)
    return so, decode(ch.get("routed_experts")), topk_mod.decode(ch.get("topk_ids"), ch.get("topk_logprobs"), start=prompt_len)


@dataclass
class _Trainer:
    backend: PrimeRLBackend
    step: int  # prime-rl's next batch step index

    async def forward_backward(self, datums: Sequence[TokenDatum], *, loss_fn: str, config: dict[str, float] | None) -> FBOut:
        b = self.backend
        if loss_fn != b.loss_kind:
            raise ValueError(f"the prime-rl trainer was launched with loss {b.loss_kind!r}, not {loss_fn!r} (deploy.trainer_toml)")
        rows = wire.samples(datums, temperature=b.temperature, env_name=b.env_name, routing=b.routing, topk=b.topk)
        t0 = time.monotonic()
        await b.io.send(self.step, rows)
        d = await b.io.wait_broadcast(self.step, timeout_s=b.step_timeout_s)
        name = f"{b.adapter_prefix}-s{self.step:06d}"
        await b.load_adapter(name, d)
        m = await b.trainer_metrics(self.step)
        m.update({"primerl/step": float(self.step), "primerl/step_s": time.monotonic() - t0, "primerl/samples": float(len(rows)),
                  "primerl/token_scale": wire.undo_token_mean(datums)})
        if b.routing is not None:
            cov = [r["routing_coverage"] for r in rows]
            m["routing/coverage_mean"], m["routing/coverage_min"] = sum(cov) / len(cov), min(cov)
            m["routing/replayed_share"] = sum(x is not None for x in wire.replay_all_or_none(rows)) / len(rows)  # what the step sends
        if b.topk is not None:
            cov = [r["sc_coverage"] for r in rows]
            sent = sum(x is not None for x in topk_mod.all_or_none(rows)) / len(rows)
            b.sc_steps += 1
            b.sc_fallbacks += sent < 1.0
            m.update({"sc/coverage_mean": sum(cov) / len(cov), "sc/coverage_min": min(cov), "sc/centered_share": sent,
                      "sc/fallback": float(sent < 1.0), "sc/fallback_steps_share": b.sc_fallbacks / b.sc_steps})
            if sent < 1.0:
                log.warning("prime-rl step %d: %d of %d sequences lack a sampler head at a trained position (coverage min %.3f): "
                            "no heads sent, the loss falls back to MIS for the whole step", self.step, sum(c < 1.0 for c in cov), len(cov), min(cov))
        for dt, row in zip(datums, rows, strict=True):  # trained: their calls' routes and heads are no longer needed (``d`` is the broadcast)
            full = wire.full_ids(dt)
            for end in wire.call_ends(dt):
                if b.routing is not None:
                    b.routing.drop(full[:end])
                if b.topk is not None:  # this sequence's own call: another call with the same tokens keeps its heads
                    b.topk.drop(full[:end], full_logprobs=topk_mod.trained_logprobs(row["logprobs"], row["mask"]))
        b.last = (self.step, d, name)
        self.step += 1
        return FBOut(target_logprobs=[], metrics=m)

    async def optim_step(self, *, learning_rate: float) -> dict[str, float]:
        if abs(learning_rate - self.backend.learning_rate) > 1e-12:
            raise ValueError(f"the prime-rl trainer was launched with lr {self.backend.learning_rate}, not {learning_rate}")
        return {}

    async def save_for_sampler(self, name: str, *, ttl_s: int | None) -> str:
        if self.backend.last is None:
            raise RuntimeError("save_for_sampler before any optimizer step")
        return str(self.backend.last[1])

    async def save_state(self, name: str, *, ttl_s: int | None) -> str:
        b = self.backend
        if b.last is None:
            raise RuntimeError("save_state before any optimizer step")
        s = b.last[0]
        t0 = time.monotonic()
        while (p := b.io.checkpoint(s)) is None:
            if time.monotonic() - t0 > b.step_timeout_s:
                raise TimeoutError(f"prime-rl wrote no checkpoint for step {s} (set ckpt.interval = 1)")
            await asyncio.sleep(b.poll_s)
        return f"primerl-ckpt:{s}:{p}"


@dataclass
class PrimeRLBackend:
    """``train.bridges.tinker.backend.Backend`` over a running prime-rl trainer and vLLM server (module doc)."""

    io: IO
    http: Http
    base_model: str  # the HF name vLLM loads (``vllm.model``)
    learning_rate: float
    loss_kind: str = "importance_sampling"
    temperature: float = 1.0  # the trainer's per-token temperature (``wire``): the loop's sampling temperature (delta_rl)
    env_name: str = "crucible"
    adapter_prefix: str = "crucible"
    step_timeout_s: float = 6 * 3600.0
    poll_s: float = 1.0
    startup_timeout_s: float = 3 * 3600.0
    # prime-rl logs a step's numbers after its checkpoint, so they land after the broadcast the bridge waits for (the
    # passing smoke, job 3474848: steps 1 and 2 reached steps.jsonl without the trainer's loss or grad norm)
    metrics_keys: tuple[str, ...] = TRAINER_METRIC_KEYS
    metrics_timeout_s: float = 600.0
    stats: dict[str, int] = field(default_factory=dict)
    adapters: dict[str, str] = field(default_factory=dict)  # adapter dir -> the name vLLM serves it under
    routing: RoutingStore | None = None  # MoE router replay (``routing``): None samples over /v1/completions without routes
    topk: TopKStore | None = None  # score centering (``topk``): the sampler's top-k head per token, over /inference/v1/generate
    sc_steps: int = 0
    sc_fallbacks: int = 0  # steps sent without heads (one sequence lacked a head at a trained position): the loss used MIS
    last: tuple[int, Path, str] | None = None

    @property
    def served_base(self) -> str:
        """The name the base model is served under. With LoRA on (always, here) prime-rl serves it as ``<model>-base``
        (``InferenceConfig.to_namespace``, v0.10.0), because vLLM rejects an adapter named like a served base model."""
        return f"{self.base_model}-base"

    async def trainer_metrics(self, step: int) -> dict[str, float]:
        """The trainer's logged numbers for ``step``, once its rows with ``metrics_keys`` are in prime-rl's file monitor.
        The wait is bounded: past ``metrics_timeout_s`` the step goes on with what is there, and
        ``primerl/metrics_complete`` = 0 says so (a missing number never stops training)."""
        t0 = time.monotonic()
        while True:
            m = dict(self.io.metrics(step))
            done = all(f"primerl/{k}" in m for k in self.metrics_keys)
            if done or time.monotonic() - t0 > self.metrics_timeout_s:
                if not done:
                    log.warning("prime-rl logged no %s for step %d in %.0fs", [k for k in self.metrics_keys if f"primerl/{k}" not in m], step,
                                self.metrics_timeout_s)
                m["primerl/metrics_complete"] = 1.0 if done else 0.0
                m["primerl/metrics_wait_s"] = time.monotonic() - t0
                return m
            await asyncio.sleep(self.poll_s)

    async def load_adapter(self, name: str, path: Path) -> None:
        await with_retries("vllm load_lora_adapter", lambda: self.http.post("/load_lora_adapter", {"lora_name": name, "lora_path": str(path)},
                                                                          timeout_s=600.0), stats=self.stats)
        self.adapters[str(path)] = name

    async def sampler(self, model: str, path: str | None = None) -> _Sampler:
        if path is None:
            return _Sampler(self.http, self.served_base, None, self.stats, routing=self.routing, topk=self.topk, train_temperature=self.temperature)
        name = self.adapters.get(str(path))
        if name is None:  # a resumed run samples from a version this process never loaded
            name = f"{self.adapter_prefix}-{Path(path).name}"
            await self.load_adapter(name, Path(path))
        return _Sampler(self.http, name, str(path), self.stats, routing=self.routing, topk=self.topk, train_temperature=self.temperature)

    async def trainer(self, model: str, *, rank: int, resume: str | None = None, seed: int | None = None) -> _Trainer:
        """The trainer process is already running (``deploy``); its first step is 1 (``FIRST_STEP``), or the checkpoint step + 1 for
        ``resume`` (``primerl-ckpt:<s>:<dir>``, written by ``save_state``). Its startup broadcast (policy v{first-1}) is
        acknowledged here, or the trainer would wait for a consumer forever."""
        first = FIRST_STEP if resume is None else parse_ckpt(resume)[0] + 1
        d = await self.io.wait_broadcast(first - 1, timeout_s=self.startup_timeout_s)
        if resume is not None:  # a fresh run's startup adapter is the zero-initialised LoRA: the base model
            await self.load_adapter(f"{self.adapter_prefix}-s{first - 1:06d}", d)
        return _Trainer(self, first)

    async def archive_url(self, path: str) -> str:
        return Path(path).as_uri()


def parse_ckpt(ref: str) -> tuple[int, str]:
    kind, _, rest = ref.partition(":")
    if kind != "primerl-ckpt":
        raise ValueError(f"not a prime-rl checkpoint reference: {ref!r}")
    s, _, p = rest.partition(":")
    return int(s), p
