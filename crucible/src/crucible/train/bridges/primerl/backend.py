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
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from crucible.train.bridges.primerl import wire
from crucible.train.bridges.primerl.routing import RoutingStore, decode
from crucible.train.bridges.tinker.backend import FBOut, SampleOut, TokenDatum, with_retries
from crucible.train.bridges.tinker.models import Prices, TinkerModel, resolve

SENDER_READY, RECEIVER_READY, FINISHED = ".sender_ready", ".receiver_ready", ".finished"
FIRST_STEP = 1  # a fresh prime-rl trainer's first step (``Progress.step``, v0.10.0); its startup broadcast is step_0


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
        d = self.output_dir / "broadcasts" / f"step_{step}"
        t0 = time.monotonic()
        while not (d / SENDER_READY).exists():
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"prime-rl trainer published no weights for step {step} in {timeout_s:.0f}s ({d})")
            await asyncio.sleep(self.poll_s)
        (d / RECEIVER_READY).touch()  # unblocks the trainer's broadcast
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

    async def sample(self, prompt_ids: Sequence[int], *, max_tokens: int, temperature: float, stop: Sequence[int] | None, seed: int | None) -> SampleOut:
        # vLLM reads a list of strings as a batch of text prompts and answers 200, so a mis-tokenised prompt would only
        # fail later, in the trainer's batch decode (smoke attempt 9: transformers 5 returned BatchEncoding keys)
        if not all(isinstance(t, int) and not isinstance(t, bool) for t in prompt_ids):
            raise TypeError(f"prompt token ids must be ints, got {list(prompt_ids)[:3]}")
        if self.routing is not None:
            return await self._generate(prompt_ids, max_tokens=max_tokens, temperature=temperature, stop=stop, seed=seed)
        body: dict[str, Any] = {"model": self.name, "prompt": list(prompt_ids), "max_tokens": int(max_tokens), "temperature": float(temperature),
                                "logprobs": 0, "return_tokens_as_token_ids": True, "skip_special_tokens": False, "stop_token_ids": list(stop or [])}
        if seed is not None:
            body["seed"] = int(seed)
        out = await with_retries("vllm completions", lambda: self.http.post("/v1/completions", body, timeout_s=self.timeout_s), stats=self.stats)
        return parse_completion(out)

    async def _generate(self, prompt_ids: Sequence[int], *, max_tokens: int, temperature: float, stop: Sequence[int] | None, seed: int | None) -> SampleOut:
        """vLLM's token-in/token-out ``/inference/v1/generate`` on prime-rl's server: the routed experts come back with
        the completion. The prefix cache is salted per episode, so every position this episode feeds was forwarded by one
        of its own calls (``routing.stitch``)."""
        from crucible.train.bridges.tinker.sampler import _EPISODE

        sp: dict[str, Any] = {"max_tokens": int(max_tokens), "temperature": float(temperature), "logprobs": 0, "stop_token_ids": list(stop or [])}
        if seed is not None:
            sp["seed"] = int(seed)
        body: dict[str, Any] = {"model": self.name, "token_ids": list(prompt_ids), "sampling_params": sp}
        if (ep := _EPISODE.get()) is not None:
            body["cache_salt"] = f"crucible-{ep}"
        out = await with_retries("vllm generate", lambda: self.http.post("/inference/v1/generate", body, timeout_s=self.timeout_s), stats=self.stats)
        res, routes = parse_generate(out)
        if routes is not None and self.routing is not None:
            self.routing.put(list(prompt_ids) + list(res.tokens), routes)
        return res


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


def parse_generate(out: dict[str, Any]) -> tuple[SampleOut, Any]:
    """prime-rl's generate response: ``token_ids``, the sampled logprobs packed as ``completion_logprobs`` (float32) or
    in ``logprobs.content``, and ``routed_experts`` (``routing.decode``)."""
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
    return so, decode(ch.get("routed_experts"))


@dataclass
class _Trainer:
    backend: PrimeRLBackend
    step: int  # prime-rl's next batch step index

    async def forward_backward(self, datums: Sequence[TokenDatum], *, loss_fn: str, config: dict[str, float] | None) -> FBOut:
        b = self.backend
        if loss_fn != b.loss_kind:
            raise ValueError(f"the prime-rl trainer was launched with loss {b.loss_kind!r}, not {loss_fn!r} (deploy.trainer_toml)")
        rows = wire.samples(datums, temperature=b.temperature, env_name=b.env_name, routing=b.routing)
        t0 = time.monotonic()
        await b.io.send(self.step, rows)
        d = await b.io.wait_broadcast(self.step, timeout_s=b.step_timeout_s)
        name = f"{b.adapter_prefix}-s{self.step:06d}"
        await b.load_adapter(name, d)
        m = dict(b.io.metrics(self.step))
        m.update({"primerl/step": float(self.step), "primerl/step_s": time.monotonic() - t0, "primerl/samples": float(len(rows)),
                  "primerl/token_scale": wire.undo_token_mean(datums)})
        if b.routing is not None:
            cov = [r["routing_coverage"] for r in rows]
            m["routing/coverage_mean"], m["routing/coverage_min"] = sum(cov) / len(cov), min(cov)
            m["routing/replayed_share"] = sum(x is not None for x in wire.replay_all_or_none(rows)) / len(rows)  # what the step sends
            for d in datums:  # trained: their calls' routes are no longer needed
                for end in wire.call_ends(d):
                    b.routing.drop(wire.full_ids(d)[:end])
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
    temperature: float = 1.0
    env_name: str = "crucible"
    adapter_prefix: str = "crucible"
    step_timeout_s: float = 6 * 3600.0
    poll_s: float = 1.0
    startup_timeout_s: float = 3 * 3600.0
    stats: dict[str, int] = field(default_factory=dict)
    adapters: dict[str, str] = field(default_factory=dict)  # adapter dir -> the name vLLM serves it under
    routing: RoutingStore | None = None  # MoE router replay (``routing``): None samples over /v1/completions without routes
    last: tuple[int, Path, str] | None = None

    @property
    def served_base(self) -> str:
        """The name the base model is served under. With LoRA on (always, here) prime-rl serves it as ``<model>-base``
        (``InferenceConfig.to_namespace``, v0.10.0), because vLLM rejects an adapter named like a served base model."""
        return f"{self.base_model}-base"

    async def load_adapter(self, name: str, path: Path) -> None:
        await with_retries("vllm load_lora_adapter", lambda: self.http.post("/load_lora_adapter", {"lora_name": name, "lora_path": str(path)},
                                                                          timeout_s=600.0), stats=self.stats)
        self.adapters[str(path)] = name

    async def sampler(self, model: str, path: str | None = None) -> _Sampler:
        if path is None:
            return _Sampler(self.http, self.served_base, None, self.stats, routing=self.routing)
        name = self.adapters.get(str(path))
        if name is None:  # a resumed run samples from a version this process never loaded
            name = f"{self.adapter_prefix}-{Path(path).name}"
            await self.load_adapter(name, Path(path))
        return _Sampler(self.http, name, str(path), self.stats, routing=self.routing)

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
