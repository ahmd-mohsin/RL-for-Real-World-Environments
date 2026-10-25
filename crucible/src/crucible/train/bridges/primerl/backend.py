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
  bridge continues at batch step s + 1. Step numbering: Crucible's version v (after v optimizer steps) is prime-rl's
  policy ``v{v-1}`` (its step index starts at 0).

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
from crucible.train.bridges.tinker.backend import FBOut, SampleOut, TokenDatum, with_retries
from crucible.train.bridges.tinker.models import Prices, TinkerModel, resolve

SENDER_READY, RECEIVER_READY, FINISHED = ".sender_ready", ".receiver_ready", ".finished"


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
        await FileSystemBatchSender(self.output_dir, self.ranks, current_step=step).send(grid)

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
            for line in p.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("step") == step:
                    out.update({f"primerl/{k}": float(v) for k, v in row.items() if k != "step" and isinstance(v, int | float)})
        return out

    def checkpoint(self, step: int) -> Path | None:
        d = self.output_dir / "checkpoints" / f"step_{step}"
        return d if d.exists() else None


@dataclass
class _Sampler:
    http: Http
    name: str
    path: str | None
    stats: dict[str, int]
    timeout_s: float = 3600.0

    async def sample(self, prompt_ids: Sequence[int], *, max_tokens: int, temperature: float, stop: Sequence[int] | None, seed: int | None) -> SampleOut:
        body: dict[str, Any] = {"model": self.name, "prompt": list(prompt_ids), "max_tokens": int(max_tokens), "temperature": float(temperature),
                                "logprobs": 0, "return_tokens_as_token_ids": True, "skip_special_tokens": False, "stop_token_ids": list(stop or [])}
        if seed is not None:
            body["seed"] = int(seed)
        out = await with_retries("vllm completions", lambda: self.http.post("/v1/completions", body, timeout_s=self.timeout_s), stats=self.stats)
        return parse_completion(out)


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


@dataclass
class _Trainer:
    backend: PrimeRLBackend
    step: int  # prime-rl's next batch step index

    async def forward_backward(self, datums: Sequence[TokenDatum], *, loss_fn: str, config: dict[str, float] | None) -> FBOut:
        b = self.backend
        if loss_fn != b.loss_kind:
            raise ValueError(f"the prime-rl trainer was launched with loss {b.loss_kind!r}, not {loss_fn!r} (deploy.trainer_toml)")
        rows = wire.samples(datums, temperature=b.temperature, env_name=b.env_name)
        t0 = time.monotonic()
        await b.io.send(self.step, rows)
        d = await b.io.wait_broadcast(self.step, timeout_s=b.step_timeout_s)
        name = f"{b.adapter_prefix}-s{self.step:06d}"
        await b.load_adapter(name, d)
        m = dict(b.io.metrics(self.step))
        m.update({"primerl/step": float(self.step), "primerl/step_s": time.monotonic() - t0, "primerl/samples": float(len(rows)),
                  "primerl/token_scale": wire.undo_token_mean(datums)})
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
    base_model: str  # the HF name vLLM serves (``vllm.model``)
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
    last: tuple[int, Path, str] | None = None

    async def load_adapter(self, name: str, path: Path) -> None:
        await with_retries("vllm load_lora_adapter", lambda: self.http.post("/load_lora_adapter", {"lora_name": name, "lora_path": str(path)},
                                                                          timeout_s=600.0), stats=self.stats)
        self.adapters[str(path)] = name

    async def sampler(self, model: str, path: str | None = None) -> _Sampler:
        if path is None:
            return _Sampler(self.http, self.base_model, None, self.stats)
        name = self.adapters.get(str(path))
        if name is None:  # a resumed run samples from a version this process never loaded
            name = f"{self.adapter_prefix}-{Path(path).name}"
            await self.load_adapter(name, Path(path))
        return _Sampler(self.http, name, str(path), self.stats)

    async def trainer(self, model: str, *, rank: int, resume: str | None = None, seed: int | None = None) -> _Trainer:
        """The trainer process is already running (``deploy``); its first step is 0, or the checkpoint step + 1 for
        ``resume`` (``primerl-ckpt:<s>:<dir>``, written by ``save_state``). Its startup broadcast (policy v{first-1}) is
        acknowledged here, or the trainer would wait for a consumer forever."""
        first = 0 if resume is None else parse_ckpt(resume)[0] + 1
        d = await self.io.wait_broadcast(first - 1, timeout_s=self.startup_timeout_s)
        if first > 0:
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
