"""The vLLM server's command line on a DeltaAI GH200 node (doc 14-deltaai §2).

**The engine.** NCSA's LLMHub vLLM container for DeltaAI (``/sw/llmhub/vllm-v0.28.0-slingshot-deltaai.sif``: upstream
``vllm/vllm-openai:v0.28.0`` arm64, Ubuntu 24.04 / CUDA 13, plus the site's Slingshot libfabric and aws-ofi-nccl), pinned
by its sha256 (``VLLM_SIF_SHA256``). It is the site's maintained build for this machine (glibc 2.38 floor, NCCL over
Slingshot); building our own aarch64 wheel would re-derive the same constraints.

**The flags that matter for equal arms:** ``--max-model-len`` is the tasks' 262144-token context; ``--seed`` fixes the
engine's sampling seed stream (each request also carries its own seed); ``--enable-lora`` with ``--max-lora-rank`` lets
the trained arm load its adapter at runtime onto the same base, so both arms share one engine build;
``--generation-config vllm`` keeps the model repo's ``generation_config.json`` from overriding the request's sampling
parameters (each arm's temperature and top_p are the request's, recorded in its report); ``--enable-prefix-caching``
(the episodes re-send a growing prompt); ``--enable-prompt-tokens-details`` (cached prefill counts in ``usage``).
Thinking is the renderer's (``TinkerChatCodec``, ``enable_thinking`` off for the student): the server never applies a
chat template, since prompts arrive as token ids.
"""

from __future__ import annotations

import asyncio
import shlex
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

VLLM_SIF = "/sw/llmhub/vllm-v0.28.0-slingshot-deltaai.sif"
VLLM_SIF_SHA256 = "4395a0c67b8fc3752d9571c898456820e9bdfc7feb21d075332be1f71497146c"
VLLM_VERSION = "v0.28.0"


@dataclass(frozen=True)
class ServeSpec:
    """One server: ``model`` is a local weights directory (or an HF id when the cache holds it), served under ``name``."""

    model: str
    name: str
    tp: int = 1
    context: int = 262_144
    port: int = 8000
    seed: int = 0
    gpu_memory_utilization: float = 0.90
    max_lora_rank: int = 64
    max_loras: int = 2
    max_num_seqs: int = 64
    enable_lora: bool = True
    extra: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)

    def vllm_argv(self) -> list[str]:
        a = ["vllm", "serve", self.model, "--served-model-name", self.name, "--host", "127.0.0.1", "--port", str(self.port),
             "--tensor-parallel-size", str(self.tp), "--max-model-len", str(self.context), "--seed", str(self.seed),
             "--gpu-memory-utilization", str(self.gpu_memory_utilization), "--max-num-seqs", str(self.max_num_seqs),
             "--generation-config", "vllm", "--enable-prefix-caching", "--enable-prompt-tokens-details"]  # v0.28 logs no requests by default
        if self.enable_lora:
            a += ["--enable-lora", "--max-lora-rank", str(self.max_lora_rank), "--max-loras", str(self.max_loras)]
        return [*a, *self.extra]

    def environment(self) -> dict[str, str]:
        e = {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True" if self.enable_lora else "False", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
             "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
        return {**e, **dict(self.env)}

    def apptainer_argv(self, *, sif: str = VLLM_SIF, binds: Sequence[str] = ()) -> list[str]:
        """``apptainer exec --nv`` of the server: the weights directory and any adapter directories bound read-only. The
        server keeps the host network (it listens on 127.0.0.1 only; the episode Worlds have none)."""
        cmd = ["apptainer", "exec", "--nv", "--cleanenv", "--no-home"]
        for k, v in self.environment().items():
            cmd += ["--env", f"{k}={v}"]
        for b in binds:
            cmd += ["-B", b if ":" in b else f"{b}:{b}:ro"]
        return [*cmd, sif, *self.vllm_argv()]

    def shell(self, **kw: Any) -> str:
        return " ".join(shlex.quote(x) for x in self.apptainer_argv(**kw))

    def record(self) -> dict[str, Any]:
        """What a run report records about its engine (both arms must match on every field but the adapter)."""
        return {"engine": "vllm", "version": VLLM_VERSION, "sif": VLLM_SIF, "sif_sha256": VLLM_SIF_SHA256, "tp": self.tp, "context": self.context,
                "seed": self.seed, "enable_lora": self.enable_lora, "max_lora_rank": self.max_lora_rank, "generation_config": "vllm",
                "prefix_caching": True, "extra": list(self.extra)}


async def wait_ready(backend: Any, *, timeout_s: float = 1800.0, every_s: float = 5.0, sleep: Any = asyncio.sleep) -> float:
    """Poll ``/v1/models`` until the server lists a model; returns the seconds it took. Raises ``TimeoutError``."""
    t0 = time.monotonic()
    last = ""
    while True:
        try:
            if await backend.models():
                return time.monotonic() - t0
        except Exception as e:  # noqa: BLE001 - not up yet
            last = f"{type(e).__name__}: {e}"[:200]
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(f"the vLLM server at {backend.url} was not ready after {timeout_s:.0f}s (last: {last})")
        await sleep(every_s)
