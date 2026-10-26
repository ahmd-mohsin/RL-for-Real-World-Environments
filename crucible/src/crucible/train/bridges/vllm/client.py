"""The service side of the vLLM bridge (package doc): a model row, and the ``sampler(model, path)`` backend over vLLM's
OpenAI-compatible server.

**Token in, token out.** ``/v1/completions`` takes the prompt as token ids (our codec rendered them; the server never
re-templates), and ``return_tokens_as_token_ids`` with ``logprobs`` returns the sampled ids (``"token_id:<n>"``) and
their logprobs, the same ``SampleOut`` the Tinker backend returns. ``skip_special_tokens`` is off so the tool-call and
end-of-turn markers survive; ``stop_token_ids`` are the codec's.

**Adapters.** ``sampler(model, path)`` with a ``path`` (a PEFT LoRA directory on the node: ``adapter_config.json`` plus
``adapter_model.safetensors``) loads it at runtime (``POST /v1/load_lora_adapter``; the server runs with
``--enable-lora`` and ``VLLM_ALLOW_RUNTIME_LORA_UPDATING=True``) under ``adapter_name(path)`` and samples through that
name; ``path`` None samples the base. A base and an adapter arm therefore share one server, one engine build and one set
of sampling settings.

**Errors.** Connection errors and 5xx are the service's (retried with backoff, then ``LLMError``: masked); a 400 that
names the context is ``ContextOverflow``; other 4xx are the request's (``backend.classify`` and ``with_retries``, shared
with the Tinker bridge).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

from crucible.actors.llm import LLMError
from crucible.train.bridges.tinker.backend import SampleOut, with_retries
from crucible.train.bridges.tinker.models import MODELS, Prices, TinkerModel

PROVIDER = "vllm"
ZERO = Prices(0.0, 0.0, 0.0, 0.0)  # our own GPUs: no per-token price (GPU-hours are the Slurm account's ledger)
DEFAULT_CONTEXT = 262_144
# max output tokens per model call on our own engine (the coordinator, 2026-10-25): replies cut at 4096 in about 90 stored
# episodes across most debug tasks, and on a migration task the cut ended episodes outright. One value for every arm of a
# comparison; each arm's settings record it (``factory.tinker.arm_settings``: ``max_tokens``).
DEFAULT_MAX_TOKENS = 8192


@dataclass(frozen=True)
class VLLMModel(TinkerModel):
    """A model row for a model we serve ourselves: the Tinker row's renderer facts (family, tool format, thinking switch,
    stop tokens), priced at 0, provider ``vllm``. ``served`` is the server's ``--served-model-name``."""

    served: str = ""

    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def ref(self) -> str:
        return f"{PROVIDER}:{self.name}"


def vllm_model(name: str, *, context: int = DEFAULT_CONTEXT, served: str | None = None) -> VLLMModel:
    """The row for ``name`` (an HF id, e.g. ``Qwen/Qwen3.6-35B-A3B``) from the Tinker table's row of the same base, with
    ``context`` the server's ``--max-model-len``. An id with no row is refused: its tool format would be a guess."""
    base = name.split(":", 1)[0]
    rows = [m for m in MODELS.values() if m.base == base]
    if not rows:
        raise KeyError(f"no model row for {base!r}: its family, tool format and thinking switch are unknown (train/bridges/tinker/models)")
    row = min(rows, key=lambda m: m.context)
    kw = {f.name: getattr(row, f.name) for f in fields(TinkerModel)}
    kw.update(name=base, context=int(context), prices=ZERO, list_prices=None, notes=f"served by vLLM (row of {row.name})", tested=())
    return VLLMModel(**kw, served=served or base)


def adapter_name(path: str) -> str:
    """The server-side name of the LoRA adapter at ``path`` (stable: the same path is one adapter)."""
    return "lora-" + hashlib.sha256(str(path).encode()).hexdigest()[:12]


class _HTTPError(Exception):
    """An HTTP status from the server, shaped for ``backend.classify`` (``status_code``)."""

    def __init__(self, status: int, text: str):
        super().__init__(f"HTTP {status}: {text[:500]}")
        self.status_code = status


def parse_choice(doc: Mapping[str, Any]) -> SampleOut:
    """A ``/v1/completions`` reply (``return_tokens_as_token_ids`` + ``logprobs``) as ``SampleOut``."""
    choices = doc.get("choices") or []
    if not choices:
        raise LLMError(f"vllm returned no choice: {str(doc)[:300]}")
    c = choices[0]
    lp = c.get("logprobs") or {}
    raw, lps = list(lp.get("tokens") or []), list(lp.get("token_logprobs") or [])
    if len(raw) != len(lps):
        raise LLMError(f"vllm returned {len(lps)} logprobs for {len(raw)} tokens")
    toks = []
    for t in raw:
        s = str(t)
        if not s.startswith("token_id:"):
            raise LLMError(f"vllm returned a token as text ({s[:40]!r}): the server needs return_tokens_as_token_ids")
        toks.append(int(s.split(":", 1)[1]))
    usage = doc.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    finish = str(c.get("finish_reason") or "stop")
    return SampleOut(tokens=toks, logprobs=[float(x) if x is not None else 0.0 for x in lps], stop_reason="length" if finish == "length" else "stop",
                     cached_prompt_tokens=int(cached) if cached is not None else None)


class _VLLMSampler:
    def __init__(self, backend: VLLMBackend, served: str, path: str | None):
        self.backend, self.served, self.path = backend, served, path

    def body(self, prompt_ids: Sequence[int], *, max_tokens: int, temperature: float, stop: Sequence[int] | None, seed: int | None) -> dict[str, Any]:
        b: dict[str, Any] = {"model": self.served, "prompt": [int(x) for x in prompt_ids], "max_tokens": int(max_tokens), "temperature": float(temperature),
                             "logprobs": 1, "return_tokens_as_token_ids": True, "skip_special_tokens": False,
                             "spaces_between_special_tokens": False, "n": 1, **self.backend.sampling}
        if stop:
            b["stop_token_ids"] = [int(x) for x in stop]
        if seed is not None:
            b["seed"] = int(seed) & 0x7FFFFFFF
        return b

    async def sample(self, prompt_ids: Sequence[int], *, max_tokens: int, temperature: float, stop: Sequence[int] | None, seed: int | None) -> SampleOut:
        body = self.body(prompt_ids, max_tokens=max_tokens, temperature=temperature, stop=stop, seed=seed)

        async def call() -> SampleOut:
            return parse_choice(await self.backend.post("/v1/completions", body))

        return await with_retries("sample", call, stats=self.backend.stats)


class VLLMBackend:
    """``sampler(model, path)`` over one vLLM server at ``url`` (``CRUCIBLE_VLLM_URL``, e.g. ``http://127.0.0.1:8000``).
    ``sampling``: extra request fields identical for every call of a run (``top_p``, ``top_k``; recorded by the caller)."""

    def __init__(self, url: str | None = None, *, timeout_s: float = 1800.0, sampling: Mapping[str, Any] | None = None, transport: Any = None,
                 engine: Mapping[str, Any] | None = None):
        import os

        self.url = (url or os.environ.get("CRUCIBLE_VLLM_URL") or "http://127.0.0.1:8000").rstrip("/")
        self.timeout_s, self.sampling = float(timeout_s), dict(sampling or {})
        self.engine = dict(engine) if engine else None  # ``ServeSpec.record()``: stamped into every arm's settings
        self.stats: dict[str, int] = {}
        self.adapters: dict[str, str] = {}  # path -> served name
        self._transport = transport
        self._client: Any = None

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(base_url=self.url, timeout=self.timeout_s, transport=self._transport)
        return self._client

    async def post(self, route: str, body: Mapping[str, Any]) -> dict[str, Any]:
        r = await self._http().post(route, json=dict(body))
        if r.status_code >= 400:
            raise _HTTPError(r.status_code, r.text)
        try:
            return r.json() if r.content else {}
        except ValueError:  # the adapter routes answer in plain text ("Success: LoRA adapter ... added")
            return {"text": r.text}

    async def get(self, route: str) -> dict[str, Any]:
        r = await self._http().get(route)
        if r.status_code >= 400:
            raise _HTTPError(r.status_code, r.text)
        return r.json() if r.content else {}

    async def models(self) -> list[str]:
        return [str(m.get("id")) for m in (await self.get("/v1/models")).get("data") or []]

    async def load_adapter(self, path: str) -> str:
        name = adapter_name(path)
        if path in self.adapters:
            return self.adapters[path]
        if name not in await self.models():
            try:
                await self.post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
            except _HTTPError as e:
                if "already" not in str(e).lower():
                    raise LLMError(f"vllm could not load the adapter at {path}: {e}") from e
        self.adapters[path] = name
        return name

    async def sampler(self, model: str, path: str | None = None) -> _VLLMSampler:
        served = await self.load_adapter(path) if path else model
        return _VLLMSampler(self, served, path)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
