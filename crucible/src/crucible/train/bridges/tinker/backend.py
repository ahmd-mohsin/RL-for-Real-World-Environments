"""The Tinker service behind one narrow interface (swarm-pipeline/09 §1): what the bridge calls, nothing else.

``Backend`` is the protocol; ``RealTinker`` implements it over the ``tinker`` SDK (0.34, Apache-2.0, an optional extra:
``pip install 'crucible[tinker]'``; imported lazily, so nothing here needs it until a real call), ``fake.FakeTinker``
implements it in memory for tests and dry runs. The bridge's own types cross the boundary (``SampleOut``,
``TokenDatum``, ``FBOut``), so no Tinker object leaks into a record.

- ``sampler(model, path)``: a sampling handle on the base model (``path`` None) or on saved sampler weights
  (``tinker://<run>/sampler_weights/<name>``). ``sample(prompt_ids, max_tokens, temperature, stop, seed)`` is one
  token-in/token-out call: the sampled ids, their logprobs, the stop reason and the prefix-cache hits the service billed.
- ``trainer(model, rank, resume)``: a LoRA training handle, new or restored *with its optimizer state* from a
  ``save_state`` path (``create_training_client_from_state_with_optimizer``). ``forward_backward(datums, loss_fn,
  config)``, ``optim_step(lr)``, ``save_for_sampler(name)`` (a sampler path), ``save_state(name)`` (a resume path).
- ``archive_url(path)``: a signed URL of a checkpoint's LoRA adapter (``adapter_model.safetensors``,
  ``adapter_config.json``; a tar archive).

**The key** is read by the SDK from ``TINKER_API_KEY`` in the process environment (the caller sources the repo's ``.env``
into a subprocess); it is never passed, stored, logged or printed here.

**Errors: infra vs task** (``classify``). A connection error, a timeout, a 5xx, a 408/409/429 or a Tinker
``RequestFailedError`` is the service's fault: retried with capped, jittered backoff (``RETRY_ATTEMPTS``, honouring a
Retry-After), then raised as ``actors.llm.LLMError`` (``provider_unavailable``, infra: the episode is masked, never
scored). A 402 or a ``BillingError`` is ``BudgetExceeded`` (``crucible:credits_exhausted``, infra). A 400 that says the
prompt is too long is ``ContextOverflow`` (the policy's episode ends truncated, scored). Any other 4xx is
``RequestRejected`` (the request is wrong: never retried, masked with harness blame).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from crucible.actors.llm import CONTEXT_OVERFLOW_MARKERS, BudgetExceeded, ContextOverflow, LLMError, RequestRejected

RETRY_ATTEMPTS = 6
RETRY_CAP_S = 60.0


@dataclass
class SampleOut:
    tokens: list[int]
    logprobs: list[float]
    stop_reason: str  # stop | length
    cached_prompt_tokens: int | None = None  # None: the service did not say (booked as uncached)


@dataclass
class TokenDatum:
    """One training sequence in next-token form: ``input_ids`` = full[:-1], every per-position list aligned with
    ``target_ids`` = full[1:]. ``mask`` 1 on the policy's own sampled tokens (never sent: a 0 advantage is what keeps the
    rest out of the gradient, since Tinker's RL losses take no mask)."""

    input_ids: list[int]
    target_ids: list[int]
    logprobs: list[float]
    advantages: list[float]
    mask: list[int]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.input_ids)
        if not (n == len(self.target_ids) == len(self.logprobs) == len(self.advantages) == len(self.mask)):
            raise ValueError("TokenDatum fields must align with input_ids")

    @property
    def n_train(self) -> int:
        return sum(self.mask)


@dataclass
class FBOut:
    target_logprobs: list[list[float]]  # per datum, the learner's logprobs of the target tokens
    metrics: dict[str, float] = field(default_factory=dict)


class SamplerHandle(Protocol):
    path: str | None

    async def sample(self, prompt_ids: Sequence[int], *, max_tokens: int, temperature: float, stop: Sequence[int] | None, seed: int | None) -> SampleOut: ...


class TrainerHandle(Protocol):
    async def forward_backward(self, datums: Sequence[TokenDatum], *, loss_fn: str, config: dict[str, float] | None) -> FBOut: ...

    async def optim_step(self, *, learning_rate: float) -> dict[str, float]: ...

    async def save_for_sampler(self, name: str, *, ttl_s: int | None) -> str: ...

    async def save_state(self, name: str, *, ttl_s: int | None) -> str: ...


class Backend(Protocol):
    async def sampler(self, model: str, path: str | None = None) -> SamplerHandle: ...

    async def trainer(self, model: str, *, rank: int, resume: str | None = None, seed: int | None = None) -> TrainerHandle: ...

    async def archive_url(self, path: str) -> str: ...


# ---- errors -----------------------------------------------------------------------------------------------------------------
def classify(e: BaseException) -> str:
    """``retry`` (the service's fault), ``budget``, ``overflow``, ``reject`` (the request's fault) or ``raise`` (not a
    service error: a bug, let it surface)."""
    name = type(e).__name__
    status = getattr(e, "status_code", None)
    text = str(e).casefold()
    if name == "BillingError" or status == 402:
        return "budget"
    if status in (400, 413) and any(m in text for m in CONTEXT_OVERFLOW_MARKERS + ("context length", "too long", "exceeds the model")):
        return "overflow"
    if name in ("APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError", "RequestFailedError", "TimeoutError",
                "ConnectionError", "RemoteProtocolError") or (isinstance(status, int) and (status >= 500 or status in (408, 409, 425, 429))):
        return "retry"
    if isinstance(status, int) and 400 <= status < 500:
        return "reject"
    if isinstance(e, (asyncio.TimeoutError, ConnectionError, OSError)):
        return "retry"
    return "raise"


def _retry_after(e: BaseException) -> float | None:
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    try:
        v = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return v if v >= 0 else None


async def with_retries(what: str, fn: Any, *, attempts: int = RETRY_ATTEMPTS, sleep: Any = asyncio.sleep, stats: dict[str, int] | None = None) -> Any:
    """``await fn()`` with the infra/task split above."""
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return await fn()
        except (asyncio.CancelledError, LLMError, BudgetExceeded):
            raise
        except Exception as e:
            kind = classify(e)
            if stats is not None:
                stats[kind] = stats.get(kind, 0) + 1
            if kind == "budget":
                raise BudgetExceeded(f"tinker refused for credits: {what}: {e}") from e
            if kind == "overflow":
                raise ContextOverflow(f"tinker: {what}: {e}") from e
            if kind == "reject":
                raise RequestRejected(f"tinker rejected {what}: {e}") from e
            if kind == "raise":
                raise
            last = e
            if i + 1 < attempts:
                ra = _retry_after(e)
                await sleep(min(ra if ra is not None else min(1.5 * 2**i, RETRY_CAP_S) * (0.5 + random.random()), RETRY_CAP_S))
    err = LLMError(f"tinker {what} failed after {attempts} attempts: {type(last).__name__}: {last}")
    err.retry_after = _retry_after(last) if last is not None else None
    raise err from last


# ---- the real service -------------------------------------------------------------------------------------------------------
def _tinker() -> Any:
    try:
        import tinker
    except ImportError as e:  # pragma: no cover -- the optional extra is missing
        raise ImportError("the Tinker bridge needs the tinker SDK: pip install 'crucible[tinker]' (or uv pip install tinker)") from e
    return tinker


class _RealSampler:
    def __init__(self, client: Any, path: str | None, stats: dict[str, int]):
        self.client, self.path, self.stats = client, path, stats

    async def sample(self, prompt_ids: Sequence[int], *, max_tokens: int, temperature: float, stop: Sequence[int] | None, seed: int | None) -> SampleOut:
        t = _tinker()
        params = t.SamplingParams(max_tokens=int(max_tokens), temperature=float(temperature), seed=seed, stop=list(stop) if stop is not None else None)
        mi = t.ModelInput.from_ints([int(x) for x in prompt_ids])

        async def call() -> Any:
            return await self.client.sample_async(prompt=mi, num_samples=1, sampling_params=params)

        r = await with_retries("sample", call, stats=self.stats)
        s = r.sequences[0]
        lps = list(s.logprobs or [])
        toks = list(s.tokens)
        if len(lps) != len(toks):
            raise LLMError(f"tinker returned {len(lps)} logprobs for {len(toks)} tokens")
        return SampleOut(tokens=[int(x) for x in toks], logprobs=[float(x) for x in lps], stop_reason=str(s.stop_reason),
                         cached_prompt_tokens=getattr(r, "prompt_cache_hit_tokens", None))


class _RealTrainer:
    def __init__(self, client: Any, stats: dict[str, int]):
        self.client, self.stats = client, stats

    async def forward_backward(self, datums: Sequence[TokenDatum], *, loss_fn: str, config: dict[str, float] | None) -> FBOut:
        t = _tinker()
        data = [t.Datum(model_input=t.ModelInput.from_ints(list(d.input_ids)),
                        loss_fn_inputs={"target_tokens": list(d.target_ids), "logprobs": [float(x) for x in d.logprobs],
                                        "advantages": [float(x) for x in d.advantages]}) for d in datums]

        async def call() -> Any:
            fut = await self.client.forward_backward_async(data, loss_fn=loss_fn, loss_fn_config=config)
            return await fut.result_async()

        out = await with_retries("forward_backward", call, attempts=2, stats=self.stats)  # not idempotent: one retry only on a transport error
        lps = []
        for o in out.loss_fn_outputs:
            v = o.get("logprobs") if isinstance(o, dict) else getattr(o, "logprobs", None)
            lps.append([float(x) for x in (v.tolist() if hasattr(v, "tolist") else (v.data if hasattr(v, "data") else v or []))])
        return FBOut(target_logprobs=lps, metrics={k: float(v) for k, v in (out.metrics or {}).items()})

    async def optim_step(self, *, learning_rate: float) -> dict[str, float]:
        t = _tinker()

        async def call() -> Any:
            fut = await self.client.optim_step_async(t.AdamParams(learning_rate=float(learning_rate), beta1=0.9, beta2=0.95, eps=1e-8))
            return await fut.result_async()

        r = await with_retries("optim_step", call, attempts=2, stats=self.stats)
        return {k: float(v) for k, v in (r.metrics or {}).items()}

    async def save_for_sampler(self, name: str, *, ttl_s: int | None) -> str:
        async def call() -> Any:
            fut = await self.client.save_weights_for_sampler_async(name, ttl_seconds=ttl_s)
            return await fut.result_async()

        return str((await with_retries("save_weights_for_sampler", call, stats=self.stats)).path)

    async def save_state(self, name: str, *, ttl_s: int | None) -> str:
        async def call() -> Any:
            fut = await self.client.save_state_async(name, ttl_seconds=ttl_s, overwrite=True)
            return await fut.result_async()

        return str((await with_retries("save_state", call, stats=self.stats)).path)


class RealTinker:
    """The Tinker service (``tinker.ServiceClient``; the SDK reads ``TINKER_API_KEY`` from the environment)."""

    def __init__(self) -> None:
        self._service: Any = None
        self.stats: dict[str, int] = {}

    def service(self) -> Any:
        if self._service is None:
            self._service = _tinker().ServiceClient()
        return self._service

    async def sampler(self, model: str, path: str | None = None) -> _RealSampler:
        svc = self.service()

        async def call() -> Any:
            if path:
                return await svc.create_sampling_client_async(model_path=path)
            return await svc.create_sampling_client_async(base_model=model)

        return _RealSampler(await with_retries("create_sampling_client", call, stats=self.stats), path, self.stats)

    async def trainer(self, model: str, *, rank: int, resume: str | None = None, seed: int | None = None) -> _RealTrainer:
        svc = self.service()

        async def call() -> Any:
            if resume:
                return await svc.create_training_client_from_state_with_optimizer_async(resume)
            return await svc.create_lora_training_client_async(base_model=model, rank=int(rank), seed=seed)

        return _RealTrainer(await with_retries("create_training_client", call, attempts=3, stats=self.stats), self.stats)

    async def archive_url(self, path: str) -> str:
        rest = self.service().create_rest_client()

        async def call() -> Any:
            return await rest.get_checkpoint_archive_url_from_tinker_path_async(path)

        r = await with_retries("checkpoint archive", call, stats=self.stats)
        return str(getattr(r, "url", r))
