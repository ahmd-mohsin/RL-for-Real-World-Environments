"""The student: the open model the factory builds tasks for and post-trains (docs/research/swarm-pipeline/08).

- **The target student** is ``Qwen/Qwen3.6-35B-A3B`` (MoE, about 3B active, 262K context), sampled and trained on Tinker:
  ``student_spec()`` is ``ModelSpec(provider="tinker", model=TARGET_MODEL, family="qwen")``, rung id
  ``tinker:Qwen/Qwen3.6-35B-A3B``. Tasks are calibrated per student model: only a model in ``TARGET_STUDENTS`` sampled by
  a final backend produces numbers that count, and failure findings and builder intents are keyed by the student model id
  (``factory.failures``).
- **The local dev model** is ``Qwen/Qwen3.5-4B`` (``DEV_MODEL``; Apache-2.0, hybrid linear + full attention, a vision
  tower, hybrid thinking), for harness development and the pipeline smoke only. A 35B MoE does not fit a 24 GB Mac at
  bf16, so the local path never serves the target. ``Qwen3-4B-Instruct-2507`` (retired on Tinker) is also a dev model.
  ``DEV_STANDINS`` lists both: their episodes never count, whatever served them. The local server is
  ``scripts/student_serve.sh`` at bf16, text only: **vLLM** on a CUDA box (``--enable-auto-tool-choice``, the tool parser
  matching the model's template) or **mlx-lm** on Apple silicon (its parser inferred from the template; the vision tower's
  weights are dropped on load). Both answer the harness's native OpenAI ``tools`` / ``tool_calls``
  (``actors.policy.LLMToolPolicy``), so no text tool protocol is needed.

- **Registry and the provider seam.** The local dev student is ``ModelSpec(provider="student", model=DEV_MODEL,
  family="qwen")`` (``dev_spec()``) on ``[llm_providers.student]`` in ``crucible.toml`` (``http://127.0.0.1:8100/v1``,
  ``api_key = "EMPTY"``), so ``scripts/lh_run.py --models student:Qwen/Qwen3.5-4B`` uses it as any rung, booking 0 USD on
  localhost. The harness only sees an ``LLMClient``, so the Tinker sampler replaces the local server without touching
  it, in one of two ways: (1) **a registered backend** (``register_backend("tinker", make, final=True)``): the Tinker
  bridge (``train/bridges/tinker``, owned elsewhere) registers a factory for its ``TinkerSampler`` client, and
  ``ModelBook`` builds every ``tinker:<model>`` rung through it (``backend("tinker").make(model, thinking=...)``);
  (2) **a provider entry** (``student_provider``): Tinker's OpenAI-compatible endpoint (``base_url`` + ``api_key =
  "env:TINKER_API_KEY"``) or token-in/token-out (``tito = true``). This module never calls Tinker.
- **Thinking** is an explicit run setting (``THINKING_DEFAULT``: off for the debugging agent). It rides in the provider's
  ``request_defaults`` as ``extra_body.chat_template_kwargs.enable_thinking`` (mlx-lm and vLLM read it per request),
  ``thinking_of`` reads it back, and ``scripts/lh_run.py`` records it in the policy config and the run's ``student.json``.
- **Backend metadata.** The backend (``mlx`` | ``vllm`` | ``tinker`` | ...) is recorded with every run. The serve script
  writes a descriptor (``student-<port>.json``: backend, model, dtype, engine version, pid, tool parser, thinking,
  ``dev_only``); ``probe`` reads it and checks the server answers (falling back to ``/v1/models``' ``owned_by``).
  ``scripts/lh_run.py`` stamps the backend into the policy id (``solver:student:<model>@mlx``) and writes the run's
  ``student.json``, so every record, trace bundle and metric row carries it.
- **Dev-only numerics.** The numbers that count come from the trained model's own sampler: ``FINAL_BACKENDS`` is
  ``tinker`` (or a backend registered ``final=True``). Local serving (MLX: other kernels, sampler and bf16 accumulation;
  vLLM: not Tinker's either) is dev and calibration only (``DEV_BACKENDS``), as is the stand-in model.
  ``is_dev_policy`` reads the stamp; ``assert_countable`` raises ``DevOnlyNumerics`` where a number would count
  (``targeting.target_verdict`` refuses an S15 verdict, ``benchmark.report`` a benchmark report, ``lh_run`` metrics give
  the window ``dev_only``). An unknown backend is dev: numerics nobody recorded never reach a final number.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from crucible.factory.models import ModelSpec

TARGET_MODEL = "Qwen/Qwen3.6-35B-A3B"  # the student tasks are built for and post-trained (Tinker)
TARGET_BACKEND = "tinker"
TARGET_STUDENTS: frozenset[str] = frozenset({TARGET_MODEL})  # models whose (final-backend) numbers count; one calibration each
STUDENT_MODEL = TARGET_MODEL
STUDENT_PROVIDER = "student"  # the local dev server's provider entry
DEV_MODEL = "Qwen/Qwen3.5-4B"  # harness development and the pipeline smoke on the local server
# dev models: their episodes are dev-only whatever engine served them (Qwen3-4B-2507 was retired on Tinker in June 2026)
DEV_STANDINS: frozenset[str] = frozenset({DEV_MODEL, "Qwen/Qwen3-4B-Instruct-2507"})
STUDENT_PROVIDERS: frozenset[str] = frozenset({STUDENT_PROVIDER, TARGET_BACKEND})  # provider prefixes that name a student rung
STUDENT_PORT = 8100
STUDENT_BASE_URL = f"http://127.0.0.1:{STUDENT_PORT}/v1"
# The numbers that count come from the sampler of the trained model (Tinker). Local serving (MLX on the Mac, vLLM on a
# CUDA box) is dev and calibration only: its kernels, sampler and accumulation are not Tinker's.
DEV_BACKENDS: frozenset[str] = frozenset({"mlx", "vllm", "sglang", "unknown"})
FINAL_BACKENDS: frozenset[str] = frozenset({"tinker"})
THINKING_DEFAULT = False  # Qwen3.5/3.6 hybrid thinking: off for the debugging agent; recorded per run (``thinking_of``)
STUDENT_MAX_TOKENS = 4096  # per reply: a 4B model that loops on a tool call should not fill the context in one turn


class DevOnlyNumerics(RuntimeError):
    """A dev-only (MLX or unrecorded) backend's episodes reached a place where a number counts."""


def student_spec(model: str = TARGET_MODEL, *, provider: str = TARGET_BACKEND) -> ModelSpec:
    """The target student's registry entry (``tinker:Qwen/Qwen3.6-35B-A3B``); the Tinker bridge supplies its client."""
    return ModelSpec(provider, model, "qwen")


def dev_spec(model: str = DEV_MODEL) -> ModelSpec:
    """The local dev student (``student:Qwen/Qwen3.5-4B`` on scripts/student_serve.sh): dev-only numerics."""
    return ModelSpec(STUDENT_PROVIDER, model, "qwen")


def student_model_of(policy: str) -> str | None:
    """The student model id inside a policy id (``solver:tinker:Qwen/Qwen3.6-35B-A3B@tinker`` -> the model), None when the
    policy is not a student rung (a hosted OpenRouter rung)."""
    p = str(policy or "").partition("@")[0]
    p = p.removeprefix("solver:")
    prov, sep, model = p.partition(":")
    return model if sep and prov in STUDENT_PROVIDERS and "/" in model else None


# ---- sampler backends: the registry lives in factory.models (where the Tinker bridge registers ``tinker``) -------------------
def _registry() -> dict[str, Any]:
    from crucible.factory import models

    return models._BACKENDS


def register_backend(name: str, make: Callable[..., Any], *, final: bool = False, doc: str = "") -> Any:
    """``models.register_backend`` (the Tinker bridge registers ``tinker`` with ``final=True``)."""
    from crucible.factory.models import register_backend as reg

    return reg(name, make, final=final, doc=doc)


def backend(name: str) -> Any:
    from crucible.factory.models import backend as get

    return get(name)


def backends() -> dict[str, Any]:
    return dict(_registry())


def is_final_backend(name: str) -> bool:
    b = _registry().get(name)
    return b.final if b is not None else name in FINAL_BACKENDS


def thinking_kwargs(thinking: bool = THINKING_DEFAULT) -> dict[str, Any]:
    """The request defaults that set Qwen3.5's thinking switch (chat-template kwarg, read per request by mlx-lm and vLLM)."""
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}}


def thinking_of(provider: Any) -> bool | None:
    """The thinking switch an ``LLMProvider`` (or its ``request_defaults`` dict) sends; None when it sends none (the model's
    own template default then decides, which a run must not leave unrecorded)."""
    rd = provider if isinstance(provider, Mapping) else (getattr(provider, "request_defaults", None) or {})
    v = ((rd.get("extra_body") or {}).get("chat_template_kwargs") or {}).get("enable_thinking")
    return None if v is None else bool(v)


def student_provider(kind: str = "local", *, base_url: str = STUDENT_BASE_URL, thinking: bool = THINKING_DEFAULT, api_key: str = "EMPTY",
                     policy_version: int | str | None = None) -> Any:
    """The ``[llm_providers.student]`` entry for one backend kind: ``local`` (student_serve.sh), ``tinker_openai`` (Tinker's
    OpenAI-compatible endpoint: ``api_key`` an ``env:`` reference) or ``tito`` (token-in/token-out through the engine stack;
    needs a ``policy_version``). The harness code is the same for all three."""
    from crucible.core.config import LLMProvider

    if kind not in ("local", "tinker_openai", "tito"):
        raise ValueError("student provider kind: local | tinker_openai | tito")
    if kind == "tinker_openai" and not api_key.startswith("env:"):
        raise ValueError("a Tinker endpoint takes its key as an env: reference")
    return LLMProvider(base_url=base_url, api_key=api_key, request_defaults=thinking_kwargs(thinking), tito=kind == "tito",
                       template_kwargs={"enable_thinking": bool(thinking)} if kind == "tito" else {}, policy_version=policy_version)


def state_dir() -> Path:
    return Path(os.environ.get("CRU_STUDENT_DIR") or Path.home() / ".cache" / "crucible" / "student")


@dataclass(frozen=True)
class StudentServer:
    backend: str  # mlx | vllm | sglang | unknown
    model: str
    base_url: str
    dtype: str = "bfloat16"
    engine_version: str = ""
    tool_parser: str = ""
    pid: int | None = None
    source: str = "descriptor"  # descriptor | models_endpoint
    thinking: bool | None = None  # the server's default (the provider's per-request setting is what a run records)

    @property
    def standin(self) -> bool:
        """A dev model (not a target student) served here."""
        return self.model in DEV_STANDINS or self.model not in TARGET_STUDENTS

    @property
    def dev_only(self) -> bool:
        return not is_final_backend(self.backend) or self.standin

    @property
    def numerics(self) -> str:
        return "dev" if self.dev_only else "final"

    def meta(self) -> dict[str, Any]:
        """The run metadata (``runs/<run>/student.json``)."""
        return {**asdict(self), "dev_only": self.dev_only, "numerics": self.numerics, "standin": self.standin}


def _port(base_url: str) -> int | None:
    from urllib.parse import urlparse

    return urlparse(base_url).port


def read_descriptor(base_url: str, *, root: Path | None = None) -> dict[str, Any] | None:
    port = _port(base_url)
    p = (root or state_dir()) / f"student-{port}.json"
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def probe(base_url: str = STUDENT_BASE_URL, *, root: Path | None = None, timeout_s: float = 5.0, check: bool = True) -> StudentServer | None:
    """The served student at ``base_url`` (None when nothing answers). ``check=False`` trusts the descriptor alone."""
    models: list[dict[str, Any]] = []
    if check:
        import httpx

        try:
            r = httpx.get(base_url.rstrip("/") + "/models", timeout=timeout_s)
            r.raise_for_status()
            models = [m for m in (r.json().get("data") or []) if isinstance(m, dict)]
        except (httpx.HTTPError, ValueError):
            return None
    d = read_descriptor(base_url, root=root)
    if d and d.get("base_url", base_url).rstrip("/") == base_url.rstrip("/"):
        return StudentServer(str(d.get("backend") or "unknown"), str(d.get("model") or ""), base_url, str(d.get("dtype") or ""),
                             str(d.get("engine_version") or ""), str(d.get("tool_parser") or ""), d.get("pid"), "descriptor",
                             None if d.get("thinking") is None else bool(d["thinking"]))
    owners = {str(m.get("owned_by") or "") for m in models}
    backend = "vllm" if "vllm" in owners else ("sglang" if "sglang" in owners else "unknown")
    return StudentServer(backend, str(models[0].get("id")) if models else "", base_url, "", "", "", None, "models_endpoint")


def backend_tag(server: StudentServer | None) -> str:
    """The policy-id tag of a backend (``@mlx``); ``unknown`` when nothing could be probed."""
    return server.backend if server is not None else "unknown"


def is_dev_policy(policy: str) -> bool:
    """A policy id stamped with a dev-only backend (``solver:student:<model>@mlx``, also ``@nudges6-mlx`` style tags), or
    one whose model is a dev stand-in (``DEV_STANDINS``)."""
    p = str(policy or "")
    head, _, tag = p.rpartition("@") if "@" in p else (p, "", "")
    if any(head.endswith(s) for s in DEV_STANDINS):
        return True
    m = student_model_of(p)
    # a student rung on a model no task is calibrated for; a Tinker context variant (``<model>:peft:262144``) is the same model
    if m is not None and m.split(":", 1)[0] not in TARGET_STUDENTS:
        return True
    dev = DEV_BACKENDS | {n for n, b in _registry().items() if not b.final}
    return any(part in dev for part in tag.replace("+", "-").split("-"))


def numerics_of(x: Any) -> str:
    """``dev`` or ``final`` from a StudentServer, its ``meta()``, a backend name or a policy id; ``final`` for anything that
    is not the student (hosted rungs are not served here)."""
    if isinstance(x, StudentServer):
        return x.numerics
    if isinstance(x, Mapping):
        if "numerics" in x:
            return str(x["numerics"])
        return "final" if is_final_backend(str(x.get("backend") or "unknown")) else "dev"
    s = str(x or "")
    if s in DEV_BACKENDS or (s in _registry() and not _registry()[s].final):
        return "dev"
    if is_final_backend(s):
        return "final"
    return "dev" if is_dev_policy(s) else "final"


def assert_countable(x: Any, purpose: str) -> None:
    """Raise ``DevOnlyNumerics`` when ``x`` (see ``numerics_of``) is dev-only and ``purpose`` would count it."""
    if numerics_of(x) == "dev":
        finals = sorted(FINAL_BACKENDS | {n for n, b in _registry().items() if b.final})
        raise DevOnlyNumerics(f"{purpose}: dev-only numerics ({x if isinstance(x, str) else 'a local or unknown backend'}); numbers that count "
                              f"come from a final backend ({finals}) sampling a target student ({sorted(TARGET_STUDENTS)})")
