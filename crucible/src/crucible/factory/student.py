"""The student: the small open model the factory targets (docs/research/swarm-pipeline/08-student-and-builder.md).

The student is ``Qwen/Qwen3.5-4B`` (Apache-2.0; hybrid linear + full attention, a vision tower, hybrid thinking), the
model trained on Tinker, so the tasks are calibrated against exactly the model that is trained. ``Qwen3-4B-Instruct-2507``
(retired on Tinker) is only a dev stand-in (``DEV_STANDINS``): its episodes never count. The local server is
``scripts/student_serve.sh`` at bf16, text only: **vLLM** on a CUDA box (``--enable-auto-tool-choice``, the tool parser
matching the model's template) or **mlx-lm** on Apple silicon (its parser inferred from the template; the vision tower's
weights are dropped on load). Both answer the harness's native OpenAI ``tools`` / ``tool_calls``
(``actors.policy.LLMToolPolicy``), so no text tool protocol is needed.

- **Registry and the provider seam.** The student is a ``ModelSpec(provider="student", model=STUDENT_MODEL,
  family="qwen")``; the provider is ``[llm_providers.student]`` in ``crucible.toml`` (``http://127.0.0.1:8100/v1``,
  ``api_key = "EMPTY"``), so ``scripts/lh_run.py --models student:Qwen/Qwen3.5-4B`` and ``scripts/lh_target.py --model
  ...`` use it as any rung, and its calls book 0 USD on localhost (``LLMClient._self_hosted``). The harness only sees an
  ``LLMClient``, so a Tinker sampler replaces the local server without touching it, in one of two ways: (1) **a provider
  entry** (``student_provider``): Tinker's OpenAI-compatible endpoint (``base_url`` + ``api_key = "env:TINKER_API_KEY"``)
  or token-in/token-out (``tito = true``, the engine stack behind ``train.world_env``'s pointer); (2) **a registered
  backend** (``register_backend("tinker", make, final=True)``): the Tinker bridge (``train/bridges/tinker``, owned
  elsewhere) registers a factory for its ``TinkerSampler`` client; ``backend(name).make(model, thinking=...)`` builds it.
  This module never calls Tinker.
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

STUDENT_PROVIDER = "student"
STUDENT_MODEL = "Qwen/Qwen3.5-4B"  # the model trained on Tinker: tasks are calibrated against exactly it
# a dev stand-in only (retired on Tinker in June 2026): its episodes are dev-only whatever engine served them
DEV_STANDINS: frozenset[str] = frozenset({"Qwen/Qwen3-4B-Instruct-2507"})
STUDENT_PORT = 8100
STUDENT_BASE_URL = f"http://127.0.0.1:{STUDENT_PORT}/v1"
# The numbers that count come from the sampler of the trained model (Tinker). Local serving (MLX on the Mac, vLLM on a
# CUDA box) is dev and calibration only: its kernels, sampler and accumulation are not Tinker's.
DEV_BACKENDS: frozenset[str] = frozenset({"mlx", "vllm", "sglang", "unknown"})
FINAL_BACKENDS: frozenset[str] = frozenset({"tinker"})
THINKING_DEFAULT = False  # Qwen3.5's hybrid thinking: off for the debugging agent; recorded per run (``thinking_of``)
STUDENT_MAX_TOKENS = 4096  # per reply: a 4B model that loops on a tool call should not fill the context in one turn


class DevOnlyNumerics(RuntimeError):
    """A dev-only (MLX or unrecorded) backend's episodes reached a place where a number counts."""


def student_spec(model: str = STUDENT_MODEL, *, provider: str = STUDENT_PROVIDER) -> ModelSpec:
    return ModelSpec(provider, model, "qwen")


# ---- student backends: where a non-HTTP sampler (Tinker) registers ------------------------------------------------------------
@dataclass(frozen=True)
class StudentBackend:
    """A registered student sampler. ``make(model, thinking=..., **kw)`` returns an ``LLMClient``-shaped object (``complete``
    with ``tools``, ``spend``, ``model``, ``lock_settings``); ``final``: its episodes count for targeting and benchmarks."""

    name: str
    make: Callable[..., Any]
    final: bool = False
    doc: str = ""


_BACKENDS: dict[str, StudentBackend] = {}


def register_backend(name: str, make: Callable[..., Any], *, final: bool = False, doc: str = "") -> StudentBackend:
    """Register a student sampler under ``name`` (the Tinker bridge registers ``tinker``, ``final=True``). The name is the
    backend recorded in run metadata and stamped into policy ids; a final backend's episodes count."""
    if not name or "@" in name or "-" in name:
        raise ValueError(f"a backend name is one word (it is a policy-id tag): {name!r}")
    b = StudentBackend(name, make, final, doc)
    _BACKENDS[name] = b
    return b


def backend(name: str) -> StudentBackend:
    if name not in _BACKENDS:
        raise KeyError(f"no student backend {name!r} registered (have {sorted(_BACKENDS)}); local servers go through [llm_providers.student]")
    return _BACKENDS[name]


def backends() -> dict[str, StudentBackend]:
    return dict(_BACKENDS)


def is_final_backend(name: str) -> bool:
    b = _BACKENDS.get(name)
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
        return self.model in DEV_STANDINS

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
    dev = DEV_BACKENDS | {n for n, b in _BACKENDS.items() if not b.final}
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
    if s in DEV_BACKENDS or (s in _BACKENDS and not _BACKENDS[s].final):
        return "dev"
    if is_final_backend(s):
        return "final"
    return "dev" if is_dev_policy(s) else "final"


def assert_countable(x: Any, purpose: str) -> None:
    """Raise ``DevOnlyNumerics`` when ``x`` (see ``numerics_of``) is dev-only and ``purpose`` would count it."""
    if numerics_of(x) == "dev":
        finals = sorted(FINAL_BACKENDS | {n for n, b in _BACKENDS.items() if b.final})
        raise DevOnlyNumerics(f"{purpose}: dev-only numerics ({x if isinstance(x, str) else 'a local or unknown backend'}); numbers that count "
                              f"come from a final backend ({finals}) sampling {STUDENT_MODEL}")
