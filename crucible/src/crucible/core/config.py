"""Configuration (``crucible.toml``) with secret references.

Discovery order: ``$CRUCIBLE_CONFIG``; ``./crucible.toml``; the nearest ``crucible.toml`` walking up
from the working directory; ``~/.config/crucible/crucible.toml``; built-in defaults.

Secrets never live in the file (agentenv rule): a value ``"env:NAME"`` is read from the process
environment, falling back to the nearest ``.env`` file. Resolved secrets are never logged or
serialized; ``Config.redacted()`` shows the references instead.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from crucible.core.schema.base import CrucibleModel
from crucible.core.secrets import register_secret

# What a local engine (vLLM, SGLang) takes in place of a key; any other api_key must be an env:NAME reference (R-8).
KEY_PLACEHOLDERS = frozenset({"", "EMPTY", "none"})


class LLMProvider(CrucibleModel):
    """Secrets are references only (R-8; PATTERN: Harbor templates sensitive env in its config serializer, Apache-2.0):
    ``api_key`` is ``env:NAME`` or a placeholder, and a header whose name looks secret (``Authorization``,
    ``X-Api-Key``, ``Cookie``) has an ``env:NAME`` value holding the whole header value. A dump of the config then
    carries names, never keys, and ``resolved_headers`` registers what it reads with the scrubber."""

    base_url: str
    api_key: str = ""  # "env:NAME", or a placeholder in KEY_PLACEHOLDERS
    headers: dict[str, str] = Field(default_factory=dict)  # secret-looking names take "env:NAME" values
    # merged into every chat request to this provider: self-hosted engines (vLLM / SGLang) need e.g.
    # {"logprobs": true, "extra_body": {"return_token_ids": true}} for token-in-token-out capture (train mode)
    request_defaults: dict[str, Any] = Field(default_factory=dict)
    # the version stamp of the fixed checkpoint this provider serves (train-mode token records need one; a trainer
    # bridge overrides it on every weight update). None: hosted APIs, whose records are never train-valid anyway.
    policy_version: int | str | None = None
    # token-in-token-out (self-hosted vLLM): render with the model's own chat template here, send token ids to the
    # engine's completions endpoint, parse tool calls here. Exact capture for training; a chat API's parser + template
    # round trip re-renders history lossily (prefix breaks). ``template_kwargs`` go to apply_chat_template.
    tito: bool = False
    template_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("api_key")
    @classmethod
    def _key_is_a_reference(cls, v: str) -> str:
        if v in KEY_PLACEHOLDERS or (v.startswith("env:") and len(v) > 4):
            return v
        raise ValueError("api_key must be env:NAME (the key read from the environment or .env) or, for a local engine, one of "
                         f"{sorted(KEY_PLACEHOLDERS)}: a literal key would sit in crucible.toml and in every dump of the config")

    @field_validator("headers")
    @classmethod
    def _secret_headers_are_references(cls, v: dict[str, str]) -> dict[str, str]:
        literal = sorted(k for k, val in v.items() if looks_like_secret_key(k) and not (val.startswith("env:") and len(val) > 4))
        if literal:
            raise ValueError(f"headers {literal} look secret: give each as env:NAME, the variable holding the whole value "
                             "(e.g. 'Bearer ...'), not the value itself")
        return v

    def resolved_headers(self, *, search_from: Path | None = None) -> dict[str, str]:
        """Headers to send: ``env:NAME`` values read (and registered with the scrubber), the others as written."""
        return {k: resolve_secret(val, search_from=search_from) if val.startswith("env:") else val for k, val in self.headers.items()}


class GpuHost(CrucibleModel):
    ssh: str  # user@host
    docker_host: str | None = None  # e.g. ssh://user@host ; default derived from ssh
    gpu_arch: str | None = None
    shared: bool = True  # shared machine: run politely (containers only, clean up, check load)


class Config(CrucibleModel):
    data_dir: str = "data"
    llm_providers: dict[str, LLMProvider] = Field(
        default_factory=lambda: {
            "openrouter": LLMProvider(base_url="https://openrouter.ai/api/v1", api_key="env:OPENROUTER_API_KEY"),
        }
    )
    # Model aliases: name -> "<provider>:<model id>"
    models: dict[str, str] = Field(
        default_factory=lambda: {
            "sim-default": "openrouter:openai/gpt-6-luna",
            "judge-default": "openrouter:anthropic/claude-haiku-5.5",
            "policy-default": "openrouter:openai/gpt-6-luna",
        }
    )
    gpu_hosts: dict[str, GpuHost] = Field(default_factory=dict)
    prices: dict[str, list[float]] = Field(default_factory=dict)  # model id -> [usd per 1M input, usd per 1M output]
    llm_budget_usd: float = 1.0  # default hard cap per LLMClient instance
    backends: dict[str, dict[str, Any]] = Field(default_factory=dict)
    root: str | None = None  # directory the config was loaded from (set by loader)

    def data_path(self) -> Path:
        p = Path(self.data_dir)
        if not p.is_absolute() and self.root:
            p = Path(self.root) / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def model(self, alias_or_id: str) -> tuple[LLMProvider, str]:
        """Resolve a model alias (or ``provider:model``) to (provider, model id)."""
        spec = self.models.get(alias_or_id, alias_or_id)
        provider_name, sep, model_id = spec.partition(":")
        if not sep or provider_name not in self.llm_providers:
            raise KeyError(f"model {alias_or_id!r} must be an alias or '<provider>:<model>' with a known provider")
        return self.llm_providers[provider_name], model_id

    def redacted(self) -> dict[str, Any]:
        """The config with every secret masked: secret-shaped keys keep only ``env:NAME`` references."""
        return _redact(self.model_dump(mode="json"))


# PORT of NeMo-Gym ``secret_utils.py`` (Apache-2.0): a key is secret-shaped when its name contains one of these, matched
# case-insensitively (``authorization``, not ``auth``, so ``author`` stays visible).
MASKED_VALUE = "****"
_SECRET_KEY_SUBSTRINGS = ("token", "key", "header", "password", "passwd", "secret", "credential", "bearer", "cookie",
                          "authorization")


def looks_like_secret_key(key: object) -> bool:
    lowered = str(key).lower()
    return any(s in lowered for s in _SECRET_KEY_SUBSTRINGS)


def _redact(node: Any, secret: bool = False) -> Any:
    if isinstance(node, dict):
        return {k: _redact(v, secret or looks_like_secret_key(k)) for k, v in node.items()}
    if isinstance(node, list):
        return [_redact(v, secret) for v in node]
    if secret and isinstance(node, str) and node and not node.startswith("env:"):
        return MASKED_VALUE
    return node


def resolve_secret(value: str, *, search_from: Path | None = None) -> str:
    """The secret ``value`` names (``env:NAME``, from the environment or .env) or the literal itself; either way it is
    registered with ``core.secrets``, so scrubbed text never carries it (R-5)."""
    if value.startswith("env:"):
        name = value[4:]
        resolved = os.environ.get(name) or _dotenv(search_from or Path.cwd()).get(name)
        if not resolved:
            raise KeyError(f"secret {name} is not set (environment or .env)")
        value = resolved
    register_secret(value)
    return value


def _dotenv(start: Path) -> dict[str, str]:
    for d in [start, *start.resolve().parents]:
        f = d / ".env"
        if f.is_file():
            out: dict[str, str] = {}
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.removeprefix("export ").strip()
                v = v.strip().strip('"').strip("'")
                out[k] = v
            return out
    return {}


def _discover() -> Path | None:
    if os.environ.get("CRUCIBLE_CONFIG"):
        return Path(os.environ["CRUCIBLE_CONFIG"])
    for d in [Path.cwd(), *Path.cwd().resolve().parents]:
        if (d / "crucible.toml").is_file():
            return d / "crucible.toml"
    home = Path.home() / ".config" / "crucible" / "crucible.toml"
    return home if home.is_file() else None


@lru_cache(maxsize=1)
def get_config() -> Config:
    path = _discover()
    if path is None:
        return Config(root=str(Path.cwd()))
    data = tomllib.loads(path.read_text())
    data["root"] = str(path.parent.resolve())
    return Config.model_validate(data)


def reset_config() -> None:
    get_config.cache_clear()
