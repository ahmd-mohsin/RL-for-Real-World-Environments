"""Configuration (``crucible.toml``) with secret references.

Layers (M0 audit 2026-10-14 R-23; PATTERN: Prime Agent ``pa-core/src/settings/manager.rs`` keeps security settings
global, MIT), each over the one before: built-in defaults; the user file ``~/.config/crucible/crucible.toml``; the
project file, ``./crucible.toml`` or the nearest one walking up from the working directory, never past the root of the
git work tree it is in (only ``./crucible.toml`` outside a repository); ``--set KEY=VALUE`` (``set_overrides``). Tables
merge key by key, so a project file no longer hides the user's prices and GPU hosts. ``$CRUCIBLE_CONFIG`` names the one
file to use instead of the user and project files (the operator's explicit choice).

The first file found used to win, so a ``crucible.toml`` in a parent directory could send the user's API key to its own
``base_url`` and raise the budget from 1.0 to 500.0. Now a ``crucible.toml`` above the working directory outside its
repository is refused, and the operator-only settings (``OPERATOR_ONLY``: where a provider's requests and credentials go,
where code runs, the spending cap) are decided by the operator's layers: a project file sets them only where the user
file says nothing, and may lower the budget but never raise it. ``Config.sources`` names every layer that was read
(``crucible config`` prints them).

Secrets never live in the file (agentenv rule): a value ``"env:NAME"`` is read from the process
environment, falling back to the nearest ``.env`` file. Resolved secrets are never logged or
serialized; ``Config.redacted()`` shows the references instead.
"""

from __future__ import annotations

import copy
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
    # model id -> USD per 1M tokens: [input, output] or [input, output, cache read] or [input, output, cache read, cache
    # write]; a cache price left out is the input price (R-19)
    prices: dict[str, list[float]] = Field(default_factory=dict)
    llm_budget_usd: float = 1.0  # default hard cap per LLMClient instance
    root: str | None = None  # directory the config was loaded from (set by loader)
    sources: list[str] = Field(default_factory=list)  # every layer read, lowest first: "user:<path>", "project:<path>", "set:<key>" (set by loader)

    @field_validator("prices")
    @classmethod
    def _price_lists(cls, v: dict[str, list[float]]) -> dict[str, list[float]]:
        bad = sorted(k for k, p in v.items() if not 2 <= len(p) <= 4 or any(x < 0 for x in p))
        if bad:
            raise ValueError(f"prices {bad}: give USD per 1M tokens as [input, output], optionally then cache read and cache write")
        return v

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
    """The nearest ``.env`` from ``start`` up, parsed with python-dotenv's grammar (R-29; vendored in
    ``crucible._vendor.dotenv_parser``, BSD-3-Clause): quotes, escapes, ``export`` and inline comments as python-dotenv
    reads them, no interpolation. The parser here split at the first ``=``, so ``KEY=sk-...  # note`` kept the comment in
    the key and every call failed 401. A later binding of a key wins, as in python-dotenv; a line it cannot parse is
    skipped."""
    import io

    from crucible._vendor.dotenv_parser import parse_stream

    for d in [start, *start.resolve().parents]:
        f = d / ".env"
        if f.is_file():
            return {b.key: b.value for b in parse_stream(io.StringIO(f.read_text(encoding="utf-8"))) if b.key and b.value is not None and not b.error}
    return {}


class ConfigError(ValueError):
    """A configuration file the layering refuses: one above the working directory outside its repository, or a project
    file overriding an operator-only setting."""


# Settings only the operator's layers decide (the user file, $CRUCIBLE_CONFIG, --set): where a provider's requests and
# credentials go, where code runs, how much one client may spend. Globs over dotpaths (``core.fingerprint``).
OPERATOR_ONLY = ("llm_providers.*.base_url", "llm_providers.*.api_key", "llm_providers.*.headers", "gpu_hosts", "llm_budget_usd")
_OVERRIDES: dict[str, Any] = {}


def user_config_path() -> Path:
    return Path.home() / ".config" / "crucible" / "crucible.toml"


def project_config_path(cwd: Path | None = None) -> Path | None:
    """The project file for ``cwd``: the nearest ``crucible.toml`` from ``cwd`` up to the root of its git work tree
    (``cwd`` alone outside a repository). Raises ConfigError for a ``crucible.toml`` beyond that, above ``cwd``."""
    here = Path(cwd or Path.cwd()).resolve()
    chain = [here, *here.parents]
    top = next((i for i, d in enumerate(chain) if (d / ".git").exists()), None)
    inside = chain[: top + 1] if top is not None else chain[:1]
    for d in inside:
        if (d / "crucible.toml").is_file():
            return d / "crucible.toml"
    for d in chain[len(inside):]:
        f = d / "crucible.toml"
        if f.is_file():
            where = f"outside its repository ({chain[top]})" if top is not None else "and it is not in a repository"
            raise ConfigError(f"{f} lies above {here} {where}: a configuration file found there is not used, since it would decide "
                              "where API keys go and how much may be spent. Point CRUCIBLE_CONFIG at it, or move its settings to "
                              f"{user_config_path()}")
    return None


def _merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in top.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _check_project(project: dict[str, Any], below: dict[str, Any], path: Path) -> None:
    """Refuse a project file that overrides an operator-only setting the layers below it decide, or raises the budget."""
    from crucible.core.fingerprint import excluded, flatten

    have = {".".join(k): v for k, v in flatten(below).items()}
    for parts, value in flatten(project).items():
        key = ".".join(parts)
        if not excluded(parts, OPERATOR_ONLY):
            continue
        if key == "llm_budget_usd":
            cap = below.get("llm_budget_usd", Config.model_fields["llm_budget_usd"].default)
            if isinstance(value, (int, float)) and value > cap:
                raise ConfigError(f"{path} raises llm_budget_usd from {cap} to {value}: a project file may lower the spending cap, never "
                                  f"raise it; set it in {user_config_path()} or pass --set llm_budget_usd={value}")
        elif key in have and have[key] != value:
            raise ConfigError(f"{path} overrides {key}, which the user configuration sets: where requests, credentials and code go "
                              "is the operator's to decide; remove it from the project file or change it in the user file")


def set_overrides(pairs: list[str] | tuple[str, ...]) -> None:
    """``--set KEY=VALUE`` (a dotted key, a TOML value; a bare word is a string): the top layer. Clears the cache."""
    out: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"--set takes KEY=VALUE, not {pair!r}")
        try:
            value = tomllib.loads(f"v = {raw.strip()}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw.strip()
        node = out
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    _OVERRIDES.clear()
    _OVERRIDES.update(out)
    reset_config()


def load_config(cwd: Path | None = None) -> Config:
    """The layered configuration for ``cwd`` (see the module doc)."""
    from crucible.core import env as knobs
    from crucible.core.fingerprint import flatten

    data: dict[str, Any] = {}
    sources: list[str] = []
    root: Path | None = None
    explicit = knobs.get("CRUCIBLE_CONFIG")
    if explicit is not None:
        files = [("explicit", Path(explicit))]
    else:
        user, project = user_config_path(), project_config_path(cwd)
        files = [*([("user", user)] if user.is_file() else []),
                 *([("project", project)] if project is not None and (not user.is_file() or not project.samefile(user)) else [])]
    for kind, path in files:
        doc = tomllib.loads(path.read_text())
        doc.pop("sources", None)
        doc.pop("root", None)
        if kind == "project":
            _check_project(doc, data, path)
        data = _merge(data, doc)
        sources.append(f"{kind}:{path.resolve()}")
        root = path.parent.resolve()
    if _OVERRIDES:
        data = _merge(data, _OVERRIDES)
        sources += [f"set:{'.'.join(k)}" for k in flatten(_OVERRIDES)]
    data["root"], data["sources"] = str(root or Path(cwd or Path.cwd()).resolve()), sources
    return Config.model_validate(data)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()


def reset_config() -> None:
    get_config.cache_clear()
