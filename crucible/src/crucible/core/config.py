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

A credential goes only to an endpoint the operator paired it with (M0 audit 2026-10-16 R-1; PATTERN: agentenv scopes a
key to its endpoint, ``config/model.py``, Apache-2.0). Checking a project file against the user file alone let it
through when there was no user file, or when it added a provider of its own: a cloned ``crucible.toml`` could point
openrouter at its own host with ``env:AWS_SECRET_ACCESS_KEY``, or add ``[llm_providers.mirror]`` holding
``env:OPENROUTER_API_KEY`` and route ``models.policy-default`` to it. Now every (endpoint, ``env:`` reference) pair in
the result must also be in the operator's view (built-in defaults, the user file, ``$CRUCIBLE_CONFIG``, ``--set``),
so a project file may still add a keyless local engine, or restate a provider the operator has.

Secrets never live in the file (agentenv rule): a value ``"env:NAME"`` is read from the process
environment, falling back to the nearest ``.env`` file. Resolved secrets are never logged or
serialized; ``Config.redacted()`` shows the references instead.
"""

from __future__ import annotations

import copy
import os
import re
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, ValidationError, field_validator

from crucible.core.schema.base import CrucibleModel
from crucible.core.secrets import register_secret

# What a local engine (vLLM, SGLang) takes in place of a key; any other api_key must be an env:NAME reference (R-8).
KEY_PLACEHOLDERS = frozenset({"", "EMPTY", "none"})
# A credential-named setting, matched on the key's last word (``auth_token``, ``X-Api-Key``, ``client_secret``), so
# ``max_tokens`` and ``return_token_ids`` stay ordinary settings (A16-R2).
CREDENTIAL_NAME = re.compile(r"(?:^|[_-])(?:api[_-]?key|apikey|password|passwd|secret|token|credentials?|authorization|cookie|bearer)$",
                             re.IGNORECASE)
# Request keywords the client sets itself; ``extra_headers`` and ``extra_query`` would be a second, unchecked route for
# credentials next to ``headers`` (A16-R2).
_CLIENT_OWNED = frozenset({"model", "messages", "tools", "stream", "timeout", "extra_headers", "extra_query"})


class LLMProvider(CrucibleModel):
    """Secrets are references only (R-8; PATTERN: Harbor templates sensitive env in its config serializer, Apache-2.0):
    ``api_key`` is ``env:NAME`` or a placeholder, and a header whose name looks secret (``Authorization``,
    ``X-Api-Key``, ``Cookie``) has an ``env:NAME`` value holding the whole header value. A dump of the config then
    carries names, never keys, and ``resolved_headers`` registers what it reads with the scrubber."""

    # a refused value is never echoed: a prefixless literal key is not something the scrubber knows (A16-R6)
    model_config = ConfigDict(hide_input_in_errors=True)

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

    @field_validator("base_url")
    @classmethod
    def _no_userinfo(cls, v: str) -> str:
        from urllib.parse import urlsplit

        if "@" in urlsplit(v).netloc:
            raise ValueError("base_url carries credentials (user@ or user:password@): give the key as api_key = env:NAME, since a "
                             "base_url is written into run locks and logs")
        return v

    @field_validator("request_defaults", "template_kwargs")
    @classmethod
    def _settings_carry_no_credentials(cls, v: dict[str, Any]) -> dict[str, Any]:
        """``request_defaults`` is merged into every request and written into run locks and ``Turn.sampling``: a header,
        query or credential there bypasses ``headers``' ``env:`` rule and the scrubber (A16-R2; PATTERN: agentenv
        ``config/model.py`` reserves the call keywords a params table may not set, Apache-2.0)."""
        from crucible.core.fingerprint import flatten

        body = v.get("extra_body")
        owned = sorted({k for k in [*v, *(body if isinstance(body, dict) else ())] if k in _CLIENT_OWNED})
        if owned:
            raise ValueError(f"{owned} may not be set here: headers go in the provider's headers (secret ones as env:NAME), and "
                             "model, messages, tools, stream and timeout are set by the client")
        secret = sorted(".".join(p) for p in flatten(v) if any(CREDENTIAL_NAME.search(seg) for seg in p))
        if secret:
            raise ValueError(f"{secret} look like credentials: give a key as api_key or a header as env:NAME, never as a setting "
                             "that is sent on every call and written into run locks")
        return v

    def resolved_headers(self, *, search_from: Path | None = None) -> dict[str, str]:
        """Headers to send: ``env:NAME`` values read (and registered with the scrubber), the others as written."""
        return {k: resolve_secret(val, search_from=search_from) if val.startswith("env:") else val for k, val in self.headers.items()}


class GpuHost(CrucibleModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    ssh: str  # user@host
    docker_host: str | None = None  # e.g. ssh://user@host ; default derived from ssh
    gpu_arch: str | None = None
    shared: bool = True  # shared machine: run politely (containers only, clean up, check load)


class Config(CrucibleModel):
    model_config = ConfigDict(hide_input_in_errors=True)

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
    # model id -> context window in tokens (prompt plus reply), for a model whose server does not say (M2 R-14):
    # ``LLMClient.context_window`` reads it first, then a self-hosted engine's /models card
    context_windows: dict[str, int] = Field(default_factory=dict)
    llm_budget_usd: float = 1.0  # default hard cap per LLMClient instance
    backends: dict[str, dict[str, Any]] = Field(default_factory=dict)
    root: str | None = None  # directory the config was loaded from (set by loader)
    sources: list[str] = Field(default_factory=list)  # every layer read, lowest first: "user:<path>", "project:<path>", "set:<key>" (set by loader)

    @field_validator("prices")
    @classmethod
    def _price_lists(cls, v: dict[str, list[float]]) -> dict[str, list[float]]:
        bad = sorted(k for k, p in v.items() if not 2 <= len(p) <= 4 or any(x < 0 for x in p))
        if bad:
            raise ValueError(f"prices {bad}: give USD per 1M tokens as [input, output], optionally then cache read and cache write")
        return v

    @field_validator("context_windows")
    @classmethod
    def _positive_windows(cls, v: dict[str, int]) -> dict[str, int]:
        bad = sorted(k for k, n in v.items() if n <= 0)
        if bad:
            raise ValueError(f"context_windows {bad}: give each model's window as a positive number of tokens")
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
    """The nearest ``.env`` from ``start`` up to the root of its git work tree (``start`` alone outside a repository),
    else ``~/.config/crucible/.env``; parsed with python-dotenv's grammar (R-29; vendored in
    ``crucible._vendor.dotenv_parser``, BSD-3-Clause): quotes, escapes, ``export`` and inline comments as python-dotenv
    reads them, no interpolation. A later binding of a key wins, as in python-dotenv; a line it cannot parse is skipped.

    The search used to walk up to ``/`` with no check, so a job in ``/tmp/job`` took its keys from another user's
    ``/tmp/.env`` (M0 audit 2026-10-16 R-5; PATTERN: OpenEnv ``auto/_discovery.py`` reads only files the user owns,
    BSD-3-Clause; mini-swe-agent reads ``.env`` from its user directory, MIT). It now stops where the project file's
    search stops, and a ``.env`` that another user owns or anyone may write raises ConfigError."""
    import io

    from crucible._vendor.dotenv_parser import parse_stream

    for d in [*_project_dirs(start), user_config_path().parent]:
        f = d / ".env"
        if f.is_file():
            st = f.stat()
            if (hasattr(os, "getuid") and st.st_uid != os.getuid()) or st.st_mode & 0o002:
                raise ConfigError(f"{f} is not used for secrets: it belongs to another user or anyone may write it; make it yours "
                                  "and not world-writable (chmod 600), or remove it")
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


def _project_dirs(cwd: Path) -> list[Path]:
    """``cwd`` and its parents up to the root of its git work tree; ``cwd`` alone outside a repository."""
    here = Path(cwd).resolve()
    chain = [here, *here.parents]
    top = next((i for i, d in enumerate(chain) if (d / ".git").exists()), None)
    return chain[: top + 1] if top is not None else chain[:1]


def project_config_path(cwd: Path | None = None) -> Path | None:
    """The project file for ``cwd``: the nearest ``crucible.toml`` from ``cwd`` up to the root of its git work tree
    (``cwd`` alone outside a repository). Raises ConfigError for a ``crucible.toml`` beyond that, above ``cwd``."""
    here = Path(cwd or Path.cwd()).resolve()
    chain = [here, *here.parents]
    inside = _project_dirs(here)
    top = len(inside) - 1 if (inside[-1] / ".git").exists() else None
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
            # compared as the validated numbers: the TOML string "500" was skipped here and then read as 500.0 (R-1)
            cap, value = _number(below.get("llm_budget_usd", Config.model_fields["llm_budget_usd"].default)), _number(value)
            if value is not None and cap is not None and value > cap:
                raise ConfigError(f"{path} raises llm_budget_usd from {cap} to {value}: a project file may lower the spending cap, never "
                                  f"raise it; set it in {user_config_path()} or pass --set llm_budget_usd={value}")
        elif key in have and have[key] != value:
            raise ConfigError(f"{path} overrides {key}, which the user configuration sets: where requests, credentials and code go "
                              "is the operator's to decide; remove it from the project file or change it in the user file")


def _number(v: Any) -> float | None:
    """``v`` as ``Config.llm_budget_usd`` will read it, or None when it is not a number (validation then refuses it)."""
    from pydantic import TypeAdapter, ValidationError

    try:
        return TypeAdapter(float).validate_python(v)
    except ValidationError:
        return None


def _endpoint(base_url: str) -> str:
    from urllib.parse import urlsplit

    u = urlsplit(base_url.strip())
    return f"{u.scheme.lower()}://{u.netloc.lower()}{u.path.rstrip('/')}"


def _credential_pairs(cfg: Config) -> dict[tuple[str, str], str]:
    """``{(endpoint, env:NAME): provider}`` for every credential a provider would send."""
    out: dict[tuple[str, str], str] = {}
    for name, p in cfg.llm_providers.items():
        refs = [p.api_key] if p.api_key not in KEY_PLACEHOLDERS else []
        refs += [v for v in p.headers.values() if v.startswith("env:")]
        for ref in refs:
            out[(_endpoint(p.base_url), ref)] = name
    return out


def _check_credentials(cfg: Config, operator: Config, path: Path) -> None:
    """Refuse a project file whose result sends a credential to an endpoint the operator's layers did not pair it with."""
    allowed = _credential_pairs(operator)
    for (endpoint, ref), provider in sorted(_credential_pairs(cfg).items()):
        if (endpoint, ref) not in allowed:
            raise ConfigError(f"{path} sends {ref} to {endpoint} (llm_providers.{provider}), a pairing the operator's configuration "
                              f"does not make: a project file may not decide where credentials go. Declare the provider in "
                              f"{user_config_path()}, or give a local engine one of {sorted(KEY_PLACEHOLDERS)} as its api_key")


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
    operator: dict[str, Any] = {}  # the layers the operator chose (user file, $CRUCIBLE_CONFIG, --set), over the defaults
    project_path: Path | None = None
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
            project_path = path
        else:
            operator = _merge(operator, doc)
        data = _merge(data, doc)
        sources.append(f"{kind}:{path.resolve()}")
        root = path.parent.resolve()
    if _OVERRIDES:
        data = _merge(data, _OVERRIDES)
        operator = _merge(operator, _OVERRIDES)
        sources += [f"set:{'.'.join(k)}" for k in flatten(_OVERRIDES)]
    data["root"], data["sources"] = str(root or Path(cwd or Path.cwd()).resolve()), sources
    cfg = _validated(data, sources)
    if project_path is not None:
        _check_credentials(cfg, _validated(operator, sources), project_path)
    return cfg


def _validated(data: dict[str, Any], sources: list[str]) -> Config:
    """``Config`` from merged layers; a refusal names the layers read and each setting, never the value it held (A16-R6;
    PATTERN: Harbor ``models/asp.py`` hides inputs on its credential models, Apache-2.0). The pydantic error printed the
    whole refused value, so ``crucible config`` showed a prefixless literal key in full and named no file."""
    try:
        return Config.model_validate(data)
    except ValidationError as e:
        lines = [f"  {'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors(include_input=False, include_url=False)]
        raise ConfigError(f"invalid configuration ({', '.join(sources) or 'built-in defaults'}):\n" + "\n".join(lines)) from None


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()


def reset_config() -> None:
    get_config.cache_clear()
