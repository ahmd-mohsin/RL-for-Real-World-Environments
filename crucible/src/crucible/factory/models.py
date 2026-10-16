"""Role -> model assignment for the task factory (M17; 05 §12, 06 "Machine (Q)").

- A role maps to a ``ModelSpec(provider, model, family)``. The family comes from the model id's prefix
  (``deepseek/deepseek-v4-flash`` -> ``deepseek``), folded through ``FAMILY_ALIASES`` where one family publishes under
  several prefixes. The cross-family rule is checked by the machine validator (``machine.validate_machine(models=...)``)
  against these families, so it is config, not a docstring.
- v1 defaults (05 §12): authoring on ``qwen/qwen3.7-flash``; reviewer, realism, second verifier, auditor (and the family
  hacker) on ``deepseek/deepseek-v4-flash``; tie-breaks on ``z-ai/glm-4.7-flash``; solver rungs ``openai/gpt-oss-20b``,
  ``meta-llama/llama-3.1-8b-instruct``, ``qwen/qwen3.7-flash``.
- Overrides come from ``crucible.toml``: ``[backends.factory.roles]`` (``role = "provider:model"`` or ``"model"``, or a
  table ``{model, provider, family}``) and ``[backends.factory] solvers = [...]``. The provider must be one of
  ``llm_providers`` (default ``openrouter``, ``api_key = "env:OPENROUTER_API_KEY"``): the key is resolved by
  ``LLMClient`` at the first call and never stored or printed here.
- Spend: every ``LLMClient`` books its provider-reported cost (OpenRouter ``usage.cost``) into the running
  ``SpendScope``; the machine settles that into the forge ``Ledger`` per state attempt. ``ModelBook.ledger(cap)`` builds a
  Ledger whose ``observe`` is the sum over the book's clients; ``booked`` books a call made outside a machine.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from crucible.actors.llm import LLMClient, SpendScope
from crucible.core.config import Config, get_config
from crucible.forge.budget import Ledger

DEFAULT_PROVIDER = "openrouter"
AUTHORING = "qwen/qwen3.7-flash"
REVIEW = "deepseek/deepseek-v4-flash"
TIE_BREAK = "z-ai/glm-4.7-flash"
SOLVER_RUNGS = ("openai/gpt-oss-20b", "meta-llama/llama-3.1-8b-instruct", "qwen/qwen3.7-flash")

DEFAULT_ROLES: dict[str, str] = {
    "planner": AUTHORING,  # spec
    "author": AUTHORING,  # World + prompt
    "oracle_author": AUTHORING,
    "verifier_writer": AUTHORING,  # separate context from the author (staging), same family is allowed by 05 §12
    "mutant_builder": AUTHORING,
    "reviewer": REVIEW,
    "realism": REVIEW,
    "second_verifier": REVIEW,
    "auditor": REVIEW,
    "hacker": REVIEW,
    "tie_break": TIE_BREAK,
}

# (M17 e2e) models whose provider-side JSON mode corrupts replies: deepseek-v4-flash with response_format=json_object
# dropped key names ('{": true, ": "..."}') in 2 of 4 realism calls and failed every retry of a state; with JSON mode off,
# 4 of 4 replies were valid JSON. Their clients ask for JSON in the prompt only. Override: [backends.factory] no_json_mode.
NO_JSON_MODE: tuple[str, ...] = ("deepseek/deepseek-v4-flash",)


class PromptJsonClient:
    """An ``LLMClient`` whose ``complete_json`` never sets the provider's JSON mode (the reply is still parsed as JSON)."""

    def __init__(self, client: LLMClient):
        self._client = client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def complete_json(self, messages: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        kw["json_mode"] = False
        return await self._client.complete_json(messages, **kw)


# Prefixes that publish one model family (a cross-family rule must not pass because a family has two org names).
FAMILY_ALIASES: dict[str, str] = {
    "meta-llama": "llama", "meta": "llama",
    "z-ai": "glm", "zhipuai": "glm", "thudm": "glm",
    "qwen": "qwen", "alibaba": "qwen",
    "deepseek": "deepseek", "deepseek-ai": "deepseek",
    "mistralai": "mistral", "mistral": "mistral",
    "google": "gemini",
    "anthropic": "claude",
    "openai": "openai",
}


def family_of(model_id: str) -> str:
    """The model family from the id's prefix (``org/model`` -> ``org``, aliased); an id with no prefix is its own family
    up to the first ``-`` (``llama-3.1-8b`` -> ``llama``)."""
    mid = model_id.strip().lower()
    if ":" in mid and "/" in mid.split(":", 1)[1]:  # "provider:org/model"
        mid = mid.split(":", 1)[1]
    prefix = mid.split("/", 1)[0] if "/" in mid else mid.split("-", 1)[0]
    return FAMILY_ALIASES.get(prefix, prefix)


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    family: str

    @property
    def ref(self) -> str:
        """``provider:model``, the form ``Config.model`` / ``LLMClient`` resolve."""
        return f"{self.provider}:{self.model}"


def parse_spec(value: Any, *, default_provider: str = DEFAULT_PROVIDER) -> ModelSpec:
    """``"provider:org/model"``, ``"org/model"``, a ``ModelSpec``, or ``{"model", "provider"?, "family"?}``."""
    if isinstance(value, ModelSpec):
        return value
    if isinstance(value, Mapping):
        model = str(value["model"])
        provider = str(value.get("provider") or default_provider)
        return ModelSpec(provider, model, str(value.get("family") or family_of(model)))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"a model assignment must be 'provider:model', 'model' or a table, got {value!r}")
    provider, sep, model = value.strip().partition(":")
    if not sep or "/" in provider:  # "org/model" (no provider), or a model id that itself has a colon
        provider, model = default_provider, value.strip()
    return ModelSpec(provider, model, family_of(model))


class ModelBook:
    """The role -> model table of one factory run, and its LLM clients (one per role, created on first use)."""

    def __init__(self, roles: Mapping[str, Any] | None = None, solvers: Sequence[Any] | None = None, *, config: Config | None = None,
                 budget_usd_per_client: float | None = None):
        merged = {**DEFAULT_ROLES, **dict(roles or {})}
        self.roles: dict[str, ModelSpec] = {r: parse_spec(v) for r, v in merged.items()}
        self.solvers: list[ModelSpec] = [parse_spec(v) for v in (solvers if solvers is not None else SOLVER_RUNGS)]
        self._config = config
        self.budget_usd_per_client = budget_usd_per_client
        self._clients: dict[str, LLMClient] = {}
        self.no_json_mode: tuple[str, ...] = NO_JSON_MODE

    @classmethod
    def from_config(cls, config: Config | None = None, *, overrides: Mapping[str, Any] | None = None, **kw: Any) -> ModelBook:
        cfg = config or get_config()
        fac = dict(cfg.backends.get("factory") or {})
        roles = {**dict(fac.get("roles") or {}), **dict(overrides or {})}
        book = cls(roles, fac.get("solvers"), config=cfg, budget_usd_per_client=kw.get("budget_usd_per_client", fac.get("budget_usd_per_client")))
        if "no_json_mode" in fac:
            book.no_json_mode = tuple(fac["no_json_mode"])
        return book

    @property
    def config(self) -> Config:
        return self._config or get_config()

    def spec(self, role: str) -> ModelSpec:
        if role not in self.roles:
            raise KeyError(f"no model is assigned to role {role!r}")
        return self.roles[role]

    def families(self) -> dict[str, str]:
        return {r: s.family for r, s in self.roles.items()}

    def check(self, machine: Mapping[str, Any]) -> list[str]:
        """The machine's problems under this assignment, the cross-family rule included."""
        from crucible.factory.machine import validate_machine

        return validate_machine(dict(machine), models=self.roles)

    def _make(self, key: str, spec: ModelSpec) -> LLMClient:
        if key not in self._clients:
            self.config.model(spec.ref)  # fails loudly on an unknown provider, before any call
            self._clients[key] = LLMClient(spec.ref, config=self.config, budget_usd=self.budget_usd_per_client)
        c = self._clients[key]
        return PromptJsonClient(c) if spec.model in self.no_json_mode else c  # type: ignore[return-value]

    def client(self, role: str) -> LLMClient:
        return self._make(f"role:{role}", self.spec(role))

    def clients(self, roles: Sequence[str] | None = None) -> dict[str, LLMClient]:
        return {r: self.client(r) for r in (roles or list(self.roles))}

    def solver_clients(self) -> list[tuple[ModelSpec, LLMClient]]:
        return [(s, self._make(f"solver:{s.ref}", s)) for s in self.solvers]

    def observed(self) -> float:
        """Total spend booked by this book's clients (provider-reported cost where given)."""
        return sum(c.spend.usd for c in self._clients.values())

    def ledger(self, cap_usd: float) -> Ledger:
        return Ledger(cap_usd=cap_usd, observe=self.observed)

    def describe(self) -> dict[str, Any]:
        """The assignment for reports (no secrets live here: the provider's key stays an ``env:`` reference)."""
        return {"roles": {r: {"ref": s.ref, "family": s.family} for r, s in sorted(self.roles.items())},
                "solvers": [{"ref": s.ref, "family": s.family} for s in self.solvers]}


@asynccontextmanager
async def booked(ledger: Ledger, who: str, *, estimate_usd: float = 0.0) -> AsyncIterator[SpendScope]:
    """Book a call made outside a machine state: reserve ``estimate_usd`` (``BudgetExhausted`` when the cap cannot pay),
    run the block in a ``SpendScope``, settle what the clients booked (also on error)."""
    ledger.reserve(who, estimate_usd)
    scope = SpendScope()
    try:
        with scope:
            yield scope
    finally:
        ledger.settle(who, scope.usd)
