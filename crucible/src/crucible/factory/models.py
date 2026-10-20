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
- **The Claude builder** (swarm-pipeline/08): ``builder="claude-code:opus-5.5"`` (or ``[backends.factory] builder``) puts
  every ``BUILDER_ROLES`` role on headless Claude Code (``actors.claude_code.ClaudeCodeClient``, the user's Claude plan,
  family ``anthropic``). The book refuses (``SameFamilyError``, at construction and at ``client(role)``) any assignment
  that puts a role of ``INDEPENDENT_ROLES`` (reviewer, realism, second verifier, auditor, hacker, tie-break, judge) on a
  Claude Code client, or on the family of a role Claude Code builds: the reviewer of a task never shares its builder's
  family (V2). ``plan_usage()`` sums the plan accounting of the book's Claude clients for receipts.
- **The student** (``factory.student``) is a solver rung like any other: ``ModelSpec("student", "Qwen/Qwen3.5-4B",
  "qwen")``, ref ``student:Qwen/Qwen3.5-4B``. Its provider ``student`` is an ``llm_providers`` entry (a local mlx-lm/vLLM
  server, Tinker's OpenAI-compatible endpoint, or token-in/token-out), so ``_make`` builds an ``LLMClient`` for it unchanged;
  a sampler with no HTTP endpoint (the Tinker bridge's ``TinkerSampler``) registers with ``student.register_backend`` and
  is recorded as the run's backend (``mlx`` | ``vllm`` | ``tinker``). Only a final backend's numbers count.
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

# The roles that build a task, and the roles that must come from another family than the builder (00 §3.2, V2).
BUILDER_ROLES: tuple[str, ...] = ("planner", "author", "oracle_author", "oracle_builder", "verifier_writer", "mutant_builder", "prompt_author",
                                  "world_expert")
INDEPENDENT_ROLES: tuple[str, ...] = ("reviewer", "realism", "second_verifier", "auditor", "hacker", "tie_break", "judge")
CLAUDE_CODE = "claude-code"  # the provider name of headless Claude Code (not an llm_providers entry: no HTTP endpoint)
CLAUDE_BUILDER = f"{CLAUDE_CODE}:opus-5.5"
# providers whose family is not in the model id
PROVIDER_FAMILIES: dict[str, str] = {CLAUDE_CODE: "anthropic"}


class SameFamilyError(ValueError):
    """A role that must be independent of the builder is assigned the builder's family (or a Claude Code client)."""


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
    "anthropic": "anthropic", "claude": "anthropic", "claude-code": "anthropic",
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
        return ModelSpec(provider, model, str(value.get("family") or PROVIDER_FAMILIES.get(provider) or family_of(model)))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"a model assignment must be 'provider:model', 'model' or a table, got {value!r}")
    provider, sep, model = value.strip().partition(":")
    if not sep or "/" in provider:  # "org/model" (no provider), or a model id that itself has a colon
        provider, model = default_provider, value.strip()
    return ModelSpec(provider, model, PROVIDER_FAMILIES.get(provider) or family_of(model))


class ModelBook:
    """The role -> model table of one factory run, and its LLM clients (one per role, created on first use)."""

    def __init__(self, roles: Mapping[str, Any] | None = None, solvers: Sequence[Any] | None = None, *, config: Config | None = None,
                 budget_usd_per_client: float | None = None, builder: Any = None, builder_kwargs: Mapping[str, Any] | None = None):
        explicit = dict(roles or {})
        builder_roles = {r: builder for r in BUILDER_ROLES if r not in explicit} if builder else {}
        merged = {**DEFAULT_ROLES, **builder_roles, **explicit}
        self.roles: dict[str, ModelSpec] = {r: parse_spec(v) for r, v in merged.items()}
        self.solvers: list[ModelSpec] = [parse_spec(v) for v in (solvers if solvers is not None else SOLVER_RUNGS)]
        self._config = config
        self.budget_usd_per_client = budget_usd_per_client
        self.builder_kwargs = dict(builder_kwargs or {})  # ClaudeCodeClient options (receipts_path, budget_usd, effort, binary)
        self._clients: dict[str, Any] = {}
        self.no_json_mode: tuple[str, ...] = NO_JSON_MODE
        problems = self.family_problems()
        if problems:
            raise SameFamilyError("; ".join(problems))
        bad = [s.ref for s in self.solvers if s.provider == CLAUDE_CODE]
        if bad:
            raise ValueError(f"{bad}: Claude Code builds tasks, it is never a solver rung (no OpenAI tool calls)")

    def family_problems(self) -> list[str]:
        """Every independent role on a Claude Code client, or on the family of a role a Claude Code client builds."""
        out = []
        built: dict[str, str] = {}
        for r, s in self.roles.items():
            if r in BUILDER_ROLES and s.provider == CLAUDE_CODE:
                built.setdefault(s.family, r)
        for r in INDEPENDENT_ROLES:
            s = self.roles.get(r)
            if s is None:
                continue
            if s.provider == CLAUDE_CODE:
                out.append(f"role {r!r} must be a different family from the builder and may never run on Claude Code ({s.ref})")
            elif s.family in built:
                out.append(f"role {r!r} ({s.ref}) shares family {s.family!r} with the Claude Code builder role {built[s.family]!r}")
        return out

    @classmethod
    def from_config(cls, config: Config | None = None, *, overrides: Mapping[str, Any] | None = None, **kw: Any) -> ModelBook:
        cfg = config or get_config()
        fac = dict(cfg.backends.get("factory") or {})
        roles = {**dict(fac.get("roles") or {}), **dict(overrides or {})}
        book = cls(roles, fac.get("solvers"), config=cfg, budget_usd_per_client=kw.get("budget_usd_per_client", fac.get("budget_usd_per_client")),
                   builder=kw.get("builder", fac.get("builder")), builder_kwargs=kw.get("builder_kwargs"))
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
            if spec.provider == CLAUDE_CODE:
                from crucible.actors.claude_code import ClaudeCodeClient

                kw = {"budget_usd": self.budget_usd_per_client, **self.builder_kwargs}
                self._clients[key] = ClaudeCodeClient(spec.ref, role=key.removeprefix("role:"), **kw)
                return self._clients[key]
            self.config.model(spec.ref)  # fails loudly on an unknown provider, before any call
            self._clients[key] = LLMClient(spec.ref, config=self.config, budget_usd=self.budget_usd_per_client)
        c = self._clients[key]
        return PromptJsonClient(c) if spec.model in self.no_json_mode else c  # type: ignore[return-value]

    def client(self, role: str) -> LLMClient:
        spec = self.spec(role)
        if role in INDEPENDENT_ROLES and spec.provider == CLAUDE_CODE:  # also after a later edit of ``roles``
            raise SameFamilyError(f"role {role!r} may never run on the Claude Code builder ({spec.ref})")
        return self._make(f"role:{role}", spec)

    def clients(self, roles: Sequence[str] | None = None) -> dict[str, LLMClient]:
        return {r: self.client(r) for r in (roles or list(self.roles))}

    def solver_clients(self) -> list[tuple[ModelSpec, LLMClient]]:
        return [(s, self._make(f"solver:{s.ref}", s)) for s in self.solvers]

    def plan_usage(self) -> dict[str, Any]:
        """The Claude plan accounting of this book's Claude Code clients, per role and summed (receipts, provenance)."""
        rows = {k.removeprefix("role:"): c.plan_usage() for k, c in self._clients.items() if hasattr(c, "plan_usage")}
        keys = ("calls", "turns", "duration_ms", "duration_api_ms", "cost_equiv_usd", "input_tokens", "output_tokens", "cache_read_tokens",
                "cache_write_tokens", "errors", "sessions")
        return {"by_role": rows, "total": {k: sum(r.get(k) or 0 for r in rows.values()) for k in keys}}

    def observed(self) -> float:
        """Total spend booked by this book's clients (provider-reported cost where given)."""
        return sum(c.spend.usd for c in self._clients.values())

    def ledger(self, cap_usd: float) -> Ledger:
        return Ledger(cap_usd=cap_usd, observe=self.observed)

    def describe(self) -> dict[str, Any]:
        """The assignment for reports (no secrets live here: the provider's key stays an ``env:`` reference)."""
        out = {"roles": {r: {"ref": s.ref, "family": s.family} for r, s in sorted(self.roles.items())},
               "solvers": [{"ref": s.ref, "family": s.family} for s in self.solvers]}
        if any(hasattr(c, "plan_usage") for c in self._clients.values()):
            out["plan_usage"] = self.plan_usage()
        return out


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
