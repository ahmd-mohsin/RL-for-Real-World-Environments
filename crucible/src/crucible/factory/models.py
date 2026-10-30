"""Role -> model assignment for the task factory (M17; 05 §12, 06 "Machine (Q)").

- A role maps to a ``ModelSpec(provider, model, family)``. The family comes from the model id's prefix
  (``deepseek/deepseek-v4-flash`` -> ``deepseek``), folded through ``FAMILY_ALIASES`` where one family publishes under
  several prefixes. The cross-family rule is checked by the machine validator (``machine.validate_machine(models=...)``)
  against these families, so it is config, not a docstring.
- **The default map** (``DEFAULT_ROLES``, preset ``tinker``; the user's decision of 2026-10-20: Claude builds, Tinker
  samples every other role):

  | role(s) | model | family |
  |---|---|---|
  | planner, oracle builder, first verifier writer, prompt writer (author, World expert, mutant builder) | ``claude-code:opus-5.5`` | anthropic |
  | second verifier writer (S9) | ``tinker:deepseek-ai/DeepSeek-V4.1-Flash:peft:262144`` | deepseek |
  | reviewer; trace auditor of the Qwen student | ``tinker:zai-org/GLM-5.3:peft:262144`` | glm |
  | cheap judges and cause labels (``judge``, ``realism``) | ``tinker:zai-org/GLM-5.3-Flash:peft:262144`` | glm |
  | hackers (``hacker``, ``hacker_2``) | DeepSeek-V4.1-Flash, ``tinker:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144`` | deepseek, nemotron |
  | alternatives (``rewriter_deepseek``, ``rewriter_glm``) | DeepSeek-V4.1-Flash, GLM-5.3 | deepseek, glm |
  | tie-break | Nemotron | nemotron (a third family) |
  | fairness / frontier solvers (``frontier``) | ``tinker:Qwen/Qwen3.5-397B-A17B:peft:262144``, DeepSeek-V4.1-Flash | qwen, deepseek |
  | target student | ``tinker:Qwen/Qwen3.6-35B-A3B`` | qwen |

  There is no difficulty ladder: the target is the only student rung (user, 2026-10-20); difficulty is the target's
  band plus the frontier solvers' fairness check. ``tinker`` is a registered sampler backend (``register_backend``;
  ``train/bridges/tinker.sampler.make``, registered on first use by ``backend``); without the optional ``tinker``
  extra ``client`` raises a ``KeyError`` naming the bridge. Nothing here calls Tinker at import.
- **The family rule, enforced at construction** (``family_problems``, ``SameFamilyError``): no independent role (reviewer,
  realism, second verifier, auditor, hackers, tie-break, judge) is on Claude Code, on a builder's family (it would grade a
  task its own family built), or on a student's family (it would audit its own family's traces: the Qwen student is never
  audited by a Qwen model); the hackers span >= 2 families. The machine validator's ``different_family_from`` is the
  second line.
- **OpenRouter is a fallback only**: the v1 map (``OPENROUTER_ROLES``: authoring on ``qwen/qwen3.7-flash``, review on
  ``deepseek/deepseek-v4-flash``, tie-breaks on ``z-ai/glm-4.7-flash``, rungs ``OPENROUTER_SOLVER_RUNGS``) is the preset
  ``openrouter``, and a bare ``org/model`` id still resolves to the ``openrouter`` provider. The default never calls it.
- Overrides come from ``crucible.toml``: ``[backends.factory.roles]`` (``role = "provider:model"`` or ``"model"``, or a
  table ``{model, provider, family}``), ``[backends.factory] solvers = [...]``, ``frontier = [...]`` and ``preset``. An
  HTTP provider must be one of ``llm_providers``: its key is resolved by ``LLMClient`` at the first call and never stored
  or printed here.
- Spend: every ``LLMClient`` books its provider-reported cost (OpenRouter ``usage.cost``) into the running
  ``SpendScope``; the machine settles that into the forge ``Ledger`` per state attempt. ``ModelBook.ledger(cap)`` builds a
  Ledger whose ``observe`` is the sum over the book's clients; ``booked`` books a call made outside a machine.
- **The Claude builder** (swarm-pipeline/08): the builder roles run on headless Claude Code
  (``actors.claude_code.ClaudeCodeClient``, the user's Claude plan, family ``anthropic``); ``builder=`` (or
  ``[backends.factory] builder``) reassigns every ``BUILDER_ROLES`` role at once. ``client(role)`` refuses a Claude Code
  client on an independent role again after a later edit of ``roles``. ``plan_usage()`` sums the plan accounting of the
  book's Claude clients for receipts.
- **The student** (``factory.student``) is a solver rung like any other: the target ``tinker:Qwen/Qwen3.6-35B-A3B``
  through the registered ``tinker`` backend, and for harness development only the local ``student:Qwen/Qwen3.5-4B`` (the
  ``student`` provider: mlx-lm or vLLM, dev-only numerics). The backend (``mlx`` | ``vllm`` | ``tinker``) is recorded
  with every run; only a target model on a final backend counts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from crucible.actors.llm import LLMClient, SpendScope
from crucible.core.config import Config, get_config
from crucible.forge.budget import Ledger

DEFAULT_PROVIDER = "openrouter"  # the fallback for a bare "org/model" id (unused by the default map, which names its provider)
CLAUDE_CODE = "claude-code"  # the provider name of headless Claude Code (not an llm_providers entry: no HTTP endpoint)
CLAUDE_BUILDER = f"{CLAUDE_CODE}:opus-5.5"
TINKER = "tinker"  # a registered sampler backend (``register_backend``), supplied by train/bridges/tinker
VLLM = "vllm"  # a registered sampler backend: our own vLLM server (train/bridges/vllm; DeltaAI GH200)

# ---- the default map (2026-10-20): Claude builds, Tinker samples every other role ------------------------------------------
DEEPSEEK = f"{TINKER}:deepseek-ai/DeepSeek-V4.1-Flash:peft:262144"
GLM = f"{TINKER}:zai-org/GLM-5.3:peft:262144"
GLM_FLASH = f"{TINKER}:zai-org/GLM-5.3-Flash:peft:262144"
NEMOTRON = f"{TINKER}:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144"
QWEN_FRONTIER = f"{TINKER}:Qwen/Qwen3.5-397B-A17B:peft:262144"
TARGET_STUDENT = f"{TINKER}:Qwen/Qwen3.6-35B-A3B"  # the student tasks are built for and post-trained (factory.student)
LADDER: tuple[str, ...] = ()  # no smaller rungs (user, 2026-10-20): the 35B target is the only student
FRONTIER_SOLVERS = (QWEN_FRONTIER, DEEPSEEK)  # fairness / frontier solvers (alternatives, the ten-competent-people test)
SOLVER_RUNGS = (*LADDER, TARGET_STUDENT)

DEFAULT_ROLES: dict[str, str] = {
    "planner": CLAUDE_BUILDER,
    "author": CLAUDE_BUILDER,  # World + prompt (the prompt writer)
    "prompt_author": CLAUDE_BUILDER,
    "oracle_author": CLAUDE_BUILDER,
    "oracle_builder": CLAUDE_BUILDER,
    "verifier_writer": CLAUDE_BUILDER,  # the first verifier writer
    "mutant_builder": CLAUDE_BUILDER,
    "world_expert": CLAUDE_BUILDER,
    "second_verifier": DEEPSEEK,  # S9, blind to the first verifier
    "reviewer": GLM,
    "realism": GLM_FLASH,  # a cheap judge
    "judge": GLM_FLASH,  # cheap judges and cause labels
    "auditor": GLM,  # the trace auditor of the Qwen student: never a Qwen model
    "hacker": DEEPSEEK,  # hackers from >= 2 families
    "hacker_2": NEMOTRON,
    "rewriter_deepseek": DEEPSEEK,  # alternatives (oracle rewrites)
    "rewriter_glm": GLM,
    "tie_break": NEMOTRON,  # a third family against the reviewer (glm) and second verifier (deepseek)
}

# v1 (05 §12, OpenRouter): kept only as a fallback preset (``ModelBook(preset="openrouter")``); the default never calls it
AUTHORING = "qwen/qwen3.7-flash"
REVIEW = "deepseek/deepseek-v4-flash"
TIE_BREAK = "z-ai/glm-4.7-flash"
OPENROUTER_SOLVER_RUNGS = ("openai/gpt-oss-20b", "meta-llama/llama-3.1-8b-instruct", "qwen/qwen3.7-flash")
OPENROUTER_ROLES: dict[str, str] = {
    "planner": AUTHORING, "author": AUTHORING, "oracle_author": AUTHORING,
    "verifier_writer": AUTHORING,  # separate context from the author (staging), same family is allowed by 05 §12
    "mutant_builder": AUTHORING, "reviewer": REVIEW, "realism": REVIEW, "second_verifier": REVIEW, "auditor": REVIEW, "hacker": REVIEW,
    "tie_break": TIE_BREAK,
}
# ---- the swarm map (user, 2026-10-23: "all next steps by my swarm intelligence, not Claude"; every role on Tinker) ------------
# The builders span three families by strength (swarm-pipeline/08 §1.1 has the evidence): DeepSeek-V4.1-Flash, the only
# model we measured at Φ = 1 on a 300-call debugging task and the cheapest 256K builder, holds the long agentic sessions in
# the repository (oracle builder, oracle author); GLM-5.3, the strongest reasoner on the price table, plans and writes the
# first (blind) verifier, a different family from the oracle builder (V2); Kimi-K2.6 (its 128K variant: the 32K base
# cannot hold a session) writes the user's report, the World notes and the mutants: text work over a brief, not 300-call
# repository sessions. Every checker is on a family nothing builds: Nemotron (Ultra for the judgement roles, Super for a
# hacker) and gpt-oss-120b; the fairness solvers are Qwen3.5-397B and Nemotron-Ultra (never a builder's family: a builder's
# family solving its own family's task is no fairness check).
KIMI = f"{TINKER}:moonshotai/Kimi-K2.6:peft:131072"
NEMOTRON_ULTRA = f"{TINKER}:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16:peft:262144"
GPT_OSS = f"{TINKER}:openai/gpt-oss-120b:peft:131072"
# the sixth non-Qwen family on Tinker (2026-10-29): Thinking Machines' Inkling, its own tokenizer and chat format
INKLING = f"{TINKER}:thinkingmachines/Inkling:peft:262144"
INKLING_SMALL = f"{TINKER}:thinkingmachines/Inkling-Small:peft:262144"
SWARM_FRONTIER = (QWEN_FRONTIER, NEMOTRON_ULTRA)
SWARM_ROLES: dict[str, str] = {
    "planner": GLM,
    "verifier_writer": GLM,  # the first (blind) verifier writer: never the oracle builder's family (V2)
    "oracle_builder": DEEPSEEK,
    "oracle_author": DEEPSEEK,
    "author": KIMI,  # World + prompt (the prompt writer)
    "prompt_author": KIMI,
    "world_expert": KIMI,
    "mutant_builder": KIMI,
    "second_verifier": NEMOTRON_ULTRA,  # S9, blind to the first verifier, a family nothing builds
    "reviewer": NEMOTRON_ULTRA,
    "auditor": NEMOTRON_ULTRA,  # the trace auditor of the Qwen student: never Qwen, never a builder's family
    "realism": GPT_OSS,
    "judge": GPT_OSS,
    "hacker": NEMOTRON,
    "hacker_2": GPT_OSS,
    "rewriter_qwen": QWEN_FRONTIER,  # alternatives: families that neither built the oracle nor wrote the verifier
    "rewriter_nemotron": NEMOTRON,
    "tie_break": GPT_OSS,  # a third family against the reviewer and the second verifier (both nemotron)
}
PRESETS: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {"tinker": (DEFAULT_ROLES, SOLVER_RUNGS),
                                                              "openrouter": (OPENROUTER_ROLES, OPENROUTER_SOLVER_RUNGS),
                                                              "swarm": (SWARM_ROLES, SOLVER_RUNGS)}
PRESET_FRONTIER: dict[str, tuple[str, ...]] = {"tinker": FRONTIER_SOLVERS, "openrouter": (), "swarm": SWARM_FRONTIER}
# presets with the strict rules: the oracle builder and the first verifier writer are two families (V2 by family, not only
# by context), and no fairness solver is a builder's family. The legacy maps keep their own: both builders on one model
# (Claude Code, qwen3.7-flash) in separate contexts (05 §12), and Qwen3.5-397B judging fairness of qwen-built P5 tasks
STRICT_PRESETS: frozenset[str] = frozenset({"swarm"})

# The roles that build a task, and the roles that judge, review, hack or audit work (00 §3.2, V2): an independent role is
# never the family of a builder (it would grade its own family's task) nor of a student (it would audit its own family's
# traces), and the hackers span >= 2 families.
BUILDER_ROLES: tuple[str, ...] = ("planner", "author", "oracle_author", "oracle_builder", "verifier_writer", "mutant_builder", "prompt_author",
                                  "world_expert")
INDEPENDENT_ROLES: tuple[str, ...] = ("reviewer", "realism", "second_verifier", "auditor", "hacker", "hacker_2", "tie_break", "judge")
# every role that checks an artifact: the independent roles plus the alternatives (an oracle rewrite by a family that built
# the oracle or wrote the verifier is no independent solution); the legacy rewriter_deepseek / rewriter_glm keep their maps
CHECKER_ROLES: tuple[str, ...] = (*INDEPENDENT_ROLES, "rewriter_qwen", "rewriter_nemotron")
HACKER_ROLES: tuple[str, ...] = ("hacker", "hacker_2")
# providers whose family is not in the model id
PROVIDER_FAMILIES: dict[str, str] = {CLAUDE_CODE: "anthropic"}


class SameFamilyError(ValueError):
    """A role that must be independent of the builder or the student is assigned their family (or a Claude Code client)."""


# ---- sampler backends: clients with no llm_providers HTTP entry (the Tinker bridge registers ``tinker``) ----------------------
@dataclass(frozen=True)
class SamplerBackend:
    """A registered sampler. ``make(model, role=..., thinking=...)`` returns an ``LLMClient``-shaped object (``complete``
    with ``tools``, ``complete_json``, ``spend``, ``model``, ``lock_settings``); ``final``: its student episodes count for
    targeting and benchmarks (``factory.student``)."""

    name: str
    make: Any
    final: bool = False
    doc: str = ""


_BACKENDS: dict[str, SamplerBackend] = {}


def register_backend(name: str, make: Any, *, final: bool = False, doc: str = "") -> SamplerBackend:
    """Register a sampler under a provider name (``tinker``: the bridge's client class, ``final=True``). ``ModelBook`` builds
    every ``<name>:<model>`` role or rung through it; the name is also the backend recorded in run metadata and stamped
    into policy ids, so it is one word."""
    if not name or "@" in name or "-" in name or ":" in name:
        raise ValueError(f"a backend name is one word (a provider name and a policy-id tag): {name!r}")
    b = SamplerBackend(name, make, final, doc)
    _BACKENDS[name] = b
    return b


def _autoregister(name: str) -> None:
    """The ``tinker`` backend registers itself on first use (factory sits above train, so the import is legal here); the
    SDK is an optional extra, so a missing install leaves it unregistered and ``backend`` names the gap."""
    if name == TINKER and name not in _BACKENDS:
        try:
            from crucible.train.bridges.tinker.sampler import make
        except ImportError:
            return
        register_backend(TINKER, make, final=True, doc="Tinker sampling client (train/bridges/tinker)")
    elif name == VLLM and name not in _BACKENDS:
        try:
            from crucible.train.bridges.vllm.sampler import make as make_vllm
        except ImportError:
            return
        register_backend(VLLM, make_vllm, final=True, doc="our own vLLM server (train/bridges/vllm; DeltaAI GH200, doc 14)")


def backend(name: str) -> SamplerBackend:
    _autoregister(name)
    if name not in _BACKENDS:
        raise KeyError(f"no sampler backend {name!r} registered (have {sorted(_BACKENDS)}): the Tinker bridge registers 'tinker' "
                       "(train/bridges/tinker); local servers go through [llm_providers]")
    return _BACKENDS[name]


def backends() -> dict[str, SamplerBackend]:
    return dict(_BACKENDS)


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
    "z-ai": "glm", "zhipuai": "glm", "thudm": "glm", "zai-org": "glm",
    "nvidia": "nemotron",
    "qwen": "qwen", "alibaba": "qwen",
    "deepseek": "deepseek", "deepseek-ai": "deepseek",
    "mistralai": "mistral", "mistral": "mistral",
    "google": "gemini",
    "anthropic": "anthropic", "claude": "anthropic", "claude-code": "anthropic",
    "openai": "openai",
    "thinkingmachines": "inkling",
    "moonshotai": "kimi", "moonshot": "kimi",
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
                 budget_usd_per_client: float | None = None, builder: Any = None, builder_kwargs: Mapping[str, Any] | None = None,
                 preset: str = TINKER, frontier: Sequence[Any] | None = None, check_families: bool = True):
        """``preset``: ``tinker`` (the default map: Claude builds), ``swarm`` (every role on Tinker, the builders over three
        families, no Claude anywhere) or ``openrouter`` (the v1 fallback). ``solvers``: the student rungs (the ladder and the
        target; default the preset's); ``frontier``: the fairness / frontier solvers. ``check_families=False`` leaves the
        family rule to the machine validator alone (a test of that second line)."""
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {sorted(PRESETS)}")
        base_roles, base_rungs = PRESETS[preset]
        explicit = dict(roles or {})
        builder_roles = {r: builder for r in BUILDER_ROLES if r not in explicit} if builder else {}
        merged = {**base_roles, **builder_roles, **explicit}
        self.preset = preset
        self.roles: dict[str, ModelSpec] = {r: parse_spec(v) for r, v in merged.items()}
        self.solvers: list[ModelSpec] = [parse_spec(v) for v in (solvers if solvers is not None else base_rungs)]
        self.frontier: list[ModelSpec] = [parse_spec(v) for v in (frontier if frontier is not None else PRESET_FRONTIER[preset])]
        self._config = config
        self.budget_usd_per_client = budget_usd_per_client
        self.builder_kwargs = dict(builder_kwargs or {})  # ClaudeCodeClient options (receipts_path, budget_usd, effort, binary)
        self._clients: dict[str, Any] = {}
        self.no_json_mode: tuple[str, ...] = NO_JSON_MODE
        problems = self.family_problems() if check_families else []
        if problems:
            raise SameFamilyError("; ".join(problems))
        bad = [s.ref for s in [*self.solvers, *self.frontier] if s.provider == CLAUDE_CODE]
        if bad:
            raise ValueError(f"{bad}: Claude Code builds tasks, it is never a solver rung (no OpenAI tool calls)")
        if preset == "swarm" and (claude := sorted(r for r, s in self.roles.items() if s.provider == CLAUDE_CODE)):
            raise SameFamilyError(f"the swarm preset runs no role on Claude Code: {claude}")

    def family_problems(self) -> list[str]:
        """Every independent role (judge, review, hack, audit) on Claude Code, on a builder's family (it would grade its own
        family's task), or on a student's family (it would audit its own family's traces); and hackers of one family."""
        out = []
        built: dict[str, str] = {}
        for r, s in self.roles.items():
            if r in BUILDER_ROLES:
                built.setdefault(s.family, r)
        students = {s.family: s.ref for s in self.solvers}
        for r in INDEPENDENT_ROLES:
            s = self.roles.get(r)
            if s is None:
                continue
            if s.provider == CLAUDE_CODE:
                out.append(f"role {r!r} must be a different family from the builder and may never run on Claude Code ({s.ref})")
            elif s.family in built:
                out.append(f"role {r!r} ({s.ref}) shares family {s.family!r} with the builder role {built[s.family]!r}")
            elif s.family in students:
                out.append(f"role {r!r} ({s.ref}) shares family {s.family!r} with the student {students[s.family]} whose work it judges")
        for r in CHECKER_ROLES:
            s = self.roles.get(r)
            if r not in INDEPENDENT_ROLES and s is not None and s.family in built:
                out.append(f"role {r!r} ({s.ref}) shares family {s.family!r} with the builder role {built[s.family]!r}")
        hackers = [self.roles[r] for r in HACKER_ROLES if r in self.roles]
        if len(hackers) > 1 and len({h.family for h in hackers}) < 2:
            out.append(f"the hackers {[h.ref for h in hackers]} must span >= 2 families")
        if self.preset in STRICT_PRESETS:
            for s in self.frontier:  # a fairness solver of a builder's family would solve its own family's task
                if s.family in built:
                    out.append(f"frontier solver {s.ref} shares family {s.family!r} with the builder role {built[s.family]!r}")
            ob, vw = self.roles.get("oracle_builder"), self.roles.get("verifier_writer")
            if ob is not None and vw is not None and ob.family == vw.family:
                out.append(f"V2: the oracle builder ({ob.ref}) and the first verifier writer ({vw.ref}) share family {ob.family!r}")
        return out

    @classmethod
    def from_config(cls, config: Config | None = None, *, overrides: Mapping[str, Any] | None = None, **kw: Any) -> ModelBook:
        cfg = config or get_config()
        fac = dict(cfg.backends.get("factory") or {})
        roles = {**dict(fac.get("roles") or {}), **dict(overrides or {})}
        book = cls(roles, fac.get("solvers"), config=cfg, budget_usd_per_client=kw.get("budget_usd_per_client", fac.get("budget_usd_per_client")),
                   builder=kw.get("builder", fac.get("builder")), builder_kwargs=kw.get("builder_kwargs"), preset=kw.get("preset", fac.get("preset", TINKER)),
                   frontier=fac.get("frontier"))
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
            _autoregister(spec.provider)
            if spec.provider in _BACKENDS and spec.provider not in self.config.llm_providers:  # the Tinker bridge's sampler
                from crucible.factory.student import THINKING_DEFAULT

                kind, _, name = key.partition(":")
                kw = {"role": name} if kind == "role" else {"role": kind, "thinking": THINKING_DEFAULT}
                self._clients[key] = backend(spec.provider).make(spec.model, **kw)
                return self._clients[key]
            if spec.provider == TINKER and spec.provider not in self.config.llm_providers:
                backend(TINKER)  # KeyError naming the bridge, before any call
            self.config.model(spec.ref)  # fails loudly on an unknown provider, before any call
            self._clients[key] = LLMClient(spec.ref, config=self.config, budget_usd=self.budget_usd_per_client)
        c = self._clients[key]
        return PromptJsonClient(c) if spec.model in self.no_json_mode else c  # type: ignore[return-value]

    def client(self, role: str) -> LLMClient:
        spec = self.spec(role)
        if role in CHECKER_ROLES and spec.provider == CLAUDE_CODE:  # also after a later edit of ``roles``
            raise SameFamilyError(f"role {role!r} may never run on the Claude Code builder ({spec.ref})")
        return self._make(f"role:{role}", spec)

    def builder_families(self) -> dict[str, list[str]]:
        """family -> the builder roles on it (the producers every checker must be independent of)."""
        out: dict[str, list[str]] = {}
        for r, s in self.roles.items():
            if r in BUILDER_ROLES:
                out.setdefault(s.family, []).append(r)
        return out

    def clients(self, roles: Sequence[str] | None = None) -> dict[str, LLMClient]:
        return {r: self.client(r) for r in (roles or list(self.roles))}

    def solver_clients(self) -> list[tuple[ModelSpec, LLMClient]]:
        return [(s, self._make(f"solver:{s.ref}", s)) for s in self.solvers]

    def frontier_clients(self) -> list[tuple[ModelSpec, LLMClient]]:
        return [(s, self._make(f"frontier:{s.ref}", s)) for s in self.frontier]

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
        out = {"preset": self.preset, "roles": {r: {"ref": s.ref, "family": s.family} for r, s in sorted(self.roles.items())},
               "solvers": [{"ref": s.ref, "family": s.family} for s in self.solvers],
               "frontier": [{"ref": s.ref, "family": s.family} for s in self.frontier]}
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
