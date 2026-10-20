"""Tinker base models: name, family, context, tool-call format and the pinned price table (swarm-pipeline/09 §2).

Every model the bridge samples or trains is a ``TinkerModel`` looked up by its Tinker id (``resolve``). The registry
answers four questions without a network call:

- **What it costs.** USD per 1M tokens for prefill, cached prefill, sample and train, copied from the Tinker docs
  "Models & Pricing" page (``PRICES_SOURCE``) on ``PRICES_AS_OF``. Where the page showed a limited-time 50% discount the
  discounted price is pinned and the list price kept in ``list_prices`` (the meter books the pinned price; a run report
  can re-price at list). A model missing from the table is refused by the spend meter, never priced at 0.
- **How long a context it holds.** ``context`` tokens. A ``:peft:<n>`` variant is a separate row (same prefill and
  sample price, higher train price). ``with_context(name, tokens)`` picks the cheapest variant that holds ``tokens``
  and refuses when none does: a long-horizon episode cap is checked before anything is sampled.
- **Which family it is.** ``family`` folds org names the way ``factory.models.FAMILY_ALIASES`` does (``Qwen`` -> qwen,
  ``deepseek-ai`` -> deepseek, ``zai-org`` -> glm, ``nvidia`` -> nemotron, ``moonshotai`` -> kimi), so the factory's
  same-family refusal sees one family per publisher. ``spec()`` is the ``factory.models.ModelSpec`` table form
  (``{"provider": "tinker", "model", "family"}``; ``parse_spec`` accepts it).
- **How it calls tools.** ``tool_format`` names the parser ``codec.TinkerChatCodec`` uses on a completion:
  ``qwen_xml`` (``<tool_call><function=f><parameter=p>v</parameter></function></tool_call>``: Qwen3.5 / 3.6 / 3.8 and the
  Nemotron 3 family), ``dsml`` (DeepSeek V4.1 ``<｜DSML｜ calls>``), ``glm`` (GLM-5.3 ``<arg_key>``/``<arg_value>``),
  ``hermes`` (``<tool_call>{json}</tool_call>``: Qwen3). ``text`` means the model's native format is not ported (Kimi's
  special-token sections, gpt-oss harmony): the codec then declares the tools in the system prompt as a plain text
  protocol and parses ``hermes`` JSON. Formats follow tinker-cookbook ``ad3b5c5`` renderers (Apache-2.0, ported in
  ``codec``); which formats were exercised against the live service is in ``tested`` (the smoke, 09 §6).

``thinking`` is the model's default reasoning mode as the HF chat template takes it (``enable_thinking``); the bridge
records the setting it used on every run and turn and defaults it off for the student (``TinkerSampler``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

PRICES_AS_OF = "2026-10-20"
PRICES_SOURCE = "https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/"
STORAGE_USD_PER_GB_MONTH = 0.10  # checkpoint storage (same page); not metered per call: checkpoints are counted instead


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens."""

    prefill: float
    prefill_cached: float
    sample: float
    train: float

    def usd(self, *, prefill: int = 0, prefill_cached: int = 0, sample: int = 0, train: int = 0) -> float:
        return (prefill * self.prefill + prefill_cached * self.prefill_cached + sample * self.sample + train * self.train) / 1e6


@dataclass(frozen=True)
class TinkerModel:
    name: str  # the Tinker id, a ``:peft:<n>`` suffix included
    family: str
    context: int
    prices: Prices
    tool_format: str  # qwen_xml | dsml | glm | hermes | text
    kind: str = "hybrid"  # hybrid (thinking optional) | reasoning (always thinks) | base
    moe: bool = False
    thinking_kw: str | None = "enable_thinking"  # the chat-template switch for reasoning (None: the template has none)
    list_prices: Prices | None = None  # the undiscounted price when the docs showed a limited-time discount
    # tokens that end an assistant turn when the template has no per-message end token (GLM-5.3: the next role token,
    # the model's own eos list in generation_config.json; the first is appended after a cut-off reply)
    stop_tokens: tuple[str, ...] = ()
    # tokens that close an assistant turn in the *history* but must not stop sampling (gpt-oss harmony: ``<|end|>`` ends
    # each channel message; sampling stops at ``<|return|>`` / ``<|call|>``): used only to frame the TITO suffix
    turn_end_tokens: tuple[str, ...] = ()
    # template arguments for "thinking off" on a model without an on/off switch (GLM-5.3: the lowest reasoning effort,
    # the cookbook's closest replacement for a non-thinking mode)
    off_kwargs: tuple[tuple[str, Any], ...] = ()
    tested: tuple[str, ...] = ()  # what the live smoke exercised: sample, tools, train (09 §6)
    notes: str = ""

    @property
    def base(self) -> str:
        """The HF repo / tokenizer name (the Tinker id without its ``:peft:`` variant)."""
        return self.name.split(":", 1)[0]

    @property
    def provider(self) -> str:
        return "tinker"

    @property
    def ref(self) -> str:
        return f"tinker:{self.name}"

    def spec(self) -> dict[str, str]:
        """``factory.models.ModelSpec`` as a table (``parse_spec`` accepts it): the family is explicit, never re-derived."""
        return {"provider": self.provider, "model": self.name, "family": self.family}

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "family": self.family, "context": self.context, "tool_format": self.tool_format, "kind": self.kind,
                "moe": self.moe, "prices_usd_per_m": vars(self.prices), "prices_as_of": PRICES_AS_OF, "tested": list(self.tested)}


def _p(prefill: float, cached: float, sample: float, train: float) -> Prices:
    return Prices(prefill, cached, sample, train)


K64, K256, K128, K32 = 65_536, 262_144, 131_072, 32_768

_ROWS: list[TinkerModel] = []


def _add(name: str, family: str, ctx: int, prices: Prices, fmt: str, *, long: tuple[int, float] | None = None, **kw: Any) -> None:
    """One model row, plus its ``:peft:<n>`` variant (``long`` = (context, train price)) when Tinker offers one."""
    m = TinkerModel(name, family, ctx, prices, fmt, **kw)
    _ROWS.append(m)
    if long is not None:
        lp = kw.get("list_prices")
        _ROWS.append(replace(m, name=f"{name}:peft:{long[0]}", context=long[0], prices=replace(prices, train=long[1]),
                             list_prices=replace(lp, train=long[1] * 2) if lp else None))


# Qwen (one family). Qwen3.6 reuses the Qwen3.5 tokenizer and chat template (cookbook model_info: "identical tokenizer,
# special tokens, preprocessor, and chat template"); Qwen3.8 keeps the tool format and preserves thinking by default.
_add("Qwen/Qwen3.5-4B", "qwen", K64, _p(0.33, 0.066, 1.005, 0.737), "qwen_xml", long=(K256, 1.34))
_add("Qwen/Qwen3.5-9B", "qwen", K64, _p(0.30, 0.06, 0.807, 1.463), "qwen_xml")
_add("Qwen/Qwen3.5-9B-Base", "qwen", K64, _p(0.30, 0.06, 0.807, 1.463), "text", kind="base", thinking_kw=None)
_add("Qwen/Qwen3.5-35B-A3B-Base", "qwen", K64, _p(0.54, 0.108, 1.335, 1.177), "text", kind="base", moe=True, thinking_kw=None)
_add("Qwen/Qwen3.5-397B-A17B", "qwen", K64, _p(1.65, 0.33, 7.50, 6.60), "qwen_xml", moe=True, long=(K256, 12.00))
_add("Qwen/Qwen3.6-35B-A3B", "qwen", K64, _p(0.45, 0.09, 1.335, 1.177), "qwen_xml", moe=True, long=(K256, 2.14),
     notes="the user's build-and-train target (2026-10-20)")
_add("Qwen/Qwen3.6-27B", "qwen", K64, _p(1.86, 0.372, 5.595, 4.103), "qwen_xml", notes="retiring 2026-10-23")
_add("Qwen/Qwen3.8-27B", "qwen", K64, _p(0.6975, 0.1395, 5.595, 4.103), "qwen_xml", long=(K256, 7.46),
     notes="template preserves thinking in history and adds a reasoning-effort instruction")
_add("Qwen/Qwen3-8B", "qwen", K32, _p(0.195, 0.039, 0.60, 0.44), "hermes")
# DeepSeek (Tinker serves V4.1-Flash only as the 256K variant)
_add("deepseek-ai/DeepSeek-V4.1-Flash:peft:262144", "deepseek", K256, _p(0.60, 0.12, 3.60, 4.98), "dsml", moe=True)
_add("deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144", "deepseek", K256, _p(0.60, 0.12, 3.60, 4.98), "dsml", moe=True)
_add("deepseek-ai/DeepSeek-V3.1", "deepseek", K32, _p(1.695, 0.339, 4.215, 3.718), "text", moe=True, notes="retiring 2026-10-23")
# GLM (256K variants only; the template always thinks: no enable_thinking switch, reasoning_effort instead)
_GLM = {"kind": "reasoning", "moe": True, "thinking_kw": None, "stop_tokens": ("<|user|>", "<|observation|>", "<|endoftext|>"),
        "off_kwargs": (("reasoning_effort", "low"),)}
_add("zai-org/GLM-5.3:peft:262144", "glm", K256, _p(3.78, 0.756, 12.15, 14.58), "glm", **_GLM)
_add("zai-org/GLM-5.3-Flash:peft:262144", "glm", K256, _p(0.84, 0.168, 2.11, 5.58), "glm", **_GLM)
# NVIDIA Nemotron (prices under a limited-time 50% discount on PRICES_AS_OF; list prices kept)
_NEMO_SUPER, _NEMO_SUPER_LIST = _p(0.57, 0.114, 1.44, 1.276), _p(1.14, 0.114, 2.88, 2.552)
_add("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "nemotron", K64, _NEMO_SUPER, "qwen_xml", moe=True, long=(K256, 2.32), list_prices=_NEMO_SUPER_LIST)
_add("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16", "nemotron", K64, _NEMO_SUPER, "qwen_xml", moe=True, long=(K256, 2.32),
     list_prices=_NEMO_SUPER_LIST)
_add("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "nemotron", K64, _p(0.184, 0.0368, 0.495, 0.44), "qwen_xml", moe=True, long=(K256, 0.80),
     list_prices=_p(0.368, 0.0368, 0.99, 0.88))
_add("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "nemotron", K64, _p(1.80, 0.36, 6.225, 5.478), "qwen_xml", moe=True, long=(K256, 9.96),
     list_prices=_p(3.60, 0.36, 12.45, 10.956))
# Kimi, gpt-oss: native tool formats not ported, the text protocol stands in
_add("moonshotai/Kimi-K2.6", "kimi", K32, _p(2.205, 0.441, 5.49, 4.84), "text", moe=True, long=(K128, 15.40))
_HARMONY = {"kind": "reasoning", "moe": True, "thinking_kw": None, "stop_tokens": ("<|return|>", "<|call|>"), "turn_end_tokens": ("<|end|>",),
            "off_kwargs": (("reasoning_effort", "low"),)}
_add("openai/gpt-oss-120b", "openai", K32, _p(0.33, 0.066, 0.84, 0.737), "text", long=(K128, 2.33), **_HARMONY)
_add("openai/gpt-oss-20b", "openai", K32, _p(0.18, 0.036, 0.45, 0.396), "text", **_HARMONY)

MODELS: dict[str, TinkerModel] = {m.name: m for m in _ROWS}

# what the live smoke of 2026-10-20 exercised (09 §6); filled from its report, so the registry says what is known to work
TESTED: dict[str, tuple[str, ...]] = {}


def resolve(name: str) -> TinkerModel:
    """The model row for a Tinker id; an unknown id is refused (it has no price, so its spend could not be capped)."""
    if name not in MODELS:
        near = sorted(n for n in MODELS if n.split(":", 1)[0] == name.split(":", 1)[0])
        hint = f"; variants of {name.split(':', 1)[0]}: {near}" if near else ""
        raise KeyError(f"unknown Tinker model {name!r} (the price table, {PRICES_AS_OF}, has no row for it){hint}")
    m = MODELS[name]
    return replace(m, tested=TESTED.get(name, m.tested)) if name in TESTED else m


def variants(base: str) -> list[TinkerModel]:
    return sorted((m for m in MODELS.values() if m.base == base.split(":", 1)[0]), key=lambda m: m.context)


def with_context(name: str, tokens: int) -> TinkerModel:
    """``name`` if it holds ``tokens``, else the cheapest variant of its base that does; none: ``ValueError`` (a long-horizon
    episode cap a model cannot hold is refused before any spend)."""
    m = resolve(name)
    if m.context >= tokens:
        return m
    fits = [v for v in variants(m.base) if v.context >= tokens]
    if not fits:
        raise ValueError(f"{name}: no Tinker variant holds {tokens} tokens (largest: {max(v.context for v in variants(m.base))})")
    return min(fits, key=lambda v: (v.prices.train, v.context))


def check_context(m: TinkerModel, *, episode_tokens: int, max_tokens: int) -> None:
    """Refuse a model whose context cannot hold an episode of ``episode_tokens`` plus one reply of ``max_tokens``."""
    if episode_tokens + max_tokens > m.context:
        raise ValueError(f"{m.name} holds {m.context} tokens; the episode cap {episode_tokens} + one reply {max_tokens} does not fit "
                         f"(variants: {[v.name for v in variants(m.base)]})")
