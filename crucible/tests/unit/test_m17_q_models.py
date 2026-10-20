"""M17 agent Q: role -> model assignment (factory/models.py). No network: clients are built, never called."""

from __future__ import annotations

import pytest

from crucible.actors.llm import LLMClient, book_scoped
from crucible.core.config import Config
from crucible.factory import models
from crucible.factory.models import ModelBook, ModelSpec, booked, family_of, parse_spec
from crucible.forge.budget import BudgetExhausted, Ledger


def test_v1_defaults_and_families() -> None:
    b = ModelBook(preset="openrouter")  # the v1 map, kept as a fallback preset (the default is Claude + Tinker)
    assert b.spec("author") == ModelSpec("openrouter", "qwen/qwen3.7-flash", "qwen")
    for r in ("reviewer", "realism", "second_verifier", "auditor"):
        assert b.spec(r).model == "deepseek/deepseek-v4-flash" and b.spec(r).family == "deepseek"
    assert b.spec("tie_break").family == "glm"
    assert [s.model for s in b.solvers] == ["openai/gpt-oss-20b", "meta-llama/llama-3.1-8b-instruct", "qwen/qwen3.7-flash"]
    assert [s.family for s in b.solvers] == ["openai", "llama", "qwen"]
    assert family_of("openrouter:z-ai/glm-4.7-flash") == "glm" == family_of("thudm/glm-4-9b")
    assert family_of("llama-3.1-8b") == "llama"


def test_parse_spec_forms() -> None:
    assert parse_spec("vllm:qwen/qwen3-8b") == ModelSpec("vllm", "qwen/qwen3-8b", "qwen")
    assert parse_spec("qwen/qwen3-8b").provider == "openrouter"
    assert parse_spec({"model": "my-ft", "provider": "vllm", "family": "qwen"}).family == "qwen"
    with pytest.raises(ValueError):
        parse_spec("")


def test_from_config_overrides_and_clients() -> None:
    cfg = Config(backends={"factory": {"roles": {"auditor": "openrouter:mistralai/mistral-small"}, "solvers": ["openai/gpt-oss-20b", "qwen/qwen3-8b"]}})
    b = ModelBook.from_config(cfg, budget_usd_per_client=0.5)
    assert b.spec("auditor").family == "mistral" and len(b.solvers) == 2
    c = b.client("auditor")
    assert isinstance(c, LLMClient) and c.model == "mistralai/mistral-small" and c.budget_usd == 0.5
    assert b.client("auditor") is c  # one client per role
    assert b.describe()["roles"]["auditor"] == {"ref": "openrouter:mistralai/mistral-small", "family": "mistral"}
    assert "key" not in str(b.describe()).lower()
    bad = ModelBook({"author": "nowhere:qwen/x"}, config=cfg)
    with pytest.raises(KeyError):
        bad.client("author")


async def test_spend_books_into_the_forge_ledger() -> None:
    ledger = Ledger(cap_usd=0.05)
    async with booked(ledger, "librarian", estimate_usd=0.01) as scope:
        book_scoped(0.03)  # what LLMClient books from OpenRouter's usage.cost
    assert scope.usd == pytest.approx(0.03) and ledger.committed == pytest.approx(0.03) and not ledger.reserved
    with pytest.raises(BudgetExhausted):
        async with booked(ledger, "librarian", estimate_usd=0.05):
            pass
    b = ModelBook()
    assert b.ledger(1.0).observe() == 0.0 and models.OPENROUTER_ROLES["verifier_writer"] == "qwen/qwen3.7-flash"
