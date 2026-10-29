"""crucible.factory.synth_wire: synth roles on the swarm, family rules, one spend meter, attribution (M4). No model is called."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from crucible.actors.llm import LLMOutputError
from crucible.factory.models import DEEPSEEK, GPT_OSS, KIMI, QWEN_FRONTIER, TARGET_STUDENT, SameFamilyError, parse_spec
from crucible.factory.synth_wire import SYNTH_ROLES, SynthWire, Usage, WireClient, check_assignment, milestone_spent
from crucible.synth.roles import ROLES


def test_the_default_assignment_passes_both_rule_sets():
    book = check_assignment(SYNTH_ROLES)
    assert set(SYNTH_ROLES) == set(ROLES)
    fams = {r: {parse_spec(m).family for m in ms} for r, ms in SYNTH_ROLES.items()}
    builders = fams["proposer"] | fams["oracle_programmer"] | fams["contract_writer"] | fams["repairer"]
    for r in ("alt_solver", "attacker", "panel", "abstractor"):
        assert not fams[r] & builders and "qwen" not in fams[r]
    assert len(fams["panel"]) == 2 and fams["contract_writer"] != fams["oracle_programmer"]
    assert book.preset == "swarm" and parse_spec(TARGET_STUDENT).family == "qwen"
    assert not any(parse_spec(m).model == parse_spec(TARGET_STUDENT).model for ms in SYNTH_ROLES.values() for m in ms)  # the student is never called


@pytest.mark.parametrize("role, models, why", [
    ("contract_writer", (DEEPSEEK,), "oracle_programmer"),
    ("panel", (GPT_OSS, GPT_OSS), ">= 2 families"),
    ("attacker", (QWEN_FRONTIER, GPT_OSS), "Qwen"),
    ("alt_solver", (KIMI, GPT_OSS), "proposer"),
    ("blind_tester", (KIMI,), "app_coder"),
])
def test_family_violations_raise_same_family_error(role, models, why):
    with pytest.raises(SameFamilyError, match=why):
        check_assignment({**SYNTH_ROLES, role: models})


async def test_wire_client_parses_retries_and_books_usage(tmp_path):
    replies = iter([SimpleNamespace(text="not json", cost_usd=0.001, raw_finish="stop"),
                    SimpleNamespace(text='{"valid": true, "confidence": 0.9}', cost_usd=0.002, raw_finish="stop")])

    class Inner:
        async def complete(self, msgs, **kw):
            return next(replies)

    u = Usage(tmp_path / "usage.jsonl")
    c = WireClient(Inner(), role="panel", model="m", family="openai", usage=u)
    doc = await c.complete_json([{"role": "user", "content": "x"}], schema=["valid"], tag="adjudicate:t1")
    assert doc["valid"] is True and c.calls == 2
    rows = [json.loads(x) for x in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["adjudicate", "adjudicate"] and u.by("stage") == {"adjudicate": 0.003}

    class Bad:
        async def complete(self, msgs, **kw):
            return SimpleNamespace(text="{}", cost_usd=0.0)

    with pytest.raises(LLMOutputError):
        await WireClient(Bad(), role="panel", model="m", family="openai", usage=Usage()).complete_json([{"role": "user", "content": "x"}], schema=["valid"])


def test_wire_builds_clients_on_one_meter_with_a_fake_maker(tmp_path):
    made = []

    def maker(model, role):
        made.append((model, role))
        return SimpleNamespace()

    w = SynthWire(tmp_path, cap_usd=1.5, make=maker)
    roles = w.run_roles()
    assert {c.family for _, c in roles.panel} == {"nemotron", "openai"} and roles.proposer.family == "kimi"
    assert roles.builder_families().isdisjoint({c.family for _, c in roles.alt})
    assert w.meter.cap_usd == 1.5 and w.meter.path == tmp_path / "spend.jsonl"
    assert set(w.worldgen_clients()) == {"tool_designer", "schema_deriver", "app_coder", "blind_tester", "debugger", "test_approver", "seeder"}
    assert all(":" not in m or "peft" in m for m, _ in made)


def test_extract_json_takes_the_last_object_after_plain_text_reasoning():
    from crucible.factory.synth_wire import extract_json

    text = 'Let me look: T-1008 has {"duplicate_of": "T-1001"} so...\nFinal:\n{"tasks": [{"skill": "s", "instruction": "x {y}"}]}'
    assert extract_json(text) == {"tasks": [{"skill": "s", "instruction": "x {y}"}]}
    assert extract_json('{"a": 1}') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no object at all")


def test_milestone_spent_sums_every_receipt(tmp_path):
    for d, usd in (("a/r1", [0.5, 0.25]), ("b/r2", [1.0])):
        p = tmp_path / d / "spend.jsonl"
        p.parent.mkdir(parents=True)
        p.write_text("".join(json.dumps({"kind": "sample", "usd": x}) + "\n" for x in usd))
    assert milestone_spent(tmp_path) == 1.75
