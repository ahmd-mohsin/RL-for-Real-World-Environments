"""crucible.synth.roles: role definitions as data, what each may see, family rules, program/contract parsing (M4)."""

from __future__ import annotations

import pytest

from crucible.synth.contract import Contract, grade
from crucible.synth.program import Fn, Lit, Ref, run
from crucible.synth.roles import (
    BUILDERS,
    ROLES,
    ContractEdit,
    ProposalOut,
    RoleFailed,
    RoleLeak,
    apply_edits,
    ask,
    family_problems,
    parse_program,
    program_doc,
)
from tests.unit import synth_fixtures as fx
from tests.unit.synth_fakes import FakeClient, seq

GOOD = {"proposer": ["kimi"], "oracle_programmer": ["deepseek"], "contract_writer": ["glm"], "repairer": ["deepseek"],
        "alt_solver": ["nemotron", "openai"], "attacker": ["nemotron", "openai"], "panel": ["nemotron", "openai"], "abstractor": ["openai"],
        "tool_designer": ["kimi"], "schema_deriver": ["kimi"], "app_coder": ["kimi"], "blind_tester": ["deepseek"], "debugger": ["kimi"],
        "test_approver": ["nemotron"], "seeder": ["kimi"], "leak_probe": ["inkling"], "leak_panel": ["nemotron", "openai"]}


def test_every_role_is_data_with_a_schema_and_visibility():
    assert set(GOOD) == set(ROLES)
    for r in ROLES.values():
        assert r.system and r.sees and hasattr(r.output, "model_validate")
        assert not set(r.sees) & set(r.never), r.name
    # the rules of part2-paper-core §5 and §7, as data
    assert "contract" in ROLES["attacker"].never and "strategies" in ROLES["attacker"].sees
    assert {"gate_code", "falsifier", "contract"} <= set(ROLES["panel"].never) and "reason_codes" in ROLES["panel"].sees
    assert "gate_code" in ROLES["repairer"].never and "counterexamples" in ROLES["repairer"].sees
    assert "code" in ROLES["blind_tester"].never and "app_coder" in ROLES["blind_tester"].differs_from
    assert set(BUILDERS) <= set(ROLES["alt_solver"].differs_from) and ROLES["panel"].min_families == 2


def test_render_refuses_what_a_role_must_not_see():
    with pytest.raises(RoleLeak, match="never see"):
        ROLES["attacker"].render(instruction="x", tools=[], strategies=[], history=[], contract={"required": []})
    with pytest.raises(RoleLeak, match="not shown"):
        ROLES["panel"].render(task="x", intent={}, trace=[], answer=None, reason_codes=[], oracle={})
    msgs = ROLES["alt_solver"].render(instruction="Close T-1.", tools=[], history=[])
    assert msgs[0]["role"] == "system" and "## instruction\nClose T-1." in msgs[1]["content"]


def test_family_rules():
    assert family_problems(GOOD) == []
    bad = {**GOOD, "contract_writer": ["deepseek"]}  # the oracle programmer's family
    assert any("contract_writer" in p and "oracle_programmer" in p for p in family_problems(bad))
    assert any("panel" in p and ">= 2" in p for p in family_problems({**GOOD, "panel": ["openai"]}))
    assert any("student" in p for p in family_problems({**GOOD, "attacker": ["qwen", "openai"]}))  # no Qwen attacks the Qwen student
    assert any("alt_solver" in p and "proposer" in p for p in family_problems({**GOOD, "alt_solver": ["kimi", "openai"]}))
    assert any("blind_tester" in p for p in family_problems({**GOOD, "blind_tester": ["kimi"]}))
    assert any("leak_panel" in p and ">= 2" in p for p in family_problems({**GOOD, "leak_panel": ["openai"]}))
    assert any("leak_probe" in p and "contract_writer" in p for p in family_problems({**GOOD, "leak_probe": ["glm"]}))  # a builder never probes leaks


def test_checker_overlap_is_reported_and_six_families_clear_it():
    """§14.1 family exhaustion: with five non-Qwen families the panel judges its own families' ALT traces; the sixth
    (inkling) allows an assignment with no overlap."""
    from crucible.synth.roles import checker_overlap

    five = checker_overlap(GOOD)
    assert five and all("panel" in p for p in five)
    six = {**GOOD, "proposer": ["deepseek"], "panel": ["inkling", "kimi"]}  # the swarm's builders: deepseek and glm (Kimi builds Worlds only)
    assert checker_overlap(six) == [] and family_problems(six) == []
    assert ROLES["leak_probe"].never and "contract" in ROLES["leak_panel"].never and ROLES["leak_panel"].min_families == 2
    assert any("test_approver" in p for p in family_problems({**GOOD, "test_approver": ["kimi"]}))


def test_parse_program_and_round_trip():
    doc = {"steps": [{"id": "s1", "tool": "desk.list_tickets", "args": {"status": "open"}},
                     {"id": "s2", "tool": "desk.set_priority", "args": {"ticket_id": {"$fn": "get", "args": [
                         {"$fn": "select", "args": [{"$ref": "s1.tickets"}, {"customer": "bob@example.com"}]}, "0.id"]}, "priority": "high"}}],
           "answer": {"$ref": "s2.priority"}}
    p = parse_program(doc, {"desk.list_tickets", "desk.set_priority"})
    a = p.steps[1].args["ticket_id"]
    assert isinstance(a, Fn) and isinstance(a.args[0], Fn) and isinstance(a.args[0].args[0], Ref) and isinstance(p.steps[0].args["status"], Lit)
    assert parse_program(program_doc(p)) == p
    with pytest.raises(ValueError, match="unknown tool"):
        parse_program({"steps": [{"id": "s1", "tool": "desk.nope"}]}, {"desk.list_tickets"})
    with pytest.raises(ValueError, match="unknown function"):
        parse_program({"steps": [{"id": "s1", "tool": "t", "args": {"x": {"$fn": "eval", "args": []}}}]})
    with pytest.raises(ValueError, match="not resolved"):
        parse_program({"steps": [{"id": "s1", "tool": "t", "args": {"x": [{"$ref": "s0.a"}]}}]})
    with pytest.raises(ValueError, match="names no earlier step"):
        parse_program({"steps": [{"id": "s1", "tool": "t", "args": {"x": {"$ref": "s2.a"}}}]})


async def test_apply_edits_relaxes_and_rejects_bad_edits():
    t = fx.t3_resolve_email()
    w = fx.helpdesk_world()
    r = await run(w, t.oracle, t.instruction)
    from crucible.synth.infer import skeleton

    sk = skeleton(r, fx.HD_N)
    rid = next(p.id for p in sk.required if p.kind == "insert")
    edits = [ContractEdit(op="relax", target=rid, field="body", pred={"op": "regex", "pattern": "(?i).*password reset.*"}),
             ContractEdit(op="allow", pattern={"id": "a_mail", "app": "mail", "coll": "outbox", "kind": "insert", "why": "follow-up mails"}),
             ContractEdit(op="relax", target="nope", pred={"op": "any"}),  # unknown pattern: rejected
             ContractEdit(op="trace", constraint={"id": "m1", "op": "before", "tool": "desk.search_kb", "then": "desk.resolve_ticket",
                                                  "stated": "after you read the manual"}),  # not in the instruction: rejected
             ContractEdit(op="allow", pattern={"id": "a2", "app": "mail", "coll": "outbox", "kind": "insert"})]  # no why: rejected
    res = apply_edits(sk, edits, t.instruction)
    assert res.applied == [0, 1] and [i for i, _ in res.rejected] == [2, 3, 4]
    c = res.contract
    assert c.required[[p.id for p in c.required].index(rid)].where["body"].op == "regex" and [a.id for a in c.allowed] == ["a_mail"]
    assert grade(r, c).passed  # every applied edit only widens what passes
    st = apply_edits(sk, [ContractEdit(op="trace", constraint={"id": "m1", "op": "before", "tool": "desk.search_kb", "then": "desk.resolve_ticket",
                                                               "stated": "Find the knowledge base article"})], t.instruction)
    assert st.applied == [0] and st.contract.trace[0].stated
    dem = apply_edits(sk, [ContractEdit(op="demote", target=rid, why="the email is optional")], t.instruction).contract
    assert rid not in {p.id for p in dem.required} and f"a_{rid}" in {p.id for p in dem.allowed}
    assert isinstance(Contract.model_validate(dem.model_dump()), Contract)


async def test_ask_retries_with_feedback_then_fails():
    good = {"tasks": [{"skill": "s", "instruction": "Resolve ticket T-3 with the refund policy article id.", "given": [], "derive": ["KB-2"]}]}
    c = FakeClient("kimi", seq({"tasks": [{"skill": "s", "instruction": "x"}]}, good))
    out = await ask(c, ROLES["proposer"], retries=1, domain="d", tools=[], world={}, cell="c", count=1)
    assert isinstance(out, ProposalOut) and len(c.calls) == 2 and "too short" in c.calls[1][-1]["content"]

    def check(o):
        raise ValueError("no")

    with pytest.raises(RoleFailed):
        await ask(FakeClient("kimi", seq(good)), ROLES["proposer"], check=check, retries=1, domain="d", tools=[], world={}, cell="c", count=1)


def test_contract_doc_round_trips_every_fixture_contract():
    from crucible.synth.roles import contract_doc, parse_contract

    for code in ("D1", "D2", "D3", "D4", "D6", "D9", "D10", "D11"):
        c = fx.planted(code).contract
        assert parse_contract(contract_doc(c)) == c, code
        assert all("op" in p["value"] for p in contract_doc(c).get("required", []) if "value" in p)  # the discriminator survives
    for f in fx.CORRECT.values():
        c = f().contract
        assert parse_contract(contract_doc(c)) == c
