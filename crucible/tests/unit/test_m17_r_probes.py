"""M17 owner R: floors, executed requirement probes and the differential, on a fake file World and on a forged toy task.

Adversarial cases: a verifier that only checks a file exists passes every floor but is caught by the probes, which name
the requirements whose checks are weak; a verifier that pays any answer fails the constant-answer floor; a second
CheckSet without the scope checks disagrees with the first on the scope-violating end states."""

from __future__ import annotations

import asyncio
import math

import pytest

from crucible.factory.cards import CheckRef, Requirement, RequirementLedger
from crucible.factory.differential import EndState, compare, differential
from crucible.factory.floors import FLOOR_KINDS, Graded, floor_solutions, floors_for_task, graded_from, grader, run_floors
from crucible.factory.probes import LIBRARY, PROBE_KINDS, LLMProbeAuthor, Probe, ProbeContext, build_probes, footprint, probe_verifier, run_probes
from tests.unit.m17_r_fakes import LEDGER_STRONG, LEDGER_WEAK, ORACLE, PROMPT, READS, REQS, STRONG, WEAK, ScriptedClient, user_payload, verifier


def ctx(**kw) -> ProbeContext:
    base = {"oracle": ORACLE, "prompt": PROMPT, "tolerances": {"r_total": 0.01}, "other_ids": {"path": ["notes.txt"]}, "read_calls": READS}
    return ProbeContext(**{**base, **kw})


def header_alternative_author() -> LLMProbeAuthor:
    """A model author that writes a valid alternative for the header (header set before the total) and nothing else."""

    def reply(messages):
        p = user_payload(messages)
        if p["requirement"]["id"] == "r_header" and "alternative" in p["want"]:
            ref = p["reference_solution"]
            return {"probes": [{"solution": [ref[0], ref[2], ref[1]], "description": "header before total"}]}
        return {"probes": []}

    return LLMProbeAuthor(ScriptedClient(reply))


# --- floors ---------------------------------------------------------------------------------------------------------


async def test_floors_generated_for_a_tool_call_task_all_zero():
    floors = floor_solutions(oracle=ORACLE, prompt=PROMPT, noop=[])
    assert {f.kind for f in floors} == set(FLOOR_KINDS)
    rep = await run_floors(floors, verifier(STRONG))
    assert rep.ok, rep.reasons()
    card = rep.card()
    assert set(card) == set(FLOOR_KINDS) and all(v == 0.0 for v in card.values())
    assert rep.rows["original_state"]["same_as"] == "no_op"


async def test_floor_catches_a_verifier_that_pays_any_answer():
    async def pays_answers(sol):
        done = [a for a in sol if a.get("kind") == "done" and a.get("answer")]
        return Graded(1.0 if done else 0.0, {"answered": bool(done)})

    rep = await run_floors(floor_solutions(oracle=ORACLE, prompt=PROMPT, noop=[]), pays_answers)
    assert not rep.ok
    assert {rep.rows[i]["kind"] for i in rep.above_zero} == {"constant_answer", "copy_of_reference"}


async def test_floor_zero_rule_is_strict_and_unmeasured_never_passes():
    for bad in (float("nan"), 0.25, -0.0 + 1e-9):
        rep = await run_floors(floor_solutions(oracle=ORACLE, prompt=PROMPT, noop=[]), lambda s, b=bad: _const(b))
        assert not rep.ok
    masked = await run_floors(floor_solutions(oracle=ORACLE, prompt=PROMPT, noop=[]), lambda s: _const(None))
    assert not masked.ok and masked.unmeasured and math.isnan(masked.card()["no_op"])
    # a non-tool-call task without the caller's floors reports them missing, never skipped
    rep = await run_floors(floor_solutions(oracle="42", noop=""), lambda s: _const(0.0))
    assert not rep.ok and {"missing:empty_action", "missing:constant_answer", "missing:original_state", "missing:copy_of_reference"} <= set(rep.unmeasured)
    rep = await run_floors(floor_solutions(oracle="42", noop="", empty="", constant_of=lambda c: c, original_state="", copy_is_task=True), lambda s: _const(0.0))
    assert rep.ok and rep.rows["copy_of_reference"]["status"] == "not_applicable"


async def _const(v):
    return Graded(v)


def test_graded_from_reads_every_grader_shape():
    assert graded_from(None).masked and graded_from(True).reward == 1.0 and graded_from(0).reward == 0.0
    g = graded_from({"reward": 1.0, "checks": {"a": True, "b": False}, "fingerprint": "x"})
    assert g.failing() == ["b"] and g.fingerprint == "x" and g.passed() is True
    with pytest.raises(TypeError):
        graded_from(object())


# --- probes on the fake World ---------------------------------------------------------------------------------------


def test_library_covers_every_kind_and_footprints_are_specific():
    assert set(LIBRARY) == set(PROBE_KINDS)
    c = ctx()
    assert footprint(REQS[0], c) == [0] and footprint(REQS[1], c) == [1] and footprint(REQS[2], c) == [2]
    assert footprint(REQS[3], c) == [0, 1, 2]  # scope: every write
    assert footprint(REQS[0], ProbeContext(oracle=ORACLE, footprints={"r_file": [2]})) == [2]


async def test_strong_verifier_kills_every_mutant_and_passes_every_alternative():
    rep = await probe_verifier(LEDGER_STRONG, ctx(), verifier(STRONG), author=header_alternative_author())
    assert rep.ok, rep.reasons()
    assert rep.oracle_ok is True
    k, n = rep.mutant_kill
    assert k == n and n >= 4
    a, an = rep.alternatives_pass
    assert a == an and an >= 4
    assert rep.per_requirement["r_scope"].equivalent  # a write on a file that does not exist changes nothing: excluded
    assert all(not x.off_target for x in rep.per_requirement.values())  # every mutant is killed by its own requirement's check


async def test_existence_only_verifier_is_caught_and_the_weak_requirements_named():
    rep = await probe_verifier(LEDGER_WEAK, ctx(), verifier(WEAK), author=header_alternative_author())
    assert not rep.ok
    assert rep.weak_requirements == ["r_header", "r_scope", "r_total"]
    assert rep.per_requirement["r_file"].kill.rate == 1.0  # deleting the file is the one thing it sees
    assert any("r_total" in r and "weak" in r for r in rep.reasons())
    k, n = rep.mutant_kill
    assert 0 < k < n


async def test_a_check_that_grades_the_oracle_internals_rejects_an_alternative():
    strict = {**STRONG, "exact_order": lambda s: True}

    async def order_bound(sol):  # also requires the oracle's own step order
        g = await verifier(strict)(sol)
        same = [a.get("function_name") for a in sol if a.get("function_name") != "fs.read"] == [a["function_name"] for a in ORACLE]
        checks = {**g.checks, "exact_order": same}
        return Graded(1.0 if all(checks.values()) else 0.0, checks, g.fingerprint)

    led = RequirementLedger(requirements=REQS, checks=[*LEDGER_STRONG.checks, CheckRef(id="exact_order", requirement_ids=["r_header"])])
    rep = await probe_verifier(led, ctx(), order_bound, author=header_alternative_author())
    assert rep.strict_requirements == ["r_header"]
    rej = rep.per_requirement["r_header"].rejected_alternatives[0]
    assert rej["failing"] == ["exact_order"] and rej["rejected_by_requirements"] == ["r_header"]


async def test_unprobed_requirements_fail_and_the_llm_author_fills_gaps():
    led = RequirementLedger(requirements=[*REQS, Requirement(id="r_native", text="Use the compiled report writer.", prompt_span=(0, 5), kind="compiled_path")],
                            checks=[*LEDGER_STRONG.checks, CheckRef(id="native", requirement_ids=["r_native"])])
    rep = await probe_verifier(led, ctx(), verifier(STRONG))
    assert not rep.ok
    assert any("r_native" in r and "unprobed" in r for r in rep.reasons())
    assert any("r_header" in r and "alternatives" in r for r in rep.reasons())  # format alternatives need the model author

    def reply(messages):
        p = user_payload(messages)
        ref = p["reference_solution"]
        if p["requirement"]["id"] != "r_native":
            return {"probes": []}
        if "mutant" in p["want"]:
            return {"probes": [{"solution": "not a list", "description": "bad shape"}, {"solution": ref[:1], "description": "python fallback"}]}
        return {"probes": [{"solution": [READS[0], *ref], "description": "read first"}]}

    author = LLMProbeAuthor(ScriptedClient(reply))
    probes, _ = await build_probes(led, ctx(), author=author)
    native = [p for p in probes if p.requirement_id == "r_native"]
    assert {p.role for p in native} == {"mutant", "alternative"} and all(p.source == "llm" for p in native)
    assert author.rejected and author.rejected[0]["why"].startswith("not a tool-call list")


async def test_masked_and_equivalent_probes_are_not_kills():
    async def g(sol):
        return None if sol == "masked" else Graded(1.0, {}, "fp0")

    probes = [Probe("m1", "r", "mutant", "x", "masked"), Probe("m2", "r", "mutant", "x", "same", can_be_equivalent=True),
              Probe("m3", "r", "mutant", "x", "survivor")]
    rep = await run_probes(probes, g, oracle="o", requirement_ids=["r", "r_unprobed"])
    r = rep.per_requirement["r"]
    assert r.unmeasured == ["m1"] and r.equivalent == ["m2"] and r.kill.pair() == (0, 1) and r.survivors[0]["id"] == "m3"
    assert any("r_unprobed" in x for x in rep.reasons())


# --- the toy pack ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_task():
    from crucible_pack_toy.forge.world_synth import forge

    col: list = []
    asyncio.run(forge(collect=col))
    return next(t for t, _ in col if t.task_id.endswith("t4_decoy"))


KINDS = {"refund": "tolerance", "note": "laziness", "status": "behaviour", "policy": "behaviour", "scope": "side_effect_scope"}


def toy_ledger(t, *, drop: str = "") -> RequirementLedger:
    reqs = []
    for p in t.properties:
        i = t.instruction.index(p.text)
        reqs.append(Requirement(id=p.id, text=p.text, prompt_span=(i, i + len(p.text)), kind=KINDS[p.id]))
    return RequirementLedger(requirements=reqs, checks=[CheckRef(id=c.id, requirement_ids=list(c.states), gating=True)
                                                        for c in t.cs.checks if not (drop and c.id.startswith(drop))])


def toy_ctx(t) -> ProbeContext:
    return ProbeContext(oracle=t.oracle, prompt=t.instruction, tolerances={"refund": 0.004},
                        footprints={"policy": [0, 1]},  # the approval request and the refund it covers
                        read_calls=[{"kind": "tool_call", "tool_call_id": "r0", "function_name": "tickets.get_ticket", "arguments": {"ticket_id": 41}}])


def _weakened(t, prefix: str):
    from crucible_pack_toy.forge.task import ToyForgedTask

    cs = t.cs.model_copy(update={"checks": [c for c in t.cs.checks if not c.id.startswith(prefix)]})
    return ToyForgedTask(t.spec, cs, t.oracle, t.negatives, pack_root=t.pack_root, world_file=t.world_file, scenario=t.sc, canary=t.canary)


async def test_toy_floors_all_zero(toy_task):
    floors = floors_for_task(toy_task)
    assert any(f.kind == "authored" for f in floors)  # A2's naive floor is reused
    rep = await run_floors(floors, grader(toy_task))
    assert rep.ok, rep.reasons()


async def test_toy_probes_strong_then_scope_checks_removed(toy_task):
    rep = await probe_verifier(toy_ledger(toy_task), toy_ctx(toy_task), grader(toy_task))
    assert rep.oracle_ok
    assert rep.ok, rep.reasons()
    assert {p for p, x in rep.per_requirement.items() if x.off_target} <= {"policy", "scope"}  # a wrong ticket also breaks the refund
    assert rep.mutant_kill[1] >= 5
    weak = _weakened(toy_task, "scope/")
    rep2 = await probe_verifier(toy_ledger(toy_task, drop="scope/"), toy_ctx(toy_task), grader(weak))
    assert "scope" in rep2.weak_requirements


async def test_toy_differential_finds_the_scope_disagreements(toy_task):
    from crucible_pack_toy.forge.task import plan_for

    second = toy_task.cs.model_copy(update={"checks": [c for c in toy_task.cs.checks if not c.id.startswith("scope/")]})
    plan2 = plan_for(second)

    async def g2(sol):
        return await toy_task._run(list(sol), plan=plan2)

    extra = {"kind": "tool_call", "tool_call_id": "x9", "function_name": "tickets.add_note", "arguments": {"ticket_id": 40, "body": "Also checked."}}
    states = [EndState("oracle", toy_task.oracle), EndState("collateral_note", [*toy_task.oracle, extra])]
    states += [EndState(n.id, n.actions) for n in toy_task.negatives]
    rep = await differential(states, grader(toy_task), g2, ledger_a=toy_ledger(toy_task), ledger_b=toy_ledger(toy_task, drop="scope/"))
    assert rep.compared == len(states) and not rep.state_mismatch
    ids = {d.item for d in rep.disagreements}
    assert ids == {"collateral_note"}  # a scope-only violation: the other negatives also fail an outcome check both share
    assert all(d.primary is False and d.secondary is True for d in rep.disagreements)
    assert rep.card == (len(ids), len(states)) and not rep.ok
    d = rep.open()[0]
    rep.assign(d.item, "second_verifier", "the second CheckSet does not check scope")
    assert len(rep.open()) == len(ids) - 1
    with pytest.raises(ValueError):
        rep.assign(d.item, "nobody", "x")  # type: ignore[arg-type]


async def test_differential_rules_on_the_fake_world():
    states = [EndState("o", ORACLE), EndState("lazy", ORACLE[:1]), EndState("wrong_total", [ORACLE[0], {**ORACLE[1], "arguments": {"path": "report.csv", "total": 50}}, ORACLE[2]])]
    rep = await differential(states, verifier(STRONG), verifier(WEAK), ledger_a=LEDGER_STRONG, ledger_b=LEDGER_WEAK)
    assert {d.item for d in rep.disagreements} == {"lazy", "wrong_total"}
    wt = next(d for d in rep.disagreements if d.item == "wrong_total")
    assert wt.requirements == ["r_total"] and wt.primary_failing == ["total_ok"]
    assert rep.agree_pass == 1
    assert compare("x", Graded(1.0, {}, "a"), Graded(1.0, {}, "b")) == "state_mismatch"
    assert compare("x", Graded(None), Graded(1.0)) == "unmeasured"
    assert compare("x", Graded(1.0, {}, "a"), Graded(1.0, {}, "a")) is None
