"""RESKILL (meta.reskill + rsi.TaskLoop): a valid task re-labelled with the skill its evidence shows, kept only if it is
a good task under the new label; RS1–RS6 each refusing what it must; the move order; skill-preserving DESCEND knobs. A
scripted deriver and student (no model is called)."""

from __future__ import annotations

import json

import pytest

from crucible.synth.meta.reskill import INTENT_DERIVER, IntentProposal, check_proposal, derivation_evidence, domain_skills, swarm_deriver
from crucible.synth.meta.rsi import TaskRules
from crucible.synth.roles import RoleFailed, RoleLeak
from tests.unit.synth_fakes import FakeClient
from tests.unit.test_synth_taskloop import BASE, ORACLE, _loop

SKILLS = [BASE.intent.skill, "multi-document reconciliation", "lookup-and-apply"]
ADV = IntentProposal(skill="multi-document reconciliation", derive=["price_variance"], why="it reconciles the invoice with the purchase order")


def _with(loop, proposal=ADV, *, p3=True, leak=(), novel=True):
    calls = []

    async def derive(task, ctx):
        calls.append(task.id)
        return proposal

    async def p3c(t):
        return p3, ([] if p3 else ["EVID: off intent"])

    async def lk(t):
        return list(leak)

    async def nv(t, cell, pool):
        return novel, ([] if novel else ["near duplicate of a pool task"])

    loop.derive_intent, loop.skills_of, loop.leak_check, loop.novel_check, loop.p3_check = derive, (lambda t: SKILLS), lk, nv, p3c
    return calls


async def test_a_mislabelled_valid_task_is_reskilled_kept_and_can_be_rolled_back(tmp_path):
    loop, task, _ = _loop(tmp_path, tag="band", rules=TaskRules(rollouts=8, kappa=50.0, looks=(8, 16, 24)))
    _with(loop)
    (d,) = await loop.run_typed_round(1)
    assert d.move == "RESKILL" and d.accepted and d.new_version == task.version + 1, d.why
    new = loop.tasks[task.id]
    assert new.intent.skill == ADV.skill and new.intent.derive == ["price_variance"] and "RESKILL from" in new.intent.notes
    lin = loop.lineages[task.id]
    assert lin.reskilled and lin.original["skill"] == BASE.intent.skill and lin.status in ("kept", "reserved")
    assert set(d.rollouts["acceptance"]).isdisjoint(d.rollouts["diagnosis"])  # RS4: fresh rollouts
    led = [json.loads(x)["rec"] for x in (tmp_path / "loop" / "probe_ledger.jsonl").read_text().splitlines()]
    rs = [r for r in led if r["metrics"].get("move") == "RESKILL"]
    assert rs and rs[0]["metrics"]["reskill"]["from"]["skill"] == BASE.intent.skill  # RS6: the original label is in the ledger
    old = loop.rollback_reskill(task.id, round=2)
    assert loop.tasks[task.id].intent.skill == BASE.intent.skill and old.version == task.version and lin.reskilled


async def test_a_too_hard_task_reskilled_to_a_harder_skill_is_reserved_under_its_honest_label(tmp_path):
    loop, task, _ = _loop(tmp_path, tag="hard", rules=TaskRules(rollouts=8, kappa=50.0))
    _with(loop)
    (d,) = await loop.run_typed_round(1)
    assert d.move == "RESKILL" and d.accepted and d.acceptance_band["decision"] == "too_hard" and loop.lineages[task.id].status == "reserved"


@pytest.mark.parametrize(("kw", "rule"), [({"proposal": IntentProposal(skill="an invented skill", derive=["price_variance"])}, "RS1"),
                                          ({"proposal": IntentProposal(skill="lookup-and-apply", derive=["not-written"])}, "RS2"),
                                          ({"p3": False}, "RS3"), ({"leak": ["the instruction states the reason"]}, "RS3"), ({"novel": False}, "RS4")])
async def test_each_reskill_rule_refuses_and_the_loop_moves_on(tmp_path, kw, rule):
    loop, task, _ = _loop(tmp_path, tag="hard", rules=TaskRules(rollouts=8, kappa=50.0))
    calls = _with(loop, **kw)
    (d,) = await loop.run_typed_round(1)
    assert d.move == "DESCEND" and d.reskill is not None and d.reskill["refused_by"][0].startswith(rule), (d.move, d.reskill)
    assert loop.tasks[task.id].intent.skill == BASE.intent.skill  # the label is unchanged
    d2 = await loop.step(loop.tasks[task.id], 2)  # RS6: one attempt per lineage
    assert calls == [task.id] and d2.reskill is None


async def test_the_pool_trust_region_and_the_round_cap_bound_reskills(tmp_path):
    loop, _, _ = _loop(tmp_path / "a", tag="band", rules=TaskRules(rollouts=8, kappa=0.0001, looks=(8, 16, 24)))
    _with(loop)
    loop.add_tasks([BASE.model_copy(update={"id": f"{BASE.id}|other"}, deep=True)])
    (d, _d2) = await loop.run_typed_round(1)
    assert d.reskill is not None and any(r.startswith("RS5: pool cell KL") for r in d.reskill["refused_by"])
    loop2, _task2, _ = _loop(tmp_path / "b", tag="band", rules=TaskRules(rollouts=8, kappa=50.0, looks=(8, 16, 24), reskill_cap=0.2))
    _with(loop2)
    t3 = BASE.model_copy(update={"id": f"{BASE.id}|band2"}, deep=True)
    loop2.add_tasks([t3])
    ds = await loop2.run_typed_round(1)
    moves = [x.move for x in ds]
    assert moves.count("RESKILL") == 1 and any(x.reskill and any(r.startswith("RS5: this round") for r in x.reskill["refused_by"]) for x in ds)


async def test_the_oracle_and_repair_come_before_reskill(tmp_path):
    loop, _task, _ = _loop(tmp_path, tag="hard", oracle=ORACLE(truths=[lambda t: "wrong"]))
    calls = _with(loop)
    (d,) = await loop.run_typed_round(1)
    assert d.move == "REPAIR" and not calls


async def test_a_confirmed_label_is_no_reskill():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        loop, _, _ = _loop(Path(td), tag="hard")
        _with(loop, proposal=IntentProposal(skill=BASE.intent.skill, given=[], derive=list(BASE.intent.derive)))
        (d,) = await loop.run_typed_round(1)
        assert d.move == "DESCEND" and "confirms the label" in " ".join(d.why) and not loop.lineages[d.task_id].reskilled


async def test_the_deriver_reads_evidence_only_and_proposals_are_checked():
    from crucible.synth.falsify import FalsifyContext

    for k in ("pass_rate", "rollouts", "current_skill", "band"):
        with pytest.raises(RoleLeak):
            INTENT_DERIVER.render(**{k: 1})
    ctx = FalsifyContext()
    ev = await derivation_evidence(BASE, ctx)
    assert {"instruction", "tools", "oracle_program", "taint", "effects"} <= set(ev) and "price_variance" in json.dumps(ev["effects"])
    assert check_proposal(ADV, SKILLS, ev) == []
    stated = BASE.model_copy(update={"instruction": BASE.instruction + " The reason code is price_variance."})
    assert any(r.startswith("RS2") for r in check_proposal(ADV, SKILLS, await derivation_evidence(stated, FalsifyContext())))  # lookup, not derivation
    replies = iter([{"skill": "made-up", "derive": []}, {"skill": "multi-document reconciliation", "derive": ["price_variance"], "why": "w"}])
    c = FakeClient("openai", lambda m, n: next(replies))
    skills = [{"id": s, "description": s} for s in SKILLS]
    p = await swarm_deriver(c, lambda t: skills)(BASE, ctx)
    assert p.skill == "multi-document reconciliation" and p.by == "openai" and len(c.calls) == 2  # the invalid skill went back as feedback
    bad = FakeClient("openai", lambda m, n: {"skill": "made-up"})
    with pytest.raises(RoleFailed):
        await swarm_deriver(bad, lambda t: skills)(BASE, FalsifyContext())

    from types import SimpleNamespace
    spec = SimpleNamespace(skills=[SimpleNamespace(id="a", description="A")])

    assert domain_skills(spec) == [{"id": "a", "description": "A"}]


async def test_skill_preserving_easier_knobs_on_a_variant_world():
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.meta.rsi import ComposeKnobs, KnobMove

    ctx = FalsifyContext()
    k = ComposeKnobs(ctx)
    from tests.unit import synth_fixtures as fx

    base = fx.in_domain(fx.t3_resolve_email())
    harder = await k.apply(base, KnobMove(knob="distractors", direction="harder", params={"k": 2}))
    if harder is None:
        pytest.skip("the fixture World takes no distractor clones")
    moves = {m.knob for m in await k.moves(harder, "easier")}
    assert {"undo", "fewer_distractors"} <= moves and "derive" not in moves
    easier = await k.apply(harder, KnobMove(knob="fewer_distractors", direction="easier"))
    assert easier is not None and easier.intent == harder.intent and easier.id != harder.id
    assert not await k.moves(base, "easier")  # a plain task: no lawful easier knob


async def test_the_default_checks_read_the_new_card():
    from crucible.synth.falsify import FalsifyContext
    from crucible.synth.meta.reskill import leak_checker, novelty_checker, p3_checker
    from tests.unit import synth_fixtures as fx

    ctx = FalsifyContext()
    leak = leak_checker(ctx)
    assert await leak(BASE) == []
    told = fx.in_domain(fx.planted("D7"))  # the instruction states the reason code the card derives
    assert any("D7" in x or "TAINT" in x for x in await leak(told))
    nov = novelty_checker(ctx)
    ok, _ = await nov(BASE, "c", [])
    dup_ok, why = await nov(BASE, "c", [BASE.model_copy(update={"id": "copy"}, deep=True)])
    assert ok and not dup_ok and why
    on, reasons = await p3_checker(ctx)(BASE)
    assert on in (True, False, None) and isinstance(reasons, list)
