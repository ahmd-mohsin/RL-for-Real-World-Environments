"""crucible.synth.meta.library + lemmas: the operator DSL, the birth gate, retirement, fictitious play, the planter loop's
acceptance and the lemma bank (part2-paper-core §9.3, §9.5; M3 acceptance).

The M2 population misses an unstated ``never(read)`` constraint (no mutant adds a read). A hand-written DSL operator
that inserts an unused read closes the gap and passes the birth gate; an operator that demands a key swap must pass
false-alarms on the certified anchors and fails it.
"""

from __future__ import annotations

import pytest

from crucible.synth.falsify import FalsifyContext, cheap, cheap_m3, falsify
from crucible.synth.meta.lemmas import Lemma, LemmaBank, Rate, lemma_acceptance, lemma_id, mcnemar_one_sided, newcombe
from crucible.synth.meta.library import (
    DropCalls,
    InsertCall,
    LabelRule,
    Library,
    MutateArg,
    OperatorSpec,
    SConst,
    SFieldValues,
    SObservationOf,
    SReadTool,
    SSiblingOf,
    STool,
    SWrittenKey,
    TOtherSeen,
    TShift,
    accept_planter,
    birth_gate,
    compile_operator,
    fictitious_play,
    retire,
)
from crucible.synth.meta.plant import Planter, PlantOp, base_planter, plant_all
from tests.unit import synth_fixtures as fx

EXTRA_READ = OperatorSpec(name="extra_read", body=InsertCall(tool=SReadTool(unused=True)), label=LabelRule(expect="fail", defect="D6", confirm="never"),
                          why="a valid solution may read more than the oracle did; an unstated never/at_most on a read rejects it")
SWAP_MUST_PASS = OperatorSpec(name="swap_must_pass", body=MutateArg(key_only=True, values=SSiblingOf(of=SWrittenKey())),
                              label=LabelRule(expect="pass", defect="D6"))
DROP_WRITES = OperatorSpec(name="drop_writes", body=DropCalls(write_only=True), label=LabelRule(expect="fail", defect="D1"))

_CACHE: dict[str, object] = {}


def _ctx() -> FalsifyContext:
    """One context for the module: falsifier results are cached per task version, so re-measuring a plant is free."""
    if "ctx" not in _CACHE:
        _CACHE["ctx"] = FalsifyContext()
    return _CACHE["ctx"]  # type: ignore[return-value]


async def _sets():
    """The anchors, the never(read) plants, and one dev plant per (operator variant, domain)."""
    if "sets" not in _CACHE:
        ctx = _ctx()
        anchors = fx.anchors()
        never = Planter(id="never", ops=[PlantOp(name="over_constrain", variant="never")], max_per_op=1)
        nev = (await plant_all(anchors, never, ctx)).kept
        dev, seen = [], set()
        for p in (await plant_all(anchors, Planter(id="dev", ops=base_planter().ops, max_per_op=1), ctx)).kept:
            if (p.op.key, p.domain) not in seen:
                seen.add((p.op.key, p.domain))
                dev.append(p)
        _CACHE["sets"] = (anchors, nev, dev)
    return _CACHE["sets"]


def test_the_dsl_is_typed_json():
    spec = OperatorSpec(name="all", body=InsertCall(tool=STool(glob="ap.*"), args={"status": SConst(values=["on_hold"]), "invoice_id": "borrow"}),
                        label=LabelRule(defect="D2", confirm="outside_delta"))
    again = OperatorSpec.model_validate_json(spec.model_dump_json())
    assert again == spec and again.cls_name == "OP:all@1"
    for body in (MutateArg(transform=TShift(by=10)), MutateArg(values=SFieldValues(field="status"), write_only=False),
                 MutateArg(transform=TOtherSeen()), InsertCall(tool=SReadTool(), args={"query": SObservationOf(tool="desk.get_ticket", path="subject")})):
        s = OperatorSpec(name="x", body=body, label=LabelRule(defect="D3"))
        assert OperatorSpec.model_validate_json(s.model_dump_json()) == s
    with pytest.raises(ValueError):
        OperatorSpec.model_validate({"name": "bad", "body": {"kind": "teleport"}, "label": {"defect": "D1"}})


async def test_compiled_operators_probe_like_mut():
    t1 = fx.in_domain(fx.t1_hold())
    ctx = FalsifyContext()
    shift = compile_operator(OperatorSpec(name="shift", body=MutateArg(arg="invoice_id", transform=TShift(by=1)), label=LabelRule(defect="D9")))
    r = await falsify(t1, shift, ctx)
    assert r.cls == "OP:shift@1" and [f.site for f in r.findings] == ["s3.invoice_id=104"] and r.c == 0  # holding 104 fails the contract: clean
    r = await falsify(fx.planted("D9"), shift, ctx)  # on the planted "any invoice" contract the shifted hold survives
    assert r.c == 1 and r.confirmed[0].defect == "D9"
    obs = compile_operator(OperatorSpec(name="obs", body=MutateArg(arg="reason", values=SObservationOf(tool="ap.find_invoice", path="reason")),
                                        label=LabelRule(defect="D3")))
    assert (await falsify(t1, obs, ctx)).n == 0  # the only reason t1 observes is null: no mutant
    seen = compile_operator(OperatorSpec(name="seen", body=MutateArg(arg="reason", transform=TOtherSeen()), label=LabelRule(defect="D3", confirm="missing_delta")))
    r = await falsify(t1, seen, ctx)
    assert r.n >= 1 and r.c == 0  # another reason code is a different Δ, rejected by the correct contract


async def test_a_dsl_operator_passes_the_birth_gate_and_one_that_false_alarms_fails():
    anchors, nev, dev = await _sets()
    ap_src = [p for p in nev if p.domain == "ap-test"][:1]
    fresh = [*[p for p in nev if p.domain == "desk-test"], *[p for p in dev if p.domain == "desk-test"]]
    ctx = _ctx()
    # the M2 population misses the never(read) plants entirely ...
    m2 = {r.cls: r.c for p in nev for r in [await falsify(p.task, f, ctx) for f in cheap_m3()]}
    assert not any(m2.values())
    g = await birth_gate(EXTRA_READ, sources=ap_src, fresh=fresh, anchors=anchors, ctx=ctx, cost_bar=10)
    assert g.passed, g.checks
    assert g.detail["transfer_domains"] == ["desk-test"] and g.detail["false_alarms"] == [] and g.detail["probes_per_task"] <= 10
    bad = await birth_gate(SWAP_MUST_PASS, sources=ap_src, fresh=fresh, anchors=anchors, ctx=ctx, cost_bar=100)
    assert not bad.passed and bad.checks["zero_false_alarms"] is False and len(bad.detail["false_alarms"]) == len(anchors)
    blind = await birth_gate(DROP_WRITES, sources=ap_src, fresh=fresh, anchors=anchors, ctx=ctx, cost_bar=100)
    assert not blind.passed and blind.checks["reproduces_sources"] is False  # cannot reproduce the counterexample it is meant to generalise
    costly = await birth_gate(EXTRA_READ, sources=ap_src, fresh=fresh, anchors=anchors, ctx=ctx, cost_bar=0.5)
    assert not costly.passed and costly.checks == {**g.checks, "under_cost_bar": False}
    same_domain = await birth_gate(EXTRA_READ, sources=ap_src, fresh=[p for p in nev if p.domain == "ap-test"], anchors=anchors, ctx=ctx, cost_bar=10)
    assert not same_domain.passed and same_domain.checks["finds_on_other_domains"] is False  # transfer must be shown elsewhere


async def test_mut_read_mutants_make_the_hand_written_read_operator_redundant():
    """DESIGN §7.7 item 20: MUT's InsertRead closes the never(read) gap itself, so the DSL operator that closed it in M3
    adds no unique hit and retirement removes it (fictitious play holds without it)."""
    _anchors, nev, dev = await _sets()
    ctx = _ctx()
    now = {r.cls: r.c for p in nev for r in [await falsify(p.task, f, ctx) for f in cheap()]}
    assert now["MUT"] > 0 and all(v == 0 for k, v in now.items() if k != "MUT")
    lib = Library().add(EXTRA_READ, round=0)
    out, gone = await retire(lib, base=cheap(), union=[*nev, *dev], ctx=ctx, round=3, window=2)
    assert gone == ["OP:extra_read@1"] and not out.ops


async def test_retirement_keeps_unique_operators_under_fictitious_play():
    _anchors, nev, dev = await _sets()
    union = [*nev, *dev]  # the sets of two past planters
    ctx = _ctx()
    lib = Library().add(EXTRA_READ, round=0).add(DROP_WRITES, round=0)
    out, gone = await retire(lib, base=cheap_m3(), union=union, ctx=ctx, round=1, window=2)
    assert gone == []  # too early
    out, gone = await retire(out, base=cheap_m3(), union=union, ctx=ctx, round=3, window=2)
    assert gone == ["OP:drop_writes@1"] and set(out.ops) == {"OP:extra_read@1"} and out.retired == ["OP:drop_writes@1"]  # MUT's DropStep covers it
    fp = await fictitious_play(cheap_m3(), [*cheap_m3(), *lib.falsifiers()], union, ctx)
    assert not fp.ok and fp.recall_new < fp.recall_old and fp.per_planter["never@1"][1] == 0.0  # dropping extra_read forgets the never planter
    keep = await fictitious_play([*cheap_m3(), *out.falsifiers()], [*cheap_m3(), *lib.falsifiers()], union, ctx)
    assert keep.ok and keep.recall_new == keep.recall_old
    assert Library.model_validate_json(out.model_dump_json()) == out


async def test_the_planter_loop_accepts_natural_planters_and_mmd_rejects_a_shifted_one():
    anchors, _, dev = await _sets()
    ref = [p.features for p in dev if p.domain == "ap-test"]  # the reference ("real-defect") features
    desk = [t for t in anchors if t.domain == "desk-test"]
    ctx = _ctx()
    ok, plants = await accept_planter(Planter(id="cand", ops=base_planter().ops, max_per_op=1), desk, ctx, reference=ref, mu=0.35, falsifiers=cheap())
    assert ok.accepted and ok.checks == {"witnessed": True, "natural_like": True, "no_leakage": True} and ok.mmd < 0.35 and plants
    assert 0.0 <= ok.miss_rate < 1.0  # its reward: what the current population misses
    shifted = Planter(id="shifted", ops=[PlantOp(name="break_solvability", variant="unknown_code"), PlantOp(name="inject_nondet", variant="pinned")])
    bad, _ = await accept_planter(shifted, desk, ctx, reference=ref, mu=0.35, falsifiers=cheap())
    assert not bad.accepted and bad.checks["natural_like"] is False and bad.checks["witnessed"] and bad.mmd > 0.35
    leaky, _ = await accept_planter(Planter(id="leaky", ops=base_planter().ops[:2], max_per_op=1), desk, ctx, reference=ref, mu=10, falsifiers=cheap(),
                                    critic=lambda p: ["names anchor desk/resolve-T7"])
    assert not leaky.accepted and leaky.checks["no_leakage"] is False
    empty, _ = await accept_planter(Planter(id="none", ops=[PlantOp(name="drop_required")]), [fx.in_domain(fx.t6_resolve())], ctx, reference=ref, mu=10,
                                    falsifiers=cheap())
    assert not empty.accepted and empty.checks["witnessed"] is False  # a planter whose plants all lack witnesses plants nothing


def test_lemma_bank_and_its_equal_budget_acceptance():
    text = "Check every value the contract pins against a source the agent can read before relaxing it."
    bank = LemmaBank().add(Lemma(id=lemma_id(text), text=text, sources=["cx-1"], defects=["D5"], roles=["writer", "repairer"]))
    assert bank.version == 2 and bank.for_role("proposer") == [] and text in bank.prompt_block("writer")
    with pytest.raises(ValueError):
        bank.add(bank.lemmas[0])
    gone = bank.retire(bank.lemmas[0].id)
    assert gone.version == 3 and gone.lemmas == [] and gone.retired == [bank.lemmas[0].id]
    assert LemmaBank.model_validate_json(bank.model_dump_json()) == bank
    # paired: the same 20 cells with and without the batch; 8 lose their first-round counterexample, 1 gains one
    pairs = [(True, False)] * 8 + [(False, True)] + [(True, True)] * 3 + [(False, False)] * 8
    b, c, p = mcnemar_one_sided(pairs)
    assert (b, c) == (8, 1) and p == pytest.approx(10 / 512)
    yes = lemma_acceptance(Rate(k=4, n=20, spend=10.0), Rate(k=11, n=20, spend=10.0), j_rises=True, pairs=pairs)
    assert yes.accepted and yes.mcnemar_p == pytest.approx(10 / 512)
    # unpaired: Newcombe's interval for the difference must lie below 0
    lo, hi = newcombe(Rate(k=2, n=40, spend=1), Rate(k=14, n=40, spend=1))
    assert lo < hi < 0
    assert lemma_acceptance(Rate(k=2, n=40, spend=5.0), Rate(k=14, n=40, spend=5.0), j_rises=True).accepted
    assert not lemma_acceptance(Rate(k=10, n=40, spend=5.0), Rate(k=12, n=40, spend=5.0), j_rises=True).accepted  # no clear fall
    no_j = lemma_acceptance(Rate(k=2, n=40, spend=5.0), Rate(k=14, n=40, spend=5.0), j_rises=False)
    assert not no_j.accepted and "J did not rise" in no_j.reasons[0]
    rich = lemma_acceptance(Rate(k=2, n=40, spend=9.0), Rate(k=14, n=40, spend=5.0), j_rises=True)
    assert not rich.accepted and "unequal budget" in rich.reasons[0]  # a lower rate bought with more budget is not the lemmas' doing
