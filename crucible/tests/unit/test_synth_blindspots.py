"""DESIGN §7.7 item 20: the measured falsifier blind spots closed (unstated trace constraints, humanised leaks) and the
L2 leak probe with its panel. Scripted fakes only: no model is called."""

from __future__ import annotations

import pytest

from crucible.synth.contract import TraceConstraint
from crucible.synth.falsify import TAINT, FalsifyContext, falsify
from crucible.synth.falsify.agents import PanelMember
from crucible.synth.falsify.leak import (
    LeakAnchor,
    LeakMatcher,
    LeakPanel,
    LeakProbe,
    SynEntry,
    SynonymMap,
    align,
    calibrate_leak_panel,
    extract_labels,
    learn_synonyms,
    number_mentions,
    stem,
    synonym_update,
)
from crucible.synth.falsify.mut import MUT, TRACE_OPS
from tests.unit import synth_fixtures as fx
from tests.unit.synth_fakes import FakeClient, section


def _with_trace(t, tc: TraceConstraint):
    return t.model_copy(update={"id": f"{t.id}/{tc.id}", "contract": t.contract.model_copy(update={"trace": [*t.contract.trace, tc]})}, deep=True)


# ---------------------------------------------------------------------------------------------------------- MUT read mutants

async def test_read_mutants_confirm_unstated_never_at_most_and_before_as_d6():
    t1 = fx.t1_hold()
    ctx = FalsifyContext()
    for tc in (TraceConstraint(id="m_never", op="never", tool="ap.get_receipts"),  # a read pi* never makes
               TraceConstraint(id="m_most", op="at_most", tool="ap.find_invoice", n=1),  # pi*'s own read count
               TraceConstraint(id="m_never_list", op="never", tool="ap.list_invoices")):
        r = await falsify(_with_trace(t1, tc), MUT(), ctx)
        assert r.confirmed and all(f.defect == "D6" for f in r.confirmed), (tc.id, [(f.op, f.site) for f in r.confirmed])
        old = await falsify(_with_trace(t1, tc), MUT(trace=False), ctx)
        if tc.op == "never":  # the M3 neighbourhood never adds a read: the gap RSI-1/2 measured
            assert not old.confirmed and {f.op for f in r.confirmed} <= set(TRACE_OPS)
        else:  # at_most of pi*'s own count: DupStep already found it (one effect, one trial)
            assert old.confirmed


async def test_a_stated_constraint_is_not_a_false_alarm_and_correct_tasks_stay_clean():
    t1 = fx.t1_hold()
    stated = t1.model_copy(update={"instruction": t1.instruction + " Do not look at goods receipts.",
                                   "contract": t1.contract.model_copy(update={"trace": [TraceConstraint(id="m", op="never", tool="ap.get_receipts",
                                                                                                         stated="Do not look at goods receipts")]})},
                           deep=True)
    ctx = FalsifyContext()
    r = await falsify(stated, MUT(), ctx)
    assert not [f for f in r.confirmed if f.op in TRACE_OPS]  # the stated rule makes a receipts read non-equivalent: no trial
    for t in fx.anchors():
        res = await falsify(t, MUT(), ctx)
        assert not res.confirmed, (t.id, [(f.op, f.site) for f in res.confirmed])
        old = await falsify(t, MUT(trace=False), ctx)
        assert res.n == old.n  # read mutants share pi*'s effect on a correct contract: n (distinct effects) is unchanged


async def test_read_mutants_reach_every_read_tool_and_entity_class():
    t3 = fx.t3_resolve_email()
    ctx = FalsifyContext()
    probes = [p for p in await MUT().propose(t3, ctx) if p.op in TRACE_OPS]
    tools = {p.data.get("tool") for p in probes if p.op in ("InsertRead", "ExtraList")}
    world = await ctx.world(t3)
    reads = {t.name for t in await world.tools() if t.read_only}
    fillable = {t for t in reads if t in tools}
    assert "desk.get_ticket" in fillable and "desk.search_kb" in fillable and len(fillable) >= len(reads) - 1, (reads, tools)
    assert {p.site.rsplit("@", 1)[1] for p in probes if p.op == "InsertRead"} == {"start", "end"}
    swaps = [p for p in probes if p.op == "SwapReads"]
    assert swaps and all(p.program.steps[0].tool == "desk.search_kb" for p in swaps)  # s1, s2 are independent reads


# ---------------------------------------------------------------------------------------------------------- structural layers

def test_each_structural_layer_finds_its_kind_of_paraphrase():
    m = LeakMatcher()
    assert m.find("price_variance", "Put it on hold: it comes down to a price variance.")[0].layer == "sep"
    assert m.find("price_variance", "the Price-Variance rule")[0].layer == "sep"
    assert m.find("PriceVariance", "a price variance")[0].layer == "sep"
    assert m.find("vendor_blocked", "Hold it, the vendor is blocked.")[0].layer == "lemma"
    assert m.find("bank_verification", "Verify the bank account first.")[0].layer == "lemma"
    assert m.find(108, "It was one hundred and eight dollars.")[0].layer == "number"
    assert m.find(25, "twenty-five units")[0].layer == "number"
    assert m.find("price_variance", "reason code price_variance")[0].layer == "exact"
    assert m.find("P1", "It is critical.", labels=["Critical"])[0].layer == "label"
    assert not m.find("T-1001", "the ticket from tuesday")  # identifiers get exact and separator matching only
    assert m.find("GRP-02", "route it to grp 02")[0].layer == "sep"
    assert not LeakMatcher(("exact",)).find("price_variance", "a price variance")  # the M2 matcher misses it


def test_an_enumeration_of_sibling_outcomes_is_not_a_give_away():
    m = LeakMatcher()
    text = "Either approve it, put it on hold or reject it as the AP policy requires."
    hits = m.find("on_hold", text, siblings=["approved", "rejected", "pending"])
    assert hits and all(h.options for h in hits) and not m.counted(hits)
    hits2 = m.find("on_hold", "Put it on hold now.", siblings=["approved", "rejected"])
    assert m.counted(hits2)


def test_labels_come_from_readable_texts_and_numbers_from_words():
    texts = ["SLA: P1 (Critical)=15 minutes, P2 (High)=1 hour.", "Reason codes: missing_receipt: Missing goods receipt; dup = Duplicate."]
    assert extract_labels("P1", texts) == ["Critical"]
    assert extract_labels("missing_receipt", texts) == ["Missing goods receipt"]
    assert extract_labels("dup", texts) == ["Duplicate"]
    assert [x for x, *_ in number_mentions("a dozen, three thousand and two, forty")] == [12.0, 3002.0, 40.0]
    assert stem("verification") == stem("verify") and stem("blocked") == "block"


def test_synonym_maps_are_learned_data_and_judge_entries_are_suspects():
    assert align("price_variance", "a price mismatch") == [(stem("variance"), stem("mismatch"))]
    assert align("price_variance", "price variance") == []
    syn = learn_synonyms([{"value": "price_variance", "quote": "price mismatch", "source": "cx-1", "kind": "judge"}])
    assert syn.entries[0].word == stem("variance") and syn.entries[0].kind == "judge"
    hit = LeakMatcher(synonyms=syn).find("price_variance", "There is a price mismatch on it.")
    assert hit and hit[0].layer == "synonym" and hit[0].judge
    up = synonym_update(SynonymMap(), syn, round=1, author_family="inkling")
    assert up.evidence.kind == "judge" and up.edits  # R8's round cap applies to a judge-taught map
    ex = SynonymMap(entries=[SynEntry(word=stem("variance"), variants=["mismatch"], kind="executed")])
    assert synonym_update(SynonymMap(), ex, round=1).evidence.kind == "executed"
    assert SynonymMap.model_validate_json(syn.model_dump_json()) == syn


async def test_taint_layers_confirm_a_humanised_leak_and_keep_m2_behaviour_selectable():
    t1 = fx.t1_hold()
    leak = t1.model_copy(update={"id": "h", "instruction": t1.instruction + " Hint: it comes down to price variance."}, deep=True)
    ctx = FalsifyContext()
    now = await falsify(leak, TAINT(), ctx)
    assert any(f.op == "derive" and f.label == "confirmed" and f.defect == "D7" and f.outcome.data["layer"] == "sep" for f in now.findings)
    m2 = await falsify(leak, TAINT(layers=("exact",)), ctx)
    assert not any(f.op == "derive" and f.label == "confirmed" for f in m2.findings)
    syn = learn_synonyms([{"value": "price_variance", "quote": "price mismatch", "kind": "judge"}])
    taught = leak.model_copy(update={"id": "j", "instruction": t1.instruction + " There is a price mismatch."}, deep=True)
    r = await falsify(taught, TAINT(synonyms=syn), ctx)
    assert any(f.op == "derive" and f.label == "suspect" and f.defect == "D7" for f in r.findings)
    for t in fx.anchors():  # no structural layer alarms on the certified anchors
        res = await falsify(t, TAINT(), ctx)
        assert not [f for f in res.confirmed if f.op == "derive"], t.id


# ---------------------------------------------------------------------------------------------------------- L2

def _panel(leak: bool, conf: float = 0.9) -> LeakPanel:
    def vote(_m, _n):
        return {"leak": leak, "confidence": conf, "why": "scripted"}

    return LeakPanel(members=[PanelMember("a", FakeClient("nemotron", vote)), PanelMember("b", FakeClient("openai", vote))], lam=0.7)


async def test_leak_probe_quotes_are_checked_then_the_panel_decides():
    t1 = fx.t1_hold()
    task = t1.model_copy(update={"id": "p", "instruction": t1.instruction + " The amounts simply do not agree with the order."}, deep=True)

    def probe(msgs, n):
        assert section(msgs, "decisions") == ["price_variance"] and "contract" not in msgs[1]["content"]
        if n == 0:  # a quote that is not in the instruction is fed back
            return {"leaks": [{"value": "price_variance", "quote": "the price is off", "confidence": 0.9}]}
        return {"leaks": [{"value": "price_variance", "quote": "The amounts simply do not agree with the order", "confidence": 0.8}]}

    client = FakeClient("inkling", probe)
    ctx = FalsifyContext()
    r = await falsify(task, LeakProbe(client, panel=_panel(True)), ctx)
    assert r.c == 1 and r.confirmed[0].defect == "D7" and r.confirmed[0].outcome.data["evidence"] == "L2" and len(client.calls) == 2
    cleared = await falsify(task, LeakProbe(FakeClient("inkling", probe), panel=_panel(False)), ctx)
    assert cleared.c == 0 and not cleared.suspects
    open_ = await falsify(task, LeakProbe(FakeClient("inkling", probe)), ctx)
    assert open_.suspects and open_.suspects[0].defect == "D7"  # no panel: a suspect, never a confirmed counterexample
    unsure = await falsify(task, LeakProbe(FakeClient("inkling", probe), panel=_panel(True, conf=0.5)), ctx)
    assert unsure.suspects and not unsure.confirmed  # below lambda: the human queue


async def test_leak_probe_skips_what_tain_finds_and_respects_its_call_cap():
    t1 = fx.t1_hold()
    exact = t1.model_copy(update={"id": "e", "instruction": t1.instruction + " Use price_variance."}, deep=True)
    client = FakeClient("inkling", lambda _m, _n: {"leaks": []})
    ctx = FalsifyContext()
    r = await falsify(exact, LeakProbe(client), ctx)
    assert r.n == 0 and not client.calls  # the structural layers confirm it already: no money spent
    capped = LeakProbe(client, max_calls=0)
    r2 = await falsify(t1, capped, ctx)
    assert r2.n == 0 and r2.findings and r2.findings[0].outcome.error == "leak probe call cap reached"
    none = await falsify(fx.t2_approve_answer(), LeakProbe(client), ctx)
    assert not none.applicable


async def test_leak_panel_calibration_uses_executed_truth():
    def vote(msgs, _n):
        q = section(msgs, "quote")
        return {"leak": "variance" in str(q), "confidence": 0.95}

    panel = LeakPanel(members=[PanelMember("a", FakeClient("nemotron", vote)), PanelMember("b", FakeClient("openai", vote))], lam=0.0)
    anchors = [LeakAnchor(instruction="x price variance", value="price_variance", quote="price variance", truth=True, kind="planted"),
               LeakAnchor(instruction="check the order", value="price_variance", quote="check the order", truth=False, kind="anchor")]
    cal = await calibrate_leak_panel(panel, anchors, min_decided=2)
    assert cal["calibrated"] and panel.lam == pytest.approx(0.7) and cal["rows"][0]["accuracy"] == 1.0
    assert not (await calibrate_leak_panel(panel, anchors))["calibrated"] and panel.lam > 1  # too few anchors: the panel decides nothing


async def test_panel_calibration_needs_enough_anchors_of_both_truths():
    """Item 20: λ = 0.95 once "calibrated" on one decided no-op anchor and then decided a suspect."""
    from crucible.synth.falsify.agents import Anchor, Attempt, Panel, calibrate

    t1 = fx.t1_hold()

    def sure_noop(msgs, _n):  # confident only on the empty attempt (truth: invalid); unsure on the oracle
        empty = section(msgs, "trace") in ([], None)
        return {"valid": not empty, "confidence": 0.97 if empty else 0.6}

    p = Panel(members=[PanelMember("a", FakeClient("nemotron", sure_noop)), PanelMember("b", FakeClient("openai", sure_noop))])
    anchors = [Anchor(task=t1, attempt=Attempt(), valid=False, kind="noop"),
               Anchor(task=t1, attempt=Attempt(trace=[{"tool": "ap.hold_invoice", "args": {}}]), valid=True, kind="oracle")]
    loose = await calibrate(p, anchors, grid=(0.7, 0.8, 0.9))
    assert loose.calibrated and loose.lam == 0.7  # the M4 rule: the 0.7 row decides only the no-op and is 1/1 right
    strict = await calibrate(p, anchors, grid=(0.7, 0.8, 0.9), require_both=True, refuse_uncalibrated=True)
    assert not strict.calibrated and p.lam > 1


async def test_l2_finds_reach_the_archive_and_teach_judge_synonyms():
    from crucible.synth.falsify.base import Counterexample
    from crucible.synth.falsify.leak import leaks_from_archive

    t1 = fx.t1_hold()
    task = t1.model_copy(update={"id": "a", "instruction": t1.instruction + " There is a price mismatch."}, deep=True)
    probe = FakeClient("inkling", lambda _m, _n: {"leaks": [{"value": "price_variance", "quote": "a price mismatch", "confidence": 0.9}]})
    r = await falsify(task, LeakProbe(probe, panel=_panel(True)), FalsifyContext())
    cx = Counterexample.of(task, r.confirmed[0])
    found = leaks_from_archive([cx])
    assert found == [{"value": "price_variance", "quote": "a price mismatch", "source": cx.id, "kind": "judge"}]
    syn = learn_synonyms(found)
    hit = LeakMatcher(synonyms=syn).find("price_variance", "Note the price mismatch.")
    assert hit and hit[0].judge  # next time the structural layer finds it, as a suspect (judge evidence)


def test_the_patch_panel_role_and_the_six_family_wire():
    from crucible.factory.models import parse_spec
    from crucible.factory.synth_wire import SYNTH_ROLES, SYNTH_ROLES_DISJOINT, check_assignment
    from crucible.synth.falsify.agents import Panel
    from crucible.synth.roles import PANEL_PATCH, checker_overlap

    p = Panel(members=[PanelMember("a", FakeClient("nemotron", lambda _m, _n: {})), PanelMember("b", FakeClient("openai", lambda _m, _n: {}))], role=PANEL_PATCH)
    assert p.role.sees == ("task", "intent", "trace", "answer", "reason_codes") and "STATUS.json" in p.role.system
    for t in (SYNTH_ROLES, SYNTH_ROLES_DISJOINT):
        check_assignment(t)  # no Qwen, family rules hold
    fams = {r: [parse_spec(m).family for m in ms] for r, ms in SYNTH_ROLES_DISJOINT.items()}
    assert checker_overlap(fams) == [] and fams["leak_probe"] and "inkling" in fams["panel"]
    assert [parse_spec(m).family for m in SYNTH_ROLES["leak_probe"]] == ["inkling"]
