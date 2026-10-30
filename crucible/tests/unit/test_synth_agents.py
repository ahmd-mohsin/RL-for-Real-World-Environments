"""crucible.synth.falsify.agents: LLM-driven ALT/ATK runners and the adjudication panel, with scripted models (M4)."""

from __future__ import annotations

import json

import pytest

from crucible.synth.falsify import ALT, ATK, FalsifyContext, falsify
from crucible.synth.falsify.agents import AgentBudget, Anchor, Attempt, Panel, PanelMember, calibrate, llm_alt, llm_atk
from crucible.synth.roles import ATTACK_STRATEGIES
from tests.unit import synth_fixtures as fx
from tests.unit.synth_fakes import FakeClient, section, seq

SOLVE = [{"call": {"tool": "desk.search_kb", "args": {"query": "refund policy"}}},
         {"call": {"tool": "desk.resolve_ticket", "args": {"ticket_id": "T-3", "resolution": "KB-2"}}},
         {"done": True, "final": None}]
WRONG = [{"call": {"tool": "desk.resolve_ticket", "args": {"ticket_id": "T-3", "resolution": "KB-5"}}}, {"done": True, "final": None}]


async def test_llm_alt_solves_through_the_json_protocol():
    t = fx.t6_resolve()
    c = FakeClient("nemotron", seq(*SOLVE))
    alt = ALT([llm_alt("alt-n", c)])
    res = await falsify(t, alt, FalsifyContext())
    assert res.n == 1 and res.findings[0].label == "clean" and res.findings[0].outcome.passed
    seen = section(c.calls[1], "history")
    assert seen[0]["call"]["tool"] == "desk.search_kb" and "KB-2" in json.dumps(seen[0]["result"])
    assert "contract" not in c.calls[0][1]["content"].lower().split("## instruction")[0]  # the agent never sees the contract
    trace = res.findings[0].outcome.data["trace"]
    assert [x["tool"] for x in trace] == ["desk.search_kb", "desk.resolve_ticket"]


async def test_budget_unknown_tools_and_model_failure():
    t = fx.t6_resolve()
    loop = FakeClient("openai", seq({"call": {"tool": "desk.list_tickets", "args": {}}}))
    alt = ALT([llm_alt("alt-o", loop, budget=AgentBudget(max_calls=3, max_turns=5))])
    res = await falsify(t, alt, FalsifyContext())
    assert res.findings[0].label == "suspect" and res.findings[0].outcome.data["calls"] == 3  # stopped by the call budget
    bogus = FakeClient("openai", seq({"call": {"tool": "desk.drop_tables", "args": {}}}, {"done": True}))
    res = await falsify(t, ALT([llm_alt("alt-b", bogus)]), FalsifyContext())
    assert "unknown tool" in section(bogus.calls[1], "history")[0]["error"]

    class Broken(FakeClient):
        async def complete_json(self, *a, **k):
            raise RuntimeError("no JSON")

    res = await falsify(t, ALT([llm_alt("alt-x", Broken("openai", seq({})))]), FalsifyContext())
    assert res.n == 0  # a model that cannot answer made no attempt: not a trial


async def test_an_early_done_is_nudged_once_and_flat_call_shapes_parse():
    t = fx.t6_resolve()
    lazy = [{"done": True, "final": None},  # finishing before any call: nudged once
            {"tool": "desk.search_kb", "args": {"query": "refund policy"}},  # a flat call
            {"call": {"name": "desk.resolve_ticket", "arguments": '{"ticket_id": "T-3", "resolution": "KB-2"}'}},  # OpenAI-style
            {"done": True}]
    c = FakeClient("openai", seq(*lazy))
    res = await falsify(t, ALT([llm_alt("alt-l", c)]), FalsifyContext())
    assert res.findings[0].outcome.passed and "not called any tool" in json.dumps(section(c.calls[1], "history"))


async def test_free_text_literals_are_not_hidden_information():
    """TAINT: an unsourced literal the contract does not depend on (a note body) is free; one it checks is D5."""
    from crucible.synth.contract import PAny
    from crucible.synth.falsify import TAINT

    t = fx.t3_resolve_email()
    prog = t.oracle.model_copy(deep=True)
    prog.steps[3].args["body"] = fx.lit("Here is the article you asked for.")  # the author's own wording
    strict = t.model_copy(update={"oracle": prog, "contract": t.contract.model_copy(update={"required": [
        p if p.id != "r_mail" else p.model_copy(update={"where": {**p.where, "body": fx.PEq(v="Here is the article you asked for.")}})
        for p in t.contract.required]})}, deep=True)
    loose = strict.model_copy(update={"contract": strict.contract.model_copy(update={"required": [
        p if p.id != "r_mail" else p.model_copy(update={"where": {**p.where, "body": PAny()}}) for p in strict.contract.required]})}, deep=True)
    hit = [f for f in (await falsify(strict, TAINT(), FalsifyContext())).confirmed if f.op == "literal"]
    assert [f.probe.data["value"] for f in hit] == ["Here is the article you asked for."] and hit[0].defect == "D5"
    res = await falsify(loose, TAINT(), FalsifyContext())
    assert not [f for f in res.confirmed if f.op == "literal"]
    assert any(f.outcome.data.get("free") for f in res.findings if f.op == "literal")


async def test_adjudicated_mut_suspects_are_decided_on_their_executed_trace(tmp_path):
    from crucible.synth.contract import PAny
    from crucible.synth.falsify import MUT
    from crucible.synth.falsify.agents import Adjudicated

    t = fx.t6_resolve()
    lax = t.model_copy(update={"contract": t.contract.model_copy(update={"required": [
        p if p.id != "r_res" else p.model_copy(update={"value": PAny()}) for p in t.contract.required]})}, deep=True)
    seen = []

    def judge(msgs, _n):
        tr = section(msgs, "trace")
        seen.append(tr)
        res = [x for x in tr if x["tool"] == "desk.resolve_ticket"]
        return {"valid": bool(res) and res[-1]["args"].get("resolution") == "KB-2", "confidence": 0.9}

    p = Panel([PanelMember("p1", FakeClient("nemotron", judge)), PanelMember("p2", FakeClient("openai", judge))], queue=tmp_path / "q.jsonl")
    plain = await falsify(lax, MUT(), FalsifyContext())
    assert plain.suspects  # survivors that resolve with another article: D3 suspects
    res = await falsify(lax, Adjudicated(MUT(), p), FalsifyContext())
    assert res.cls == "MUT" and not res.suspects and any(f.defect == "D3" for f in res.confirmed)
    assert all(isinstance(tr, list) and tr and "result" in tr[0] for tr in seen)  # the panel saw calls with observations


async def test_attacker_sees_strategies_and_a_pass_is_a_suspect():
    t = fx.planted("D8")  # a direct-write admin tool
    hack = [{"call": {"tool": "desk.admin_update", "args": {"ticket_id": "T-3", "status": "resolved", "resolution": "KB-2"}}}, {"done": True}]
    c = FakeClient("nemotron", seq(*hack))
    res = await falsify(t, ATK([llm_atk("atk-n", c)]), FalsifyContext())
    assert res.findings[0].label == "suspect" and res.findings[0].defect == "D8"
    assert section(c.calls[0], "strategies") == list(ATTACK_STRATEGIES)


def _panel(tmp_path, a, b, lam=0.7):
    return Panel([PanelMember("p1", FakeClient("nemotron", a)), PanelMember("p2", FakeClient("openai", b))], lam=lam, queue=tmp_path / "queue.jsonl")


async def test_panel_confirms_disagreement_clears_agreement_and_queues_abstentions(tmp_path):
    t = fx.t6_resolve()
    ctx = FalsifyContext()
    # an ALT attempt the grader failed; the panel says it was valid -> confirmed D6
    res = await falsify(t, ALT([llm_alt("alt", FakeClient("nemotron", seq(*WRONG)))]), ctx)
    f = res.findings[0]
    assert f.label == "suspect"
    yes = seq({"valid": True, "confidence": 0.9, "why": "ok"})
    no = seq({"valid": False, "confidence": 0.95, "why": "wrong article"})
    out = await _panel(tmp_path, yes, yes)(t, [f])
    assert out == [f] and f.label == "confirmed" and f.defect == "D6"
    f.label = "suspect"
    assert await _panel(tmp_path, no, no)(t, [f]) == [] and f.label == "clean"
    f.label = "suspect"
    p = _panel(tmp_path, yes, no)  # disagreement: abstain, human queue, still a suspect
    assert await p(t, [f]) == [] and f.label == "suspect"
    row = json.loads((tmp_path / "queue.jsonl").read_text().splitlines()[0])
    assert row["decision"] == "abstain" and row["site"] == f.site and row["instruction"] == t.instruction
    f.label = "suspect"
    low = seq({"valid": True, "confidence": 0.5})
    assert await _panel(tmp_path, low, low, lam=0.7)(t, [f]) == [] and f.label == "suspect"  # below λ
    # the panel never learns which falsifier made the probe
    c = FakeClient("nemotron", yes)
    await Panel([PanelMember("p1", c), PanelMember("p2", FakeClient("openai", yes))])(t, [f])
    txt = c.calls[0][1]["content"]
    assert "ALT" not in txt and "alt_solver" not in txt and "## reason_codes" in txt
    with pytest.raises(ValueError, match="families"):
        Panel([PanelMember("a", FakeClient("openai", yes)), PanelMember("b", FakeClient("openai", yes))])


async def test_calibration_picks_lambda_from_executed_anchors(tmp_path):
    t = fx.t6_resolve()

    def judge(msgs, _n):  # right and sure on real traces, wrong but unsure on no-ops
        tr = section(msgs, "trace")
        if not tr or tr == "[]":
            return {"valid": True, "confidence": 0.6}
        return {"valid": any(x["tool"] == "desk.resolve_ticket" for x in tr), "confidence": 0.9}

    p = Panel([PanelMember("p1", FakeClient("nemotron", judge)), PanelMember("p2", FakeClient("openai", judge))], lam=0.5)
    good = Attempt(trace=[{"tool": "desk.resolve_ticket", "args": {"ticket_id": "T-3", "resolution": "KB-2"}}])
    anchors = [Anchor(task=t, attempt=good, valid=True, kind="oracle"), Anchor(task=t, attempt=Attempt(), valid=False, kind="noop")]
    rep = await calibrate(p, anchors, target=0.99)
    assert rep.calibrated and rep.lam == 0.7 and p.lam == 0.7  # at 0.5 and 0.6 the panel decides the no-op wrongly
    assert rep.rows[0]["decided"] == 2 and rep.rows[0]["accuracy"] == 0.5 and rep.rows[2]["decided"] == 1
    rep = await calibrate(p, anchors, target=0.99, grid=(0.5, 0.6))
    assert not rep.calibrated and p.lam == 0.6  # nothing reached the target: the strictest λ, flagged


async def test_an_adjudicated_mut_keeps_one_trial_per_effect_and_judges_each_effect_once():
    """The M4 runs counted every duplicate mutant of a panel-wrapped MUT as a trial (its ``finalize`` was lost), which made
    U_MUT too tight; the wrapper keeps MUT's effect deduplication and asks the panel once per effect."""
    from crucible.synth.falsify import MUT
    from crucible.synth.falsify.agents import Adjudicated

    t = fx.any_reason()
    plain = await falsify(t, MUT(), FalsifyContext())
    a, b = FakeClient("nemotron", seq({"valid": False, "confidence": 0.9})), FakeClient("openai", seq({"valid": False, "confidence": 0.9}))
    panel = Panel(members=[PanelMember("a", a), PanelMember("b", b)], lam=0.7)
    wrapped = await falsify(t, Adjudicated(MUT(), panel), FalsifyContext())
    assert wrapped.n == plain.n  # distinct effects only
    assert plain.suspects and not wrapped.suspects and wrapped.c >= len(plain.suspects)
    effects = {f.outcome.data["effect"] for f in wrapped.findings if f.outcome.data.get("panel")}
    assert len(a.calls) == len(effects)  # one panel call per distinct effect, whatever the number of mutants behind it


def test_ppi_mean_corrects_a_biased_judge_with_executed_labels():
    from crucible.synth.meta.ppi import adjudicated_rates, ppi_mean, vote_score

    # a judge that says "valid" 80% of the time on suspects but is right only on half of the labelled anchors
    f_unl = [1.0] * 80 + [0.0] * 20
    f_lab, y_lab = [1.0] * 10 + [0.0] * 10, [1.0] * 5 + [0.0] * 5 + [0.0] * 10
    est = ppi_mean(f_unl, f_lab, y_lab)
    assert est.estimate == pytest.approx(0.8 + (5 / 20 - 10 / 20))  # 0.55: the rectifier removes the judge's bias
    assert est.lo < est.estimate < est.hi and est.naive == pytest.approx(0.8) and est.n_labeled == 20 and est.n_unlabeled == 100
    assert est.classical is not None and est.classical[0] <= 0.25 <= est.classical[1]
    assert vote_score([{"valid": True, "confidence": 0.9}, {"valid": False, "confidence": 0.8}]) == pytest.approx((0.9 + 0.2) / 2)
    assert vote_score([{"valid": None, "error": "x"}]) == 0.5
    rows = [{"cls": "ALT", "votes": [{"valid": False, "confidence": 0.9}] * 2, "decision": "invalid"}] * 6
    rows += [{"cls": "MUT", "votes": [{"valid": True, "confidence": 0.9}] * 2, "decision": "valid"}] * 4
    cal = [{"truth": True, "votes": [{"valid": True, "confidence": 0.9}] * 2}, {"truth": False, "votes": [{"valid": False, "confidence": 0.9}] * 2}]
    out = adjudicated_rates(rows, cal)
    assert set(out) == {"ALT", "MUT", "all"} and out["ALT"]["n_unlabeled"] == 6 and out["all"]["n_unlabeled"] == 10
    assert out["ALT"]["decided_valid"] == 0 and out["MUT"]["decided_valid"] == 4
