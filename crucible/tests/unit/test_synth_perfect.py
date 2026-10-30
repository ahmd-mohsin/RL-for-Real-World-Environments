"""P2-P4, the perfect-task check and the frontier loop (synth.learn, synth.intent, synth.novelty, synth.perfect,
synth.frontier) on the fixture domains with a scripted student: no model is called."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest

from crucible.synth.contract import Contract, EffectPattern, PAny, PEq
from crucible.synth.falsify import FalsifyContext
from crucible.synth.falsify.agents import AgentSpec
from crucible.synth.frontier import (
    Attribution,
    BatchResult,
    FrontierPolicy,
    PoolEntry,
    TaskOutcome,
    admit,
    attribute,
    compare,
    export_attributions,
    frontier_round,
    knob_kept,
    requests_from,
)
from crucible.synth.iface import PyWorld, ToolInfo
from crucible.synth.intent import EVID, SWAP, evidence_sites, p3, pool_report
from crucible.synth.learn import (
    AgentStudentProbe,
    BandSpec,
    ContextPool,
    DifficultyStore,
    Predictor,
    Rollout,
    band_examples,
    band_verdicts,
    beta_cdf,
    calibrated,
    cost_report,
    decide,
    features,
    fit,
    loo,
    masses,
    measure,
    operating_characteristic,
    predictor_accuracy,
    training_rows,
)
from crucible.synth.novelty import assign_cell, calibrate, coverage, diversity, known_pairs, novel, score, signature
from crucible.synth.perfect import admit_pool, export_schemas, perfect
from crucible.synth.program import Program, Step, lit
from crucible.synth.task import IntentCard, Task, WorldRef
from tests.unit import synth_fixtures as fx

MOD = "tests.unit.test_synth_perfect"


def run(c: Any) -> Any:
    return asyncio.run(c)


# ---------------------------------------------------------------------------------------------------------- a World with an enum

class EnumDesk(PyWorld):
    """The helpdesk World with ``set_priority``'s priority declared as an enum (the generated Worlds declare enums)."""

    async def tools(self) -> list[ToolInfo]:
        out = []
        for t in await super().tools():
            if t.name == "desk.set_priority":
                s = dict(t.input_schema)
                props = dict(s.get("properties") or {})
                props["priority"] = {**props.get("priority", {}), "enum": ["low", "normal", "high"]}
                t = t.model_copy(update={"input_schema": {**s, "properties": props}})
            out.append(t)
        return out


def enum_desk_world() -> PyWorld:
    w = fx.helpdesk_world()
    return EnumDesk(list(w.apps.values()))


def t_priority(relaxed: bool = False) -> Task:
    val = PAny() if relaxed else PEq(v="high")
    return Task(id="desk/prio-T2" + ("-relaxed" if relaxed else ""), domain="desk-test", world=WorldRef(factory=f"{MOD}:enum_desk_world"),
                instruction="Set the priority of ticket T-2 to high.",
                oracle=Program(steps=[Step(id="s1", tool="desk.set_priority", args={"ticket_id": lit("T-2"), "priority": lit("high")})]),
                contract=Contract(required=[EffectPattern(id="r", app="desk", coll="tickets", key="T-2", field="priority", value=val)], normalizers=fx.HD_N),
                intent=IntentCard(skill="apply a stated priority"))


def t3_lax_mail() -> Task:
    """t3 with the email's recipient unchecked: the customer's address (evidence) is no longer needed."""
    t = fx.t3_resolve_email()
    req = [p.model_copy(update={"where": {**p.where, "to": PAny()}}) if p.id == "r_mail" else p for p in t.contract.required]
    return t.model_copy(update={"id": "desk/resolve-T7-lax", "contract": t.contract.model_copy(update={"required": req})}, deep=True)


# ---------------------------------------------------------------------------------------------------------- the scripted student

class LLMOutputError(Exception):
    """Named like the client's protocol error: a rollout ending on it is the student's failure, not masked."""


def scripted(task: Task, plan: dict[int, Any], calls: list[int]) -> Any:
    """runner(tag) for ``AgentStudentProbe``: rollout i follows ``plan[i]``: "oracle" (replays π*'s resolved calls and
    answer), "noop", "crash" (masked), "protocol", or a list of (tool, args) and an optional answer."""
    ctx = FalsifyContext()
    oracle = run(ctx.oracle(task))

    def runner(tag: str) -> Any:
        i = int(tag.rsplit(":", 1)[1])

        async def go(view: Any, world: Any) -> Any:
            calls.append(i)
            b = plan.get(i, "noop")
            if b == "oracle":
                for c in oracle.calls:
                    await world.call(c.tool, dict(c.args))
                return oracle.answer
            if b == "crash":
                raise RuntimeError("tinker 500")
            if b == "protocol":
                raise LLMOutputError("bad json")
            if b == "noop":
                return None
            seq, ans = b if isinstance(b, tuple) else (b, None)
            for tool, args in seq:
                await world.call(tool, dict(args))
            return ans

        return go

    return runner


def probe_of(task: Task, plan: dict[int, Any], calls: list[int]) -> AgentStudentProbe:
    return AgentStudentProbe(runner=scripted(task, plan, calls), contexts=ContextPool(FalsifyContext, size=2), student="fake/student", concurrency=2)


# ---------------------------------------------------------------------------------------------------------- P2: the band test

def test_beta_cdf_matches_the_binomial_identity() -> None:
    for a, b, x in [(1, 9, 0.1), (3, 5, 0.4), (2, 2, 0.6), (7, 1, 0.9)]:
        n = a + b - 1
        want = sum(math.comb(n, j) * x**j * (1 - x) ** (n - j) for j in range(a, n + 1))
        assert beta_cdf(x, a, b) == pytest.approx(want, abs=1e-9)
    m = masses(0.5 + 2, 0.5 + 6, BandSpec())
    assert sum(m.values()) == pytest.approx(1.0, abs=1e-5)


def test_two_look_decisions_and_stated_error() -> None:
    s = BandSpec()
    early = decide(4, 4, s, final=False)
    assert early is not None and early.decision == "too_easy" and early.confident
    assert decide(0, 4, s, final=False) is None  # 0/4 does not decide too_hard under Jeffreys: look again
    hard = decide(0, 8, s, final=True)
    assert hard is not None and hard.decision == "too_hard" and not hard.confident and 0 < hard.error < 0.3
    band = decide(2, 8, s, final=True)
    assert band is not None and band.decision == "in_band"
    oc = operating_characteristic(s)
    for r in oc["grid"]:
        assert sum(r[c] for c in ("too_hard", "in_band", "too_easy")) == pytest.approx(1.0, abs=1e-3)
        assert 4 <= r["expected_n"] <= 8
    assert set(oc["worst_error"]) == {"too_hard", "in_band", "too_easy"}
    assert oc["grid"][-1]["expected_n"] == 4.0  # p = 1 stops at the first look


def test_measure_with_a_scripted_student_and_the_store(tmp_path: Any) -> None:
    t = fx.t1_hold()
    store = DifficultyStore(tmp_path / "records.jsonl")
    calls: list[int] = []
    rec = run(measure(t, probe_of(t, {i: "oracle" for i in range(8)}, calls), store=store))
    assert rec.decision == "too_easy" and rec.looks == [4] and rec.n == 4 and rec.passes == 4 and len(calls) == 4
    t2 = fx.t2_approve_answer()
    calls2: list[int] = []
    plan = {0: "oracle", 1: "crash", 2: "noop", 3: "protocol", 5: "oracle"}
    rec2 = run(measure(t2, probe_of(t2, plan, calls2), store=store))
    assert rec2.masked == 1 and rec2.n == 8 and rec2.passes == 2 and rec2.decision == "in_band" and rec2.looks == [4, 8]
    assert any(r.stop == "protocol" and r.passed is False for r in rec2.rollouts)  # a protocol break is the student's failure
    assert rec2.key.student == "fake/student" and rec2.key.task_digest == t2.digest()
    # resumed: the stored record answers, no rollout is bought again
    calls3: list[int] = []
    again = run(measure(t2, probe_of(t2, plan, calls3), store=store))
    assert again.key.id() == rec2.key.id() and calls3 == []
    assert {r.key.task_id for r in store.all()} == {t.id, t2.id}
    rep = cost_report(store.all())
    assert rep["decisions"] == 2 and rep["stopped_at_first_look"] == 1
    v = band_verdicts(store.all())
    assert "p_hat" not in v[0].model_dump() and "passes" not in v[0].model_dump()


def test_predictor_fit_loo_trust_and_skip() -> None:
    rows = []
    for i in range(16):
        x = float(i % 4)
        k = [7, 3, 1, 0][i % 4]
        rows.append((f"t{i}", {"bias": 1.0, "steps": x}, k, 8))
    rep = loo(rows, keys=("bias", "steps"))
    assert rep["mae"] < rep["mae_baseline"] and rep["band_accuracy"] >= 0.7
    m = calibrated(rows, keys=("bias", "steps"), trust_min=12, trust_accuracy=0.5)
    assert m.trusted and m.calibration["tasks"] == 16
    assert m.p({"steps": 0.0}) > 0.6 and m.p({"steps": 3.0}) < 0.1
    # a trusted predictor with an out-of-band prior skips the rollouts
    m.prior_strength = 40.0
    t = fx.t1_hold()
    calls: list[int] = []
    rec = run(measure(t, probe_of(t, {}, calls), predictor=m, feats={"bias": 1.0, "steps": 3.0}))
    assert rec.source == "predictor" and rec.decision == "too_hard" and calls == []
    # an untrusted predictor never skips; its call is recorded beside the measured decision
    m.trusted = False
    rec2 = run(measure(t, probe_of(t, {}, calls), predictor=m, feats={"bias": 1.0, "steps": 3.0}))
    assert rec2.source == "measured" and rec2.predictor["band_call"] == "too_hard" and rec2.decision == "too_hard"
    acc = predictor_accuracy([rec2])
    assert acc["n"] == 1 and acc["band_call_accuracy"] == 1.0
    f = features(t, run(FalsifyContext().oracle(t)), alt=(1, 3))
    assert f["steps"] == 3 and f["writes"] == 1 and f["alt_share"] == pytest.approx(1 / 3)
    assert training_rows([rec2])[0][2:] == (0, 8)
    p = fit([(f, 2, 8)], keys=("bias", "steps"))
    assert isinstance(p, Predictor) and 0 < p.p(f) < 1
    ex = band_examples([rec2], m)
    assert ex and ex[0].decision == "too_hard" and ex[0].explaining == ["steps"]


# ---------------------------------------------------------------------------------------------------------- P3

def test_evidence_removal_on_and_off_intent() -> None:
    ctx = FalsifyContext()
    t = fx.t3_resolve_email()
    o = run(ctx.oracle(t))
    sites = evidence_sites(t, o)
    assert [(s.coll, s.key, s.field) for s in sites] == [("tickets", "T-7", "customer")]
    good = run(p3(t, ctx))
    assert good.on_intent is True and good.trials["EVID"] == 1 and good.confirmed["EVID"] == 0
    # the lax variant no longer checks the recipient: an unchecked value is incidental, so it has no evidence site (the
    # pilot's finding: π* copying a ticket's own priority back was called off intent); its laxness is P1's (MUT)
    lax = run(p3(t3_lax_mail(), ctx, classes=[EVID()]))
    assert lax.not_applicable["EVID"].startswith("no derived value")
    # an aggregate answer (a receipts total) has no single evidence site: not a false off-intent
    assert evidence_sites(fx.t2_approve_answer(), run(ctx.oracle(fx.t2_approve_answer()))) == []


def test_evidence_removal_agent_probe_is_a_suspect() -> None:
    ctx = FalsifyContext()
    t = fx.t1_hold()
    oracle = run(ctx.oracle(t))

    async def lucky(view: Any, world: Any) -> Any:  # holds INV-1003 without reading the purchase order
        for c in oracle.calls:
            if not c.read_only:
                await world.call(c.tool, dict(c.args))
        return None

    r = run(p3(t, ctx, agents=[AgentSpec(name="alt", family="fake", runner=lucky)]))
    assert r.trials.get("EVID", 0) >= 1 and r.suspects["EVID"] >= 1  # the decision's evidence (the PO) was not needed


def test_instruction_swap_enum_and_pool() -> None:
    ctx = FalsifyContext()
    good = run(p3(t_priority(), ctx, classes=[SWAP()]))
    assert good.on_intent is True and good.trials["SWAP"] == 2 and good.confirmed["SWAP"] == 0
    bad = run(p3(t_priority(relaxed=True), ctx, classes=[SWAP()]))
    assert bad.on_intent is False and bad.confirmed["SWAP"] >= 1
    # pool: an always-pass contract accepts a peer task's solution
    ap = fx.always_pass()
    r = run(p3(ap, ctx, pool=[fx.t2_approve_answer(), fx.t1_hold()], classes=[SWAP(pool=[fx.t2_approve_answer(), fx.t1_hold()])]))
    assert r.suspects["SWAP"] >= 1 and r.confirmed["SWAP"] == 0  # a peer's solution passing is for a judge
    # a peer with the same effect is a duplicate (no trial), reported for P4
    twin = fx.t1_hold().model_copy(update={"id": "ap/hold-1003-copy", "instruction": "Hold INV-1003 for price variance after checking its PO."})
    r2 = run(p3(fx.t1_hold(), ctx, classes=[SWAP(pool=[twin])]))
    assert r2.duplicates == ["ap/hold-1003-copy"] and r2.trials["SWAP"] == 0
    rep = pool_report([good, bad, r2])
    assert rep["measured"] == 2 and rep["share"] == 0.5 and rep["unmeasured"] == 1


def test_evid_not_applicable_without_perturb() -> None:
    class NoPerturb:
        def __init__(self) -> None:
            self.w = fx.helpdesk_world()

        def __getattr__(self, k: str) -> Any:
            if k == "perturb":
                raise AttributeError(k)
            return getattr(self.w, k)

    ctx = FalsifyContext(make_world=lambda t: NoPerturb())
    r = run(p3(fx.t3_resolve_email(), ctx, classes=[EVID()]))
    assert r.not_applicable["EVID"].startswith("the World cannot edit")


# ---------------------------------------------------------------------------------------------------------- P4

def test_signatures_near_duplicates_and_calibration() -> None:
    ctx = FalsifyContext()
    ts = [fx.in_domain(f()) for f in fx.CORRECT.values()] + [fx.in_domain(fx.t6_resolve()), fx.in_domain(t_priority())]
    sigs = [signature(t, run(ctx.oracle(t))) for t in ts]
    by = {s.task_id: s for s in sigs}
    assert by["ap/hold-1003"].tools == ["R:ap.find_invoice", "R:ap.get_purchase_order", "W:ap.hold_invoice"]
    assert by["desk/priority-bob"].tools == ["R:desk.list_tickets", "W:desk.set_priority+"]
    assert by["ap/hold-1003"].depth == 2 and "ap.purchase_orders" in by["ap/hold-1003"].read
    assert by["ap/hold-1003"].choices == ["ap.hold_invoice.reason=price_variance"]  # named in the tool's description
    assert by["desk/resolve-T3"].choices == []  # its resolution is a ref: data flow, not a stated decision
    cells = {t.id: ("decision" if "hold" in t.id or "approve" in t.id else "kb" if "resolve" in t.id else "rule") for t in ts}
    pairs = run(known_pairs(ts, ctx, cells=cells, paraphrase=lambda s: "Please " + s[0].lower() + s[1:]))
    kinds = {k for *_, k in pairs}
    assert {"twin", "twin+paraphrase", "cross_domain", "enum_swap"} <= kinds and all(not lab for *_, lab, k in pairs if k == "enum_swap")
    cal = calibrate(pairs)
    assert cal.youden > 0.5 and cal.tpr >= 0.8 and cal.fpr <= 0.2
    twin_pair = next((a, b) for a, b, lab, k in pairs if k == "twin")
    assert score(*twin_pair, cal.w) >= cal.theta
    n1 = novel(by["ap/hold-1003"], [twin_pair[1].model_copy(update={"task_id": "x"})], w=cal.w, theta=cal.theta)
    assert not n1.novel
    n2 = novel(by["ap/hold-1003"], [by["desk/resolve-T7"]], w=cal.w, theta=cal.theta)
    assert n2.novel and n2.new_key
    d = diversity(sigs, w=cal.w, theta=cal.theta)
    assert d["tasks"] == 7 and d["distinct_signatures"] == 7 and 0 < d["mean_pairwise_distance"] <= 1


def test_coverage_against_a_plain_dict_grid() -> None:
    grid = {"decide": {"match": ["hold", "approve", "invoice"]}, "kb": {"match": ["knowledge", "article"]}, "triage": "set ticket priority triage"}
    ts = [fx.t1_hold(), fx.t3_resolve_email(), fx.t2_approve_answer()]
    assigned = {t.id: assign_cell(t, grid) for t in ts}
    assert assigned == {"ap/hold-1003": "decide", "desk/resolve-T7": "kb", "ap/approve-1001": "decide"}
    assert assign_cell(fx.t4_priority(), grid, recorded="triage") == "triage"
    cov = coverage(assigned, grid)
    assert cov["covered"] == 2 and cov["coverage"] == pytest.approx(2 / 3, abs=1e-3) and cov["empty_cells"] == ["triage"]

    class Spec:  # a DomainSpec-like object: cells plus its own assignment
        cells = grid

        def cell_of(self, task: Task) -> str:
            return "triage"

    assert assign_cell(fx.t1_hold(), Spec()) == "triage"


# ---------------------------------------------------------------------------------------------------------- perfect

def test_perfect_needs_all_four(tmp_path: Any) -> None:
    ctx = FalsifyContext()
    t = fx.t3_resolve_email()
    sig = signature(t, run(ctx.oracle(t)))
    p3r = run(p3(t, ctx))
    store = DifficultyStore(tmp_path / "r.jsonl")
    rec = run(measure(t, probe_of(t, {0: "oracle", 5: "oracle", 6: "oracle"}, []), store=store))
    assert rec.decision == "in_band"
    v = perfect(t, [], store.all(), status="certified", p3=p3r, signature=sig)
    assert v.admit and v.valid and v.learnable and v.on_intent and v.novel and not v.reasons
    assert not perfect(t, [], store.all(), status="provisional", p3=p3r, signature=sig).valid
    assert perfect(t, [], store.all(), status="provisional", p3=p3r, signature=sig, accept_provisional=True).admit
    nolearn = perfect(t, [], [], status="certified", p3=p3r, signature=sig)
    assert not nolearn.admit and nolearn.reasons["learnable"][0].startswith("unmeasured")
    dup = perfect(t, [sig.model_copy(update={"task_id": "other"})], store.all(), status="certified", p3=p3r, signature=sig)
    assert not dup.novel and not dup.admit
    off = perfect(t, [], store.all(), status="certified", p3=run(p3(t_priority(relaxed=True), ctx, classes=[SWAP()])), signature=sig)
    assert not off.on_intent
    verdicts, summ = admit_pool([(t, "certified", p3r, sig), (t, "certified", p3r, sig.model_copy(update={"task_id": "again"}))], store.all())
    assert [x.admit for x in verdicts] == [True, False] and summ["admit"] == 1  # the second copy is not novel
    sch = export_schemas()
    assert {"BandVerdict", "BandExample", "Attribution", "TargetRequest", "PerfectVerdict"} <= set(sch)


# ---------------------------------------------------------------------------------------------------------- frontier

def _roll(task: Task, plan: Any) -> Rollout:
    calls: list[int] = []
    probe = probe_of(task, {0: plan}, calls)
    return run(probe.rollouts(task, 1))[0]


def test_snapshot_replay_attribution_kinds() -> None:
    ctx = FalsifyContext()
    t1 = fx.t1_hold()
    find = ("ap.find_invoice", {"number": "INV-1003"})
    # held another invoice: a side effect π* never makes
    a = run(attribute(t1, _roll(t1, [find, ("ap.hold_invoice", {"invoice_id": 104, "reason": "price_variance"})]), ctx, cell="decide"))
    assert a.kind == "forbidden" and a.step == 1 and a.mode == "frame_discipline" and a.recoverable == [True, True, False]
    # approved the target instead of holding it: no way back
    b = run(attribute(t1, _roll(t1, [find, ("ap.approve_invoice", {"invoice_id": 103})]), ctx, cell="decide"))
    assert b.kind == "irreversible" and b.step == 1 and b.mode == "evidence_gathering" and "ap.purchase_orders" in b.evidence_skipped
    # read only
    c = run(attribute(t1, _roll(t1, [find]), ctx))
    assert c.kind == "no_action" and c.mode == "acting"
    # a recoverable wrong value: the right field, another value
    t4 = fx.t4_priority()
    d = run(attribute(t4, _roll(t4, [("desk.list_tickets", {"status": "open"}), ("desk.set_priority", {"ticket_id": "T-2", "priority": "low"}),
                                     ("desk.set_priority", {"ticket_id": "T-5", "priority": "high"})]), ctx))
    assert d.kind == "wrong_write" and d.step == 1 and d.patterns == ["r_t2"] and d.mode == "derivation"
    # stopped early
    t5 = fx.t5_batch()
    e = run(attribute(t5, _roll(t5, [("ap.find_invoice", {"number": "INV-1002"}), ("ap.approve_invoice", {"invoice_id": 102})]), ctx))
    assert e.kind == "omission" and e.step == 2 and "r105s" in e.patterns
    # the changes pass, the answer does not
    t2 = fx.t2_approve_answer()
    f = run(attribute(t2, _roll(t2, ([("ap.find_invoice", {"number": "INV-1001"}), ("ap.approve_invoice", {"invoice_id": 101})], 999.0)), ctx))
    assert f.kind == "answer" and f.mode == "reporting"
    # routed, never targeted
    g = run(attribute(t1, _roll(t1, "protocol"), ctx))
    assert g.kind == "protocol" and g.routed == "harness"
    ok = run(attribute(t1, _roll(t1, "oracle"), ctx))
    assert ok.kind == "passed"
    ex = export_attributions([a, b, ok])
    assert len(ex) == 2 and ex[0]["first_wrong_step"] == 1 and ex[0]["knobs"]["harden"] == {"distractors": 1}


def _att(task: str, mode: str, cell: str = "decide", i: int = 0) -> Attribution:
    return Attribution(id=f"at-{task}-{i}", task_id=task, kind="irreversible", mode=mode, cell=cell, student="s", rollout=i)


def test_requests_admission_cap_r4_and_comparison(tmp_path: Any) -> None:
    from crucible.synth.learn import DifficultyKey, DifficultyRecord

    recs = [DifficultyRecord(key=DifficultyKey(task_id="a", task_version=1, task_digest="d", student="s"), decision="too_hard")]
    reqs = requests_from([_att("a", "decision", i=0), _att("a", "decision", i=1), _att("b", "completion", i=0)], recs)
    assert len(reqs) == 1 and reqs[0].direction == "ease" and reqs[0].knobs == {"derive_vs_given": -1} and reqs[0].traces == 2
    # only rollouts the outcome classifier called a skill gap are targets (one of the two is not: no cluster of 2 left)
    assert requests_from([_att("a", "decision", i=0), _att("a", "decision", i=1)], recs, skill_gaps={("a", 1, "s", 0)}) == []
    from crucible.synth.frontier import request_text

    txt = request_text(reqs[0], "a decision on one entity")
    assert txt.startswith("a decision on one entity") and "is now stated in the request" in txt
    pool = [PoolEntry(task_id=f"o{i}", cell=c) for i, c in enumerate(["decide", "kb", "rule", "decide", "kb", "rule", "decide", "kb"])]
    cands = [PoolEntry(task_id=f"x{i}", cell="decide", targeted=True, request="r") for i in range(4)]
    admitted, refused, info = admit(pool, cands, FrontierPolicy(kappa=1.0, h_min=0.0))
    assert len(admitted) == 2 and all("cap" in w for _, w in refused)  # 2 of 10 = 20%
    assert info["targeted_share"] <= 0.2 and all(e in [*pool, *admitted] for e in pool)
    _, ref2, _ = admit(pool, cands[:1], FrontierPolicy(kappa=0.001, cap_frac=1.0))
    assert ref2 and ref2[0][1].startswith("R4: KL")
    _, ref3, _ = admit(pool, [PoolEntry(task_id="y", cell="decide")], FrontierPolicy(kappa=1.0, h_min=5.0))
    assert ref3 and ref3[0][1].startswith("R4: H")
    tb = BatchResult(arm="targeted", budget_usd=1.0, spent_usd=1.0, tasks=[TaskOutcome(task_id="x0", perfect=True, decision="in_band", request=reqs[0].id,
                                                                                       failed_modes=["decision"]), TaskOutcome(task_id="x1")])
    cb = BatchResult(arm="control", budget_usd=1.0, spent_usd=0.9, tasks=[TaskOutcome(task_id="c0"), TaskOutcome(task_id="c1")])
    cmp = compare(tb, cb, reqs)
    assert cmp["equal_budget"] and cmp["targeted"]["perfect"] == 1 and cmp["targeted_mode_hit_rate"] == 0.5
    assert compare(tb, cb.model_copy(update={"spent_usd": 0.1}), reqs)["verdict"].startswith("unequal")
    assert knob_kept("ease", True, 0.0, 0.25)[0] and not knob_kept("ease", True, 0.3, 0.1)[0] and not knob_kept("harden", False, 0.5, 0.1)[0]

    seen: list[Any] = []

    def propose(requests: Any, budget: float) -> list[Task]:
        seen.append((requests, budget))
        base = fx.t1_hold()
        tag = "t" if requests else "c"
        return [base.model_copy(update={"id": f"{tag}{i}"}) for i in range(2)]

    def evaluate(tasks: Any, arm: str) -> BatchResult:
        return BatchResult(arm=arm, budget_usd=1.0, spent_usd=1.0, tasks=[TaskOutcome(task_id=t.id, cell="decide", perfect=True,
                                                                                      request=reqs[0].id if arm == "targeted" else None) for t in tasks])

    out = run(frontier_round(pool, [_att("a", "decision", i=0), _att("a", "decision", i=1)], recs, propose=propose, evaluate=evaluate, budget_usd=1.0,
                             policy=FrontierPolicy(kappa=1.0, h_min=0.0)))
    assert seen[0][1] == seen[1][1] == 1.0 and seen[1][0] is None  # the control batch: same budget, no requests
    assert out["comparison"]["equal_budget"] and sum(1 for a in out["admitted"] if a["targeted"]) <= 0.2 * (len(pool) + len(out["admitted"])) + 1e-9
