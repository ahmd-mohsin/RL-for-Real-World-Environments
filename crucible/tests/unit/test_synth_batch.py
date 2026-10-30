"""crucible.synth.batch / yieldkit / offload / failures: the batch runner on the fixture helpdesk World with scripted roles.
No model is called."""

from __future__ import annotations

import asyncio
import json

import pytest

from crucible.actors.llm import BudgetExceeded
from crucible.synth.batch import Arm, ArmSetup, BatchConfig, BatchDomain, Cand, CellSpec, Resilient, arm_metrics, bootstrap_ratio, cand_costs, load_cands, report_md, run_batch
from crucible.synth.failures import alt_mode, mine_run, primary_reason, summarise
from crucible.synth.falsify import FalsifyContext, cheap, run_classes
from crucible.synth.falsify.agents import ALT, AgentSpec, llm_alt
from crucible.synth.offload import ConcurrentAgents, FalsifyPool
from crucible.synth.pipeline import DomainEnv, RunConfig, RunRoles
from crucible.synth.task import WorldRef
from crucible.synth.yieldkit import Deduper, agreement, decision_values, entities, lemma_clusters, lemma_leaks, pre_checks, reconcile
from tests.unit import synth_fixtures as fx
from tests.unit.synth_fakes import FakeClient, const, section, seq
from tests.unit.test_synth_pipeline import ORACLES, T_PRIO, T_RESOLVE, alt_script, panel_script

LEAKY = "Set the priority of ticket T-4 to high now."
GHOST = "Resolve ticket T-999 with the refund policy article."


def proposer_script(msgs, n):
    cell = section(msgs, "cell")
    if "evidence" in str(cell):
        return {"tasks": [{"skill": "answer from the kb", "instruction": T_RESOLVE, "derive": ["KB-2"]},
                          {"skill": "triage", "instruction": T_PRIO, "derive": []}]}
    # the second cell repeats the first cell's tasks (duplicates), plus a leaked decision and a ticket the World lacks
    return {"tasks": [{"skill": "answer from the kb", "instruction": T_RESOLVE + " ", "derive": ["KB-2"]},
                      {"skill": "triage", "instruction": LEAKY, "derive": ["high"]},
                      {"skill": "ghost", "instruction": GHOST, "derive": []}]}


PRIO_T4 = {"steps": [{"id": "s1", "tool": "desk.set_priority", "args": {"ticket_id": "T-4", "priority": "high"}}]}


def oracle_any(msgs, _n):
    return ORACLES.get(section(msgs, "task"), PRIO_T4)


def alt_any(msgs, n):
    return alt_script(msgs, n) if section(msgs, "instruction") in ORACLES else {"done": True, "final": None}


def roles() -> RunRoles:
    return RunRoles(proposer=FakeClient("kimi", proposer_script), oracle_programmer=FakeClient("deepseek", oracle_any),
                    contract_writer=FakeClient("glm", const({"edits": []})), repairer=FakeClient("deepseek", const({"give_up": True})),
                    alt=[("alt-n", FakeClient("nemotron", alt_any)), ("alt-o", FakeClient("openai", alt_any))],
                    atk=[("atk-n", FakeClient("nemotron", seq({"done": True, "final": None})))],
                    panel=[("p-n", FakeClient("nemotron", panel_script)), ("p-o", FakeClient("openai", panel_script))],
                    abstractor=FakeClient("openai", const({"lemmas": []})))


def env() -> DomainEnv:
    return DomainEnv(name="synth-test-batch", description="An IT helpdesk with a knowledge base and mail.", seeds=(0,), normalizers=fx.HD_N,
                     world=lambda s: WorldRef(factory=f"{fx.MOD}:helpdesk_world"))


def domain(**kw) -> BatchDomain:
    return BatchDomain(env(), cells=("a decision that needs evidence from two records", "a figure the user asks for"), quota=1, max_proposals=3, **kw)


def setup(arm: Arm, rl: RunRoles | None = None) -> ArmSetup:
    return ArmSetup(arm=arm, roles=rl or roles(), cost=lambda: {"total_usd": 0.0})


def cfg(**kw) -> BatchConfig:
    base = {"run": "t", "run_cfg": RunConfig(calibrate=False), "batch": 3, "propose_workers": 1, "build_workers": 2, "refine_workers": 2,
            "model_calls": 4, "falsify_procs": 0}
    return BatchConfig(**{**base, **kw})


async def test_two_arms_pipelined_with_dedup_prefilters_and_state_files(tmp_path):
    out = await run_batch([setup(Arm.baseline("a0")), setup(Arm(name="a1", pre=True))], [domain()], cfg(), tmp_path)
    for a in ("a0", "a1"):
        cands = {c.proposal["instruction"].strip(): c for c in load_cands(tmp_path / a) if c.status != "duplicate"}
        assert (tmp_path / a / "metrics.json").exists() and (tmp_path / a / "props").exists()
        assert cands[T_RESOLVE].status == "provisional" and cands[T_PRIO].status == "provisional"  # short programs: MUT n < 59
        dups = [c for c in load_cands(tmp_path / a) if c.status == "duplicate"]
        assert dups and all(c.reasons == ["dedup:exact"] and c.dup_of for c in dups)  # the repeated instruction (one trailing space)
        m = out["arms"][a]
        f = m["funnel"]
        assert f["proposed"] == len(load_cands(tmp_path / a)) and f["unique"] == f["proposed"] - len(dups)
        assert f["contract"] <= f["oracle_executed"] <= f["prefilter_pass"] <= f["unique"]
        assert m["speed"]["serial_equivalent_s"] > 0 and m["speed"]["wall_s"] > 0
    a1 = {c.proposal["instruction"]: c for c in load_cands(tmp_path / "a1")}
    assert a1[LEAKY].status == "filtered" and any(r.startswith("prefilter:leak") for r in a1[LEAKY].reasons)
    assert a1[GHOST].status == "filtered" and "prefilter:unknown_entity:T999" in a1[GHOST].reasons[0]
    a0 = {c.proposal["instruction"]: c for c in load_cands(tmp_path / "a0")}
    assert a0[LEAKY].status != "filtered"  # the baseline arm has no pre-filter
    assert "| admitted per USD" in report_md(out)


async def test_resume_finishes_open_candidates_without_new_proposals(tmp_path):
    rl = roles()
    await run_batch([setup(Arm.baseline("a0"), rl)], [domain()], cfg(), tmp_path)
    cands = load_cands(tmp_path / "a0")
    done = {c.id: c.status for c in cands}
    # simulate a crash: one candidate cut off mid-refine, one before its build
    a, b = [c for c in cands if c.task is not None][:2]
    a.stage, a.status = "refining", "pending"
    b.stage, b.status, b.task, b.record = "proposed", "pending", None, None
    for c in (a, b):
        (tmp_path / "a0" / "cands" / (c.id.replace("/", "_") + ".json")).write_text(c.model_dump_json())
    rl2 = roles()
    await run_batch([setup(Arm.baseline("a0"), rl2)], [domain()], cfg(), tmp_path)
    after = {c.id: c for c in load_cands(tmp_path / "a0")}
    assert not rl2.proposer.calls  # every cell had reached its proposal maximum: nothing new is proposed
    assert after[a.id].attempt == 2 and after[a.id].status == done[a.id] and after[a.id].task_id.endswith("~2")
    assert after[b.id].status == done[b.id] and after[b.id].stage == "done"
    assert all(after[k].status == v for k, v in done.items())  # the same outcome as the uninterrupted run


async def test_the_offloaded_deterministic_classes_give_the_in_process_results():
    t = fx.anchors()[3]
    ctx = FalsifyContext()
    try:
        local = {r.cls: (r.applicable, r.n, r.c, sorted(f.label for f in r.findings if f.counted)) for r in await run_classes(t, cheap(), ctx)}
        pool = FalsifyPool(2)
        try:
            off = {r.cls: (r.applicable, r.n, r.c, sorted(f.label for f in r.findings if f.counted)) for r in await run_classes(t, pool.falsifiers(), ctx)}
        finally:
            pool.close()
    finally:
        await ctx.close()
    assert off == local and pool.jobs == 1  # one worker job served every class of the task version


async def test_concurrent_agents_run_at_once_and_keep_the_labels():
    t = fx.anchors()[4]
    running, peak = 0, 0

    def runner(delay):
        async def go(view, world):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(delay)
            running -= 1
        return go

    alt = ALT([AgentSpec("a", "nemotron", runner(0.05)), AgentSpec("b", "openai", runner(0.05))])
    ctx = FalsifyContext()
    try:
        (r,) = await run_classes(t, [ConcurrentAgents(alt)], ctx)
    finally:
        await ctx.close()
    assert peak == 2 and r.n == 2 and all(f.label == "suspect" for f in r.findings)  # an attempt that does nothing fails: a suspect D6


def test_duplicate_agent_names_are_refused():
    c = FakeClient("nemotron", const({"done": True}))
    with pytest.raises(ValueError, match="unique"):
        ALT([llm_alt("alt-nemotron", c), llm_alt("alt-nemotron", c)])


async def test_the_finish_nudge_asks_once_for_the_write():
    t = fx.anchors()[4]
    replies = iter([{"call": {"tool": "desk.get_ticket", "args": {"ticket_id": "T-4"}}}, {"done": True, "final": None},
                    {"call": {"tool": "desk.set_priority", "args": {"ticket_id": "T-4", "priority": "high"}}}, {"done": True, "final": None}])
    c = FakeClient("nemotron", lambda _m, _n: next(replies))
    from crucible.synth.falsify.agents import FINISH_NUDGE, run_agent

    ctx = FalsifyContext()
    try:
        r = await run_agent(llm_alt("a", c, finish_nudge=True), t, ctx)
    finally:
        await ctx.close()
    assert [x.tool for x in r.calls] == ["desk.get_ticket", "desk.set_priority"]
    assert FINISH_NUDGE in c.calls[2][1]["content"] and FINISH_NUDGE not in c.calls[1][1]["content"]


async def test_resilient_retries_transient_errors_only_and_bounds_calls_in_flight():
    gate = asyncio.Semaphore(2)
    slept: list[float] = []

    async def nap(s):
        slept.append(s)

    class Flaky:
        family, model = "x", "m"

        def __init__(self):
            self.n, self.live, self.peak = 0, 0, 0

        async def complete_json(self, messages, **kw):
            self.n += 1
            self.live += 1
            self.peak = max(self.peak, self.live)
            await asyncio.sleep(0.01)
            self.live -= 1
            if kw.get("tag") == "boom" and self.n < 3:
                raise ConnectionError("reset")
            if kw.get("tag") == "budget":
                raise BudgetExceeded("cap")
            return {"ok": True}

    f = Flaky()
    r = Resilient(f, gate, retries=3, backoff_s=1.0, sleep=nap)
    assert await r.complete_json([], tag="boom") == {"ok": True} and r.retried == 2 and len(slept) == 2
    with pytest.raises(BudgetExceeded):
        await r.complete_json([], tag="budget")
    f2 = Flaky()
    r2 = Resilient(f2, gate)
    await asyncio.gather(*(r2.complete_json([], tag="x") for _ in range(6)))
    assert f2.peak <= 2


def test_dedup_exact_and_near_but_not_the_same_template_on_another_entity():
    d = Deduper(0.8)
    d.add("c1", "w/0", "Process pending invoice INV-50504 from Granite Print under the AP policy.", ["duplicate"])
    assert d.check("w/0", "process pending invoice INV-50504 from granite print under the ap policy", ["duplicate"])[0] == "exact"
    assert d.check("w/0", "Please decide invoice INV-50504 (Granite Print) by the AP policy and apply it.", ["duplicate"]) == ("near", "c1")
    assert d.check("w/0", "Process pending invoice INV-77777 from Granite Print under the AP policy.", ["duplicate"]) == (None, None)
    assert d.check("w/1", "Process pending invoice INV-50504 from Granite Print under the AP policy.", ["duplicate"]) == (None, None)
    assert entities("ticket T-1003 and PO-5009, INV 12") == frozenset({"T1003", "PO5009"})


def test_prefilters_agreement_and_reconcile():
    world = json.dumps({"erp": {"invoices": {"104": {"number": "INV-50504"}}}})
    assert pre_checks("Process invoice INV-50504.", ["duplicate"], world) == []
    assert pre_checks("Process invoice INV-99999.", [], world) == ["prefilter:unknown_entity:INV99999"]
    assert pre_checks("Reject invoice INV-50504 as a duplicate.", ["duplicate"], world)[0].startswith("prefilter:leak")
    eff = [{"change": "erp.invoices[104].status", "new": "on_hold"}, {"change": "erp.invoices[104].reason", "new": "price_variance"},
           {"change": "erp.invoices[104].updated_at", "new": "2026-01-02T10:00:00Z"}]
    assert decision_values(eff) == frozenset({"on_hold", "price_variance"})  # clock values are not decisions
    assert agreement(["price_variance"], eff) is None
    assert agreement(["missing_receipt"], eff) == "prefilter:disagree:proposer:missing_receipt"
    assert reconcile(["missing_receipt"], eff, None, "Decide invoice INV-50504 under the AP policy.") == ["on_hold", "price_variance"]
    assert reconcile(["price_variance"], eff, None, "Put invoice INV-50504 on_hold.") == ["price_variance"]  # a stated value is not derived


def test_scheduler_takes_the_largest_deficit_and_closes_satisfied_cells(tmp_path):
    from crucible.synth.batch import ArmRun

    a = ArmRun(setup(Arm.baseline()), [domain()], cfg(), tmp_path, asyncio.Semaphore(1), None)
    c0, c1 = a.cells
    assert isinstance(c0, CellSpec) and a.pick() == c0  # nothing proposed: the first cell
    a.cands["x"] = Cand(id="x", n=1, arm="a0", domain=c0.domain, seed=0, cell=c0.kind, cell_key=c0.key, batch_key=f"{c0.key}-b0", status="certified")
    assert a.pick() == c1  # cell 0 met its quota of 1
    for i in range(3):
        a.cands[f"y{i}"] = Cand(id=f"y{i}", n=2 + i, arm="a0", domain=c1.domain, seed=0, cell=c1.kind, cell_key=c1.key, batch_key=f"{c1.key}-b0",
                                status="rejected")
    assert a.pick() is None  # cell 1 reached its proposal maximum


def test_costs_per_candidate_and_bootstrap_interval():
    cs = [Cand(id=f"a0/c-b0-{i}", n=i, arm="a0", domain="d", seed=0, cell="k", cell_key="c", batch_key="c-b0", status=s, task={} if s != "duplicate" else None)
          for i, s in enumerate(["certified", "rejected", "duplicate"])]
    usage = [{"tag": "propose:a0/c-b0", "stage": "propose", "usd": 0.3}, {"tag": "oracle:a0/c-b0-0", "stage": "oracle", "usd": 0.1},
             {"tag": "ALT:a0/c-b0-1~2", "stage": "ALT", "usd": 0.2}, {"tag": "calibrate", "stage": "calibrate", "usd": 0.2}]
    per, st = cand_costs(cs, usage)
    assert per["a0/c-b0-0"] == pytest.approx(0.1 + 0.1 + 0.1) and per["a0/c-b0-1"] == pytest.approx(0.1 + 0.2 + 0.1)
    assert per["a0/c-b0-2"] == pytest.approx(0.1) and st["propose"] == 0.3
    p, lo, hi = bootstrap_ratio([1, 0, 1, 0], [1.0, 1.0, 1.0, 1.0])
    assert p == 0.5 and 0.0 <= lo <= p <= hi <= 1.0
    m = arm_metrics(cs, usage, wall_s=10.0)
    assert m["admitted"] == 1 and m["yield"]["n"] == 2 and m["admitted_per_usd"]["point"] == pytest.approx(1 / 0.8, rel=1e-3)


def test_failure_modes_and_primary_reasons(tmp_path):
    assert primary_reason("rejected", ["MUT:upper=0.06>0.05", "ALT:no_pass", "TWIN:confirmed=2"]) == "TWIN:confirmed"
    assert primary_reason("rejected", ["MUT:upper=0.06>0.05", "ALT:no_pass"]) == "ALT:no_pass"
    v = lambda ok, why: {"valid": ok, "confidence": 0.9, "why": why}
    assert alt_mode({"decision": "invalid", "votes": [v(False, "The trace is empty."), v(False, "No actions were taken.")]}) == "no_action"
    assert alt_mode({"decision": "invalid", "votes": [v(False, "It approved the invoice instead of rejecting it."), v(False, "wrong")]}) == "wrong_decision"
    assert alt_mode({"decision": "invalid", "votes": [v(False, "The trace only reads data and never makes a write call."), v(False, "x")]}) == "no_write"
    assert alt_mode({"decision": "abstain", "votes": [v(False, "x"), v(True, "y")]}) == "split"
    assert alt_mode({"decision": "valid", "votes": [v(True, "fine"), v(True, "fine")]}) == "valid"


async def test_mining_a_run_directory(tmp_path):
    await run_batch([setup(Arm.baseline("a0"))], [domain(truth=lambda t: "right")], cfg(), tmp_path)
    rows = mine_run(tmp_path / "a0" / "synth-test-batch", truth=lambda t: "right")
    s = summarise(rows)
    assert s["tasks"] == len(rows) > 0 and set(s["status"]) <= {"provisional", "rejected", "certified", "dropped", "error", "filtered", "duplicate"}
    assert all(r["oracle_truth"] == "right" for r in rows if r["stage"] == "refined")


def test_lemma_clusters_and_the_leakage_critic():
    pairs = [{"task": "d/r/t1", "cx": ["TWIN/twin/D9"], "changed": ["contract"], "notes": "pinned the key <id>",
              "evidence": {"task_ids": ["d/r/t1"], "entity_keys": ["T1003"], "literals": ["GRP-02"]}}] * 2
    cl = lemma_clusters(pairs)
    assert cl[0]["size"] == 2 and cl[0]["counterexamples"] == "TWIN/twin/D9" and "task" not in cl[0]
    leaks = lemma_leaks(["Pin the ticket key the instruction names.", "Route ticket T1003 to GRP-02."], pairs)
    assert leaks["Pin the ticket key the instruction names."] == [] and leaks["Route ticket T1003 to GRP-02."]


async def test_the_oracle_programmer_decides_blind_to_derive_and_the_screen_can_reconcile(tmp_path):
    rl = roles()
    arm = Arm(name="a1", oracle_sees_derive=False, agree="reconcile")
    await run_batch([setup(arm, rl)], [domain()], cfg(), tmp_path)
    shown = [m[1]["content"] for m in rl.oracle_programmer.calls]
    assert shown and all('"derive"' not in s for s in shown)
    c = next(x for x in load_cands(tmp_path / "a1") if x.proposal["instruction"] == T_RESOLVE)
    assert c.checks.get("agree") is None and c.status == "provisional"  # the blind oracle reached the proposer's KB-2


def test_stage_meter_caps_a_stage_and_restarts_from_the_receipts(tmp_path):
    from crucible.factory.synth_wire import stage_meter
    from crucible.train.bridges.tinker.spend import SpendMeter, TinkerCapReached

    (tmp_path / "spend.jsonl").write_text(json.dumps({"kind": "sample", "model": "m", "usd": 0.3, "stage": "ALT"}) + "\n")
    run = SpendMeter(5.0, path=tmp_path / "spend.jsonl")
    alt = stage_meter(run, "ALT", 0.5)
    assert alt.spent == pytest.approx(0.3)  # resumed from the run's receipt rows tagged with the stage
    with alt.hold(0.15, "a"), pytest.raises(TinkerCapReached), alt.hold(0.1, "b"):
        pass
    assert run.refused == 0 and alt.refused == 1  # the run cap was not the one that refused
    alt.book("checkpoint", None)
    assert '"stage": "ALT"' in (tmp_path / "spend.jsonl").read_text().splitlines()[-1]


def test_unique_alt_names_on_the_wire(tmp_path):
    from crucible.factory.synth_wire import SynthWire

    w = SynthWire(tmp_path, cap_usd=1.0, make=lambda model, role: object())
    names = [n for n, _ in w.run_roles().alt]
    assert len(names) == len(set(names)) == 3 and "alt_solver-nemotron-0" in names


async def test_panel_calibration_does_not_drop_tasks_built_meanwhile(tmp_path):
    """Smoke run 1: calibration once swapped ``Runner.ready`` while other builds appended to it, and a built task was lost."""
    out = await run_batch([setup(Arm.baseline("a0"))], [domain()], cfg(run_cfg=RunConfig(calibrate=True, calib_tasks=1), build_workers=4), tmp_path)
    cands = load_cands(tmp_path / "a0")
    assert not [c for c in cands if c.status == "error"], [(c.id, c.reasons) for c in cands]
    assert (tmp_path / "a0" / "synth-test-batch" / "calibration.json").exists() and out["arms"]["a0"]["funnel"]["falsified"] >= 2


async def test_a_wrapped_cap_refusal_is_a_budget_stop_not_a_model_failure(tmp_path):
    """yield-2: the engine wrapped TinkerCapReached in EngineCallError; the wire took it for a failed call, the panel
    abstained, the ALT attempts failed, and the arm never stopped."""
    from crucible.factory.synth_wire import Usage, WireClient
    from crucible.train.bridges.tinker.spend import TinkerCapReached
    from crucible.train.world_env import EngineCallError

    class Capped:
        async def complete(self, msgs, **kw):
            try:
                raise TinkerCapReached("cap 4.5 USD: refused")
            except TinkerCapReached as e:
                raise EngineCallError(f"engine: {e}", failure_kind=e.failure_kind) from e

    w = WireClient(Capped(), role="panel", model="m", family="nemotron", usage=Usage())
    with pytest.raises(BudgetExceeded):
        await w.complete_json([{"role": "user", "content": "x"}], tag="adjudicate:t")
    slept: list[float] = []

    async def nap(s):
        slept.append(s)

    class Inner:
        family, model = "x", "m"

        async def complete_json(self, messages, **kw):
            try:
                raise TinkerCapReached("cap")
            except TinkerCapReached as e:
                raise EngineCallError("engine", failure_kind=e.failure_kind) from e

    with pytest.raises(BudgetExceeded):
        await Resilient(Inner(), asyncio.Semaphore(1), sleep=nap).complete_json([], tag="ALT:t")
    assert slept == []  # never retried


async def test_serial_mode_with_backpressure_does_not_wait_forever_for_calibration(tmp_path):
    """speed-ser: the paused proposer (backlog 2) left 3 built tasks while calibration waited for 4 anchors."""
    c = cfg(run_cfg=RunConfig(calibrate=True, calib_tasks=4), serial=True, max_candidates=5)
    out = await asyncio.wait_for(run_batch([setup(Arm.baseline("a0"))], [domain()], c, tmp_path), timeout=120)
    assert out["serial"] and out["arms"]["a0"]["funnel"]["falsified"] >= 1
    assert not [x for x in load_cands(tmp_path / "a0") if x.status in ("pending", "error")]


async def test_two_attempts_of_one_model_get_two_seeds(tmp_path):
    """yield-1/2: both Ultra ALT attempts sent one prompt on seed 0, and 120 of 216 pairs were one trajectory twice. The
    pipeline tags each agent's calls with its name and the batch wire seeds by tag."""
    from crucible.factory.synth_wire import SynthWire

    seeds: list[int] = []

    class Inner:
        async def complete(self, msgs, **kw):
            seeds.append(kw["seed"])

            class R:
                text, raw_finish, cost_usd = '{"done": true, "final": null}', "stop", 0.0
            return R()

    w = SynthWire(tmp_path, cap_usd=1.0, make=lambda model, role: Inner(), seed_by_tag=True)
    (n0, c0), (n1, c1) = w.run_roles().alt[:2]
    await c0.complete_json([], tag=f"ALT:t1#{n0}")
    await c1.complete_json([], tag=f"ALT:t1#{n1}")
    assert seeds[0] != seeds[1]


def test_a_read_that_errors_is_evidence_not_a_dead_oracle():
    """yield-1: the blind oracle read the purchase order a po_mismatch invoice bills; it does not exist, and the oracle was
    dropped as dead. A read-only error is evidence; a refused write still is a problem."""
    from crucible.synth.iface import Obs
    from crucible.synth.pipeline import run_problems
    from crucible.synth.program import CallRecord, Run

    read = CallRecord(step="s1", tool="erp.get_purchase_order", obs=Obs(error="no purchase order PO-8021"), read_only=True)
    write = CallRecord(step="s2", tool="erp.hold_invoice", obs=Obs(value={"ok": True}), read_only=False)
    assert run_problems(Run(calls=[read, write])) == []
    bad = CallRecord(step="s2", tool="erp.hold_invoice", obs=Obs(error="bad reason"), read_only=False)
    assert run_problems(Run(calls=[read, bad]))[0].startswith("step s2 (erp.hold_invoice) returned an error")
