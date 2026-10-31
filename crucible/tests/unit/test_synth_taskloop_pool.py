"""DESIGN §7.7 item 23: denser superset Worlds for stored tasks (``synth.densify``, the finance_ap pack's ``draw(extra=)``),
the full re-certification and pool helpers of the per-task loop (``meta.loopkit``), and TaskLoop's P1 precondition,
concurrent rounds and RESKILL in cell space. Scripted fakes only: no model is called."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from crucible.synth.archive import Archive
from crucible.synth.densify import check_dense, densified, densify_job, grew, measure_level, pick_level
from crucible.synth.falsify import FalsifyContext
from crucible.synth.meta.loopkit import DomainClassifier, Recertifier, TaskKeyedContext, label_cell
from crucible.synth.meta.reskill import IntentProposal
from crucible.synth.meta.rsi import KnobMove, ProbeOutcome, TaskLoop, TaskRules, archive_recheck, replay_program
from crucible.synth.program import Program, Step, lit, ref
from crucible.synth.task import WorldRef
from tests.unit import synth_fixtures as fx

# ---------------------------------------------------------------------------------------------------------- the finance_ap superset


def _h(w) -> str:
    return hashlib.sha256(json.dumps([w.vendors, w.pos, w.receipts, w.invoices, [c.__dict__ for c in w.cases], w.tol_pm, w.hold_order],
                                     default=str).encode()).hexdigest()


def test_the_base_finance_ap_worlds_are_unchanged():
    from crucible_pack_finance_ap.world import draw

    # golden digests of generator v3 before the extra stream existed (item 23 refactor)
    assert _h(draw(0)) == "fd8079b57057ef4ebb703c57b5b3b2c6ee7ed72c54e2fa5cc93872850d656a78"
    assert _h(draw(3, 30)) == "397b5c5872be751c87337bb505aa2f8335baf4da6b6beae0277a75e29734c06f"
    assert _h(draw(11, 14)) == "c0b240461ac23b61ae313fc8298fef359c329f6c618e521c0128f0fdd38525d5"
    assert _h(draw(3, 30, 0)) == _h(draw(3, 30))


def test_a_denser_finance_ap_world_is_a_superset_with_every_decision_kept():
    from crucible_pack_finance_ap.world import decide, draw

    for seed in range(3):
        b, d = draw(seed, 30), draw(seed, 30, 20)
        for coll in ("vendors", "pos", "receipts", "invoices"):
            assert all(x in getattr(d, coll) for x in getattr(b, coll))
        assert len(d.cases) == 50
        cd = {c.invoice_id: c for c in d.cases}
        for c in b.cases:
            assert cd[c.invoice_id].__dict__ == c.__dict__
            inv = next(x for x in d.invoices if x["id"] == c.invoice_id)
            assert decide(d, inv)[0] == c.expected
        base_vendors = {v["id"] for v in b.vendors}
        new = [x for x in d.invoices if x not in b.invoices]
        assert new and not any(x["vendor_id"] in base_vendors for x in new)  # no base vendor gains an invoice
        assert _h(draw(seed, 30, 20)) == _h(d)  # deterministic


def test_the_finance_ap_densifier_and_its_policy_truth():
    from crucible_pack_finance_ap.world import draw

    from crucible.factory.synth_wire import finance_ap_denser, finance_ap_truth

    ref0 = WorldRef(factory="crucible.factory.synth_wire:finance_ap_world", params={"seed": 1, "work": "/w", "invoices": 30})
    r = finance_ap_denser(ref0, 15)
    assert r.params == {"seed": 1, "work": "/w", "invoices": 30, "extra_invoices": 15}
    assert finance_ap_denser(ref0, 0) is None and finance_ap_denser(WorldRef(factory="x:open_world"), 15) is None
    c = draw(1, 30).cases[0]
    task = {"world": r.model_dump(), "contract": {"required": [
        {"app": "erp", "coll": "invoices", "key": str(c.invoice_id), "field": "status", "value": {"op": "eq", "v": c.expected[0]}}]}}
    assert finance_ap_truth(task) == "right"
    wrong = "approved" if c.expected[0] != "approved" else "rejected"
    task["contract"]["required"][0]["value"]["v"] = wrong
    assert finance_ap_truth(task) == "wrong"


# ---------------------------------------------------------------------------------------------------------- densify (fixture World)


def fixture_denser(ref: WorldRef, level: int) -> WorldRef | None:
    return WorldRef(factory=ref.factory, params={**ref.params, "extra": int(level)}) if ref.factory.endswith(":ap_world") else None


def fixture_world_maker(work: str):
    return lambda task: task.world.build()


def test_grew_and_densified():
    assert grew({"invoices": [1, 2], "total": 2}, {"invoices": [1, 2, 3], "total": 3})
    assert not grew({"invoices": [1, 2]}, {"invoices": [1, 3, 4]})
    assert not grew({"id": 1, "status": "pending"}, {"id": 1, "status": "approved"})
    t = fx.t1_hold()
    d = densified(t, fixture_denser(t.world, 40))
    assert d.version == t.version + 1 and d.world.params["extra"] == 40
    assert (d.instruction, d.contract, d.oracle, d.intent, d.id) == (t.instruction, t.contract, t.oracle, t.intent, t.id)


async def test_an_aggregate_over_a_grown_collection_is_reported_density_sensitive():
    ctx = FalsifyContext()
    try:
        t = fx.t1_hold()
        ok = await check_dense(t, densified(t, fixture_denser(t.world, 30)), ctx)
        assert ok.ok and ok.passes and ok.same_effect and set(ok.observations.values()) == {"same"}
        # "how many invoices are pending?": an aggregate over a collection the denser World grows
        from crucible.synth.contract import AnswerCheck, Contract
        from crucible.synth.program import fn

        n = sum(1 for r in fx.INVOICES.values() if r["status"] == "pending")
        cnt = fx._with(t, id="ap/count", instruction="How many invoices are pending?",
                       oracle=Program(steps=[Step(id="s1", tool="ap.list_invoices", args={})], answer=fn("len", ref("s1.invoices"))),
                       contract=Contract(answer=[AnswerCheck(id="q", op="eq", value=n)], normalizers=fx.AP_N))
        bad = await check_dense(cnt, densified(cnt, fixture_denser(cnt.world, 30)), ctx)
        assert not bad.ok and not bad.passes and not bad.same_answer and bad.observations["s1"] == "grew"
        assert any("answer differs" in p for p in bad.problems)
    finally:
        await ctx.close()


async def test_errors_that_differ_only_by_the_worlds_paths_are_the_same():
    from crucible.synth.densify import _masked

    a = "tool_error: FileNotFoundError: No such file: '/tmp/work/ap-2-4fa788dc/w/apps/kb/x.md'"
    b = "tool_error: FileNotFoundError: No such file: '/tmp/work/ap-2-c98e012d/w/apps/kb/x.md'"
    assert _masked(a) == _masked(b) and _masked("other") != _masked(a)


async def test_levels_are_measured_on_the_shown_split_and_the_smallest_sufficient_one_is_picked():
    ctx = FalsifyContext(max_mutants=480)
    try:
        t = fx.t1_hold()
        lv0 = await measure_level(t, t, ctx, level=0)
        lv = await measure_level(t, densified(t, fixture_denser(t.world, 60)), ctx, level=60)
        assert lv.n_shown > lv0.n_shown and lv.n_heldout > 0 and lv.c_shown == 0
        assert pick_level([lv0, lv], target=lv.n_shown) == lv and pick_level([lv0], target=lv.n_shown) is None
    finally:
        await ctx.close()


def test_the_densify_worker_stops_at_the_first_sufficient_level(tmp_path):
    import asyncio

    t = fx.t1_hold()
    ctx = FalsifyContext(max_mutants=480)
    lv = asyncio.run(measure_level(t, densified(t, fixture_denser(t.world, 60)), ctx, level=60))
    asyncio.run(ctx.close())
    doc = json.loads(densify_job(t.model_dump_json(), str(tmp_path), [10, 60, 90], "tests.unit.test_synth_taskloop_pool:fixture_denser", 0.2, 480,
                                 target=lv.n_shown, make_world="tests.unit.test_synth_taskloop_pool:fixture_world_maker"))
    assert doc["chosen"] == 60 and [x["level"] for x in doc["levels"]] == [0, 10, 60]  # stops at the first sufficient level
    assert doc["dense_task"]["world"]["params"]["extra"] == 60


# ---------------------------------------------------------------------------------------------------------- loopkit


def test_label_cell_maps_free_text_labels_to_spec_skills():
    from crucible.synth.domain import finance_ap_spec
    from crucible.synth.meta.reskill import domain_skills

    sk = domain_skills(finance_ap_spec())
    assert label_cell("finance_ap", "AP invoice decision: duplicate detection with a near-identical earlier invoice", sk) == "finance_ap:duplicate_check"
    assert label_cell("finance_ap", "vendor_risk", sk) == "finance_ap:vendor_risk"
    assert label_cell("finance_ap", "AP invoice decision with bank verification hold", sk) == "finance_ap:vendor_risk"
    assert label_cell("finance_ap", "something else entirely", sk) == "finance_ap:unmapped"


async def test_a_task_keyed_context_gives_each_task_its_own_world():
    ctx = TaskKeyedContext(max_worlds=2)
    try:
        a, b = fx.t1_hold(), fx._with(fx.t1_hold(), id="ap/other")
        assert (await ctx.world(a)) is not (await ctx.world(b))
        assert (await ctx.world(a)) is (await ctx.world(a))
        await ctx.world(fx._with(fx.t1_hold(), id="ap/third"))
        assert len(ctx._worlds) == 2  # least recently used closed
    finally:
        await ctx.close()


async def test_the_recertifier_rejects_with_shown_counterexamples_only(tmp_path):
    rec = Recertifier(make_ctx=FalsifyContext, archive=Archive(tmp_path / "a"), holdout=0.2, truths=lambda t: [], paid=None,
                      builders_of=lambda t: [], log=tmp_path / "recert.jsonl")
    good = fx.t1_hold()
    assert await rec.recheck(good) == [] and (await rec.validity(good))[0] in ("certified", "provisional")
    bad = fx.planted("D2")  # a widened frame: MUT's extra writes on distractors survive (confirmed D2)
    st, cxs = await rec.validity(bad)
    assert st == "rejected" and cxs and all(c.cls != "MUT-HO" for c in cxs)
    assert (await rec.recheck(bad))[0].startswith("re-certification rejected")
    rec.seed(bad, "on_hold", ["ALT:suspects=1"])
    assert (await rec.validity(bad))[0] == "on_hold"  # a seeded (manifest) status is not re-measured
    assert (tmp_path / "recert.jsonl").read_text().count("\n") == 2


# ---------------------------------------------------------------------------------------------------------- TaskLoop: P1, concurrency, RESKILL


class R:
    def __init__(self, i, passed, program):
        self.i, self.passed, self.program = i, passed, program


class Student:
    def __init__(self, ctx):
        self.ctx = ctx

    async def rollouts(self, task, n, *, start=0):
        prog = task.oracle.model_dump(mode="json")
        return [R(i, (await replay_program(task, prog, self.ctx)) if i % 3 == 0 else False, prog if i % 3 == 0 else {"steps": [], "answer": None})
                for i in range(start, start + n)]


class Clf:
    async def classify(self, task, rollouts):
        return [ProbeOutcome(i=r.i, passed=r.passed, kind="intended_pass" if r.passed else "intended_fail", evidence="executed", program=r.program)
                for r in rollouts]


class NoKnobs:
    async def moves(self, task, direction):
        return []

    async def apply(self, task, move: KnobMove):
        return None


def _loop(tmp_path, tasks, *, validity=None, repair=None, derive=None, cell_of=None, ctx=None):
    ctx = ctx or TaskKeyedContext()
    arch = Archive(tmp_path / "arch")

    async def rep(t, cxs):
        return None

    async def p3(t):
        return True, []

    async def alt(t):
        return True

    loop = TaskLoop(tmp_path / "loop", probe=Student(ctx), classify=Clf(), repair=repair or rep, replay=lambda t, p: replay_program(t, p, ctx),
                    recheck=archive_recheck(arch, ctx), knobs=NoKnobs(), ctx=ctx, p3_check=p3, alt_check=alt, cell_of=cell_of or (lambda t: t.intent.skill),
                    archive=arch, rules=TaskRules(kappa=5.0, student_family="student"), validity=validity, derive_intent=derive,
                    skills_of=lambda t: ["three_way_match", "vendor_risk", "duplicate_check"])
    loop.add_tasks(tasks)
    return loop


async def test_a_rejected_task_is_repaired_from_its_counterexamples_and_otherwise_held(tmp_path):
    seen = []

    async def validity(t):
        return ("rejected", [fx_cx(t)]) if "bad" in t.id else ("on_hold", []) if "hold" in t.id else ("certified", [])

    async def repair(t, cxs):
        seen.append((t.id, [c.id for c in cxs]))

    tasks = [fx._with(fx.t1_hold(), id=f"ap/{k}") for k in ("bad", "onhold", "fine")]
    loop = _loop(tmp_path, tasks, validity=validity, repair=repair)
    decs = {d.task_id: d for d in await loop.run_typed_round(1, concurrency=3)}
    assert decs["ap/bad"].move == "REPAIR" and not decs["ap/bad"].accepted and seen == [("ap/bad", ["cx-test"])]
    assert loop.lineages["ap/bad"].status == "held"  # an invalid task is never reserved as a good task
    assert decs["ap/onhold"].move == "HOLD" and any("P1" in w for w in decs["ap/onhold"].why) and loop.lineages["ap/onhold"].status == "held"
    assert decs["ap/fine"].move in ("KEEP", "RESERVE", "HOLD") and decs["ap/fine"].oracle["certificate"] == "certified"
    await loop.run_typed_round(2)
    assert not [x for x in loop.decisions if x.round == 2 and x.task_id in ("ap/bad", "ap/onhold")]  # held lineages are not re-probed


def fx_cx(t):
    from crucible.synth.falsify.base import Counterexample, Probe

    return Counterexample(id="cx-test", domain=t.domain, task_id=t.id, task_version=t.version, cls="MUT", op="ExtraWrite", site="s",
                          defect="D2", probe=Probe(cls="MUT", op="ExtraWrite", site="s", program=t.oracle))


async def test_a_reskill_into_the_same_cell_confirms_the_label(tmp_path):
    calls = []

    async def derive(t, ctx):
        calls.append(t.id)
        return IntentProposal(skill="three_way_match", derive=["price_variance"], why="matches the PO")

    t = fx._with(fx.t1_hold(), id="ap/r1")
    loop = _loop(tmp_path, [t], derive=derive, cell_of=lambda x: "c:three_way_match" if "match" in x.intent.skill else "c:other")
    (d,) = await loop.run_typed_round(1)
    assert calls == ["ap/r1"] and d.move != "RESKILL" and any("confirms the label" in w for w in d.why)


async def test_concurrent_steps_reserve_the_pool_gates(tmp_path):
    loop = _loop(tmp_path, [fx._with(fx.t1_hold(), id=f"ap/c{i}") for i in range(4)])
    before = loop.pool_cells()
    loop._reserve({"x": 1.0})
    assert loop.pool_cells()["x"] == 1.0
    loop._reserve({"x": 1.0}, -1.0)
    assert loop.pool_cells() == before and loop._reserved == {}
    decs = await loop.run_typed_round(1, concurrency=4)
    assert len(decs) == 4 and len({d.task_id for d in decs}) == 4


async def test_a_failing_step_is_recorded_and_the_round_goes_on(tmp_path):
    class Boom(Clf):
        async def classify(self, task, rollouts):
            if task.id.endswith("x"):
                raise RuntimeError("boom")
            return await super().classify(task, rollouts)

    loop = _loop(tmp_path, [fx._with(fx.t1_hold(), id="ap/x"), fx._with(fx.t1_hold(), id="ap/y")])
    loop.classify = Boom()
    errs = []
    decs = await loop.run_typed_round(1, concurrency=2, on_error=lambda i, e: errs.append(i))
    assert errs == ["ap/x"] and [d.task_id for d in decs] == ["ap/y"]


# ---------------------------------------------------------------------------------------------------------- the CLI's pool helpers


def _cli():
    p = Path(__file__).resolve().parents[2] / "scripts" / "synth_taskloop.py"
    spec = importlib.util.spec_from_file_location("synth_taskloop_cli", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_unique_ids_and_rekeyed_archive_records(tmp_path):
    from crucible.synth.recert import ManifestRow

    m = _cli()
    rows = [ManifestRow(key=f"{s}::a1/x@v1", task_id="a1/x", version=1, digest="d", domain="ap-test", source=s, kinds=[], task_path="",
                        old_status="", status="provisional") for s in ("yield-1/a1", "yield-2/a1")]
    rows.append(ManifestRow(key="yield-1/a1::a1/y@v1", task_id="a1/y", version=1, digest="d", domain="ap-test", source="yield-1/a1", kinds=[],
                            task_path="", old_status="", status="provisional"))
    ids = m.unique_ids(rows)
    assert ids == {"yield-1/a1::a1/x@v1": "yield-1/a1/x", "yield-2/a1::a1/x@v1": "yield-2/a1/x"}
    t = fx.in_domain(fx._with(fx.t1_hold(), id="a1/x"))
    Archive(tmp_path / "src").add_all([fx_cx(t)])
    assert m.copy_records(tmp_path / "src", tmp_path / "dst", t.domain, "a1/x", "yield-2/a1/x") == 1
    (c,) = Archive(tmp_path / "dst").records(t.domain)
    assert c.task_id == "yield-2/a1/x" and c.id != "cx-test"
    assert DomainClassifier({}).records == []


async def test_stored_rollouts_of_the_exact_version_are_the_round_one_diagnosis(tmp_path):
    t = fx._with(fx.t1_hold(), id="ap/stored")
    loop = _loop(tmp_path, [t])
    asked = []
    inner = loop.probe

    class Spy:
        async def rollouts(self, task, n, *, start=0):
            asked.append((start, n))
            return await inner.rollouts(task, n, start=start)

    loop.probe = Spy()
    prog = t.oracle.model_dump(mode="json")
    stored = [R(i, i % 2 == 0, prog if i % 2 == 0 else {"steps": [], "answer": None}) for i in range(8)]
    loop.stored = lambda task: stored if task.id == "ap/stored" and task.version == t.version else None
    (d,) = await loop.run_typed_round(1)
    assert any("stored rollouts" in w for w in d.why) and d.rollouts["diagnosis"][:8] == list(range(8))
    assert all(start >= 8 for start, _ in asked)  # nothing re-bought at the stored indices


async def test_a_failing_deriver_is_a_refused_reskill_and_an_inapplicable_knob_reserves(tmp_path):
    async def derive(t, ctx):
        raise RuntimeError("intent_deriver: RS2: derive value 'invoice_id' is nothing the solution writes or answers")

    class Knobs:
        async def moves(self, task, direction):
            return [KnobMove(knob="distractors", direction=direction), KnobMove(knob="paginate", direction=direction)]

        async def apply(self, task, move):
            return None  # the World cannot take either (finance_ap's SQL World)

    t = fx._with(fx.t1_hold(), id="ap/k")
    loop = _loop(tmp_path, [t], derive=derive)
    loop.knobs = Knobs()

    class Easy(Student):
        async def rollouts(self, task, n, *, start=0):
            prog = task.oracle.model_dump(mode="json")
            return [R(i, await replay_program(task, prog, self.ctx), prog) for i in range(start, start + n)]

    loop.probe = Easy(loop.ctx)
    (d,) = await loop.run_typed_round(1)
    assert d.band["decision"] == "too_easy" and any("no lawful proposal" in w for w in d.why) and loop.lineages["ap/k"].reskilled
    assert d.move == "RESERVE" and "applies to this task's World" in d.why[-1] and loop.lineages["ap/k"].status == "reserved"


async def test_labels_graded_against_a_wrong_oracle_are_void_in_its_repair(tmp_path):
    from crucible.synth.falsify.oracle import ORACLE

    t = fx._with(fx.t1_hold(), id="ap/wrong")
    wrong_reason = Program(steps=[Step(id="s1", tool="ap.find_invoice", args={"number": lit("INV-1003")}),
                                  Step(id="s3", tool="ap.hold_invoice", args={"invoice_id": ref("s1.invoices.0.id"), "reason": lit("missing_receipt")})])

    async def repair(task, cxs):  # the repaired reference decision: the student's "failing" decision was the right one
        return task.next_version(oracle=wrong_reason, contract=task.contract.model_copy(update={"required": fx._hold("103", "missing_receipt")}))

    loop = _loop(tmp_path, [t], repair=repair)
    loop.oracle = ORACLE(truths=[lambda x: "wrong" if x.version == 1 else "right"])

    async def clean(task):
        return []

    loop.recheck = clean

    class Applies(Student):  # every rollout holds with missing_receipt: a fail under v1, a pass under the repair
        async def rollouts(self, task, n, *, start=0):
            prog = wrong_reason.model_dump(mode="json")
            return [R(i, await replay_program(task, prog, self.ctx), prog) for i in range(start, start + n)]

    loop.probe = Applies(loop.ctx)
    (d,) = await loop.run_typed_round(1)
    assert d.move == "REPAIR" and d.accepted and not any(r.startswith("T2") for r in d.refused_by)
    assert any("label(s) void" in w for w in d.why)


async def test_a_refused_knob_is_not_retried_on_fresh_rollouts(tmp_path):
    applied = []

    class Same:
        async def moves(self, task, direction):
            return [KnobMove(knob="paginate", direction=direction)]

        async def apply(self, task, move):
            applied.append(task.id)
            return task.model_copy(update={"id": f"{task.id}~paginate", "version": 1}, deep=True)  # as easy as before

    class Easy(Student):
        async def rollouts(self, task, n, *, start=0):
            prog = task.oracle.model_dump(mode="json")
            return [R(i, await replay_program(task, prog, self.ctx), prog) for i in range(start, start + n)]

    t = fx._with(fx.t1_hold(), id="ap/same")
    loop = _loop(tmp_path, [t])
    loop.knobs, loop.probe = Same(), Easy(loop.ctx)
    (d1,) = await loop.run_typed_round(1)
    assert d1.move == "ASCEND" and not d1.accepted and any(r.startswith("G4") for r in d1.refused_by)
    (d2,) = await loop.run_typed_round(2)
    assert applied == ["ap/same"] and d2.move == "RESERVE" and loop.lineages["ap/same"].refused_knobs == ["paginate"]
