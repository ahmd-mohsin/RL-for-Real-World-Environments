"""The RSI-2 machinery on the fixture domains with scripted proposers (no model is called): dev headroom, the paired
loop with its decision split (never shown to a proposer), continuous runs that resume, stop rules, the E10 ablation
harness, and the per-task student-probe → classify → repair → lift loop with a scripted (fake) student."""

from __future__ import annotations

import json

import pytest

from crucible.synth.falsify import FalsifyContext
from crucible.synth.meta.ablate import e10_summary, e10_table, run_e10
from crucible.synth.meta.headroom import LABELS as HEAD_LABELS
from crucible.synth.meta.headroom import catchable, expressible_bodies
from crucible.synth.meta.plant import PlantOp
from crucible.synth.meta.rsi import ABLATIONS, EvolveConfig, Loop, Pool, Proposers, StopRules, verdicts
from crucible.synth.meta.update import StatsConfig, UpdateConfig
from tests.unit.test_synth_rsi import EXTRA_READ, _setup


async def test_headroom_finds_what_an_expressible_operator_catches_and_never_counts_an_alarming_one():
    ctx, _clean, _, dev_src, locked_src, dev, locked, _, _, _ = await _setup()
    from crucible.synth.falsify import cheap

    plants = [*dev.plants, *locked.plants]  # in a test the fixture's locked plants are just more plants
    vs, _ = await verdicts(plants, cheap(), ctx)
    missed = [p.id for p in plants if not vs[p.id].hit]
    assert missed  # the fixture planters include gaps of the base population (over_constrain:never, eq_to_any)
    h = await catchable(plants, missed, [*dev_src, *locked_src], ctx)
    assert set(h.catchable) <= set(missed) and h.missed == len(missed) and h.plants == len(plants)
    assert h.bodies == len(expressible_bodies()) and h.operators == h.bodies * len(HEAD_LABELS) and 0 < h.admissible < h.operators
    never = [p for p in plants if p.op.key.startswith("over_constrain") and p.op.variant == "never" and p.id in missed]
    assert never and all(p.id in h.catchable for p in never)  # an unused inserted read is in the grammar (EXTRA_READ's body)
    assert any("insert_call" in h.witnesses[p.id] or "agent" in h.witnesses[p.id] for p in never)
    assert 0 < h.headroom_abs <= h.headroom_rel <= 1


def _paired_cfg(**kw) -> EvolveConfig:
    st = StatsConfig(paired=True, t_max=12, n_max=8, n_min=2, chunk=4, n_futility=4, aa_replicates=5)
    base = {"rounds": 2, "candidates": 2, "view_frac": 0.8, "gate_fresh": 8, "cost_bar": 64, "decide_frac": 0.5, "headroom": True,
            "planter_reward": "headroom", "update": UpdateConfig(rounds=2, min_band=0.0, stats=st)}
    base.update(kw)
    return EvolveConfig(**base)


def _proposers(seen: list | None = None) -> Proposers:
    async def ops(*, k, round, misses, clusters, **_):
        if seen is not None:
            seen.append({"misses": misses, "clusters": clusters})
        return [(EXTRA_READ.model_copy(update={"name": f"x{round}"}), [misses[0]["cluster"]] if misses else [])], {"lemmas": [], "strategies": []}

    async def planters(*, k, round, **_):
        return [PlantOp(name="widen_frame", variant="coll", version=101, params={"why": "edits here are fine"})], {}

    async def guidance(*, guidance, gate_outcomes):
        return [*guidance, "cite the clusters you generalise"], {}

    return Proposers(ops=ops, planters=planters, guidance=guidance, family="openai")


def _loop(tmp_path, cfg: EvolveConfig, *, own_worlds: bool = True) -> Loop:
    ctx, _clean, _, dev_src, locked_src, dev, locked, _, nat, nat_tasks = _C["s"]
    pool = Pool(dev_src)
    pool.add("P@1", dev.plants)
    if own_worlds:
        return Loop(tmp_path, pool, locked, locked_src, nat, ctx=ctx, report_ctx=FalsifyContext(), config=cfg, dev_sources=dev_src, builders=["deepseek"],
                    natural_sources=nat_tasks, make_ctx=FalsifyContext)
    return Loop(tmp_path, pool, locked, locked_src, nat, ctx=ctx, report_ctx=_C.setdefault("rctx", FalsifyContext()), config=cfg, dev_sources=dev_src,
                builders=["deepseek"], natural_sources=nat_tasks, seeded=_C.setdefault("seeded", {}))


_C: dict = {}


async def _ready():
    if "s" not in _C:
        _C["s"] = await _setup()
    return _C["s"]


async def test_the_paired_loop_keeps_the_decision_split_away_from_proposers_and_resumes(tmp_path):
    await _ready()
    seen: list = []
    cfg = _paired_cfg()
    loop = _loop(tmp_path, cfg)
    rows = await loop.run(_proposers(seen), control=False)
    arm = loop.arms[0]
    decide = set(arm.pool.decide_ids())
    assert arm.pool.decide_sources and decide and set(arm.pool.proposal_ids()).isdisjoint(decide)
    shown = json.dumps(seen)
    for pid in decide:
        assert pid not in shown  # cluster member ids never reach a proposer anyway; and no decision unit is in an evidence plant
    assert all(p.id not in decide for p in loop.evidence_plants(arm))
    r0 = rows[0]
    assert r0.info["noise_floor"]["aa"]["replicates"] if "replicates" in r0.info["noise_floor"]["aa"] else len(r0.info["noise_floor"]["aa"]["replicate_means"]) == 5
    assert {"f_recall", "headroom_abs", "headroom_rel", "pool_kl", "decide_left", "alpha_spent"} <= set(rows[-1].game)
    led = [json.loads(x)["rec"] for x in (tmp_path / "loop" / "ledger.jsonl").read_text().splitlines()]
    tests = [r for r in led if r["kind"] == "test"]
    upd = [r for r in led if r["kind"] == "update" and r["evaluated"] and r["component"] == "F"]
    assert len(tests) >= len(upd) and all(set(t["metrics"]["reserved"]) <= decide for t in tests if t["metrics"]["reserved"])
    for r in upd:  # every evaluated F decision ran on fresh decision units and reports its effect with intervals
        assert "effect" in r["metrics"] and r["metrics"]["alpha_t"] > 0
    used_once = [u for t in tests for u in t["metrics"]["reserved"]]
    consumed = arm.engine.used_units
    assert consumed <= decide and len(consumed) <= len(used_once)
    st = json.loads((tmp_path / "loop" / "state.json").read_text())
    assert st["round"] == 2 and rows[-1].stop == "max_rounds"
    # resume: a new Loop over the same directory continues from round 2 to round 3; spent α and used units carry over
    cfg3 = cfg.model_copy(update={"rounds": 3})
    loop2 = _loop(tmp_path, cfg3)
    rows2 = await loop2.run(_proposers(), control=False, resume=True)
    arm2 = loop2.arms[0]
    assert [r.round for r in rows2] == [0, 1, 2, 3] and arm2.engine.used_units >= consumed
    assert len(arm2.engine.spender.spent) >= len(arm.engine.spender.spent) and arm2.engine.floor == arm.engine.floor


async def test_stop_rules_end_a_continuous_run(tmp_path):
    await _ready()
    cfg = _paired_cfg(rounds=10, stop=StopRules(futility_rounds=1, min_headroom=1.0), planter_loop=False, headroom=False)
    loop = _loop(tmp_path, cfg, own_worlds=False)

    async def nothing(*, k, round, **_):
        return [], {}

    rows = await loop.run(Proposers(ops=nothing, family="openai"), control=False)
    assert rows[-1].stop == "futility" and rows[-1].round == 1
    loop2 = _loop(tmp_path / "b", _paired_cfg(rounds=10, headroom=False, stop=StopRules(max_spend_usd=0.0)), own_worlds=False)
    rows2 = await loop2.run(_proposers(), control=False)
    assert rows2[-1].stop == "spend" and rows2[-1].round == 0  # the spend cap is checked before every round, the first included
    arm = loop2.arms[0]
    loop2.cfg.stop.max_spend_usd = None
    while not arm.engine.spender.exhausted:
        arm.engine.spender.next("x")
    assert loop2.stop_reason(arm, 1) == "alpha_spent"
    arm.engine.spender.spent.clear()
    arm.sampler.consumed |= set(arm.pool.decide_ids())
    assert loop2.stop_reason(arm, 1) == "decision_split_dry"


def test_ablations_switch_off_one_rule_each():
    base = _paired_cfg()
    for name, spec in ABLATIONS.items():
        c = base.ablated(name)
        assert c.ablation == name and set(c.update.disabled) == set(spec.get("disabled", []))
    assert base.ablated("R1").fixed_view and not base.ablated("NL").natural_likeness and base.ablated("none").update.disabled == []


async def test_e10_harness_runs_equal_budget_arms_over_three_seeds(tmp_path):
    await _ready()
    cfg = _paired_cfg(rounds=1, headroom=False, planter_reward="miss")
    rows = await run_e10(lambda c, d: _loop(d, c, own_worlds=False), lambda s: _proposers(), cfg, tmp_path, ablations=("none", "R7", "NL"), seeds=(0, 1, 2))
    assert len(rows) == 9 and {r.ablation for r in rows} == {"none", "R7", "NL"}
    s = e10_summary(rows)
    assert "vs_none" in s["R7"]["j"] and "vs_none" not in s["none"]["j"] and len(s["NL"]["j"]["ci95"]) == 2
    t = e10_table(rows)
    assert "| R7 | 3 |" in t and "Δ" in t
    with pytest.raises(ValueError):
        await run_e10(lambda c, d: _loop(d, c), lambda s: _proposers(), cfg, tmp_path, seeds=(0, 1))


# ---------------------------------------------------------------------------------------------------------- the per-task probe loop


class FakeRollout:
    def __init__(self, i, passed, program, name):
        self.i, self.passed, self.program, self.name = i, passed, program, name


class FakeStudent:
    """Scripted probe rollouts (no model): each is a literal program, graded here by the task's own contract."""

    def __init__(self, programs, ctx):
        self.programs, self.ctx, self.calls = programs, ctx, 0
        self.student, self.checkpoint, self.simulator, self.time_mode = "student-fam", "base", "json-agent/v1", "frozen"

    async def rollouts(self, task, n, *, start=0):
        from crucible.synth.meta.rsi import replay_program

        self.calls += 1
        out = []
        for j in range(n):
            name, prog = self.programs[(start + j) % len(self.programs)]
            out.append(FakeRollout(start + j, await replay_program(task, prog, self.ctx), prog, name))
        return out


class FakeClassifier:
    """Labels each rollout from a script keyed by program name (the learnability agent's classifier is injected the
    same way)."""

    def __init__(self, labels):
        self.labels = labels

    async def classify(self, task, rollouts):
        from crucible.synth.meta.rsi import ProbeOutcome

        out = []
        for r in rollouts:
            name = r.name
            kind, mech, ev, extra = self.labels[name]
            out.append(ProbeOutcome(i=r.i, passed=r.passed, kind=kind, mechanism=mech, evidence=ev, program=r.program, **extra))
        return out


def _progs():
    from tests.unit import synth_fixtures as fx

    t = fx.t3_resolve_email()
    full = t.oracle.model_dump(mode="json")
    shortcut = t.oracle.model_copy(update={"steps": t.oracle.steps[:3]}).model_dump(mode="json")  # no customer email
    from crucible.synth.program import lit

    wrong = t.oracle.model_copy(deep=True)
    wrong.steps[2].args["resolution"] = lit("KB-1")  # a wrong article: a genuine failure of the skill
    return {"full": ("full", full), "shortcut": ("shortcut", shortcut), "wrong": ("wrong", wrong.model_dump(mode="json"))}


def _probe_loop(tmp_path, labels, repair, *, reprobe=0):
    from crucible.synth.archive import Archive
    from crucible.synth.meta.rsi import ProbeLoop, ProbeRules, archive_recheck, replay_program
    from tests.unit import synth_fixtures as fx

    ctx = FalsifyContext()
    p = _progs()
    student = FakeStudent([p["full"], p["shortcut"], p["wrong"]], ctx)
    arch = Archive(tmp_path / "arch")
    planted = fx.in_domain(fx.planted("D1"))  # the customer email is no longer required: the shortcut passes

    async def replay(task, prog):
        return await replay_program(task, prog, ctx)

    loop = ProbeLoop(tmp_path / "probe", probe=student, classify=FakeClassifier(labels), repair=repair, replay=replay,
                     recheck=archive_recheck(arch, ctx), archive=arch, rules=ProbeRules(rollouts=3, student_family="student-fam", builder_families=["deepseek"],
                                                                                       reprobe=reprobe))
    loop.add_tasks([planted])
    return loop, planted, student, arch


LABELS = {"full": ("intended_pass", "", "executed", {}), "shortcut": ("unintended_pass", "frame_hole", "executed", {}),
          "wrong": ("intended_fail", "", "executed", {})}


def _fixed(task):
    from tests.unit import synth_fixtures as fx

    good = fx.in_domain(fx.t3_resolve_email())
    return task.model_copy(update={"contract": good.contract, "version": task.version + 1})


async def test_an_executed_unintended_pass_is_archived_repaired_and_lifted(tmp_path):
    calls = []

    async def repair(task, cxs):
        calls.append([c.id for c in cxs])
        return _fixed(task)

    lifted = []

    loop, planted, _student, arch = _probe_loop(tmp_path, LABELS, repair, reprobe=3)

    async def lift(old, new, cxs):
        lifted.append(len(cxs))
        return ["lemma-1"]

    loop.lift_lemmas = lift
    (rec,) = await loop.run_round(1)
    assert rec.counted == {"intended_pass": 1, "unintended_pass": 1, "intended_fail": 1}
    assert rec.repair == "accepted" and rec.new_version == planted.version + 1 and rec.refused_by == []
    assert len(rec.counterexamples) == 1 and calls == [rec.counterexamples] and arch.records(planted.domain)[0].cls == "PROBE"
    assert rec.lessons["archive"] == rec.counterexamples and rec.lessons["lemma"] == ["lemma-1"] and lifted == [1]
    assert rec.drift is not None and "unexplained" in rec.drift  # the shortcut's pass is explained: the re-probe drops it
    (again,) = await loop.run_round(2)  # the repaired task: the shortcut now fails, labelled by the classifier as before;
    assert again.version == planted.version + 1  # resumable state: the current version is what the next round probes
    led = [json.loads(x)["rec"]["kind"] for x in (tmp_path / "probe" / "probe_ledger.jsonl").read_text().splitlines()]
    assert led[:2] == ["probe", "repair"]
    from crucible.synth.meta.rsi import ProbeLoop

    back = ProbeLoop(tmp_path / "probe", probe=None, classify=None, repair=None, replay=None, recheck=None)  # type: ignore[arg-type]
    assert back.tasks[planted.id].version == planted.version + 1 and len(back.history) == 2


async def test_a_repair_that_eases_a_skill_gap_or_over_constrains_is_refused(tmp_path):
    from crucible.synth.contract import TraceConstraint

    async def ease(task, cxs):  # fixes the shortcut AND drops the resolution requirement: the wrong resolution now passes
        from crucible.synth.contract import EffectPattern

        t = _fixed(task)
        c = t.contract.model_copy(update={"required": [p for p in t.contract.required if p.id != "r_res"],
                                          "allowed": [EffectPattern(id="a_res", app="desk", coll="tickets", key="T-7", field="resolution", why="any article")]})
        return t.model_copy(update={"contract": c})

    loop, _, _, _ = _probe_loop(tmp_path / "a", LABELS, ease)
    (rec,) = await loop.run_round(1)
    assert rec.repair == "refused" and any(x.startswith("T2") for x in rec.refused_by), rec.refused_by

    async def tighten(task, cxs):  # fixes the shortcut and demands an unstated order: the intended pass now fails
        t = _fixed(task)
        tc = TraceConstraint(id="m_x", op="before", tool="mail.send", then="desk.get_ticket")
        return t.model_copy(update={"contract": t.contract.model_copy(update={"trace": [tc]})})

    loop2, _, _, _ = _probe_loop(tmp_path / "b", LABELS, tighten)
    (rec2,) = await loop2.run_round(1)
    assert rec2.repair == "refused" and any(x.startswith("T3") for x in rec2.refused_by)

    async def nothing(task, cxs):  # a "repair" that changes nothing: the shortcut still passes
        return task.model_copy(update={"version": task.version + 1})

    loop3, _, _, _ = _probe_loop(tmp_path / "c", LABELS, nothing)
    (rec3,) = await loop3.run_round(1)
    assert rec3.repair == "refused" and any(x.startswith("T4") for x in rec3.refused_by)


async def test_skill_gaps_and_unevidenced_labels_never_trigger_a_repair(tmp_path):
    called = []

    async def repair(task, cxs):
        called.append(1)
        return _fixed(task)

    gaps = {"full": LABELS["full"], "shortcut": ("intended_fail", "", "executed", {}), "wrong": ("intended_fail", "", "executed", {})}
    loop, _, _, _ = _probe_loop(tmp_path / "a", gaps, repair)
    (rec,) = await loop.run_round(1)
    assert rec.repair == "none" and not called
    judged = {**LABELS, "shortcut": ("unintended_pass", "leak", "judged", {}), "wrong": ("unintended_fail", "ambiguous", "none", {})}
    loop2, _, _, _ = _probe_loop(tmp_path / "b", judged, repair)
    (rec2,) = await loop2.run_round(1)
    assert rec2.repair == "none" and not called and rec2.counted.get("suspect") == 1 and rec2.counted.get("unattributed_fail") == 1
    assert len(rec2.demoted) == 2 and rec2.suspects == ["rollout1"]
    alt = {**LABELS, "shortcut": ("intended_pass", "", "executed", {}),
           "wrong": ("unintended_fail", "hidden_info", "calibrated", {"alt_solved": True, "alt_family": "nemotron"})}
    loop3, _, _, _ = _probe_loop(tmp_path / "c", alt, repair)
    (rec3,) = await loop3.run_round(1)
    assert rec3.repair == "none" and not called and any("T7" in x for x in rec3.demoted)


async def test_world_bugs_become_world_builder_fixes_and_repairs_are_capped(tmp_path):
    async def repair(task, cxs):
        return None  # the repairer declines

    wb = {**LABELS, "shortcut": ("intended_pass", "", "executed", {}), "wrong": ("unintended_fail", "tool_bug", "executed", {})}
    loop, planted, _, _ = _probe_loop(tmp_path, wb, repair)
    (rec,) = await loop.run_round(1)
    assert rec.repair == "declined" and loop.world_fixes and loop.world_fixes[0]["mechanism"] == "tool_bug" and rec.lessons["world_fix"]
    loop.repairs[planted.id] = loop.rules.max_repairs
    (rec2,) = await loop.run_round(2)
    assert rec2.repair == "capped" and rec2.refused_by[0].startswith("T6")


def test_the_tuning_monitor_alarms_on_unexplained_pass_rate_moves():
    from crucible.synth.meta.rsi import TuningMonitor

    m = TuningMonitor(threshold=0.1, min_tasks=5)
    for _ in range(6):  # the fixed defects explain the change: unintended passes removed, nothing else moved
        r = m.add(n_before=8, passes_before=4, unintended_passes=2, unintended_fails=0, n_after=8, passes_after=2)
        assert r["unexplained"] == 0.0
    assert not m.alarm()["alarm"]
    t = TuningMonitor(threshold=0.1, min_tasks=5)
    for _ in range(6):  # pass rates rise far beyond what the fixed unintended fails could explain: the task drifted to the student
        t.add(n_before=8, passes_before=2, unintended_passes=0, unintended_fails=1, n_after=8, passes_after=7)
    a = t.alarm()
    assert a["alarm"] and a["mean_unexplained"] > 0.1
    assert not TuningMonitor().alarm()["alarm"]  # too few tasks: no alarm, and it says why
