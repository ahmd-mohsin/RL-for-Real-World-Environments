"""crucible.synth.meta.rsi + abstract: the regularised RSI loop end to end on the fixture domains, with scripted
proposers (no model is called; the scripted operator is a hand-written test fixture)."""

from __future__ import annotations

import json
import random

import pytest

from crucible.synth.falsify import FalsifyContext, cheap, falsify
from crucible.synth.falsify.base import Counterexample
from crucible.synth.meta.abstract import (
    GUIDANCE_EDITOR,
    OPERATOR_ABSTRACTOR,
    PLANTER_PROPOSER,
    OperatorsOut,
    PlantVerdict,
    miss_clusters,
    natural_clusters,
    parse_operators,
    random_operator,
)
from crucible.synth.meta.library import InsertCall, LabelRule, OperatorSpec, SReadTool, compile_operator
from crucible.synth.meta.plant import Planter, PlantOp, base_planter, held_out_planter
from crucible.synth.meta.recall import LockedSetError, SetStore
from crucible.synth.meta.rsi import (
    EvolveConfig,
    Loop,
    Pool,
    Proposers,
    Sampler,
    build_sets,
    e1_estimates,
    natural_features,
    recheck,
    refusals,
    rows_markdown,
    split_sources,
    swarm_proposers,
)
from crucible.synth.meta.update import UpdateConfig
from crucible.synth.roles import RoleLeak
from tests.unit import synth_fixtures as fx
from tests.unit.synth_fakes import FakeClient, section

EXTRA_READ = OperatorSpec(name="extra_read", body=InsertCall(tool=SReadTool(unused=True)), label=LabelRule(expect="fail", defect="D6", confirm="never"),
                          why="a valid solution may read more than the reference did")
DEV = Planter(id="dev", ops=[PlantOp(name="over_constrain", variant="before"), PlantOp(name="drop_required"), PlantOp(name="relax_value", variant="eq_to_any")],
              max_per_op=1)
LOCKED = Planter(id="held-out", ops=[PlantOp(name="over_constrain", variant="never"), PlantOp(name="drop_required", variant="last")], max_per_op=1)
_C: dict[str, object] = {}


async def _natural() -> tuple[list[Counterexample], dict]:
    out, tasks = [], {}
    for code in ("D2", "D4", "D9"):
        t = fx.in_domain(fx.planted(code))
        tasks[t.id] = t
        for f in cheap():
            r = await falsify(t, f, FalsifyContext())
            out += [Counterexample.of(t, x) for x in r.confirmed]
    return out, tasks


async def _setup():
    if "s" not in _C:
        ctx = FalsifyContext()
        clean, infos = await recheck([(t, _info(t)) for t in fx.anchors()], ctx)
        dev_src, locked_src = split_sources(clean, seed=0)
        dev, locked, info = await build_sets(dev_src, locked_src, ctx, name="t", dev_planter=DEV, locked_planter=LOCKED)
        nat, nat_tasks = await _natural()
        _C["s"] = (ctx, clean, infos, dev_src, locked_src, dev, locked, info, nat, nat_tasks)
    return _C["s"]


def _info(t):
    from crucible.synth.meta.rsi import SourceInfo

    return SourceInfo(task_id=t.id, domain=t.domain, run="fixture", status="certified")


async def test_sources_split_and_the_two_sets_share_neither_tasks_nor_variants(tmp_path):
    ctx, clean, infos, dev_src, locked_src, dev, locked, _, _, _ = await _setup()
    assert len(clean) == 5 and all(not i.alarms for i in infos)  # the fixture anchors are clean
    assert {t.id for t in dev_src}.isdisjoint({t.id for t in locked_src}) and {t.domain for t in dev_src} == {"ap-test", "desk-test"}
    assert {p.source_id for p in dev.plants} <= {t.id for t in dev_src} and {p.source_id for p in locked.plants} <= {t.id for t in locked_src}
    assert {p.op.key for p in dev.plants}.isdisjoint({p.op.key for p in locked.plants}) and dev.plants and locked.plants
    with pytest.raises(ValueError, match="share"):
        await build_sets(dev_src, locked_src, ctx, name="x", dev_planter=DEV, locked_planter=DEV)
    assert not ({o.key for o in base_planter().ops} & {o.key for o in held_out_planter().ops})
    store = SetStore(tmp_path / "dev", tmp_path / "sealed")
    store.write(locked)
    with pytest.raises(LockedSetError):
        store.read(locked.name, "locked")
    assert store.read(locked.name, "locked", purpose="report").digest == locked.digest


async def test_clusters_hide_tasks_and_the_roles_refuse_them():
    _, _, _, _, _, dev, _, _, nat, _ = await _setup()
    vs = {p.id: PlantVerdict(plant_id=p.id) for p in dev.plants}  # pretend nothing was caught
    ms = miss_clusters(dev.plants, vs)
    ns = natural_clusters(nat)
    assert ms and ns and {c.kind for c in ms} == {"miss"} and {c.kind for c in ns} == {"natural"}
    text = json.dumps([c.shown() for c in [*ms, *ns]])
    for p in dev.plants:  # no task id, plant id or entity of any plant reaches the abstractor
        assert p.source_id not in text and p.id not in text
    assert "members" not in text and "INV-" not in text and "T-3" not in text
    with pytest.raises(RoleLeak):
        OPERATOR_ABSTRACTOR.render(task_ids=["x"], clusters=[])
    for role in (OPERATOR_ABSTRACTOR, PLANTER_PROPOSER, GUIDANCE_EDITOR):
        assert "locked_set" in role.never and "anchors" in role.never
    f = natural_features(nat[0])
    assert f[f"d:{nat[0].defect}"] == 1.0 and sum(v for k, v in f.items() if k.startswith("t:")) == 1.0


def test_random_operators_are_valid_dsl_and_reply_parsing_renames():
    rng = random.Random(0)
    specs = [random_operator(rng, f"r{i}") for i in range(40)]
    for s in specs:
        assert OperatorSpec.model_validate_json(s.model_dump_json()) == s and compile_operator(s).name == f"OP:{s.name}@1"
    assert len({s.body.kind for s in specs}) == 4
    out = OperatorsOut(operators=[EXTRA_READ.model_dump(mode="json"), {"name": "bad", "body": {"kind": "teleport"}, "label": {"defect": "D1"}}],
                       cites=[["m0"], []])
    got, errs = parse_operators(out, prefix="r1", limit=4)
    assert [s.name for s, _ in got] == ["r1_0_extra_read"] and got[0][1] == ["m0"] and len(errs) == 1


async def test_the_swarm_proposers_render_only_what_the_abstractor_may_see():
    reply = {"operators": [EXTRA_READ.model_dump(mode="json")], "cites": [["m0"]], "strategies": ["read more"],
             "lemmas": [{"applies_to": "proposer", "rule": "say every rule the checker uses"}]}
    c = FakeClient("openai", lambda m, n: reply)
    p = swarm_proposers(c)
    got, info = await p.ops(clusters=[{"cluster": "n0"}], misses=[{"cluster": "m0"}], guidance=["be generic"], library=[], k=2, round=1)
    assert got[0][0].name.endswith("extra_read") and info["strategies"] == ["read more"] and info["lemmas"][0]["applies_to"] == "proposer"
    assert section(c.calls[0], "misses") == [{"cluster": "m0"}] and "## dsl" in c.calls[0][1]["content"]
    pl = FakeClient("openai", lambda m, n: {"planters": [{"name": "leak_decision", "variant": "reason", "params": {"template": "Use {v}.", "evil": 1}}]})
    ops, _ = await swarm_proposers(c, planter_client=pl).planters(miss_rates={"x": 1.0}, natural_mix={}, k=2, round=1)
    assert ops[0].params == {"template": "Use {v}."} and ops[0].version == 101
    held = FakeClient("openai", lambda m, n: {"planters": [{"name": "over_constrain", "variant": "never"}]})  # a locked-set variant: refused
    with pytest.raises(Exception, match="never"):
        await swarm_proposers(c, planter_client=held).planters(miss_rates={}, natural_mix={}, k=1, round=1)
    assert "never" not in pl.calls[0][0]["content"].split("over_constrain")[1].split(";")[0]  # the planter doc lists dev variants only


async def test_views_are_fresh_stratified_and_carry_every_planter():
    _, _, _, _, _, dev, _, _, _, _ = await _setup()
    pool = Pool([])
    pool.add("P@1", dev.plants)
    s = Sampler(pool, frac=0.6, seed=3)
    v1 = await s(1, "decide", frozenset())
    v2 = await s(1, "decide", frozenset({v1.hash}))
    assert v1.hash != v2.hash and v1.planters == ["P@1"]
    assert {pool.plants[i].domain for i in v1.payload["plants"]} == {p.domain for p in dev.plants}  # stratified by domain


async def test_the_loop_runs_both_arms_with_the_update_rule_and_reports(tmp_path):
    ctx, _clean, _, dev_src, locked_src, dev, locked, _, nat, nat_tasks = await _setup()
    pool = Pool(dev_src)
    pool.add("P@1", dev.plants)

    async def ops(*, k, round, misses, **_):
        return [(EXTRA_READ.model_copy(update={"name": f"x{round}"}), [misses[0]["cluster"]] if misses else [])], {"lemmas": [], "strategies": []}

    async def planters(*, k, round, **_):
        return [PlantOp(name="over_constrain", variant="never"), PlantOp(name="widen_frame", variant="coll", version=101,
                                                                        params={"why": "edits here are fine"})], {}

    async def guidance(*, guidance, gate_outcomes):
        return [*guidance, "cite the clusters you generalise"], {}

    cfg = EvolveConfig(rounds=2, candidates=2, view_frac=0.7, gate_fresh=8, cost_bar=64, update=UpdateConfig(rounds=2, min_band=0.0))
    loop = Loop(tmp_path, pool, locked, locked_src, nat, ctx=ctx, report_ctx=FalsifyContext(), config=cfg, dev_sources=dev_src, builders=["deepseek"],
                natural_sources=nat_tasks)
    rows = await loop.run(Proposers(ops=ops, planters=planters, guidance=guidance, family="openai"))
    arms = {r.arm for r in rows}
    assert arms == {"loop", "control"} and [r.round for r in rows if r.arm == "loop"] == [0, 1, 2]
    r1 = next(r for r in rows if r.arm == "loop" and r.round == 1)
    (cand,) = r1.candidates
    assert cand.origin == "swarm:A" and cand.gate and cand.gate["evidence_domain"] in ("ap-test", "desk-test")
    assert cand.accepted or cand.refused_by  # decided either way, and every refusal names its rule or gate check
    r2 = next(r for r in rows if r.arm == "loop" and r.round == 2)  # the meta loop: A and A' each propose; the gate pass rates decide
    assert r2.guidance_decision is not None and {c.origin for c in r2.candidates} == {"swarm:A", "swarm:A'"}
    assert r2.guidance_decision["accepted"] or r2.guidance_decision["rules"]
    pd = {d["op"]: d for d in r1.planter_decisions}
    assert pd["over_constrain@1:never"]["refused_by"] == ["locked-variant"]  # the planter loop never plants the locked variants
    assert "widen_frame@101:coll" in pd
    ctrl = next(r for r in rows if r.arm == "control" and r.round == 1)
    assert len(ctrl.candidates) == 2 and all(c.origin.startswith("random") for c in ctrl.candidates)
    for r in rows:  # every round's locked measurement is a report-only matrix over the same locked set
        assert r.locked["n"] == len(locked.plants) and r.locked["fa_n"] == len(locked_src)
    led = (tmp_path / "loop" / "ledger.jsonl").read_text().splitlines()
    kinds = [json.loads(x)["rec"]["kind"] for x in led]
    assert kinds[0] == "calibrate" and "update" in kinds
    assert loop.arms[0].engine.ledger.records()  # the hash chain verifies
    md = rows_markdown(rows)
    assert "| loop | 1 |" in md and "| control | 1 |" in md
    ref = refusals(rows)
    assert set(ref) <= {"loop", "control"}


async def test_e1_estimates_from_the_archive_and_the_locked_matrix():
    from crucible.synth.meta.recall import measure

    ctx, _, _, _, locked_src, _, locked, _, nat, _ = await _setup()
    m = await measure(locked.plants, locked_src, cheap(), ctx, set_name=locked.name, set_kind="locked", purpose="report")
    res, cap = e1_estimates(m, nat)
    assert res and all(r["n_found"] >= 1 for r in res)
    assert {r["domain"] for r in res} <= {"ap-test", "desk-test"}
    assert all(set(c["classes"]) <= {"MUT", "TWIN", "RED", "NULL", "ATK", "ALT"} for c in cap)


def test_the_natural_reference_uses_only_independently_found_validity_defects(tmp_path):
    from crucible.synth.meta.rsi import load_real_defects, plant_key, real_defect_features

    recs = [{"id": "a", "independent": True, "quality": "P1", "d_class": "D5", "mechanism": "unstated_rule"},
            {"id": "b", "independent": True, "quality": "P2", "d_class": None, "mechanism": "too_hard"},  # learnability, no class
            {"id": "c", "independent": False, "quality": "P1", "d_class": "D9", "mechanism": "archive"},  # falsifier-found
            {"id": "d", "independent": True, "quality": "P1", "d_class": "D6", "mechanism": "stale_visible_test"}]
    p = tmp_path / "rd.json"
    p.write_text(json.dumps({"built_at": "x", "debug_lane_and_finance_ap": recs}))
    f = real_defect_features(load_real_defects(p))
    assert len(f) == 2 and f[0]["d:D5"] == 1.0 and f[0]["t:instruction"] == 1.0 and f[1]["t:required"] == 1.0
    t = fx.in_domain(fx.t1_hold())
    from crucible.synth.meta.plant import Plant, Witness

    def mk(why: str) -> Plant:
        c = t.contract.model_copy(update={"allowed": [t.contract.required[0].model_copy(update={"id": "a1", "why": why})]})
        return Plant(id=f"p-{why}", defect="D1", op=PlantOp(name="drop_required"), domain=t.domain, source_id=t.id, source_version=1, source_digest="x",
                     task=t.model_copy(update={"contract": c}), witness=Witness(kind="lenient"))

    assert plant_key(mk("one reason")) == plant_key(mk("another reason"))  # a why text plants nothing new
