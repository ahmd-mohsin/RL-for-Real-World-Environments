"""crucible.synth.meta.plant + recall: M3 acceptance for the planter and the FR/FA matrix (part2-paper-core §6, §8, §14).

Every §8 operator plants a task whose witness re-validates by execution on the fixture domains, the M2 population
catches it, the matrix prints, the certified anchors raise no false alarm, a plant without a valid witness is discarded,
and the residual estimates match a hand computation.
"""

from __future__ import annotations

import json
import math

import pytest

from crucible.synth.falsify import FalsifyContext, cheap, cheap_m3
from crucible.synth.meta.plant import OPS, Plant, Planter, PlantOp, Witness, base_planter, feature_matrix, held_out_planter, median_gamma, mmd, mmd2, plant, plant_all, validate
from crucible.synth.meta.recall import (
    LockedSetError,
    PlantSet,
    SetStore,
    capture_recapture,
    load_set,
    measure,
    report_json,
    report_markdown,
    residual,
    split_sets,
    wilson,
)
from crucible.synth.program import Program
from crucible.synth.task import Task
from tests.unit import synth_fixtures as fx

_CACHE: dict[str, object] = {}


def _population() -> Planter:
    return Planter(id="m3-test", ops=[*base_planter().ops, *held_out_planter().ops], max_per_op=1)


async def _plants() -> list[Plant]:
    """One plant per (operator variant, domain): every operator on both fixture domains, small enough to run in seconds."""
    if "plants" not in _CACHE:
        kept, seen = [], set()
        for p in (await plant_all(fx.anchors(), _population(), FalsifyContext())).kept:
            if (p.op.key, p.domain) not in seen:
                seen.add((p.op.key, p.domain))
                kept.append(p)
        _CACHE["plants"] = kept
    return _CACHE["plants"]  # type: ignore[return-value]


async def _matrix(pop: str = "now"):
    k = f"matrix-{pop}"
    if k not in _CACHE:
        fs = cheap() if pop == "now" else cheap_m3()
        _CACHE[k] = await measure(await _plants(), fx.anchors(), fs, FalsifyContext(), set_name="fixtures-dev")
    return _CACHE[k]


def _src(p: Plant) -> Task:
    return next(t for t in fx.anchors() if t.id == p.source_id)


async def test_every_operator_plants_a_witnessed_defect_and_the_population_catches_it():
    plants = await _plants()
    m = await _matrix()
    print("\n" + report_markdown(m))  # the FR/FA matrix of the M2 population on the fixture planted set
    by_op = {name: [p for p in plants if p.op.name == name] for name in OPS}
    assert all(by_op.values()), {k: len(v) for k, v in by_op.items()}  # every operator yields a plant
    ctx = FalsifyContext()
    for p in plants:  # every kept plant carries an executed witness that re-validates in a fresh context
        ok, why = await validate(p, _src(p), ctx)
        assert ok, (p.op.key, p.change, why)
        assert p.defect == OPS[p.op.name].defect and p.features[f"d:{p.defect}"] == 1.0
    hit = {o.plant_id for o in m.plants if o.hit}
    for name, ps in by_op.items():  # and the M2 population confirms a counterexample on at least one plant of each
        assert any(p.id in hit for p in ps), name
    # the certified anchors raise no false alarm in any class
    assert m.fa_population.hits == 0 and all(c.hits == 0 for c in m.fa.values()) and m.fa_population.n == 5
    # the class that should own each defect does: (class, defect) -> recall
    for cls, d in [("MUT", "D1"), ("MUT", "D2"), ("RED", "D4"), ("TAINT", "D5"), ("MUT", "D6"), ("TAINT", "D7"), ("NULL", "D8"), ("MUT", "D9"),
                   ("MUT", "D10"), ("REP", "D11")]:
        assert m.fr[cls][d].hits > 0 and m.fr[cls][d].matched > 0, (cls, d)
    # REP gates the deterministic classes on nondeterministic plants (§5.1 rule 3)
    assert all({"MUT", "TWIN", "PERT"} <= set(o.not_applicable) for o in m.plants if o.defect == "D11")


async def test_the_matrix_measures_real_gaps_of_the_m2_population():
    """Not every plant is caught: the matrix is a measurement, and these misses are what the falsifier loop must close.
    (The M3 population, ``cheap_m3``: the gaps as RSI-1/2 measured them.)"""
    m = await _matrix("m3")
    never = [o for o in m.plants if o.op.startswith("over_constrain@1:never")]
    assert never and not any(o.hit for o in never)  # an unstated never(read) is invisible to MUT (no mutant adds a read)
    human = [o for o in m.plants if o.op.startswith("leak_decision@1:humanised") and o.domain == "ap-test"]
    assert human and not any(o.hit for o in human)  # "price variance" leaks price_variance; TAINT's token match misses it
    relax = [o for o in m.plants if o.op.startswith("relax_value@1:eq_to_any")]
    assert relax and all("MUT" in o.suspect_only or "MUT" not in o.hit for o in relax)  # MUT's PerturbValue survivors are suspects (§5)


async def test_the_item_20_population_closes_the_never_and_humanised_gaps_without_alarms():
    """DESIGN §7.7 item 20: MUT's read mutants confirm the unstated never(read) as D6, TAINT's separator layer confirms
    the humanised leak as D7, and no anchor alarms in any class."""
    m = await _matrix()
    never = [o for o in m.plants if o.op.startswith("over_constrain@1:never")]
    assert never and all("MUT" in o.matched for o in never)
    human = [o for o in m.plants if o.op.startswith("leak_decision@1:humanised")]
    assert human and all("TAINT" in o.matched for o in human)
    assert m.fa_population.hits == 0
    relax = [o for o in m.plants if o.op.startswith("relax_value@1:eq_to_any")]
    assert relax and all("MUT" in o.suspect_only or "MUT" not in o.hit for o in relax)  # still suspects: the third gap is the panel's


async def test_a_plant_without_a_valid_witness_is_discarded():
    ctx = FalsifyContext()
    t1 = fx.in_domain(fx.t1_hold())
    b = await plant(t1, Planter(ops=[PlantOp(name="drop_required"), PlantOp(name="relax_value", variant="tolerance", params={"factor": 10})]), ctx)
    # every write of t1 also logs an audit row, so no run passes with one requirement dropped; t1 has no tolerance at all
    assert not [p for p in b.kept if p.op.name == "drop_required"] and {d.reason for d in b.discarded if d.op.startswith("drop_required")} == {"no witness"}
    t2 = fx.in_domain(fx.t2_approve_answer())
    b2 = await plant(t2, Planter(ops=[PlantOp(name="relax_value", variant="tolerance", params={"factor": 10})]), ctx)
    assert not b2.kept and b2.discarded  # x10 widens 0.01 to 0.1: no WrongAnswer lands inside it, so the plant has no witness
    b3 = await plant(t2, Planter(ops=[PlantOp(name="relax_value", variant="tolerance", params={"factor": 100})]), ctx)
    (good,) = b3.kept  # x100 does: 1201 is within 1.0 of 1200
    assert good.witness.kind == "lenient" and good.witness.source_op == "WrongAnswer"
    # a forged witness does not validate, so it could never have been kept
    forged = good.model_copy(update={"witness": Witness(kind="lenient", program=t2.oracle)})
    assert (await validate(forged, t2, ctx)) == (False, "the witness passes the source task (no defect)")
    empty = good.model_copy(update={"witness": Witness(kind="lenient", program=Program())})
    assert not (await validate(empty, t2, ctx))[0]
    assert (await validate(good, t2.next_version(), ctx)) == (False, "source task changed")


async def test_planted_worlds_are_data_and_behave():
    ctx = FalsifyContext()
    plants = await _plants()
    aff = next(p for p in plants if p.op.key.startswith("plant_affordance@1:solve_button") and p.domain == "desk-test")
    assert aff.task.world.factory == "crucible.synth.meta.plant:planted_world"
    again = Task.model_validate_json(aff.task.model_dump_json())  # a planted World round-trips as JSON
    w = await again.world.build()
    tools = {t.name for t in await w.tools()}
    assert aff.change["tool"] in tools and aff.change["tool"].startswith("desk.quick_") and "desk.resolve_ticket" in tools
    assert aff.witness.program is not None and [s.tool for s in aff.witness.program.steps] == [aff.change["tool"]]  # one call, no reads
    nd = next(p for p in plants if p.op.key.startswith("inject_nondet@1:pinned"))
    r1, r2 = await ctx.run(nd.task, nd.task.oracle), await ctx.run(nd.task, nd.task.oracle)
    assert ctx.grade(r1, nd.task.contract).passed != ctx.grade(r2, nd.task.contract).passed  # the stamp alternates by episode
    w2 = await nd.task.world.build()
    await w2.reset(0)
    ref = await w2.snapshot()
    s0 = (await w2.canon()).fingerprint()
    await w2.call(nd.change["tool"], dict(next(c.args for c in r1.calls if c.tool == nd.change["tool"])))
    assert (await w2.canon()).fingerprint() != s0
    await w2.restore(ref)
    assert (await w2.canon()).fingerprint() == s0  # snapshots carry the overlay


def test_residual_estimate_matches_a_hand_computation():
    r = residual("D2", n_found=4, hits=8, n=10)
    se = 4 / 0.8**2 * math.sqrt(0.8 * 0.2 / 10)  # 0.790569
    assert r.estimate == pytest.approx(1.0) and r.lo == 0.0 and r.hi == pytest.approx(1 + 1.959964 * se)
    b = residual("D2", n_found=4, hits=8, n=10, method="bootstrap", seed=1)
    assert b.estimate == pytest.approx(1.0) and b.lo <= 1.0 <= b.hi and b.lo >= 0
    assert residual("D3", n_found=2, hits=0, n=6).unbounded  # an unmeasured class is not a clean one
    assert residual("D3", n_found=0, hits=5, n=5).estimate == 0.0
    cr = capture_recapture("D2", [f"x{i}" for i in range(10)], [f"x{i}" for i in range(6, 14)])
    assert (cr.n_a, cr.n_b, cr.overlap) == (10, 8, 4)
    assert cr.lincoln_petersen == pytest.approx(20.0) and cr.chapman == pytest.approx(11 * 9 / 5 - 1)
    assert cr.chapman_var == pytest.approx(11 * 9 * 6 * 4 / (25 * 6)) and cr.lo >= 14  # never below the 14 distinct defects found
    assert capture_recapture("D2", ["a"], ["b"]).lincoln_petersen is None
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)


async def test_reports_are_json_and_markdown():
    m = await _matrix()
    res = [residual(d, 1, m.population[d].hits, m.population[d].n) for d in m.defects]
    doc = json.loads(report_json(m, res, [capture_recapture("D2", ["a", "b"], ["b"])]))
    assert doc["schema_version"] == "crucible.synth.recall/v1" and doc["fr"]["MUT"]["D2"]["hits"] == m.fr["MUT"]["D2"].hits
    assert all(r["estimate"] is None or isinstance(r["estimate"], float) for r in doc["residuals"])
    md = report_markdown(m, res)
    assert md.count("\n| ") >= len(m.classes) + 2 and "| **any** |" in md and "residual" in md


async def test_the_locked_set_is_read_for_reporting_only(tmp_path):
    plants = await _plants()
    held = {op.key for op in held_out_planter().ops}
    dev, locked = split_sets(plants, locked_domains=["desk-test"], locked_ops=held, name="fx")
    assert locked.plants and dev.plants
    assert not {p.id for p in dev.plants} & {p.id for p in locked.plants}
    assert all(p.domain == "desk-test" and p.op.key in held for p in locked.plants) and not any(p.op.key in held or p.domain == "desk-test" for p in dev.plants)
    store = SetStore(tmp_path / "dev", tmp_path / "sealed" / "synth-locked")
    store.write(dev)
    pl = store.write(locked)
    with pytest.raises(FileExistsError):
        store.write(dev)  # write-once
    assert store.read("fx-dev").digest == dev.digest
    with pytest.raises(LockedSetError):
        store.read("fx-locked", "locked")
    with pytest.raises(LockedSetError):
        load_set(pl)  # by path too: the directory is registered as locked
    copy = tmp_path / "elsewhere.json"
    copy.write_text(pl.read_text())
    with pytest.raises(LockedSetError):
        load_set(copy)  # and a copy still says what it is
    assert store.read("fx-locked", "locked", purpose="report").digest == locked.digest
    with pytest.raises(LockedSetError):
        await measure(locked.plants, [], cheap(), FalsifyContext(), set_kind="locked")
    m = await measure(locked.plants[:2], [], cheap(), FalsifyContext(), set_kind="locked", purpose="report")
    assert m.set_kind == "locked" and len(m.plants) == 2
    with pytest.raises(ValueError):
        SetStore(tmp_path / "sealed" / "synth-locked" / "dev", tmp_path / "sealed" / "synth-locked")
    tampered = PlantSet(name="t", plants=dev.plants[:1], digest=dev.digest)
    (tmp_path / "t.json").write_text(tampered.model_dump_json())
    with pytest.raises(ValueError):
        load_set(tmp_path / "t.json")


async def test_natural_likeness_mmd():
    plants = await _plants()
    ref = feature_matrix(p.features for p in plants if p.domain == "ap-test")
    assert mmd(ref, ref) == 0.0 and mmd2(ref, ref, median_gamma(ref)) == 0.0
    same = feature_matrix(p.features for p in plants if p.domain == "desk-test")
    shifted = feature_matrix(p.features for p in plants if p.defect in ("D10", "D11"))
    assert mmd(same, ref) < mmd(shifted, ref)
    with pytest.raises(ValueError):
        mmd2([], ref, 1.0)
