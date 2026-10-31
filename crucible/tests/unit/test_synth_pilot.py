"""The fixes the 10 x 10 pilot's real runs forced (DESIGN §7.7 item 26; runs/pilot-10x10.md). Every model reply is
scripted; no Tinker call."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.synth.families import FAMILY_SLICE_CHARS, FamilyConfig, InstanceDomain, build_family, family_slice, instance_config, make_instance
from crucible.synth.roles import cjson
from tests.unit.synth_world_fixtures import family_clients


@pytest.fixture(scope="module")
def instance(tmp_path_factory):
    import asyncio

    root = tmp_path_factory.mktemp("worlds")
    st = asyncio.run(build_family("IT service desk ticket handling", root / "service_desk", family_clients(), cfg=FamilyConfig(k_min=40)))
    assert st.status == "frozen", st.reasons
    fdir = root / "service_desk"
    rep = make_instance(fdir, instance_config(fdir.name, 0))
    return Path(rep.dir)


def _usable(inst: Path, thin: list[str]) -> None:
    (inst / "gates.json").write_text(json.dumps({"usable": True, "gates": {"density": {"details": {"thin_tools": thin}}}}))


async def test_the_family_slice_shows_the_rules_and_every_table_within_its_limit(instance, tmp_path):
    d = InstanceDomain(instance, tmp_path / "work")
    assert d.env.slice is not None and d.env.slice_chars == FAMILY_SLICE_CHARS
    w = await d.env.world(0).build()
    try:
        await w.reset(0)
        canon = await w.canon()
        tables = {f"{a}.{c}" for a, cs in canon.apps.items() for c in cs}
        s = await family_slice(8000)(w)
        assert set(s) == tables  # every table is shown, none cut off by the table order
        docs = next(k for k in s if k.endswith(".documents"))
        assert any(isinstance(v, dict) and "..." in v for v in s.values())  # rows beyond a table's share are counted, not dropped silently
        assert len(cjson(s)) <= 8000  # within the limit, so compact never cuts it in key order
        full = await family_slice(10**6)(w)
        assert full[docs] == dict(canon.apps[docs.split(".")[0]]["documents"])  # the documents whole when they fit
    finally:
        await w.close()


def test_instance_cells_fill_the_quota_after_the_thin_filter(instance, tmp_path):
    spec = json.loads((instance / "domain.json").read_text())
    skills = spec["skills"]
    keep = next(s for s in skills if s.get("tools"))
    thin = sorted({t for s in skills for t in s.get("tools") or [] if t not in set(keep["tools"])})
    _usable(instance, thin)
    d = InstanceDomain(instance, tmp_path / "work", n_cells=10)
    only_thin = {s["id"] for s in skills if s.get("tools") and set(s["tools"]) <= set(thin)}
    assert len(d.cells) == 10 and len(set(d.cells)) == 10  # pilot-10x10: a draw of 2 x n left 6 of 10 cells
    assert not any(c.startswith(f"skill {sid}:") for c in d.cells for sid in only_thin)


async def test_refs_inside_a_literal_list_or_object_are_resolved_and_keys_are_schema():
    from crucible.synth.program import Fn, run
    from crucible.synth.roles import parse_program, program_doc
    from tests.unit import synth_fixtures as fx

    doc = {"steps": [{"id": "s1", "tool": "ap.find_invoice", "args": {"number": "INV-1003"}}],
           "answer": {"$fn": "concat", "args": [["id", {"$ref": "s1.invoices.0.id"}], [{"vendor": {"$ref": "s1.invoices.0.vendor"}, "n": 1}]]}}
    p = parse_program(doc)  # pilot-10x10: refused before ("a $ref or $fn inside a literal list or object is not resolved")
    a = p.answer
    assert isinstance(a, Fn) and [x.name for x in a.args] == ["list", "list"] and a.args[1].args[0].name == "obj"
    assert parse_program(program_doc(p)) == p  # round trip
    r = await run(fx.ap_world(), p, "Look up invoice INV-1003.", seed=0)
    assert r.error is None and r.answer == ["id", 103, {"vendor": "Cobalt Office", "n": 1}]
    keys = [e for e in r.taint if e.value in ("vendor", "n") and "fn:obj" in e.loc]
    assert keys and all(s.kind == "structural" for e in keys for s in e.sources)  # an object's keys are schema, not task information
    plain = parse_program({"steps": [{"id": "s1", "tool": "ap.find_invoice", "args": {"number": "INV-1003", "x": [1, {"a": 2}]}}]})
    assert plain.steps[0].args["x"].op == "lit"  # a container without refs stays a literal: earlier programs are unchanged


def test_the_wire_maps_the_policy_ladder_to_models_and_the_batch_certifies_with_l0(tmp_path):
    from types import SimpleNamespace

    from crucible.factory.synth_wire import GPT_OSS, INKLING_SMALL, LADDER_MODELS, NEMOTRON, SYNTH_ROLES, SynthWire, ladder_configs, ladder_models
    from crucible.synth.solvers import ladder_rungs

    assert {n for r in ladder_rungs() for n in r.configs} <= set(LADDER_MODELS)  # every configuration the policy names is wired
    assert ladder_models(0) == (GPT_OSS, NEMOTRON, INKLING_SMALL)  # R-ALT-whole's L0 certifying set
    w = SynthWire(tmp_path, cap_usd=1.0, make=lambda m, r: SimpleNamespace(), table=dict(SYNTH_ROLES) | {"alt_solver": ladder_models()})
    cfg = ladder_configs(w)
    assert set(cfg) == set(LADDER_MODELS) and cfg["nemotron-3-ultra:v2"].harness == "v2" and cfg["inkling-small:v3"].harness == "v3"
    assert {c.family for c in cfg.values()}.isdisjoint({"deepseek", "glm", "qwen"})  # no builder and no student family solves
    w0 = SynthWire(tmp_path / "b", cap_usd=1.0, make=lambda m, r: SimpleNamespace(), table=dict(SYNTH_ROLES) | {"alt_solver": ladder_models(0)})
    assert len(w0.clients("alt_solver")) == 3 and len({c.family for c in w0.clients("alt_solver")}) == 3  # three families certify


async def test_candidates_a_cap_stopped_are_picked_up_on_resume(tmp_path):
    """pilot-10x10: the oracle stage cap bound; its candidates were marked budget/done and a resumed batch skipped them."""
    from crucible.synth.batch import Arm, load_cands, run_batch
    from tests.unit.test_synth_batch import cfg, domain, roles, setup

    await run_batch([setup(Arm.baseline("a0"), roles())], [domain()], cfg(), tmp_path)
    cands = load_cands(tmp_path / "a0")
    done = {c.id: c.status for c in cands}
    a, b = [c for c in cands if c.task is not None][:2]
    a.status, a.stage, a.reasons = "budget", "done", ["budget: stage oracle cap"]  # stopped while queued for refine
    b.status, b.stage, b.reasons, b.task, b.record = "budget", "done", ["engine: TinkerCapReached: stage oracle"], None, None  # refused in its build
    for c in (a, b):
        (tmp_path / "a0" / "cands" / (c.id.replace("/", "_") + ".json")).write_text(c.model_dump_json())
    rl2 = roles()
    await run_batch([setup(Arm.baseline("a0"), rl2)], [domain()], cfg(), tmp_path)
    after = {c.id: c for c in load_cands(tmp_path / "a0")}
    assert after[a.id].status == done[a.id] and after[b.id].status == done[b.id]  # refined / rebuilt, the uninterrupted outcome
    assert after[b.id].task is not None and not rl2.proposer.calls


async def test_an_uncalibrated_panel_gets_more_anchors_then_abstains_without_deadlock(tmp_path):
    """R-panel: uncalibrated -> add_calibration_anchors while more built tasks can come; then the panel abstains (λ > 1)."""
    from crucible.synth.batch import Arm, run_batch
    from crucible.synth.pipeline import RunConfig, RunRoles
    from tests.unit.synth_fakes import FakeClient, const
    from tests.unit.test_synth_batch import cfg, domain, roles, setup

    yes = const({"valid": True, "confidence": 0.95, "why": "looks fine"})  # calls every no-op valid: never calibrated
    rl = roles()
    rl = RunRoles(**{**rl.__dict__, "panel": [("p-n", FakeClient("nemotron", yes)), ("p-o", FakeClient("openai", yes))]})
    await run_batch([setup(Arm.baseline("a0"), rl)], [domain()], cfg(run_cfg=RunConfig(calibrate=True, calib_tasks=1), calib_max_tasks=4), tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "a0" / "calibration_remedies.jsonl").read_text().splitlines()]
    assert rows and rows[0]["action"] == "add_calibration_anchors" and rows[0]["rule"] == "R-panel"
    assert rows[-1]["anchor_tasks"] >= rows[0]["anchor_tasks"]
    cal = json.loads(next((tmp_path / "a0").glob("*/calibration.json")).read_text())
    assert cal["calibrated"] is False and cal["lam"] > 1  # it abstains: suspects stay open, nothing is decided by it


async def test_a_resumed_domain_built_before_its_calibration_is_anchored_from_its_tasks(tmp_path):
    """pilot-10x10: built tasks resumed straight into refine left calibration without anchors; refine waited for it and the
    proposer for the backlog (a deadlock). The anchors now come from the stored tasks."""
    import asyncio

    from crucible.synth.batch import Arm, load_cands, run_batch
    from crucible.synth.pipeline import RunConfig
    from tests.unit.test_synth_batch import cfg, domain, roles, setup

    rc = RunConfig(calibrate=True, calib_tasks=2)
    await run_batch([setup(Arm.baseline("a0"), roles())], [domain()], cfg(run_cfg=rc), tmp_path)
    cands = load_cands(tmp_path / "a0")
    done = {c.id: c.status for c in cands}
    for p in (tmp_path / "a0").glob("*/calibration.json"):
        p.unlink()  # the crash came before the domain's calibration
    for c in [c for c in cands if c.task is not None]:
        c.stage, c.status = "built", "pending"
        (tmp_path / "a0" / "cands" / (c.id.replace("/", "_") + ".json")).write_text(c.model_dump_json())
    await asyncio.wait_for(run_batch([setup(Arm.baseline("a0"), roles())], [domain()], cfg(run_cfg=rc), tmp_path), timeout=120)
    after = {c.id: c.status for c in load_cands(tmp_path / "a0")}
    assert after == done
    cal = json.loads(next((tmp_path / "a0").glob("*/calibration.json")).read_text())
    assert cal["anchor_tasks"] >= 1  # calibrated on the stored tasks' executed oracles


async def test_a_solver_that_never_calls_ends_its_attempt_and_notes_only_turns_are_answered():
    from crucible.synth.falsify.agents import AgentView, llm_agent
    from crucible.synth.roles import variant
    from tests.unit import synth_fixtures as fx
    from tests.unit.synth_fakes import FakeClient, const

    w = fx.ap_world()
    await w.reset(0)
    view = AgentView(instruction="Hold invoice INV-1003.", tools=list(await w.tools()))
    notes = FakeClient("nemotron", const({"notes": "1. find the invoice 2. hold it"}))
    r = await llm_agent(notes, variant("alt_solver", "v3"), max_malformed=4)(view, w)
    assert r is None and len(notes.calls) == 4  # four notes-only turns in a row end the attempt (pilot-10x10)
    assert "Your notes are saved" in str(notes.calls[-1])
    notes2 = FakeClient("nemotron", const({"notes": "plan"}))
    await llm_agent(notes2, variant("alt_solver", "v3"))(view, w)
    assert len(notes2.calls) > 4  # the default (None, the student probe's) never ends an attempt early


def test_stored_rows_hold_a_task_rejected_only_on_open_suspects(tmp_path):
    from crucible.synth.certificate import certify
    from crucible.synth.recert import Source, collect, stored_rows
    from tests.unit import synth_fixtures as fx

    run = tmp_path / "arm" / "dom"
    rows = []
    for n, (t, why) in enumerate([(fx.t1_hold(), ["ALT:suspects=2", "MUT:upper=0.06>0.05"]), (fx.t2_approve_answer(), ["ALT:no_pass", "MUT:suspects=1"])], 1):
        cert = certify(t, []).model_dump(mode="json") | {"status": "rejected", "reasons": why}
        p = run / "tasks" / f"t{n:02d}" / "cegis.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"task_id": t.id, "domain": t.domain, "status": "rejected", "reasons": why, "rounds": [], "certificate": cert,
                                 "task": t.model_dump(mode="json")}))
        rows.append({"n": n, "id": t.id, "seed": 0, "status": "rejected"})
    (run / "report.json").write_text(json.dumps({"domain": "synth-test", "run_id": "r", "started": 0, "tasks": rows, "families": {}}))
    got = {r.task_id: r.status for r in stored_rows(collect([Source(name="p", kind="pilot", dir=str(tmp_path / "arm"), archive_root=str(tmp_path),
                                                                    statuses=["rejected"])]))}
    assert got == {fx.t1_hold().id: "on_hold", fx.t2_approve_answer().id: "rejected"}  # a failed rule (no ALT pass) stays rejected


async def test_rule_derived_dates_are_computed_and_sourced():
    from crucible.synth.program import LIBRARY, LibraryError, run
    from crucible.synth.roles import parse_program
    from tests.unit import synth_fixtures as fx

    assert LIBRARY["date_add"]("2026-02-10", 5) == "2026-02-15" and LIBRARY["date_add"]("2026-01-01T00:00:04Z", -1) == "2025-12-31"
    assert LIBRARY["days_between"]("2025-10-13", "2025-12-23") == 71
    with pytest.raises(LibraryError):
        LIBRARY["date_add"]("soon", 5)
    ins = "Invoice INV-1003 was promised for 2026-02-10. Report the follow-up date, 5 days after the promise."
    p = parse_program({"steps": [{"id": "s1", "tool": "ap.find_invoice", "args": {"number": "INV-1003"}}],
                       "answer": {"$fn": "date_add", "args": ["2026-02-10", 5]}})
    r = await run(fx.ap_world(), p, ins, seed=0)
    assert r.answer == "2026-02-15"
    src = {e.value: {s.kind for s in e.sources} for e in r.taint}
    assert "instruction" in src["2026-02-10"] and "instruction" in src[5]  # every input sourced: no D5 literal date (pilot-10x10)


async def test_a_calibrated_panel_decides_an_on_hold_certificate_in_the_loop(tmp_path):
    """R-panel's panel_decide (pilot-10x10: the loop hard-coded panel_calibrated False, so on-hold tasks waited forever)."""
    from tests.unit import synth_fixtures as fx
    from tests.unit.test_synth_taskloop_pool import _loop

    status = {"ap/held-cal": "on_hold", "ap/held-uncal": "on_hold"}
    readj = []

    async def validity(t):
        return status[t.id], []

    async def readjudicate(t):  # the calibrated panel decides the open suspects: here it clears them
        readj.append(t.id)
        status[t.id] = "certified"

    tasks = [fx._with(fx.t1_hold(), id=k) for k in status]
    loop = _loop(tmp_path, tasks, validity=validity)
    loop.panel_calibrated, loop.readjudicate = (lambda t: t.id.endswith("-cal")), readjudicate
    decs = {d.task_id: d for d in await loop.run_typed_round(1, concurrency=2)}
    assert readj == ["ap/held-cal"] and decs["ap/held-cal"].oracle["certificate"] == "certified"
    assert any("panel_decide" in w for w in decs["ap/held-cal"].why) and not any("P1: the certificate is on_hold" in w for w in decs["ap/held-cal"].why)
    assert decs["ap/held-uncal"].move == "HOLD" and loop.lineages["ap/held-uncal"].status == "held"  # uncalibrated: human queue
    assert any(r["action"] == "human_queue" for r in decs["ap/held-uncal"].remedies)


def test_the_loop_calibrates_per_world_family(tmp_path):
    import importlib.util

    from crucible.synth.task import WorldRef
    from tests.unit import synth_fixtures as fx

    spec = importlib.util.spec_from_file_location("stl", Path(__file__).resolve().parents[2] / "scripts" / "synth_taskloop.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    inst = tmp_path / "insurance_claims_intake" / "instances" / "i00-standard-s1"
    inst.mkdir(parents=True)
    (inst / "instance.json").write_text("{}")
    t = fx.t1_hold().model_copy(update={"world": WorldRef(factory="crucible.synth.worldgen:open_world", params={"dir": str(inst), "work": "w"})})
    assert m.family_of(t) == "insurance_claims_intake" and m.family_of(fx.t1_hold()) == fx.t1_hold().domain


def test_the_batch_alt_harness_is_its_own_setting():
    from crucible.synth.batch import Arm
    from crucible.synth.pipeline import RunConfig

    a = Arm.improved(alt_variant="v3", alt_progress=True, alt_window=16)
    assert a.variant == "v2" and a.alt_variant == "v3"  # the builders keep v2, the solvers get the v3 harness
    assert RunConfig().alt_variant is None  # unset: the solver follows ``variant`` as before


def test_stored_rows_carry_the_batch_certificate_without_recertifying(tmp_path):
    from crucible.synth.certificate import certify
    from crucible.synth.recert import Source, collect, stored_rows
    from tests.unit import synth_fixtures as fx

    run = tmp_path / "arm" / "dom"
    t = fx.t1_hold()
    cert = certify(t, []).model_dump(mode="json")
    cert["status"], cert["reasons"] = "rejected", ["ALT:no_pass"]
    p = run / "tasks" / "t01" / "cegis.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"task_id": t.id, "domain": t.domain, "status": "rejected", "reasons": ["ALT:no_pass"], "rounds": [], "certificate": cert,
                             "task": t.model_dump(mode="json")}))
    (run / "report.json").write_text(json.dumps({"domain": t.domain, "run_id": "r", "started": 0, "tasks": [{"n": 1, "id": t.id, "seed": 0, "status": "rejected"}],
                                                 "families": {"proposer": [{"family": "deepseek"}]}}))
    items = collect([Source(name="p/a", kind="pilot", dir=str(tmp_path / "arm"), archive_root=str(tmp_path), statuses=["certified", "provisional", "rejected"])])
    [row] = stored_rows(items)
    assert row.status == "rejected" and row.reasons == ["ALT:no_pass"] and row.certificate == str(p) and row.load_task().digest() == t.digest()
    assert row.policy_version == cert.get("policy_version", "") and row.outcome == "" and row.status_h0 == ""
