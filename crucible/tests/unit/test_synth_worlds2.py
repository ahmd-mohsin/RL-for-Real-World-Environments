"""worlds-2: the fixes the second World-families run forced (DESIGN §7.7 item 16). The listing rule at the design and
code stages, density measured and fed back at the data stage on the shown mutants, the data generator's restarts, and
the realism panel's negative anchors from gate-rejected instances. Every model reply is scripted."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from crucible.synth.families import (
    FamilyConfig,
    build_family,
    instance_config,
    make_instance,
    realism_design_problems,
    unlisted_targets,
)
from crucible.synth.roles import ToolDesignOut
from crucible.synth.worldgates import Anchor, RealismPanel, calibrate_panel, density_gate, listing_problems, rejected_anchors, synthetic_seed
from tests.unit.synth_fakes import FakeClient, const, seq, user_text
from tests.unit.synth_world_fixtures import CODE, DATAGEN, DESIGN, REALISTIC, SCHEMA, SPEC, family_clients


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import asyncio

    root = tmp_path_factory.mktemp("worlds2")
    st = asyncio.run(build_family("IT service desk ticket handling", root / "service_desk", family_clients(), cfg=FamilyConfig(k_min=40)))
    assert st.status == "frozen", st.reasons
    return root / "service_desk"


def _no_ticket_list(d: dict) -> dict:
    d = copy.deepcopy(d)
    t = next(x for x in d["tools"] if x["name"] == "list_tickets")
    t["name"] = "list_queue"  # pages, but is named for no collection
    return d


def test_the_listing_rule_at_the_design_stage():
    # worlds-1: insurance had no list tool for claims (29-38% findable), returns none for refunds
    assert unlisted_targets(DESIGN) == [] and unlisted_targets(DESIGN, SCHEMA) == []
    bad = _no_ticket_list(DESIGN)
    assert unlisted_targets(bad) == ["tickets"]
    probs = " ".join(realism_design_problems(ToolDesignOut.model_validate(bad)))
    assert "no paged list or search tool: ['tickets']" in probs
    # a head-noun key (line_id) naming several written collections counts for none of them before a schema exists
    d = copy.deepcopy(DESIGN)
    d["tools"].append({"name": "split_line", "description": "x", "read_only": False, "writes": ["order_lines", "invoice_lines"],
                       "params": [{"name": "line_id", "type": "string"}], "errors": ["NOT_FOUND"]})
    assert unlisted_targets(d) == []


async def test_the_listing_rule_executed_on_a_synthetic_state(tmp_path):
    from crucible.synth.domain import DomainSpec

    spec = DomainSpec.model_validate(SPEC)
    seed = synthetic_seed(DESIGN, SCHEMA, spec)
    assert len(seed["tables"]["tickets"]) == 6 and [r["priority"] for r in seed["tables"]["tickets"][:4]] == ["low", "medium", "high", "critical"]
    assert {r["role"] for r in seed["tables"]["staff"]} == {"agent", "lead", "manager"}
    app = tmp_path / "app"
    app.mkdir()
    for name, obj in (("design.json", DESIGN), ("schema.json", SCHEMA), ("kit.json", {"kinds": ["docs", "policy", "actors"]})):
        (app / name).write_text(json.dumps(obj))
    (app / "app.py").write_text(CODE)
    probs, det = await listing_problems(app, tmp_path / "w", DESIGN, SCHEMA, spec)
    assert probs == [] and det["cover"]["tickets"] == 1.0
    # list_tickets returning staff rows, not tickets: the tool exists by name but lists another collection
    (app / "app.py").write_text(CODE.replace('rows = [dict(r) for k, r in sorted(ctx.tables["tickets"].items())', 'rows = [dict(r) for k, r in sorted(ctx.tables["staff"].items())'))
    probs, det = await listing_problems(app, tmp_path / "w", DESIGN, SCHEMA, spec)
    assert probs and "['tickets']" in probs[0] and det["cover"]["tickets"] == 0.0


async def test_the_coder_gets_the_listing_problem_back(tmp_path):
    bad = CODE.replace('rows = [dict(r) for k, r in sorted(ctx.tables["tickets"].items())', 'rows = [dict(r) for k, r in sorted(ctx.tables["staff"].items())')
    coder = FakeClient("kimi", seq({"code": bad}, {"code": CODE}))
    st = await build_family("desk", tmp_path / "f", family_clients(app_coder=coder), cfg=FamilyConfig(density=False))
    assert st.status == "frozen" and len(coder.calls) == 2
    assert "no paged list or search tool lists the records of ['tickets']" in user_text(coder.calls[1])


async def test_density_is_measured_on_the_shown_mutants_with_the_held_out_share(built):
    rep = make_instance(built, instance_config(built.name, 1))
    g, _ = await density_gate(Path(rep.dir), diagnose=True)
    d = g.details
    assert d["split"] == "shown" and d["holdout"] == 0.2
    per = {p["tool"]: p for p in d["per_tool"]}
    close = per["close_ticket"]
    assert close["n_heldout"] > 0 and d["heldout_by_collection"]["tickets"] > 0
    g0, _ = await density_gate(Path(rep.dir), holdout=0)
    whole = {p["tool"]: p for p in g0.details["per_tool"]}["close_ticket"]["n_distinct"]
    assert close["n_distinct"] < whole <= close["n_distinct"] + close["n_heldout"] + 2  # the bar is on the shown part
    reach = close["reach"]
    assert reach["table"] == "tickets" and 0 < reach["ok"] < reach["rows"] and any("INVALID_STATE" in k for k in reach["refusals"])


async def test_a_thin_generator_is_sent_back_with_its_density(tmp_path):
    # worlds-1: returns' Customer and Refund collections reached n = 12-39, found only at the gate after freezing
    thin = copy.deepcopy(DATAGEN)
    thin["tables"][1]["rows"] = 12  # too few sibling tickets for any write tool to reach 59 distinct shown mutants
    thin["tables"][1]["fields"]["status"]["weights"] = [1, 0.1, 9, 0.01]  # and most of them closed
    writer = FakeClient("kimi", seq(thin, DATAGEN))
    st = await build_family("desk", tmp_path / "f", family_clients(data_spec_writer=writer), cfg=FamilyConfig(k_min=12))
    assert st.status == "frozen", st.reasons
    assert len(writer.calls) == 2
    fb = user_text(writer.calls[1])
    assert "DENSITY" in fb and "tickets" in fb and "succeeded on" in fb and "INVALID_STATE" in fb
    rows = [json.loads(x) for x in (tmp_path / "f" / "app" / "datagen_density.jsonl").read_text().splitlines()]
    assert rows[0]["short"] and rows[0]["best_by_collection"]["tickets"] < rows[-1]["best_by_collection"]["tickets"]
    assert json.loads((tmp_path / "f" / "app" / "datagen.json").read_text())["tables"][1]["rows"] == 60  # the dense generator is kept


async def test_a_redesign_moves_the_old_generator_problems_aside(tmp_path):
    cl = family_clients()
    st = await build_family("desk", tmp_path / "f", cl, cfg=FamilyConfig(density=False))
    app = tmp_path / "f" / "app"
    (app / "datagen_attempts.jsonl").write_text(json.dumps({"attempt": 0, "problem": "table purchase_orders: an old design's table"}) + "\n")
    writer = FakeClient("kimi", const(DATAGEN))
    st = await build_family("desk", tmp_path / "f", family_clients(data_spec_writer=writer), cfg=FamilyConfig(density=False), restart_from="design")
    assert st.status == "frozen" and not (app / "datagen_attempts.jsonl").exists()
    assert list(app.glob("datagen_attempts.prev-*.jsonl"))
    assert "purchase_orders" not in user_text(writer.calls[0])


async def test_gate_rejected_instances_are_negative_anchors(built, tmp_path):
    import shutil

    root = tmp_path / "worlds"
    shutil.copytree(built, root / built.name, ignore=shutil.ignore_patterns("work"))
    insts = []
    for i, gate in ((0, "obscurity"), (1, "density")):
        rep = make_instance(root / built.name, instance_config(built.name, i))
        (Path(rep.dir) / "gates.json").write_text(json.dumps({"gates": {gate: {"gate": gate, "passed": False}}}))
        insts.append(rep)
    anchors = await rejected_anchors(root)
    assert [a.kind for a in anchors] == ["gate:obscurity"] and anchors[0].truth is False  # a density failure is no realism verdict
    p = RealismPanel(members=[("a", FakeClient("nemotron", const(REALISTIC))), ("b", FakeClient("gpt-oss", const(REALISTIC)))])
    both = [Anchor(kind="reviewed", truth=True, summary=anchors[0].summary), *anchors]
    cal = await calibrate_panel(p, both, min_decided=1, require_both=False, refuse_uncalibrated=False)  # measured under the M4 rule
    assert cal["by_kind"]["gate:obscurity"] == {"n": 1, "decided": 1, "correct": 0}  # a panel that passes everything is measured wrong
    assert cal["by_kind"]["reviewed"]["correct"] == 1
    cal = await calibrate_panel(p, both)  # R-panel: too few anchors, and the one negative is wrong: it decides nothing
    assert not cal["calibrated"] and p.lam > 1 and cal["by_kind"]["gate:obscurity"]["decided"] == 0


def test_failing_gates_resolve_to_the_policy_remedy():
    # DESIGN item 24, R-density: reseed denser, else rebuild the stage, else not used; the remedy is recorded
    from crucible.synth.worldgates import GateResult, InstanceGates, remedies_for

    g = InstanceGates(family="f", instance="i00", dir="x", gates={"density": GateResult(gate="density", passed=False),
                                                                   "obscurity": GateResult(gate="obscurity", passed=False),
                                                                   "build": GateResult(gate="build", passed=True)})
    r = {x["gate"]: x for x in remedies_for(g)}
    assert set(r) == {"density", "obscurity"}
    assert r["density"]["action"] == "reseed_denser" and r["density"]["rule"] == "R-density" and r["density"]["policy_version"].startswith("policy-v")
    assert r["obscurity"]["action"] == "not_used"  # an instance gate run does not rebuild a frozen family unless allowed
    r = {x["gate"]: x for x in remedies_for(g, reseed_tries=2, restartable=True)}
    assert r["density"]["action"] == "not_used" and r["obscurity"]["action"] == "rebuild_stage" and r["obscurity"]["stage"] == "design"
    assert {x["action"] for x in remedies_for(g, rebuild_tries=2, restartable=True, can_reseed=False)} == {"not_used"}


async def test_an_uncalibrated_panel_defers_to_the_executed_realism_gates(built, tmp_path):
    from crucible.synth.worldgates import gate_instance

    rep = make_instance(built, instance_config(built.name, 2))
    p = RealismPanel(members=[("a", FakeClient("nemotron", const(REALISTIC))), ("b", FakeClient("gpt-oss", const(REALISTIC)))], lam=1.01)
    g = await gate_instance(built, Path(rep.dir), panel=p, only=["integrity", "surface", "obscurity", "panel"])
    d = g.gates["panel"].details
    assert g.gates["panel"].passed and d["remedy"] == "executed_check" and d["decided_by"] == ["integrity", "obscurity", "surface"]
    assert not p.members[0][1].calls  # an uncalibrated panel is not asked
    g = await gate_instance(built, Path(rep.dir), panel=p, only=["panel"])
    assert g.gates["panel"].passed  # the earlier executed results are reused
    (Path(rep.dir) / "gates.json").unlink()
    g = await gate_instance(built, Path(rep.dir), panel=p, only=["panel"])
    assert not g.gates["panel"].passed and g.gates["panel"].details["remedy"] == "human_queue"  # no executed evidence: held for a person


async def test_a_reseeded_instance_regenerates_from_its_own_config(built):
    from crucible.synth.worldgates import determinism_gate

    cfg = instance_config(built.name, 3)
    a = make_instance(built, cfg)
    b = make_instance(built, cfg.model_copy(update={"k_min": 90}))
    assert b.rows["tickets"] > a.rows["tickets"] and b.config.k_min == 90
    r = await determinism_gate(built, Path(b.dir))
    assert r.passed, r.reasons


async def test_the_executed_listing_check_asks_only_what_the_design_rule_asked(tmp_path):
    # worlds-2: a collection keyed by an argument of the tool that creates it (a shipment's tracking number) is a target
    # only once the schema exists; the design rule never demanded its lister, so the coder must not be refused for it
    from crucible.synth.domain import DomainSpec

    d = copy.deepcopy(DESIGN)
    d["tools"].append({"name": "log_shipment", "description": "Log a parcel.", "read_only": False, "writes": ["parcels", "tickets"],
                       "params": [{"name": "ticket_id", "type": "string"}, {"name": "tracking_no", "type": "string"}], "errors": ["NOT_FOUND"]})
    s = copy.deepcopy(SCHEMA)
    s["tables"].append({"name": "parcels", "key": "tracking_no", "fields": {"tracking_no": "string", "ticket_id": "string"}})
    from crucible.synth.families import design_targets, targets

    assert "parcels" in targets(d, s) and "parcels" not in design_targets(d)
    app = tmp_path / "app"
    app.mkdir()
    for name, obj in (("design.json", d), ("schema.json", s)):
        (app / name).write_text(json.dumps(obj))
    (app / "app.py").write_text(CODE.replace("    @app.tool(writes=[\"tickets\", \"audit\"])\n    def close_ticket", "    @app.tool(writes=[\"parcels\", \"tickets\"])\n    def log_shipment(ctx, ticket_id: str, tracking_no: str) -> dict:\n        \"\"\"Log.\"\"\"\n        return {}\n\n    @app.tool(writes=[\"tickets\", \"audit\"])\n    def close_ticket"))
    probs, det = await listing_problems(app, tmp_path / "w", d, s, DomainSpec.model_validate(SPEC))
    assert probs == [] and det["targets"] == ["tickets"]


def test_a_listed_item_with_only_its_key_and_a_timestamp_counts():
    # worlds-2: a policy-exception list returned key, enums and created_at "2025-12-15T09:00:00Z"; its colons read as
    # another table's synthetic string, so every row was refused and three coder attempts were spent
    from crucible.synth.worldgates import _item_rows

    rows = [{"pe_id": "PE-1", "notes": "pe:notes:0"}]
    assert _item_rows([{"pe_id": "PE-1", "status": "open", "created_at": "2025-12-15T09:00:00Z"}], rows, "pe_id", "pe") == {"PE-1"}
    assert _item_rows([{"pe_id": "PE-1", "title": "notice:title:3"}], rows, "pe_id", "pe") == set()  # another table's row linking it


def test_the_clock_is_told_and_a_date_failure_is_hinted_as_a_dispute():
    # worlds-2: purchase requisition's t11 fixture (required_by 2024-06-01) failed "requisition expired" against the clock
    # 2026-01-01 through three debug rounds; neither the tester nor the coder had been told the clock
    from datetime import UTC, datetime

    from crucible.synth.datagen import TODAY
    from crucible.synth.iface import PyWorld
    from crucible.synth.roles import APP_API_DOC, CLOCK_T0, TEST_FORMAT_DOC
    from crucible.synth.worldgen import _hint

    assert PyWorld([]).t0 == TODAY == datetime.fromisoformat(CLOCK_T0).astimezone(UTC)
    assert "2026-01-01" in APP_API_DOC and "2026-01-01" in TEST_FORMAT_DOC
    assert "dispute" in _hint("call 0 create_purchase_order: expected success, got error ToolError: INVALID_STATE: requisition expired")
    assert _hint("call 0 x: expected success, got error ToolError: NOT_FOUND: no such row") == ""
