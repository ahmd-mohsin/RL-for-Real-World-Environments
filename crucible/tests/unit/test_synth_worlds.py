"""crucible.synth.families / worldgates / worldkit: World families (built once, frozen), World instances (no model
call), and the gates. Every model reply is scripted (``synth_world_fixtures``)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from crucible.synth.domain import DomainSpec
from crucible.synth.families import (
    DocsOut,
    FamilyConfig,
    build_family,
    check_docs,
    instance_config,
    make_instance,
    make_instances,
    realism_design_problems,
    targets,
)
from crucible.synth.iface import App, PyWorld
from crucible.synth.roles import ToolDesignOut
from crucible.synth.worldgates import RealismPanel, anchors_from, calibrate_panel, degrade, gate_instance, instance_summary
from crucible.synth.worldkit import register
from tests.unit.synth_fakes import FakeClient, const, seq, user_text
from tests.unit.synth_world_fixtures import DESIGN, DOCS, REALISTIC, SCHEMA, SPEC, UNREALISTIC, family_clients


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import asyncio

    root = tmp_path_factory.mktemp("worlds")
    cl = family_clients()
    st = asyncio.run(build_family("IT service desk ticket handling", root / "service_desk", cl, cfg=FamilyConfig(k_min=40)))
    return root / "service_desk", st, cl


def test_a_family_builds_frozen_with_stage_records(built):
    fdir, st, cl = built
    assert st.status == "frozen", st.reasons
    assert list(st.stages) == ["spec", "design", "schema", "code", "tests", "debug", "datagen", "docs", "actors", "base", "freeze"]
    assert all(r.status == "done" and r.elapsed_s >= 0 for r in st.stages.values())
    man = json.loads((fdir / "app" / "world.json").read_text())
    assert set(man["files"]) >= {"app.py", "tests.json", "datagen.json", "docs.json", "actors.json", "domain.json", "kit.json"}
    schema = json.loads((fdir / "app" / "schema.json").read_text())
    assert {t["name"] for t in schema["tables"]} >= {"documents", "policy_params", "colleagues"}  # the kit tables
    tester = user_text(cl["blind_tester"].calls[0])
    assert "## code" not in tester and "policy_params" in tester  # blind, but told the policy parameters
    assert "PERMISSION_DENIED" in user_text(cl["family_designer"].calls[0])  # the realism surface reaches the designer


async def test_resuming_a_frozen_family_calls_no_model(built):
    fdir, _st, cl = built
    before = {r: len(c.calls) for r, c in cl.items()}
    st = await build_family("IT service desk ticket handling", fdir, cl)
    assert st.status == "frozen" and {r: len(c.calls) for r, c in cl.items()} == before


async def test_a_resumed_family_restarts_at_the_failed_stage(tmp_path):
    bad_docs = copy.deepcopy(DOCS)
    bad_docs["documents"][3]["kind"] = "faq"
    cl = family_clients(doc_writer=FakeClient("glm", const(bad_docs)))
    st = await build_family("desk", tmp_path / "f", cl)
    assert st.status == "rejected:docs" and "data_standards" in st.reasons[0]
    calls = len(cl["app_coder"].calls)
    cl2 = family_clients(doc_writer=FakeClient("glm", const(DOCS)), app_coder=cl["app_coder"])
    st = await build_family("desk", tmp_path / "f", cl2)
    assert st.status == "frozen" and len(cl["app_coder"].calls) == calls  # code, tests and data were not rebuilt
    assert st.stages["docs"].attempts == 2


def test_realism_design_problems_and_feedback():
    d = copy.deepcopy(DESIGN)
    for t in d["tools"]:
        t["errors"] = [e for e in t.get("errors", []) if e != "PERMISSION_DENIED"]
        t["params"] = [p for p in t["params"] if p["name"] not in ("page", "page_size")]
    probs = " ".join(realism_design_problems(ToolDesignOut.model_validate(d)))
    assert "no read tool pages" in probs and "PERMISSION_DENIED" in probs
    d2 = copy.deepcopy(DESIGN)
    d2["tools"].append({"name": "read_document", "description": "x", "read_only": True, "params": []})
    assert "platform provides" in " ".join(realism_design_problems(ToolDesignOut.model_validate(d2)))
    d3 = copy.deepcopy(DESIGN)  # pilot 1: the acting user passed as an argument is an impersonation shortcut
    d3["tools"][3]["params"].append({"name": "acting_user_id", "type": "string"})
    assert "acting user as an argument" in " ".join(realism_design_problems(ToolDesignOut.model_validate(d3)))


async def test_the_designer_gets_the_realism_problems_back(tmp_path):
    d = copy.deepcopy(DESIGN)
    for t in d["tools"]:
        t["params"] = [p for p in t["params"] if p["name"] not in ("page", "page_size")]
    designer = FakeClient("kimi", seq(d, DESIGN))
    st = await build_family("desk", tmp_path / "f", family_clients(family_designer=designer))
    assert st.status == "frozen" and len(designer.calls) == 2 and "no read tool pages" in user_text(designer.calls[1])


def test_docs_must_state_every_parameter_and_the_data_standards():
    spec = DomainSpec.model_validate(SPEC)
    check_docs(DocsOut.model_validate(DOCS), spec, stale_values=["archived"])
    d = copy.deepcopy(DOCS)
    d["documents"][0]["body"] = d["documents"][0]["body"].replace("{{max_open_per_agent}}", "six")
    with pytest.raises(ValueError, match="max_open_per_agent"):
        check_docs(DocsOut.model_validate(d), spec)
    d = copy.deepcopy(DOCS)
    d["documents"][1]["body"] += " {{bogus}}"
    with pytest.raises(ValueError, match="unknown parameters"):
        check_docs(DocsOut.model_validate(d), spec)


def test_targets_are_written_collections_named_by_key():
    assert targets(DESIGN, SCHEMA) == ["tickets"]  # the audit log is written but never named
    assert targets(DESIGN) == ["audit", "tickets"]


def test_instances_are_distinct_deterministic_and_dense(built):
    fdir, _st, _cl = built
    reps = make_instances(fdir, 3)
    assert [r.config.profile for r in reps] == ["standard", "dense", "noisy"]
    assert len({r.seed_digest for r in reps}) == 3
    again = make_instance(fdir, instance_config(fdir.name, 1))
    assert again.seed_digest == reps[1].seed_digest
    assert reps[0].params["max_open_per_agent"] == 6  # instance 0: the spec's defaults
    seed = json.loads((Path(reps[1].dir) / "seed.json").read_text())["tables"]
    assert len(seed["tickets"]) >= 64 and reps[2].noise["near_duplicate"] > reps[0].noise["near_duplicate"]
    staff = {s["staff_id"]: s for s in seed["staff"]}
    assert staff[reps[1].params["acting_user_id"]]["role"] in ("lead", "manager")  # the actor config honours the ref's filter
    docs = {d["doc_id"]: d["body"] for d in seed["documents"]}
    assert f"at most {reps[1].params['max_open_per_agent']} open tickets" in docs["DOC-1"]  # documents rendered with the policy variant
    assert seed["colleagues"][0]["facts"][0]["text"].startswith("Ticket ")


def test_a_tampered_family_yields_no_instance(built, tmp_path):
    import shutil

    fdir, _st, _cl = built
    copy_dir = tmp_path / fdir.name
    shutil.copytree(fdir, copy_dir, ignore=shutil.ignore_patterns("instances", "work"))
    p = copy_dir / "app" / "app.py"
    p.write_text(p.read_text() + "\n# edited\n")
    with pytest.raises(ValueError, match="changed since it was frozen"):
        make_instance(copy_dir, instance_config(fdir.name, 0))


def _panel(ok: dict, tmp: Path, conf: float = 0.9) -> RealismPanel:
    a, b = dict(ok, confidence=conf), dict(ok, confidence=conf)
    return RealismPanel(members=[("r-nemotron", FakeClient("nemotron", const(a))), ("r-gptoss", FakeClient("gpt-oss", const(b)))], lam=0.7,
                        queue=tmp / "human_queue.jsonl")


async def test_every_gate_passes_on_a_good_instance(built, tmp_path):
    fdir, _st, _cl = built
    rep = make_instance(fdir, instance_config(fdir.name, 1))
    g = await gate_instance(fdir, Path(rep.dir), panel=_panel(REALISTIC, tmp_path))
    assert g.usable, g.why
    d = g.gates["density"].details
    assert d["reached"] >= 2 and d["best"] >= 59 and d["targets"]["tickets"] >= 40
    assert g.gates["determinism"].details["regenerated_identical"] and all(g.gates["determinism"].details["replays"].values())
    assert g.gates["surface"].details["permission_tests"] == ["assign_perm", "prio_critical"]
    assert g.gates["surface"].details["impersonation_accepted"] == []  # the acting user cannot be passed as an argument
    assert g.gates["obscurity"].details["params_unreachable"] == [] and g.gates["obscurity"].details["discoverable"]["tickets"] == 1.0
    assert json.loads((Path(rep.dir) / "gates.json").read_text())["usable"] is True


async def test_gates_fail_with_reasons(built, tmp_path):
    fdir, _st, _cl = built
    rep = make_instance(fdir, instance_config(fdir.name, 2))
    inst = Path(rep.dir)
    g = await gate_instance(fdir, inst, k_min=500, only=["density"])
    assert not g.gates["density"].passed and "fewer than 500" in g.gates["density"].reasons[0]
    seed = json.loads((inst / "seed.json").read_text())
    seed["tables"]["documents"][0]["body"] = seed["tables"]["documents"][0]["body"].replace("open tickets", "open tickets (value redacted)", 1)
    pp = next(p for p in seed["tables"]["policy_params"] if p["name"] == "max_open_per_agent")
    pp["value"] = 999  # the App would now behave unlike the documents
    (inst / "seed.json").write_text(json.dumps(seed))
    g = await gate_instance(fdir, inst, only=["integrity", "determinism", "build"])
    assert not g.gates["integrity"].passed and any("max_open_per_agent" in r for r in g.gates["integrity"].reasons)
    assert not g.gates["determinism"].passed  # the stored start state no longer regenerates from its config
    assert g.gates["build"].passed
    (inst / "app.py").write_text((inst / "app.py").read_text() + "\n")
    g = await gate_instance(fdir, inst, only=["build"])
    assert not g.gates["build"].passed and "differs from the frozen family" in g.gates["build"].reasons[0]
    assert not g.usable


async def test_the_panel_abstains_into_the_human_queue_and_rejects_toys(built, tmp_path):
    fdir, _st, _cl = built
    rep = make_instance(fdir, instance_config(fdir.name, 0))
    split = RealismPanel(members=[("a", FakeClient("nemotron", const(REALISTIC))), ("b", FakeClient("gpt-oss", const(UNREALISTIC)))], lam=0.7,
                         queue=tmp_path / "q.jsonl")
    g = await gate_instance(fdir, Path(rep.dir), panel=split, only=["panel"])
    assert not g.gates["panel"].passed and g.gates["panel"].details["decision"] == "abstain"
    assert json.loads((tmp_path / "q.jsonl").read_text().splitlines()[0])["id"].endswith(rep.id)
    g = await gate_instance(fdir, Path(rep.dir), panel=_panel(UNREALISTIC, tmp_path), only=["panel"])
    assert g.gates["panel"].details["decision"] == "unrealistic"
    with pytest.raises(ValueError):
        RealismPanel(members=[("a", FakeClient("nemotron", const(REALISTIC))), ("b", FakeClient("nemotron", const(REALISTIC)))])


async def test_calibration_sets_lambda_from_known_anchors(built):
    fdir, _st, _cl = built
    rep = make_instance(fdir, instance_config(fdir.name, 0))
    s = await instance_summary(Path(rep.dir))
    assert {"tools", "collections", "documents", "colleagues", "noise_examples"} <= set(s)
    for k in ("placeholder", "toy", "incoherent"):
        assert degrade(s, k) != s
    anchors = anchors_from([s])

    def judge(msgs, _n):  # a rater that tells the reviewed summary from its degradations, sure only of the toys
        txt = user_text(msgs)
        if "does stuff" in txt or "lorem ipsum" in txt or "Qx" in txt or "Amounts can be negative" in txt:
            return dict(UNREALISTIC, confidence=0.95)
        if '"rows": 2' in txt:
            return dict(UNREALISTIC, confidence=0.65)
        return dict(REALISTIC, confidence=0.85)

    p = RealismPanel(members=[("a", FakeClient("nemotron", judge)), ("b", FakeClient("gpt-oss", judge))])
    cal = await calibrate_panel(p, anchors)
    assert cal["calibrated"] and cal["lam"] == 0.6 and p.lam == 0.6
    assert cal["rows"][0]["accuracy"] == 1.0


def test_the_kit_tools():
    import asyncio

    app = App("desk", {"documents": {"D1": {"doc_id": "D1", "title": "Refund policy", "kind": "policy", "tags": ["refunds"], "body": "Refunds above 500 need approval.",
                                            "version": "1", "effective_date": "2025-01-01"},
                                     "D2": {"doc_id": "D2", "title": "FAQ", "kind": "faq", "tags": [], "body": "Ask the lead.", "version": "1", "effective_date": "2025-01-01"}},
                       "colleagues": {"C1": {"colleague_id": "C1", "name": "Rui", "title": "lead", "topics": ["payroll"],
                                             "facts": [{"topic": "payroll", "keywords": ["payroll"], "text": "INC-1 blocks payroll."}]}}})
    register(app, ["docs", "actors"])
    w = PyWorld([app])

    async def go() -> None:
        await w.reset(0)
        o = await w.call("desk.list_documents", {"page_size": 1})
        assert o.value["total"] == 2 and o.value["next_page"] == 2 and len(o.value["items"]) == 1
        assert (await w.call("desk.list_documents", {"kind": "faq"})).value["items"][0]["doc_id"] == "D2"
        assert (await w.call("desk.search_documents", {"query": "refunds 500"})).value["results"][0]["doc_id"] == "D1"
        assert (await w.call("desk.read_document", {"doc_id": "D9"})).error.startswith("ToolError: NOT_FOUND")
        assert "blocks payroll" in (await w.call("desk.ask_colleague", {"colleague_id": "C1", "question": "Anything about payroll?"})).value["reply"]
        assert "can help with: payroll" in (await w.call("desk.ask_colleague", {"colleague_id": "C1", "question": "vacation plans?"})).value["reply"]

    asyncio.run(go())


def test_the_wire_binds_world_roles_with_family_separation():
    from crucible.factory.models import DEEPSEEK, GLM, KIMI, SameFamilyError
    from crucible.factory.synth_wire import SYNTH_ROLES, WORLD_ROLE_TABLE, SynthWire, check_assignment

    check_assignment({**SYNTH_ROLES, **WORLD_ROLE_TABLE})
    with pytest.raises(SameFamilyError, match="realism_rater"):
        check_assignment({**SYNTH_ROLES, **WORLD_ROLE_TABLE, "realism_rater": (GLM, DEEPSEEK)})  # a builder's family judging its own World
    with pytest.raises(SameFamilyError, match="Qwen|qwen"):
        check_assignment({**SYNTH_ROLES, **WORLD_ROLE_TABLE, "doc_writer": ("tinker:Qwen/Qwen3.6-35B-A3B:peft:262144",)})
    with pytest.raises(SameFamilyError, match="blind_tester"):
        check_assignment({**SYNTH_ROLES, **WORLD_ROLE_TABLE, "blind_tester": (KIMI,)})
    w = SynthWire("/nonexistent-unused", cap_usd=0.0, table={**SYNTH_ROLES, **WORLD_ROLE_TABLE}, make=lambda m, r: object())
    assert set(w.family_clients()) >= {"family_designer", "data_spec_writer", "doc_writer", "blind_tester"}
