"""crucible.synth.domain (DomainSpec, the cell grid, loaders, the swarm generator) and crucible.synth.catalogue
(the domain catalogue and its dedupe rule). Every model reply is scripted."""

from __future__ import annotations

import copy
import json
from collections import Counter

import pytest

from crucible.synth.catalogue import Catalogue, CatalogueEntry, add_entries, build_catalogue, duplicate_of, load, pick
from crucible.synth.domain import DomainSpec, bind_design, check_spec, finance_ap_spec, generate_spec, load_spec, spec_from_world
from tests.unit.synth_fakes import FakeClient, seq, user_text
from tests.unit.synth_world_fixtures import DESIGN, SCHEMA, SPEC


def test_the_fixture_spec_passes_and_its_cells_are_balanced():
    s = DomainSpec.model_validate(SPEC)
    check_spec(s)
    knobs = s.all_knobs()
    assert {k.name for k in knobs} >= {"queue_load", "requester_vip", "evidence", "distractor", "authority", "scope"}
    assert s.grid_size() == 6 * 2 * 2 * 4 * 4 * 2 * 2
    cells = s.sample_cells(48, seed=1)
    assert Counter(c.skill for c in cells) == Counter({sk.id: 8 for sk in s.skills})
    for k in knobs:  # every value of every knob is drawn
        assert {c.knobs[k.name] for c in cells} == set(k.values)
    assert s.sample_cells(48, seed=1) == cells and s.sample_cells(48, seed=2) != cells
    t = s.cell_text(cells[0])
    assert t.startswith(f"skill {cells[0].skill}:") and "distractor:" in t


@pytest.mark.parametrize("edit, msg", [
    (lambda d: d.update(skills=d["skills"][:3]), "at least 5 skills"),
    (lambda d: d.update(knobs=[]), "domain-specific knobs"),
    (lambda d: d["policy_params"][1].update(default=99), "outside"),
    (lambda d: d.update(policy_params=[p for p in d["policy_params"] if p["name"] != "acting_user_id"]), "acting_user_id"),
    (lambda d: d.update(common_practice=False), "common enterprise practice"),
])
def test_check_spec_names_the_problem(edit, msg):
    d = copy.deepcopy(SPEC)
    edit(d)
    with pytest.raises(ValueError, match=msg):
        check_spec(DomainSpec.model_validate(d))


async def test_generate_spec_feeds_problems_back_to_the_swarm():
    bad = copy.deepcopy(SPEC)
    bad["skills"] = bad["skills"][:2]
    c = FakeClient("glm", seq(bad, SPEC))
    s = await generate_spec("IT service desk", c, function="IT")
    assert s.name == "service_desk" and len(c.calls) == 2
    assert "at least 5 skills" in user_text(c.calls[1])


def test_loaders_and_binding(tmp_path):
    s = bind_design(DomainSpec.model_validate(SPEC), DESIGN, SCHEMA)
    app = s.apps[0]
    assert app.name == "desk" and "assign_ticket" in app.writes and "list_tickets" in app.read_only
    assert s.targetable() == ["audit", "tickets"]
    assert any("assign_ticket" in sk.tools for sk in s.skills)
    fa = finance_ap_spec()
    check_spec_ok = fa.model_copy()
    assert check_spec_ok.name == "finance_ap" and fa.cells(limit=5)
    (tmp_path / "design.json").write_text(json.dumps(DESIGN))
    (tmp_path / "schema.json").write_text(json.dumps(SCHEMA))
    g = spec_from_world(tmp_path)  # a worldgen World without domain.json: a skill per write tool
    assert {sk.id for sk in g.skills} == {"assign_ticket", "set_priority", "close_ticket"}
    (tmp_path / "domain.json").write_text(s.model_dump_json())
    assert load_spec("service_desk", tmp_path).apps[0].name == "desk"
    with pytest.raises(ValueError):
        load_spec("other")


def _e(kw: str, fn: str = "finance", records: tuple[str, ...] = ()) -> CatalogueEntry:
    return CatalogueEntry(keyword=kw, function=fn, records=list(records))


def test_dedupe_rule():
    kept = [_e("accounts payable invoice processing", records=("invoice", "purchase order", "vendor"))]
    assert duplicate_of(_e("invoice processing for accounts payable"), kept) == "accounts payable invoice processing"
    assert duplicate_of(_e("supplier invoice handling", records=("invoices", "purchase orders", "vendors")), kept) == "accounts payable invoice processing"
    assert duplicate_of(_e("supplier invoice handling", fn="procurement", records=("invoices", "purchase orders", "vendors")), kept) is None
    assert duplicate_of(_e("expense report approval", records=("expense report", "receipt", "employee")), kept) is None
    cat = Catalogue(entries=list(kept))
    added = add_entries(cat, [_e("expense report approval"), _e("AP invoice processing management"), CatalogueEntry(keyword="crypto mining rigs",
                                                                                                                  function="IT", common_practice=False)])
    assert [e.keyword for e in added] == ["expense report approval"]
    assert cat.duplicates[0]["duplicate_of"] == "accounts payable invoice processing" and cat.refused[0]["keyword"] == "crypto mining rigs"


async def test_build_catalogue_is_resumable_and_pick_is_diverse(tmp_path):
    def reply(msgs, n):
        area = user_text(msgs).split("## function_area\n")[1].split("\n")[0]
        fn = area.split(":")[0]
        return {"domains": [{"keyword": f"{fn} {('alpha ledger', 'beta intake')[i]}", "function": fn, "records": [("ledger entry", "intake form")[i]],
                            "common_practice": True} for i in range(2)]}

    c = FakeClient("deepseek", reply)
    path = tmp_path / "catalogue.json"
    areas = ("finance: ap", "HR: leave", "IT: desk")
    cat = await build_catalogue(c, per_area=2, areas=areas[:2], path=path)
    assert len(cat.entries) == 4 and len(c.calls) == 2
    cat = await build_catalogue(c, per_area=2, areas=areas, cat=load(path), path=path)
    assert len(c.calls) == 3 and len(cat.entries) == 6  # the two areas already asked are skipped
    assert "## existing" in c.calls[-1][1]["content"]  # the swarm sees the catalogue so far
    p = pick(cat, 3)
    assert len({e.function for e in p}) == 3
