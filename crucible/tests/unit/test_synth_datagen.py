"""crucible.synth.datagen: the deterministic instance data generator, its noise and its integrity checks."""

from __future__ import annotations

import copy

import pytest

from crucible.synth.datagen import (
    PROFILES,
    DataSpec,
    DataSpecError,
    Profile,
    compile_spec,
    eval_expr,
    generate,
    integrity,
    near_variant,
    normalize_spec,
    reads_aggregate,
    rewrite_implies,
    rule_holds,
)
from tests.unit.synth_world_fixtures import DATAGEN, SCHEMA


def _schema() -> dict:
    return {t["name"]: {"key": t["key"], "fields": t["fields"]} for t in SCHEMA["tables"]}


def test_generation_is_deterministic_per_seed_and_profile():
    spec = DataSpec.model_validate(DATAGEN)
    a = generate(spec, _schema(), seed=3)
    b = generate(spec, _schema(), seed=3)
    c = generate(spec, _schema(), seed=4)
    assert a.model_dump() == b.model_dump()
    assert a.tables["tickets"] != c.tables["tickets"]
    d = generate(spec, _schema(), seed=3, profile=PROFILES["dense"])
    assert len(d.tables["tickets"]) > len(a.tables["tickets"])


def test_the_fixture_spec_is_clean_and_plants_every_noise_kind():
    spec = DataSpec.model_validate(DATAGEN)
    g = generate(spec, _schema(), seed=0, profile=Profile(min_rows={"tickets": 80}))
    assert integrity(spec, _schema(), g) == []
    kinds = {n.kind for n in g.noise}
    assert kinds == {"near_duplicate", "stale", "conflict"}
    base = [r for r in g.tables["tickets"] if r["ticket_id"] not in {str(n.key) for n in g.noise if n.kind != "conflict"}]
    assert len(base) >= 80
    staff = {r["staff_id"] for r in g.tables["staff"]}
    assert all(r["assignee_id"] in staff for r in g.tables["tickets"] if r["assignee_id"])  # referential integrity
    for n in g.noise:
        if n.kind == "near_duplicate":
            twin = next(r for r in g.tables[n.table] if r[_schema()[n.table]["key"]] == n.of)
            new = next(r for r in g.tables[n.table] if r[_schema()[n.table]["key"]] == n.key)
            assert [f for f in new if new[f] != twin[f]] == [_schema()[n.table]["key"], n.field]  # one field varied
        if n.kind == "stale":
            new = next(r for r in g.tables[n.table] if r[_schema()[n.table]["key"]] == n.key)
            assert new["status"] == "archived"
    assert all(r["due_at"] >= r["created_at"] for r in g.tables["tickets"])
    assert all(r["created_at"] <= "2026-01-01" for r in g.tables["tickets"])  # history before the World clock's today
    closed = [r for r in g.tables["tickets"] if r["status"] == "closed"]
    assert closed and all(r["resolution"] for r in closed) and all(r["resolution"] is None for r in g.tables["tickets"] if r["status"] == "open")


def test_integrity_catches_broken_references_values_dates_rules_and_unplanted_conflicts():
    spec = DataSpec.model_validate(DATAGEN)
    g = generate(spec, _schema(), seed=1)
    bad = copy.deepcopy(g)
    plain = [r for r in bad.tables["tickets"] if r["ticket_id"] not in {str(n.key) for n in bad.noise}]
    plain[0]["assignee_id"] = "S-999"
    plain[1]["priority"] = "urgent"
    plain[2]["created_at"] = "2027-05-01"
    plain[3]["due_at"] = "2020-01-01"
    with_name = next(r for r in plain[4:] if r["assignee_id"])
    with_name["assignee_name"] = "Someone Else"
    probs = " | ".join(integrity(spec, _schema(), bad))
    for want in ("S-999", "urgent", "after today", "before created_at", "disagrees with its master"):
        assert want in probs, want
    bad.tables["staff"][0].pop("team")
    assert any("no field team" in p for p in integrity(spec, _schema(), bad))


def test_compile_refuses_bad_specs_with_feedback():
    s = _schema()
    d = copy.deepcopy(DATAGEN)
    d["tables"][0]["fields"]["role"] = {"kind": "magic"}
    with pytest.raises(DataSpecError, match="unknown kind"):
        compile_spec(DataSpec.model_validate(d), s)
    d = copy.deepcopy(DATAGEN)
    del d["tables"][1]["fields"]["subject"]
    with pytest.raises(DataSpecError, match="no generator for schema fields"):
        compile_spec(DataSpec.model_validate(d), s)
    d = copy.deepcopy(DATAGEN)
    d["tables"][1]["fields"]["created_at"] = {"kind": "date_offset", "from": "due_at"}
    with pytest.raises(DataSpecError, match="cycle"):
        compile_spec(DataSpec.model_validate(d), s)
    d = copy.deepcopy(DATAGEN)  # worlds-1: a stale marker the live rows never take ("inactive") is accepted, on stale rows only
    d["tables"][1]["stale"] = {"set": {"status": "gone"}, "older": ["created_at"]}
    g = generate(DataSpec.model_validate(d), s, seed=0, profile=Profile(stale=0.2))
    assert integrity(DataSpec.model_validate(d), s, g) == []
    gone = [r for r in g.tables["tickets"] if r["status"] == "gone"]
    assert gone and {r["ticket_id"] for r in gone} <= {str(n.key) for n in g.noise if n.kind == "stale"}


def test_aggregates_and_expressions():
    schema = {"orders": {"key": "id", "fields": {"id": "string", "total": "number", "lines": "integer"}},
              "lines": {"key": "lid", "fields": {"lid": "integer", "order_id": "string", "qty": "integer", "price": "number", "amount": "number"}}}
    spec = DataSpec.model_validate({"tables": [
        {"table": "orders", "rows": 5, "fields": {"id": {"kind": "seq", "prefix": "O-"}, "total": {"kind": "sum", "table": "lines", "fk": "order_id", "field": "amount"},
                                                  "lines": {"kind": "count", "table": "lines", "fk": "order_id"}}},
        {"table": "lines", "rows": 20, "fields": {"lid": {"kind": "seq"}, "order_id": {"kind": "ref", "table": "orders"}, "qty": {"kind": "int", "min": 1, "max": 9},
                                                  "price": {"kind": "number", "min": 1, "max": 50}, "amount": {"kind": "calc", "expr": "qty * price", "round": 2}}}]})
    g = generate(spec, schema, seed=0, profile=Profile(near_duplicate=0, stale=0, conflict=0))
    assert integrity(spec, schema, g) == []
    for o in g.tables["orders"]:
        mine = [ln for ln in g.tables["lines"] if ln["order_id"] == o["id"]]
        assert o["lines"] == len(mine) and abs(o["total"] - sum(ln["amount"] for ln in mine)) < 0.02
    assert all(isinstance(ln["lid"], int) for ln in g.tables["lines"])  # an integer key with no prefix stays an integer
    assert eval_expr("a + b * 2 if c in ['x'] else 0", {"a": 1, "b": 2, "c": "x"}) == 5
    for bad in ("a.__class__", "open('x')", "[x for x in a]", "lambda: 1"):
        with pytest.raises(DataSpecError):
            eval_expr(bad, {"a": 1})


def test_near_variants_always_differ():
    import random

    rng = random.Random(0)
    for v in ("Acme Inc.", "Bo", "Nordic Supplies Ltd", 1200, 19.5, True, "x"):
        for _ in range(20):
            assert near_variant(v, rng) != v


def test_rows_are_rejection_sampled_against_rules_and_absolute_spreads_do_not_overflow():
    # pilot 1: independent draws broke "used <= entitled"; an sd given in the value's units overflowed the log sigma
    schema = {"bal": {"key": "id", "fields": {"id": "string", "entitled": "integer", "used": "integer", "amount": "number"}}}
    spec = DataSpec.model_validate({"tables": [{"table": "bal", "rows": 60, "rules": ["used <= entitled"], "fields": {
        "id": {"kind": "seq", "prefix": "B-"}, "entitled": {"kind": "int", "min": 5, "max": 30}, "used": {"kind": "int", "min": 0, "max": 30},
        "amount": {"kind": "number", "dist": "lognormal", "mean": 2500, "sd": 4000, "min": 10, "max": 90000}}}]})
    g = generate(spec, schema, seed=0, profile=Profile(near_duplicate=0, stale=0, conflict=0))
    assert integrity(spec, schema, g) == []
    assert [r["id"] for r in g.tables["bal"]] == [f"B-{i}" for i in range(1, 61)]  # redraws release their sequence numbers
    assert len({r["amount"] for r in g.tables["bal"]}) > 30


def test_cross_record_rules_hold_and_are_checked():
    # pilot 1: every loss notice fell after its policy's expiry, so the App refused them all
    schema = {"policy": {"key": "pid", "fields": {"pid": "string", "start": "string", "expiry": "string", "tier": "string"}},
              "notice": {"key": "nid", "fields": {"nid": "string", "policy_id": "string", "loss_date": "string", "cap": "number"}}}
    spec = DataSpec.model_validate({"tables": [
        {"table": "policy", "rows": 20, "fields": {"pid": {"kind": "seq", "prefix": "P-"}, "start": {"kind": "date", "days_ago": [200, 700]},
                                                  "expiry": {"kind": "date_offset", "from": "start", "min_days": 30, "max_days": 365},
                                                  "tier": {"kind": "choice", "values": ["gold", "basic"]}}},
        {"table": "notice", "rows": 50, "rules": ["loss_date >= policy_id.start", "loss_date <= policy_id.expiry"], "fields": {
            "nid": {"kind": "seq", "prefix": "N-"}, "policy_id": {"kind": "ref", "table": "policy"}, "loss_date": {"kind": "date", "days_ago": [1, 700]},
            "cap": {"kind": "number", "min": 100, "max": 900, "when": "policy_id.tier == 'gold'", "else": {"kind": "number", "min": 10, "max": 90}}}}]})
    g = generate(spec, schema, seed=0, profile=Profile(near_duplicate=0, stale=0, conflict=0))
    assert integrity(spec, schema, g) == []
    pol = {p["pid"]: p for p in g.tables["policy"]}
    assert all(pol[n["policy_id"]]["start"] <= n["loss_date"] <= pol[n["policy_id"]]["expiry"] for n in g.tables["notice"])
    assert all((n["cap"] >= 100) == (pol[n["policy_id"]]["tier"] == "gold") for n in g.tables["notice"])  # a generator as the else branch
    g.tables["notice"][0]["loss_date"] = "2025-12-31"
    pol[g.tables["notice"][0]["policy_id"]]["expiry"] = "2025-01-01"
    assert any("policy_id.expiry" in p for p in integrity(spec, schema, g))
    bad = spec.model_copy(deep=True)
    bad.tables[1].rules = ["loss_date <= nid.expiry"]
    with pytest.raises(DataSpecError, match="not a ref field"):
        compile_spec(bad, schema)


def test_noise_keeps_declared_values_and_dates_can_follow_a_referenced_row():
    # pilot 1: a near-duplicate varying a status wrote a typo ("deneid") that failed the declared-values check; an
    # approval date had to follow its requisition's date
    schema = {"req": {"key": "rid", "fields": {"rid": "string", "raised": "string", "status": "string"}},
              "appr": {"key": "aid", "fields": {"aid": "string", "req_id": "string", "decided": "string"}}}
    spec = DataSpec.model_validate({"tables": [
        {"table": "req", "rows": 40, "near_duplicate": {"vary": ["status"]}, "fields": {
            "rid": {"kind": "seq", "prefix": "R-"}, "raised": {"kind": "date", "days_ago": [30, 300]},
            "status": {"kind": "choice", "values": ["open", "approved", "denied"]}}},
        {"table": "appr", "rows": 30, "fields": {"aid": {"kind": "seq", "prefix": "A-"}, "req_id": {"kind": "ref", "table": "req"},
                                                "decided": {"kind": "date_offset", "from": "req_id.raised", "min_days": 1, "max_days": 9}}}]})
    g = generate(spec, schema, seed=2, profile=Profile(near_duplicate=0.3, stale=0, conflict=0))
    assert integrity(spec, schema, g) == []
    assert all(r["status"] in ("open", "approved", "denied") for r in g.tables["req"])
    raised = {r["rid"]: r["raised"] for r in g.tables["req"]}
    assert all(a["decided"] > raised[a["req_id"]] for a in g.tables["appr"])


def test_two_way_links_are_backrefs_and_null_is_accepted():
    # pilot 1: a notice links its claim and a claim its notice (refused as a cycle); the swarm writes null in rules
    schema = {"notice": {"key": "nid", "fields": {"nid": "string", "claim_id": "string?"}},
              "claim": {"key": "cid", "fields": {"cid": "string", "notice_id": "string"}}}
    spec = DataSpec.model_validate({"tables": [
        {"table": "notice", "rows": 20, "fields": {"nid": {"kind": "seq", "prefix": "N-"}, "claim_id": {"kind": "backref", "table": "claim", "fk": "notice_id"}}},
        {"table": "claim", "rows": 12, "rules": ["notice_id is not null"],
         "fields": {"cid": {"kind": "seq", "prefix": "C-"}, "notice_id": {"kind": "ref", "table": "notice", "unique": True}}}]})
    g = generate(spec, schema, seed=0, profile=Profile(near_duplicate=0, stale=0, conflict=0))
    assert integrity(spec, schema, g) == []
    by = {c["cid"]: c["notice_id"] for c in g.tables["claim"]}
    linked = [n for n in g.tables["notice"] if n["claim_id"]]
    assert len(linked) == 12 and all(by[n["claim_id"]] == n["nid"] for n in linked)
    cyc = copy.deepcopy(spec.model_dump())
    cyc["tables"][0]["fields"]["claim_id"] = {"kind": "ref", "table": "claim"}
    with pytest.raises(DataSpecError, match="backref"):
        compile_spec(DataSpec.model_validate(cyc), schema)


def test_every_static_problem_comes_back_at_once():
    # worlds-1: the purchase-requisition writer got one problem per round over four rounds (32 logged refusals)
    d = copy.deepcopy(DATAGEN)
    d["tables"][0]["fields"]["role"] = {"kind": "magic"}
    d["tables"][1]["rules"].append("due_at >= ghost")
    d["tables"][1]["near_duplicate"] = {"vary": ["nothing"]}
    d["tables"][2]["fields"]["ticket_id"] = {"kind": "ref", "table": "nowhere"}
    with pytest.raises(DataSpecError) as e:
        compile_spec(DataSpec.model_validate(d), _schema())
    msg = str(e.value)
    assert msg.startswith("4 problems") and all(w in msg for w in ("unknown kind", "ghost", "nothing", "nowhere"))


def test_implies_chains_and_sql_nulls():
    # worlds-1: `implies` was invalid syntax, `po_id.requisition_id.raised_at` was refused, `limit >= 0` on null was a TypeError
    assert rewrite_implies("a == 1 implies b > 2") == "(not (a == 1)) or (b > 2)"
    assert rewrite_implies("x and (a implies b) or c") == "x and ((not (a)) or (b)) or c"
    assert rewrite_implies("s == 'implies' or t") == "s == 'implies' or t"  # a string literal is not the operator
    assert eval_expr("status == 'approved' implies budget != 'exceeded'", {"status": "approved", "budget": "exceeded"}) is False
    assert eval_expr("status == 'approved' implies budget != 'exceeded'", {"status": "draft", "budget": "exceeded"}) is True
    assert eval_expr("limit >= 0", {"limit": None}) is None and rule_holds(None)
    assert eval_expr("limit >= 0", {"limit": -1}) is False and not rule_holds(False)
    assert eval_expr("a == null or a > b", {"a": None, "b": 1}) is True
    assert eval_expr("a > 1 or b > 1", {"a": None, "b": 0}) is None  # unknown, not False
    assert eval_expr("a > 1 and b > 1", {"a": None, "b": 0}) is False
    assert eval_expr("not (a > 1)", {"a": None}) is None and eval_expr("a + 1", {"a": None}) is None and eval_expr("round(a)", {"a": None}) is None
    schema = {"req": {"key": "rid", "fields": {"rid": "string", "raised": "string", "limit": "number?"}},
              "po": {"key": "pid", "fields": {"pid": "string", "req_id": "string", "ordered": "string"}},
              "inv": {"key": "iid", "fields": {"iid": "string", "po_id": "string", "dated": "string"}}}
    spec = DataSpec.model_validate({"tables": [
        {"table": "req", "rows": 20, "rules": ["limit >= 0"], "fields": {"rid": {"kind": "seq", "prefix": "R-"}, "raised": {"kind": "date", "days_ago": [200, 300]},
                                                                      "limit": {"kind": "number", "min": 10, "max": 90, "null_p": 0.5}}},
        {"table": "po", "rows": 20, "fields": {"pid": {"kind": "seq", "prefix": "P-"}, "req_id": {"kind": "ref", "table": "req"},
                                              "ordered": {"kind": "date_offset", "from": "req_id.raised", "min_days": 1, "max_days": 20}}},
        {"table": "inv", "rows": 30, "rules": ["dated >= po_id.req_id.raised", "po_id.req_id.limit == null implies dated >= po_id.ordered"],
         "fields": {"iid": {"kind": "seq", "prefix": "I-"}, "po_id": {"kind": "ref", "table": "po"},
                    "dated": {"kind": "date_offset", "from": "po_id.req_id.raised", "min_days": 0, "max_days": 60}}}]})
    g = generate(spec, schema, seed=0, profile=Profile(near_duplicate=0, stale=0, conflict=0))
    assert integrity(spec, schema, g) == []
    assert any(r["limit"] is None for r in g.tables["req"])
    raised = {r["rid"]: r["raised"] for r in g.tables["req"]}
    po = {r["pid"]: r for r in g.tables["po"]}
    assert all(i["dated"] >= raised[po[i["po_id"]]["req_id"]] for i in g.tables["inv"])
    g.tables["inv"][0]["dated"] = "2020-01-01"
    assert any("po_id.req_id.raised" in p for p in integrity(spec, schema, g))
    bad = spec.model_copy(deep=True)
    bad.tables[2].rules = ["dated >= po_id.ordered.raised"]
    with pytest.raises(DataSpecError, match="ordered is not a ref field of po"):
        compile_spec(bad, schema)


def test_a_unique_template_numbers_its_overflow():
    schema = {"inv": {"key": "iid", "fields": {"iid": "string", "ref": "string"}}}
    spec = DataSpec.model_validate({"tables": [{"table": "inv", "rows": 30, "fields": {
        "iid": {"kind": "seq", "prefix": "I-"}, "ref": {"kind": "template", "template": "INV-{n}", "parts": {"n": ["1", "2", "3"]}, "unique": True}}}]})
    g = generate(spec, schema, seed=0, profile=Profile(near_duplicate=0, stale=0, conflict=0))
    refs = [r["ref"] for r in g.tables["inv"]]
    assert len(set(refs)) == 30 and integrity(spec, schema, g) == []


def test_a_two_way_link_written_as_two_refs_is_normalised_to_a_backref():
    # worlds-1: told "use a backref" three times, the writer wrote requisitions -> purchase_orders -> requisitions again
    schema = {"req": {"key": "rid", "fields": {"rid": "string", "po_id": "string?"}},
              "po": {"key": "pid", "fields": {"pid": "string", "req_id": "string"}}}
    raw = DataSpec.model_validate({"tables": [
        {"table": "req", "rows": 20, "fields": {"rid": {"kind": "seq", "prefix": "R-"}, "po_id": {"kind": "ref", "table": "po"}}},
        {"table": "po", "rows": 8, "fields": {"pid": {"kind": "seq", "prefix": "P-"}, "req_id": {"kind": "ref", "table": "req", "unique": True}}}]})
    with pytest.raises(DataSpecError, match="cycle"):
        compile_spec(raw, schema)
    spec, notes = normalize_spec(raw, schema)
    assert notes and spec.tables[0].fields["po_id"] == {"kind": "backref", "table": "po", "fk": "req_id"}
    g = generate(spec, schema, seed=0, profile=Profile(near_duplicate=0, stale=0, conflict=0))
    assert integrity(spec, schema, g) == [] and sum(1 for r in g.tables["req"] if r["po_id"]) == 8
    both = copy.deepcopy(schema)
    both["po"]["fields"]["req_id"] = "string?"  # both sides optional: no side is clearly the reverse link, so it stays refused
    assert normalize_spec(raw, both)[1] == []


def test_a_copy_rule_makes_its_field_a_lookup():
    # worlds-2b: the purchase-requisition writer wrote "requester_id.department_id == department_id" with department_id
    # sampled on its own; the integrity check refused most rows and said "generate it with lookup"
    schema = {"dept": {"key": "did", "fields": {"did": "string"}},
              "emp": {"key": "eid", "fields": {"eid": "string", "dept_id": "string"}},
              "req": {"key": "rid", "fields": {"rid": "string", "requester_id": "string", "department_id": "string", "cost_centre": "string"}}}
    raw = DataSpec.model_validate({"tables": [
        {"table": "dept", "rows": 300, "fields": {"did": {"kind": "seq", "prefix": "D-"}}},  # too many for rejection sampling
        {"table": "emp", "rows": 20, "fields": {"eid": {"kind": "seq", "prefix": "E-"}, "dept_id": {"kind": "ref", "table": "dept"}}},
        {"table": "req", "rows": 40, "fields": {"rid": {"kind": "seq", "prefix": "R-"}, "requester_id": {"kind": "ref", "table": "emp"},
                                                "department_id": {"kind": "ref", "table": "dept"},
                                                "cost_centre": {"kind": "calc", "expr": "department_id"}},
         "rules": ["requester_id.dept_id == department_id", "cost_centre == requester_id.dept_id"]}]})
    g = generate(raw, schema, seed=0, profile=Profile(near_duplicate=0, stale=0, conflict=0))
    assert any("generate it with lookup" in p for p in integrity(raw, schema, g))
    spec, notes = normalize_spec(raw, schema)
    req = spec.tables[2].fields
    assert req["department_id"] == {"kind": "lookup", "ref": "requester_id", "field": "dept_id"}
    assert req["cost_centre"] == {"kind": "calc", "expr": "department_id"}  # already derived: left to the writer
    assert len(notes) == 1 and "copy" in notes[0]
    for sd in range(3):
        g = generate(spec, schema, seed=sd, profile=Profile(near_duplicate=0, stale=0, conflict=0))
        assert integrity(spec, schema, g) == []
    two = raw.model_copy(deep=True)  # two copy rules naming different sources: ambiguous, left alone
    two.tables[2].rules.append("department_id == requester_id.eid")
    assert normalize_spec(two, schema)[0].tables[2].fields["department_id"] == {"kind": "ref", "table": "dept"}


def test_if_then_rules_and_conflicts_never_land_on_a_copied_master():
    # worlds-2: the insurance writer wrote 16 rules as "if A then B"; a conflict planted on a loss notice's insured name
    # (itself a copy of the policy's) broke the claim that copies it
    assert eval_expr("if status == 'merged' then target != null", {"status": "merged", "target": None}) is False
    assert eval_expr("if status == 'merged' then target != null", {"status": "open", "target": None}) is True
    assert eval_expr("if a > 1 then 'big' else 'small'", {"a": 0}) == "small"
    schema = {"pol": {"key": "pid", "fields": {"pid": "string", "name": "string"}},
              "notice": {"key": "nid", "fields": {"nid": "string", "pid": "string", "name": "string"}},
              "claim": {"key": "cid", "fields": {"cid": "string", "nid": "string", "name": "string"}}}
    spec = DataSpec.model_validate({"tables": [
        {"table": "pol", "rows": 30, "fields": {"pid": {"kind": "seq", "prefix": "P-"}, "name": {"kind": "template", "template": "{a} {b}",
                                                                                             "parts": {"a": ["Ann", "Bo", "Cy"], "b": ["Lee", "Ng", "Roy"]}}}},
        {"table": "notice", "rows": 40, "fields": {"nid": {"kind": "seq", "prefix": "N-"}, "pid": {"kind": "ref", "table": "pol"},
                                                  "name": {"kind": "lookup", "ref": "pid", "field": "name"}}},
        {"table": "claim", "rows": 40, "fields": {"cid": {"kind": "seq", "prefix": "C-"}, "nid": {"kind": "ref", "table": "notice"},
                                                 "name": {"kind": "lookup", "ref": "nid", "field": "name"}}}]})
    for sd in range(4):
        g = generate(spec, schema, seed=sd, profile=Profile(conflict=0.3, near_duplicate=0, stale=0))
        assert integrity(spec, schema, g) == []
        assert {n.table for n in g.noise if n.kind == "conflict"} == {"claim"}  # the notice's name is the claim's master


def test_a_planted_conflict_does_not_break_the_copy_rule_it_violates_by_design():
    # worlds-2b: purchase orders copy their requisition's department (a lookup, stated again as a rule); a conflict
    # planted on the copy broke the rule on every conflicted row, so two clean generators were refused
    schema = {"req": {"key": "rid", "fields": {"rid": "string", "dept": "string"}},
              "po": {"key": "pid", "fields": {"pid": "string", "req_id": "string", "dept": "string", "amount": "number"}}}
    spec = DataSpec.model_validate({"tables": [
        {"table": "req", "rows": 40, "fields": {"rid": {"kind": "seq", "prefix": "R-"}, "dept": {"kind": "pool", "values": ["Finance", "IT", "Operations"]}}},
        {"table": "po", "rows": 60, "fields": {"pid": {"kind": "seq", "prefix": "P-"}, "req_id": {"kind": "ref", "table": "req"},
                                               "dept": {"kind": "lookup", "ref": "req_id", "field": "dept"},
                                               "amount": {"kind": "number", "min": 10, "max": 100}},
         "rules": ["dept == req_id.dept", "amount > 0"]}]})
    g = generate(spec, schema, seed=0, profile=Profile(conflict=0.3, near_duplicate=0, stale=0))
    assert {n.table for n in g.noise if n.kind == "conflict"} == {"po"}
    assert integrity(spec, schema, g) == []
    bad = copy.deepcopy(g)  # a conflicted row still answers to the rules that do not read the conflicted field
    key = str(next(n.key for n in g.noise if n.kind == "conflict"))
    next(r for r in bad.tables["po"] if r["pid"] == key)["amount"] = -5
    assert any("amount > 0" in p for p in integrity(spec, schema, bad))


def test_a_rule_on_a_backref_is_left_to_integrity_not_to_rejection_sampling():
    # worlds-2b: "status == 'converted' implies po_id != null" with po_id a backref (null while the row is drawn) refused
    # every converted requisition, so the orders' ref where status == 'converted' found no row in every data attempt
    schema = {"req": {"key": "rid", "fields": {"rid": "string", "status": "string", "po_id": "string?"}},
              "po": {"key": "pid", "fields": {"pid": "string", "req_id": "string"}}}
    spec = DataSpec.model_validate({"tables": [
        {"table": "req", "rows": 30, "fields": {"rid": {"kind": "seq", "prefix": "R-"},
                                                "status": {"kind": "choice", "values": ["open", "converted"], "weights": [0.7, 0.3]},
                                                "po_id": {"kind": "backref", "table": "po", "fk": "req_id"}},
         "rules": ["status == 'converted' implies po_id != null"]},
        {"table": "po", "rows": 30, "fields": {"pid": {"kind": "seq", "prefix": "P-"},
                                               "req_id": {"kind": "ref", "table": "req", "where": {"status": "converted"}}}}]})
    tabs = {t.table: t for t in spec.tables}
    assert reads_aggregate(spec.tables[0].rules[0], spec.tables[0], tabs)
    assert not reads_aggregate("status != 'x'", spec.tables[0], tabs)
    assert reads_aggregate("req_id.po_id != null", spec.tables[1], tabs)  # a chain ending on a backref
    g = generate(spec, schema, seed=0, profile=Profile(near_duplicate=0, stale=0, conflict=0))
    conv = [r for r in g.tables["req"] if r["status"] == "converted"]
    assert conv  # drawn (it used to be redrawn away), then judged once the backrefs are filled
    held = [p for p in integrity(spec, schema, g) if "does not hold" in p]
    assert all("backref, sum or count" in p for p in held)
    assert {r["rid"] for r in conv if r["po_id"] is None} == {p.split("[")[1].split("]")[0] for p in held}


def test_json_text_for_an_array_field_is_the_array():
    # worlds-2b: line items (an array of objects) have no object generator; the writer drew them as JSON text from a
    # choice and every row was refused as "not a array"
    schema = {"po": {"key": "pid", "fields": {"pid": "string", "lines": "array", "meta": "object?", "note": "string"}}}
    spec = DataSpec.model_validate({"tables": [{"table": "po", "rows": 10, "fields": {
        "pid": {"kind": "seq", "prefix": "P-"},
        "lines": {"kind": "choice", "values": ['[{"sku": "A1", "qty": 2}]', '[{"sku": "B2", "qty": 1}, {"sku": "C3", "qty": 5}]']},
        "meta": {"kind": "const", "value": '{"terms": "net 30"}'}, "note": {"kind": "const", "value": "[1, 2]"}}}]})
    g = generate(spec, schema, seed=0, profile=Profile(near_duplicate=0, stale=0, conflict=0))
    assert integrity(spec, schema, g) == []
    r = g.tables["po"][0]
    assert isinstance(r["lines"], list) and r["lines"][0]["sku"] in ("A1", "B2") and r["meta"] == {"terms": "net 30"}
    assert r["note"] == "[1, 2]"  # a string field keeps its text


def test_a_rule_written_b_if_a_reads_as_a_implies_b():
    # worlds-2b: "approval_limit >= 0 if approval_limit != null" was refused ("expected 'else' after 'if' expression")
    assert eval_expr("limit >= 0 if limit != null", {"limit": -1}) is False
    assert eval_expr("limit >= 0 if limit != null", {"limit": None}) is True
    assert eval_expr("'big' if a > 1 else 'small'", {"a": 0}) == "small"  # a full conditional is unchanged
    with pytest.raises(DataSpecError):
        eval_expr("a >= if", {"a": 1})
