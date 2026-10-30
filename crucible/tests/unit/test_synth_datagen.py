"""crucible.synth.datagen: the deterministic instance data generator, its noise and its integrity checks."""

from __future__ import annotations

import copy

import pytest

from crucible.synth.datagen import PROFILES, DataSpec, DataSpecError, Profile, compile_spec, eval_expr, generate, integrity, near_variant
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
    d = copy.deepcopy(DATAGEN)
    d["tables"][1]["stale"] = {"set": {"status": "gone"}}
    with pytest.raises(DataSpecError, match="not one of its choice values"):
        compile_spec(DataSpec.model_validate(d), s)


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
