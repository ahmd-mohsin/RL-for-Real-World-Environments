"""crucible.synth.canon: the canonical state, the exact diff, the normalizers and the canonicalizers (part2-paper-core §3.2)."""

from __future__ import annotations

import copy
import os
import random
import sqlite3

import pytest

from crucible.synth.canon import (
    MISSING,
    Address,
    CanonState,
    Change,
    FieldScope,
    FloatRound,
    IdCanon,
    Normalizers,
    TextNorm,
    canon_fs,
    canon_gym,
    canon_json,
    canon_mail,
    canon_sql,
    canon_tables,
    diff,
    same_diff,
    same_value,
)


def _rand_state(rng: random.Random) -> dict:
    vals = [None, True, False, 0, 1, 2.5, -3, "a", "B b", "", [1, 2], {"k": 1}]
    out: dict = {}
    for app in ("x", "y"):
        for coll in ("c1", "c2"):
            rows = {}
            for k in rng.sample(range(8), rng.randint(0, 6)):
                rows[str(k)] = {f: rng.choice(vals) for f in rng.sample(["f", "g", "h", "id"], rng.randint(1, 4))}
            out.setdefault(app, {})[coll] = rows
    return out


def _apply(s: CanonState, changes: list[Change]) -> CanonState:
    apps = copy.deepcopy(s.apps)
    for c in changes:
        a = c.addr
        rows = apps.setdefault(a.app, {}).setdefault(a.coll, {})
        if c.kind == "insert":
            rows[a.key] = dict(c.new)
        elif c.kind == "delete":
            del rows[a.key]
        else:
            rows[a.key][a.field] = c.new
    return CanonState.build(apps)


def _strip_none_fields(s: CanonState, ref: CanonState) -> CanonState:
    """An update to a field absent on one side is recorded with None: compare on the fields the reference holds."""
    apps = {a: {c: {k: {f: v for f, v in r.items() if ref.entity(a, c, k) is None or f in ref.entity(a, c, k)} for k, r in rows.items()}
                for c, rows in colls.items()} for a, colls in s.apps.items()}
    return CanonState.build(apps)


def test_diff_is_exact_and_deterministic_on_random_states():
    """Δ(s, s) = ∅; applying Δ(s, s2) to s gives s2; the same inputs give the same sorted diff."""
    rng = random.Random(7)
    for _ in range(200):
        s, s2 = CanonState.build(_rand_state(rng)), CanonState.build(_rand_state(rng))
        assert diff(s, s) == []
        d = diff(s, s2)
        assert d == sorted(d, key=Change.sort_key)
        assert d == diff(CanonState.model_validate_json(s.model_dump_json()), CanonState.model_validate_json(s2.model_dump_json()))
        rebuilt = _strip_none_fields(_apply(s, d), s2)
        for app, coll, key, row in s2.entities():
            assert rebuilt.entity(app, coll, key) is not None
            assert same_value(rebuilt.entity(app, coll, key), row)
        assert {(a, c, k) for a, c, k, _ in rebuilt.entities()} == {(a, c, k) for a, c, k, _ in s2.entities()}


def test_update_is_per_field_and_insert_delete_are_whole_entities():
    s = CanonState.build({"ap": {"inv": {"1": {"status": "pending", "reason": None, "amt": 5}, "2": {"status": "x"}}}})
    s2 = CanonState.build({"ap": {"inv": {"1": {"status": "on_hold", "reason": "pv", "amt": 5}, "3": {"status": "y"}}}})
    d = diff(s, s2)
    assert [(str(c.addr), c.kind) for c in d] == [("ap.inv[1].reason", "update"), ("ap.inv[1].status", "update"), ("ap.inv[2]", "delete"), ("ap.inv[3]", "insert")]
    assert d[1].old == "pending" and d[1].new == "on_hold"
    assert d[3].new == {"status": "y"} and d[2].old == {"status": "x"}


def test_value_equality_is_type_aware():
    assert same_value(1, 1.0) and not same_value(True, 1) and not same_value(0, False) and not same_value("1", 1)
    assert same_value([1, {"a": 2.0}], [1.0, {"a": 2}]) and not same_value([1, 2], [2, 1])
    s = CanonState.build({"a": {"c": {"k": {"f": 1, "g": True}}}})
    assert diff(s, CanonState.build({"a": {"c": {"k": {"f": 1.0, "g": True}}}})) == []
    assert [c.addr.field for c in diff(s, CanonState.build({"a": {"c": {"k": {"f": 1, "g": 1}}}}))] == ["g"]


def test_ignore_fields_float_round_text_norm_and_order_insensitive():
    s = CanonState.build({"a": {"c": {"k": {"at": "t1", "x": 1.0000001, "name": "  Ann  Lee", "tags": ["b", "a"]}}}})
    s2 = CanonState.build({"a": {"c": {"k": {"at": "t2", "x": 1.0000002, "name": "ann lee", "tags": ["a", "b"]}}}})
    assert len(diff(s, s2)) == 4
    n = Normalizers(ignore_fields=[FieldScope(field="at")], float_round=[FloatRound(digits=4)], text_norm=[TextNorm(field="name")],
                    order_insensitive=[FieldScope(field="tags")])
    assert diff(s, s2, n) == []
    assert diff(s, s2, Normalizers(text_norm=[TextNorm(field="name", fold_case=False)], ignore_fields=[FieldScope(field="*")]))  == []
    # scope is respected: a normalizer on another collection changes nothing
    assert len(diff(s, s2, Normalizers(ignore_fields=[FieldScope(coll="other", field="at")]))) == 4


def test_normalizers_are_idempotent():
    rng = random.Random(3)
    n = Normalizers(ignore_fields=[FieldScope(field="h")], float_round=[FloatRound(digits=1)], text_norm=[TextNorm()],
                    order_insensitive=[FieldScope()])
    for _ in range(100):
        s = CanonState.build(_rand_state(rng))
        once = n.apply(s)
        assert once.fingerprint() == n.apply(once).fingerprint()


def test_id_canon_keys_inserts_by_content():
    base = {"ap": {"log": {"1": {"id": 1, "act": "x"}}, "inv": {"7": {"s": "p"}}}}
    s0 = CanonState.build(base)
    run_a = copy.deepcopy(base)
    run_a["ap"]["log"]["2"] = {"id": 2, "act": "hold"}
    run_b = copy.deepcopy(base)
    run_b["ap"]["log"]["5"] = {"id": 5, "act": "hold"}  # another autoincrement value, same effect
    n = Normalizers(id_canon=[IdCanon(app="ap", coll="log")])
    da, db = diff(s0, CanonState.build(run_a), n), diff(s0, CanonState.build(run_b), n)
    assert same_diff(da, db) and da[0].addr.key.startswith("h:") and da[0].new == {"act": "hold"}
    assert not same_diff(diff(s0, CanonState.build(run_a)), diff(s0, CanonState.build(run_b)))
    # two identical inserts are two changes (a duplicate is counted, never merged)
    run_c = copy.deepcopy(run_a)
    run_c["ap"]["log"]["3"] = {"id": 3, "act": "hold"}
    dc = diff(s0, CanonState.build(run_c), n)
    assert len(dc) == 2 and dc[0].addr.key != dc[1].addr.key and dc[1].addr.key.endswith("~1")
    # a pre-existing row keeps its key (deleted, updated rows are not re-keyed)
    gone = copy.deepcopy(base)
    del gone["ap"]["log"]["1"]
    assert [(c.kind, c.addr.key) for c in diff(s0, CanonState.build(gone), n)] == [("delete", "1")]


def test_canon_state_api_and_json_roundtrip():
    s = CanonState.build({"a": {"c": {"k": {"f": 1}}}}, unobserved=["z"])
    assert s.get(Address(app="a", coll="c", key="k", field="f")) == 1
    assert s.get(Address(app="a", coll="c", key="k", field="g")) is MISSING
    assert s.get(Address(app="a", coll="c", key="k")) == {"f": 1}
    assert CanonState.model_validate_json(s.model_dump_json()).fingerprint() == s.fingerprint()
    assert copy.deepcopy(MISSING) is MISSING


def test_canon_sql_primary_key_natural_key_and_row_hash(tmp_path):
    db = tmp_path / "x.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE inv (id INTEGER PRIMARY KEY, status TEXT, amount REAL, raw BLOB);
        CREATE TABLE line (inv INTEGER, pos INTEGER, sku TEXT, PRIMARY KEY (inv, pos));
        CREATE TABLE vendor (code TEXT, name TEXT);
        CREATE TABLE log (msg TEXT);
        INSERT INTO inv VALUES (42, 'pending', 10.5, x'0a0b'), (7, 'approved', 3.0, NULL);
        INSERT INTO line VALUES (42, 1, 'A'), (42, 2, 'B');
        INSERT INTO vendor VALUES ('V1', 'Acme'), ('V2', 'Bore');
        INSERT INTO log VALUES ('x'), ('x'), ('y');
    """)
    con.commit()
    s = canon_sql("erp", db, natural_keys={"vendor": ["code"]})
    assert s.entity("erp", "inv", "42") == {"id": 42, "status": "pending", "amount": 10.5, "raw": "b:0a0b"}
    assert sorted(s.collection("erp", "line")) == ["[42,1]", "[42,2]"]
    assert sorted(s.collection("erp", "vendor")) == ["V1", "V2"]
    logs = sorted(s.collection("erp", "log"))
    assert len(logs) == 3 and all(k.startswith("#") for k in logs) and sum(k.endswith("~1") for k in logs) == 1
    # row order and storage layout never matter; a value change is one per-field update
    con.execute("UPDATE inv SET status = 'on_hold' WHERE id = 42")
    con.commit()
    d = diff(s, canon_sql("erp", con, natural_keys={"vendor": ["code"]}))
    assert [(str(c.addr), c.new) for c in d] == [("erp.inv[42].status", "on_hold")]
    assert canon_sql("erp", db, include=["inv"]).apps["erp"].keys() == {"inv"}
    con.close()


def test_canon_fs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("print(1)\n")
    (tmp_path / "big.txt").write_text("x" * 5000)
    (tmp_path / "bin").write_bytes(b"\xff\xfe\x00")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref")
    os.symlink("src/a.py", tmp_path / "link")
    os.chmod(tmp_path / "src" / "a.py", 0o755)
    s = canon_fs("ws", tmp_path)
    files = s.collection("ws", "files")
    assert set(files) == {"src/a.py", "big.txt", "bin", "link"}
    assert files["src/a.py"]["text"] == "print(1)\n" and files["src/a.py"]["mode"] == "755"
    assert "text" not in files["big.txt"] and "text" not in files["bin"] and files["link"] == {"link": "src/a.py", "mode": files["link"]["mode"]}
    (tmp_path / "src" / "a.py").write_text("print(2)\n")
    assert {c.addr.field for c in diff(s, canon_fs("ws", tmp_path))} == {"sha256", "text"}


def test_canon_json_pointer_flatten():
    doc = {"tickets": [{"id": 7, "status": "open"}, {"id": 9, "status": "done"}], "tags": ["b", "a"], "meta": {"v": 1, "x/y": 2}}
    s = canon_json("app", "doc", doc, list_keys={"/tickets": "id"})
    rows = s.collection("app", "doc")
    assert rows["/tickets"] == {"/[id=7]/id": 7, "/[id=7]/status": "open", "/[id=9]/id": 9, "/[id=9]/status": "done"}
    assert rows["/tags"] == {"": ["b", "a"]} and rows["/meta"] == {"/v": 1, "/x~1y": 2}
    # keyed lists: reordering is no change; positional lists: it is
    swapped = {**doc, "tickets": list(reversed(doc["tickets"]))}
    assert diff(s, canon_json("app", "doc", swapped, list_keys={"/tickets": "id"})) == []
    assert diff(canon_json("app", "doc", doc), canon_json("app", "doc", swapped)) != []
    deep = canon_json("app", "doc", doc, depth=2, list_keys={"/tickets": "id"})
    assert deep.entity("app", "doc", "/tickets/[id=7]") == {"/id": 7, "/status": "open"}


def test_canon_mail_and_gym():
    s = canon_mail("mail", [{"id": "m2", "to": ["a@x"], "body": "hi"}, {"id": "m1", "to": ["b@x"], "body": "yo"}, {"to": ["c"], "body": "no id"}])
    rows = s.collection("mail", "messages")
    assert rows["m1"] == {"to": ["b@x"], "body": "yo"} and len(rows) == 3 and any(k.startswith("#") for k in rows)
    g = canon_gym("env", [0.5, -1.0])
    assert g.entity("env", "obs", "1") == {"value": -1.0}
    with pytest.raises(ValueError):
        canon_gym("env", [float("nan")])


def test_canon_tables_positional_collections():
    t = {"q": {"a": {"s": 1}, "b": {"s": 2}}, "k": {5: {"s": 3}}}
    s = canon_tables("app", t, positional=["q"])
    assert sorted(s.collection("app", "q")) == ["0", "1"] and list(s.collection("app", "k")) == ["5"]
