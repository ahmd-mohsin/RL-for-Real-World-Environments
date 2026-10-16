"""M17 P: the append-only label and decision store: supersede chains, item-hash binding, torn writes, tampering, pairing,
kappa against hand-computed values, interval helpers."""

from __future__ import annotations

import json
import math
import multiprocessing as mp

import pytest

from crucible.factory import labels as LB
from crucible.factory.labels import (
    CorruptStore,
    DuplicateLabel,
    LabelStore,
    LabelStoreError,
    StaleItem,
    clopper_pearson,
    cohen_kappa,
    fleiss_kappa,
    item_sha256,
    wilson,
)

H1, H2 = item_sha256({"v": 1}), item_sha256({"v": 2})
A, B, C = "human:alice", "human:bob", "model:judge"


@pytest.fixture
def store(tmp_path):
    return LabelStore(tmp_path / "f" / "labels.jsonl")


def add(store, item="i1", who=A, value="pass", sha=H1, **kw):
    return store.add_label(item=item, item_sha256=sha, labeller=who, value=value, **kw)


def test_append_and_read_back(store):
    lab = add(store, reason="clean")
    assert store.get_label(lab.id) == lab and store.labels("i1") == [lab] and store.verify() == 1
    assert store.path.read_text().endswith("\n") and len(store.path.read_text().splitlines()) == 1
    assert LabelStore(store.path).labels() == [lab]  # a second process view


def test_there_is_no_edit_or_delete_api():
    assert not [m for m in dir(LabelStore) if any(w in m for w in ("edit", "update", "delete", "remove", "set_", "overwrite"))]


# --- supersede chains ----------------------------------------------------------------------------------------------------


def test_supersede_chain_and_effective_label(store):
    l1 = add(store, value="fail")
    l2 = add(store, value="pass", supersedes=l1.id)
    l3 = add(store, value="fail", supersedes=l2.id)
    assert [x.id for x in store.chain(l1.id)] == [l1.id, l2.id, l3.id] == [x.id for x in store.chain(l3.id)]
    assert store.labels("i1") == [l3] and store.labels("i1", include_superseded=True) == [l1, l2, l3]
    assert store.is_superseded(l1.id) and not store.is_superseded(l3.id)
    assert store.verify() == 3


@pytest.mark.parametrize("case", ["unknown", "other_labeller", "other_item", "fork", "self", "decision_id"])
def test_forged_supersedes_are_refused(store, case):
    mine = add(store, who=A, item="i1")
    other_item = add(store, who=A, item="i2")
    theirs = add(store, who=B, item="i1")
    newer = add(store, who=A, item="i3")
    add(store, who=A, item="i3", value="fail", supersedes=newer.id)
    dec = store.add_decision(subject="t1", actor=A, action="keep", reason="r", recipe_version="r1")
    target, who, item = {"unknown": ("lbl-does-not-exist", A, "i1"), "other_labeller": (theirs.id, A, "i1"),
                         "other_item": (other_item.id, A, "i1"), "fork": (newer.id, A, "i3"), "self": ("same", A, "i1"),
                         "decision_id": (dec.id, A, "i1")}[case]
    before = store.path.read_bytes()
    with pytest.raises((LabelStoreError, ValueError)):  # SupersedeError, or a validation error (a self supersede never reaches the store)
        store.add_label(item=item, item_sha256=H1, labeller=who, value="x", supersedes=target, id="same" if case == "self" else None)
    assert store.path.read_bytes() == before  # a refused append writes nothing
    assert mine.id and store.verify() == 6


def test_second_label_by_same_labeller_must_supersede(store):
    add(store)
    with pytest.raises(DuplicateLabel):
        add(store, value="fail")
    assert add(store, who=B)  # another labeller is fine


def test_duplicate_ids_refused_across_labels_and_decisions(store):
    add(store, id="x1")
    with pytest.raises(LabelStoreError, match="already exists"):
        add(store, who=B, id="x1")
    with pytest.raises(LabelStoreError, match="already exists"):
        store.add_decision(subject="s", actor=A, action="keep", reason="r", recipe_version="r1", id="x1")
    store.add_decision(subject="s", actor=A, action="keep", reason="r", recipe_version="r1", id="d1")
    with pytest.raises(LabelStoreError, match="already exists"):
        add(store, who=C, id="d1")


# --- item-hash binding ----------------------------------------------------------------------------------------------------


def test_label_on_stale_item_hash_is_refused(store):
    with pytest.raises(StaleItem):
        add(store, sha=H1, current_item_sha256=H2)
    assert not store.path.exists() or store.verify() == 0
    assert add(store, sha=H2, current_item_sha256=H2)


def test_changed_item_makes_labels_stale(store):
    la, lb = add(store, who=A), add(store, who=B)
    assert store.effective("i1", H1) == [la, lb] and store.stale("i1", H1) == []
    assert store.effective("i1", H2) == [] and store.stale("i1", H2) == [la, lb]
    assert store.needs_labels({"i1": H1}) == [] and store.needs_labels({"i1": H2}) == ["i1"]
    # the labeller re-judges the new content by superseding, never by a second independent label
    with pytest.raises(DuplicateLabel):
        add(store, who=A, sha=H2)
    add(store, who=A, sha=H2, supersedes=la.id)
    assert [x.labeller for x in store.effective("i1", H2)] == [A]
    assert store.needs_labels({"i1": H2}, labeller=A) == []
    assert store.needs_labels({"i1": H2}, labeller=B) == ["i1"]


def test_item_sha256_is_canonical():
    assert item_sha256({"a": 1, "b": [1, 2]}) == item_sha256({"b": [1, 2], "a": 1}) != item_sha256({"a": 1, "b": [2, 1]})


# --- crash safety and tampering ---------------------------------------------------------------------------------------------


def test_torn_last_line_is_ignored_then_truncated_by_the_next_append(store):
    l1 = add(store)
    good = store.path.read_bytes()
    full = good + store.path.read_bytes().splitlines()[0]  # a second copy of a full record, then cut it
    store.path.write_bytes(good + full[len(good):][:40])  # a torn write: no trailing newline
    fresh = LabelStore(store.path)
    assert fresh.labels() == [l1] and fresh.torn_bytes == 40 and fresh.verify() == 1
    l2 = add(fresh, who=B)  # an append repairs the file
    assert fresh.torn_bytes == 0 and fresh.labels("i1") == [l1, l2] and fresh.verify() == 2
    assert store.path.read_bytes().startswith(good)
    assert all(line.endswith(b"}") for line in store.path.read_bytes().splitlines())


def test_complete_json_without_newline_is_still_unacknowledged(store):
    add(store)
    store.path.write_bytes(store.path.read_bytes().rstrip(b"\n"))  # the newline is what commits a record
    fresh = LabelStore(store.path)
    assert fresh.labels() == [] and fresh.torn_bytes > 0
    assert add(fresh).labeller == A and fresh.verify() == 1


def test_torn_line_in_the_middle_is_corruption(store):
    add(store)
    add(store, who=B)
    lines = store.path.read_bytes().split(b"\n")
    store.path.write_bytes(lines[0][:30] + b"\n" + lines[1] + b"\n")
    with pytest.raises(CorruptStore, match="line 1"):
        LabelStore(store.path).labels()
    with pytest.raises(CorruptStore):
        add(LabelStore(store.path), who=C)  # appends refuse too


def test_edited_record_breaks_the_chain(store):
    add(store, value="fail")
    add(store, who=B, value="fail")
    row = json.loads(store.path.read_text().splitlines()[0])
    row["rec"]["value"] = "pass"  # rewrite history
    lines = store.path.read_text().splitlines()
    lines[0] = json.dumps(row)
    store.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(CorruptStore, match="chain"):
        LabelStore(store.path).labels()


def test_removed_or_reordered_line_breaks_the_chain(store):
    for who in (A, B, C):
        add(store, who=who)
    lines = store.path.read_text().splitlines()
    store.path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(CorruptStore):
        LabelStore(store.path).verify()
    store.path.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n")
    with pytest.raises(CorruptStore):
        LabelStore(store.path).verify()


def test_hand_appended_forged_supersede_is_corruption(store):
    mine = add(store, who=A)
    theirs = add(store, who=B)
    # an attacker appends a correctly chained line whose supersede crosses labellers
    s = LabelStore(store.path)
    st = s._state()
    rec = {"id": "forged", "item": "i1", "item_sha256": H1, "labeller": A, "value": "pass", "reason": "", "at": "2026-01-01T00:00:00Z", "supersedes": theirs.id}
    line = {"seq": st.seq + 1, "prev": st.head, "kind": "label", "rec": rec, "h": LB._line_hash(st.head, st.seq + 1, "label", rec)}
    store.path.write_text(store.path.read_text() + json.dumps(line) + "\n")
    with pytest.raises(CorruptStore, match="invalid supersede"):
        LabelStore(store.path).labels()
    assert mine.id


def test_unknown_kind_and_garbage_are_corruption(store):
    add(store)
    store.path.write_text(store.path.read_text() + "not json\n")
    with pytest.raises(CorruptStore, match="line 2"):
        LabelStore(store.path).labels()


def _worker(path, who, n):
    s = LabelStore(path)
    for i in range(n):
        s.add_label(item=f"{who}-{i}", item_sha256=H1, labeller=who, value="pass")


def test_concurrent_appenders_keep_one_valid_chain(tmp_path):
    path = tmp_path / "labels.jsonl"
    procs = [mp.get_context("spawn").Process(target=_worker, args=(path, f"human:p{k}", 15)) for k in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    assert LabelStore(path).verify() == 60


# --- pairing and agreement ------------------------------------------------------------------------------------------------


def test_needs_labels_counts_distinct_fresh_labellers(store):
    items = {"i1": H1, "i2": H1, "i3": H1}
    add(store, item="i1", who=A)
    add(store, item="i1", who=B)
    add(store, item="i2", who=A)
    assert store.needs_labels(items) == ["i2", "i3"]
    assert store.needs_labels(items, raters=3) == ["i1", "i2", "i3"]
    assert store.needs_labels(items, labeller=A) == ["i3"]
    assert store.needs_labels(items, labeller=C) == ["i2", "i3"]  # i1 already has its two raters


def test_agreement_cohen_over_fresh_labels_only(store):
    # 50 items, the textbook table: both yes 20, A yes/B no 5, A no/B yes 10, both no 15 -> po 0.70, pe 0.50, kappa 0.40
    items = {f"i{n}": H1 for n in range(50)}
    plan = [("pass", "pass")] * 20 + [("pass", "fail")] * 5 + [("fail", "pass")] * 10 + [("fail", "fail")] * 15
    for n, (a, b) in enumerate(plan):
        add(store, item=f"i{n}", who=A, value=a)
        add(store, item=f"i{n}", who=B, value=b)
    ag = store.agreement(items)
    assert ag.method == "cohen" and ag.n_items == 50 and ag.raters == (A, B)
    assert ag.observed == pytest.approx(0.7) and ag.kappa == pytest.approx(0.4)
    # change five items: their labels go stale and drop out, so the number is over what is still bound
    changed = {k: (H2 if int(k[1:]) < 5 else H1) for k in items}
    assert store.agreement(changed).n_items == 45


def test_agreement_fleiss_for_three_raters_and_undefined_cases(store):
    items = {f"i{n}": H1 for n in range(4)}
    votes = [("a", "a", "a"), ("a", "a", "b"), ("b", "b", "b"), ("a", "b", "b")]
    for n, row in enumerate(votes):
        for who, v in zip((A, B, C), row):
            add(store, item=f"i{n}", who=who, value=v)
    ag = store.agreement(items)
    assert ag.method == "fleiss" and ag.raters == tuple(sorted((A, B, C))) and ag.n_items == 4
    assert ag.kappa == pytest.approx(fleiss_kappa([[3, 0], [2, 1], [0, 3], [1, 2]]))
    assert store.agreement(items, raters=[A]).kappa is None and store.agreement(items, raters=[A]).method == "none"
    assert store.agreement({}, raters=[A, B]).kappa is None
    assert store.agreement({"nope": H1}).kappa is None


def test_cohen_kappa_hand_computed():
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa(list("yyyy"), list("yyyy")) is None  # chance agreement 1: undefined
    assert cohen_kappa(list("yn"), list("ny")) == pytest.approx(-1.0)  # po 0, pe 0.5
    assert cohen_kappa(list("yynn"), list("yynn")) == pytest.approx(1.0)
    assert cohen_kappa([], []) is None
    with pytest.raises(ValueError):
        cohen_kappa(["a"], ["a", "b"])
    assert cohen_kappa(list("xxyyzz"), list("xxyyzz")) == pytest.approx(1.0)


def test_cohen_three_category_by_hand():
    a, b = list("xxyyzz"), list("xyyzzz")  # agree at positions 0, 2, 4, 5
    # po = 4/6; marg a: x2 y2 z2; marg b: x1 y2 z3; pe = (2*1 + 2*2 + 2*3)/36 = 12/36 = 1/3; kappa = (2/3 - 1/3) / (2/3) = 0.5
    assert cohen_kappa(a, b) == pytest.approx(0.5)


def test_fleiss_kappa_wikipedia_example():
    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
             [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    # Pbar = 3.780 / 10 = 0.378; Pe = 0.2128; kappa = (0.378 - 0.2128) / (1 - 0.2128) = 0.2099
    assert fleiss_kappa(table) == pytest.approx(0.2099, abs=5e-5)
    assert fleiss_kappa([[3, 0], [0, 3]]) == pytest.approx(1.0)
    assert fleiss_kappa([[3, 0], [3, 0]]) is None
    assert fleiss_kappa([]) is None
    for bad in ([[2, 1], [1, 1]], [[1]], [[3, -1], [1, 1]]):
        with pytest.raises(ValueError):
            fleiss_kappa(bad)


# --- intervals --------------------------------------------------------------------------------------------------------------


def test_wilson_hand_values():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(3.8416 / 13.8416, abs=1e-4)  # z^2 / (n + z^2)
    lo, hi = wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-4) and lo + hi == pytest.approx(1.0)
    assert wilson(0, 0) == (0.0, 1.0)
    assert wilson(40, 40)[1] == 1.0 and wilson(40, 40)[0] == pytest.approx(40 / (40 + 3.8416), abs=1e-4)
    lo90, hi90 = wilson(5, 10, 0.90)
    assert lo90 > wilson(5, 10)[0] and hi90 < wilson(5, 10)[1]
    for bad in ((-1, 5), (6, 5)):
        with pytest.raises(ValueError):
            wilson(*bad)


def test_clopper_pearson_hand_values():
    assert clopper_pearson(0, 10)[0] == 0.0 and clopper_pearson(0, 10)[1] == pytest.approx(1 - 0.025 ** 0.1, abs=1e-9)
    assert clopper_pearson(10, 10)[1] == 1.0 and clopper_pearson(10, 10)[0] == pytest.approx(0.025 ** 0.1, abs=1e-9)
    assert clopper_pearson(5, 10) == pytest.approx((0.1871, 0.8129), abs=5e-5)
    assert clopper_pearson(0, 0) == (0.0, 1.0)
    # exact is wider than Wilson at the edges, and both bracket the point estimate
    for k, n in [(0, 40), (1, 40), (3, 40), (39, 40)]:
        lo, hi = clopper_pearson(k, n)
        assert lo <= k / n <= hi and hi - lo >= wilson(k, n)[1] - wilson(k, n)[0] - 1e-12
    # the defining property: the upper tail at the bounds is alpha/2
    lo, hi = clopper_pearson(7, 30)
    def tail(p, k):
        return sum(math.comb(30, j) * p ** j * (1 - p) ** (30 - j) for j in range(k + 1))

    assert tail(hi, 7) == pytest.approx(0.025, abs=1e-9) and 1 - tail(lo, 6) == pytest.approx(0.025, abs=1e-9)


# --- decisions --------------------------------------------------------------------------------------------------------------


def test_decision_queries_by_criterion_and_recipe_version(store):
    d1 = store.add_decision(subject="t1", actor=A, action="drop", reason="ambiguous prompt", criteria=["ambiguity"], recipe_version="r1")
    d2 = store.add_decision(subject="t2", actor=B, action="revise", reason="unstated format", criteria=["ambiguity", "format"], recipe_version="r2")
    d3 = store.add_decision(subject="t1", actor=C, action="keep", reason="fixed", recipe_version="r2")
    assert store.decisions() == [d1, d2, d3]
    assert store.decisions(criterion="ambiguity") == [d1, d2]
    assert store.decisions(criterion="format") == [d2]
    assert store.decisions(recipe_version="r2") == [d2, d3]
    assert store.decisions(criterion="ambiguity", recipe_version="r2") == [d2]
    assert store.decisions(subject="t1", action="keep") == [d3] and store.decisions(actor=B) == [d2]
    assert store.decisions(criterion="missing") == []
    assert store.latest_decision("t1") == d3 and store.latest_decision("zzz") is None
    assert store.verify() == 3


def test_labels_and_decisions_share_one_chain(store):
    add(store)
    store.add_decision(subject="t", actor=A, action="approve", reason="ok", recipe_version="r1")
    add(store, who=B)
    assert store.verify() == 3 and [json.loads(x)["kind"] for x in store.path.read_text().splitlines()] == ["label", "decision", "label"]


def test_bad_label_input_is_refused_before_writing(store):
    for kw in ({"who": "alice"}, {"value": ""}, {"sha": "zz"}, {"item": "a b"}):
        with pytest.raises(ValueError):
            add(store, **kw)
    assert not store.path.exists()
