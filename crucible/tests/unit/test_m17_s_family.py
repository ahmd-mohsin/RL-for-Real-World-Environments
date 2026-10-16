"""M17 owner S: family acceptance (factory/family.py, 05 §8). The v1 gold-set gate is per class (36 truly-failing and 36
truly-passing measured items for a 10% Wilson bound); real solver outputs count only with a human label; a candidate
intent can never make a family acceptable."""

from __future__ import annotations

from typing import Any

from crucible.factory.family import GOLD_LARGE_SCALE, GOLD_V1, GoldRecord, _per_class_need, accept_family, gold_report
from crucible.factory.labels import LabelStore


def records(n_fail: int, n_pass: int, *, real: int = 0, wrong: int = 0) -> list[GoldRecord]:
    out = [GoldRecord(f"t{i % 3}", "mutant", {"edits": [{"m": i}]}, 0.0, {"c1": False}) for i in range(n_fail)]
    out += [GoldRecord(f"t{i % 3}", "alternative", {"edits": [{"a": i}]}, 0.0 if i < wrong else 1.0, {"c1": i >= wrong}) for i in range(n_pass)]
    out += [GoldRecord(f"t{i % 3}", "real_fail", {"edits": [{"r": i}]}, 0.0, {"c1": False}) for i in range(real)]
    return out


def test_per_class_need_matches_the_wilson_bound() -> None:
    assert _per_class_need(GOLD_V1) == 35  # zero errors: 3.84 / (n + 3.84) <= 0.10 (05 rounds this up to 36)
    assert _per_class_need(GOLD_LARGE_SCALE) == 73


def test_constructed_items_fill_both_classes_but_real_items_need_humans(tmp_path: Any) -> None:
    g = gold_report(records(40, 40, real=10))
    assert g["per_class"]["truly_failing"] == 40 and g["per_class"]["truly_passing"] == 40
    assert g["constructed_only"]["fp"]["k"] == 0 and g["constructed_only"]["fp"]["wilson"][1] <= 0.10
    assert not g["ok"] and any("human truth label" in r for r in g["full"]["reasons"])  # 10 real items, no human label
    store = LabelStore(tmp_path / "labels.jsonl")
    from crucible.factory.family import gold_items

    items, _ = gold_items(records(40, 40, real=10))
    for it in items:
        if it.source == "real_fail":
            store.add_label(item=it.id, item_sha256=it.sha256, labeller="human:ann", value="fail", reason="read the diff")
    g2 = gold_report(records(40, 40, real=10), store.labels())
    assert not any("human truth label" in r for r in g2["full"]["reasons"])
    assert g2["full"]["real_share"] < 0.5 and not g2["ok"]  # still under half real: the v1 rule holds


def test_too_few_passing_items_cannot_prove_the_false_fail_bound() -> None:
    g = gold_report(records(40, 20))
    assert any("false-fail" in r for r in g["full"]["reasons"])


def test_a_wrong_verdict_on_a_valid_alternative_is_a_false_fail() -> None:
    g = gold_report(records(40, 40, wrong=2))
    assert g["full"]["fn"]["k"] == 2 and g["full"]["errors"]["false_fail"]


def test_accept_family_reports_every_shortfall() -> None:
    cards = [{"id": f"t{i}", "intent": "intent-x", "prompt": f"request number {i} about thing {i * 7}", "family": "intent-x"} for i in range(5)]
    extra = {c["id"]: {"surface": {"repo": "a" if i < 4 else "b"}} for i, c in enumerate(cards)}
    res = accept_family(cards=cards, extra=extra, gold=records(40, 40), transfer={"ok": False, "reasons": ["no paired gain"]}, probes=[],
                        provisional_intent=True)
    r = " | ".join(res["reasons"])
    assert not res["ok"]
    assert "5 accepted tasks, need 20" in r and "2 surfaces, need 3" in r and "0 held-out probes" in r
    assert "transfer proxy: no paired gain" in r and "provisional_intent=True" in r
    assert "orthogonality" in r  # surface a holds 80% of the family
