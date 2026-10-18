"""P2 gold set (``factory.goldset``): executed labels decide real items (a human label still overrides them, a conflict
with a construction is reported), the zero-error sample sizes (36 for 10%, 73 for 5%), per-class bounds, pair scoring
at a finer unit, the 10% human spot-check queue, and the family's verifier card."""

from __future__ import annotations

from crucible.factory.cards import Label
from crucible.factory.floors import Frac
from crucible.factory.goldset import (
    EXECUTED,
    GoldConfig,
    GoldItem,
    class_bounds,
    executed_label,
    pair_fracs,
    score,
    spot_check_queue,
    truths,
    verifier_card,
    zero_error_n,
)
from crucible.factory.labels import item_sha256

AT = "2026-10-18T00:00:00Z"


def item(i: str, src: str) -> GoldItem:
    return GoldItem(i, src, {"x": i}, item_sha256({"x": i}), family="window")  # type: ignore[arg-type]


def test_zero_error_targets() -> None:
    assert zero_error_n(0.10) == 35 and zero_error_n(0.05) == 73  # the spec rounds 35 up to 36
    b = class_bounds(Frac(0, 73), Frac(0, 36))
    assert b["truly_failing"]["targets"]["<= 5%"]["met"] and b["truly_passing"]["targets"]["<= 10%"]["met"]
    assert not b["truly_passing"]["targets"]["<= 5%"]["met"] and b["truly_passing"]["wilson_upper"] <= 0.10
    assert not class_bounds(Frac(1, 200), Frac(0, 0))["truly_failing"]["targets"]["<= 10%"]["met"]


def test_executed_labels_decide_real_items_and_humans_override() -> None:
    its = [item("r1", "real_fail"), item("r2", "real_fail"), item("m1", "mutant"), item("o1", "oracle"), item("f1", "floor")]
    labels = [executed_label(its[0], False, reason="strict suite fails", at=AT), executed_label(its[1], True, reason="strict suite passes", at=AT),
              executed_label(its[2], True, reason="an equivalent mutant", at=AT)]
    t, info = truths(its, labels)
    assert t == {"r1": False, "r2": True, "m1": True, "o1": True, "f1": False}
    assert info["proof_conflict"] == ["m1"] and info["unlabelled_real"] == []
    human = Label(id="h1", item="r2", item_sha256=its[1].sha256, labeller="human:ann", value="fail", at=AT)
    t2, _ = truths(its, [*labels, human])
    assert t2["r2"] is False
    assert labels[0].labeller == EXECUTED
    rep = score(its, {"r1": 0.0, "r2": 0.0, "m1": 0.0, "o1": 1.0, "f1": 0.0}, labels, config=GoldConfig(min_items=1))
    assert rep.errors == {"false_pass": [], "false_fail": ["r2", "m1"]} and rep.counts["oracle"] == 1 and rep.counts["floor"] == 1


def test_pair_fracs_queue_and_card() -> None:
    fp, fn = pair_fracs([(True, True), (True, False), (False, False), (False, None), (None, True), (False, True)])
    assert (fp.k, fp.n, fn.k, fn.n) == (1, 2, 1, 2)
    its = [item(f"r{i}", "real_fail") for i in range(30)] + [item(f"m{i}", "mutant") for i in range(8)] + [item("o", "oracle")]
    truth = {it.id: it.source == "oracle" for it in its}
    q = spot_check_queue(its, truth, share=0.10, seed="s")
    assert len(q) == 4 and {r["source"] for r in q} == {"real_fail", "mutant", "oracle"} and all(r["task"] == "confirm" for r in q)
    assert q == spot_check_queue(its, truth, share=0.10, seed="s")  # deterministic
    q2 = spot_check_queue([*its, item("d1", "real_fail")], {**truth, "d1": None}, share=0.10, seed="s")
    assert q2[-1]["item"] == "d1" and q2[-1]["task"] == "decide" and q2[-1]["label"] is None
    labels = [executed_label(it, False, reason="x", at=AT) for it in its if it.source == "real_fail"]
    rep = score(its, {it.id: (1.0 if it.source == "oracle" else 0.0) for it in its}, labels, config=GoldConfig(min_items=1))
    card = verifier_card(rep, verifier_version="v1", family="window", mutant_kill=(8, 8), alternatives_pass=(2, 2), differential=(1, 50),
                         units={"milestone": (Frac(0, 80), Frac(0, 40))})
    assert card["card"]["fp"] == [0, 38] and card["card"]["fn"] == [0, 1] and card["card"]["mutant_kill"] == [8, 8]
    assert card["bounds"]["truly_failing"]["targets"]["<= 10%"]["met"] and card["units"]["milestone"]["truly_failing"]["targets"]["<= 5%"]["met"]
