"""M17 owner R: family-level evidence. Orthogonality (four layers, the intent+cause duplicate rule, skew), the CPU
transfer proxy (paired gain, surface overlap, placebo) and the VerifierGoldSet (FP/FN with Wilson bounds, κ against human
labels from the label store). Adversarial cases: a verifier that passes everything; a gain that lives only on probes
sharing the family's surface; a goldset at the v1 minimum size that cannot reach the v1 bound."""

from __future__ import annotations

import pytest

from crucible.factory.goldset import GoldConfig, GoldItem, run_goldset, score
from crucible.factory.labels import LabelStore, item_sha256
from crucible.factory.ortho import OrthoConfig, TaskProfile, orthogonality, pair_scores, pearson
from crucible.factory.transfer import ProbeTask, TransferConfig, best_of_n, disjoint, exemplar_block, overlap, transfer_proxy

# --- orthogonality --------------------------------------------------------------------------------------------------


def prof(id: str, intent: str, causes: dict, *, domain: str, prompt: str = "", **kw) -> TaskProfile:
    return TaskProfile(id=id, intent=intent, prompt=prompt or f"Task {id}: reconcile {domain} records for case {id} and report the result.",
                       surface={"domain": domain, "world": f"w_{domain}"}, causes=causes, start_fp=f"s{id}", end_fp=f"e{id}", **kw)


def test_duplicate_rule_is_same_intent_and_same_failure_cause():
    ps = [prof("a", "restated_invoice", {"skill_gap": 5, "budget": 1}, domain="finance"),
          prof("b", "restated_invoice", {"skill_gap": 3}, domain="support"),  # same intent, same cause: duplicate
          prof("c", "restated_invoice", {"unstated_format": 4, "skill_gap": 1}, domain="legal"),  # same intent, other cause: kept
          prof("d", "aliased_vendor", {"skill_gap": 4}, domain="ops"),  # other intent: kept
          prof("e", "aliased_vendor", {}, domain="hr")]  # no failures yet: no cause to compare, kept
    r = orthogonality(ps)
    assert r["kept"] == ["a", "c", "d", "e"]
    assert r["dropped"]["b"]["dup_of"] == "a" and "same intent" in r["dropped"]["b"]["rule"]
    assert r["ok"] and r["skew"]["axes"]["domain"]["share"] == 0.25


def test_forge_dedup_layers_still_drop_near_copies():
    a = prof("a", "i1", {"skill_gap": 1}, domain="finance")
    same_state = prof("b", "i2", {"budget": 1}, domain="ops")
    same_state.start_fp, same_state.end_fp = a.start_fp, a.end_fp
    r = orthogonality([a, same_state])
    assert r["dropped"]["b"]["rule"] == "forge dedup L2_state"

    def embed(texts):
        return [[1.0, 0.0] if "refund" in t else [0.0, 1.0] for t in texts]

    x = prof("x", "i1", {"skill_gap": 1}, domain="a", prompt="Issue the refund for the damaged order in full today.")
    y = prof("y", "i1", {"budget": 1}, domain="b", prompt="Please refund the broken order completely, approved by the lead.")
    r2 = orthogonality([x, y], embed=embed)
    assert "semantic" in r2["dropped"]["y"]["rule"]


def test_skew_and_behavioural_layers():
    ps = [prof(str(i), f"i{i}", {"skill_gap": 1}, domain="finance" if i < 3 else f"d{i}") for i in range(5)]
    r = orthogonality(ps)
    assert not r["ok"] and "domain='finance' is 3/5" in r["reasons"][0]
    panel = {f"p{j}": j % 2 == 0 for j in range(8)}
    a = prof("a", "i1", {"skill_gap": 2}, domain="x", passes=panel, checks=["c1", "c2"], footprint=["t:1"])
    b = prof("b", "i2", {"skill_gap": 2}, domain="y", passes=dict(panel), checks=["c2", "c3"], footprint=["t:1"])
    s = pair_scores(a, b)
    assert s["pass_corr"] == 1.0 and s["checks"] == pytest.approx(1 / 3, abs=1e-4) and s["footprint"] == 1.0 and s["cause_cos"] == 1.0
    assert orthogonality([a, b], config=OrthoConfig(top_surface_max=1.0))["behaviourally_redundant"] == [["a", "b"]]
    assert pearson([1, 1, 1], [0, 1, 0]) is None  # constant vector: undefined, never a correlation of 0


# --- transfer proxy -------------------------------------------------------------------------------------------------


FAMILY = [ProbeTask(f"f{i}", f"Reconcile the restated invoice INV-{i} against the purchase order and flag the delta.", {"domain": "finance"},
                    world="ap_world", template="tpl_ap", verifier="ver_ap") for i in range(5)]


def probes(n: int = 12) -> list[ProbeTask]:
    out = []
    for i in range(n):
        near = i % 2 == 0  # half the probes copy the family's wording
        prompt = (f"Reconcile the restated invoice INV-{i} against the purchase order and flag the delta." if near
                  else f"A clinic changed a lab order {i} after billing; work out which charge stands and record why.")
        out.append(ProbeTask(f"p{i}", prompt, {"domain": "health"}, world="clinic", template="tpl_clinic", verifier="ver_clinic"))
    return out


def arm(rate_of):
    def run(p, a):
        return a < round(rate_of(p) * 4)

    return run


async def test_transfer_gain_on_held_out_probes_is_accepted():
    ps = probes()
    rep = await transfer_proxy(ps, arm(lambda p: 0.0), arm(lambda p: 0.75), family=FAMILY, run_placebo=arm(lambda p: 0.25))
    assert rep["ok"], rep["reasons"]
    assert rep["paired"]["improved"] and rep["paired"]["positive"] == 12 and rep["paired"]["sign_p"] < 0.001
    assert rep["surface"]["low_overlap_gain"] == 0.75 and rep["vs_placebo"]["improved"]
    assert disjoint(ps, FAMILY, held_out_axis="domain") == []


async def test_gain_explained_by_surface_or_generic_context_is_rejected():
    ps = probes()
    near = {p.id for p in ps if overlap(p, FAMILY)["lexical"] > 0.5}
    assert len(near) == 6
    surf = await transfer_proxy(ps, arm(lambda p: 0.0), arm(lambda p: 1.0 if p.id in near else 0.0), family=FAMILY)
    assert not surf["ok"] and any("surface overlap" in r for r in surf["reasons"])
    placebo = await transfer_proxy(ps, arm(lambda p: 0.0), arm(lambda p: 0.75), family=FAMILY, run_placebo=arm(lambda p: 0.75))
    assert not placebo["ok"] and any("placebo" in r for r in placebo["reasons"])
    none = await transfer_proxy(ps, arm(lambda p: 0.25), arm(lambda p: 0.25), family=FAMILY)
    assert not none["ok"] and any("no paired gain" in r for r in none["reasons"])
    few = await transfer_proxy(ps[:4], arm(lambda p: 0.0), arm(lambda p: 1.0), family=FAMILY)
    assert any("need 10" in r for r in few["reasons"])
    solved = await transfer_proxy(ps, arm(lambda p: 0.75), arm(lambda p: 1.0), family=FAMILY, config=TransferConfig(base_max=0.5))
    assert any("already solves" in r for r in solved["reasons"])


def test_probe_set_must_be_disjoint_from_the_family():
    leaky = [ProbeTask("p", "x", {"domain": "finance"}, world="ap_world", template="tpl_other", verifier="ver_ap")]
    errs = disjoint(leaky, FAMILY, held_out_axis="domain")
    assert any("world" in e for e in errs) and any("verifier" in e for e in errs) and any("domain" in e for e in errs)
    assert exemplar_block(["t1", "t2", "t3"], 2).count("Example") == 2 and exemplar_block([], 3) == ""


async def test_best_of_n_selects_with_the_family_verifier_and_grades_with_the_probes():
    cands = ["bad", "good", "meh"]
    family_score = {"bad": 0.0, "good": 1.0, "meh": 0.5}
    ok = await best_of_n(lambda i: cands[i], lambda c: family_score[c], lambda c: c == "good", 3)
    assert ok is True
    assert await best_of_n(lambda i: None, lambda c: 1.0, lambda c: True, 3) is None


# --- goldset --------------------------------------------------------------------------------------------------------


def items(n_each: int) -> list[GoldItem]:
    out = []
    for src, n in (("real_pass", n_each), ("real_fail", n_each), ("mutant", n_each), ("alternative", n_each)):
        for i in range(n):
            sol = {"src": src, "i": i}
            out.append(GoldItem(f"{src}_{i}", src, sol, item_sha256(sol)))  # type: ignore[arg-type]
    return out


def label_reals(store: LabelStore, its: list[GoldItem], *, model_wrong: int = 0, share: float = 1.0) -> None:
    reals = [it for it in its if it.source.startswith("real")]
    for j, it in enumerate(reals):
        truth = "pass" if it.source == "real_pass" else "fail"
        store.add_label(item=it.id, item_sha256=it.sha256, labeller="model:reviewer", value=("fail" if truth == "pass" else "pass") if j < model_wrong else truth)
        if j < share * len(reals):
            store.add_label(item=it.id, item_sha256=it.sha256, labeller="human:ann", value=truth)


def perfect(sol):
    return 1.0 if sol["src"] in ("real_pass", "alternative") else 0.0


async def test_goldset_accepts_a_correct_verifier_and_rejects_one_that_passes_everything(tmp_path):
    its = items(40)
    store = LabelStore(tmp_path / "labels.jsonl")
    label_reals(store, its)
    good = await run_goldset(its, perfect, store)
    assert good.ok, good.reasons()
    assert good.card() == {"fp": (0, 80), "fn": (0, 80), "kappa": 1.0}
    assert good.fp.interval[1] < 0.05 and good.real_share == 0.5

    bad = await run_goldset(its, lambda s: 1.0, store)
    assert not bad.ok and bad.fp.pair() == (80, 80) and len(bad.errors["false_pass"]) == 80
    assert any("false-pass" in r for r in bad.reasons())


async def test_v1_minimum_set_cannot_reach_the_v1_bound(tmp_path):
    its = items(10)  # 40 items: 20 truly passing, 20 truly failing
    store = LabelStore(tmp_path / "labels.jsonl")
    label_reals(store, its)
    rep = await run_goldset(its, perfect, store)
    assert rep.fp.pair() == (0, 20) and rep.fp.interval[1] > 0.05
    assert not rep.ok and any("Wilson upper" in r for r in rep.reasons())


async def test_kappa_human_gate_stale_and_conflicting_labels(tmp_path):
    its = items(40)
    store = LabelStore(tmp_path / "labels.jsonl")
    label_reals(store, its, model_wrong=20, share=0.25)  # the model reviewer is wrong on 20 of 80; humans label 20
    rep = score(its, {it.id: perfect(it.solution) for it in its}, store, config=GoldConfig(human_lane=True))
    assert rep.kappa["n"] == 20 and rep.kappa["model_reviewed"] == 80 and rep.kappa["kappa"] < 0.7
    assert rep.human_gate  # below the bar: the human stays a gate for the family
    assert any("without a human truth label" in r for r in rep.reasons())  # 60 real items lack a human truth

    its2 = items(40)
    s2 = LabelStore(tmp_path / "l2.jsonl")
    label_reals(s2, its2)
    target = its2[0]
    first = s2.effective(target.id, target.sha256)
    changed = GoldItem(target.id, target.source, {"edited": True}, item_sha256({"edited": True}))  # the item changed: its labels are stale
    s2.add_label(item=its2[1].id, item_sha256=its2[1].sha256, labeller="human:bob", value="fail")  # two humans disagree
    mut = its2[85]  # a 'mutant' a human says is right: the construction is contradicted
    s2.add_label(item=mut.id, item_sha256=mut.sha256, labeller="human:ann", value="pass")
    its2 = [changed, *its2[1:]]
    rep2 = score(its2, {it.id: 1.0 if it.source in ("real_pass", "alternative") else 0.0 for it in its2}, s2)
    assert first and rep2.info["stale_labels"] == [target.id] and target.id in rep2.info["unlabelled_real"]
    assert rep2.info["human_disagreement"] == [its2[1].id] and rep2.info["proof_conflict"] == [mut.id]
    assert any("contradict" in r for r in rep2.reasons())
    # the verifier rejected that 'mutant', which the human says is right: a false fail
    assert mut.id in rep2.errors["false_fail"]


async def test_goldset_composition_rules():
    its = [it for it in items(40) if it.source != "alternative"]
    rep = score(its, {}, [])
    reasons = " ".join(rep.reasons())
    assert "kind ['alternative']" in reasons and "real solver outputs" not in reasons
    masked = score(items(40), {}, [])
    assert masked.unmeasured and masked.fp.n == 0 and not masked.ok
