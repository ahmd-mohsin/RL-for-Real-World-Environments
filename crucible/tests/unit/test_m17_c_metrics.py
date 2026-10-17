"""M17 study 08 §8: long-horizon metrics (pass@1, pass^k, final Φ, losses, I6 compaction loss, calibration, give-up, hack
rate, cost per solve), the paired base-vs-trained report with its MDE, and the held-out sealing helpers."""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

from crucible.bench import lh_metrics as lm
from crucible.factory import sealing


def ep(task, passed, *, intent="I1", series=(), comp=(), claims=(), gave_up=False, hack=False, tokens=1000, cost=0.01, masked=False):
    return lm.LHEpisode(task=task, intent=intent, passed=passed, masked=masked, series=[tuple(s) for s in series], compaction_turns=list(comp),
                        claims=list(claims), gave_up=gave_up, hack=hack, tokens=tokens, cost_usd=cost)


def test_pass_metrics_and_cost():
    es = [ep("t1", True), ep("t1", True), ep("t1", False), ep("t1", True), ep("t2", False), ep("t2", False), ep("t2", False), ep("t2", True),
          ep("t3", True, masked=True)]
    s = lm.summarize(es, k=2)
    assert s["pass@1"] == pytest.approx((3 / 4 + 1 / 4) / 2) and s["masked"] == 1 and s["tasks"] == 2
    assert s["pass^2"] == pytest.approx((math.comb(3, 2) / math.comb(4, 2) + 0.0) / 2)
    assert lm.pass_hat_k(3, 3, 4) is None and lm.pass_hat_k(4, 4, 4) == 1.0
    assert s["tokens_per_solved"] == pytest.approx(8000 / 4) and s["usd_per_solved"] == pytest.approx(0.08 / 4)
    assert lm.summarize([ep("t", False)])["usd_per_solved"] is None


def test_phi_losses_and_compaction_loss():
    # spans: (0,10] no boundary, (10,20] starts at the boundary c=10 and loses 2, (20,30] loses 1 elsewhere
    a = ep("t1", False, series=[(10, 0.5, ["a", "b"], [], True), (20, 0.0, [], ["a", "b"], True), (25, 0.0, [], [], False), (30, 0.5, ["a"], [], True),
                                (40, 0.25, [], ["a"], True)], comp=[10])
    assert a.final_phi == 0.25 and a.lost == 3
    assert a.compaction_loss() == (2, 1, 1, 3)
    s = lm.summarize([a, ep("t2", True, series=[(5, 1.0, ["a"], [], True)])])
    c = s["compaction_loss"]
    assert c["post_boundary_per_span"] == 2.0 and c["elsewhere_per_span"] == pytest.approx(1 / 4) and c["ratio"] == pytest.approx(8.0)
    assert s["mean_final_phi"] == pytest.approx((0.25 + 1.0) / 2) and s["milestones_lost_per_episode"] == pytest.approx(1.5)
    assert ep("t", False, series=[(3, 0.5, [], [], False)]).final_phi is None  # nothing measured: unknown, not 0


def test_calibration_give_up_and_hack():
    es = [ep("t1", False, claims=[(0.9, 0), (0.1, 0)], gave_up=True), ep("t1", True, claims=[(1.0, 1)]), ep("t2", True, gave_up=True, hack=True),
          ep("t3", False, gave_up=True)]
    s = lm.summarize(es)
    assert s["calibration"]["brier"] == pytest.approx((0.81 + 0.01 + 0.0) / 3) and s["calibration"]["claims"] == 3
    assert lm.ece([(0.9, 0), (0.1, 0), (1.0, 1)]) == pytest.approx(1 / 3 * 0.9 + 1 / 3 * 0.1 + 1 / 3 * 0.0)
    g = s["give_up"]
    assert g["rate"] == 0.75 and g["correct"] == pytest.approx(2 / 3) and g["premature"] == pytest.approx(1 / 3)  # t1 was solved by another attempt
    assert s["hack_rate"] == 0.25
    bi = lm.by_intent([*es, ep("t9", True, intent="I6")])
    assert set(bi) == {"I1", "I6", "overall"} and bi["I6"]["pass@1"] == 1.0


def test_from_progress_reads_the_sidecar_summary():
    prog = {"phi0": 0.0, "series": [[2, 0.5, ["a"], [], True, ["commit"]], [4, 0.0, [], ["a"], True, ["compaction"]]], "compaction_turns": [4],
            "give_up": "stuck", "honesty": {"brier": 0.5, "scored": {"r1": [1.0, 0]}}}
    e = lm.LHEpisode.from_progress("t", prog, intent="I1", passed=False, hack=True)
    assert e.final_phi == 0.0 and e.lost == 1 and e.gave_up and e.claims == [(1.0, 0)] and e.brier == 0.5 and e.compaction_turns == [4]


def test_paired_bootstrap_and_mde_deterministic():
    rng = random.Random(1)
    base, trained = [], []
    for t in range(30):
        pb, pt = rng.uniform(0.1, 0.6), None
        pt = min(1.0, pb + 0.3)
        for k in range(4):
            base.append(ep(f"t{t}", rng.random() < pb))
            trained.append(ep(f"t{t}", rng.random() < pt))
    r = lm.paired(base, trained, n_boot=1000, seed=3)
    assert r == lm.paired(base, trained, n_boot=1000, seed=3)  # seeded: reproducible
    assert r["n_tasks"] == 30 and r["mean_diff"] > 0 and r["ci"][0] < r["mean_diff"] < r["ci"][1]
    assert r["mde_assumed"] == pytest.approx(2.8 * 0.25 / math.sqrt(30)) and 0.12 < r["mde_assumed"] < 0.13  # study 08: ~13 points at n=30
    assert r["mde"] == pytest.approx(2.8 * r["sd"] / math.sqrt(30)) and "minimum detectable change" in r["note"]
    assert lm.paired(base[:88], trained[:88], n_boot=200)["mde_assumed"] == pytest.approx(2.8 * 0.25 / math.sqrt(22))  # ~16 points at n=22
    same = lm.paired(base, base, n_boot=300)
    assert same["mean_diff"] == 0.0 and not same["significant"] and "no effect shown" in same["note"]
    grouped = lm.paired(base, trained, n_boot=300, group_of=lambda t: int(t[1:]) % 5)
    assert grouped["units"] == 5
    phi = lm.paired([ep("a", False, series=[(1, 0.2, [], [], True)])], [ep("a", False, series=[(1, 0.6, [], [], True)])], metric="phi", n_boot=50)
    assert phi["mean_diff"] == pytest.approx(0.4) and phi["only_base"] == [] and lm.paired([], [])["mean_diff"] is None


def W(i, repo, date, prompt="", diff=""):
    return {"id": i, "repo": repo, "date_b": date, "prompt": prompt, "diff": diff}


TEXT = ("Add a streaming parser for the configuration format that accepts nested sections, reports the line of every error, "
        "keeps comments attached to the keys they precede and round trips a file byte for byte when nothing was edited")


def test_seal_disjoint_time_split_and_contamination():
    train = [W("tr1", "org/a", "2025-03-01", "Implement caching of resolved paths in the loader", "def f():\n    return cache_lookup(path_key)\n"),
             W("tr2", "org/b", "2025-06-01", "Add retries with jitter to the HTTP client", "")]
    held = [W("h1", "org/c", "2025-09-01", TEXT, "def parse_stream(lines):\n    for no, line in enumerate(lines):\n        yield no, line\n"),
            W("h2", "org/d", "2025-10-01", "Support timezone aware timestamps in the scheduler", "")]
    ok = sealing.seal(train, held, model_cutoff="2025-08-15")
    assert ok.ok and not ok.contamination and ok.counts == {"train": 2, "heldout": 2} and not ok.cutoff_violations
    bad = sealing.seal([*train, W("tr3", "org/c", "2025-12-01", TEXT, "")], held, model_cutoff="2025-09-15")
    assert not bad.ok and bad.shared_repos == ["org/c"]
    assert {v["heldout"] for v in bad.time_violations} == {"h1", "h2"} and bad.time_violations[0]["latest_train"] == "tr3"
    assert [v["heldout"] for v in bad.cutoff_violations] == ["h1"]  # reported; fatal only with strict_cutoff
    assert {h["task"] for h in bad.contamination} == {"h1"} and any(h["source"] == "tr3:prompt" for h in bad.contamination)
    with pytest.raises(sealing.SealError, match="repos on both sides"):
        sealing.assert_sealed([*train, W("tr3", "org/c", "2025-12-01", TEXT, "")], held)
    strict = sealing.seal(train, held, model_cutoff="2025-09-15", strict_cutoff=True)
    assert not strict.ok and strict.problems() == ["1 held-out windows not after the model cutoff"]
    undated = sealing.seal(train, [W("h3", "org/e", None)])
    assert not undated.ok and undated.undated == ["h3"]
    store = SimpleNamespace(windows=lambda: held)  # a window store: anything with ``windows``
    assert sealing.seal(train, store).ok


def test_publish_heldout_needs_a_clean_seal(monkeypatch):
    import crucible.factory.publish as pub

    calls = []
    monkeypatch.setattr(pub, "publish_family", lambda cards, splits, **kw: calls.append(splits) or "published")
    bad = sealing.SealReport(shared_repos=["org/c"])
    with pytest.raises(sealing.SealError, match="unsealed"):
        sealing.publish_heldout([], ["a"], ["b"], bad)
    assert not calls
    assert sealing.publish_heldout([], ["a"], ["b"], sealing.SealReport()) == "published" and calls == [{"train": ["a"], "heldout": ["b"]}]
