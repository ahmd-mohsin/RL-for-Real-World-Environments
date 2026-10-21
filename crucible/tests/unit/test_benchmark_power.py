"""Benchmark power (``bench.power``): A/A noise from attempt halves, the MDE of a paired comparison, the gate, the planner,
P(trained > control) of the paired bootstrap (``lh_metrics.paired``), the S15 band-decision power, and the factory's stamp
(``factory.tinker.comparison_power``) and W&B view (``observe.bench.power_scalars``)."""

from __future__ import annotations

import math
import random

import pytest

from crucible.bench import power as pw
from crucible.bench.lh_metrics import LHEpisode, paired


def _bernoulli_rows(ps: list[float], k: int, seed: int = 0, task_prefix: str = "t") -> list[dict]:
    rng = random.Random(seed)
    return [{"task": f"{task_prefix}{i}", "passed": rng.random() < p, "reward": float(rng.random() < p), "final_phi": None, "masked": False}
            for i, p in enumerate(ps) for _ in range(k)]


def test_z_sum_is_the_two_sided_80_percent_constant_and_matches_rl_env_use_143() -> None:
    assert pw.z_sum() == pytest.approx(1.959964 + 0.841621, abs=1e-5)
    # rl-env-use: mde = 1.43 * noise, noise = 1.96 * SE_diff (the A/A half-width over sqrt 2): the same 2.80
    assert 1.43 * 1.959964 == pytest.approx(pw.z_sum(), abs=0.01)


def test_aa_noise_recovers_the_within_task_variance() -> None:
    ps = [0.1, 0.3, 0.5, 0.7, 0.0, 1.0] * 50
    rows = _bernoulli_rows(ps, 8, seed=1)
    nz = pw.aa_noise(pw.by_task(rows, "pass"), metric="pass", splits=50)
    true = sum(p * (1 - p) for p in ps) / len(ps)
    assert nz["n_tasks"] == len(ps) and nz["k"] == 8
    assert nz["sigma2"] == pytest.approx(true, rel=0.12)
    assert nz["sigma2_by_split"]["direct"] == pytest.approx(true, rel=0.12)
    assert abs(nz["aa"]["diff"]) < 3 * nz["aa"]["se"]  # halves are exchangeable
    assert nz["control_mean"] == pytest.approx(sum(ps) / len(ps), abs=0.03)


def test_all_zero_tasks_have_no_aa_noise_but_the_bounded_trained_variance_keeps_the_mde_honest() -> None:
    rows = _bernoulli_rows([0.0] * 10, 8)
    nz = pw.aa_noise(pw.by_task(rows), metric="pass")
    assert nz["sigma2"] == 0.0 and nz["control_mean"] == 0.0
    g = pw.power_gate(nz, 10, 8, target=0.2)
    assert g["mde_aa"] == 0.0  # rl-env-use's A/A assumption claims any effect is detectable
    assert g["sigma2_trained"] == pytest.approx(0.2 * 0.8) and g["mde"] == pytest.approx(pw.z_sum() * math.sqrt(0.16 / 80))
    assert g["verdict"] == "powered"


def test_mde_and_n_needed_agree() -> None:
    s2c, s2t, k, tgt = 0.04, 0.2475, 8, 0.15
    n = pw.n_needed(tgt, s2c, s2t, k)
    assert pw.mde(pw.se_diff(s2c, s2t, n, k)) <= tgt + 1e-12 < pw.mde(pw.se_diff(s2c, s2t, n - 1, k))
    # heterogeneity of the effect across tasks only ever needs more tasks; more attempts never fewer
    assert pw.n_needed(tgt, s2c, s2t, k, tau=0.2) > n >= pw.n_needed(tgt, s2c, s2t, 16)
    with pytest.raises(ValueError):
        pw.n_needed(0.0, s2c, s2t, k)


def test_gate_verdicts_powered_underpowered_and_unmeasured() -> None:
    nz = pw.aa_noise(pw.by_task(_bernoulli_rows([0.0, 0.1, 0.2, 0.3, 0.5] * 4, 8, seed=2)), metric="pass")
    ok = pw.power_gate(nz, 40, 8, target=0.20)
    assert ok["verdict"] == "powered" and ok["why"] == "" and not ok["headline"].startswith("UNDER")
    small = pw.power_gate(nz, 6, 8, target=0.20)
    assert small["verdict"] == "underpowered" and "6 tasks < 10" in small["why"] and small["headline"].startswith("UNDERPOWERED")
    tight = pw.power_gate(nz, 12, 8, target=0.05)
    assert tight["verdict"] == "underpowered" and "MDE" in tight["why"] and tight["n_needed"] > 12
    assert pw.power_gate({"n_tasks": 0, "sigma2": None}, 10, 8)["verdict"] == "unmeasured"
    with pytest.raises(ValueError):
        pw.power_gate(nz, 10, 8, variance="other")


def test_plan_rows_and_floor() -> None:
    nz = pw.aa_noise(pw.by_task(_bernoulli_rows([0.0, 0.1, 0.25] * 5, 8, seed=3)), metric="pass")
    p = pw.plan(nz, targets=(0.2, 0.15), ks=(4, 8), taus=(0.0, 0.2))
    assert len(p["rows"]) == 8 and all(r["n"] >= 10 and r["n"] >= r["raw"] for r in p["rows"])
    by = {(r["target"], r["k"], r["tau"]): r["raw"] for r in p["rows"]}
    assert by[(0.15, 8, 0.0)] > by[(0.2, 8, 0.0)] and by[(0.2, 4, 0.0)] > by[(0.2, 8, 0.0)] and by[(0.2, 8, 0.2)] > by[(0.2, 8, 0.0)]


def test_halves_handle_odd_attempt_counts_and_masked_rows() -> None:
    rows = [{"task": "a", "passed": x, "masked": False} for x in (1, 0, 1, 0, 1, 0, 1)] + [{"task": "a", "passed": True, "masked": True}]
    rows += [{"task": "b", "passed": True, "masked": False}]  # one attempt: no split exists
    vals = pw.by_task(rows)
    assert len(vals["a"]) == 7 and vals["b"] == [1.0]
    nz = pw.aa_noise(vals)
    assert nz["n_tasks"] == 1 and nz["per_task"]["a"]["var_direct"] == pytest.approx(statistics_var([1, 0, 1, 0, 1, 0, 1]))


def statistics_var(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def test_by_task_reads_lh_episodes_and_reward_rows() -> None:
    eps = [LHEpisode(task="x", passed=True), LHEpisode(task="x", passed=False), LHEpisode(task="y", passed=False, masked=True)]
    assert pw.by_task(eps, "pass") == {"x": [1.0, 0.0]}
    assert pw.by_task([{"task": "x", "reward": 0.25, "final_phi": 0.5}], "reward") == {"x": [0.25]}
    assert pw.by_task([{"task": "x", "reward": 0.25, "final_phi": 0.5}], "phi") == {"x": [0.5]}
    with pytest.raises(ValueError):
        pw.by_task([{"task": "x"}], "other")


def test_paired_bootstrap_reports_p_better() -> None:
    base = [LHEpisode(task=f"t{i}", passed=False) for i in range(12) for _ in range(4)]
    better = [LHEpisode(task=f"t{i}", passed=j < 2) for i in range(12) for j in range(4)]
    d = paired(base, better, n_boot=500)
    assert d["p_better"] == pytest.approx(1.0) and d["mean_diff"] == pytest.approx(0.5)
    same = paired(base, base, n_boot=200)
    assert same["p_better"] == 0.0 and same["mean_diff"] == 0.0
    assert paired([], [], n_boot=10)["p_better"] is None


def test_study_pools_datasets_as_distinct_units() -> None:
    a = _bernoulli_rows([0.0, 0.25, 0.5], 8, seed=4)
    b = _bernoulli_rows([0.0, 0.25, 0.5], 8, seed=5)  # the same task ids measured again: two units, not one
    out = pw.study({"a": a, "b": b}, n=10, k=8, targets=(0.2,), ks=(8,), taus=(0.0,), trained_means=(None,), splits=20)
    assert out["sets"]["pooled"]["pass"]["noise"]["n_tasks"] == 6
    g = out["sets"]["pooled"]["pass"]["gates"]["target=0.20|bounded|tau=0.0"]
    assert g["verdict"] in ("powered", "underpowered") and g["mde"] >= g["mde_aa"] - 1e-12


def test_tinker_rl_power_command_reads_both_arm_layouts(tmp_path, capsys) -> None:
    import importlib.util
    import json
    from pathlib import Path
    from types import SimpleNamespace

    p = Path(__file__).resolve().parents[2] / "scripts" / "tinker_rl.py"
    spec = importlib.util.spec_from_file_location("tinker_rl_power", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s15 = tmp_path / "s15"  # an S15 arm: one directory per task
    for i, rows in enumerate([_bernoulli_rows([0.25], 8, seed=8, task_prefix="a"), _bernoulli_rows([0.0], 8, seed=9, task_prefix="b")]):
        (s15 / f"task{i}").mkdir(parents=True)
        (s15 / f"task{i}" / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    bench = tmp_path / "bench"  # a benchmark arm: one episodes.jsonl
    bench.mkdir()
    (bench / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _bernoulli_rows([0.5, 0.1], 8, seed=10)))
    assert len(mod.arm_rows(s15)) == 16 and len(mod.arm_rows(bench)) == 16
    out = tmp_path / "power.json"
    mod.cmd_power(SimpleNamespace(arms=[f"s15={s15}", str(bench)], n=10, k=8, target=[0.2, 0.15], json=str(out)))
    doc = json.loads(out.read_text())
    assert set(doc["sets"]) == {"s15", "bench", "pooled"} and doc["sets"]["pooled"]["pass"]["noise"]["n_tasks"] == 4
    assert doc["inputs"]["s15"]["episodes"] == 16 and "MDE" in capsys.readouterr().out


def test_comparison_power_preregisters_from_the_baseline_and_says_so() -> None:
    from crucible.factory.tinker import comparison_power

    base = _bernoulli_rows([0.0, 0.1, 0.3] * 4, 8, seed=6)
    pre = comparison_power(n=12, k=8, baseline_rows=base, control_rows=[])
    assert pre["preregistered"] and pre["source"].startswith("baseline") and pre["verdict"] == pre["metrics"]["pass"]["gate"]["verdict"]
    assert set(pre["metrics"]) == {"pass", "reward"}
    post = comparison_power(n=3, k=8, baseline_rows=[], control_rows=base)
    assert not post["preregistered"] and "post hoc" in post["source"] and post["verdict"] == "underpowered"


def test_power_scalars_flatten_for_wandb() -> None:
    from crucible.observe.bench import power_scalars, power_verdict

    rep = pw.power_report(_bernoulli_rows([0.0, 0.2] * 6, 8, seed=7), n=12, k=8)
    sc = power_scalars(rep)
    assert set(sc) == {"pass", "reward"} and isinstance(sc["pass"]["powered"], bool) and sc["pass"]["n_tasks"] == 12
    assert power_verdict(rep) == rep["pass"]["gate"]["verdict"] and power_verdict(None) is None


def test_keep_verdict_reports_whether_the_band_decision_is_powered() -> None:
    from crucible.factory.targeting import KeepRule, band_power, keep_verdict

    r = KeepRule()
    undecided = band_power(1, 8, split="train", rule=r)  # 0.125, Wilson [0.02, 0.47]: crosses train_lo 0.1
    assert not undecided["decided"] and undecided["crosses"] == [0.1] and (undecided["n_needed"] is None or undecided["n_needed"] > 8)
    mid = band_power(3, 8, split="train", rule=r)  # 0.375, [0.14, 0.69]: crosses train_hi 0.6
    assert mid["crosses"] == [0.6] and not mid["decided"]
    held = band_power(0, 8, split="heldout", rule=r)  # 0, [0, 0.32]: crosses bench_hi 0.3
    assert held["crosses"] == [0.3] and held["n_needed"] == 9
    sure = band_power(0, 16, split="heldout", rule=r)  # [0, 0.19]
    assert sure["decided"] and sure["n_needed"] == 16 and sure["crosses"] == []
    v = keep_verdict(split="train", passes=1, n=8, on_intent=6, labelled=7, frontier={"f": [True]})
    assert v["verdict"] == "train" and v["power"]["decided"] is False  # a flag, never a gate
