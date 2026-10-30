"""crucible.synth.meta.stats + the paired mode of the update rule (runs/rsi-stats.md): A/A noise floors, betting tests,
alpha spending over a whole run (with a simulation of its error guarantee), R1-strict decision units, the reusable
holdout, power, and the continuous loop's state. Everything is a scripted fake: no model is called."""

from __future__ import annotations

import json
import math
import random
import zlib

import pytest

from crucible.synth.meta.stats import (
    AlphaSpender,
    Thresholdout,
    aa_floor,
    betting_lcb,
    betting_test,
    bootstrap_ci,
    effect,
    fisher_one_sided,
    mcnemar_power,
    mde,
    plants_needed,
    power_table,
)
from crucible.synth.meta.update import Ledger, Measurement, StatsConfig, ThetaStore, Update, UpdateConfig, UpdateEngine, View

# ---------------------------------------------------------------------------------------------------------- the tests by betting


def _null_draw(rng: random.Random, n: int, mean: float = 0.0) -> list[float]:
    """Paired differences with ties and seed noise: P(+1) = q + mean/2, P(−1) = q − mean/2, else 0."""
    q = 0.06
    out = []
    for _ in range(n):
        u = rng.random()
        out.append(1.0 if u < q + mean / 2 else -1.0 if u < 2 * q else 0.0)
    return out


def test_the_betting_test_holds_its_level_at_the_null_boundary_and_has_power():
    rng = random.Random(0)
    m0 = 0.02  # the boundary of H: E[D] <= m0
    rej = sum(betting_test(_null_draw(rng, 120, mean=m0), m0, 0.05).reject for _ in range(1500))
    assert rej / 1500 <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 1500)
    alt = sum(betting_test(_null_draw(rng, 120, mean=0.10), m0, 0.05).reject for _ in range(300))
    assert alt / 300 > 0.5  # a 0.10 gain on 120 units is found most of the time
    d = [1.0] * 30 + [0.0] * 170
    random.Random(1).shuffle(d)
    r = betting_test(d, 0.01, 0.05 / 20)
    assert r.reject and r.stop is not None and r.stop <= len(d) and r.max_wealth >= 20 / 0.05
    lo = betting_lcb(d, 0.05 / 20)
    assert 0.01 < lo < 0.15 and not betting_test(d, lo + 0.01, 0.05 / 20).reject
    assert not betting_test(d, 0.01, 0.0).reject  # no α left: never rejects
    assert betting_lcb([], 0.05) == -math.inf


def test_bootstrap_effects_and_the_cluster_interval_is_wider_for_correlated_units():
    d = [1.0] * 10 + [0.0] * 50
    lo, hi = bootstrap_ci(d)
    assert lo < 1 / 6 < hi
    clusters = [f"t{i // 10}" for i in range(60)]  # all ten gains in one source task
    clo, chi = bootstrap_ci(d, clusters=clusters)
    assert (chi - clo) > (hi - lo)
    e = effect(d, 0.01, clusters=clusters)
    assert e.gains == 10 and e.losses == 0 and e.n == 60 and e.ci95_cluster is not None


def test_the_aa_floor_needs_five_replicates_and_measures_paired_noise():
    rng = random.Random(3)
    base = [1.0 if rng.random() < 0.9 else 0.0 for _ in range(80)]

    def noisy() -> list[float]:
        return [1.0 - x if rng.random() < 0.03 else x for x in base]

    with pytest.raises(ValueError):
        aa_floor([(noisy(), noisy()) for _ in range(4)])
    f = aa_floor([(noisy(), noisy()) for _ in range(6)], seeds=[(1, 2)] * 6)
    assert f.replicates == 6 and 0 < f.delta_aa < 0.1 and 0.02 < f.discord_rate < 0.12 and f.delta_aa >= f.max_abs
    exact = aa_floor([(base, base)] * 5)
    assert exact.delta_aa == 0.0 and exact.discord_rate == 0.0  # a deterministic evaluator has no paired noise


def test_alpha_spending_schedules_sum_to_alpha():
    s = AlphaSpender(alpha=0.05, schedule="bonferroni", t_max=10)
    levels = [s.next(f"u{i}").alpha_t for i in range(12)]
    assert levels[:10] == [0.005] * 10 and levels[10:] == [0.0, 0.0] and s.exhausted and s.total == pytest.approx(0.05)
    g = AlphaSpender(alpha=0.05, schedule="series")
    assert sum(g.level(t) for t in range(1, 100000)) <= 0.05 and g.level(1) == pytest.approx(0.05 * 6 / math.pi**2)


def test_the_run_level_error_guarantee_holds_by_simulation_and_the_naive_rule_breaks_it():
    """FWER = P(some non-improving update of a run is accepted). Every candidate here is null (true gain 0, the A/A
    distribution); each run tests T of them. Ours: betting at α_t = α/T against δ_AA. Naive: a per-test one-sided normal
    LCB_95 > 0, no correction (M3's rule with a floor of 0)."""
    rng = random.Random(11)
    runs, T, n = 400, 8, 60
    ours = naive = 0
    for _ in range(runs):
        aa = aa_floor([(_null_draw(rng, n), [0.0] * n) for _ in range(5)])  # replicate means of null differences
        hit_ours = hit_naive = False
        sp = AlphaSpender(alpha=0.05, t_max=T)
        for t in range(T):
            d = _null_draw(rng, n)
            hit_ours |= betting_test(d, aa.delta_aa, sp.next(f"c{t}").alpha_t, c=0.75).reject
            m = sum(d) / n
            sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
            hit_naive |= m - 1.644854 * sd / math.sqrt(n) > 0
        ours += hit_ours
        naive += hit_naive
    se = math.sqrt(0.05 * 0.95 / runs)
    assert ours / runs <= 0.05 + 3 * se
    assert naive / runs > 0.15  # T uncorrected looks at noise: the family-wise error is far above 0.05


def test_exact_tests_and_power():
    assert fisher_one_sided(2, 2, 0, 2) == pytest.approx(1 / 6)  # RSI-1's meta-loop acceptances: 2/2 vs 0/2 is p = 0.17
    assert fisher_one_sided(0, 4, 4, 4) == pytest.approx(1.0)
    assert plants_needed(0.2) == 33 and plants_needed(0.3) < plants_needed(0.2) < plants_needed(0.1)
    assert mde(30) == pytest.approx(0.215) and mde(1) is None and mde(12) > mde(30)
    assert mcnemar_power(33, 0.2) >= 0.8 > mcnemar_power(20, 0.2)
    assert plants_needed(0.2, p01=0.02) > plants_needed(0.2)  # seed-noise losses cost power
    rows = power_table({"D1": 30, "D6": 12, "D9": 0}, gains=(0.2,), classes_tested=11)
    assert rows[0]["mde"] == pytest.approx(0.215) and rows[0]["mde_bonferroni"] > rows[0]["mde"] and rows[2]["mde"] is None


def test_thresholdout_answers_from_training_logs_every_access_and_fails_closed(tmp_path):
    held = {"close": 0.30, "far": 0.90}
    ho = Thresholdout(lambda q: held[q], threshold=0.05, sigma=0.0, budget=1, log=tmp_path / "access.jsonl")
    a = ho.ask("close", 0.31)
    assert a.answer == 0.31 and not a.over_threshold and a.budget_left == 1  # agreement: the holdout reveals nothing
    b = ho.ask("far", 0.20)
    assert b.over_threshold and b.answer == pytest.approx(0.90) and b.budget_left == 0
    c = ho.ask("close", 0.31)
    assert c.refused
    log = [json.loads(x) for x in (tmp_path / "access.jsonl").read_text().splitlines()]
    assert len(log) == 3 and [x["access"] for x in log] == [1, 2, 3] and log[2]["refused"]
    assert all("0.9" not in json.dumps({k: v for k, v in x.items() if k != "at"}) for x in log)  # the held-out value is never logged


# ---------------------------------------------------------------------------------------------------------- paired mode of the engine


def _h(*parts: object) -> float:
    return (zlib.crc32("/".join(map(str, parts)).encode()) % 10_000) / 10_000


class ToyEval:
    """Units are ids; the unchanged Θ catches a unit with p = 0.7; each evaluator seed flips a unit with p = 0.03 (the
    paired noise A/A replicates measure); Θ.F["gain"] = g catches g of the units Θ misses."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, theta, view: View, budget):
        self.calls += 1
        f = theta.data["F"]
        seed = view.eval_seed or 0
        js, doms = [], []
        for u in view.payload["plants"]:
            y = 1.0 if _h(u, "base") < 0.7 else 0.0
            if y == 0.0 and _h(u, "gain") < f.get("gain", 0.0):
                y = 1.0
            if _h(u, "flip", seed) < 0.03:
                y = 1.0 - y
            js.append(y)
            doms.append("A" if _h(u, "dom") < 0.5 else "B")
        extra = 0.0
        return Measurement(j_units=[min(1.0, y + extra) for y in js], unit_domains=doms, fr={"D1": sum(js) / max(1, len(js))}, fa=0.0,
                           cost=1.0 if budget is None else budget, unit_ids=list(view.payload["plants"]), unit_classes=["D1"] * len(js),
                           unit_clusters=[u.split("-")[0] for u in view.payload["plants"]], n_anchors=0)


class ToySampler:
    """Calibration views from a proposal bank; decision views of fresh units from a decision bank (R1-strict)."""

    def __init__(self, n_decide: int = 2000, n_max: int = 60) -> None:
        self.n, self.n_max = 0, n_max
        self.bank = [f"d{i // 8}-{i}" for i in range(n_decide)]
        self.consumed: set[str] = set()
        self.reserved: set[str] = set()
        self.reuse: list[str] | None = None

    async def __call__(self, round, purpose, used):
        self.n += 1
        if purpose in ("decide", "prune"):
            rem = [u for u in self.bank if u not in self.consumed and u not in self.reserved]
            ids = self.reuse or rem[: self.n_max]
            self.reserved |= set(ids)
            return View(hash=f"v{self.n}", round=round, domains=["A", "B"], planters=["P@1"], seed=self.n, strict=True, payload={"plants": ids, "anchors": []})
        return View(hash=f"c{self.n}", round=round, domains=["A", "B"], planters=["P@1"], seed=self.n, payload={"plants": [f"p{i}" for i in range(self.n_max)], "anchors": []})

    def consume(self, view, units):
        self.consumed |= set(units)
        self.reserved -= set(units)

    def release(self, view, units):
        self.reserved -= set(units)


def _paired_engine(tmp_path=None, **st) -> tuple[UpdateEngine, ToyEval, ToySampler]:
    store = ThetaStore(tmp_path / "theta" if tmp_path is not None else None)
    store.init("F", {"gain": 0.0})
    store.init("P", {"planters": ["dev"]})
    ev, s = ToyEval(), ToySampler(n_max=st.get("n_max", 60))
    cfg = UpdateConfig(rounds=5, stats=StatsConfig(paired=True, **st))
    eng = UpdateEngine(store, ev, s, config=cfg, ledger=Ledger(tmp_path / "ledger.jsonl" if tmp_path is not None else None))
    return eng, ev, s


def _upd(uid: str, gain: float) -> Update:
    from crucible.synth.meta.update import Evidence

    return Update(id=uid, component="F", data={"gain": gain}, edits=["e"], author_family="openai", evidence=Evidence(domains=["A"]))


async def test_paired_mode_calibrates_from_aa_replicates_and_keeps_the_floor_rising():
    eng, _, _ = _paired_engine()
    fl = await eng.calibrate()
    assert fl.aa is not None and fl.aa.replicates == 5 and fl.delta == fl.aa.delta_aa and fl.delta_spread is not None and fl.delta > 0
    assert len({tuple(p) for p in fl.aa.seeds}) == 5  # five disjoint seed pairs on one view
    again = await eng.calibrate()
    assert again.delta >= fl.delta  # δ only moves up
    with pytest.raises(PermissionError):
        eng.set_floor(0.0)


async def test_paired_mode_accepts_a_real_gain_refuses_a_null_and_spends_alpha_write_ahead(tmp_path):
    eng, _, s = _paired_engine(tmp_path, t_max=4, n_max=200, chunk=20)
    await eng.calibrate()
    good = await eng.decide(_upd("good", 0.9), round=0)
    assert good.accepted, good.failed
    m = good.metrics
    assert m["t"] == 1 and m["alpha_t"] == pytest.approx(0.0125) and m["e_value"] >= 80 and m["stopped"] == "crossed" and m["effect"]["lcb_anytime"] > m["delta"]
    assert m["released"] and not (set(m["released"]) & s.consumed)  # stopped early: the unevaluated units stay fresh
    null = await eng.decide(_upd("null", 0.9), round=0)  # the same Θ again: no gain over the incumbent
    assert not null.accepted and "R2" in null.rules and null.metrics["stopped"] == "futility" and null.metrics["n_units"] == 20
    recs = eng.ledger.records()
    tests = [r for r in recs if r.kind == "test"]
    assert [t.metrics["t"] for t in tests] == [1, 2] and all(t.metrics["reserved"] for t in tests)
    for i in range(2):
        await eng.decide(_upd(f"x{i}", 0.9 + 0.01 * i), round=1)
    out = await eng.decide(_upd("late", 1.0), round=2)
    assert out.rules == ["R2"] and "budget is spent" in out.failed[0].detail and not out.evaluated  # t_max = 4 tests


async def test_r1_strict_refuses_reused_decision_units_and_a_dry_split():
    eng, _, s = _paired_engine(n_max=40)
    await eng.calibrate()
    d = await eng.decide(_upd("a", 0.9), round=0)
    s.reuse = d.metrics.get("released", []) or []
    s.reuse = [u for u in s.bank if u in s.consumed][:40]
    again = await eng.decide(_upd("b", 0.95), round=0)
    assert again.rules == ["R1"] and "earlier decision" in again.failed[0].detail and len(eng.spender.spent) == 1  # refused before any α is spent
    s.reuse = None
    s.bank = s.bank[:45]
    dry = await eng.decide(_upd("c", 0.95), round=0)
    assert dry.rules == ["R1"] and "fresh units left" in dry.failed[0].detail


async def test_the_engine_state_round_trips_and_replays_write_ahead_tests(tmp_path):
    eng, _, _ = _paired_engine(tmp_path, t_max=10)
    await eng.calibrate()
    await eng.decide(_upd("a", 0.9), round=0)
    st = eng.state()
    eng2, _, _ = _paired_engine(tmp_path, t_max=10)
    eng2.ledger = eng.ledger
    eng2.store = ThetaStore(tmp_path / "theta")
    # a crash after a write-ahead test record and before its decision: the spend and the reserved units stay spent
    eng._spend("crashed", 1, View(hash="vx", round=1, payload={"plants": ["d999-1", "d999-2"], "anchors": []}))
    eng2.load_state(st)
    assert eng2.floor == eng.floor and len(eng2.spender.spent) == 2 and {"d999-1", "d999-2"} <= eng2.used_units
    assert eng2.used_units >= eng.used_units - {"d999-1", "d999-2"}


async def test_the_holdout_gate_refuses_with_rule_g():
    eng, _, _ = _paired_engine(t_max=4, n_max=200, chunk=20)

    async def gate(theta, cand, u, train_gain):
        from crucible.synth.meta.update import RuleFailure

        return [RuleFailure(rule="G", detail="locked-set gain -0.05")]

    eng.holdout_gate = gate
    await eng.calibrate()
    d = await eng.decide(_upd("good", 0.9), round=0)
    assert d.rules == ["G"] and not d.accepted


async def test_the_whole_engine_keeps_the_run_level_error_under_alpha():
    """The machinery end to end: many runs of T null updates each (no candidate is better than the incumbent) through
    calibrate (A/A) and decide (paired mode); no run may accept more often than α."""
    runs, T = 120, 6
    bad = 0
    for k in range(runs):
        eng, _ev, s = _paired_engine(t_max=T, n_max=40, chunk=10, seeds=list(range(100 * k + 1, 100 * k + 13)))
        s.bank = [f"r{k}-{i // 8}-{i}" for i in range(T * 40 + 40)]
        await eng.calibrate()
        acc = False
        for t in range(T):
            acc |= (await eng.decide(_upd(f"n{t}", 0.0), round=t)).accepted
        bad += acc
    assert bad / runs <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / runs)
