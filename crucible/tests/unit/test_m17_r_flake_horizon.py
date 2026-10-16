"""M17 owner R: flake (reruns × concurrency, Clopper–Pearson bound, verifier nondeterminism kept apart from state
variation) and the long-horizon tests (milestones, order, redundancy, concatenation, shortcut). Adversarial cases: a
verifier that flips on the same end state; a task that is three independent short tasks fails the concatenation test;
a World with a one-call bulk tool fails the shortcut probe."""

from __future__ import annotations

import asyncio
import random

import pytest

from crucible.factory.flake import FlakeConfig, measure_flake, separate
from crucible.factory.floors import Graded
from crucible.factory.horizon import (
    HorizonConfig,
    concatenation_test,
    horizon_card,
    milestone_test,
    series_by_replay,
    series_from_progress,
    shortcut_candidates,
    shortcut_probe,
    steps_of,
)
from tests.unit.m17_r_fakes import STRONG, call, replay, verifier

# --- flake ----------------------------------------------------------------------------------------------------------


async def test_deterministic_oracle_passes_with_the_default_60_reruns():
    inflight = {"now": 0, "max": {}}

    async def run(i):
        inflight["now"] += 1
        await asyncio.sleep(0)
        inflight["max"][i] = inflight["now"]
        inflight["now"] -= 1
        return Graded(1.0, {"a": True}, "fp")

    rep = await measure_flake(run)
    assert rep.ok, rep.reasons()
    assert rep.n == 60 and rep.upper < 0.05 and set(rep.levels) == {1, 4}
    assert max(inflight["max"][i] for i in range(30)) == 1  # level 1 runs one at a time
    assert max(inflight["max"].values()) <= 4


async def test_ten_reruns_per_level_cannot_meet_the_v1_bound():
    rep = await measure_flake(lambda i: _ok(), config=FlakeConfig(reruns=10))
    assert rep.false_negatives == 0 and rep.upper == pytest.approx(1 - 0.05 ** (1 / 20))
    assert not rep.ok and "upper bound" in rep.reasons()[0]


async def _ok():
    return Graded(1.0, {}, "fp")


async def test_verifier_nondeterminism_is_separated_from_state_variation():
    async def flips(i):  # same end state, verdict alternates: the verifier is nondeterministic
        return Graded(float(i % 2), {"c": bool(i % 2)}, "same")

    rep = await measure_flake(flips)
    assert rep.verifier_nondeterministic and not rep.environment_nondeterministic and not rep.ok
    assert any("verifier nondeterminism" in r for r in rep.reasons())

    async def env(i):  # oracle replays end in two states, both pass: environment nondeterminism, no verifier flake
        return Graded(1.0, {"c": True}, f"s{i % 2}")

    rep2 = await measure_flake(env)
    assert rep2.environment_nondeterministic and not rep2.verifier_nondeterministic and rep2.ok
    # policy rollouts: different end states with different verdicts are agent stochasticity, never a verifier flake
    att = separate([Graded(1.0, {"c": True}, "a"), Graded(0.0, {"c": False}, "b"), Graded(1.0, {"c": True}, "a"), Graded(None)])
    assert att == {"states": 2, "verifier_inconsistent_states": [], "states_all_pass": 1, "states_all_fail": 1, "unattributed": 0}


async def test_crashes_and_masks_are_unmeasured_not_passes():
    async def run(i):
        if i % 3 == 0:
            raise RuntimeError("node died")
        return None if i % 3 == 1 else Graded(1.0, {}, "fp")

    rep = await measure_flake(run, config=FlakeConfig(reruns=12))
    assert rep.masked == 16 and rep.n == 8 and not rep.ok  # 4 measured per level < 10


async def test_toy_oracle_reruns_are_one_state_one_verdict():
    from crucible_pack_toy.forge.world_synth import forge

    from crucible.factory.floors import grader

    col: list = []
    await forge(collect=col)
    t = next(t for t, _ in col if t.task_id.endswith("t2_approval"))
    g = grader(t)
    rep = await measure_flake(lambda i: g(t.oracle), config=FlakeConfig(reruns=3, concurrency=(1, 3), min_reruns=3))
    assert rep.false_negatives == 0 and rep.attribution["states"] == 1 and not rep.verifier_nondeterministic
    assert not rep.ok  # 6 reruns bound the rate at 39%: the card says so


# --- horizon --------------------------------------------------------------------------------------------------------


def test_milestones_order_and_redundancy():
    ms = ["m1", "m2", "m3", "m4", "m5"]
    series = [frozenset(ms[:k]) for k in range(6)]
    ok = milestone_test(series, order=ms)
    assert ok["ok"] and ok["first_step"] == {m: i + 1 for i, m in enumerate(ms)}
    assert not milestone_test(series[:4])["ok"]  # three milestones < 5
    together = [frozenset(), frozenset({"m1", "m2"}), *[frozenset({"m1", "m2", *ms[2:k]}) for k in range(3, 6)]]
    r = milestone_test(together)
    assert ["m1", "m2"] in r["simultaneous"] and not r["ok"]
    r2 = milestone_test(series, order=["m2", "m1", "m3", "m4", "m5"])
    assert r2["out_of_order"] == ["m2 after m1"]
    # redundant: identical over every observed state; another end state that tells them apart clears it
    assert r["redundant"] == [["m1", "m2"]]
    assert milestone_test(together, other_states=[frozenset({"m2"})])["redundant"] == []
    pre = [frozenset({"x"}), *[frozenset({"x", *ms[:k]}) for k in range(1, 6)]]
    assert not milestone_test(pre, universe=["x", *ms])["redundant"]  # x already holds at the start: not a milestone


def test_series_from_progress_rows():
    rows = [{"step": 0, "gained": ["a0"], "lost": []}, {"step": 1, "gained": ["a"], "lost": []}, {"step": 2, "gained": ["b"], "lost": ["a"]},
            {"step": -1, "gained": ["a"], "lost": []}]
    s = series_from_progress(rows)
    assert s == [frozenset({"a0"}), frozenset({"a0", "a"}), frozenset({"a0", "a", "b"})]
    assert milestone_test(s, config=HorizonConfig(min_milestones=2))["regressions"] == []


async def test_series_by_replay_reads_world_state():
    ms = {"created": lambda s: "report.csv" in s, "total": lambda s: bool(s.get("report.csv", {}).get("total")),
          "header": lambda s: bool(s.get("report.csv", {}).get("header"))}
    oracle = [call("fs.create", 0, path="report.csv"), call("report.set_total", 1, path="report.csv", total=42.5),
              call("report.set_header", 2, path="report.csv", header="ID,TOTAL")]
    s = await series_by_replay(oracle, lambda prefix: {k: f(replay(prefix)) for k, f in ms.items()})
    r = milestone_test(s, order=["created", "total", "header"], config=HorizonConfig(min_milestones=3))
    assert r["ok"] and r["first_step"] == {"created": 1, "total": 2, "header": 3}


def _episodes(rates: list[float], n: int, rng: random.Random, *, dependent: bool = False) -> list[frozenset]:
    out = []
    for _ in range(n):
        got: set[str] = set()
        for i, p in enumerate(rates):
            q = p * (0.35 if dependent and i == 2 else 1.0)  # in context, milestone 3 needs what milestone 1 left behind
            if rng.random() < q:
                got.add(f"m{i + 1}")
            else:
                break
        out.append(frozenset(got))
    return out


def test_three_independent_short_tasks_fail_the_concatenation_test():
    rng = random.Random(0)
    rates = [0.8, 0.7, 0.9]
    iso = {f"m{i + 1}": (round(p * 200), 200) for i, p in enumerate(rates)}
    order = ["m1", "m2", "m3"]
    r = concatenation_test(_episodes(rates, 300, rng), iso, order=order)
    assert not r["ok"] and not r["intrinsic"] and "concatenation" in r["reasons"][0]
    dep = concatenation_test(_episodes(rates, 300, random.Random(1), dependent=True), iso, order=order)
    assert dep["ok"] and dep["intrinsic"] and dep["milestones"][2]["dependent"]
    # a tiny sample cannot show dependence: it fails (unpowered is not evidence)
    assert not concatenation_test(_episodes(rates, 6, random.Random(2), dependent=True), {k: (v[0] // 40, 5) for k, v in iso.items()}, order=order)["ok"]
    assert not concatenation_test(_episodes(rates, 50, rng), {"m1": (1, 2)}, order=order)["ok"]  # milestones never posed alone


async def test_shortcut_probe_finds_a_one_call_bulk_path():
    oracle = [call("fs.create", 0, path="report.csv"), call("fs.read", 1, path="report.csv"), call("report.set_total", 2, path="report.csv", total=42.5),
              call("fs.read", 3, path="report.csv"), call("report.set_header", 4, path="report.csv", header="ID,TOTAL"), call("fs.read", 5, path="report.csv")]
    assert steps_of(oracle) == 6 and all(steps_of(s) < 3 for _, s in shortcut_candidates(oracle))
    clean = await shortcut_probe(oracle, verifier(STRONG))
    assert clean["ok"] and clean["attempts"] > 5 and clean["floor"] == 3
    bulk = await shortcut_probe(oracle, verifier(STRONG), extra=[("llm/bulk", [call("report.bulk", 0, path="report.csv")])])
    assert not bulk["ok"] and bulk["found"][0]["id"] == "llm/bulk" and bulk["found"][0]["steps"] == 1
    card = horizon_card(milestones={"ok": True, "reasons": [], "milestones": ["a"] * 5}, concatenation={"ok": True, "reasons": []}, shortcut=bulk,
                        expert_minutes=12.0)
    assert not card["ok"] and card["oracle_steps"] == 6 and card["expert_minutes"] == 12.0 and card["shortcut"]["found"] == ["llm/bulk"]
