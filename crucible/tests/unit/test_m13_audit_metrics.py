"""M13 audit regression tests for ``bench.{metrics, compare, report, run}`` and their callers (docs/audit/2026-10-13:
M13-study-defects-H1.md V1-V18, M13-sweep-G1.md D1-D14, M13-sweep-G2.md G2-1/6/7/8/18/19). Each test fails on the
M12 commit (b6d89fc) and passes after the fix.

Run directories are written with the real M10 writers (``RunJournal.finish``, ``RunRecord.mark_failure``, ``SampleKey``,
``attempt_id``), as the audit's ``synth.py`` does, or by the real ``Runner`` with a fake harness. A fuzz test checks the
estimators against an independent reference (product-form pass@k, brute-force max@k) on 50 seeded random runs."""

from __future__ import annotations

import asyncio
import html.parser
import itertools
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.bench import compare as cmp
from crucible.bench.metrics import Episode, final_milestones, first_irrecoverable_step, heatmap, load_run, progress_auc, regressions, stage_first_reached, stage_reach, summarize, summarize_group
from crucible.core.schema.run import Failure, RunRecord
from crucible.runtime.journal import RunJournal
from crucible.runtime.keys import SampleKey, attempt_id, derive_seed, identity

NOW = "2026-10-13T00:00:00Z"


# ---- synthetic run directories (the real writers) ------------------------------------------------------------------------
def journal(root: Path, run_id: str, tasks, policies=("p",), harnesses=("h",), k=1) -> RunJournal:
    doc = {"run_id": run_id, "tasks": list(tasks), "policies": list(policies), "harnesses": list(harnesses), "k": k, "run_seed": 0, "tier": "full"}
    return RunJournal(root, run_id, identity(doc), doc)


def put(j: RunJournal, task: str, idx: int = 0, n: int = 1, *, reward: float | None = 1.0, cost: float = 0.0, uncertain: float = 0.0,
        kind: str | None = None, detail: str | None = None, terminal: bool = True, policy: str = "p", harness: str = "h", meta: dict | None = None,
        steps: int = 1) -> tuple[RunRecord, Path]:
    sk = SampleKey(task=task, world="", policy=policy, harness=harness, sample_idx=idx, seed=derive_seed(0, task, policy, harness, idx), tier="full")
    rec = RunRecord(run_id=j.dir.name, task=task, world=task, policy=policy, seed=sk.seed, started_at=NOW, ended_at=NOW, steps=steps, outcome="completed",
                    reward=reward, cost_usd=cost, cost_uncertain_usd=uncertain, sample=sk.model_dump(mode="json"), attempt=attempt_id(sk.id(), n),
                    meta={"harness": harness, "timings": {"agent": 1.0}, **(meta or {})})
    if kind:
        rec.mark_failure(Failure(kind=kind, stage="agent", reason="x", detail=detail))
    adir = j.attempt_dir(sk.id(), n)
    adir.mkdir(parents=True, exist_ok=True)
    j.finish(rec, terminal=terminal)
    return rec, adir


def over(s: dict, policy: str = "p") -> dict:
    return next(m for m in s["models"] if m["policy"] == policy)["overall"]


def det(s: dict, policy: str = "p") -> dict:
    return over(s, policy)["deterministic"]


def ep(task, r, i=0, policy="p", harness="h", **kw) -> Episode:
    return Episode(task=task, policy=policy, harness=harness, sample_idx=i, attempt=f"{task}-{policy}-{harness}-{i}", reward=r, **kw)


class _Tag(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: list[tuple[str, list]] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, attrs))


# ---- H1 V1 / V2 / V3: progress rows ---------------------------------------------------------------------------------------
class _SSet:
    def __init__(self, truth):
        from crucible.verify.progress import Subgoal

        self.subgoals = [Subgoal("a", frozenset({"ticket:1"})), Subgoal("b", frozenset({"ticket:2"}))]
        self.truth = truth

    async def evaluate(self, world, ids):
        return {i: self.truth[i] for i in ids}


async def _track(tmp: Path, initial, script, end_truth=None):
    from crucible.verify.progress import ProgressTracker, read_rows

    truth = dict(initial)
    tr = ProgressTracker(_SSet(truth), out=tmp / "progress.jsonl", full_every=0)
    world = SimpleNamespace(log=[])
    await tr.on_start(world)
    for step, (tid, after) in enumerate(script, start=1):
        truth.update(after)
        await tr.on_step(world, step, "agent", SimpleNamespace(arguments={"ticket_id": tid}, tool_call_id=f"c{step}"), None)
    truth.update(end_truth or {})
    end = await tr.on_end(world)
    return read_rows(tmp / "progress.jsonl"), end


def test_v1_end_correction_row_is_the_end_state():
    rows = [{"step": 1, "phi": 0.5, "gained": ["a"], "lost": []}, {"step": 2, "phi": 1.0, "gained": ["b"], "lost": []},
            {"step": -1, "phi": 0.5, "gained": [], "lost": ["a"]}]
    assert final_milestones(rows) == {"b"} and regressions(rows) == 1
    assert stage_reach(final_milestones(rows), [("s", ["a", "b"])])[0] == -1
    hm = heatmap([ep("t", 0.5, steps=2, rows=rows, phi_final=0.5)])["p@h"]["t"]
    assert hm[0] == 0.0 and hm[-1] == 0.5  # Φ_0 first, the corrected Φ_T last
    assert progress_auc(rows, 2) == pytest.approx((0.5 + 0.5) / 2)  # Φ_1 = 0.5, Φ_2 corrected to 0.5


async def test_v1_real_tracker_drift_caught_at_the_end(tmp_path):
    rows, end = await _track(tmp_path, {"a": False, "b": False}, [(1, {"a": True, "b": True}), (2, {})], end_truth={"a": False})
    assert end["phi_final"] == 0.5 and final_milestones(rows) == {"b"} and regressions(rows) == 1


async def test_v2_start_state_is_a_row_and_counts(tmp_path):
    rows, end = await _track(tmp_path, {"a": True, "b": False}, [(2, {"b": True})])
    assert rows[0]["step"] == 0 and rows[0]["gained"] == ["a"] and rows[0]["phi"] == 0.5 and end["phi0"] == 0.5
    e = ep("t", 1.0, steps=1, rows=rows, phi_final=end["phi_final"])
    d = summarize_group([e], stages=[("s", ["a", "b"])])["deterministic"]
    assert d["stage_reach"] == 0.0 and d["stage_first_step"] == {"s": 1.0} and d["progress_auc"] >= 0.5
    rows2, _ = await _track(tmp_path / "x", {"a": True, "b": False}, [(2, {"b": True}), (1, {"a": False})])
    assert regressions(rows2) == 1  # losing a milestone held at the start is a regression
    assert abs(sum(r["delta_phi"] for r in rows2) - (0.5 - 0.5)) < 1e-9  # the stream still telescopes to Φ_T - Φ_0


def test_v3_auc_counts_the_state_each_step_reaches():
    assert progress_auc([{"step": 1, "phi": 1.0}], 1) == 1.0
    assert progress_auc([{"step": 1, "phi": 0.0}, {"step": 2, "phi": 1.0}], 2) == 0.5
    assert progress_auc([], 5) == 0.0


# ---- H1 V4 / G1 D4: spend of every attempt -----------------------------------------------------------------------------
class _RetryHarness:
    id = "fake@1"

    async def run(self, attempt):
        from crucible.runtime.harness.base import HarnessResult

        if attempt.n < 3:  # two infra faults, retried by the runner, each after $1.00 of model calls
            return HarnessResult(failure=Failure(kind="transport_timeout", stage="agent", reason="x"), cost_usd=1.0)
        return HarnessResult(reward=1.0, steps=1, cost_usd=0.10, cost_uncertain_usd=0.5)


async def test_v4_cost_counts_masked_retries_and_uncertain_spend(tmp_path):
    from crucible.runtime.run import Runner, RunSpec

    await Runner(tmp_path).run(RunSpec(run_id="c", tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p", estimate_usd=0.0)],
                                       harnesses=[_RetryHarness()], k=1, retry_backoff_s=0.0))
    c = over(summarize(load_run(tmp_path / "c")))["cost"]
    assert c["usd"] == pytest.approx(2.1) and c["per_success"]["usd"] == pytest.approx(2.1)
    assert c["usd_uncertain"] == pytest.approx(0.5) and c["per_success"]["usd_with_uncertain"] == pytest.approx(2.6)
    assert c["retries"] == 2 and c["attempts"] == 3  # G1 D9: the retry count


# ---- H1 V5: a stopped run is not the benchmark ------------------------------------------------------------------------
class _StopHarness:
    id = "fake@1"

    async def run(self, attempt):
        from crucible.runtime.harness.base import HarnessResult

        t = attempt.task.id
        if t.startswith("easy"):
            return HarnessResult(reward=1.0, steps=1, cost_usd=0.1)
        if t.startswith("flaky"):
            return HarnessResult(failure=Failure(kind="crucible:stale_image", stage="seed", reason="x"), cost_usd=0.1)
        return HarnessResult(reward=0.0, steps=1, cost_usd=0.1)


async def test_v5_stopped_run_reports_plan_owed_and_stop_reason(tmp_path):
    from crucible.bench.run import run_bench

    tasks = [SimpleNamespace(id=f"easy{i:02d}") for i in range(10)] + [SimpleNamespace(id=f"flaky{i:02d}") for i in range(10)] + \
            [SimpleNamespace(id=f"hard{i:02d}") for i in range(20)]
    res = await run_bench("demo", root=tmp_path, tasks=tasks, policies=[SimpleNamespace(id="p", estimate_usd=0.0)], harnesses=[_StopHarness()], k=1)
    s = res["summary"]
    assert s["run"]["stopped"].startswith("health") and s["run"]["planned"] == 40 and s["run"]["owed"] == 20 and not s["run"]["complete"]
    cov = det(s)["coverage"]
    assert cov["planned"] == 40 and cov["owed"] == 20 and cov["fully_masked_tasks"] == 30
    page = Path(res["paths"]["report"]).read_text()
    assert "the runner stopped" in page and "mask fraction" in page
    on_disk = json.loads(Path(res["paths"]["summary"]).read_text())["summary"]
    assert on_disk["run"]["stopped"].startswith("health") and any("planned samples have no result" in w for w in on_disk["warnings"])


def test_v5_killed_run_without_summary_is_incomplete(tmp_path):
    from crucible.bench.metrics import run_info

    j = journal(tmp_path, "r", ["a", "b"], k=2)
    put(j, "a", 0)
    put(j, "a", 1)
    info = run_info(tmp_path / "r")
    assert info["planned"] == 4 and info["owed"] == 2 and info["stopped"].startswith("incomplete")


# ---- H1 V6: torn tail ----------------------------------------------------------------------------------------------------
def test_v6_torn_tail_is_skipped_and_middle_corruption_raises(tmp_path):
    j = journal(tmp_path, "r", [f"t{i}" for i in range(6)])
    for i in range(6):
        put(j, f"t{i}")
    with (tmp_path / "r" / "results.jsonl").open("a") as f:
        f.write('{"run_id": "r", "ta')
    assert sum(not e.masked for e in load_run(tmp_path / "r")) == 6
    lines = (tmp_path / "r" / "results.jsonl").read_text().splitlines()
    (tmp_path / "r" / "results.jsonl").write_text("\n".join([lines[0], "{oops", *lines[1:6]]) + "\n")
    with pytest.raises(json.JSONDecodeError):
        load_run(tmp_path / "r")


# ---- H1 V7 / G1 D1 / D7 / G2-8: compare -----------------------------------------------------------------------------------
def test_v7_compare_refuses_several_models_and_duplicate_keys():
    two_harness = [ep(f"t{t}", float(i % 2), i, harness=h) for t in range(3) for h in ("h1", "h2") for i in range(4)]
    with pytest.raises(ValueError, match="harness"):
        cmp.compare(two_harness, list(reversed(two_harness)))
    one = [e for e in two_harness if e.harness == "h1"]
    r = cmp.compare(one, list(reversed(one)))
    assert r["flips"]["pass_to_fail"] == r["flips"]["fail_to_pass"] == 0 and r["mean_delta"] == 0.0 and r["support"]["tasks"] == 3
    with pytest.raises(ValueError, match="two episodes"):
        cmp.compare([ep("t", 1.0, 0), ep("t", 0.0, 0)], [ep("t", 1.0, 0)])


def test_g1_d1_cli_compare_refuses_an_ambiguous_run(tmp_path):
    from click.testing import CliRunner

    from crucible.cli.main import cli

    j = journal(tmp_path, "r", ["t"], policies=("p1", "p2"), k=2)
    for p, r in (("p1", 1.0), ("p2", 0.0)):
        for i in range(2):
            put(j, "t", i, reward=r, policy=p)
    res = CliRunner().invoke(cli, ["bench", "compare", str(tmp_path / "r"), str(tmp_path / "r"), "--json"])
    assert res.exit_code != 0 and "policy, harness" in res.output
    res = CliRunner().invoke(cli, ["bench", "compare", str(tmp_path / "r"), str(tmp_path / "r"), "--base-policy", "p1", "--cand-policy", "p2", "--json"])
    assert res.exit_code == 0 and json.loads(res.output)["flips"]["pass_to_fail"] == 1


def test_g1_d7_flips_are_per_task_not_per_sample_order():
    base = [ep(f"t{t}", float(i % 2), i, policy="A") for t in range(10) for i in range(2)]
    cand = [ep(f"t{t}", float((i + 1) % 2), i, policy="B") for t in range(10) for i in range(2)]  # same rates, swapped order
    r = cmp.compare(base, cand)
    assert r["flips"]["regressions"] == [] and r["flips"]["improvements"] == [] and r["mean_delta"] == 0.0 and r["mcnemar_p"] == 1.0


def test_g2_8_samples_of_a_task_are_not_independent_pairs():
    base = [ep(t, 0.0, i, policy="A") for t in ("t1", "t2") for i in range(8)]
    cand = [ep("t1", 1.0, i, policy="B") for i in range(8)] + [ep("t2", 0.0, i, policy="B") for i in range(8)]
    r = cmp.compare(base, cand)
    assert r["unit"] == "task" and r["mcnemar_p"] == 1.0 and r["permutation_p"] == 1.0 and r["delta_ci"]["approximate"]
    assert r["flips"]["fail_to_pass"] == 1  # one task improved, not 8 samples


# ---- H1 V8 / G1 D2 / D3 / G2-6 / G2-7: one task set, task-mean headline ----------------------------------------------------
def test_v8_macro_mean_and_one_task_set_for_every_k():
    eps = [ep("A", 1.0, i) for i in range(8)] + [ep("B", 0.0, 0)] + [ep("B", None, i) for i in range(1, 8)]
    d = summarize_group(eps, ks=(1, 2))["deterministic"]
    assert d["mean_reward"] == 0.5 and d["mean_reward_tasks"] == 2 and d["mean_reward_episodes"] == pytest.approx(8 / 9)
    assert d["pass@1"] == 0.5 and d["pass@2"] is None and d["pass^2"] is None and d["max@2"] is None
    assert d["pass@2_excluded"] == 1 and d["pass@2_tasks"] == 1 and d["pass@1_tasks"] == 2
    g26 = [ep("A", 0.0, i) for i in range(4)] + [ep("B", 1.0, 0)] + [ep("B", None, i) for i in range(1, 4)]
    d = summarize_group(g26, ks=(1, 2))["deterministic"]
    assert d["pass@1"] == 0.5 and d["pass@2"] is None  # never pass@2 < pass@1 on one row (G2-6)


# ---- H1 V9 / G1 D5: the CI covers ---------------------------------------------------------------------------------------
def test_v9_t_quantile_matches_tables():
    from crucible.bench.metrics import t_ppf

    for df, q in ((1, 12.706204736), (2, 4.302652730), (4, 2.776445105), (9, 2.262157163), (29, 2.045229642), (200, 1.971896224)):
        assert t_ppf(0.975, df) == pytest.approx(q, abs=1e-6)


def test_v9_ci_on_task_means_covers_at_five_tasks():
    rng = random.Random(5)
    hit = total = 0
    for _ in range(300):
        eps = []
        for t in range(5):
            p = rng.betavariate(1, 1)
            for i in range(4):
                eps.append(ep(f"t{t}", None if rng.random() < 0.2 else float(rng.random() < p), i))
        ci = summarize_group(eps)["deterministic"].get("mean_reward_ci")
        if ci is None:
            continue
        lo, hi = (ci["lo"], ci["hi"]) if isinstance(ci, dict) else ci
        total += 1
        hit += lo <= 0.5 <= hi
    assert total >= 290 and hit / total >= 0.92
    ci = summarize_group([ep(f"t{t}", float(t % 2)) for t in range(5)])["deterministic"]["mean_reward_ci"]
    assert ci["n_tasks"] == 5 and ci["approximate"]


# ---- H1 V10: probes from the World trace ----------------------------------------------------------------------------------
def test_v10_probe_attempts_read_from_the_trace(tmp_path):
    j = journal(tmp_path, "r", ["t"])
    _, adir = put(j, "t")
    (adir / "trace.json").write_text(json.dumps({"outcome": {"details": {"probes": {"attempted": {"git_log_all": 2, "reflog": 1}, "ran": {}, "exposed": {}}}}}))
    assert det(summarize(load_run(tmp_path / "r")))["probes"]["attempted"] == 3


# ---- H1 V11 / V17: recovery and first irrecoverable step ------------------------------------------------------------------
def test_v11_recovery_and_branches_reach_the_report(tmp_path):
    from crucible.bench.run import report_run

    j = journal(tmp_path, "r", ["t"], k=2)
    r0, _ = put(j, "t", 0)
    put(j, "t", 1, reward=0.0)
    (tmp_path / "r" / "branches.json").write_text(json.dumps({r0.attempt: [[2, 0, 8], [4, 0, 8]]}))
    d = det(report_run("x", tmp_path / "r", tags_of=lambda t: ["fault_start"])["summary"])
    assert d["recovery_rate"] == 0.5 and d["recovery_episodes"] == 2
    assert d["branches_insufficient"] == 1 and d["first_irrecoverable_step"] is None and d["irrecoverable_min_branches"] == 73


async def test_v11_run_bench_reads_fault_start_from_the_tasks(tmp_path):
    from crucible.bench.run import run_bench

    class H:
        id = "h@1"

        async def run(self, attempt):
            from crucible.runtime.harness.base import HarnessResult

            return HarnessResult(reward=1.0, steps=1)

    res = await run_bench("f", root=tmp_path, tasks=[SimpleNamespace(id="t", tags=["fault_start"])], policies=[SimpleNamespace(id="p", estimate_usd=0.0)],
                          harnesses=[H()], k=1)
    assert det(res["summary"])["recovery_rate"] == 1.0


def test_v17_first_irrecoverable_step_is_a_suffix_condition():
    assert first_irrecoverable_step([(3, 0, 80), (5, 4, 8)]) is None
    assert first_irrecoverable_step([(3, 0, 80), (5, 0, 80)]) == 3
    assert first_irrecoverable_step([(1, 5, 8), (3, 0, 80), (5, 0, 80)]) == 3


# ---- H1 V12: the mutant classes the old tests let through ---------------------------------------------------------------
def test_v12_masked_never_enters_mean_coverage_gates_or_recovery():
    eps = [ep("t", 1.0, 0, gates={"g": 1}, tags=["fault_start"]), ep("t", None, 1, gates={"g": 0}, tags=["fault_start"]),
           ep("u", 0.0, 0, gates={"g": 0})]
    d = summarize_group(eps)["deterministic"]
    assert d["mean_reward"] == 0.5 and d["coverage"]["measured"] == 2 and d["coverage"]["masked"] == 1
    assert d["gate_profile"]["gates"]["g"] == {"passed": 1, "total": 2, "pass_rate": 0.5}
    assert d["recovery_rate"] == 1.0


def test_v12_compare_drops_masked_and_load_run_keeps_the_latest_terminal_row(tmp_path):
    r = cmp.compare([ep("t", 1.0, 0), ep("u", 1.0, 0)], [ep("t", None, 0, policy="q"), ep("u", 1.0, 0, policy="q")])
    assert r["support"]["tasks"] == 1 and r["flips"]["pass_to_fail"] == 0
    j = journal(tmp_path, "r", ["t"])
    put(j, "t", 0, 1, reward=0.0)
    put(j, "t", 0, 2, reward=1.0)  # a regrade: the newest terminal row wins
    (e,) = load_run(tmp_path / "r")
    assert e.reward == 1.0 and e.attempt.endswith("-a2")


def test_v12_max_at_k_exposure_ci_cost_heatmap_and_stage_reset():
    from crucible.bench.metrics import max_at_k

    assert max_at_k([ep("t", 1.0, 0, exposed=True), ep("t", 0.5, 1)], 1) == 0.25
    ci = summarize_group([ep(f"t{i}", 0.9 + 0.01 * i) for i in range(10)])["deterministic"]["mean_reward_ci"]
    lo, hi = (ci["lo"], ci["hi"]) if isinstance(ci, dict) else ci
    assert 0.9 < lo < hi < 1.0
    c = summarize_group([ep("t", 1.0, 0, cost_usd=1.0), ep("t", None, 1, cost_usd=3.0)])["cost"]
    assert c["usd"] == 4.0 and c["per_success"]["usd"] == 4.0
    rows = [{"step": 1, "phi": 1.0, "gained": ["a", "b"], "lost": []}]
    assert heatmap([ep("t", 1.0, steps=1, rows=rows, phi_final=1.0)], bins=2)["p@h"]["t"] == [0.0, 1.0]
    rows = [{"step": 1, "phi": 0.5, "gained": ["a"], "lost": []}, {"step": 2, "phi": 0.0, "gained": [], "lost": ["a"]},
            {"step": 3, "phi": 0.5, "gained": ["b"], "lost": []}, {"step": 4, "phi": 1.0, "gained": ["a"], "lost": []}]
    assert stage_first_reached(rows, [("s", ["a", "b"])]) == {"s": 4}


# reference implementation (independent of crucible.bench): product-form pass@k, brute-force max@k
def _ref_pass_at(n, c, k):
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def _ref_max_at(xs, k):
    subs = list(itertools.combinations(xs, k))
    return sum(max(s) for s in subs) / len(subs)


@dataclass
class _Truth:
    task: str
    policy: str
    idx: int
    final: float | None = None
    owed: bool = False
    spend: list = field(default_factory=list)


def _random_run(root: Path, rng: random.Random) -> tuple[list[_Truth], int]:
    tasks = [f"t{i}" for i in range(rng.randint(1, 5))]
    pols = ("p", "q") if rng.random() < 0.3 else ("p",)
    k = rng.randint(1, 4)
    p_mask = rng.choice([0.0, 0.2, 0.6, 1.0])
    cont = rng.random() < 0.3
    j = journal(root, "r", tasks, policies=pols, k=k)
    out = []
    for t in tasks:
        pt = rng.random()
        for p in pols:
            for i in range(k):
                tr = _Truth(t, p, i)
                n = 0
                while rng.random() < 0.3 and n < 2:
                    n += 1
                    c = round(rng.uniform(0.01, 1.0), 4)
                    put(j, t, i, n, reward=None, cost=c, kind="transport_timeout", terminal=False, policy=p)
                    tr.spend.append(c)
                if rng.random() < 0.1:
                    tr.owed = True
                    out.append(tr)
                    continue
                n += 1
                c = round(rng.uniform(0.01, 1.0), 4)
                reward = round(rng.random(), 3) if cont else float(rng.random() < pt)
                u = rng.random()
                kind = "crucible:backend_error" if u < p_mask else ("agent_timeout" if u < p_mask + 0.1 else None)
                put(j, t, i, n, reward=reward, cost=c, kind=kind, policy=p)
                tr.final = None if kind == "crucible:backend_error" else reward
                tr.spend.append(c)
                out.append(tr)
    return out, k


def test_v12_fuzz_against_a_reference(tmp_path):
    for seed in range(50):
        rng = random.Random(seed)
        truths, k = _random_run(tmp_path / str(seed), rng)
        ks = tuple(range(1, k + 1))
        s = summarize(load_run(tmp_path / str(seed) / "r"), ks=ks)
        for m in s["models"]:
            mine = [t for t in truths if t.policy == m["policy"]]
            d, c = m["overall"]["deterministic"], m["overall"]["cost"]
            scored = [t for t in mine if t.final is not None]
            assert d["coverage"]["measured"] == len(scored) and d["coverage"]["owed"] == sum(t.owed for t in mine)
            assert d["coverage"]["masked"] == len(mine) - len(scored) - sum(t.owed for t in mine)
            assert c["usd"] == pytest.approx(sum(sum(t.spend) for t in mine), abs=1e-6)
            if not scored:
                assert "mean_reward" not in d
                continue
            by: dict[str, list[float]] = {}
            for t in scored:
                by.setdefault(t.task, []).append(t.final)
            assert d["mean_reward"] == pytest.approx(sum(sum(v) / len(v) for v in by.values()) / len(by), abs=1e-8)
            for kk in ks:
                if any(len(v) < kk for v in by.values()):
                    assert d[f"pass@{kk}"] is None and d[f"pass@{kk}_excluded"] == sum(len(v) < kk for v in by.values())
                    continue
                want = sum(_ref_pass_at(len(v), sum(x >= 1.0 for x in v), kk) for v in by.values()) / len(by)
                assert d[f"pass@{kk}"] == pytest.approx(want, abs=1e-8)
                want = sum(math.comb(sum(x >= 1.0 for x in v), kk) / math.comb(len(v), kk) for v in by.values()) / len(by)
                assert d[f"pass^{kk}"] == pytest.approx(want, abs=1e-8)
                assert d[f"max@{kk}"] == pytest.approx(sum(_ref_max_at(v, kk) for v in by.values()) / len(by), abs=1e-8)


# ---- H1 V13: digest by path ------------------------------------------------------------------------------------------------
def test_v13_digest_keeps_a_task_and_gate_named_cost():
    from crucible.bench.metrics import deterministic_digest

    def run(step, usd=0.0):  # a stage named "cost" first reached at ``step``: the only difference sits under that key
        e = ep("cost", 1.0, steps=2, cost_usd=usd, rows=[{"step": step, "phi": 1.0, "gained": ["a"], "lost": []}])
        return summarize([e], stages_of=lambda t: [("cost", ["a"])])

    assert deterministic_digest(run(1)) != deterministic_digest(run(2))
    assert deterministic_digest(run(1)) == deterministic_digest(run(1, usd=5.0))  # spend alone never moves the digest


# ---- H1 V14 / V15: policy actors and unobserved GPU -----------------------------------------------------------------------
def test_v14_policy_actors_come_from_the_attempt(tmp_path):
    j = journal(tmp_path, "r", ["t"])
    _, adir = put(j, "t")
    (adir / "calls.json").write_text(json.dumps({"buyer": [], "seller": []}))
    ev = [{"actor": "buyer", "type": "tool_call"}, {"actor": "buyer", "type": "message", "data": {"to": "seller"}},
          {"actor": "seller", "type": "message", "data": {"to": "user"}}, {"actor": "user", "type": "message"}]
    (adir / "trace.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    d = det(summarize(load_run(tmp_path / "r")))
    assert d["tool_calls"] == 1 and d["messages"] == 2 and d["peer_messages"] == 1


def test_v15_gpu_seconds_unobserved_is_none():
    c = summarize_group([ep("t", 1.0)])["cost"]
    assert c["gpu_s"] is None and c["per_success"]["gpu_s"] is None and c["observability"]["gpu_s"] == 0.0 and "wall_s" in c["observability"]


# ---- H1 V16 / G1 D6: ks from k ------------------------------------------------------------------------------------------------
def test_v16_ks_derive_from_k(tmp_path):
    from click.testing import CliRunner

    from crucible.bench.metrics import default_ks
    from crucible.bench.run import report_run
    from crucible.cli.main import cli

    assert default_ks(2) == (1, 2) and default_ks(8) == (1, 2, 4, 8) and default_ks(6) == (1, 2, 4, 6)
    j = journal(tmp_path, "r", ["t"], k=2)
    put(j, "t", 0)
    put(j, "t", 1, reward=0.0)
    d = det(report_run("x", tmp_path / "r")["summary"])
    assert d["pass@2"] == 1.0 and d["pass^2"] == 0.0
    res = CliRunner().invoke(cli, ["bench", "report", str(tmp_path / "r"), "--ks", "1,2", "--json"])
    assert res.exit_code == 0 and json.loads(res.output)["summary"]["ks"] == [1, 2]


# ---- H1 V18 / G1 D14: presentation, McNemar ------------------------------------------------------------------------------
def test_v18_mcnemar_unrounded_and_formatted():
    assert 0 < cmp.mcnemar_exact(0, 40) < 1e-9 and cmp.fmt_p(cmp.mcnemar_exact(0, 40)) == "< 1e-9"
    assert cmp.mcnemar_exact(0, 6) == pytest.approx(2 / 64)


def test_g1_d14_mcnemar_large_counts_fast_and_exact():
    t0 = time.monotonic()
    p = cmp.mcnemar_exact(4999, 5201)
    assert time.monotonic() - t0 < 1.0
    n, k = 1201, 560
    exact = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    assert cmp.mcnemar_exact(k, n - k) == pytest.approx(exact, rel=1e-9) and 0.04 < p < 0.06


def test_v18_pareto_and_stage_reach_normalized():
    from crucible.bench.report import _pareto_svg

    svg = _pareto_svg([("a", 1.0, 2.0), ("b", 2.0, -1.0)])
    ys = [float(x.split('"')[0]) for x in svg.split('cy="')[1:]]
    assert all(36 <= y <= 220 - 36 for y in ys)
    eps = [ep("t1", 1.0, rows=[{"step": 1, "phi": 1.0, "gained": ["a"], "lost": []}], phi_final=1.0),
           ep("t2", 1.0, rows=[{"step": 1, "phi": 1.0, "gained": ["a"], "lost": []}], phi_final=1.0)]
    stages = {"t1": [("s1", ["a"])], "t2": [("s1", ["a"]), ("s2", ["b"]), ("s3", ["c"])]}
    d = summarize_group(eps, stages_of=stages.get)["deterministic"]
    assert d["stage_reach"] is None and d["stage_reach_frac"] == pytest.approx((1.0 + 1 / 3) / 2)


# ---- G1 D9 / D10 / D11 / D12 / D13 ---------------------------------------------------------------------------------------
def test_g1_d9_masks_by_kind_and_retries(tmp_path):
    j = journal(tmp_path, "r", ["t", "u"], k=2)
    put(j, "t", 0, 1, reward=None, kind="transport_timeout", terminal=False)
    put(j, "t", 0, 2)
    put(j, "t", 1, kind="crucible:backend_error", reward=None)
    put(j, "u", 0, kind="crucible:stale_image", reward=None)
    put(j, "u", 1, reward=0.0, kind="agent_timeout")
    s = summarize(load_run(tmp_path / "r"))
    assert det(s)["coverage"]["masked_by_kind"] == {"crucible:backend_error": 1, "crucible:stale_image": 1}
    assert det(s)["failures"]["scored_by_kind"] == {"agent_timeout": 1} and over(s)["cost"]["retries"] == 1


def test_g1_d10_d11_spread_and_subsets():
    eps = [ep(f"t{t}", float(i == 0), i) for t in range(4) for i in range(3)]
    s = summarize(eps, subsets_of=lambda t: ["even" if int(t[1]) % 2 == 0 else "odd"])
    sp = det(s)["spread"]
    assert sp["repeats"] == 3 and sp["std_across_repeats"] == pytest.approx(math.sqrt(1 / 3)) and sp["tasks_with_repeats"] == 4
    assert set(s["models"][0]["subsets"]) == {"even", "odd"} and s["models"][0]["subsets"]["odd"]["deterministic"]["coverage"]["tasks"] == 2


def test_g1_d12_tokens_split_and_latency(tmp_path):
    j = journal(tmp_path, "r", ["t"], k=2)
    for i, wall in ((0, 2.0), (1, 4.0)):
        _, adir = put(j, "t", i, meta={"timings": {"agent": wall}})
        (adir / "calls.json").write_text(json.dumps({"agent": [{"usage": {"prompt_tokens": 100, "completion_tokens": 10}}]}))
    c = over(summarize(load_run(tmp_path / "r")))["cost"]
    assert c["tokens_in"] == 200 and c["tokens_out"] == 20 and c["perf"]["latency_s_p50"] == 2.0 and c["perf"]["latency_s_p99"] == 4.0


async def test_g1_d13_attempt_timeout_masks_under_its_own_detail(tmp_path):
    from crucible.runtime.infra import RetryRule
    from crucible.runtime.run import Runner, RunSpec

    class Slow:
        id = "slow@1"

        async def run(self, attempt):
            await asyncio.sleep(5)

    await Runner(tmp_path).run(RunSpec(run_id="r", tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p", estimate_usd=0.0)],
                                       harnesses=[Slow()], k=1, attempt_timeout_s=0.05, retry=RetryRule(max_attempts=1)))
    d = det(summarize(load_run(tmp_path / "r")), "p")
    assert d["coverage"]["masked_by_kind"] == {"agent_run_error/attempt_timeout": 1}


# ---- G1 §4: exports -----------------------------------------------------------------------------------------------------
def test_g1_exports_harbor_nemo_gym_and_dual_headline(tmp_path):
    from crucible.bench.report import headline
    from crucible.bench.run import report_run

    j = journal(tmp_path, "r", ["t", "u"], k=2)
    put(j, "t", 0)
    put(j, "t", 1, reward=0.0)
    put(j, "u", 0, reward=None, kind="crucible:backend_error")
    put(j, "u", 1, reward=1.0)
    res = report_run("x", tmp_path / "r")
    hb = json.loads(Path(res["paths"]["harbor"]).read_text())
    ev = hb["evals"]["h__p__crucible"]
    assert hb["n_errored_trials"] == 1 and ev["n_trials"] == 3 and ev["exception_stats"] == {"crucible:backend_error": [ev["exception_stats"]["crucible:backend_error"][0]]}
    assert ev["metrics"] == [{"mean": 0.75}] and "excluded" in hb["crucible"]["masked"]
    ng = json.loads(Path(res["paths"]["nemo_gym"]).read_text())[0]
    assert ng["agent_metrics"]["coverage/masked_rollouts"] == 1 and ng["agent_metrics"]["pass@1/accuracy"] == 75.0
    assert [r["repeat"] for r in ng["repeat_level_metrics"]] == [0, 1]
    assert headline(det(res["summary"])).startswith("0.750 (0.500) over 2 tasks")


# ---- G2-1 / G2-19: exposure is a scored failure; common support ---------------------------------------------------------
def test_g2_1_exposed_counts_as_failure_and_hacker_does_not_outrank(tmp_path):
    j = journal(tmp_path, "r", [f"t{i}" for i in range(10)], policies=("honest", "hacker"))
    for i in range(10):
        put(j, f"t{i}", reward=1.0 if i < 5 else 0.0, policy="honest")
        if i < 5:
            put(j, f"t{i}", reward=1.0, policy="hacker")
        else:
            put(j, f"t{i}", reward=None, kind="crucible:answer_exposed", policy="hacker")
    s = summarize(load_run(tmp_path / "r"))
    h, x = det(s, "honest"), det(s, "hacker")
    assert x["pass@1"] == h["pass@1"] == 0.5 and x["mean_reward"] == 0.5
    assert x["exposure"] == {"exposed": 5, "measured": 10, "rate": 0.5} and x["coverage"]["measured"] == 10
    assert s["common_support"]["tasks"] == 10 and s["common_support"]["models"]["hacker@h"]["pass@1"] == 0.5


def test_g2_19_exposed_reward_never_counts():
    d = summarize_group([ep("t", 1.0, 0, exposed=True), ep("t", 0.0, 1)])["deterministic"]
    assert d["mean_reward"] == 0.0 and d["pass@1"] == 0.0
    r = cmp.compare([ep("t", 0.0, 0)], [ep("t", 1.0, 0, policy="q", exposed=True)])
    assert r["mean_delta"] == 0.0


def test_g2_1_common_support_drops_tasks_one_model_lacks():
    eps = [ep("a", 1.0, policy="m1"), ep("b", 0.0, policy="m1"), ep("a", 1.0, policy="m2"), ep("b", None, policy="m2")]
    cs = summarize(eps)["common_support"]
    assert cs["tasks"] == 1 and cs["models"]["m2@h"]["dropped_tasks"] == [] and cs["models"]["m1@h"]["dropped_tasks"] == ["b"]
    r = cmp.compare([e for e in eps if e.policy == "m1"], [e for e in eps if e.policy == "m2"])
    assert r["support"] == {"tasks": 1, "base_only": ["b"], "cand_only": [], "neither": []}


# ---- G2-18: configuration block --------------------------------------------------------------------------------------------
async def test_g2_18_summary_carries_the_configuration(tmp_path):
    from crucible.bench.run import run_bench

    class H:
        id = "h@2.1"

        async def run(self, attempt):
            from crucible.runtime.harness.base import HarnessResult

            return HarnessResult(reward=1.0, steps=1)

    res = await run_bench("cfg", root=tmp_path, tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p", estimate_usd=0.0)], harnesses=[H()],
                          k=2, config={"judge": {"model": "j", "temperature": 0}})
    cfg = res["summary"]["config"]
    assert cfg["k"] == 2 and cfg["ks"] == [1, 2] and cfg["harnesses"] == [{"id": "h@2.1"}] and cfg["judge"] == {"model": "j", "temperature": 0}
    assert cfg["threshold"] == 1.0 and cfg["run_seed"] == 0 and "Configuration" in Path(res["paths"]["report"]).read_text()


# ---- determinism and escaping of the new sections ----------------------------------------------------------------------
def test_report_new_sections_escape_and_render_deterministically(tmp_path):
    from crucible.bench.run import report_run

    bad = "<img src=x onerror=alert(1)>"
    for d in ("a", "b"):
        j = journal(tmp_path / d, "r", [bad, "u"], policies=("'><script>alert(1)</script>",), k=1)
        put(j, bad, reward=None, kind="crucible:backend_error", policy="'><script>alert(1)</script>")
        put(j, "u", reward=1.0, policy="'><script>alert(1)</script>")
    pages = [Path(report_run("t", tmp_path / d / "r")["paths"]["report"]).read_text() for d in ("a", "b")]
    assert pages[0] == pages[1]
    p = _Tag()
    p.feed(pages[0])
    assert not any(t in ("script", "img") for t, _ in p.tags) and not any(a.startswith("on") for _, attrs in p.tags for a, _ in attrs)
    assert "Masked samples by kind" in pages[0] and "Per task" in pages[0] and "Read before using these numbers" in pages[0]


# ---- M13 bench campaign (M13-bench-campaign.md) ------------------------------------------------------------------------
class _ControlHarness:
    """Reward by policy: the oracle solves every task but ``broken`` (B5), the no-op solves ``vacuous`` (B4)."""

    id = "ctl@1"

    async def run(self, attempt):
        from crucible.runtime.harness.base import HarnessResult

        t, p = attempt.task.id, attempt.policy.id
        if p == "control:oracle":
            return HarnessResult(reward=0.0 if t == "broken" else 1.0, steps=1)
        if p == "control:noop":
            return HarnessResult(reward=1.0 if t == "vacuous" else 0.0, steps=1)
        return HarnessResult(reward=1.0 if t in ("vacuous", "ok1") else 0.0, steps=1)


async def test_b4_b5_controls_exclude_vacuous_and_broken_tasks(tmp_path):
    from crucible.bench.run import run_bench

    tasks = [SimpleNamespace(id=t, oracle=[{"function_name": "x.y"}]) for t in ("ok1", "ok2", "vacuous", "broken")]
    pols = [SimpleNamespace(id="m1", estimate_usd=0.0), SimpleNamespace(id="m2", estimate_usd=0.0)]
    res = await run_bench("v", root=tmp_path, tasks=tasks, policies=pols, harnesses=[_ControlHarness()], k=2, controls=True)
    s = res["summary"]
    assert s["invalid_tasks"]["vacuous"]["reasons"] == ["noop_passes"] and s["invalid_tasks"]["broken"]["reasons"] == ["broken_oracle"]
    d = det(s, "m1")
    assert d["coverage"]["tasks"] == 2 and d["mean_reward"] == 0.5  # ok1 1.0, ok2 0.0; vacuous no longer lifts it
    assert s["validity"]["status"] == "manifest" and (tmp_path / "bench-v" / "validity.json").exists()
    assert set(res["extras"]["irt"]["respondents"]) == {"m1@ctl@1", "m2@ctl@1"} and "vacuous" not in res["extras"]["irt"]["items"]
    assert any("invalid tasks are excluded" in w for w in s["warnings"])
    base = [e for e in res["episodes"] if e.policy == "m1"]
    cand = [e for e in res["episodes"] if e.policy == "m2"]
    r = cmp.compare(base, cand, exclude_tasks=s["invalid_tasks"])
    assert r["invalid_tasks"] == ["broken", "vacuous"] and r["support"]["tasks"] == 2


def test_b4_no_manifest_says_validity_unknown():
    s = summarize([ep("t", 1.0)])
    assert s["validity"]["status"] == "unknown"
    from crucible.bench.report import warnings

    assert any(w.startswith("validity unknown") for w in warnings(s))


def test_b4_validity_from_controls_reasons():
    from crucible.bench.metrics import validity_from_controls

    eps = [ep("a", 1.0, i, policy="o") for i in range(3)] + [ep("b", float(i % 2), i, policy="o") for i in range(3)] + \
          [ep("c", 1.0, 0, policy="n"), ep("a", 0.0, 0, policy="n")]
    v = validity_from_controls(eps, oracle="o", noop="n")
    assert v["a"]["valid"] is True and v["b"]["reasons"] == ["flaky_oracle"] and v["c"]["valid"] is False and v["c"]["reasons"] == ["noop_passes"]


def test_b7_duplicate_task_ids_refused_and_live_dirs_kept(tmp_path):
    from crucible.runtime.run import Runner, RunSpec

    rs = RunSpec(run_id="r", tasks=[SimpleNamespace(id="harbor/task_00000"), SimpleNamespace(id="harbor/task_00000")],
                 policies=[SimpleNamespace(id="p")], harnesses=[SimpleNamespace(id="h")])
    with pytest.raises(ValueError, match="duplicate task ids"):
        Runner(tmp_path).plan(rs)
    j = journal(tmp_path, "j", ["t"])
    live = j.attempt_dir("sid", 1)
    live.mkdir(parents=True)
    j.live.add(live)
    assert j.next_attempt("sid") == 2 and live.exists()


async def test_b10_report_saved_keeps_the_digest_and_the_original(tmp_path):
    from crucible.bench.run import report_saved, run_bench

    class H:
        id = "h@1"

        async def run(self, attempt):
            from crucible.runtime.harness.base import HarnessResult

            return HarnessResult(reward=1.0, steps=2)

    stages = {"t": [("s1", ["a"])]}
    res = await run_bench("b10", root=tmp_path, tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p", estimate_usd=0.0)], harnesses=[H()],
                          k=2, stages_of=stages.get, horizon_of=lambda t: "h3", extras={"manifest": {"counts": 1}})
    orig = Path(res["paths"]["summary"]).read_text()
    again = report_saved(tmp_path / "bench-b10")
    assert json.loads(Path(again["paths"]["summary"]).read_text())["digest"] == json.loads(orig)["digest"]
    assert Path(again["paths"]["summary"]).parent.name == "bench-report" and Path(res["paths"]["summary"]).read_text() == orig
    assert json.loads(Path(again["paths"]["summary"]).read_text())["manifest"] == {"counts": 1}


def test_b14_cli_errors_on_bad_input(tmp_path):
    from click.testing import CliRunner

    from crucible.cli.main import cli

    j = journal(tmp_path, "r", ["t"])
    put(j, "t")
    res = CliRunner().invoke(cli, ["bench", "compare", str(tmp_path / "r"), str(tmp_path / "r"), "--base-policy", "typo"])
    assert res.exit_code != 0 and "no base episodes" in res.output
    (tmp_path / "empty").mkdir()
    res = CliRunner().invoke(cli, ["bench", "report", str(tmp_path / "empty")])
    assert res.exit_code != 0 and "holds no run" in res.output


def test_b12_hack_metrics_count_tampering_and_flagged_passes(tmp_path):
    j = journal(tmp_path, "r", ["t", "u"])
    put(j, "t", reward=0.0, kind="crucible:policy_misbehavior")
    _, adir = put(j, "u", reward=1.0)
    (adir / "rollout.json").write_text(json.dumps({"flags": [{"rule": "intercepted:grader_paths"}]}))
    h = det(summarize(load_run(tmp_path / "r")))["hacks"]
    assert h["grader_tampered"] == 1 and h["flagged_passes"] == 1 and h["flag_rules"] == {"intercepted:grader_paths": 1} and h["rate"] == 1.0


def test_b17_cli_benchmarks_plain_tasks_with_controls(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from crucible.cli.main import cli

    (tmp_path / "b17mod.py").write_text('''
from crucible_pack_toy.longhorizon.task import BacklogTask
from crucible.actors.policy import ScriptedPolicy
from crucible.runtime.harness.native import PolicyRef
_T = {}

def tasks():
    if not _T:
        _T["t"] = BacklogTask(seed=3, n=4)
    return [_T["t"]]

def policies():
    return [PolicyRef(id="noop", make=lambda a: {"agent": ScriptedPolicy("agent", [])})]
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    res = CliRunner().invoke(cli, ["bench", "run", "--tasks", "b17mod:tasks", "--policies", "b17mod:policies", "--k", "1", "--out", str(tmp_path / "o"), "--json"],
                             catch_exceptions=False)
    assert res.exit_code == 0, res.output
    s = json.loads(res.output)["summary"]
    assert s["validity"]["status"] == "manifest" and s["validity"]["valid_tasks"] == 1 and det(s, "noop")["mean_reward"] == 0.0


def test_b16_controls_stay_out_of_irt(tmp_path):
    from crucible.bench.run import report_run

    j = journal(tmp_path, "r", ["t", "u"], policies=("control:oracle", "m1", "m2"))
    for t in ("t", "u"):
        put(j, t, policy="control:oracle")
        put(j, t, policy="m1", reward=1.0 if t == "t" else 0.0)
        put(j, t, policy="m2", reward=0.0)
    res = report_run("x", tmp_path / "r")
    assert res["extras"]["irt"]["respondents"] == ["m1@h", "m2@h"] and next(m for m in res["summary"]["models"] if m["policy"] == "control:oracle")["control"]


def test_v12_anchors_saved_and_reused_by_the_cli(tmp_path):
    """dynamic batch V12 at the call site: ``bench run`` saves the anchor parameters; ``--anchors`` links the next run."""
    from click.testing import CliRunner

    from crucible.cli.main import cli

    res = CliRunner().invoke(cli, ["bench", "run", "toy", "--fresh", "0", "--k", "1", "--no-controls", "--out", str(tmp_path / "a"), "--json"],
                             catch_exceptions=False)
    assert res.exit_code == 0, res.output
    anchors = Path(json.loads(res.output)["paths"]["anchors"])
    assert anchors.exists() and json.loads(anchors.read_text())["meta"]["bench"] == "toy-bench"
    res = CliRunner().invoke(cli, ["bench", "run", "toy", "--fresh", "0", "--k", "1", "--no-controls", "--anchors", str(anchors), "--out", str(tmp_path / "b"),
                                   "--json"], catch_exceptions=False)
    assert res.exit_code == 0 and json.loads(res.output)["summary"]["validity"]["status"] == "unknown"
