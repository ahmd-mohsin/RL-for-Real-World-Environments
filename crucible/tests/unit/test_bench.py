"""M13 bench. The exit gate's toy part: a toy benchmark is generated (2 fixed anchors + 4 fresh items from HMAC-keyed
held-out seeds, each validated: gold solves it, no-op does not) and reported, for three policies. The policies are the
scripted oracle, a half-done oracle and a no-op; the backlog World gives every step a Φ.

Every step metric is checked, along with IRT ability order, splits, contamination hits and a paired comparison.
A re-run in a fresh directory gives the identical deterministic digest and byte-identical HTML. Unit checks cover the
estimators: pass@k / pass^k / max@k, progress AUC, regressions, first irrecoverable step, exact McNemar, IRT recovery
on simulated data, fixed-anchor calibration and drift."""

from __future__ import annotations

import json
import math
import random

import pytest

from crucible.bench import compare, contamination, dynamic, irt, splits
from crucible.bench.metrics import Episode, first_irrecoverable_step, max_at_k, pass_at_k, pass_hat_k, progress_auc, regressions


# ---- estimators ---------------------------------------------------------------------------------------------------------
def ep(task, r, i=0, policy="p", rows=(), steps=10):
    return Episode(task=task, policy=policy, harness="h", sample_idx=i, attempt=f"{task}-{policy}-{i}", reward=r, rows=list(rows), steps=steps)


def test_estimators():
    eps = [ep("t", 1.0, 0), ep("t", 0.0, 1), ep("t", 1.0, 2), ep("t", None, 3)]  # one masked: not a sample
    assert pass_at_k(eps, 1) == pytest.approx(2 / 3) and pass_at_k(eps, 2) == pytest.approx(1.0)
    assert pass_hat_k(eps, 2) == pytest.approx(1 / 3)  # C(2,2)/C(3,2)
    assert max_at_k([ep("t", 0.2, 0), ep("t", 0.6, 1), ep("t", 0.4, 2)], 2) == pytest.approx((0.4 * 1 + 0.6 * 2) / 3)
    rows = [{"step": 2, "phi": 0.5, "gained": ["a"], "lost": []}, {"step": 6, "phi": 0.25, "gained": [], "lost": ["a"]},
            {"step": 8, "phi": 1.0, "gained": ["a", "b"], "lost": []}]
    assert progress_auc(rows, 10) == pytest.approx((0.5 * 4 + 0.25 * 2 + 1.0 * 2) / 10)
    assert regressions(rows) == 1
    assert first_irrecoverable_step([(3, 4, 8), (4, 0, 40), (5, 0, 80)]) == 5  # 0/40 is not yet implausible (Wilson upper 0.088)
    assert compare.mcnemar_exact(0, 6) == pytest.approx(2 / 64) and compare.mcnemar_exact(3, 3) == 1.0


def test_irt_recovers_and_anchors():
    rng = random.Random(0)
    thetas = {f"m{j}": rng.gauss(0, 1) for j in range(24)}
    items = {f"i{k}": irt.Item(b=rng.uniform(-2, 2), a=rng.uniform(0.7, 2.0)) for k in range(30)}
    counts = {}
    for m, t in thetas.items():
        for i, it in items.items():
            n = 8
            s = sum(rng.random() < 1 / (1 + math.exp(-it.a * (t - it.b))) for _ in range(n))
            counts[(m, i)] = (float(s), n)
    f = irt.fit(counts, model="2pl")

    def corr(x, y):
        mx, my = sum(x) / len(x), sum(y) / len(y)
        return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))

    assert corr([f.theta[m] for m in thetas], list(thetas.values())) > 0.9
    assert corr([f.items[i].b for i in items], [it.b for it in items.values()]) > 0.9
    # fixed-anchor calibration: the 10 anchors keep their parameters; fresh items land on the same scale
    anchors = {i: items[i] for i in list(items)[:10]}
    g = irt.fit(counts, anchors=anchors, model="2pl")
    assert all(g.items[i].b == anchors[i].b for i in anchors)
    assert corr([g.items[i].b for i in items if i not in anchors], [items[i].b for i in items if i not in anchors]) > 0.9
    th, se = irt.ability(g.items, {i: counts[("m0", i)] for i in items})
    assert abs(th - thetas["m0"]) < 3 * se + 0.3
    # an anchor whose difficulty moved (contaminated: everyone now solves it) is quarantined
    moved = {k: ((float(v[1]), v[1]) if k[1] == "i0" else v) for k, v in counts.items()}
    assert "i0" in irt.drift(anchors, moved) and irt.next_item(g.items, 0.0) in items


def test_splits_and_contamination():
    items = [splits.SplitItem(f"t{i}", lineage=f"L{i // 3}", axes={"repo": "pcl" if i in (0, 1, 2) else "r"}) for i in range(30)]
    items.append(splits.SplitItem("task04", showcase=True))
    items += [splits.SplitItem("pr1", source_kind="pr", source_time="2026-08-01T00:00:00Z"), splits.SplitItem("pr0", source_kind="pr", source_time="2025-01-01T00:00:00Z")]
    r = splits.assign(items, salt=b"s3cret", bench_frac=0.3, ood={"repo": {"pcl"}}, time_cutoff="2026-06-01T00:00:00Z", guard_days=30)
    sp = r["splits"]
    assert sp["task04"] == "showcase" and sp["t0"] == "ood:repo" and sp["pr1"] == "time"
    assert not splits.check(items, sp, ood={"repo": {"pcl"}}) and splits.verify_seal(sp, b"s3cret", r["seal"])
    assert not splits.verify_seal(sp, b"other", r["seal"])
    assert splits.eligible(items[-2], "2026-03-01T00:00:00Z") and not splits.eligible(items[-1], "2026-03-01T00:00:00Z")
    text = "Refund the damaged item on order O-7270 in full on ticket 39 and set ticket 39 to resolved after the supervisor approves it"
    can = contamination.new_canary("toy", "v1")
    bench = [contamination.fingerprint("t39", can + "\n" + text, code="UPDATE refunds SET amount_usd = 93.96 WHERE ticket_id = 39 AND order_id = 'O-7270'")]
    assert "crucible-canary" not in bench[0].text
    corpus = [("clean", "a completely unrelated document about berth planning in a container port"), ("leak", f"training sample: {text}"),
              ("canary", f"... {can} ..."), ("code", "fix: update refunds set amount_usd = 93.96 where ticket_id = 39 and order_id = 'o-7270';")]
    kinds = {(h["source"], h["kind"]) for h in contamination.scan(bench, corpus, canary=can)}
    assert ("leak", "13-gram") in kinds and ("leak", "near-duplicate") in kinds and ("canary", "canary") in kinds and ("code", "code-substring") in kinds
    assert not any(s == "clean" for s, _ in kinds)
    assert contamination.leaks({"a": "fix the crash from #4512", "b": "no refs"}) == {"a": ["#4512"]}


# ---- the toy benchmark (exit gate, toy part) ---------------------------------------------------------------------------
def _policies():
    from crucible.actors.policy import ScriptedPolicy
    from crucible.core.schema.actions import Message, ToolCall
    from crucible.runtime.harness.native import PolicyRef

    def acts(task, frac):
        a = task.oracle_actions()
        a = a[: int(len(a) * frac)]
        return [ToolCall(tool_call_id=x["tool_call_id"], function_name=x["function_name"], arguments=x.get("arguments") or {}) if x["kind"] == "tool_call"
                else Message(to=x["to"], text=x["text"]) for x in a]

    return [PolicyRef(id=f"scripted:{name}", make=lambda att, f=frac: {"agent": ScriptedPolicy("agent", acts(att.task, f))})
            for name, frac in (("oracle", 1.0), ("half", 0.5), ("noop", 0.0))]


async def _toy_bench(root):
    from crucible_pack_toy.longhorizon.task import BacklogTask

    from crucible.actors.policy import EpisodeLimits
    from crucible.bench.run import run_bench
    from crucible.runtime.harness.native import NativeHarness

    made = []

    def gen(seed):
        made.append(BacklogTask(seed=seed % 100_000, n=12))
        return made[-1]

    async def gold_and_noop(task):
        from crucible.actors.policy import ScriptedPolicy, run_episode
        from crucible.core.schema.actions import Message, ToolCall

        acts = [ToolCall(tool_call_id=a["tool_call_id"], function_name=a["function_name"], arguments=a.get("arguments") or {}) if a["kind"] == "tool_call"
                else Message(to=a["to"], text=a["text"]) for a in task.oracle_actions()]
        rewards = []
        for script in (acts, []):
            w = await task.open(root / "validate" / f"{task.id.replace('/', '_')}-{len(script)}")
            try:
                await run_episode(w, {"agent": ScriptedPolicy("agent", script)}, limits=EpisodeLimits(max_steps=500), reset=False)
                rewards.append((await task.grade(w, root / "validate" / "g")).reward)
            finally:
                await w.close()
        ok = rewards[0] == 1.0 and rewards[1] < 1.0
        return ok, "pass" if ok else f"gold {rewards[0]} / noop {rewards[1]}"

    anchors = [(t.id, "backlog", t) for t in (BacklogTask(seed=11, n=12), BacklogTask(seed=12, n=12))]
    made += [t for _, _, t in anchors]
    key = b"toy-bench-v1-key"
    bm = await dynamic.build("toy-backlog", "v1", key=key, anchors=anchors, generators={"backlog": gen}, n_fresh=4, checks=[("gold_and_noop", gold_and_noop)])
    tasks = [i.task for i in bm.items]
    stages = {t.id: [(f"q{q}", [g.id for g in t.subgoals.subgoals][: 3 * q]) for q in (1, 2, 3, 4)] for t in tasks}
    items = [splits.SplitItem(i.id, lineage=i.id, axes={"family": i.family}) for i in bm.items]
    sp = splits.assign(items, salt=b"toy-salt", bench_frac=0.5)
    try:
        res = await run_bench("toy-backlog-v1", root=root, tasks=tasks, policies=_policies(),
                              harnesses=[NativeHarness(limits=EpisodeLimits(max_steps=500))], k=2, stages_of=stages.get,
                              horizon_of=lambda t: "h12", extras={"manifest": bm.manifest(), "splits": sp})
    finally:
        for t in made:
            t.cleanup()
    return bm, res


async def test_toy_benchmark_generated_and_reported_reproducibly(tmp_path):
    bm, res = await _toy_bench(tmp_path / "a")
    man = bm.manifest()
    assert man["counts"] == {"anchor": 2, "fresh": 4} and man["commitment"] == dynamic.commitment(b"toy-bench-v1-key")
    assert "seed" not in json.dumps(man["items"])  # the manifest carries ids, never seeds
    by = {m["policy"]: m["overall"]["deterministic"] for m in res["summary"]["models"]}
    o, h, n = by["scripted:oracle"], by["scripted:half"], by["scripted:noop"]
    assert o["coverage"]["measured"] == 12 and o["coverage"]["masked"] == 0
    assert o["pass@1"] == 1.0 and h["pass@1"] == 0.0 and n["pass@1"] == 0.0 and o["pass^1"] == 1.0
    assert o["progress_auc"] > h["progress_auc"] > n["progress_auc"] == 0.0
    assert 0 < h["phi_final"] < 1 and o["phi_final"] == 1.0
    assert o["stage_reach"] == 3.0 and n["stage_reach"] == -1.0 and 0 <= h["stage_reach"] < 3
    assert o["stage_first_step"]["q1"] < o["stage_first_step"]["q4"] and o["regressions"] == 0
    assert o["tool_calls"] > h["tool_calls"] > 0 and o["messages"] > 0 and n["messages_per_success"] is None
    th = res["extras"]["irt"]["theta"]
    assert th["scripted:oracle@native:llm-tool@1"] > th["scripted:half@native:llm-tool@1"] >= th["scripted:noop@native:llm-tool@1"]
    eps = res["episodes"]
    cmp = compare.compare([e for e in eps if e.policy == "scripted:oracle"], [e for e in eps if e.policy == "scripted:half"])
    assert cmp["pairs"] == 12 and cmp["flips"]["pass_to_fail"] == 12 and cmp["mcnemar_p"] < 0.001 and cmp["delta_ci95_paired_bootstrap"][1] < 0
    html = (tmp_path / "a" / "bench-toy-backlog-v1" / "bench" / "report.html").read_text()
    assert "Progress Φ per normalized step" in html and "Gate profile" not in html  # the backlog grades by subgoals, not gates
    # reproducible: a fresh run gives the same deterministic components and byte-identical HTML
    bm2, _ = await _toy_bench(tmp_path / "b")
    assert bm2.manifest() == man
    s1 = json.loads((tmp_path / "a" / "bench-toy-backlog-v1" / "bench" / "summary.json").read_text())
    s2 = json.loads((tmp_path / "b" / "bench-toy-backlog-v1" / "bench" / "summary.json").read_text())
    assert s1["digest"] == s2["digest"]
    assert html == (tmp_path / "b" / "bench-toy-backlog-v1" / "bench" / "report.html").read_text()
