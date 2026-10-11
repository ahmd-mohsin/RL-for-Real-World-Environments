"""Step-level benchmark metrics (DESIGN §11.1), computed from the M10 run journal.

An ``Episode`` is one terminal attempt: its scored reward (None when masked: a masked sample is not a sample, DESIGN
P16), gate bits, step count, cost, its progress rows (``progress.jsonl``: Φ after each World step, milestones gained and
lost) and counts read from its trace. Every metric is a pure function of episodes, so a re-run with the same seeds gives
identical deterministic components (cost and wall time are reported, never part of that digest).

| Metric | Definition here |
|---|---|
| coverage | measured vs masked episodes, measured tasks, fully masked tasks; an all-masked group reports coverage only, never a 0 (NeMo-Gym's mask semantics) |
| mean reward | over measured episodes, with the raw mean (masked as 0) beside it so masking is visible (verifiers' dual headline) |
| pass@k, pass^k | unbiased per (task, policy, harness) group at a reward threshold; pass^k = all k succeed (reliability; prime-rl, tau-bench) |
| max@k | the unbiased expected best of k for non-binary rewards (order statistics; not the order-biased max of the first k) |
| gate profile | per-gate pass rate over scored episodes that report the gate, plus EnterpriseOps-Gym's ``verifier_level_pass_rate`` (passed gates / reported gates) |
| progress AUC | area under the step function Φ(t) over normalized steps t/T in [0, 1], T = the episode's steps |
| stage reach | the furthest stage (in workflow order) whose exit milestones are all satisfied at the end, and the step each stage was first reached (ToolSandbox's milestone-to-step mapping, as a pattern) |
| first irrecoverable step | the first step whose branch-rollout success rate has a Wilson upper bound below ``threshold`` (C3) |
| regressions | milestones lost after being gained |
| recovery rate | pass rate on episodes started from a fault-injected state (``tags`` contains ``fault_start``) |
| cost | tokens, tool calls, wall seconds, USD, GPU seconds; per success = all spend (masked included) / measured successes |
| communication | messages the policy sent, and per success |
| probe profile | shortcut probes attempted / exposed; an exposed episode never counts as a pass |

Uncertainty: seeded bootstrap CIs over tasks (deterministic for a seed). Study: ``docs/research/bench-studio/01``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.credit.operators.branch_value import wilson


@dataclass
class Episode:
    task: str
    policy: str
    harness: str
    sample_idx: int
    attempt: str
    reward: float | None  # scored reward; None = masked
    gates: dict[str, int] = field(default_factory=dict)
    steps: int = 0
    stop: str | None = None
    cost_usd: float = 0.0
    tokens: int = 0
    tool_calls: int = 0
    messages: int = 0
    wall_s: float = 0.0
    gpu_s: float = 0.0
    rows: list[dict[str, Any]] = field(default_factory=list)  # progress rows: step, phi, gained, lost
    phi_final: float | None = None
    probes_attempted: int = 0
    exposed: bool = False
    tags: list[str] = field(default_factory=list)
    progress_tracked: bool = True  # Φ was measured at every step (progress.jsonl); recorded external runs carry only an end-state Φ

    @property
    def masked(self) -> bool:
        return self.reward is None

    def passed(self, threshold: float = 1.0) -> bool:
        return self.reward is not None and not self.exposed and self.reward >= threshold


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load_run(run_dir: Path, *, tags_of: Callable[[str], list[str]] | None = None) -> list[Episode]:
    """Every terminal attempt of a run (``data/runs/<run_id>``), latest attempt per sample, in sample order."""
    from crucible.core.schema.run import RunRecord

    run_dir = Path(run_dir)
    latest: dict[str, RunRecord] = {}
    for row in _jsonl(run_dir / "results.jsonl"):
        r = RunRecord.model_validate(row)
        latest[(r.attempt or r.run_id).rsplit("-a", 1)[0]] = r
    out = []
    for sid, r in sorted(latest.items()):
        sk = r.sample or {}
        n = (r.attempt or "-a1").rsplit("-a", 1)[1]
        adir = run_dir / "samples" / sid / f"a{n}"
        tool_calls = messages = 0
        policy_actors = {"agent"}
        for e in _jsonl(adir / "trace.jsonl"):
            if e.get("actor") in policy_actors and e.get("type") == "tool_call":
                tool_calls += 1
            elif e.get("actor") in policy_actors and e.get("type") == "message":
                messages += 1
        ro = json.loads((adir / "rollout.json").read_text()) if (adir / "rollout.json").exists() else {}
        tokens = int(sum((r.tokens or {}).values()))
        if not tokens and (adir / "calls.json").exists():  # the run record carries no token counts: sum the model calls' usage
            for cs in json.loads((adir / "calls.json").read_text()).values():
                tokens += sum(int((c.get("usage") or {}).get("prompt_tokens") or 0) + int((c.get("usage") or {}).get("completion_tokens") or 0) for c in cs)
        comps = ro.get("reward_components") or {}
        timings = r.meta.get("timings") or {}
        task = str(sk.get("task") or r.task or "")
        out.append(Episode(
            task=task, policy=str(sk.get("policy") or r.policy), harness=str(sk.get("harness") or r.meta.get("harness") or ""),
            sample_idx=int(sk.get("sample_idx") or 0), attempt=r.attempt or "", reward=r.scored_reward(), gates=dict(r.meta.get("gates") or {}),
            steps=int(r.steps or 0), stop=r.meta.get("stop"), cost_usd=float(r.cost_usd or 0.0), tokens=tokens,
            tool_calls=tool_calls, messages=messages, wall_s=float(sum(v for v in timings.values() if isinstance(v, (int, float)))),
            gpu_s=float(r.meta.get("gpu_s") or 0.0), rows=_jsonl(adir / "progress.jsonl"),
            phi_final=(r.meta.get("progress") or {}).get("phi_final"), probes_attempted=int(comps.get("probe/attempted") or 0),
            progress_tracked=(adir / "progress.jsonl").exists(),
            exposed=bool(r.failure and "exposed" in (r.failure.kind or "")), tags=list(tags_of(task) if tags_of else [])))
    return out


# ---- per-episode metrics --------------------------------------------------------------------------------------------
def progress_auc(rows: Sequence[dict[str, Any]], total_steps: int, *, phi0: float = 0.0) -> float:
    """Area under the step function Φ(t) on normalized steps: Φ holds its value from a row's step until the next row's."""
    T = max(int(total_steps), max((int(r["step"]) for r in rows), default=0), 1)
    pts = sorted((int(r["step"]), float(r["phi"])) for r in rows)
    area, x, phi = 0.0, 0, phi0
    for step, value in pts:
        step = min(max(step, 0), T)
        area += phi * (step - x)
        x, phi = step, value
    area += phi * (T - x)
    return round(area / T, 9)


def final_milestones(rows: Sequence[dict[str, Any]]) -> set[str]:
    have: set[str] = set()
    for r in sorted(rows, key=lambda r: int(r["step"])):
        have |= set(r.get("gained") or [])
        have -= set(r.get("lost") or [])
    return have


def regressions(rows: Sequence[dict[str, Any]]) -> int:
    """Milestones lost after being gained (each loss of a previously held milestone counts once)."""
    have: set[str] = set()
    n = 0
    for r in sorted(rows, key=lambda r: int(r["step"])):
        have |= set(r.get("gained") or [])
        for m in r.get("lost") or []:
            if m in have:
                n += 1
                have.discard(m)
    return n


def stage_first_reached(rows: Sequence[dict[str, Any]], stages: Sequence[tuple[str, Sequence[str]]]) -> dict[str, int | None]:
    """The step at which each stage's exit milestones were first all held (None: never)."""
    have: set[str] = set()
    out: dict[str, int | None] = {name: None for name, _ in stages}
    for r in sorted(rows, key=lambda r: int(r["step"])):
        have |= set(r.get("gained") or [])
        have -= set(r.get("lost") or [])
        for name, exits in stages:
            if out[name] is None and exits and set(exits) <= have:
                out[name] = int(r["step"])
    return out


def stage_reach(milestones: set[str], stages: Sequence[tuple[str, Sequence[str]]]) -> tuple[int, str | None]:
    """(index, name) of the furthest stage whose exit milestones are all satisfied; (-1, None) if none is."""
    best = (-1, None)
    for i, (name, exits) in enumerate(stages):
        if exits and set(exits) <= milestones:
            best = (i, name)
    return best


def first_irrecoverable_step(branches: Sequence[tuple[int, int, int]], *, threshold: float = 0.05, z: float = 1.96) -> int | None:
    """``branches``: (step, successes, n) from C3 branch rollouts on a sampled subset. The first step from which success is
    implausible: the Wilson upper bound of its branch success rate is below ``threshold``."""
    for step, s, n in sorted(branches):
        if n > 0 and wilson(s, n, z)[1] < threshold:
            return step
    return None


# ---- aggregates ---------------------------------------------------------------------------------------------------------
def pass_at_k(eps: Iterable[Episode], k: int, *, threshold: float = 1.0) -> float | None:
    groups: dict[tuple[str, str, str], list[bool]] = {}
    for e in eps:
        if not e.masked:
            groups.setdefault((e.task, e.policy, e.harness), []).append(e.passed(threshold))
    vals = []
    for g in groups.values():
        n, c = len(g), sum(g)
        if n >= k:
            vals.append(1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k))
    return round(sum(vals) / len(vals), 9) if vals else None


def pass_hat_k(eps: Iterable[Episode], k: int, *, threshold: float = 1.0) -> float | None:
    """pass^k: the probability that k samples drawn without replacement all pass (C(c, k) / C(n, k))."""
    groups: dict[tuple[str, str, str], list[bool]] = {}
    for e in eps:
        if not e.masked:
            groups.setdefault((e.task, e.policy, e.harness), []).append(e.passed(threshold))
    vals = [math.comb(sum(g), k) / math.comb(len(g), k) for g in groups.values() if len(g) >= k]
    return round(sum(vals) / len(vals), 9) if vals else None


def max_at_k(eps: Iterable[Episode], k: int) -> float | None:
    """The unbiased expected maximum of k rewards drawn without replacement: sorted ascending, value i (1-based) is the max
    with probability C(i-1, k-1) / C(n, k)."""
    groups: dict[tuple[str, str, str], list[float]] = {}
    for e in eps:
        if not e.masked:
            groups.setdefault((e.task, e.policy, e.harness), []).append(0.0 if e.exposed else float(e.reward or 0.0))
    vals = []
    for g in groups.values():
        n = len(g)
        if n >= k:
            xs = sorted(g)
            vals.append(sum(xs[i - 1] * math.comb(i - 1, k - 1) for i in range(k, n + 1)) / math.comb(n, k))
    return round(sum(vals) / len(vals), 9) if vals else None


def coverage(eps: Sequence[Episode]) -> dict[str, int]:
    tasks: dict[str, list[bool]] = {}
    for e in eps:
        tasks.setdefault(e.task, []).append(e.masked)
    return {"episodes": len(eps), "measured": sum(not e.masked for e in eps), "masked": sum(e.masked for e in eps),
            "measured_tasks": sum(not all(v) for v in tasks.values()), "fully_masked_tasks": sum(all(v) for v in tasks.values())}


def bootstrap_ci(per_task: dict[str, list[float]], *, iters: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float] | None:
    """A seeded percentile bootstrap of the mean, resampling tasks (each task's episodes stay together)."""
    import random

    keys = sorted(k for k, v in per_task.items() if v)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        vals = [x for k in (keys[rng.randrange(len(keys))] for _ in keys) for x in per_task[k]]
        means.append(sum(vals) / len(vals))
    means.sort()
    return (round(means[int(alpha / 2 * iters)], 6), round(means[min(iters - 1, int((1 - alpha / 2) * iters))], 6))


def gate_profile(eps: Iterable[Episode]) -> dict[str, Any]:
    per: dict[str, list[int]] = {}
    for e in eps:
        if e.masked:
            continue
        for g, bit in e.gates.items():
            per.setdefault(g, []).append(int(bit))
    gates = {g: {"passed": sum(v), "total": len(v), "pass_rate": round(sum(v) / len(v), 9)} for g, v in sorted(per.items())}
    total = sum(x["total"] for x in gates.values())
    return {"gates": gates, "verifier_level_pass_rate": round(sum(x["passed"] for x in gates.values()) / total, 9) if total else None,
            "macro_pass_rate": round(sum(x["pass_rate"] for x in gates.values()) / len(gates), 9) if gates else None}


def _mean(xs: Sequence[float]) -> float | None:
    return round(sum(xs) / len(xs), 9) if xs else None


def summarize_group(eps: Sequence[Episode], *, ks: Sequence[int] = (1,), threshold: float = 1.0,
                    stages: Sequence[tuple[str, Sequence[str]]] | None = None,
                    branches: dict[str, Sequence[tuple[int, int, int]]] | None = None, ci_seed: int = 0,
                    stages_of: Callable[[str], Sequence[tuple[str, Sequence[str]]] | None] | None = None) -> dict[str, Any]:
    """Every §11.1 metric over one group of episodes (one task × policy × harness, or any union of them)."""
    scored = [e for e in eps if not e.masked]
    cov = coverage(eps)
    if not scored:  # an all-masked group: coverage only, never an invented 0
        return {"deterministic": {"coverage": cov}, "cost": {"usd": round(sum(e.cost_usd for e in eps), 6)}}
    succ = [e for e in scored if e.passed(threshold)]
    aucs = [progress_auc(e.rows, e.steps) for e in scored if e.progress_tracked and e.phi_final is not None]  # only where Φ was measured per step
    st_of = (lambda t: stages_of(t)) if stages_of else (lambda t: stages)
    staged = [(e, st_of(e.task)) for e in scored if st_of(e.task)]
    reach = [stage_reach(final_milestones(e.rows), st)[0] for e, st in staged]
    first = [stage_first_reached(e.rows, st) for e, st in staged]
    names = sorted({n for f in first for n in f})
    fis = [first_irrecoverable_step((branches or {}).get(e.attempt, ())) for e in scored if (branches or {}).get(e.attempt)]
    faults = [e for e in scored if "fault_start" in e.tags]
    n_succ = len(succ)
    by_task: dict[str, list[float]] = {}
    for e in scored:
        by_task.setdefault(e.task, []).append(float(e.reward or 0.0))
    det = {
        "coverage": cov,
        "mean_reward": _mean([float(e.reward or 0.0) for e in scored]),
        "mean_reward_raw": _mean([0.0 if e.masked else float(e.reward or 0.0) for e in eps]),  # masked as 0: shown, never used
        "mean_reward_ci": bootstrap_ci(by_task, seed=ci_seed),
        **{f"pass@{k}": pass_at_k(eps, k, threshold=threshold) for k in ks},
        **{f"pass^{k}": pass_hat_k(eps, k, threshold=threshold) for k in ks},
        **{f"max@{k}": max_at_k(eps, k) for k in ks},
        "gate_profile": gate_profile(eps),
        "progress_auc": _mean(aucs),
        "phi_final": _mean([float(e.phi_final) for e in scored if e.phi_final is not None]),
        "stage_reach": _mean([float(r) for r in reach]) if staged else None,
        "stage_reach_hist": {str(i): reach.count(i) for i in sorted(set(reach))} if staged else None,
        "stage_first_step": {n: _mean([float(f[n]) for f in first if f.get(n) is not None]) for n in names} if staged else None,
        "first_irrecoverable_step": _mean([float(x) for x in fis if x is not None]) if fis else None,
        "irrecoverable_episodes": sum(x is not None for x in fis) if fis else None,
        "regressions": sum(regressions(e.rows) for e in scored),
        "recovery_rate": _mean([float(e.passed(threshold)) for e in faults]) if faults else None,
        "steps": _mean([float(e.steps) for e in scored]),
        "tool_calls": sum(e.tool_calls for e in scored),
        "messages": sum(e.messages for e in scored),
        "messages_per_success": round(sum(e.messages for e in scored) / n_succ, 6) if n_succ else None,
        "probes": {"attempted": sum(e.probes_attempted for e in eps), "exposed": sum(e.exposed for e in eps)},
    }
    cost = {"usd": round(sum(e.cost_usd for e in eps), 6), "tokens": sum(e.tokens for e in eps), "wall_s": round(sum(e.wall_s for e in eps), 3),
            "gpu_s": round(sum(e.gpu_s for e in eps), 3)}
    cost["per_success"] = {k: (round(v / n_succ, 6) if n_succ else None) for k, v in cost.items()} | {
        "tool_calls": round(sum(e.tool_calls for e in scored) / n_succ, 6) if n_succ else None}
    cost["observability"] = {"tokens": round(sum(e.tokens > 0 for e in eps) / len(eps), 6), "usd": round(sum(e.cost_usd > 0 for e in eps) / len(eps), 6)}
    return {"deterministic": det, "cost": cost}


def summarize(eps: Sequence[Episode], *, ks: Sequence[int] = (1,), threshold: float = 1.0,
              stages_of: Callable[[str], Sequence[tuple[str, Sequence[str]]] | None] | None = None,
              horizon_of: Callable[[str], str] | None = None,
              branches: dict[str, Sequence[tuple[int, int, int]]] | None = None) -> dict[str, Any]:
    """Per (policy, harness) × task, and aggregated per horizon and overall (DESIGN §11.1: every metric per task and per
    horizon)."""
    by: dict[tuple[str, str], dict[str, list[Episode]]] = {}
    for e in eps:
        by.setdefault((e.policy, e.harness), {}).setdefault(e.task, []).append(e)
    models = []
    for (pol, har), tasks in sorted(by.items()):
        per_task = {t: summarize_group(es, ks=ks, threshold=threshold, stages_of=stages_of, branches=branches) for t, es in sorted(tasks.items())}
        horizons: dict[str, list[Episode]] = {}
        for t, es in tasks.items():
            horizons.setdefault(horizon_of(t) if horizon_of else "all", []).extend(es)
        models.append({"policy": pol, "harness": har, "tasks": per_task,
                       "horizons": {h: summarize_group(es, ks=ks, threshold=threshold, branches=branches, stages_of=stages_of) for h, es in sorted(horizons.items())},
                       "overall": summarize_group([e for es in tasks.values() for e in es], ks=ks, threshold=threshold, branches=branches,
                                                  stages_of=stages_of)})
    return {"models": models, "threshold": threshold, "ks": list(ks)}


def heatmap(eps: Sequence[Episode], *, bins: int = 10) -> dict[str, dict[str, list[float | None]]]:
    """Mean Φ per normalized-step bin, per model (policy@harness) and task: the report's per-step heatmap."""
    acc: dict[str, dict[str, list[list[float]]]] = {}
    for e in eps:
        if e.masked or not e.progress_tracked:
            continue
        T = max(e.steps, max((int(r["step"]) for r in e.rows), default=0), 1)
        pts = sorted((int(r["step"]), float(r["phi"])) for r in e.rows)
        cells = acc.setdefault(f"{e.policy}@{e.harness}", {}).setdefault(e.task, [[] for _ in range(bins)])
        for b in range(bins):
            t = (b + 1) * T / bins  # Φ at the end of the bin
            phi = 0.0
            for step, value in pts:
                if step <= t:
                    phi = value
            cells[b].append(phi)
    return {m: {t: [round(sum(c) / len(c), 6) if c else None for c in cells] for t, cells in sorted(ts.items())} for m, ts in sorted(acc.items())}


def deterministic_digest(summary: dict[str, Any]) -> str:
    """A hash of every deterministic component (cost and wall time excluded): equal across re-runs with the same seeds."""
    import hashlib

    def strip(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: strip(v) for k, v in sorted(x.items()) if k != "cost"}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    return hashlib.sha256(json.dumps(strip(summary), sort_keys=True, default=str).encode()).hexdigest()[:16]


def episodes_json(eps: Sequence[Episode]) -> list[dict[str, Any]]:
    return [{k: v for k, v in asdict(e).items() if k != "rows"} for e in eps]
