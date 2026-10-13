"""Step-level benchmark metrics (DESIGN §11.1), computed from the M10 run journal.

An ``Episode`` is one planned sample: its terminal attempt's scored reward (None when masked: a masked sample is not a
sample, DESIGN P16; None and ``owed`` when the runner never finished it), gate bits, step count, the spend of **every**
attempt of the sample, its progress rows (``progress.jsonl``: Φ after each World step, milestones gained and lost, and
the start state as a ``step=0`` row) and counts read from its trace. Every metric is a pure function of episodes, so a
re-run with the same seeds gives identical deterministic components (cost and wall time are reported, never part of
that digest).

Every number states its denominator: ``*_tasks`` / ``*_episodes`` counts sit next to each value, coverage carries the
planned count, and a run the runner stopped is never presented as the whole benchmark (``run_info``, ``summary["run"]``).

| Metric | Definition here |
|---|---|
| coverage | planned / terminal / measured / masked / owed samples, measured tasks, tasks with nothing measured, masks by kind; an all-masked group reports coverage only, never a 0 (NeMo-Gym's mask semantics) |
| mean reward | the **mean of per-task means** over tasks with a measured sample (G1 D3, H1 V8, G2-7: a pooled mean reweights tasks by their mask rate); the pooled per-episode mean (``mean_reward_episodes``) and the raw mean (masked as 0, verifiers' dual headline) beside it |
| mean reward CI | a t interval on the task means, the headline's own statistic (H1 V9 / G1 D5: the percentile bootstrap covered 0.76 at 3 tasks); ``n_tasks`` beside it, ``approximate`` below 10 tasks |
| pass@k, pass^k, max@k | unbiased per (task, policy, harness) group, averaged over **one task set for every k** (G1 D2, H1 V8, G2-6): if a measured task has fewer than k measured samples the k is refused (None) and ``pass@k_excluded`` says how many; ``pass@k_tasks`` is the set size |
| exposure | an exposed episode (``crucible:answer_exposed``) is a **scored failure (0) with a flag** in every bench view (G2-1: the runner masks it for training, and dropping it let a hacker outrank an honest model); ``exposure.rate`` = exposed / measured |
| gate profile | per-gate pass rate over measured episodes that report the gate, plus EnterpriseOps-Gym's ``verifier_level_pass_rate`` (passed gates / reported gates) |
| progress AUC | (1/T) Σ_{t=1..T} Φ_t: Φ_t is the state after step t, Φ_0 the start state (H1 V3: the left endpoint gave a one-step solve AUC 0) |
| stage reach | the furthest stage (in workflow order) whose exit milestones all hold at the end, counting milestones held at the start (H1 V2) and the end-of-episode correction row (H1 V1); ``stage_reach_frac`` normalizes by each task's stage count (H1 V18) |
| first irrecoverable step | the first step t such that every sampled step >= t has a branch-success Wilson upper bound below ``threshold`` (a suffix condition, H1 V17); episodes whose branch budget cannot resolve the threshold are counted as ``branches_insufficient`` (H1 V11) |
| regressions | milestones lost after being held (gained, or held at the start) |
| recovery rate | pass rate on measured episodes started from a fault-injected state (``tags`` contains ``fault_start``), with its n |
| spread | std and standard error across sample indices (repeats) of the task mean, and the mean within-task std (G1 D10) |
| cost | USD (settled), uncertain USD, tokens (in / out), wall seconds, GPU seconds (None when unobserved, H1 V15) over every attempt, masked retries included (H1 V4 / G1 D4); per success = all spend / measured successes, with the success count |
| perf | wall-time p50 / p90 / p99 of the graded attempts and mean steps (G1 D12) |
| communication | messages the policy actors sent, and per success; actors come from the attempt's calls and trajectories (H1 V14) |
| probe profile | shortcut probes attempted (from the World trace, H1 V10) / exposed |

Study: ``docs/research/bench-studio/01``; audit: ``docs/audit/2026-10-13/M13-study-defects-H1.md``, ``M13-sweep-G1.md``,
``M13-sweep-G2.md``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.credit.operators.branch_value import wilson

EXPOSED_KINDS = frozenset({"crucible:answer_exposed"})
CI_MIN_TASKS = 10  # below this the task-level CI is labelled approximate (G1 D5)


@dataclass
class Episode:
    task: str
    policy: str
    harness: str
    sample_idx: int
    attempt: str
    reward: float | None  # scored reward; None = masked (or owed)
    gates: dict[str, int] = field(default_factory=dict)
    steps: int = 0
    stop: str | None = None
    cost_usd: float = 0.0  # settled spend of every attempt of the sample (H1 V4)
    tokens: int = 0
    tool_calls: int = 0
    messages: int = 0
    wall_s: float = 0.0  # wall time of every attempt
    gpu_s: float | None = None  # None: not observed (H1 V15)
    rows: list[dict[str, Any]] = field(default_factory=list)  # progress rows: step, phi, gained, lost
    phi_final: float | None = None
    probes_attempted: int = 0
    exposed: bool = False
    tags: list[str] = field(default_factory=list)
    progress_tracked: bool = True  # Φ was measured at every step (progress.jsonl); recorded external runs carry only an end-state Φ
    # added with the M13 audit (2026-10-13)
    owed: bool = False  # planned, but no terminal attempt (the runner stopped or was killed): H1 V5
    failure_kind: str | None = None  # the terminal attempt's failure kind (masked or scored), G1 D9
    failure_detail: str | None = None
    attempts: int = 1  # attempts with a record (masked retries included); 0 for an owed sample never dispatched
    aborted_attempts: int = 0  # attempts that left no record (process death): spend unknown
    cost_uncertain_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float | None = None  # wall time of the graded (terminal) attempt
    phi0: float | None = None  # Φ of the start state (meta.progress.phi0), H1 V2
    peer_messages: int = 0  # messages from one policy actor to another (P28 communication cost)
    seed: int | None = None
    flags: list[str] = field(default_factory=list)  # rule flags the harness raised (rollout.json ``flags``): campaign B1

    @property
    def masked(self) -> bool:
        """Not measured: a masked terminal attempt or an owed sample (``owed`` tells the two apart)."""
        return self.reward is None

    @property
    def value(self) -> float | None:
        """The reward every bench aggregate uses: None when not measured, 0 when exposed (G2-1, G2-19)."""
        if self.reward is None:
            return None
        return 0.0 if self.exposed else float(self.reward)

    def passed(self, threshold: float = 1.0) -> bool:
        return self.reward is not None and not self.exposed and self.reward >= threshold


# ---- reading a run directory ---------------------------------------------------------------------------------------------
def _jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL with the journal's torn-tail rule (H1 V6): an undecodable **last** line is a crash mid-append and is skipped;
    an undecodable line anywhere else is corruption and raises."""
    if not path.exists():
        return []
    lines = [x for x in path.read_text().splitlines() if x.strip()]
    out = []
    for i, x in enumerate(lines):
        try:
            out.append(json.loads(x))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue
            raise
    return out


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (json.JSONDecodeError, OSError):
        return None


def _sample_tuple(sk: Mapping[str, Any], r: Any) -> tuple[str, str, str, int]:
    return (str(sk.get("task") or r.task or ""), str(sk.get("policy") or r.policy), str(sk.get("harness") or (r.meta or {}).get("harness") or ""),
            int(sk.get("sample_idx") or 0))


def _attempt_n(attempt: str | None) -> int:
    tail = (attempt or "-a1").rsplit("-a", 1)[-1]
    return int(tail) if tail.isdigit() else 1


def _usage(calls: Any) -> tuple[int, int]:
    tin = tout = 0
    if isinstance(calls, dict):
        for cs in calls.values():
            for c in cs or []:
                u = (c or {}).get("usage") or {}
                tin += int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
                tout += int(u.get("completion_tokens") or u.get("output_tokens") or 0)
    return tin, tout


def _attempt_stats(adir: Path, rec: Any) -> dict[str, Any]:
    """Spend, tokens and wall time of one attempt (its own record and model calls)."""
    timings = (rec.meta or {}).get("timings") or {}
    tin, tout = _usage(_json(adir / "calls.json"))
    rt = rec.tokens or {}
    if rt and not (tin or tout):
        tin = int(rt.get("prompt") or rt.get("input") or rt.get("prompt_tokens") or 0)
        tout = int(rt.get("completion") or rt.get("output") or rt.get("completion_tokens") or 0)
        if not (tin or tout):
            tin = int(sum(v for v in rt.values() if isinstance(v, (int, float))))
    gpu = (rec.meta or {}).get("gpu_s")
    return {"usd": float(rec.cost_usd or 0.0), "usd_uncertain": float(rec.cost_uncertain_usd or 0.0), "tokens_in": tin, "tokens_out": tout,
            "wall_s": float(sum(v for v in timings.values() if isinstance(v, (int, float)))), "gpu_s": None if gpu is None else float(gpu)}


def _sum_stats(stats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gpus = [s["gpu_s"] for s in stats if s["gpu_s"] is not None]
    tin, tout = sum(s["tokens_in"] for s in stats), sum(s["tokens_out"] for s in stats)
    return {"cost_usd": round(sum(s["usd"] for s in stats), 9), "cost_uncertain_usd": round(sum(s["usd_uncertain"] for s in stats), 9),
            "tokens_in": tin, "tokens_out": tout, "tokens": tin + tout, "wall_s": round(sum(s["wall_s"] for s in stats), 6),
            "gpu_s": round(sum(gpus), 6) if gpus else None, "attempts": len(stats)}


def _policy_actors(adir: Path) -> set[str]:
    """The attempt's policy actors (H1 V14): the keys of ``calls.json`` and the ``trajectory.<actor>.json`` artifacts."""
    calls = _json(adir / "calls.json")
    actors = set(calls) if isinstance(calls, dict) else set()
    actors |= {p.name[len("trajectory."):-len(".json")] for p in adir.glob("trajectory.*.json")}
    return actors or {"agent"}


def _probes_attempted(adir: Path, comps: Mapping[str, Any]) -> int:
    """Probe attempts (H1 V10): the training path writes ``probe/attempted``; the M10 runner leaves it in the World
    trace (``trace.json`` → ``outcome.details.probes.attempted``, a count per probe kind)."""
    if comps.get("probe/attempted"):
        return int(comps["probe/attempted"])
    tr = _json(adir / "trace.json") or {}
    att = (((tr.get("outcome") or {}).get("details") or {}).get("probes") or {}).get("attempted") or {}
    return int(sum(att.values())) if isinstance(att, dict) else int(att or 0)


def run_spec(run_dir: Path) -> dict[str, Any]:
    """The run's spec as the runner recorded it in ``run.json`` (tasks, policies, harnesses, k, run_seed, tier)."""
    return dict((_json(Path(run_dir) / "run.json") or {}).get("spec") or {})


def _plan(run_dir: Path) -> list[tuple[str, str, str, int]] | None:
    """The planned samples from ``run.json``'s spec (the runner's plan: tasks × policies × harnesses × k)."""
    spec = run_spec(run_dir)
    if "tasks" not in spec:
        return None
    return [(str(t), str(p), str(h), i) for t in spec.get("tasks") or [] for p in spec.get("policies") or [] for h in spec.get("harnesses") or []
            for i in range(int(spec.get("k") or 1))]


def load_run(run_dir: Path, *, tags_of: Callable[[str], Iterable[str]] | None = None, plan: bool = True) -> list[Episode]:
    """Every planned sample of a run (``data/runs/<run_id>``): the latest terminal attempt per sample, plus each planned
    sample without one as an ``owed`` episode (H1 V5; ``plan=False`` leaves them out). Spend, tokens and wall time sum
    every attempt of the sample, masked retries and regrades included (H1 V4); quality comes from the terminal attempt.
    An exposed terminal attempt is scored 0 with ``exposed`` set (G2-1). Ordered by (task, policy, harness, index)."""
    from crucible.core.schema.run import RunRecord

    run_dir = Path(run_dir)
    latest: dict[str, RunRecord] = {}
    for row in _jsonl(run_dir / "results.jsonl"):
        r = RunRecord.model_validate(row)
        latest[(r.attempt or r.run_id).rsplit("-a", 1)[0]] = r  # the newest terminal row of a sample wins (a regrade)
    attempts: dict[str, list[tuple[Path, RunRecord]]] = {}  # every attempt with a record; masked retries live only here and in failures.jsonl
    aborted: dict[str, int] = {}
    sdir = run_dir / "samples"
    if sdir.exists():
        for d in sorted(p for p in sdir.iterdir() if p.is_dir()):
            for a in sorted(d.glob("a*/record.json"), key=lambda p: _attempt_n("-" + p.parent.name)):
                try:
                    attempts.setdefault(d.name, []).append((a.parent, RunRecord.model_validate_json(a.read_text())))
                except (ValueError, OSError):
                    continue
            ab = d / "aborted"
            aborted[d.name] = sum(1 for p in ab.iterdir() if p.is_dir()) if ab.exists() else 0
    out: list[Episode] = []
    seen: set[tuple[str, str, str, int]] = set()
    for sid, r in latest.items():
        sk = r.sample or {}
        key = _sample_tuple(sk, r)
        seen.add(key)
        adir = run_dir / "samples" / sid / f"a{_attempt_n(r.attempt)}"
        policy_actors = _policy_actors(adir)
        tool_calls = messages = peer = 0
        for e in _jsonl(adir / "trace.jsonl"):
            if e.get("actor") in policy_actors and e.get("type") == "tool_call":
                tool_calls += 1
            elif e.get("actor") in policy_actors and e.get("type") == "message":
                messages += 1
                peer += int(((e.get("data") or {}).get("to")) in (policy_actors - {e.get("actor")}))
        ro = _json(adir / "rollout.json") or {}
        comps = ro.get("reward_components") or {}
        flags = sorted({str(f.get("rule") if isinstance(f, dict) else f) for f in (ro.get("flags") or [])})
        ats = attempts.get(sid, [])
        term = _attempt_stats(adir, r)
        stats = [_attempt_stats(p, rec) for p, rec in ats]
        if not any(rec.attempt == r.attempt for _, rec in ats):  # the terminal row has no record file (imported): count it
            stats.append(term)
        kind = r.failure.kind if r.failure else None
        exposed = bool(kind and (kind in EXPOSED_KINDS or "exposed" in kind))
        reward = 0.0 if exposed else r.scored_reward()  # G2-1: masked for training by the runner, a scored failure here
        prog = (r.meta or {}).get("progress") or {}
        tot = _sum_stats(stats)
        out.append(Episode(
            task=key[0], policy=key[1], harness=key[2], sample_idx=key[3], attempt=r.attempt or "", reward=reward,
            gates=dict((r.meta or {}).get("gates") or {}), steps=int(r.steps or 0), stop=(r.meta or {}).get("stop"),
            tool_calls=tool_calls, messages=messages, peer_messages=peer, latency_s=term["wall_s"], **tot,
            rows=_jsonl(adir / "progress.jsonl"), phi_final=prog.get("phi_final"), phi0=prog.get("phi0"),
            probes_attempted=_probes_attempted(adir, comps), progress_tracked=(adir / "progress.jsonl").exists(), exposed=exposed,
            tags=list(tags_of(key[0]) if tags_of else []), failure_kind=kind, failure_detail=r.failure.detail if r.failure else None,
            aborted_attempts=aborted.get(sid, 0), seed=sk.get("seed"), flags=flags))
    if plan:
        spent: dict[tuple[str, str, str, int], list[str]] = {}  # samples without a terminal row: their attempts' spend is still spend
        for sid, ats in attempts.items():
            if sid not in latest:
                spent.setdefault(_sample_tuple(ats[-1][1].sample or {}, ats[-1][1]), []).append(sid)
        for key in _plan(run_dir) or []:
            if key in seen:
                continue
            seen.add(key)
            sids = spent.get(key, [])
            recs = [(p, rec) for s in sids for p, rec in attempts.get(s, [])]
            last = recs[-1][1] if recs else None
            out.append(Episode(task=key[0], policy=key[1], harness=key[2], sample_idx=key[3], attempt="", reward=None, owed=True,
                               progress_tracked=False, **_sum_stats([_attempt_stats(p, rec) for p, rec in recs]),
                               aborted_attempts=sum(aborted.get(s, 0) for s in sids),
                               failure_kind=last.failure.kind if last is not None and last.failure else None,
                               tags=list(tags_of(key[0]) if tags_of else [])))
    out.sort(key=lambda e: (e.task, e.policy, e.harness, e.sample_idx))
    return out


def run_info(run_dir: Path, eps: Sequence[Episode] | None = None) -> dict[str, Any]:
    """What the runner planned and why it stopped (H1 V5): the plan from ``run.json``, the stop reason from the journal's
    ``summary`` / ``stop_rule`` events. A run with owed samples and no summary event is reported as incomplete."""
    run_dir = Path(run_dir)
    eps = list(eps) if eps is not None else load_run(run_dir)
    ev = _jsonl(run_dir / "journal.jsonl")
    summaries = [e for e in ev if e.get("event") == "summary"]
    stops = [e.get("reason") for e in ev if e.get("event") == "stop_rule" and e.get("reason")]
    stopped = (summaries[-1].get("stopped") if summaries else None) or (stops[-1] if stops else None)
    owed = sum(e.owed for e in eps)
    if stopped is None and owed and not summaries:
        stopped = "incomplete: the runner left no summary (killed, or still running)"
    planned = _plan(run_dir)
    return {"planned": len(planned) if planned is not None else len(eps), "terminal": sum(not e.owed for e in eps),
            "measured": sum(not e.masked for e in eps), "masked": sum(e.masked and not e.owed for e in eps), "owed": owed,
            "stopped": stopped, "complete": owed == 0 and stopped is None,
            "attempts": sum(e.attempts for e in eps), "aborted_attempts": sum(e.aborted_attempts for e in eps)}


# ---- per-episode metrics -------------------------------------------------------------------------------------------------
def _ordered(rows: Sequence[dict[str, Any]], total_steps: int | None = None) -> list[tuple[int, dict[str, Any]]]:
    """Rows in episode order with their effective step (H1 V1): the end-of-episode correction row (``step=-1``, written
    by ``ProgressTracker.on_end``) is the end state, placed at T after every step row; a ``step=0`` row is the start
    state. Stable for rows at the same step."""
    T = max([int(total_steps or 0), *(int(r["step"]) for r in rows if int(r["step"]) >= 0), 0])
    keyed = sorted(((int(r["step"]) if int(r["step"]) >= 0 else math.inf), i) for i, r in enumerate(rows))
    return [(T if s == math.inf else int(s), rows[i]) for s, i in keyed]


def progress_auc(rows: Sequence[dict[str, Any]], total_steps: int, *, phi0: float | None = None) -> float:
    """(1/T) Σ_{t=1..T} Φ_t on the step function: Φ_t is the value after step t (the last row at or before t), Φ_0 the
    start state (``phi0``, else a ``step=0`` row, else 0). The right endpoint (H1 V3): the state a step reaches counts
    for that step, so a one-step solve scores 1 and a no-op 0."""
    T = max(int(total_steps), max((int(r["step"]) for r in rows), default=0), 1)
    vals: dict[int, float] = {}
    for s, r in _ordered(rows, T):
        vals[min(max(s, 0), T)] = float(r["phi"])
    cur = float(phi0) if phi0 is not None else vals.get(0, 0.0)
    area, last = 0.0, 0
    for s in sorted(vals):
        if s <= 0:
            continue
        area += cur * (s - 1 - last) + vals[s]
        cur, last = vals[s], s
    area += cur * (T - last)
    return round(area / T, 9)


def final_milestones(rows: Sequence[dict[str, Any]], initial: Iterable[str] = ()) -> set[str]:
    """The milestones held at the end: ``initial`` plus every row in episode order (start row first, correction last)."""
    have = set(initial)
    for _, r in _ordered(rows):
        have |= set(r.get("gained") or [])
        have -= set(r.get("lost") or [])
    return have


def regressions(rows: Sequence[dict[str, Any]], initial: Iterable[str] = ()) -> int:
    """Milestones lost after being held (gained, or held at the start); each loss of a held milestone counts once."""
    have = set(initial)
    n = 0
    for _, r in _ordered(rows):
        have |= set(r.get("gained") or [])
        for m in r.get("lost") or []:
            if m in have:
                n += 1
                have.discard(m)
    return n


def stage_first_reached(rows: Sequence[dict[str, Any]], stages: Sequence[tuple[str, Sequence[str]]],
                        initial: Iterable[str] = ()) -> dict[str, int | None]:
    """The step at which each stage's exit milestones were first all held (0: at the start; None: never)."""
    have = set(initial)
    out: dict[str, int | None] = {name: (0 if exits and set(exits) <= have else None) for name, exits in stages}
    for s, r in _ordered(rows):
        have |= set(r.get("gained") or [])
        have -= set(r.get("lost") or [])
        for name, exits in stages:
            if out[name] is None and exits and set(exits) <= have:
                out[name] = s
    return out


def stage_reach(milestones: set[str], stages: Sequence[tuple[str, Sequence[str]]]) -> tuple[int, str | None]:
    """(index, name) of the furthest stage whose exit milestones are all satisfied; (-1, None) if none is."""
    best: tuple[int, str | None] = (-1, None)
    for i, (name, exits) in enumerate(stages):
        if exits and set(exits) <= milestones:
            best = (i, name)
    return best


def min_branches(threshold: float = 0.05, z: float = 1.96) -> int:
    """The fewest branches whose 0-success Wilson upper bound is below ``threshold`` (73 at 0.05): a step sampled with
    fewer can never be called irrecoverable (H1 V11; C3's default k = 8 gives an upper bound of 0.324)."""
    n = 1
    while wilson(0, n, z)[1] >= threshold:
        n += 1
    return n


def first_irrecoverable_step(branches: Sequence[tuple[int, int, int]], *, threshold: float = 0.05, z: float = 1.96) -> int | None:
    """``branches``: (step, successes, n) from C3 branch rollouts on a sampled subset. DESIGN: "the first t after which
    no continuation reaches success". The smallest sampled step t such that **every** sampled step >= t has a Wilson
    upper bound below ``threshold`` (a suffix condition, H1 V17: a later step that still succeeds clears earlier ones)."""
    best = None
    for step, s, n in sorted(branches, reverse=True):
        if n <= 0:
            continue
        if wilson(s, n, z)[1] < threshold:
            best = step
        else:
            break
    return best


# ---- aggregates ------------------------------------------------------------------------------------------------------------
def _groups(eps: Iterable[Episode]) -> dict[tuple[str, str, str], list[Episode]]:
    g: dict[tuple[str, str, str], list[Episode]] = {}
    for e in eps:
        if not e.masked:
            g.setdefault((e.task, e.policy, e.harness), []).append(e)
    return g


def _per_k(eps: Iterable[Episode], k: int, f: Callable[[list[Episode]], float]) -> tuple[float | None, int, int]:
    """(value, tasks used, tasks excluded) over **one task set** (G1 D2): every group with a measured sample. If any has
    fewer than k measured samples, the k is refused (None): averaging only the larger groups silently changes which
    tasks the number covers (pass@2 < pass@1 on one row)."""
    gs = _groups(eps)
    small = sum(1 for g in gs.values() if len(g) < k)
    if not gs or small:
        return None, len(gs) - small, small
    return round(sum(f(g) for g in gs.values()) / len(gs), 9), len(gs), 0


def _pass_at(g: list[Episode], k: int, threshold: float) -> float:
    n, c = len(g), sum(e.passed(threshold) for e in g)
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _pass_hat(g: list[Episode], k: int, threshold: float) -> float:
    return math.comb(sum(e.passed(threshold) for e in g), k) / math.comb(len(g), k)


def _max_at(g: list[Episode], k: int) -> float:
    xs = sorted(float(e.value or 0.0) for e in g)
    return sum(xs[i - 1] * math.comb(i - 1, k - 1) for i in range(k, len(xs) + 1)) / math.comb(len(xs), k)


def pass_at_k(eps: Iterable[Episode], k: int, *, threshold: float = 1.0) -> float | None:
    """Unbiased pass@k (Chen et al. 2021) per group, averaged over one task set (None if a task has < k measured)."""
    return _per_k(eps, k, lambda g: _pass_at(g, k, threshold))[0]


def pass_hat_k(eps: Iterable[Episode], k: int, *, threshold: float = 1.0) -> float | None:
    """pass^k: the probability that k samples drawn without replacement all pass (C(c, k) / C(n, k))."""
    return _per_k(eps, k, lambda g: _pass_hat(g, k, threshold))[0]


def max_at_k(eps: Iterable[Episode], k: int) -> float | None:
    """The unbiased expected maximum of k rewards drawn without replacement (exposed = 0): sorted ascending, value i
    (1-based) is the max with probability C(i-1, k-1) / C(n, k)."""
    return _per_k(eps, k, lambda g: _max_at(g, k))[0]


def coverage(eps: Sequence[Episode]) -> dict[str, Any]:
    """Planned / terminal / measured / masked / owed samples and tasks (H1 V5): ``fully_masked_tasks`` are tasks with
    nothing measured (masked or owed); ``masked_by_kind`` names each mask, ``kind/detail`` where a detail exists (G1 D9)."""
    tasks: dict[str, list[bool]] = {}
    kinds: dict[str, int] = {}
    for e in eps:
        tasks.setdefault(e.task, []).append(e.masked)
        if e.masked and not e.owed:
            k = e.failure_kind or "unknown"
            k = f"{k}/{e.failure_detail}" if e.failure_detail else k
            kinds[k] = kinds.get(k, 0) + 1
    return {"planned": len(eps), "episodes": sum(not e.owed for e in eps), "measured": sum(not e.masked for e in eps),
            "masked": sum(e.masked and not e.owed for e in eps), "owed": sum(e.owed for e in eps), "tasks": len(tasks),
            "measured_tasks": sum(not all(v) for v in tasks.values()), "fully_masked_tasks": sum(all(v) for v in tasks.values()),
            "masked_by_kind": dict(sorted(kinds.items()))}


# Student t quantile without scipy (production code stays scipy-free): the regularized incomplete beta by Lentz's
# continued fraction (Numerical Recipes betacf), inverted by bisection. Checked against scipy in the audit tests.
def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    ib = _betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - 0.5 * ib if t > 0 else 0.5 * ib


def t_ppf(p: float, df: float) -> float:
    """The Student t quantile (a two-sided 1 - alpha interval uses p = 1 - alpha/2)."""
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2.0


def task_mean_ci(task_means: Sequence[float], *, alpha: float = 0.05) -> dict[str, Any] | None:
    """A t interval on task means, the headline's own statistic (G1 D5 / H1 V9: the percentile bootstrap undercovered,
    0.76 at 3 tasks, 0.87 at 5). Tasks are the independent units; a task's samples are not. ``approximate`` below
    ``CI_MIN_TASKS`` tasks; None below 2 tasks."""
    xs = [float(x) for x in task_means]
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = t_ppf(1 - alpha / 2, n - 1) * sd / math.sqrt(n)
    return {"lo": round(m - h, 6), "hi": round(m + h, 6), "n_tasks": n, "level": 1 - alpha, "method": "t on task means",
            "approximate": n < CI_MIN_TASKS}


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


def _pct(xs: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (deterministic, no interpolation choice)."""
    if not xs:
        return None
    ys = sorted(xs)
    return round(ys[min(len(ys) - 1, max(0, math.ceil(q * len(ys)) - 1))], 6)


def _std(xs: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


TAMPER_KINDS = frozenset({"crucible:grader_tampered", "crucible:policy_misbehavior"})


def _hacks(scored: Sequence[Episode], threshold: float) -> dict[str, Any]:
    """Hack metrics (G2-1, campaign B1/B12) over measured episodes: exposures and grader tampering (both scored 0), and
    passes that carry a rule flag (``flagged_passes``: reported, not rescored here; the grade side decides)."""
    tampered = sum(1 for e in scored if e.failure_kind in TAMPER_KINDS)
    flagged = [e for e in scored if e.flags]
    fp = sum(1 for e in flagged if e.passed(threshold))
    n = len(scored)
    return {"exposed": sum(e.exposed for e in scored), "grader_tampered": tampered, "flagged": len(flagged), "flagged_passes": fp,
            "measured": n, "rate": round((sum(e.exposed for e in scored) + tampered + fp) / n, 9) if n else None,
            "flag_rules": _count(f for e in flagged for f in e.flags)}


def _count(xs: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def default_ks(k: int) -> tuple[int, ...]:
    """Every power of two up to k, and k itself (G1 D6 / H1 V16: callers kept ks=(1,) while sampling k = 2)."""
    out = {1, max(1, int(k))}
    p = 2
    while p <= k:
        out.add(p)
        p *= 2
    return tuple(sorted(out))


def _initial(e: Episode) -> set[str]:
    """Milestones held at the start: the ``step=0`` row M6 writes in ``ProgressTracker.on_start`` (H1 V2)."""
    return {m for r in e.rows if int(r.get("step", -1)) == 0 for m in (r.get("gained") or [])}


def _cost(eps: Sequence[Episode], scored: Sequence[Episode], threshold: float) -> dict[str, Any]:
    """All spend of every attempt (masked retries and owed samples' attempts included; H1 V4 / G1 D4) per measured success."""
    attempted = [e for e in eps if e.attempts]
    n_succ = sum(e.passed(threshold) for e in scored)
    gpus = [e.gpu_s for e in eps if e.gpu_s is not None]
    cost: dict[str, Any] = {
        "usd": round(sum(e.cost_usd for e in eps), 6), "usd_uncertain": round(sum(e.cost_uncertain_usd for e in eps), 6),
        "tokens": sum(e.tokens for e in eps), "tokens_in": sum(e.tokens_in for e in eps), "tokens_out": sum(e.tokens_out for e in eps),
        "wall_s": round(sum(e.wall_s for e in eps), 3), "gpu_s": round(sum(gpus), 3) if gpus else None,
        "attempts": sum(e.attempts for e in eps), "retries": sum(max(0, e.attempts - (0 if e.owed else 1)) for e in eps),
        "aborted_attempts": sum(e.aborted_attempts for e in eps), "successes": n_succ,
    }
    ps: dict[str, Any] = {k: (round(cost[k] / n_succ, 6) if n_succ and cost[k] is not None else None) for k in ("usd", "tokens", "wall_s", "gpu_s")}
    ps["usd_with_uncertain"] = round((cost["usd"] + cost["usd_uncertain"]) / n_succ, 6) if n_succ else None
    ps["tool_calls"] = round(sum(e.tool_calls for e in scored) / n_succ, 6) if n_succ else None
    cost["per_success"] = ps
    n = len(attempted) or 1
    cost["observability"] = {"episodes": len(attempted), "tokens": round(sum(e.tokens > 0 for e in attempted) / n, 6),
                             "usd": round(sum(e.cost_usd > 0 for e in attempted) / n, 6),
                             "gpu_s": round(sum(e.gpu_s is not None for e in attempted) / n, 6),
                             "wall_s": round(sum(e.wall_s > 0 for e in attempted) / n, 6)}
    lat = [float(e.latency_s) for e in scored if e.latency_s]
    cost["perf"] = {"latency_s_p50": _pct(lat, 0.5), "latency_s_p90": _pct(lat, 0.9), "latency_s_p99": _pct(lat, 0.99), "latency_episodes": len(lat),
                    "steps_mean": _mean([float(e.steps) for e in scored]),
                    "tokens_per_attempted_episode": round(cost["tokens"] / len(attempted), 3) if attempted else None}
    return cost


def summarize_group(eps: Sequence[Episode], *, ks: Sequence[int] = (1,), threshold: float = 1.0,
                    stages: Sequence[tuple[str, Sequence[str]]] | None = None,
                    branches: Mapping[str, Sequence[tuple[int, int, int]]] | None = None, ci_seed: int = 0,
                    stages_of: Callable[[str], Sequence[tuple[str, Sequence[str]]] | None] | None = None,
                    irrecoverable_threshold: float = 0.05) -> dict[str, Any]:
    """Every §11.1 metric over one group of episodes (one task × policy × harness, or any union of them). ``ci_seed`` is
    accepted for old callers; the interval is analytic now."""
    del ci_seed
    scored = [e for e in eps if not e.masked]
    cov = coverage(eps)
    cost = _cost(eps, scored, threshold)
    if not scored:  # an all-masked group: coverage only, never an invented 0
        return {"deterministic": {"coverage": cov}, "cost": cost}
    succ = [e for e in scored if e.passed(threshold)]
    aucs = [progress_auc(e.rows, e.steps, phi0=e.phi0) for e in scored if e.progress_tracked and e.phi_final is not None]  # Φ measured per step only
    st_of = (lambda t: stages_of(t)) if stages_of else (lambda t: stages)
    staged = [(e, st_of(e.task)) for e in scored if st_of(e.task)]
    reach = [stage_reach(final_milestones(e.rows, _initial(e)), st)[0] for e, st in staged]
    first = [stage_first_reached(e.rows, st, _initial(e)) for e, st in staged]
    names = sorted({n for f in first for n in f})
    n_stage_maps = {len(st) for _, st in staged}
    br = branches or {}
    with_br = [e for e in scored if br.get(e.attempt)]
    fis = [first_irrecoverable_step(br[e.attempt], threshold=irrecoverable_threshold) for e in with_br]
    need = min_branches(irrecoverable_threshold)
    faults = [e for e in scored if "fault_start" in e.tags]
    by_task: dict[str, list[float]] = {}
    raw_task: dict[str, list[float]] = {}
    by_idx: dict[int, dict[str, float]] = {}
    for e in scored:
        by_task.setdefault(e.task, []).append(float(e.value or 0.0))
        by_idx.setdefault(e.sample_idx, {})[e.task] = float(e.value or 0.0)
    for e in eps:
        if not e.owed:
            raw_task.setdefault(e.task, []).append(0.0 if e.masked else float(e.value or 0.0))
    task_means = [sum(v) / len(v) for _, v in sorted(by_task.items())]
    repeat_means = [sum(v.values()) / len(v) for _, v in sorted(by_idx.items())]
    sd_rep = _std(repeat_means)
    task_sds = [s for s in (_std(v) for v in by_task.values()) if s is not None]
    exposed = sum(e.exposed for e in scored)
    det: dict[str, Any] = {
        "coverage": cov,
        "mean_reward": _mean(task_means),  # the headline: mean of per-task means (G1 D3)
        "mean_reward_tasks": len(task_means),
        "mean_reward_episodes": _mean([float(e.value or 0.0) for e in scored]),  # pooled over measured episodes
        "mean_reward_raw": _mean([sum(v) / len(v) for _, v in sorted(raw_task.items())]),  # masked as 0: shown, never used
        "mean_reward_ci": task_mean_ci(task_means),
    }
    for k in ks:
        det[f"pass@{k}"], used, excl = _per_k(scored, k, lambda g, k=k: _pass_at(g, k, threshold))
        det[f"pass^{k}"] = _per_k(scored, k, lambda g, k=k: _pass_hat(g, k, threshold))[0]
        det[f"max@{k}"] = _per_k(scored, k, lambda g, k=k: _max_at(g, k))[0]
        det[f"pass@{k}_tasks"], det[f"pass@{k}_excluded"] = used, excl  # one set for pass@k, pass^k and max@k
    det.update({
        "spread": {"repeats": len(repeat_means), "std_across_repeats": None if sd_rep is None else round(sd_rep, 9),
                   "stderr_across_repeats": None if sd_rep is None else round(sd_rep / math.sqrt(len(repeat_means)), 9),
                   "mean_task_std": _mean(task_sds), "tasks_with_repeats": len(task_sds)},
        "exposure": {"exposed": exposed, "measured": len(scored), "rate": round(exposed / len(scored), 9)},
        "hacks": _hacks(scored, threshold),
        "failures": {"scored_by_kind": _count(e.failure_kind for e in scored if e.failure_kind),
                     "stop_reasons": _count(str(e.stop) for e in scored if e.stop)},
        "gate_profile": gate_profile(eps),
        "progress_auc": _mean(aucs),
        "progress_auc_episodes": len(aucs),
        "phi_final": _mean([float(e.phi_final) for e in scored if e.phi_final is not None]),
        "stage_reach": (_mean([float(r) for r in reach]) if len(n_stage_maps) == 1 else None) if staged else None,  # raw index: one stage count only
        "stage_reach_frac": _mean([(r + 1) / len(st) for r, (_, st) in zip(reach, staged, strict=True)]) if staged else None,
        "stage_reach_episodes": len(staged),
        "stage_reach_hist": {str(i): reach.count(i) for i in sorted(set(reach))} if staged else None,
        "stage_first_step": {n: _mean([float(f[n]) for f in first if f.get(n) is not None]) for n in names} if staged else None,
        "first_irrecoverable_step": _mean([float(x) for x in fis if x is not None]) if fis else None,
        "irrecoverable_episodes": sum(x is not None for x in fis) if fis else None,
        "branch_episodes": len(with_br),
        "branches_insufficient": sum(1 for e in with_br if all(n < need for _, _, n in br[e.attempt])) if with_br else None,
        "irrecoverable_min_branches": need,
        "regressions": sum(regressions(e.rows, _initial(e)) for e in scored),
        "recovery_rate": _mean([float(e.passed(threshold)) for e in faults]) if faults else None,
        "recovery_episodes": len(faults),
        "steps": _mean([float(e.steps) for e in scored]),
        "tool_calls": sum(e.tool_calls for e in scored),
        "messages": sum(e.messages for e in scored),
        "peer_messages": sum(e.peer_messages for e in scored),
        "messages_per_success": round(sum(e.messages for e in scored) / len(succ), 6) if succ else None,
        "successes": len(succ),
        "probes": {"attempted": sum(e.probes_attempted for e in eps), "exposed": sum(e.exposed for e in eps)},
    })
    return {"deterministic": det, "cost": cost}


def summarize(eps: Sequence[Episode], *, ks: Sequence[int] = (1,), threshold: float = 1.0,
              stages_of: Callable[[str], Sequence[tuple[str, Sequence[str]]] | None] | None = None,
              horizon_of: Callable[[str], str] | None = None,
              branches: Mapping[str, Sequence[tuple[int, int, int]]] | None = None,
              subsets_of: Callable[[str], Iterable[str]] | None = None, validity: Mapping[str, Mapping[str, Any]] | None = None,
              controls: Iterable[str] = ()) -> dict[str, Any]:
    """Per (policy, harness) × task, aggregated per horizon, per subset (``subsets_of(task)`` labels such as
    ``repo=pcl`` or a category; G1 D11, G2-18) and overall (DESIGN §11.1: every metric per task and per horizon). With
    two or more models, ``common_support`` scores them on the tasks measured for all (G2-1).

    ``validity`` (campaign B4/B5): a manifest {task: {"valid": bool, "reasons": [...]}} from M9 acceptance, an import
    report or a controls run (``validity_from_controls``). Invalid tasks (a broken or flaky oracle, a no-op that passes)
    leave every model's metrics and are reported under ``invalid_tasks`` with their reasons; without a manifest the
    summary says ``validity: unknown`` and warns. ``controls``: control policy ids (oracle, no-op), marked
    ``control: true`` and kept out of common support (and IRT, B16)."""
    controls = set(controls)
    vstat = validity_status(eps, validity)
    invalid = set(vstat["invalid_tasks"])
    by: dict[tuple[str, str], dict[str, list[Episode]]] = {}
    inv_by: dict[str, dict[str, list[Episode]]] = {}
    for e in eps:
        if e.task in invalid:
            inv_by.setdefault(e.task, {}).setdefault(f"{e.policy}@{e.harness}", []).append(e)
            continue
        by.setdefault((e.policy, e.harness), {}).setdefault(e.task, []).append(e)
    kw: dict[str, Any] = {"ks": ks, "threshold": threshold, "stages_of": stages_of, "branches": branches}
    models = []
    for (pol, har), tasks in sorted(by.items()):
        horizons: dict[str, list[Episode]] = {}
        subsets: dict[str, list[Episode]] = {}
        for t, es in tasks.items():
            horizons.setdefault(horizon_of(t) if horizon_of else "all", []).extend(es)
            for s in (subsets_of(t) if subsets_of else ()):
                subsets.setdefault(str(s), []).extend(es)
        m: dict[str, Any] = {"policy": pol, "harness": har, "tasks": {t: summarize_group(es, **kw) for t, es in sorted(tasks.items())},
                             "horizons": {h: summarize_group(es, **kw) for h, es in sorted(horizons.items())},
                             "overall": summarize_group([e for es in tasks.values() for e in es], **kw)}
        if subsets_of:
            m["subsets"] = {s: summarize_group(es, **kw) for s, es in sorted(subsets.items())}
        if pol in controls:
            m["control"] = True
        models.append(m)
    out: dict[str, Any] = {"models": models, "threshold": threshold, "ks": list(ks), "validity": {k: v for k, v in vstat.items() if k != "invalid_tasks"}}
    out["invalid_tasks"] = {t: {"reasons": list(vstat["invalid_tasks"][t]),
                                "models": {m: {"coverage": coverage(es), "mean_reward": _mean([float(e.value or 0.0) for e in es if not e.masked])}
                                           for m, es in sorted(inv_by.get(t, {}).items())}}
                            for t in sorted(vstat["invalid_tasks"])}
    rated = [e for e in eps if e.task not in invalid and e.policy not in controls]
    if len({(e.policy, e.harness) for e in rated}) >= 2:
        out["common_support"] = common_support(rated, threshold=threshold)
    return out


INVALID_REASONS = ("broken_oracle", "flaky_oracle", "noop_passes")


def validity_from_controls(eps: Sequence[Episode], *, oracle: str, noop: str, threshold: float = 1.0, min_oracle_runs: int = 3,
                           source: str = "controls") -> dict[str, dict[str, Any]]:
    """A validity manifest from control runs (campaign B4/B5; Repo2RLEnv's "gold 1.0, no-op 0.0: that is the contract"):
    per task, the oracle policy's measured samples must all pass (none passing = ``broken_oracle``, some =
    ``flaky_oracle``) and no no-op sample may pass (``noop_passes``). A task whose oracle was never measured, or that
    has no oracle, is ``valid: None`` (unknown), never silently valid; fewer than ``min_oracle_runs`` measured oracle
    samples is noted as ``weak``."""
    out: dict[str, dict[str, Any]] = {}
    for t in sorted({e.task for e in eps}):
        o = [e for e in eps if e.task == t and e.policy == oracle and not e.masked]
        n = [e for e in eps if e.task == t and e.policy == noop and not e.masked]
        reasons = []
        po = sum(e.passed(threshold) for e in o)
        if o and po == 0:
            reasons.append("broken_oracle")
        elif o and po < len(o):
            reasons.append("flaky_oracle")
        if any(e.passed(threshold) for e in n):
            reasons.append("noop_passes")
        out[t] = {"valid": (False if reasons else (True if o else None)), "reasons": reasons, "source": source,
                  "oracle": {"passed": po, "measured": len(o), "mean": _mean([float(e.value or 0.0) for e in o])},
                  "noop": {"passed": sum(e.passed(threshold) for e in n), "measured": len(n), "mean": _mean([float(e.value or 0.0) for e in n])},
                  **({"weak": True} if 0 < len(o) < min_oracle_runs else {})}
    return out


def validity_status(eps: Sequence[Episode], validity: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    """``status``: ``unknown`` (no manifest: nothing shows the tasks are solvable and not vacuous), ``manifest`` (every
    task has evidence) or ``partial``; the invalid tasks with reasons, and the tasks with no evidence."""
    tasks = sorted({e.task for e in eps})
    if validity is None:
        return {"status": "unknown", "valid_tasks": 0, "unknown_tasks": tasks, "invalid_tasks": {}}
    invalid = {t: list(validity[t].get("reasons") or ["invalid"]) for t in tasks if t in validity and validity[t].get("valid") is False}
    unknown = [t for t in tasks if t not in validity or validity[t].get("valid") is None]
    return {"status": "manifest" if not unknown else "partial", "valid_tasks": sum(1 for t in tasks if t in validity and validity[t].get("valid") is True),
            "unknown_tasks": unknown, "invalid_tasks": invalid,
            "sources": sorted({str(validity[t].get("source")) for t in tasks if t in validity and validity[t].get("source")})}


def common_support(eps: Sequence[Episode], *, threshold: float = 1.0) -> dict[str, Any]:
    """Every model scored on the tasks measured for **all** models (G2-1: models measured on different task sets are not
    comparable); each model's dropped tasks are listed."""
    measured: dict[str, set[str]] = {}
    for e in eps:
        measured.setdefault(f"{e.policy}@{e.harness}", set())
        if not e.masked:
            measured[f"{e.policy}@{e.harness}"].add(e.task)
    common = set.intersection(*measured.values()) if measured else set()
    models = {}
    for m in sorted(measured):
        bt: dict[str, list[Episode]] = {}
        for e in eps:
            if f"{e.policy}@{e.harness}" == m and e.task in common and not e.masked:
                bt.setdefault(e.task, []).append(e)
        models[m] = {"mean_reward": _mean([sum(float(x.value or 0.0) for x in v) / len(v) for _, v in sorted(bt.items())]),
                     "pass@1": _mean([sum(x.passed(threshold) for x in v) / len(v) for _, v in sorted(bt.items())]),
                     "measured_tasks": len(measured[m]), "dropped_tasks": sorted(measured[m] - common)}
    return {"tasks": len(common), "models": models}


def heatmap(eps: Sequence[Episode], *, bins: int = 10) -> dict[str, dict[str, list[float | None]]]:
    """Mean Φ per normalized-step bin, per model (policy@harness) and task: the report's per-step heatmap. Φ at the end of
    each bin is the last row at or before it in episode order (the ``step=-1`` correction at T, H1 V1); Φ_0 before any."""
    acc: dict[str, dict[str, list[list[float]]]] = {}
    for e in eps:
        if e.masked or not e.progress_tracked:
            continue
        T = max(e.steps, max((int(r["step"]) for r in e.rows), default=0), 1)
        pts = [(s, float(r["phi"])) for s, r in _ordered(e.rows, T)]
        start = float(e.phi0) if e.phi0 is not None else next((v for s, v in pts if s == 0), 0.0)
        cells = acc.setdefault(f"{e.policy}@{e.harness}", {}).setdefault(e.task, [[] for _ in range(bins)])
        for b in range(bins):
            t = (b + 1) * T / bins  # Φ at the end of the bin
            phi = start
            for step, value in pts:
                if step <= t:
                    phi = value
            cells[b].append(phi)
    return {m: {t: [round(sum(c) / len(c), 6) if c else None for c in cells] for t, cells in sorted(ts.items())} for m, ts in sorted(acc.items())}


def strip_cost(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The summary without its cost blocks, located **by path** (H1 V13: stripping every key named ``cost`` also dropped
    a task or gate called ``cost``): ``models[i].{overall, tasks.*, horizons.*, subsets.*}.cost`` and ``run.spend``."""
    s = json.loads(json.dumps(summary, default=str))
    for m in s.get("models") or []:
        blocks = [m.get("overall")] + [v for key in ("tasks", "horizons", "subsets") for v in (m.get(key) or {}).values()]
        for b in blocks:
            if isinstance(b, dict):
                b.pop("cost", None)
    if isinstance(s.get("run"), dict):
        s["run"].pop("spend", None)
    return s


def deterministic_digest(summary: dict[str, Any]) -> str:
    """A hash of every deterministic component (the cost blocks excluded by path): equal across re-runs with the same seeds."""
    import hashlib

    return hashlib.sha256(json.dumps(strip_cost(summary), sort_keys=True, default=str).encode()).hexdigest()[:16]


def episodes_json(eps: Sequence[Episode]) -> list[dict[str, Any]]:
    return [{k: v for k, v in asdict(e).items() if k != "rows"} for e in eps]
