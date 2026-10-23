"""Difficulty at factory time: the generator controller, the cheap predictor, checkpoint-conditioned difficulty and
checkpoint starts (swarm-pipeline/12). The training-time half (online p̂, the sampler, the reverse curriculum) is
``forge.difficulty`` and ``train.curriculum``.

1. **Controller** (``control``; pattern: R-Zero, Huang et al. 2025, arXiv 2508.05004: a Challenger is rewarded for
   questions the Solver answers about half the time, ``1 - 2|p̂ - ½|``). Here the "challenger" is the swarm's task
   generator and the "question" is a task variant: from a measured pass rate and the failure labels the controller emits
   one typed ``KnobAction``: ``ease`` (the K1-K6 knobs of ``factory.calibration.KNOBS``, ordered by the failure modes, or
   a checkpoint start), ``harden`` (H1-H3: add a real fault, take back an easing knob, a longer span), ``keep``,
   ``sample_more``, or ``fix_task`` / ``budget_bound`` when difficulty is not the problem (a spec defect, an unfair task,
   failures that are budget stops). Difficulty moves only through knobs: never a vaguer prompt, never a lie (V5), never
   anything in ``calibration.NEVER``. Every action carries its reason and the generator reward, and is journaled.
   Held-out tasks are refused (``HeldOutRefused``): they are eased only for defects, never from their traces (V9).
2. **Predictor** (``fit`` / ``Predictor``): p before the k=8 student run, from static features (faults, executed edges,
   modules, oracle lines, milestone tests, lane language, tool limit, fairness flags) and optional cheap probes (a few
   low-budget student rollouts, the frontier's pass rate) fused as a Beta update. A ridge logistic fit on the measured
   S15 records; ``loo`` reports leave-one-task-out error against the constant baseline. **Advisory only** (V6): it orders
   the expensive runs and suggests skips; it never keeps or rejects a task.
3. **Checkpoint-conditioned difficulty** (``conditional``): from each episode's milestone vector, through
   ``factory.checkpoint.credited`` and ``potential`` (the same C1 rule and weights as the step credit): p(full pass),
   mean Φ, p(m) and p(m | every prerequisite of m credited), and the stall point (the deepest DAG level reached).
4. **Checkpoint starts** (``checkpoint_levels``, ``checkpoint_start``, ``check_start``): the start tree with the real
   fixes of the first L milestones (a prerequisite-closed set, in DAG order) applied, built by the task's own
   ``simplify_units`` (the counterfactual control's machinery): the pre-applied milestones leave the task with their
   requirements, prompt lines and mutants, so they can never earn credit, and every gate runs again on the variant. The
   sidecar grades the variant's start as Φ0, so step credit counts only progress made after the start.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.calibration import KNOBS
from crucible.factory.checkpoint import credited, normalized, potential, uniform_weights
from crucible.forge.learnability import BANDS, Band, PassRate, decided, learnability

HARDEN: tuple[tuple[str, str, str], ...] = (
    ("H1", "Add a real fault", "re-break one more real fix from the same window pool that interacts with a kept fault (a new milestone, its own tests)"),
    ("H2", "Take back an easing knob", "remove the last K-knob applied to this task (a hint, a reproduction, an order), never anything the report needs"),
    ("H3", "A longer span", "an earlier start release, so more dependent real commits lie between the start and the oracle"),
)
CHECKPOINT = ("C1", "Checkpoint start", "start from the tree with the first fixes applied (the stall is after them): a reverse curriculum")

# failure mode (targeting ``modes`` keys: "<what>/<skill>") -> the easing knobs that answer it, best first
MODE_KNOBS: dict[str, tuple[str, ...]] = {
    "never_progressed": ("K3", "K2", "K4"),
    "multi_fault_localization": ("K3", "K2"),
    "reproduce_from_report": ("K4", "K6"),
    "claimed_done_without_progress": ("K4", "K6"),
    "stalled": ("C1", "K2", "K5"),
    "interacting_faults": ("C1", "K2", "K5"),
    "lost_progress": ("K5", "C1"),
    "regression": ("K5",),
}
BUDGET_KNOBS = ("C1", "K1")  # a budget-bound task gets a shorter horizon; the 300-call budget itself never moves
NOT_DIFFICULTY = {"spec_defect": "fix_task", "world_bug": "fix_task", "world_fault": "fix_task", "unfair": "fix_task", "budget": "budget_bound",
                  "env_flake": "fix_task"}


class HeldOutRefused(ValueError):
    """A difficulty action was asked for a held-out task (V9: never eased or hardened from its traces)."""


@dataclass(frozen=True)
class KnobAction:
    task: str
    action: str  # ease | harden | keep | sample_more | fix_task | budget_bound
    knobs: tuple[str, ...]
    reason: str
    p: float | None
    lo: float | None
    hi: float | None
    n: int
    generator_reward: float | None  # R-Zero's uncertainty reward 1 - 2|p - 1/2| of the variant that produced this measurement
    at: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["knob_text"] = {k: _knob_text(k) for k in self.knobs}
        return d


def _knob_text(k: str) -> str:
    for kid, name, what in (*KNOBS, *HARDEN, CHECKPOINT):
        if kid == k:
            return f"{name}: {what}"
    return k


def uncertainty_reward(p: float | None) -> float | None:
    """R-Zero's challenger reward: 1 at p = 1/2, 0 at p in {0, 1}."""
    return None if p is None else max(0.0, 1.0 - 2.0 * abs(float(p) - 0.5))


def _ease_order(modes: Mapping[str, int], conditional_in_band: bool) -> tuple[str, ...]:
    """K-knobs ordered by how many failures each answers (ties: the KNOBS order); the checkpoint start leads when the
    later milestones are in band once their prerequisites hold."""
    score: dict[str, float] = {}
    for key, n in modes.items():
        for part in str(key).split("/"):
            for rank, k in enumerate(MODE_KNOBS.get(part, ())):
                score[k] = score.get(k, 0.0) + float(n) / (rank + 1)
    if conditional_in_band:
        score["C1"] = score.get("C1", 0.0) + 1e6
    base = [k for k, _, _ in KNOBS]
    order = sorted(set(score) | set(base), key=lambda k: (-score.get(k, 0.0), base.index(k) if k in base else -1))
    return tuple(order)


def control(task: str, rate: PassRate, *, split: str = "train", band: Band | None = None, final: bool = True, causes: Mapping[str, int] | None = None,
            modes: Mapping[str, int] | None = None, skill_min: float = 0.7, conditional_rates: Mapping[str, float | None] | None = None,
            frontier_passes: bool | None = None, applied: Sequence[str] = ()) -> KnobAction:
    """One difficulty action for a measured task. ``causes``: failure cause -> count (the targeting cause labels:
    skill_gap, budget, spec_defect, ...); ``modes``: failure mode -> count; ``conditional_rates``: p(m | prerequisites)
    per milestone; ``applied``: knobs already applied (H2 takes the last one back)."""
    if str(split or "train") != "train":
        raise HeldOutRefused(f"{task} is {split}: held-out tasks are never eased or hardened from their traces (V9)")
    band = band or BANDS["train"]
    gr = uncertainty_reward(rate.p if rate.n else None)

    def act(action: str, knobs: Sequence[str], why: str) -> KnobAction:
        return KnobAction(task, action, tuple(knobs), why, rate.p if rate.n else None, rate.lo if rate.n else None, rate.hi if rate.n else None, rate.n, gr)

    if rate.n == 0:
        return act("sample_more", (), f"no measured draw ({rate.masked} masked)")
    if not final and not decided(rate, band):
        return act("sample_more", (), f"p = {rate.p:.2f}, interval [{rate.lo:.2f}, {rate.hi:.2f}] does not decide the band yet")
    causes = dict(causes or {})
    failures = sum(causes.values())
    if failures:
        top, top_n = max(causes.items(), key=lambda kv: kv[1])
        skill = causes.get("skill_gap", 0) / failures
        if top in NOT_DIFFICULTY and top_n / failures >= 0.5:
            if NOT_DIFFICULTY[top] == "budget_bound":  # the horizon is too long for the fixed budget: shorten it, never raise it (NEVER)
                return act("budget_bound", tuple(k for k in BUDGET_KNOBS if k not in applied),
                           f"{top_n}/{failures} failures are budget stops: shorten the horizon (a checkpoint start or one fault fewer), never a bigger budget")
            return act(NOT_DIFFICULTY[top], (), f"{top_n}/{failures} failures are {top}: not a difficulty problem")
        if skill < skill_min and (band.lo is None or rate.p < band.lo):
            return act("fix_task", (), f"only {skill:.0%} of failures are skill gaps (< {skill_min:.0%}): fix the task before easing it")
    if frontier_passes is False and (band.lo is not None and rate.p < band.lo):
        return act("fix_task", (), "no frontier solver passes either: check the report and the World before easing (unfair)")
    if band.hi is not None and rate.p > band.hi:
        knobs = ("H2", "H1", "H3") if applied else ("H1", "H3")
        return act("harden", knobs, f"p = {rate.p:.2f} above {band.hi}: little signal left (learnability {learnability(rate.p):.2f})"
                   + (f"; take back {applied[-1]} first" if applied else ""))
    if band.lo is not None and rate.p < band.lo:
        cond_ok = any(v is not None and (band.lo or 0) <= v <= (1.0 if band.hi is None else band.hi) for v in (conditional_rates or {}).values())
        knobs = tuple(k for k in _ease_order(modes or {}, cond_ok) if k not in applied)
        return act("ease", knobs, f"p = {rate.p:.2f} below {band.lo}: all-fail groups carry no signal"
                   + ("; later milestones are in band once their prerequisites hold: checkpoint start first" if cond_ok else ""))
    return act("keep", (), f"p = {rate.p:.2f} inside [{band.lo}, {band.hi}]")


@dataclass
class Journal:
    """Append-only JSONL of controller actions (``difficulty_actions.jsonl``)."""

    path: Path

    def add(self, a: KnobAction, **extra: Any) -> dict[str, Any]:
        row = {**a.to_json(), **extra}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return row

    def rows(self) -> list[dict[str, Any]]:
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()] if self.path.exists() else []


# ---- checkpoint-conditioned difficulty -------------------------------------------------------------------------------------


def _mid(m: Any) -> str:
    return str(m["id"] if isinstance(m, Mapping) else m.id)


def _req(m: Any) -> list[str]:
    r = m.get("requires") if isinstance(m, Mapping) else getattr(m, "requires", None)
    return [str(x) for x in r or ()]


def depth(milestones: Sequence[Any]) -> dict[str, int]:
    """DAG level per milestone (0: no prerequisite)."""
    req = {_mid(m): _req(m) for m in milestones}
    memo: dict[str, int] = {}

    def d(x: str, seen: frozenset[str] = frozenset()) -> int:
        if x in memo:
            return memo[x]
        v = 1 + max((d(r, seen | {x}) for r in req.get(x, ()) if r in req and r not in seen), default=-1)
        memo[x] = v
        return v

    return {m: d(m) for m in req}


def conditional(episodes: Iterable[Mapping[str, Any]], milestones: Sequence[Any], weights: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Full-task vs checkpoint-conditioned difficulty from episode rows (``milestones``: {id: bool}, ``passed``,
    ``masked``, ``stop``). Masked episodes and budget-stopped failures are left out of the rates (counted apart)."""
    from crucible.forge.difficulty import BUDGET_STOPS

    w = normalized(weights, milestones) if weights else uniform_weights(milestones)
    ids = [_mid(m) for m in milestones]
    req = {_mid(m): _req(m) for m in milestones}
    lv = depth(milestones)
    rows = [e for e in episodes if not e.get("masked")]
    budget = [e for e in rows if not e.get("passed") and str(e.get("stop") or "") in BUDGET_STOPS]
    live = [e for e in rows if e not in budget]
    per: dict[str, dict[str, Any]] = {}
    for m in ids:
        got = sum(1 for e in live if m in credited(e.get("milestones") or {}, milestones))
        base = [e for e in live if all(r in credited(e.get("milestones") or {}, milestones) for r in req[m])]
        cond = sum(1 for e in base if m in credited(e.get("milestones") or {}, milestones))
        per[m] = {"p": got / len(live) if live else None, "p_given_prereqs": cond / len(base) if base else None, "n_given": len(base),
                  "requires": req[m], "level": lv[m]}
    phis = [potential(e.get("milestones") or {}, milestones, w) for e in live]
    reached = [max((lv[m] for m in credited(e.get("milestones") or {}, milestones)), default=-1) for e in live]
    full = sum(1 for e in live if e.get("passed"))
    return {"n": len(live), "masked": sum(1 for e in episodes if e.get("masked")), "budget": len(budget),
            "p_full": full / len(live) if live else None, "phi_mean": statistics.fmean(phis) if phis else None,
            "phi_var": statistics.pvariance(phis) if len(phis) > 1 else 0.0, "milestones": per,
            "stall_level": statistics.median(reached) if reached else None,
            "stalled_after_first": (sum(1 for x in reached if x == 0) / len(reached)) if reached else None}


# ---- checkpoint starts -----------------------------------------------------------------------------------------------------


def checkpoint_levels(milestones: Sequence[Any]) -> list[list[str]]:
    """The pre-applied set of each checkpoint start, earliest first: level L applies the first L milestones in DAG order
    (by depth, then history order), always prerequisite-closed, and leaves at least one milestone. Level 0 (nothing
    applied) is the task itself and is not listed."""
    ids = [_mid(m) for m in milestones]
    req = {_mid(m): _req(m) for m in milestones}
    lv = depth(milestones)
    order = sorted(ids, key=lambda m: (lv[m], ids.index(m)))
    out: list[list[str]] = []
    for L in range(1, len(order)):
        pre = order[:L]
        if all(r in pre for m in pre for r in req[m]):
            out.append(pre)
    return out


def checkpoint_start(task: Any, pre: Sequence[str], *, level: int, root: Path | None = None) -> Any:
    """The variant that starts from ``task``'s start with the real fixes of milestones ``pre`` applied
    (``GeneratedTask.simplify_units``; kind ``checkpoint``). It carries ``curriculum = (task id, level)`` for the
    sampler, and ``checkpoint_of`` with the pre-applied milestones. A new task: S5, S8 and the start check run again."""
    if str(getattr(task, "split", "train") or "train") != "train":
        raise HeldOutRefused(f"{getattr(task, 'task_id', task)} is held out: no curriculum variants (V9)")
    by_id = {_mid(m): m for m in task.milestones}
    units = {str(by_id[m]["unit"] if isinstance(by_id[m], Mapping) else by_id[m].unit) for m in pre}
    v = task.simplify_units(units, kind="checkpoint", root=root)
    v.curriculum = (str(task.task_id), int(level))
    v.checkpoint_of = {"task": str(task.task_id), "level": int(level), "pre_applied": sorted(pre)}
    return v


def check_start(original_milestones: Sequence[Any], pre: Sequence[str], status_at_start: Mapping[str, bool | None]) -> list[str]:
    """Problems with a checkpoint start, from the ORIGINAL task's milestone status graded on the variant's start tree:
    every pre-applied milestone must be credited there (the fixes really hold), every other one must fail (something is
    left to solve). Unmeasured is a problem (never read as pass or fail)."""
    got = credited({k: bool(v) for k, v in status_at_start.items() if v is not None}, original_milestones)
    probs = [f"{m}: unmeasured at the checkpoint start" for m in (_mid(x) for x in original_milestones) if status_at_start.get(m) is None]
    probs += [f"{m}: pre-applied but not credited at the start" for m in pre if m not in got and status_at_start.get(m) is not None]
    probs += [f"{m}: already passes at the start, nothing to solve" for m in (_mid(x) for x in original_milestones)
              if m not in pre and status_at_start.get(m)]
    return probs


async def gate_start(variant: Any, *, original_milestones: Sequence[Any], pre: Sequence[str],
                     status: Callable[[Any], Any], gates: Callable[[Any], Any]) -> dict[str, Any]:
    """Run the start check and the variant's gates (``gates(variant)`` -> a mapping with ``ok``, e.g.
    ``generators.base.deterministic_gates``). ``status(variant)`` grades the original milestones on the variant's start."""
    st = await status(variant)
    probs = check_start(original_milestones, pre, st)
    g = await gates(variant) if not probs else {"ok": False, "skipped": "start check failed"}
    ok = not probs and bool((g or {}).get("ok"))
    return {"ok": ok, "start_problems": probs, "gates": g, "task": str(getattr(variant, "task_id", "")), "pre_applied": sorted(pre)}


# ---- the cheap predictor ---------------------------------------------------------------------------------------------------

LANGS = ("go", "python", "rust", "node", "java", "cuda")
FEATURES = ("bias", "faults", "edges", "modules", "oracle_lines_log", "milestone_tests_log", "tool_limit_300", "fairness_flags",
            *(f"lang_{x}" for x in LANGS))


def features(rec: Mapping[str, Any], *, tool_limit: int | None = None) -> dict[str, float]:
    """Static features from a candidate record (``data/debug/candidates.json`` rows: faults, edges, modules,
    oracle_lines, milestone_tests, lang, fairness_flags) and the episode budget."""
    lang = str(rec.get("lang") or "")
    f = {"bias": 1.0, "faults": float(rec.get("faults") or 0), "edges": float(rec.get("edges") or 0), "modules": float(rec.get("modules") or 0),
         "oracle_lines_log": math.log1p(float(rec.get("oracle_lines") or 0)), "milestone_tests_log": math.log1p(float(rec.get("milestone_tests") or 0)),
         "tool_limit_300": float(tool_limit or rec.get("tool_limit") or 300) / 300.0, "fairness_flags": float(len(rec.get("fairness_flags") or []))}
    f.update({f"lang_{x}": 1.0 if lang == x else 0.0 for x in LANGS})
    return f


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


@dataclass
class Predictor:
    coef: dict[str, float]
    mean: dict[str, float]
    scale: dict[str, float]
    n_tasks: int
    prior_strength: float = 4.0  # pseudo-draws of the static prediction in the probe fusion

    def p(self, feats: Mapping[str, float]) -> float:
        z = sum(self.coef.get(k, 0.0) * ((feats.get(k, 0.0) - self.mean.get(k, 0.0)) / self.scale.get(k, 1.0) if k != "bias" else 1.0) for k in self.coef)
        return _sig(z)

    def predict(self, feats: Mapping[str, float], *, probe: tuple[int, int] | None = None, frontier: float | None = None) -> dict[str, Any]:
        """The prediction, fused with a cheap probe (k passes of n low-budget rollouts) as a Beta update. ``frontier``: a
        frontier solver's pass rate; 0 marks the task as possibly unfair (the controller decides, not this)."""
        p0 = self.p(feats)
        a, b = p0 * self.prior_strength, (1 - p0) * self.prior_strength
        if probe is not None:
            a, b = a + probe[0], b + (probe[1] - probe[0])
        mean = a / (a + b)
        sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        return {"p_static": round(p0, 4), "p": round(mean, 4), "lo": round(max(0.0, mean - 1.96 * sd), 4), "hi": round(min(1.0, mean + 1.96 * sd), 4),
                "probe": list(probe) if probe else None, "frontier": frontier, "maybe_unfair": frontier == 0.0, "advisory": True}

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def fit(rows: Sequence[tuple[Mapping[str, float], int, int]], *, l2: float = 1.0, iters: int = 3000, lr: float = 0.05) -> Predictor:
    """Ridge logistic regression on (features, passes, n) with a binomial likelihood (standardized features, bias not
    penalized), by gradient descent: stdlib only (numpy is an optional extra)."""
    keys = list(FEATURES)
    cols = {k: [r[0].get(k, 0.0) for r in rows] for k in keys}
    mean = {k: (statistics.fmean(v) if k != "bias" else 0.0) for k, v in cols.items()}
    scale = {k: (statistics.pstdev(v) or 1.0) if k != "bias" else 1.0 for k, v in cols.items()}
    X = [[1.0 if k == "bias" else (r[0].get(k, 0.0) - mean[k]) / scale[k] for k in keys] for r in rows]
    w = [0.0] * len(keys)
    N = sum(r[2] for r in rows) or 1
    for _ in range(iters):
        g = [0.0] * len(keys)
        for x, (_, k, n) in zip(X, rows, strict=True):
            mu = _sig(sum(a * b for a, b in zip(w, x, strict=True)))
            for j, xj in enumerate(x):
                g[j] += (n * mu - k) * xj
        for j in range(len(keys)):
            pen = 0.0 if keys[j] == "bias" else l2 * w[j]
            w[j] -= lr * (g[j] / N + pen / N)
    return Predictor(dict(zip(keys, w, strict=True)), mean, scale, n_tasks=len({id(r[0]) for r in rows}))


def loo(rows: Sequence[tuple[str, Mapping[str, float], int, int]], **kw: Any) -> dict[str, Any]:
    """Leave-one-task-out: every run of a task is held out together. Returns per-task predictions and the mean
    absolute error of p against the constant baseline (the training rows' pooled rate), so the predictor shows whether
    it beats guessing."""
    tasks = sorted({r[0] for r in rows})
    preds, errs, base_errs = [], [], []
    for t in tasks:
        train = [(f, k, n) for tid, f, k, n in rows if tid != t]
        test = [(f, k, n) for tid, f, k, n in rows if tid == t]
        if not train or not test:
            continue
        m = fit(train, **kw)
        pooled = sum(k for _, k, _ in train) / max(1, sum(n for _, _, n in train))
        for f, k, n in test:
            p_hat, p_true = m.p(f), k / n if n else 0.0
            preds.append({"task": t, "p_true": round(p_true, 4), "p_pred": round(p_hat, 4), "baseline": round(pooled, 4), "n": n})
            errs.append(abs(p_hat - p_true))
            base_errs.append(abs(pooled - p_true))
    return {"tasks": len(tasks), "rows": len(preds), "mae": round(statistics.fmean(errs), 4) if errs else None,
            "mae_baseline": round(statistics.fmean(base_errs), 4) if base_errs else None, "predictions": preds}


def plan(preds: Mapping[str, Mapping[str, Any]], *, skip_below: float = 0.02, skip_above: float = 0.98) -> list[dict[str, Any]]:
    """The order to run the expensive k=8 measurement in: most learnable first (|p - 1/2| ascending), with a skip
    *suggestion* when the whole interval is near 0 or 1. Advisory: a suggested skip is still measured before any keep or
    reject, because only executed runs decide (V6)."""
    out = []
    for t, d in preds.items():
        out.append({"task": t, "p": d["p"], "lo": d.get("lo"), "hi": d.get("hi"), "learnability": round(learnability(float(d["p"])), 4),
                    "suggest_skip": bool((d.get("hi") is not None and d["hi"] < skip_below) or (d.get("lo") is not None and d["lo"] > skip_above)),
                    "advisory": True})
    return sorted(out, key=lambda r: (abs(float(r["p"]) - 0.5), r["task"]))


def training_rows(targeting_runs: Sequence[Mapping[str, Any]], candidates: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, dict[str, float], int, int]]:
    """(task, features, passes, n) from S15 targeting run JSONs (``tasks[]``: task, passes, n, budget_stops) and the
    candidate records. Budget stops are taken out of n (they are not skill observations). Held-out rows are never used:
    the predictor orders training-task work, and held-out traces must not shape training tasks (V9)."""
    out = []
    for run in targeting_runs:
        tl = int(((run.get("settings") or {}).get("env_budget") or {}).get("tool_limit") or 150)
        for t in run.get("tasks") or []:
            if str(t.get("split") or "train") != "train":
                continue
            tid = str(t.get("task"))
            rec = dict(candidates.get(tid) or {"faults": t.get("faults"), "lang": t.get("lang")})
            stops = (t.get("budget_stops") or {}).get("student") or 0
            n = int(t.get("n") or 0) - int(stops)
            if n <= 0:
                continue
            out.append((tid, features(rec, tool_limit=tl), int(t.get("passes") or 0), n))
    return out


LIMIT_STOPS = frozenset({"max_steps", "tool_limit", "budget"})


def backfill_budget_stops(run: Mapping[str, Any], arms: Path | None) -> dict[str, Any]:
    """A targeting run whose rows predate the ``budget_stops`` field (the 150-call S15 run) with it filled in from the
    traces: per task, the student's failed, unmasked episodes that ended on the step limit (``episodes.jsonl`` under
    ``<arms>/student/<task>``). Without the episodes, the row's own stop counts (``stops['max_steps']``, at most the
    failures) stand in. Rows that already carry ``budget_stops`` are unchanged; each filled row says where it came from."""
    out = json.loads(json.dumps(run, default=str))
    for t in out.get("tasks") or []:
        if t.get("budget_stops") is not None:
            continue
        ep = Path(arms) / "student" / str(t.get("task")) / "episodes.jsonl" if arms is not None else None
        if ep is not None and ep.exists():
            rows = [json.loads(x) for x in ep.read_text().splitlines() if x.strip()]
            n = sum(1 for r in rows if not r.get("masked") and not r.get("passed") and r.get("stop") in LIMIT_STOPS)
            t["budget_stops"], t["budget_stops_source"] = {"student": n}, "episodes"
        else:
            fails = max(0, int(t.get("n") or 0) - int(t.get("passes") or 0))
            t["budget_stops"] = {"student": min(fails, int((t.get("stops") or {}).get("max_steps") or 0))}
            t["budget_stops_source"] = "stop counts"
    return out


def predictor_report(targeting_runs: Sequence[Mapping[str, Any]], candidates: Mapping[str, Mapping[str, Any]], **kw: Any) -> dict[str, Any]:
    rows = training_rows(targeting_runs, candidates)
    rep = loo(rows, **kw)
    rep["fit"] = fit([(f, k, n) for _, f, k, n in rows], **kw).to_json() if rows else None
    return rep
