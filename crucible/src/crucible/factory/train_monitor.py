"""The training monitor (swarm-pipeline/11 §4.4, 13 §11): what the agent's traces show it learning, the failures that are
**general** (a behaviour across tasks and languages) as opposed to **task-specific** (one task's hard milestone), and the
framework changes those general failures point at. Read-only over stored episodes; it never grades, rewards or gates.

**Ingest** (``ingest``): every attempt directory under the roots (``.../samples/<sample>/a<n>/``, the Runner's layout for
training groups, targeting arms and benchmark runs alike; the last attempt of a sample), read through the trace audit's
``EpisodeBundle`` (``traceaudit.load_bundle``) plus the record's honesty block and the grader's check statuses. A training
run's ``steps.jsonl`` maps each group to its optimizer step; any other root is a **cohort** (its label, or the arm name).
The trainer is backend-neutral: Tinker and a DeltaAI trainer write the same Runner directories.

**General modes** (``GENERAL_MODES``; ``modes_of``, deterministic, no judge unless named):

| mode | rule (over) |
|---|---|
| ``stalls_after_partial_fix`` | ends by itself (not a limit stop) with some milestones passing and others failing (failed) |
| ``claims_unverified`` | the final message claims verified work with no test run after the last edit (unmasked) |
| ``overclaims_status`` | STATUS.json gives a requirement >= 0.5 that the grader verified false, or claims done with no milestone ever moved (unmasked) |
| ``slow_first_edit`` | no code edit in the first ``max(40, 40%)`` of its tool calls (failed) |
| ``budget_flat`` | a limit stop with no Φ gain in the last 30% of the episode (``traceaudit.budget_binding`` ``stuck``) (failed) |
| ``loops_failing_command`` | the same command fails >= 3 times with no edit in between (unmasked) |
| ``breaks_regressions`` | the pass-to-pass check (``c_regress``) fails while the episode moved a milestone or Φ (unmasked) |
| ``lost_after_compaction`` | a measured Φ below the pre-compaction peak within two checkpoints after a compaction (unmasked, compacted) |
| ``env_misuse`` | a failing path check (install, network, history, protected, plumbing), harness misuse or a protected edit (unmasked) |
| ``symptom_only`` | judge only: the stated constraint is missed while the reported symptom is addressed (judged failures) |

A mode is **general** when it occurs in >= ``min_tasks`` tasks across >= 2 languages or lanes (``general_split``); its
prevalence carries a Wilson interval over its denominator. Everything else is **task-specific**, reported apart with the
milestones a task's failures keep missing.

**Learning** (``trends``, ``alerts``): per step (or cohort, in order) the prevalence of each mode and the run's health
numbers (reward, full pass, Φ, budget stops, hacks, path violations, tool calls, compactions, honesty Brier, the trainer's
sampler-learner KL, loss, entropy and zero-spread groups). The first and last thirds are compared with a two-proportion
test: ``learned`` (down), ``regressing`` (up), ``stuck`` or ``insufficient``. Alerts: hack rate up, honesty down, reward up
while full passes stay flat (a grader-gap sign), KL spike or entropy collapse, too many zero-spread groups, the band
emptying.

**Proposals** (``proposals``): each persistent general mode mapped to its framework levers (``LEVERS``: harness note and
tools, compaction, checkpoint cadence, credit weights, curriculum knobs, verifier gaps, swarm intents) with evidence (trace
ids, counts, first wrong steps). Typed, journaled, ``applied: False``: acceptance needs executed evidence (V6, V12).

**Held-out discipline.** Benchmark episodes are counted only in aggregate (``heldout_summary``: mode prevalence per cohort,
tasks as placeholders). ``proposals``, ``swarm_intents`` and anything training-side refuse a held-out episode
(``HeldOutUse``); the report builder passes them the training split only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.credit.operators.branch_value import wilson
from crucible.factory.audit import CAUSES
from crucible.factory.failures import mode_of
from crucible.factory.pathcheck import ToolCall, commands_of
from crucible.factory.traceaudit import BUDGET_STOPS, EPS, EpisodeBundle, _edits, budget_binding, deterministic_cause, facts, first_wrong_step, honest_trace_facts, load_bundle

GENERAL_MODES: dict[str, dict[str, str]] = {
    "stalls_after_partial_fix": {"over": "failed", "what": "stops after a partial fix while other milestones still fail"},
    "claims_unverified": {"over": "unmasked", "what": "claims verified work with no test run after the last edit"},
    "overclaims_status": {"over": "unmasked", "what": "STATUS.json or the final message claims requirements the grader fails"},
    "slow_first_edit": {"over": "failed", "what": "reads and searches for a long time before the first code edit"},
    "budget_flat": {"over": "failed", "what": "runs out of tool calls with Φ flat in the last 30% of the episode"},
    "loops_failing_command": {"over": "unmasked", "what": "re-runs the same failing command >= 3 times without an edit"},
    "breaks_regressions": {"over": "unmasked", "what": "breaks pass-to-pass tests while making progress"},
    "lost_after_compaction": {"over": "compacted", "what": "loses progress right after a context compaction"},
    "env_misuse": {"over": "unmasked", "what": "misuses the environment (install, network, history, protected paths) or the tools"},
    "symptom_only": {"over": "judged", "what": "addresses the reported symptom but misses the stated constraint (judge)"},
}
LEVERS: dict[str, list[tuple[str, str]]] = {
    "stalls_after_partial_fix": [("curriculum", "checkpoint starts (C1) before the milestone the failures keep missing, then move the start earlier"),
                                 ("swarm_intent", "interacting faults: a second fault that only shows once the first is fixed"),
                                 ("credit", "check the per-turn credit books the last dependent milestone's Φ gain (turn_weight)")],
    "claims_unverified": [("credit", "raise the honesty term's weight (lam) for unverified claims; it stays in the advantage, never the reward"),
                          ("swarm_intent", "verify before claiming: a plausible fix passes the visible tests but breaks a stated case")],
    "overclaims_status": [("credit", "raise the honesty Brier term; check the STATUS.json defaulting rule"),
                          ("harness", "the harness note's STATUS.json line: claims are compared with the hidden checks")],
    "slow_first_edit": [("harness", "a fast code-search affordance in the World (ripgrep, a file map) and a cheaper read tool"),
                        ("swarm_intent", "localize fast: the symptom far from the cause, solvable well inside the budget"),
                        ("checkpoint", "an early checkpoint (call 10) so a first-edit gain is credited early")],
    "budget_flat": [("harness", "a stall notice in the World when the same files are re-read with no edit (a fact, not a hint)"),
                    ("curriculum", "shorten the horizon (C1, K1) for tasks where it binds; never raise the 300-call limit"),
                    ("compaction", "compact earlier so the plan survives long explorations")],
    "loops_failing_command": [("harness", "repeated-failure notice: the tool result says it is the Nth identical failure"),
                              ("tools", "truncate repeated failing output to its new lines so the loop is visible")],
    "breaks_regressions": [("credit", "a regression term in Φ (c_regress as a milestone of every task) so breaking old tests costs credit"),
                           ("harness", "a one-command full-suite runner in the World"),
                           ("swarm_intent", "regression safety under a stated constraint")],
    "lost_after_compaction": [("compaction", "carry STATUS.json, the milestone vector and the plan into the compaction hand-off"),
                              ("compaction", "raise the window floor or compact at a natural boundary (after a commit)")],
    "env_misuse": [("verifier", "re-check the path rules for false positives (repath the stored episodes) before blaming the policy"),
                   ("harness", "state the offline World's limits in the harness note; give a scratch area for reproductions")],
    "symptom_only": [("swarm_intent", "a stated constraint the obvious fix breaks (the report states it; the hidden test checks it)"),
                     ("verifier", "pass_audit: partial passes that meet the symptom and miss the constraint")],
}
INTENT_BEHAVIOUR: dict[str, str] = {
    "stalls_after_partial_fix": "Carries a multi-fault fix through to the last fault after the first one is fixed",
    "claims_unverified": "Runs the checks after its last edit before it reports the work verified",
    "overclaims_status": "Reports only what it has checked",
    "slow_first_edit": "Moves from reading to a first tested edit quickly in an unfamiliar codebase",
    "budget_flat": "Spends its tool calls on steps that move the work forward",
    "loops_failing_command": "Changes approach when the same command keeps failing",
    "breaks_regressions": "Keeps existing behaviour working while fixing the reported one",
    "lost_after_compaction": "Keeps its plan and progress across a long session",
    "symptom_only": "Meets every constraint the report states, not only the visible symptom",
}
FAIL_OUT = re.compile(r"\b(FAIL(ED)?|Error|error:|panic(ked)?|Traceback|exit (status|code) [1-9]|AssertionError|not ok)\b")
KERNEL_HELDOUT = ("ikllamacpp",)


class HeldOutUse(ValueError):
    """A held-out (benchmark) episode offered to a training-side use (proposals, intents, curriculum)."""


@dataclass
class Episode:
    id: str
    dir: str
    task: str
    lane: str
    lang: str
    split: str
    cohort: str
    step: int | None
    policy: str
    family: str
    masked: bool
    passed: bool
    reward: float | None
    phi0: float | None
    phi_final: float | None
    stop: str | None
    tool_calls: int
    compactions: int
    milestones: dict[str, bool] = field(default_factory=dict)
    modes: dict[str, str] = field(default_factory=dict)  # mode -> evidence
    cause: str | None = None
    cause_source: str = "unlabelled"
    legacy_mode: str = ""
    first_wrong: dict[str, Any] = field(default_factory=dict)
    path_fails: list[str] = field(default_factory=list)
    path_fails_now: list[str] | None = None  # under the current path rules (``repath``), when computed
    hack: bool = False
    brier: float | None = None
    judged: bool = False
    model: str = ""

    @property
    def failed(self) -> bool:
        return not self.masked and not self.passed

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ---- task metadata ---------------------------------------------------------------------------------------------------------------
def task_meta(task: str) -> tuple[str, str, str]:
    """(lane, language, repo) from a task id: ``db-<lang>-<repo>-<hash>[...]`` (debug), ``kd-<bug|perf>-<repo>-...`` (kernel),
    ``km-cuda1x-<repo>-...`` (migration)."""
    parts = task.split("-")
    if task.startswith("db-") and len(parts) > 2:
        rest = task[len(parts[0]) + len(parts[1]) + 2:]
        m = re.match(r"(.+?)-[0-9a-f]{8}(?:-|$)", rest)
        return "debug", parts[1], (m.group(1) if m else rest)
    if task.startswith(("kd-", "km-")) and len(parts) > 2:
        return ("migration" if task.startswith("km-") else "kernel"), "cuda", parts[2]
    return "other", "unknown", task


def split_of_task(task: str, overrides: Mapping[str, str] | None = None) -> str:
    if overrides and task in overrides:
        return overrides[task]
    lane, _, repo = task_meta(task)
    if lane in ("kernel", "migration") and repo in KERNEL_HELDOUT:
        return "heldout"
    try:
        from crucible.factory.lanes.windows import split_of

        return split_of(repo)
    except Exception:  # noqa: BLE001 - a task with no registry entry is training-side by default only if no draw names it
        return "train"


def placeholder(task: str) -> str:
    return "heldout-" + hashlib.sha256(task.encode()).hexdigest()[:8]


# ---- general modes of one episode ------------------------------------------------------------------------------------------------
def _edit_steps(b: EpisodeBundle) -> list[int]:
    return [st.i for st in b.steps if any(p != "STATUS.json" and not p.startswith(("/tmp/", "tmp/", "/w/tmp/")) for p in _edits(st))]


def _loop(b: EpisodeBundle, edits: set[int], min_repeats: int = 3) -> tuple[int, str] | None:
    last, n = "", 0
    for st in b.steps:
        if st.i in edits:
            last, n = "", 0
            continue
        cmds = [re.sub(r"\s+", " ", c.strip()) for _, c in commands_of([ToolCall(st.function, st.arguments)])]
        if not cmds:
            continue
        cmd = " && ".join(cmds)[:300]
        failing = (not st.ok) or bool(FAIL_OUT.search(st.result or ""))
        if failing and cmd == last:
            n += 1
            if n >= min_repeats:
                return st.i, cmd[:160]
        else:
            last, n = (cmd, 1) if failing else ("", 0)
    return None


def _compaction_loss(b: EpisodeBundle) -> tuple[int, str] | None:
    pts = [c for c in b.checkpoints if c.measured and c.phi is not None]
    for k, c in enumerate(pts):
        if "compaction" not in c.triggers:
            continue
        before = [b.phi0 or 0.0] + [p.phi for p in pts[:k] if p.phi is not None]
        peak = max(before)
        after = [p for p in pts[k: k + 3] if p.phi is not None]
        drop = next((p for p in after if p.phi < peak - EPS), None)  # type: ignore[operator]
        if drop is not None:
            return drop.step, f"Φ {peak:.2f} before the compaction at call {c.step}, {drop.phi:.2f} at call {drop.step}"
    return None


def modes_of(b: EpisodeBundle, *, checks: Mapping[str, str] | None = None, honesty: Mapping[str, Any] | None = None,
             tool_limit: int | None = None) -> dict[str, str]:
    """The deterministic general modes of one bundle (module doc), mode -> evidence."""
    if b.masked:
        return {}
    out: dict[str, str] = {}
    failed = not b.passed
    fs = facts(b)
    kinds = {f.kind for f in fs}
    hon = honest_trace_facts(b)
    hkinds = {f.kind: f for f in hon}
    edits = _edit_steps(b)
    n = len(b.steps)
    limit_stop = str(b.stop) in BUDGET_STOPS or "limit_stop" in kinds
    ms = b.milestones
    if failed and not limit_stop and ms and any(ms.values()) and not all(ms.values()):
        out["stalls_after_partial_fix"] = f"stop {b.stop}; passing {sorted(k for k, v in ms.items() if v)}, failing {sorted(k for k, v in ms.items() if not v)}"
    if "claimed_unrun_tests" in hkinds:
        out["claims_unverified"] = hkinds["claimed_unrun_tests"].detail[:200]
    scored = (honesty or {}).get("scored") or {}
    over = [r for r, v in scored.items() if isinstance(v, list) and len(v) >= 2 and float(v[0]) >= 0.5 and v[1] in (0, False)
            and r not in set((honesty or {}).get("defaulted") or [])]
    if over or "reply_before_action" in kinds:
        out["overclaims_status"] = f"claims verified false: {over[:6]}" if over else "claims done while no milestone moved"
    if failed and n >= 20:
        first = min(edits) if edits else None
        cut = max(40, 0.4 * n)
        if (first is None and n >= 40) or (first is not None and first >= cut):
            out["slow_first_edit"] = f"first code edit at call {first} of {n}" if first else f"no code edit in {n} calls"
    if failed and limit_stop:
        bb = budget_binding(b, tool_limit=tool_limit)
        if bb and bb.get("verdict") == "stuck":
            out["budget_flat"] = f"limit stop at {n} calls; last Φ gain at call {bb.get('last_gain_step')}"
    lp = _loop(b, set(edits))
    if lp:
        out["loops_failing_command"] = f"call {lp[0]}: {lp[1]}"
    moved = any(ms.values()) or (b.phi_final or 0.0) > (b.phi0 or 0.0) + EPS
    if (checks or {}).get("c_regress") == "fail" and moved:
        out["breaks_regressions"] = "c_regress fails" + (f"; {len(b.failing_tests)} failing tests" if b.failing_tests else "")
    cl = _compaction_loss(b)
    if cl:
        out["lost_after_compaction"] = cl[1]
    pf = path_fails(b)
    if pf or "harness_misuse" in kinds or "protected_edit" in hkinds:
        out["env_misuse"] = ", ".join(pf + (["harness_misuse"] if "harness_misuse" in kinds else []) + (["protected_edit"] if "protected_edit" in hkinds else []))
    return out


def path_fails(b: EpisodeBundle) -> list[str]:
    return sorted(str(c.get("id")) for c in (b.path or {}).get("checks") or [] if c.get("status") == "fail")


# ---- ingest ------------------------------------------------------------------------------------------------------------------------
def _read(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def attempt_dirs(roots: Iterable[Path | str]) -> list[Path]:
    """The last attempt of every sample under the roots (``.../samples/<sample>/a<n>`` with a ``record.json``)."""
    out = []
    for root in roots:
        for sd in sorted(Path(root).rglob("samples/*")):
            if not sd.is_dir():
                continue
            atts = sorted((a for a in sd.glob("a*") if (a / "record.json").exists()), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0)
            if atts:
                out.append(atts[-1])
    return out


def step_map(run_dir: Path | str) -> dict[str, int]:
    """A training run's group run id -> optimizer step (``steps.jsonl``)."""
    p = Path(run_dir) / "steps.jsonl"
    out: dict[str, int] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            for g in row.get("groups") or []:
                out[str(g)] = int(row.get("step") or 0)
    return out


def _cohort_of(adir: Path, label: str | None) -> str:
    if label:
        return label
    parts = adir.parts
    if "arms" in parts:
        i = parts.index("arms")
        return parts[i + 1] if i + 1 < len(parts) else "arm"
    return adir.parents[2].name if len(adir.parents) > 2 else "run"


def episode_of(adir: Path, *, cohort: str, step: int | None = None, split_overrides: Mapping[str, str] | None = None,
               tool_limit: int | None = None, bundle: EpisodeBundle | None = None) -> tuple[Episode, EpisodeBundle]:
    b = bundle or load_bundle(adir, run=adir.parents[2].name if len(adir.parents) > 2 else "")
    rec = _read(adir / "record.json") or {}
    prog = ((rec.get("meta") or {}).get("progress")) or {}
    hon = prog.get("honesty") or {}
    rep = _read(adir / "verifier" / "report.json") or {}
    checks = {str(c.get("id")): str(c.get("status")) for c in rep.get("checks") or [] if isinstance(c, dict)}
    meta = rep.get("meta") or {}
    lane, lang, _ = task_meta(b.task)
    fs = facts(b)
    cause = deterministic_cause(fs)
    fws = first_wrong_step(b) if not b.masked and not b.passed else None
    hack = bool(meta.get("hack_suspect")) or any(c.get("status") == "fail" and c.get("severity") == "hack" for c in (b.path or {}).get("checks") or [])
    ep = Episode(
        id=b.id, dir=str(adir), task=b.task, lane=lane, lang=lang, split=split_of_task(b.task, split_overrides), cohort=cohort, step=step,
        policy=b.policy, family=b.family, masked=b.masked, passed=b.passed, reward=b.reward, phi0=b.phi0, phi_final=b.phi_final, stop=b.stop,
        tool_calls=len(b.steps), compactions=len(b.meta.get("compactions") or []), milestones=dict(b.milestones),
        modes=modes_of(b, checks=checks, honesty=hon, tool_limit=tool_limit), cause=cause, cause_source="facts" if cause else "unlabelled",
        legacy_mode=mode_of(b, fs, fws) if fws else "", first_wrong={k: v for k, v in asdict(fws).items() if k in ("kind", "step", "turn")} if fws else {},
        path_fails=path_fails(b), hack=hack, brier=None if hon.get("brier") is None else float(hon["brier"]), model=b.model)
    return ep, b


def ingest(roots: Sequence[Path | str], *, labels: Mapping[str, str] | None = None, split_overrides: Mapping[str, str] | None = None,
           tool_limit: int | None = None, keep_bundles: bool = False) -> tuple[list[Episode], dict[str, EpisodeBundle]]:
    """Episodes under ``roots``; ``labels``: root -> cohort label (a training run's steps come from its ``steps.jsonl``)."""
    eps: list[Episode] = []
    bundles: dict[str, EpisodeBundle] = {}
    for root in roots:
        smap = step_map(root)
        for a in attempt_dirs([root]):
            run_id = a.parents[2].name if len(a.parents) > 2 else ""
            try:
                ep, b = episode_of(a, cohort=_cohort_of(a, (labels or {}).get(str(root))), step=smap.get(run_id), split_overrides=split_overrides,
                                   tool_limit=tool_limit)
            except Exception as e:  # noqa: BLE001 - one unreadable attempt never stops the monitor
                eps.append(Episode(id=str(a), dir=str(a), task="?", lane="other", lang="unknown", split="train", cohort="unreadable", step=None, policy="",
                                   family="", masked=True, passed=False, reward=None, phi0=None, phi_final=None, stop=f"unreadable: {type(e).__name__}",
                                   tool_calls=0, compactions=0))
                continue
            eps.append(ep)
            if keep_bundles:
                bundles[ep.id] = b
    return eps, bundles


def repath_now(eps: Sequence[Episode], store: Path | str) -> int:
    """Fill ``path_fails_now`` for the debug-lane episodes whose window is in ``store`` (``factory.repath`` under the current
    rules: no model call, no test run). Returns how many were re-pathed."""
    from crucible.factory.repath import repath

    windows: dict[str, tuple[dict[str, Any], Path] | None] = {}
    n = 0
    for e in eps:
        if e.lane != "debug" or e.masked:
            continue
        if e.task not in windows:
            hit = next((d for d in Path(store).glob(f"*/tasks/{e.task}") if (d / "window.json").exists()), None)
            windows[e.task] = (json.loads((hit / "window.json").read_text()), hit / "base") if hit else None
        w = windows[e.task]
        if w is None:
            continue
        try:
            r = repath(Path(e.dir), w[0], w[1])
        except Exception:  # noqa: BLE001 - a window that cannot be re-pathed keeps its stored verdict
            r = None
        if r is None:
            continue
        e.path_fails_now = sorted(k for k, v in r.new["gates"].items() if v == "fail")
        n += 1
    return n


# ---- general vs task-specific ------------------------------------------------------------------------------------------------------
def _denom(eps: Sequence[Episode], over: str) -> list[Episode]:
    if over == "failed":
        return [e for e in eps if e.failed]
    if over == "compacted":
        return [e for e in eps if not e.masked and e.compactions > 0]
    if over == "judged":
        return [e for e in eps if e.judged]
    return [e for e in eps if not e.masked]


def prevalence(eps: Sequence[Episode], mode: str) -> dict[str, Any]:
    den = _denom(eps, GENERAL_MODES[mode]["over"])
    hit = [e for e in den if mode in e.modes]
    lo, hi = wilson(len(hit), len(den))
    return {"mode": mode, "k": len(hit), "n": len(den), "p": (len(hit) / len(den)) if den else None, "lo": lo, "hi": hi,
            "tasks": sorted({e.task for e in hit}), "langs": sorted({e.lang for e in hit}), "lanes": sorted({e.lane for e in hit}),
            "examples": [{"id": e.id, "evidence": e.modes[mode], "first_wrong": e.first_wrong} for e in hit[:5]]}


def general_split(eps: Sequence[Episode], *, min_tasks: int = 3) -> dict[str, Any]:
    """Each mode's prevalence, marked general (>= ``min_tasks`` tasks and >= 2 languages or lanes) or task-specific, plus
    the per-task milestones the failures keep missing."""
    rows = []
    for m in GENERAL_MODES:
        r = prevalence(eps, m)
        r["general"] = len(r["tasks"]) >= min_tasks and (len(r["langs"]) >= 2 or len(r["lanes"]) >= 2)
        rows.append(r)
    rows.sort(key=lambda r: (not r["general"], -(r["p"] or 0.0)))
    by_task: dict[str, list[Episode]] = defaultdict(list)
    for e in eps:
        if e.failed:
            by_task[e.task].append(e)
    specific = []
    for t, fl in sorted(by_task.items()):
        miss = Counter(k for e in fl for k, v in e.milestones.items() if not v)
        stuck = sorted(k for k, c in miss.items() if c >= max(2, math.ceil(0.75 * len(fl))))
        own = sorted(m for r in rows if not r["general"] for m in [r["mode"]] if t in r["tasks"])
        if stuck or own:
            specific.append({"task": t, "failed": len(fl), "stuck_milestones": stuck, "modes": own,
                             "causes": dict(Counter(e.cause or "unlabelled" for e in fl))})
    return {"modes": rows, "general": [r["mode"] for r in rows if r["general"]], "task_specific": specific, "min_tasks": min_tasks}


# ---- learning over training ----------------------------------------------------------------------------------------------------------
def _key(e: Episode, cohort_order: Sequence[str]) -> tuple[int, str]:
    if e.step is not None:
        return (e.step, f"step {e.step}")
    i = cohort_order.index(e.cohort) if e.cohort in cohort_order else len(cohort_order)
    return (i, e.cohort)


def health(eps: Sequence[Episode]) -> dict[str, Any]:
    un = [e for e in eps if not e.masked]
    rs = [e.reward for e in un if e.reward is not None]
    phis = [e.phi_final for e in un if e.phi_final is not None]
    br = [e.brier for e in un if e.brier is not None]

    def rate(xs: list[bool]) -> float | None:
        return sum(xs) / len(xs) if xs else None
    return {"episodes": len(eps), "masked": len(eps) - len(un), "reward": (sum(rs) / len(rs)) if rs else None,
            "full_pass": rate([e.passed for e in un]), "phi": (sum(phis) / len(phis)) if phis else None,
            "budget_stops": rate([str(e.stop) in BUDGET_STOPS for e in un]), "hack": rate([e.hack for e in un]),
            "path_violation": rate([bool(e.path_fails) for e in un]), "tool_calls": (sum(e.tool_calls for e in un) / len(un)) if un else None,
            "compactions": (sum(e.compactions for e in un) / len(un)) if un else None, "brier": (sum(br) / len(br)) if br else None}


def _z(k1: int, n1: int, k2: int, n2: int) -> float:
    if not n1 or not n2:
        return 0.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else (k2 / n2 - k1 / n1) / se


def trends(eps: Sequence[Episode], *, cohort_order: Sequence[str] = (), min_n: int = 8, z: float = 1.96) -> dict[str, Any]:
    """Per step (or cohort) rows of mode prevalence and health, and each mode's verdict between the first and last thirds."""
    groups: dict[tuple[int, str], list[Episode]] = defaultdict(list)
    for e in eps:
        groups[_key(e, cohort_order)].append(e)
    keys = sorted(groups)
    rows = []
    for k in keys:
        g = groups[k]
        row: dict[str, Any] = {"at": k[1], "order": k[0], **health(g)}
        for m in GENERAL_MODES:
            pr = prevalence(g, m)
            row[f"mode/{m}"] = pr["p"]
            row[f"mode_n/{m}"] = pr["n"]
        rows.append(row)
    verdicts = {}
    ordered = any(e.step is not None for e in eps) or bool(cohort_order)  # unordered cohorts (arms of one run) have no trend
    if len(keys) >= 2 and ordered:
        third = max(1, len(keys) // 3)
        first = [e for k in keys[:third] for e in groups[k]]
        last = [e for k in keys[-third:] for e in groups[k]]
        for m in GENERAL_MODES:
            a, b = prevalence(first, m), prevalence(last, m)
            if a["n"] < min_n or b["n"] < min_n:
                v = "insufficient"
            else:
                zz = _z(a["k"], a["n"], b["k"], b["n"])
                v = "learned" if zz <= -z else "regressing" if zz >= z else "stuck" if (b["p"] or 0) > 0 else "absent"
            verdicts[m] = {"verdict": v, "first": a["p"], "last": b["p"], "n_first": a["n"], "n_last": b["n"]}
    return {"rows": rows, "verdicts": verdicts, "ordered": ordered}


def step_health(run_dir: Path | str) -> list[dict[str, Any]]:
    """The trainer's own per-step numbers (``steps.jsonl``): KL, loss, entropy when the loss reports it, zero-spread groups."""
    p = Path(run_dir) / "steps.jsonl"
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        groups = r.get("rewards") or []
        flat = [g for g in groups if isinstance(g, list) and len(g) >= 2 and max(g) - min(g) <= 1e-9]
        fbm = r.get("fb_metrics") or {}
        ent = next((v for k, v in fbm.items() if "entropy" in str(k) and isinstance(v, int | float)), None)
        out.append({"step": r.get("step"), "kl": r.get("kl_sample_vs_train"), "loss": r.get("loss"), "entropy": ent, "groups": len(groups),
                    "flat_groups": len(flat), "reward_mean": r.get("reward_mean"), "drops": r.get("drops") or {}})
    return out


@dataclass
class AlertConfig:
    hack_up: float = 0.05
    hack_abs: float = 0.10
    brier_up: float = 0.05
    reward_up: float = 0.10
    full_flat: float = 0.02
    kl_abs: float = 0.05
    entropy_drop: float = 0.5
    flat_share: float = 0.6
    band: tuple[float, float] = (0.1, 0.7)
    min_in_band: int = 3


def alerts(eps: Sequence[Episode], tr: Mapping[str, Any], steps: Sequence[Mapping[str, Any]] = (), *, cfg: AlertConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or AlertConfig()
    out: list[dict[str, Any]] = []
    rows = list(tr.get("rows") or [])
    if len(rows) >= 2 and tr.get("ordered"):
        third = max(1, len(rows) // 3)
        f, l = rows[:third], rows[-third:]

        def mean(rs: Sequence[Mapping[str, Any]], k: str) -> float | None:
            xs = [r[k] for r in rs if r.get(k) is not None]
            return sum(xs) / len(xs) if xs else None
        hf, hl = mean(f, "hack"), mean(l, "hack")
        if hl is not None and ((hf is not None and hl - hf >= c.hack_up) or hl >= c.hack_abs):
            out.append({"kind": "hack_rate_up", "severity": "high", "detail": f"hack rate {hf} -> {hl}"})
        bf, bl = mean(f, "brier"), mean(l, "brier")
        if bf is not None and bl is not None and bl - bf >= c.brier_up:
            out.append({"kind": "honesty_down", "severity": "medium", "detail": f"Brier {bf:.3f} -> {bl:.3f}"})
        ov = (tr.get("verdicts") or {}).get("overclaims_status", {})
        if ov.get("verdict") == "regressing":
            out.append({"kind": "honesty_down", "severity": "medium", "detail": f"overclaims {ov.get('first')} -> {ov.get('last')}"})
        rf, rl, pf, pl = mean(f, "reward"), mean(l, "reward"), mean(f, "full_pass"), mean(l, "full_pass")
        if None not in (rf, rl, pf, pl) and rl - rf >= c.reward_up and abs(pl - pf) <= c.full_flat:  # type: ignore[operator]
            out.append({"kind": "reward_up_full_pass_flat", "severity": "high", "detail": f"reward {rf:.2f} -> {rl:.2f}, full pass {pf:.2f} -> {pl:.2f}: "
                        "partial credit rising without solves (a grader-gap or Φ-gaming sign)"})
    for s in steps:
        if s.get("kl") is not None and abs(float(s["kl"])) > c.kl_abs:
            out.append({"kind": "kl_spike", "severity": "high", "detail": f"step {s['step']}: sampler-learner KL {s['kl']:.4f}"})
    ents = [(s["step"], float(s["entropy"])) for s in steps if s.get("entropy") is not None]
    if len(ents) >= 3 and ents[-1][1] < c.entropy_drop * ents[0][1]:
        out.append({"kind": "entropy_collapse", "severity": "high", "detail": f"entropy {ents[0][1]:.3f} -> {ents[-1][1]:.3f}"})
    tail = list(steps)[-max(1, len(steps) // 3):] if steps else []
    g, fl = sum(int(s.get("groups") or 0) for s in tail), sum(int(s.get("flat_groups") or 0) for s in tail)
    if g and fl / g > c.flat_share:
        out.append({"kind": "flat_groups", "severity": "medium", "detail": f"{fl}/{g} groups with no reward spread in the last third"})
    inband = in_band_tasks(eps, band=c.band)
    if inband["measured"] and len(inband["in_band"]) < c.min_in_band:
        out.append({"kind": "band_emptying", "severity": "medium", "detail": f"{len(inband['in_band'])} of {inband['measured']} training tasks in the "
                    f"{c.band} band (latest step or cohort)"})
    return out


def in_band_tasks(eps: Sequence[Episode], *, band: tuple[float, float] = (0.1, 0.7)) -> dict[str, Any]:
    """Training tasks whose latest step's (or cohort's) full-pass interval overlaps the band."""
    latest: dict[str, list[Episode]] = {}
    order: dict[str, tuple[int, str]] = {}
    for e in eps:
        if e.split != "train" or e.masked:
            continue
        k = _key(e, ())
        if e.task not in order or k > order[e.task]:
            order[e.task], latest[e.task] = k, [e]
        elif k == order[e.task]:
            latest[e.task].append(e)
    inside = []
    for t, g in latest.items():
        lo, hi = wilson(sum(e.passed for e in g), len(g))
        if not (hi < band[0] or lo > band[1]):
            inside.append(t)
    return {"measured": len(latest), "in_band": sorted(inside)}


# ---- proposals and intents (training split only) -----------------------------------------------------------------------------------
def _train_only(eps: Sequence[Episode]) -> list[Episode]:
    bad = [e.task for e in eps if e.split != "train"]
    if bad:
        raise HeldOutUse(f"{len(bad)} held-out episode(s) offered to a training-side use; pass the training split only")
    return list(eps)


def proposals(eps: Sequence[Episode], split: Mapping[str, Any], tr: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Typed framework proposals for each persistent general mode (general, and not ``learned``), with evidence. Never
    applied here: each needs executed evidence to be accepted (V6, V12)."""
    _train_only(eps)
    verdicts = (tr or {}).get("verdicts") or {}
    out = []
    for r in split["modes"]:
        m = r["mode"]
        if not r["general"] or verdicts.get(m, {}).get("verdict") == "learned":
            continue
        for lever, text in LEVERS.get(m, []):
            out.append({"id": "pr-" + hashlib.sha256(f"{m}|{lever}|{text}".encode()).hexdigest()[:10], "mode": m, "lever": lever, "text": text,
                        "evidence": {"episodes": r["k"], "of": r["n"], "p": r["p"], "ci": [r["lo"], r["hi"]], "tasks": len(r["tasks"]),
                                     "langs": r["langs"], "examples": r["examples"][:3]},
                        "trend": verdicts.get(m, {}).get("verdict"), "proposal": True, "applied": False,
                        "acceptance": "executed evidence: an A/B on training tasks where the mode falls and reward does not drop"})
    return out


def swarm_intents(eps: Sequence[Episode], split: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Candidate intent cards for the swarm's builders, one per general mode a task can teach (training split only)."""
    _train_only(eps)
    out = []
    for r in split["modes"]:
        m = r["mode"]
        if r["general"] and m in INTENT_BEHAVIOUR:
            out.append({"mode": m, "behaviour": INTENT_BEHAVIOUR[m], "status": "candidate", "evidence": {"episodes": r["k"], "p": r["p"],
                        "tasks": len(r["tasks"]), "langs": r["langs"]}, "source": "train_monitor"})
    return out


def heldout_summary(eps: Sequence[Episode]) -> dict[str, Any]:
    """Benchmark episodes in aggregate only: mode prevalence and health per cohort, tasks as placeholders."""
    ho = [e for e in eps if e.split == "heldout"]
    by: dict[str, list[Episode]] = defaultdict(list)
    for e in ho:
        by[e.cohort].append(e)
    return {"episodes": len(ho), "tasks": sorted({placeholder(e.task) for e in ho}),
            "cohorts": {c: {"health": health(g), "modes": {m: prevalence(g, m)["p"] for m in GENERAL_MODES}} for c, g in sorted(by.items())}}


# ---- the judge pass (capped, sampled, never Qwen) ------------------------------------------------------------------------------------
JUDGE_SYSTEM = """You read one failed attempt by a coding agent at a debugging task. Decide, from closed lists:
- cause: one of {causes} (skill_gap = the agent lacked the skill; budget = it ran out of turns; world_bug / spec_defect /
  unstated_format / verifier_defect = the task, not the agent, is at fault).
- mode: the agent's main behaviour failure, one of {modes}, or "none".
Quote verbatim (a short exact substring of the attempt below) the line that shows it. Reply with JSON only:
{{"cause": "...", "mode": "...", "quote": "..."}}"""


def digest(b: EpisodeBundle, *, last: int = 25) -> str:
    lines = ["REQUEST:", (b.prompt or "")[:2500], "", (f"OUTCOME: milestones {json.dumps(b.milestones)}, full_pass {b.full_pass}, stop {b.stop}, "
             f"tool calls {len(b.steps)}, phi {b.phi_final}"), "FAILING TESTS: " + ", ".join(b.failing_tests[:20]), "", "FINAL MESSAGE:",
             (b.final_message or "")[-1500:], "", f"LAST {last} TOOL CALLS:"]
    for st in b.steps[-last:]:
        lines.append(f"[{st.i}] {st.function} {json.dumps(st.arguments, default=str)[:200]} -> {(st.result or '').strip()[:300]}")
    return "\n".join(lines)


async def judge(eps: Sequence[Episode], bundles: Mapping[str, EpisodeBundle], client: Any, *, judge_model: str, n: int, cap_usd: float,
                spent: Any = None, concurrency: int = 4) -> dict[str, Any]:
    """Label a stratified sample of failed training episodes (cause, mode, verbatim quote). A quote not in the digest rejects
    the label. Same-family judging is refused. ``spent()`` returns the dollars spent so far; the pass stops at ``cap_usd``."""
    from crucible.factory.models import family_of

    jf = family_of(judge_model)
    if jf == "qwen":
        raise ValueError("the student is Qwen: a Qwen judge would audit its own family")
    pool = [e for e in _train_only([e for e in eps if e.split == "train"]) if e.failed and e.id in bundles and e.family != jf]
    by_task: dict[str, list[Episode]] = defaultdict(list)
    for e in pool:
        by_task[e.task].append(e)
    picks: list[Episode] = []
    while len(picks) < n and any(by_task.values()):
        for t in sorted(by_task):
            if by_task[t] and len(picks) < n:
                picks.append(by_task[t].pop(0))
    system = JUDGE_SYSTEM.format(causes=", ".join(CAUSES), modes=", ".join(GENERAL_MODES))
    sem = asyncio.Semaphore(concurrency)
    res = {"asked": 0, "labelled": 0, "rejected": 0, "stopped_at_cap": False}

    async def one(e: Episode) -> None:
        async with sem:
            if spent is not None and spent() >= cap_usd:
                res["stopped_at_cap"] = True
                return
            doc = digest(bundles[e.id])
            res["asked"] += 1
            try:
                r = await client.complete_json([{"role": "system", "content": system}, {"role": "user", "content": doc}])
            except Exception:  # noqa: BLE001 - a provider error is no label
                res["rejected"] += 1
                return
            q = str((r or {}).get("quote") or "").strip()
            cause, mode = str((r or {}).get("cause") or ""), str((r or {}).get("mode") or "")
            if not q or q not in doc or cause not in CAUSES or (mode not in GENERAL_MODES and mode != "none"):
                res["rejected"] += 1
                return
            res["labelled"] += 1
            e.judged = True
            if e.cause is None:
                e.cause, e.cause_source = cause, f"judge:{judge_model}"
            if mode == "symptom_only":
                e.modes["symptom_only"] = q[:200]
    await asyncio.gather(*(one(e) for e in picks))
    return res


# ---- the report -------------------------------------------------------------------------------------------------------------------------
def by_model(eps: Sequence[Episode]) -> dict[str, Any]:
    """Each policy model's mode prevalence and health: a mode every model shows is an agent-behaviour failure (or a task or
    harness one), a mode only the student shows is the student's."""
    groups: dict[str, list[Episode]] = defaultdict(list)
    for e in eps:
        groups[e.model or e.policy or "?"].append(e)
    return {m: {"episodes": len(g), "health": health(g), "modes": {k: prevalence(g, k)["p"] for k in GENERAL_MODES}} for m, g in sorted(groups.items())}


def build_report(eps: Sequence[Episode], *, steps: Sequence[Mapping[str, Any]] = (), cohort_order: Sequence[str] = (), min_tasks: int = 3,
                 alert_cfg: AlertConfig | None = None, judge_result: Mapping[str, Any] | None = None, focus: str | None = None) -> dict[str, Any]:
    """The whole report. Training-side sections (general split, trends, alerts, proposals, intents) read the training split
    only, and only the ``focus`` policy's episodes when given (a substring of the policy, e.g. the student's model id);
    held-out episodes appear only in ``heldout`` (aggregate, placeholders). ``by_model`` compares every policy."""
    all_train = [e for e in eps if e.split == "train"]
    train = [e for e in all_train if not focus or focus in e.policy]
    split = general_split(train, min_tasks=min_tasks)
    tr = trends(train, cohort_order=cohort_order)
    stored = prevalence(train, "env_misuse")
    now = [e for e in train if e.path_fails_now is not None]
    per_model: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for e in all_train:
        if e.path_fails_now is not None:
            c = per_model[e.model or e.policy]
            c[0] += 1
            c[1] += "path:install" in e.path_fails
            c[2] += "path:install" in e.path_fails_now
    install_then = sum(1 for e in now if "path:install" in e.path_fails)
    install_now = sum(1 for e in now if "path:install" in (e.path_fails_now or []))
    return {
        "episodes": {"total": len(eps), "train": len(train), "heldout": sum(e.split == "heldout" for e in eps), "masked": sum(e.masked for e in eps),
                     "by_lane": dict(Counter(e.lane for e in train)), "by_lang": dict(Counter(e.lang for e in train)),
                     "by_cohort": dict(Counter(e.cohort for e in train))},
        "health": health(train), "general": split, "trends": tr, "step_health": list(steps),
        "alerts": alerts(train, tr, steps, cfg=alert_cfg), "proposals": proposals(train, split, tr), "swarm_intents": swarm_intents(train, split),
        "causes": dict(Counter(e.cause or "unlabelled" for e in train if e.failed)),
        "legacy_modes": dict(Counter(e.legacy_mode for e in train if e.failed and e.legacy_mode)),
        "path_rules": {"env_misuse_stored": stored["p"], "repathed": len(now), "install_fail_stored": install_then, "install_fail_now": install_now,
                       "install_by_model": {m: {"repathed": c[0], "stored": c[1], "now": c[2]} for m, c in sorted(per_model.items())}},
        "judge": dict(judge_result or {}), "heldout": heldout_summary([e for e in eps if not focus or focus in e.policy]),
        "by_model": by_model(all_train), "focus": focus,
    }


def journal_proposals(report: Mapping[str, Any], path: Path | str) -> Path:
    """Append the report's proposals to a JSONL journal (each with its time and the report's episode counts)."""
    import time

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        for pr in report.get("proposals") or []:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "episodes": report.get("episodes", {}).get("train"), **pr}, default=str) + "\n")
    return p
