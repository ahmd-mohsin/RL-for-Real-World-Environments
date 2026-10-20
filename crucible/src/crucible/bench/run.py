"""Run a benchmark through the M10 runner and report it (DEPEND: the runner's journal is the result store).

``run_bench`` runs tasks × policies × harnesses with k samples (masked attempts retried by the runner), loads the
episodes, computes the summary, optionally fits IRT (respondent = policy@harness, item = task), and writes
``summary.json`` / ``episodes.json`` / ``report.html`` (plus Harbor- and NeMo-Gym-shaped exports) next to the run.

What the M13 audit (``docs/audit/2026-10-13``) changed here:
- the runner's ``RunSummary`` is kept: ``summary["run"]`` carries the planned count, owed samples and the stop reason,
  so a stopped run is never presented as the whole benchmark (H1 V5, campaign B6); ``report_run`` on an existing
  directory reads the same from ``run.json`` and the journal;
- ``ks`` defaults to every power of two up to the run's k, and k (H1 V16 / G1 D6, B9);
- fault-start tags come from the tasks (``task.tags``) and C3 branch counts from ``branches=`` or the run's
  ``branches.json``, so recovery rate and the first irrecoverable step are computed (H1 V11);
- ``summary["config"]``: policies, harnesses with their step caps, k, ks, threshold, seeds, sampling and the judge
  (G2-18), plus the stage, horizon and tag maps, so ``report_saved`` re-derives the same digest from the run alone
  (campaign B10: ``bench report`` dropped stages, extras and title and overwrote the original);
- **validity** (campaign B4/B5): ``controls=True`` runs an oracle and a no-op control (``control:oracle``,
  ``control:noop``, 3 samples) on every task first, and a task whose oracle fails or is flaky, or whose no-op passes, is
  excluded from every model's metrics, from IRT and from compare (``invalid_tasks``, with reasons). A manifest can also
  come from M9 acceptance (``studio.build``) or an import report (``validity=``); it is saved as ``validity.json`` in
  the run dir. Without one the summary says ``validity: unknown`` and warns;
- control policies never enter IRT as respondents (B16).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.bench import irt, report
from crucible.bench.metrics import Episode, default_ks, load_run, run_info, run_spec, summarize, validity_from_controls

ORACLE, NOOP = "control:oracle", "control:noop"
CONTROL_PREFIX = "control:"


def irt_counts(eps: Sequence[Episode], *, threshold: float = 1.0, exclude_tasks: Iterable[str] = (),
               exclude_policies: Iterable[str] = ()) -> dict[tuple[str, str], tuple[float, int]]:
    """Successes and trials per (respondent, item). Invalid tasks and control policies are not data (campaign B4, B16)."""
    xt, xp = set(exclude_tasks), set(exclude_policies)
    out: dict[tuple[str, str], list[int]] = {}
    for e in eps:
        if e.masked or e.task in xt or e.policy in xp or e.policy.startswith(CONTROL_PREFIX):
            continue  # an exposed episode is measured and failed (G2-1)
        c = out.setdefault((f"{e.policy}@{e.harness}", e.task), [0, 0])
        c[0] += int(e.passed(threshold))
        c[1] += 1
    return {k: (float(v[0]), v[1]) for k, v in out.items()}


def _harness_doc(h: Any) -> dict[str, Any]:
    lim = getattr(h, "limits", None)
    return {"id": getattr(h, "id", str(h)), **({"max_steps": getattr(lim, "max_steps", None)} if lim is not None else {})}


def control_policies() -> list[Any]:
    """The oracle (the task's own oracle actions replayed) and the no-op (an empty episode) as ``PolicyRef``s."""
    from crucible.actors.policy import ScriptedPolicy
    from crucible.runtime.harness.native import PolicyRef

    # M10 lab audit R-4: a policy's configuration is its run identity (``PolicyRef.config``); these two are scripts whose content is
    # the task's own oracle actions (the task's identity covers them) or nothing
    return [PolicyRef(id=ORACLE, make=lambda att: {"agent": ScriptedPolicy("agent", _oracle_actions(att.task) or [])}, config={"script": "task oracle"}),
            PolicyRef(id=NOOP, make=lambda att: {"agent": ScriptedPolicy("agent", [])}, config={"script": "noop"})]


async def run_controls(root: Path, run_id: str, tasks: Sequence[Any], harnesses: Sequence[Any], *, k: int = 3, threshold: float = 1.0,
                       spec: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Oracle and no-op controls on every task (its own run, ``<run_id>-controls``) → a validity manifest. A task with no
    oracle actions is ``valid: None`` (unknown: its oracle cannot be checked), never valid by default."""
    from crucible.runtime.run import Runner, RunSpec

    # M14 campaign B14: a step limit below the oracle's length cut the oracle off, and the task was dropped silently as
    # broken_oracle. The controls run with a step limit sized to the longest oracle, so they test the task, and a task
    # whose oracle is longer than the run's own limit is reported as oracle_truncated (no policy can finish it under it)
    lengths = {t.id: len(a) for t in tasks if (a := _oracle_actions(t)) is not None}
    limit = _step_limit(harnesses)
    sized = [_sized(h, max(lengths.values(), default=0) + 1) for h in harnesses]
    await Runner(root).run(RunSpec(run_id=f"{run_id}-controls", tasks=list(tasks), policies=control_policies(), harnesses=sized, k=k,
                                   **dict(spec or {})))
    eps = load_run(Path(root) / f"{run_id}-controls")
    man = validity_from_controls(eps, oracle=ORACLE, noop=NOOP, threshold=threshold)
    for t in tasks:
        e = man.get(t.id)
        n = lengths.get(t.id)
        if e is None or n is None:
            continue
        cut = [x for x in eps if x.task == t.id and x.policy == ORACLE and x.stop == "max_steps" and not x.passed(threshold)]
        if (limit is not None and n > limit) or (cut and "broken_oracle" in e["reasons"]):
            fits = not any(r in e["reasons"] for r in ("broken_oracle", "flaky_oracle")) or bool(cut)
            reasons = [r for r in e["reasons"] if not (fits and r in ("broken_oracle", "flaky_oracle"))]
            e |= {"valid": False, "reasons": ["oracle_truncated", *reasons], "oracle_steps": n, "max_steps": limit,
                  "note": f"the oracle takes {n} steps and the run's step limit is {limit}: no policy can finish this task under it; "
                          "raise max_steps to measure it"}
    for t in tasks:
        if _oracle_actions(t) is None and t.id in man:
            man[t.id] |= {"valid": False if "noop_passes" in man[t.id]["reasons"] else None, "reasons": [r for r in man[t.id]["reasons"] if r == "noop_passes"],
                          "note": "no oracle actions: the oracle could not be checked"}
    return man


def _step_limit(harnesses: Sequence[Any]) -> int | None:
    """The smallest step limit among the run's harnesses (None when none declares one)."""
    caps = [m for h in harnesses if isinstance(m := getattr(getattr(h, "limits", None), "max_steps", None), int)]
    return min(caps) if caps else None


def _sized(h: Any, need: int) -> Any:
    """``h`` with a step limit of at least ``need`` (a copy; the run's own harness is untouched)."""
    import copy
    import dataclasses

    lim = getattr(h, "limits", None)
    if not dataclasses.is_dataclass(lim) or not isinstance(getattr(lim, "max_steps", None), int) or lim.max_steps >= need:
        return h
    c = copy.copy(h)
    c.limits = dataclasses.replace(lim, max_steps=need)
    return c


async def run_bench(name: str, *, root: Path, tasks: Sequence[Any], policies: Sequence[Any], harnesses: Sequence[Any], k: int = 1,
                    ks: Sequence[int] | None = None, threshold: float = 1.0, stages_of: Callable[[str], Any] | None = None,
                    horizon_of: Callable[[str], str] | None = None, anchors: Mapping[str, irt.Item] | None = None,
                    extras: dict[str, Any] | None = None, run_id: str | None = None, tags_of: Callable[[str], Iterable[str]] | None = None,
                    branches: Mapping[str, Sequence[tuple[int, int, int]]] | None = None, subsets_of: Callable[[str], Iterable[str]] | None = None,
                    config: Mapping[str, Any] | None = None, spec: Mapping[str, Any] | None = None, controls: bool = False,
                    validity: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """``spec``: further ``RunSpec`` fields (run_seed, budget_usd, attempt_timeout_s, sampling, concurrency, ...).
    ``config``: anything else the report must disclose (the judge model and version, token caps). ``controls``: run the
    oracle and no-op controls first and exclude invalid tasks; ``validity``: a manifest from elsewhere (M9, import)."""
    from crucible.runtime.journal import assert_free
    from crucible.runtime.run import Runner, RunSpec

    run_id = run_id or f"bench-{name}"
    # M10 lab audit R-4: a second bench of the same name is refused before it writes anything beside the first (the controls
    # run, validity.json); Runner.run's lease stays the authority
    for rid in (run_id, f"{run_id}-controls") if controls else (run_id,):
        assert_free(Path(root) / rid)
    if controls:
        man = await run_controls(root, run_id, tasks, harnesses, threshold=threshold, spec=spec)
        validity = {**man, **dict(validity or {})}
    rs = RunSpec(run_id=run_id, tasks=list(tasks), policies=list(policies), harnesses=list(harnesses), k=k, **dict(spec or {}))
    summ = await Runner(root).run(rs)
    if validity is not None:
        (Path(root) / run_id / "validity.json").write_text(json.dumps(validity, indent=1, sort_keys=True, default=str) + "\n")
    if tags_of is None:  # fault_start and other tags travel on the task objects (H1 V11)
        tag_map = {t.id: [str(x) for x in (getattr(t, "tags", None) or ())] for t in tasks}
        tags_of = lambda t: tag_map.get(t, [])
    cfg = {"harnesses": [_harness_doc(h) for h in harnesses], "sampling": dict(rs.sampling), "run_seed": rs.run_seed,
           "attempt_timeout_s": rs.attempt_timeout_s, "budget_usd": rs.budget_usd, **({"controls": f"{run_id}-controls"} if controls else {}),
           **dict(config or {})}
    return report_run(name, Path(root) / run_id, ks=ks if ks is not None else default_ks(k), threshold=threshold, stages_of=stages_of,
                      horizon_of=horizon_of, anchors=anchors, extras=extras, tags_of=tags_of, branches=branches, subsets_of=subsets_of,
                      config=cfg, runner={"planned": summ.planned, "stopped": summ.stopped, "attempts": summ.attempts, "spend": summ.spend},
                      validity=validity)


def check_key_file(f: Path) -> Path:
    """A key file must be a regular file only its owner can read (M15 audit H2 V15, as ssh treats private keys): a symlink
    or a group- / world-accessible file is refused, never used silently."""
    import os
    import stat

    f = Path(f)
    st = os.lstat(f)
    if stat.S_ISLNK(st.st_mode):
        raise PermissionError(f"bench key file {f} is a symlink: give the key file itself")
    if not stat.S_ISREG(st.st_mode):
        raise PermissionError(f"bench key file {f} is not a regular file")
    if st.st_mode & 0o077:
        raise PermissionError(f"bench key file {f} has mode {oct(st.st_mode & 0o777)}: other users can read it; run chmod 600 {f}")
    return f


def bench_key(out: Path, *, key_file: Path | None = None, notes: list[str] | None = None) -> bytes:
    """The HMAC item key for a bench or build under ``out`` (G2-4 with the dynamic batch: there is no public default).
    ``key_file`` or ``$CRUCIBLE_BENCH_KEY`` / ``$CRUCIBLE_BENCH_KEY_FILE`` first; otherwise ``out/bench.key`` (0600),
    created with a fresh random key on first use, so the key holder can regenerate the items and nobody else can.

    M15 audit: every key file must be 0600 and not a symlink (H2 V15); a key set in the environment that is invalid is an
    error, never replaced by a random one (G2 D9); a new key inside ``out`` is announced in ``notes`` (the CLI prints it),
    since ``out`` is what gets shared (keep ``bench.key`` out of it, or set ``$CRUCIBLE_BENCH_KEY_FILE`` to a key kept
    elsewhere)."""
    import os

    from crucible.bench import dynamic

    if key_file is not None:
        return dynamic.load_key(file=check_key_file(Path(key_file)))
    from crucible.core import env as knobs

    if (key_file_env := knobs.get(dynamic.KEY_FILE_ENV)) is not None:
        return dynamic.load_key(file=check_key_file(key_file_env))
    if knobs.get(dynamic.KEY_ENV):
        return dynamic.load_key()  # an invalid key raises (G2 D9)
    f = Path(out) / "bench.key"
    if f.exists() or f.is_symlink():
        return dynamic.load_key(file=check_key_file(f))
    f.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(dynamic.new_key().hex())
    if notes is not None:
        notes.append(f"created the bench key {f} (mode 0600); it is a secret: keep it out of what you share, or set "
                     f"${dynamic.KEY_FILE_ENV} to a key kept elsewhere")
    return dynamic.load_key(file=f)


def _oracle_actions(task: Any) -> list[Any] | None:
    """The task's gold actions: ``bench_gold`` (a pack's ``hooks.tasks`` pairs), ``oracle_actions()`` or ``oracle``."""
    from crucible.core.schema.actions import Message, ToolCall

    raw = getattr(task, "bench_gold", None) or (task.oracle_actions() if callable(getattr(task, "oracle_actions", None)) else getattr(task, "oracle", None))
    if not raw:
        return None
    if not all(isinstance(a, dict) for a in raw):  # already actions (a pack's hooks.tasks gold)
        return list(raw)
    return [ToolCall(tool_call_id=a.get("tool_call_id") or f"o{i}", function_name=a["function_name"], arguments=a.get("arguments") or {})
            if a.get("kind", "tool_call") == "tool_call" else Message(to=a["to"], text=a["text"]) for i, a in enumerate(raw)]


def gold_and_noop_check(work: Path, *, max_steps: int = 1000, best: float = 1.0, threshold: float | None = None) -> tuple[str, Callable[[Any], Any]]:
    """The validation floor every fresh item passes before it counts (R:21/58/72: gold scores the top of the pack's reward
    range, a no-op does not count as solved): the task's own oracle replayed, then an empty episode. A task without an
    oracle is rejected (fail closed). ``best`` / ``threshold`` are the pack's ``reward.range`` top and ``label.threshold``
    (M14 audit V12: a pack on [0, 10] was validated against 1.0); a non-finite reward never passes."""
    import math

    thr = best if threshold is None else threshold

    async def check(task: Any) -> tuple[bool, str]:
        from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode

        acts = _oracle_actions(task)
        if acts is None:
            return False, "no oracle to validate against"
        rewards = []
        for name, script in (("gold", acts), ("noop", [])):
            d = Path(work) / "validate" / f"{str(getattr(task, 'id', 'task')).replace('/', '_').replace('@', '_')}-{name}"
            w = await task.open(d)
            try:
                await run_episode(w, {"agent": ScriptedPolicy("agent", script)}, limits=EpisodeLimits(max_steps=max_steps), reset=False)
                rewards.append((await task.grade(w, d / "grade")).reward)
            finally:
                await w.close()
        fin = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        ok = fin(rewards[0]) and abs(rewards[0] - best) <= 1e-9 and (rewards[1] is None or (fin(rewards[1]) and rewards[1] < thr))
        return ok, "pass" if ok else f"gold {rewards[0]} / noop {rewards[1]}"

    return ("gold_and_noop", check)


def _branches(run_dir: Path) -> dict[str, list[tuple[int, int, int]]] | None:
    """C3 branch counts saved next to the run: ``branches.json`` = {attempt id: [[step, successes, n], ...]}."""
    f = Path(run_dir) / "branches.json"
    if not f.exists():
        return None
    return {a: [(int(s), int(c), int(n)) for s, c, n in rows] for a, rows in json.loads(f.read_text()).items()}


def load_validity(run_dir: Path) -> dict[str, dict[str, Any]] | None:
    f = Path(run_dir) / "validity.json"
    return json.loads(f.read_text()) if f.exists() else None


def summarize_run(run_dir: Path, *, ks: Sequence[int] | None = None, threshold: float = 1.0, stages_of: Callable[[str], Any] | None = None,
                  horizon_of: Callable[[str], str] | None = None, tags_of: Callable[[str], Iterable[str]] | None = None,
                  branches: Mapping[str, Sequence[tuple[int, int, int]]] | None = None, subsets_of: Callable[[str], Iterable[str]] | None = None,
                  config: Mapping[str, Any] | None = None, runner: Mapping[str, Any] | None = None,
                  validity: Mapping[str, Mapping[str, Any]] | None = None, controls: Iterable[str] = ()) -> tuple[dict[str, Any], list[Episode]]:
    """The full summary of a run directory without writing anything: metrics, ``run`` (plan, owed, stop reason),
    ``config`` and validity. Every reader (``report_run``, the CLI, the studio explorer) goes through here. Raises on a
    directory that holds no run (campaign B14: an empty report used to be written)."""
    run_dir = Path(run_dir)
    if not (run_dir / "run.json").exists() and not (run_dir / "results.jsonl").exists():
        raise ValueError(f"{run_dir} holds no run (no run.json or results.jsonl)")
    eps = load_run(run_dir, tags_of=tags_of)
    spec = run_spec(run_dir)
    k = int(spec.get("k") or (max((e.sample_idx for e in eps), default=0) + 1))
    ks = list(ks) if ks is not None else list(default_ks(k))
    validity = validity if validity is not None else load_validity(run_dir)
    ctl = sorted(set(controls) | {e.policy for e in eps if e.policy.startswith(CONTROL_PREFIX)})
    summary = summarize(eps, ks=ks, threshold=threshold, stages_of=stages_of, horizon_of=horizon_of,
                        branches=branches if branches is not None else _branches(run_dir), subsets_of=subsets_of, validity=validity, controls=ctl)
    info = run_info(run_dir, eps)
    if runner:
        info["stopped"] = runner.get("stopped") or info["stopped"]
        info["complete"] = info["owed"] == 0 and info["stopped"] is None
        if runner.get("spend") is not None:
            info["spend"] = runner["spend"]  # the ledger's view; excluded from the digest by path
    summary["run"] = info
    tasks = sorted({e.task for e in eps})
    summary["config"] = {"policies": list(spec.get("policies") or sorted({e.policy for e in eps})),
                         "harnesses": list(spec.get("harnesses") or sorted({e.harness for e in eps})), "tasks": len(tasks),
                         "k": k, "ks": ks, "threshold": threshold, "run_seed": spec.get("run_seed"), "tier": spec.get("tier"),
                         "judge": None, "controls": ctl or None,
                         # the maps a re-report needs (B10): data, not the pack's callables
                         "stages": {t: [[n, list(x)] for n, x in st] for t in tasks if stages_of and (st := stages_of(t))} or None,
                         "horizons": {t: horizon_of(t) for t in tasks} if horizon_of else None,
                         "tags": {t: list(tg) for t in tasks if tags_of and (tg := list(tags_of(t)))} or None,
                         **dict(config or {})}
    summary["warnings"] = report.warnings(summary)
    return summary, eps


def report_run(name: str, run_dir: Path, *, ks: Sequence[int] | None = None, threshold: float = 1.0, stages_of: Callable[[str], Any] | None = None,
               horizon_of: Callable[[str], str] | None = None, anchors: Mapping[str, irt.Item] | None = None, extras: dict[str, Any] | None = None,
               out_dir: Path | None = None, tags_of: Callable[[str], Iterable[str]] | None = None,
               branches: Mapping[str, Sequence[tuple[int, int, int]]] | None = None, subsets_of: Callable[[str], Iterable[str]] | None = None,
               config: Mapping[str, Any] | None = None, runner: Mapping[str, Any] | None = None,
               validity: Mapping[str, Mapping[str, Any]] | None = None, controls: Iterable[str] = ()) -> dict[str, Any]:
    summary, eps = summarize_run(run_dir, ks=ks, threshold=threshold, stages_of=stages_of, horizon_of=horizon_of, tags_of=tags_of,
                                 branches=branches, subsets_of=subsets_of, config=config, runner=runner, validity=validity, controls=controls)
    counts = irt_counts(eps, threshold=threshold, exclude_tasks=summary.get("invalid_tasks") or {}, exclude_policies=summary["config"].get("controls") or ())
    ex = dict(extras or {})
    f = None
    if counts and len({r for r, _ in counts}) >= 2:
        f = irt.fit(counts, anchors=anchors)
        ex["irt"] = f.to_json() | {"item_qa_flagged": irt.item_qa(f), "respondents": sorted({r for r, _ in counts}),
                                   "excluded": {"control_policies": summary["config"].get("controls") or [], "invalid_tasks": sorted(summary.get("invalid_tasks") or {})}}
        if anchors:
            ex["irt"]["anchor_drift"] = irt.drift(anchors, counts)
        if ex["irt"].get("converged") is False:
            summary["warnings"].append(f"IRT did not converge ({ex['irt'].get('iterations')} iterations, gradient norm {ex['irt'].get('grad_norm')}): abilities are unreliable")
    paths = report.write(out_dir or (Path(run_dir) / "bench"), name, summary, eps, extras=ex)
    return {"summary": summary, "episodes": eps, "extras": ex, "paths": paths, "irt_fit": f}


def report_saved(run_dir: Path, *, out_dir: Path | None = None, ks: Sequence[int] | None = None, threshold: float | None = None,
                 force: bool = False) -> dict[str, Any]:
    """Re-report a run from what its own report saved (campaign B10): title, ks, threshold, stage / horizon / tag maps,
    validity and extras (the IRT block is recomputed). Writes to ``out_dir`` or ``<run>/bench-report``; it overwrites
    ``<run>/bench`` only with ``force``, so the run's own report is never replaced by a lesser one."""
    run_dir = Path(run_dir)
    prev = run_dir / "bench" / "summary.json"
    doc = json.loads(prev.read_text()) if prev.exists() else {}
    s = doc.get("summary") or {}
    cfg = dict(s.get("config") or {})
    stages = {t: [(n, list(x)) for n, x in v] for t, v in (cfg.get("stages") or {}).items()}
    horizons, tags = cfg.get("horizons"), cfg.get("tags") or {}
    keep = {k: v for k, v in cfg.items() if k not in ("ks", "threshold", "stages", "horizons", "tags", "controls")}  # re-derived from the maps
    out = out_dir or (run_dir / "bench" if force else run_dir / "bench-report")
    return report_run(doc.get("title") or run_dir.name, run_dir, ks=ks if ks is not None else s.get("ks"),
                      threshold=threshold if threshold is not None else float(s.get("threshold") or 1.0),
                      stages_of=stages.get if stages else None, horizon_of=(lambda t: horizons.get(t, "all")) if horizons else None,
                      tags_of=(lambda t: tags.get(t, [])), extras={k: v for k, v in doc.items() if k not in ("title", "digest", "summary", "irt")},
                      out_dir=out, config=keep, runner={"stopped": (s.get("run") or {}).get("stopped"), "spend": (s.get("run") or {}).get("spend")} if s else None,
                      controls=cfg.get("controls") or ())
