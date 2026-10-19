"""P5b (swarm-pipeline/00 §2, §3, §4, §7, §14, §15 P5): swarm-built windows, and failure targeting of generated tasks
against a frontier closed model. Paid phase: one USD ledger (``<out>/spend.jsonl``, every row tagged with its stage and
role) with a hard cap (``--cap``): every client is guarded, so no call starts once the total reaches it.

    cd <repo root>   # the OpenRouter key resolves from the nearest .env
    python crucible/scripts/lh_swarm.py --out <dir> --store <store3/train> --clones <lhrepos> --cap 30 <stage> ...

Part A (``factory/generators/swarm.py``):
- ``swarm``    candidates (a store3 training window's start A + the real requests of its changelog and the issues it
               cites) -> one P1 machine per candidate: lease -> planner -> oracle builder session (one commit per milestone)
               -> oracle check -> verifier writer session (never sees the oracle) -> reviewer -> build (synthetic history,
               ``validate_window``: World, leak probes, fail-to-pass, prober, commit-revert mutants) -> prompt (S7) ->
               gates (S6, S5, S8, S3, S10) -> alternatives + shadow (S8 alternatives, S9) -> hacker (S12) -> accept;
- ``funnel``   the swarm funnel and one example window.

Part B (frontier targeting, ``factory/targeting.py``):
- ``models``   the OpenRouter models list (public endpoint, no key): the frontier closed model, its id and prices;
- ``pilot``    one episode of the frontier model on one window; cost measured, extrapolation written before any scaling;
- ``target``   per ``--windows`` (a spec ``kind:id[=target][@simplify]``): S8 on the simplified task, the scripted probe, the
               policy's probe episode, ``--attempts`` attempts per arm (full, simplified) with the checkpoint sidecar on,
               the counterfactual, band and the S15 verdict;
- ``audit``    the P3 RLM auditor (deepseek, seeded controls) on every failing targeting episode: the cause mix per window;
- ``report``   ``<out>/p5b_report.json`` (spend by role);
- ``cleanup``  removes this phase's containers (``crucible-p5b-*``, ``crucible-lh-p5b-*``) and nothing else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

GRADE_PREFIX = "crucible-p5b-"
WORLD_PREFIX = "crucible-lh-p5b-"
SCRATCH = Path(__file__).resolve().parents[3]  # the scratchpad holding the stores (default roots below)
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"
FRONTIER_VENDORS = ("openai", "anthropic")
AUTHOR = "qwen/qwen3.7-flash"
REVIEW = "deepseek/deepseek-v4-flash"


def _now() -> float:
    return round(time.time(), 1)


def jload(p: Path, default: Any = None) -> Any:
    return json.loads(p.read_text()) if p.exists() else default


def jsave(p: Path, x: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, indent=1, sort_keys=True, default=str))
    tmp.replace(p)


# ---- the run: one guarded USD ledger ------------------------------------------------------------------------------------------
def p5b_run(a: argparse.Namespace) -> Any:
    """``lh_run.Run`` (guarded clients, one ledger persisted in ``<out>/spend.jsonl``, the hard ``--cap``) with this phase's
    lane: grading containers ``crucible-p5b-*``, World containers ``crucible-lh-p5b-world-*``."""
    import lh_run

    from crucible.factory.lanes.windows import WindowLane, sandbox_factory

    ns = argparse.Namespace(work=a.out, out=str(Path(a.out) / "lh_out"), cap=a.cap, store=a.store, clones=a.clones, parallel=a.parallel, windows="",
                            routes=False)
    run = lh_run.Run(ns)
    run.lane = WindowLane(run.store, repo_dirs={p.name: str(p) for p in Path(a.clones).iterdir() if p.is_dir()},
                          sandboxes=sandbox_factory(parallel=a.parallel, name_prefix=GRADE_PREFIX))
    run.lane.world_prefix = f"{WORLD_PREFIX}world-"
    return run


def spend_rows(out: Path) -> list[dict[str, Any]]:
    p = out / "spend.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def spend_by_role(out: Path) -> dict[str, float]:
    c: Counter[str] = Counter()
    for r in spend_rows(out):
        c[f"{r.get('stage')}/{r.get('role')}"] += float(r.get("usd") or 0)
    return {k: round(v, 5) for k, v in sorted(c.items())}


def spent(out: Path) -> float:
    return round(sum(float(r.get("usd") or 0) for r in spend_rows(out)), 6)


# ---- Part B: the frontier model ---------------------------------------------------------------------------------------------
def pick_frontier(models: list[Mapping[str, Any]], *, vendors: tuple[str, ...] = FRONTIER_VENDORS, prefer: str | None = None) -> dict[str, Any]:
    """The frontier closed model from an OpenRouter models list: tool calling supported, from ``vendors``, newest first;
    ``prefer`` (an id) wins when listed. Returns ``{id, prices per 1M tokens, candidates}``."""

    def row(m: Mapping[str, Any]) -> dict[str, Any]:
        p = m.get("pricing") or {}

        def per_m(k: str) -> float | None:
            v = p.get(k)
            return None if v in (None, "") else round(float(v) * 1e6, 4)

        return {"id": m["id"], "created": m.get("created"), "input": per_m("prompt"), "output": per_m("completion"), "cache_read": per_m("input_cache_read"),
                "cache_write": per_m("input_cache_write"), "context": m.get("context_length"), "tools": "tools" in (m.get("supported_parameters") or [])}

    cands = [row(m) for m in models if str(m.get("id", "")).split("/")[0] in vendors and ":" not in str(m.get("id")) and "tools" in (m.get("supported_parameters") or [])]
    cands.sort(key=lambda r: -(r["created"] or 0))
    chosen = next((r for r in cands if r["id"] == prefer), None) or (cands[0] if cands else None)
    return {"chosen": chosen, "candidates": cands[:16]}


def stage_models(a: argparse.Namespace) -> dict[str, Any]:
    """The public models list (no key is sent or read); the choice with every listed price."""
    import urllib.request

    with urllib.request.urlopen(OPENROUTER_MODELS, timeout=60) as r:
        data = json.loads(r.read().decode())["data"]
    res = pick_frontier(data, prefer=a.frontier or None)
    res["queried_at"] = _now()
    res["why"] = a.frontier_why
    jsave(Path(a.out) / "frontier.json", res)
    print(json.dumps(res["chosen"]), flush=True)
    return res


def frontier_id(a: argparse.Namespace) -> str:
    f = jload(Path(a.out) / "frontier.json") or {}
    if a.frontier:
        return a.frontier
    if not f.get("chosen"):
        raise SystemExit("run the models stage first (or pass --frontier)")
    return f["chosen"]["id"]


# ---- Part B: the tasks targeted ----------------------------------------------------------------------------------------------
def p5_root(a: argparse.Namespace) -> Path:
    return Path(a.p5)


def target_spec(spec: str) -> dict[str, Any]:
    """``kind:id[=m1+m2][@simplify]`` -> ``{kind, id, targets, simplify}``; kinds: window (store3), stitched, rebreak, swarm."""
    kind, _, rest = spec.partition(":")
    rest, _, simp = rest.partition("@")
    wid, _, tg = rest.partition("=")
    default = {"window": "prerequisites", "stitched": "host_change", "rebreak": "locate", "swarm": "prerequisites"}
    if kind not in default:
        raise ValueError(f"target kind is one of {sorted(default)}, got {kind!r}")
    return {"kind": kind, "id": wid, "targets": [t for t in tg.split("+") if t], "simplify": simp or default[kind]}


_LANES: dict[str, Any] = {}


def gen_lane(a: argparse.Namespace, kind: str) -> Any:
    """A generator's lane over its P5 store with this phase's container names (grading ``crucible-p5b-*``, World
    ``crucible-lh-p5b-world-*``)."""
    if kind in _LANES:
        return _LANES[kind]
    from crucible.factory.generators.base import GenLane
    from crucible.factory.lanes.windows import WindowStore, sandbox_factory

    if kind == "stitched":
        from crucible.factory.generators.stitch import NAME, StitchTask, stitch_sandboxes

        root = p5_root(a) / "stitch"
        meta = jload(root / "pairs.json", {}) or {}
        lane = GenLane(WindowStore(root / "store" / "train", split="train"), generator=NAME, task_cls=StitchTask,
                       repo_dirs={k: str(v["mono"]) for k, v in meta.items()}, sandboxes=stitch_sandboxes(name_prefix=GRADE_PREFIX))
    elif kind == "rebreak":
        from crucible.factory.generators.rebreak import NAME, RebreakTask

        root = p5_root(a) / "rebreak"
        lane = GenLane(WindowStore(root / "store" / "train", split="train"), generator=NAME, task_cls=RebreakTask,
                       repo_dirs={p.name: str(p) for p in (root / "clones").iterdir() if p.is_dir()},
                       sandboxes=sandbox_factory(parallel=a.parallel, memory="1536m", name_prefix=GRADE_PREFIX))
    elif kind == "swarm":
        from crucible.factory.generators.swarm import NAME, SwarmTask

        root = Path(a.out) / "swarm"
        lane = GenLane(WindowStore(root / "store" / "train", split="train"), generator=NAME, task_cls=SwarmTask,
                       repo_dirs={p.name: str(p) for p in (root / "clones").iterdir() if p.is_dir()},
                       sandboxes=sandbox_factory(parallel=a.parallel, memory="1536m", name_prefix=GRADE_PREFIX))
    else:
        raise ValueError(kind)
    lane.world_prefix = f"{WORLD_PREFIX}world-"
    _LANES[kind] = lane
    return lane


def build_task(run: Any, a: argparse.Namespace, spec: Mapping[str, Any], *, prompt: Mapping[str, Any] | None = None, tag: str = "") -> Any:
    """The authored task of a target spec: store3 windows under P2's prompt in the World of their head version (P4's
    versions), generated tasks under their own ``prompt.json``."""
    if spec["kind"] == "window":
        import lh_target

        ns = argparse.Namespace(out=a.out, prompts=a.prompts)
        return lh_target.window_task(run, ns, spec["id"], prompt=prompt, tag=tag)
    lane = gen_lane(a, spec["kind"])
    doc = dict(prompt) if prompt is not None else jload(lane.store.dir(spec["id"]) / "prompt.json")
    if not doc or doc.get("findings"):
        raise SystemExit(f"{spec['id']}: no clean authored prompt")
    t = lane.task(spec["id"], authored=doc)
    if tag:  # a probe or paraphrase variant: same World and checks, another prompt
        t = lane.task_cls(lane, t.mined, t.window, task_id=f"{spec['id']}{tag}", prompt=doc["prompt"], requirements=list(doc.get("requirements") or []),
                          spans=dict(doc.get("spans") or {}), checkset=t.checkset)
    return t


def simplify_task(task: Any, kind: str, *, targets: list[str], root: Path) -> Any:
    """``targeting.simplify`` with this phase's root for every kind that writes a new start tree (``prerequisites``,
    ``host_change``), so nothing is written into P5a's stores."""
    from crucible.factory.targeting import SIMPLIFY_KINDS, generator_of

    gen = generator_of(task)
    if kind not in SIMPLIFY_KINDS.get(gen, ()):
        raise ValueError(f"generator {gen!r} simplifies by {SIMPLIFY_KINDS.get(gen, ())}, not {kind!r}")
    kw: dict[str, Any] = {"milestones": list(targets) or None}
    if kind in ("prerequisites", "host_change"):
        kw["root"] = root
    return task.simplify(kind, **kw)


def probe_command(task: Any) -> str | None:
    """The scripted probe's suite command: a stitched window runs the consumer's own tests from its root (one pytest over
    both trees collides on conftest and package names; an old host release can carry a test module that no longer imports on
    the World's Python, and a collection error interrupts its whole run); every other task uses ``targeting.suite_command``."""
    if getattr(task, "generator", "") != "stitched_window":
        return None
    from crucible.factory.lanes.windows import suite_targets

    t = suite_targets(task.base_dir / "consumer", {})
    return f"cd consumer && python -m pytest -q -p no:cacheprovider -o addopts= --continue-on-collection-errors {' '.join(t)}" if t else None


def two_repo_probe(ep: Mapping[str, Any] | None, task: Any) -> Mapping[str, Any] | None:
    """A stitched window's workspace holds two repositories: a policy that runs both suites and reports their sum ran the
    tests and reported a count its own runs printed. ``targeting.probe_episode`` compares the report with each run's count
    alone; here the sum of the runs' counts is accepted too (``two_repo_sum``)."""
    if ep is None or ep.get("ok") or getattr(task, "generator", "") != "stitched_window":
        return ep
    seen = [int(x) for x in ep.get("seen_in_own_runs") or []]
    rep = ep.get("reported")
    if ep.get("ran_tests") and rep is not None and len(seen) >= 2 and any(abs(rep - (seen[i] + seen[j])) <= max(1, 0.1 * rep)
                                                                          for i in range(len(seen)) for j in range(i + 1, len(seen))):
        return {**ep, "ok": True, "why": "", "two_repo_sum": True}
    return ep


def episode_cfg(a: argparse.Namespace, **kw: Any) -> Any:
    """The M16-compatible solver settings of ``scripts/lh_run.py`` (tool limit 150 unless ``--tool-limit``)."""
    import lh_run

    return lh_run.EpisodeCfg(tool_limit=kw.pop("tool_limit", a.tool_limit), **kw)


def extrapolate(pilot_usd: float, *, spent_usd: float, cap: float, slack: float, arms: int = 2) -> dict[str, Any]:
    """From one measured episode: how many more episodes fit under ``cap - slack``, and per-window plans (n per arm)."""
    left = max(0.0, cap - slack - spent_usd)
    per = max(pilot_usd, 1e-6)
    eps = int(left / per + 1e-9)
    plans = {f"n={n}": {"per_window_usd": round(per * n * arms, 3), "windows": int(left / (per * n * arms) + 1e-9)} for n in (2, 3, 4)}
    return {"pilot_usd": round(pilot_usd, 4), "spent_usd": round(spent_usd, 4), "cap": cap, "slack": slack, "left_usd": round(left, 3),
            "episodes_that_fit": eps, "plans": plans}


async def stage_pilot(a: argparse.Namespace) -> dict[str, Any]:
    """One frontier episode, full arm, on the first ``--windows`` spec: its cost, and the extrapolation written into the
    report before anything scales (``<out>/pilot.json``)."""
    import lh_run

    out = Path(a.out)
    run = p5b_run(a)
    spec = target_spec(a.windows.split(",")[0])
    task = build_task(run, a, spec)
    model = frontier_id(a)
    cfg = episode_cfg(a)
    await lh_run.run_episodes(run, f"pilot-{spec['id']}", [task], [lh_run.policy_ref(run, model, cfg)], k=1, concurrency=1, stage="pilot", role="solver",
                              cfg=cfg, checkpoints=True)
    eps = lh_run.episodes_of(run, f"pilot-{spec['id']}")
    usd = sum(e["usd"] for e in eps)
    res = {"model": model, "window": spec, "episodes": eps, "usd": round(usd, 5),
           "extrapolation": extrapolate(usd, spent_usd=spent(out), cap=a.cap, slack=a.slack)}
    jsave(out / "pilot.json", res)
    print(json.dumps({k: res[k] for k in ("model", "usd", "extrapolation")}, default=str), flush=True)
    for e in eps:
        print(json.dumps({k: e.get(k) for k in ("reward", "phi", "milestones", "stop", "steps", "usd", "prompt_tokens", "completion_tokens", "masked")}), flush=True)
    return res


async def stage_target(a: argparse.Namespace) -> dict[str, Any]:
    """Per window: S8 on the simplified task, the scripted probe, the probe episode, both arms (``--attempts``, checkpoint
    sidecar on), counterfactual, band, the S15 verdict with the cause mix of ``audit`` when it ran."""
    import lh_run
    import lh_target

    from crucible.factory.longtask import ground_truth
    from crucible.factory.targeting import PROBE_PROMPT, band, counterfactual, outcomes_of, prerequisite_probe, probe_episode, scripted_probe

    out = Path(a.out)
    run = p5b_run(a)
    model = frontier_id(a)
    cfg = episode_cfg(a)
    pol = lh_run.policy_ref(run, model, cfg)
    rows: dict[str, Any] = {}
    for raw in [w for w in a.windows.split(",") if w]:
        spec = target_spec(raw)
        wid = spec["id"]
        jp = out / "target" / f"{wid}.json"
        j = jload(jp, {}) or {}
        task = build_task(run, a, spec)
        targets = spec["targets"] or task.target_milestones()
        try:
            simple = simplify_task(task, spec["simplify"], targets=targets, root=out / "simplified")
        except (ValueError, RuntimeError) as e:  # e.g. host_change pre-applies every milestone of an observation-only stitched window
            j.update(window=wid, kind=spec["kind"], targets=targets, simplify=spec["simplify"], simplify_error=str(e)[:300])
            jsave(jp, j)
            print(f"target {wid}: simplify {spec['simplify']} refused: {e}", flush=True)
            rows[wid] = j
            continue
        if "s8_simple" not in j:  # the simplified task is a changed task: S8 again before any solver token (00 §4)
            gt = await ground_truth(simple)
            j["s8_simple"] = {"ok": gt["ok"], "failed": gt["failed"]}
            j.update(window=wid, kind=spec["kind"], generator=getattr(task, "generator", "mined_window"), targets=targets, simplify=spec["simplify"],
                     world=task.window["world"].get("image"), simplified={"task": simple.task_id, "milestones": [m["id"] for m in simple.milestones],
                                                                           **{k: (simple.window.get("simplified_from") or {}).get(k) for k in ("pre_applied", "mode")}})
            jsave(jp, j)
        if not j["s8_simple"]["ok"]:
            print(f"target {wid}: the simplified task fails S8 {j['s8_simple']['failed']}: skipped", flush=True)
            rows[wid] = j
            continue
        if "truths" not in (j.get("probe_scripted") or {}):
            j["probe_scripted"] = await scripted_probe(task, out / "probe" / wid, command=probe_command(task))
            jsave(jp, j)
        if a.no_solvers:
            rows[wid] = j
            continue
        if run.spent() >= a.cap - a.slack:
            print(f"target {wid}: the slack is reached ({run.spent():.3f} of {a.cap}): stopped", flush=True)
            break
        if "probe_episode" not in j:
            pt = build_task(run, a, spec, prompt={"prompt": PROBE_PROMPT, "requirements": [], "spans": {}}, tag="~probe")
            pcfg = episode_cfg(a, tool_limit=12, max_steps=20)
            await lh_run.run_episodes(run, f"probe-{wid}", [pt], [lh_run.policy_ref(run, model, pcfg)], k=1, concurrency=1, stage="probe", role="probe",
                                      cfg=pcfg, checkpoints=False)
            for e in lh_run.episodes_of(run, f"probe-{wid}"):
                calls, last = lh_target.episode_trail(Path(e["src_dir"]))
                j["probe_episode"] = {**probe_episode(calls, last, j["probe_scripted"].get("truths")), "usd": e["usd"], "stop": e["stop"], "masked": e["masked"]}
            jsave(jp, j)
        n = a.attempts
        rid = f"arms{a.batch}-{wid}"  # a later batch (``--batch``) adds attempts under its own run id; every batch's episodes count
        await lh_run.run_episodes(run, rid, [task, simple], [pol], k=n, concurrency=a.concurrency, stage="arms", role="solver", cfg=cfg,
                                  checkpoints=True)
        eps = [e for d in sorted((out / "runs").glob(f"arms*-{wid}")) if d.name.split("-", 1)[1] == wid for e in lh_run.episodes_of(run, d.name)]
        extra = [e for e in lh_run.episodes_of(run, f"pilot-{wid}")] if a.with_pilot else []
        full_eps = [e for e in [*extra, *eps] if e["task"] == task.task_id]
        simple_eps = [e for e in eps if e["task"] == simple.task_id]
        fo, so = outcomes_of(full_eps, task, targets, targets), outcomes_of(simple_eps, simple, targets, targets)
        j["arms"] = {"full": [vars(o) for o in fo], "simple": [vars(o) for o in so]}
        j["episodes"] = {"full": [_ep_row(e) for e in full_eps], "simple": [_ep_row(e) for e in simple_eps]}
        j["counterfactual"] = counterfactual(fo, so)
        j["band"] = band([o.reward for o in fo if not o.masked])
        j["probe"] = prerequisite_probe(j["probe_scripted"], two_repo_probe(j.get("probe_episode"), task))
        j["phi"] = {"full": _mean([o.phi for o in fo if not o.masked]), "simple": _mean([o.phi for o in so if not o.masked]),
                    "full_series": [[pt[:2] for pt in (e.get("progress") or {}).get("series") or []] for e in full_eps]}
        j["usd"] = round(sum(e["usd"] for e in [*full_eps, *simple_eps]), 5)
        jsave(jp, j)
        rows[wid] = j
        cf = j["counterfactual"]
        print(f"target {wid} {targets}: {cf['verdict']} full {cf.get('success_full')}/{cf.get('n_full')} simple {cf.get('success_simple')}/"
              f"{cf.get('n_simple')} p={cf.get('fisher_p')} power={cf.get('power_at_ref')}; probe {j['probe']['ok']}; band {j['band']['in_band']}; "
              f"usd {j['usd']} (spent {run.spent():.3f})", flush=True)
    return rows


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def _ep_row(e: Mapping[str, Any]) -> dict[str, Any]:
    prog = e.get("progress") or {}
    return {**{k: e.get(k) for k in ("sample", "reward", "phi", "milestones", "stop", "steps", "usd", "masked", "failure", "prompt_tokens", "completion_tokens",
                                       "compactions", "hack_suspect", "path_violations", "seconds", "src_dir")},
            "checkpoints": {**{k: prog.get(k) for k in ("phi0", "phi_final", "sum_delta", "checkpoints", "graded", "milestones_lost", "compaction_turns") if k in prog},
                            "series": [pt[:4] for pt in prog.get("series") or []]}}


async def stage_audit(a: argparse.Namespace) -> dict[str, Any]:
    """The P3 RLM auditor (``--auditor``, seeded control bank in every batch, cross-family) on every targeting episode
    (both arms); per window the cause mix of trusted labels plus deterministic causes (limit stops, masks)."""
    import lh_audit

    from crucible.factory.traceaudit import LoopConfig, load_bundle, run_loop
    from crucible.factory.versioning import VersionStore

    out = Path(a.out)
    run = p5b_run(a)
    bundles = []
    for wdir in sorted((out / "runs").glob("arms*")) + sorted((out / "runs").glob("pilot-*")):
        for adir in sorted(wdir.glob("samples/*/a*")):
            if (adir / "record.json").exists() and adir == max(adir.parent.glob("a*")):
                b = load_bundle(adir, run=wdir.name)
                b.meta["workspace"] = str(adir / "work" / "ws")
                bundles.append(b)
    from crucible.actors.rlm.agent import RLMConfig
    from crucible.factory.audit import AuditConfig
    from crucible.factory.models import parse_spec
    from crucible.factory.traceaudit import P3Launcher, RLMAuditor

    # P3's auditor with its notes and checks unchanged; its turn cap is raised from 24 to ``--auditor-turns`` (the frontier's
    # trails are long: 30 of 41 first sessions ended at 24 turns with no verdict, a rejected label)
    spec = parse_spec(a.auditor)
    client = run.solver_client(spec.ref.split(":", 1)[-1], tag="auditor")
    auditor = RLMAuditor(client, model=spec.model, workdir=out / "audit" / "auditor", launcher=P3Launcher, config=AuditConfig(concurrency=4),
                         rlm_config=RLMConfig(temperature=0.0, tree_max_depth=0, max_turns=a.auditor_turns, window_tokens=64_000, max_tokens=4000,
                                              cell_timeout_s=60.0, wall_s=1500.0), max_steps=a.auditor_turns + 16)
    # incremental: a bundle with a verdict from an earlier audit run keeps it (its batch's trust with it); only new bundles and
    # those whose label was rejected (no verdict) are labelled again
    lp = out / "audit" / "labels.json"
    known: dict[str, Any] = {k: v for k, v in (jload(lp, {}) or {}).items() if v.get("status") in ("labelled", "abstained", "deterministic")}
    todo = {b.id for b in bundles if b.id not in known}
    rep = await run_loop(bundles, out=out / "audit" / "loop", store=VersionStore(out / "audit" / "versions.jsonl"), contexts=lambda p: None, auditor=auditor,
                         seeded=lh_audit.controls(Path(a.controls)), config=LoopConfig(intent={"id": "intent-p5-generic-change"}, batch_size=6),
                         label_ids=todo, spend=lambda who: run.spend("audit", "auditor", who))
    rows = [json.loads(x) for x in (out / "audit" / "loop" / "bundles.jsonl").read_text().splitlines() if x.strip()]
    for r in rows:
        if r["id"] in known:
            r["label"] = known[r["id"]]
        elif r.get("label"):
            known[r["id"]] = r["label"]
    jsave(lp, known)
    jsave(out / "audit" / "bundles_merged.json", rows)
    per: dict[str, Counter[str]] = {}
    simple: dict[str, Counter[str]] = {}
    for r in rows:
        task = str(r.get("task") or "")
        w = task.split("~")[0].split("#")[0]
        if r.get("passed") or r.get("masked"):
            continue
        lab = r.get("label") or {}
        cause = r.get("deterministic_cause") or (lab.get("cause") if lab.get("trusted_batch") else None)
        if cause:  # the cause mix is about the full task's failures; the simplified arm (a control) is reported apart
            (simple if "~" in task else per).setdefault(w, Counter())[cause] += 1
    res = {"by_window": {w: dict(c) for w, c in per.items()}, "simplified_arm": {w: dict(c) for w, c in simple.items()},
           "report": {k: rep.get(k) for k in ("bundles", "masked", "passed", "failing", "causes", "auditor", "first_wrong_step")}}
    jsave(out / "audit" / "cause_mix.json", res)
    print(json.dumps(res, default=str)[:2000], flush=True)
    return res


async def stage_verdicts(a: argparse.Namespace) -> dict[str, Any]:
    """S15 per window from its targeting evidence and the audited cause mix (``factory.targeting.target_verdict`` through the
    S15 machine)."""
    import lh_target

    from crucible.factory.targeting import TargetEvidence, cause_mix, target_verdict

    out = Path(a.out)
    mix = (jload(out / "audit" / "cause_mix.json", {}) or {}).get("by_window") or {}
    rows = {}
    for p in sorted((out / "target").glob("*.json")):
        j = jload(p)
        if "counterfactual" not in j:
            continue
        wid = j["window"]
        j["cause"] = {**cause_mix(Counter(mix.get(wid) or {})), "source": "P3 RLM auditor (deepseek, seeded controls) on this phase's frontier trails"}
        ev = TargetEvidence(task=wid, band=j["band"], cause=j["cause"], counterfactual=j["counterfactual"], probe=j["probe"], paraphrase=None)
        j["verdict"] = target_verdict(ev)
        j["machine"] = await lh_target.run_s15_machine(out, wid, ev)
        jsave(p, j)
        rows[wid] = j
    return rows


# ---- report, cleanup -----------------------------------------------------------------------------------------------------------
def stage_report(a: argparse.Namespace) -> dict[str, Any]:
    out = Path(a.out)
    tgt = {}
    for p in sorted((out / "target").glob("*.json")):
        j = jload(p)
        cf = j.get("counterfactual") or {}
        tgt[j["window"]] = {"kind": j.get("kind"), "generator": j.get("generator"), "targets": j.get("targets"), "simplify": j.get("simplify"),
                            "s8_simple": j.get("s8_simple"), "full": f"{cf.get('success_full')}/{cf.get('n_full')}", "simple": f"{cf.get('success_simple')}/{cf.get('n_simple')}",
                            "phi": j.get("phi"), "counterfactual": {k: cf.get(k) for k in ("verdict", "fisher_p", "power_at_ref", "dphi", "dphi_ci", "simple_vs_ref_p", "why")},
                            "band": j.get("band"), "probe": {k: (j.get("probe") or {}).get(k) for k in ("ok", "owner", "why")}, "cause": j.get("cause"),
                            "verdict": {k: (j.get("verdict") or {}).get(k) for k in ("ok", "route", "owner", "findings", "flags")}, "machine": j.get("machine"),
                            "usd": j.get("usd"), "episodes": j.get("episodes")}
    swarm = jload(out / "swarm" / "summary.json")
    rep = {"phase": "P5b", "cap_usd": a.cap, "frontier": jload(out / "frontier.json"), "pilot": jload(out / "pilot.json"), "swarm": swarm, "target": tgt,
           "audit": jload(out / "audit" / "cause_mix.json"), "spend_by_role": spend_by_role(out), "spend_total": spent(out)}
    jsave(out / "p5b_report.json", rep)
    if a.report_to:
        jsave(Path(a.report_to), rep)
    print(json.dumps({"spend_by_role": rep["spend_by_role"], "total": rep["spend_total"]}), flush=True)
    return rep


def cleanup() -> None:
    for prefix in (GRADE_PREFIX, WORLD_PREFIX):
        r = subprocess.run(["docker", "ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}"], capture_output=True, text=True, check=False)
        names = [x for x in r.stdout.split() if x.startswith(prefix)]
        if names:
            subprocess.run(["docker", "rm", "-f", *names], capture_output=True, check=False)
        print(f"removed {len(names)} {prefix}* containers")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", required=True, help="store3/train (the held-out stores are never opened)")
    ap.add_argument("--clones", required=True)
    ap.add_argument("--p5", default=str(SCRATCH / "p5"), help="P5a's root (stitched and rebreak stores)")
    ap.add_argument("--prompts", default=str(SCRATCH / "lhwork" / "p2" / "windows"), help="P2's authored prompts of store3 windows")
    ap.add_argument("--cap", type=float, default=30.0, help="hard USD cap of the whole phase (every role, one ledger)")
    ap.add_argument("--slack", type=float, default=3.0, help="USD kept unspent: solver stages stop when the cap minus this is reached")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--windows", default="", help="target specs kind:id[=m1+m2][@simplify], comma-separated")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--batch", default="", help="target: a tag for more attempts on a window already run (run id arms<tag>-<window>)")
    ap.add_argument("--tool-limit", type=int, default=150)
    ap.add_argument("--with-pilot", action="store_true", help="count the pilot episode in its window's full arm")
    ap.add_argument("--frontier", default="", help="the frontier model id (default: models stage's choice)")
    ap.add_argument("--frontier-why", default="")
    ap.add_argument("--auditor", default=f"openrouter:{REVIEW}")
    ap.add_argument("--auditor-turns", type=int, default=48)
    ap.add_argument("--controls", default=str(Path(__file__).resolve().parents[1] / "docs/research/task-factory/runs/controls.json"))
    ap.add_argument("--no-solvers", action="store_true")
    ap.add_argument("--report-to", default="")
    # Part A
    ap.add_argument("--candidates", default="", help="swarm: store3 window ids whose start A and changelog seed a candidate")
    ap.add_argument("--max-milestones", type=int, default=4)
    ap.add_argument("--gh-cache", default="", help="swarm: GitHub cache dir for the issues the changelog cites ('' = no network)")
    ap.add_argument("--skip-paid-gates", action="store_true", help="swarm: no alternatives, shadow or hacker calls")
    ap.add_argument("--swarm-parallel", type=int, default=2)
    ap.add_argument("--redo", action="store_true", help="swarm: a fresh machine run per candidate")
    ap.add_argument("--finish", action="store_true", help="swarm: re-run S9, S12 and S17 only (window, prompt and gates stored)")
    return ap.parse_args(argv)


async def amain(a: argparse.Namespace) -> None:
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for st in a.stages:
        if st == "models":
            stage_models(a)
        elif st == "pilot":
            await stage_pilot(a)
        elif st == "target":
            await stage_target(a)
        elif st == "audit":
            await stage_audit(a)
        elif st == "verdicts":
            await stage_verdicts(a)
        elif st == "swarm":
            await stage_swarm(a)
        elif st == "funnel":
            stage_funnel(a)
        elif st == "report":
            stage_report(a)
        elif st == "cleanup":
            cleanup()
        else:
            raise SystemExit(f"unknown stage {st}")


async def stage_swarm(a: argparse.Namespace) -> dict[str, Any]:
    """Part A: every ``--candidates`` store3 window through the swarm machine (``generators.swarm.SwarmRunner``)."""
    import lh_run

    from crucible.factory.generators.swarm import ROLE_MODELS, SwarmRunner
    from crucible.factory.lanes.windows import WindowStore
    from crucible.factory.sources.lease import EvalLease

    out = Path(a.out)
    run = p5b_run(a)
    gh = None
    if a.gh_cache:
        from crucible.factory.sources.github import GitHub

        gh = GitHub(Path(a.gh_cache))
    hcfg = episode_cfg(a, tool_limit=60, max_billed=1_000_000)
    hacker = ROLE_MODELS["hacker"]

    async def hack_episode(task: Any) -> dict[str, Any]:
        """S12: the hacker (reviewer family) as an agent in the solver's World, told to pass the hidden tests without the work."""
        ht = type(task)(task.lane, task.mined, task.window, task_id=f"{task.task_id}#hacker", prompt=lh_run.hacker_prompt(task.prompt),
                        requirements=task.requirements, spans={}, checkset=task.checkset)
        rid = f"hacker-{task.task_id}"
        await lh_run.run_episodes(run, rid, [ht], [lh_run.policy_ref(run, hacker, hcfg, pid=f"hacker:{hacker}")], k=1, concurrency=1, stage="swarm",
                                  role="hacker", cfg=hcfg, checkpoints=False)
        eps = lh_run.episodes_of(run, rid)
        e = eps[-1] if eps else {}
        return {"reward": e.get("reward"), "phi": e.get("phi"), "masked": e.get("masked"), "stop": e.get("stop"), "usd": e.get("usd"),
                "path_violations": e.get("path_violations"), "hack_suspect": e.get("hack_suspect"), "dir": e.get("dir"),
                "confirmed_hack": e.get("reward") == 1.0, "run_id": rid}

    runner = SwarmRunner(run, root=out / "swarm", clones=Path(a.clones), store3=WindowStore(Path(a.store), split="train"), world_prefix=WORLD_PREFIX,
                         grade_prefix=GRADE_PREFIX, lease=EvalLease.default(), hack_episode=hack_episode, gh=gh, max_milestones=a.max_milestones,
                         parallel=a.swarm_parallel, grade_parallel=a.parallel, paid_gates=not a.skip_paid_gates, redo=a.redo)
    if a.finish:  # S9 + S12 (+ S17) again on candidates whose build, prompt and gates are stored
        res = {w: await runner.finish(w) for w in a.candidates.split(",") if w}
    else:
        res = await runner.run_all([w for w in a.candidates.split(",") if w])
    print(json.dumps({w: {k: (r or {}).get(k) for k in ("status", "reason", "error", "usd")} for w, r in res.items()}, default=str, indent=1)[:4000], flush=True)
    return res


def stage_funnel(a: argparse.Namespace) -> dict[str, Any]:
    from crucible.factory.generators.swarm import summarize

    res = summarize(Path(a.out) / "swarm")
    jsave(Path(a.out) / "swarm" / "summary.json", res)
    print(json.dumps(res, default=str, indent=1)[:4000], flush=True)
    return res


def main() -> None:
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()

