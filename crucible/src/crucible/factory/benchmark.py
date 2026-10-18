"""The sealed held-out benchmark (swarm-pipeline/00 §10, §0 V9, §15 P6; study 08 §8).

``build`` turns a held-out window store (``lhwork/store3/heldout``) into a benchmark split, with no model call:

1. **Candidates**: the store's dependent windows (``mutants_valid``, not ``independent``), each a ``WindowTask`` with no
   prompt. Held-out prompts are **not authored here**: every task is ``prompt_pending`` until a dedicated author lease
   writes it under sealing (no factory role that writes training tasks reads the held-out set).
2. **Sealing** (``factory.sealing.seal`` plus the org rule) against the training store's trainable windows (or all mined
   ones, ``train_filter="all"``): disjoint repositories, disjoint **organisations** (``windows.REPO_URLS`` owners), the
   **time split** (every held-out window's B after every training window's B, compared in UTC; the base model's cutoff is
   reported), and **contamination** (``bench.contamination``: oracle diffs and prompts, exact n-grams, code substrings,
   near duplicates, containment). A task is in the sealed split only when every rule holds for it
   (``time_policy="strict"``); ``"report"`` keeps time violators in, flagged, and marks the manifest ``sealed: false``.
3. **Gates**: the training gates, executed (Docker, ``crucible-lh-p6-*`` / ``crucible-p6-*`` containers): S3
   (``windows_s3.world_s3``: the World builds, A's suite runs, leak probes, 60 clean reruns, the verifier does not move),
   S5 (``longtask.clean_world_solvable`` + ``null_agent_score``; the fairness lint waits for the prompt), S6
   (``longtask.check_longtask``: an executed ``requires`` edge, a well-formed DAG), S8 (``longtask.ground_truth``: oracle
   and alternatives 1, floors 0, every mutant fails its milestone) and S10 (``flake.measure_flake``: oracle grades at
   concurrency 1 and 4, Clopper-Pearson bound <= 5%). Results are journaled per task under ``gates/`` and reused while
   the task's content hash is unchanged (V11).
4. **Manifest**: per task its content hashes (window record, A tree, oracle patch, hidden tests, check set, verifier
   version, World digest), category (generator x horizon class x intent), sealing verdicts, gate verdicts and split
   membership; the split's portable rows (``factory.envs.task_row``, ``tasks.jsonl``); the metrics plan; a ``not_verified``
   list; and the sha256 of all of it. ``verify`` recomputes every hash from the store.

**Reporting** (``report`` / ``paired_report``, through ``bench.lh_metrics``): pass@k and pass^k (k = 1..5 from n >= 5
attempts) by category with confidence intervals that resample **tasks**; mean final Φ; milestones lost; compaction loss;
Brier / ECE; cost per passing episode with the max/min ratio across tasks; paired base vs trained per task with the MDE
(about 13 points at n = 30 when σ_d ≈ 0.25).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import random
import time
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.lanes.windows import HELD_OUT_REPOS, REPO_URLS

SPLIT = "heldout"
GATES = ("s3", "s5", "s6", "s8", "s10")
GRADE_PREFIX = "crucible-p6-grade-"
WORLD_PREFIX = "crucible-lh-p6-"
VERSION = "p6-1"

METRICS_PLAN: dict[str, Any] = {
    "unit": "task (resampled for every CI); episodes are nested in tasks",
    "attempts": "n >= 5 per (task, policy); k = 1..5",
    "categories": "generator x horizon class (H1 same repo class, H2 transfer surface) x intent",
    "capability": "pass@1 and pass@k by category, task-clustered bootstrap CI (bench.lh_metrics.pass_hat_k per task)",
    "reliability": "pass^k (all k of k pass, tau-bench) by category, task-clustered bootstrap CI",
    "progress": "mean final Phi (last measured checkpoint), milestones lost per episode, compaction-boundary loss ratio",
    "calibration": "STATUS.json Brier and ECE over (claimed p, verified y) per requirement",
    "cost": "USD per passing episode by category, with the max/min ratio across tasks (cost per episode rewards giving up)",
    "transfer": "paired base vs trained per task (bench.lh_metrics.paired): mean difference, repo-clustered bootstrap CI, MDE at 80% power",
    "held_out_traces": "only intent-level counts leave the benchmark (V9; swarm-pipeline/00 §8.6)",
    "masked": "infrastructure-masked attempts are counted, never scored (V8)",
}


# ---- candidates ------------------------------------------------------------------------------------------------------------
def utc(d: Any) -> str | None:
    if d is None:
        return None
    x = d if isinstance(d, dt.datetime) else dt.datetime.fromisoformat(str(d))
    if x.tzinfo is None:
        x = x.replace(tzinfo=dt.UTC)
    return x.astimezone(dt.UTC).isoformat()


def org_of(repo: str) -> str:
    return REPO_URLS.get(repo, f"{repo}/{repo}").split("/")[0].lower()


def trainable(rec: Mapping[str, Any]) -> bool:
    return rec.get("stage") == "mutants_valid" and rec.get("independent") is False


def sha_text(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()


def windows_of(root: Path, split: str, *, only: Callable[[Mapping[str, Any]], bool] | None = trainable) -> list[dict[str, Any]]:
    from crucible.factory.lanes.windows import WindowStore

    store = WindowStore(Path(root), split=split)
    out = []
    for wid in store.window_ids():
        rec = store.window(wid)
        if only is None or only(rec):
            out.append(rec)
    return out


def make_lane(root: Path, split: str, *, parallel: int = 4, sandboxes: Callable[[str], Any] | None = None) -> Any:
    from crucible.factory.lanes.windows import WindowLane, WindowStore, sandbox_factory

    lane = WindowLane(WindowStore(Path(root), split=split), sandboxes=sandboxes or sandbox_factory(parallel=parallel, name_prefix=GRADE_PREFIX))
    lane.world_prefix = f"{WORLD_PREFIX}world-"
    return lane


def unauthored_task(lane: Any, rec: Mapping[str, Any]) -> Any:
    """The window with no prompt (``prompt_pending``): every gate but the prompt lint runs on it."""
    from crucible.factory.lanes.windows import WindowTask, mined_of, window_checkset

    return WindowTask(lane, mined_of(rec, lane.store.dir(rec["id"]) / "base"), rec, task_id=str(rec["id"]), prompt="", requirements=[], spans={},
                      checkset=window_checkset(rec, []))


def solution_text(task: Any) -> str:
    """What a training task would have to contain to leak the held-out solution: the lines the oracle **adds** to the
    project's Python sources (tests, packaging, CI, ``docs/`` and import lines left out, diff headers dropped). Measured
    on store3: the full oracle diff gave 132 code-substring hits, every one shared boilerplate (GitHub Actions steps, diff
    headers, Sphinx ``conf.py``, ``from __future__ import annotations``)."""
    import re

    from crucible.factory.lanes.executable import is_protected

    files = {p: t for p, t in (task.window.get("oracle_files") or {}).items()
             if p.endswith(".py") and not is_protected(p) and not p.startswith(("docs/", "doc/"))}
    if not files:
        return ""
    imp = re.compile(r"^\s*(import\s|from\s+\S+\s+import\s)")
    out = []
    for ln in task.patch_of(files).splitlines():
        if ln.startswith("+") and not ln.startswith("+++"):
            body = ln[1:]
            if body.strip() and not imp.match(body):
                out.append(body)
    return "\n".join(out)


def content_hashes(task: Any, rec: Mapping[str, Any]) -> dict[str, str]:
    from crucible.bench.contamination import tree_fingerprint

    w = rec.get("world") or {}
    return {"window_record": sha_text(json.dumps(rec, sort_keys=True, default=str)), "a_tree": tree_fingerprint(task.base_dir),
            "oracle_patch": sha_text(task.oracle_patch()), "hidden_tests": sha_text(json.dumps(task.grader_tests, sort_keys=True)),
            "checkset": sha_text(json.dumps(task.checkset, sort_keys=True)), "verifier_version": str(task.verifier_version),
            "world_digest": str(w.get("digest") or "")}


# ---- sealing ---------------------------------------------------------------------------------------------------------------
def seal(train: Sequence[Mapping[str, Any]], heldout: Sequence[Mapping[str, Any]], *, train_prompts: Mapping[str, str] | None = None,
         diffs: Mapping[str, str] | None = None, model_cutoff: Any = None, contamination_check: bool = True) -> dict[str, Any]:
    """``factory.sealing.seal`` over (train, held-out) window records, plus the org rule and per-task verdicts.
    ``diffs``: window id -> oracle patch (contamination); ``train_prompts``: train window id -> authored prompt."""
    from crucible.factory.sealing import seal as base_seal

    diffs = dict(diffs or {})
    tp = dict(train_prompts or {})

    def row(r: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        return {"id": r["id"], "repo": r["repo"], "date_b": utc(r.get("b_date")), "prompt": prompt, "diff": diffs.get(r["id"], "")}

    tr = [row(r, tp.get(r["id"], "")) for r in train]
    ho = [row(r, "") for r in heldout]
    rep = base_seal(tr, ho, model_cutoff=utc(model_cutoff) if model_cutoff else None, contamination_check=contamination_check)
    train_orgs = {org_of(r["repo"]) for r in train}
    shared_orgs = sorted(train_orgs & {org_of(r["repo"]) for r in heldout})
    latest = max(((x["date_b"], x["id"]) for x in tr if x["date_b"]), default=(None, None))
    hit_by: dict[str, list[dict[str, Any]]] = {}
    for h in rep.contamination:
        hit_by.setdefault(str(h.get("task") or h.get("bench") or h.get("id") or ""), []).append(h)
    per = {}
    for r in heldout:
        d = utc(r.get("b_date"))
        per[r["id"]] = {"repo_ok": r["repo"] not in rep.shared_repos, "org": org_of(r["repo"]), "org_ok": org_of(r["repo"]) not in shared_orgs,
                        "time_ok": d is not None and latest[0] is not None and d > latest[0], "b_date_utc": d,
                        "after_model_cutoff": None if not model_cutoff else bool(d and d > utc(model_cutoff)),
                        "contamination": hit_by.get(r["id"], [])[:5], "contamination_ok": not hit_by.get(r["id"])}
    return {"report": rep.to_dict(), "shared_orgs": shared_orgs, "train_orgs": sorted(train_orgs), "latest_train": {"date_utc": latest[0], "window": latest[1]},
            "per_task": per, "train_windows": len(train), "heldout_windows": len(heldout)}


def sealed_ok(v: Mapping[str, Any], *, time_policy: str = "strict") -> tuple[bool, list[str]]:
    why = [k for k in ("repo_ok", "org_ok", "contamination_ok") if not v.get(k)]
    if not v.get("time_ok"):
        why.append("time_ok")
    blocking = [w for w in why if not (w == "time_ok" and time_policy == "report")]
    return not blocking, why


# ---- gates -----------------------------------------------------------------------------------------------------------------
@dataclass
class GateConfig:
    reruns: int = 60
    concurrency: int = 4
    flake_reruns_per_level: int = 30
    gates: tuple[str, ...] = GATES


async def run_gates(task: Any, rec: Mapping[str, Any], *, cfg: GateConfig | None = None, log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    """S3 / S5 / S6 / S8 / S10 for one held-out task (see the module doc); ``ok`` needs every gate run."""
    from crucible.factory.flake import FlakeConfig, measure_flake
    from crucible.factory.lanes.windows_s3 import world_s3
    from crucible.factory.longtask import check_longtask, clean_world_solvable, ground_truth, null_agent_score

    cfg = cfg or GateConfig()
    out: dict[str, Any] = {}
    t0 = time.monotonic()
    if "s6" in cfg.gates:
        found = check_longtask(task, need_prompt=False)
        edges = sum(len(m.get("requires") or []) for m in task.milestones)
        out["s6"] = {"ok": not found and edges >= 1, "findings": found, "executed_edges": edges}
        log(f"{task.task_id} s6 {out['s6']['ok']}")
    if "s5" in cfg.gates:
        cw = await clean_world_solvable(task)
        null = await null_agent_score(task)
        out["s5"] = {"ok": cw["ok"] and null["ok"], "clean_world": cw, "null_agent": null, "lint": "deferred: prompt_pending"}
        log(f"{task.task_id} s5 {out['s5']['ok']}")
    if "s8" in cfg.gates:
        gt = await ground_truth(task)
        out["s8"] = {"ok": gt["ok"], "failed": gt["failed"], "rows": gt["rows"], "alternatives": [c.id for c in task.alternatives()],
                     "note": "alternatives: the oracle as its unit steps (no second-family rewrite for held-out windows)"}
        log(f"{task.task_id} s8 {out['s8']['ok']}")
    if "s10" in cfg.gates:
        patch = task.oracle_patch()

        async def one(_i: int) -> Any:
            rep = await task.grade_patch(patch, trail=[])
            return {"reward": None if (rep.infra or {}).get("status") != "ok" else rep.reward, "checks": {k: v == "pass" for k, v in rep.gates.items()},
                    "fingerprint": sha_text(patch)}

        fl = await measure_flake(one, config=FlakeConfig(reruns=cfg.flake_reruns_per_level, concurrency=(1, cfg.concurrency)))
        out["s10"] = {"ok": fl.ok, **fl.to_dict()}
        log(f"{task.task_id} s10 {out['s10']['ok']} upper {fl.upper:.4f}")
    if "s3" in cfg.gates:
        s3 = await world_s3(rec, Path(task.base_dir), str((rec.get("world") or {})["image"]), reruns=cfg.reruns, concurrency=cfg.concurrency,
                            name_prefix=f"{WORLD_PREFIX}s3-")
        out["s3"] = {**s3, "ok": bool(s3.get("ok"))}
        log(f"{task.task_id} s3 {out['s3']['ok']}")
    out["ok"] = all(out.get(g, {}).get("ok") for g in cfg.gates)
    out["seconds"] = round(time.monotonic() - t0, 1)
    return out


# ---- build -----------------------------------------------------------------------------------------------------------------
@dataclass
class BuildConfig:
    heldout_root: Path
    train_root: Path
    out_dir: Path
    train_filter: str = "trainable"  # trainable | all
    time_policy: str = "strict"  # strict | report
    model_cutoff: str | None = None
    train_prompts: Path | None = None  # <dir>/<window id>/prompt.json (P2's authored training prompts) for the contamination scan
    gates: GateConfig = field(default_factory=GateConfig)
    run_gates: bool = True
    only: tuple[str, ...] = ()  # restrict to these held-out window ids (a partial build is marked partial)
    intent: str = "unassigned"
    sandboxes: Callable[[str], Any] | None = None  # grading sandboxes (default: Docker, ``crucible-p6-grade-*``)


def _jdump(x: Any) -> str:
    return json.dumps(x, indent=1, sort_keys=True, default=str)


def manifest_sha(m: Mapping[str, Any]) -> str:
    return sha_text(json.dumps({k: v for k, v in m.items() if k != "manifest_sha256"}, sort_keys=True, default=str))


async def build(cfg: BuildConfig, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Build (or resume) the sealed benchmark under ``cfg.out_dir`` and return the manifest (see the module doc)."""
    from crucible.factory.envs import task_row, write_task_rows

    if cfg.time_policy not in ("strict", "report") or cfg.train_filter not in ("trainable", "all"):
        raise ValueError("time_policy is strict|report and train_filter is trainable|all")
    out = Path(cfg.out_dir)
    (out / "gates").mkdir(parents=True, exist_ok=True)
    held = windows_of(cfg.heldout_root, "heldout")
    if cfg.only:
        held = [r for r in held if r["id"] in set(cfg.only)]
    train = windows_of(cfg.train_root, "train", only=trainable if cfg.train_filter == "trainable" else None)
    bad_split = [r["id"] for r in held if r["repo"] not in HELD_OUT_REPOS]
    if bad_split:
        raise ValueError(f"windows of non-held-out repositories in the held-out store: {bad_split[:5]}")
    hl = make_lane(cfg.heldout_root, "heldout", parallel=cfg.gates.concurrency, sandboxes=cfg.sandboxes)
    tl = make_lane(cfg.train_root, "train", sandboxes=cfg.sandboxes)
    tasks = {r["id"]: unauthored_task(hl, r) for r in held}
    log(f"held-out candidates {len(held)}; training windows {len(train)} ({cfg.train_filter})")
    diffs = {wid: solution_text(t) for wid, t in tasks.items()}
    for r in train:
        diffs[r["id"]] = solution_text(unauthored_task(tl, r))
    tp = {}
    if cfg.train_prompts:
        for r in train:
            f = Path(cfg.train_prompts) / r["id"] / "prompt.json"
            if f.exists():
                tp[r["id"]] = json.loads(f.read_text()).get("prompt", "")
    sealing = seal(train, held, train_prompts=tp, diffs=diffs, model_cutoff=cfg.model_cutoff)
    (out / "seal.json").write_text(_jdump(sealing))
    log(f"sealing: shared repos {sealing['report']['shared_repos']}, shared orgs {sealing['shared_orgs']}, "
        f"latest training B {sealing['latest_train']}, contamination hits {len(sealing['report']['contamination'])}")
    rows, members = [], []
    for r in held:
        t = tasks[r["id"]]
        hashes = content_hashes(t, r)
        key = sha_text(json.dumps(hashes, sort_keys=True), json.dumps(list(cfg.gates.gates)), str(cfg.gates.reruns))
        gfile = out / "gates" / f"{r['id']}.json"
        gates: dict[str, Any] = {}
        if gfile.exists() and json.loads(gfile.read_text()).get("key") == key:
            gates = json.loads(gfile.read_text())["gates"]
            log(f"{r['id']}: gates reused")
        elif cfg.run_gates:
            log(f"{r['id']}: gates running")
            gates = await run_gates(t, r, cfg=cfg.gates, log=log)
            gfile.write_text(_jdump({"key": key, "gates": gates}))
        sv = sealing["per_task"][r["id"]]
        s_ok, s_why = sealed_ok(sv, time_policy=cfg.time_policy)
        g_ok = bool(gates.get("ok"))
        excluded = [f"sealing:{w}" for w in s_why if not (w == "time_ok" and cfg.time_policy == "report")]
        excluded += [] if g_ok else [f"gate:{g}" for g in cfg.gates.gates if not (gates.get(g) or {}).get("ok")] or ["gates: not run"]
        in_split = s_ok and g_ok
        cls = HELD_OUT_REPOS[r["repo"]]
        rows.append({"id": r["id"], "repo": r["repo"], "org": org_of(r["repo"]), "horizon_class": cls, "generator": "mined_window", "intent": cfg.intent,
                     "category": f"mined_window/{cls}/{cfg.intent}", "a": r.get("a"), "b": r.get("b"), "a_date": r.get("a_date"), "b_date": r.get("b_date"),
                     "b_date_utc": sv["b_date_utc"], "milestones": len(t.milestones), "executed_edges": sum(len(m.get("requires") or []) for m in t.milestones),
                     "image": (r.get("world") or {}).get("image"), "prompt_status": "prompt_pending", "hashes": hashes, "sealing": sv,
                     "sealing_flags": s_why, "gates": {g: bool((gates.get(g) or {}).get("ok")) for g in cfg.gates.gates},
                     "gate_seconds": gates.get("seconds"), "in_split": in_split, "excluded": [] if in_split else excluded})
        if in_split:
            members.append(t)
    task_rows = [task_row(t, template="mined_window", runtime_config={"prompt_status": "prompt_pending"}) for t in members]
    for tr in task_rows:
        tr["split"] = SPLIT
        tr["args"]["ref"]["split"] = SPLIT
    tr_info = write_task_rows(task_rows, out / "tasks.jsonl", allow_heldout=True) if task_rows else {"rows": 0, "sha256": None}
    partial = bool(cfg.only) or not cfg.run_gates
    manifest: dict[str, Any] = {
        "benchmark": "crucible-lh-heldout", "version": VERSION, "split": SPLIT, "built_utc": utc(dt.datetime.now(dt.UTC)),
        "sealed": all(not x["sealing_flags"] for x in rows if x["in_split"]) and not partial, "partial": partial,
        "policy": {"time_policy": cfg.time_policy, "train_filter": cfg.train_filter, "model_cutoff": cfg.model_cutoff,
                   "rules": ["disjoint repositories", "disjoint organisations", "time split (UTC, B after every training B)", "no contamination hit"],
                   "contamination_scope": "the lines each oracle adds to Python sources (tests, packaging, CI, docs, imports "
                                          "excluded) vs the same for training windows plus their authored prompts",
                   "gates": list(cfg.gates.gates), "s3_reruns": cfg.gates.reruns, "s10_reruns_per_level": cfg.gates.flake_reruns_per_level},
        "sources": {"heldout_store": str(cfg.heldout_root), "train_store": str(cfg.train_root), "train_windows": len(train),
                    "train_prompts": len(tp), "heldout_candidates": len(held)},
        "sealing_summary": {"shared_repos": sealing["report"]["shared_repos"], "shared_orgs": sealing["shared_orgs"], "latest_train": sealing["latest_train"],
                            "time_violations": len(sealing["report"]["time_violations"]), "contamination_hits": len(sealing["report"]["contamination"]),
                            "cutoff_violations": len(sealing["report"]["cutoff_violations"])},
        "tasks": rows, "split_members": [t.task_id for t in members], "tasks_jsonl_sha256": tr_info.get("sha256"),
        "metrics_plan": METRICS_PLAN,
        "prompt_authoring": {"status": "pending", "rule": "held-out prompts are written later, under sealing, by a dedicated author lease; no model "
                             "authored any held-out text in P6", "lint": "S5's fairness lint and the S7 ledger run after authoring"},
        "not_verified": ["S5 fairness lint and the S7 prompt ledger (prompt_pending)", "S9 second verifier and S11-S14 (no prompt, no solver spend in P6)",
                         "S8 valid alternatives from another family (only the oracle's unit steps)", "strong-frontier pass@1 <= 50% (paid; needs approval)",
                         ">= 20 tasks per intent (spec 10.2): see the split size"],
    }
    manifest["manifest_sha256"] = manifest_sha(manifest)
    (out / "manifest.json").write_text(_jdump(manifest))
    log(f"manifest: {len(members)} of {len(held)} tasks in the sealed split; sha256 {manifest['manifest_sha256'][:16]}")
    return manifest


def verify(out_dir: Path, *, sandboxes: Callable[[str], Any] | None = None) -> list[str]:
    """Every way the built benchmark no longer matches its manifest (empty = intact): the manifest's own hash, the rows
    file, and every member's content hashes recomputed from the held-out store."""
    out = Path(out_dir)
    m = json.loads((out / "manifest.json").read_text())
    probs = []
    if manifest_sha(m) != m.get("manifest_sha256"):
        probs.append("manifest.json was edited (sha256 mismatch)")
    if m.get("tasks_jsonl_sha256") and hashlib.sha256((out / "tasks.jsonl").read_bytes()).hexdigest() != m["tasks_jsonl_sha256"]:
        probs.append("tasks.jsonl changed")
    lane = make_lane(Path(m["sources"]["heldout_store"]), "heldout", sandboxes=sandboxes)
    for row in m["tasks"]:
        if not row["in_split"]:
            continue
        rec = lane.store.window(row["id"])
        got = content_hashes(unauthored_task(lane, rec), rec)
        diff = sorted(k for k in row["hashes"] if got.get(k) != row["hashes"][k])
        if diff:
            probs.append(f"{row['id']}: {diff} changed")
    return probs


def sealed_ids(manifest: Mapping[str, Any] | Path) -> set[str]:
    """Every id a training row must never name (all candidates, in the split or not: the whole held-out store is sealed)."""
    m = json.loads(Path(manifest).read_text()) if isinstance(manifest, (str, Path)) else manifest
    return {r["id"] for r in m["tasks"]}


def publish(manifest: Mapping[str, Any], cards: Sequence[Any], *, train_cards: Sequence[Any] = (), **kw: Any) -> Any:
    """Publish the split as its own agentenv Eval (``sealing.publish_heldout``) once every member has an accepted, authored
    card; refused while any member is ``prompt_pending``."""
    from crucible.factory.sealing import SealError, SealReport, publish_heldout

    by = {c.id: c for c in cards}
    pending = [i for i in manifest["split_members"] if i not in by or not str(getattr(by[i], "prompt", "")).strip()]
    if pending:
        raise SealError(f"{len(pending)} held-out task(s) are prompt_pending: authoring comes first, under its own lease ({pending[:3]})")
    if not manifest.get("sealed"):
        raise SealError("the manifest is not sealed (a time violation kept under time_policy='report', or a partial build)")
    rep = SealReport(counts={"heldout": len(manifest["split_members"])})
    return publish_heldout([*train_cards, *cards], [c.id for c in train_cards], list(manifest["split_members"]), rep, **kw)


# ---- reporting -------------------------------------------------------------------------------------------------------------
def _boot_ci(per_task: Mapping[str, float], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
             group_of: Callable[[str], Hashable] | None = None) -> tuple[float | None, float | None, float | None]:
    """(mean, lo, hi) of per-task values with tasks (or ``group_of`` clusters) resampled."""
    if not per_task:
        return None, None, None
    units: dict[Hashable, list[float]] = {}
    for t, v in per_task.items():
        units.setdefault(group_of(t) if group_of else t, []).append(v)
    gs = list(units.values())
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        pick = [v for g in (rng.choice(gs) for _ in gs) for v in g]
        boots.append(sum(pick) / len(pick))
    boots.sort()
    mean = sum(per_task.values()) / len(per_task)
    return mean, boots[int(alpha / 2 * n_boot)], boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def report(episodes: Iterable[Any], *, category_of: Callable[[str], str], ks: Sequence[int] = (1, 2, 3, 4, 5), n_boot: int = 2000, seed: int = 0
           ) -> dict[str, Any]:
    """``METRICS_PLAN`` over ``bench.lh_metrics.LHEpisode``s, per category and overall."""
    from crucible.bench.lh_metrics import pass_hat_k, summarize

    eps = list(episodes)
    cats: dict[str, list[Any]] = {}
    for e in eps:
        cats.setdefault(category_of(e.task), []).append(e)
    cats["overall"] = eps
    out: dict[str, Any] = {}
    for cat, es in sorted(cats.items()):
        live = [e for e in es if not e.masked]
        by_task: dict[str, list[Any]] = {}
        for e in live:
            by_task.setdefault(e.task, []).append(e)
        row: dict[str, Any] = {"summary": summarize(es, k=max(ks)), "tasks": len(by_task)}
        for k in ks:
            at = {t: 1.0 - (pass_hat_k(len(v), len(v) - sum(x.passed for x in v), k) or 0.0) if len(v) >= k else None for t, v in by_task.items()}
            hat = {t: pass_hat_k(len(v), sum(x.passed for x in v), k) for t, v in by_task.items()}
            row[f"pass@{k}"] = _boot_ci({t: v for t, v in at.items() if v is not None}, n_boot=n_boot, seed=seed)
            row[f"pass^{k}"] = _boot_ci({t: v for t, v in hat.items() if v is not None}, n_boot=n_boot, seed=seed)
        phi = {t: sum(f for x in v if (f := x.final_phi) is not None) / max(1, sum(1 for x in v if x.final_phi is not None)) for t, v in by_task.items()}
        row["mean_final_phi"] = _boot_ci(phi, n_boot=n_boot, seed=seed)
        cpp = {t: sum(x.cost_usd for x in v) / s for t, v in by_task.items() if (s := sum(x.passed for x in v))}
        row["usd_per_passing_episode"] = {"by_task": cpp, "mean": (sum(cpp.values()) / len(cpp)) if cpp else None,
                                          "max_min_ratio": (max(cpp.values()) / min(cpp.values())) if cpp and min(cpp.values()) > 0 else None}
        out[cat] = row
    return out


def paired_report(base: Iterable[Any], trained: Iterable[Any], *, group_of: Callable[[str], Hashable] | None = None, **kw: Any) -> dict[str, Any]:
    """Base vs trained per task on pass rate and final Φ (``bench.lh_metrics.paired``: repo-clustered when ``group_of``)."""
    from crucible.bench.lh_metrics import paired

    b, t = list(base), list(trained)
    return {"pass": paired(b, t, metric="pass", group_of=group_of, **kw), "phi": paired(b, t, metric="phi", group_of=group_of, **kw)}


def run(cfg: BuildConfig, **kw: Any) -> dict[str, Any]:
    return asyncio.run(build(cfg, **kw))
