"""Running a sealed benchmark (``factory.benchseal``) as one arm: the before/after protocol's fresh baseline and, later,
the trained arm (swarm-pipeline/14 §4; ``scripts/deltaai/bench.py`` is the CLI).

A sealed benchmark is a manifest of hashes and seeds, not the tasks. This module checks that the tasks on disk are the
ones that were sealed, packs the least data an off-machine run needs, and turns a finished arm into the numbers that
count. It holds no held-out content and writes none into git (every path it writes is the caller's, under
``crucible/data``).

- ``verify(manifest, debug=...)``: the manifest's own sha256, and per task the sha256 of its window record, oracle,
  hidden test ids and report recomputed from the store (``benchseal.task_hashes``). Any mismatch is a refusal: the
  benchmark changed after it was sealed.
- ``measured_harness(targeting, ids, tool_limit=...)``: the harness ids of the student measurements the keep rule read,
  at the lane's budget (``generators.debug.ENV_BUDGET``), per task id (a hardened ``-d1`` / ``-h1`` / ``-s1`` variant
  also counts its parent's). ``harness_problems`` refuses an eval harness that is not that single id.
- ``bundle(manifest, debug, out, clones=...)``: the debug lane's data for the sealed tasks only (their candidate rows,
  authored prompts, gate records, store windows and the held-out repositories' clones). No training data, no other
  held-out task: the bundle is what a compute node reads.
- ``world_images(manifest, debug)``: each task's World image and the digest its window recorded.
- ``save_episodes`` / ``load_episodes``: ``bench.lh_metrics.LHEpisode`` rows as JSONL, so a later
  ``benchmark.paired_report`` reads this arm without re-running it.
- ``arm_summary(...)``: ``benchmark.report`` (task-resampled bootstrap CIs) plus pass@1 (mean over k), pass@k, the
  per-task pass rate, masked and host-failed episodes, stop reasons and the GPU-hours (``gpu_hours`` over ``sacct``).
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# a hardened variant of a mined task (``<parent>-d1``, ``-h1``, ``-s1``): the keep rule's earlier measurements of the
# parent ran under the same harness and count for it
VARIANT = re.compile(r"-[a-z]\d+$")


def load_manifest(manifest: Mapping[str, Any] | Path | str) -> dict[str, Any]:
    return dict(manifest) if isinstance(manifest, Mapping) else json.loads(Path(manifest).read_text())


def task_dir(manifest: Mapping[str, Any], tid: str, debug: Path | None = None) -> Path:
    """The task's store directory: the sealed path, or the same task under ``debug`` (a bundle on another machine)."""
    return Path(debug) / "gen" / "store" / "heldout" / "tasks" / tid if debug is not None else Path(manifest["tasks"][tid]["store"])


def prompt_file(tid: str, debug: Path, sealed: Path | None = None) -> Path | None:
    """The report the seal hashed (``lh_debug.py seal``: the sealed prompts dir first, then the lane's ``prompts/``)."""
    for p in ([Path(sealed) / "prompts" / f"{tid}.json"] if sealed else []) + [Path(debug) / "prompts" / f"{tid}.json"]:
        if p.is_file():
            return p
    return None


def verify(manifest: Mapping[str, Any] | Path | str, *, debug: Path, sealed: Path | None = None, relocated: bool = False) -> dict[str, Any]:
    """Every reason the tasks on disk are not the sealed ones (``problems`` empty: they are). ``relocated``: the stores
    are read under ``debug`` (a bundle), not at the sealed paths."""
    from crucible.factory.benchseal import _sha, task_hashes

    m = load_manifest(manifest)
    problems: list[str] = []
    want = m.get("manifest_sha256")
    got = _sha({x: y for x, y in m.items() if x != "manifest_sha256"})
    if want != got:
        problems.append(f"manifest sha256 {got[:12]} is not the sealed {str(want)[:12]}")
    rows = {}
    for tid, t in sorted(m["tasks"].items()):
        d = task_dir(m, tid, debug if relocated else None)
        if not d.is_dir():
            problems.append(f"{tid}: no store directory at {d}")
            continue
        h = task_hashes(d, prompt_file(tid, debug, sealed))
        bad = sorted(k for k, v in (t.get("hashes") or {}).items() if h.get(k) != v)
        if bad:
            problems.append(f"{tid}: {', '.join(bad)} changed since the seal")
        rows[tid] = {"ok": not bad, "seed": t.get("seed"), "n_hidden_tests": h.get("n_hidden_tests")}
    return {"name": m.get("name"), "k": m.get("k"), "tasks": rows, "problems": problems, "ok": not problems}


def parent_of(tid: str) -> str:
    return VARIANT.sub("", tid)


def measured_harness(targeting: Path, ids: Iterable[str], *, tool_limit: int, arm: str = "student") -> dict[str, list[str]]:
    """harness id -> ``<phase>/<task>`` of every student measurement of ``ids`` (or their parents) at ``tool_limit``."""
    want = {i for t in ids for i in (t, parent_of(t))}
    out: dict[str, list[str]] = {}
    for f in sorted(Path(targeting).glob(f"*/arms/{arm}/*/metrics.json")):
        tid = f.parent.name
        if tid not in want:
            continue
        try:
            s = json.loads(f.read_text()).get("settings") or {}
        except json.JSONDecodeError:
            continue
        if s.get("tool_limit") == tool_limit and s.get("harness"):
            out.setdefault(str(s["harness"]), []).append(f"{f.parents[3].name}/{tid}")
    return out


def measured_sampling(targeting: Path, ids: Iterable[str], *, tool_limit: int, arm: str = "student") -> dict[str, Any]:
    """How the keep rule's student measurements of ``ids`` (or their parents) at ``tool_limit`` sampled: the counts of
    each ``max_tokens`` and backend (a run that predates the ``backend`` field ran on Tinker). An arm records it beside its
    own settings, so a difference (e.g. 8192 tokens per reply on our vLLM against 4096 on Tinker) is stated, never silent."""
    want = {i for t in ids for i in (t, parent_of(t))}
    toks: Counter[str] = Counter()
    backs: Counter[str] = Counter()
    for f in sorted(Path(targeting).glob(f"*/arms/{arm}/*/metrics.json")):
        if f.parent.name not in want:
            continue
        try:
            s = json.loads(f.read_text()).get("settings") or {}
        except json.JSONDecodeError:
            continue
        if s.get("tool_limit") == tool_limit:
            toks[str(s.get("max_tokens"))] += 1
            backs[str(s.get("backend") or "tinker")] += 1
    return {"runs": sum(toks.values()), "max_tokens": dict(sorted(toks.items())), "backend": dict(sorted(backs.items())), "tool_limit": tool_limit}


def harness_problems(eval_ids: Mapping[str, str], measured: Mapping[str, Sequence[str]]) -> list[str]:
    """``eval_ids``: task -> the harness id this run builds. Refused unless the measurements used one id and every task's
    eval harness is it."""
    if not measured:
        return ["no student measurement at the lane's budget: nothing to match the eval harness against"]
    if len(measured) > 1:
        return [f"the measurements used {len(measured)} harness ids ({', '.join(sorted(measured))}): which one the benchmark was measured with is ambiguous"]
    (want,) = measured
    return [f"{t}: eval harness {h} is not the measured {want}" for t, h in sorted(eval_ids.items()) if h != want]


def world_images(manifest: Mapping[str, Any], *, debug: Path | None = None) -> dict[str, dict[str, Any]]:
    """image -> {"digest": the window's recorded image id, "tasks": [...]} over the sealed tasks."""
    out: dict[str, dict[str, Any]] = {}
    for tid in sorted(manifest["tasks"]):
        d = task_dir(manifest, tid, debug)
        wf = d / "window.json" if (d / "window.json").is_file() else d / "record.json"
        w = (json.loads(wf.read_text()).get("world") or {})
        row = out.setdefault(str(w.get("image")), {"digest": w.get("digest"), "tasks": []})
        if row["digest"] != w.get("digest"):
            row["digest_conflict"] = True
        row["tasks"].append(tid)
    return out


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():  # a re-bundle replaces the tree (copytree cannot overwrite symlinks)
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst)


def bundle(manifest: Mapping[str, Any], *, debug: Path, out: Path, clones: Mapping[str, Path], sealed: Path | None = None) -> dict[str, Any]:
    """The debug lane's data dir for the sealed tasks only, at ``out`` (see the module doc). ``clones``: repository name (a
    candidate row's ``repo``) -> its clone. The manifest goes along as ``out/manifest.json``."""
    debug, out = Path(debug), Path(out)
    ids = sorted(manifest["tasks"])
    rows = [r for r in json.loads((debug / "candidates.json").read_text()) if r["id"] in set(ids)]
    missing = sorted(set(ids) - {r["id"] for r in rows})
    if missing:
        raise ValueError(f"not candidates: {missing}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.json").write_text(json.dumps(rows, indent=1, sort_keys=True))
    (out / "funnel.jsonl").touch()
    files = []
    for tid in ids:
        for rel in (f"authored/{tid}.json", f"gates/{tid}.json", f"prompts/{tid}.json"):
            if (debug / rel).is_file():
                _copy(debug / rel, out / rel)
                files.append(rel)
        _copy(task_dir(manifest, tid), out / "gen" / "store" / "heldout" / "tasks" / tid)
        p = prompt_file(tid, debug, sealed)
        if p is not None and not (out / "prompts" / f"{tid}.json").exists():
            _copy(p, out / "prompts" / f"{tid}.json")
    split = debug / "gen" / "store" / "heldout" / "split.json"
    if split.is_file():
        _copy(split, out / "gen" / "store" / "heldout" / "split.json")
    repos = sorted({r["repo"] for r in rows})
    absent = [r for r in repos if r not in clones]
    if absent:
        raise ValueError(f"no clone for {absent}")
    for r in repos:
        src = Path(clones[r])
        _copy(src, out / "src" / src.name)
        lane = debug / "gen" / "clones" / r  # the lane's working clone: its own objects (fetched fixes) + alternates into src/
        if lane.is_dir():
            _copy(lane, out / "gen" / "clones" / r)
            alt = out / "gen" / "clones" / r / ".git" / "objects" / "info" / "alternates"
            if alt.is_file():  # relative to the objects dir, so the bundle works wherever it lands
                alt.write_text(f"../../../../../src/{src.name}/.git/objects\n")
    (out / "manifest.json").write_text(json.dumps(dict(manifest), indent=1, sort_keys=True))
    return {"tasks": ids, "repos": repos, "files": len(files), "out": str(out)}


# ---- the arm's episodes ----------------------------------------------------------------------------------------------------
def save_episodes(path: Path, episodes: Iterable[Any]) -> int:
    rows = [dataclasses.asdict(e) for e in episodes]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return len(rows)


def load_episodes(path: Path | Iterable[Path]) -> list[Any]:
    """``LHEpisode``s from one ``lh_episodes.jsonl`` or several (an arm's tasks), for ``report`` / ``paired_report``."""
    from crucible.bench.lh_metrics import LHEpisode

    paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]
    out = []
    for p in paths:
        for ln in p.read_text().splitlines():
            if ln.strip():
                d = json.loads(ln)
                d["series"] = [tuple(x) for x in d.get("series") or []]
                d["claims"] = [tuple(x) for x in d.get("claims") or []]
                out.append(LHEpisode(**d))
    return out


def host_failed(row: Mapping[str, Any]) -> bool:
    """A masked episode the host lost: its failure kind's blame (``core.schema.run.DEFAULT_BLAME``) is ``infra`` (the
    World, the grader's container, the sampler service), or it has no kind at all."""
    from crucible.core.schema.run import default_blame

    f = row.get("failure")
    return bool(row.get("masked")) and (not f or default_blame(str(f)) == "infra")


def gpu_hours(sacct: str) -> dict[str, Any]:
    """GPU-hours from ``sacct -X -n -P -o JobID,Elapsed,AllocTRES,State``: per job the elapsed time times its GPUs, and
    the billed hours (``billing=`` over 1,000 per GPU-hour, DeltaAI's unit) when the TRES string carries them."""
    jobs, tot, bill = [], 0.0, 0.0
    for ln in sacct.splitlines():
        parts = ln.strip().split("|")
        if len(parts) < 3 or not parts[0]:
            continue
        el = parts[1]
        days, _, hms = el.rpartition("-")
        h, m, s = (int(x) for x in hms.split(":")) if hms.count(":") == 2 else (0, *(int(x) for x in hms.split(":")))
        hours = int(days or 0) * 24 + h + m / 60 + s / 3600
        tres = dict(x.split("=", 1) for x in parts[2].split(",") if "=" in x)
        g = int(tres.get("gres/gpu", "0") or 0)
        b = float(tres.get("billing", "0") or 0) / 1000.0
        jobs.append({"job": parts[0], "hours": round(hours, 4), "gpus": g, "gpu_hours": round(hours * g, 4), "billed_gpu_hours": round(hours * b, 4) if b else None,
                     "state": parts[3] if len(parts) > 3 else None})
        tot += hours * g
        bill += hours * b
    return {"jobs": jobs, "gpu_hours": round(tot, 3), "billed_gpu_hours": round(bill, 3) if bill else None}


def arm_summary(episodes: Sequence[Any], rows: Sequence[Mapping[str, Any]], *, k: int, policy: str, category_of: Callable[[str], str] | None = None,
                n_boot: int = 2000, seed: int = 0, gpu: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The arm's numbers (module doc). ``episodes``: its ``LHEpisode``s; ``rows``: its ``episodes.jsonl`` rows (stop and
    failure per attempt). Ids stay in ``per_task``; ``aggregate`` holds numbers only."""
    from crucible.factory.benchmark import report

    rep = report(episodes, category_of=category_of or (lambda _t: "all"), ks=tuple(sorted({1, k})), n_boot=n_boot, seed=seed, policy=policy)
    ov = rep["overall"]
    per: dict[str, dict[str, Any]] = {}
    for e in episodes:
        r = per.setdefault(e.task, {"n": 0, "passed": 0, "masked": 0})
        r["n"] += 1
        r["masked"] += int(e.masked)
        r["passed"] += int(e.passed and not e.masked)
    for r in per.values():
        live = r["n"] - r["masked"]
        r["pass_rate"] = r["passed"] / live if live else None
    stops = Counter(str(r.get("stop") or "none") for r in rows)
    agg = {"tasks": len(per), "episodes": len(episodes), "masked": sum(r["masked"] for r in per.values()), "host_failed": sum(host_failed(r) for r in rows),
           "pass@1": ov["pass@1"], f"pass@{k}": ov[f"pass@{k}"], "mean_final_phi": ov["mean_final_phi"], "stops": dict(sorted(stops.items())),
           "tasks_short_of_k": sum(1 for r in per.values() if r["n"] - r["masked"] < k)}
    if gpu is not None:
        agg["gpu"] = dict(gpu)
    return {"policy": policy, "k": k, "aggregate": agg, "per_task": per, "report": rep}
