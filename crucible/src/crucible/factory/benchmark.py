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

**Benchmark v2** (2026-10-18; P6 sealed 1 of 11). The time rule is now a fixed **training cutoff**
(``TRAINING_CUTOFF``, 2026-07-03T00:00Z: just after the latest B of any trainable training window, pyparsing's
2026-07-02T21:45Z). A held-out window must END after it; any training source that ends after it is **embargoed**
(listed in the manifest, whose ids ``sealed_ids`` returns, so the training row writers refuse them). Organisations and
repositories are checked against **every training source** (``training_sources``: every window of every training store,
the generators' source repositories, the stitched pairs, the scouted repositories the lease let through, the training
repository list), through ``sealing.seal_sources``. The held-out repositories were chosen before mining (``BENCH_V2``,
with the excluded candidates and why) and mined by ``mine_heldout``: release windows whose B is after the cutoff and
8-unit commit runs wholly after it, newest first, ``cap`` per repository, through ``windows.validate_window`` (tolerant
ownership, the executed prober, revert mutants, recipe-3 Worlds, leak probes) with the timing-test rule
(``lanes.windows_timing``). S10 also reads each oracle grade's failing tests: a checked test whose verdict moves across
grades of one end state is load-dependent; the window is validated again without it (``revalidate``) and re-gated once.

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
GRADE_PREFIX = "crucible-bench-grade-"
WORLD_PREFIX = "crucible-lh-bench-"
VERSION = "bench-v2"
# the training cutoff: just after the latest B of any trainable training window (store3: pyparsing, 2026-07-02T21:45Z)
TRAINING_CUTOFF = "2026-07-03T00:00:00+00:00"
HELD_OUT_RUN_SIZE = 8  # qualifying units per commit run (the 108 days after the cutoff hold few 12-unit runs)

# the v2 selection, fixed before any mining (org disjointness against ``training_sources``, licence from the LICENSE file,
# pure Python, a pytest suite, >= 1 release window or 8-unit run ending after the cutoff; dry-run counts on 2026-10-18)
BENCH_V2: dict[str, Any] = {
    "selected": {
        "fsspec": "fsspec/filesystem_spec BSD (1 release + 15 runs after the cutoff)",
        "filelock": "tox-dev/filelock MIT (4 + 12)",
        "sphinx-autodoc-typehints": "tox-dev/sphinx-autodoc-typehints MIT (2 + 31)",
        "w3lib": "scrapy/w3lib BSD (0 + 9)",
        "starlette": "Kludex/starlette BSD (3 + 1)",
        "pyjwt": "jpadilla/pyjwt MIT (2 + 0)",
        "jsonpickle": "jsonpickle/jsonpickle BSD (0 + 2)",
        "peewee": "coleifer/peewee MIT (8 + 2; the optional Cython playhouse module is not built)",
        "streamlink": "streamlink/streamlink BSD (2 + 3)",
        "pydash": "dgilland/pydash MIT (1 + 6)",
        "pint": "hgrecco/pint BSD (0 + 9)",
        "dateparser": "scrapinghub/dateparser BSD (1 + 5)",
        "graphql-core": "graphql-python/graphql-core MIT (1 + 3)",
        "alembic": "sqlalchemy/alembic MIT (0 + 2)",
    },
    "kept": {"sqlparse": "andialbrecht", "tomlkit": "python-poetry (a scouted python-poetry record was refused by the lease: never a "
             "training source)", "python-dotenv": "theskumar (no window ends after the cutoff)", "voluptuous": "alecthomas",
             "mistune": "lepture"},
    "dropped": {"jinja": "pallets also owns click (training); its windows stay held out, out of the benchmark",
                # found after selection, before or while mining (they stay held out: no training role may use them)
                "peewee": "before mining: its suite is unittest modules not named test_*.py, so pytest collects nothing",
                "filelock": "while mining: its lock tests hang on the bind-mounted workspace (Docker Desktop file sharing); the "
                            "containers survive docker kill, so no window finished",
                "streamlink": "while mining: after the pytest-minversion fix its suite still ran past the 600 s sandbox timeout "
                              "(2 CPUs, single core busy) on every run of 5 windows",
                "alembic": "while mining: its tests write SQLite files into the working directory, and the grading sandbox mounts "
                           "the tree read-only (no test result at B)"},
    "reserves": ["parsel", "validators", "huey", "tenacity", "cattrs", "loguru", "nox", "litestar", "tablib", "jsonpath-ng"],
    "reserve_rule": "mined only if the selection seals fewer than 20 tasks; then the per-repository cap rises from 8 to 12",
    "excluded": {
        "org shared with a training source (a scouted consumer the lease let through)": ["PyCQA/flake8", "PyCQA/pycodestyle", "PyCQA/pyflakes",
            "PyCQA/isort", "executablebooks/markdown-it-py", "eliben/pycparser", "pydantic/pydantic-settings", "dry-python/returns",
            "tobymao/sqlglot", "PyGithub/PyGithub", "RDFLib/rdflib", "trentm/python-markdown2"],
        "the grader's pytest imports it (a pytest11 plugin or a pytest dependency: the agent's edit would reach the grader, V4)": [
            "pypa/packaging", "pytest-dev/pluggy", "pytest-dev/pytest-mock", "agronholm/anyio", "agronholm/typeguard", "joke2k/faker",
            "tox-dev/platformdirs"],
        "no release window or 8-unit run ends after the cutoff": ["python-attrs/attrs", "hynek/structlog", "encode/httpx", "carpedm20/emoji",
            "cloudpipe/cloudpickle", "hukkin/mdformat", "konradhalas/dacite", "lidatong/dataclasses-json", "mitmproxy/pdoc", "pyinvoke/invoke",
            "pytransitions/transitions", "seperman/deepdiff", "prompt-toolkit/python-prompt-toolkit", "python-jsonschema/jsonschema",
            "FactoryBoy/factory_boy", "Kludex/python-multipart", "jaraco/inflect", "kurtmckee/feedparser", "prettytable/prettytable",
            "pypa/wheel", "python-gitlab/python-gitlab", "python-lsp/python-lsp-server", "python-rope/rope", "savoirfairelinux/num2words",
            "sphinx-doc/sphinx", "sqlalchemy/mako"],
        "compiled core or a build that downloads data": ["tox-dev/pipdeptree (Rust)", "jawah/charset_normalizer (mypyc/Cython)",
            "falconry/falcon (Cython)", "python-babel/babel (CLDR download)"],
        "suite too slow for 3 + 3 reruns per window": ["sqlfluff/sqlfluff"],
    },
}

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
    project's Python sources (tests, packaging, CI, ``docs/`` and import lines left out). Measured on store3: the full
    oracle diff gave 132 code-substring hits, every one shared boilerplate (GitHub Actions steps, diff headers, Sphinx
    ``conf.py``, ``from __future__ import annotations``)."""
    return added_lines(Path(task.base_dir), task.window.get("oracle_files") or {})


def added_lines(base_dir: Path, oracle_files: Mapping[str, str | None]) -> str:
    """``solution_text`` from a window's start tree and oracle files alone (any training store: mined, rebreak,
    stitched, open), through ``difflib`` (the same ``+`` lines as the unified diff)."""
    import difflib
    import re

    from crucible.factory.lanes.executable import is_protected

    imp = re.compile(r"^\s*(import\s|from\s+\S+\s+import\s)")
    out = []
    for p, new in sorted(oracle_files.items()):
        if not (p.endswith(".py") and not is_protected(p) and not p.startswith(("docs/", "doc/"))) or not isinstance(new, str):
            continue
        f = Path(base_dir) / p
        old = f.read_text(errors="replace").splitlines() if f.is_file() else []
        for ln in difflib.unified_diff(old, new.splitlines(), lineterm="", n=0):
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


# ---- training sources (v2) -------------------------------------------------------------------------------------------------
def _split_pair(repo: str) -> list[str]:
    """A stitched record's repository ``consumer--host`` is two sources."""
    return [x for x in str(repo).split("--") if x]


def training_sources(paths: Iterable[Path | str], *, include_repo_list: bool = True) -> tuple[list[Any], list[dict[str, Any]]]:
    """Every training source under ``paths`` as ``sealing.Source`` rows, and the contamination corpus (window records of
    training stores, with their store dir). A path is read by what it is:

    - a window store (``split.json`` says ``train``): every window, trainable or not; its end is B (a generator's
      synthetic history is dated when it was built, the conservative reading);
    - a scout ``records.json`` (a list with ``type`` and ``lease``): every record the lease let through, undated;
    - a stitched ``pairs.json`` (consumer, host): both repositories, undated;
    - a directory of clones named ``owner__name``: each one, undated;
    - a JSON list of ``{"repo": ..., "end"?: ...}``: each one.

    ``include_repo_list`` adds the training repositories of ``windows.REPO_URLS`` (every name not held out). A held-out
    store is refused: it is never a training source."""
    from crucible.factory.lanes.windows import WindowStore
    from crucible.factory.sealing import Source, org_from

    out: list[Any] = []
    corpus: list[dict[str, Any]] = []

    def add(repo: str, *, end: Any = None, kind: str, origin: str, sid: str = "") -> None:
        for r in _split_pair(repo):
            out.append(Source(repo=r, org=org_from(r, REPO_URLS), end=utc(end) if end else None, kind=kind, origin=origin, id=sid))

    for raw in paths:
        p = Path(raw)
        if (p / "split.json").is_file():
            split = json.loads((p / "split.json").read_text()).get("split")
            if split != "train":
                raise ValueError(f"{p} is a {split} store: it is never a training source")
            store = WindowStore(p, split="train")
            for wid in store.window_ids():
                rec = store.window(wid)
                add(str(rec["repo"]), end=rec.get("b_date"), kind="window_store", origin=str(p), sid=str(rec["id"]))
                corpus.append({"rec": rec, "base": store.dir(wid) / "base"})
        elif p.is_dir():
            for d in sorted(p.iterdir()):
                if d.is_dir() and "__" in d.name:
                    add(d.name, kind="clone", origin=str(p))
        elif p.is_file():
            doc = json.loads(p.read_text())
            rows = doc if isinstance(doc, list) else list(doc.values()) if isinstance(doc, dict) else []
            for r in rows:
                if not isinstance(r, Mapping):
                    continue
                if "lease" in r and "type" in r:
                    if (r.get("lease") or {}).get("ok") and r.get("repo"):
                        add(str(r["repo"]), kind=f"scout:{r['type']}", origin=str(p), sid=str(r.get("id") or ""))
                elif "consumer" in r and "host" in r:
                    add(str(r["consumer"]), kind="pairs", origin=str(p))
                    add(str(r["host"]), kind="pairs", origin=str(p))
                elif r.get("repo"):
                    add(str(r["repo"]), end=r.get("end"), kind=str(r.get("kind") or "list"), origin=str(p), sid=str(r.get("id") or ""))
        else:
            raise FileNotFoundError(f"training source {p} does not exist")
    if include_repo_list:
        for name in REPO_URLS:
            if name not in HELD_OUT_REPOS:
                add(name, kind="repo_list", origin="windows.REPO_URLS")
    return out, corpus


# ---- mining the held-out windows (v2) --------------------------------------------------------------------------------------
def heldout_windows(repo_dir: Path, name: str, *, cutoff: str = TRAINING_CUTOFF, run_size: int = HELD_OUT_RUN_SIZE, min_units: int = 5,
                    max_units: int = 25, cap: int = 8) -> list[Any]:
    """The candidate windows of one held-out repository: release windows whose B is after ``cutoff`` (5-25 qualifying
    units), then ``run_size``-unit commit runs wholly after it that overlap no release window; newest first, at most
    ``cap``."""
    from crucible.factory.lanes.windows_git import _bodies, release_windows, run_windows

    cut = utc(cutoff) or ""
    rel = [w for w in release_windows(Path(repo_dir), name, since=cut[:4] + "-01-01", min_units=min_units, max_units=max_units)
           if (utc(w.b_date) or "") > cut]
    taken = {u.sha for w in rel for u in w.units}
    runs = [w for w in run_windows(Path(repo_dir), name, size=max(min_units, min(run_size, max_units)), since=cut[:10], taken=taken)
            if (utc(w.b_date) or "") > cut and all((utc(u.date) or "") > cut for u in w.units)]
    out = sorted(rel + runs, key=lambda w: utc(w.b_date) or "", reverse=True)[:cap]
    for w in out:
        _bodies(Path(repo_dir), w.units)
    return out


async def mine_heldout(repos: Mapping[str, Path], store: Any, *, cutoff: str = TRAINING_CUTOFF, cap: int = 8, run_size: int = HELD_OUT_RUN_SIZE,
                       extra: Sequence[Mapping[str, Any]] = (), images: Any = None, sandboxes: Callable[[str], Any] | None = None, reruns: int = 3,
                       parallel_windows: int = 3, deadline: float | None = None, log: Callable[[dict], None] | None = None,
                       leak_prefix: str = f"{WORLD_PREFIX}leak-", redo: Sequence[str] = ("deadline", "error")) -> dict[str, Any]:
    """Validate every ``heldout_windows`` candidate of ``repos`` (plus ``extra``: window records mined earlier whose B
    is after the cutoff, re-validated under the v2 rules; a fresh window sharing a unit with one is dropped) into the
    held-out ``store`` through ``windows.validate_window``. A window already in the store is skipped unless its stop
    reason starts with ``redo``. Returns the candidates per repository."""
    import time as _time

    from crucible.factory.lanes.windows import ImageBook, Window, sandbox_factory, validate_window

    images = images or ImageBook()
    sandboxes = sandboxes or sandbox_factory(parallel=6, name_prefix="crucible-bench-mine-")
    cut = utc(cutoff) or ""
    wins: list[Any] = []
    for rec in extra:
        if rec["repo"] in repos and (utc(rec.get("b_date")) or "") > cut:
            wins.append(Window.from_json({**rec, "repo_dir": str(repos[rec["repo"]])}))
    seen = {u.sha for w in wins for u in w.units}
    plan: dict[str, Any] = {}
    for name, path in repos.items():
        store.guard(name)
        fresh = [w for w in await asyncio.to_thread(heldout_windows, Path(path), name, cutoff=cutoff, run_size=run_size, cap=cap)
                 if not any(u.sha in seen for u in w.units)]
        wins += fresh
        plan[name] = {"fresh": [w.id for w in fresh], "extra": [w.id for w in wins if w.repo == name and w not in fresh]}
    todo = [w for w in wins if not (store.dir(w.id) / "window.json").exists() or str(store.window(w.id).get("reason", "")).startswith(tuple(redo))]
    sem = asyncio.Semaphore(parallel_windows)

    async def go(w: Any) -> None:
        async with sem:
            if deadline is not None and _time.time() > deadline:
                store.save_window({**{k: v for k, v in w.to_json().items() if k != "repo_dir"}, "split": store.split, "stage": "windows",
                                   "reason": "deadline: not validated", "lane": "window"})
                return
            try:
                await validate_window(w, store, images, sandboxes, reruns=reruns, log=log, leak_prefix=leak_prefix)
            except Exception as e:  # noqa: BLE001 - one window's failure is recorded, never the run's
                store.save_window({**{k: v for k, v in w.to_json().items() if k != "repo_dir"}, "split": store.split, "stage": "windows",
                                   "reason": f"error: {type(e).__name__}: {e}"[:500], "lane": "window"})

    await asyncio.gather(*[go(w) for w in todo])
    return {"cutoff": cut, "run_size": run_size, "cap": cap, "candidates": len(wins), "validated": len(todo), "per_repo": plan}


async def revalidate(rec: Mapping[str, Any], store: Any, repo_dir: Path, exclude: Mapping[str, str], *, images: Any = None,
                     sandboxes: Callable[[str], Any] | None = None, reruns: int = 3, log: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """Validate a held-out window again with ``exclude`` (test id -> reason, e.g. S10's load-dependent tests) never a
    check: ownership, the prober and the mutants run on what is left. The store's record is replaced."""
    from crucible.factory.lanes.windows import ImageBook, Window, sandbox_factory, validate_window

    win = Window.from_json({**rec, "repo_dir": str(repo_dir)})
    excl = {**dict(rec.get("timing_rule") or {}), **dict(exclude)}
    new = await validate_window(win, store, images or ImageBook(), sandboxes or sandbox_factory(parallel=6, name_prefix="crucible-bench-mine-"),
                                reruns=reruns, log=log, leak_prefix=f"{WORLD_PREFIX}leak-", exclude=excl)
    new["revalidated"] = {"exclude": excl, "previous_stage": rec.get("stage")}
    store.save_window(new)
    return new


async def refresh_world(rec: Mapping[str, Any], store: Any, images: Any = None, *, leak_prefix: str = f"{WORLD_PREFIX}leak-") -> dict[str, Any]:
    """The window's World under the current recipe (``windows_world.EXTRAS_FALLBACK``: A's installed copy keeps only its
    metadata, so an image shared by windows with the same packaging no longer carries another window's A). A record
    whose image tag is current is returned as it is; otherwise the image is built, the leak probes run again, and the
    record (with ``world_previous``) is saved. ``world_refresh.ok`` is False when the build or the probes fail. The
    classification (fail-to-pass, owners, prober, mutants) stands: the installed copy never served an import."""
    from crucible.factory.lanes.windows import ImageBook
    from crucible.factory.lanes.windows_s3 import new_names
    from crucible.factory.lanes.windows_world import leak_probes, project_name

    images = images or ImageBook()
    base = store.dir(str(rec["id"])) / "base"
    tag, _ = images.tag_for(str(rec["repo"]), base)
    w = dict(rec.get("world") or {})
    if w.get("image") == tag:
        return dict(rec)
    card = await images.ensure(str(rec["repo"]), base)
    new = {**rec, "world_previous": {k: w.get(k) for k in ("image", "digest", "recipe", "leak")}}
    new["world"] = {k: v for k, v in card.items() if k != "build_log"}
    if not card["ok"]:
        new["world_refresh"] = {"ok": False, "why": "the World image did not build", "tail": str(card.get("build_log", ""))[-600:]}
    else:
        leak = await leak_probes(card["image"], base, package=card.get("project") or project_name(base), version=rec.get("b_version"),
                                 new_test_names=new_names(rec, base), name_prefix=leak_prefix, installed=card.get("projects"))
        new["world"]["leak"] = leak
        new["world_refresh"] = {"ok": bool(leak.get("ok")), "why": "" if leak.get("ok") else f"leak probe failed: {leak.get('probes')}"}
    store.save_window(new)
    return new


# ---- sealing ---------------------------------------------------------------------------------------------------------------
def seal(train: Sequence[Mapping[str, Any]], heldout: Sequence[Mapping[str, Any]], *, train_prompts: Mapping[str, str] | None = None,
         diffs: Mapping[str, str] | None = None, model_cutoff: Any = None, contamination_check: bool = True, sources: Sequence[Any] | None = None,
         cutoff: Any = None, idiom_chars: int | None = None) -> dict[str, Any]:
    """``factory.sealing.seal`` over (train, held-out) window records, plus the org rule and per-task verdicts.
    ``diffs``: window id -> oracle patch (contamination); ``train_prompts``: train window id -> authored prompt.
    v2: ``sources`` (``training_sources``) adds ``sealing.seal_sources``'s organisation and repository rules against every
    training source, and ``cutoff`` replaces "after the latest training B" with "after the training cutoff" (the
    training sources after it are the embargo). ``idiom_chars`` (v2): a code-substring hit between two oracles whose
    longest shared normalised run is shorter than this (100: two of the scan's 50-character windows, about one source
    line) is an idiom, recorded per task as ``contamination_idioms`` and not blocking; None keeps P6's rule (every hit
    blocks). Measured on v2: ``return get_type_hints(obj, globalns, localns, include_extras=True)``, an alphabet literal,
    ``def __init__(self, format: Optional[str] = None``."""
    from crucible.factory.sealing import seal as base_seal
    from crucible.factory.sealing import seal_sources

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
    src = None
    if sources is not None or cutoff is not None:
        src = seal_sources(ho, list(sources or []), cutoff=utc(cutoff) if cutoff else None, urls=REPO_URLS)
        shared_orgs = sorted(set(shared_orgs) | set(src.shared_orgs))
    hit_by: dict[str, list[dict[str, Any]]] = {}
    idioms: dict[str, list[dict[str, Any]]] = {}
    for h in rep.contamination:
        tid = str(h.get("task") or h.get("bench") or h.get("id") or "")
        run = shared_run(diffs.get(tid, ""), diffs.get(str(h.get("source") or "").rsplit(":", 1)[0], "")) if idiom_chars else None
        if run is not None and h.get("kind") == "code-substring" and str(h.get("source", "")).endswith(":diff") and len(run) < idiom_chars:
            idioms.setdefault(tid, []).append({**h, "shared": run, "shared_chars": len(run)})
        else:
            hit_by.setdefault(tid, []).append(h)
    per = {}
    for r in heldout:
        d = utc(r.get("b_date"))
        sv = src.per_heldout.get(r["id"], {}) if src else {}
        time_ok = (d is not None and d > utc(cutoff)) if cutoff else (d is not None and latest[0] is not None and d > latest[0])
        per[r["id"]] = {"repo_ok": r["repo"] not in rep.shared_repos and sv.get("repo_ok", True), "org": org_of(r["repo"]),
                        "org_ok": org_of(r["repo"]) not in shared_orgs, "org_sources": sv.get("org_sources", []), "time_ok": time_ok, "b_date_utc": d,
                        "time_rule": f"B after the training cutoff {utc(cutoff)}" if cutoff else "B after the latest training B",
                        "after_model_cutoff": None if not model_cutoff else bool(d and d > utc(model_cutoff)),
                        "contamination": hit_by.get(r["id"], [])[:5], "contamination_ok": not hit_by.get(r["id"]),
                        "contamination_idioms": idioms.get(r["id"], [])[:5]}
    out = {"report": rep.to_dict(), "shared_orgs": shared_orgs, "train_orgs": sorted(train_orgs), "latest_train": {"date_utc": latest[0], "window": latest[1]},
           "per_task": per, "train_windows": len(train), "heldout_windows": len(heldout)}
    if src is not None:
        out["sources"] = src.to_dict()
        out["cutoff"] = src.cutoff
    return out


def shared_run(a: str, b: str) -> str:
    """The longest run two solution texts share after ``bench.contamination.norm_code`` (the scan's normalisation)."""
    import difflib

    from crucible.bench.contamination import norm_code

    x, y = norm_code(a), norm_code(b)
    m = difflib.SequenceMatcher(None, x, y, autojunk=False).find_longest_match(0, len(x), 0, len(y))
    return x[m.a: m.a + m.size]


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
        from crucible.factory.lanes.windows_timing import load_dependent

        patch = task.oracle_patch()
        failing: list[list[str] | None] = []

        async def one(_i: int) -> Any:
            rep = await task.grade_patch(patch, trail=[])
            ok = (rep.infra or {}).get("status") == "ok"
            failing.append(list((rep.meta or {}).get("failing_tests") or []) if ok else None)
            return {"reward": None if not ok else rep.reward, "checks": {k: v == "pass" for k, v in rep.gates.items()}, "fingerprint": sha_text(patch)}

        fl = await measure_flake(one, config=FlakeConfig(reruns=cfg.flake_reruns_per_level, concurrency=(1, cfg.concurrency)))
        # the empirical timing rule: one end state, so a checked test whose verdict moves is load-dependent
        moving = load_dependent(failing, [t for ts in task.check_tests.values() for t in ts])
        out["s10"] = {"ok": fl.ok and not moving, **fl.to_dict(), "load_dependent": moving}
        log(f"{task.task_id} s10 {out['s10']['ok']} upper {fl.upper:.4f} load-dependent {len(moving)}")
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
    sandboxes: Callable[[str], Any] | None = None  # grading sandboxes (default: Docker, ``crucible-bench-grade-*``)
    # v2: the training cutoff (None: P6's "after the latest training B"), every training source (``training_sources``
    # paths; the train store is always one), the held-out clones (``revalidate`` after S10 finds load-dependent tests)
    cutoff: str | None = None
    train_sources: tuple[Path, ...] = ()
    repo_dirs: Mapping[str, Path] = field(default_factory=dict)
    revalidate_on_load: bool = True
    selection: Mapping[str, Any] | None = None  # recorded in the manifest (``BENCH_V2``)
    task_parallel: int = 1  # tasks gated at once (each gate keeps its own concurrency levels)
    gate_sealed_only: bool = False  # v2: a task that fails sealing is excluded without running its gates
    refresh_worlds: bool = False  # v2: rebuild every candidate's World under the current recipe first (``refresh_world``)
    idiom_chars: int | None = None  # v2: code-substring hits sharing a shorter run are idioms, not blocking (``seal``)


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
    if cfg.refresh_worlds:  # every candidate's World under the current recipe (an image shared across windows held another A)
        from crucible.factory.lanes.windows import ImageBook

        book = ImageBook(parallel=3)
        held = list(await asyncio.gather(*[refresh_world(r, hl.store, book) for r in held]))
        log(f"worlds refreshed: {sum(1 for r in held if r.get('world_refresh'))}; failed: "
            f"{[r['id'] for r in held if not (r.get('world_refresh') or {}).get('ok', True)]}")
    tasks = {r["id"]: unauthored_task(hl, r) for r in held}
    v2 = cfg.cutoff is not None or bool(cfg.train_sources)
    sources: list[Any] | None = None
    corpus: list[dict[str, Any]] = []
    if v2:  # every training source: the train store (every window, trainable or not) and the other training stores and lists
        sources, corpus = training_sources([cfg.train_root, *cfg.train_sources])
    log(f"held-out candidates {len(held)}; training windows {len(train)} ({cfg.train_filter}); training sources {len(sources or [])}")
    diffs = {wid: solution_text(t) for wid, t in tasks.items()}
    for r in train:  # any training window (a non-trainable one has no milestones): its start tree and oracle files
        diffs[r["id"]] = added_lines(tl.store.dir(r["id"]) / "base", r.get("oracle_files") or {})
    train_all = list(train)
    have = {r["id"] for r in train}
    for c in corpus:  # contamination against every training store: windows whose start tree and oracle exist
        rec = c["rec"]
        if rec["id"] in have or not rec.get("oracle_files") or not Path(c["base"]).is_dir():
            continue
        have.add(rec["id"])
        diffs[rec["id"]] = added_lines(Path(c["base"]), rec["oracle_files"])
        train_all.append(rec)
    tp = {}
    if cfg.train_prompts:
        for r in train:
            f = Path(cfg.train_prompts) / r["id"] / "prompt.json"
            if f.exists():
                tp[r["id"]] = json.loads(f.read_text()).get("prompt", "")
    sealing = seal(train_all, held, train_prompts=tp, diffs=diffs, model_cutoff=cfg.model_cutoff, sources=sources, cutoff=cfg.cutoff,
                   idiom_chars=cfg.idiom_chars)
    (out / "seal.json").write_text(_jdump(sealing))
    log(f"sealing: shared repos {sealing['report']['shared_repos']}, shared orgs {sealing['shared_orgs']}, "
        f"latest training B {sealing['latest_train']}, cutoff {sealing.get('cutoff')}, contamination hits {len(sealing['report']['contamination'])}")
    sem = asyncio.Semaphore(max(1, cfg.task_parallel))

    async def one(r0: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
        async with sem:
            return await _one(r0)

    async def _one(r0: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
        r = dict(r0)
        t = tasks[r["id"]]
        hashes = content_hashes(t, r)
        key = sha_text(json.dumps(hashes, sort_keys=True), json.dumps(list(cfg.gates.gates)), str(cfg.gates.reruns))
        gfile = out / "gates" / f"{r['id']}.json"
        gates: dict[str, Any] = {}
        s_ok0, _ = sealed_ok(sealing["per_task"][r["id"]], time_policy=cfg.time_policy)
        if gfile.exists() and json.loads(gfile.read_text()).get("key") == key:
            gates = json.loads(gfile.read_text())["gates"]
            log(f"{r['id']}: gates reused")
        elif not (r.get("world_refresh") or {}).get("ok", True):
            gates = {"ok": False, "world_refresh": r["world_refresh"]}
        elif cfg.run_gates and (s_ok0 or not cfg.gate_sealed_only):
            log(f"{r['id']}: gates running")
            gates = await run_gates(t, r, cfg=cfg.gates, log=log)
            gfile.write_text(_jdump({"key": key, "gates": gates}))
        moving = (gates.get("s10") or {}).get("load_dependent") or {}
        if moving and cfg.run_gates and cfg.revalidate_on_load and r["repo"] in cfg.repo_dirs and not r.get("revalidated"):
            # the empirical timing rule: validate the window again without the load-dependent tests, then gate it once more
            log(f"{r['id']}: {len(moving)} load-dependent test(s); validating again without them")
            new = await revalidate(r, hl.store, Path(cfg.repo_dirs[r["repo"]]), moving, sandboxes=cfg.sandboxes,
                                   log=lambda ev: log(json.dumps(ev, default=str)[:300]))
            if trainable(new):
                r, t = new, unauthored_task(hl, new)
                hashes = content_hashes(t, r)
                key = sha_text(json.dumps(hashes, sort_keys=True), json.dumps(list(cfg.gates.gates)), str(cfg.gates.reruns))
                gates = await run_gates(t, r, cfg=cfg.gates, log=log)
                gfile.write_text(_jdump({"key": key, "gates": gates, "revalidated": {"load_dependent": moving}}))
            else:
                gates = {**gates, "ok": False, "revalidation": f"{new.get('stage')}: {new.get('reason')}"}
                gfile.write_text(_jdump({"key": key, "gates": gates}))
        sv = sealing["per_task"][r["id"]]
        s_ok, s_why = sealed_ok(sv, time_policy=cfg.time_policy)
        g_ok = bool(gates.get("ok"))
        excluded = [f"sealing:{w}" for w in s_why if not (w == "time_ok" and cfg.time_policy == "report")]
        excluded += [] if g_ok else ([f"gate:{g}" for g in cfg.gates.gates if not (gates.get(g) or {}).get("ok")]
                                     or ([f"revalidation:{gates['revalidation']}"] if gates.get("revalidation") else
                                         [f"world_refresh:{gates['world_refresh']['why']}"] if gates.get("world_refresh") else ["gates: not run"]))
        if not gates and not s_ok0 and cfg.gate_sealed_only:
            excluded = [f"sealing:{w}" for w in s_why if not (w == "time_ok" and cfg.time_policy == "report")] + ["gates: not run (sealing failed)"]
        in_split = s_ok and g_ok
        cls = HELD_OUT_REPOS[r["repo"]]
        row = {"id": r["id"], "repo": r["repo"], "org": org_of(r["repo"]), "horizon_class": cls, "generator": "mined_window", "intent": cfg.intent,
               "category": f"mined_window/{cls}/{cfg.intent}", "a": r.get("a"), "b": r.get("b"), "a_date": r.get("a_date"), "b_date": r.get("b_date"),
               "b_date_utc": sv["b_date_utc"], "milestones": len(t.milestones), "executed_edges": sum(len(m.get("requires") or []) for m in t.milestones),
               "image": (r.get("world") or {}).get("image"), "prompt_status": "prompt_pending", "hashes": hashes, "sealing": sv,
               "sealing_flags": s_why, "gates": {g: bool((gates.get(g) or {}).get("ok")) for g in cfg.gates.gates},
               "gate_seconds": gates.get("seconds"), "in_split": in_split, "excluded": [] if in_split else excluded,
               "timing_excluded": sorted((r.get("timing_rule") or {}).items())[:20], "revalidated": bool(r.get("revalidated"))}
        return row, (t if in_split else None)

    done = await asyncio.gather(*[one(r) for r in held])
    rows = [x[0] for x in done]
    members = [x[1] for x in done if x[1] is not None]
    task_rows = [task_row(t, template="mined_window", runtime_config={"prompt_status": "prompt_pending"}) for t in members]
    for tr in task_rows:
        tr["split"] = SPLIT
        tr["args"]["ref"]["split"] = SPLIT
    tr_info = write_task_rows(task_rows, out / "tasks.jsonl", allow_heldout=True) if task_rows else {"rows": 0, "sha256": None}
    partial = bool(cfg.only) or not cfg.run_gates
    src = sealing.get("sources") or {}
    embargo = list(src.get("embargo") or [])
    if v2:
        (out / "embargo.json").write_text(_jdump(embargo))
    from crucible.factory.lanes.windows import WindowStore, funnel

    manifest: dict[str, Any] = {
        "benchmark": "crucible-lh-heldout", "version": VERSION, "split": SPLIT, "built_utc": utc(dt.datetime.now(dt.UTC)),
        "sealed": all(not x["sealing_flags"] for x in rows if x["in_split"]) and not partial, "partial": partial,
        "policy": {"time_policy": cfg.time_policy, "train_filter": cfg.train_filter, "model_cutoff": cfg.model_cutoff,
                   "training_cutoff": utc(cfg.cutoff) if cfg.cutoff else None,
                   "rules": ["disjoint repositories (every training source)", "disjoint organisations (every training source)",
                             f"time: B after the training cutoff {utc(cfg.cutoff)} (UTC); training sources after it are embargoed"
                             if cfg.cutoff else "time split (UTC, B after every training B)", "no contamination hit",
                             "timing tests are never checks (lanes.windows_timing: static rule at mining, load-dependent at S10)"],
                   "contamination_scope": "the lines each oracle adds to Python sources (tests, packaging, CI, docs, imports "
                                          "excluded) vs the same for training windows plus their authored prompts",
                   "contamination_idiom_chars": cfg.idiom_chars,
                   "gates": list(cfg.gates.gates), "s3_reruns": cfg.gates.reruns, "s10_reruns_per_level": cfg.gates.flake_reruns_per_level},
        "sources": {"heldout_store": str(cfg.heldout_root), "train_store": str(cfg.train_root), "train_windows": len(train),
                    "train_prompts": len(tp), "heldout_candidates": len(held), "train_sources": [str(x) for x in cfg.train_sources],
                    "contamination_corpus_windows": len(train_all), "training_sources": src.get("counts")},
        "training_embargo": [e["id"] for e in embargo], "embargo_sha256": sha_text(_jdump(embargo)) if v2 else None,
        "selection": dict(cfg.selection) if cfg.selection else None,
        "heldout_funnel": funnel(WindowStore(Path(cfg.heldout_root), split="heldout"))["total"],
        "sealing_summary": {"shared_repos": sealing["report"]["shared_repos"], "shared_orgs": sealing["shared_orgs"], "latest_train": sealing["latest_train"],
                            "training_cutoff": sealing.get("cutoff"), "source_shared_orgs": src.get("shared_orgs"), "embargoed_sources": len(embargo),
                            "heldout_not_after_cutoff": len(src.get("heldout_not_after_cutoff") or []),
                            "time_violations": len(sealing["report"]["time_violations"]), "contamination_hits": len(sealing["report"]["contamination"]),
                            "cutoff_violations": len(sealing["report"]["cutoff_violations"])},
        "tasks": rows, "split_members": [t.task_id for t in members], "tasks_jsonl_sha256": tr_info.get("sha256"),
        "metrics_plan": METRICS_PLAN,
        "prompt_authoring": {"status": "pending", "rule": "held-out prompts are written later, under sealing, by a dedicated author lease; no model "
                             "authored any held-out text (P6, v2)", "lint": "S5's fairness lint and the S7 ledger run after authoring"},
        "not_verified": ["S5 fairness lint and the S7 prompt ledger (prompt_pending)", "S9 second verifier and S11-S14 (no prompt, no solver spend)",
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
    """Every id a training row must never name: all candidates, in the split or not (the whole held-out store is
    sealed), and (v2) every embargoed training source (it ends after the training cutoff)."""
    m = json.loads(Path(manifest).read_text()) if isinstance(manifest, (str, Path)) else manifest
    return {r["id"] for r in m["tasks"]} | {str(x) for x in m.get("training_embargo") or []}


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


def report(episodes: Iterable[Any], *, category_of: Callable[[str], str], ks: Sequence[int] = (1, 2, 3, 4, 5), n_boot: int = 2000, seed: int = 0,
           policy: str = "") -> dict[str, Any]:
    """``METRICS_PLAN`` over ``bench.lh_metrics.LHEpisode``s, per category and overall. ``policy``: the policy id the
    episodes came from; a dev-only student backend (``factory.student``: ``...@mlx``) raises ``DevOnlyNumerics``."""
    from crucible.bench.lh_metrics import pass_hat_k, summarize
    from crucible.factory.student import assert_countable

    assert_countable(policy, "benchmark report")
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


def paired_report(base: Iterable[Any], trained: Iterable[Any], *, group_of: Callable[[str], Hashable] | None = None, policies: Sequence[str] = (),
                  **kw: Any) -> dict[str, Any]:
    """Base vs trained per task on pass rate and final Φ (``bench.lh_metrics.paired``: repo-clustered when ``group_of``).
    ``policies``: the two arms' policy ids; a dev-only student backend raises ``DevOnlyNumerics`` (``factory.student``)."""
    from crucible.bench.lh_metrics import paired
    from crucible.factory.student import assert_countable

    for p in policies:
        assert_countable(p, "paired benchmark report")
    b, t = list(base), list(trained)
    return {"pass": paired(b, t, metric="pass", group_of=group_of, **kw), "phi": paired(b, t, metric="phi", group_of=group_of, **kw)}


def run(cfg: BuildConfig, **kw: Any) -> dict[str, Any]:
    return asyncio.run(build(cfg, **kw))
