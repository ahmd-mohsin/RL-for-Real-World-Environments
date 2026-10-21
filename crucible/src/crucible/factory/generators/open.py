"""Open tasks, Type B (swarm-pipeline/00 §2, §7.4, §15 P5): make a real repository faster under its full test suite. No
ceiling: the reward keeps rising past the best known version.

**Sources** (``perf_units``): first-parent commits whose message says faster, speed up, performance, optimise, quadratic
or slow, that change source ``.py`` files and nothing but prose and tests (``real_perf`` records). The task starts at the
commit's parent A; the commit itself (P) is the known faster version.

**The metric** is the runtime of the repository's *own* code paths on **fresh inputs**:
1. ``discover`` (executed at A, in the World image): the repository's tests run under a profiler plugin (``TRACE_PLUGIN``)
   that records every outermost call from a test into the project whose dynamic extent runs a function P changed (an
   *entry*), with its arguments when they are plain data (numbers, strings, lists, tuples, dicts, sets, bytes). Entries are
   module-level functions, so any implementation can be called the same way.
2. The workload (``BENCH``) calls each entry on inputs generated from a per-grade seed: the recorded arguments with every
   digit redrawn, numbers rescaled, lists resampled and grown, multi-line text repeated. A grade never reuses an input, so
   memoisation and replay cannot pay (the ``memo``, ``replay`` floors prove it); lazy results are materialised inside the
   timed loop.
3. Each grade runs, in one ``--network none`` container on one CPU, the start tree (A, read-only at ``/ref``) and the
   candidate (at ``/w``) as separate processes, alternating, ``rounds`` times on the same fresh inputs; no other container
   this process started runs meanwhile (``QUIET``: timing exclusive, regression tests shared). The baseline is
   re-measured in every grade. Quiet host: the start tree's coefficient of variation across rounds must stay under
   ``max_cv``, else the grade is re-measured once, then masked (infrastructure, never a 0, V8).

**Reward** (correctness gate, then log-speedup, uncapped): the candidate must pass every test of A's suite that passes at
A and P (the regression check) and return the start tree's results on every fresh input (exceptions by type); then, with
``s`` the geometric-mean speedup over rounds and ``lo`` its lower 95% bound, reward = ``log(s) / log(s_bar)`` when
``lo > MIN_LO`` = 1.05 (else 0; an unchanged tree once measured lo 1.008 against itself), where ``s_bar`` is P's speedup measured at build. Milestones: ``m_correct``; ``m_faster`` (``lo > 1.05``;
gated by ``m_correct``); ``m_bar`` (``s >= sqrt(s_bar)``, half the known headroom in log terms; gated by ``m_faster``).
Phi weighs ``m_faster`` and ``m_bar`` 0.5 each with their gates, transitively; ``m_correct`` weighs 0 (the start state is correct:
correctness gates the progress, it is not progress).

**Headroom.** P must speed the workload up significantly at build (``lo_bar > 1 + min_gain``): *headroom proven*. A
workload P does not speed up is ``headroom_unproven`` and never gate-eligible.

**Long-horizon.** An open task's milestones are *gated* (``gated_by``: ``m_faster`` counts only on a correct end state),
not executed dependencies, so they carry no ``requires`` edge: ``longtask.check_longtask`` labels the task short, and the
funnel reports S6 as not applicable (``open_no_ceiling``), never as passed. Open tasks are long because they have no
ceiling (``00`` §2).

**Simplify** (``simplify('baseline_bar')``): the bar at the baseline: ``m_bar`` = ``m_faster`` (any significant speedup is
a full pass).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from crucible.factory.generators.base import WORLD_PREFIX, Candidate, Funnel, changed_paths, git, out
from crucible.factory.lanes.executable import build_tree, is_protected, is_test_path, sha256_text
from crucible.factory.lanes.windows import (
    ImageBook,
    WindowStore,
    _module_file,
    changed_defs,
    normalize_ids,
    solution_files,
    suite_targets,
)
from crucible.factory.lanes.windows_git import RepoGit, Unit, _bodies, first_parent_units
from crucible.factory.lanes.windows_world import docker
from crucible.factory.longtask import GroundCase, LongTaskRef, status_claim_patch
from crucible.factory.pathcheck import PathRules, ToolCall, check_paths, find_trail
from crucible.factory.sources.lease import EvalLease
from crucible.factory.sources.records import SourceRecord

NAME = "open"
PERF_RX = re.compile(r"\b(perf(ormance)?|speed\s?ups?|speed(s|ing)? up|faster|optimi[sz](e[ds]?|ation)|quadratic|linear[- ]time|slow(ness|er)?|"
                     r"inefficien\w*|O\(n)\b", re.IGNORECASE)
NOT_PERF_RX = re.compile(r"\b(import time|ci|tox|tests?|coverage|docs?|build|install|wheel|mypy|typing|lint)\b", re.IGNORECASE)
PROSE = re.compile(r"\.(md|rst|txt)$|(^|/)(docs?|changes|changelog|history|news)(/|$)|(^|/)(CHANGES|CHANGELOG|HISTORY|NEWS)[^/]*$", re.IGNORECASE)
BENCH_PREFIX = f"{WORLD_PREFIX}bench-"
TRACE_PREFIX = f"{WORLD_PREFIX}trace-"
VERIFIER_RECIPE = "open-grader/p5.1"
# the smallest speedup that counts: the lower 95% bound must clear 5%. An unchanged tree measured against itself reached
# lo 1.008 once (padded's claims-only floor, P5 run): significance alone (lo > 1) lets noise pay
MIN_LO = 1.05

TRACE_PLUGIN = r'''
import json, os, sys

ROOTS = set(filter(None, os.environ.get("P5_ROOTS", "").split(",")))
HOT = [tuple(x) for x in json.loads(os.environ.get("P5_HOT", "[]"))]
MAXS = 40
_calls = {}
_active = []


def _enc(x, depth=0):
    if depth > 6:
        raise TypeError("deep")
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            raise TypeError("nonfinite")
        return x
    if isinstance(x, bytes):
        return {"__b__": x.decode("latin-1")}
    if isinstance(x, list):
        return [_enc(e, depth + 1) for e in x]
    if isinstance(x, tuple):
        return {"__t__": [_enc(e, depth + 1) for e in x]}
    if isinstance(x, (set, frozenset)):
        return {"__s__": sorted((_enc(e, depth + 1) for e in x), key=repr)}
    if isinstance(x, dict) and all(isinstance(k, str) for k in x):
        return {"__d__": {k: _enc(v, depth + 1) for k, v in x.items()}}
    raise TypeError(type(x).__name__)


def _snapshot(frame):
    co = frame.f_code
    names = co.co_varnames
    n, k = co.co_argcount, co.co_kwonlyargcount
    loc = frame.f_locals
    args = [loc[x] for x in names[:n]]
    kw = {x: loc[x] for x in names[n:n + k]}
    i = n + k
    if co.co_flags & 0x04:
        args += list(loc[names[i]])
        i += 1
    if co.co_flags & 0x08:
        kw.update(loc[names[i]])
    return _enc([args, kw])


def _is_test(frame):
    g = frame.f_globals.get("__name__", "")
    f = frame.f_code.co_filename
    return g.split(".")[0] not in ROOTS and ("/test" in f or "test_" in f.rsplit("/", 1)[-1])


def _hot(co):
    q = getattr(co, "co_qualname", co.co_name)
    for path, name in HOT:
        if co.co_filename.endswith("/" + path) and (q == name or q.endswith("." + name) or co.co_name == name.rsplit(".", 1)[-1]):
            return True
    return False


def _prof(frame, event, arg):
    if event == "call":
        g = frame.f_globals.get("__name__", "")
        if g.split(".")[0] not in ROOTS:
            return
        if not _active:
            if frame.f_back is None or not _is_test(frame.f_back):
                return
            q = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
            try:
                snap = _snapshot(frame) if ("." not in q and "<" not in q) else None
            except Exception:
                snap = None
            _active.append([frame, g, q, snap, False])
        if _hot(frame.f_code):
            _active[-1][4] = True
    elif event == "return" and _active and _active[-1][0] is frame:
        fr, g, q, snap, hot = _active.pop()
        if hot:
            d = _calls.setdefault(g + ":" + q, {"module": g, "function": q, "hot": 0, "samples": [], "unsimple": 0, "method": "." in q or "<" in q})
            d["hot"] += 1
            if snap is None:
                d["unsimple"] += 1
            else:
                s = json.dumps(snap, sort_keys=True)
                seen = d.setdefault("_seen", set())
                if s not in seen:
                    seen.add(s)
                    if len(d["samples"]) < MAXS:
                        d["samples"].append(snap)


def pytest_configure(config):
    sys.setprofile(_prof)


def pytest_sessionfinish(session, exitstatus):
    sys.setprofile(None)
    for d in _calls.values():
        d.pop("_seen", None)
    os.write(1, b"\n@@P5TRACE@@" + json.dumps(_calls).encode() + b"@@P5END@@\n")
'''

BENCH = r'''
import hashlib, importlib, itertools, json, math, random, re, sys, time, types

SPEC = json.load(open(sys.argv[1]))
SEED = int(sys.argv[2])
LIMIT = 10000


def dec(x):
    if isinstance(x, list):
        return [dec(e) for e in x]
    if isinstance(x, dict):
        if "__t__" in x:
            return tuple(dec(e) for e in x["__t__"])
        if "__s__" in x:
            return set(dec(e) for e in x["__s__"])
        if "__b__" in x:
            return x["__b__"].encode("latin-1")
        if "__d__" in x:
            return {k: dec(v) for k, v in x["__d__"].items()}
    return x


def mut(x, rng, scale):
    if x is None or isinstance(x, bool):
        return x
    if isinstance(x, int):
        if abs(x) < 2:
            return x
        d = len(str(abs(x)))
        return (1 if x > 0 else -1) * rng.randint(10 ** (d - 1), 10 ** d - 1)
    if isinstance(x, float):
        return x if x == 0 else x * rng.uniform(0.5, 2.0)
    if isinstance(x, str):
        s = "".join(rng.choice("0123456789") if c.isdigit() else c for c in x)
        if "\n" in x and scale > 1:
            s = "\n".join(s for _ in range(rng.randint(1, scale)))
        return s
    if isinstance(x, bytes):
        return bytes(rng.randrange(48, 58) if 48 <= b < 58 else b for b in x)
    if isinstance(x, list):
        base = [mut(e, rng, scale) for e in x]
        if x and scale > 1:
            base += [mut(rng.choice(x), rng, scale) for _ in range(len(x) * (rng.randint(1, scale) - 1))]
        return base
    if isinstance(x, tuple):
        return tuple(mut(e, rng, scale) for e in x)
    if isinstance(x, set):
        return set(mut(e, rng, scale) for e in x)
    if isinstance(x, dict):
        return {k: mut(v, rng, scale) for k, v in x.items()}
    return x


def materialize(r):
    if isinstance(r, (types.GeneratorType, map, filter, zip, range, itertools.chain, itertools.islice)) or (
            hasattr(r, "__next__") and hasattr(r, "__iter__") and not isinstance(r, (str, bytes, list, tuple, dict, set))):
        return list(itertools.islice(r, LIMIT))
    return r


def canon(r, depth=0):
    if depth > 8:
        return "..."
    if isinstance(r, float):
        return repr(r)
    if r is None or isinstance(r, (bool, int, str)):
        return r
    if isinstance(r, bytes):
        return ["b", r.decode("latin-1")]
    if isinstance(r, (list, tuple)):
        return [type(r).__name__, [canon(e, depth + 1) for e in r]]
    if isinstance(r, (set, frozenset)):
        return ["set", sorted(json.dumps(canon(e, depth + 1), sort_keys=True) for e in r)]
    if isinstance(r, dict):
        return ["dict", [[json.dumps(canon(k, depth + 1), sort_keys=True), canon(v, depth + 1)] for k, v in r.items()]]
    return ["obj", type(r).__name__, re.sub(r" at 0x[0-9a-fA-F]+", "", repr(r))[:2000]]


def inputs(entry, seed):
    rng = random.Random(f"{seed}:{entry['module']}:{entry['function']}")
    samples = [dec(s) for s in entry["samples"]]
    sc = entry.get("scale", 1)
    outp = []
    for _ in range(entry["n"]):
        a, k = rng.choice(samples)
        outp.append(([mut(e, rng, sc) for e in a], {n: mut(v, rng, sc) for n, v in k.items()}))
    return outp


def main():
    total, hashes, raised = 0.0, [], 0
    for entry in SPEC["entries"]:
        fn = getattr(importlib.import_module(entry["module"]), entry["function"])
        warm = inputs(entry, SEED + 7919)[: max(3, entry["n"] // 20)]
        for a, k in warm:
            try:
                materialize(fn(*a, **k))
            except Exception:
                pass
        data = inputs(entry, SEED)
        res = []
        t0 = time.perf_counter()
        for a, k in data:
            try:
                res.append(materialize(fn(*a, **k)))
            except Exception as e:
                res.append(("__raise__", type(e).__name__))
        total += time.perf_counter() - t0
        for r in res:
            if isinstance(r, tuple) and len(r) == 2 and r[0] == "__raise__":
                raised += 1
                hashes.append("raise:" + r[1])
            else:
                hashes.append(hashlib.sha256(json.dumps(canon(r), sort_keys=True, default=str).encode()).hexdigest()[:16])
    sys.stdout.write("@@P5BENCH@@" + json.dumps({"t": total, "hashes": hashes, "raised": raised}) + "@@END@@\n")


main()
'''

DRIVER = r'''
import json, os, subprocess, sys, time
spec, seed, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
which = sys.argv[4].split(",")
paths = {"ref": "/ref/src:/ref", "cand": "/w/src:/w"}
out = {k: [] for k in which}
first = {}
for r in range(rounds):
    order = which if r % 2 == 0 else list(reversed(which))
    for k in order:
        p = subprocess.run([sys.executable, "/p5/bench.py", spec, seed], capture_output=True, text=True, cwd="/tmp", timeout=600,
                           env={**os.environ, "PYTHONPATH": paths[k], "PYTHONHASHSEED": "0"})
        txt = p.stdout
        if "@@P5BENCH@@" not in txt:
            out[k].append({"error": (p.stderr or txt)[-1500:]})
            continue
        d = json.loads(txt.split("@@P5BENCH@@", 1)[1].split("@@END@@", 1)[0])
        if k not in first:
            first[k] = d
        out[k].append({"t": d["t"], "raised": d["raised"]})
res = {"rounds": out, "hashes": {k: v["hashes"] for k, v in first.items()}}
sys.stdout.write("@@P5DRV@@" + json.dumps(res) + "@@END@@\n")
'''


# ---- sources ----------------------------------------------------------------------------------------------------------------------
def perf_units(repo_dir: Path, *, since: str, head: str = "HEAD") -> list[Unit]:
    """First-parent units since ``since`` whose message says faster / performance / speed up / optimise / quadratic and
    that change source ``.py`` (tests and prose allowed, nothing else)."""
    start = out(git(repo_dir, "rev-list", "--first-parent", "-1", f"--before={since}", head, check=False))
    if not start:
        start = out(git(repo_dir, "rev-list", "--max-parents=0", head)).split()[-1]
    units = first_parent_units(repo_dir, start, head)
    _bodies(repo_dir, units)
    keep = []
    for u in units:
        text = f"{u.subject}\n{u.body}"
        if not PERF_RX.search(text) or NOT_PERF_RX.search(u.subject) or not u.src_files:
            continue
        if all(PROSE.search(p) or (p.endswith(".py") and (is_test_path(p) or not is_protected(p))) for p in u.files):
            keep.append(u)
    return keep


def hot_functions(repo: RepoGit, u: Unit) -> list[tuple[str, str]]:
    """``(source path, qualified name)`` of every function or method the perf commit changed."""
    outp = []
    for p in u.src_files:
        for q in changed_defs(repo.show(u.parent, p), repo.show(u.sha, p)):
            outp.append((p.split("/", 1)[1] if p.startswith("src/") else p, q))
    return outp


def _docker_run(args: Sequence[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return docker(*args, timeout=timeout)
    except subprocess.TimeoutExpired:
        name = next((args[i + 1] for i, a in enumerate(args) if a == "--name"), None)
        if name:
            docker("rm", "-f", name, timeout=60)
        return subprocess.CompletedProcess(list(args), 124, "", "timeout")


async def discover(image: str, tree: Path, roots: Sequence[str], hot: Sequence[tuple[str, str]], targets: Sequence[str], *,
                   timeout: float = 900.0) -> dict[str, Any]:
    """The entries of the workload (see the module doc), executed in ``image`` on ``tree`` (A)."""
    with tempfile.TemporaryDirectory(prefix="p5tr-") as td:
        plug = Path(td)
        (plug / "p5trace.py").write_text(TRACE_PLUGIN)
        name = f"{TRACE_PREFIX}{uuid.uuid4().hex[:10]}"
        args = ["run", "--rm", "--name", name, "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=256m", "--memory", "2g", "--cpus", "2",
                "-v", f"{Path(tree).resolve()}:/w:ro", "-v", f"{plug.resolve()}:/p5:ro", "-w", "/w", "-e", "PYTHONPATH=/w/src:/w:/p5", "-e", "HOME=/tmp",
                "-e", f"P5_ROOTS={','.join(roots)}", "-e", f"P5_HOT={json.dumps([list(h) for h in hot])}", image, "python", "-m", "pytest", "-p", "p5trace",
                "-q", "-p", "no:cacheprovider", "-o", "addopts=", "--timeout=120", *targets]
        r = await asyncio.to_thread(_docker_run, args, timeout)
    m = re.search(r"@@P5TRACE@@(.*?)@@P5END@@", r.stdout, re.DOTALL)
    if not m:
        return {"ok": False, "why": f"no trace (exit {r.returncode}): {(r.stderr or r.stdout)[-600:]}", "entries": {}}
    return {"ok": True, "entries": json.loads(m.group(1))}


def speedup_stats(ref: Sequence[float], cand: Sequence[float]) -> dict[str, float]:
    """Geometric-mean speedup over paired rounds with a 95% t-interval on the log ratios, and the baseline's CV. From 7
    rounds on, the lowest and the highest log ratio are trimmed first: one host hiccup in one round otherwise decides the
    verdict (P5: 1 of 60 oracle grades missed its bar on a single slow round)."""
    logs = [math.log(r / c) for r, c in zip(ref, cand, strict=False) if r > 0 and c > 0]
    if len(logs) >= 7:
        logs = sorted(logs)[1:-1]
    n = len(logs)
    if n == 0:
        return {"s": 1.0, "lo": 0.0, "hi": math.inf, "n": 0, "cv_ref": math.inf}
    mu = statistics.fmean(logs)
    sd = statistics.stdev(logs) if n > 1 else math.inf
    tcrit = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}.get(n - 1, 2.0)
    half = tcrit * sd / math.sqrt(n) if n > 1 else math.inf
    cv = statistics.stdev(ref) / statistics.fmean(ref) if len(ref) > 1 and statistics.fmean(ref) > 0 else math.inf
    return {"s": math.exp(mu), "lo": math.exp(mu - half) if math.isfinite(half) else 0.0, "hi": math.exp(mu + half) if math.isfinite(half) else math.inf,
            "n": n, "cv_ref": cv}


class QuietLock:
    """The quiet-host rule inside one process: a timing run holds the lock alone (``exclusive``); the other container
    work of open tasks (regression tests, traces) holds it ``shared``, so grades can run concurrently while no timing
    ever overlaps another container this process started."""

    def __init__(self) -> None:
        self._cond: asyncio.Condition | None = None
        self.readers = 0
        self.writer = False

    def _c(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    @contextlib.asynccontextmanager
    async def shared(self) -> Any:
        c = self._c()
        async with c:
            await c.wait_for(lambda: not self.writer)
            self.readers += 1
        try:
            yield
        finally:
            async with c:
                self.readers -= 1
                c.notify_all()

    @contextlib.asynccontextmanager
    async def exclusive(self) -> Any:
        c = self._c()
        async with c:
            await c.wait_for(lambda: not self.writer and self.readers == 0)
            self.writer = True
        try:
            yield
        finally:
            async with c:
                self.writer = False
                c.notify_all()


QUIET = QuietLock()



async def measure(image: str, ref_tree: Path, cand_tree: Path, spec: Mapping[str, Any], *, seed: int, rounds: int = 7, which: str = "ref,cand",
                  timeout: float = 900.0, cpus: str = "1") -> dict[str, Any]:
    """One timing run (see the module doc): ``{ok, s, lo, hi, cv_ref, equal, mismatches, raised, rounds}``."""
    with tempfile.TemporaryDirectory(prefix="p5b-") as td:
        d = Path(td)
        (d / "bench.py").write_text(BENCH)
        (d / "driver.py").write_text(DRIVER)
        (d / "spec.json").write_text(json.dumps(spec))
        name = f"{BENCH_PREFIX}{uuid.uuid4().hex[:10]}"
        args = ["run", "--rm", "--name", name, "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=256m", "--memory", "1g", "--cpus", cpus,
                "-v", f"{Path(ref_tree).resolve()}:/ref:ro", "-v", f"{Path(cand_tree).resolve()}:/w:ro", "-v", f"{d.resolve()}:/p5:ro", "-e", "HOME=/tmp",
                image, "python", "/p5/driver.py", "/p5/spec.json", str(seed), str(rounds), which]
        async with QUIET.exclusive():
            r = await asyncio.to_thread(_docker_run, args, timeout)
    m = re.search(r"@@P5DRV@@(.*?)@@END@@", r.stdout, re.DOTALL)
    if not m:
        return {"ok": False, "why": f"no bench result (exit {r.returncode}): {(r.stderr or r.stdout)[-500:]}"}
    res = json.loads(m.group(1))
    rs = res["rounds"]
    errs = {k: [x["error"] for x in v if "error" in x][:1] for k, v in rs.items()}
    outp: dict[str, Any] = {"ok": True, "rounds": {k: [round(x.get("t", -1), 5) for x in v] for k, v in rs.items()}, "errors": {k: v for k, v in errs.items() if v}}
    hs = res.get("hashes") or {}
    if "ref" in hs and "cand" in hs:
        mism = [i for i, (a, b) in enumerate(zip(hs["ref"], hs["cand"], strict=False)) if a != b] + ([-1] if len(hs["ref"]) != len(hs["cand"]) else [])
        outp.update(equal=not mism, mismatches=len(mism), n_inputs=len(hs["ref"]), raised_ref=sum(1 for h in hs["ref"] if h.startswith("raise:")))
    elif "ref" in hs:
        outp.update(n_inputs=len(hs["ref"]), raised_ref=sum(1 for h in hs["ref"] if h.startswith("raise:")))
    if "ref" in rs and "cand" in rs:
        ref = [x["t"] for x in rs["ref"] if "t" in x]
        cand = [x["t"] for x in rs["cand"] if "t" in x]
        outp.update(speedup_stats(ref, cand))
        outp["ok"] = bool(ref) and bool(cand) and not outp["errors"].get("ref")
    return outp


# ---- the task -----------------------------------------------------------------------------------------------------------------------
def patch_from_files(base_dir: Path, files: Mapping[str, str | None]) -> str:
    """``files`` over ``base_dir`` as the git patch a workspace harness collects."""
    env = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "c", "GIT_AUTHOR_EMAIL": "c@c", "GIT_COMMITTER_NAME": "c", "GIT_COMMITTER_EMAIL": "c@c",
           "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
    with tempfile.TemporaryDirectory(prefix="p5d-") as td:
        t = Path(td) / "t"
        shutil.copytree(base_dir, t, symlinks=True)
        env.update(HOME=td, GIT_CEILING_DIRECTORIES=td)

        def g(*a: str) -> str:
            return subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false", *a], cwd=t, env=env, capture_output=True,
                                  check=True).stdout.decode("utf-8", "replace")

        g("init", "-q")
        g("add", "-A", "-f")
        g("commit", "-q", "--allow-empty", "-m", "base")
        for p, text in files.items():
            f = t / p
            if text is None:
                f.unlink(missing_ok=True)
            else:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(text)
        g("add", "-A", "-f")
        return g("-c", "core.quotepath=off", "diff", "--cached", "--binary", "HEAD")


class OpenTask:
    """One open task (a ``factory.longtask.LongTask``): A's tree, the workload spec, the regression tests, the bar."""

    generator = NAME
    horizon = "H2"

    def __init__(self, gen: OpenGen, rec: Mapping[str, Any], *, prompt: str = "", requirements: Sequence[Mapping[str, Any]] = (),
                 bar: str = "headroom", rounds: int = 9, max_cv: float = 0.25):
        self.gen, self.rec = gen, dict(rec)
        self.task_id = str(rec["id"])
        self.id = self.task_id
        self.base_dir = Path(gen.store.dir(self.task_id)) / "base"
        self.image = rec["world"]["image"]
        self.instruction = prompt
        self.requirements = [dict(r) for r in requirements] or [
            {"id": "R1", "milestone": "m_correct", "text": "(unauthored) everything that works today keeps working, with the same results", "kind": "behaviour"},
            {"id": "R2", "milestone": "m_faster", "text": "(unauthored) the hot code paths run measurably faster", "kind": "behaviour"},
            {"id": "R3", "milestone": "m_bar", "text": "(unauthored) at least half of the known possible speedup", "kind": "behaviour"}]
        self.bar = bar
        self.s_bar = float(rec[NAME]["s_bar"])
        self.rounds, self.max_cv = rounds, max_cv
        self.milestones = [
            {"id": "m_correct", "tests": list(rec["regression"]) + ["bench:equal"], "requires": []},
            {"id": "m_faster", "tests": ["bench:lo>1"], "requires": [], "gated_by": ["m_correct"]},
            {"id": "m_bar", "tests": ["bench:bar"], "requires": [], "gated_by": ["m_faster"]}]
        roots = tuple(sorted(rec.get("roots") or []))
        self.path_rules = PathRules(project_names=tuple(sorted({rec["repo"], *roots})))
        self.oracle = [{"function_name": "apply_files", "arguments": {"files": dict(rec["oracle_files"])}}]
        self.provenance = [str(s) for s in rec.get("sources") or []] or [f"obs-{self.task_id}"]
        self.verifier_version = sha256_text(VERIFIER_RECIPE, json.dumps(rec[NAME]["spec"], sort_keys=True), str(self.s_bar), bar,
                                            json.dumps(rec["regression"]))[:16]
        self.world_prefix = f"{WORLD_PREFIX}world-"

    # -- LongTask members ------------------------------------------------------------------------------------------------------------
    @property
    def world(self) -> dict[str, Any]:
        return {"kind": "repo_open", "repos": [self.rec["repo"]], "image": self.image, "digest": self.rec["world"].get("digest"),
                "base_dir": str(self.base_dir), "start": self.rec["a"], "network": "none", "history": "cut"}

    @property
    def prompt(self) -> str:
        return self.instruction

    def bar_value(self) -> float:
        return 1.0 if self.bar == "baseline" else math.sqrt(self.s_bar)

    def oracle_patch(self) -> str:
        if not hasattr(self, "_oracle_patch"):
            self._oracle_patch = patch_from_files(self.base_dir, self.rec["oracle_files"])
        return self._oracle_patch

    def _files(self, solution: Any) -> tuple[dict[str, str | None], list[str], list[str]]:
        if isinstance(solution, list) and solution and all(isinstance(s, Mapping) and s.get("function_name") == "apply_files" for s in solution):
            files: dict[str, str | None] = {}
            for s in solution:
                files.update((s.get("arguments") or {}).get("files") or {})
            return {p: t for p, t in files.items() if not is_protected(p)}, sorted(p for p in files if is_protected(p)), []
        return solution_files(self.base_dir, solution)

    async def _regression(self, tree: Path) -> dict[str, str]:
        sb = self.gen.sandboxes(self.image)
        files = sorted({t.split("::", 1)[0] for t in self.rec["regression"]})
        async with QUIET.shared():
            tr = await sb.run(tree, files)
        if tr.infra != "ok":
            raise RuntimeError(f"masked: {tr.why}")
        return normalize_ids(tr.results, files)

    async def evaluate(self, solution: Any, *, seed: int | None = None) -> dict[str, Any]:
        """The end state: regression tests, the fresh-input differential and the timing (re-measured once when the host is
        noisy; masked when it stays noisy)."""
        files, protected, errors = await asyncio.to_thread(self._files, solution)
        if errors:
            return {"infra": "ok", "apply_errors": errors, "milestones": dict.fromkeys(("m_correct", "m_faster", "m_bar"), False)}
        seed = seed if seed is not None else int.from_bytes(uuid.uuid4().bytes[:4], "big")
        with tempfile.TemporaryDirectory(prefix="p5o-") as td:
            tree = Path(td) / "t"
            await asyncio.to_thread(build_tree, self.base_dir, tree, solution={"files": files})
            try:
                st = await self._regression(tree)
            except RuntimeError as e:
                return {"infra": "masked", "why": str(e)}
            failing = sorted(t for t in self.rec["regression"] if st.get(t) != "passed")
            m: dict[str, Any] = {}
            for attempt in range(2):
                m = await measure(self.image, self.base_dir, tree, self.rec[NAME]["spec"], seed=seed + attempt, rounds=self.rounds)
                if m.get("ok") and m.get("cv_ref", math.inf) <= self.max_cv:
                    break
        if not m.get("ok"):
            if (m.get("errors") or {}).get("cand") and not (m.get("errors") or {}).get("ref"):
                m.update(equal=False, s=0.0, lo=0.0, hi=0.0, cv_ref=0.0)  # the candidate crashed the workload: incorrect, not infra
            else:
                return {"infra": "masked", "why": m.get("why") or str(m.get("errors"))[:300]}
        if m.get("cv_ref", math.inf) > self.max_cv:
            return {"infra": "masked", "why": f"noisy host: baseline CV {m.get('cv_ref'):.3f} > {self.max_cv}", "bench": m}
        correct = not failing and bool(m.get("equal"))
        faster = correct and m["lo"] > MIN_LO
        bar = faster and m["s"] >= self.bar_value()
        return {"infra": "ok", "milestones": {"m_correct": correct, "m_faster": faster, "m_bar": bar}, "failing": failing[:30], "bench": m,
                "protected_writes": protected, "seed": seed}

    WEIGHTS: ClassVar[dict[str, float]] = {"m_correct": 0.0, "m_faster": 0.5, "m_bar": 0.5}  # correctness is a gate, not progress: the start state is correct

    def phi(self, ms: Mapping[str, bool]) -> float:
        """Weighted milestones passing with every milestone they need, transitively (speed never counts without
        correctness)."""
        by = {m["id"]: m for m in self.milestones}

        def held(i: str) -> bool:
            return bool(ms.get(i)) and all(held(r) for r in [*by[i]["requires"], *by[i].get("gated_by", [])])

        return round(sum(self.WEIGHTS[i] for i in by if held(i)), 6)

    def reward_of(self, ev: Mapping[str, Any]) -> float:
        ms = ev.get("milestones") or {}
        if not ms.get("m_faster"):
            return 0.0
        s = float(ev["bench"]["s"])
        return round(math.log(s) / math.log(self.s_bar), 6) if self.s_bar > 1 else 0.0

    async def grade_patch(self, patch: Any, workdir: Path | None = None, *, trail: Sequence[Any] | None = None,
                          network_events: Sequence[str] | None = None) -> Any:
        """``GradeReport``: reward = log(s)/log(s_bar) on a correct, significantly faster end state (uncapped), else 0; a hack
        path violation zeroes it; meta carries the milestones, Phi, the timing and ``reward_capped`` (min(reward, 1))."""
        from crucible.verify.runtime.plan import CheckResult, GradeReport

        t0 = time.monotonic()
        ev = await self.evaluate(patch)
        if ev.get("infra") != "ok":
            return GradeReport(task=self.task_id, tier="full", checks=[], gates={}, reward=0.0, outcome=0.0, infra={"status": "masked", "why": ev.get("why")},
                               trusted=False, elapsed_s=time.monotonic() - t0, meta={"why": ev.get("why")})
        ms = ev["milestones"]
        if trail is None:
            calls, source = find_trail(workdir)
        else:
            calls = [t if isinstance(t, ToolCall) else ToolCall(str(t.get("function") or t.get("function_name") or ""), t.get("arguments")) for t in trail]
            source = "given"
        pr = check_paths(self.path_rules, patch=patch if isinstance(patch, str) else "", trail=calls, trail_source=source, network_events=network_events)
        reward = 0.0 if (pr.hack_suspect or ev.get("apply_errors")) else self.reward_of(ev)
        full = bool(ms.get("m_bar")) and pr.ok
        phi = self.phi(ms)
        gates = {m: ("pass" if ok else "fail") for m, ok in ms.items()} | pr.gates()
        checks = [CheckResult(id=m, gate=m, status=("pass" if ok else "fail"), detail={}) for m, ok in ms.items()]
        meta = {"milestones": ms, "phi": phi, "full_pass": full, "outcome_reward": reward, "reward_capped": min(reward, 1.0), "bench": ev.get("bench"),
                "failing_tests": ev.get("failing"), "path": pr.to_json(), "hack_suspect": pr.hack_suspect, "apply_errors": ev.get("apply_errors") or [],
                "reward_rule": "0 unless correct and lo>1.05; else log(s)/log(s_bar), uncapped; 0 if hack_suspect", "s_bar": self.s_bar, "seed": ev.get("seed"),
                # M7 R6: the milestones (correct, faster, at the bar) are outcomes; the path checker's checks are the stated
                # rules (as ``lanes/windows.py`` declares them), so a correct below-bar patch keeps its per-turn credit
                "kinds": {m: "outcome" for m in ms} | {g: "gate" for g in pr.gates()}}
        return GradeReport(task=self.task_id, tier="full", checks=checks, gates=gates, reward=reward, outcome=reward, infra={"status": "ok"}, trusted=True,
                           elapsed_s=time.monotonic() - t0, meta=meta)

    async def milestone_status(self, patch: str) -> dict[str, bool]:
        ev = await self.evaluate(patch)
        if ev.get("infra") != "ok":
            raise RuntimeError(f"milestone status unmeasured: {ev.get('why')}")
        return dict(ev["milestones"])

    def entry_files(self) -> dict[str, tuple[str, str]]:
        """Entry function -> (module source path, its text at A)."""
        outp = {}
        for e in self.rec[NAME]["spec"]["entries"]:
            f = _module_file(self.base_dir, e["module"])
            if f is not None:
                outp[e["function"]] = (f.relative_to(self.base_dir).as_posix(), f.read_text())
        return outp

    def _wrap(self, kind: str) -> dict[str, str]:
        files: dict[str, str] = {}
        for fn, (path, text) in self.entry_files().items():
            cur = files.get(path, text)
            if kind == "memo":
                add = (f"\n\n_p5_orig_{fn} = {fn}\n_p5_cache_{fn} = {{}}\n\n\ndef {fn}(*args, **kwargs):\n    key = repr((args, sorted(kwargs.items())))\n"
                       f"    if key not in _p5_cache_{fn}:\n        _p5_cache_{fn}[key] = _p5_orig_{fn}(*args, **kwargs)\n    return _p5_cache_{fn}[key]\n")
            elif kind == "replay":
                samples = json.dumps(next(e for e in self.rec[NAME]["spec"]["entries"] if e["function"] == fn)["samples"])
                add = (f"\n\n_p5_orig_{fn} = {fn}\n_p5_known_{fn} = {{}}\n\n\ndef _p5_dec_{fn}(x):\n    if isinstance(x, list):\n        return [_p5_dec_{fn}(e) for e in x]\n"
                       f"    if isinstance(x, dict):\n        if '__t__' in x:\n            return tuple(_p5_dec_{fn}(e) for e in x['__t__'])\n"
                       f"        if '__d__' in x:\n            return {{k: _p5_dec_{fn}(v) for k, v in x['__d__'].items()}}\n    return x\n\n\n"
                       f"for _a, _k in [_p5_dec_{fn}(s) for s in __import__('json').loads({samples!r})]:\n    try:\n"
                       f"        _p5_known_{fn}[repr((_a, sorted(_k.items())))] = _p5_orig_{fn}(*_a, **_k)\n    except Exception:\n        pass\n\n\n"
                       f"def {fn}(*args, **kwargs):\n    key = repr((list(args), sorted(kwargs.items())))\n    if key in _p5_known_{fn}:\n        return _p5_known_{fn}[key]\n"
                       f"    return _p5_orig_{fn}(*args, **kwargs)\n")
            else:  # fast_wrong
                add = f"\n\ndef {fn}(*args, **kwargs):\n    return None\n"
            files[path] = cur + add
        return files

    def floors(self) -> list[GroundCase]:
        """Score 0: the start state, a STATUS-only claim, memoisation and replay of the recorded inputs (fresh inputs: no
        hit), a fast but wrong entry (the differential fails)."""
        return [GroundCase("no_op", "floor", "", note="the start state"), GroundCase("claims_only", "floor", status_claim_patch(self)),
                GroundCase("memo", "floor", patch_from_files(self.base_dir, self._wrap("memo")), note="results cached by argument"),
                GroundCase("replay", "floor", patch_from_files(self.base_dir, self._wrap("replay")), note="the recorded inputs' results precomputed")]

    def mutants(self) -> list[Any]:
        from crucible.factory.lanes.windows import MilestoneMutant

        return [MilestoneMutant(id="fast_wrong", family="open", solution=patch_from_files(self.base_dir, self._wrap("fast_wrong")),
                                description="every entry returns None at once", milestone="m_correct"),
                MilestoneMutant(id="revert_perf", family="commit_revert", solution="", description="the perf commit reverted (the start)",
                                milestone="m_faster")]

    def alternatives(self) -> list[GroundCase]:
        return [GroundCase("oracle_files", "alternative", self.oracle, note="P as files")]

    def simplify(self, kind: str, **kw: Any) -> OpenTask:
        if kind != "baseline_bar":
            raise ValueError(f"an open task simplifies by baseline_bar, not {kind!r}")
        return OpenTask(self.gen, self.rec, prompt=self.instruction, requirements=self.requirements, bar="baseline", rounds=self.rounds, max_cv=self.max_cv)

    def card(self, *, intent: str = "unassigned", recipe_version: str = "p5", status: str = "draft", **_: Any) -> Any:
        from crucible.factory.cards import RequirementLedger, TaskCard, VerifierCard

        if not self.instruction.strip():
            raise ValueError("an unauthored open task has no card (no prompt)")
        return TaskCard(id=self.task_id, family=f"open-{self.rec['repo']}", intent=intent,
                        surface={"domain": "python library", "world": self.rec["repo"], "tools": "shell workspace", "data_shape": "repository",
                                 "prompt_style": "performance request"},
                        prompt=self.instruction, ledger=RequirementLedger(), world_ref=f"image:{self.image}@{self.rec['world'].get('digest')}",
                        checkset_ref="sha256:" + sha256_text(json.dumps(self.rec[NAME]["spec"], sort_keys=True)), oracle_ref=f"git:{self.rec['repo']}@{self.rec['b']}",
                        verifier=VerifierCard(verifier_version=self.verifier_version),
                        stats={"split": "train", "s_bar": self.s_bar, "headroom": self.rec[NAME].get("headroom")},
                        lineage={"generator": NAME, "perf_commit": self.rec["b"], "a": self.rec["a"], "bar": self.bar}, recipe_version=recipe_version, status=status)

    def ref(self) -> LongTaskRef:
        return LongTaskRef(lane=NAME, task_id=self.task_id, source_id=self.task_id, store=str(self.gen.store.root), split="train",
                           prompt_sha256=sha256_text(self.instruction), checkset_sha256=sha256_text(json.dumps(self.rec[NAME]["spec"], sort_keys=True)),
                           verifier_version=self.verifier_version, simplified={"kind": "baseline_bar"} if self.bar == "baseline" else {})

    async def open(self, workdir: Path) -> Any:
        from crucible.factory.lanes.windows_world import open_world

        return await open_world(image=self.image, base_dir=self.base_dir, workdir=Path(workdir), instruction=self.instruction, name_prefix=self.world_prefix)


# ---- gates -----------------------------------------------------------------------------------------------------------------------
async def open_gates(task: OpenTask, *, flake_reruns: int = 30, s3_reruns: int = 60, levels: tuple[int, ...] = (1, 2), stop_on_fail: bool = True,
                     s3_sandboxes: Any = None, gates_run: tuple[str, ...] = ("S5", "S8", "S3", "S10"), s3_prefix: str = f"{WORLD_PREFIX}s3-") -> dict[str, Any]:
    """S6 (n/a: no executed edge by design), S5 (oracle: correct, faster, at the bar; null agent 0), S8 (floors 0 with Phi
    0, mutants fail their milestone, alternative full pass), S3 (``windows_s3.world_s3`` with no milestone), S10 (oracle
    full pass in ``flake_reruns`` grades at each level; timing runs are serialised, so a level is the grading
    concurrency of everything but the timing). ``gates_run``: a subset (Phase A runs S3 inside the World fix's
    acceptance); ``ok`` covers the gates run."""
    from crucible.factory.flake import FlakeConfig, measure_flake
    from crucible.factory.lanes.windows_s3 import world_s3
    from crucible.factory.longtask import null_agent_score

    gates: dict[str, Any] = {"S6": {"ok": None, "findings": ["n/a: open tasks have gate edges, not executed dependencies (no ceiling, 00 §2)"],
                                    "evidence": {"executed_edges": 0, "long_kind": "open_no_ceiling"}}}
    failed = False

    async def run(name: str, fn: Callable[[], Any]) -> None:
        nonlocal failed
        if failed and stop_on_fail:
            gates[name] = {"ok": None, "findings": ["not_run: an earlier gate failed"]}
            return
        t0 = time.monotonic()
        try:
            res = await fn()
        except Exception as e:  # noqa: BLE001
            res = {"ok": False, "findings": [f"error: {type(e).__name__}: {e}"[:400]]}
        res["seconds"] = round(time.monotonic() - t0, 1)
        gates[name] = res
        failed = failed or not res["ok"]

    async def s5() -> dict[str, Any]:
        rep = await task.grade_patch(task.oracle_patch(), trail=[])
        m = rep.meta or {}
        null = await null_agent_score(task)
        ok = rep.infra.get("status") == "ok" and bool(m.get("full_pass")) and m.get("phi") == 1.0 and null["ok"]
        return {"ok": ok, "findings": [] if ok else [f"oracle reward {rep.reward} phi {m.get('phi')} full {m.get('full_pass')}; null {null['failed']}"],
                "evidence": {"oracle_reward": rep.reward, "oracle_s": (m.get("bench") or {}).get("s"), "null": null["rows"]}}

    async def s8() -> dict[str, Any]:
        rows, bad = {}, []
        for c in task.floors():
            r = await task.grade_patch(c.solution, trail=[])
            m = r.meta or {}
            ok = r.infra.get("status") == "ok" and r.reward == 0.0 and (m.get("phi") or 0.0) == 0.0
            rows[c.id] = {"reward": r.reward, "phi": m.get("phi"), "s": (m.get("bench") or {}).get("s"), "lo": (m.get("bench") or {}).get("lo"),
                          "equal": (m.get("bench") or {}).get("equal"), "ok": ok}
            bad += [] if ok else [c.id]
        for mu in task.mutants():
            r = await task.grade_patch(mu.solution, trail=[])
            ms = (r.meta or {}).get("milestones") or {}
            ok = r.infra.get("status") == "ok" and ms.get(mu.milestone) is False and not (r.meta or {}).get("full_pass")
            rows[mu.id] = {"reward": r.reward, "milestones": ms, "ok": ok}
            bad += [] if ok else [mu.id]
        for c in task.alternatives():
            r = await task.grade_patch(c.solution, trail=[])
            ok = r.infra.get("status") == "ok" and bool((r.meta or {}).get("full_pass"))
            rows[c.id] = {"reward": r.reward, "ok": ok}
            bad += [] if ok else [c.id]
        return {"ok": not bad, "findings": [f"ground truth fails: {x}" for x in bad], "evidence": {"rows": rows}}

    async def s3() -> dict[str, Any]:
        rec = {**task.rec, "milestones": [], "tests_b": {}}
        r = await world_s3(rec, task.base_dir, task.image, reruns=s3_reruns, sandboxes=s3_sandboxes, name_prefix=s3_prefix)
        return {"ok": r["ok"], "findings": [r["why"]] if r.get("why") else [], "evidence": {"checks": r["checks"], "reruns": r.get("reruns")}}

    async def s10() -> dict[str, Any]:
        patch = task.oracle_patch()
        rep = await measure_flake(lambda i: _full(task, patch), config=FlakeConfig(reruns=flake_reruns, concurrency=levels))
        d = rep.to_dict()
        return {"ok": rep.ok, "findings": d["reasons"], "evidence": {k: d[k] for k in ("levels", "n", "false_negatives", "masked", "upper")}}

    for name, fn in (("S5", s5), ("S8", s8), ("S3", s3), ("S10", s10)):
        if name in gates_run:
            await run(name, fn)
    order = tuple(g for g in ("S5", "S8", "S3", "S10") if g in gates_run)
    return {"task": task.task_id, "generator": NAME, "gates": gates, "ok": all(gates[g]["ok"] for g in order), "passed": [g for g in order if gates[g]["ok"]],
            "long_kind": "open_no_ceiling"}


async def _full(task: OpenTask, patch: str) -> Any:
    """One oracle grade as a flake draw: the verdict is the full pass (the bar), reward 1 when met."""
    from crucible.factory.floors import Graded

    rep = await task.grade_patch(patch, trail=[])
    if rep.infra.get("status") != "ok":
        return Graded(None)
    return Graded(1.0 if (rep.meta or {}).get("full_pass") else 0.0, {k: bool(v) for k, v in ((rep.meta or {}).get("milestones") or {}).items()},
                  sha256_text(patch)[:16])


# ---- the generator ---------------------------------------------------------------------------------------------------------------
class OpenGen:
    """Open tasks from real perf commits of local training clones (``Generator``)."""

    name = NAME

    def __init__(self, root: Path, clones: Mapping[str, Path], *, funnel: Funnel, lease: EvalLease, images: ImageBook | None = None, sandboxes: Any = None,
                 log: Callable[[dict], None] | None = None, since_years: int = 8, min_gain: float = 0.05, per_repo: int = 3, rounds_build: int = 9,
                 target_seconds: float = 0.4):
        from crucible.factory.generators.base import GRADE_PREFIX
        from crucible.factory.lanes.windows import sandbox_factory

        self.root = Path(root)
        self.funnel, self.lease, self.log = funnel, lease, log
        self.images = images or ImageBook()
        self.sandboxes = sandboxes or sandbox_factory(parallel=4, memory="1536m", name_prefix=GRADE_PREFIX)
        self.clones = {}
        for n, p in clones.items():
            lease.guard_repo(n)
            self.clones[n] = Path(p)
        self.since_years, self.min_gain, self.per_repo, self.rounds_build, self.target_seconds = since_years, min_gain, per_repo, rounds_build, target_seconds
        self.store = WindowStore(self.root / "store" / "train", split="train")
        self.records: list[SourceRecord] = []

    def task(self, record_id: str, *, authored: Mapping[str, Any] | None = None) -> OpenTask:
        rec = self.store.window(record_id)
        if authored:
            return OpenTask(self, rec, prompt=authored["prompt"], requirements=authored["requirements"])
        return OpenTask(self, rec)

    async def candidates(self) -> list[Candidate]:
        outp: list[Candidate] = []
        for name, repo_dir in self.clones.items():
            head = out(git(repo_dir, "rev-parse", "HEAD"))
            year = int(out(git(repo_dir, "log", "-1", "--format=%aI", head))[:4])
            units = await asyncio.to_thread(perf_units, repo_dir, since=f"{year - self.since_years}-01-01", head=head)
            for u in units:
                rec = SourceRecord("real_perf", name, sha=u.sha, title=u.subject, text=u.body[:1500], created=u.date, evidence={"executed": False})
                self.lease.take(rec)
                self.records.append(rec)
                cid = f"op-{name}-{u.sha[:8]}"
                self.funnel.note(NAME, cid, "S1_source", rec.lease["ok"], rec.lease.get("reason", ""), subject=u.subject[:120])
                if rec.lease["ok"]:
                    outp.append(Candidate(cid, NAME, name, sources=[rec.to_json()], params={"sha": u.sha, "parent": u.parent, "date": u.date,
                                                                                             "subject": u.subject}))
        return outp

    async def build(self, cand: Candidate) -> dict[str, Any]:
        """A at the perf commit's parent; World; entries discovered at A; workload sized and P's speedup measured (headroom);
        the regression tests (A's suite passing at A and at P)."""
        self.lease.guard_repo(cand.repo)
        d = self.store.dir(cand.id)
        if (d / "window.json").exists():
            return self.store.window(cand.id)
        repo_dir = self.clones[cand.repo]
        repo = RepoGit(repo_dir)
        sha, parent = cand.params["sha"], cand.params["parent"]
        rec: dict[str, Any] = {"id": cand.id, "repo": cand.repo, "lane": NAME, "split": "train", "a": parent, "b": sha, "b_date": cand.params["date"],
                               "stage": "source", "reason": "", "sources": [s["id"] for s in cand.sources]}

        def stop(stage: str, reason: str, **ev: Any) -> dict[str, Any]:
            rec.update(stage=stage, reason=reason)
            rec.setdefault(NAME, {}).update(ev)
            self.store.save_window(rec)
            return rec

        base = d / "base"
        if not base.exists():
            ok, bad = await asyncio.to_thread(repo.build, parent, [], base)
            if not ok:
                self.funnel.note(NAME, cand.id, "S3_world", False, f"A does not check out ({bad})")
                return stop("source", f"A does not check out ({bad})")
        card = await self.images.ensure(cand.repo, base)
        rec["world"] = {k: v for k, v in card.items() if k != "build_log"}
        if not card["ok"]:
            self.funnel.note(NAME, cand.id, "S3_world", False, "world image did not build")
            return stop("source", "world image did not build")
        self.funnel.note(NAME, cand.id, "S3_world", True)
        u = Unit(sha=sha, parent=parent, merge=False, date=cand.params["date"], subject=cand.params["subject"], files=changed_paths(repo_dir, parent, sha))
        oracle_files = {}
        for p in u.files:
            if p.endswith(".py") and not is_test_path(p) and not is_protected(p):
                oracle_files[p] = repo.show(sha, p)
        rec["oracle_files"] = oracle_files
        hot = hot_functions(repo, u)
        from crucible.factory.lanes.windows_imports import project_roots

        roots = sorted(project_roots(base, list(oracle_files)) - {"tests", "test", "testing", "docs", "doc", "benchmarks", "examples"})
        rec["roots"] = roots
        names = {q.rsplit(".", 1)[-1] for _, q in hot}
        targets = suite_targets(base, {})
        mention = [t for t in targets if any(re.search(rf"\b{re.escape(n)}\b", (base / t).read_text(errors="replace")) for n in names)] or targets
        disc = await discover(card["image"], base, roots, hot, mention)
        if not disc["ok"]:
            self.funnel.note(NAME, cand.id, "S4_oracle", False, f"no trace: {disc['why'][:200]}")
            return stop("world", f"trace failed: {disc['why'][:300]}", hot=hot)
        entries = [e for e in disc["entries"].values() if not e["method"] and len(e["samples"]) >= 2]
        entries.sort(key=lambda e: (-len(e["samples"]), -e["hot"]))
        entries = entries[:3]
        if not entries:
            why = f"no module-level entry with plain-data arguments reaches the changed code ({len(disc['entries'])} entries, hot={hot[:4]})"
            self.funnel.note(NAME, cand.id, "S4_oracle", False, why)
            return stop("world", why, hot=hot, entries_seen=[{k: e[k] for k in ("module", "function", "hot", "unsimple", "method")}
                                                              for e in disc["entries"].values()][:10])
        spec = {"entries": [{"module": e["module"], "function": e["function"], "samples": e["samples"], "n": 100, "scale": 3} for e in entries]}
        with tempfile.TemporaryDirectory(prefix="p5p-") as td:
            ptree = Path(td) / "p"
            await asyncio.to_thread(build_tree, base, ptree, solution={"files": oracle_files})
            # size the workload so one baseline run takes about target_seconds
            cal = await measure(card["image"], base, base, spec, seed=1, rounds=1, which="ref")
            if not cal.get("ok"):
                why = f"the workload does not run at A: {str(cal.get('errors') or cal.get('why'))[:200]}"
                self.funnel.note(NAME, cand.id, "S4_oracle", False, why)
                return stop("world", why, spec=spec)
            t1 = max(1e-4, (cal["rounds"]["ref"] or [1e-4])[0])
            if cal.get("raised_ref", 0) > 0.5 * cal.get("n_inputs", 1):
                why = f"most fresh inputs raise at A ({cal['raised_ref']}/{cal['n_inputs']})"
                self.funnel.note(NAME, cand.id, "S4_oracle", False, why)
                return stop("world", why, spec=spec)
            factor = self.target_seconds / t1
            for e in spec["entries"]:
                e["n"] = int(min(20000, max(30, e["n"] * factor)))
            hb = await measure(card["image"], base, ptree, spec, seed=2, rounds=self.rounds_build)
            st_a = await self.sandboxes(card["image"]).run(base, targets)
            st_p = await self.sandboxes(card["image"]).run(ptree, targets)
        if not hb.get("ok") or not hb.get("equal"):
            why = f"the perf commit changes results on fresh inputs ({hb.get('mismatches')} of {hb.get('n_inputs')})" if hb.get("ok") else \
                f"timing failed: {str(hb.get('errors') or hb.get('why'))[:200]}"
            self.funnel.note(NAME, cand.id, "S4_oracle", False, why)
            return stop("world", why, spec=spec, headroom_run=hb)
        ra, rp = normalize_ids(st_a.results, targets), normalize_ids(st_p.results, targets)
        regression = sorted(t for t, s in ra.items() if s == "passed" and rp.get(t) == "passed")
        headroom = "proven" if hb["lo"] > max(1.0 + self.min_gain, MIN_LO) else "unproven"
        rec.update(regression=regression, targets=targets, tests_b={}, p2p=regression, collateral=[], milestones=[], units=[])
        ok4 = headroom == "proven" and bool(regression)
        self.funnel.note(NAME, cand.id, "S4_oracle", ok4, "" if ok4 else (f"headroom_unproven: P's speedup {hb['s']:.3f} (lo {hb['lo']:.3f})"
                                                                          if headroom != "proven" else "no regression test passes at A and P"),
                         s_bar=round(hb["s"], 4), lo=round(hb["lo"], 4))
        return stop("accepted_source" if ok4 else "headroom_unproven", "" if ok4 else "not gate-eligible", spec=spec, s_bar=hb["s"],
                    lo_bar=hb["lo"], cv_ref=hb["cv_ref"], headroom=headroom, hot=hot, headroom_run={k: hb[k] for k in ("rounds", "n_inputs", "raised_ref")},
                    gate_eligible=ok4)


def spec_digest(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
