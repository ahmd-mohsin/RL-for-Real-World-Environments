"""The window lane's S3 gate and World fixes (swarm-pipeline/00 §4 S3, §8.3 ``world_bug``; P4).

**S3** (``world_s3``) on one World image for one window, all executed in ``crucible-lh-*`` containers:
- ``builds``: the image exists (its digest);
- ``existing_tests``: A's own test files run through pytest in the image (no B test file is written) with a JUnit report
  and at least one passing test;
- ``leak``: ``windows_world.leak_probes`` (B's release cannot be downloaded or found; no file defines a new test of the
  window; the seeded workspace's history is one commit tagged ``base``);
- ``reruns``: the existing suite ``reruns`` more times (half at concurrency 1, half at ``concurrency``); a rerun is clean
  when its status vector equals the first run's. The Clopper-Pearson upper bound on the unclean rate must be <= 5%
  (60 clean runs bound it at 4.9%). Timing tests (``windows_timing``: the record's ``excluded['timing']`` and every
  test of A's suite the static rule names) are not compared: they are never checks, and their verdict is the host's
  load (benchmark v2); they are listed as ``ignored_timing``;
- ``grading``: the window's own hidden tests in the image: the oracle passes every milestone test and the start state
  passes none of them (a World change must not move the verifier: P2 of ``00`` §5.3).

**World fixes** (``world_fix``, the World expert's job, P3 left them at ``reproduced_needs_fix``): the window's World
rebuilt under the current ``windows_world.IMAGE_RECIPE`` (recipe 3: ``world_needs``). ``WindowWorldPort`` is the
``traceaudit.WorldPort`` the executed acceptance runs: the reproduction on the oracle state in the current World, then in
the fixed one, then S3 on the fixed one. An accepted fix becomes a task version (``traceaudit.version_accepted``), and
``world_of_version`` is how a consumer (P4 targeting, a solver run) picks the World of the head version.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.lanes.windows import grader_tests, passed_group, run_tests, sandbox_factory, suite_targets
from crucible.factory.lanes.windows_git import _defs
from crucible.factory.lanes.windows_timing import timing_tests
from crucible.factory.lanes.windows_world import _DIST_SEL, IMAGE_RECIPE, PREFIX, ImageBook, image_id, leak_probes, window_touched, workspace_modules

S3_PREFIX = f"{PREFIX}p4-"  # World containers of this phase (P4): crucible-lh-p4-*


def cp_upper(fails: int, n: int, alpha: float = 0.05) -> float:
    from crucible.verify.calibrate import clopper_pearson_upper

    return float(clopper_pearson_upper(fails, n, alpha))


def new_names(rec: Mapping[str, Any], base_dir: Path) -> set[str]:
    """Test functions B's test files define that A's tree never mentions (the leak probe's names)."""
    from crucible.factory.langs.base import active

    if (lang := active()) is not None:
        return lang.new_test_names(rec.get("tests_b") or {}, {}, Path(base_dir))
    a_text = "\n".join(f.read_text(errors="replace") for f in Path(base_dir).rglob("*.py") if f.is_file())
    names = {q.rsplit("::", 1)[-1] for txt in (rec.get("tests_b") or {}).values() for q in _defs(txt)}
    return {n for n in names if n.startswith("test") and n not in a_text}


# a run-level cause the profile names in ``why`` (``golang.parse_report``): the host's (masked) or one test's (a panic)
HOST_RUN = re.compile(r"\btimeout: |fatal error: runtime: out of memory|\bkilled: ")
PANIC = re.compile(r"\bpanic: (\S+)")


def run_level(ref: Mapping[str, str], tr: Any, differs: Sequence[str]) -> str:
    """Why a rerun's differences are the run's, not its tests' ('' when they are tests'): every differing test is a
    passing test of the first run that failed or vanished, and in each package they touch at least half of the first
    run's passing tests (and 10) went at once: a test binary that died (a crash, an OOM kill, go test's own timeout), not
    dozens of flaky tests. excelize 824ac2ce-d1 (2026-10-27): one rerun of 60, infra ok, its whole root package failed;
    the flip counts then named every test of calc_test.go."""
    if not differs or any(ref.get(t) != "passed" or tr.results.get(t) == "passed" for t in differs):
        return ""
    pkg = lambda t: t.split("::", 1)[0].rsplit("/", 1)[0] if "/" in t.split("::", 1)[0] else ""
    passing: dict[str, int] = {}
    for t, v in ref.items():
        if v == "passed":
            passing[pkg(t)] = passing.get(pkg(t), 0) + 1
    hit: dict[str, int] = {}
    for t in differs:
        hit[pkg(t)] = hit.get(pkg(t), 0) + 1
    if all(n >= 10 and n >= passing.get(p, 0) / 2 for p, n in hit.items()):
        return f"{sum(hit.values())} passing tests of {len(hit)} package(s) failed at once ({', '.join(sorted(p or '.' for p in hit))[:120]})"
    return ""


def rerun_verdict(ref: Mapping[str, str], tr: Any, ignored: set[str]) -> tuple[str, list[str]]:
    """One S3 rerun against the first run: ``masked`` when the host failed it, never a verdict on the tree (V8): the
    whole suite killed at the time limit with no result (k8sgpt 855ff255, 4 of 5 unclean reruns, 2026-10-26), a sandbox
    that never started or a run killed from outside (``infra == "masked"``: reviewdog 4da03bc4-s1, "sandbox did not
    start (exit 125)", 2026-10-27), a run-level death the profile names (go test's timeout, an OOM kill) or a whole
    package lost at once with no test-level cause (``run_level``). A panic is one test's: only the panicking test
    differs, so the per-function flaky rule sees one flip. Else ``clean`` or ``unclean`` with the tests that differ
    (every one: the flip counts read them). P5: an id built from the clock (marshmallow parametrizes on datetime.now())
    exists in one run only; ids are compared where both runs have them, and the runs must collect the same number of
    tests (a crashed module still changes the count)."""
    why = str(tr.why or "")
    if (why.startswith("timeout after") and not tr.results) or tr.infra == "masked" or HOST_RUN.search(why):
        return "masked", []
    common = (set(ref) & set(tr.results)) - ignored
    # the counts are compared without the ignored tests (smoke 2026-10-29, huma 815e1ff8): an excluded load-dependent test
    # that panicked was missing from the rerun, so the count differed and the rerun was unclean with nothing differing
    if tr.infra != "ok" or len(set(tr.results) - ignored) != len(set(ref) - ignored) or any(ref[k] != tr.results[k] for k in common):
        differs = sorted(k for k in common if ref.get(k) != tr.results.get(k)) or sorted((set(ref) ^ set(tr.results)) - ignored)
        if panics := sorted({m.group(1) for m in PANIC.finditer(why)}):
            fn = lambda t: t.split("::", 1)[-1].split("/", 1)[0]
            own = [t for t in differs if fn(t) in panics]
            if own:
                return "unclean", own
            dead = {t.split("::", 1)[0].rpartition("/")[0] for t in ignored if fn(t) in panics}
            if set(panics) <= {fn(t) for t in ignored}:
                # every panic is an ignored test's: what its dying test binary took with it (its package's other tests) was
                # not measured, so that part of the run is no evidence; a difference elsewhere still is
                rest = [t for t in differs if t.split("::", 1)[0].rpartition("/")[0] not in dead]
                return ("unclean", rest) if rest else ("masked", [])
            return "unclean", differs
        if run_level(ref, tr, [t for t in differs if t in ref]) and not (set(tr.results) - set(ref) - ignored):
            return "masked", []
        return "unclean", differs
    return "clean", []


async def world_s3(rec: Mapping[str, Any], base_dir: Path, image: str, *, reruns: int = 60, concurrency: int = 4,
                   sandboxes: Callable[[str], Any] | None = None, max_flake: float = 0.05, name_prefix: str = f"{S3_PREFIX}s3-") -> dict[str, Any]:
    """S3 for ``image`` as the World of window ``rec`` (see the module doc). ``ok`` needs every check."""
    if not name_prefix.startswith(PREFIX):
        raise ValueError(f"World containers are named {PREFIX}*, got {name_prefix!r}")
    base_dir = Path(base_dir)
    out: dict[str, Any] = {"image": image, "checks": {}}
    digest = image_id(image)
    out["digest"] = digest
    out["checks"]["builds"] = digest is not None
    if digest is None:
        return {**out, "ok": False, "why": "the World image does not exist (did not build)"}
    sbx = sandboxes or sandbox_factory(parallel=concurrency, name_prefix=name_prefix, timeout_s=600.0)
    sb = sbx(image)
    targets = suite_targets(base_dir, {})
    first = await run_tests(sb, base_dir, {}, {}, targets)
    counts = {s: sum(1 for v in first.results.values() if v == s) for s in sorted(set(first.results.values()))}
    ran = first.infra == "ok" and bool(first.results) and counts.get("passed", 0) > 0
    out["existing_tests"] = {"targets": len(targets), "counts": counts, "infra": first.infra, "why": first.why, "exit": first.exit_code,
                             "failing": sorted(t for t, v in first.results.items() if v in ("failed", "error"))[:50],
                             "tail": first.tail[-600:] if not ran else ""}
    out["checks"]["existing_tests_run"] = ran
    w = rec.get("world") or {}
    from crucible.factory.langs.base import active

    leak = await (lang.leak_probes if (lang := active()) is not None else leak_probes)(image, base_dir, package=w.get("project") or rec.get("repo"), version=rec.get("b_version"),
                             new_test_names=new_names(rec, base_dir), name_prefix=f"{name_prefix}leak-", installed=w.get("projects"),
                             touched=window_touched(rec))
    out["leak"] = leak
    out["checks"]["leak_probes_hold"] = bool(leak.get("ok"))
    # reruns of the existing suite: half alone, half ``concurrency`` at a time
    ref = first.results
    a_tests = {p: f.read_text(errors="replace") for p in {t.split("::", 1)[0] for t in ref} if (f := base_dir / p).is_file()}
    ignored = set((rec.get("excluded") or {}).get("timing") or []) | set(timing_tests(ref, a_tests))
    out["ignored_timing"] = sorted(ignored)[:50]
    unclean: list[dict[str, Any]] = []
    masked: list[str] = []
    done = 0
    half = reruns // 2

    async def one() -> None:
        nonlocal done
        tr = await run_tests(sb, base_dir, {}, {}, targets)
        verdict, diff = rerun_verdict(ref, tr, ignored)
        if verdict == "masked":
            masked.append(str(tr.why) or run_level(ref, tr, [t for t in ref if tr.results.get(t) != ref[t]]) or "masked")
            return
        done += 1
        if verdict == "unclean":
            unclean.append({"infra": tr.infra, "why": tr.why, "n_differs": len(diff), "differs": diff[:10], "flips": diff,
                            "tail": str(tr.tail or "")[-1500:]})

    if ran:
        for _ in range(half):
            await one()
        sem = asyncio.Semaphore(max(1, concurrency))

        async def gated() -> None:
            async with sem:
                await one()

        await asyncio.gather(*(gated() for _ in range(reruns - half)))
        while done < reruns and len(masked) <= reruns // 5:  # a masked rerun is replaced, alone, while masks stay rare
            await one()
    bound = cp_upper(len(unclean), done) if done else 1.0
    flips: dict[str, int] = {}
    for u in unclean:
        for t in u.pop("flips"):
            flips[t] = flips.get(t, 0) + 1
    out["reruns"] = {"n": done, "clean": done - len(unclean), "unclean": unclean[:5], "cp_upper": round(bound, 4), "concurrency": [1, concurrency],
                     "masked": len(masked), "masked_why": sorted(set(masked))[:10], "flips": dict(sorted(flips.items(), key=lambda x: -x[1])[:40])}
    out["checks"]["reruns_clean"] = bool(done) and bound <= max_flake  # too many masks leave fewer reruns: the bound fails
    # the verifier does not move: the oracle passes every milestone test, the start state none
    tests, _ = grader_tests(rec, base_dir)
    ms = list(rec.get("milestones") or [])
    mtargets = sorted({x.split("::", 1)[0] for m in ms for x in m["tests"]})
    orc = await run_tests(sb, base_dir, dict(rec.get("oracle_files") or {}), tests, mtargets, rec.get("tests_deleted") or ())
    st = await run_tests(sb, base_dir, {}, tests, mtargets, rec.get("tests_deleted") or ())
    o_ms = {m["id"]: passed_group(orc.results, m["tests"]) for m in ms}
    s_ms = {m["id"]: passed_group(st.results, m["tests"], require_all=False) for m in ms}
    out["grading"] = {"oracle_milestones": o_ms, "start_any_milestone_test": s_ms, "infra": [orc.infra, st.infra]}
    out["checks"]["grading_unchanged"] = orc.infra == st.infra == "ok" and all(o_ms.values()) and not any(
        passed_group(st.results, m["tests"]) for m in ms)
    bad = [k for k, v in out["checks"].items() if not v]
    return {**out, "ok": not bad, "why": "" if not bad else f"S3 fails: {', '.join(bad)}"}


# ---- World fixes ---------------------------------------------------------------------------------------------------------
async def world_fix(rec: Mapping[str, Any], base_dir: Path, *, images: ImageBook | None = None) -> dict[str, Any]:
    """The window's World rebuilt under the current recipe: ``{"type": "world_image", "recipe", "image", "digest", "from_image",
    "needs"}``. ``ok`` is false when the build failed (no fix to propose)."""
    images = images or ImageBook()
    card = await images.ensure(rec["repo"], Path(base_dir))
    needs = card.get("needs") or {}
    return {"type": "world_image", "recipe": IMAGE_RECIPE, "image": card["image"], "digest": card["digest"], "ok": bool(card["ok"]),
            "from_image": (rec.get("world") or {}).get("image"), "needs": {k: needs.get(k) for k in ("apt", "pip_build", "extras", "import_deps", "pins")},
            "why": needs.get("why"), "unresolved": needs.get("unresolved"), "build_log": "" if card["ok"] else str(card.get("build_log"))[-800:]}


def world_of_version(rec: Mapping[str, Any], store: Any, subject: str | None = None) -> dict[str, Any]:
    """``rec`` with the World of the head task version of ``subject`` (default: the window id): the last accepted
    ``world_image`` fix replaces ``world.image`` / ``world.digest`` and is recorded as ``world.version``. Unchanged when the
    store holds no World fix for it."""
    head = store.head(subject or rec["id"], "task") if store is not None else None
    fixes = [f for f in ((head.content.get("world_fixes") or []) if head is not None else []) if isinstance(f, Mapping) and f.get("type") == "world_image"]
    if not fixes:
        return dict(rec)
    f = fixes[-1]
    return {**rec, "world": {**(rec.get("world") or {}), "image": f["image"], "digest": f["digest"], "version": head.id, "recipe": f.get("recipe")}}


SITE_MODULE = re.compile(r"site-packages/([A-Za-z_][\w]*)(?:/|\.py)")
RELEASE = re.compile(r"^\d+(\.\d+)*$")


def implicated_modules(tail: str) -> list[str]:
    """Modules a failing test's traceback runs through, most implicated first: names called on the failing source lines
    (pytest's ``>`` and ``E`` lines, in order: the first block is the root of a chained error), then the modules of
    ``site-packages`` frames."""
    out: list[str] = []
    for ln in tail.splitlines():
        if ln.startswith((">", "E ")):
            out += [m for m in re.findall(r"\b([A-Za-z_]\w*)\.[A-Za-z_]\w*", ln) if m not in out]
    out += [m for m in SITE_MODULE.findall(tail) if m not in out]
    return out


def _vkey(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def version_ladder(versions: Sequence[str], pin: str | None, *, current: str | None = None) -> list[str]:
    """Candidate releases for an executed version search: final releases only, the newest patch of each minor, ascending
    from the pin (closest to what A was tested with first); without a pin, descending from the newest below ``current``."""
    finals = sorted({v for v in versions if RELEASE.match(v)}, key=_vkey)
    best: dict[tuple[int, ...], str] = {}
    for v in finals:
        best[_vkey(v)[:2]] = v
    ladder = sorted(best.values(), key=_vkey)
    if pin and RELEASE.match(pin):
        return [v for v in ladder if _vkey(v) >= _vkey(pin)[:2] and v != current]
    return [v for v in reversed(ladder) if v != current]


async def _docker(*args: str, timeout: float = 600) -> tuple[int, str]:
    p = await asyncio.create_subprocess_exec("docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        o, _ = await asyncio.wait_for(p.communicate(), timeout=timeout)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            p.kill()
        return -1, "timeout"
    return p.returncode or 0, o.decode(errors="replace")


async def repair_versions(rec: Mapping[str, Any], base_dir: Path, image: str, *, sandboxes: Callable[[str], Any] | None = None,
                          max_builds: int = 8) -> dict[str, Any]:
    """The World expert's executed version search (recipe 3): when A's own tests fail in the World at A (upstream released
    A with them passing), find the installed distribution their tracebacks run through (``site-packages/<module>``) and
    try its releases (``version_ladder`` from A's pin) as one-line image layers on ``image``. The first release with
    which A's failing tests pass and no other A test starts failing is the fix: ``{"image", "dist", "version", "tried"}``.
    ``ok`` is false when nothing failed, nothing was implicated or no release repairs it."""
    base_dir = Path(base_dir)
    sbx = sandboxes or sandbox_factory(parallel=4, name_prefix=f"{S3_PREFIX}repair-")
    targets = suite_targets(base_dir, {})
    before = await run_tests(sbx(image), base_dir, {}, {}, targets)
    failing = sorted(t for t, st in before.results.items() if st in ("failed", "error"))
    out: dict[str, Any] = {"image": image, "failing_at_a": failing, "tried": [], "ok": False}
    if not failing:
        return {**out, "why": "A's own tests all pass in this World"}
    trace = await run_tests(sbx(image), base_dir, {}, {}, failing[:20])  # the failing tests alone: their whole tracebacks
    mods = implicated_modules(trace.tail)
    name = f"{S3_PREFIX}pip-{uuid.uuid4().hex[:8]}"
    script = ("import importlib.metadata as m, json, sys; pd = m.packages_distributions(); "
              "print(json.dumps({k: [pd.get(k, [None])[0], m.version(pd[k][0]) if pd.get(k) else None] for k in sys.argv[1:]}))")
    rc, txt = await _docker("run", "--rm", "--network", "none", "--name", name, image, "python", "-c", script, *mods)
    dists = json.loads(txt.strip().splitlines()[-1]) if rc == 0 and txt.strip() else {}
    project = {str((rec.get("world") or {}).get("project") or "").lower(), str(rec.get("repo") or "").lower()}
    implicated = [(mod, d, v) for mod, (d, v) in ((m, dists.get(m) or [None, None]) for m in mods)
                  if d and d.lower() not in (project | {"pytest", "pluggy", "pytest-timeout"}) - {""}]
    out["implicated"] = [{"module": m, "dist": d, "version": v} for m, d, v in implicated]
    if not implicated:
        return {**out, "why": "no installed distribution in the failing tests' tracebacks"}
    pins = world_needs_pins(base_dir)
    builds = 0
    for mod, dist, cur in implicated:
        rc, txt = await _docker("run", "--rm", "--name", f"{S3_PREFIX}pip-{uuid.uuid4().hex[:8]}", image, "pip", "index", "versions", dist, timeout=180)
        m = re.search(r"Available versions:\s*(.+)", txt)
        versions = [v.strip() for v in m.group(1).split(",")] if m else []
        pin = next((v for k, v in pins.items() if k.lower() == dist.lower()), None)
        for v in version_ladder(versions, pin, current=cur):
            if builds >= max_builds:
                break
            builds += 1
            tag = f"{image}-{re.sub(r'[^a-z0-9]+', '', dist.lower())}-{v}"
            with tempfile.TemporaryDirectory(prefix="p4fix-") as td:
                (Path(td) / "Dockerfile").write_text(f"FROM {image}\nRUN pip install --no-cache-dir {dist}=={v}\n")
                brc, blog = await _docker("build", "-q", "-t", tag, td, timeout=900)
            row: dict[str, Any] = {"dist": dist, "version": v, "built": brc == 0}
            if brc != 0:
                row["why"] = blog.strip().splitlines()[-1][:200] if blog.strip() else "build failed"
                out["tried"].append(row)
                continue
            after = await run_tests(sbx(tag), base_dir, {}, {}, targets)
            still = sorted(t for t in failing if after.results.get(t) != "passed")
            new = sorted(t for t, st in after.results.items() if st in ("failed", "error") and t not in failing)
            row.update(still_failing=still, new_failures=new[:10])
            out["tried"].append(row)
            if not still and not new and after.infra == "ok":
                return {**out, "ok": True, "image": tag, "dist": dist, "version": v, "from_version": cur, "pin": pin, "module": mod,
                        "digest": image_id(tag)}
            await _docker("rmi", tag, timeout=120)
    return {**out, "why": f"no release of {[d for _, d, _ in implicated]} repairs A's failing tests ({builds} builds)"}


def world_needs_pins(base_dir: Path) -> dict[str, str]:
    from crucible.factory.lanes.windows_world import world_needs

    return dict(world_needs(base_dir).get("pins") or {})


class WindowWorldPort:
    """A ``traceaudit.WorldPort`` over a mined window: the start tree plus a state, run in the window's World image (the
    current one, or a fix's image) in a ``crucible-lh-p4-*`` container (``--network none``) at ``/w``. A recorded command's
    absolute workspace path (the solver's World mounted it at its host path) is rewritten to ``/w``; trailing pagers are cut
    (``traceaudit.repro_command``)."""

    def __init__(self, rec: Mapping[str, Any], base_dir: Path, *, image: str | None = None, s3_reruns: int = 60, concurrency: int = 4,
                 timeout_s: float = 300.0, sandboxes: Callable[[str], Any] | None = None, name_prefix: str = S3_PREFIX):
        if not name_prefix.startswith(PREFIX):
            raise ValueError(f"World containers are named {PREFIX}*, got {name_prefix!r}")
        self.rec, self.base = dict(rec), Path(base_dir)
        self.image = image or (rec.get("world") or {}).get("image")
        self.s3_reruns, self.concurrency, self.timeout_s, self.sandboxes = s3_reruns, concurrency, timeout_s, sandboxes
        self.name_prefix = name_prefix
        self.s3_results: list[dict[str, Any]] = []

    def _tree(self, dest: Path, state: Any) -> Path:
        shutil.copytree(self.base, dest, symlinks=True)
        files = (state or {}).get("files") if isinstance(state, Mapping) else None
        for p, txt in (files or {}).items():
            f = dest / p
            if txt is None:
                f.unlink(missing_ok=True)
            else:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(txt)
        return dest

    async def run(self, command: str, *, state: Any, image: str) -> tuple[int, str]:
        with tempfile.TemporaryDirectory(prefix="p4-ww-") as d:
            t = self._tree(Path(d) / "t", state)
            name = f"{self.name_prefix}repro-{uuid.uuid4().hex[:10]}"
            argv = ["docker", "run", "--rm", "--network", "none", "--name", name, "--memory", "2g", "--cpus", "2", "-v", f"{t}:/w", "-w", "/w",
                    "-e", "PYTHONPATH=/w/src:/w", "-e", "HOME=/tmp", image, "sh", "-c", command]
            p = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                o, _ = await asyncio.wait_for(p.communicate(), timeout=self.timeout_s)
                return p.returncode or 0, o.decode(errors="replace")
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    p.kill()
                await asyncio.to_thread(subprocess.run, ["docker", "rm", "-f", name], capture_output=True, check=False)
                return -1, "timeout"

    async def reproduce(self, repro: Mapping[str, Any], *, state: Any, fix: Mapping[str, Any] | None = None) -> dict[str, Any]:
        from crucible.factory.traceaudit import fault_regex, repro_command

        args = repro.get("arguments") or {}
        cmd = str(args.get("command") or "") if isinstance(args, Mapping) else ""
        if not cmd:
            return {"failed": False, "exit": None, "output": "the reproduction names no shell command"}
        ws = str(repro.get("workspace") or "")
        if ws:
            cmd = cmd.replace(ws, "/w")
        cmd = repro_command(cmd)
        image = str((fix or {}).get("image") or self.image)
        rc, out = await self.run(cmd, state=state, image=image)
        rx = fault_regex(str(repro.get("signature") or ""))
        same = bool(rx.search(out)) if rx is not None else str(repro.get("observed") or "")[:80] in out
        return {"failed": rc != 0 and same, "exit": rc, "output": out[-1500:], "same_signature": same, "command": cmd, "image": image}

    async def s3(self, *, fix: Mapping[str, Any] | None = None) -> dict[str, Any]:
        image = str((fix or {}).get("image") or self.image)
        res = await world_s3(self.rec, self.base, image, reruns=self.s3_reruns, concurrency=self.concurrency, sandboxes=self.sandboxes,
                             name_prefix=f"{self.name_prefix}s3-")
        self.s3_results.append(res)
        return res

    def content(self) -> dict[str, Any]:
        return {"image": self.image}


# ---- Phase A: the site-packages leak, evidence and the executed World fix -----------------------------------------------------
# Benchmark v2 found that one World image served every window whose A had the same packaging, so its installed copy of the
# project was the A of whichever window built it first (a later A is the solution). ``EXTRAS_FALLBACK`` fix 3 removes it at
# build and ``no_installed_project_source`` probes it. Phase A rebuilds every World built before the fix, with evidence.
LEAK_OBSERVED = "LEAK: installed project source files present"
_TESTISH = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")


LEAK_REPRO = ("import importlib.metadata as m, re, sys; " + _DIST_SEL + "c = sum(1 for d in m.distributions() if own(d) for f in (d.files or []) "
              "if str(f).endswith('.py') and not any(p.endswith('.dist-info') for p in f.parts) and f.parts[0] != '..'); "
              f"print('{LEAK_OBSERVED}: %d' % c if c else 'clean'); sys.exit(1 if c else 0)")


def leak_repro_command(names: Sequence[str], modules: Sequence[str] = ()) -> str:
    """A shell command that fails (exit 1, printing ``LEAK_OBSERVED`` and the count) when the World's installed copy of any
    of ``names`` (or of a distribution installing one of A's own ``modules``) holds Python source: the
    ``no_installed_project_source`` probe as a ``world_bug`` reproduction. ``-I``: site-packages only (the workspace on
    ``PYTHONPATH`` is not the installed copy)."""
    args = [*(n for n in names if n), *(f"mod:{m}" for m in modules)]
    return f"python -I -c {shlex.quote(LEAK_REPRO)} " + " ".join(shlex.quote(x) for x in args)


def _added(a: str | None, b: str | None) -> set[str]:
    """Non-trivial lines of ``b`` that ``a`` does not contain (stripped; short lines, comments and docstring quotes out)."""
    al = {x.strip() for x in (a or "").splitlines()}
    return {x.strip() for x in (b or "").splitlines() if len(x.strip()) >= 12 and not x.strip().startswith(("#", '"""', "'''")) and x.strip() not in al}


def compare_installed(files: Mapping[str, str], base_dir: Path, oracle_files: Mapping[str, str | None]) -> dict[str, Any]:
    """The installed copy (``windows_world.installed_copy`` files) against the window's A (``base_dir``) and B (A plus
    ``oracle_files``). Each installed file is mapped to the workspace path that ends with its site-packages path (the
    shortest: ``src/`` layouts and a stitched workspace's ``host/`` / ``consumer/``) and classed ``unchanged`` (A == B
    == installed), ``b`` (equals B, differs from A), ``a`` (equals A, differs from B), ``other`` or ``unmatched``. For
    every source file the oracle changes (tests out): whether the installed copy equals B, and the share of the oracle's
    added lines it contains. ``solution_present``: some changed file is installed as B, or at least half of the oracle's
    added lines are in the installed copy."""
    base_dir = Path(base_dir)
    paths = sorted({str(f.relative_to(base_dir)) for f in base_dir.rglob("*.py") if f.is_file() and ".git" not in f.parts} | set(oracle_files))

    def where(rel: str) -> str | None:
        hits = [p for p in paths if p == rel or p.endswith("/" + rel)]
        return min(hits, key=len) if hits else None

    def a_text(p: str) -> str | None:
        f = base_dir / p
        return f.read_text(errors="replace") if f.is_file() else None

    classes: dict[str, int] = {}
    mapped: dict[str, str] = {}
    for rel, txt in files.items():
        p = where(rel)
        if p is None:
            k = "unmatched"
        else:
            mapped[p] = txt
            a = a_text(p)
            b = oracle_files.get(p, a)
            k = "unchanged" if txt == a == b else "b" if txt == b else "a" if txt == a else "other"
        classes[k] = classes.get(k, 0) + 1
    changed = []
    tot_added = tot_present = 0
    for p, b in sorted(oracle_files.items()):
        if not p.endswith(".py") or _TESTISH.search(p):
            continue
        a = a_text(p)
        add = _added(a, b)
        inst = mapped.get(p)
        present = len(add & {x.strip() for x in (inst or "").splitlines()}) if inst is not None else 0
        tot_added += len(add)
        tot_present += present
        changed.append({"path": p, "installed": "absent" if inst is None else "b" if inst == b else "a" if inst == a else "other",
                        "added_lines": len(add), "present": present})
    share = round(tot_present / tot_added, 3) if tot_added else None
    return {"installed_files": len(files), "classes": classes, "changed": changed, "added_lines": tot_added, "added_present": tot_present,
            "added_share": share, "solution_present": any(c["installed"] == "b" for c in changed) or bool(share and share >= 0.5)}


def identify_copy(files: Mapping[str, str], trees: Mapping[str, Path]) -> dict[str, Any]:
    """Which known start tree the installed copy is (``trees``: label -> a window's A): the label with the most installed
    files equal to the tree's file at the same mapped path, and whether every file matched."""
    best: dict[str, Any] = {"label": None, "equal": 0, "of": len(files), "exact": False}
    for label, root in trees.items():
        root = Path(root)
        idx = {str(f.relative_to(root)): f for f in root.rglob("*.py") if f.is_file() and ".git" not in f.parts}
        eq = 0
        for rel, txt in files.items():
            hits = [p for p in idx if p == rel or p.endswith("/" + rel)]
            if hits and idx[min(hits, key=len)].read_text(errors="replace") == txt:
                eq += 1
        if eq > best["equal"]:
            best = {"label": label, "equal": eq, "of": len(files), "exact": eq == len(files)}
    return best


async def leak_world_fix(rec: Mapping[str, Any], base_dir: Path, *, images: Any, versions: Any, names: Sequence[str], current_image: str | None = None,
                         sandboxes: Callable[[str], Any] | None = None, name_prefix: str = S3_PREFIX, s3_reruns: int = 60, concurrency: int = 4,
                         carry_repair: Mapping[str, Any] | None = None, s3_rec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The leak as a ``world_bug`` through the executed acceptance (``traceaudit.accept_world_bug``, 00 §8.3): the
    reproduction (``leak_repro_command``) must fail on the oracle state in ``current_image`` (the World solvers used: the
    record's, or its head version's), pass in the rebuilt World (``images.ensure`` under the current recipe, plus a
    carried version-repair layer ``{dist, version}`` from an earlier fix), and the rebuilt World must pass S3 (60 reruns,
    every leak probe including ``no_installed_project_source`` over ``names``). An accepted fix becomes a task version
    (``traceaudit.version_accepted``: parent = the head, or a new root). ``s3_rec``: the record S3 runs on (an open task's
    has no milestones). Returns ``{status, reasons, proposal, fix, version, before, after, s3}``."""
    from crucible.factory.traceaudit import Proposal, TaskContext, accept_world_bug, version_accepted

    base_dir = Path(base_dir)
    wid = str(rec["id"])
    old = current_image or (rec.get("world") or {}).get("image")
    p = Proposal(kind="world_bug", task=wid, traces=[], source="s3", why="the World's installed copy of the project holds source (site-packages leak)",
                 artifact={"type": "repro", "function": "repo.shell", "arguments": {"command": leak_repro_command(names, workspace_modules(base_dir))},
                           "observed": LEAK_OBSERVED,
                           "signature": "installed project source in site-packages"}, change={"world_fix": None})
    built: dict[str, Any] = {}

    async def fixer(_p: Any) -> dict[str, Any] | None:
        card = await images.ensure(str(rec["repo"]), base_dir)
        built["card"] = {k: v for k, v in card.items() if k != "build_log"}
        if not card["ok"]:
            built["build_log"] = str(card.get("build_log") or "")[-800:]
            return None
        image, digest = card["image"], card["digest"]
        if carry_repair and carry_repair.get("dist") and carry_repair.get("version"):
            dist, ver = str(carry_repair["dist"]), str(carry_repair["version"])
            tag = f"{image}-{re.sub(r'[^a-z0-9]+', '', dist.lower())}-{ver}"
            if image_id(tag) is None:
                with tempfile.TemporaryDirectory(prefix="pafix-") as td:
                    (Path(td) / "Dockerfile").write_text(f"FROM {image}\nRUN pip install --no-cache-dir {dist}=={ver}\n")
                    brc, blog = await _docker("build", "-q", "-t", tag, td, timeout=900)
                if brc != 0:
                    built["build_log"] = blog[-800:]
                    return None
            image, digest = tag, image_id(tag)
        return {"type": "world_image", "recipe": IMAGE_RECIPE, "fix": "installed-sources-removed", "image": image, "digest": digest, "from_image": old,
                "repair": dict(carry_repair) if carry_repair else None}

    s3r = dict(s3_rec or rec)
    s3r["world"] = {**(s3r.get("world") or {}), "projects": list(names)}
    port = WindowWorldPort(s3r, base_dir, image=old, s3_reruns=s3_reruns, concurrency=concurrency, sandboxes=sandboxes, name_prefix=name_prefix)
    p = await accept_world_bug(p, port, oracle={"files": dict(rec.get("oracle_files") or {})}, fixer=fixer)
    out: dict[str, Any] = {"status": p.status, "reasons": list(p.reasons), "proposal": p.id, "fix": p.change.get("world_fix"), "built": built,
                           "before": p.evidence.get("before_fix_on_oracle"), "after": p.evidence.get("after_fix_on_oracle"), "version": None}
    s3 = p.evidence.get("s3") or (port.s3_results[-1] if port.s3_results else None)
    out["s3"] = {k: s3.get(k) for k in ("ok", "why", "image", "digest", "checks", "existing_tests", "reruns", "leak", "grading", "ignored_timing")} if s3 else None
    if p.status == "accepted":
        v = version_accepted(p, versions, TaskContext(wid, prompt=""))
        out["version"] = v.id if v else None
    return out
