"""The window lane's S3 gate and World fixes (swarm-pipeline/00 §4 S3, §8.3 ``world_bug``; P4).

**S3** (``world_s3``) on one World image for one window, all executed in ``crucible-lh-*`` containers:
- ``builds``: the image exists (its digest);
- ``existing_tests``: A's own test files run through pytest in the image (no B test file is written) with a JUnit report
  and at least one passing test;
- ``leak``: ``windows_world.leak_probes`` (B's release cannot be downloaded or found; no file defines a new test of the
  window; the seeded workspace's history is one commit tagged ``base``);
- ``reruns``: the existing suite ``reruns`` more times (half at concurrency 1, half at ``concurrency``); a rerun is clean
  when its status vector equals the first run's. The Clopper-Pearson upper bound on the unclean rate must be <= 5%
  (60 clean runs bound it at 4.9%);
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
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.lanes.windows import grader_tests, passed_group, run_tests, sandbox_factory, suite_targets
from crucible.factory.lanes.windows_git import _defs
from crucible.factory.lanes.windows_world import IMAGE_RECIPE, PREFIX, ImageBook, image_id, leak_probes

S3_PREFIX = f"{PREFIX}p4-"  # World containers of this phase (P4): crucible-lh-p4-*


def cp_upper(fails: int, n: int, alpha: float = 0.05) -> float:
    from crucible.verify.calibrate import clopper_pearson_upper

    return float(clopper_pearson_upper(fails, n, alpha))


def new_names(rec: Mapping[str, Any], base_dir: Path) -> set[str]:
    """Test functions B's test files define that A's tree never mentions (the leak probe's names)."""
    a_text = "\n".join(f.read_text(errors="replace") for f in Path(base_dir).rglob("*.py") if f.is_file())
    names = {q.rsplit("::", 1)[-1] for txt in (rec.get("tests_b") or {}).values() for q in _defs(txt)}
    return {n for n in names if n.startswith("test") and n not in a_text}


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
    leak = await leak_probes(image, base_dir, package=w.get("project") or rec.get("repo"), version=rec.get("b_version"),
                             new_test_names=new_names(rec, base_dir), name_prefix=f"{name_prefix}leak-")
    out["leak"] = leak
    out["checks"]["leak_probes_hold"] = bool(leak.get("ok"))
    # reruns of the existing suite: half alone, half ``concurrency`` at a time
    ref = first.results
    unclean: list[dict[str, Any]] = []
    done = 0
    half = reruns // 2

    async def one() -> None:
        nonlocal done
        tr = await run_tests(sb, base_dir, {}, {}, targets)
        done += 1
        # P5: an id built from the clock (marshmallow parametrizes on datetime.now()) exists in one run only; ids are
        # compared where both runs have them, and the runs must collect the same number of tests (a crashed module still
        # changes the count)
        common = set(ref) & set(tr.results)
        if tr.infra != "ok" or len(tr.results) != len(ref) or any(ref[k] != tr.results[k] for k in common):
            diff = sorted(k for k in common if ref.get(k) != tr.results.get(k)) or sorted(set(ref) ^ set(tr.results))
            unclean.append({"infra": tr.infra, "why": tr.why, "differs": diff[:10]})

    if ran:
        for _ in range(half):
            await one()
        sem = asyncio.Semaphore(max(1, concurrency))

        async def gated() -> None:
            async with sem:
                await one()

        await asyncio.gather(*(gated() for _ in range(reruns - half)))
    bound = cp_upper(len(unclean), done) if done else 1.0
    out["reruns"] = {"n": done, "clean": done - len(unclean), "unclean": unclean[:5], "cp_upper": round(bound, 4), "concurrency": [1, concurrency]}
    out["checks"]["reruns_clean"] = bool(done) and bound <= max_flake
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
                 timeout_s: float = 300.0, sandboxes: Callable[[str], Any] | None = None):
        self.rec, self.base = dict(rec), Path(base_dir)
        self.image = image or (rec.get("world") or {}).get("image")
        self.s3_reruns, self.concurrency, self.timeout_s, self.sandboxes = s3_reruns, concurrency, timeout_s, sandboxes
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
            name = f"{S3_PREFIX}repro-{uuid.uuid4().hex[:10]}"
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
        res = await world_s3(self.rec, self.base, image, reruns=self.s3_reruns, concurrency=self.concurrency, sandboxes=self.sandboxes)
        self.s3_results.append(res)
        return res

    def content(self) -> dict[str, Any]:
        return {"image": self.image}
