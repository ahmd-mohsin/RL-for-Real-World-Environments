"""Crucible in agentenv's explorer (M15 UI): an ``ExplorerPlugin`` mounted at ``/api/v1/crucible``, registered through the
``agent_env.explorer_plugins`` entry point (DEPEND: agentenv explorer; routes shaped like Harbor's viewer API: list,
detail, report).

| route | what |
|---|---|
| ``GET /packs`` | every pack under ``packs/`` with its manifest |
| ``GET /packs/{name}/check`` | ``crucible pack check`` (static level) as JSON, run in its own interpreter |
| ``GET /runs`` | run directories under the data root, with episode counts; ``run_id`` is the path relative to the root |
| ``GET /runs/{run_id}/summary`` | the M13 summary (every step metric, recomputed from the journal) |
| ``GET /runs/{run_id}/report`` | the HTML report, served under ``Content-Security-Policy: sandbox`` |
| ``GET /runs/{run_id}/episodes`` | per-episode rows (reward, gates, steps, cost), paginated |
| ``GET /datasets`` | ``crucible label`` outputs under the data root (counts per file) |

The data root comes from ``[plugins.crucible] data_root`` in agentenv's config, then ``CRUCIBLE_DATA``, then ``data``.

M15 audit (H2 V13 V14, G2 D2-D5 D13):
- a pack is named, never a path: only names in the pack index are checked, and the check runs in a subprocess (the
  static level imports the pack's code; it ran in the explorer process and resolved ``..`` against the server's cwd);
- a run id is the run's path relative to the data root (unique; M9's internal accept sub-runs are not listed); every
  file the API reads is resolved and must stay inside the root (a symlinked ``report.html`` served a file from outside);
- the report is shared HTML: it is served under agentenv's sandbox CSP for stored HTML (no script runs on the explorer
  origin), and a run without one is rendered in memory (a GET never writes);
- listings are bounded (depth, count) and cached briefly; errors never echo the absolute data root; a corrupt
  ``results.jsonl`` is a 422, not an empty summary; ``offset`` is clamped at 0.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from agent_env.explorer.plugin import ExplorerPlugin

SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$")  # a pack name: no dots, no path
SAFE_PART = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")  # one component of a run path: no leading dot, so no . / ..
SAFE = SAFE_PART  # kept for importers
MAX_DEPTH = 8  # run dirs and label outputs sit within this many levels of the root
MAX_DIRS = 20_000  # a listing visits at most this many directories
CACHE_S = 5.0
SANDBOX = {"Content-Security-Policy": "sandbox", "X-Content-Type-Options": "nosniff"}  # agentenv objects router, for stored HTML


def data_root() -> Path:
    try:
        from agent_env.plugins import settings
    except ImportError:  # no agentenv: fall back
        root = None
    else:
        root = (settings("crucible") or {}).get("data_root")  # a malformed config raises (never silently replaced)
    from crucible.core import env as knobs

    return Path(root or knobs.get("CRUCIBLE_DATA") or "data").resolve()


def contained(p: Path, root: Path) -> bool:
    """``p`` resolved (symlinks followed) is inside ``root``."""
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def _walk(root: Path, want: str) -> list[Path]:
    """Directories under ``root`` (depth <= MAX_DEPTH, no symlinked dirs, no hidden dirs) holding a file named ``want``."""
    out: list[Path] = []
    base = len(root.parts)
    for seen, (d, dirs, files) in enumerate(os.walk(root, followlinks=False), 1):
        p = Path(d)
        if seen > MAX_DIRS:
            break
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and len(p.parts) - base < MAX_DEPTH)
        if want in files:
            out.append(p)
    return out


def _is_subrun(rel: Path) -> bool:
    """M9's internal runs (``<build>/accept/<hash>/...``) are evidence of an acceptance check, not runs to browse."""
    return "accept" in rel.parts[:-1]


class _Index:
    def __init__(self) -> None:
        self.cache: dict[tuple[str, str], tuple[float, list[Path]]] = {}

    def find(self, root: Path, want: str) -> list[Path]:
        key = (str(root), want)
        hit = self.cache.get(key)
        if hit and time.monotonic() - hit[0] < CACHE_S:
            return hit[1]
        found = _walk(root, want)
        self.cache[key] = (time.monotonic(), found)
        return found


def _runs(root: Path, index: _Index | None = None) -> list[Path]:
    found = (index or _Index()).find(root, "results.jsonl")
    return sorted(d for d in found if (d / "samples").is_dir() and contained(d, root) and not _is_subrun(d.relative_to(root)))


def _run_dir(root: Path, run_id: str, index: _Index | None = None) -> Path:
    from fastapi import HTTPException

    parts = run_id.split("/")
    if not run_id or not all(SAFE_PART.match(x) for x in parts):
        raise HTTPException(400, "invalid run id")
    d = root.joinpath(*parts)
    if not contained(d, root) or d.is_symlink():
        raise HTTPException(400, "invalid run id")
    if d not in _runs(root, index):
        raise HTTPException(404, f"no run {run_id!r} under the data root")
    return d


def _file(root: Path, run: Path, rel: str) -> Path | None:
    """A file of a run, only when it is a regular file that resolves inside the root (G2 D2)."""
    f = run / rel
    if not f.exists() and not f.is_symlink():
        return None
    from fastapi import HTTPException

    if f.is_symlink() or not f.is_file() or not contained(f, root):
        raise HTTPException(403, f"{rel} of this run is not a regular file inside the data root")
    return f


def _check_run_files(root: Path, run: Path) -> None:
    """Every file the summary readers open (the journal, samples) stays inside the root, and the results parse."""
    from fastapi import HTTPException

    n = 0
    for d, dirs, files in os.walk(run, followlinks=False):
        for name in [*files, *dirs]:
            p = Path(d) / name
            n += 1
            if n > 200_000:
                raise HTTPException(413, "this run directory is too large to serve")
            if p.is_symlink() and not contained(p, root):
                raise HTTPException(403, "this run links to files outside the data root")
    res = _file(root, run, "results.jsonl")
    if res is not None:
        for i, line in enumerate(res.read_text(errors="replace").splitlines(), 1):
            if line.strip():
                try:
                    json.loads(line)
                except ValueError:
                    raise HTTPException(422, f"results.jsonl line {i} is not JSON: the run is corrupt") from None


class CrucibleExplorerPlugin(ExplorerPlugin):
    type = "crucible"

    def __init__(self, root: Path | None = None):
        self._root = root
        self._router = None
        self._index = _Index()

    @property
    def root(self) -> Path:
        return Path(self._root).resolve() if self._root else data_root()

    @property
    def router(self):
        if self._router is None:
            self._router = self._build()
        return self._router

    def _build(self):
        from fastapi import APIRouter, HTTPException
        from fastapi.concurrency import run_in_threadpool
        from fastapi.responses import HTMLResponse

        r = APIRouter(prefix="/api/v1/crucible", tags=["crucible"])

        @r.get("/packs")
        def packs() -> list[dict[str, Any]]:
            from crucible.studio.pack import all_packs, load_manifest

            out = []
            for d in all_packs():
                try:
                    out.append({"dir": d.name, **load_manifest(d).model_dump(mode="json")})
                except Exception as e:  # noqa: BLE001 - shown, not hidden
                    out.append({"dir": d.name, "error": f"{type(e).__name__}: {e}"[:300]})
            return out

        @r.get("/packs/{name}/check")
        async def pack_check(name: str) -> dict[str, Any]:
            from crucible.studio.pack import check_in_subprocess, pack_index

            if not SAFE_NAME.match(name):
                raise HTTPException(400, "invalid pack name")
            hits = pack_index(installed=True).get(name) or []
            if len(hits) != 1:
                raise HTTPException(404 if not hits else 409, f"no pack named {name!r}" if not hits else f"pack name {name!r} is ambiguous")
            rep = await run_in_threadpool(check_in_subprocess, hits[0], level="static", timeout=300.0)  # never in this process
            return rep.to_json()

        @r.get("/runs")
        def runs() -> list[dict[str, Any]]:
            root = self.root
            out = []
            for d in _runs(root, self._index):
                try:  # one unreadable run is shown without a count; it never fails the listing
                    res = _file(root, d, "results.jsonl")
                    n: int | None = sum(1 for x in res.read_text(errors="replace").splitlines() if x.strip()) if res else 0
                except (HTTPException, OSError):
                    n = None
                rel = d.relative_to(root).as_posix()
                out.append({"run_id": rel, "name": d.name, "path": rel, "results": n, "has_report": (d / "bench" / "report.html").is_file()})
            return out

        @r.get("/runs/{run_id:path}/summary")
        def summary(run_id: str) -> dict[str, Any]:
            from crucible.bench.metrics import deterministic_digest
            from crucible.bench.run import summarize_run

            d = _run_dir(self.root, run_id, self._index)
            _check_run_files(self.root, d)
            s, _ = summarize_run(d)  # the plan, owed samples and stop reason ride along (M13 H1 V5)
            return {"run_id": run_id, "digest": deterministic_digest(s), "summary": s}

        @r.get("/runs/{run_id:path}/report", response_class=HTMLResponse)
        def report(run_id: str) -> HTMLResponse:
            from crucible.bench import report as rep
            from crucible.bench.run import summarize_run

            root = self.root
            d = _run_dir(root, run_id, self._index)
            f = _file(root, d, "bench/report.html")
            if f is not None:
                return HTMLResponse(f.read_text(errors="replace"), headers=SANDBOX)
            _check_run_files(root, d)
            s, eps = summarize_run(d)  # rendered in memory: a GET writes nothing (H2 V14)
            return HTMLResponse(rep.render(d.name, s, eps), headers=SANDBOX)

        @r.get("/runs/{run_id:path}/episodes")
        def episodes(run_id: str, offset: int = 0, limit: int = 100) -> dict[str, Any]:
            from crucible.bench.metrics import episodes_json, load_run

            d = _run_dir(self.root, run_id, self._index)
            _check_run_files(self.root, d)
            rows = episodes_json(load_run(d))
            off = max(0, offset)
            return {"total": len(rows), "offset": off, "items": rows[off: off + max(1, min(limit, 1000))]}

        @r.get("/datasets")
        def datasets() -> list[dict[str, Any]]:
            root = self.root
            out = []
            for d in sorted(self._index.find(root, "summary.json")):
                s = d / "summary.json"
                if s.is_symlink() or not contained(s, root):
                    continue
                try:
                    doc = json.loads(s.read_text())
                except (ValueError, OSError):
                    continue
                if isinstance(doc, dict) and doc.get("command") == "label":
                    out.append({"path": d.relative_to(root).as_posix(), "datasets": doc.get("datasets"), "source": doc.get("pack") or doc.get("run")})
            return out

        return r
