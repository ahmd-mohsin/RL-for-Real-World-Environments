"""Crucible in agentenv's explorer (M15 UI): an ``ExplorerPlugin`` mounted at ``/api/v1/crucible``, registered through the
``agent_env.explorer_plugins`` entry point (DEPEND: agentenv explorer; routes shaped like Harbor's viewer API: list,
detail, report).

| route | what |
|---|---|
| ``GET /packs`` | every pack under ``packs/`` with its manifest |
| ``GET /packs/{name}/check`` | ``crucible pack check`` (static level) as JSON |
| ``GET /runs`` | run directories under the data root, with episode counts |
| ``GET /runs/{run_id}/summary`` | the M13 summary (every step metric, recomputed from the journal) |
| ``GET /runs/{run_id}/report`` | the self-contained HTML report |
| ``GET /runs/{run_id}/episodes`` | per-episode rows (reward, gates, steps, cost), paginated |
| ``GET /datasets`` | ``crucible label`` outputs under the data root (counts per file) |

The data root comes from ``[plugins.crucible] data_root`` in agentenv's config, then ``CRUCIBLE_DATA``, then ``data``.
Run ids are validated (``[A-Za-z0-9._-]+``) and resolved inside the root, so the API cannot read outside it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from agent_env.explorer.plugin import ExplorerPlugin

SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def data_root() -> Path:
    try:
        from agent_env.plugins import settings

        root = (settings("crucible") or {}).get("data_root")
    except Exception:  # noqa: BLE001 - no agentenv config: fall back
        root = None
    return Path(root or os.environ.get("CRUCIBLE_DATA") or "data").resolve()


def _runs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("results.jsonl") if (p.parent / "samples").is_dir()})


def _run_dir(root: Path, run_id: str) -> Path:
    from fastapi import HTTPException

    if not SAFE.match(run_id):
        raise HTTPException(400, "invalid run id")
    hits = [d for d in _runs(root) if d.name == run_id]
    if not hits:
        raise HTTPException(404, f"no run {run_id!r} under {root}")
    return hits[0]


class CrucibleExplorerPlugin(ExplorerPlugin):
    type = "crucible"

    def __init__(self, root: Path | None = None):
        self._root = root
        self._router = None

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
            from crucible.studio.pack import check

            if not SAFE.match(name):
                raise HTTPException(400, "invalid pack name")
            try:
                return (await check(name, level="static")).to_json()
            except ValueError as e:
                raise HTTPException(404, str(e)) from e

        @r.get("/runs")
        def runs() -> list[dict[str, Any]]:
            root = self.root
            out = []
            for d in _runs(root):
                n = sum(1 for x in (d / "results.jsonl").read_text().splitlines() if x.strip())
                out.append({"run_id": d.name, "path": str(d.relative_to(root)), "results": n, "has_report": (d / "bench" / "report.html").exists()})
            return out

        @r.get("/runs/{run_id}/summary")
        def summary(run_id: str) -> dict[str, Any]:
            from crucible.bench.metrics import deterministic_digest, load_run, summarize

            s = summarize(load_run(_run_dir(self.root, run_id)))
            return {"run_id": run_id, "digest": deterministic_digest(s), "summary": s}

        @r.get("/runs/{run_id}/report", response_class=HTMLResponse)
        def report(run_id: str) -> str:
            from crucible.bench.run import report_run

            d = _run_dir(self.root, run_id)
            f = d / "bench" / "report.html"
            if not f.exists():
                report_run(run_id, d)
            return f.read_text()

        @r.get("/runs/{run_id}/episodes")
        def episodes(run_id: str, offset: int = 0, limit: int = 100) -> dict[str, Any]:
            from crucible.bench.metrics import episodes_json, load_run

            rows = episodes_json(load_run(_run_dir(self.root, run_id)))
            return {"total": len(rows), "offset": offset, "items": rows[offset: offset + max(1, min(limit, 1000))]}

        @r.get("/datasets")
        def datasets() -> list[dict[str, Any]]:
            root = self.root
            out = []
            for s in sorted(root.rglob("summary.json")):
                try:
                    doc = json.loads(s.read_text())
                except ValueError:
                    continue
                if doc.get("command") == "label":
                    out.append({"path": str(s.parent.relative_to(root)), "datasets": doc.get("datasets"), "source": doc.get("pack") or doc.get("run")})
            return out

        return r
