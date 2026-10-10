"""Byte-identical MiMo export (ARCHITECTURE M11 gate: "one MiMo env is imported, run and re-exported byte-identical";
research m11-m12/05 §3.7). The export emits the **original files verbatim** from the source env recorded at import —
never a regenerated ``state.db`` or a re-serialized ``manifest.json`` (MiMo's are 1-space-indented JSON) — and checks every
file against the sha256 manifest taken at import, so a run (which works on a bundle copy) can never leak into it."""

from __future__ import annotations

import shutil
from pathlib import Path

from crucible.interop.mimo.importer import MimoEnv, tree_hashes


def export(env: MimoEnv, out: Path) -> dict:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(env.path, out, ignore=shutil.ignore_patterns("__pycache__", ".cache"))
    got = tree_hashes(out)
    want = env.files or {}
    diff = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
    if diff:
        raise ValueError(f"MiMo export differs from the imported env in {len(diff)} files: {diff[:5]}")
    return {"files": len(got), "identical": True}
