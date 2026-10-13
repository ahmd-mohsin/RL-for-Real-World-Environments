"""Byte-identical MiMo export (ARCHITECTURE M11 gate: "one MiMo env is imported, run and re-exported byte-identical";
research m11-m12/05 §3.7). The export emits the **original files verbatim** from the source env recorded at import —
never a regenerated ``state.db`` or a re-serialized ``manifest.json`` (MiMo's are 1-space-indented JSON) — and checks every
file against the sha256 manifest taken at import, so a run (which works on a bundle copy) can never leak into it."""

from __future__ import annotations

import shutil
from pathlib import Path

from crucible.interop.mimo.importer import MimoEnv, tree_hashes


def export(env: MimoEnv, out: Path) -> dict:
    """Re-emit the source env's bytes. This proves the import never altered its source, not that a World round-trips:
    there is no Crucible-World → MiMo writer yet (M11 audit G3 D13, H3 V10)."""
    out, src = Path(out).resolve(), Path(env.path).resolve()
    if out == src or src.is_relative_to(out) or out.is_relative_to(src):  # rmtree(out) would delete the source (H3 V10)
        raise ValueError(f"export target {out} overlaps the source env {src}")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(env.path, out, ignore=shutil.ignore_patterns("__pycache__", ".cache"))
    got = tree_hashes(out)
    want = env.files or {}
    diff = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
    if diff:
        raise ValueError(f"MiMo export differs from the imported env in {len(diff)} files: {diff[:5]}")
    return {"files": len(got), "identical": True}
