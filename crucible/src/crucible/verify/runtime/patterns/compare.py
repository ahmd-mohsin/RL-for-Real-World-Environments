"""Comparators for differential checks. Standard library only; packs plug in their own (numpy) comparators.

Every comparator returns a list of failure dicts (empty = equal). Comparators run in the grader's root process on
artifacts the candidate wrote into a sealed directory.

- ``bitwise_tree``: the same file set with identical bytes (concurrency equivalence, bit-exact outputs);
- ``json_close``: nested JSON with per-key absolute/relative tolerances;
- ``TwoTier``: strict when the trajectories agree; a bounded relaxed tier when a *bifurcation detector* says the
  reference itself takes a different discrete branch (DESIGN §8.2, task04 KinFu-LS). The relaxed tier must still kill
  every negative, which ``calibrate`` verifies; whichever tier was used is reported (P20: the prompt must state it).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bitwise_tree(cand: Path, ref: Path, *, include: tuple = ("*",), exclude: tuple = ()) -> list[dict]:
    def files(root: Path) -> dict[str, Path]:
        out = {}
        for p in Path(root).rglob("*"):
            if p.is_file() and not p.is_symlink():
                rel = p.relative_to(root).as_posix()
                if any(fnmatch.fnmatch(rel, i) for i in include) and not any(fnmatch.fnmatch(rel, x) for x in exclude):
                    out[rel] = p
        return out

    a, b = files(cand), files(ref)
    fails = [{"file": f, "error": "missing"} for f in sorted(set(b) - set(a))]
    fails += [{"file": f, "error": "unexpected"} for f in sorted(set(a) - set(b))]
    fails += [{"file": f, "error": "differs"} for f in sorted(set(a) & set(b)) if sha256(a[f]) != sha256(b[f])]
    return fails


def json_close(cand: Any, ref: Any, *, atol: float = 0.0, rtol: float = 0.0, tol: dict | None = None, path: str = "") -> list[dict]:
    """Recursive compare; ``tol`` maps a key path (``a.b``) or key name to ``(atol, rtol)``."""
    tol = tol or {}
    fails: list[dict] = []
    if isinstance(ref, dict):
        if not isinstance(cand, dict):
            return [{"key": path, "error": "type"}]
        for k, v in ref.items():
            kp = f"{path}.{k}" if path else str(k)
            if k not in cand:
                fails.append({"key": kp, "error": "missing"})
                continue
            a, r = tol.get(kp, tol.get(str(k), (atol, rtol)))
            fails += json_close(cand[k], v, atol=a, rtol=r, tol=tol, path=kp)
        return fails
    if isinstance(ref, list):
        if not isinstance(cand, list) or len(cand) != len(ref):
            return [{"key": path, "error": "length", "cand": len(cand) if isinstance(cand, list) else None, "ref": len(ref)}]
        for i, (x, y) in enumerate(zip(cand, ref, strict=True)):
            fails += json_close(x, y, atol=atol, rtol=rtol, tol=tol, path=f"{path}[{i}]")
        return fails
    if isinstance(ref, bool) or not isinstance(ref, (int, float)):
        return [] if cand == ref else [{"key": path, "cand": cand, "ref": ref}]
    if not isinstance(cand, (int, float)) or isinstance(cand, bool) or not math.isfinite(float(cand)):
        return [{"key": path, "cand": cand, "ref": ref, "error": "not a finite number"}]
    if abs(float(cand) - float(ref)) > atol + rtol * abs(float(ref)):
        return [{"key": path, "cand": cand, "ref": ref, "atol": atol, "rtol": rtol}]
    return []


@dataclass
class TwoTier:
    """``strict(c, r)`` first; if it fails and ``bifurcated(c, r)`` says the reference took a different discrete branch
    on this input, ``relaxed(c, r)`` decides. Returns ``(failures, tier)``."""

    strict: Callable[[Any, Any], list]
    relaxed: Callable[[Any, Any], list]
    bifurcated: Callable[[Any, Any], bool]

    def __call__(self, cand: Any, ref: Any) -> tuple[list, str]:
        f = self.strict(cand, ref)
        if not f:
            return [], "strict"
        if self.bifurcated(cand, ref):
            g = self.relaxed(cand, ref)
            return g, "relaxed"
        return f, "strict"


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())
