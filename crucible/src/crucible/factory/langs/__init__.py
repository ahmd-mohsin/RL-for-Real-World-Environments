"""Language profiles for the task factory (swarm-pipeline/07): what a source file, a test file and a protected path are in
one language, how a set of test files runs and reports per-test results, how a World image is built and leak-probed, and
how test functions are found, compared and hidden. ``base`` holds the ``LangProfile`` protocol, the active-profile
context the window machinery consults (``active``, ``use``) and what every compiled-language profile shares; ``python``
adapts the existing behaviour (no change), ``golang``, ``rust``, ``node`` (TypeScript / JavaScript under vitest or jest)
and ``java`` (maven) are new; ``cuda`` is the kernel debug lane's (swarm-pipeline/10), chosen by that lane, never detected.

This package init stays import-free: the low lanes (``lanes.executable``, ``lanes.windows_git``) import ``langs.base``,
and a profile module imports the lanes, so profiles load lazily through ``get``."""

from __future__ import annotations

import importlib
from typing import Any

NAMES = ("python", "go", "rust", "node", "java", "cuda")
_MODULES = {"python": "python", "go": "golang", "rust": "rust", "node": "node", "java": "java", "cuda": "cuda"}
_MADE: dict[str, Any] = {}


def get(name: str) -> Any:
    """The profile called ``name`` (one instance per process)."""
    key = {"golang": "go", "typescript": "node", "javascript": "node", "ts": "node", "js": "node"}.get(name, name)
    if key not in _MODULES:
        raise KeyError(f"no language profile {name!r} (known: {', '.join(NAMES)})")
    if key not in _MADE:
        _MADE[key] = importlib.import_module(f"crucible.factory.langs.{_MODULES[key]}").PROFILE
    return _MADE[key]


def detect(tree: Any) -> str | None:
    """The language of a source tree by its build files (first match in ``NAMES`` order but Python last: a Go or Rust
    repository often carries Python helper scripts)."""
    for n in ("go", "rust", "node", "java", "python"):
        try:
            prof = get(n)
        except ImportError:  # a profile not built in this tree
            continue
        if prof.detect(tree):
            return n
    return None
