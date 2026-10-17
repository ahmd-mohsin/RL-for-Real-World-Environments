"""Fail-closed identities over a settings document (R-1; PATTERN: NeMo-RL
``nemo_rl/algorithms/single_controller_utils/rollout_checkpoint.py`` bootstrap fingerprint, Apache-2.0; slime freezes
every field not declared mutable).

An allowlist fails open: a setting nobody thought to list is ignored, so two different experiments share one identity.
Here every leaf of the document counts unless an ``exclude`` glob names it as operational (concurrency, budgets,
paths). ``project`` flattens a document to ``{dotpath: value}`` (list items are indexed: ``policies.0.settings.model``),
drops excluded paths and scrubs secret values; ``fingerprint`` hashes the result in canonical JSON; ``diff`` names what
changed between two projections. Globs match whole dotpath segments: ``*`` is one segment, ``**`` any number, and a
glob that matches a prefix excludes everything under it (``retry_backoff_s``, ``**.api_key``).
"""

from __future__ import annotations

import fnmatch
import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

SCHEME = "crucible.compat_identity/v1"

# Never part of an identity: credentials may rotate without changing an experiment, and are never written down.
SECRET_GLOBS = ("**.api_key", "**.*_api_key", "**.apikey", "**.password", "**.*_password", "**.secret", "**.*_secret", "**.headers")


def flatten(doc: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    """``{path: leaf}`` of a JSON-like document; an empty mapping or list is a leaf."""
    if isinstance(doc, Mapping) and doc:
        return {p: v for k, sub in doc.items() for p, v in flatten(sub, (*prefix, str(k))).items()}
    if isinstance(doc, (list, tuple)) and doc:
        return {p: v for i, sub in enumerate(doc) for p, v in flatten(sub, (*prefix, str(i))).items()}
    return {prefix: list(doc) if isinstance(doc, tuple) else doc}


def excluded(path: tuple[str, ...], globs: Iterable[str]) -> bool:
    """Whether a glob matches ``path`` or one of its prefixes."""
    return any(_match(path[:n], tuple(g.split("."))) for g in globs for n in range(1, len(path) + 1))


def _match(path: tuple[str, ...], glob: tuple[str, ...]) -> bool:
    if not glob:
        return not path
    if glob[0] == "**":
        return any(_match(path[i:], glob[1:]) for i in range(len(path) + 1))
    return bool(path) and fnmatch.fnmatchcase(path[0], glob[0]) and _match(path[1:], glob[1:])


def project(doc: Any, exclude: Iterable[str] = ()) -> dict[str, Any]:
    """The identity-bearing settings of ``doc``: flattened to dotpaths, excluded and secret paths dropped, and every
    string scrubbed of registered and credential-shaped secrets (``core.secrets``)."""
    from crucible.core.secrets import scrub_obj

    globs = (*exclude, *SECRET_GLOBS)
    flat = {".".join(p): v for p, v in flatten(doc).items() if not excluded(p, globs)}
    return dict(sorted(scrub_obj(flat).items()))  # type: ignore[union-attr]


def fingerprint(settings: Mapping[str, Any]) -> str:
    """``sha256:<hex>`` of the projected settings in canonical JSON, with the scheme inside the hashed bytes."""
    from crucible.core.canonical import dumps

    return "sha256:" + hashlib.sha256(dumps({"scheme": SCHEME, "settings": dict(settings)}).encode()).hexdigest()


ABSENT = "<absent>"


def diff(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """``{dotpath: (old, new)}`` for every setting added, removed or changed; ``ABSENT`` marks a missing side."""
    return {k: (old.get(k, ABSENT), new.get(k, ABSENT)) for k in sorted(set(old) | set(new)) if old.get(k, ABSENT) != new.get(k, ABSENT)}


def describe(changes: Mapping[str, tuple[Any, Any]], limit: int = 20) -> str:
    """At most ``limit`` changes, one per line, values shortened."""

    def short(v: Any) -> str:
        s = v if v == ABSENT else repr(v)
        return s if len(s) <= 60 else s[:57] + "..."

    lines = [f"  {k}: {short(a)} -> {short(b)}" for k, (a, b) in list(changes.items())[:limit]]
    more = len(changes) - limit
    return "\n".join(lines + ([f"  ... and {more} more"] if more > 0 else []))
