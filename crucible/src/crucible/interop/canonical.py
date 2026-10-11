"""Cross-runtime identity (ARCHITECTURE M11 exit gate; research m11-m12/05 §2): what "the same reward and the same dense
channel through every runtime" means, and the normalizations that make it true.

- ``jcs(obj)``: canonical JSON (RFC 8785 style: sorted keys, shortest round-trip float ``repr``, no whitespace). The gate
  compares canonical forms, not on-disk bytes (each runtime's writer formats differently).
- ``call_id(episode_key, ordinal)``: model-call ids are **derived**, never copied (``episode_key = "<task id>|<episode>"``
  everywhere) — Responses item ids, verifiers node ids,
  ATIF step ids and gateway event ids all differ across runtimes; runtime ids are kept in ``extra`` for joins.
- ``canonical_tool(name)``: one tool naming. Crucible ``<app>.<tool>``; agentenv / OpenAI ``<app>__<tool>``; NeMo-Gym
  strips its MCP namespace (``<server>/<tool>`` or ``<server>.<tool>``).
- ``canonical_args(args)``: JSON-normalized arguments (stringified JSON parsed; key order irrelevant).
- ``reward_doc`` / ``steps_rows``: the comparable projections (volatile fields removed: timing, ``cost_s``, runtime
  ``extra``; floats canonical).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from crucible.core.canonical import normalize

VOLATILE_ROW_KEYS = {"cost_s", "extra", "atif_step_id"}


def _norm(o: Any) -> Any:
    """The core normalizer (finite floats, ``-0.0`` folded, NFC strings): one canonical form for every hash."""
    return normalize(o)


def jcs(obj: Any) -> str:
    return json.dumps(_norm(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def call_id(episode_key: str, ordinal: int) -> str:
    return "c" + hashlib.sha256(f"{episode_key}|{ordinal}".encode()).hexdigest()[:16]


def canonical_tool(name: str, *, known: Iterable[str] = ()) -> str:
    """Map a runtime's tool name to ``<app>.<tool>``. ``known``: the World's qualified names, used to resolve ambiguity."""
    known = set(known)
    if name in known:
        return name
    for sep in ("__", "/", "."):
        if sep in name:
            app, _, tool = name.partition(sep)
            q = f"{app}.{tool}"
            if not known or q in known:
                return q
    matches = [k for k in known if k.split(".", 1)[-1] == name]
    return matches[0] if len(matches) == 1 else name


def canonical_args(args: Any) -> dict:
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            return {"__unparseable__": args}
    return json.loads(jcs(args or {}))


def reward_doc(path: Path) -> dict | None:
    """The comparable ``reward.json`` (None when absent: a masked grade has none)."""
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def steps_rows(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.append({k: v for k, v in r.items() if k not in VOLATILE_ROW_KEYS})
    return out


def same(a: Any, b: Any) -> bool:
    return jcs(a) == jcs(b)
