"""slime's JSONL input rows (03 §5 item 1; swarm-pipeline/00 §11 item 1).

One row per training task, read by slime with ``--input-key prompt --label-key label --metadata-key metadata``
(``run_qwen38_27b_fully_async.sh:169``): ``prompt`` is the task prompt plus the harness epilogue; ``metadata`` carries
``instance_id``, ``image`` (the World at A), ``workdir``, ``agent`` (``claude_code`` | ``codex`` | ``crucible_rlm``;
``scripted`` for the equivalence run), ``crucible_task_ref`` (the ``LongTaskRef`` JSON) and ``split="train"``.

**Held-out rows are never written (V9).** ``write_rows`` refuses a row whose split, or whose ref's split, is not
``train``, and any row naming a sealed id (``sealed``: a benchmark manifest's task and source ids).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

AGENTS = ("claude_code", "codex", "crucible_rlm", "scripted")


class SealedRowError(ValueError):
    """A held-out (sealed) task offered to the training rows."""


def slime_row(*, instance_id: str, prompt: str, image: str, workdir: str, agent: str, crucible_task_ref: Mapping[str, Any], epilogue: str = "",
              split: str = "train", label: str | None = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if agent not in AGENTS:
        raise ValueError(f"agent is one of {AGENTS}, got {agent!r}")
    text = prompt if not epilogue or epilogue in prompt else f"{prompt.rstrip()}\n\n{epilogue.strip()}\n"
    md = {"instance_id": instance_id, "image": image, "workdir": workdir, "agent": agent, "crucible_task_ref": dict(crucible_task_ref), "split": split,
          **dict(extra or {})}
    return {"prompt": text, "label": label, "metadata": md}


def check_row(row: Mapping[str, Any], sealed: Iterable[str] = ()) -> list[str]:
    """Why ``row`` may not train (empty = it may)."""
    md = row.get("metadata") or {}
    ref = md.get("crucible_task_ref") or {}
    out = []
    if md.get("split") != "train":
        out.append(f"split is {md.get('split')!r}, not train")
    if ref.get("split") not in (None, "train"):
        out.append(f"the task ref is split {ref.get('split')!r}")
    seal = set(sealed)
    hit = sorted({str(x) for x in (md.get("instance_id"), ref.get("task_id"), ref.get("source_id")) if x is not None} & seal)
    if hit:
        out.append(f"sealed ids {hit}")
    for k in ("instance_id", "image", "workdir", "agent"):
        if not md.get(k):
            out.append(f"metadata.{k} missing")
    if not str(row.get("prompt") or "").strip():
        out.append("empty prompt")
    return out


def write_rows(rows: Iterable[Mapping[str, Any]], path: Path, *, sealed: Iterable[str] = ()) -> dict[str, Any]:
    """Write the rows as JSONL (atomically), refusing the whole file on any held-out or malformed row."""
    seal = set(sealed)
    rows = list(rows)
    bad = {str((r.get("metadata") or {}).get("instance_id")): p for r in rows if (p := check_row(r, seal))}
    if bad:
        raise SealedRowError(f"{len(bad)} row(s) may not train: {dict(list(bad.items())[:3])}")
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return {"path": str(path), "rows": len(rows), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
