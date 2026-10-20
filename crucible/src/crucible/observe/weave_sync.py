"""Upload the local Weave-shaped spans to Weave once a W&B key exists (swarm-pipeline/11 §6).

``sync(store)`` replays ``spans.jsonl`` parents first through ``WeaveClient.create_call(op, inputs, parent, attributes,
display_name, started_at=, _call_id_override=)`` and ``finish_call(call, output, ended_at=)``: the call ids and
timestamps are the ones written offline, so the links in W&B tables (``naming.call_url``) open these calls. Calls
already uploaded are listed in ``synced.json`` and skipped (a sync can be re-run after an interruption). The spans
were redacted when they were written; Weave's own key redaction (``redact_sensitive_keys``) runs again on upload.

``_call_id_override`` is a private keyword of ``create_call`` (weave 0.53): when a later version drops it, the sync
falls back to Weave's ids and writes the mapping to ``links.json`` (old id -> new id and URL), which
``scripts/observe.py links`` prints.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from crucible.observe.naming import call_url, entity
from crucible.observe.run import has_key
from crucible.observe.traces import Span, TraceStore

log = logging.getLogger(__name__)


class NoKey(RuntimeError):
    """``WANDB_API_KEY`` is not in the environment: traces stay local."""


def _t(s: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(s) if s else None


def weave_client(project: str, ent: str | None = None) -> Any:
    """``weave.init("<entity>/<project>")`` (network: only after ``has_key()``)."""
    if not has_key():
        raise NoKey("WANDB_API_KEY is not set: run `wandb login` or export the key, then sync")
    import weave  # optional extra ``observe``

    return weave.init(f"{ent}/{project}" if ent else project)


def _ordered(spans: list[Span]) -> list[Span]:
    """Parents before children (a span whose parent is missing is uploaded as a root)."""
    by_id = {s.id: s for s in spans}
    depth: dict[str, int] = {}

    def d(s: Span) -> int:
        if s.id not in depth:
            depth[s.id] = 0 if not s.parent_id or s.parent_id not in by_id else d(by_id[s.parent_id]) + 1
        return depth[s.id]

    return sorted(spans, key=lambda s: (d(s), s.started_at, s.id))


def sync(store: TraceStore, *, client: Any = None, make_client: Callable[[str, str | None], Any] | None = None, ent: str | None = None,
         limit: int | None = None) -> dict[str, Any]:
    """Upload every span of ``store`` not yet in ``synced.json``. ``client``: a ``WeaveClient`` (tests pass a fake)."""
    ent = ent or entity()
    done_p = store.dir / "synced.json"
    done: dict[str, str] = json.loads(done_p.read_text()) if done_p.exists() else {}
    todo = [s for s in _ordered(store.spans()) if s.id not in done]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        return {"project": store.project, "uploaded": 0, "already": len(done)}
    c = client or (make_client or weave_client)(store.project, ent)
    calls: dict[str, Any] = {}
    links: dict[str, dict[str, str]] = json.loads((store.dir / "links.json").read_text()) if (store.dir / "links.json").exists() else {}
    override = True
    n = 0
    try:
        for s in todo:
            parent = calls.get(s.parent_id) if s.parent_id else None
            if s.parent_id and parent is None and s.parent_id in done:
                parent = _Ref(done[s.parent_id], s.trace_id)  # uploaded by an earlier sync: link by id
            kw: dict[str, Any] = {"parent": parent, "attributes": s.attributes, "display_name": s.display_name, "use_stack": False,
                                  "started_at": _t(s.started_at)}
            if override:
                try:
                    call = c.create_call(s.op, s.inputs, _call_id_override=s.id, **kw)
                except TypeError:
                    override = False
                    log.warning("this weave has no _call_id_override: using Weave's call ids (links.json maps them)")
                    call = c.create_call(s.op, s.inputs, **kw)
            else:
                call = c.create_call(s.op, s.inputs, **kw)
            c.finish_call(call, output=s.output, ended_at=_t(s.ended_at))
            new_id = str(getattr(call, "id", s.id))
            calls[s.id] = call
            done[s.id] = new_id
            if new_id != s.id:
                links[s.id] = {"id": new_id, "url": call_url(store.project, new_id, ent)}
            n += 1
    finally:
        done_p.write_text(json.dumps(done, indent=0, sort_keys=True))
        if links:
            (store.dir / "links.json").write_text(json.dumps(links, indent=1, sort_keys=True))
        flush = getattr(c, "flush", None)
        if callable(flush):
            flush()
    return {"project": store.project, "uploaded": n, "already": len(done) - n, "ids_kept": override}


class _Ref:
    """A parent uploaded by an earlier sync (only its ids are needed to attach a child)."""

    def __init__(self, id: str, trace_id: str):
        self.id, self.trace_id = id, trace_id
        self._children: list[Any] = []


def sync_all(root: Path | str, **kw: Any) -> list[dict[str, Any]]:
    """Every project's spans under ``<root>/weave``."""
    return [sync(TraceStore(root, p.name), **kw) for p in sorted((Path(root) / "weave").glob("*")) if (p / "spans.jsonl").exists()]
