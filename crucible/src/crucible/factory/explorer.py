"""The human-labelling intake for agentenv's explorer (M17; study 04 §3.3): an ``ExplorerPlugin`` at
``/api/v1/crucible/labels`` over the append-only ``crucible.factory.labels.LabelStore``. agentenv's explorer has no write
routes and its ``ReviewStore`` holds one continue / abort decision per step with a 7-day TTL, so labels live in Crucible's
own store and this plugin is the door to it.

| route | what |
|---|---|
| ``GET /queue`` | items still needing labels (``family``, ``limit``, ``offset``); with an identity, only those the labeller has not labelled |
| ``GET /item?id=`` | one item's payload and its labels (others' labels are hidden for a ``blind`` item) |
| ``POST /label`` | one label: ``{item, item_sha256, value, reason, supersedes}``; 409 on a stale hash, a forged or forking supersede, or a second label by one labeller |
| ``GET /decisions`` | the decision store, filtered by ``criterion``, ``recipe_version``, ``subject``, ``actor``, ``action`` |
| ``GET /kappa`` | Cohen's or Fleiss' kappa per family (or one ``family``), over fresh labels |

Items come from ``<data_root>/factory/label_items.jsonl`` (one ``{"id", "family", "payload", "blind"?, "choices"?}`` per
line; the hash a label binds to is computed here from the payload, never taken from the file) or from an ``items``
callable. Labels go to ``<data_root>/factory/labels.jsonl``.

Identity. The labeller is read from a request header the plugin trusts **only when configured**: set
``[plugins.crucible] label_identity_header`` (for example ``X-Labeller``, set by the proxy in front of the explorer) and,
so a direct caller cannot forge it, ``label_identity_secret`` (the request must also carry it in
``X-Crucible-Identity-Secret``). With no header configured, ``POST /label`` answers 403: nobody is authenticated, so
nobody may label. The identity must look like ``human:<id>`` or ``model:<id>``.
"""

# no ``from __future__ import annotations``: FastAPI resolves the route annotations declared inside ``_build``
import hmac
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from agent_env.explorer.plugin import ExplorerPlugin

from crucible.core.ids import validate_id
from crucible.factory.cards import check_actor
from crucible.factory.labels import (
    CorruptStore,
    DuplicateLabel,
    LabelStore,
    LabelStoreError,
    StaleItem,
    SupersedeError,
    item_sha256,
)

SECRET_HEADER = "X-Crucible-Identity-Secret"
MAX_LIMIT = 500
MAX_ITEMS = 100_000


@dataclass(frozen=True)
class Item:
    id: str
    family: str
    payload: Any
    sha256: str
    blind: bool = False
    choices: tuple[str, ...] = field(default_factory=tuple)


def _root() -> Path:
    try:
        from agent_env.plugins import settings
    except ImportError:  # pragma: no cover
        root = None
    else:
        root = (settings("crucible") or {}).get("data_root")
    return Path(root or os.environ.get("CRUCIBLE_DATA") or "data").resolve()


def _settings() -> dict[str, Any]:
    try:
        from agent_env.plugins import settings
    except ImportError:  # pragma: no cover
        return {}
    return dict(settings("crucible") or {})


def read_items(path: Path) -> list[Item]:
    """Items from a JSONL file; a bad line is skipped, never a crash (the file is written by other tools)."""
    out: dict[str, Item] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return []
    for line in lines[:MAX_ITEMS]:
        try:
            row = json.loads(line)
            item_id = validate_id(row["id"])
            family = validate_id(row["family"])
            payload = row["payload"]
            choices = tuple(str(c) for c in row.get("choices", ()))
            out.setdefault(item_id, Item(item_id, family, payload, item_sha256(payload), bool(row.get("blind", False)), choices))
        except (ValueError, KeyError, TypeError):
            continue
    return list(out.values())


class LabelsExplorerPlugin(ExplorerPlugin):
    type: ClassVar[str] = "crucible_labels"

    def __init__(self, root: Path | None = None, *, store: LabelStore | None = None, items: Callable[[], list[Item]] | None = None,
                 identity_header: str | None = None, identity_secret: str | None = None, raters: int = 2):
        self._root = root
        self._store = store
        self._items = items
        self.identity_header = identity_header
        self.identity_secret = identity_secret
        self.raters = raters
        self._router = None

    @classmethod
    def from_config(cls) -> "LabelsExplorerPlugin":
        cfg = _settings()
        return cls(identity_header=cfg.get("label_identity_header") or None, identity_secret=cfg.get("label_identity_secret") or None,
                   raters=int(cfg.get("label_raters", 2)))

    @property
    def root(self) -> Path:
        return Path(self._root).resolve() if self._root else _root()

    @property
    def store(self) -> LabelStore:
        return self._store or LabelStore(self.root / "factory" / "labels.jsonl")

    def items(self) -> dict[str, Item]:
        found = self._items() if self._items else read_items(self.root / "factory" / "label_items.jsonl")
        return {i.id: i for i in found}

    @property
    def router(self):
        if self._router is None:
            self._router = self._build()
        return self._router

    def _identity(self, request) -> str:
        from fastapi import HTTPException

        if not self.identity_header:
            raise HTTPException(403, "labelling is not configured: no identity header is trusted")
        if self.identity_secret is not None:
            given = request.headers.get(SECRET_HEADER, "")
            if not hmac.compare_digest(given.encode(), self.identity_secret.encode()):
                raise HTTPException(403, "identity header not trusted: bad or missing secret")
        who = request.headers.get(self.identity_header)
        if not who:
            raise HTTPException(401, f"missing {self.identity_header} header")
        try:
            return check_actor(who)
        except ValueError:
            raise HTTPException(400, "labeller must look like 'human:<id>' or 'model:<id>'") from None

    def _build(self):
        from fastapi import APIRouter, HTTPException, Query, Request
        from pydantic import BaseModel, ConfigDict, Field

        plugin = self
        r = APIRouter(prefix="/api/v1/crucible/labels", tags=["crucible-labels"])

        class LabelBody(BaseModel):
            model_config = ConfigDict(extra="forbid")
            item: str = Field(max_length=512)
            item_sha256: str = Field(max_length=64)
            value: str = Field(min_length=1, max_length=200)
            reason: str = Field(default="", max_length=10_000)
            supersedes: str | None = Field(default=None, max_length=512)

        def store() -> LabelStore:
            return plugin.store

        def guarded(fn):
            try:
                return fn()
            except CorruptStore:
                raise HTTPException(500, "the label store is corrupt; it refuses to serve") from None

        def optional_identity(request: Request) -> str | None:
            if not plugin.identity_header:
                return None
            try:
                return plugin._identity(request)
            except HTTPException:
                return None

        @r.get("/queue")
        def queue(request: Request, family: str | None = None, limit: int = Query(50, ge=1, le=MAX_LIMIT), offset: int = Query(0, ge=0)) -> dict[str, Any]:
            who = optional_identity(request)
            items = {k: v for k, v in plugin.items().items() if family is None or v.family == family}
            st = store()
            todo = guarded(lambda: st.needs_labels({k: v.sha256 for k, v in items.items()}, raters=plugin.raters, labeller=who))
            rows = [{"item": k, "family": items[k].family, "item_sha256": items[k].sha256, "blind": items[k].blind,
                     "choices": list(items[k].choices),
                     "fresh_labels": len(st.effective(k, items[k].sha256)), "raters": plugin.raters} for k in todo]
            return {"total": len(rows), "offset": offset, "labeller": who, "items": rows[offset: offset + limit]}

        @r.get("/item")
        def item(request: Request, id: str) -> dict[str, Any]:
            it = plugin.items().get(id)
            if it is None:
                raise HTTPException(404, "no such item")
            who = optional_identity(request)
            labels = guarded(lambda: store().effective(id))
            shown = [lab for lab in labels if not it.blind or lab.labeller == who]
            return {"item": it.id, "family": it.family, "item_sha256": it.sha256, "blind": it.blind, "choices": list(it.choices),
                    "payload": it.payload, "labels": [{**lab.model_dump(mode="json"), "fresh": lab.item_sha256 == it.sha256} for lab in shown],
                    "hidden_labels": len(labels) - len(shown)}

        @r.post("/label")
        def post_label(request: Request, body: LabelBody) -> dict[str, Any]:
            who = plugin._identity(request)
            it = plugin.items().get(body.item)
            if it is None:
                raise HTTPException(404, "no such item")
            if it.choices and body.value not in it.choices:
                raise HTTPException(422, f"value must be one of {list(it.choices)}")
            try:
                label = store().add_label(item=it.id, item_sha256=body.item_sha256, labeller=who, value=body.value, reason=body.reason,
                                          supersedes=body.supersedes, current_item_sha256=it.sha256)
            except StaleItem:
                raise HTTPException(409, {"error": "stale item hash", "current_item_sha256": it.sha256}) from None
            except (SupersedeError, DuplicateLabel) as e:
                raise HTTPException(409, str(e)) from None
            except CorruptStore:
                raise HTTPException(500, "the label store is corrupt; it refuses to serve") from None
            except (LabelStoreError, ValueError) as e:
                raise HTTPException(422, str(e)) from None
            return label.model_dump(mode="json")

        @r.get("/decisions")
        def decisions(criterion: str | None = None, recipe_version: str | None = None, subject: str | None = None,
                      actor: str | None = None, action: str | None = None, limit: int = Query(200, ge=1, le=MAX_LIMIT)) -> dict[str, Any]:
            found = guarded(lambda: store().decisions(subject=subject, actor=actor, action=action,  # type: ignore[arg-type]
                                                      criterion=criterion, recipe_version=recipe_version))
            return {"total": len(found), "decisions": [d.model_dump(mode="json") for d in found[-limit:]]}

        @r.get("/kappa")
        def kappa(family: str | None = None, raters: str | None = Query(None, description="comma-separated labellers")) -> dict[str, Any]:
            names = [x for x in (raters or "").split(",") if x] or None
            by_family: dict[str, dict[str, str]] = {}
            for it in plugin.items().values():
                if family is None or it.family == family:
                    by_family.setdefault(it.family, {})[it.id] = it.sha256
            st = store()
            out = {}
            for fam, items in sorted(by_family.items()):
                a = guarded(lambda items=items: st.agreement(items, names))
                out[fam] = {"method": a.method, "kappa": a.kappa, "n_items": a.n_items, "raters": list(a.raters),
                            "observed_agreement": a.observed, "items_in_family": len(items)}
            return {"families": out}

        return r
