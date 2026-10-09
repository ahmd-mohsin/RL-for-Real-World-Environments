"""JSON API App: a small REST-like service whose whole state is one JSON document (``json_doc``).

Models third-party systems (vendor portal, CRM, ticket API) without writing a server. Resources live
under collections: ``{"invoices": {"INV-1": {...}}, "vendors": {...}}``.
Tools: list_resources(collection, filter={}), get_resource(collection, id),
create_resource(collection, id, data), update_resource(collection, id, patch), delete_resource(collection, id).
Params: ``collections`` (allowed names; default: whatever the seed has), ``read_only_collections``,
``seed`` (host JSON file copied in at provision).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from crucible.core.schema.actions import ToolSpec
from crucible.world.apps.base import App, ToolError


class JsonApiApp(App):
    template = "common/json_api"
    PATH_PARAMS = ("path",)

    async def provision(self) -> None:
        seed = self.spec.params.get("seed")
        doc = json.loads(self.ctx.host_seed(seed).read_text()) if seed else {c: {} for c in self.spec.params.get("collections", [])}
        await self._save(doc)

    async def _load(self) -> dict[str, Any]:
        return json.loads(await self.ctx.exec.read_text(self.state_params["path"]))

    async def _save(self, doc: dict[str, Any]) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(doc, f, indent=2, sort_keys=True)
        try:
            await self.ctx.exec.run(f"mkdir -p $(dirname {self.state_params['path']})")
            await self.ctx.exec.push_file(Path(f.name), self.state_params["path"])
        finally:
            Path(f.name).unlink(missing_ok=True)

    def _check(self, doc: dict[str, Any], collection: str, write: bool) -> dict[str, Any]:
        allowed = self.spec.params.get("collections")
        if collection not in doc or (allowed and collection not in allowed):
            raise ToolError(f"unknown collection {collection!r}")
        if write and collection in self.spec.params.get("read_only_collections", []):
            raise ToolError(f"collection {collection!r} is read-only")
        return doc[collection]

    async def list_tools(self) -> list[ToolSpec]:
        s, o = {"type": "string"}, {"type": "object"}
        return [
            ToolSpec(app=self.id, name="list_resources", read_only=True, description="List resources in a collection, optionally filtered by exact field values.",
                     input_schema={"type": "object", "properties": {"collection": s, "filter": o}, "required": ["collection"]}),
            ToolSpec(app=self.id, name="get_resource", read_only=True, description="Get one resource.", input_schema={"type": "object", "properties": {"collection": s, "id": s}, "required": ["collection", "id"]}),
            ToolSpec(app=self.id, name="create_resource", description="Create a resource.", input_schema={"type": "object", "properties": {"collection": s, "id": s, "data": o}, "required": ["collection", "id", "data"]}),
            ToolSpec(app=self.id, name="update_resource", description="Merge fields into a resource.", input_schema={"type": "object", "properties": {"collection": s, "id": s, "patch": o}, "required": ["collection", "id", "patch"]}),
            ToolSpec(app=self.id, name="delete_resource", description="Delete a resource.", input_schema={"type": "object", "properties": {"collection": s, "id": s}, "required": ["collection", "id"]}),
        ]

    async def tool_list_resources(self, collection: str, filter: dict[str, Any] | None = None) -> Any:  # noqa: A002
        col = self._check(await self._load(), collection, False)
        items = [{"id": k, **v} for k, v in sorted(col.items())]
        if filter:
            items = [x for x in items if all(x.get(k) == v for k, v in filter.items())]
        return items

    async def tool_get_resource(self, collection: str, id: str) -> Any:  # noqa: A002
        col = self._check(await self._load(), collection, False)
        if id not in col:
            raise ToolError(f"{collection}/{id} not found")
        return {"id": id, **col[id]}

    async def tool_create_resource(self, collection: str, id: str, data: dict[str, Any]) -> Any:  # noqa: A002
        doc = await self._load()
        col = self._check(doc, collection, True)
        if id in col:
            raise ToolError(f"{collection}/{id} already exists")
        col[id] = data
        await self._save(doc)
        return {"created": f"{collection}/{id}"}

    async def tool_update_resource(self, collection: str, id: str, patch: dict[str, Any]) -> Any:  # noqa: A002
        doc = await self._load()
        col = self._check(doc, collection, True)
        if id not in col:
            raise ToolError(f"{collection}/{id} not found")
        col[id].update(patch)
        await self._save(doc)
        return {"id": id, **col[id]}

    async def tool_delete_resource(self, collection: str, id: str) -> Any:  # noqa: A002
        doc = await self._load()
        col = self._check(doc, collection, True)
        if col.pop(id, None) is None:
            raise ToolError(f"{collection}/{id} not found")
        await self._save(doc)
        return {"deleted": f"{collection}/{id}"}
