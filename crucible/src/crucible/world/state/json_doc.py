"""JSON-document State Adapter (``json_doc``): state is one JSON file (e.g. a mocked HTTP API's data).

Fingerprint = sha256 of canonical JSON (sorted keys, compact). Diff reports changed JSON paths.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from crucible.core.ids import sha256_text
from crucible.core.schema.state import StateDiff, StateRef
from crucible.world.state.base import SnapshotMismatch, StateAdapter


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _paths(obj: Any, prefix: str = "") -> dict[str, str]:
    if isinstance(obj, dict) and obj:
        out: dict[str, str] = {}
        for k, v in obj.items():
            out.update(_paths(v, f"{prefix}/{k}"))
        return out
    if isinstance(obj, list) and obj:
        out = {}
        for i, v in enumerate(obj):
            out.update(_paths(v, f"{prefix}/{i}"))
        return out
    return {prefix or "/": canonical(obj)}


class JsonDocAdapter(StateAdapter):
    kind = "json_doc"
    OPS: ClassVar[dict[str, str]] = {"get": "value at a /-separated path: {path}", "doc": "the whole document"}

    async def _load(self) -> Any:
        try:
            return json.loads(await self.exec.read_text(self.params["path"]))
        except FileNotFoundError:
            return None

    async def fingerprint(self) -> str:
        return sha256_text(canonical(await self._load()))

    async def snapshot(self) -> StateRef:
        doc = await self._load()
        text = canonical(doc)
        key = self.store.put_bytes(text.encode())
        return StateRef(app=self.app, adapter=self.kind, fingerprint=sha256_text(text), blob=key, size_bytes=len(text), meta={"manifest": key})

    async def restore(self, ref: StateRef) -> None:
        assert ref.blob is not None
        text = self.store.get_bytes(ref.blob).decode()
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps(json.loads(text), indent=2, sort_keys=True))
        try:
            await self.exec.push_file(Path(f.name), self.params["path"])
        finally:
            Path(f.name).unlink(missing_ok=True)
        fp = await self.fingerprint()
        if fp != ref.fingerprint:
            raise SnapshotMismatch(f"{self.app}: restored json {fp[:12]} != {ref.fingerprint[:12]}")

    async def diff(self, a: StateRef, b: StateRef) -> StateDiff:
        pa = _paths(json.loads(self.store.get_bytes(a.blob))) if a.blob else {}
        pb = _paths(json.loads(self.store.get_bytes(b.blob))) if b.blob else {}
        changed = sorted(p for p in set(pa) | set(pb) if pa.get(p) != pb.get(p))
        return StateDiff(app=self.app, adapter=self.kind, before=a.fingerprint, after=b.fingerprint, detail={"changed_paths": changed})

    async def q_get(self, path: str) -> Any:
        node = await self._load()
        for part in [p for p in path.split("/") if p]:
            node = node[int(part)] if isinstance(node, list) else node[part]
        return node

    async def q_doc(self) -> Any:
        return await self._load()
