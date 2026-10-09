"""State Adapter base: the verifier's typed view of one App's state (ARCHITECTURE §1.2).

Adapters are verifier-side only — the policy never receives one. Every adapter provides:

- ``fingerprint()``: canonical hash; equal hash <=> equal state, on every backend;
- ``snapshot()`` -> ``StateRef`` whose payload is in the content-addressed store;
- ``restore(ref)``: brings the live state back to ``ref`` and *verifies* the fingerprint
  (a mismatch raises ``SnapshotMismatch``, classified ``crucible:snapshot_mismatch``);
- ``diff(a, b)``: what changed between two captured states;
- ``query(q)``: adapter-specific read ops (documented in ``OPS``).
"""

from __future__ import annotations

import importlib.resources
import os
import shlex
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from crucible.core.schema.state import Query, QueryResult, StateDiff, StateRef
from crucible.core.store import Store
from crucible.world.exec import Exec, ExecError


class SnapshotMismatch(RuntimeError):
    failure_kind = "crucible:snapshot_mismatch"


def snippet(name: str) -> str:
    return importlib.resources.files("crucible.world.state.snippets").joinpath(f"{name}.py").read_text()


class StateAdapter(ABC):
    kind: ClassVar[str]
    OPS: ClassVar[dict[str, str]] = {}

    def __init__(self, app: str, exec_: Exec, params: dict[str, Any], store: Store):
        self.app = app
        self.exec = exec_
        self.params = dict(params)
        self.store = store

    @abstractmethod
    async def fingerprint(self) -> str: ...

    @abstractmethod
    async def snapshot(self) -> StateRef: ...

    @abstractmethod
    async def restore(self, ref: StateRef) -> None: ...

    @abstractmethod
    async def diff(self, a: StateRef, b: StateRef) -> StateDiff: ...

    async def query(self, q: Query) -> QueryResult:
        handler = getattr(self, f"q_{q.op}", None)
        if handler is None:
            return QueryResult(ok=False, error=f"{self.kind} adapter has no op {q.op!r}; ops: {sorted(self.OPS)}")
        t0 = time.monotonic()
        try:
            value = await handler(**q.args)
        except (ExecError, OSError, ValueError, KeyError) as e:
            return QueryResult(ok=False, error=f"{type(e).__name__}: {e}", duration_s=time.monotonic() - t0)
        if isinstance(value, QueryResult):
            return value
        return QueryResult(ok=True, value=value, duration_s=time.monotonic() - t0)

    # ---- helpers ----
    def _tmp_remote(self, suffix: str) -> str:
        return f"/tmp/.crucible_{self.app}_{os.getpid()}_{time.monotonic_ns()}{suffix}"

    async def _pull_to_store(self, remote: str) -> tuple[str, int]:
        """Move a payload produced inside the exec into the store; returns (key, size)."""
        if self.exec.is_local:
            key = self.store.put_file(remote)
            size = Path(remote).stat().st_size
            Path(remote).unlink(missing_ok=True)
            return key, size
        with tempfile.TemporaryDirectory() as d:
            local = Path(d) / "payload"
            await self.exec.pull_file(remote, local)
            await self.exec.run(f"rm -f {shlex.quote(remote)}")
            return self.store.put_file(local), local.stat().st_size

    async def _push_from_store(self, key: str, suffix: str, *, copy: bool = False) -> str:
        """Place a stored payload inside the exec; returns its path there (caller removes copies)."""
        if self.exec.is_local and not copy:
            return str(self.store.blob_path(key))
        if self.exec.is_local:
            import shutil

            tmp = self._tmp_remote(suffix)
            shutil.copyfile(self.store.blob_path(key), tmp)
            return tmp
        remote = self._tmp_remote(suffix)
        await self.exec.push_file(self.store.blob_path(key), remote)
        return remote
