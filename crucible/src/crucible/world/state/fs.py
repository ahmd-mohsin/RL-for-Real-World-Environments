"""File-tree State Adapters: ``fs`` (plain directory) and ``fs_git`` (git-aware managed set).

Params: ``root`` (path in the exec), ``include`` / ``exclude`` (globs with ``**``), ``git`` (fs_git only).
For ``fs_git`` the managed set is tracked + untracked-but-not-ignored files: build outputs a
``.gitignore`` hides are derived state, not state — exactly what the task's collect hook diffs.
"""

from __future__ import annotations

import json
from typing import Any

from crucible.core.schema.state import FileChange, StateDiff, StateRef
from crucible.world.state.base import SnapshotMismatch, StateAdapter, snippet

FS_SNIPPET = snippet("fs")


class FsAdapter(StateAdapter):
    kind = "fs"
    GIT = False
    OPS = {
        "read": "read a file: {path, max_bytes}",
        "exists": "path exists: {path}",
        "list": "managed paths, optionally filtered by glob: {pattern}",
        "grep": "regex over managed text files: {pattern, glob, max_hits}",
        "manifest": "full manifest {entries, fingerprint}",
        "run": "run a shell command in root: {cmd, timeout_s}",
    }

    def _args(self, **extra: Any) -> dict[str, Any]:
        root = self.params.get("root")
        if not root:
            raise ValueError(f"{self.kind} adapter for app {self.app!r} needs params.root")
        return {
            "root": root,
            "git": bool(self.params.get("git", self.GIT)),
            "include": list(self.params.get("include", [])),
            "exclude": list(self.params.get("exclude", [])),
            **extra,
        }

    async def _snip(self, op: str, **extra: Any) -> dict[str, Any]:
        return await self.exec.run_python(FS_SNIPPET, self._args(op=op, **extra))

    async def fingerprint(self) -> str:
        return (await self._snip("manifest"))["fingerprint"]

    async def snapshot(self) -> StateRef:
        out = self._tmp_remote(".tar.gz")
        m = await self._snip("pack", out=out)
        key, size = await self._pull_to_store(out)
        manifest_key = self.store.put_bytes(json.dumps(m["entries"], separators=(",", ":")).encode())
        return StateRef(
            app=self.app,
            adapter=self.kind,
            fingerprint=m["fingerprint"],
            blob=key,
            size_bytes=size,
            meta={"files": m["count"], "bytes": m["bytes"], "head": m.get("head"), "manifest": manifest_key,
                  **({"anomalies": m["anomalies"]} if m.get("anomalies") else {}),
                  **({"unpackable": m["unpackable"]} if m.get("unpackable") else {})},
        )

    async def restore(self, ref: StateRef) -> None:
        if ref.adapter != self.kind or ref.blob is None:
            raise ValueError(f"cannot restore {ref.adapter} ref (blob={ref.blob}) with {self.kind} adapter")
        if ref.meta.get("unpackable"):
            raise SnapshotMismatch(f"{self.app}: snapshot holds files that cannot be restored (sockets, devices, oversize): {ref.meta['unpackable'][:5]}")
        tar = await self._push_from_store(ref.blob, ".tar.gz")
        try:
            m = await self._snip("unpack", tar=tar)
        finally:
            if not self.exec.is_local:
                await self.exec.run(f"rm -f {tar}")
        if m["fingerprint"] != ref.fingerprint:
            raise SnapshotMismatch(f"{self.app}: restored fingerprint {m['fingerprint'][:12]} != snapshot {ref.fingerprint[:12]}")

    async def diff(self, a: StateRef, b: StateRef) -> StateDiff:
        ma = self._manifest_of(a)
        mb = self._manifest_of(b)
        changes: list[FileChange] = []
        for path in sorted(set(ma) | set(mb)):
            x, y = ma.get(path), mb.get(path)
            if x == y:
                continue
            if x is None:
                changes.append(FileChange(path=path, change="added", after=y[1]))
            elif y is None:
                changes.append(FileChange(path=path, change="removed", before=x[1]))
            elif x[1] == y[1]:
                changes.append(FileChange(path=path, change="mode", before=x[0], after=y[0]))
            else:
                changes.append(FileChange(path=path, change="modified", before=x[1], after=y[1]))
        return StateDiff(app=self.app, adapter=self.kind, before=a.fingerprint, after=b.fingerprint, files=changes)

    def _manifest_of(self, ref: StateRef) -> dict[str, tuple[str, str]]:
        key = ref.meta.get("manifest")
        if not key:
            raise ValueError(f"ref {ref.fingerprint[:12]} carries no manifest; diff needs captured snapshots")
        return {rel: (mode, h) for rel, mode, h in self.store.get_json(key)}

    # ---- query ops ----
    async def q_read(self, path: str, max_bytes: int = 200_000) -> Any:
        return await self._snip("read", path=path, max_bytes=max_bytes)

    async def q_exists(self, path: str) -> Any:
        return await self._snip("exists", path=path)

    async def q_list(self, pattern: str | None = None) -> Any:
        return (await self._snip("list", pattern=pattern))["paths"]

    async def q_grep(self, pattern: str, glob: str | None = None, max_hits: int = 200) -> Any:
        return await self._snip("grep", pattern=pattern, glob=glob, max_hits=max_hits)

    async def q_manifest(self) -> Any:
        return await self._snip("manifest")

    async def q_run(self, cmd: str, timeout_s: float = 600, env: dict[str, str] | None = None) -> Any:
        from crucible.core.schema.state import QueryResult

        out = await self.exec.run(cmd, cwd=self.params["root"], timeout_s=timeout_s, env=env)
        return QueryResult(ok=out.ok, exit_code=out.rc, stdout=out.stdout[-20000:], stderr=out.stderr[-20000:], duration_s=out.duration_s,
                           value={"timed_out": out.timed_out})


class FsGitAdapter(FsAdapter):
    kind = "fs_git"
    GIT = True
    OPS = {**FsAdapter.OPS, "git_diff": "git diff against a ref (default 'base'), excluding ignored files: {ref, stat}"}

    async def q_git_diff(self, ref: str = "base", stat: bool = False) -> Any:
        from crucible.core.schema.state import QueryResult

        # Diff the working tree (incl. untracked, minus ignored) against `ref` without touching the
        # repo's real index: a throwaway index file (GIT_INDEX_FILE) is used and deleted.
        root = self.params["root"]
        flag = "--stat" if stat else "--binary"
        cmd = (
            f'set -e; T=$(mktemp); trap "rm -f $T" EXIT; export GIT_INDEX_FILE=$T; '
            f"G='git -c safe.directory=* -C {root}'; $G read-tree {ref} && $G add -A -- . 2>/dev/null; $G diff --cached {flag} {ref}"
        )
        out = await self.exec.run(cmd, timeout_s=600)
        return QueryResult(ok=out.ok, exit_code=out.rc, stdout=out.stdout, stderr=out.stderr[-4000:], duration_s=out.duration_s)
