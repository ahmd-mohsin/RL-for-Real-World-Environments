"""File-tree State Adapters: ``fs`` (plain directory) and ``fs_git`` (git-aware managed set).

Params: ``root`` (path in the exec), ``include`` / ``exclude`` (globs with ``**``), ``git`` (fs_git only).
For ``fs_git`` the managed set is tracked + untracked-but-not-ignored files: build outputs a
``.gitignore`` hides are derived state, not state — exactly what the task's collect hook diffs.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any, ClassVar

from crucible.core.schema.state import FileChange, StateDiff, StateRef
from crucible.world.state.base import SnapshotMismatch, StateAdapter, snippet

FS_SNIPPET = snippet("fs")
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@^~{}-]{0,200}$")


def check_ref(ref: str) -> str:
    """A git revision safe to put on a command line (no options, no shell metacharacters)."""
    if not _REF.match(ref):
        raise ValueError(f"invalid git ref {ref!r}")
    return ref


def git_shell(root: str, git_dir: str | None = None, extra_env: str = "") -> str:
    """A git command line that reads no system/global config, runs no hooks or fsmonitor, and (with ``git_dir``)
    uses a root-owned copy of the repository instead of the policy-writable ``.git``."""
    q = shlex.quote
    loc = f"--git-dir={q(git_dir)} --work-tree={q(root)} -C {q(root)}" if git_dir else f"-C {q(root)}"
    return (f"env -i PATH={SAFE_PATH} HOME=/nonexistent LC_ALL=C GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 {extra_env} "
            f"git -c safe.directory='*' -c core.fsmonitor=false -c core.hooksPath=/dev/null -c core.quotepath=off {loc}")


def git_diff_command(root: str, ref: str = "base", stat: bool = False, git_dir: str | None = None) -> str:
    """Working tree (incl. untracked, minus ignored) vs ``ref`` through a throwaway index; the repo's index is untouched."""
    ref = shlex.quote(check_ref(ref))
    flag = "--stat" if stat else "--binary"
    g = git_shell(root, git_dir, 'GIT_INDEX_FILE="$T"')
    return (f'set -e; T=$(mktemp); trap \'rm -f "$T"\' EXIT; {g} read-tree {ref}; {g} add -A -- . 2>/dev/null || true; '
            f"{g} diff --cached --no-ext-diff --no-textconv {flag} {ref}")


class FsAdapter(StateAdapter):
    kind = "fs"
    GIT = False
    OPS: ClassVar[dict[str, str]] = {
        "read": "read a file: {path, max_bytes}",
        "exists": "path exists: {path}",
        "list": "managed paths, optionally filtered by glob: {pattern}",
        "grep": "regex over managed text files: {pattern, glob, paths, max_hits}",
        "manifest": "full manifest {entries, fingerprint}",
        "run": "run a shell command in root: {cmd, timeout_s}",
    }

    # Set by an isolating backend: a root-owned copy of the repo's .git taken before the policy ran, and the uid
    # that owns what a restore writes (the policy's, so its files stay writable after a reset).
    trusted_git_dir: str | None = None
    owner_uid: int | None = None

    def _args(self, **extra: Any) -> dict[str, Any]:
        root = self.params.get("root")
        if not root:
            raise ValueError(f"{self.kind} adapter for app {self.app!r} needs params.root")
        args = {
            "root": root,
            "git": bool(self.params.get("git", self.GIT)),
            "include": list(self.params.get("include", [])),
            "exclude": list(self.params.get("exclude", [])),
            **extra,
        }
        if self.trusted_git_dir:
            args["git_dir"] = self.trusted_git_dir
        if self.owner_uid is not None:
            args["owner"] = self.owner_uid
        return args

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

    async def q_grep(self, pattern: str, glob: str | None = None, max_hits: int = 200, paths: list[str] | None = None) -> Any:
        return await self._snip("grep", pattern=pattern, glob=glob, max_hits=max_hits, paths=paths)

    async def q_manifest(self) -> Any:
        return await self._snip("manifest")

    async def q_run(self, cmd: str, timeout_s: float = 600, env: dict[str, str] | None = None, user: str | None = None) -> Any:
        from crucible.core.schema.state import QueryResult

        out = await self.exec.run(cmd, cwd=self.params["root"], timeout_s=timeout_s, env=env, **({"user": user} if user else {}))
        return QueryResult(ok=out.ok, exit_code=out.rc, stdout=out.stdout[-20000:], stderr=out.stderr[-20000:], duration_s=out.duration_s,
                           value={"timed_out": out.timed_out})


class FsGitAdapter(FsAdapter):
    kind = "fs_git"
    GIT = True
    OPS: ClassVar[dict[str, str]] = {**FsAdapter.OPS, "git_diff": "git diff against a ref (default 'base'), excluding ignored files: {ref, stat}",
                                     "block_changed": "whether the block an anchor regex starts differs from the base ref's: {path, anchor, ref}"}

    async def q_block_changed(self, path: str, anchor: str, ref: str = "base") -> Any:
        """The same answer ``FileStateView`` gives offline: ``{changed, found}`` for the block at ``anchor`` in ``path``
        now vs at ``ref`` (git show through the trusted git dir)."""
        from crucible.core.schema.state import QueryResult
        from crucible.core.textblock import block

        if path.startswith("/") or ".." in path.split("/"):
            return QueryResult(ok=False, error=f"bad path {path!r}")
        try:
            now = (await self.q_read(path)).get("text")
        except Exception:  # noqa: BLE001 - no such file now
            now = None
        g = git_shell(self.params["root"], self.trusted_git_dir)
        out = await self.exec.run(f"{g} show {shlex.quote(check_ref(ref) + ':' + path)}", timeout_s=120)
        base = out.stdout if out.ok else None
        if now is None or base is None:
            return QueryResult(ok=True, value={"changed": now != base, "found": False})
        b0, b1 = block(base, anchor), block(now, anchor)
        return QueryResult(ok=True, value={"changed": b0 != b1, "found": b1 is not None})

    async def q_git_diff(self, ref: str = "base", stat: bool = False) -> Any:
        from crucible.core.schema.state import QueryResult

        try:
            cmd = git_diff_command(self.params["root"], ref, stat, self.trusted_git_dir)
        except ValueError as e:
            return QueryResult(ok=False, error=str(e))
        out = await self.exec.run(cmd, timeout_s=600)
        return QueryResult(ok=out.ok, exit_code=out.rc, stdout=out.stdout, stderr=out.stderr[-4000:], duration_s=out.duration_s)
