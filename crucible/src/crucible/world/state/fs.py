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


def plain_git() -> str:
    """A git command line with no system or global config, for reading one config file (``git config --file``)."""
    return f"env -i PATH={SAFE_PATH} HOME=/nonexistent LC_ALL=C GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null git"


def filter_overrides_sh(git_dir: str = ".git", names: str = '"$W/f"', git: str = "git") -> str:
    """Shell that sets the positional parameters to ``-c`` overrides making each filter driver of the repository's own
    config files a pass-through (``verify.collect.filter_flags``, read inside the exec, since the repository may be in a
    container), and exits 3 on a driver name no ``-c`` can name exactly. ``names`` is a scratch file. One copy for the
    in-container snapshot (``verify.collect.snapshot_command``) and the verifier view (``FsGitAdapter.q_git_diff``)."""
    return (f": > {names}; for c in {git_dir}/config {git_dir}/config.worktree; do if [ -f \"$c\" ]; then {git} config --file \"$c\" --includes --name-only "
            f"--get-regexp '^filter[.]' >> {names} 2>/dev/null || true; fi; done; set --; "
            "while IFS= read -r k; do n=$(printf '%s' \"$k\" | sed 's/^filter[.]//; s/[.][^.]*$//'); "
            "case \"$n\" in ''|*[!A-Za-z0-9._+-]*) echo \"a filter driver git cannot override: $n\" >&2; exit 3;; esac; "
            f"set -- \"$@\" -c \"filter.$n.clean=\" -c \"filter.$n.smudge=\" -c \"filter.$n.process=\" -c \"filter.$n.required=false\"; done < {names}; ")


# the exit status of a refusal by ``repo_guard_sh`` or ``filter_overrides_sh`` (git itself exits 0, 1, 128 or 129)
REFUSED_RC = 3


def repo_guard_sh(root: str, git: str | None = None) -> str:
    """Shell that exits ``REFUSED_RC`` with the reason when the repository at ``root`` was redirected: the rules of
    ``verify.collect.repo_hazards``, read inside the exec. ``.git`` a symlink or a ``gitdir:`` file, a ``commondir``,
    ``core.bare`` set, or ``core.worktree`` naming another tree (resolved from ``.git``, as git resolves it): git would read
    another repository or tree. A missing ``.git`` passes (git then fails, as before)."""
    git = git or plain_git()
    g, r, q = shlex.quote(root) + "/.git", REFUSED_RC, shlex.quote(root)
    no = "echo 'the repository is refused:"
    return (f"if [ -L {g} ]; then {no} the workspace .git is a symlink (the repository was redirected)' >&2; exit {r}; fi; "
            f"if [ -e {g} ] && [ ! -d {g} ]; then {no} the workspace .git is a file, not the repository (a gitdir redirect)' >&2; exit {r}; fi; "
            f"if [ -e {g}/commondir ]; then {no} the workspace repository has a commondir redirect' >&2; exit {r}; fi; "
            f"for c in {g}/config {g}/config.worktree; do if [ -f \"$c\" ]; then "
            f"if [ \"$({git} config --file \"$c\" --includes --bool --get core.bare 2>/dev/null)\" = true ]; then "
            f"{no} core.bare is set in the workspace repository' >&2; exit {r}; fi; "
            f"w=$({git} config --file \"$c\" --includes --get core.worktree 2>/dev/null || true); "
            f"if [ -n \"$w\" ]; then case \"$w\" in /*) p=\"$w\";; *) p={g}/\"$w\";; esac; "
            f"if [ \"$(cd \"$p\" 2>/dev/null && pwd -P)\" != \"$(cd {q} && pwd -P)\" ]; then "
            f"{no} core.worktree points outside the workspace' >&2; exit {r}; fi; fi; fi; done; ")


def refusal(out: Any) -> str | None:
    """The reason a guarded git command was refused (its last stderr line), or None."""
    return ((out.stderr or "").strip().splitlines() or ["the repository is refused"])[-1] if out.rc == REFUSED_RC else None


def git_diff_command(root: str, ref: str = "base", stat: bool = False, git_dir: str | None = None, *, config: dict[str, str] | None = None,
                     guard: bool = False) -> str:
    """Working tree (incl. untracked, minus ignored) vs ``ref`` through a throwaway index; the repo's index is untouched.
    ``config``: extra settings at command scope (``GIT_CONFIG_COUNT``, above the repository's own config): the checkpoint
    sidecar passes empty commands for the repository's filter drivers (harness-reward audit R6, 2026-10-28). ``guard``
    (the repository is the agent's own, no trusted copy stands in for it: ``FsGitAdapter.q_git_diff``): a redirected
    repository is refused (``repo_guard_sh``) and its own filter drivers are pass-throughs (``filter_overrides_sh``), both
    read inside the exec. Without ``guard`` the command is unchanged."""
    ref = shlex.quote(check_ref(ref))
    flag = "--stat" if stat else "--binary"
    cfg = dict(config or {})
    env = "".join(f" GIT_CONFIG_KEY_{i}={shlex.quote(k)} GIT_CONFIG_VALUE_{i}={shlex.quote(v)}" for i, (k, v) in enumerate(cfg.items()))
    g = git_shell(root, git_dir, 'GIT_INDEX_FILE="$T"' + (f" GIT_CONFIG_COUNT={len(cfg)}{env}" if cfg else ""))
    if not guard:
        return (f'set -e; T=$(mktemp); trap \'rm -f "$T"\' EXIT; {g} read-tree {ref}; {g} add -A -- . 2>/dev/null || true; '
                f"{g} diff --cached --no-ext-diff --no-textconv {flag} {ref}")
    if git_dir:
        raise ValueError("guard is for the agent's own repository; a trusted git dir needs none")
    pre = repo_guard_sh(root) + filter_overrides_sh(shlex.quote(root) + "/.git", '"$F"', plain_git())
    g = f'{g} "$@"'
    return (f'set -e; T=$(mktemp); F=$(mktemp); trap \'rm -f "$T" "$F"\' EXIT; {pre}{g} read-tree {ref}; {g} add -A -- . 2>/dev/null || true; '
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
        # without a trusted copy the repository is the agent's (as in q_git_diff): a redirected one would answer from another
        # repository's ``ref``, and its own textconv driver would run and rewrite the base text
        pre, opt = ("", "") if self.trusted_git_dir else (repo_guard_sh(self.params["root"]), "--no-textconv ")
        out = await self.exec.run(f"{pre}{g} show {opt}{shlex.quote(check_ref(ref) + ':' + path)}", timeout_s=120)
        if pre and (why := refusal(out)) is not None:
            return QueryResult(ok=False, exit_code=out.rc, stderr=out.stderr[-4000:], error=why)
        base = out.stdout if out.ok else None
        if now is None or base is None:
            return QueryResult(ok=True, value={"changed": now != base, "found": False})
        b0, b1 = block(base, anchor), block(now, anchor)
        return QueryResult(ok=True, value={"changed": b0 != b1, "found": b1 is not None})

    async def q_git_diff(self, ref: str = "base", stat: bool = False) -> Any:
        """The verifier's view of the working tree against ``ref``. Without a ``trusted_git_dir`` the repository is the
        agent's own (2026-10-28, the item harness-reward audit R6 left open): its filter drivers ran at ``git add`` (a
        command the policy chose, and any content it liked in the diff) and a redirected ``.git`` made git diff another
        repository or tree. The drivers are now pass-throughs and a redirected repository is refused
        (``git_diff_command(guard=True)``): an error, never a diff."""
        from crucible.core.schema.state import QueryResult

        try:
            cmd = git_diff_command(self.params["root"], ref, stat, self.trusted_git_dir, guard=self.trusted_git_dir is None)
        except ValueError as e:
            return QueryResult(ok=False, error=str(e))
        out = await self.exec.run(cmd, timeout_s=600)
        why = refusal(out) if self.trusted_git_dir is None else None
        return QueryResult(ok=out.ok, exit_code=out.rc, stdout=out.stdout if why is None else "", stderr=out.stderr[-4000:], duration_s=out.duration_s,
                           error=why)
