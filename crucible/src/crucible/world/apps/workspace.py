"""Workspace and git-repo Apps: files + (optionally) a shell, as most coding harnesses expose them.

Tool surface (names follow common agent harness conventions):
  list_dir(path=".", depth=1), read_file(path, offset=1, limit=400), write_file(path, content),
  edit_file(path, old_string, new_string, replace_all=False), search(pattern, glob=None, max_hits=100),
  shell(command, timeout_s=600)           [only when params.shell is true]
  git_status(), git_diff(ref="base")      [GitRepoApp]

Params: ``shell`` (bool), ``read_only`` (bool), ``seed_dir`` (host dir copied in at provision),
``init_git`` (git init + commit + tag ``base`` after seeding), ``max_output`` (chars).
Paths are confined to ``state.params.root``; the shell is confined only by the backend's sandbox,
so ``shell`` on the local backend requires ``params.allow_unsafe_local_shell``.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from crucible.core.schema.actions import ToolSpec
from crucible.world.apps.base import App, PolicyToolError, ToolError, confine, push_seed_tree
from crucible.world.state.base import snippet

WS_SNIPPET = snippet("ws")
FS_SNIPPET = snippet("fs")

_S = {"type": "string"}
_I = {"type": "integer"}
_B = {"type": "boolean"}


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


# errors a well-formed call of the policy caused: its old_string, its regex, its path (fault ``policy``, never the World's)
POLICY_CAUSED = re.compile(r"old_string (not found|occurs \d+ times)|re\.error|re\._constants\.error|re\.PatternError|unterminated character set|"
                           r"nothing to repeat|bad escape|missing \)|unbalanced parenthesis|unterminated subpattern|"
                           r"FileNotFoundError|IsADirectoryError|NotADirectoryError|No such file or directory|is a directory|Not a directory")


def _last_error_line(text: str) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return (lines[-1] if lines else "error").removeprefix("python snippet failed: ")[:2000]


class WorkspaceApp(App):
    template = "common/workspace"

    async def stop(self) -> None:
        """End what the policy's commands left running (local backend; a container backend ends with its container)."""
        kill = getattr(getattr(self, "policy_exec", None), "kill_sessions", None)
        if callable(kill):
            kill()

    @property
    def root(self) -> str:
        return self.state_params["root"]

    @property
    def max_output(self) -> int:
        return int(self.spec.params.get("max_output", 30000))

    async def provision(self) -> None:
        seed = self.spec.params.get("seed_dir")
        await self.ctx.exec.run(f"mkdir -p {shlex.quote(self.root)}")
        if seed:
            await push_seed_tree(self, self.ctx.host_seed(seed), self.root)  # a tarball unique to this World (M14 campaign B3)
        if self.spec.params.get("init_git"):
            out = await self.ctx.exec.run(
                "git init -q -b main . && git add -A && git -c user.email=crucible@local -c user.name=crucible commit -qm base --allow-empty && git tag -f base",
                cwd=self.root,
            )
            if not out.ok:
                raise RuntimeError(f"git init in {self.id} failed: {out.stderr[-500:]}")

    async def list_tools(self) -> list[ToolSpec]:
        ro = bool(self.spec.params.get("read_only"))
        tools = [
            ToolSpec(app=self.id, name="list_dir", read_only=True, description="List files and directories (relative to the workspace root).",
                     input_schema=_obj({"path": _S, "depth": _I}, [])),
            ToolSpec(app=self.id, name="read_file", read_only=True, description="Read a text file with line numbers; use offset/limit for long files.",
                     input_schema=_obj({"path": _S, "offset": _I, "limit": _I}, ["path"])),
            ToolSpec(app=self.id, name="search", read_only=True, description="Regex search over workspace text files; returns path:line:text hits.",
                     input_schema=_obj({"pattern": _S, "glob": _S, "max_hits": _I}, ["pattern"])),
        ]
        if not ro:
            tools += [
                ToolSpec(app=self.id, name="write_file", description="Create or overwrite a file with the given content.",
                         input_schema=_obj({"path": _S, "content": _S}, ["path", "content"])),
                ToolSpec(app=self.id, name="edit_file", description="Replace an exact, unique string in a file (set replace_all to replace every occurrence).",
                         input_schema=_obj({"path": _S, "old_string": _S, "new_string": _S, "replace_all": _B}, ["path", "old_string", "new_string"])),
            ]
        if self.spec.params.get("shell"):
            tools.append(ToolSpec(app=self.id, name="shell", description="Run a bash command in the workspace root; returns exit code and output.",
                                  input_schema=_obj({"command": _S, "timeout_s": _I}, ["command"])))
        return tools

    async def _ws(self, op: str, **kw: Any) -> dict[str, Any]:
        try:
            return await self.policy_exec.run_python(WS_SNIPPET, {"op": op, **kw})
        except Exception as e:  # file errors are the policy's problem; the traceback (snippet paths) is not its business
            msg = _last_error_line(str(e))
            if POLICY_CAUSED.search(str(e)):
                raise PolicyToolError(msg) from e
            raise ToolError(msg) from e

    async def tool_list_dir(self, path: str = ".", depth: int = 1) -> Any:
        return "\n".join((await self._ws("list", path=confine(self.root, path), depth=min(max(depth, 1), 5)))["entries"]) or "(empty)"

    async def tool_read_file(self, path: str, offset: int = 1, limit: int = 400) -> Any:
        r = await self._ws("read", path=confine(self.root, path), offset=offset, limit=min(limit, 2000))
        text = r["text"]
        if r.get("more"):
            text += f"\n... ({r['total_lines']} lines total; continue with offset={offset + limit})"
        return text

    async def tool_write_file(self, path: str, content: str) -> Any:
        r = await self._ws("write", path=confine(self.root, path), content=content)
        return f"wrote {r['bytes']} bytes to {path}"

    async def tool_edit_file(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> Any:
        r = await self._ws("edit", path=confine(self.root, path), old=old_string, new=new_string, all=replace_all)
        return f"replaced {r['replacements']} occurrence(s) in {path}"

    async def tool_search(self, pattern: str, glob: str | None = None, max_hits: int = 100) -> Any:
        args = {"op": "grep", "root": self.root, "git": False, "include": [], "exclude": [".git/**"], "pattern": pattern, "glob": glob, "max_hits": min(max_hits, 500)}
        try:
            r = await self.policy_exec.run_python(FS_SNIPPET, args)
        except Exception as e:
            msg = f"search failed: {_last_error_line(str(e))}"
            raise (PolicyToolError if POLICY_CAUSED.search(str(e)) else ToolError)(msg) from e
        lines = [f"{p}:{n}:{t}" for p, n, t in r["hits"]]
        return "\n".join(lines) + ("\n... (truncated)" if r["truncated"] else "") if lines else "no matches"

    async def tool_shell(self, command: str, timeout_s: int = 600) -> Any:
        if not self.spec.params.get("shell"):
            raise ToolError("shell is disabled for this workspace")
        if self.policy_exec.is_local and not self.spec.params.get("allow_unsafe_local_shell"):
            raise ToolError("shell refused: the local backend has no sandbox (set allow_unsafe_local_shell for trusted policies only)")
        out = await self.policy_exec.run(command, cwd=self.root, timeout_s=self.deadline(timeout_s, self.spec.params.get("max_timeout_s", 3600)))
        text = (out.stdout + (("\n[stderr]\n" + out.stderr) if out.stderr.strip() else ""))
        if len(text) > self.max_output:
            text = text[: self.max_output // 2] + f"\n... [{len(text) - self.max_output} chars elided] ...\n" + text[-self.max_output // 2 :]
        status = "TIMEOUT" if out.timed_out else f"exit {out.rc}"
        return f"[{status}]\n{text}"


class GitRepoApp(WorkspaceApp):
    template = "common/git_repo"

    async def list_tools(self) -> list[ToolSpec]:
        return [
            *await super().list_tools(),
            ToolSpec(app=self.id, name="git_status", read_only=True, description="git status --short of the repository.", input_schema=_obj({}, [])),
            ToolSpec(app=self.id, name="git_diff", read_only=True, description="Diff of the working tree (including new files) against a ref (default: base).",
                     input_schema=_obj({"ref": _S, "stat": _B}, [])),
        ]

    async def tool_git_status(self) -> Any:
        out = await self.policy_exec.run("git -c core.quotepath=off status --short", cwd=self.root, timeout_s=120)
        return out.stdout or "clean"

    async def tool_git_diff(self, ref: str = "base", stat: bool = False) -> Any:
        from crucible.world.state.fs import git_diff_command

        try:  # the policy's own view of its own repo, run as the policy: its hooks and config are its business
            cmd = git_diff_command(self.root, ref, stat)
        except ValueError as e:
            raise ToolError(str(e)) from e
        out = await self.policy_exec.run(cmd, timeout_s=600)
        if not out.ok:
            raise ToolError(_last_error_line(out.stderr or out.stdout or "git diff failed"))
        text = out.stdout or "(no changes)"
        return text if len(text) <= self.max_output else text[: self.max_output] + "\n... (truncated)"
