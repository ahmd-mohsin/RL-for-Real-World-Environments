"""The replay test (DESIGN §5.3.2, M5 gate S1): materialize the state at step boundaries and reproduce the outcomes the
trace recorded. A trace that does not replay is **quarantined**: its states cannot be trusted, so no milestone is
mined from it and no credit is computed on it (Patronus's "dropped dependency" risk; GitSwarm's exact-source
repeats show why every replayed metric needs a tolerance band).

What "the recorded outcome" is depends on the trace:

- **agent sessions with editor calls** (``EditReplayer``, generic): re-apply every recorded edit to the base snapshot
  and compare the reconstructed final files with the recorded final patch. An edit that no longer applies, or a
  file the final patch changed that no replayable call touched (a ``sed -i`` or a script did it), is a divergence;
- **CI runs** (pack replayers): rebuild the state the CI tested and re-run the job, comparing its conclusion;
- **World episodes / event logs** (pack replayers): re-issue the recorded actions into a fresh World and compare the
  recorded check events and the final state.

Replayers are registry entries (kind ``replayer``) keyed by trace family prefix, like signal mappers.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.core.schema.trace import Trace
from crucible.workflow.filestate import EditError, FileState, parse_patch


class ReplayCheck(CrucibleModel):
    step: int | None = None  # trace seq or step index where the outcome was recorded
    what: str  # e.g. "final_patch:src/a.py", "ci:Formatting", "check:refund_over_50"
    recorded: Any = None
    replayed: Any = None
    match: bool
    tolerance: str | None = None  # how the comparison tolerates noise (exact, whitespace, ±5%)


class ReplayResult(CrucibleModel):
    trace_id: str
    replayer: str
    ok: bool
    quarantined: bool = False
    reason: str | None = None
    steps_replayed: int = 0
    checks: list[ReplayCheck] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Replayer(Protocol):
    name: str

    def replay(self, trace: Trace) -> ReplayResult: ...


def _norm(text: str | None, mode: str) -> str | None:
    if text is None or mode == "exact":
        return text
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


class EditReplayer:
    """Generic edit replay for agent sessions (ATIF / capture / World traces).

    ``base``: path -> base content (None if the file did not exist), e.g. a ``git show <base>:<path>`` reader;
    ``final_patch``: the recorded final state as a unified diff (mini-swe-agent ``model.patch``, an OpenHands
    ``model_patch``, a PR's diff); ``path_of``: maps a tool call's path argument to a repo-relative path (agents use
    absolute container paths such as ``/workspace/<repo>/src/x.py``); ``compare``: ``exact`` or ``whitespace``
    (trailing whitespace and final blank lines ignored, since a final patch is often produced by ``git diff`` of
    files the editor wrote with a different trailing newline).
    """

    name = "edit_replay"

    def __init__(self, base: Callable[[str], str | None], final_patch: str, *, path_of: Callable[[str], str | None] | None = None,
                 compare: str = "whitespace", ignore: Callable[[str], bool] | None = None, cwd: str = "/"):
        self.base, self.final_patch, self.compare, self.cwd = base, final_patch, compare, cwd
        self.path_of = path_of or (lambda p: p.lstrip("/"))
        self.ignore = ignore or (lambda p: False)

    def states(self, trace: Trace) -> tuple[list[tuple[int, FileState]], list[dict[str, Any]]]:
        """File states after every state-changing call (seq, state), and the calls that could not be replayed."""
        from crucible.workflow.actions import SHELL_TOOLS, classify, command_of
        from crucible.workflow.shellops import apply_shell

        fs = FileState(base=self.base)
        cwd = self.cwd
        out: list[tuple[int, FileState]] = []
        problems: list[dict[str, Any]] = []
        for e in trace.events:
            if e.type != "tool_call":
                continue
            tool = str(e.data.get("tool") or e.data.get("function") or "")
            base = tool.rsplit(".", 1)[-1]
            args = e.data.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    problems.append({"seq": e.seq, "tool": tool, "problem": "arguments truncated or not JSON"})
                    continue
            args = args or {}
            try:
                if base in ("str_replace_editor", "str_replace_based_edit_tool"):
                    if args.get("command") in (None, "view"):
                        continue
                    path = self.path_of(str(args.get("path") or ""))
                    if path is None:
                        continue
                    fs.str_replace_editor(args, path)
                elif base in ("Write", "Edit", "MultiEdit"):
                    path = self.path_of(str(args.get("file_path") or ""))
                    if path is None:
                        continue
                    fs.claude_edit(base, args, path)
                elif base in ("write_file", "edit_file", "create_file"):
                    path = self.path_of(str(args.get("path") or ""))
                    if path is None:
                        continue
                    if base == "edit_file":
                        fs.claude_edit("Edit", {"old_string": args.get("old_string"), "new_string": args.get("new_string"),
                                                "replace_all": args.get("replace_all")}, path)
                    else:
                        fs.write(path, args.get("content") or args.get("file_text") or "")
                elif base in SHELL_TOOLS:
                    cmd = command_of(args)
                    eff = apply_shell(fs, cmd, cwd=cwd, path_of=self.path_of)
                    cwd = eff.cwd  # the harness keeps one shell session: cd persists across calls
                    if eff.unreplayable:
                        problems.append({"seq": e.seq, "tool": tool, "problem": "shell: " + eff.unreplayable[0], "command": cmd[:200]})
                    if not eff.applied:
                        continue
                elif classify(tool, args) == "edit":
                    problems.append({"seq": e.seq, "tool": tool, "problem": "unknown editor", "command": command_of(args)[:200]})
                    continue
                else:
                    continue
            except EditError as err:
                problems.append({"seq": e.seq, "tool": tool, "problem": str(err)[:200]})
                continue
            out.append((e.seq, fs.copy()))
        return out, problems

    def replay(self, trace: Trace) -> ReplayResult:
        states, problems = self.states(trace)
        final = states[-1][1] if states else FileState(base=self.base)
        want = FileState(base=self.base)
        try:
            want_paths = [p for p in want.apply_patch(self.final_patch) if not self.ignore(p)]
        except EditError as err:
            return ReplayResult(trace_id=trace.trace_id, replayer=self.name, ok=False, quarantined=True,
                                reason=f"recorded final patch does not apply to the base: {err}", steps_replayed=len(states))
        checks = []
        for p in sorted(set(want_paths) | {p for p in final.touched if not self.ignore(p)}):
            rec = want.files.get(p) if p in want.files else self.base(p)
            got = final.files.get(p) if p in final.files else self.base(p)
            checks.append(ReplayCheck(what=f"final:{p}", recorded=None if rec is None else len(rec), replayed=None if got is None else len(got),
                                      match=_norm(rec, self.compare) == _norm(got, self.compare), tolerance=self.compare))
        bad = [c.what for c in checks if not c.match]
        ok = not bad
        reason = None if ok else f"{len(bad)} file(s) differ from the recorded final state: {bad[:5]}"
        if not ok and problems:
            reason += f"; {len(problems)} call(s) not replayable (first: {problems[0]['problem']} at seq {problems[0]['seq']})"
        return ReplayResult(trace_id=trace.trace_id, replayer=self.name, ok=ok, quarantined=not ok, reason=reason, steps_replayed=len(states),
                            checks=checks, meta={"unreplayable": problems[:20], "n_unreplayable": len(problems),
                                                 "files_in_final_patch": len(parse_patch(self.final_patch))})


def replayer_for(family: str | None) -> Any:
    from crucible.core.registry import get_registry

    reg = get_registry()
    names = sorted((n for n in reg.names("replayer") if family and (family == n or family.startswith(n + "/"))), key=len, reverse=True)
    return reg.get("replayer", names[0]) if names else None
