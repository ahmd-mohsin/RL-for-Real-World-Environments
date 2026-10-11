"""What a tool call does, from its tool name and arguments: the vocabulary segmentation and the tracker share.

Agent harnesses expose very different tools (mini-swe-agent: one ``bash``; Claude Code: Read / Edit / Bash / Grep;
OpenHands: ``str_replace_editor`` + ``execute_bash``; a Crucible World: ``ws.read_file`` / ``ws.shell`` / app tools),
so a call is classified by what it *does*:

  investigate  reads state without changing it (cat, grep, ls, git diff, read_file, get_ticket)
  edit         changes files (Edit/Write, str_replace_editor create/str_replace, sed -i, cat > f, apply_patch)
  build        compiles or installs (cmake, make, ninja, nvcc, pip install, setup.py, cargo build, go build)
  test         runs tests or checks (pytest, ctest, go test, npm test, run_ci, compute-sanitizer)
  run          executes the program under work (python script.py, ./bin, benchmarks)
  communicate  talks to someone (send_message, send_email, reply)
  act          changes an App's state through its tools (set_status, issue_refund, post_invoice)
  submit       ends the attempt (submit, done, COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT)

A shell command chaining several kinds takes the most consequential one (edit > build > test > run > investigate),
because that is what moved the state. The tracker evaluates medium-cost milestones at build/test calls (DESIGN §5.3).
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from crucible.core.agent_events import EDIT_TOOLS

ActionKind = Literal["investigate", "edit", "build", "test", "run", "communicate", "act", "submit", "other"]
ORDER: tuple[ActionKind, ...] = ("submit", "edit", "build", "test", "run", "act", "communicate", "investigate", "other")

SUBMIT = re.compile(r"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|MINI_SWE_AGENT_FINAL_OUTPUT|\bsubmit\b\s*$|^\s*(?:exit|finish)\s*$", re.MULTILINE)
SHELL_EDIT = re.compile(  # changes files in place: an edit wherever it happens
    r"\bsed\s+(?:-[a-zA-Z]*i|--in-place)|\bperl\s+-[a-zA-Z]*i|\bpatch\b\s|\bgit\s+(?:apply|am|checkout\s+--|restore|stash\s+pop|cherry-pick|revert|reset\s+--hard)"
    r"|\bclang-format\s+-i\b|\bblack\s(?!--check)|\bruff\s+format\b|\bisort\s|\.write_text\(|\.write\(|open\([^)]*['\"][wa]\+?['\"]")
REDIRECT = re.compile(r"(?:^|[^0-9&<>])>>?\s*(?!&)(['\"]?)([^\s;&|<>'\"]+)\1")  # > file / >> file (not 2>&1)
FILE_OPS = re.compile(r"(?:^|[\s;&|(])(?:git\s+mv|git\s+rm|mv|cp|rm|mkdir|touch|truncate|ln|install)\s+((?:-[\w-]+\s+)*)([^\n;&|]+)")
SCRATCH = ("/tmp", "/dev/", "/var/tmp", "/proc", "$TMPDIR")


BUILD = re.compile(r"\b(?:cmake|make|ninja|nvcc|g\+\+|gcc|clang\+\+|clang|cc|ld|meson|bazel|cargo\s+build|go\s+build|mvn|gradle|tsc|npm\s+(?:install|ci|run\s+build)"
                   r"|pip3?\s+install|uv\s+pip\s+install|python3?\s+-m\s+pip\s+install|setup\.py|python3?\s+-m\s+build|maturin|conda\s+install)\b")
TEST = re.compile(r"\b(?:pytest|py\.test|ctest|unittest|nosetests|tox|go\s+test|cargo\s+test|npm\s+(?:run\s+)?test|yarn\s+test|jest|mocha|rspec|phpunit"
                  r"|compute-sanitizer|valgrind|run_ci|run_tests?|make\s+(?:check|test)|ninja\s+test|doctest|\w*_tests?\b(?:\s|$))")
RUN = re.compile(r"(?:^|[\s;&|(])(?:python3?|node|\./[\w./-]+|java|ruby|bash\s+[\w./-]+\.sh|sh\s+[\w./-]+\.sh|nsys|ncu|perf\s|time\s|hyperfine)\b")
READ_TOOLS = {"Read", "Grep", "Glob", "LS", "read_file", "list_dir", "search", "view", "find_file", "search_file", "search_dir", "open",
              "git_status", "git_diff", "list_tickets", "get_ticket", "list_inbox", "read_email", "search_docs", "read_doc", "query", "WebFetch", "WebSearch"}
COMM_TOOLS = {"send_message", "send_email", "reply", "post_comment", "notify"}
SUBMIT_TOOLS = {"submit", "finish", "done", "task_complete", "attempt_completion"}
SHELL_TOOLS = {"bash", "Bash", "shell", "execute_bash", "run_command", "terminal", "exec", "run", "cmd"}
_SEGMENT = re.compile(r"&&|\|\||;|\n")
_CD = re.compile(r"^\s*(?:\(?\s*)cd\s+(['\"]?)([^\s'\";&|]+)\1")


def _redirect_targets(segment: str) -> list[str]:
    """Files a shell segment redirects output into; quote-aware, so ``python -c "print(1 >= 2)"`` has none."""
    from crucible.workflow.shellops import _tokens

    toks = _tokens(segment)
    if toks is None:
        return [r.group(2) for r in REDIRECT.finditer(segment)]
    return [toks[i + 1] for i, t in enumerate(toks[:-1]) if t in (">", ">>") and toks[i + 1] not in ("&",) and not toks[i + 1].startswith("&")]


def _targets_outside_scratch(command: str) -> tuple[bool, bool]:
    """(writes a workspace file via redirect, touches workspace files via mv/cp/rm/mkdir/...). Scratch paths (/tmp,
    /dev) do not count; a relative path is resolved against the last ``cd`` in the chain (here-doc bodies are
    skipped), so ``cd /tmp/probe && cat > p.cpp`` is scratch work."""
    body = re.sub(r"<<-?\s*(['\"]?)(\w+)\1(.*?)\n.*?\n\2\b", lambda m: "<<" + m.group(3), command, flags=re.DOTALL)  # drop here-doc bodies
    cwd = ""
    red = ops = False

    def real(path: str) -> bool:
        if not path or path.startswith("-"):
            return False
        full = path if path.startswith(("/", "$", "~")) else (cwd.rstrip("/") + "/" + path if cwd else path)
        return not full.startswith(SCRATCH)

    for seg in _SEGMENT.split(body):
        m = _CD.match(seg)
        if m:
            cwd = m.group(2)
            continue
        red = red or any(real(t) for t in _redirect_targets(seg))
        ops = ops or any(any(real(tok) for tok in f.group(2).split()) for f in FILE_OPS.finditer(seg))
    return red, ops


_CMD_FIELD = re.compile(r'"(?:command|cmd|script|code)"\s*:\s*"((?:[^"\\]|\\.)*)')


def command_of(arguments: Any) -> str:
    """The shell command in a call's arguments: a dict, JSON text, or JSON text truncated by a connector."""
    if isinstance(arguments, dict):
        for k in ("command", "cmd", "script", "code", "input"):
            if isinstance(arguments.get(k), str):
                return arguments[k]
        return json.dumps(arguments)
    if isinstance(arguments, str):
        try:
            return command_of(json.loads(arguments))
        except ValueError:
            m = _CMD_FIELD.search(arguments)  # truncated JSON: recover the command string and unescape it
            if m:
                try:
                    return json.loads(f'"{m.group(1)}"')
                except ValueError:
                    return m.group(1).encode().decode("unicode_escape", "replace")
            return arguments
    return ""


_cmd = command_of


def classify_shell(command: str) -> ActionKind:
    """Most consequential kind in a (possibly chained) command. In-place edits and writes to workspace files are
    edits; file shuffling (mv/cp/rm/mkdir) in the workspace is an edit only if nothing else happened; scratch work
    under /tmp (probe programs, reference builds) is not an edit at all."""
    if SUBMIT.search(command):
        return "submit"
    redirect, file_ops = _targets_outside_scratch(command)
    if SHELL_EDIT.search(command) or redirect:
        return "edit"
    for kind, rx in (("build", BUILD), ("test", TEST), ("run", RUN)):
        if rx.search(command):
            return kind  # type: ignore[return-value]
    return "edit" if file_ops else "investigate"


def classify(tool: str, arguments: Any = None) -> ActionKind:
    """Kind of one tool call (``tool`` may be app-qualified, e.g. ``ws.shell``)."""
    base = tool.rsplit(".", 1)[-1]
    if base in SUBMIT_TOOLS:
        return "submit"
    if tool in EDIT_TOOLS or base in EDIT_TOOLS:
        args = arguments if isinstance(arguments, dict) else {}
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                args = parsed if isinstance(parsed, dict) else {}
            except ValueError:
                args = {"command": "view"} if re.search(r'"command"\s*:\s*"view"', arguments) else {}
        if base in ("str_replace_editor", "str_replace_based_edit_tool") and args.get("command") == "view":
            return "investigate"
        return "edit"
    if base in SHELL_TOOLS:
        return classify_shell(_cmd(arguments))
    if base in READ_TOOLS or base.startswith(("get_", "list_", "read_", "search_", "find_", "view_", "query_")):
        return "investigate"
    if base in COMM_TOOLS or base.startswith("send_"):
        return "communicate"
    if base.startswith(("run_ci", "run_test", "check_")):
        return "test"
    return "act" if "." in tool else "other"  # an App tool changes that App; an unknown bare tool (think, plan) is other
