"""Part 3: Interface. The tools the agent gets, as JSON schemas plus their implementation.

The tool set is part of the environment: the same model can score very differently with
different tools, so it is versioned with the task. This set is the de facto standard for coding
agents (SWE-agent's ACI, Anthropic's bash + text_editor tools, mini-swe-agent, MiMo's
bash/read/write/edit):

    bash         run any shell command in /testbed (search, run tests, inspect)
    view         read a file with line numbers, or list a directory
    str_replace  edit a file by replacing one exact, unique snippet
    create       write a whole new file
    submit       end the episode; grading starts

Every tool returns plain text. Errors are returned as text, never raised, so the agent can recover.
"""

from __future__ import annotations

import posixpath
from typing import Any

from runtime.sandbox import Sandbox

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "bash",
        "description": (
            "Run a bash command in the repository root (/testbed) and return its combined "
            "stdout/stderr and exit code. There is no internet access. Commands time out after "
            "120 seconds. Each call runs in a fresh shell: `cd` and environment variables do not persist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    },
    {
        "name": "view",
        "description": (
            "Show a file with line numbers, or list a directory two levels deep. "
            "Use view_range [start, end] (1-indexed, inclusive, end=-1 for EOF) for long files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path, or relative to /testbed."},
                "view_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            },
            "required": ["path"],
        },
    },
    {
        "name": "str_replace",
        "description": (
            "Replace old_str with new_str in a file. old_str must match exactly (including "
            "whitespace) and occur exactly once; include surrounding lines to make it unique."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "create",
        "description": "Create a new file, or overwrite an existing one, with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "file_text": {"type": "string"}},
            "required": ["path", "file_text"],
        },
    },
    {
        "name": "submit",
        "description": "Call when you are done. Your changes in /testbed are then graded by hidden tests.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


class Tools:
    """Dispatches tool calls to a running sandbox."""

    def __init__(self, sandbox: Sandbox):
        self.sb = sandbox
        self.submitted = False

    def call(self, name: str, args: dict[str, Any]) -> str:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"Error: unknown tool '{name}'. Available: {[t['name'] for t in TOOL_SCHEMAS]}"
        try:
            return handler(**args)
        except TypeError as e:
            return f"Error: bad arguments for '{name}': {e}"
        except Exception as e:  # noqa: BLE001 - surface infra problems to the agent as text
            return f"Error: {type(e).__name__}: {e}"

    def _abs(self, path: str) -> str:
        return path if path.startswith("/") else posixpath.join(self.sb.workdir, path)

    def _tool_bash(self, command: str) -> str:
        res = self.sb.exec(command)
        return f"{res.output}\n[exit code: {res.returncode}]"

    def _tool_view(self, path: str, view_range: list[int] | None = None) -> str:
        path = self._abs(path)
        if self.sb.exec(f"test -d '{path}'").returncode == 0:
            res = self.sb.exec(f"find '{path}' -maxdepth 2 -not -path '*/.git*' | sort")
            return res.output
        lines = self.sb.read_file(path).splitlines()
        start, end = 1, len(lines)
        if view_range:
            start, end = view_range[0], (len(lines) if view_range[1] == -1 else view_range[1])
            if not (1 <= start <= max(end, 1)):
                return f"Error: invalid view_range {view_range} for a file with {len(lines)} lines"
        return "\n".join(f"{i:6}\t{lines[i - 1]}" for i in range(start, min(end, len(lines)) + 1))

    def _tool_str_replace(self, path: str, old_str: str, new_str: str) -> str:
        path = self._abs(path)
        text = self.sb.read_file(path)
        count = text.count(old_str)
        if count == 0:
            return f"Error: old_str not found in {path}. Nothing was changed."
        if count > 1:
            return f"Error: old_str occurs {count} times in {path}; add context to make it unique."
        self.sb.write_file(path, text.replace(old_str, new_str, 1))
        line = text[: text.index(old_str)].count("\n") + 1
        return f"Edited {path} at line {line}."

    def _tool_create(self, path: str, file_text: str) -> str:
        path = self._abs(path)
        self.sb.write_file(path, file_text)
        return f"Wrote {len(file_text)} characters to {path}."

    def _tool_submit(self) -> str:
        self.submitted = True
        return "Submitted. Your changes will now be graded."
