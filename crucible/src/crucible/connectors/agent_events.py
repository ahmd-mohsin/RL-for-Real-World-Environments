"""What every agent-session connector agrees on (ATIF transcripts, the capture proxy, telemetry): which tool calls
write files, and which user texts mean "stop" or "no". One definition, so M4 sees the same event types whatever
the source."""

from __future__ import annotations

import re
from typing import Any

# tool name -> argument holding the path it writes (Claude Code, OpenHands / SWE-agent editors, MCP file tools)
EDIT_TOOLS = {"Edit": "file_path", "Write": "file_path", "MultiEdit": "file_path", "NotebookEdit": "notebook_path",
              "str_replace_editor": "path", "str_replace_based_edit_tool": "path", "edit_file": "path", "write_file": "path",
              "create_file": "path", "apply_patch": None}
INTERRUPT = re.compile(r"\[Request interrupted by user|Interrupted by user", re.IGNORECASE)
REJECTED = re.compile(r"user doesn't want to proceed with this tool use|tool use was rejected|user rejected|permission denied by the user", re.IGNORECASE)


def writes_of(fn: str, args: Any) -> list[str]:
    """``file:<path>`` objects a tool call wrote (M4 attributes later corrections of the file to this call)."""
    key = EDIT_TOOLS.get(fn, "missing")
    if key == "missing" or not isinstance(args, dict):
        return []
    if key is None:  # apply_patch: paths are in the patch text
        patch = str(args.get("input") or args.get("patch") or "")
        return sorted({f"file:{m}" for m in re.findall(r"^\*\*\* (?:Update|Add|Delete) File: (\S+)", patch, re.MULTILINE)})
    path = args.get(key)
    return [f"file:{path}"] if isinstance(path, str) and path else []


def user_type(text: str) -> str:
    return "interrupt" if INTERRUPT.search(text) else "user_message"
