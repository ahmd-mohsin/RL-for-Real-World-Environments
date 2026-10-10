"""File-level state of a workspace, reconstructed from recorded actions (DESIGN §5.3.2 state reconstruction).

A ``FileState`` maps paths to contents (``None`` = deleted). It is built from a base snapshot plus the edits a trace
recorded, applied with each editor's exact semantics:

- OpenHands / SWE-agent / Anthropic ``str_replace_editor``: ``create`` (overwrites), ``str_replace`` (``old_str`` must
  occur exactly once), ``insert`` (after line ``insert_line``; 0 = top), ``undo_edit`` (restores the file's previous
  content), ``view`` (no change);
- Claude Code ``Write`` / ``Edit`` (``replace_all``) / ``MultiEdit``;
- unified diffs (``git diff`` / ``model.patch``), applied strictly: every hunk's context and removed lines must match.

An edit that does not apply raises ``EditError``: the replay test (``replay.py``) counts it as a divergence instead of
guessing. Shell commands that edit files (``sed -i``, scripts) cannot be replayed from text; the replayer flags them.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


class EditError(Exception):
    pass


@dataclass
class FileState:
    files: dict[str, str | None] = field(default_factory=dict)  # path -> content (None = deleted)
    base: Callable[[str], str | None] | None = None  # lazily reads the base snapshot (git show, raw fetch)
    _history: dict[str, list[str | None]] = field(default_factory=dict)
    touched: set[str] = field(default_factory=set)

    def read(self, path: str) -> str | None:
        if path in self.files:
            return self.files[path]
        content = self.base(path) if self.base else None
        self.files[path] = content
        return content

    def write(self, path: str, content: str | None) -> None:
        self._history.setdefault(path, []).append(self.read(path))
        self.files[path] = content
        self.touched.add(path)

    def copy(self) -> FileState:
        return FileState(files=dict(self.files), base=self.base, _history={k: list(v) for k, v in self._history.items()}, touched=set(self.touched))

    # ------------------------------------------------------------------ editors
    def str_replace_editor(self, args: dict[str, Any], path: str) -> None:
        cmd = args.get("command")
        if cmd in (None, "view"):
            return
        if cmd == "create":
            self.write(path, args.get("file_text") or "")
            return
        cur = self.read(path)
        if cur is None:
            raise EditError(f"{cmd} on missing file {path}")
        if cmd == "str_replace":
            old, new = args.get("old_str") or "", args.get("new_str") or ""
            n = cur.count(old)
            if not old or n != 1:
                # the editor expands tabs before matching; retry on that form (OpenHands/SWE-agent behaviour)
                exp = cur.expandtabs()
                if old and exp.count(old.expandtabs()) == 1:
                    self.write(path, exp.replace(old.expandtabs(), new.expandtabs(), 1))
                    return
                raise EditError(f"old_str occurs {n} times in {path}")
            self.write(path, cur.replace(old, new, 1))
        elif cmd == "insert":
            line = int(args.get("insert_line") or 0)
            lines = cur.split("\n")
            if not 0 <= line <= len(lines):
                raise EditError(f"insert_line {line} outside {path} ({len(lines)} lines)")
            text = args.get("new_str") or ""
            self.write(path, "\n".join(lines[:line] + text.split("\n") + lines[line:]))
        elif cmd == "undo_edit":
            hist = self._history.get(path)
            if not hist:
                raise EditError(f"undo_edit with no history for {path}")
            self.files[path] = hist.pop()
        else:
            raise EditError(f"unknown str_replace_editor command {cmd!r}")

    def claude_edit(self, tool: str, args: dict[str, Any], path: str) -> None:
        if tool == "Write":
            self.write(path, args.get("content") or "")
            return
        cur = self.read(path)
        if cur is None:
            raise EditError(f"{tool} on missing file {path}")
        edits = args.get("edits") if tool == "MultiEdit" else [args]
        for ed in edits or []:
            old, new = ed.get("old_string") or "", ed.get("new_string") or ""
            if ed.get("replace_all"):
                if old not in cur:
                    raise EditError(f"old_string not in {path}")
                cur = cur.replace(old, new)
            else:
                if cur.count(old) != 1:
                    raise EditError(f"old_string occurs {cur.count(old)} times in {path}")
                cur = cur.replace(old, new, 1)
        self.write(path, cur)

    # ------------------------------------------------------------------ unified diffs
    def apply_patch(self, patch: str, *, strip: int = 1) -> list[str]:
        """Apply a git-style unified diff strictly; returns the paths it changed."""
        changed = []
        for fp in parse_patch(patch):
            src = _strip(fp.old, strip)
            dst = _strip(fp.new, strip)
            if fp.binary:
                raise EditError(f"binary patch for {dst or src}")
            if dst is None:  # deletion
                self.write(src, None)  # type: ignore[arg-type]
                changed.append(src)
                continue
            base = "" if src is None else (self.read(src) or "")
            if src is not None and src != dst:  # rename
                self.write(src, None)
            self.write(dst, apply_hunks(base, fp.hunks, dst))
            changed.append(dst)
        return changed

    def diff(self, other_base: Callable[[str], str | None] | None = None, paths: Iterable[str] | None = None) -> str:
        """Unified diff of the current contents against the base snapshot (for the given or all touched paths)."""
        base = other_base or self.base or (lambda p: None)
        out = []
        for p in sorted(paths or self.touched):
            a, b = base(p), self.files.get(p)
            if a == b:
                continue
            out.extend(difflib.unified_diff((a or "").splitlines(keepends=True), (b or "").splitlines(keepends=True),
                                            fromfile=f"a/{p}" if a is not None else "/dev/null", tofile=f"b/{p}" if b is not None else "/dev/null"))
        return "".join(out)


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[str]  # with their ' ', '-', '+' prefix; '\\ No newline' markers kept


@dataclass
class FilePatch:
    old: str | None
    new: str | None
    hunks: list[Hunk]
    binary: bool = False


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(patch: str) -> list[FilePatch]:
    out: list[FilePatch] = []
    cur: FilePatch | None = None
    lines = patch.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("diff --git "):
            cur = None
            m = re.match(r"diff --git (\S+) (\S+)$", ln)
            j = i + 1
            mode = None
            while j < len(lines) and not lines[j].startswith(("diff --git ", "--- ", "@@", "Binary files ", "GIT binary patch")):
                if lines[j].startswith(("new file mode", "deleted file mode")):
                    mode = lines[j].split()[0]
                j += 1
            if m and mode and (j >= len(lines) or lines[j].startswith("diff --git ")):  # empty file added / deleted
                out.append(FilePatch(old=None if mode == "new" else m.group(1), new=None if mode == "deleted" else m.group(2), hunks=[]))
        elif ln.startswith("--- "):
            old = ln[4:].split("\t")[0].strip()
            new = lines[i + 1][4:].split("\t")[0].strip() if i + 1 < len(lines) and lines[i + 1].startswith("+++ ") else None
            cur = FilePatch(old=None if old == "/dev/null" else old, new=None if new in (None, "/dev/null") else new, hunks=[])
            out.append(cur)
            i += 2
            continue
        elif ln.startswith(("Binary files ", "GIT binary patch")):
            m = re.match(r"Binary files (\S+) and (\S+) differ", ln)
            out.append(FilePatch(old=m.group(1) if m else None, new=m.group(2) if m else None, hunks=[], binary=True))
        elif (m := _HUNK.match(ln)) and cur is not None:
            h = Hunk(int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1), [])
            i += 1
            need_old, need_new = h.old_len, h.new_len
            while i < len(lines) and (need_old > 0 or need_new > 0 or (i < len(lines) and lines[i].startswith("\\"))):
                body = lines[i]
                if body.startswith("\\"):
                    h.lines.append(body)
                elif body.startswith("-"):
                    h.lines.append(body)
                    need_old -= 1
                elif body.startswith("+"):
                    h.lines.append(body)
                    need_new -= 1
                else:  # context (an empty line in a patch is a context line whose content is empty)
                    h.lines.append(" " + body[1:] if body.startswith(" ") else " " + body)
                    need_old -= 1
                    need_new -= 1
                i += 1
            while i < len(lines) and lines[i].startswith("\\"):
                h.lines.append(lines[i])
                i += 1
            cur.hunks.append(h)
            continue
        i += 1
    return out


def apply_hunks(text: str, hunks: list[Hunk], name: str = "") -> str:
    src = text.split("\n") if text else []
    trailing_newline = text.endswith("\n")
    if trailing_newline:
        src = src[:-1]
    out: list[str] = []
    pos = 0  # next unconsumed source line (0-based)
    new_no_eol = None
    for h in hunks:
        old_block = [ln[1:] for ln in h.lines if ln[:1] in (" ", "-")]
        new_block = [ln[1:] for ln in h.lines if ln[:1] in (" ", "+")]
        start = max(h.old_start - 1, 0) if h.old_len else h.old_start  # a pure insertion at line N goes after line N
        if src[start:start + len(old_block)] != old_block:
            found = next((s for d in range(1, 200) for s in (start - d, start + d) if pos <= s <= len(src) - len(old_block)
                          and src[s:s + len(old_block)] == old_block), None)  # tolerate small offsets, like `patch`
            if found is None:
                raise EditError(f"hunk @@ -{h.old_start},{h.old_len} does not apply to {name}")
            start = found
        if start < pos:
            raise EditError(f"overlapping hunks in {name}")
        out.extend(src[pos:start])
        out.extend(new_block)
        pos = start + len(old_block)
        markers = [i for i, ln in enumerate(h.lines) if ln.startswith("\\")]
        for mi in markers:  # "\ No newline at end of file" after a '+'/' ' line refers to the new file
            prev = h.lines[mi - 1][:1] if mi > 0 else ""
            if prev in ("+", " "):
                new_no_eol = True
            elif prev == "-":
                new_no_eol = new_no_eol if new_no_eol is not None else False
    touches_end = pos == len(src)  # the last hunk reached the end of the old file: its new side decides the final newline
    out.extend(src[pos:])
    eol = (not new_no_eol) if touches_end else trailing_newline
    if not out:
        return ""
    return "\n".join(out) + ("\n" if eol else "")


def _strip(path: str | None, n: int) -> str | None:
    if path is None:
        return None
    parts = path.split("/")
    return "/".join(parts[n:]) if len(parts) > n else path
