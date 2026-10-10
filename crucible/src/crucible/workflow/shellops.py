"""Replaying the file effects of shell commands from their text (state reconstruction for bash-driven agents).

Agents edit through the shell as much as through editors: they ``rm`` scratch files, ``mv`` and ``cp`` files, write
files with here-docs, ``git checkout --`` a file back. These effects are deterministic given the command text and
the current files, so they can be replayed exactly. Effects that depend on program execution (``sed -i`` with
regexes, a Python script that writes files, a build that generates sources) cannot; ``apply_shell`` reports those
instead of guessing, and the replay test counts them as divergences.

Parsing is quote-aware (``shlex`` with punctuation), so ``python -c "print(1 >= 2)"`` is not a redirect, and
here-doc bodies are taken verbatim. Paths are resolved against the ``cd`` chain and mapped to repo-relative paths
by ``path_of`` (scratch paths outside the repo map to None and are ignored).
"""

from __future__ import annotations

import fnmatch
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from posixpath import join, normpath

from crucible.workflow.filestate import FileState

HEREDOC = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_][\w-]*)\2")
OPS = {";", "&&", "||", "|", "&", "\n"}
REPLAYABLE = {"cd", "rm", "mv", "cp", "touch", "mkdir", "cat", "echo", "printf", "git", "true", ":", "ls", "pwd"}
WRITES_FROM_CODE = re.compile(r"\.write_text\(|\.write\(|open\([^)]*['\"][wax]\+?b?['\"]|shutil\.(?:copy|move|rmtree)|os\.(?:remove|unlink|rename)|\.unlink\(")


@dataclass
class ShellEffect:
    applied: list[str] = field(default_factory=list)  # repo paths changed
    unreplayable: list[str] = field(default_factory=list)  # reasons
    cwd: str = ""


def _heredocs(command: str) -> tuple[str, list[str]]:
    """Strip here-doc bodies out of the command (replaced by markers) and return them in order."""
    bodies: list[str] = []
    lines = command.split("\n")
    out_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        found = list(HEREDOC.finditer(line))
        out_lines.append(line)
        i += 1
        for m in found:
            tag, dash = m.group(3), m.group(1)
            body = []
            while i < len(lines) and (lines[i].strip() if dash else lines[i]) != tag:
                body.append(lines[i].lstrip("\t") if dash else lines[i])
                i += 1
            i += 1  # the terminator
            bodies.append("\n".join(body) + "\n")
    return "\n".join(out_lines), bodies


def _tokens(command: str) -> list[str] | None:
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lex.whitespace_split = True
        lex.whitespace = " \t\r"  # newlines separate commands
        lex.commenters = "#"
        return list(lex)
    except ValueError:
        return None


def _commands(tokens: list[str]) -> list[list[str]]:
    cmds, cur = [], []
    for t in tokens:
        if t in OPS or t.strip() == "" and "\n" in t:
            if cur:
                cmds.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        cmds.append(cur)
    return cmds


def apply_shell(fs: FileState, command: str, *, cwd: str, path_of: Callable[[str], str | None]) -> ShellEffect:
    """Apply the file effects of ``command`` to ``fs``; ``cwd`` is the absolute directory it starts in."""
    eff = ShellEffect(cwd=cwd)
    text, bodies = _heredocs(command.replace("\\\n", " "))
    toks = _tokens(text.replace("\n", " \n "))
    if toks is None:
        eff.unreplayable.append("unparseable shell text")
        return eff
    body_i = 0

    def abspath(p: str) -> str:
        return normpath(p if p.startswith("/") else join(eff.cwd or "/", p))

    def repo(p: str) -> str | None:
        return path_of(abspath(p))

    def known(pattern_abs: str) -> list[str]:
        rel = path_of(pattern_abs)
        if rel is None:
            return []
        if any(ch in rel for ch in "*?["):
            return [p for p, c in fs.files.items() if c is not None and fnmatch.fnmatch(p, rel)]
        return [rel]

    for cmd in _commands(toks):
        # redirections inside this command
        out_target, append, src_heredoc = None, False, False
        args: list[str] = []
        i = 0
        while i < len(cmd):
            t = cmd[i]
            if t in (">", ">>") and i + 1 < len(cmd):
                out_target, append = cmd[i + 1], t == ">>"
                i += 2
                continue
            if t in ("<<", "<<-") and i + 1 < len(cmd):
                src_heredoc = True
                i += 2
                continue
            if t in ("2>", "1>", "2>>", "&>") or re.fullmatch(r"\d?>&\d", t):
                i += 2 if t in ("2>", "1>", "2>>", "&>") else 1
                continue
            if t == "<" and i + 1 < len(cmd):
                i += 2
                continue
            args.append(t)
            i += 1
        if not args:
            continue
        prog, rest = args[0], args[1:]
        heredoc_body = None
        if src_heredoc:
            heredoc_body = bodies[body_i] if body_i < len(bodies) else ""
            body_i += 1
        if prog == "cd":
            eff.cwd = abspath(rest[0]) if rest else eff.cwd
            continue
        if out_target is not None and out_target not in ("/dev/null",) and not out_target.startswith("&"):
            target = repo(out_target)
            if target is not None:
                if prog == "cat" and heredoc_body is not None and not [a for a in rest if not a.startswith("-")]:
                    content = heredoc_body
                elif prog in ("echo", "printf") and not any("$" in a or "`" in a for a in rest):
                    flags = [a for a in rest if a in ("-n", "-e", "-ne", "-en")]
                    words = [a for a in rest if a not in flags]
                    content = " ".join(words) + ("" if "-n" in "".join(flags) or prog == "printf" else "\n")
                    if prog == "printf" or "e" in "".join(flags):
                        content = content.encode().decode("unicode_escape")
                else:
                    eff.unreplayable.append(f"output of `{prog}` redirected to {target}")
                    continue
                prev = fs.read(target) or "" if append else ""
                fs.write(target, prev + content)
                eff.applied.append(target)
            continue
        if prog in ("rm", "unlink"):
            for a in [x for x in rest if not x.startswith("-")]:
                p = abspath(a)
                rel = path_of(p.rstrip("/"))
                if rel is None:
                    continue
                hits = known(p) if any(ch in a for ch in "*?[") else [rel]
                prefix = rel.rstrip("/") + "/"
                hits += [k for k, c in fs.files.items() if c is not None and k.startswith(prefix)]  # rm -r dir/
                for h in dict.fromkeys(hits):
                    if fs.read(h) is not None:
                        fs.write(h, None)
                        eff.applied.append(h)
            continue
        if prog in ("mv", "cp") and len([x for x in rest if not x.startswith("-")]) == 2:
            src, dst = [x for x in rest if not x.startswith("-")]
            s, d = repo(src), repo(dst)
            content = fs.read(s) if s is not None else None
            if s is not None and content is None:
                eff.unreplayable.append(f"{prog} of an unknown file {src}")
                continue
            if d is not None:
                if dst.endswith("/") and s is not None:
                    d = join(d, s.rsplit("/", 1)[-1])
                fs.write(d, content)
                eff.applied.append(d)
            if prog == "mv" and s is not None:
                fs.write(s, None)
                eff.applied.append(s)
            continue
        if prog == "touch":
            for a in [x for x in rest if not x.startswith("-")]:
                r = repo(a)
                if r is not None and fs.read(r) is None:
                    fs.write(r, "")
                    eff.applied.append(r)
            continue
        if prog == "git" and rest[:1] in (["checkout"], ["restore"]) and ("--" in rest or rest[:1] == ["restore"]):
            paths = rest[rest.index("--") + 1:] if "--" in rest else [x for x in rest[1:] if not x.startswith("-")]
            base = fs.base or (lambda p: None)
            for a in paths:
                r = repo(a)
                if r is None:
                    continue
                if r in (".", "") or a in (".", "./"):
                    for k in list(fs.touched):
                        fs.write(k, base(k))
                        eff.applied.append(k)
                else:
                    fs.write(r, base(r))
                    eff.applied.append(r)
            continue
        if prog == "git" and rest[:1] in (["stash"], ["apply"], ["am"], ["reset"], ["clean"], ["mv"], ["rm"]):
            eff.unreplayable.append(f"git {rest[0]}")
            continue
        if prog in ("sed", "perl") and any(a.startswith("-i") or a == "--in-place" for a in rest):
            eff.unreplayable.append(f"{prog} -i")
            continue
        if prog in ("python", "python3") and WRITES_FROM_CODE.search(command):
            eff.unreplayable.append("python code that writes files")
            continue
        if prog in ("patch",):
            eff.unreplayable.append("patch")
            continue
    return eff
