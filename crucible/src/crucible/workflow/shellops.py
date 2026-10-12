"""Replaying the file effects of shell commands from their text (state reconstruction for bash-driven agents).

Agents edit through the shell as much as through editors: they ``rm`` scratch files, ``mv`` and ``cp`` files, write
files with here-docs, ``git checkout --`` a file back. These effects are deterministic given the command text and
the current files, so they can be replayed exactly. Effects that depend on program execution (``sed -i`` with
regexes, a Python script that writes files, a build that generates sources) cannot; ``apply_shell`` reports those
instead of guessing, and the replay test counts them as divergences.

Parsing is quote-aware (``shlex`` with punctuation), so ``python -c "print(1 >= 2)"`` is not a redirect, and
quoted here-doc bodies are taken verbatim. Paths are resolved against the ``cd`` chain (a ``( ... )`` subshell keeps
its own) and mapped to repo-relative paths by ``path_of`` (scratch paths outside the repo map to None and are
ignored).

M5 audit (2026-10-12): every command the shared write detector (``core.agent_events.shell_writes``, the one M4 uses)
says writes a repo file is either applied or reported -- ``sed -Ei``, ``--in-place=``, ``perl -pi``, ``tee``, ``dd``,
``truncate``, ``python3.11`` and ``mv a b 2>/dev/null`` were dropped without a word. Effects are no longer guessed:
``git restore --staged`` leaves files alone, ``mv f dir`` moves into a known directory, ``printf`` with a format and
here-docs that expand ``$`` are reported, a ``#`` inside a word is not a comment, and a write behind ``||`` or behind
``&&`` after a command whose success is unknown (``false``, ``test``, ``grep``) is reported rather than applied.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from posixpath import basename, join, normpath

from crucible.workflow.filestate import FileState

HEREDOC = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_][\w-]*)\2")
SEPARATORS = {";", "&&", "||", "|", "&", "\n"}
REDIRECTS = {">", ">>", "<", "<<", "<<-", ">&", "&>", "&>>", ">|"}
REPLAYABLE = {"cd", "rm", "mv", "cp", "touch", "mkdir", "cat", "echo", "printf", "git", "true", ":", "ls", "pwd"}
# commands whose success the replay assumes (they are replayed, or change nothing): `x && y` after them runs y
ASSUMED_OK = REPLAYABLE | {"unlink", "rmdir", "export", "set", "source", "."}
WRITES_FROM_CODE = re.compile(r"\.write_text\(|\.write\(|open\([^)]*['\"][wax]\+?b?['\"]|shutil\.(?:copy|move|rmtree)|os\.(?:remove|unlink|rename)|\.unlink\(")
PUNCT = re.compile(r"&&|\|\||>>|<<-|<<|&>>|&>|>&|>\||[;&|<>()]")


@dataclass
class ShellEffect:
    applied: list[str] = field(default_factory=list)  # repo paths changed
    unreplayable: list[str] = field(default_factory=list)  # reasons
    uncertain: list[str] = field(default_factory=list)  # repo paths an unreplayed command wrote (content now unknown)
    overwritten: list[str] = field(default_factory=list)  # repo paths given wholly new content (or deleted)
    cwd: str = ""


def _heredocs(command: str) -> tuple[str, list[tuple[str, bool]]]:
    """Strip here-doc bodies out of the command and return them in order, with whether the tag was quoted (a quoted
    tag takes the body verbatim; an unquoted one expands ``$`` and backquotes)."""
    bodies: list[tuple[str, bool]] = []
    lines = command.split("\n")
    out_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        found = list(HEREDOC.finditer(line))
        out_lines.append(line)
        i += 1
        for m in found:
            tag, dash, quoted = m.group(3), m.group(1), bool(m.group(2))
            body = []
            while i < len(lines) and (lines[i].strip() if dash else lines[i]) != tag:
                body.append(lines[i].lstrip("\t") if dash else lines[i])
                i += 1
            i += 1  # the terminator
            bodies.append(("\n".join(body) + "\n", quoted))
    return "\n".join(out_lines), bodies


def _strip_comments(text: str) -> str:
    """Drop ``# ...`` comments: a ``#`` that starts a word, outside quotes (``echo a#b`` keeps its ``#``)."""
    out, quote, prev = [], None, " "
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            out.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < len(text):
                out.append(text[i + 1])
                i += 1
            elif ch == quote:
                quote = None
        elif ch == "\\" and i + 1 < len(text):
            out += [ch, text[i + 1]]
            i += 1
        elif ch in "'\"":
            quote = ch
            out.append(ch)
        elif ch == "#" and (prev.isspace() or prev in ";&|()"):
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        else:
            out.append(ch)
        prev = text[i] if i < len(text) else " "
        i += 1
    return "".join(out)


def _tokens(command: str) -> list[str] | None:
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
        lex.whitespace_split = True
        lex.whitespace = " \t\r"  # newlines separate commands
        lex.commenters = ""
        raw = list(lex)
    except ValueError:
        return None
    out: list[str] = []
    for t in raw:  # shlex glues adjacent punctuation (");", "&&(") into one token
        if t and all(c in ";&|<>()" for c in t):
            out += PUNCT.findall(t)
        else:
            out.append(t)
    return out


def _commands(tokens: list[str]) -> list[tuple[str, list[str], int]]:
    """(operator before it, tokens, subshell depth) per simple command; ``(`` / ``)`` change the depth."""
    cmds: list[tuple[str, list[str], int]] = []
    cur: list[str] = []
    op, depth, start_depth = ";", 0, 0
    for t in tokens:
        if t in SEPARATORS or (t.strip() == "" and "\n" in t) or t in ("(", ")"):
            if cur:
                cmds.append((op, cur, start_depth))
                cur = []
            if t in ("(", ")"):
                depth += 1 if t == "(" else -1
                if t == ")":  # the subshell's commands are done: the next one starts back in the outer shell
                    cmds.append((")", [], depth))
                continue
            op = "\n" if t.strip() == "" else t
            start_depth = depth
            continue
        if not cur:
            start_depth = depth
        cur.append(t)
    if cur:
        cmds.append((op, cur, start_depth))
    return cmds


def apply_shell(fs: FileState, command: str, *, cwd: str, path_of: Callable[[str], str | None]) -> ShellEffect:
    """Apply the file effects of ``command`` to ``fs``; ``cwd`` is the absolute directory it starts in."""
    from crucible.core.agent_events import shell_writes

    eff = ShellEffect(cwd=cwd)
    text, bodies = _heredocs(command.replace("\\\n", " "))
    toks = _tokens(_strip_comments(text).replace("\n", " \n "))
    if toks is None:
        eff.unreplayable.append("unparseable shell text")
        return eff
    body_i = 0
    cwd_stack: list[str] = []
    prev_prog, prev_uncertain = "", False

    cur_declared: list[str] = []

    def report(reason: str) -> None:
        """This command was not (fully) replayed: what it writes is now unknown."""
        if reason not in eff.unreplayable:
            eff.unreplayable.append(reason)
        eff.uncertain.extend(p for p in cur_declared if p not in eff.uncertain)

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

    def is_dir(rel: str) -> bool:
        return rel in fs.dirs or any(k.startswith(rel.rstrip("/") + "/") and c is not None for k, c in fs.files.items())

    def covered(rel: str, done: list[str]) -> bool:
        return any(d == rel or d.startswith(rel.rstrip("/") + "/") for d in done)

    for op, cmd, depth in _commands(toks):
        if op == ")":  # leaving a subshell: its `cd` does not leak out
            if cwd_stack:
                eff.cwd = cwd_stack.pop()
            continue
        while len(cwd_stack) < depth:
            cwd_stack.append(eff.cwd)
        # redirections inside this command
        out_target, append, src_heredoc = None, False, False
        args: list[str] = []
        i = 0
        while i < len(cmd):
            t = cmd[i]
            nxt = cmd[i + 1] if i + 1 < len(cmd) else None
            if t.isdigit() and nxt in (">", ">>", ">&", "<", ">|"):  # an fd redirect: `2>/dev/null`, `1>out`, `2>&1`
                if t == "1" and nxt in (">", ">>", ">|"):
                    out_target, append = (cmd[i + 2] if i + 2 < len(cmd) else None), nxt == ">>"
                i += 3
                continue
            if t in (">", ">>", ">|") and nxt is not None:
                out_target, append = nxt, t == ">>"
                i += 2
                continue
            if t in ("&>", "&>>") and nxt is not None:
                out_target, append = nxt, t == "&>>"
                i += 2
                continue
            if t in ("<<", "<<-") and nxt is not None:
                src_heredoc = True
                i += 2
                continue
            if t in (">&", "<") and nxt is not None:
                i += 2
                continue
            args.append(t)
            i += 1
        if not args:
            continue
        prog, rest = basename(args[0]), args[1:]
        heredoc_body, heredoc_quoted = None, True
        if src_heredoc:
            heredoc_body, heredoc_quoted = bodies[body_i] if body_i < len(bodies) else ("", True)
            body_i += 1
        # whether this command runs at all: after `||`, or after `&&` behind a command whose outcome is unknown
        uncertain = op == "||" or (op == "&&" and (prev_uncertain or prev_prog not in ASSUMED_OK))
        guard = prev_prog
        prev_prog, prev_uncertain = prog, uncertain
        seg = shlex.join(cmd)
        declared = [r for p in shell_writes(seg) if (r := repo(p)) is not None]  # what M4 says this command writes
        if out_target is not None and (r := repo(out_target)) is not None and r not in declared:
            declared.append(r)
        cur_declared[:] = declared
        if prog == "cd":
            eff.cwd = abspath(rest[0]) if rest else eff.cwd
            continue
        if uncertain:
            if declared or prog in ("rm", "unlink", "mv", "cp", "touch") or (prog == "git" and rest[:1] in (["checkout"], ["restore"])):
                cur_declared[:] = declared + [r for a in rest if not a.startswith("-") and (r := repo(a)) is not None]
                report(f"`{prog}` runs only if `{guard or '?'}` {'fails' if op == '||' else 'succeeds'}")
            continue
        applied_before = len(eff.applied)
        if prog == "mkdir":
            for a in [x for x in rest if not x.startswith("-")]:
                if (r := repo(a)) is not None:
                    fs.dirs.add(r.rstrip("/"))
            continue
        if out_target is not None and out_target not in ("/dev/null",) and not out_target.startswith("&"):
            target = repo(out_target)
            if target is not None:
                content = None
                words = [a for a in rest if not a.startswith("-")]
                if prog == "cat" and heredoc_body is not None and not words:
                    if heredoc_quoted or not re.search(r"[$`]", heredoc_body):
                        content = heredoc_body
                    else:
                        report(f"here-doc with shell expansion written to {target}")
                elif prog == "echo" and not any("$" in a or "`" in a for a in rest):
                    flags = [a for a in rest if a in ("-n", "-e", "-ne", "-en", "-E")]
                    content = " ".join(a for a in rest if a not in flags) + ("" if "n" in "".join(flags) else "\n")
                    if "e" in "".join(flags).replace("-E", ""):
                        content = content.encode().decode("unicode_escape")
                elif prog == "printf" and len(rest) == 1 and "%" not in rest[0] and "$" not in rest[0] and "`" not in rest[0]:
                    content = rest[0].encode().decode("unicode_escape")  # a plain format string: printf writes it, escapes decoded
                else:
                    report(f"output of `{prog}` redirected to {target}")
                if content is not None:
                    prev = fs.read(target) or "" if append else ""
                    fs.write(target, prev + content)
                    eff.applied.append(target)
                    if not append:
                        eff.overwritten.append(target)
            continue
        if prog in ("rm", "unlink", "rmdir"):
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
                        eff.overwritten.append(h)
                fs.dirs.discard(rel)
            continue
        if prog in ("mv", "cp"):
            plain = [x for x in rest if not x.startswith("-")]
            if len(plain) != 2:
                if declared:
                    report(f"{prog} with {len(plain)} operands")
                continue
            src, dst = plain
            s, d = repo(src), repo(dst)
            content = fs.read(s) if s is not None else None
            if s is not None and content is None:
                report(f"{prog} of an unknown file {src}")
                continue
            if d is not None:
                if s is not None and (dst.endswith("/") or is_dir(d)):  # into a directory
                    d = join(d.rstrip("/"), s.rsplit("/", 1)[-1])
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
        if prog == "git" and rest[:1] == ["checkout"] and "--" not in rest:
            # `git checkout <file>...` (optionally after HEAD): a restore when the arguments are files, a branch
            # switch otherwise (not replayable)
            cand = [x for x in rest[1:] if not x.startswith("-")]
            if cand[:1] == ["HEAD"]:
                cand = cand[1:]
            files = [x for x in cand if (r := repo(x)) is not None and (fs.read(r) is not None or (fs.base and fs.base(r) is not None))]
            if cand and len(files) == len(cand):
                rest = ["checkout", "--", *files]
            elif cand:
                report("git checkout <ref>")
                continue
        if prog == "git" and rest[:1] == ["restore"]:
            flags = [x for x in rest[1:] if x.startswith("-")]
            if any(f.startswith(("--source", "-s")) for f in flags):
                report("git restore --source")
                continue
            staged = any(f in ("--staged", "-S") or (f.startswith("-") and not f.startswith("--") and "S" in f) for f in flags)
            worktree = any(f in ("--worktree", "-W") or (f.startswith("-") and not f.startswith("--") and "W" in f) for f in flags)
            if staged and not worktree:
                continue  # unstaging changes no file
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
        if prog == "git" and rest[:1] in (["stash"], ["apply"], ["am"], ["reset"], ["clean"], ["mv"], ["rm"], ["switch"], ["merge"], ["rebase"], ["pull"],
                                          ["cherry-pick"], ["revert"]):
            report(f"git {rest[0]}")
            continue
        if prog in ("sed", "perl") and any(re.match(r"-[a-zA-Z]*i", a) or a.startswith("--in-place") for a in rest):
            report(f"{prog} -i")
            continue
        if re.fullmatch(r"python[0-9.]*", prog) and WRITES_FROM_CODE.search(command):
            report("python code that writes files")
            continue
        if prog in ("patch", "dd", "truncate", "tee", "install", "rsync", "tar", "unzip"):
            if declared or prog in ("patch", "dd", "truncate", "tar", "unzip"):
                report(prog)
            continue
        # anything else that the shared write detector says writes a repo file, and that was not applied: reported
        done = eff.applied[applied_before:]
        missed = [r for r in declared if not covered(r, done)]
        if missed:
            cur_declared[:] = missed
            report(f"`{prog}` writes {', '.join(sorted(missed))}")
    return eff
