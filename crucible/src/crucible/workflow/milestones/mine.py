"""Mining candidate milestones from expert traces (DESIGN §5.3.3).

Three sources, each producing ``Candidate`` milestones with provenance; the filters in ``filters.py`` decide which
survive.

- **Diff-derived** (``from_diff``): from an expert's change (a final patch, a PR diff, a commit), per changed file:
  * *construct removed*: an identifier or call the expert deleted and that no longer occurs in that file afterwards
    -> ``grep(pattern, paths=[file]) count_eq 0``;
  * *construct added*: a distinctive identifier or call the expert introduced (absent from the base file) ->
    ``grep count_ge 1``;
  * *file created / deleted* -> ``exists`` truthy / ``count_eq 0``.
  Constructs are tokens, not lines: a different but equivalent edit that uses the same API still satisfies them,
  while copying the gold line is not required. Trivial tokens (keywords, very common names, short words) are skipped.
- **CI-derived** (``from_ci``): tests that go fail -> pass across the trace's recorded test runs (SWE-smith's F2P logic
  applied per step instead of per instance) -> a medium-cost ``run`` milestone.
- **Signal-derived** (``from_signals``): a negative expert signal anchored at a file/line (a review correction) -> the
  check the reviewer demanded (C2 turns it into a hidden check; here it is a candidate milestone).

Candidates are deduplicated by (query, expectation); ids are stable (derived from content).
"""

from __future__ import annotations

import hashlib
import keyword
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from crucible.core.schema.milestone import Expectation, Milestone, MilestoneCheck, Provenance
from crucible.core.schema.state import Query
from crucible.workflow.filestate import FileState, parse_patch

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*\(")
COMMON = frozenset(keyword.kwlist) | {
    "self", "cls", "None", "True", "False", "return", "print", "len", "str", "int", "float", "list", "dict", "set", "tuple", "range",
    "isinstance", "super", "init", "args", "kwargs", "value", "values", "result", "data", "name", "type", "text", "path", "file",
    "test", "assert", "assertEqual", "pytest", "import", "from", "def", "class", "const", "auto", "void", "char", "bool", "size_t",
    "std", "include", "define", "endif", "ifdef", "ifndef", "struct", "static", "inline", "unsigned", "double", "template",
    "typename", "public", "private", "protected", "this", "nullptr", "NULL", "true", "false", "else", "elif", "while", "for",
    "other", "with", "that", "when", "then", "into", "only", "also", "some", "each", "used", "uses", "using", "have", "will",
}


def _mid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:10]


@dataclass
class Candidate:
    milestone: Milestone
    source: str  # diff | ci | signal | policy
    evidence: dict[str, Any] = field(default_factory=dict)


def _tokens(lines: Iterable[str], min_len: int = 4) -> set[str]:
    out: set[str] = set()
    for ln in lines:
        s = ln.strip()
        if s.startswith(("#", "//", "*", "/*")) and not s.startswith(("#include", "#define", "#pragma")):
            continue  # comments carry no behaviour
        for c in CALL.findall(ln):
            if c.rsplit(".", 1)[-1] not in COMMON and len(c) >= min_len:
                out.add(c + "(")
        for t in IDENT.findall(ln):
            if t not in COMMON and len(t) >= min_len and (not t.isupper() or len(t) >= 6):  # short ALLCAPS are macros/noise
                out.add(t)
    return out


DEF = re.compile(r"^(\s*)(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)|^(\s*)(?:[\w:<>,*&\s]+?\s)?([A-Za-z_][\w:]*)\s*\([^;]*\)\s*(?:const\s*)?\{?\s*$")


def _enclosing_anchors(base_text: str, hunks: list) -> list[tuple[str, str]]:
    """(anchor regex, label) of the innermost def/class/function enclosing each hunk's first changed line in the base
    file, so "the place the expert changed was changed" is checkable however the change was written."""
    lines = base_text.split("\n")
    out: list[tuple[str, str]] = []
    for h in hunks:
        offset = next((i for i, ln in enumerate(h.lines) if ln.startswith(("-", "+"))), 0)
        ctx_before = sum(1 for ln in h.lines[:offset] if ln.startswith(" "))
        line_no = max(h.old_start - 1 + ctx_before, 0)
        target_ind = None
        for i in range(min(line_no, len(lines) - 1), -1, -1):
            ln = lines[i]
            if ln.lstrip().startswith(("*", "//", "/*", "#", "--")):
                continue  # comments ("Software License Agreement (BSD License)") and preprocessor lines are not blocks
            m = DEF.match(ln)
            if not m:
                continue
            ind = len(ln) - len(ln.lstrip())
            if target_ind is not None and ind >= target_ind:
                continue
            name = m.group(2) or m.group(4)
            if not name or name in COMMON:
                continue
            anchor = r"^\s*" + re.escape(ln.strip()[:120])
            if (anchor, name) not in out:
                out.append((anchor, name))
            break
    return out


def _pattern(token: str) -> str:
    return re.escape(token[:-1]) + r"\s*\(" if token.endswith("(") else r"\b" + re.escape(token) + r"\b"


def _grep_milestone(app: str, path: str, token: str, kind: str, weight: float, prov: Provenance) -> Milestone:
    pat = _pattern(token)
    exp = Expectation(kind="count_eq", value=0) if kind == "removed" else Expectation(kind="count_ge", value=1)
    return Milestone(id=f"diff/{kind}{'_call' if token.endswith('(') else ''}/{path.rsplit('/', 1)[-1]}:{token.rstrip('(')}#{_mid(path, token, kind)}",
                     check=MilestoneCheck(app=app, query=Query(op="grep", args={"pattern": pat, "paths": [path]}), expect=exp),
                     weight=weight, cost="cheap", provenance=[prov])


def from_diff(patch: str, base: Callable[[str], str | None], *, app: str = "repo", source: str = "expert_end_state",
              max_per_file: int = 8, include: Callable[[str], bool] | None = None) -> list[Candidate]:
    """Diff-derived candidates from one expert change. ``include``: which files count (e.g. skip tests, docs)."""
    after = FileState(base=base)
    after.apply_patch(patch)
    out: list[Candidate] = []
    for fp in parse_patch(patch):
        path = (fp.new or fp.old or "")[2:] if (fp.new or fp.old or "").startswith(("a/", "b/")) else (fp.new or fp.old or "")
        if not path or (include and not include(path)):
            continue
        prov = Provenance(source=source, note=f"diff of {path}")
        before_text, after_text = base(path), after.files.get(path)
        if before_text is None and after_text is not None:
            out.append(Candidate(Milestone(id=f"diff/created/{path}#{_mid(path, 'created')}", weight=0.5, cost="cheap", provenance=[prov],
                                           check=MilestoneCheck(app=app, query=Query(op="exists", args={"path": path}), expect=Expectation(kind="truthy"))),
                                 "diff", {"path": path, "change": "created"}))
        if after_text is None and before_text is not None:
            out.append(Candidate(Milestone(id=f"diff/deleted/{path}#{_mid(path, 'deleted')}", weight=0.5, cost="cheap", provenance=[prov],
                                           check=MilestoneCheck(app=app, query=Query(op="exists", args={"path": path}),
                                                                expect=Expectation(kind="equals", value={"exists": False, "is_file": False, "is_dir": False}))),
                                 "diff", {"path": path, "change": "deleted"}))
            continue
        code_file = not re.search(r"(CMakeLists\.txt|\.cmake|\.txt|\.md|\.rst|\.ya?ml|\.json|\.toml|\.cfg|\.ini)$", path)
        for anchor in (_enclosing_anchors(before_text or "", fp.hunks) if code_file else []):
            out.append(Candidate(Milestone(id=f"diff/located/{path.rsplit('/', 1)[-1]}:{anchor[1][:40]}#{_mid(path, anchor[0])}", weight=1.0, cost="cheap",
                                           provenance=[prov], check=MilestoneCheck(app=app, query=Query(op="block_changed", args={"path": path, "anchor": anchor[0]}),
                                                                                   expect=Expectation(kind="equals", value={"changed": True, "found": True}))),
                                 "diff", {"path": path, "change": "located", "anchor": anchor[1]}))
        removed = [ln[1:] for h in fp.hunks for ln in h.lines if ln.startswith("-")]
        added = [ln[1:] for h in fp.hunks for ln in h.lines if ln.startswith("+")]
        before_tokens = _tokens((before_text or "").split("\n"))
        after_tokens = _tokens((after_text or "").split("\n"))
        def occurs(tok: str, text: str | None) -> bool:
            return bool(text) and re.search(_pattern(tok), text or "", re.MULTILINE) is not None

        gone = sorted(t for t in _tokens(removed) if t in before_tokens and occurs(t, before_text) and not occurs(t, after_text))
        new = sorted(t for t in _tokens(added) if t in after_tokens and occurs(t, after_text) and not occurs(t, before_text))
        # a call token makes its bare name redundant (`helper(` implies `helper`)
        gone = [t for t in gone if t.endswith("(") or t + "(" not in gone]
        new = [t for t in new if t.endswith("(") or t + "(" not in new]
        # the most specific first: calls before bare names, longer before shorter
        for kind, toks in (("removed", gone), ("added", new)):
            toks.sort(key=lambda t: (not t.endswith("("), -len(t)))
            for t in toks[:max_per_file]:
                out.append(Candidate(_grep_milestone(app, path, t, kind, 1.0, prov), "diff", {"path": path, "token": t, "change": kind}))
    seen: set[str] = set()
    uniq = []
    for c in out:
        key = c.milestone.check.model_dump_json()
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def from_ci(test_runs: list[tuple[int, dict[str, str]]], *, app: str = "repo", command: Callable[[str], str] | None = None) -> list[Candidate]:
    """``test_runs``: (step, {test_name: pass|fail|error|skip}) in trace order. A test failing at some step and passing
    at the last run where it appears (fail -> pass) becomes a candidate; one that ever passed and ends failing is a
    regression signal, not a milestone."""
    first_fail: dict[str, int] = {}
    last: dict[str, tuple[int, str]] = {}
    for step, results in test_runs:
        for name, st in results.items():
            if st in ("fail", "error") and name not in first_fail:
                first_fail[name] = step
            last[name] = (step, st)
    out = []
    for name, (step, st) in sorted(last.items()):
        if st == "pass" and name in first_fail and first_fail[name] < step:
            cmd = command(name) if command else f"python -m pytest -q {name!r}"
            out.append(Candidate(Milestone(id=f"ci/f2p/{name}#{_mid(name)}", weight=1.0, cost="medium",
                                           provenance=[Provenance(source="ci_fail_to_pass", note=f"failed at step {first_fail[name]}, passed at {step}")],
                                           check=MilestoneCheck(app=app, query=Query(op="run", args={"cmd": cmd, "timeout_s": 900}), expect=Expectation(kind="ok"))),
                                 "ci", {"test": name, "failed_at": first_fail[name], "passed_at": step}))
    return out


def from_signals(signals: Iterable[Any], *, app: str = "repo") -> list[Candidate]:
    """Negative signals anchored at a file region (``target.object = "file:<path>"``, ``target.span``) -> "the region
    the expert flagged was changed" candidates (the precise behavioural check is C2's job)."""
    out = []
    for s in signals:
        tgt = getattr(s, "target", None)
        if not tgt or not tgt.object or not str(tgt.object).startswith("file:") or getattr(s, "polarity", 0) >= 0:
            continue
        path = str(tgt.object)[5:]
        text = (getattr(s, "text", None) or "")[:200]
        out.append(Candidate(Milestone(id=f"signal/{s.type}/{path.rsplit('/', 1)[-1]}#{_mid(s.id)}", weight=1.0, cost="cheap",
                                       provenance=[Provenance(signal=s.id, note=text)],
                                       check=MilestoneCheck(app=app, query=Query(op="exists", args={"path": path}), expect=Expectation(kind="truthy"))),
                             "signal", {"signal": s.id, "type": s.type, "path": path, "span": tgt.span}))
    return out
