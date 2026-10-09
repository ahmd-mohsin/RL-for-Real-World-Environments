"""Written policy -> traces (coding pack ``policy_docs`` connector, M3; DESIGN §5.1, §5.2 "Policy text").

Sources are CONTRIBUTING guides, style guides, migration guides, ADRs, CODEOWNERS and formatter / linter configs,
read either from git at a pinned revision (the policy that applied when a PR was written) or from files on disk
(our tasks' ``MIGRATION_LOG.md`` / ``DESIGN_NOTES.md``). Each document becomes one trace:

  ``policy_section``  one per heading (Markdown ``#`` or reStructuredText underlines): the heading path, line span
  ``policy_rule``     one per normative sentence or list item, ``reply_to`` its section, with ``data.modality``
                      (``must_not`` > ``must`` > ``should``) and ``target.span`` (its lines in the document)
  ``config_rule``     one per setting of a config file (``.clang-format`` key, ``.clang-tidy`` check, CODEOWNERS
                      pattern), which tools enforce mechanically

Rules are *candidate* rubric items with low reliability until the predictive-validity gate (§6.5) shows they
predict outcomes. Code blocks and HTML comments are skipped. For documents read from git, ``outcome.details``
records the revision, the number of commits that changed the document and the dates of its first and last change
(how settled the policy is).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from crucible.connectors.base import RawRecord, SourceRef
from crucible.core.schema.signal import SignalTarget
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource

FAMILY = "coding/policy_doc"
MODALITY = [  # strongest first: a sentence takes the first modality it matches
    ("must_not", re.compile(r"\b(?:must not|mustn't|shall not|should not|shouldn't|do not|don't|never|avoid|not allowed|prohibited|forbidden|no longer)\b", re.IGNORECASE)),
    ("must", re.compile(r"\b(?:must|shall|required?|have to|has to|need to|always|mandatory|strict)\b", re.IGNORECASE)),
    ("should", re.compile(r"\b(?:should|prefer(?:red|ably)?|recommended|please|ideally|encouraged)\b", re.IGNORECASE)),
]
RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")
MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
LIST_ITEM = re.compile(r"^\s*(?:[*\-+]|\d+[.)])\s+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(*])(?<!e\.g\. )(?<!i\.e\. )")
CONFIG = {".clang-format", ".clang-tidy", "CODEOWNERS", ".editorconfig"}


def modality(sentence: str) -> str | None:
    return next((name for name, rx in MODALITY if rx.search(sentence)), None)


def _strip_markup(s: str) -> str:
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)  # markdown links
    s = re.sub(r"`([^`<]+?)\s*<[^>]+>`_", r"\1", s)  # rst links
    return re.sub(r"\*\*|__|``|`", "", s).strip()


def sections(lines: list[str]) -> list[dict[str, Any]]:
    """Heading-delimited sections with the prose units (paragraph sentences, list items) inside each."""
    out: list[dict[str, Any]] = [{"heading": [], "first": 1, "units": []}]
    levels: list[str] = []  # rst underline characters in order of first use = heading levels
    fence = comment = False
    code_indent: int | None = None
    para: list[tuple[int, str]] = []

    def flush() -> None:
        if para:
            text = _strip_markup(" ".join(t.strip() for _, t in para))
            for sentence in SENTENCE_END.split(text):  # a '.' inside a URL or "e.g." does not end a sentence
                if sentence.strip():
                    out[-1]["units"].append((para[0][0], para[-1][0], sentence.strip()))
            para.clear()

    def open_section(title: str, level: int, line: int) -> None:
        flush()
        parent = out[-1]["heading"][: level - 1]
        out.append({"heading": [*parent, _strip_markup(title)], "first": line, "units": []})

    i = 0
    while i < len(lines):
        ln, raw = i + 1, lines[i]
        s = raw.strip()
        if s.startswith("```"):
            fence = not fence
            flush()
        elif fence:
            pass
        elif "<!--" in s or comment:
            comment = "-->" not in s
            flush()
        elif code_indent is not None and (not s or len(raw) - len(raw.lstrip()) > code_indent):
            pass  # inside an rst directive body
        elif s.startswith(".. "):
            flush()
            code_indent = len(raw) - len(raw.lstrip())
        elif m := MD_HEADING.match(s):
            code_indent = None
            open_section(m.group(2), len(m.group(1)), ln)
        elif s and i + 1 < len(lines) and RST_UNDERLINE.match(lines[i + 1]) and len(lines[i + 1].strip()) >= len(s) * 0.6:
            code_indent = None
            ch = lines[i + 1].strip()[0]
            if ch not in levels:
                levels.append(ch)
            open_section(s, levels.index(ch) + 1, ln)
            i += 1
        elif not s or RST_UNDERLINE.match(s):
            code_indent = None
            flush()
        elif LIST_ITEM.match(raw):
            code_indent = None
            flush()
            para.append((ln, LIST_ITEM.sub("", raw)))
        else:
            code_indent = None
            para.append((ln, raw))
        i += 1
    flush()
    for a, b in zip(out, [*out[1:], None], strict=False):
        a["last"] = (b["first"] - 1) if b else len(lines)
    return [s for s in out if s["units"] or s["heading"]]


def config_rules(path: str, lines: list[str]) -> list[tuple[int, str, dict[str, Any]]]:
    name = Path(path).name
    out = []
    if name == "CODEOWNERS":
        for n, ln in enumerate(lines, 1):
            if ln.strip() and not ln.lstrip().startswith("#"):
                pattern, *owners = ln.split()
                out.append((n, f"{pattern} is owned by {' '.join(owners)}", {"pattern": pattern, "owners": owners}))
    elif name == ".clang-tidy":
        text = "\n".join(lines)
        m = re.search(r"^Checks:\s*(?:'|\"|>-?\s*)?([^\n]*(?:\n\s+[^\n:]+)*)", text, re.MULTILINE)
        for c in re.split(r"[,\s'\"]+", m.group(1) if m else ""):
            if c:
                line = next((n for n, ln in enumerate(lines, 1) if c in ln), 1)
                out.append((line, f"clang-tidy check {'disabled' if c.startswith('-') else 'enabled'}: {c.lstrip('-')}",
                            {"check": c.lstrip("-"), "enabled": not c.startswith("-")}))
    else:  # .clang-format / .editorconfig: key: value (nested keys joined with '.')
        stack: list[tuple[int, str]] = []
        for n, ln in enumerate(lines, 1):
            m = re.match(r"^(\s*)([\w.\[\]*-]+)\s*[:=]\s*(.*?)\s*$", ln)
            if not m or ln.lstrip().startswith(("#", ";")):
                continue
            ind = len(m.group(1))
            while stack and stack[-1][0] >= ind:
                stack.pop()
            if m.group(3):
                key = ".".join([k for _, k in stack] + [m.group(2)])
                out.append((n, f"{key} = {m.group(3)}", {"key": key, "value": m.group(3)}))
            else:
                stack.append((ind, m.group(2)))
    return out


class PolicyDocConnector:
    """scope: {"git_dir": ..., "rev": "master", "paths": [...], "repo": "owner/name"} (documents in git at a revision)
    or {"root": dir, "glob": "*/MIGRATION_LOG.md"} (documents on disk)."""

    name = "policy_docs"

    def __init__(self, privacy: str = "internal"):
        self.privacy = privacy

    def discover(self, scope: dict[str, Any]) -> Iterator[SourceRef]:
        if "git_dir" in scope:
            rev = _git(scope["git_dir"], "rev-parse", scope.get("rev", "HEAD")).strip()
            listed = set(_git(scope["git_dir"], "ls-tree", "-r", "--name-only", rev).splitlines())
            for p in scope["paths"]:
                if p in listed:
                    yield SourceRef(connector=self.name, key=f"{scope.get('repo', Path(scope['git_dir']).stem)}@{rev[:12]}:{p}",
                                    meta={"git_dir": str(scope["git_dir"]), "rev": rev, "path": p, "privacy": scope.get("privacy", self.privacy)})
        else:
            root = Path(scope["root"])
            for f in sorted(root.glob(scope["glob"])):
                yield SourceRef(connector=self.name, key=f.relative_to(root).as_posix(), meta={"file": str(f), "path": f.relative_to(root).as_posix(),
                                                                                               "privacy": scope.get("privacy", self.privacy)})

    def fetch(self, ref: SourceRef) -> RawRecord:
        m = ref.meta
        if "git_dir" in m:
            text = _git(m["git_dir"], "show", f"{m['rev']}:{m['path']}")
            log = _git(m["git_dir"], "log", "--format=%H %aI", m["rev"], "--", m["path"]).splitlines()
            history = {"rev": m["rev"], "commits": len(log), "first_change": log[-1].split()[1] if log else None,
                       "last_change": log[0].split()[1] if log else None, "last_commit": log[0].split()[0] if log else None}
        else:
            text, history = Path(m["file"]).read_text(errors="replace"), {}
        return RawRecord(ref=ref, payload={"text": text, "history": history}, privacy=m.get("privacy", self.privacy))

    def normalize(self, raw: RawRecord) -> Trace:
        path, lines = raw.ref.meta["path"], raw.payload["text"].splitlines()
        hist = raw.payload["history"]
        time = hist.get("last_change")
        events: list[TraceEvent] = []
        counts = {"must_not": 0, "must": 0, "should": 0}
        if Path(path).name in CONFIG:
            for n, text, data in config_rules(path, lines):
                events.append(TraceEvent(seq=len(events), time=time, kind="state", actor="doc", type="config_rule", text=text,
                                         target=SignalTarget(object=f"doc:{path}", span=[n, n]), data={"path": path, "line": n, **data}))
        else:
            for sec in sections(lines):
                sseq = len(events)
                title = " > ".join(sec["heading"]) or "(preamble)"
                events.append(TraceEvent(seq=sseq, time=time, kind="state", actor="doc", type="policy_section", text=title,
                                         target=SignalTarget(object=f"doc:{path}", span=[sec["first"], sec["last"]]),
                                         data={"path": path, "heading": sec["heading"], "units": len(sec["units"])}))
                for a, b, text in sec["units"]:
                    mod = modality(text) or ("must" if "checklist" in title.lower() else None)  # checklist items are requirements
                    if mod is None or len(text) < 12:
                        continue
                    counts[mod] += 1
                    events.append(TraceEvent(seq=len(events), time=time, kind="state", actor="doc", type="policy_rule", text=text[:600],
                                             reply_to=sseq, target=SignalTarget(object=f"doc:{path}", span=[a, b]),
                                             data={"path": path, "section": title, "modality": mod}))
        details = {"path": path, **hist, "rules": counts, "config_rules": sum(e.type == "config_rule" for e in events)}
        return Trace(trace_id=f"policy:{raw.ref.key}", family=FAMILY,
                     source=TraceSource(connector=self.name, ref=raw.ref.key, privacy=raw.privacy),  # type: ignore[arg-type]
                     actors=[TraceActor(id="doc", role="policy", kind="system")], events=events,
                     outcome=TraceOutcome(label="unknown", label_source="policy text: candidate rubric items, unvalidated (§6.5)", details=details))


def _git(git_dir: str | Path, *args: str) -> str:
    out = subprocess.run(["git", f"--git-dir={git_dir}", *args], capture_output=True, timeout=300, check=True)
    return out.stdout.decode("utf-8", "replace")
