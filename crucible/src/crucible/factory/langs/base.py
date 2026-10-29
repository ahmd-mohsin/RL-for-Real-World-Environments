"""The ``LangProfile`` protocol, the active-profile context, and the text tools every C-like profile shares
(swarm-pipeline/07 §1). Standard library only: ``lanes.executable`` and ``lanes.windows_git`` import this module.

**The active profile.** The window machinery (``windows.validate_window``, the prober, the mutants, grading, the path
checks, rebreak) was written for Python and decides "source", "test", "protected" through a handful of predicates
(``executable.is_test_path`` / ``is_protected``, ``windows_git.is_source`` / ``is_test_file`` / ``_defs`` /
``key_of_test``, ``windows.suite_targets`` / ``changed_defs``). Each of them asks ``active()`` first: ``None`` (no
profile, or the Python profile) keeps the Python behaviour byte for byte; any other profile answers instead. A
profile is made active with ``use(profile)`` (a ``contextvars`` scope: it follows ``asyncio`` tasks created inside it
and ``asyncio.to_thread``, never another task), so two languages can run in one process. The debug lane sets it around
everything it calls and ``DebugTask`` around each of its own methods.

**Test identity.** A test id is ``<file>::<key>``: the file is the test file the runner ran (a valid run target), the key
is what the language's runner reports under it (Go ``TestX/sub``, libtest ``mod::name``, vitest/jest ``describe::title``,
surefire ``method``). ``key_of_test`` maps an id to the definition that owns it (Go ``TestX``, parametrized cases
dropped). The executed fail-to-pass set is the truth about which tests a change makes pass; ``test_defs`` (regex plus a
string- and comment-aware bracket matcher, no parser dependency) only proposes which test functions a commit added or
changed, to hide them from the agent's tree.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

_ACTIVE: contextvars.ContextVar[Any] = contextvars.ContextVar("crucible_lang_profile", default=None)


def active() -> Any:
    """The active non-Python profile, or None (no profile, or Python: the original behaviour)."""
    p = _ACTIVE.get()
    return None if p is None or getattr(p, "name", "python") == "python" else p


def current() -> Any:
    """The active profile including Python (None when none was set)."""
    return _ACTIVE.get()


@contextlib.contextmanager
def use(profile: Any) -> Iterator[Any]:
    tok = _ACTIVE.set(profile)
    try:
        yield profile
    finally:
        _ACTIVE.reset(tok)


@runtime_checkable
class LangProfile(Protocol):
    """One language (the stable contract other lanes build on: swarm-pipeline/07 §2). Paths are repository-relative POSIX
    paths. The first block is what the window machinery asks through ``active()``; the second is what the debug lane
    and ``runtime`` call. ``clike.CLikeProfile`` implements both from a few patterns and five hooks; ``python.PythonProfile``
    adapts the existing lane (its runtime members are the Python lane's own image book and sandbox).

    Runtime members (not checked by ``isinstance``, documented here): ``images(**kw) -> ImageBook-like`` (``ensure(repo,
    base_dir) -> card`` with ``image``, ``digest``, ``ok``, ``project``, ``projects``); ``packaging_files(tree)``,
    ``image_info(tree)``, ``dockerfile(tree, info)``, ``build_files(tree, info)`` (the World recipe, keyed by
    ``recipe`` and ``base_image``); ``run_script(tree, targets) -> shell`` that writes its machine report to
    ``$O/report`` and ``parse_report(report, tree, targets, tail) -> ({test id: passed|failed|error|skipped}, notes)``
    (a note ``build failed: ...`` marks a target that did not compile); ``leak_probes(image, base_dir, *,
    new_test_names, ...)`` with ``fetch_probe(project)``, ``own_copy_probe(tree, project)``, ``leak_needle(name)``,
    ``leak_globs``; ``world_setup_cmd(ws)``; ``timeout_s``, ``tmpfs``, ``plumbing_words`` (the epilogue)."""

    name: str
    extensions: tuple[str, ...]

    def detect(self, tree: Path) -> bool: ...
    def is_source(self, path: str) -> bool: ...  # code the agent fixes (a fix's source revert, the prober, mutants)
    def is_test_file(self, path: str) -> bool: ...  # a file that defines tests
    def is_test_path(self, path: str) -> bool: ...  # anything test-side: test files, fixtures, snapshots, test data
    def is_protected(self, path: str) -> bool: ...  # test paths, build and CI configuration, lockfiles, plumbing
    def is_plumbing(self, path: str) -> bool: ...  # what could steer a test run (path:plumbing, hack severity)
    def test_defs(self, text: str | None, path: str = "") -> dict[str, str]: ...
    def strip_tests(self, text: str, keys: set[str], path: str = "") -> str | None: ...
    def changed_defs(self, before: str | None, after: str | None, path: str = "") -> list[str]: ...
    def key_of_test(self, test_id: str) -> tuple[str, str]: ...
    def suite_targets(self, tree: Path) -> list[str]: ...


# ---- path rules shared by profiles -------------------------------------------------------------------------------------
NOT_SOURCE = re.compile(r"(^|/)(docs?|examples?|benchmarks?|bench|scripts?|tools|\.github|vendor|node_modules|third_party|dist|build|target|coverage)/")
CI_CONFIG = re.compile(r"^\.github/|^\.gitlab-ci\.yml$|^\.travis\.yml$|^(azure-pipelines|appveyor|codecov)\.ya?ml$|^\.circleci/|(^|/)(GNU)?[Mm]akefile$"
                       r"|^\.pre-commit-config\.yaml$|^\.editorconfig$|^\.gitattributes$|^\.gitignore$")
PROSE = re.compile(r"\.(md|rst|txt|adoc)$|(^|/)(docs?|changes|changelog|history|news)(/|$)|(^|/)(CHANGES|CHANGELOG|HISTORY|NEWS|RELEASE[-_ ]?NOTES)[^/]*$",
                   re.IGNORECASE)


def ext_of(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name[name.rfind("."):] if "." in name else ""


# ---- a string- and comment-aware bracket matcher for C-like languages ------------------------------------------------------
_OPEN, _CLOSE = "([{", ")]}"
_REGEX_PREV = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_WORDS = ("return", "typeof", "case", "do", "else", "in", "of", "delete", "void", "throw", "new", "yield", "await")


def _skip_string(text: str, i: int, q: str, multiline: bool = False) -> int:
    """Index after the string literal opened by ``q`` at ``i`` (escapes honoured; a newline ends a broken one, except in
    a language whose plain strings span lines: Rust)."""
    n = len(text)
    j = i + 1
    while j < n:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == q:
            return j + 1
        if c == "\n" and q != "`" and not multiline:
            return j
        j += 1
    return n


def _skip_template(text: str, i: int, lang: str) -> int:
    """A JS template literal from ``i`` (the backtick), ``${...}`` expressions matched recursively."""
    n = len(text)
    j = i + 1
    while j < n:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "`":
            return j + 1
        if c == "$" and j + 1 < n and text[j + 1] == "{":
            end = match_bracket(text, j + 1, lang)
            if end < 0:
                return n
            j = end
            continue
        j += 1
    return n


def _skip_regex(text: str, i: int) -> int:
    n = len(text)
    j = i + 1
    cls = False
    while j < n:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "\n":
            return j
        if cls:
            cls = c != "]"
        elif c == "[":
            cls = True
        elif c == "/":
            j += 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            return j
        j += 1
    return n


def _regex_allowed(text: str, i: int) -> bool:
    k = i - 1
    while k >= 0 and text[k] in " \t\r\n":
        k -= 1
    if k < 0 or text[k] in _REGEX_PREV:
        return True
    m = re.search(r"([A-Za-z_$][\w$]*)$", text[max(0, k - 12): k + 1])
    return bool(m and m.group(1) in _REGEX_WORDS)


def _skip_rust_raw(text: str, i: int) -> int | None:
    """``r"..."``, ``r#"..."#``, ``br#"..."#`` starting at ``i`` (the ``r`` or ``b``), or None."""
    m = re.match(r"b?r(#*)\"", text[i: i + 12])
    if not m or (i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_")):
        return None
    close = '"' + m.group(1)
    end = text.find(close, i + m.end())
    return len(text) if end < 0 else end + len(close)


def skip_noncode(text: str, i: int, lang: str) -> int | None:
    """When ``text[i]`` starts a comment or a literal, the index after it; else None."""
    c = text[i]
    nxt = text[i + 1] if i + 1 < len(text) else ""
    if c == "/" and nxt == "/":
        j = text.find("\n", i)
        return len(text) if j < 0 else j
    if c == "/" and nxt == "*":
        if lang == "rust":  # nested block comments
            depth, j = 1, i + 2
            while j < len(text) and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            return j
        j = text.find("*/", i + 2)
        return len(text) if j < 0 else j + 2
    if c == '"':
        if lang == "java" and text.startswith('"""', i):
            j = text.find('"""', i + 3)
            return len(text) if j < 0 else j + 3
        # a Rust string spans lines (gleam's formatter tests: "pub fn main() {\n ... \n}" as one literal, whose "{" hid
        # 13 of 35 tests in a file from the profile)
        return _skip_string(text, i, '"', multiline=lang == "rust")
    if c == "`":
        if lang == "go":
            j = text.find("`", i + 1)
            return len(text) if j < 0 else j + 1
        if lang == "node":
            return _skip_template(text, i, lang)
    if c == "'":
        if lang == "rust":  # a char literal, or a lifetime / label ('a)
            m = re.match(r"'(\\.[^']*|[^\\'\n])'", text[i: i + 12])
            return i + m.end() if m else i + 1
        return _skip_string(text, i, "'")
    if lang == "rust" and c in "rb":
        return _skip_rust_raw(text, i)
    if lang == "node" and c == "/" and _regex_allowed(text, i):
        return _skip_regex(text, i)
    if c == "#" and lang == "python":
        j = text.find("\n", i)
        return len(text) if j < 0 else j
    return None


def match_bracket(text: str, i: int, lang: str) -> int:
    """Index just after the bracket that closes ``text[i]`` (one of ``([{``), skipping strings, comments, template
    literals and regex literals; -1 when it never closes."""
    stack = [text[i]]
    j = i + 1
    n = len(text)
    while j < n:
        c = text[j]
        s = skip_noncode(text, j, lang)
        if s is not None and s > j:
            j = s
            continue
        if c in _OPEN:
            stack.append(c)
        elif c in _CLOSE:
            if not stack or _OPEN.index(stack[-1]) != _CLOSE.index(c):
                return -1
            stack.pop()
            if not stack:
                return j + 1
        j += 1
    return -1


def code_mask(text: str, lang: str) -> str:
    """``text`` with comments and literals blanked (same length, newlines kept): regexes then find only real code."""
    out = list(text)
    j = 0
    n = len(text)
    while j < n:
        s = skip_noncode(text, j, lang)
        if s is not None and s > j + 1:
            for k in range(j + 1, min(s, n) - 1):
                if out[k] != "\n":
                    out[k] = " "
            j = s
            continue
        j += 1
    return "".join(out)


def line_start(text: str, i: int) -> int:
    return text.rfind("\n", 0, i) + 1


def line_end(text: str, i: int) -> int:
    """Index after the newline that ends the line holding ``i`` (or the text's end)."""
    j = text.find("\n", i)
    return len(text) if j < 0 else j + 1


def body_hash(text: str) -> str:
    """What "the same test" means: its text with whitespace runs collapsed."""
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()[:16]


def remove_spans(text: str, spans: Sequence[tuple[int, int]]) -> str:
    """``text`` without the (start, end) spans, each widened to whole lines; blank-line runs left behind collapse."""
    out = text
    for a, b in sorted(spans, key=lambda s: -s[0]):
        a2, b2 = line_start(out, a), line_end(out, max(a, b - 1))
        out = out[:a2] + out[b2:]
    return re.sub(r"\n{3,}", "\n\n", out)


def changed_keys(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    return [k for k in after if before.get(k) != after[k]]


CHANGELOG_FILE = re.compile(r"(^|/)(CHANGES|CHANGELOG|HISTORY|NEWS|RELEASE[-_ ]?NOTES)[^/]*$", re.IGNORECASE)


def fix_refs(subject: str, body: str, sha: str) -> set[str]:
    """What a release changelog cites a fix by: its PR or issue numbers (``#1571``) and its short sha (``884502a``)."""
    return {f"#{n}" for n in re.findall(r"#(\d+)", f"{subject}\n{body}")} | {sha[:7]}


SUBJECT_REF = "subject:"
_CC_PREFIX = re.compile(r"^\s*(?:[^\w\s]+\s*)?[a-z]+(?:\([^)]*\))?!?:\s*", re.IGNORECASE)  # a gitmoji may lead: "🐞 fix(x): "
_STOP = frozenset("a an the is are was be of to in on at for when while it its and or by with from as that this not no".split())


def _prose(text: str) -> str:
    """Lower case, markup and punctuation dropped, whitespace collapsed: how a changelog line and a commit subject compare."""
    return " ".join(re.sub(r"[`*_\[\]()\"'.,;:!?]", " ", text.lower()).split())


def _stem(word: str) -> str:
    """A crude stem, the same for "update" / "updating" / "updates" and "drop" / "dropping": release notes turn a fix's
    imperative into a gerund ("`setValue` not updating ..." for "fix(setValue): update ...")."""
    for suf in ("ing", "ed", "es", "s", "e"):
        if len(word) > len(suf) + 2 and word.endswith(suf):
            word = word[:-len(suf)]
            break
    return word[:-1] if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou" else word


def _content(prose: str) -> set[str]:
    return {_stem(w) for w in prose.split() if len(w) > 1 and w not in _STOP}


def subject_ref(subject: str) -> str | None:
    """A fix cited by its subject alone: a changelog entry that repeats the commit subject (``fix(static): preserve
    matched handler 404s`` -> "preserve matched handler 404s") with no PR, issue or sha to match (echo 48128ab, 2026-10-24:
    its changelog line kept the PR URL in the start tree). The conventional-commit prefix and a trailing ``(#n)`` are
    dropped; a subject under 4 words or 20 characters cites nothing (too generic to match safely)."""
    s = _prose(re.sub(r"\s*\(#\d+\)\s*$", "", _CC_PREFIX.sub("", subject.strip())))
    return f"{SUBJECT_REF}{s}" if len(s) >= 20 and len(s.split()) >= 4 else None


def strip_ref_entries(text: str, refs: Iterable[str]) -> tuple[str, int]:
    """``text`` without the changelog entries that cite one of ``refs`` (``fix_refs``): a list item with its wrapped lines,
    else the line. A changelog written at release time names the fixes after the fact, so the fixes' own diffs never
    added these lines (``strip_added_lines`` cannot see them): mvdan-sh's "as well as ``$-`` - #1398", h3's "Call
    ``onError`` for stream errors in Node.js handlers (#1571)". Returns the text and the number of lines removed.
    A subject of 6 or more distinct words also cites the entries holding all of its words but one: release notes reword a
    fix slightly (isort 9f56767 "Fix pylint disable-next comments at top of imports" is "... at the start of imports" in
    its changelog), and an entry removed by mistake costs nothing while one left in announces the fix. A subject of 5 or
    more content words (stop words dropped, words stemmed) also cites the entries holding all of them, all but one from
    6 on: react-hook-form's "🐞 fix(setValue): update field array rows when the parent object is set" is "`setValue` not
    updating field array rows when a parent object is set" in its changelog."""
    refs = list(refs)
    subjects = [r[len(SUBJECT_REF):] for r in refs if r.startswith(SUBJECT_REF)]
    loose = [w for x in subjects if len(w := set(x.split())) >= 6]
    stems = [w for x in subjects if len(w := _content(x)) >= 5]
    pats = [re.compile(rf"(?<![\w/])#{re.escape(r[1:])}(?!\d)|/(?:pull|issues)/{re.escape(r[1:])}(?!\d)") if r.startswith("#")
            else re.compile(rf"(?<![0-9a-fA-F]){re.escape(r)}(?![0-9a-fA-F])") for r in refs if not r.startswith(SUBJECT_REF)]
    lines = text.splitlines(keepends=True)
    item = re.compile(r"^(\s*)([-*+]|\d+[.)])\s")
    out, k, removed = [], 0, 0
    while k < len(lines):
        end = k + 1
        if m := item.match(lines[k]):  # the item's wrapped lines: indented deeper, not another item
            while end < len(lines) and lines[end].strip() and not item.match(lines[end]) and len(lines[end]) - len(lines[end].lstrip()) > len(m.group(1)):
                end += 1
        chunk = lines[k:end]
        prose = f" {_prose(''.join(chunk))} "
        if (any(p.search(ln) for ln in chunk for p in pats) or any(f" {x} " in prose for x in subjects)
                or any(len(w - set(prose.split())) <= 1 for w in loose)
                or any(len(w - _content(prose)) <= (len(w) >= 6) for w in stems)):
            removed += len(chunk)
        else:
            out += chunk
        k = end
    return "".join(out), removed


def strip_added_lines(text: str, added: Sequence[str]) -> str:
    """``text`` without one occurrence of each line in ``added`` (exact, newline excluded), last occurrence first: the
    lines a fix added to a prose file (its changelog entry), so the start tree does not announce the fix."""
    lines = text.splitlines(keepends=True)
    for a in added:
        if not a.strip():
            continue
        for k in range(len(lines) - 1, -1, -1):
            if lines[k].rstrip("\r\n") == a:
                del lines[k]
                break
    return "".join(lines)
