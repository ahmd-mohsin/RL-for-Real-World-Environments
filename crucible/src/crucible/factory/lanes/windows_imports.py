"""Import-tolerant grading for the window lane (report 11 F8; swarm-pipeline/00 §6).

A hidden test file that imports, at module level, a name the window adds late (``from semver import cmd_check``) fails
to collect on every tree that lacks the name: every test in the file fails, including tests that never touch it. Prefix
ownership then collapses (no test passes before the unit that adds the last name) and a solver that finishes four of
five milestones scores zero on all of them.

``tolerant_tests`` rewrites the grader's copy of B's test files (never the agent's tree; the grader writes them over it)
so that each module-level ``from <project module> import name`` (and ``import <project module>``) is guarded per name:
when the import raises ``ImportError`` the name is bound to a placeholder that raises ``MissingName`` on any use (call,
attribute, item, iteration, ``isinstance``, subclassing). Each test then fails or passes on its own:

- a test that uses the missing name fails (``MissingName`` derives from ``BaseException``, so ``pytest.raises(Exception)``
  and ``except Exception`` in a test never swallow it);
- a test that does not use it runs as it would on a complete tree;
- the placeholder answers ``__name__`` with the imported name, so ``parametrize`` ids are the same as on the oracle;
- an import that raises anything other than ``ImportError`` (a syntax error in the agent's module) still fails the
  collection, as before.

Only top-level statements that sit alone on their lines are rewritten; everything else (a guarded import the test file
already has, ``import *``, relative imports of test helpers, imports of other packages) is left byte for byte. A window
records whether it was mined this way (``rec["import_tolerant"]``); grading always uses the window's own mode, because
fail-to-pass tests and their owners were classified under that mode (a strict-mined window graded tolerantly would let
the start state pass tests that were fail-to-pass only through the import).
"""

from __future__ import annotations

import ast
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path

GUARD_MARK = "# crucible: import-tolerant grading (F8)"
PRELUDE = f'''{GUARD_MARK}
class _CrucibleMissingName(BaseException):
    """A name a hidden test imports that the tree under test does not define."""


class _CrucibleMissing:
    def __init__(self, module, name):
        self.__dict__.update(__name__=name, __qualname__=name, __module__=module, _crucible_missing=module + "." + name)

    def _crucible_fail(self, *args, **kwargs):
        raise _CrucibleMissingName("cannot import name %r from %r" % (self.__name__, self.__module__))

    def __getattr__(self, attr):
        if attr.startswith("_") or attr.startswith("pytest"):  # what pytest's collection probes (fixture markers, marks)
            raise AttributeError(attr)
        self._crucible_fail()

    __call__ = __getitem__ = __iter__ = __enter__ = __exit__ = _crucible_fail
    __instancecheck__ = __subclasscheck__ = __mro_entries__ = _crucible_fail

'''


def project_roots(base_dir: Path, extra_paths: Iterable[str] = ()) -> set[str]:
    """Top-level import names the project defines: packages and modules at the tree root or under ``src/``, plus the roots
    of ``extra_paths`` (B's source files, for a package the window adds)."""
    out: set[str] = set()
    for root in (Path(base_dir), Path(base_dir) / "src"):
        if not root.is_dir():
            continue
        for p in root.iterdir():
            if p.is_file() and p.suffix == ".py" and p.stem.isidentifier() and p.stem not in ("setup", "conftest", "noxfile"):
                out.add(p.stem)
            elif p.is_dir() and (p / "__init__.py").is_file() and p.name.isidentifier():
                out.add(p.name)
    for path in extra_paths:
        parts = path.split("/")
        if parts[0] == "src":
            parts = parts[1:]
        if parts and parts[-1].endswith(".py"):
            head = parts[0][:-3] if len(parts) == 1 else parts[0]
            if head.isidentifier() and head not in ("setup", "conftest", "noxfile"):
                out.add(head)
    return out


def _alone(body: list[ast.stmt], i: int, lines: list[str]) -> bool:
    """Statement i occupies its own lines: nothing before it on its first line, nothing after it on its last."""
    n = body[i]
    if n.col_offset != 0 or n.end_lineno is None or n.end_col_offset is None:
        return False
    if i > 0 and (body[i - 1].end_lineno or 0) >= n.lineno:
        return False
    if i + 1 < len(body) and body[i + 1].lineno <= n.end_lineno:
        return False
    rest = lines[n.end_lineno - 1][n.end_col_offset:]
    return rest.strip() == "" or rest.lstrip().startswith("#")


def _guard_from(node: ast.ImportFrom) -> str:
    out = []
    for a in node.names:
        bound = a.asname or a.name
        imp = f"from {node.module} import {a.name}" + (f" as {a.asname}" if a.asname else "")
        out.append(f"try:\n    {imp}\nexcept ImportError:\n    {bound} = _CrucibleMissing({node.module!r}, {a.name!r})\n")
    return "".join(out)


def _guard_import(node: ast.Import) -> str:
    out = []
    for a in node.names:
        if a.asname:
            out.append(f"try:\n    import {a.name} as {a.asname}\nexcept ImportError:\n    {a.asname} = _CrucibleMissing({a.name!r}, {a.name.rsplit('.', 1)[-1]!r})\n")
            continue
        root = a.name.split(".")[0]
        if "." in a.name:  # ``import pkg.sub`` binds ``pkg``: keep ``pkg`` when only the submodule is missing
            out.append(f"try:\n    import {a.name}\nexcept ImportError:\n    try:\n        import {root}\n    except ImportError:\n"
                       f"        {root} = _CrucibleMissing({root!r}, {root!r})\n")
        else:
            out.append(f"try:\n    import {a.name}\nexcept ImportError:\n    {root} = _CrucibleMissing({root!r}, {root!r})\n")
    return "".join(out)


def tolerant_source(text: str, roots: set[str]) -> tuple[str, list[str]]:
    """One test file with its module-level project imports guarded per name. Returns (text, guarded names); the text is
    returned unchanged when nothing is guarded (or it does not parse)."""
    if GUARD_MARK in text:
        return text, []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError:
        return text, []
    lines = text.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []
    names: list[str] = []
    for i, n in enumerate(tree.body):
        if isinstance(n, ast.ImportFrom):
            if n.level != 0 or not n.module or n.module == "__future__" or n.module.split(".")[0] not in roots or any(a.name == "*" for a in n.names):
                continue
            new = _guard_from(n)
            got = [a.asname or a.name for a in n.names]
        elif isinstance(n, ast.Import):
            if not all(a.name.split(".")[0] in roots for a in n.names):
                continue
            new = _guard_import(n)
            got = [a.asname or a.name.split(".")[0] for a in n.names]
        else:
            continue
        if not _alone(tree.body, i, lines):
            continue
        edits.append((n.lineno, n.end_lineno or n.lineno, new))
        names += got
    if not edits:
        return text, []
    for k, (start, end, new) in enumerate(sorted(edits, reverse=True)):
        if k == len(edits) - 1:  # the first guarded import: the placeholder is defined just before it
            new = PRELUDE + new
        lines[start - 1:end] = [new]
    return "".join(lines), names


def tolerant_tests(tests: Mapping[str, str], roots: set[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """The grader's copy of B's test files, import-tolerant. Returns (files, {path: guarded names})."""
    out: dict[str, str] = {}
    guarded: dict[str, list[str]] = {}
    for p, text in tests.items():
        if not p.endswith(".py"):
            out[p] = text
            continue
        new, names = tolerant_source(text, roots)
        out[p] = new
        if names:
            guarded[p] = names
    return out, guarded
