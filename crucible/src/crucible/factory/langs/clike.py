"""What the compiled-language profiles share (swarm-pipeline/07 §1): path rules from per-language patterns, test and
function definitions from ``_find_tests`` / ``_find_funcs`` spans (``base.match_bracket``), hiding tests by span
removal, suite targets, and the runtime hooks (``runtime``: image book, sandbox, leak probes) wired to the profile.

A subclass sets the patterns and implements ``_find_tests(text, path) -> [(key, start, end)]`` (the whole definition,
attributes and decorators included), ``_find_funcs(text, path) -> [(name, start, end)]`` (functions and methods, for
``simplify('locate')`` and the faulty-name lint), the image recipe (``dockerfile``, ``packaging_files``,
``image_info``), ``run_script`` and ``parse_report``, and the leak hooks (``fetch_probe``, ``own_copy_probe``,
``leak_needle``, ``leak_globs``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.langs.base import CI_CONFIG, NOT_SOURCE, PROSE, body_hash, ext_of, remove_spans

SKIP_DIRS = {".git", "node_modules", "target", "vendor", "dist", "build", ".venv", "__pycache__", "coverage", ".next", ".turbo"}


class CLikeProfile:
    name = "clike"
    lang = "clike"  # the lexer dialect of ``base.match_bracket``
    extensions: tuple[str, ...] = ()
    test_file_rx: re.Pattern[str] = re.compile(r"$^")
    test_path_rx: re.Pattern[str] = re.compile(r"$^")
    protected_names: frozenset[str] = frozenset()
    protected_rx: re.Pattern[str] = re.compile(r"$^")
    plumbing_rx: re.Pattern[str] = re.compile(r"$^")
    not_source_rx: re.Pattern[str] = NOT_SOURCE
    recipe = "1"
    base_image = ""
    timeout_s = 900.0
    tmpfs = "1g"
    leak_globs: tuple[str, ...] = ()
    plumbing_words = ""  # the epilogue's list of what counts as test plumbing in this language
    install_words = "install packages"

    # -- paths -------------------------------------------------------------------------------------------------------------
    def is_test_file(self, path: str) -> bool:
        return ext_of(path) in self.extensions and bool(self.test_file_rx.search(path))

    def is_test_path(self, path: str) -> bool:
        return self.is_test_file(path) or bool(self.test_path_rx.search(path))

    def is_plumbing(self, path: str) -> bool:
        return bool(self.plumbing_rx.search(path))

    def is_protected(self, path: str) -> bool:
        name = path.rsplit("/", 1)[-1]
        return (self.is_test_path(path) or name in self.protected_names or bool(self.protected_rx.search(path)) or self.is_plumbing(path)
                or bool(CI_CONFIG.search(path)))

    def is_source(self, path: str) -> bool:
        return (ext_of(path) in self.extensions and not self.is_test_path(path) and not self.not_source_rx.search(path) and not self.is_protected(path))

    def is_prose(self, path: str) -> bool:
        return bool(PROSE.search(path))

    def detect(self, tree: Path) -> bool:
        return False

    # -- definitions ---------------------------------------------------------------------------------------------------------
    def _find_tests(self, text: str, path: str) -> list[tuple[str, int, int]]:
        raise NotImplementedError

    def _find_funcs(self, text: str, path: str) -> list[tuple[str, int, int]]:
        raise NotImplementedError

    def test_defs(self, text: str | None, path: str = "") -> dict[str, str]:
        """``key -> body hash`` of every test the file defines (the ``windows_git._defs`` contract)."""
        if not text:
            return {}
        out: dict[str, str] = {}
        for k, a, b in self._find_tests(text, path):
            out.setdefault(k, body_hash(text[a:b]))
        return out

    def strip_tests(self, text: str, keys: set[str], path: str = "") -> str | None:
        """``text`` without the tests ``keys``; None when no test of the file is left (the caller removes the file: an
        empty suite file fails some runners)."""
        found = self._find_tests(text, path)
        spans = [(a, b) for k, a, b in found if k in keys]
        if not spans:
            return text
        if all(k in keys for k, _, _ in found):
            return None
        return self.after_strip(remove_spans(text, spans), path)

    def after_strip(self, text: str, path: str) -> str:
        return text

    def changed_defs(self, before: str | None, after: str | None, path: str = "") -> list[str]:
        def defs(t: str | None) -> dict[str, str]:
            if not t:
                return {}
            out: dict[str, str] = {}
            for n, a, b in self._find_funcs(t, path):
                out.setdefault(n, body_hash(t[a:b]))
            return out

        a, b = defs(before), defs(after)
        return [q for q in b if a.get(q) != b[q] and not q.rsplit(".", 1)[-1].startswith(("Test", "test"))]

    def key_of_test(self, test_id: str) -> tuple[str, str]:
        f, _, k = test_id.partition("::")
        return f, k

    # -- suites -------------------------------------------------------------------------------------------------------------
    def walk(self, tree: Path) -> list[str]:
        tree = Path(tree)
        out: list[str] = []
        for p in tree.rglob("*"):
            rel = p.relative_to(tree)
            if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts[:-1]):
                continue
            if p.is_file():
                out.append(rel.as_posix())
        return sorted(out)

    def suite_targets(self, tree: Path) -> list[str]:
        return [p for p in self.walk(tree) if self.is_test_file(p) and not p.startswith(("docs/", "doc/", "examples/", "example/"))]

    def source_text(self, tree: Path) -> str:
        """Every code file of the tree, concatenated (what ``new_test_names`` reads to drop names A already has)."""
        parts = []
        for rel in self.walk(tree):
            if ext_of(rel) in self.extensions:
                try:
                    parts.append((Path(tree) / rel).read_text(errors="replace"))
                except OSError:
                    continue
        return "\n".join(parts)

    def new_test_names(self, tests_b: Mapping[str, str], a_tests: Mapping[str, str | None], tree: Path) -> set[str]:
        """Leaf names of tests B's files define that A's tree never mentions (the leak probe's needles)."""
        names: set[str] = set()
        for p, txt in tests_b.items():
            before = set(self.test_defs(a_tests.get(p), p))
            names |= {self.leaf(k) for k in set(self.test_defs(txt, p)) - before}
        a_text = self.source_text(tree)
        return {n for n in names if n and len(n) >= 6 and n not in a_text}

    def leaf(self, key: str) -> str:
        return key.rsplit("::", 1)[-1]

    # -- runtime hooks ---------------------------------------------------------------------------------------------------------
    def images(self, **kw: Any) -> Any:
        from crucible.factory.langs.runtime import LangImageBook

        return LangImageBook(self, **kw)

    def packaging_files(self, tree: Path) -> list[str]:
        raise NotImplementedError

    def image_info(self, tree: Path) -> dict[str, Any]:
        return {}

    def build_files(self, tree: Path, info: Mapping[str, Any]) -> dict[str, str]:
        return {}

    def dockerfile(self, tree: Path, info: Mapping[str, Any]) -> str:
        raise NotImplementedError

    def run_script(self, tree: Path, targets: Sequence[str]) -> str:
        raise NotImplementedError

    def parse_report(self, report: str, tree: Path, targets: Sequence[str], tail: str = "") -> tuple[dict[str, str], list[str]]:
        raise NotImplementedError

    async def leak_probes(self, image: str, base_dir: Path, *, new_test_names: set[str], name_prefix: str | None = None, **kw: Any) -> dict[str, Any]:
        from crucible.factory.langs.runtime import PREFIX, lang_leak_probes

        return await lang_leak_probes(image, base_dir, profile=self, new_test_names=new_test_names, name_prefix=name_prefix or f"{PREFIX}leak-")

    def fetch_probe(self, project: str | None) -> str:
        return "exit 1"

    def own_copy_probe(self, tree: Path, project: str | None) -> str:
        return "true"

    def leak_needle(self, name: str) -> str:
        return name

    def world_setup_cmd(self, ws: str) -> str:
        return ""

    def tests_in_target(self, tree: Path, target: str) -> list[str]:
        """Ids of the tests a target file defines (``target::key``), from the file in ``tree``: what a target whose build
        failed reports as ``error``."""
        f = Path(tree) / target
        try:
            text = f.read_text(errors="replace")
        except OSError:
            return []
        return [f"{target}::{k}" for k in self.test_defs(text, target)]
