"""The C++/CUDA profile (swarm-pipeline/10 §2): the kernel debugging lane's language, on the debug lane's ``LangProfile``
protocol (swarm-pipeline/07 §2).

- **Paths.** Source: ``.cu``/``.cuh`` and the C++ that hosts kernels (``.cpp``/``.cc``/``.h``/``.hpp``/``.inl``) outside
  tests, benchmarks, examples, docs and vendored trees. Tests: ``tests/test-*.cpp`` (ggml's op test is the harness),
  ``test_*.cu``/``*_test.cu``, gtest and Catch2 files. Protected: tests, every build file (``CMakeLists.txt``,
  ``*.cmake``, Makefiles, ``cmake/``), CI. Plumbing (hack severity): build files, because they choose the CUDA
  architectures, flags and what gets compiled.
- **Definitions.** ``_find_funcs`` finds C++/CUDA function definitions (free functions, templates, kernels with
  ``__global__``/``__device__``/``__host__``/``static``/``inline`` qualifiers, methods) with a comment-, string- and
  char-literal-aware brace matcher; ``kernels`` lists the ``__global__`` ones. ``changed_defs`` (the faulty-name lint, the
  locate simplification) diffs their bodies. Tests: gtest ``TEST*``, Catch2 ``TEST_CASE``; a ggml op-test case is a
  ``test_cases.emplace_back(new test_<op>(...))`` line, keyed by its text (the grader enumerates cases by executing
  the harness, never from this text).

The class subclasses the shared ``CLikeProfile`` when the debug lane's language layer is present (its runtime hooks:
image book, sandbox) and is self-sufficient otherwise: every method the kernel lane calls is defined here. Register it
in ``langs.NAMES`` (``"cuda": "cuda"``) when the two lanes meet.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

try:  # the debug lane's shared C-like base (factory/langs/clike.py); absent on a tree built before it merges
    from crucible.factory.langs.clike import CLikeProfile as _Base
except ImportError:  # pragma: no cover - exercised only without the debug lane's language layer
    _Base = object  # type: ignore[assignment,misc]

EXTENSIONS = (".cu", ".cuh", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hh", ".inl")
TEST_FILE = re.compile(r"(^|/)tests?/test-[^/]+\.(cpp|cc|cu|c)$|(^|/)test_[^/]+\.(cu|cpp|cc)$|_test\.(cu|cpp|cc)$|(^|/)(unit_?tests?|gtest)/")
TEST_PATH = re.compile(r"(^|/)(tests?|testing|testdata|test_data|fixtures?)/")
NOT_SOURCE = re.compile(r"(^|/)(docs?|examples?|benchmarks?|bench|scripts?|tools|third_party|vendor|external|extern|\.github|build|ci)/")
BUILD = re.compile(r"(^|/)(CMakeLists\.txt|[^/]+\.cmake|GNUmakefile|[Mm]akefile|[^/]+\.mk|CMakePresets\.json|meson\.build|setup\.py|pyproject\.toml)$|(^|/)cmake/")
CI = re.compile(r"^\.github/|^\.gitlab-ci\.yml$|^ci/|^\.devops/|^\.circleci/")
QUALIFIERS = r"(?:(?:static|inline|__inline__|__forceinline__|__device__|__global__|__host__|constexpr|extern\s+\"C\"|template\s*<[^;{}]*>|" \
             r"__launch_bounds__\s*\([^)]*\)|__noinline__|virtual|explicit|friend)\s+)*"
FUNC = re.compile(QUALIFIERS + r"(?P<ret>[A-Za-z_][\w:<>,\s\*&]*?[\s\*&]+)(?:__launch_bounds__\s*\([^)]*\)\s*)?(?P<name>(?:[A-Za-z_]\w*::)*~?[A-Za-z_]\w*)\s*\((?P<args>[^;{}]*?)\)"
                  r"\s*(?:const\s*)?(?:noexcept\s*)?(?:->\s*[^{;]+)?\s*\{", re.DOTALL)
KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "catch", "else", "do", "case", "decltype", "static_assert", "alignas", "defined", "constexpr",
            "consteval", "noexcept", "typeid", "alignof", "requires", "operator"}
GLOBAL = re.compile(r"__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?(?:static\s+)?void\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(")
GTEST = re.compile(r"^\s*(?:TEST|TEST_F|TEST_P|TYPED_TEST)\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", re.MULTILINE)
CATCH = re.compile(r"^\s*(?:TEST_CASE|SCENARIO|C2H_TEST)\s*\(\s*\"([^\"]+)\"", re.MULTILINE)
GGML_CASE = re.compile(r"^\s*test_cases\.emplace_back\(new\s+(test_\w+)\s*\((.*)\)\)\s*;\s*$", re.MULTILINE)


def code_mask(text: str) -> str:
    """``text`` with comments, string and char literals blanked (same length, newlines kept)."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        end = None
        if c == "/" and nxt == "/":
            end = text.find("\n", i)
            end = n if end < 0 else end
        elif c == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
        elif c == '"' or (c == "'" and not (i > 0 and (text[i - 1].isalnum()))):  # a char literal, not a digit separator 1'000
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            end = min(j + 1, n)
        if end is not None and end > i:
            for k in range(i + (1 if c in "\"'" else 0), end - (1 if c in "\"'" else 0)):
                if out[k] != "\n":
                    out[k] = " "
            i = end
            continue
        i += 1
    return "".join(out)


def match_brace(masked: str, i: int) -> int:
    """Index after the brace that closes ``masked[i]`` (a ``{``), or -1."""
    depth = 0
    for j in range(i, len(masked)):
        ch = masked[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def body_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()[:16]


def find_funcs(text: str) -> list[tuple[str, int, int]]:
    """``(name, start, end)`` of every function definition (qualified names keep their ``Class::``), outermost only."""
    masked = code_mask(text)
    out: list[tuple[str, int, int]] = []
    pos = 0
    while True:
        m = FUNC.search(masked, pos)
        if not m:
            break
        name = m.group("name")
        leaf = name.rsplit("::", 1)[-1]
        if leaf in KEYWORDS or leaf.isupper() or leaf.startswith("__") or m.group("ret").split()[-1] in ("return", "else", "new", "delete", "throw", "case", "if", "while", "for", "switch", "do", "goto"):
            pos = m.start("name") + len(name)
            continue
        end = match_brace(masked, m.end() - 1)
        if end < 0:
            break
        out.append((name, m.start(), end))
        pos = end
    return out


class CudaProfile(_Base):  # type: ignore[valid-type,misc]
    name = "cuda"
    lang = "clike"
    extensions = EXTENSIONS
    plumbing_words = "CMakeLists.txt, *.cmake files, Makefiles and anything under cmake/"
    install_words = "download toolkits or packages"

    # -- paths ----------------------------------------------------------------------------------------------------------------
    def detect(self, tree: Path) -> bool:
        tree = Path(tree)
        if any(tree.rglob("*.cu")):
            return True
        cm = tree / "CMakeLists.txt"
        return cm.is_file() and bool(re.search(r"\bCUDA\b", cm.read_text(errors="replace")))

    def is_test_file(self, path: str) -> bool:
        return path.endswith(EXTENSIONS) and bool(TEST_FILE.search(path))

    def is_test_path(self, path: str) -> bool:
        return self.is_test_file(path) or bool(TEST_PATH.search(path))

    def is_plumbing(self, path: str) -> bool:
        return bool(BUILD.search(path))

    def is_protected(self, path: str) -> bool:
        return self.is_test_path(path) or self.is_plumbing(path) or bool(CI.search(path))

    def is_source(self, path: str) -> bool:
        return path.endswith(EXTENSIONS) and not self.is_test_path(path) and not NOT_SOURCE.search(path) and not self.is_protected(path)

    def is_kernel_source(self, path: str) -> bool:
        """Device code: ``.cu``/``.cuh``, or a header next to them in a CUDA backend directory."""
        return self.is_source(path) and (path.endswith((".cu", ".cuh")) or bool(re.search(r"(^|/)[^/]*cuda[^/]*/", path)))

    # -- definitions ----------------------------------------------------------------------------------------------------------
    def _find_funcs(self, text: str, path: str = "") -> list[tuple[str, int, int]]:
        return find_funcs(text)

    def _find_tests(self, text: str, path: str = "") -> list[tuple[str, int, int]]:
        out: list[tuple[str, int, int]] = []
        masked = code_mask(text)
        for rx, key in ((GTEST, lambda m: f"{m.group(1)}.{m.group(2)}"), (CATCH, lambda m: m.group(1))):
            for m in rx.finditer(text):
                b = masked.find("{", m.end())
                e = match_brace(masked, b) if b >= 0 else -1
                out.append((key(m), m.start(), e if e > 0 else m.end()))
        for m in GGML_CASE.finditer(text):
            out.append((f"{m.group(1)}#{body_hash(m.group(2))[:10]}", m.start(), m.end()))
        return sorted(out, key=lambda t: t[1])

    def kernels(self, text: str) -> list[str]:
        """Names of the ``__global__`` functions defined in ``text``."""
        return sorted(set(GLOBAL.findall(code_mask(text))))

    def test_defs(self, text: str | None, path: str = "") -> dict[str, str]:
        if not text:
            return {}
        out: dict[str, str] = {}
        for k, a, b in self._find_tests(text, path):
            out.setdefault(k, body_hash(text[a:b]))
        return out

    def strip_tests(self, text: str, keys: set[str], path: str = "") -> str | None:
        found = self._find_tests(text, path)
        spans = [(a, b) for k, a, b in found if k in keys]
        if not spans:
            return text
        if all(k in keys for k, _, _ in found):
            return None
        out = text
        for a, b in sorted(spans, key=lambda s: -s[0]):
            a2 = out.rfind("\n", 0, a) + 1
            e = out.find("\n", max(a, b - 1))
            out = out[:a2] + out[(len(out) if e < 0 else e + 1):]
        return out

    def changed_defs(self, before: str | None, after: str | None, path: str = "") -> list[str]:
        def defs(t: str | None) -> dict[str, str]:
            return {} if not t else {n: body_hash(t[a:b]) for n, a, b in find_funcs(t)}

        a, b = defs(before), defs(after)
        return [q for q in b if a.get(q) != b[q]] + [q for q in a if q not in b]

    def key_of_test(self, test_id: str) -> tuple[str, str]:
        f, _, k = test_id.partition("::")
        return f, k

    def suite_targets(self, tree: Path) -> list[str]:
        tree = Path(tree)
        return sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file() and self.is_test_file(p.relative_to(tree).as_posix()))

    def leaf(self, key: str) -> str:
        return key.rsplit("::", 1)[-1]


PROFILE = CudaProfile()


def changed_names(before: str | None, after: str | None) -> list[str]:
    """Leaf names of the functions whose bodies differ between ``before`` and ``after`` (added, changed or removed)."""
    return sorted({n.rsplit("::", 1)[-1] for n in PROFILE.changed_defs(before, after)})


def kernels_of(texts: Sequence[str]) -> list[str]:
    return sorted({k for t in texts for k in PROFILE.kernels(t)})
