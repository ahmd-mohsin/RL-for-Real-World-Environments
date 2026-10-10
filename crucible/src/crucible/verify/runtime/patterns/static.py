"""Static and source checks (DESIGN §8.1 "static and binary checks"). Standard library only.

Generalized from the task03/04 graders (``static_checks``: comment-stripped regex rules over chosen roots, forbidden
files, required constructs), hardened against the A13 evasions found in the 2026-10-09 seed audit:

- **what is scanned is what is built**: ``prune`` deletes every excluded path from the grader's copy *before* the scan,
  so a banned construct in a "skipped" directory cannot be compiled in afterwards;
- **aliases**: a ``#define`` whose body contains a banned construct makes its name banned too (transitively), and a
  token-pasting macro (``##``) that assembles a banned identifier is reported;
- **dynamic imports**: ``importlib.import_module`` / ``__import__`` / ``getattr(..., "...")`` with string literals
  (adjacent literals and ``+`` concatenations joined) are matched against banned module names;
- **renamed binaries**: files whose content starts with an ELF/Mach-O magic or a shebang are reported wherever the rule
  set forbids native artifacts, whatever their suffix;
- **runner tampering**: candidate source that references the test runner (``pytest``, ``_pytest``, doctest config
  macros, frame inspection) is flagged, because in-process forgery of results needs it (see ``hidden_tests``).
"""

from __future__ import annotations

import fnmatch
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

C_LIKE = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp", ".hxx", ".inl", ".ipp", ".cl", ".metal"}
PY = {".py", ".pyi", ".pyx", ".pxd"}
NATIVE_MAGIC = (b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"MZ")
RUNNER_TAMPER_PY = [r"\bimport\s+_?pytest\b", r"\bfrom\s+_?pytest\b", r"sys\.modules\s*\[\s*['\"]_?pytest", r"\bsys\._getframe\b",
                    r"\binspect\.(stack|currentframe|getouterframes)\b", r"\bjunitxml\b", r"PYTEST_CURRENT_TEST", r"\bos\._exit\s*\(",
                    r"crucible_verify"]
RUNNER_TAMPER_C = [r"\bDOCTEST_CONFIG_\w+", r"\bdoctest::(Context|detail|IReporter)\b", r"\bCATCH_CONFIG_\w+", r"\bgtest_\w*filter\b",
                   r"\b_exit\s*\(\s*0\s*\)", r"\bquick_exit\s*\("]


def strip_c(text: str) -> str:
    """Remove C/C++ comments, keeping string literals and line structure."""
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            seg = text[i:n if j < 0 else j + 2]
            out.append("\n" * seg.count("\n"))
            i = n if j < 0 else j + 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def strip_cmake(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def strip_py(text: str) -> str:
    """Remove Python comments (string-aware), keep strings: a banned name in a string is still evidence."""
    lines = []
    for line in text.splitlines():
        q, buf, k = None, [], 0
        while k < len(line):
            c = line[k]
            if q:
                buf.append(c)
                if c == "\\" and k + 1 < len(line):
                    buf.append(line[k + 1])
                    k += 2
                    continue
                if c == q:
                    q = None
            elif c in "\"'":
                q = c
                buf.append(c)
            elif c == "#":
                break
            else:
                buf.append(c)
            k += 1
        lines.append("".join(buf))
    return "\n".join(lines)


@dataclass
class Rule:
    id: str
    pattern: str  # regex over comment-stripped code
    suffixes: set = field(default_factory=set)  # empty = every text file
    roots: tuple = ()  # top-level directories the rule applies to (empty = all)
    flags: int = 0


@dataclass
class RuleSet:
    forbid: list[Rule] = field(default_factory=list)
    require: list[Rule] = field(default_factory=list)  # each must match somewhere
    forbidden_files: set = field(default_factory=set)  # suffixes or exact names
    forbid_native: bool = False  # any ELF/Mach-O/PE content, any suffix
    banned_modules: set = field(default_factory=set)  # Python modules (dynamic import check)
    runner_tamper: bool = True
    tamper_roots: tuple = ()  # top-level directories the runner-tamper rules apply to (empty = all)
    skip: tuple = ()  # top-level directories that are not scanned (they must then be frozen: unchanged from the base)
    frozen_base: str | None = None  # the pristine tree ``skip`` directories are compared with (A13: no code hidden in skipped dirs)
    exclude: tuple = (".git", "__pycache__")  # pruned before scanning (and therefore never built)
    text_suffixes: set = field(default_factory=lambda: C_LIKE | PY | {".txt", ".cmake", ".toml", ".cfg", ".in", ".sh", ".mk", ".json", ".yaml", ".yml"})


def prune(root: Path, exclude: tuple) -> list[str]:
    """Delete excluded paths (glob patterns on relative paths or names) from the grader's copy."""
    removed = []
    for p in sorted(Path(root).rglob("*"), key=lambda q: len(q.parts)):
        if not p.exists() and not p.is_symlink():
            continue
        rel = p.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(p.name, pat) or rel.startswith(pat.rstrip("/") + "/") for pat in exclude):
            removed.append(rel)
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
    return removed


def _join_literals(text: str) -> str:
    """'tor' 'ch' and 'tor' + 'ch' -> 'torch' (string assembly used to dodge a literal match)."""
    lit = r"(['\"])([^'\"\\\n]*)\1"
    prev = None
    while prev != text:
        prev = text
        text = re.sub(lit + r"\s*\+?\s*" + lit, lambda m: m.group(1) + m.group(2) + m.group(4) + m.group(1), text)
    return text


def scan(root: Path, rules: RuleSet) -> dict:
    root = Path(root)
    pruned = prune(root, rules.exclude) if rules.exclude else []
    violations: list[dict] = []
    required = {r.id: False for r in rules.require}
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()]
    for p in files:
        rel = p.relative_to(root).as_posix()
        if rel.split("/", 1)[0] in rules.skip:
            continue
        if p.suffix in rules.forbidden_files or p.name in rules.forbidden_files:
            violations.append({"rule": "forbidden_file", "file": rel})
            continue
        head = p.open("rb").read(4)
        if rules.forbid_native and any(head.startswith(m) for m in NATIVE_MAGIC):
            violations.append({"rule": "native_binary", "file": rel})
            continue
        if p.suffix not in rules.text_suffixes:
            continue
        top = rel.split("/", 1)[0]
        if top in rules.skip:
            continue
        raw = p.read_text(errors="replace")
        is_cmake = p.name == "CMakeLists.txt" or p.suffix == ".cmake"
        code = strip_c(raw) if p.suffix in C_LIKE else strip_py(raw) if p.suffix in PY else strip_cmake(raw) if is_cmake else raw
        active = [r for r in rules.forbid if (not r.suffixes or p.suffix in r.suffixes or p.name in r.suffixes) and (not r.roots or top in r.roots)]
        for r in active:
            for m in re.finditer(r.pattern, code, r.flags):
                violations.append({"rule": r.id, "file": rel, "match": m.group(0)[:120]})
        for r in rules.require:
            if (not r.suffixes or p.suffix in r.suffixes or p.name in r.suffixes) and (not r.roots or top in r.roots) and re.search(r.pattern, code, r.flags):
                required[r.id] = True
        if p.suffix in C_LIKE and "##" in code:  # token pasting: join the arguments of each call and rescan
            for d in re.finditer(r"(?m)^\s*#\s*define\s+(\w+)\(([^)]*)\)\s+(.*##.*)$", code):
                for call in re.finditer(rf"\b{re.escape(d.group(1))}\s*\(([^()]*)\)(\s*\()?", code):
                    joined = "".join(x.strip() for x in call.group(1).split(",")) + ("(" if call.group(2) else "")
                    for r in active:
                        if re.search(r.pattern, joined, r.flags):
                            violations.append({"rule": f"{r.id}:token_paste", "file": rel, "match": call.group(0).strip()[:120]})
        if p.suffix in PY and rules.banned_modules:
            joined = _join_literals(code)
            for m in re.finditer(r"(?:importlib\.import_module|__import__|import_module|find_spec|getattr)\s*\(\s*[^)]*?['\"]([\w.]+)['\"]", joined):
                if m.group(1).split(".")[0] in rules.banned_modules:
                    violations.append({"rule": "dynamic_import", "file": rel, "match": m.group(0)[:120]})
        if rules.runner_tamper and (not rules.tamper_roots or top in rules.tamper_roots):
            pats = RUNNER_TAMPER_PY if p.suffix in PY else RUNNER_TAMPER_C if p.suffix in C_LIKE else []
            for pat in pats:
                for m in re.finditer(pat, code):
                    violations.append({"rule": "runner_tamper", "file": rel, "match": m.group(0)[:80]})
    # aliases: expand object-like macros (transitively) and rescan; a construct that only appears after expansion was
    # hidden behind a macro
    macros: dict[str, str] = {}
    for p in files:
        if p.suffix in C_LIKE and p.exists():
            for d in re.finditer(r"(?m)^[ \t]*#[ \t]*define[ \t]+(\w+)[ \t]+(.*)$", strip_c(p.read_text(errors="replace"))):
                macros[d.group(1)] = d.group(2).strip()
    if macros:
        names = re.compile(r"\b(" + "|".join(sorted(map(re.escape, macros), key=len, reverse=True)) + r")\b")
        for p in files:
            if p.suffix not in C_LIKE or not p.exists():
                continue
            rel = p.relative_to(root).as_posix()
            top = rel.split("/", 1)[0]
            raw = "\n".join(ln for ln in strip_c(p.read_text(errors="replace")).splitlines() if not re.match(r"\s*#\s*define\b", ln))
            exp = raw
            for _ in range(4):
                nxt = names.sub(lambda m: macros[m.group(1)], exp)
                if nxt == exp:
                    break
                exp = nxt
            if exp == raw:
                continue
            for r in rules.forbid:
                if (r.suffixes and p.suffix not in r.suffixes) or (r.roots and top not in r.roots):
                    continue
                before = len(re.findall(r.pattern, raw, r.flags))
                hits = re.findall(r.pattern, exp, r.flags)
                if len(hits) > before:
                    violations.append({"rule": f"{r.id}:alias", "file": rel, "match": str(hits[-1])[:120]})
    for rid, ok in required.items():
        if not ok:
            violations.append({"rule": f"missing:{rid}", "file": "-"})
    if rules.skip and rules.frozen_base:  # skipped directories are not scanned: no buildable file there may be new or modified
        import hashlib

        def buildable(q: Path) -> bool:
            return q.suffix in C_LIKE | PY | {".cmake", ".cu", ".cuh"} or q.name == "CMakeLists.txt"

        def digest(top: Path) -> dict:
            if not top.exists():
                return {}
            return {q.relative_to(top).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest()
                    for q in top.rglob("*") if q.is_file() and not q.is_symlink() and buildable(q)}

        for d in rules.skip:
            if d == ".git":
                continue
            a, b = digest(root / d), digest(Path(rules.frozen_base) / d)
            changed = sorted(k for k in a if a.get(k) != b.get(k))  # new or modified; deleting code hides nothing
            violations += [{"rule": "frozen_dir_changed", "file": f"{d}/{k}"} for k in changed[:20]]
    return {"pass": not violations, "violations": violations[:200], "count": len(violations), "pruned": pruned[:50]}
