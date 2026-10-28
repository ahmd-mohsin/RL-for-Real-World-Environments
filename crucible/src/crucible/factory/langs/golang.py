"""The Go profile (swarm-pipeline/07 §3.1): ``go test -json`` over the packages of the target test files.

- **Paths.** Source: ``*.go`` outside tests, ``testdata/``, ``vendor/`` and tooling dirs. Tests: ``*_test.go``; test
  paths add ``testdata/``. Protected: tests, ``go.mod`` / ``go.sum`` / ``go.work``, ``vendor/``, CI and build config.
  Plumbing (hack): ``go.work`` and ``vendor/`` (either one redirects the module graph a test run builds against).
- **Tests.** Top-level ``func TestX`` / ``ExampleX`` / ``FuzzX`` (gofmt puts the closing brace of a top-level function
  in column 0; the matcher does not rely on it). An id is ``<dir>/<file>_test.go::TestX/sub``: ``go test -json`` reports
  the package and the test name, the file is the one defining ``TestX`` in that package. A package whose build fails
  reports every test of its files as ``error`` (and the run says which). ``key_of_test`` drops subtests.
- **Hiding.** Removing a test can leave an import only it used, which Go refuses to compile: an import the removed
  text used and the rest of the file no longer names is dropped too (an import whose package name cannot be read from
  its path is kept).
- **World.** ``golang`` pinned by digest; ``go mod download`` and a build of every non-main-module package the tests
  import (standard library included) warm ``GOMODCACHE`` / ``GOCACHE`` at build time; the main module is never
  compiled into the image. ``GOTOOLCHAIN=local``, ``GOPROXY=off``, ``GOFLAGS=-buildvcs=false``, ``CGO_ENABLED=0``.
- **Leaks.** ``go mod download <module>@latest`` fails; the module cache holds no copy of the module itself; no ``.go``
  file in the image defines two of the new test functions.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.langs.base import code_mask, line_start, match_bracket, remove_spans
from crucible.factory.langs.clike import CLikeProfile

BASE_IMAGE = "golang@sha256:5c3e9b43cfd3bb95c7937462e29d87ada57b481a96fa23ad0eb61778d7a00110"  # golang:1-bookworm, go1.27.3
_TEST = re.compile(r"(?m)^func[ \t]+((?:Test|Example|Fuzz)[A-Za-z0-9_]*)[ \t]*\(")
_FUNC = re.compile(r"(?m)^func[ \t]+(?:\([ \t]*\w*[ \t]*\*?[ \t]*(\w+)(?:\[[^\]]*\])?[ \t]*\)[ \t]*)?(\w+)[ \t]*(?:\[[^\]]*\])?\(")
_IMPORT_SPEC = re.compile(r'^[ \t]*(?:([A-Za-z_]\w*|\.|_)[ \t]+)?"([^"]+)"[ \t]*(?://.*)?$')


def module_path(tree: Path) -> str | None:
    f = Path(tree) / "go.mod"
    if not f.is_file():
        return None
    m = re.search(r"(?m)^module\s+(\S+)", f.read_text(errors="replace"))
    return m.group(1).strip('"') if m else None


def escape_module(path: str) -> str:
    """The module cache's case encoding (``!`` + lower case for each upper-case letter)."""
    return re.sub(r"[A-Z]", lambda m: "!" + m.group(0).lower(), path)


def import_name(path: str) -> str | None:
    """The package name an import path most likely binds, or None when it cannot be read from the path."""
    parts = path.split("/")
    last = parts[-1]
    if re.fullmatch(r"v\d+", last) and len(parts) > 1:
        last = parts[-2]
    last = re.sub(r"\.v\d+$", "", last)  # gopkg.in/yaml.v3
    return last if re.fullmatch(r"[A-Za-z_]\w*", last) else None


def _import_lines(text: str) -> list[tuple[int, int, str | None, str]]:
    """(start, end, alias, path) of every import spec line (block or single form)."""
    out = []
    for m in re.finditer(r"(?m)^import[ \t]*\(", text):
        end = match_bracket(text, m.end() - 1, "go")
        if end < 0:
            continue
        pos = m.end()
        for ln in text[m.end(): end - 1].splitlines(keepends=True):
            s = _IMPORT_SPEC.match(ln.rstrip("\n"))
            if s:
                out.append((pos, pos + len(ln), s.group(1), s.group(2)))
            pos += len(ln)
    for m in re.finditer(r'(?m)^import[ \t]+(?:([A-Za-z_]\w*|\.|_)[ \t]+)?"([^"]+)"[^\n]*\n?', text):
        out.append((m.start(), m.end(), m.group(1), m.group(2)))
    return out


def prune_imports(before: str, after: str) -> str:
    """``after`` without imports that ``before`` used and ``after`` no longer names (see the module doc)."""
    def uses(text: str, name: str, skip: list[tuple[int, int]]) -> bool:
        masked = code_mask(text, "go")
        for a, b in sorted(skip, reverse=True):
            masked = masked[:a] + " " * (b - a) + masked[b:]
        return re.search(rf"(?<![\w.]){re.escape(name)}\s*\.", masked) is not None

    specs_after = _import_lines(after)
    specs_before = {p: (a, b, al) for a, b, al, p in _import_lines(before)}
    drop = []
    for a, b, alias, path in specs_after:
        if alias in (".", "_"):
            continue
        name = alias or import_name(path)
        if not name or path not in specs_before:
            continue
        if uses(before, name, [(x, y) for x, y, _, _ in _import_lines(before)]) and not uses(after, name, [(x, y) for x, y, _, _ in specs_after]):
            drop.append((a, b))
    out = after
    for a, b in sorted(drop, reverse=True):
        out = out[:a] + out[b:]
    return re.sub(r"(?m)^import[ \t]*\([ \t]*\n[ \t]*\)[ \t]*\n", "", out)  # an import block left empty


def _code_lines(text: str) -> list[str]:
    """``text``'s lines with string, rune and raw-string literals and comments blanked, by character (``code_mask`` shifts
    on multi-byte runes: mvdan-sh's ``○`` left a comment's ``)`` unmasked and cut a table short)."""
    out, cur, i, n = [], [], 0, len(text)
    state = None  # None | '"' | "'" | '`' | '/*' | '//'
    while i < n:
        c = text[i]
        if c == "\n":
            out.append("".join(cur))
            cur = []
            if state in ('"', "'", "//"):
                state = None
            i += 1
            continue
        if state is None:
            if text.startswith("//", i):
                state, i = "//", i + 2
                continue
            if text.startswith("/*", i):
                state, i = "/*", i + 2
                continue
            if c in "\"'`":
                state = c
                cur.append(" ")
            else:
                cur.append(c)
            i += 1
        elif state in ('"', "'"):
            if c == "\\":
                i += 2
                continue
            if c == state:
                state = None
            i += 1
        elif state == "`":
            if c == "`":
                state = None
            i += 1
        elif state == "/*":
            if text.startswith("*/", i):
                state, i = None, i + 2
            else:
                i += 1
        else:  # //
            i += 1
    out.append("".join(cur))
    return out


def _counts(line: str) -> tuple[int, int, int]:
    return line.count("{") - line.count("}"), line.count("(") - line.count(")"), line.count("[") - line.count("]")


def table_spans(text: str) -> list[tuple[int, int]]:
    """Line spans [start, end) of package-level ``var`` declarations that hold a composite literal: the test tables
    (``var printTests = []struct{...}{...}``, ``var (...)``) that table-driven Go tests range over. Brackets inside string
    literals and comments do not count (``_code_lines``): shell and template inputs are full of them."""
    raw = text.splitlines(keepends=True)
    masked = _code_lines(text)
    out, k = [], 0
    while k < len(raw):
        if re.match(r"^var\b", raw[k]) and ("{" in masked[k] or masked[k].rstrip().endswith("(")):
            depth, j = 0, k
            while j < len(raw):
                c = _counts(masked[j])
                depth += c[0] + c[1]
                if depth <= 0:
                    break
                j += 1
            out.append((k, min(j, len(raw) - 1) + 1))
            k = j + 1
        else:
            k += 1
    return out


def strip_table_entries(text: str, runs: Sequence[Sequence[str]]) -> tuple[str, int]:
    """Strip mode's residue in Go: the table entries a fix added or changed stay readable in the agent's tree when only the
    test functions are stripped (mvdan-sh dc951aeb-e1, 2026-10-24: the fixes' printer cases sat in package-level tables of
    files whose functions were stripped). ``runs`` are the fix diffs' runs of consecutive added lines; each run found as a
    contiguous block inside a package-level table is removed with the whole entries it belongs to (a changed line inside a
    multi-line entry takes that entry), only when the removed block's brackets (outside string literals) balance at the
    table's element level: never half an entry. Returns the text and the number of lines removed."""
    raw = text.splitlines(keepends=True)
    masked = _code_lines(text)
    spans = table_spans(text)
    drop: set[int] = set()
    depth_at: dict[int, int] = {}  # bracket depth before each line, from its table's declaration
    for a, b in spans:
        d = 0
        for i in range(a, b):
            depth_at[i] = d
            d += sum(_counts(masked[i]))
    for run in runs:
        want = [ln.rstrip() for ln in run]
        if not want or not any(w.strip() for w in want):
            continue
        for a, b in spans:
            hit = next((i for i in range(a + 1, b - len(want)) if all(raw[i + j].rstrip() == w for j, w in enumerate(want))
                        and not drop & set(range(i, i + len(want)))), None)
            if hit is None:
                continue
            lo, hi = hit, hit + len(want)  # [lo, hi)
            if depth_at[lo] > 1:  # a changed line inside a multi-line entry: the whole entry goes (its want string changed)
                while lo > a + 1 and depth_at[lo] > 1:
                    lo -= 1
                while hi < b - 1 and (depth_at.get(hi, 1) > 1 or not masked[hi - 1].rstrip().endswith(",")):
                    hi += 1
            idx = range(lo, hi)
            # whole elements only: the block starts at the table's element level, closes every bracket it opens, and
            # ends an element (half an entry would leave ``{"in", }``: too few values)
            whole = (depth_at[lo] == 1 and depth_at.get(hi, 1) == 1 and [sum(_counts(masked[i])[x] for i in idx) for x in range(3)] == [0, 0, 0]
                     and masked[hi - 1].rstrip().endswith(",") and not drop & set(idx))
            if whole:
                drop |= set(idx)
            break
    return "".join(ln for i, ln in enumerate(raw) if i not in drop), len(drop)


class GoProfile(CLikeProfile):
    name = "go"
    lang = "go"
    extensions = (".go",)
    test_file_rx = re.compile(r"_test\.go$")
    test_path_rx = re.compile(r"_test\.go$|(^|/)testdata/")
    protected_names = frozenset({"go.mod", "go.sum", "go.work", "go.work.sum"})
    protected_rx = re.compile(r"(^|/)vendor/|(^|/)\.golangci\.ya?ml$|(^|/)\.goreleaser\.ya?ml$")
    plumbing_rx = re.compile(r"(^|/)go\.work(\.sum)?$|(^|/)vendor/")
    recipe = "go-1"
    base_image = BASE_IMAGE
    timeout_s = 900.0
    tmpfs = "1g"
    leak_globs = ("*.go",)
    plumbing_words = "go.work files or a vendor directory"

    def detect(self, tree: Path) -> bool:
        return (Path(tree) / "go.mod").is_file()

    def is_source(self, path: str) -> bool:
        return super().is_source(path) and not path.endswith("_test.go")

    # -- definitions ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def _extent(text: str, masked: str, m: re.Match[str]) -> tuple[int, int] | None:
        """(start incl. the doc comment, end incl. the closing brace's line) of the function whose ``func`` ``m``
        matched."""
        pe = match_bracket(text, m.end() - 1, "go")
        if pe < 0:
            return None
        ob = masked.find("{", pe)
        if ob < 0:
            return None
        end = match_bracket(text, ob, "go")
        if end < 0:
            return None
        start = m.start()
        while True:  # the contiguous // comment lines just above belong to it
            prev = line_start(text, start - 1) if start > 0 else 0
            if start == 0 or not text[prev:start].lstrip().startswith("//"):
                break
            start = prev
        return start, end

    def _find_tests(self, text: str, path: str) -> list[tuple[str, int, int]]:
        masked = code_mask(text, "go")
        out = []
        for m in _TEST.finditer(masked):
            ext = self._extent(text, masked, m)
            if ext:
                out.append((m.group(1), *ext))
        return out

    def _find_funcs(self, text: str, path: str) -> list[tuple[str, int, int]]:
        masked = code_mask(text, "go")
        out = []
        for m in _FUNC.finditer(masked):
            ext = self._extent(text, masked, m)
            if ext:
                out.append((f"{m.group(1)}.{m.group(2)}" if m.group(1) else m.group(2), *ext))
        return out

    def strip_table_entries(self, text: str, runs: Sequence[Sequence[str]]) -> tuple[str, int]:
        return strip_table_entries(text, runs)

    def strip_tests(self, text: str, keys: set[str], path: str = "") -> str | None:
        found = self._find_tests(text, path)
        spans = [(a, b) for k, a, b in found if k in keys]
        if not spans:
            return text
        if all(k in keys for k, _, _ in found):
            return None
        return prune_imports(text, remove_spans(text, spans))

    def key_of_test(self, test_id: str) -> tuple[str, str]:
        f, _, k = test_id.partition("::")
        return f, k.split("/", 1)[0]

    def leaf(self, key: str) -> str:
        return key.split("/", 1)[0]

    # -- World ---------------------------------------------------------------------------------------------------------------
    def packaging_files(self, tree: Path) -> list[str]:
        return [f for f in ("go.mod", "go.sum", "go.work", "go.work.sum") if (Path(tree) / f).is_file()]

    def image_info(self, tree: Path) -> dict[str, Any]:
        mod = module_path(tree)
        return {"project": mod, "projects": [mod] if mod else []}

    def dockerfile(self, tree: Path, info: Mapping[str, Any]) -> str:
        deps = ("{{if not .Standard}}{{if .Module}}{{if not .Module.Main}}{{.ImportPath}}{{end}}{{end}}{{end}}")
        return "\n".join([
            f"FROM {BASE_IMAGE}",
            ("ENV GOTOOLCHAIN=local GOFLAGS=-buildvcs=false CGO_ENABLED=0 GOCACHE=/opt/gocache GOMODCACHE=/opt/gomod GOPATH=/opt/gopath "
             "GOSUMDB=off GONOSUMDB=* GOTELEMETRY=off"),
            "COPY a /w",
            "WORKDIR /w",
            "RUN GOPROXY=https://proxy.golang.org,direct go mod download && (GOPROXY=https://proxy.golang.org,direct go mod download all || true)",
            # the dependencies the tests build against, and the standard library; never the main module (module doc)
            (f"RUN pkgs=$(GOPROXY=off go list -deps -test -f '{deps}' ./... 2>/dev/null | sort -u); "
             "if [ -n \"$pkgs\" ]; then GOPROXY=off go build $pkgs || true; fi; GOPROXY=off go build std"),
            "ENV GOPROXY=off",
            "RUN rm -rf /w /opt/gopath/pkg && mkdir -p /w && chmod -R a+rwX /opt/gocache",
            "WORKDIR /w",
        ]) + "\n"

    # -- runs ----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def packages_of(targets: Sequence[str]) -> list[str]:
        dirs = sorted({("./" + t.rsplit("/", 1)[0]) if "/" in t else "." for t in targets if t.endswith(".go")})
        return dirs

    def run_script(self, tree: Path, targets: Sequence[str]) -> str:
        pk = " ".join(self.packages_of(targets)) or "./..."
        return (f'go test -json -count=1 -vet=off -p 2 -timeout 1200s {pk} > "$O/report" 2> "$O/err"; echo "go test exit $?"; '
                'tail -c 3000 "$O/err"')

    def parse_report(self, report: str, tree: Path, targets: Sequence[str], tail: str = "") -> tuple[dict[str, str], list[str]]:
        mod = module_path(tree) or ""
        files: dict[str, dict[str, str]] = {}

        def dir_of(pkg: str) -> str | None:
            pkg = pkg.split(" ", 1)[0]
            if pkg == mod:
                return ""
            return pkg[len(mod) + 1:] if mod and pkg.startswith(mod + "/") else None

        def index(d: str) -> dict[str, str]:
            if d not in files:
                files[d] = {}
                root = Path(tree) / d if d else Path(tree)
                for f in sorted(root.glob("*_test.go")):
                    rel = f"{d}/{f.name}" if d else f.name
                    for k in self.test_defs(f.read_text(errors="replace"), rel):
                        files[d].setdefault(k, rel)
            return files[d]

        out: dict[str, str] = {}
        build_failed: set[str] = set()
        killed: list[str] = []
        deaths: list[str] = []
        st = {"pass": "passed", "fail": "failed", "skip": "skipped"}
        for line in report.splitlines():
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            act = ev.get("Action")
            # a test binary killed from outside (the OOM killer): go test reports it on the package, not on a test
            if act == "output" and not ev.get("Test") and str(ev.get("Output") or "").strip() == "signal: killed":
                killed.append(str(ev.get("Package") or ""))
                continue
            if act == "output":  # a test binary that died: go test's own timeout (the run's), a panic (one test's), a fatal error
                text = str(ev.get("Output") or "")
                if "panic: test timed out after" in text:
                    deaths.append(f"timeout: {ev.get('Package') or ''}")
                elif text.startswith("panic: ") and ev.get("Test"):
                    deaths.append(f"panic: {str(ev['Test']).split('/', 1)[0]} ({ev.get('Package') or ''})")
                elif text.startswith("fatal error: "):
                    deaths.append(f"{text.strip()[:80]} ({ev.get('Package') or ''})")
            if act == "build-fail" and ev.get("ImportPath"):
                build_failed.add(ev["ImportPath"].split(" ", 1)[0])
                continue
            if act == "fail" and not ev.get("Test") and ev.get("FailedBuild"):
                build_failed.add(str(ev["FailedBuild"]).split(" ", 1)[0])
                build_failed.add(str(ev.get("Package") or ""))
                continue
            if act not in st or not ev.get("Test") or not ev.get("Package"):
                continue
            d = dir_of(ev["Package"])
            if d is None:
                continue
            top = ev["Test"].split("/", 1)[0]
            f = index(d).get(top)
            if f is None:
                continue
            out[f"{f}::{ev['Test']}"] = st[act]
        notes = [f"killed: signal: killed ({p})" for p in sorted(set(killed))] + sorted(set(deaths))
        for pkg in sorted(p for p in build_failed if p):
            d = dir_of(pkg)
            if d is None:
                continue
            notes.append(f"build failed: {pkg}")
            for k, f in index(d).items():
                out.setdefault(f"{f}::{k}", "error")
        return out, notes

    # -- leaks ---------------------------------------------------------------------------------------------------------------
    def fetch_probe(self, project: str | None) -> str:
        mod = project or "example.com/none"
        return f"cd /tmp && GOFLAGS=-mod=mod GOPROXY=https://proxy.golang.org GONOSUMDB='*' go mod download -json {mod}@latest"

    def own_copy_probe(self, tree: Path, project: str | None) -> str:
        if not project:
            return "true"
        e = escape_module(project)
        return f"ls -d /opt/gomod/{e}@* /opt/gomod/cache/download/{e}/@v/*.zip 2>/dev/null"

    def leak_needle(self, name: str) -> str:
        return f"func {name}("


PROFILE = GoProfile()
