"""The Rust profile (swarm-pipeline/07 §3.2): ``cargo test`` per test target, libtest result lines.

- **Paths.** Source: ``*.rs`` outside tests, benches and examples. Tests: integration tests (``tests/**.rs``) and test
  modules kept in their own files (``tests.rs``, ``*_test(s).rs``); a test written inline in a source file is not a
  test file (its fix cannot be hidden without hiding source, so such fixes never qualify). Test paths add snapshots
  (``*.snap``, ``snapshots/``) and test data. Protected: tests, ``Cargo.toml`` / ``Cargo.lock``, ``build.rs``,
  ``rust-toolchain*``, ``.cargo/``. Plumbing (hack): ``build.rs`` and ``.cargo/`` (code run at build time and the
  cargo configuration both steer what a test run compiles).
- **Tests.** ``#[test]`` (and ``#[<crate>::test]``) functions; the key is the module path libtest prints
  (``mod::name``): ``mod`` blocks of the file plus the file's place in its test binary (``tests/a/b.rs`` -> ``b::``) or
  in the crate (``src/x/tests.rs`` -> ``x::tests::``). An id is ``<file>::<key>``.
- **Runs.** One ``cargo test --offline -p <package> --test <binary>`` (or ``--lib``) per target, each delimited in the
  report, so a target that does not compile fails only its own tests (``error``).
- **World.** ``rust`` pinned by digest, git and python3 added; ``cargo fetch`` and ``cargo test --no-run`` (all
  features when they compile together, else the defaults) at build time into ``CARGO_TARGET_DIR=/opt/target``, then
  every artifact of the workspace's own crates is deleted (only dependencies stay compiled). A lockfile the repository
  does not commit is generated at build and copied into each run. ``CARGO_NET_OFFLINE=true``.
- **Leaks.** ``cargo search`` fails; the registry holds no copy of a workspace crate; no ``.rs`` file in the image
  defines two of the new test functions.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.langs.base import code_mask, line_start, match_bracket
from crucible.factory.langs.clike import CLikeProfile

BASE_IMAGE = "rust@sha256:a24e292b4af7bdee351a31d3a6379972a5652f0ea22a1b6d6883324ac4f5097c"  # rust:1-slim-bookworm, cargo 1.99.1
_TEST_ATTR = re.compile(r"#\[\s*(?:[A-Za-z_]\w*::)*test\b[^\]]*\]")
_FN = re.compile(r"(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+([A-Za-z_]\w*)")
_MOD = re.compile(r"(?<![\w])mod\s+([A-Za-z_]\w*)\s*\{")
_RESULT = re.compile(r"^test (\S+) \.\.\. (ok|FAILED|ignored|bench)\b", re.MULTILINE)
CLEAN = r'''import json, os, re, shutil, subprocess, sys
meta = json.loads(subprocess.run(["cargo", "metadata", "--no-deps", "--format-version", "1", "--offline"], capture_output=True, text=True, check=True).stdout)
if sys.argv[1:] == ["--names"]:
    print(" ".join(p["name"] for p in meta["packages"]))
    sys.exit(0)
names, pkgs = set(), set()
for p in meta["packages"]:
    pkgs.add(p["name"])
    names.add(p["name"].replace("-", "_"))
    for t in p.get("targets") or []:
        names.add(t["name"].replace("-", "_")); names.add(t["name"])
root = os.environ.get("CARGO_TARGET_DIR", "target")
gone = 0
for prof in ("debug", "release"):
    d = os.path.join(root, prof)
    if not os.path.isdir(d):
        continue
    for sub, pat in (("deps", names), ("incremental", names), (".fingerprint", pkgs), ("build", pkgs), ("", names)):
        dd = os.path.join(d, sub)
        if not os.path.isdir(dd):
            continue
        rx = re.compile(r"^(lib)?(" + "|".join(re.escape(n) for n in pat) + r")(-[0-9a-f]{16})?(\.|$|-)")
        for f in os.listdir(dd):
            if rx.match(f) and (sub or not os.path.isdir(os.path.join(dd, f)) or f in pat):
                pth = os.path.join(dd, f)
                shutil.rmtree(pth, ignore_errors=True) if os.path.isdir(pth) else os.remove(pth)
                gone += 1
print("removed", gone, "workspace artifacts")
'''


def _toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(errors="replace"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def crate_of(tree: Path, path: str) -> tuple[str, str] | None:
    """(crate dir relative to the tree, package name) of the nearest ``Cargo.toml`` with a ``[package]`` above ``path``."""
    parts = path.split("/")[:-1]
    for i in range(len(parts), -1, -1):
        d = "/".join(parts[:i])
        m = _toml(Path(tree) / d / "Cargo.toml") if (Path(tree) / d / "Cargo.toml").is_file() else {}
        name = (m.get("package") or {}).get("name")
        if name:
            return d, str(name)
    return None


def packages(tree: Path) -> list[str]:
    out = []
    for f in sorted(Path(tree).rglob("Cargo.toml")):
        rel = f.relative_to(tree).as_posix()
        if any(p in ("target", ".git", "vendor") for p in rel.split("/")):
            continue
        n = (_toml(f).get("package") or {}).get("name")
        if n:
            out.append(str(n))
    return out


def module_prefix(path: str, crate_dir: str = "") -> str:
    """The module path a file's tests sit under in their test binary or crate (see the module doc)."""
    rel = path[len(crate_dir) + 1:] if crate_dir and path.startswith(crate_dir + "/") else path
    parts = rel.split("/")
    if parts[0] == "tests":
        inner = parts[2:] if len(parts) > 2 else []  # tests/<bin>/<mod>.rs
        mods = [p.removesuffix(".rs") for p in inner]
        mods = [m for m in mods if m not in ("main", "mod")]
        return "".join(f"{m}::" for m in mods)
    if parts[0] == "src":
        mods = [p.removesuffix(".rs") for p in parts[1:]]
        mods = [m for m in mods if m not in ("lib", "main", "mod")]
        return "".join(f"{m}::" for m in mods)
    return ""


def target_of(tree: Path, path: str) -> tuple[str, str, str] | None:
    """(package, ``--test <bin>`` | ``--lib``, binary root) of a test file."""
    c = crate_of(tree, path)
    if c is None:
        return None
    d, pkg = c
    rel = path[len(d) + 1:] if d and path.startswith(d + "/") else path
    parts = rel.split("/")
    if parts[0] == "tests" and len(parts) >= 2:
        name = parts[1].removesuffix(".rs")
        root = f"{d}/tests/{parts[1]}" if d else f"tests/{parts[1]}"
        if len(parts) > 2:  # a module of the test binary tests/<dir>/main.rs; a helper dir (tests/helpers/mod.rs) is no binary
            root += "/main.rs"
            if not (Path(tree) / root).is_file():
                return None
        for t in _toml(Path(tree) / d / "Cargo.toml").get("test") or []:
            if str(t.get("path") or "").removeprefix("./") in (rel, "/".join(parts[:2]), root.removeprefix(f"{d}/") if d else root):
                name = str(t.get("name") or name)
        return pkg, f"--test {name}", root
    return pkg, "--lib", f"{d}/src/lib.rs" if d else "src/lib.rs"


class RustProfile(CLikeProfile):
    name = "rust"
    lang = "rust"
    extensions = (".rs",)
    test_file_rx = re.compile(r"(^|/)tests/.*\.rs$|(^|/)tests\.rs$|_tests?\.rs$")
    test_path_rx = re.compile(r"(^|/)tests/|(^|/)snapshots/|\.snap$|(^|/)(testdata|test-data|test_data|fixtures)/")
    protected_names = frozenset({"Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain", "rust-toolchain.toml", "deny.toml"})
    protected_rx = re.compile(r"(^|/)\.cargo/")
    plumbing_rx = re.compile(r"(^|/)\.cargo/|(^|/)build\.rs$")
    not_source_rx = re.compile(r"(^|/)(docs?|examples?|benches|benchmarks?|scripts?|tools|xtask|fuzz|\.github|vendor|target)/")
    recipe = "rust-2"
    base_image = BASE_IMAGE
    timeout_s = 1500.0
    tmpfs = "1g"
    leak_globs = ("*.rs",)
    plumbing_words = "build scripts (build.rs) or cargo configuration under .cargo"

    def detect(self, tree: Path) -> bool:
        return (Path(tree) / "Cargo.toml").is_file()

    def suite_targets(self, tree: Path) -> list[str]:
        """Test binaries (``tests/x.rs``, ``tests/x/main.rs``) and test-module files of the crates' libraries; a module of a
        test binary or a shared helper is not a target of its own."""
        out = []
        for p in super().suite_targets(tree):
            tg = target_of(tree, p)
            if tg is not None and (tg[1] == "--lib" or tg[2] == p):
                out.append(p)
        return out

    # -- definitions ---------------------------------------------------------------------------------------------------------
    def _spans(self, text: str, rx: re.Pattern[str], attr: bool) -> list[tuple[str, int, int]]:
        masked = code_mask(text, "rust")
        mods = []
        for m in _MOD.finditer(masked):
            e = match_bracket(text, m.end() - 1, "rust")
            if e > 0:
                mods.append((m.group(1), m.start(), e))
        out = []
        for m in rx.finditer(masked):
            if attr:
                f = _FN.search(masked, m.end())
                if f is None or re.search(r"[;{}]", masked[m.end(): f.start()]):
                    continue
            else:
                f = m
            ob = masked.find("{", f.end())
            semi = masked.find(";", f.end())
            if ob < 0 or (0 <= semi < ob):
                continue
            end = match_bracket(text, ob, "rust")
            if end < 0:
                continue
            start = line_start(text, m.start())
            while start > 0:  # attributes and doc comments just above
                prev = line_start(text, start - 1)
                s = text[prev:start].strip()
                if not s.startswith(("#[", "//")):
                    break
                start = prev
            path = "".join(f"{n}::" for n, a, b in sorted(mods, key=lambda x: x[1]) if a < f.start() < b)
            out.append((path + f.group(1), start, end))
        return out

    def _find_tests(self, text: str, path: str) -> list[tuple[str, int, int]]:
        pre = ""
        if path.startswith(("tests/", "src/")):
            pre = module_prefix(path)
        elif path:  # a workspace crate: the prefix from its crate dir
            hits = [i for i in (path.find("/tests/"), path.find("/src/")) if i >= 0]
            pre = module_prefix(path[min(hits) + 1:]) if hits else ""
        return [(pre + k, a, b) for k, a, b in self._spans(text, _TEST_ATTR, True)]

    def _find_funcs(self, text: str, path: str) -> list[tuple[str, int, int]]:
        return self._spans(text, _FN, False)

    def leaf(self, key: str) -> str:
        return key.rsplit("::", 1)[-1]

    # -- World ---------------------------------------------------------------------------------------------------------------
    def packaging_files(self, tree: Path) -> list[str]:
        out = []
        for f in sorted(Path(tree).rglob("Cargo.toml")):
            rel = f.relative_to(tree).as_posix()
            if not any(p in ("target", ".git") for p in rel.split("/")):
                out.append(rel)
        return out + [f for f in ("Cargo.lock", "rust-toolchain", "rust-toolchain.toml") if (Path(tree) / f).is_file()]

    def image_info(self, tree: Path) -> dict[str, Any]:
        ps = packages(tree)
        root = (_toml(Path(tree) / "Cargo.toml").get("package") or {}).get("name")
        return {"project": root or (ps[0] if ps else None), "projects": ps, "locked": (Path(tree) / "Cargo.lock").is_file()}

    def build_files(self, tree: Path, info: Mapping[str, Any]) -> dict[str, str]:
        return {"crl_clean.py": CLEAN}

    def dockerfile(self, tree: Path, info: Mapping[str, Any]) -> str:
        return "\n".join([
            f"FROM {BASE_IMAGE}",
            "RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git python3 pkg-config ca-certificates >/dev/null && rm -rf /var/lib/apt/lists/*",
            "RUN git config --system safe.directory '*' && git config --system user.email agent@sandbox && git config --system user.name agent",
            ("ENV CARGO_TARGET_DIR=/opt/target CARGO_TERM_COLOR=never CARGO_INCREMENTAL=0 RUST_BACKTRACE=0 CARGO_NET_RETRY=3 "
             "CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0"),  # no debug info: the warm target dir is several times smaller
            "COPY crl_clean.py /opt/crl_clean.py",
            "COPY a /w",
            "WORKDIR /w",
            "RUN cargo fetch",
            # all features when they compile together, else the defaults, else each package that compiles on its own (a
            # workspace member binding Python or wasm cannot block the rest); a run uses what compiled
            ("RUN if cargo test --workspace --all-features --no-run -j 4; then echo --all-features > /opt/crl_features; "
             "elif cargo test --workspace --no-run -j 4; then : > /opt/crl_features; "
             "else : > /opt/crl_features; for p in $(python3 /opt/crl_clean.py --names); do cargo test -p $p --no-run -j 4 || true; done; fi"),
            "RUN cp Cargo.lock /opt/Cargo.lock && python3 /opt/crl_clean.py",
            "ENV CARGO_NET_OFFLINE=true",
            "RUN rm -rf /w && mkdir -p /w && chmod -R a+rwX /opt/target /usr/local/cargo",
            "WORKDIR /w",
        ]) + "\n"

    # -- runs ----------------------------------------------------------------------------------------------------------------
    def run_script(self, tree: Path, targets: Sequence[str]) -> str:
        groups: dict[tuple[str, str], list[str]] = {}
        for t in targets:
            if not t.endswith(".rs"):
                continue
            tg = target_of(tree, t)
            if tg:  # an integration binary is named by its root file (its ids use it)
                files = groups.setdefault((tg[0], tg[1]), [])
                for f in ([tg[2]] if tg[1] != "--lib" else [t]):
                    if f not in files:
                        files.append(f)
        lines = ["[ -f Cargo.lock ] || cp /opt/Cargo.lock Cargo.lock", 'F=$(cat /opt/crl_features 2>/dev/null)']
        for (pkg, sel), files in sorted(groups.items()):
            lines.append(f'echo "@@T {" ".join(files)}" >> "$O/report"')
            lines.append(f'cargo test --offline $F -p {pkg} {sel} -- --test-threads=2 >> "$O/report" 2>&1; echo "@@RC $?" >> "$O/report"')
        lines.append('echo "cargo done"')
        return "\n".join(lines)

    def parse_report(self, report: str, tree: Path, targets: Sequence[str], tail: str = "") -> tuple[dict[str, str], list[str]]:
        out: dict[str, str] = {}
        notes = []
        st = {"ok": "passed", "FAILED": "failed", "ignored": "skipped", "bench": "skipped"}
        for sec in report.split("@@T ")[1:]:
            head, _, body = sec.partition("\n")
            files = head.split()
            if not files:
                continue
            rc = re.search(r"@@RC (\d+)", body)
            results = _RESULT.findall(body)
            tg = target_of(tree, files[0])
            lib = bool(tg and tg[1] == "--lib")
            c = crate_of(tree, files[0])
            prefixes = sorted(((module_prefix(f, c[0] if c else ""), f) for f in files), key=lambda x: -len(x[0]))
            for name, s in results:
                f = next((f for p, f in prefixes if p and name.startswith(p)), None) if lib else files[0]
                if f is not None:
                    out[f"{f}::{name}"] = st[s]
            if not results and rc and rc.group(1) != "0":
                notes.append(f"build failed: {' '.join(files)}")
                for f in files:
                    for t in self.tests_in_target(tree, f):
                        out.setdefault(t, "error")
        return out, notes

    # -- leaks ---------------------------------------------------------------------------------------------------------------
    def fetch_probe(self, project: str | None) -> str:
        return f"cd /tmp && CARGO_NET_OFFLINE=false CARGO_HTTP_TIMEOUT=5 CARGO_NET_RETRY=0 cargo search {project or 'none-such-crate-x'} --limit 1"

    def own_copy_probe(self, tree: Path, project: str | None) -> str:
        names = packages(tree) or ([project] if project else [])
        return "; ".join(f"ls -d /usr/local/cargo/registry/src/*/{n}-[0-9]* 2>/dev/null" for n in names) or "true"

    def leak_needle(self, name: str) -> str:
        return f"fn {name}("


PROFILE = RustProfile()
