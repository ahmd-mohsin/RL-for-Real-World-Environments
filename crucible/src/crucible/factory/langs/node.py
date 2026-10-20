"""The TypeScript / JavaScript profile (swarm-pipeline/07 §3.3): single-package repositories tested with vitest or jest.

- **Paths.** Source: ``.js .mjs .cjs .ts .mts .cts .jsx .tsx`` outside tests, ``dist/``, ``node_modules/`` and tooling
  dirs. Tests: ``*.test.*`` / ``*.spec.*``, anything under ``__tests__/``, ``test/`` or ``tests/`` (and date-fns style
  ``<fn>/test.ts``); test paths add snapshots, fixtures and mocks. Protected: tests, ``package.json``, every lockfile,
  ``tsconfig*.json``, vitest / vite / jest / babel configuration, package-manager rc files. Plumbing (hack): ``.npmrc``,
  ``.yarnrc*``, ``.pnpmfile.cjs`` and anything under ``node_modules/`` (each one changes what a run resolves).
- **Tests.** ``describe`` / ``it`` / ``test`` calls with a literal title; the key is the describe titles and the title
  joined by ``::``, as the runner's JSON names them (``ancestorTitles`` + ``title``). Titles built at run time
  (``it.each``, template expressions) are not definitions: their tests still run and count, they are only never hidden
  or owned by text. A file that fails to load reports each test it defines as ``error``.
- **Runner.** ``vitest run --reporter=json`` or ``jest --ci --json --runTestsByPath``, read from the report file, never
  the console. Chosen from ``package.json`` (dependency or test script); a repository with neither is refused.
- **World.** ``node`` pinned by digest; the lockfile's install (``npm ci`` / ``pnpm install --frozen-lockfile`` /
  ``yarn install --frozen-lockfile``) at build time into ``/opt/deps/node_modules``; each run links it into the
  copied tree. Yarn Berry installs with the ``node-modules`` linker (never Plug'n'Play); in a workspace only the root package's tests run (members keep their own dependency trees).
- **Leaks.** ``npm view <name>`` fails; no package in the dependency tree is the project itself (a tool that depends
  on the project would ship a release of it: the World build deletes every such copy, the probe checks it); no source file in the image holds two of the new test titles.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.langs.base import code_mask, line_end, line_start, match_bracket
from crucible.factory.langs.clike import CLikeProfile

BASE_IMAGE = "node@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392"  # node:22-bookworm-slim, v22.23.3
EXTS = (".js", ".mjs", ".cjs", ".ts", ".mts", ".cts", ".jsx", ".tsx")
LOCKS = ("package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml", "yarn.lock", "bun.lockb", "bun.lock")
_CALL = re.compile(r"(?<![\w$.])(describe|suite|context|it|test)((?:\s*\.\s*(?:only|skip|concurrent|sequential|todo|fails|shuffle))*)\s*\(")
_FUNCS = (re.compile(r"(?<![\w$.])(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*[<(]"),
          re.compile(r"(?<![\w$.])(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=\n]+)?=\s*(?:async\s*)?(?:function\b|\([^)\n]*\)\s*(?::[^=\n]+)?=>|[A-Za-z_$][\w$]*\s*=>)"),
          re.compile(r"(?m)^[ \t]+(?:(?:public|private|protected|static|async|get|set|override|readonly)\s+)*(?!if\b|for\b|while\b|switch\b|catch\b|return\b|function\b)"
                     r"([A-Za-z_$][\w$]*)\s*(?:<[^>\n]*>)?\([^)\n]*\)\s*(?::\s*[^{\n]+)?\{"))


def _literal(text: str, i: int) -> str | None:
    """The string literal at ``i`` (after whitespace), unescaped; None when it is not a plain literal."""
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    if i >= len(text) or text[i] not in "'\"`":
        return None
    q = text[i]
    j = i + 1
    buf = []
    while j < len(text):
        c = text[j]
        if c == "\\" and j + 1 < len(text):
            buf.append(text[j + 1])
            j += 2
            continue
        if c == q:
            return "".join(buf)
        if q == "`" and c == "$" and text[j + 1: j + 2] == "{":
            return None
        if c == "\n" and q != "`":
            return None
        buf.append(c)
        j += 1
    return None


def package_json(tree: Path) -> dict[str, Any]:
    f = Path(tree) / "package.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return {}


def runner_of(tree: Path) -> str | None:
    pj = package_json(tree)
    deps = {**(pj.get("dependencies") or {}), **(pj.get("devDependencies") or {})}
    script = str((pj.get("scripts") or {}).get("test") or "")
    for r in ("vitest", "jest"):
        if r in deps or re.search(rf"\b{r}\b", script):
            return r
    return None


def runner_config(tree: Path) -> str | None:
    """The configuration file the repository's own test script passes its runner (``jest --config scripts/jest.config.js``),
    or None (the runner finds its default)."""
    runner = runner_of(tree)
    scripts = package_json(tree).get("scripts") or {}
    for key in sorted(scripts, key=lambda k: (k != "test", k)):
        cmd = str(scripts[key])
        if runner and re.search(rf"\b{runner}\b", cmd):
            first = re.split(r"&&|;|\|\|", cmd)[0] if key == "test" else cmd
            m = re.search(r"(?:--config|-c)[= ]+([^\s&;|]+)", first)
            if m and (Path(tree) / m.group(1)).is_file():
                return m.group(1).removeprefix("./")
            if key == "test" and re.search(rf"\b{runner}\b", first):
                return None  # the main test script runs the runner with its default configuration
    return None


def manager_of(tree: Path) -> str:
    t = Path(tree)
    if (t / "pnpm-lock.yaml").is_file():
        return "pnpm"
    if (t / "yarn.lock").is_file():
        return "yarn"
    if (t / "bun.lockb").is_file() or (t / "bun.lock").is_file():
        return "bun"
    return "npm"


class NodeProfile(CLikeProfile):
    name = "node"
    lang = "node"
    extensions = EXTS
    test_file_rx = re.compile(r"[._-](test|spec)\.[cm]?[jt]sx?$|(^|/)__tests__/|(^|/)tests?/|(^|/)test\.[cm]?[jt]sx?$")
    test_path_rx = re.compile(r"(^|/)(__snapshots__|__fixtures__|__mocks__|fixtures?|testdata|test-data|tests?|__tests__)/|\.snap$")
    protected_names = frozenset({"package.json", *LOCKS, ".npmrc", ".yarnrc", ".yarnrc.yml", ".pnpmfile.cjs", ".nvmrc", ".node-version", ".npmignore"})
    protected_rx = re.compile(r"(^|/)tsconfig[^/]*\.json$|(^|/)(vitest|vite|jest|babel|vitest\.workspace|rollup|tsup|esbuild|webpack)\.config\.[cm]?[jt]s(on)?$"
                              r"|(^|/)\.babelrc|(^|/)jest\.config\.json$|(^|/)\.swcrc$|(^|/)\.yarn/")
    plumbing_rx = re.compile(r"(^|/)(\.npmrc|\.yarnrc(\.yml)?|\.pnpmfile\.cjs)$|(^|/)node_modules(/|$)")
    recipe = "node-3"
    base_image = BASE_IMAGE
    timeout_s = 900.0
    tmpfs = "1g"
    leak_globs = ("*.js", "*.mjs", "*.cjs", "*.ts", "*.mts", "*.cts")
    plumbing_words = ".npmrc, .yarnrc or .pnpmfile.cjs files, or anything under node_modules"

    def detect(self, tree: Path) -> bool:
        return (Path(tree) / "package.json").is_file()

    def supported(self, tree: Path) -> str:
        """Why the repository cannot be a World of this profile ("" when it can)."""
        pj = package_json(tree)
        if not pj:
            return "no package.json"
        if runner_of(tree) is None:
            return "no vitest or jest"
        return ""

    def suite_targets(self, tree: Path) -> list[str]:
        """The root package's test files: a workspace's member packages (a directory with its own ``package.json``) have
        dependency trees of their own and are left out."""
        files = super().suite_targets(tree)
        nested = {p.rsplit("/", 1)[0] + "/" for p in self.walk(tree) if p.endswith("/package.json") and "node_modules" not in p}
        return [f for f in files if not any(f.startswith(n) for n in nested)]

    # -- definitions ---------------------------------------------------------------------------------------------------------
    def _calls(self, text: str) -> list[tuple[str, str, int, int]]:
        """(kind, title, start, end) of every describe / it / test call with a literal title."""
        masked = code_mask(text, "node")
        out = []
        for m in _CALL.finditer(masked):
            if "skipIf" in m.group(0) or "each" in m.group(0):
                continue
            title = _literal(text, m.end())
            if title is None:
                continue
            end = match_bracket(text, m.end() - 1, "node")
            if end < 0:
                continue
            if text[end: end + 1] == ";":
                end += 1
            out.append(("group" if m.group(1) in ("describe", "suite", "context") else "test", title, m.start(), end))
        return out

    def _find_tests(self, text: str, path: str) -> list[tuple[str, int, int]]:
        calls = self._calls(text)
        groups = [c for c in calls if c[0] == "group"]
        out = []
        for kind, title, a, b in calls:
            if kind != "test":
                continue
            anc = [g[1] for g in sorted(groups, key=lambda g: g[2]) if g[2] < a and b <= g[3]]
            out.append(("::".join([*anc, title]), line_start(text, a), line_end(text, b - 1)))
        return out

    def _find_funcs(self, text: str, path: str) -> list[tuple[str, int, int]]:
        masked = code_mask(text, "node")
        out = []
        for rx in _FUNCS:
            for m in rx.finditer(masked):
                ob = masked.find("{", m.end() - 1)
                nl = masked.find("\n", m.end())
                if ob < 0 or (nl >= 0 and ob > nl and rx is _FUNCS[1]):
                    out.append((m.group(1), m.start(), line_end(text, m.end())))
                    continue
                end = match_bracket(text, ob, "node")
                out.append((m.group(1), m.start(), end if end > 0 else line_end(text, m.end())))
        return out

    def leaf(self, key: str) -> str:
        return key.rsplit("::", 1)[-1]

    def new_test_names(self, tests_b: Mapping[str, str], a_tests: Mapping[str, str | None], tree: Path) -> set[str]:
        names = super().new_test_names(tests_b, a_tests, tree)
        return {n for n in names if len(n) >= 12 and "\n" not in n}  # titles are prose: short ones are common words

    # -- World ---------------------------------------------------------------------------------------------------------------
    def packaging_files(self, tree: Path) -> list[str]:
        return [f for f in ("package.json", *LOCKS, ".npmrc", ".yarnrc.yml") if (Path(tree) / f).is_file()]

    def image_info(self, tree: Path) -> dict[str, Any]:
        pj = package_json(tree)
        name = pj.get("name")
        return {"project": name, "projects": [name] if name else [], "runner": runner_of(tree), "manager": manager_of(tree),
                "package_manager": pj.get("packageManager") or ""}

    def dockerfile(self, tree: Path, info: Mapping[str, Any]) -> str:
        mgr = info.get("manager") or "npm"
        install = {"npm": "if [ -f package-lock.json ] || [ -f npm-shrinkwrap.json ]; then npm ci --no-audit --no-fund; else npm install --no-audit --no-fund; fi",
                   "pnpm": ("(corepack enable && pnpm install --frozen-lockfile) || (corepack disable; npm i -g pnpm@10 && pnpm install --frozen-lockfile) "
                         "|| (npm i -g pnpm@9 && pnpm install --frozen-lockfile)"),  # a packageManager corepack cannot fetch
                   "yarn": "corepack enable && (YARN_NODE_LINKER=node-modules YARN_ENABLE_IMMUTABLE_INSTALLS=false yarn install || yarn install --frozen-lockfile)",
                   "bun": "npm install -g bun && bun install --frozen-lockfile"}[mgr]
        return "\n".join([
            f"FROM {BASE_IMAGE}",
            "RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git ca-certificates >/dev/null && rm -rf /var/lib/apt/lists/*",
            "RUN git config --system safe.directory '*' && git config --system user.email agent@sandbox && git config --system user.name agent",
            "ENV CI=1 HUSKY=0 npm_config_update_notifier=false npm_config_fund=false COREPACK_ENABLE_DOWNLOAD_PROMPT=0 NO_COLOR=1 FORCE_COLOR=0",
            "COPY a /w",
            "WORKDIR /w",
            f"RUN {install}",
            "RUN mkdir -p /opt/deps && mv /w/node_modules /opt/deps/node_modules && rm -rf /w && mkdir -p /w && chmod -R a+rwX /opt/deps",
            # a dependency that depends on the project itself ships a release of it (the Python lane's installed-copy leak):
            # every such copy is deleted, the dependents keep working only if the tests never load them
            *([f"RUN find /opt/deps/node_modules -path '*/node_modules/{info['project']}' -prune -exec rm -rf {{}} + ; true"]
              if info.get("project") and re.fullmatch(r"(@[\w.-]+/)?[\w.-]+", str(info["project"])) else []),
            "ENV npm_config_offline=true",
            "WORKDIR /w",
        ]) + "\n"

    def world_setup_cmd(self, ws: str) -> str:
        return (f"cd {ws} && ln -sfn /opt/deps/node_modules node_modules && mkdir -p .git/info && "
                "(grep -qx node_modules .git/info/exclude 2>/dev/null || echo node_modules >> .git/info/exclude)")

    # -- runs ----------------------------------------------------------------------------------------------------------------
    def run_script(self, tree: Path, targets: Sequence[str]) -> str:
        files = " ".join(_q(t) for t in targets)
        cfg = runner_config(tree)
        conf = f"--config {_q(cfg)} " if cfg else ""
        link = "[ -e node_modules ] || ln -s /opt/deps/node_modules node_modules; export NODE_OPTIONS=--max-old-space-size=1536; "
        if runner_of(tree) == "jest":
            return (link + f'node_modules/.bin/jest --ci --json --outputFile="$O/report" --coverage=false --watchman=false --passWithNoTests {conf}'
                    f'--runTestsByPath {files}; echo "jest exit $?"')
        return (link + f'node_modules/.bin/vitest run --reporter=json --outputFile="$O/report" --coverage.enabled=false --passWithNoTests {conf}'
                f'{files}; echo "vitest exit $?"')

    def parse_report(self, report: str, tree: Path, targets: Sequence[str], tail: str = "") -> tuple[dict[str, str], list[str]]:
        try:
            doc = json.loads(report)
        except ValueError:
            return {}, ["the runner wrote no JSON report"]
        st = {"passed": "passed", "failed": "failed", "pending": "skipped", "skipped": "skipped", "todo": "skipped", "disabled": "skipped"}
        out: dict[str, str] = {}
        notes = []
        for tr in doc.get("testResults") or []:
            name = str(tr.get("name") or tr.get("testFilePath") or "")
            rel = name[3:] if name.startswith("/w/") else name.lstrip("/")
            ars = tr.get("assertionResults") or tr.get("testResults") or []
            for a in ars:
                key = "::".join([*(a.get("ancestorTitles") or []), str(a.get("title") or "")])
                tid = f"{rel}::{key}"
                s = st.get(str(a.get("status")), "failed")
                if tid not in out or out[tid] == "passed":
                    out[tid] = s
            if not ars and str(tr.get("status")) == "failed":
                notes.append(f"load failed: {rel}")
                for t in self.tests_in_target(tree, rel):
                    out.setdefault(t, "error")
        return out, notes

    # -- leaks ---------------------------------------------------------------------------------------------------------------
    def fetch_probe(self, project: str | None) -> str:
        return f"cd /tmp && npm_config_offline=false npm view {_q(project or 'none-such-package-x')} version --registry=https://registry.npmjs.org --fetch-timeout=5000 --fetch-retries=0"

    def own_copy_probe(self, tree: Path, project: str | None) -> str:
        if not project:
            return "true"
        return (f"find /opt/deps/node_modules -path '*/node_modules/{project}/package.json' 2>/dev/null; "
                f"grep -rlF --include=package.json '\"name\": \"{project}\"' /opt/deps/node_modules 2>/dev/null | head -3")


def _q(s: str) -> str:
    import shlex

    return shlex.quote(s)


PROFILE = NodeProfile()
