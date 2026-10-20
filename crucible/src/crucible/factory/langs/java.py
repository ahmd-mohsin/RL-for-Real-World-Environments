"""The Java profile (swarm-pipeline/07 §3.4): single-module Maven projects, surefire XML reports.

- **Paths.** Source: ``src/main/**.java``. Tests: ``src/test/**.java``; test paths are all of ``src/test/`` (resources
  included). Protected: tests, ``pom.xml``, Gradle and wrapper files, ``.mvn/``. Plumbing (hack): ``.mvn/`` (JVM and
  extension configuration steer every build).
- **Tests.** Methods annotated ``@Test`` / ``@ParameterizedTest`` / ``@RepeatedTest`` / ``@TestFactory``; the key is the
  method name (``Inner::method`` inside a nested class). An id is ``<test file>::<key>``; surefire's ``classname``
  names the file (``a.b.FooTest$Inner`` -> ``src/test/java/a/b/FooTest.java``, key ``Inner::m``), parameter suffixes
  (``m(int)[1]``) are dropped by ``key_of_test``.
- **Runs.** ``mvn -o -B test -Dtest=<classes>`` with failing-on-no-tests switched off and the usual report plugins
  skipped; every ``TEST-*.xml`` surefire wrote is the report. A compile failure reports every test of the targets as
  ``error``.
- **World.** ``maven`` (Temurin 21) pinned by digest; ``dependency:go-offline``, and one full test run at build time fill ``~/.m2`` (plugins and providers included); ``/w`` with its compiled classes is deleted. Maven
  runs offline (``-o``).
- **Leaks.** The central repository is unreachable; ``~/.m2`` holds no artifact of the project's own group and
  artifact id; no ``.java`` file in the image defines two of the new test methods.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.langs.base import code_mask, line_start, match_bracket
from crucible.factory.langs.clike import CLikeProfile

BASE_IMAGE = "maven@sha256:d23342b1a070a07555f4ef20c085a81aa40e5f5b3288ebf8ddae2f67e9d760c8"  # maven:3-eclipse-temurin-21
_ANN = re.compile(r"@(?:Test|ParameterizedTest|RepeatedTest|TestFactory|TestTemplate)\b")
_METHOD = re.compile(r"(?:(?:public|protected|private|static|final|synchronized|abstract|default)\s+)*(?:<[^>]*>\s*)?[\w<>\[\],.? ]+?\s+([A-Za-z_]\w*)\s*\(")
_CLASS = re.compile(r"(?<![\w.])(?:class|interface|enum|record)\s+([A-Za-z_]\w*)[^{;]*\{")
_CASE = re.compile(r'<testcase\b([^>]*?)(/>|>(.*?)</testcase>)', re.DOTALL)
SKIPS = ("-Djacoco.skip=true -Dcheckstyle.skip=true -Dspotbugs.skip=true -Dpmd.skip=true -Denforcer.skip=true -Danimal.sniffer.skip=true "
         "-Drat.skip=true -Dlicense.skip=true -Dmaven.javadoc.skip=true -Dgpg.skip=true -Djapicmp.skip=true -Dspotless.check.skip=true")


def _attr(s: str, name: str) -> str:
    m = re.search(rf'\b{name}="([^"]*)"', s)
    return m.group(1) if m else ""


def pom_coords(tree: Path) -> tuple[str, str]:
    f = Path(tree) / "pom.xml"
    if not f.is_file():
        return "", ""
    t = re.sub(r"<parent>.*?</parent>", "", f.read_text(errors="replace"), flags=re.DOTALL)
    t = re.sub(r"<(dependencies|build|profiles|dependencyManagement)>.*", "", t, flags=re.DOTALL)
    g = re.search(r"<groupId>([^<]+)</groupId>", t)
    a = re.search(r"<artifactId>([^<]+)</artifactId>", t)
    if not g:
        pg = re.search(r"<parent>.*?<groupId>([^<]+)</groupId>", f.read_text(errors="replace"), re.DOTALL)
        g = pg
    return (g.group(1).strip() if g else ""), (a.group(1).strip() if a else "")


class JavaProfile(CLikeProfile):
    name = "java"
    lang = "java"
    extensions = (".java",)
    test_file_rx = re.compile(r"(^|/)src/test/.*\.java$")
    test_path_rx = re.compile(r"(^|/)src/test/")
    protected_names = frozenset({"pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts", "gradle.properties", "gradlew",
                                 "gradlew.bat", "mvnw", "mvnw.cmd"})
    protected_rx = re.compile(r"(^|/)\.mvn/|(^|/)gradle/")
    plumbing_rx = re.compile(r"(^|/)\.mvn/")
    recipe = "java-2"
    base_image = BASE_IMAGE
    timeout_s = 1200.0
    tmpfs = "1g"
    leak_globs = ("*.java",)
    plumbing_words = "Maven configuration under .mvn"

    def detect(self, tree: Path) -> bool:
        return (Path(tree) / "pom.xml").is_file()

    def is_source(self, path: str) -> bool:
        return super().is_source(path) and "src/main/" in "/" + path

    def supported(self, tree: Path) -> str:
        t = (Path(tree) / "pom.xml").read_text(errors="replace") if (Path(tree) / "pom.xml").is_file() else ""
        if not t:
            return "no pom.xml"
        return "a multi-module build" if "<modules>" in t else ""

    # -- definitions ---------------------------------------------------------------------------------------------------------
    def _classes(self, text: str, masked: str) -> list[tuple[str, int, int]]:
        out = []
        for m in _CLASS.finditer(masked):
            e = match_bracket(text, m.end() - 1, "java")
            if e > 0:
                out.append((m.group(1), m.start(), e))
        return out

    def _method_at(self, text: str, masked: str, i: int) -> tuple[str, int] | None:
        m = _METHOD.search(masked, i)
        if m is None:
            return None
        pe = match_bracket(text, m.end() - 1, "java")
        if pe < 0:
            return None
        ob = masked.find("{", pe)
        semi = masked.find(";", pe)
        if ob < 0 or (0 <= semi < ob):
            return None
        e = match_bracket(text, ob, "java")
        return (m.group(1), e) if e > 0 else None

    def _find_tests(self, text: str, path: str) -> list[tuple[str, int, int]]:
        masked = code_mask(text, "java")
        classes = self._classes(text, masked)
        out = []
        for a in _ANN.finditer(masked):
            got = self._method_at(text, masked, a.end())
            if not got:
                continue
            name, end = got
            inner = [c for c in sorted(classes, key=lambda c: c[1]) if c[1] < a.start() < c[2]][1:]  # the file's own class is the file
            out.append(("::".join([*(c[0] for c in inner), name]), line_start(text, a.start()), end))
        return out

    def _find_funcs(self, text: str, path: str) -> list[tuple[str, int, int]]:
        masked = code_mask(text, "java")
        out = []
        for m in re.finditer(r"(?m)^[ \t]+(?:(?:public|protected|private|static|final|synchronized|abstract)\s+)+[\w<>\[\],.? ]+?\s+([A-Za-z_]\w*)\s*\(", masked):
            got = self._method_at(text, masked, m.start())
            if got:
                out.append((got[0], m.start(), got[1]))
        return out

    def key_of_test(self, test_id: str) -> tuple[str, str]:
        f, _, k = test_id.partition("::")
        return f, re.sub(r"\(.*$|\[.*$", "", k)

    def leaf(self, key: str) -> str:
        return key.rsplit("::", 1)[-1]

    # -- World ---------------------------------------------------------------------------------------------------------------
    def packaging_files(self, tree: Path) -> list[str]:
        return [f for f in ("pom.xml",) if (Path(tree) / f).is_file()]

    def image_info(self, tree: Path) -> dict[str, Any]:
        g, a = pom_coords(tree)
        return {"project": f"{g}:{a}" if g and a else a or None, "projects": [f"{g}:{a}"] if g and a else [], "group": g, "artifact": a}

    def dockerfile(self, tree: Path, info: Mapping[str, Any]) -> str:
        return "\n".join([
            f"FROM {BASE_IMAGE}",
            "RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*",
            "RUN git config --system safe.directory '*' && git config --system user.email agent@sandbox && git config --system user.name agent",
            "ENV MAVEN_OPTS=-Xmx1g MAVEN_CONFIG=/opt/m2",
            "COPY a /w",
            "WORKDIR /w",
            f"RUN mvn -B -q -Dmaven.repo.local=/opt/m2/repository dependency:go-offline {SKIPS} || true",
            # the whole suite once: surefire resolves its test-framework provider only when it runs tests
            f"RUN mvn -B -q -Dmaven.repo.local=/opt/m2/repository test -Dmaven.test.failure.ignore=true {SKIPS} > /tmp/mvn-build.log 2>&1 || (tail -50 /tmp/mvn-build.log; exit 1)",
            "RUN rm -rf /w && mkdir -p /w && chmod -R a+rwX /opt/m2",
            "WORKDIR /w",
        ]) + "\n"

    # -- runs ----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def class_of(path: str) -> str | None:
        m = re.search(r"src/test/java/(.+)\.java$", path)
        return m.group(1).replace("/", ".") if m else None

    def run_script(self, tree: Path, targets: Sequence[str]) -> str:
        classes = sorted({c for t in targets if (c := self.class_of(t))})
        sel = f"-Dtest={','.join(classes)}" if classes else ""
        return (f"mvn -o -B -Dmaven.repo.local=/opt/m2/repository test {sel} -DfailIfNoTests=false -Dsurefire.failIfNoSpecifiedTests=false "
                f"-Dmaven.test.failure.ignore=true {SKIPS} > \"$O/mvn.log\" 2>&1; echo \"mvn exit $?\" | tee -a \"$O/report\"; tail -c 3000 \"$O/mvn.log\"; "
                "grep -q 'COMPILATION ERROR' \"$O/mvn.log\" && echo '@@COMPILE-FAILED' >> \"$O/report\"; "
                "cat target/surefire-reports/TEST-*.xml >> \"$O/report\" 2>/dev/null; true")

    def parse_report(self, report: str, tree: Path, targets: Sequence[str], tail: str = "") -> tuple[dict[str, str], list[str]]:
        out: dict[str, str] = {}
        for m in _CASE.finditer(report):
            attrs, body = m.group(1), m.group(3) or ""
            cls, name = _attr(attrs, "classname"), _attr(attrs, "name")
            if not cls or not name:
                continue
            outer, _, inner = cls.partition("$")
            f = "src/test/java/" + outer.replace(".", "/") + ".java"
            key = "::".join([*([x for x in inner.split("$") if x] if inner else []), re.sub(r"\(.*$", "", name)]) + (re.search(r"\[.*$", name).group(0) if "[" in name else "")
            st = "failed" if "<failure" in body else "error" if "<error" in body else "skipped" if "<skipped" in body else "passed"
            tid = f"{f}::{key}"
            if tid not in out or out[tid] == "passed":
                out[tid] = st
        notes = []
        if "@@COMPILE-FAILED" in report:
            notes.append("build failed: test compilation")
            for t in targets:
                for x in self.tests_in_target(tree, t):
                    out.setdefault(x, "error")
        return out, notes

    # -- leaks ---------------------------------------------------------------------------------------------------------------
    def fetch_probe(self, project: str | None) -> str:
        return "getent hosts repo.maven.apache.org"

    def own_copy_probe(self, tree: Path, project: str | None) -> str:
        g, a = pom_coords(tree)
        if not g or not a:
            return "true"
        return f"ls -d /opt/m2/repository/{g.replace('.', '/')}/{a} 2>/dev/null"

    def leak_needle(self, name: str) -> str:
        return f" {name}("


PROFILE = JavaProfile()
