"""The debug lane's language layer (swarm-pipeline/07): path rules, test definitions, hiding, result parsing, the
active-profile hooks and the Python no-regression rule. No Docker, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.factory import langs
from crucible.factory.langs.base import active, code_mask, match_bracket, strip_added_lines, use
from crucible.factory.langs.golang import escape_module, import_name, prune_imports

GO = langs.get("go")
RUST = langs.get("rust")
NODE = langs.get("node")
PY = langs.get("python")

GO_TEST = '''package pkg

import (
\t"strings"
\t"testing"
)

// TestOld checks the old behaviour.
func TestOld(t *testing.T) {
\tif Add(1, 2) != 3 {
\t\tt.Fatal("}")
\t}
}

func TestNew(t *testing.T) {
\tif !strings.HasPrefix(Name(), "x") {
\t\tt.Fatal(`raw } brace`)
\t}
}
'''


def test_bracket_matcher_skips_literals_and_comments() -> None:
    js = "f(a, '(', `${g(1)})`, /\\)/, /* ) */ b) + 1"
    assert match_bracket(js, 1, "node") == js.index(" + 1")
    go = 'f("(", `)`, \'(\') // )\n'
    assert match_bracket(go, 1, "go") == go.index(" //")
    rs = "f(r#\")\"#, 'a', x) // )"
    assert match_bracket(rs, 1, "rust") == rs.index(" //")
    m = code_mask('x := "a{b}"', "go")
    assert "{" not in m and len(m) == len('x := "a{b}"')


def test_go_paths_defs_and_strip() -> None:
    assert GO.is_test_file("a/b_test.go") and not GO.is_source("a/b_test.go") and GO.is_source("a/b.go")
    assert GO.is_test_path("x/testdata/in.yaml") and GO.is_protected("go.sum") and GO.is_protected("vendor/x/y.go")
    assert GO.is_plumbing("go.work") and not GO.is_source("vendor/x/y.go") and not GO.is_source("docs/x.go")
    defs = GO.test_defs(GO_TEST, "pkg/a_test.go")
    assert set(defs) == {"TestOld", "TestNew"}
    out = GO.strip_tests(GO_TEST, {"TestNew"}, "pkg/a_test.go")
    assert "TestNew" not in out and "TestOld" in out and '"strings"' not in out and '"testing"' in out  # the import only it used
    assert GO.strip_tests(GO_TEST, {"TestNew", "TestOld"}) is None  # nothing left: the caller removes the file
    assert GO.key_of_test("pkg/a_test.go::TestNew/case_1") == ("pkg/a_test.go", "TestNew")
    assert GO.changed_defs("package p\nfunc A() int { return 1 }\n", "package p\nfunc A() int { return 2 }\nfunc (s *S) B() {}\n") == ["A", "S.B"]


def test_go_import_names() -> None:
    assert import_name("gopkg.in/yaml.v3") == "yaml" and import_name("github.com/x/y/v2") == "y" and import_name("github.com/x/go-bar") is None
    assert escape_module("github.com/BurntSushi/toml") == "github.com/!burnt!sushi/toml"
    keep = 'package p\nimport (\n\tbar "github.com/x/go-bar"\n)\nfunc f() { bar.X() }\n'
    assert prune_imports(keep, keep) == keep


def test_go_parse_report(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/m\n\ngo 1.22\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a_test.go").write_text(GO_TEST)
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "b_test.go").write_text("package bad\nfunc TestB(t *testing.T) {}\n")
    evs = [{"Action": "pass", "Package": "example.com/m/pkg", "Test": "TestOld"}, {"Action": "fail", "Package": "example.com/m/pkg", "Test": "TestNew/sub"},
           {"Action": "build-fail", "ImportPath": "example.com/m/bad [example.com/m/bad.test]"}]
    res, notes = GO.parse_report("\n".join(json.dumps(e) for e in evs), tmp_path, ["pkg/a_test.go", "bad/b_test.go"])
    assert res == {"pkg/a_test.go::TestOld": "passed", "pkg/a_test.go::TestNew/sub": "failed", "bad/b_test.go::TestB": "error"}
    assert notes == ["build failed: example.com/m/bad"]
    assert GO.packages_of(["pkg/a_test.go", "x_test.go"]) == [".", "./pkg"]


NODE_TEST = '''import { describe, it, expect } from "vitest";
import { parse } from "../src/index";

describe("parse", () => {
  it("handles empty input", () => {
    expect(parse("")).toEqual([]);
  });

  it(`keeps "quotes" and } braces`, () => {
    expect(parse("a}")).toEqual(["a}"]);
  });
  it.each([1, 2])("case %s", (n) => expect(n).toBeTruthy());
});

test('top level', () => { expect(/\\)/.test(")")).toBe(true) });
'''


def test_node_paths_defs_strip_and_report(tmp_path: Path) -> None:
    assert NODE.is_test_file("src/a.test.ts") and NODE.is_test_file("__tests__/x.js") and NODE.is_test_file("src/addDays/test.ts")
    assert NODE.is_source("src/a.ts") and not NODE.is_source("dist/a.js") and NODE.is_protected("pnpm-lock.yaml")
    assert NODE.is_protected("vitest.config.ts") and NODE.is_plumbing(".npmrc") and NODE.is_plumbing("node_modules/x/index.js")
    defs = NODE.test_defs(NODE_TEST, "test/p.test.ts")
    assert set(defs) == {"parse::handles empty input", 'parse::keeps "quotes" and } braces', "top level"}  # it.each is not a definition
    out = NODE.strip_tests(NODE_TEST, {"parse::handles empty input"}, "test/p.test.ts")
    assert "handles empty input" not in out and "keeps" in out and "top level" in out
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "p.test.ts").write_text(NODE_TEST)
    rep = {"testResults": [{"name": "/w/test/p.test.ts", "status": "failed", "assertionResults": [
        {"ancestorTitles": ["parse"], "title": "handles empty input", "status": "passed"},
        {"ancestorTitles": [], "title": "top level", "status": "failed"}, {"ancestorTitles": ["parse"], "title": "skip me", "status": "pending"}]},
        {"name": "/w/test/broken.test.ts", "status": "failed", "assertionResults": [], "message": "SyntaxError"}]}
    (tmp_path / "test" / "broken.test.ts").write_text("it('first broken test', () => {})\n")
    res, notes = NODE.parse_report(json.dumps(rep), tmp_path, ["test/p.test.ts"])
    assert res["test/p.test.ts::parse::handles empty input"] == "passed" and res["test/p.test.ts::top level"] == "failed"
    assert res["test/p.test.ts::parse::skip me"] == "skipped" and res["test/broken.test.ts::first broken test"] == "error"
    assert notes == ["load failed: test/broken.test.ts"]


def test_node_runner_and_workspace(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(json.dumps({"name": "lib", "devDependencies": {"vitest": "^3"}, "workspaces": ["docs"]}))
    (tmp_path / "pnpm-lock.yaml").write_text("x")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.test.ts").write_text("it('x', () => {})\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "package.json").write_text("{}")
    (tmp_path / "docs" / "b.test.ts").write_text("it('y', () => {})\n")
    assert NODE.supported(tmp_path) == "" and NODE.suite_targets(tmp_path) == ["src/a.test.ts"]
    info = NODE.image_info(tmp_path)
    assert info["runner"] == "vitest" and info["manager"] == "pnpm" and info["project"] == "lib"
    assert "vitest run --reporter=json" in NODE.run_script(tmp_path, ["src/a.test.ts"])


RUST_TEST = '''use mycrate::parse;

#[test]
fn parses_empty() {
    assert_eq!(parse(""), Ok(vec![]));
}

mod nested {
    use super::*;

    /// Doc comment.
    #[test]
    #[ignore]
    fn handles_brace() {
        let s = "}";
        let c = '}';
        assert!(parse(s).is_err() && c == '}');
    }

    fn helper<'a>(x: &'a str) -> &'a str { x }
}
'''


def test_rust_defs_strip_and_report(tmp_path: Path) -> None:
    assert RUST.is_test_file("tests/parse.rs") and RUST.is_test_file("src/de/tests.rs") and not RUST.is_test_file("src/lib.rs")
    assert RUST.is_protected("Cargo.lock") and RUST.is_plumbing("build.rs") and RUST.is_plumbing(".cargo/config.toml") and RUST.is_test_path("tests/snapshots/a.snap")
    assert set(RUST.test_defs(RUST_TEST, "tests/parse.rs")) == {"parses_empty", "nested::handles_brace"}
    assert set(RUST.test_defs("#[test]\nfn t() {}\n", "src/de/tests.rs")) == {"de::tests::t"}
    out = RUST.strip_tests(RUST_TEST, {"nested::handles_brace"}, "tests/parse.rs")
    assert "handles_brace" not in out and "#[ignore]" not in out and "Doc comment" not in out and "fn helper" in out and "parses_empty" in out
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "mycrate"\nversion = "0.1.0"\n')
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "parse.rs").write_text(RUST_TEST)
    (tmp_path / "tests" / "other.rs").write_text("#[test]\nfn other_one() {}\n")
    rep = ("@@T tests/parse.rs\nrunning 2 tests\ntest parses_empty ... ok\ntest nested::handles_brace ... ignored\n@@RC 0\n"
           "@@T tests/other.rs\nerror[E0425]: cannot find value\nerror: could not compile `mycrate`\n@@RC 101\n")
    res, notes = RUST.parse_report(rep, tmp_path, ["tests/parse.rs", "tests/other.rs"])
    assert res == {"tests/parse.rs::parses_empty": "passed", "tests/parse.rs::nested::handles_brace": "skipped", "tests/other.rs::other_one": "error"}
    assert notes == ["build failed: tests/other.rs"]
    script = RUST.run_script(tmp_path, ["tests/parse.rs"])
    assert "-p mycrate --test parse" in script
    for p in ("tests/helpers/mod.rs", "tests/multi/main.rs", "tests/multi/part.rs"):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text("#[test]\nfn t() {}\n")
    assert RUST.suite_targets(tmp_path) == ["tests/multi/main.rs", "tests/other.rs", "tests/parse.rs"]  # helpers and binary modules are no targets
    assert "@@T tests/multi/main.rs" in RUST.run_script(tmp_path, ["tests/multi/part.rs"])


def test_active_profile_hooks_and_python_no_regression() -> None:
    from crucible.factory.generators.rebreak import allowed_file
    from crucible.factory.lanes.executable import is_protected, is_test_path
    from crucible.factory.lanes.windows_git import Unit, is_source, key_of_test
    from crucible.factory.pathcheck import PathRules, check_paths

    assert active() is None and is_test_path("tests/test_a.py") and not is_test_path("a_test.go") and is_source("pkg/a.py")
    with use(PY):  # the Python profile is "no profile": every predicate takes the original path
        assert active() is None and not is_test_path("a_test.go") and key_of_test("t.py::C::f[1]") == ("t.py", "C::f")
    with use(GO):
        assert active() is GO and is_test_path("a_test.go") and is_protected("go.mod") and is_source("pkg/a.go") and not is_source("pkg/a.py")
        u = Unit(sha="s", parent="p", merge=False, date="", subject="fix", files=["pkg/a.go", "pkg/a_test.go", "pkg/testdata/x.golden", "CHANGELOG.md"])
        assert u.src_files == ["pkg/a.go"] and u.test_files == ["pkg/a_test.go"]
        assert all(allowed_file(p) for p in u.files) and not allowed_file("go.mod") and not allowed_file("go.work")
        rep = check_paths(PathRules(), patch="", trail=[], entries=[("A", "go.work"), ("M", "pkg/a_test.go")])
        assert "path:plumbing" in rep.violations and "path:protected" in rep.violations and rep.hack_suspect
    assert active() is None


def test_pathcheck_language_package_managers() -> None:
    from crucible.factory.pathcheck import ToolCall, trail_hits

    h = trail_hits([ToolCall("shell", {"command": "go get github.com/x/y@v1"}), ToolCall("shell", {"command": "npm view mylib version"}),
                    ToolCall("shell", {"command": "cargo add serde"}), ToolCall("shell", {"command": "go test ./..."})], projects=["mylib"])
    assert any("go get" in x for x in h.install) and any("cargo add" in x for x in h.install) and any("npm view" in x for x in h.fetch)
    assert not any("go test" in x for x in h.install + h.fetch)


def test_lint_debug_and_epilogue() -> None:
    from crucible.factory.generators.debug import epilogue_for, lint_debug

    bad = lint_debug("Calling scanTokens on `a}` crashes; see TestScanBrace in lexer_test.go.", faulty={"scanTokens"}, hidden={"TestScanBrace"})
    kinds = {f.split(":", 1)[0] for f in bad}
    assert {"lint faulty_name", "lint hidden_test", "lint file_path"} <= kinds
    assert lint_debug("Parsing a document whose last key is quoted drops that key.", faulty={"scanTokens"}, hidden={"TestScanBrace"}) == []
    assert "go.work files or a vendor directory" in epilogue_for("go") and "conftest.py" in epilogue_for("python")
    assert epilogue_for("node").startswith("How this works:")


def test_packaging_key_keeps_lockfile_versions(tmp_path: Path) -> None:
    from crucible.factory.langs.runtime import packaging_key

    (tmp_path / "Cargo.toml").write_text('[package]\nname = "x"\nversion = "1.0.0"\n')
    (tmp_path / "Cargo.lock").write_text('[[package]]\nname = "aho-corasick"\nversion = "1.1.3"\n')
    k1 = packaging_key(tmp_path, ["Cargo.toml", "Cargo.lock"])
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "x"\nversion = "1.0.1"\n')
    assert packaging_key(tmp_path, ["Cargo.toml", "Cargo.lock"]) == k1  # a release bump changes no dependency
    (tmp_path / "Cargo.lock").write_text('[[package]]\nname = "aho-corasick"\nversion = "1.1.4"\n')
    assert packaging_key(tmp_path, ["Cargo.toml", "Cargo.lock"]) != k1  # a dependency bump does


def test_strip_added_lines() -> None:
    assert strip_added_lines("# Changes\n- fix a\n- fix b\n", ["- fix b"]) == "# Changes\n- fix a\n"


def test_profiles_registry() -> None:
    assert langs.get("golang") is GO and langs.get("ts") is NODE
    with pytest.raises(KeyError):
        langs.get("cobol")


JAVA_TEST = '''package a.b;

import org.junit.jupiter.api.Test;

public class FooTest {
    @Test
    public void parsesEmpty() {
        assertEquals("}", Foo.parse("{"));
    }

    @Nested
    class Inner {
        @ParameterizedTest
        @ValueSource(ints = {1, 2})
        void handlesInts(int n) { assertTrue(n > 0); }
    }

    private static String helper() { return "x"; }
}
'''


def test_java_defs_and_report(tmp_path: Path) -> None:
    java = langs.get("java")
    assert java.is_test_file("src/test/java/a/b/FooTest.java") and java.is_source("src/main/java/a/b/Foo.java") and java.is_protected("pom.xml")
    assert set(java.test_defs(JAVA_TEST, "src/test/java/a/b/FooTest.java")) == {"parsesEmpty", "Inner::handlesInts"}
    out = java.strip_tests(JAVA_TEST, {"parsesEmpty"}, "src/test/java/a/b/FooTest.java")
    assert "parsesEmpty" not in out and "handlesInts" in out and "helper" in out
    rep = ('<?xml version="1.0"?><testsuite><testcase classname="a.b.FooTest" name="parsesEmpty"/>'
           '<testcase classname="a.b.FooTest$Inner" name="handlesInts(int)[1]"><failure message="x"/></testcase></testsuite>')
    res, notes = java.parse_report(rep, tmp_path, [])
    assert res == {"src/test/java/a/b/FooTest.java::parsesEmpty": "passed", "src/test/java/a/b/FooTest.java::Inner::handlesInts[1]": "failed"}
    assert java.key_of_test("src/test/java/a/b/FooTest.java::Inner::handlesInts[1]")[1] == "Inner::handlesInts" and notes == []
    assert "-Dtest=a.b.FooTest" in java.run_script(tmp_path, ["src/test/java/a/b/FooTest.java"])


def test_node_runner_flags(tmp_path: Path) -> None:
    """Node flags the repository's own runner script passes (jest's ESM support) reach NODE_OPTIONS; nothing else does."""
    from crucible.factory.langs.node import runner_node_flags

    pkg = {"scripts": {"test": "run-s test-jest", "test-jest": "node --experimental-vm-modules node_modules/jest/bin/jest.js",
                       "lint": "node --inspect eslint"}, "devDependencies": {"jest": "1"}}
    (tmp_path / "package.json").write_text(json.dumps(pkg))
    assert runner_node_flags(tmp_path) == ["--experimental-vm-modules"]
    assert "NODE_OPTIONS='--max-old-space-size=1536 --experimental-vm-modules'" in NODE.run_script(tmp_path, ["a.test.mjs"])
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "vitest run"}, "devDependencies": {"vitest": "1"}}))
    assert runner_node_flags(tmp_path) == [] and "NODE_OPTIONS=--max-old-space-size=1536;" in NODE.run_script(tmp_path, ["a.test.ts"])


def test_fairness_flags() -> None:
    """A hidden milestone test named after a changed non-API function, and strip mode's visible test data, are flagged
    for review (never a gate); an exported Go name is the package's API and is not flagged."""
    from crucible.factory.generators.debug import NAME, fairness_flags, task_intent

    rec = {"lang": "python", "milestones": [{"id": "m01", "unit": "u", "requires": [], "tests": [
        "tests/test_black.py::BlackTestCase::test_lines_with_leading_tabs_expanded", "tests/test_format.py::test_simple_format[docstring]"]}],
           NAME: {"hidden": {"mode": "strip", "visible_test_changes": ["tests/data/case.py"]}}}
    flags = fairness_flags(rec, {"lines_with_leading_tabs_expanded", "format_str"})
    assert any(f.startswith("internal_function_check: m01 test test_lines_with_leading_tabs_expanded") for f in flags)
    assert any(f.startswith("visible_test_data:") for f in flags) and len(flags) == 2
    go = {"lang": "go", "milestones": [{"id": "m01", "unit": "u", "requires": [], "tests": ["a_test.go::TestEllipsis/cjk", "a_test.go::TestparseInt64"]}]}
    assert fairness_flags(go, {"Ellipsis", "parseInt64"}) == ["internal_function_check: m01 test TestparseInt64 is named after the changed function parseInt64"]
    assert task_intent(go, {"Ellipsis"})["fairness_flags"] == []
