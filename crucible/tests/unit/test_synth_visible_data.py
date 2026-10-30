"""DESIGN §7.7 item 20: answers leaked through visible test data on the repository substrate (TAINT ``visible_data``,
static; ``visible_check``, executed) and the re-created strip-mode defect version. Fixture repo, in-process runner."""

from __future__ import annotations

import asyncio
from pathlib import Path

from crucible.factory.synth_patch_wire import _drop_functions
from crucible.synth.falsify.base import falsify
from crucible.synth.falsify.patch import PatchTAINT, visible_data
from crucible.synth.patch import PatchEnv
from tests.unit.synth_patch_fixtures import FIXED, HIDDEN_TESTS, TEST, fixture_runner, make_repo, spec_for


def run(c):
    return asyncio.run(c)


def _env(tmp_path: Path, **kw) -> PatchEnv:
    base = make_repo(tmp_path, **{k: kw.pop(k) for k in ("visible",) if k in kw})
    for p, t in kw.pop("extra_files", {}).items():
        (base / p).parent.mkdir(parents=True, exist_ok=True)
        (base / p).write_text(t)
    return PatchEnv(spec_for(base, **kw))


def _by_op(res, op):
    return [f for f in res.findings if f.op == op and f.outcome.valid]


def test_a_visible_test_the_fix_turns_green_is_an_executed_leak(tmp_path):
    env = _env(tmp_path, visible=HIDDEN_TESTS)  # the hidden checks were left in the agent's tree
    runner = fixture_runner(env)
    res = run(falsify(env.task(), PatchTAINT(env, runner), env.context(runner)))
    (f,) = _by_op(res, "visible_check")
    assert f.label == "confirmed" and f.defect == "D7"
    assert set(f.outcome.data["milestone_tests"]) == {f"{TEST}::test_add", f"{TEST}::test_mul"}
    clean = _env(tmp_path / "ok")
    r2 = run(falsify(clean.task(), PatchTAINT(clean, fixture_runner(clean)), clean.context(fixture_runner(clean))))
    assert r2.c == 0 and not r2.suspects  # the fixture's visible test passes at A′ and at B: nothing to see


def test_fix_answers_and_shared_cases_are_static_evidence(tmp_path):
    fixed = FIXED + '\n\ndef format_total(x):\n    return "the total is " + str(x)\n'
    hidden = HIDDEN_TESTS + ('\n\ndef test_format():\n    assert format_total(5) == "the total is 5"\n'
                             '    assert describe("one two three") == "three words here"\n    assert describe("alpha beta") == "two words here"\n')
    readme = "Formatting: format_total(5) prints the total is 5.\n"
    env = _env(tmp_path, oracle_files={"calc.py": fixed}, hidden_tests={TEST: hidden}, extra_files={"docs/usage.md": readme})
    ev = visible_data(env, env.base_text)
    assert ev["fix_definitions"] == 1 and ev["fix_literals"] == 1
    assert {a["file"] for a in ev["fix_answers"]} == {"docs/usage.md"} and set(ev["fix_answers"][0]["tokens"]) == {"format_total", "the total is "}
    res = run(falsify(env.task(), PatchTAINT(env), env.context(fixture_runner(env))))
    (f,) = _by_op(res, "visible_data")
    assert f.label == "confirmed" and f.defect == "D7"
    cases = ('    assert describe("one two three") == "three words here"\n    assert describe("alpha beta") == "two words here"\n')
    env2 = _env(tmp_path / "cases", hidden_tests={TEST: hidden}, extra_files={"tests/data/cases.txt": cases})
    ev2 = visible_data(env2, env2.base_text)
    assert ev2["hidden_cases"] and ev2["hidden_cases"][0]["file"] == "tests/data/cases.txt" and ev2["hidden_cases"][0]["lines"] == 2
    res2 = run(falsify(env2.task(), PatchTAINT(env2), env2.context(fixture_runner(env2))))
    (g,) = _by_op(res2, "visible_data")
    assert g.label == "suspect" and g.defect == "D7"  # shared case lines alone: whether they give the answer is a judgement


def test_imports_comments_and_common_lines_are_not_evidence(tmp_path):
    fixed = FIXED + "\n\n# e.g. def format_total(x) would print it\n"
    hidden = HIDDEN_TESTS + '\n\nimport collections.abc as abc_module_name\n'
    env = _env(tmp_path, oracle_files={"calc.py": fixed}, hidden_tests={TEST: hidden},
               extra_files={"docs/a.md": "import collections.abc as abc_module_name\nformat_total\n", "docs/b.md": "format_total\n"})
    ev = visible_data(env, env.base_text)
    assert ev == {"hidden_cases": [], "fix_answers": [], "fix_literals": 0, "fix_definitions": 0}


def test_the_pre_table_strip_version_drops_only_the_stripped_functions():
    go = 'package x\n\nvar cases = []string{\n\t"a",\n\t"b",\n}\n\n// TestNew checks b.\nfunc TestNew(t *testing.T) {\n\tif f() {\n\t}\n}\n\nfunc TestOld(t *testing.T) {\n}\n'
    out = _drop_functions(go, ["TestNew"], "go")
    assert "TestNew" not in out and "TestOld" in out and '"b",' in out and "checks b" not in out
    py = "CASES = [1, 2]\n\n\ndef test_new():\n    assert 1\n\n    assert 2\n\n\ndef test_old():\n    pass\n"
    out2 = _drop_functions(py, ["test_new"], "python")
    assert "test_new" not in out2 and "def test_old" in out2 and "CASES = [1, 2]" in out2
