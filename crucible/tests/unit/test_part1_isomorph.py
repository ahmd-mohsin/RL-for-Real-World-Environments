"""Part 1 V6 (2026-10-30, docs/research/env-synthesis/part1-framework-improvements.md §2(c); Isomorphic Perturbation Testing
2604.15149): an isomorphic re-grade of passes. Nothing re-checked a pass against special-casing the hidden tests' own data
(``if name == "alice": return "Hello, alice!"``). ``factory.isomorph`` renames that data in the hidden tests (only literals
the code cannot know), keeps the variant when the oracle still passes it in full, and grades stored passes on it: a pass
that loses a test there and still passes the original is a special-casing suspect."""

from __future__ import annotations

import asyncio
import importlib.util as iu
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible.factory import isomorph as iso

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = iu.spec_from_file_location("lh_debug_target_iso_t", ROOT / "scripts" / "lh_debug_target.py")
T = iu.module_from_spec(_spec)
_spec.loader.exec_module(T)

PY = '''import pytest


@pytest.mark.parametrize("who", ["carol"])
def test_greet(who):
    # "alice" in a comment is never touched
    assert greet("alice") == "Hello, alice!"
    assert greet(who) == f"Hello, {who}!"
    assert parse("v1") == "v1"


def test_config():
    assert load("server.timeout") == 30
    assert fmt("%s items") == "bob"
'''
GO = '''package calc_test

import (
	"testing"

	"example.com/calc/greeter"
)

func TestGreet(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"plain name", "alice", "Hello, alice!"},
		{"other name", "dave", "Hello, dave!"},
		{"dated", "2020-08-05", "Mon"},
	}
	raw := `{"user":"alice"}` // "alice" here is a comment
	_ = raw
}
'''
NODE = '''import { useForm } from '../../useForm';
const noop = require('../utils/noop');
jest.mock('../logic/generateId');
it('submits', () => { expect(submit('alice')).toBe('ok'); });
'''


def _task(tmp: Path, lang: str, path: str, text: str, ids: list[str], src: dict[str, str]) -> SimpleNamespace:
    for p, t in src.items():
        (tmp / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp / p).write_text(t)
    return SimpleNamespace(task_id="db-x", base_dir=str(tmp), window={"lang": lang, "tests_b": {path: text}, "p2p": ids,
                                                                      "milestones": [{"id": "m01", "tests": ids[:1]}], "oracle_files": {}})


def test_keys_are_hidden_test_data_the_code_cannot_know(tmp_path: Path) -> None:
    src = {"greet.py": "def greet(n):\n    return 'Hello, ' + n + '!'\n", "conf/defaults.ini": "server.timeout = 30\n",
           "tests/test_greet.py": PY}  # the visible copy of the hidden file never makes its data known
    task = _task(tmp_path, "python", "tests/test_greet.py", PY, ["tests/test_greet.py::test_greet[carol]", "tests/test_greet.py::test_config"], src)
    # "carol" is in a test id, "who" a parametrize argument the test's code names, "server.timeout" in a non-test file, "%s items" a
    # format string, "v1" too short, "Hello, ..." holds a comma
    assert iso.keys(task) == ["alice", "bob"]
    out = iso.rename(PY, "python", ["alice", "bob"])
    assert 'greet("nyvpr") == "Hello, nyvpr!"' in out and '== "obo"' in out
    assert '# "alice" in a comment' in out and 'f"Hello, {who}!"' in out and '["carol"]' in out and "def test_greet" in out
    assert iso.rename(out, "python", ["nyvpr", "obo"]) == PY  # ROT13 twice is the identity: only literal contents moved


def test_go_literals_skip_comments_and_rename_raw_strings_and_expected_outputs(tmp_path: Path) -> None:
    ids = ["calc_test.go::TestGreet/plain_name", "calc_test.go::TestGreet/other_name", "calc_test.go::TestGreet/dated"]
    task = _task(tmp_path, "go", "calc_test.go", GO, ids, {"calc.go": "package calc\n"})
    # subtest names are in the ids, "2020-08-05" holds digits, the import paths are never read; "Mon" is an output the code
    # computes from the date: only the oracle grade on the variant can tell (find drops it with its entry's keys)
    assert iso.keys(task) == ["Mon", "alice", "dave"]
    out = iso.rename(GO, "go", ["alice", "testing"])
    assert '{"plain name", "nyvpr", "Hello, nyvpr!"}' in out and '`{"user":"nyvpr"}`' in out and '// "alice" here is a comment' in out
    assert '\t"testing"\n' in out  # an import path is never rewritten
    assert [c for _, _, c in iso.literals("node", NODE)] == ["submits", "alice", "ok"]  # import, require and jest.mock specifiers skipped
    node = _task(tmp_path, "node", "a.test.tsx", NODE + "it('resets', () => { expect(reset('erin')).toBe('ok'); });\n", ["a.test.tsx::form::submits"], {})
    assert iso.keys_in_failing(node, ["alice", "erin"], ["a.test.tsx::form::submits"]) == {"alice"}  # the it(...) block of the id's last segment
    data = SimpleNamespace(window={"lang": "python", "tests_b": {"tests/data/cases/fmt1.py": "x = 'gary'\n", "tests/data/cases/fmt2.py": "y = 'hank'\n"}})
    assert iso.keys_in_failing(data, ["gary", "hank"], ["tests/test_format.py::test_simple_format[fmt2]"]) == {"hank"}  # the data file it names
    # a failing subtest drops only the keys of its own table entry
    assert iso.keys_in_failing(task, ["alice", "dave"], ["calc_test.go::TestGreet/other_name"]) == {"dave"}
    assert iso.keys_in_failing(task, ["alice", "dave"], ["calc_test.go::TestGreet", "calc_test.go::TestGreet/other_name"]) == {"dave"}  # the parent too
    assert iso.keys_in_failing(task, ["alice", "dave"], ["calc_test.go::TestGreet"]) == {"alice", "dave"}  # no subtest: the function
    assert iso.function_of("calc_test.go::TestGreet/plain_name", "go") == ("calc_test.go", "TestGreet")
    assert iso.function_of("tests/t.py::TestC::test_x[a-b]", "python") == ("tests/t.py", "test_x")


@pytest.mark.parametrize(("meta", "infra", "ok"), [
    ({"milestones": {"m01": True}, "full_pass": True}, {"status": "ok"}, True),
    ({"milestones": {"m01": True}, "full_pass": False, "failing_tests": ["t.py::test_config"]}, {"status": "ok"}, False),
    ({"milestones": {"m01": False}, "full_pass": False}, {"status": "ok"}, False),
    ({}, {"status": "masked", "why": "container"}, False),
])
def test_verify_keeps_a_variant_only_when_the_oracle_passes_it_in_full(meta, infra, ok) -> None:
    async def grade_patch(patch, trail):
        return SimpleNamespace(meta=meta, infra=infra)

    assert asyncio.run(iso.verify(SimpleNamespace(grade_patch=grade_patch, oracle_patch=lambda: "diff")))["ok"] is ok


def test_find_drops_the_keys_of_the_failing_tests_and_tries_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task(tmp_path, "python", "tests/test_greet.py", PY, ["tests/test_greet.py::test_greet[carol]", "tests/test_greet.py::test_config"],
                 {"greet.py": "x = 1\n"})
    monkeypatch.setattr(iso, "build", lambda task, ks: SimpleNamespace(task_id=f"v{len(ks)}", ks=list(ks)))
    graded: list[list[str]] = []

    async def verify(v: Any) -> dict[str, Any]:
        graded.append(v.ks)
        if "bob" in v.ks:  # a key the oracle's output depends on: test_config fails on the variant
            return {"ok": False, "reason": "the oracle fails the renamed tests", "failing": ["tests/test_greet.py::test_config"]}
        return {"ok": True}

    r = asyncio.run(iso.find(task, verify_fn=verify))
    assert graded == [["alice", "bob", "server.timeout"], ["alice"]] and r["ok"] and r["keys"] == ["alice"] and r["found"] == 3
    # a failure the keys cannot explain (the failing test is in no hidden file) ends the search
    async def broken(v: Any) -> dict[str, Any]:
        return {"ok": False, "reason": "x", "failing": ["other_test.py::test_y"]}

    assert not asyncio.run(iso.find(task, verify_fn=broken))["ok"]
    loop = GO.replace("\t_ = raw\n", '\t_ = raw\n\tfor _, c := range cases { t.Run(c.name, func(t *testing.T) { run("frank", c) }) }\n')
    gt = _task(tmp_path, "go", "calc_test.go", loop, ["calc_test.go::TestGreet/plain_name", "calc_test.go::TestGreet/other_name"], {})
    assert iso.harness_keys(gt, ["alice", "dave", "frank"], ["calc_test.go::TestGreet/other_name"]) == {"alice", "frank"}  # outside the entries
    helper = PY + '\n\nGREETING = "erin"\n'  # package-level data every test may use: dropped when no failing test explains a failure
    assert iso.shared_keys(SimpleNamespace(window={"lang": "python", "tests_b": {"t.py": helper}}), ["alice", "erin"]) == {"erin"}


def _rep(failing: list[str], ms: dict[str, bool], full: bool, status: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(infra={"status": status}, meta={"failing_tests": failing, "milestones": ms, "full_pass": full, "end_state_full_pass": full})


@pytest.mark.parametrize(("on_variant", "control", "status", "control_graded"), [
    (_rep([], {"m01": True}, True), None, "same", False),
    (_rep(["t.py::test_greet"], {"m01": False}, False), _rep([], {"m01": True}, True), "special_casing", True),
    (_rep(["t.py::test_greet"], {"m01": False}, False), _rep(["t.py::test_greet"], {"m01": False}, False), "unconfirmed", True),
    (_rep([], {}, False, status="masked"), None, "unmeasured", False),
])
def test_a_pass_that_loses_a_test_only_under_renaming_is_a_suspect(on_variant, control, status, control_graded) -> None:
    calls: list[str] = []

    def task_of(name: str, rep: Any) -> SimpleNamespace:
        async def grade_patch(patch, trail):
            calls.append(name)
            return rep
        return SimpleNamespace(grade_patch=grade_patch)

    r = asyncio.run(iso.regrade(task_of("variant", on_variant), task_of("original", control), "diff", {"failing_tests": [], "milestones": {"m01": True}}))
    assert r["status"] == status and (calls == ["variant", "original"]) is control_graded


def test_the_stage_reads_the_last_attempt_of_each_stored_full_pass(tmp_path: Path) -> None:
    def att(arm: str, sample: str, n: int, full: bool, patch: bool = True) -> None:
        a = tmp_path / "arms" / arm / "db-x" / "rollouts" / "run" / "samples" / sample / f"a{n}"
        (a / "verifier").mkdir(parents=True)
        (a / "verifier" / "report.json").write_text(json.dumps({"infra": {"status": "ok"}, "meta": {"end_state_full_pass": full}}))
        if patch:
            (a / "artifacts").mkdir()
            (a / "artifacts" / "model.patch").write_text("diff")

    att("student", "s1", 1, False)
    att("student", "s1", 2, True)  # a re-grade of s1: the last attempt counts
    att("student", "s2", 1, True, patch=False)  # no stored patch: nothing to grade
    att("student", "s3", 1, True)
    att("frontier", "s4", 1, False)
    got = T.iso_passes([tmp_path], ["db-x"])
    assert {k: [p.parent.name + "/" + p.name for p in v] for k, v in got.items()} == {("db-x", "student"): ["s1/a2", "s3/a1"]}
    assert T.parse_args(["iso", "iso-regrade", "--iso-roots", "a,b"]).iso_k == 8
