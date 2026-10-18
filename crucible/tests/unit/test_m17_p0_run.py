"""P0 run fixes (report 11): D7, the policy's tool-output cap holds the World's whole 30k-character tool output (a
400-line ``read_file`` page is never cut in the middle); D8, findings sent back to the prompt author come from the failing
hidden test's own source and parameters, and an auditor's free text is refused."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from crucible.core.schema.actions import Observation, ToolResult
from crucible.factory.lanes.windows import findings_from_tests, hidden_test_facts
from crucible.factory.lanes.windows_world import world_spec

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("lh_run_p0", ROOT / "scripts" / "lh_run.py")
lh = importlib.util.module_from_spec(_spec)
sys.modules["lh_run_p0"] = lh
_spec.loader.exec_module(lh)

SEMVER_TEST = '''import pytest

from semver import cmd_check


@pytest.mark.parametrize("version", ["1.2.3", "0.0.1-alpha"])
def test_should_check_valid(version):
    assert cmd_check(version) is None


def test_should_raise_on_invalid():
    with pytest.raises(ValueError, match="not valid"):
        cmd_check("1.2")
'''
REC = {"milestones": [{"id": "m03", "tests": ["test_semver.py::test_should_check_valid[1.2.3]", "test_semver.py::test_should_raise_on_invalid"]}],
       "p2p": [], "collateral": [], "tests_b": {"test_semver.py": SEMVER_TEST}}


def test_d7_the_tool_output_cap_holds_the_worlds_whole_output(tmp_path: Path) -> None:
    world_max = world_spec(tmp_path, tmp_path).apps[0].params["max_output"]
    cfg = lh.EpisodeCfg()
    assert cfg.max_tool_output >= world_max == 30000
    page = "".join(f"{i:4d}  def function_number_{i}(argument_a, argument_b):  # a long module\n" for i in range(400))  # read_file's 400 lines
    assert 25_000 < len(page) <= world_max

    class _Nil:
        pass

    pol = lh.CappedToolPolicy("agent", _Nil(), max_tool_output=cfg.max_tool_output)
    pol._ingest(Observation(actor="agent", step=1, tool_results=[ToolResult(tool_call_id="c1", function_name="repo.read_file", ok=True, content=page)]))
    assert pol.messages[-1]["content"] == page  # whole: nothing elided
    old = lh.CappedToolPolicy("agent", _Nil(), max_tool_output=8000)  # run 2's cap cut the middle out
    old._ingest(Observation(actor="agent", step=1, tool_results=[ToolResult(tool_call_id="c1", function_name="repo.read_file", ok=True, content=page)]))
    assert "characters truncated" in old.messages[-1]["content"]


def test_d8_test_facts_come_from_the_tests_own_source() -> None:
    f = hidden_test_facts(REC["tests_b"], "test_semver.py::test_should_check_valid[1.2.3]")
    assert f["calls"] == ["cmd_check"] and f["asserts"] == ["cmd_check(version) is None"]
    assert f["params"] == ["'version' = '1.2.3'", "'version' = '0.0.1-alpha'"]
    g = hidden_test_facts(REC["tests_b"], "test_semver.py::test_should_raise_on_invalid")
    assert g["raises"] == ["ValueError from cmd_check('1.2')"]
    assert hidden_test_facts(REC["tests_b"], "test_semver.py::test_missing") is None


def test_d8_only_hidden_test_ids_reach_the_author() -> None:
    sent = ["test_semver.py::test_should_check_valid[1.2.3]", {"test": "test_semver.py::test_should_raise_on_invalid"},
            "The check command should print a capitalised True",  # run 2's wrong auditor guess
            "test_semver.py::test_not_in_this_window"]
    findings, refused = findings_from_tests(REC, sent)
    assert len(findings) == 2 and len(refused) == 2
    assert all(f.startswith("spec_fact (change m03)") for f in findings)
    assert "cmd_check(version) is None" in findings[0] and "'0.0.1-alpha'" in findings[0]
    assert "ValueError from cmd_check('1.2')" in findings[1]
    assert not any("test_should" in f or "test_semver.py" in f for f in findings)  # no test name, no file path
    assert "capitalised True" in refused[0] and "never sent to the author" in refused[0]
    assert lh.author_findings(REC, sent) == (findings, refused)  # the run script's send-back path is the same function
