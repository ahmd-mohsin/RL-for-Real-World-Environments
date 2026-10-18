"""P2 fairness lint (``factory.fairness``, swarm-pipeline/00 §5.8): source-text and source-structure reads, private-name
checks, unstated free-text formats (a stated one and an echoed input are fine), a gating check with no requirement, and
the executed ten-competent-people rule."""

from __future__ import annotations

from crucible.factory.fairness import is_free_text, lint_checks, ten_competent_people

TESTS = {"tests/test_m.py": '''
import inspect
import pytest
from lib import m


def _src(f):
    return inspect.getsource(f)


def test_reads_source():
    assert "cache_size" in inspect.getsource(m.f)


def test_reads_source_via_helper():
    assert "total" in _src(m.f)


def test_code_object():
    assert m.f.__code__.co_varnames[0] == "x"


def test_private_name():
    assert hasattr(m.Cache, "_store")


def test_private_in_dir():
    assert "_hidden" in dir(m)


def test_unstated_message():
    with pytest.raises(ValueError) as e:
        m.f(-1)
    assert str(e.value) == "the value must not be negative here"


def test_unstated_match():
    with pytest.raises(ValueError, match="value must not be negative"):
        m.f(-1)


def test_stated_message():
    with pytest.raises(ValueError) as e:
        m.g(-1)
    assert str(e.value) == "Size must be a positive integer"


def test_echoed_input(capsys):
    m.echo("Pause to standard error")
    assert capsys.readouterr().err == "Pause to standard error\\n"


def test_behaviour():
    assert m.f(2) == 4 and m.describe(3) == "3 items"


class TestNested:
    def test_file_read(self):
        assert "def f" in open(m.__file__).read()
'''}
PROMPT = "Release request. g() raises ValueError with the message 'Size must be a positive integer' for bad sizes."


def checks(*names: str, reqs=("R1",)) -> list[dict]:
    return [{"id": f"c_{n}", "requirement_ids": list(reqs), "gating": True, "spec": {"tests": [f"tests/test_m.py::{n}"]}} for n in names]


def test_static_rules() -> None:
    names = ["test_reads_source", "test_reads_source_via_helper", "test_code_object", "test_private_name", "test_private_in_dir", "test_unstated_message",
             "test_unstated_match", "test_stated_message", "test_echoed_input", "test_behaviour", "TestNested::test_file_read"]
    rep = lint_checks(checks(*names), TESTS, PROMPT)
    kinds = {f.check: f.kind for f in rep.findings}
    assert kinds == {"c_test_reads_source": "source_text", "c_test_reads_source_via_helper": "source_text", "c_test_code_object": "source_structure",
                     "c_test_private_name": "private_name", "c_test_private_in_dir": "private_name", "c_test_unstated_message": "unstated_format",
                     "c_test_unstated_match": "unstated_format", "c_TestNested::test_file_read": "source_text"}
    assert sorted(f.check for f in rep.spec_defects()) == ["c_test_unstated_match", "c_test_unstated_message"]
    assert all(f.owner == "spec" for f in rep.spec_defects()) and "c_test_unstated_message" not in rep.refused_checks()
    assert rep.tests_seen == len(names) and rep.tests_unresolved == [] and not rep.ok
    miss = lint_checks([{"id": "c_x", "requirement_ids": [], "gating": True, "spec": {"tests": ["tests/test_m.py::test_gone", "tests/other.py::t"]}}],
                       TESTS, PROMPT)
    assert [f.kind for f in miss.findings] == ["no_requirement"] and len(miss.tests_unresolved) == 2
    assert is_free_text("the value must not be negative") and not is_free_text("3 items") and not is_free_text("a_b_c_d_e_f_g_h")


def test_ten_competent_people() -> None:
    ct = {"c1": ["t1", "t2"], "c2": ["t3"], "c3": ["t4"]}
    pos = {"oracle": {"t1": "passed", "t2": "passed", "t3": "passed", "t4": "passed"},
           "alt_a": {"t1": "passed", "t2": "failed", "t3": "passed", "t4": "passed"}, "alt_b": None}
    ff = ten_competent_people(ct, pos, gating={"c1": True, "c2": True, "c3": False})
    assert [(f.check, f.test) for f in ff] == [("c1", "t2")] and "alt_a" in ff[0].evidence
