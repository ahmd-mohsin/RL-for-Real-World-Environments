"""M17 report 11, defect found on the first run: B's test file imports new public names the prompt never stated
(python-semver: ``from semver import cmd_bump, cmd_check, cmd_compare``), so a solution missing one name fails collection
of the whole file and every milestone in it reads 0. ``required_names`` finds those names; the author must state each
(``unstated_names`` is a finding that sends the draft back)."""

from __future__ import annotations

from pathlib import Path

from crucible.factory.lanes.windows import required_names, unstated_names

A_PKG = {"pkg/__init__.py": "from .core import parse, VERSION\n", "pkg/core.py": "import re\nVERSION = '1'\n\n\ndef parse(s):\n    return s\n",
         "pkg/sub/__init__.py": "", "star/__init__.py": "from .x import *\n", "star/x.py": "def a():\n    pass\n", "flat.py": "def old():\n    pass\n"}
TESTS = {"tests/test_core.py": ("import pytest\nimport flat as fl\nfrom pkg import parse, bump, sub\nfrom pkg.core import VERSION, re, Spec\n"
                                "from star import anything\nfrom os import path\n\n\ndef test_x():\n    from pkg.core import _helper, check\n"
                                "    assert fl.old() is None and fl.new_one() and fl._priv\n")}


def _tree(tmp: Path) -> Path:
    for p, t in A_PKG.items():
        (tmp / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp / p).write_text(t)
    return tmp


def test_required_names_finds_what_the_hidden_tests_import_that_a_lacks(tmp_path):
    req = required_names(TESTS, _tree(tmp_path))
    # module level: bump (pkg), Spec (pkg.core); parse/VERSION/re are bound at A, sub is a subpackage, star re-exports (unknown),
    # os is not the project's
    assert req["module_level"] == ["Spec", "bump"]
    assert req["in_function"] == ["check"]
    assert req["attribute"] == ["new_one"]
    assert req["private"] == ["_helper", "_priv"]


def test_dynamic_modules_and_dunders_are_unknown_not_required(tmp_path):
    (tmp_path / "hook").mkdir()
    (tmp_path / "hook" / "__init__.py").write_text("from . import _build\n")
    (tmp_path / "hook" / "_build.py").write_text("import sys\nsys.meta_path.append(object())\n")
    (tmp_path / "lazy.py").write_text("def __getattr__(name):\n    return name\n")
    (tmp_path / "plain.py").write_text("x = 1\n")
    t = "import hook\nimport lazy\nimport plain\nfrom lazy import thing\n\n\ndef test():\n    hook.curry, lazy.anything, plain.__file__, plain.y\n"
    assert required_names({"t.py": t}, tmp_path) == {"module_level": [], "in_function": [], "attribute": ["y"], "private": []}


def test_required_names_with_src_layout_and_syntax_errors(tmp_path):
    (tmp_path / "src" / "lib").mkdir(parents=True)
    (tmp_path / "src" / "lib" / "__init__.py").write_text("def a():\n    pass\n")
    req = required_names({"t.py": "from lib import a, b\n", "bad.py": "def (:\n"}, tmp_path)
    assert req["module_level"] == ["b"]


def test_unstated_names_requires_each_name_verbatim():
    payload = {"changes": [{"id": "m1", "must_name": ["cmd_bump"]}, {"id": "m2"}], "must_name_any": ["Spec"]}
    text = "The CLI gains cmd_bump() for bumping. A Specification class is added."
    out = unstated_names(payload, text)
    assert len(out) == 1 and "'Spec'" in out[0]
    assert unstated_names(payload, "cmd_bump and Spec exist.") == []
    assert unstated_names(payload, "pkg.cmd_bump_all and Spec") and not unstated_names(payload, "pkg.cmd_bump() and pkg.Spec")
    assert not unstated_names({"changes": []}, "")
