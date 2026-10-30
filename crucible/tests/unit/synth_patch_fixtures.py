"""A tiny repository task for the ``synth.patch`` tests: two interacting bugs in ``calc.py``, hidden tests graded by an
in-process runner (no Docker). The runner applies the lane's rule: protected writes are dropped before the tests run."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from crucible.synth.patch import FORBIDDEN_ROLES, ExecResult, FixUnit, FuncRunner, Milestone, PatchEnv, PatchSpec, Requirement, RevertMutant, default_roles

BUGGY = '''"""A calculator."""


def add(a, b):
    return a - b


def ident(x):
    return x


# helpers below


def mul(a, b):
    return a + b
'''

FIXED = '''"""A calculator."""


def add(a, b):
    return a + b


def ident(x):
    return x


# helpers below (fixed)


def mul(a, b):
    return a * b
'''

VISIBLE_TESTS = '''from calc import ident


def test_ident():
    assert ident(4) == 4
'''

HIDDEN_TESTS = '''from calc import add, ident, mul


def test_add():
    assert add(2, 3) == 5


def test_mul():
    assert mul(2, 3) == 6


def test_ident():
    assert ident(4) == 4
'''

TEST = "tests/test_calc.py"


def make_repo(root: Path, *, changelog: str = "# Changes\n\n- initial release\n", visible: str = VISIBLE_TESTS) -> Path:
    base = root / "base"
    (base / "tests").mkdir(parents=True)
    (base / "calc.py").write_text(BUGGY)
    (base / "util.py").write_text("def two():\n    return 2\n")
    (base / TEST).write_text(visible)
    (base / "CHANGELOG.md").write_text(changelog)
    (base / "pyproject.toml").write_text("[project]\nname = 'calc'\n")
    (base / "testdata").mkdir()
    (base / "testdata" / "golden.txt").write_text("5\n")
    return base


def spec_for(base: Path, **kw: Any) -> PatchSpec:
    ms = kw.pop("milestones", None) or [Milestone(id="m01", tests=[f"{TEST}::test_add"]), Milestone(id="m02", tests=[f"{TEST}::test_mul"])]
    checks = kw.pop("checks", None) or {**{f"c_{m.id}": list(m.tests) for m in ms}, "c_regress": [f"{TEST}::test_ident"]}
    add_only = FIXED.replace("return a * b", "return a + b").replace("# helpers below (fixed)", "# helpers below")
    mul_only = FIXED.replace("return a + b", "return a - b", 1)
    data: dict[str, Any] = {
        "task_id": "fx-calc", "lang": "python", "base_dir": str(base), "comment": "#",
        "instruction": "Adding two numbers gives the wrong answer, and so does multiplying them. 2 plus 3 should be 5; 2 times 3 should be 6.",
        "requirements": [Requirement(id="R1", text="2 plus 3 is 5", milestone="m01"), Requirement(id="R2", text="2 times 3 is 6", milestone="m02"),
                         Requirement(id="r_regress", text="everything else keeps working")],
        "milestones": ms, "checks": checks, "guards": ["c_regress"], "oracle_files": {"calc.py": FIXED},
        "mutants": [RevertMutant(id="revert_m01", milestone="m01", files={"calc.py": mul_only.replace("# helpers below", "# helpers below (fixed)")},
                                 observed={"m01": False, "m02": True}),
                    RevertMutant(id="revert_m02", milestone="m02", files={"calc.py": add_only}, observed={"m01": True, "m02": False})],
        "hidden_tests": {TEST: HIDDEN_TESTS},
        "units": [FixUnit(sha="a" * 40, subject="fix: add returns the difference (#123)"), FixUnit(sha="b" * 40, subject="fix: mul adds instead of multiplying")],
    }
    data.update(kw)
    return PatchSpec(**data)


def tree_of(base: Path, files: Mapping[str, str | None], roles: Callable[[str], str] = default_roles) -> dict[str, str]:
    out = {p.relative_to(base).as_posix(): p.read_text() for p in base.rglob("*") if p.is_file()}
    for p, t in files.items():
        if roles(p) in FORBIDDEN_ROLES:  # the lane restores protected paths before its tests run
            continue
        if t is None:
            out.pop(p, None)
        else:
            out[p] = t
    return out


def run_tests(tree: Mapping[str, str], tests: str) -> dict[str, str]:
    ns: dict[str, Any] = {}
    try:
        exec(compile(tree.get("calc.py", ""), "calc.py", "exec"), ns)  # noqa: S102 - the fixture's own source
    except Exception:  # noqa: BLE001
        return {}
    mod: dict[str, Any] = {}
    import types

    calc = types.ModuleType("calc")
    calc.__dict__.update(ns)
    import sys

    sys.modules["calc"] = calc
    try:
        exec(compile(tests, "t.py", "exec"), mod)  # noqa: S102
    except Exception:  # noqa: BLE001
        return {}
    finally:
        sys.modules.pop("calc", None)
    out = {}
    for name, f in sorted(mod.items()):
        if name.startswith("test_") and callable(f):
            try:
                f()
                out[f"{TEST}::{name}"] = "passed"
            except Exception:  # noqa: BLE001
                out[f"{TEST}::{name}"] = "failed"
    return out


def fixture_runner(env: PatchEnv, *, honour_conftest: bool = False, flaky: bool = False) -> FuncRunner:
    calls = {"n": 0}

    def fn(files: dict[str, str | None], mode: str) -> ExecResult:
        tree = tree_of(env.base_dir, files, env.roles)
        if mode == "visible":
            st = run_tests(tree, tree.get(TEST, ""))
            return ExecResult(checks={}, statuses=st)
        st = run_tests(tree, HIDDEN_TESTS)
        if honour_conftest and "conftest.py" in tree:
            st = dict.fromkeys(st, "passed")
        calls["n"] += 1
        if flaky and calls["n"] % 2 == 0:
            st = {**st, f"{TEST}::test_ident": "failed"}
        checks = {cid: bool(ts) and all(st.get(t) == "passed" for t in ts) for cid, ts in env.spec.checks.items()}
        return ExecResult(checks=checks, statuses=st)

    return FuncRunner(fn)
