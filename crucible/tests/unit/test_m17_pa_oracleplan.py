"""Phase A, S4.5: the oracle against its plan, executed (``factory/oracleplan.py``; its wiring in ``generators/swarm.py``).

Synthetic repositories on the host interpreter, plus P5b's real cachetools swarm window when the scratch store is present
(the seed: the oracle deprecated ``MRUCache`` / ``mru_cache`` where the plan removes them)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible.factory import oracleplan as op
from crucible.factory.generators import swarm as sw

A_INIT = '''__all__ = ("LRU", "MRU", "keep")


class LRU:
    def __init__(self, n=1):
        self.n = n


class MRU:
    def __init__(self, n=1):
        self.n = n


def keep(x, y=0):
    return x + y


def legacy(x):
    return x
'''
A_FUNC = '''from . import MRU


def mru_cache(n=1):
    return MRU(n)


def lru_cache(n=1):
    return n
'''
TEST_MRU = '''from pkg import MRU


def test_mru():
    assert MRU(2).n == 2
'''
TEST_FUNC = '''import pkg.func


class Base:
    DECORATOR = None

    def test_decorator(self):
        assert self.DECORATOR(3) is not None


class TestMRU(Base):
    DECORATOR = staticmethod(pkg.func.mru_cache)


class TestLRU(Base):
    DECORATOR = staticmethod(pkg.func.lru_cache)


def test_keep():
    assert pkg.keep(1, 2) == 3
'''

DEPRECATED_INIT = A_INIT.replace('__all__ = ("LRU", "MRU", "keep")', 'import warnings\n\n__all__ = ("LRU", "keep")').replace(
    "class MRU:\n    def __init__(self, n=1):\n", "class MRU:\n    def __init__(self, n=1):\n        warnings.warn('MRU is deprecated', DeprecationWarning, stacklevel=2)\n")
DEPRECATED_FUNC = A_FUNC.replace("def mru_cache(n=1):\n", "def mru_cache(n=1):\n    import warnings\n\n    warnings.warn('deprecated', DeprecationWarning)\n")
REMOVED_INIT = A_INIT.replace('("LRU", "MRU", "keep")', '("LRU", "keep")').replace("class MRU:\n    def __init__(self, n=1):\n        self.n = n\n\n\n", "")
REMOVED_FUNC = "def lru_cache(n=1):\n    return n\n"

REMOVE_PLAN = {"milestones": [
    {"id": "m1", "title": "Remove MRU and mru_cache", "requests": ["R1"], "requirement": "Remove the MRU class from pkg and the mru_cache decorator from pkg.func.",
     "observable": "from pkg import MRU raises ImportError; from pkg.func import mru_cache raises ImportError.", "public_names": ["MRU", "mru_cache"]}]}
DEPRECATE_PLAN = {"milestones": [
    {"id": "m1", "title": "Deprecate legacy", "requests": ["R2"], "requirement": "Deprecate legacy(): it keeps working and warns.",
     "observable": "calling legacy(1) emits a DeprecationWarning and returns 1", "public_names": ["legacy"]}]}
ADD_PLAN = {"milestones": [
    {"id": "m1", "title": "Add totient", "requests": ["R3"], "requirement": "Add pkg.totient(n), Euler's totient.", "observable": "totient(9) == 6",
     "public_names": ["totient"]},
    {"id": "m2", "title": "keep gains a scale option", "requests": ["R4"], "requirement": "keep() accepts a new keyword parameter scale.",
     "observable": "keep(1, 2, scale=2) == 6", "public_names": ["keep", "scale"]}]}
OPTION_REMOVAL_PLAN = {"milestones": [
    {"id": "m1", "title": "Drop y", "requests": ["R5"], "requirement": "Remove the y parameter from keep.", "observable": "keep(1, y=2) raises TypeError",
     "public_names": ["keep", "y"]}]}


def make_a(root: Path) -> Path:
    a = root / "a"
    (a / "pkg").mkdir(parents=True)
    (a / "tests").mkdir()
    (a / "pkg" / "__init__.py").write_text(A_INIT)
    (a / "pkg" / "func.py").write_text(A_FUNC)
    (a / "tests" / "test_mru.py").write_text(TEST_MRU)
    (a / "tests" / "test_func.py").write_text(TEST_FUNC)
    (a / "pyproject.toml").write_text('[project]\nname = "pkg"\nversion = "1.0"\n')
    return a


REGRESSION = ["tests/test_mru.py::test_mru", "tests/test_func.py::TestMRU::test_decorator", "tests/test_func.py::TestLRU::test_decorator",
              "tests/test_func.py::test_keep"]


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def verdicts(res: dict[str, Any]) -> dict[str, str]:
    return {r["name"]: r["verdict"] for r in res["probes"]}


# ---- removal: the cachetools seed, synthetic -----------------------------------------------------------------------------------
def test_a_deprecated_oracle_fails_a_planned_removal_and_a_removing_one_passes(tmp_path: Path) -> None:
    a = make_a(tmp_path)
    bad = run(op.check(REMOVE_PLAN, a, {"pkg/__init__.py": DEPRECATED_INIT, "pkg/func.py": DEPRECATED_FUNC}, runner=op.host_runner(a),
                       regression_ids=REGRESSION, regression_on_oracle={t: "passed" for t in REGRESSION}))
    assert not bad["ok"] and bad["owner"] == "oracle_builder"
    assert verdicts(bad) == {"MRU": "deprecated_not_removed", "mru_cache": "deprecated_not_removed"}
    kinds = {(d["kind"], d["name"]) for d in bad["defects"]}
    assert ("contradicted_regression", "MRU") in kinds and ("contradicted_regression", "mru_cache") in kinds
    assert any("deprecated, not removed" in f for f in bad["findings"])
    # the regression tests that use a removed name contradict the plan: they leave the regression set (TestLRU and keep stay)
    assert bad["drop_from_regression"] == ["tests/test_func.py::TestMRU::test_decorator", "tests/test_mru.py::test_mru"]
    good = run(op.check(REMOVE_PLAN, a, {"pkg/__init__.py": REMOVED_INIT, "pkg/func.py": REMOVED_FUNC}, runner=op.host_runner(a), regression_ids=REGRESSION,
                        regression_on_oracle={"tests/test_mru.py::test_mru": "error", "tests/test_func.py::TestMRU::test_decorator": "failed"}))
    assert good["ok"] and verdicts(good) == {"MRU": "ok", "mru_cache": "ok"} and good["drop_from_regression"] == bad["drop_from_regression"]


def test_a_removal_that_only_drops_the_name_from_all_is_not_a_removal(tmp_path: Path) -> None:
    a = make_a(tmp_path)
    res = run(op.check(REMOVE_PLAN, a, {"pkg/__init__.py": A_INIT.replace('"MRU", ', "")}, runner=op.host_runner(a)))
    assert verdicts(res) == {"MRU": "not_removed", "mru_cache": "not_removed"} and not res["ok"]


# ---- deprecation vs removal ------------------------------------------------------------------------------------------------------
def test_deprecation_is_told_apart_from_removal(tmp_path: Path) -> None:
    a = make_a(tmp_path)
    warns = A_INIT.replace("def legacy(x):\n", "def legacy(x):\n    import warnings\n\n    warnings.warn('use keep', DeprecationWarning)\n")
    assert verdicts(run(op.check(DEPRECATE_PLAN, a, {"pkg/__init__.py": warns}, runner=op.host_runner(a)))) == {"legacy": "ok"}
    gone = A_INIT.replace("\n\ndef legacy(x):\n    return x\n", "\n")
    res = run(op.check(DEPRECATE_PLAN, a, {"pkg/__init__.py": gone}, runner=op.host_runner(a)))
    assert verdicts(res) == {"legacy": "removed_not_deprecated"} and res["defects"][0]["owner"] == "oracle_builder"
    silent = run(op.check(DEPRECATE_PLAN, a, {"pkg/__init__.py": A_INIT + "\n"}, runner=op.host_runner(a)))
    assert verdicts(silent) == {"legacy": "not_deprecated"}


# ---- additions and options -------------------------------------------------------------------------------------------------------
def test_additions_and_new_options_need_the_oracle_to_have_them(tmp_path: Path) -> None:
    a = make_a(tmp_path)
    full = A_INIT.replace("def keep(x, y=0):\n    return x + y\n", "def keep(x, y=0, scale=1):\n    return (x + y) * scale\n") + "\n\ndef totient(n):\n    return n\n"
    res = run(op.check(ADD_PLAN, a, {"pkg/__init__.py": full}, runner=op.host_runner(a)))
    assert res["ok"] and verdicts(res) == {"totient": "ok", "keep": "exists_at_start", "scale": "ok"}
    res = run(op.check(ADD_PLAN, a, {"pkg/__init__.py": A_INIT + "\n"}, runner=op.host_runner(a)))
    assert verdicts(res) == {"totient": "not_added", "keep": "exists_at_start", "scale": "option_missing"}
    assert {d["kind"] for d in res["defects"]} == {"not_added", "option_missing"}


def test_an_option_removal_is_checked_on_the_signature(tmp_path: Path) -> None:
    a = make_a(tmp_path)
    assert [(p.kind, p.name) for p in op.plan_probes(OPTION_REMOVAL_PLAN)] == [("addition", "keep"), ("removal", "y")]
    kept = run(op.check(OPTION_REMOVAL_PLAN, a, {"pkg/__init__.py": A_INIT + "\n"}, runner=op.host_runner(a)))
    assert verdicts(kept)["y"] == "option_not_removed" and not kept["ok"]
    done = A_INIT.replace("def keep(x, y=0):\n    return x + y\n", "def keep(x):\n    return x\n")
    assert verdicts(run(op.check(OPTION_REMOVAL_PLAN, a, {"pkg/__init__.py": done}, runner=op.host_runner(a))))["y"] == "ok"


def test_removal_targets_are_the_objects_of_the_verb() -> None:
    plan = {"milestones": [{"id": "m1", "title": "t", "requirement": "Add frobnicate(), which removes the need for legacy() calls in user code.",
                            "observable": "", "public_names": ["frobnicate", "legacy"]},
                           {"id": "m2", "title": "t", "requirement": "Remove the y parameter from keep; X is removed too.", "observable": "",
                            "public_names": ["keep", "y", "X"]}]}
    kinds = {(p.milestone, p.name): p.kind for p in op.plan_probes(plan)}
    assert kinds == {("m1", "frobnicate"): "addition", ("m1", "legacy"): "removal", ("m2", "keep"): "addition", ("m2", "y"): "removal",
                     ("m2", "X"): "removal"}


def test_a_removal_of_a_name_the_start_lacks_goes_to_the_planner(tmp_path: Path) -> None:
    a = make_a(tmp_path)
    plan = {"milestones": [{"id": "m1", "title": "Remove Ghost", "requests": ["R1"], "requirement": "Remove the Ghost class.", "observable": "",
                            "public_names": ["Ghost"]}]}
    res = run(op.check(plan, a, {"pkg/__init__.py": A_INIT + "\n"}, runner=op.host_runner(a)))
    assert res["owner"] == "planner" and res["defects"][0]["kind"] == "removal_of_absent_name"


# ---- the swarm wiring ------------------------------------------------------------------------------------------------------------
def test_the_swarm_machine_runs_s4_5_between_the_oracle_check_and_the_verifier() -> None:
    from crucible.factory.machine import validate_machine

    m = sw.swarm_machine()
    assert validate_machine(m, models=sw.ROLE_MODELS) == []
    t = {(x["from"], x["to"]) for x in m["transitions"]}
    assert {("oracle_check", "oracle_plan"), ("oracle_plan", "verifier"), ("oracle_plan", "oracle"), ("oracle_plan", "plan")} <= t
    assert ("oracle_check", "verifier") not in t
    assert "oracle_plan" in next(s for s in m["states"] if s["id"] == "verifier")["isolated_from"]


def test_s4_5_sends_a_defect_back_to_the_oracle_builder_and_passes_a_clean_oracle(tmp_path: Path) -> None:
    from crucible.factory.generators.base import Funnel

    results = [{"ok": False, "owner": "oracle_builder", "findings": ["m1 (R1) plans the removal of MRU ..."], "defects": [{}], "probes": []},
               {"ok": True, "findings": [], "probes": [{}, {}], "drop_from_regression": ["t::x"]},
               {"ok": False, "owner": "planner", "findings": ["m1 plans the removal of Ghost ..."], "defects": [{}], "probes": []}]

    async def fn(shas: list[str], plan: dict) -> dict:
        return results.pop(0)

    svc = sw.SwarmServices(cand=sw.SwarmCandidate("c", "pkg", {"a": "x"}, []), clone=tmp_path, work=tmp_path, funnel=Funnel(tmp_path / "f.jsonl"))
    svc.oracle_plan_fn = fn

    def ctx(entry: int) -> Any:
        return SimpleNamespace(services=svc, inputs={"check": {"shas": ["s1"]}, "plan": REMOVE_PLAN}, entry=entry)

    v = run(sw.oracle_plan_state(ctx(1))).outputs["verdict"]
    assert not v["ok"] and v["owner"] == "oracle_builder" and v["retry"] and "removal of MRU" in v["findings"][0]
    v = run(sw.oracle_plan_state(ctx(2))).outputs["verdict"]
    assert v["ok"] and v["drop_from_regression"] == ["t::x"]
    v = run(sw.oracle_plan_state(ctx(1))).outputs["verdict"]
    assert not v["ok"] and v["owner"] == "planner" and not v["retry"]


def test_regressions_do_not_count_tests_that_use_a_removed_name(tmp_path: Path) -> None:
    a = make_a(tmp_path)
    contra = sw.SwarmRunner.contradicting(None, a, REMOVE_PLAN, REGRESSION)  # type: ignore[arg-type]
    assert contra == {"tests/test_mru.py::test_mru": ["MRU"], "tests/test_func.py::TestMRU::test_decorator": ["mru_cache"]}


# ---- the real window: P5b's cachetools swarm-built task --------------------------------------------------------------------------
SCRATCH = Path(os.environ.get("CRUCIBLE_SCRATCH", str(Path(__file__).resolve().parents[4])))
CACHETOOLS = SCRATCH / "p5b/run/swarm/store/train/tasks/sw-cachetools-aadb387d"


@pytest.mark.skipif(not (CACHETOOLS / "window.json").exists(), reason="P5b's swarm store is not present")
def test_the_real_cachetools_swarm_window_is_caught(tmp_path: Path) -> None:
    w = json.loads((CACHETOOLS / "window.json").read_text())
    res = run(op.check(w["swarm_window"]["plan"], CACHETOOLS / "base", w["oracle_files"], runner=op.host_runner(CACHETOOLS / "base"),
                       regression_ids=w["p2p"]))
    v = verdicts(res)
    assert v["MRUCache"] == "deprecated_not_removed" and v["mru_cache"] == "deprecated_not_removed" and v["condition"] == "ok"
    assert not res["ok"] and res["owner"] == "oracle_builder"
    # the 23 MRU tests P5b kept as regression checks contradict R1
    assert len(res["drop_from_regression"]) == 23 and all("MRU" in t or "mru" in t for t in res["drop_from_regression"])


# ---- names that are not module-level: methods, dotted names, parameters, run-time attributes (P5b's real plans) -------------
CLS_A = A_INIT + '''

class Version:
    def __init__(self, major=0):
        self.major = major

    def shift(self, n=1):
        return Version(self.major + n)


class Doc:
    def render(self):
        self.parts = []
        return ""


def split(src, sep=None, maxsplit=None):
    return [src]
'''


def test_methods_dotted_names_parameters_and_runtime_attributes(tmp_path: Path) -> None:
    a = make_a(tmp_path)
    (a / "pkg" / "__init__.py").write_text(CLS_A)
    plan = {"milestones": [
        {"id": "m1", "title": "to_tuple", "requests": ["R1"], "requirement": "Version.to_tuple() returns (major,); Version supports v[0].",
         "observable": "Version(1).to_tuple() == (1,)", "public_names": ["to_tuple", "__getitem__"]},
        {"id": "m2", "title": "shift check", "requests": ["R2"], "requirement": "I want Version.shift to accept a check parameter.",
         "observable": "Version(1).shift(1, check=False)", "public_names": ["Version.shift", "check"]},
        {"id": "m3", "title": "split", "requests": ["R3"], "requirement": "split with maxsplit=0 returns the values.", "observable": "split([1], maxsplit=0)",
         "public_names": ["split", "maxsplit"]},
        {"id": "m4", "title": "toc", "requests": ["R4"], "requirement": "Each Doc.render fills parts and the new toc_tokens list.", "observable": "doc.toc_tokens",
         "public_names": ["parts", "toc_tokens"]}]}
    good = CLS_A.replace("    def shift(self, n=1):\n", "    def to_tuple(self):\n        return (self.major,)\n\n    def __getitem__(self, i):\n"
                         "        return self.to_tuple()[i]\n\n    def shift(self, n=1, check=True):\n").replace(
        "        self.parts = []\n", "        self.parts = []\n        self.toc_tokens = []\n")
    res = run(op.check(plan, a, {"pkg/__init__.py": good}, runner=op.host_runner(a)))
    assert res["ok"], res["findings"]
    assert verdicts(res) == {"to_tuple": "ok", "__getitem__": "ok", "Version.shift": "exists_at_start", "check": "ok", "split": "exists_at_start",
                             "maxsplit": "exists_at_start", "parts": "exists_at_start", "toc_tokens": "unexercised"}
    bad = run(op.check(plan, a, {"pkg/__init__.py": CLS_A + "\n"}, runner=op.host_runner(a)))
    v = verdicts(bad)
    assert v["to_tuple"] == v["__getitem__"] == "not_added" and v["check"] == "option_missing" and v["toc_tokens"] == "not_added"
    assert v["maxsplit"] == v["parts"] == "exists_at_start"
