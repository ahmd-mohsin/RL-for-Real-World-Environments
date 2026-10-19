"""P5b swarm-built windows (``factory/generators/swarm.py``; swarm-pipeline/00 §2, §3, §4, §15 P5), with scripted models ($0) and
no Docker:

- requests from a real changelog (items, filters, issue references, lease on every record);
- the plan guard (S2), the oracle guard (S4: one commit per milestone, in order), the hidden-test guards (S5) and the
  outcome findings that never carry a value the oracle produced (V2);
- the synthetic window: tree k = oracle commit k with test and protected paths back at A plus the hidden files of
  milestones 1..k;
- the machine end to end on a local kernel: lease -> planner (a plan with no dependency is sent back, the second is
  accepted) -> oracle builder session (two commits) -> oracle check (fetched into the derived clone) -> verifier writer
  session (its private clone holds no oracle commit) -> reviewer (blocks once, then advisory: V6) -> build (the real
  ``synthesize``) -> prompt / gates / second / hack (scripted) -> accept.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from crucible.actors.llm import LLMResponse
from crucible.factory.generators import swarm as sw
from crucible.factory.generators.base import Funnel
from crucible.factory.handoff import TaskRepo, accept_state, lease_state
from crucible.factory.machine import MachineRun, validate_machine
from crucible.factory.sessions import SessionPool

GOOD_PLAN = {"milestones": [
    {"id": "m1", "title": "mul", "requests": ["R1"], "requirement": "calc.mul(a, b) returns the product.", "observable": "mul(2, 3) == 6",
     "public_names": ["mul"], "requires": []},
    {"id": "m2", "title": "power via mul", "requests": ["R2"], "requirement": "calc.power(a, n) multiplies a by itself n times.",
     "observable": "power(2, 3) == 8", "public_names": ["power"], "requires": ["m1"], "why_requires": "power is repeated mul"}],
    "expected_causes": ["forgets mul"], "verifier_plan": "values"}


def sh(cwd: Path, cmd: str) -> str:
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, check=True).stdout


def make_repo(root: Path) -> Path:
    src = root / "calc"
    (src / "tests").mkdir(parents=True)
    (src / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (src / "tests" / "test_calc.py").write_text("from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    (src / "CHANGES.rst").write_text("Changes\n=======\n")
    g = "git -c user.email=a@b -c user.name=a"
    sh(src, f"git init -q -b main && git add -A && {g} commit -qm 'release A' && git tag A")
    (src / "CHANGES.rst").write_text("Changes\n=======\n\nVersion 1.1\n-----------\n\nFeatures\n--------\n\n- Added ``mul()`` to multiply two numbers (#12).\n"
                                     "- Added ``power()``, built on multiplication.\n  It accepts a non-negative exponent.\n- Fixed a typo in the docs.\n"
                                     "- Improved CI caching.\n")
    sh(src, f"{g} commit -qam 'changelog' && git tag B")
    return src


# ---- S1 -------------------------------------------------------------------------------------------------------------------------
def test_requests_come_from_the_changelog_with_issue_text_and_the_lease(tmp_path: Path) -> None:
    src = make_repo(tmp_path)
    a, b = sh(src, "git rev-parse A").strip(), sh(src, "git rev-parse B").strip()
    items, source = sw.request_items(src, a, b)
    assert source == "changelog" and items == ["Added ``mul()`` to multiply two numbers (#12).",
                                                "Added ``power()``, built on multiplication. It accepts a non-negative exponent."]

    class GH:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def fetch(self, url: str) -> tuple[int, dict]:
            self.calls.append(url)
            return 200, {"title": "Please add mul", "body": "I need products.", "html_url": "https://x/12"}

    class Lease:
        def take(self, rec: Any) -> bool:
            rec.lease = {"ok": True}
            return True

    gh = GH()
    recs = sw.request_records("calc", {"id": "w1", "a": a, "b": b, "b_date": "2020-01-01"}, src, gh=gh, owner_repo="o/calc", lease=Lease())
    assert [r.type for r in recs] == ["request", "request"] and gh.calls == ["https://api.github.com/repos/o/calc/issues/12"]
    assert "Please add mul" in recs[0].text and recs[0].evidence["rid"] == "R1" and recs[0].lease == {"ok": True}
    assert recs[1].evidence["refs"] == [] and recs[1].url == ""


def test_split_requests_and_filters() -> None:
    lines = ["Version 2.0", "=====", "Released: 2020-01-01", "Bug Fixes", "---------", "* Fixed ``parse()`` on empty input", "  when strict is set.",
             "- Bump version", "- Update dependency foo to v2 (#3)", "- ``semver.parse``", "[x.y]: https://docs/x"]
    items = sw.split_requests(lines)
    assert items[0] == "Fixed ``parse()`` on empty input when strict is set."
    assert [i for i in items if sw.is_request(i)] == ["Fixed ``parse()`` on empty input when strict is set."]


# ---- S2 / S4 / S5 guards -----------------------------------------------------------------------------------------------------------
def test_plan_problems() -> None:
    assert sw.plan_problems(GOOD_PLAN, request_ids=["R1", "R2"]) == []
    flat = json.loads(json.dumps(GOOD_PLAN))
    flat["milestones"][1]["requires"] = []
    assert any("must be dependent" in p for p in sw.plan_problems(flat, request_ids=["R1", "R2"]))
    bad = json.loads(json.dumps(GOOD_PLAN))
    bad["milestones"][0]["requires"] = ["m2"]
    bad["milestones"][1].update(id="m3", public_names=["_private"], requests=["R9"])
    probs = sw.plan_problems(bad, request_ids=["R1", "R2"], max_milestones=1)
    assert any("need 2 to 1" in p for p in probs) and any("not an earlier milestone" in p for p in probs)
    assert any("ids are m1, m2" in p for p in probs) and any("not public Python identifiers" in p for p in probs) and any("unknown request ids" in p for p in probs)
    assert sw.plan_problems({"x": 1}, request_ids=[]) == ["the plan has no milestones list"]
    dunder = json.loads(json.dumps(GOOD_PLAN))
    dunder["milestones"][0]["public_names"] = ["VersionInfo.__getitem__"]
    assert sw.plan_problems(dunder, request_ids=["R1", "R2"]) == [  # a dunder is public; the dependency must name it
        "m2: why_requires must name what the required milestone adds (['__getitem__']) and how this one uses it"]
    dunder["milestones"][1]["why_requires"] = "power reads the parts through __getitem__"
    assert sw.plan_problems(dunder, request_ids=["R1", "R2"]) == []
    assert sw.renumber({"milestones": [{"id": "a"}, {"id": "b", "requires": ["a"]}]})["milestones"][1] == {"id": "m2", "requires": ["m1"]}
    norm = sw.normalize_plan(GOOD_PLAN)
    assert norm["milestones"][1]["text"] == GOOD_PLAN["milestones"][1]["requirement"] and norm["milestones"][1]["requires"] == ["m1"]


def test_oracle_and_test_guards() -> None:
    assert sw.oracle_problems(["m1: mul", "m2: power"], GOOD_PLAN) == []
    assert sw.oracle_problems(["m1: mul"], GOOD_PLAN)[0].startswith("1 commits for 2 milestones")
    assert "must start with 'm2:'" in sw.oracle_problems(["m1: mul", "power"], GOOD_PLAN)[0]
    tests = {"m1": "def test_a():\n    from calc import mul\n    assert mul(2, 3) == 6\n",
             "m2": "import inspect\n\ndef test_b():\n    import calc\n    assert 'power' in inspect.getsource(calc)\n"}
    probs = sw.test_problems(tests, GOOD_PLAN)
    assert probs == ["m2: test_m2.py reads source or code structure (inspect.): test behaviour"]
    assert sw.test_problems({"m1": "def helper(:\n"}, GOOD_PLAN)[0].startswith("m1: test_m1.py does not parse")


def test_outcome_findings_name_ids_never_values(tmp_path: Path) -> None:
    base = make_repo(tmp_path)
    f1, f2 = sw.hidden_path(base, "m1"), sw.hidden_path(base, "m2")
    assert f1 == "tests/test_swarm_m1.py"
    at_a = {f"{f1}::test_a": "failed", f"{f1}::test_b": "passed", f"{f2}::test_c": "error"}
    at_b = {f"{f1}::test_a": "passed", f"{f1}::test_b": "passed", f"{f2}::test_c": "failed"}
    findings, per = sw.outcome_findings(at_a, at_b, GOOD_PLAN, base)
    assert per == {"m1": {"tests": 2, "f2p": 1, "pass_on_start": 1, "fail_on_reference": 0},
                   "m2": {"tests": 1, "f2p": 0, "pass_on_start": 0, "fail_on_reference": 1}}
    assert any("already pass on the current repository" in f and "test_b" in f for f in findings)
    assert any(f.startswith("m2: these tests fail on a reference implementation of the plan (failed): ['test_c']") for f in findings)
    assert not any(re.search(r"assert|==|\d+ != \d+", f) for f in findings)  # outcome kinds and ids only (V2)


# ---- the synthetic window -----------------------------------------------------------------------------------------------------------
def test_synthesize_resets_tests_and_protected_paths_and_adds_hidden_files_per_milestone(tmp_path: Path) -> None:
    src = make_repo(tmp_path)
    a = sh(src, "git rev-parse A").strip()
    g = "git -c user.email=a@b -c user.name=a"
    sh(src, "git checkout -q -b oracle A")
    (src / "calc.py").write_text("def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n")
    (src / "tests" / "test_calc.py").write_text("def test_builder_own():\n    pass\n")  # the builder touched a test: reset to A
    (src / "setup.cfg").write_text("[metadata]\n")  # a protected path: removed (new at the oracle)
    sh(src, f"git add -A && {g} commit -qm 'm1: mul'")
    (src / "calc.py").write_text((src / "calc.py").read_text() + "\n\ndef power(a, n):\n    out = 1\n    for _ in range(n):\n        out = mul(out, a)\n    return out\n")
    sh(src, f"{g} commit -qam 'm2: power'")
    shas = sh(src, "git rev-list --reverse A..oracle").split()
    base = tmp_path / "base"
    sh(tmp_path, f"git -C {src} worktree add -q {base} A")
    tests = {"m1": "def test_m1():\n    from calc import mul\n    assert mul(2, 3) == 6\n", "m2": "def test_m2():\n    from calc import power\n    assert power(2, 3) == 8\n"}
    win, info = sw.synthesize(src, a, shas, tests, sw.normalize_plan(GOOD_PLAN), base, cid="sw-calc-1", a_ref="A", a_date="2020-01-01T00:00:00+00:00", repo="calc")
    assert len(win.units) == 2 and win.a == a and win.b_version is None and win.kind == "swarm"
    assert info["reset_paths"] == ["setup.cfg", "tests/test_calc.py"] and set(info["unit_of"]) == {"m1", "m2"}
    u1, u2 = info["chain"]
    ls1, ls2 = sh(src, f"git ls-tree -r --name-only {u1}").split(), sh(src, f"git ls-tree -r --name-only {u2}").split()
    assert "tests/test_swarm_m1.py" in ls1 and "tests/test_swarm_m2.py" not in ls1 and "tests/test_swarm_m2.py" in ls2 and "setup.cfg" not in ls2
    assert sh(src, f"git show {u2}:tests/test_calc.py") == sh(src, "git show A:tests/test_calc.py")
    assert "def power" in sh(src, f"git show {u2}:calc.py") and "def power" not in sh(src, f"git show {u1}:calc.py")
    assert sh(src, "git rev-parse refs/crucible/swarm/sw-calc-1").strip() == u2
    assert sw.SwarmTask.generator == "swarm_window"
    with pytest.raises(ValueError):
        sw.synthesize(src, a, shas[:1], tests, sw.normalize_plan(GOOD_PLAN), base, cid="x", a_ref="A", a_date="", repo="calc")


# ---- the machine, end to end on a local kernel ---------------------------------------------------------------------------------------
def cell(code: str, i: int) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[{"id": f"py{i}", "name": "python", "arguments": {"code": code}}], usage={"prompt_tokens": 100, "completion_tokens": 10})


def W(call: str) -> str:
    return f"r = await world.call({call}); print(r['ok'], str(r['content'])[:3000])"


def oracle_cells(task: str, k: int) -> str:
    g = "git add -A && git commit -qm"
    return [W("'repo.write_file', path='calc.py', content='def add(a, b):\\n    return a + b\\n\\n\\ndef mul(a, b):\\n    return a * b\\n'"),
            W(f"'repo.shell', command='{g} \"m1: mul\" && echo C1'"),
            W("'repo.shell', command='printf \"\\\\n\\\\ndef power(a, n):\\\\n    out = 1\\\\n    for _ in range(n):\\\\n        out = mul(out, a)\\\\n"
              "    return out\\\\n\" >> calc.py && git commit -qam \"m2: power\" && echo C2'"),
            "rlm.finish('two commits')"][min(k, 3)]


def verifier_cells(task: str, k: int) -> str:
    m1 = "def test_mul():\\n    from calc import mul\\n    assert mul(2, 3) == 6\\n"
    m2 = "def test_power():\\n    from calc import power\\n    assert power(2, 3) == 8\\n" + ("\\n\\ndef test_power_zero():\\n    from calc import power\\n"
                                                                                           "    assert power(5, 0) == 1\\n" if "zero" in task else "")
    return [W("'repo.shell', command='git log --all --format=%s'"),
            W(f"'repo.write_file', path='{sw.HIDDEN_DIR}/test_m1.py', content='{m1}'"),
            W(f"'repo.write_file', path='{sw.HIDDEN_DIR}/test_m2.py', content='{m2}'"),
            "rlm.finish('tests written')"][min(k, 3)]


class Script:
    """Scripted session models: the RLM cells per role (``ROLE:`` line of TASK), every TASK and tool output recorded."""

    model = "scripted"

    def __init__(self) -> None:
        self.tasks: dict[str, list[str]] = {}
        self.outputs: dict[str, list[str]] = {}

    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> LLMResponse:
        task = next((m["content"] for m in messages if m.get("role") == "user"), "")
        role = re.search(r"^ROLE: (\S+)", task, re.MULTILINE).group(1)  # type: ignore[union-attr]
        k = sum(1 for m in messages if m.get("role") == "tool")
        if k == 0:
            self.tasks.setdefault(role, []).append(task)
        else:
            self.outputs.setdefault(role, []).append(str(next(m["content"] for m in reversed(messages) if m.get("role") == "tool")))
        r = cell({"oracle_builder": oracle_cells, "verifier_writer": verifier_cells}[role](task, k), k)
        r.model = self.model
        return r


class Planner:
    def __init__(self) -> None:
        self.seen: list[list[dict[str, str]]] = []

    async def complete_json(self, messages: list[dict[str, str]], **kw: Any) -> dict:
        self.seen.append(messages)
        if len(self.seen) == 1:  # no dependency: sent back with the guard's findings
            flat = json.loads(json.dumps(GOOD_PLAN))
            flat["milestones"][1]["requires"] = []
            return flat
        return GOOD_PLAN


class Reviewer:
    def __init__(self) -> None:
        self.n = 0

    async def complete_json(self, messages: list[dict[str, str]], **kw: Any) -> dict:
        self.n += 1
        assert "def mul(a, b):\n    return a * b" not in messages[1]["content"]  # the reviewer never sees the oracle
        return {"blocking": ["test_m2.py: power(x, 0) is never checked -> add a zero exponent test"] if self.n == 1 else [], "advisory": []}


async def test_machine_end_to_end_with_sessions_on_a_local_kernel(tmp_path: Path) -> None:
    src = make_repo(tmp_path)
    a = sh(src, "git rev-parse A").strip()
    family = TaskRepo(tmp_path / "family", "swarm-window")
    script, planner, reviewer = Script(), Planner(), Reviewer()
    m = sw.machine_with_briefs("python -m pytest -q tests")
    assert validate_machine(m, models=sw.ROLE_MODELS) == []
    spec = {"repo_name": "calc", "requests": [{"id": "R1", "text": "Added mul()"}, {"id": "R2", "text": "Added power(), built on multiplication"}],
            "repository": {"files": ["calc.py"]}, "sources": ["req-1", "req-2"]}
    params = {"spec": spec, "repo": str(src), "base": a, "task_id": "sw-calc-e2e", "repo_name": "calc", "scope": {"repo": "calc", "start": a},
              "abort_criteria": ["upstream ships it"], "holder": "root/1", "parent": "calc@A"}
    pool = SessionPool(tmp_path / "factory", repo=family, machine=m, run_id="sw-calc-e2e", params=params, kernel="local", llm_for=lambda role: script)
    svc = sw.SwarmServices(cand=sw.SwarmCandidate("sw-calc-e2e", "calc", {"a": a}, []), clone=src, work=tmp_path / "w", funnel=Funnel(tmp_path / "funnel.jsonl"),
                           clients={"planner": planner, "reviewer": reviewer}, pool=pool)
    built: dict[str, Any] = {}

    async def build(ctx: Any) -> Any:
        tests = sw.hidden_tests(Path(ctx.artifact_root) / ctx.inputs["tests"]["path"], ctx.inputs["plan"])
        base = tmp_path / "base"
        if not base.exists():
            sh(tmp_path, f"git -C {src} worktree add -q {base} {a}")
        win, info = sw.synthesize(src, a, ctx.inputs["check"]["shas"], tests, ctx.inputs["plan"], base, cid="sw-calc-e2e", a_ref="A", a_date="", repo="calc")
        built.update(win=win, info=info, tests=tests)
        return sw.verdict(True, stage="mutants_valid")

    async def ok(ctx: Any) -> Any:
        built.setdefault("states", []).append(ctx.state_id)
        return sw.verdict(True)

    svc.build_fn, svc.prompt_fn, svc.gates_fn, svc.second_fn, svc.hack_fn = build, ok, ok, ok, ok
    run = MachineRun(m, run_dir=tmp_path / "machine", run_id="sw-calc-e2e", models=sw.ROLE_MODELS, params=pool.params,
                     handlers=sw.handlers(svc, lease=lease_state(family), accept=accept_state(family)), services=svc, **pool.machine_kwargs())
    try:
        res = await run.run()
    finally:
        await pool.close()
    assert res.status == "done", (res.reason, res.decision_request)
    assert res.entries == {"lease": 1, "plan": 2, "oracle": 1, "oracle_check": 1, "verifier": 2, "review": 2, "build": 1, "prompt": 1, "gates": 1,
                           "second": 1, "hack": 1, "accept": 1}
    assert "must be dependent" in planner.seen[1][1]["content"]  # the S2 guard's finding reached the second plan
    assert res.outputs["oracle"]["oracle"]["subjects"] == ["m1: mul", "m2: power"] and len(res.outputs["oracle_check"]["verdict"]["shas"]) == 2
    # generation and verification never shared a context: the verifier's clone has no oracle commit, its TASK no oracle code
    vlog = "\n".join(script.outputs["verifier_writer"])
    assert "m1: mul" not in vlog and "release A" in vlog and all("return a * b" not in t for t in script.tasks["verifier_writer"])
    assert "add a zero exponent test" in script.tasks["verifier_writer"][1]  # the reviewer's blocking finding, once
    assert "test_power_zero" in built["tests"]["m2"] and len(built["win"].units) == 2
    assert sh(src, f"git show {built['info']['chain'][1]}:tests/test_swarm_m2.py").count("def test_") == 2
    ho = json.loads(next(family.root.glob("runs/sw-calc-e2e/verifier/*/handoff.json")).read_text())
    assert ho["spec"] == spec and set(ho["ports"]) == {"plan"}  # typed hand-off with the full original spec
    assert [x["task_id"] for x in family.accepted()] == ["sw-calc-e2e"] and built["states"] == ["prompt", "gates", "second", "hack"]
    raw = [json.loads(x) for x in (tmp_path / "funnel.jsonl").read_text().splitlines()]  # every row; the summary keeps the last per stage
    assert [(r["stage"], r["ok"]) for r in raw] == [("S2_plan", False), ("S2_plan", True), ("S4_oracle", True), ("S11_review", False), ("S11_review", True)]


def test_pick_frontier_and_extrapolate_and_target_specs() -> None:
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("lh_swarm_p5b", root / "scripts" / "lh_swarm.py")
    lh = importlib.util.module_from_spec(spec)
    sys.modules["lh_swarm_p5b"] = lh
    spec.loader.exec_module(lh)  # type: ignore[union-attr]
    models = [{"id": "openai/a", "created": 3, "pricing": {"prompt": "0.000002", "completion": "0.00001", "input_cache_read": "0.0000001"},
               "supported_parameters": ["tools"]},
              {"id": "openai/a:batch", "created": 4, "pricing": {}, "supported_parameters": ["tools"]},
              {"id": "anthropic/b", "created": 5, "pricing": {"prompt": "0.00001", "completion": "0.00005"}, "supported_parameters": []},
              {"id": "qwen/c", "created": 9, "pricing": {}, "supported_parameters": ["tools"]}]
    got = lh.pick_frontier(models)
    assert got["chosen"]["id"] == "openai/a" and got["chosen"]["input"] == 2.0 and got["chosen"]["cache_read"] == 0.1
    assert lh.pick_frontier(models, prefer="openai/a")["chosen"]["id"] == "openai/a" and [c["id"] for c in got["candidates"]] == ["openai/a"]
    ex = lh.extrapolate(0.1, spent_usd=2.0, cap=30.0, slack=3.0)
    assert ex["left_usd"] == 25.0 and ex["episodes_that_fit"] == 250 and ex["plans"]["n=4"] == {"per_window_usd": 0.8, "windows": 31}
    assert lh.target_spec("stitched:st-x=m02c") == {"kind": "stitched", "id": "st-x", "targets": ["m02c"], "simplify": "host_change"}
    assert lh.target_spec("rebreak:rb-1@faulty_function")["simplify"] == "faulty_function"
    with pytest.raises(ValueError):
        lh.target_spec("open:x")


def test_mentioned_names_and_call_links_from_the_start_tree(tmp_path: Path) -> None:
    text = "Added ``week_start`` to ``floor()`` and [`iterutils.remap`][x]; fixed has_path() and `_private`."
    assert sw.mentioned_names(text) == ["week_start", "floor", "remap", "has_path"]
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text(
        "def remap(x):\n    return list(x)\n\n\ndef research(x):\n    return remap(x)\n\n\nclass Arrow:\n    def span(self):\n        return 1\n\n"
        "    def floor(self):\n        return self.span()\n\n    def append(self, v):\n        return len(v)\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_core.py").write_text("from pkg.core import research\n\n\ndef test_r():\n    assert research([1]) == [1]\n")
    links = sw.call_links(tmp_path, ["research", "remap", "floor"])
    assert {"caller": "research", "callee": "remap", "module": "pkg/core.py", "both_mentioned": "true"} == links[0]
    assert {"caller": "Arrow.floor", "callee": "span", "module": "pkg/core.py", "both_mentioned": "false"} in links
    assert not any(x["callee"] == "len" or x["module"].startswith("tests/") for x in links)


async def test_reasoning_cap_bounds_thinking_on_json_calls() -> None:
    seen: list[dict] = []

    class C:
        model = "m"

        async def complete_json(self, messages: list, **kw: Any) -> dict:
            seen.append(kw)
            return {}

    cap = sw.ReasoningCap(C(), reasoning_tokens=5)
    await cap.complete_json([], max_tokens=10)
    await cap.complete_json([], extra_body={"reasoning": {"effort": "low"}, "x": 1})
    assert seen == [{"max_tokens": 10, "extra_body": {"reasoning": {"max_tokens": 5}}}, {"extra_body": {"reasoning": {"effort": "low"}, "x": 1}}]
    assert cap.model == "m"


def test_import_paths_the_hidden_tests_use_are_stated_by_the_lane() -> None:
    from crucible.factory.lanes.windows import EPILOGUE

    tests = {"t1": "def test_a():\n    from funcy import memoize, SkipMemoization\n    from funcy.calc import cache\n", "t2": "import os\nfrom .x import y\n"}
    assert sw.test_imports(tests) == [("funcy", "memoize"), ("funcy", "SkipMemoization"), ("funcy.calc", "cache")]
    prompt = "Opening.\n\n- [R1] expose .skip as the SkipMemoization class\n\n" + EPILOGUE
    stated = sw.unstated_imports([["funcy", "SkipMemoization"], ["funcy.calc", "cache"]], prompt + " `funcy.calc.cache`")
    assert stated == ["from funcy import SkipMemoization"]  # the class is named, its import path is not
    out = sw.state_imports(prompt, stated)
    assert "These names must be importable exactly as written: `from funcy import SkipMemoization`." in out and out.endswith(EPILOGUE)
    assert out.index("importable") < out.index(EPILOGUE[:40]) and sw.state_imports(prompt, []) == prompt
