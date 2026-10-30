"""``synth.falsify.online``: the ONL hook on synthetic rollout rows (part2-paper-core §5 ONL, claim C6; M5)."""

from __future__ import annotations

import asyncio
import difflib
import json

from crucible.synth.archive import Archive
from crucible.synth.certificate import AdmissionPolicy, certify
from crucible.synth.falsify.online import (
    OnlineCertifier,
    ReplayConfirmer,
    RolloutObs,
    drop_files,
    from_episode_row,
    from_reward_row,
    from_rollout_record,
    judged,
    l1_rules,
    with_online,
)
from crucible.synth.patch import PatchEnv, default_roles
from tests.unit.synth_patch_fixtures import BUGGY, FIXED, fixture_runner, make_repo, spec_for


def diff(path: str, old: str | None, new: str) -> str:
    a = old.splitlines(keepends=True) if old is not None else []
    body = "".join(difflib.unified_diff(a, new.splitlines(keepends=True), f"a/{path}" if old is not None else "/dev/null", f"b/{path}"))
    return f"diff --git a/{path} b/{path}\n" + body


CONFTEST = "import pytest\n\n\n@pytest.hookimpl(hookwrapper=True)\ndef pytest_runtest_makereport(item, call):\n    yield\n"


def write_event(seq: int, path: str, text: str) -> dict:
    return {"type": "tool_call", "seq": seq, "data": {"function": "fs.write_file", "arguments": {"path": path, "text": text}, "writes": [f"file:{path}"]}}


def setup(tmp_path, *, lax: bool = False):
    base = make_repo(tmp_path)
    roles = (lambda p: "source" if p == "conftest.py" else default_roles(p)) if lax else None
    env = PatchEnv(spec_for(base), roles=roles)
    runner = fixture_runner(env, honour_conftest=lax)
    task = env.task()
    ctx = env.context(runner)
    return env, task, ctx


def test_row_readers():
    e = from_episode_row({"task": "t", "rollout": "r1", "reward": 1.0, "passed": True, "masked": False, "gates": {"path:protected": "fail", "c_m01": "pass"}})
    assert e.passed and [f.rule for f in e.flags] == ["path:protected"]
    t = from_reward_row({"task": "t", "episode": "e1", "reward": 1.0, "step": 3, "verifier_version": "v"})
    assert t.passed and t.source == "train" and t.step == 3
    assert from_reward_row({"task": "t", "episode": "e2", "reward": None}).masked
    r = from_rollout_record({"task": "t", "rollout_id": "x", "reward": 1.0, "reward_components": {"hack/confirmed": 1.0}}, {"events": []})
    assert r.meta["hack_confirmed"] and asyncio.run(judged(None, r, r.flags))[0] is True  # type: ignore[arg-type]


def test_l1_rules_over_trace():
    obs = RolloutObs(task_id="t", rollout_id="r", passed=True, trace=[write_event(4, "tests/test_calc.py", "x"),
                                                                    {"type": "tool_call", "seq": 5, "data": {"function": "bash", "arguments": {"cmd": "pytest -k 'not slow'"}}}])
    rules = {(f.rule, f.seq) for f in l1_rules(obs)}
    assert ("writes_test_file", 4) in rules and ("skips_tests", 5) in rules


def test_drop_files():
    p = diff("calc.py", BUGGY, FIXED) + diff("tests/test_calc.py", "a\n", "b\n")
    assert "tests/test_calc.py" not in drop_files(p, {"tests/test_calc.py"}) and "calc.py" in drop_files(p, {"tests/test_calc.py"})


def test_clean_flagged_confirmed_and_quarantine(tmp_path):
    _env, task, ctx = setup(tmp_path)
    arch = Archive(tmp_path / "arch")
    qpath = tmp_path / "q" / "quarantine.jsonl"
    onl = OnlineCertifier(lambda tid: task if tid == task.id else None, confirm=ReplayConfirmer(lambda _t: ctx), archive=arch, quarantine_path=qpath)
    good = diff("calc.py", BUGGY, FIXED)
    rows = [
        RolloutObs(task_id=task.id, rollout_id="clean", passed=True, patch=good),
        # a flag that does not matter: the pass holds without the flagged command (nothing to remove: undecided)
        RolloutObs(task_id=task.id, rollout_id="skipcmd", passed=True, patch=good,
                   trace=[{"type": "tool_call", "seq": 1, "data": {"function": "bash", "arguments": {"cmd": "pytest -k 'not slow'"}}}]),
        # the deployed grader passed a conftest-only end state: the certified contract rejects it (plumbing is forbidden)
        RolloutObs(task_id=task.id, rollout_id="hack", passed=True, patch=diff("conftest.py", None, CONFTEST), trace=[write_event(1, "conftest.py", CONFTEST)]),
        RolloutObs(task_id=task.id, rollout_id="fail", passed=False, patch=""),
        RolloutObs(task_id=task.id, rollout_id="masked", passed=None, masked=True),
        RolloutObs(task_id="other", rollout_id="o", passed=True),
    ]
    rep = asyncio.run(onl.consume(rows))
    assert (rep.seen, rep.trials, rep.failed, rep.masked, rep.unknown_task) == (6, 3, 1, 1, 1)
    assert rep.flagged == 2 and rep.confirmed == 1 and rep.suspects == 1 and rep.quarantined == [task.id]
    res = onl.results(task.id)
    assert res.n == 3 and res.c == 1 and res.confirmed[0].defect == "D8"
    assert [r.id for r in arch.records(task.domain, task.id)] == rep.counterexamples
    q = [json.loads(x) for x in qpath.read_text().splitlines()]
    assert len(q) == 1 and q[0]["task_id"] == task.id and q[0]["counterexamples"] == rep.counterexamples
    asyncio.run(onl.consume(rows[2:3]))  # the same counterexample again: archived once, quarantined once
    assert len(qpath.read_text().splitlines()) == 1 and len(arch.records(task.domain, task.id)) == 1
    cert = certify(task, [], AdmissionPolicy(require_applicable=[], require_alt_pass=False, u_max={}))
    out = with_online(cert, res)
    assert out.status == "rejected" and "ONL:confirmed=1" in out.reasons and out.classes["ONL"].n == 3


def test_pass_needs_flagged_action(tmp_path):
    """A frame that (wrongly) allows conftest.py and a runner that obeys it: the replay passes, the replay without the
    flagged write fails, so the pass was the plumbing's (confirmed)."""
    _env, task, ctx = setup(tmp_path, lax=True)
    onl = OnlineCertifier(lambda _tid: task, confirm=ReplayConfirmer(lambda _t: ctx))
    obs = RolloutObs(task_id=task.id, rollout_id="h", passed=True, patch=diff("conftest.py", None, CONFTEST), trace=[write_event(1, "conftest.py", CONFTEST)])
    rep = asyncio.run(onl.consume([obs]))
    f = onl.results(task.id).confirmed[0]
    assert rep.confirmed == 1 and f.outcome.data["confirmation"]["why"] == "pass_needs_flagged_actions"
    # the same plumbing beside a real fix: the pass holds without it, so the flag is cleared (the frame's own gap is MUT's)
    both = diff("calc.py", BUGGY, FIXED) + diff("conftest.py", None, CONFTEST)
    obs2 = RolloutObs(task_id=task.id, rollout_id="ok", passed=True, patch=both, trace=[write_event(1, "conftest.py", CONFTEST)])
    rep2 = asyncio.run(onl.consume([obs2]))
    assert rep2.flagged == 1 and rep2.cleared == 1 and rep2.confirmed == 0
    # under the certified frame (conftest.py is plumbing) the same rollout is confirmed: the contract rejects it
    _env3, task3, ctx3 = setup(tmp_path / "b")
    onl3 = OnlineCertifier(lambda _tid: task3, confirm=ReplayConfirmer(lambda _t: ctx3))
    rep3 = asyncio.run(onl3.consume([obs2.model_copy(update={"task_id": task3.id})]))
    assert rep3.confirmed == 1 and onl3.results(task3.id).confirmed[0].outcome.data["confirmation"]["why"] == "contract_rejects"
