"""The framework gaps the open-risks pass (2026-10-28, commit 32ffaca1) left open, fixed (docs/research/harness-reward-audit.md
§5 and §8 R2, R6, R10):

- R6 follow-up: ``FsGitAdapter.q_git_diff`` (the native harness's verifier view) honoured the agent's repo-local filter
  drivers and followed a redirected repository when it ran without a ``trusted_git_dir``; the host checkpoint snapshot
  followed a redirected repository.
- R2 follow-up: the RLM's model calls (turns, sub-calls, compaction summaries, child agents) ran on the agent clock of an
  ``agent_clock="own"`` episode.
- §5 / R10: ``export_agentenv`` published Tasks whose World step cannot run; ``_restore_triggers`` lost or shifted
  wall-clock triggers.

Every test fails on the code before its fix. Host git in tmp dirs, a host kernel for the RLM (trusted scripted code), no
Docker, no model."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible.actors import policy as pm
from crucible.actors.llm import LLMResponse
from crucible.core.schema.state import Query
from crucible.core.store import Store
from crucible.world.exec import LocalExec
from crucible.world.state.fs import FsGitAdapter, git_diff_command

ENV = {"GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": "a@a", "GIT_COMMITTER_NAME": "a", "GIT_COMMITTER_EMAIL": "a@a", "GIT_CONFIG_NOSYSTEM": "1",
       "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}


def git(ws: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(ws), *a], check=True, capture_output=True, text=True, env={**ENV, "HOME": str(ws.parent)}).stdout


def repo(tmp_path: Path, name: str = "ws") -> Path:
    ws = tmp_path / name
    ws.mkdir()
    git(ws, "init", "-q")
    (ws / "a.py").write_text("x = 1\n")
    git(ws, "add", "-A")
    git(ws, "commit", "-qm", "base")
    git(ws, "tag", "base")
    return ws


def adapter(tmp_path: Path, ws: Path) -> FsGitAdapter:
    return FsGitAdapter("repo", LocalExec(), {"root": str(ws)}, Store(tmp_path / "store"))


# ---- R6 follow-up: the verifier's git view of the agent's own repository ---------------------------------------------------------
async def test_q_git_diff_runs_no_filter_driver_of_the_agent(tmp_path):
    ws = repo(tmp_path)
    sentinel = tmp_path / "pwned"
    git(ws, "config", "filter.evil.clean", f"sh -c 'touch {sentinel}; echo hacked'")
    (ws / ".gitattributes").write_text("*.py filter=evil\n")
    (ws / "a.py").write_text("x = 2\n")
    d = await adapter(tmp_path, ws).query(Query(op="git_diff", args={"ref": "base"}))
    assert not sentinel.exists(), "the policy's clean command ran in the verifier's view"  # before the fix: it ran
    assert d.ok and "+x = 2" in d.stdout and "hacked" not in d.stdout


async def test_q_git_diff_refuses_a_filter_driver_no_override_can_name(tmp_path):
    ws = repo(tmp_path)
    git(ws, "config", "filter.a=b.clean", "true")
    d = await adapter(tmp_path, ws).query(Query(op="git_diff", args={"ref": "base"}))
    assert not d.ok and "filter driver" in (d.error or "")


def _gitdir_file(ws: Path, tmp_path: Path) -> None:
    other = repo(tmp_path, "elsewhere")  # a host repository holding a base of its own
    shutil.rmtree(ws / ".git")
    (ws / ".git").write_text(f"gitdir: {other / '.git'}\n")


def _symlink(ws: Path, tmp_path: Path) -> None:
    other = repo(tmp_path, "elsewhere")
    shutil.rmtree(ws / ".git")
    os.symlink(other / ".git", ws / ".git")


def _commondir(ws: Path, tmp_path: Path) -> None:
    other = repo(tmp_path, "elsewhere")
    (ws / ".git" / "commondir").write_text(str(other / ".git") + "\n")


def _worktree(ws: Path, tmp_path: Path) -> None:
    other = tmp_path / "tree"
    other.mkdir()
    (other / "a.py").write_text("x = 1\n")  # another tree: the agent's edit disappears from the diff
    git(ws, "config", "core.worktree", str(other))


def _bare(ws: Path, tmp_path: Path) -> None:
    git(ws, "config", "core.bare", "true")


@pytest.mark.parametrize("redirect,why", [(_gitdir_file, "gitdir"), (_symlink, "symlink"), (_commondir, "commondir"), (_worktree, "core.worktree"),
                                          (_bare, "core.bare")])
async def test_q_git_diff_refuses_a_redirected_repository(tmp_path, redirect, why):
    ws = repo(tmp_path)
    (ws / "a.py").write_text("x = 3\n")
    redirect(ws, tmp_path)
    d = await adapter(tmp_path, ws).query(Query(op="git_diff", args={"ref": "base"}))
    # before the fix: a diff of another repository or tree (or of nothing), read as the agent's work
    assert not d.ok and why in (d.error or "") and "refused" in d.error and not d.stdout


async def test_q_block_changed_refuses_a_redirected_repository(tmp_path):
    ws = repo(tmp_path)
    _gitdir_file(ws, tmp_path)
    r = await adapter(tmp_path, ws).query(Query(op="block_changed", args={"path": "a.py", "anchor": "x"}))
    assert not r.ok and "gitdir" in (r.error or "")


async def test_a_worktree_naming_the_workspace_itself_still_diffs(tmp_path):
    ws = repo(tmp_path)
    git(ws, "config", "core.worktree", "..")  # relative to .git: the workspace itself, as git init --separate layouts write it
    (ws / "a.py").write_text("x = 4\n")
    d = await adapter(tmp_path, ws).query(Query(op="git_diff", args={"ref": "base"}))
    assert d.ok and "+x = 4" in d.stdout


def test_the_unguarded_commands_are_unchanged():
    """The policy's own ``git_diff`` tool, the checkpoint sidecar and a trusted-copy view keep their exact command."""
    plain = git_diff_command("/w", "base")
    assert '"$@"' not in plain and "refused" not in plain and plain.startswith("set -e; T=$(mktemp); trap 'rm -f \"$T\"' EXIT; ")
    assert "refused" not in git_diff_command("/w", "base", git_dir="/trusted.git")
    with pytest.raises(ValueError):
        git_diff_command("/w", "base", git_dir="/trusted.git", guard=True)


async def test_checkpoint_snapshot_refuses_a_redirected_repository(tmp_path):
    from crucible.factory.checkpoint import workspace_snapshot

    ws = repo(tmp_path)
    sha = git(ws, "rev-parse", "HEAD").strip()
    _gitdir_file(ws, tmp_path)
    with pytest.raises(RuntimeError, match="gitdir"):  # before the fix: host git diffed the other repository
        await workspace_snapshot(ws, sha)(None)


# ---- R2 follow-up: the RLM's model calls hold the agent clock under agent_clock="own" -------------------------------------------
def _text(m: dict) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c, default=str)


def _role(messages: list, tools: Any) -> str:
    system = next((_text(m) for m in messages if m.get("role") == "system"), "")
    if not system:
        return "sub"
    if not tools:
        return "summary"
    return "child" if "You are a child agent" in system else "root"


def _cell(code: str, i: int = 0) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[{"id": f"py{i}", "name": "python", "arguments": {"code": code}}])


def _decide(messages: list, tools: Any, role: str) -> LLMResponse:
    """A root that compacts, makes a sub-call, runs a child agent and finishes: every kind of RLM model call."""
    if role == "summary":
        return LLMResponse(text="SUMMARY: compacted on request")
    if role == "sub":
        return LLMResponse(text="sub answer")
    if role == "child":
        return _cell("rlm.finish('child done')")
    last = next((_text(m) for m in reversed(messages) if m.get("role") == "tool"), "")
    if "CHILD" in last:
        return _cell("rlm.finish('done')", 4)
    if "SUBS" in last:
        return _cell("await rlm.agent.spawn('say done', name='c1')\nr = await rlm.agent.result('c1', wait_s=60)\nprint('CHILD', r['status'])", 3)
    if "ASKED" in last:
        return _cell("print('SUBS', await rlm.batch(['q']))", 2)
    return _cell("print('ASKED', await rlm.compact(instructions='keep it'))", 1)


class ClockLLM:
    """Records, per call, whether the episode's agent clock stood still while the model answered."""

    model = "script"

    def __init__(self, delay: float = 0.0):
        self.delay, self.held = delay, []

    async def complete(self, messages, *, tools=None, max_tokens=2000, **kw):
        role, clock = _role(messages, tools), pm._MODEL_CLOCK.get()
        if self.delay:
            await asyncio.sleep(self.delay)
        self.held.append((role, clock is not None and clock.when() is None))
        r = _decide(messages, tools, role)
        r.usage = {"prompt_tokens": pm.approx_tokens(messages), "completion_tokens": 8}
        r.model = self.model
        return r


async def _rlm_episode(tmp_path: Path, llm: ClockLLM, limits: pm.EpisodeLimits):
    from crucible_pack_toy.runtime import HelpdeskTask

    from crucible.actors.rlm.agent import RLMConfig
    from crucible.actors.rlm.kernel import LocalLauncher
    from crucible.actors.rlm.policy import RLMPolicy

    task = HelpdeskTask()
    w = await task.open(tmp_path / "work")
    cfg = RLMConfig(temperature=0.0, kernel="local", allow_host=True, window_tokens=64_000, max_tokens=2_000, cell_timeout_s=30.0)
    p = RLMPolicy("agent", llm, ledger_dir=tmp_path / "ledger", launcher=LocalLauncher(allow_host=True), config=cfg)
    try:
        return await pm.run_episode(w, {"agent": p}, limits=limits)
    finally:
        await w.close()
        await p.aclose()


@pytest.mark.parametrize("clock", ["own", "wall"])
async def test_every_rlm_model_call_holds_the_agent_clock_only_under_own(tmp_path, clock):
    llm = ClockLLM()
    ep = await _rlm_episode(tmp_path, llm, pm.EpisodeLimits(max_steps=500, timeout_s=600, agent_clock=clock))
    assert ep.stop_condition == "agent_done", (ep.stop_condition, ep.failures)
    assert {r for r, _ in llm.held} == {"root", "summary", "sub", "child"}, llm.held
    # before the fix: no RLM call went through model_time, so under "own" the agent clock ran through every one of them
    assert all(h is (clock == "own") for _, h in llm.held), llm.held


@pytest.mark.parametrize("clock,stop", [("own", "agent_done"), ("wall", "timeout")])
async def test_rlm_sampler_latency_decides_the_timeout_only_on_the_wall_clock(tmp_path, clock, stop):
    llm = ClockLLM(delay=1.0)  # 6 calls x 1 s of sampler queue: past the 4 s agent budget on the wall clock
    ep = await _rlm_episode(tmp_path, llm, pm.EpisodeLimits(max_steps=500, timeout_s=4.0, agent_clock=clock))
    assert ep.stop_condition == stop, (ep.stop_condition, ep.failures, llm.held)


def test_rlm_harness_ids_are_unchanged_at_the_default_clock():
    from crucible.runtime.harness.rlm import RLMHarness

    # the ids the RLM harness had before the follow-up (and before agent_clock existed): evals keep their identity
    assert RLMHarness().id == "rlm:b92f2cc0f1@1#f9f28dbf9ee1"
    assert RLMHarness(limits=pm.EpisodeLimits(max_steps=500, timeout_s=3600)).id == "rlm:b92f2cc0f1@1#0d025a456f1a"
    assert RLMHarness(limits=pm.EpisodeLimits(max_steps=500, timeout_s=3600, agent_clock="wall")).id == "rlm:b92f2cc0f1@1#0d025a456f1a"
    assert RLMHarness(limits=pm.EpisodeLimits(max_steps=500, timeout_s=3600, agent_clock="own")).id != "rlm:b92f2cc0f1@1#0d025a456f1a"


# ---- §5: export_agentenv published Tasks that cannot run ---------------------------------------------------------------------------
def test_export_agentenv_refuses_rows_whose_task_cannot_run(monkeypatch):
    from types import SimpleNamespace

    from crucible.factory import envs, publish

    published = []
    monkeypatch.setattr(publish, "publish_task", lambda c, **kw: published.append(c.id) or SimpleNamespace(card_id=c.id))
    monkeypatch.setattr(publish, "publish_eval", lambda *a, **kw: SimpleNamespace())
    ref = {"lane": "window", "task_id": "w1", "source_id": "s1", "store": "/nowhere", "split": "train", "prompt_sha256": "p", "checkset_sha256": "c",
           "verifier_version": "v"}
    row = {"args": {"ref": ref}}
    with pytest.raises(NotImplementedError):  # what the published Task's crucible_world step does on every run
        envs.RowTask(backend="agentenv", row=row)
    card = SimpleNamespace(id="c1", family="f")
    with pytest.raises(publish.PublishError, match="could not run"):  # before the fix: published, then failed at run time
        envs.export_agentenv([card], {"c1": row}, {"train": ["c1"]})
    assert published == []


# ---- §5: restoring the delegated triggers --------------------------------------------------------------------------------------
T0 = "2026-03-02T09:00:00Z"


def _t(minutes: float) -> str:
    import datetime

    return (datetime.datetime(2026, 3, 2, 9, tzinfo=datetime.UTC) + datetime.timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")


def _oracle(seed: int, mean_s: float, n: int, start: str = T0) -> list[str]:
    """agentenv's marks for ``every: {dist: exp}`` with no ``at``: t0 + d0, then + d1, ... from ``random.Random(seed)``."""
    import datetime
    import random

    rng, m, out = random.Random(seed), datetime.datetime.fromisoformat(start), []
    for _ in range(n):
        m = m + datetime.timedelta(seconds=rng.expovariate(1.0 / mean_s))
        out.append(m.isoformat().replace("+00:00", "Z"))
    return out


NL = [{"type": "nl", "instruction": "{}"}]
BODY = [
    {"id": "A", "when": {"type": "time", "at": _t(10)}, "actions": NL},
    {"id": "D", "when": {"type": "time", "after": "A", "offset": "PT600S"}, "actions": NL},
    {"id": "E", "when": {"type": "time", "every": "PT300S", "count": 5}, "actions": NL},
    {"id": "S", "when": {"type": "time", "every": {"dist": "exp", "mean": "PT300S"}, "seed": 7, "count": 10}, "actions": NL},
    {"id": "U", "when": {"type": "time", "every": "PT3600S"}, "actions": NL},
]
S = _oracle(7, 300.0, 10)


def _row(tid: str, n: int, status: str, mark: str | None) -> dict:
    return {"id": tid, "fire_count": n, "status": status, "next_mark": mark}


# the engine's state at a snapshot: A fired and anchored D at its fire time + 10 min, E fired twice, S three times, U never
STATE = {"triggers": [_row("A", 1, "fired", None), _row("D", 0, "armed", _t(21)), _row("E", 2, "armed", _t(15)), _row("S", 3, "armed", S[3]),
                      _row("U", 0, "armed", _t(60))]}


class FakeGateway:
    def __init__(self):
        self.registered: list[list[dict]] = []

    async def post(self, path, json=None):
        if path == "/triggers/register":
            self.registered.append(json["triggers"])
        return SimpleNamespace(status_code=200, text="", raise_for_status=lambda: None)


def _backend():
    from crucible.core.schema.world import ClockSpec
    from crucible.interop.agentenv.world_backend import AgentEnvBackend

    b = AgentEnvBackend.__new__(AgentEnvBackend)
    b._trigger_body, b._trigger_plan, b.http = {"watch_roles": ["agent"], "triggers": BODY}, {}, FakeGateway()
    b._world = SimpleNamespace(spec=SimpleNamespace(clock=ClockSpec(mode="wall", t0=T0, rate=3600)))
    return b


def _when(reg: list[dict], tid: str) -> dict:
    return next(t["when"] for t in reg if t["id"] == tid)


async def test_restore_keeps_anchored_and_unfired_wall_clock_triggers_at_their_marks():
    b = _backend()
    await b._restore_triggers(STATE)
    reg = b.http.registered[-1]
    assert "A" not in {t["id"] for t in reg}  # fired one-shot: stays fired
    # before the fix: D was registered anchored on A again, which never fires again, so D was lost
    assert _when(reg, "D") == {"type": "time", "at": _t(21)}
    # before the fix: an unfired ``every`` restarted one interval after the restore's clock (its new t0)
    assert _when(reg, "U") == {"type": "time", "at": _t(60), "every": "PT3600S"}
    assert _when(reg, "E") == {"type": "time", "at": _t(15), "every": "PT300S", "count": 3}


async def test_restore_continues_a_seeded_stream_where_it_stood():
    b = _backend()
    await b._restore_triggers(STATE)
    reg = b.http.registered[-1]
    arrivals = [t for t in reg if t["id"].startswith("S")]
    # before the fix: one trigger at S[3] whose later arrivals drew from the start of the seed's stream again
    assert [t["when"] for t in arrivals] == [{"type": "time", "at": m} for m in S[3:]]
    assert all(t["actions"] == NL for t in arrivals)


async def test_a_restore_of_a_restore_keeps_counts_streams_and_fired_one_shots():
    b = _backend()
    await b._restore_triggers(STATE)
    first = b.http.registered[-1]
    rows = [_row(t["id"], 0, "armed", t["when"].get("at")) for t in first]
    for r in rows:  # on the restored registration: S's next two arrivals and one more E fired, then a snapshot
        if r["id"] in ("S~3", "S~4"):
            r.update(fire_count=1, status="fired", next_mark=None)
        if r["id"] == "E":
            r.update(fire_count=1, next_mark=_t(20))
    prog = b.trigger_progress({"triggers": rows})
    assert prog["S"]["fired"] == 5 and prog["E"]["fired"] == 3 and prog["A"]["done"]
    await b._restore_triggers({"triggers": rows}, prog)
    second = b.http.registered[-1]
    assert "A" not in {t["id"] for t in second}  # (re-registered, it would fire a second time)
    assert [t["when"]["at"] for t in second if t["id"].startswith("S")] == S[5:]
    assert _when(second, "E") == {"type": "time", "at": _t(20), "every": "PT300S", "count": 2}


async def test_an_unbounded_seeded_stream_is_expanded_then_continued(monkeypatch):
    from crucible.interop.agentenv import world_backend as wb

    monkeypatch.setattr(wb, "MAX_EXPAND", 3)
    body = [{"id": "S", "when": {"type": "time", "every": {"dist": "exp", "mean": "PT300S"}, "seed": 7}, "actions": NL}]
    specs, plan = wb.plan_restore(body, {"S": {"fired": 3, "done": False, "next_mark": S[3], "draw": 4, "seed": 7}}, None)
    assert [s["when"]["at"] for s in specs] == S[3:7]  # three exact arrivals, then the tail starts at the fourth's exact mark
    tail = specs[-1]["when"]
    assert tail["every"] == {"dist": "exp", "mean": "PT300S"} and tail["seed"] != 7 and "count" not in tail
    assert plan["S"]["parts"][-1]["kind"] == "native"


async def test_live_gateway_replays_a_seeded_stream_across_a_restore(tmp_path):
    """The planned registration is one agentenv's own engine accepts, and the arrivals after a restore land on the marks the
    stream would have given without it."""
    pytest.importorskip("agent_env")
    from crucible.core.schema.world import WorldSpec
    from crucible.world.world import World

    spec = WorldSpec.model_validate({
        "world": "test/restore", "network": "no-network", "backend": "agentenv",
        "clock": {"mode": "wall", "t0": T0, "rate": 300, "step_dt_s": 60},
        "apps": [{"id": "mail", "template": "common/mailbox", "state": {"adapter": "sql", "params": {"path": "mail.db"}}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["mail.*"]}, {"id": "boss", "kind": "system"}],
        "events": [{"id": "ping", "actor": "boss", "when": {"every": {"dist": "exp", "mean": "5m"}, "seed": 7, "count": 6},
                    "action": {"type": "permission", "to": "agent", "grant": ["mail.list_inbox"]}}]})
    w = World(spec, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()

        async def state():
            return (await w.backend.http.get("/triggers/state")).json()

        async def fired_marks():
            return [e["mark"] for e in (await state())["events"] if e.get("kind") == "fired"]

        deadline = asyncio.get_running_loop().time() + 30
        while len(await fired_marks()) < 2 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)
        ref = await w.snapshot()
        done = next(t["fire_count"] for t in ref.backend["triggers"]["triggers"] if t["id"] == "ping")  # the first registration: its own count
        while len(await fired_marks()) < 6 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)
        original = await fired_marks()
        await w.restore(ref)
        while len(await fired_marks()) < 6 - done and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)
        assert 2 <= done <= 4 and len(original) == 6  # at least two arrivals after the restore: the second is drawn
        assert await fired_marks() == original[done:]  # before the fix: the arrivals after the restore drew a new stream
    finally:
        await w.close()
