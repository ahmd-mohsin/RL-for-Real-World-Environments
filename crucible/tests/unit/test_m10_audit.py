"""M10 audit regressions (docs/audit/2026-10-12/M10-runtime.md): one test per reproduced defect (V1-V17) and per pick.

Every test fails on the M9 audit commit (6efff7a). Symbols the audit added are imported inside the tests.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import subprocess
from dataclasses import dataclass

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.runtime.harness.base import HarnessResult
from crucible.runtime.harness.workspace import WorkspaceHarness
from crucible.runtime.run import Runner, RunSpec


def _sh(i, cmd):
    return ToolCall(tool_call_id=f"s{i}", function_name="ws.shell", arguments={"command": cmd})


async def _fib(tmp_path, name, acts, **kw):
    from crucible_pack_toy.runtime import FibTask, scripted

    spec = RunSpec(run_id=name, tasks=[kw.pop("task", None) or FibTask()], policies=[scripted(name, acts)],
                   harnesses=[kw.pop("harness", None) or WorkspaceHarness()], k=1, **kw)
    s = await Runner(tmp_path).run(spec)
    return s, json.loads((tmp_path / name / "results.jsonl").read_text().splitlines()[-1])


# ---------------------------------------------------------------- masks a policy can trigger (V1, V5, V6, V7)


@pytest.mark.parametrize("cmd", ["printf garbage > .git/HEAD", "echo x > a.txt && printf junk > .git/index", "echo x > notes.txt && chmod 000 notes.txt"])
async def test_v1_breaking_collection_is_scored_not_masked(tmp_path, cmd):
    import os

    if "chmod" in cmd and hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root reads a chmod 000 file")
    s, r = await _fib(tmp_path, "junk", [_sh(1, cmd)])
    assert r["reward"] == 0.0 and not r["mask_sample"] and s.attempts == 1


def test_v1_collection_classifies_agent_path_errors():
    from crucible.runtime.collect import Collected, classify

    c = Collected("", "tolerant", True, ['error: open("notes.txt"): Permission denied\nerror: unable to index file \'notes.txt\''], ["notes.txt"])
    assert classify(c) is None
    assert classify(Collected("", "tolerant", True, ["fatal: index corrupt"], ["a.py"])) is not None  # the hook's own failure


async def test_v2_recollect_uses_the_recorded_base(tmp_path):
    from crucible_pack_toy.runtime import FibTask

    from crucible.runtime.harness.base import Attempt
    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.keys import SampleKey

    def g(ws, *a):
        return subprocess.run(["git", "-C", str(ws), "-c", "user.email=a@b", "-c", "user.name=a", *a], capture_output=True, text=True, check=True).stdout.strip()

    sd = tmp_path / "samples" / "s"
    a1, a2 = sd / "a1", sd / "a2"
    ws = a1 / "preserved_ws"
    ws.mkdir(parents=True)
    a2.mkdir(parents=True)
    g(ws, "init", "-q")
    (ws / "README").write_text("x\n")
    g(ws, "add", "-A")
    g(ws, "commit", "-qm", "base")
    g(ws, "tag", "base")
    (a1 / "base_sha").write_text(g(ws, "rev-parse", "HEAD") + "\n")
    (ws / "fib.py").write_text("a, b = 0, 1\nfor _ in range(10):\n    print(a)\n    a, b = b, a + b\n")
    g(ws, "add", "-A")
    g(ws, "commit", "-qm", "work")
    g(ws, "tag", "-f", "base", "HEAD")  # the agent moved the tag
    (a1 / "record.json").write_text(json.dumps({"mask_sample": True, "failure": {"kind": "crucible:empty_collection", "stage": "collect"}, "meta": {}}))
    att = Attempt(sample=SampleKey(task="t", policy="p", harness="h", sample_idx=0, seed=1), id="s-a2", n=2, dir=a2, task=FibTask(),
                  policy=PolicyRef(id="p", make=lambda a: {}))
    res = await WorkspaceHarness().run(att)
    assert res.stop == "recollected" and res.reward == 1.0


async def test_v5_policy_background_jobs_end_with_the_world(tmp_path):
    from crucible.world.exec import LocalExec

    ex = LocalExec()
    tag = f"crucible-m10-v5-{secrets.token_hex(4)}"
    await ex.run(f"nohup bash -c 'exec -a {tag} sleep 30' >/dev/null 2>&1 &")
    await asyncio.sleep(0.2)
    alive = await asyncio.to_thread(lambda: subprocess.run(["pgrep", "-f", tag], capture_output=True, check=False).returncode == 0)
    ex.kill_sessions()
    await asyncio.sleep(0.2)
    gone = await asyncio.to_thread(lambda: subprocess.run(["pgrep", "-f", tag], capture_output=True, check=False).returncode != 0)
    assert alive and gone


async def test_v6_v7_policy_exceptions_are_classified(tmp_path):
    from crucible_pack_toy.runtime import HelpdeskTask

    from crucible.actors.llm import LLMResponse
    from crucible.actors.policy import ContextConfig, LLMToolPolicy, ScriptedPolicy
    from crucible.runtime.harness.native import NativeHarness, PolicyRef
    from crucible.runtime.inject import FakeLLM

    class Raiser(ScriptedPolicy):
        async def act(self, obs):
            raise IndexError("out-of-range action index")

    class Socket(ScriptedPolicy):
        async def act(self, obs):
            raise TimeoutError("socket read timed out")

    def compact(a):
        llm = FakeLLM([LLMResponse(text="", tool_calls=[{"id": "k1", "name": "context_compact", "arguments": {"keep_recent_turns": "all", "summary": "s"}}])] * 4)
        return {"agent": LLMToolPolicy("agent", llm, context=ContextConfig(mode="action"))}

    async def run(name, make):
        s = await Runner(tmp_path).run(RunSpec(run_id=name, tasks=[HelpdeskTask()], policies=[PolicyRef(id=name, make=make)], harnesses=[NativeHarness()], k=1))
        return s, json.loads((tmp_path / name / "results.jsonl").read_text().splitlines()[-1])

    _, r = await run("raises", lambda a: {"agent": Raiser("agent", [])})
    assert r["reward"] == 0.0 and not r["mask_sample"] and r["failure"]["kind"] == "crucible:policy_error"
    _, r = await run("socket", lambda a: {"agent": Socket("agent", [])})
    assert r["mask_sample"] and r["failure"]["kind"] == "transport_timeout" and r["meta"]["stop"] != "timeout"
    _, r = await run("compact", compact)
    assert not r["mask_sample"]  # a malformed compaction argument is a format problem, not an unknown mask


# ---------------------------------------------------------------- retries, budget, resume, journal (V3, V4, V8-V11)


async def test_v3_train_regrade_never_reruns_the_agent(tmp_path):
    from crucible_pack_toy.runtime import FibTask

    from crucible.actors.llm import LLMResponse
    from crucible.actors.policy import LLMToolPolicy
    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.inject import FakeLLM, FlakyGradeTask

    code = "a, b = 0, 1\nfor _ in range(10):\n    print(a)\n    a, b = b, a + b\n"
    toks = [{"prompt_token_ids": [1, 2, 3], "completion_token_ids": [7, 8], "logprobs": [-0.1, -0.2], "policy_version": 3},
            {"prompt_token_ids": [1, 2, 3, 7, 8, 100], "completion_token_ids": [7, 8], "logprobs": [-0.1, -0.2], "policy_version": 3}]
    u = {"prompt_tokens": 10, "completion_tokens": 2}
    runs = {"n": 0}

    def make(a):
        runs["n"] += 1
        return {"agent": LLMToolPolicy("agent", FakeLLM([LLMResponse(text="", tool_calls=[{"id": "x1", "name": "ws__write_file", "arguments": {
            "path": "fib.py", "content": code}}], usage=u, tokens=toks[0]), LLMResponse(text="done", usage=u, tokens=toks[1])]))}

    task = FlakyGradeTask(FibTask(), "perf rounds contaminated: ['candidate/m']", on_attempts=("a1",))
    s = await Runner(tmp_path).run(RunSpec(run_id="t", tasks=[task], policies=[PolicyRef(id="fake", make=make)], harnesses=[WorkspaceHarness()],
                                           k=1, rollout_mode="train"))
    assert runs["n"] == 1 and s.mean_reward == 1.0


@dataclass
class _T:
    id: str = "t@1"


@dataclass
class _P:
    id: str = "p"
    estimate_usd: float = 0.0


class _H:
    id = "h@1"

    def __init__(self, cost=0.0, unc=0.0):
        self.cost, self.unc = cost, unc

    async def run(self, a):
        await asyncio.sleep(0.02)
        return HarnessResult(reward=1.0, cost_usd=self.cost, cost_uncertain_usd=self.unc)


async def test_v4_budget_holds_and_uncertain_spend_stops(tmp_path):
    s = await Runner(tmp_path).run(RunSpec(run_id="b", tasks=[_T()], policies=[_P()], harnesses=[_H(0.6)], k=8, concurrency=4, budget_usd=1.0))
    assert s.attempts <= 4 and s.stopped  # 5 attempts and $3.00 with zero reservations; now only the first wave runs
    s = await Runner(tmp_path).run(RunSpec(run_id="b2", tasks=[_T()], policies=[_P(estimate_usd=0.6)], harnesses=[_H(0.6)], k=8, concurrency=4, budget_usd=1.0))
    assert s.spend["spent_run"] <= 1.0  # a declared estimate keeps the cap exactly
    s = await Runner(tmp_path).run(RunSpec(run_id="u", tasks=[_T()], policies=[_P()], harnesses=[_H(0.0, unc=0.5)], k=4, concurrency=1, budget_usd=100.0))
    assert s.stopped and "uncertain" in s.stopped


async def test_v11_harness_exceptions_are_contained(tmp_path):
    class Boom:
        id = "boom@1"

        async def run(self, a):
            raise RuntimeError("harness bug")

    s = await Runner(tmp_path).run(RunSpec(run_id="c", tasks=[_T()], policies=[_P()], harnesses=[Boom(), _H()], k=2, retry_backoff_s=0.0))
    assert s.terminal == 4 and s.scored == 2 and s.masked_terminal == 2


async def test_pick_attempt_deadline(tmp_path):
    class Hang:
        id = "hang@1"

        async def run(self, a):
            await asyncio.sleep(30)

    s = await Runner(tmp_path).run(RunSpec(run_id="d", tasks=[_T()], policies=[_P()], harnesses=[Hang()], k=1, attempt_timeout_s=0.1, retry_backoff_s=0.0))
    assert s.masked_terminal == 1


def test_v10_torn_tail_is_cut_before_appending(tmp_path):
    from crucible.runtime.journal import _append, _read_jsonl

    p = tmp_path / "results.jsonl"
    p.write_text('{"a": 1}\n{"b": ')
    _append(p, {"c": 3})
    assert _read_jsonl(p) == [{"a": 1}, {"c": 3}]


async def test_v8_v9_resume_chains_and_charges_only_new_calls(tmp_path):
    from crucible_pack_toy.longhorizon.task import BacklogTask
    from crucible_pack_toy.longhorizon.world import diligent

    from crucible.actors.policy import ContextConfig, EpisodeLimits, LLMToolPolicy
    from crucible.runtime.harness.native import NativeHarness, PolicyRef
    from crucible.runtime.inject import ReplayLLM

    class Costed(ReplayLLM):
        async def complete(self, messages, **kw):
            r = await super().complete(messages, **kw)
            r.cost_usd = 0.01
            self.spend.usd += 0.01
            return r

    t = BacklogTask(4, 40)
    acts = diligent(t.b)

    def make(attempt):
        c = {1: (150, ConnectionError("worker lost")), 2: (90, ConnectionError("worker lost again"))}.get(attempt.n)
        return {"agent": LLMToolPolicy("agent", Costed(acts, crash_at=c), context=ContextConfig(budget_tokens=5_000, keep_recent_turns=2))}

    s = await Runner(tmp_path).run(RunSpec(run_id="r", tasks=[t], policies=[PolicyRef(id="replay", make=make)],
                                           harnesses=[NativeHarness(limits=EpisodeLimits(max_steps=100_000, max_tool_errors=None), checkpoint_every=50)],
                                           k=1, retry_backoff_s=0.0))
    assert s.mean_reward == 1.0
    sd = next((tmp_path / "r" / "samples").iterdir())
    recs = {a.name: json.loads((a / "record.json").read_text()) for a in sorted(sd.glob("a*"))}
    assert recs["a3"]["meta"]["artifacts"]["resumed_from"] == "a2@step150"
    calls = sum(len(json.loads((sd / a / "calls.json").read_text())["agent"]) for a in recs if (sd / a / "calls.json").exists())
    assert s.spend["spent_run"] < 0.01 * calls  # restored calls are not charged twice
    assert recs["a1"]["cost_usd"] > 0  # a masked attempt paid for its calls


# ---------------------------------------------------------------- records and secrets (V12)


async def test_v12_secrets_never_reach_records(tmp_path):
    from crucible.actors.llm import LLMError
    from crucible.core.schema.run import Failure
    from crucible.runtime.journal import _append

    canary = "sk-ant-api03-" + secrets.token_hex(24)
    assert canary not in str(LLMError(f"upstream 401 for key {canary}"))
    assert canary not in Failure(kind="provider_unavailable", stage="agent", reason=f"x {canary}").reason
    p = tmp_path / "f.jsonl"
    _append(p, {"reason": f"echo {canary}", "url": "https://h/v1?key=" + secrets.token_hex(12), "token_ids": [1, 2, 3, 4, 5]})
    row = json.loads(p.read_text())
    assert canary not in p.read_text() and row["token_ids"] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------- environment results and edges (V13, V15, V17)


def test_v13_non_finite_return_is_not_full_reward():
    from crucible.runtime.control_task import ControlTask

    t = ControlTask.__new__(ControlTask)
    t.return_range = (0.0, 100.0)
    with pytest.raises(ValueError):
        t.score(float("nan"))


def test_v15_self_matching_lint():
    from crucible.runtime.remote import SELF_MATCH

    assert all(SELF_MATCH.search(c) for c in ("pgrep -af foo", "pgrep -fl foo", "pkill --full foo", "pkill -9 -f foo"))
    assert not SELF_MATCH.search('pkill -f "[f]oo"')


def test_v17_preflight_needs_every_gpu_to_match():
    from crucible.runtime.gpu import preflight

    def run(cmd):
        return (0, "tini\n") if "comm" in cmd else (0, "12.0\n8.9\n")

    assert preflight(run, gpu_arch="12.0") is not None


# ---------------------------------------------------------------- picks: the taxonomy


def test_pick_taxonomy_registered_unmapped_and_non_retryable():
    from crucible.core.schema.run import Failure, register_failure_kind
    from crucible.runtime.infra import RetryRule, failure

    register_failure_kind("m10test:wrong_answer", blame="policy")
    assert failure("m10test:wrong_answer", "agent").who == "policy"  # a pack kind keeps its blame
    assert failure("SomeHarborError", "agent").who == "unknown"  # unmapped: counted by the stop rule
    rr = RetryRule()
    assert not rr.allows(Failure(kind="crucible:stale_image", stage="verification"), 1)
    assert not rr.allows(Failure(kind="agent_timeout", stage="agent", mask=True, blame="harness"), 1)  # a masked limit stop
    assert not rr.allows(Failure(kind="crucible:harness_unsupported", stage="seed"), 1)
