"""M10 trajectories and rollout records: every model call, World action, ATIF step and steps.jsonl row joins; a training
rollout without complete token records is masked (``crucible:capture_incomplete``), never guessed."""

from __future__ import annotations

import json
from types import SimpleNamespace

from crucible_pack_toy.runtime import HelpdeskTask

from crucible.actors.llm import LLMResponse, token_record
from crucible.actors.policy import LLMToolPolicy
from crucible.core.schema.reward import StepRow
from crucible.core.schema.rollout import RolloutRecord
from crucible.interop.harbor.reconcile import reconcile
from crucible.interop.harbor.trajectories import align, to_atif
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.inject import FakeLLM
from crucible.runtime.run import Runner, RunSpec
from crucible.workflow.steps import segment

TOK = {"prompt_token_ids": [1, 2, 3], "completion_token_ids": [7, 8], "logprobs": [-0.1, -0.2], "policy_version": 3}


def toks(n: int) -> list[dict]:
    """Engine-style records: each prompt extends the previous prompt + completion (token-in-token-out)."""
    out, prompt = [], [1, 2, 3]
    for i in range(n):
        out.append({**TOK, "prompt_token_ids": list(prompt)})
        prompt = prompt + TOK["completion_token_ids"] + [100 + i]
    return out


def script(tokens: bool):
    t = toks(4) if tokens else [None] * 4
    u = {"prompt_tokens": 100, "completion_tokens": 2}
    return [LLMResponse(text="", tool_calls=[{"id": "x1", "name": "tickets__list_tickets", "arguments": {"status": "open"}}], usage=u, tokens=t[0]),
            LLMResponse(text="", usage=u, tokens=t[1]),  # blank: the policy nudges past it, no action
            LLMResponse(text="", tool_calls=[{"id": "x2", "name": "tickets__get_ticket", "arguments": {"ticket_id": 1}},
                                             {"id": "x3", "name": "tickets__get_ticket", "arguments": {"ticket_id": 2}}], usage=u, tokens=t[2]),
            LLMResponse(text="All reviewed.", usage=u, tokens=t[3])]


def ref(tokens: bool) -> PolicyRef:
    return PolicyRef(id=f"fake:tok{int(tokens)}", make=lambda a: {"agent": LLMToolPolicy("agent", FakeLLM(script(tokens)))}, config={"tokens": tokens})  # M10 lab audit R-4


async def run(tmp_path, tokens: bool, mode: str = "eval", run_id: str = "r"):
    s = await Runner(tmp_path).run(RunSpec(run_id=run_id, tasks=[HelpdeskTask()], policies=[ref(tokens)], harnesses=[NativeHarness()], k=1, rollout_mode=mode))
    adirs = sorted((tmp_path / run_id / "samples").glob("*/a*"))
    return s, adirs


async def test_trajectory_joins_calls_actions_and_steps(tmp_path):
    _s, adirs = await run(tmp_path, tokens=True)
    a = adirs[0]
    calls = json.loads((a / "calls.json").read_text())["agent"]
    assert [c["actions"] for c in calls] == [["tool:x1"], [], ["tool:x2", "tool:x3"], ["done"]]
    traj = json.loads((a / "trajectory.agent.json").read_text())
    agent = [s for s in traj["steps"] if s["source"] == "agent"]
    assert [s["extra"]["call_id"] for s in agent] == ["agent:c1", "agent:c2", "agent:c3", "agent:c3", "agent:c4"]
    assert [s.get("llm_call_count") for s in agent] == [1, 1, 1, 0, 1]  # one metrics block per call
    assert agent[1]["extra"]["no_action"] and agent[1]["metrics"]["completion_token_ids"] == [7, 8]  # the blank call has its own step
    assert [s["step_id"] for s in traj["steps"]] == list(range(1, len(traj["steps"]) + 1))
    fm = traj["final_metrics"]
    assert fm["total_prompt_tokens"] == 400 and fm["extra"]["unjoined_steps"] == [] and fm["extra"]["unconsumed_actions"] == 0
    ro = RolloutRecord.model_validate_json((a / "rollout.json").read_text())
    assert ro.reward is not None and len(ro.turns) == 4 and ro.turns[2].atif_step_ids == [s["step_id"] for s in agent[2:4]]
    assert ro.valid_for_training() == (True, "ok") and ro.oldest_version() == 3
    assert ro.turns[0].prompt_token_ids == [1, 2, 3] and not ro.turns[0].prompt_is_delta  # the first call of a segment: the whole prompt
    assert ro.turns[1].prompt_is_delta and ro.turns[1].prompt_token_ids == [100]  # later calls: only what they add (linear storage)


async def test_steps_jsonl_rows_align_to_atif(tmp_path):
    from harbor.models.trajectories import Trajectory

    from crucible.core.schema.trace import Trace

    _s, adirs = await run(tmp_path, tokens=False)
    a = adirs[0]
    trace = Trace.model_validate_json((a / "trace.json").read_text())
    calls = json.loads((a / "calls.json").read_text())["agent"]
    traj = to_atif(trace, "agent", calls=calls)
    assert Trajectory.model_validate_json(traj.model_dump_json())  # Harbor's own validator accepts it
    steps = [s for s in segment(trace) if s.actor == "agent"]
    rows = [StepRow(step=s.index, phi=0.0, delta_phi=0.0) for s in steps]
    out = align(rows, steps, traj)
    assert out and all(r.atif_step_id for r in out) and {r.call_id for r in out} <= {c["call_id"] for c in calls}


async def test_reconcile_and_train_mode_masking(tmp_path):
    s, adirs = await run(tmp_path, tokens=False, mode="train", run_id="t")
    ro = RolloutRecord.model_validate_json((adirs[0] / "rollout.json").read_text())
    assert ro.mask_sample and ro.reward is None and ro.failure_kind == "crucible:capture_incomplete" and ro.blame == "infra"
    assert s.scored == 0 and s.by_failure.get("crucible:capture_incomplete")  # masked in the run as well, never a 0
    s2, adirs2 = await run(tmp_path, tokens=True, mode="train", run_id="t2")
    assert s2.scored == 1 and not RolloutRecord.model_validate_json((adirs2[0] / "rollout.json").read_text()).mask_sample
    _, e = await run(tmp_path, tokens=False, mode="eval", run_id="e")
    assert not RolloutRecord.model_validate_json((e[0] / "rollout.json").read_text()).mask_sample  # evaluation never needs tokens
    from harbor.models.trajectories import Trajectory

    traj = Trajectory.model_validate_json((adirs2[0] / "trajectory.agent.json").read_text())
    # M10 lab audit R-1: the capture cross-check is the OpenEnv reconcile port. The trajectory has 5 agent steps for 4
    # calls (llm_call_count [1, 1, 1, 0, 1], 2 completion tokens per call); the llm_call_count 0 step is a second action
    # of call 3, not a model call
    atif = json.loads(traj.model_dump_json())
    call = lambda n, t=2: {"call_id": f"c{n}", "usage": {"completion_tokens": t}}  # noqa: E731
    ok = reconcile(atif, [call(i) for i in range(4)])
    assert (ok.level, ok.code) == ("info", "turns_match")
    assert reconcile(atif, [call(i) for i in range(3)]).level == "fatal"  # ATIF logged a call the capture never saw
    bad = reconcile(atif, [call(0), call(1, 9), call(2), call(3)])
    assert (bad.level, bad.code) == ("fatal", "turn_mismatch")


def test_token_record_parsing():
    choice = SimpleNamespace(model_extra={"token_ids": [5, 6]}, logprobs=SimpleNamespace(content=[SimpleNamespace(logprob=-0.5), SimpleNamespace(logprob=-1.0)]))
    resp = SimpleNamespace(model_extra={"prompt_token_ids": [1, 2]})
    assert token_record(resp, choice, 7) == {"prompt_token_ids": [1, 2], "completion_token_ids": [5, 6], "logprobs": [-0.5, -1.0], "policy_version": 7}
    assert token_record(SimpleNamespace(model_extra={}), SimpleNamespace(model_extra={}, logprobs=None), 7) is None  # hosted API: nothing, never re-tokenized


async def test_blank_usage_tolerant_mode_books_uncertain_spend(tmp_path, monkeypatch):
    import pytest

    from crucible.actors.llm import LLMClient, UsageBlank

    msg = SimpleNamespace(content="hi", tool_calls=None, model_dump=lambda exclude_none=True: {})
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop", model_extra={}, logprobs=None)], usage=None, model="m", id="r1", model_extra={})

    class Fake:
        class chat:
            class completions:
                @staticmethod
                async def create(**kw):
                    return resp

    strict = LLMClient(cache_dir=tmp_path / "c1", cache_mode="off")
    strict._client = Fake
    with pytest.raises(UsageBlank):
        await strict.complete([{"role": "user", "content": "x"}])
    tol = LLMClient(cache_dir=tmp_path / "c2", cache_mode="off", require_usage=False)
    tol._client = Fake
    monkeypatch.setitem(tol.config.prices, tol.model, (1.0, 4.0))
    r = await tol.complete([{"role": "user", "content": "x"}], max_tokens=1000)
    assert r.usage["source"] == "none" and r.usage["prompt_tokens"] == 0 and tol.spend.blank_usage_calls == 1 and tol.spend.uncertain_usd >= 1000 / 1e6 * 4.0  # worst case, not $0


def test_provider_policy_version_and_named_gaps(tmp_path):
    """A served checkpoint's version comes from the provider config (lab M10: every train rollout was masked because an
    external vLLM had no version stamp), and an invalid record names what is missing."""
    from crucible.actors.llm import LLMClient
    from crucible.core.config import Config, LLMProvider
    from crucible.core.schema.rollout import Turn

    cfg = Config(llm_providers={"lab": LLMProvider(base_url="http://127.0.0.1:1/v1", policy_version=0)}, models={"p": "lab:m"}, data_dir=str(tmp_path))
    assert LLMClient("p", config=cfg, cache_mode="off").policy_version == 0
    t = Turn(call_id="c1", actor="agent", prompt_token_ids=[1, 2], completion_token_ids=[3, 4], logprobs=[-0.1, -0.2], policy_version=None)
    assert t.missing() == ["policy_version (set the provider's policy_version for a served checkpoint)"]
    assert Turn(call_id="c2", actor="agent", prompt_token_ids=[1], completion_token_ids=[3], logprobs=[-0.1], policy_version=0).missing() == []
