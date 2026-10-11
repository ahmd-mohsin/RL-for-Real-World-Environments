"""M12 bridges: a crucible.rollout/v1 record flattens into per-segment training sequences exactly (masks, logprobs,
advantages, versions), and each trainer adapter produces its framework's fields consistently."""

from __future__ import annotations

from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.train.bridges.adapters import rllm_steps, skyrl_generator_output, trl_rollout, verl_agent_loop_output
from crucible.train.bridges.common import flatten


def record() -> RolloutRecord:
    # segment 0: two chained calls (call 2's prompt = call 1's prompt + completion + obs [9, 9]); segment 1 after compaction
    return RolloutRecord(rollout_id="r", sample="r", task="t", harness="h", policy="p", reward=1.0, turns=[
        Turn(call_id="a:c1", actor="agent", segment=0, prompt_token_ids=[1, 2], completion_token_ids=[5, 6], logprobs=[-0.1, -0.2], policy_version=3,
             advantage=0.5),
        Turn(call_id="a:c2", actor="agent", segment=0, prompt_is_delta=True, prompt_token_ids=[9, 9], completion_token_ids=[7], logprobs=[-0.3],
             policy_version=3, version_spans=None),
        Turn(call_id="a:c3", actor="agent", segment=1, kind="compaction", prompt_token_ids=[1, 2, 42], completion_token_ids=[8, 8], logprobs=[-0.4, -0.5],
             policy_version=4, version_spans=[(0, 4), (1, 5)])])


def test_flatten_segments_masks_and_advantages():
    seqs = flatten(record(), seq_adv=0.25, turn_weight=1.0)
    assert len(seqs) == 2  # one per compaction segment
    s0, s1 = seqs
    assert s0.prompt_ids == [1, 2] and s0.response_ids == [5, 6, 9, 9, 7] and s0.response_mask == [1, 1, 0, 0, 1]
    assert s0.logprobs == [-0.1, -0.2, 0.0, 0.0, -0.3] and s0.advantages == [0.75, 0.75, 0.0, 0.0, 0.25]  # turn credit only on its own call
    assert s0.call_spans == [("a:c1", 0, 2), ("a:c2", 4, 5)] and s0.num_turns == 2
    assert s1.prompt_ids == [1, 2, 42] and s1.versions == [4, 5]  # a call that crossed a publish keeps per-token versions
    assert flatten(record().model_copy(update={"mask_sample": True})) == []  # masked rollouts never become training data


def test_adapters_shapes():
    seqs = flatten(record(), seq_adv=0.25)
    v = verl_agent_loop_output(seqs[0])
    assert len(v["response_ids"]) == len(v["response_mask"]) == len(v["response_logprobs"]) == len(v["extra_fields"]["crucible_token_advantages"])
    sk = skyrl_generator_output(seqs)
    assert all(len(r) == len(m) for r, m in zip(sk["rewards"], sk["loss_masks"], strict=True))
    assert sk["rewards"][0][-1] == 1.0 and sum(sk["rewards"][0]) == 1.0  # outcome on the last sampled token (GRPO sums token rewards)
    t = trl_rollout(seqs)
    assert t["env_mask"][0] == [1, 1, 0, 0, 1]
    steps = rllm_steps(seqs[0])
    assert [s["id"] for s in steps] == ["a:c1", "a:c2"] and steps[1]["prompt_ids"] == [1, 2, 5, 6, 9, 9] and steps[1]["advantage"] == [0.25]


def test_verl_crucible_delta_estimator():
    import torch

    from crucible.train.bridges.verl_bridge import crucible_delta, token_rewards

    s0, _ = flatten(record(), seq_adv=0.0)
    r = token_rewards(s0, turn_credit={"a:c1": 0.1, "a:c2": -0.2})
    assert r == [0.1, 0.1, 0.0, 0.0, 1.0]  # credit on call 1's tokens; the outcome on the last sampled token
    other = [0.0, 0.0, 0.0, 0.0, 0.0]  # a group member that failed (R = 0), same mask
    tlr = torch.tensor([r, other])
    mask = torch.tensor([s0.response_mask, s0.response_mask])
    adv, _ = crucible_delta(tlr, mask, index=["g", "g"])
    assert torch.allclose(adv[0], torch.tensor([0.6, 0.6, 0.0, 0.0, 0.5])) and torch.allclose(adv[1], torch.tensor([-0.5, -0.5, 0.0, 0.0, -0.5]))
    masked = torch.zeros(1, 5)
    adv2, _ = crucible_delta(torch.cat([tlr, torch.zeros(1, 5)]), torch.cat([mask, masked.long()]), index=["g", "g", "g"])
    assert torch.allclose(adv2[:2], adv) and adv2[2].abs().sum() == 0  # a masked rollout neither trains nor moves the baseline


def test_trl_rollout_func_runs_world_episodes():
    import json as _json

    from crucible_pack_toy.runtime import HelpdeskTask
    from crucible_pack_toy.verify import GOLD

    from crucible.core.schema.actions import openai_name
    from crucible.train.bridges.trl_bridge import make_rollout_func
    from crucible.train.chat import ByteChatCodec
    from crucible.train.engine import GenRequest, ScriptedEngine
    from crucible.train.world_env import world_rollout

    codec = ByteChatCodec()

    def script(rid, k, prompt):
        k = bytes(t for t in prompt if t < 256).count(b"<tool_call>")  # stateless: position from the context, like a real model
        if k < len(GOLD):
            return list(f"<tool_call>{_json.dumps({'name': openai_name(GOLD[k].function_name), 'arguments': GOLD[k].arguments})}</tool_call>".encode()) + [codec.EOS]
        return list(b"done") + [codec.EOS]

    eng = ScriptedEngine(script, stop_ids=[codec.EOS])
    calls = {"n": 0}

    def gen_factory(trainer):
        async def gen(ids, max_tokens, temperature):
            calls["n"] += 1
            return await eng.generate(GenRequest(rid=f"x{trainer}{calls['n'] // 100}", call_id=str(calls["n"]), prompt_ids=ids, max_tokens=max_tokens))
        return gen

    rf = make_rollout_func(world_rollout(lambda p: HelpdeskTask(), codec, max_steps=len(GOLD) + 3, max_tokens=2048), num_generations=2, gen_factory=gen_factory)
    out = rf(["p0"], trainer="T")
    assert len(out["completion_ids"]) == 2 and all(len(c) == len(m) for c, m in zip(out["completion_ids"], out["env_mask"], strict=True))
    assert out["crucible_reward"] == [1.0, 1.0]
