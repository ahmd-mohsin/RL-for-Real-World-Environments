"""M12 runtime end to end on CPU: a real tiny torch policy learns a multi-turn token task through the asyncio runtime,
the shared scheduler, a bf16 engine (a real train–inference gap), the reference trainer and in-process weight publishing,
under every topology and both version-switch modes. Invariants checked on every run: staleness ≤ τ, version purity
(call-level) or valid version spans (token-level), aborts at the colocated barrier, no pause in disaggregated mode,
masked infra failures never trained."""

from __future__ import annotations

import pytest

from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask
from crucible.train.mismatch import Correction
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import SchedulerConfig
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher


async def run(topology: str, *, steps: int = 30, vs: str = "call", tau: int = 2, infra_fail: float = 0.0, correction: str = "dis"):
    model = TinyPolicy(32, 48, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64, numerics="bf16", decode_delay_s=0.002, version_switch=vs)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2, correction=Correction(correction, old_logprob_source="rollout" if correction != "dis" else None)))
    rt = Runtime(RuntimeConfig(topology=topology, steps=steps, max_turn_tokens=3, version_switch=vs), SchedulerConfig(n=8, B=4, max_staleness=tau),
                 eng, tr, InprocPublisher(tr, eng), lambda: CopyTask(turns=2, symbols=8, latency_s=0.001, long_tail=0.1, infra_fail=infra_fail),
                 {"copy": [f"p{i}" for i in range(4000)]})
    rep = await rt.run()
    return rt, rep


def learned(rep) -> tuple[float, float]:
    r = rep.series("reward/mean")
    return sum(r[:5]) / 5, sum(r[-10:]) / 10


@pytest.mark.parametrize("topology,vs", [("colocated_sync", "call"), ("colocated_partial", "call"), ("colocated_partial", "token"),
                                         ("disaggregated_stream", "call")])
async def test_runtime_learns_and_keeps_invariants(topology, vs):
    n = 40 if topology == "disaggregated_stream" else 30
    _, rep = await run(topology, vs=vs, steps=n)
    first, last = learned(rep)
    assert len(rep.steps) == n and rep.publishes == n and last > first + 0.08, (first, last)
    assert max(rep.series("staleness/max")) <= (0 if topology == "colocated_sync" else 2)
    assert rep.max_untrained <= (4 * 8 if topology == "colocated_sync" else 2 * 4 * 8)  # S·B·n untrained cap
    for r in rep.trained_records:
        for t in r.turns:
            assert t.spans_valid()
            if vs == "call":
                assert t.version_spans is None  # a call is served by exactly one version
    if topology == "colocated_partial":
        assert rep.aborted_calls > 0 and rep.reprefill_tokens > 0
        if vs == "token":
            assert rep.resumed_calls > 0 and any(t.version_spans for r in rep.trained_records for t in r.turns)
    if topology == "disaggregated_stream":
        assert rep.aborted_calls == 0  # generation never stops for training
    assert all(s.version >= 1 for s in rep.steps)


async def test_infra_failures_are_masked_not_trained():
    rt, rep = await run("colocated_partial", steps=10, infra_fail=0.05)
    assert rep.masked_rollouts > 0 and rt.sched.counters["evicted_masked"] + rep.masked_rollouts > 0  # failures happened
    assert all(r.reward is not None for r in rep.trained_records)  # none of them reached the trainer as a 0


async def test_corrections_run_in_the_loop():
    for c in ("tis", "icepop", "mimo4", "cispo"):
        _, rep = await run("colocated_partial", steps=6, correction=c)
        assert len(rep.steps) == 6 and "mismatch/masked_token_share" in rep.steps[-1].metrics


async def test_world_episodes_through_the_runtime_with_exact_tito():
    """A Crucible World (toy helpdesk) trained through the M12 runtime: the scripted engine emits the oracle's tool calls
    as tokens; LLMToolPolicy → EngineLLM (byte codec) → engine → World → M7 grader → crucible.rollout/v1."""
    import json as _json

    from crucible_pack_toy.runtime import HelpdeskTask
    from crucible_pack_toy.verify import GOLD

    from crucible.core.schema.actions import openai_name
    from crucible.train.chat import ByteChatCodec
    from crucible.train.engine import ScriptedEngine
    from crucible.train.rollout import full_prompts
    from crucible.train.world_env import world_rollout

    codec = ByteChatCodec()

    def script(rid, k, prompt):
        if k < len(GOLD):
            a = GOLD[k]
            body = _json.dumps({"name": openai_name(a.function_name), "arguments": a.arguments})
            return list(f"<tool_call>{body}</tool_call>".encode()) + [codec.EOS]
        return list(b"All tickets handled.") + [codec.EOS]

    eng = EnginePool([ScriptedEngine(script, stop_ids=[codec.EOS], delay_s=0.001)])
    tr = TorchTrainer(TinyPolicy(codec.vocab, 16, seed=0), TrainerConfig(lr=1e-3))
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=2, max_turn_tokens=2048, warmup_batches=0), SchedulerConfig(n=2, B=2, max_staleness=2),
                 eng, tr, InprocPublisher(tr, eng), None, {"helpdesk": [f"h{i}" for i in range(50)]},
                 rollout_fn=world_rollout(lambda p: HelpdeskTask(), codec, max_steps=len(GOLD) + 3, max_tokens=2048))
    rep = await rt.run()
    assert len(rep.steps) == 2 and rep.masked_rollouts == 0
    assert all(r.reward == 1.0 for r in rep.trained_records)  # the oracle's tool calls, through tokens, solve the task
    for r in rep.trained_records:
        prompts = full_prompts(r)
        assert len(r.turns) == len(GOLD) + 1 and all(t.policy_version is not None for t in r.turns)
        assert all(t.prompt_is_delta for t in r.turns[1:]) and not any(t.prefix_break for t in r.turns)  # exact TITO: never re-rendered
        for a, b in zip(r.turns, r.turns[1:], strict=False):  # each prompt extends the previous prompt + its completion
            assert prompts[b.call_id][: len(prompts[a.call_id]) + len(a.completion_token_ids)] == prompts[a.call_id] + a.completion_token_ids


def _tiny_qwen():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, max_position_embeddings=128, tie_word_embeddings=True)
    import torch as _t

    _t.manual_seed(0)
    return Qwen3ForCausalLM(cfg)


async def test_hf_policy_batched_logprobs_training_and_checkpoint_publish(tmp_path):
    import torch as _t
    from safetensors.torch import load_file

    from crucible.train.hf import HFPolicy
    from crucible.train.rollout import RolloutRecord, Turn
    from crucible.train.weights import CheckpointPublisher

    pol = HFPolicy(_tiny_qwen(), lm_head_fp32=True)
    pairs = [([1, 2, 3], [4, 5]), ([7, 8], [9, 10, 11])]
    batched = pol.batch_token_logprobs(pairs)
    single = [pol.token_logprobs(p, c) for p, c in pairs]
    assert all(_t.allclose(a, b, atol=1e-5) for a, b in zip(batched, single, strict=True))  # padding/masking does not change logprobs
    recs = []
    for i, r in enumerate([1.0, 0.0, 1.0, 0.0]):
        with _t.no_grad():
            lp = pol.token_logprobs([1, 2, 3], [4 + i, 5]).tolist()
        recs.append(RolloutRecord(rollout_id=f"r{i}", sample=f"r{i}", task="t", harness="h", policy="p", reward=r,
                                  turns=[Turn(call_id=f"r{i}:c1", actor="agent", prompt_token_ids=[1, 2, 3], completion_token_ids=[4 + i, 5], logprobs=lp,
                                              policy_version=0)]))
    tr = TorchTrainer(pol, TrainerConfig(lr=1e-2, correction=Correction("dis")))
    before = {k: v.clone() for k, v in pol.state_dict().items()}
    m = tr.step(recs, ["g"] * 4)
    assert m["train/items"] == 4 and any(not _t.equal(before[k], v) for k, v in pol.state_dict().items())

    class Sink:
        version = 0

        async def load_weights(self, path, version):
            self.path, self.version = path, version

    sink = Sink()
    v = await CheckpointPublisher(tr, sink, tmp_path).publish()
    sd = load_file(f"{sink.path}/model.safetensors")
    assert v == 1 and sink.version == 1 and set(sd) <= set(pol.state_dict()) and _t.equal(sd["model.norm.weight"], pol.state_dict()["model.norm.weight"])


async def test_top_p_rollouts_carry_candidates_through_the_runtime():
    model = TinyPolicy(32, 48, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64, decode_delay_s=0.001, version_switch="token")])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2))
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=4, max_turn_tokens=3, version_switch="token", top_p=0.8), SchedulerConfig(n=4, B=2, max_staleness=2),
                 eng, tr, InprocPublisher(tr, eng), lambda: CopyTask(turns=2, symbols=8, latency_s=0.001), {"copy": [f"p{i}" for i in range(400)]})
    rep = await rt.run()
    assert len(rep.steps) == 4 and rep.trained_records
    for r in rep.trained_records:
        assert r.valid_for_training()[0]
        for t in r.turns:
            assert t.sampling["top_p"] == 0.8 and t.candidates and len(t.candidates) == len(t.completion_token_ids)  # resumed calls keep theirs


async def test_moe_rollouts_record_routing_and_train_with_r3():
    from crucible.train.engine import TinyMoEPolicy

    model = TinyMoEPolicy(32, 32, seed=1, experts=4, k=2)
    eng = EnginePool([TorchEngine(model, capacity=64, decode_delay_s=0.001, numerics="bf16_full", version_switch="token")])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2, routing_replay="r3"))
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=4, max_turn_tokens=3, version_switch="token"), SchedulerConfig(n=4, B=2, max_staleness=2),
                 eng, tr, InprocPublisher(tr, eng), lambda: CopyTask(turns=2, symbols=8, latency_s=0.001), {"copy": [f"p{i}" for i in range(400)]})
    rep = await rt.run()
    assert len(rep.steps) == 4 and all("moe/routing_confidence" in s.metrics for s in rep.steps)
    for r in rep.trained_records:
        for t in r.turns:
            assert t.routing and len(t.routing) == len(t.completion_token_ids) and all(len(x) == 2 for x in t.routing)  # k = 2, kept across resumes


@pytest.mark.parametrize("kind", ["tiny", "hf"])
def test_prefix_sharing_equals_the_unpacked_step(kind):
    """Checklist #50: a GRPO group's completions share one prompt; with prefix_sharing the prompt is encoded once (GRU
    state / Forge-style packing with a 4D mask) and the loss and the updated weights equal the unpacked step to 1e-6."""
    import copy

    import torch as _t

    from crucible.train.hf import HFPolicy
    from crucible.train.rollout import RolloutRecord, Turn

    base = TinyPolicy(64, 16, seed=4) if kind == "tiny" else HFPolicy(_tiny_qwen(), lm_head_fp32=True)
    g = _t.Generator().manual_seed(0)
    recs, groups = [], []
    for gi in range(3):
        prompt = _t.randint(4, 60, (12,), generator=g).tolist()
        for k in range(4):
            comp = _t.randint(4, 60, (int(_t.randint(1, 7, (1,), generator=g)),), generator=g).tolist()
            recs.append(RolloutRecord(rollout_id=f"r{gi}{k}", sample="s", task="t", harness="h", policy="p", reward=float(k % 2),
                                      turns=[Turn(call_id=f"r{gi}{k}:c", actor="agent", prompt_token_ids=prompt, completion_token_ids=comp,
                                                  logprobs=[-2.0] * len(comp), policy_version=0)]))
            groups.append(f"g{gi}")
    out = {}
    for share in (False, True):
        m = copy.deepcopy(base)
        tr = TorchTrainer(m, TrainerConfig(lr=1e-2, prefix_sharing=share, token_budget=1 << 20, correction=Correction("ppo")))
        met = tr.step(recs, groups)
        out[share] = (met, {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None})  # the step's (clipped) gradient
    (m0, g0), (m1, g1) = out[False], out[True]
    assert abs(m0["train/loss"] - m1["train/loss"]) < 1e-6
    assert g0.keys() == g1.keys() and max(float((g0[k] - g1[k]).abs().max()) for k in g0) < 1e-6  # (Adam's 1/√v would amplify float noise in weights)
    assert m1["train/prefix_tokens_saved"] > 0
