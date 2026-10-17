"""M12 audit, framework bridges (docs/audit/2026-10-12: M12-study-defects-H4 V19-V28, M12-sweep-G1 H7, G2 H6/H7 + pick 8,
G4 H4-H6 + picks 4-6, G5 D4). Every bridge test runs the pinned upstream framework's own code (``_upstream.upstream``:
the files of RL-Environments/06-frameworks-and-env-code executed unchanged, with only their missing third-party imports
stubbed), so a bridge is checked against exactly what that framework consumes."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
import threading
import time
import tomllib
from contextlib import nullcontext
from importlib.metadata import EntryPoint
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.interop.verifiers.v1_import import trace_to_record
from crucible.train.bridges.common import flatten
from crucible.world.intercept import RuleInterceptor

sys.path.insert(0, str(Path(__file__).parent))
from _upstream import FRAMEWORKS, ROOTS, upstream

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


class AttrDict(dict):
    __getattr__ = dict.get


def _record(rid: str = "r", reward: float | None = 1.0, phis: tuple[float, float] = (0.0, 0.0), **kw) -> RolloutRecord:
    """Two chained calls: call 2's prompt = call 1's prompt + completion + observation [9, 9]."""
    return RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", reward=reward, turns=[
        Turn(call_id=f"{rid}:c1", actor="agent", prompt_token_ids=[1, 2], completion_token_ids=[5, 6], logprobs=[-0.1, -0.2], policy_version=3,
             advantage=phis[0]),
        Turn(call_id=f"{rid}:c2", actor="agent", prompt_is_delta=True, prompt_token_ids=[9, 9], completion_token_ids=[7, 8], logprobs=[-0.3, -0.4],
             policy_version=4, advantage=phis[1])], **kw)


# ---------------------------------------------------------------------------------------------------------------- verl

VERL_MODULES = ["verl.trainer.ppo.core_algos", "verl.trainer.ppo.ray_trainer", "verl.trainer.ppo.v1.utils", "verl.experimental.agent_loop.agent_loop"]


def _forget_crucible_verl() -> None:
    for k in ("crucible.train.bridges.verl_plugin", "crucible.train.bridges.verl_agent_loop"):
        sys.modules.pop(k, None)


def _verl_plugin_discovery(monkeypatch) -> None:
    """verl's own entry-point loop (``verl/__init__.py``: ``_plugins_policy`` … ``_ep.load()``), executed as written, over
    the ``verl.plugins`` entry points this checkout's pyproject declares (the installed metadata may predate it)."""
    import importlib.metadata

    eps = tomllib.loads(PYPROJECT.read_text()).get("project", {}).get("entry-points", {}).get("verl.plugins", {})
    monkeypatch.setattr(importlib.metadata, "entry_points",
                        lambda group=None: [EntryPoint(n, v, "verl.plugins") for n, v in eps.items()] if group == "verl.plugins" else [])
    src = (FRAMEWORKS / ROOTS["verl"] / "verl" / "__init__.py").read_text()
    block = src[src.index("_plugins_policy = "): src.index('if os.getenv("VERL_USE_MODELSCOPE"')]
    exec(compile(block, "verl/__init__.py", "exec"), {"os": __import__("os"), "logging": __import__("logging")})  # noqa: S102 - verl's own source, run as written


@pytest.fixture
def verl(monkeypatch):
    _forget_crucible_verl()
    with upstream("verl", VERL_MODULES) as ns:
        _verl_plugin_discovery(monkeypatch)
        yield ns
    _forget_crucible_verl()


def _verl_batch(dicts: list[dict], uids: list[str]):
    """What verl v1's ``_compute_advantage`` builds from the TransferQueue rows (``v1/trainer_base.py:1743-1760``):
    right-padded ``rm_scores``/``response_mask``, ``token_level_scores = rm_scores``, ``token_level_rewards`` = scores."""
    import numpy as np

    T = max(d["responses"].size(0) for d in dicts)

    def pad(t):
        return torch.cat([t, t.new_zeros(T - t.size(0))])

    rm = torch.stack([pad(d["rm_scores"]) for d in dicts])
    batch = {"response_mask": torch.stack([pad(d["response_mask"]) for d in dicts]), "rm_scores": rm, "token_level_scores": rm,
             "token_level_rewards": rm}
    return SimpleNamespace(batch=batch, non_tensor_batch={"uid": np.array(uids, dtype=object)})


def _verl_advantages(verl, dicts, uids, sessions=None, config=None):
    from crucible.train.bridges.verl_bridge import crucible_delta

    if "crucible_delta" not in verl.core_algos.ADV_ESTIMATOR_REGISTRY:  # (registration itself is V20's test: this checks the math)
        verl.core_algos.register_adv_est("crucible_delta")(crucible_delta)
    data = _verl_batch(dicts, uids)
    keys = [f"{u}_{s}_{i}" for i, (u, s) in enumerate(zip(uids, sessions or range(len(uids)), strict=True))]
    data = verl.utils.compute_advantage_for_multi_trajectories(data, batch_keys=keys, adv_estimator="crucible_delta",
                                                               config=config if config is not None else AttrDict(use_kl_in_reward=False))
    return data.batch["advantages"]


def test_V19_verl_adapter_is_a_valid_agent_loop_output(verl):
    s = flatten(_record())[0]
    from crucible.train.bridges.adapters import verl_agent_loop_output

    out = verl.agent_loop.AgentLoopOutput(**verl_agent_loop_output(s))  # upstream pydantic model: metrics is required (agent_loop.py:107)
    d = out.as_dict()
    assert d["responses"].tolist() == [5, 6, 9, 9, 7, 8] and d["response_mask"].tolist() == [1, 1, 0, 0, 1, 1]
    assert d["rollout_log_probs"].tolist() == pytest.approx([-0.1, -0.2, 0.0, 0.0, -0.3, -0.4])
    assert out.extra_fields["min_global_steps"] == 3 and out.extra_fields["max_global_steps"] == 4  # staleness tags (agent_loop_tq.py:224-227)
    assert float(d["rm_scores"].sum()) == 1.0


def test_V20_verl_plugin_entry_point_registers_estimator_and_agent_loop(verl):
    # registered by verl's own import-time plugin discovery, as in every Ray actor that imports verl (main_ppo.py:102)
    from crucible.train.bridges.verl_bridge import crucible_delta

    assert verl.core_algos.get_adv_estimator_fn("crucible_delta") is crucible_delta
    reg = verl.agent_loop._agent_loop_registry
    assert reg["crucible_agent"] == {"_target_": "crucible.train.bridges.verl_agent_loop.CrucibleAgentLoop"}


def test_V21_turn_credit_reaches_verl_advantages(verl):
    """Two rollouts of one uid with M6 turn credit: the advantages verl's v1 trainer computes from the rows the Crucible
    agent loop writes equal Crucible's own formula: (R − mean R) + the call's group-processed credit on its tokens."""
    from crucible.train.bridges.adapters import verl_agent_loop_output
    from crucible.train.bridges.verl_agent_loop import CrucibleAgentLoopOutput
    from crucible.train.bridges.verl_bridge import token_rewards
    from crucible.train.credit import turn_credit_map

    recs = [_record("a", 1.0, phis=(0.6, 0.2)), _record("b", 0.0, phis=(0.1, -0.3))]
    credit = turn_credit_map(recs, ["u", "u"], eta=0.5)
    assert any(abs(v) > 1e-6 for v in credit.values())
    dicts = []
    for r in recs:
        s = flatten(r)[0]
        per = {cid: c for (rid, cid), c in credit.items() if rid == r.rollout_id}
        dicts.append(CrucibleAgentLoopOutput(**verl_agent_loop_output(s), crucible_token_rewards=token_rewards(s, turn_credit=per)).as_dict())
        assert float(dicts[-1]["rm_scores"].sum()) == pytest.approx(r.reward)  # verl's metrics and outcome estimators still read R
    adv = _verl_advantages(verl, dicts, ["u", "u"])
    for k, r in enumerate(recs):
        a_out = r.reward - 0.5
        c1, c2 = credit[(r.rollout_id, f"{r.rollout_id}:c1")], credit[(r.rollout_id, f"{r.rollout_id}:c2")]
        want = [a_out + c1, a_out + c1, 0.0, 0.0, a_out + c2, a_out]  # the final sampled token carries the outcome only
        assert adv[k].tolist() == pytest.approx(want, abs=1e-6)


def test_V28_H6_verl_outcome_kept_when_a_response_ends_on_observation_tokens(verl):
    # verl puts the scalar at the last response position (agent_loop.py:143-148), here a tool-output token (mask 0)
    rows = [verl.agent_loop.AgentLoopOutput(prompt_ids=[1], response_ids=[5, 6, 9, 9], response_mask=[1, 1, 0, 0], reward_score=r,
                                            metrics={}).as_dict() for r in (1.0, 0.0)]
    adv = _verl_advantages(verl, rows, ["g", "g"])
    assert adv.tolist() == [[0.5, 0.5, 0.0, 0.0], [-0.5, -0.5, 0.0, 0.0]]


def test_V28_verl_kl_in_reward_refused_and_invalid_reward_value_masked(verl):
    rows = [verl.agent_loop.AgentLoopOutput(prompt_ids=[1], response_ids=[5, 6], response_mask=[1, 1], reward_score=r, metrics={}).as_dict()
            for r in (1.0, 0.0, -999.0)]
    with pytest.raises(ValueError, match="use_kl_in_reward"):
        _verl_advantages(verl, rows, ["g"] * 3, config=AttrDict(use_kl_in_reward=True))
    adv = _verl_advantages(verl, rows, ["g"] * 3, config=AttrDict(use_kl_in_reward=False, invalid_reward_value=-999))
    assert adv.tolist() == [[0.5, 0.5], [-0.5, -0.5], [0.0, 0.0]]  # MiMo-verl's sentinel neither trains nor moves the mean


def _agent_loop(verl, server, rollout, **kw):
    from crucible.train.bridges.verl_agent_loop import CrucibleAgentLoop

    rc = AttrDict(prompt_length=4096, response_length=4096, n=3, top_p=1.0, top_k=-1)
    cfg = AttrDict(actor_rollout_ref=AttrDict(rollout=rc))
    A = verl.agent_loop
    return CrucibleAgentLoop(trainer_config=A.DictConfigWrap(cfg), server_manager=server, tokenizer=SimpleNamespace(pad_token_id=0),
                             processor=None, dataset_cls=None, data_config=A.DictConfigWrap(AttrDict()), rollout=rollout, **kw)


def test_G2pick8_crucible_agent_loop_runs_world_episodes_through_verl(verl):
    """verl's worker contract end to end: the registered agent loop is built the way ``_run_agent_loop`` builds it, runs
    a toy Crucible World per session against a verl-shaped ``LLMServerClient`` (TokenOutput fields), and its rows go
    through verl's ``as_dict`` and v1 advantage path. A masked session keeps its place with no trainable token."""
    import asyncio

    from crucible_pack_toy.runtime import HelpdeskTask
    from crucible_pack_toy.verify import GOLD

    from crucible.core.schema.actions import openai_name
    from crucible.train.chat import ByteChatCodec
    from crucible.train.world_env import world_rollout

    codec = ByteChatCodec()

    class Server:  # verl LLMServerClient.generate (llm_server.py:99-164) -> TokenOutput (replica.py:34-46)
        def __init__(self):
            self.params = []

        async def generate(self, request_id, *, prompt_ids, sampling_params, **_):
            self.params.append(dict(sampling_params))
            session = int(request_id.split(":")[2])
            k = bytes(t for t in prompt_ids if t < 256).count(b"<tool_call>")
            if session == 1 or k >= len(GOLD):  # session 1 gives up at once (R = 0)
                toks = list(b"done") + [codec.EOS]
            else:
                toks = list(f"<tool_call>{json.dumps({'name': openai_name(GOLD[k].function_name), 'arguments': GOLD[k].arguments})}</tool_call>".encode()) \
                    + [codec.EOS]
            return SimpleNamespace(token_ids=toks, log_probs=[-0.05] * len(toks), extra_fields={"global_steps": 7}, stop_reason="completed",
                                   num_preempted=None)

    base = world_rollout(lambda p: HelpdeskTask(), codec, max_steps=len(GOLD) + 3, max_tokens=2048)

    async def rollout(rid, prompt, seed, gen, *, temperature=1.0):
        assert prompt == "ticket-7"
        return None if seed == 2 else await base(rid, prompt, seed, gen, temperature=temperature)  # session 2: masked

    server = Server()
    loop = _agent_loop(verl, server, lambda _loop: rollout)
    kw = {"uid": "u0", "raw_prompt": [{"role": "user", "content": "ticket-7"}], "agent_name": "crucible_agent"}

    async def sessions():
        return await asyncio.gather(*[loop.run({"temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": False}, session_id=i, **kw)
                                      for i in range(3)])

    outs = asyncio.run(sessions())
    assert all(isinstance(o, list) and len(o) == 1 for o in outs)
    rows = [o[0].as_dict() for o in outs]
    assert all(p["logprobs"] is True and p["max_tokens"] == 2048 for p in server.params)  # behaviour logprobs always requested
    assert rows[0]["response_mask"].sum() > 0 and rows[2]["response_mask"].tolist() == [0]
    assert outs[0][0].extra_fields["min_global_steps"] == 7 and outs[2][0].extra_fields["crucible_masked"]
    assert [float(r["rm_scores"].sum()) for r in rows] == [1.0, 0.0, 0.0]
    adv = _verl_advantages(verl, rows, ["u0"] * 3)
    m = [r["response_mask"].bool() for r in rows]
    assert torch.allclose(adv[0][: m[0].numel()][m[0]], torch.tensor(0.5)) and torch.allclose(adv[1][: m[1].numel()][m[1]], torch.tensor(-0.5))
    assert adv[2].abs().sum() == 0  # the masked session: no trainable token, out of the mean (0.5, not 1/3)


def test_G2pick8_verl_agent_loop_refuses_truncated_sampling(verl):
    from crucible.train.bridges.verl_agent_loop import CrucibleAgentLoop

    rc = AttrDict(prompt_length=64, response_length=64, n=2, top_p=0.9, top_k=-1)
    A = verl.agent_loop
    with pytest.raises(ValueError, match="top_p"):
        CrucibleAgentLoop(trainer_config=A.DictConfigWrap(AttrDict(actor_rollout_ref=AttrDict(rollout=rc))), server_manager=None,
                          tokenizer=None, processor=None, dataset_cls=None, data_config=A.DictConfigWrap(AttrDict()), rollout="x:y")


def test_G4S5_flatten_refuses_token_incomplete_records():
    r = _record()
    r.turns[1].logprobs = None  # used to be zero-filled: trained as if π_rollout = 1
    assert flatten(r) == []
    r = _record()
    r.turns[1].policy_version = None
    assert flatten(r) == []
    assert flatten(_record(), frozen_actors=["agent"]) == []  # nothing trainable
    ok = _record()
    ok.turns[0].actor = "judge"
    ok.turns[0].logprobs = None  # a frozen actor's call is context: its logprobs are never read
    assert [s.actor for s in flatten(ok, frozen_actors=["judge"])] == ["judge", "agent"]


def test_V21_agent_loop_sessions_meet_for_m6_group_credit(verl):
    """With ``turn_weight`` the loop's sessions of one uid apply M6's group rule together (one rendezvous per uid)."""
    import asyncio

    from crucible.train.credit import turn_credit_map

    recs = {0: _record("a", 1.0, phis=(0.6, 0.2)), 1: _record("b", 0.0, phis=(0.1, -0.3))}

    async def rollout(rid, prompt, seed, gen, *, temperature=1.0):
        await asyncio.sleep(0.01 * (1 - seed))  # sessions finish in either order
        return recs[seed]

    loop = _agent_loop(verl, None, lambda _l: rollout, turn_weight=0.5)
    loop.rollout_config["n"] = 2

    async def sessions():
        return await asyncio.gather(*[loop.run({"temperature": 1.0}, uid="u", session_id=i, raw_prompt=[]) for i in range(2)])

    rows = [o[0].as_dict() for o in asyncio.run(sessions())]
    adv = _verl_advantages(verl, rows, ["u", "u"])
    credit = turn_credit_map(list(recs.values()), ["u", "u"], eta=0.5)
    for k, r in recs.items():
        a_out, c1, c2 = r.reward - 0.5, credit[(r.rollout_id, f"{r.rollout_id}:c1")], credit[(r.rollout_id, f"{r.rollout_id}:c2")]
        assert adv[k].tolist() == pytest.approx([a_out + c1, a_out + c1, 0.0, 0.0, a_out + c2, a_out], abs=1e-6)


# ----------------------------------------------------------------------------------------------------------------- TRL

GRPO = FRAMEWORKS / ROOTS["trl"] / "trl" / "trainer" / "grpo_trainer.py"


@pytest.fixture(scope="module")
def trl():
    with upstream("trl", ["trl.generation.vllm_generation", "trl.trainer.utils", "trl.trainer.grpo_trainer"]) as ns:
        mods = SimpleNamespace(vg=ns.vllm_generation, utils=ns.utils, grpo=ns.grpo_trainer)
    mods.vg.SamplingParams = lambda **k: SimpleNamespace(**k)  # vllm.SamplingParams: a plain record of the kwargs
    return mods


def _block(start: str, end: str) -> str:
    """Upstream source lines from the line holding ``start`` through the line holding ``end`` (dedented)."""
    lines = Path(GRPO).read_text().splitlines()
    a = next(i for i, ln in enumerate(lines) if start in ln)
    b = next(i for i in range(a, len(lines)) if end in lines[i])
    return textwrap.dedent("\n".join(lines[a:b + 1]))


# ---- fake vLLM engines, returning what vLLM returns (RequestOutput / the server's JSON) ---------------------------------

class FakeLLM:
    """``vllm.LLM`` in colocate mode: ``script(prompt_ids, max_tokens) -> token ids``. With ``logprobs=1`` vLLM returns
    the top-1 *and* the sampled token; here the sampled token is ranked second, so rank 1 is never the sampled one."""

    def __init__(self, script, delay: float = 0.0):
        self.script, self.delay = script, delay
        self.calls: list[dict] = []
        self.in_flight = self.max_in_flight = 0
        self.lock = threading.Lock()

    @staticmethod
    def lp(tok: int) -> float:
        return -0.01 * (tok % 50) - 0.5

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay)
            self.calls.append({"n": len(prompts), "max_tokens": sampling_params.max_tokens, "temperature": sampling_params.temperature,
                               "thread": threading.get_ident()})
            outs = []
            for p in prompts:
                ids = list(self.script(list(p["prompt_token_ids"])))[: sampling_params.max_tokens]
                lps = [{tok: SimpleNamespace(logprob=self.lp(tok), rank=2), tok + 100000: SimpleNamespace(logprob=-0.01, rank=1)} for tok in ids]
                outs.append(SimpleNamespace(prompt_token_ids=p["prompt_token_ids"], outputs=[SimpleNamespace(token_ids=ids, logprobs=lps)]))
            return outs
        finally:
            with self.lock:
                self.in_flight -= 1


def colocate(trl, llm, *, temperature=1.0, max_completion_length=64, logprobs=1):
    vg = trl.vg.VLLMGeneration.__new__(trl.vg.VLLMGeneration)
    vg.__dict__.update(mode="colocate", accelerator=SimpleNamespace(process_index=0, num_processes=1, is_main_process=True), model=None,
                       temperature=temperature, top_p=1.0, top_k=-1, min_p=None, repetition_penalty=1.0, max_completion_length=max_completion_length,
                       logprobs=logprobs, structured_outputs_regex=None, generation_kwargs={}, enable_sleep_mode=False, tensor_parallel_size=1, llm=llm)
    return SimpleNamespace(vllm_generation=vg, state=SimpleNamespace(global_step=7), eos_token_ids=[2],
                           processing_class=SimpleNamespace(pad_token_id=0, eos_token_id=2), accelerator=vg.accelerator)


def chained_record(rid: str, results: list, reward: float | None) -> RolloutRecord:
    turns = []
    for k, (prompt, res) in enumerate(results):
        turns.append(Turn(call_id=f"{rid}:c{k + 1}", actor="agent", prompt_is_delta=k > 0, prompt_token_ids=prompt, completion_token_ids=res.completion_ids,
                          logprobs=res.logprobs, policy_version=res.version, finish=res.finish))
    return RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", reward=reward, turns=turns)


def scripted_rollout(turns_for, reward_for=lambda prompt, seed: 1.0, *, masked=(), max_tokens=8, log=None):
    """A World-shaped rollout: ``turns_for(prompt)`` engine calls, each prompt extending the previous call."""
    async def rollout(rid, prompt, seed, gen, *, temperature=1.0):
        if log is not None:
            log.append((rid, prompt, seed))
        full, results = [1, 3 + seed], []
        for _ in range(turns_for(prompt)):
            res = await gen(full, max_tokens, temperature)
            results.append((full if not results else [9], res))
            full = full + res.completion_ids + [9]
        if prompt in masked:
            return None
        return chained_record(rid, results, reward_for(prompt, seed))
    return rollout


def run_trl_scoring(trl, out: dict, prompts: list, num_generations: int):
    """TRL's own consumption of a ``rollout_func`` output: the required-key check, ``env_mask`` pop, the extra_fields
    merge (``grpo_trainer.py:2670-2677``), ``_calculate_rewards`` with a reward function returning ``crucible_reward``,
    and the advantage block (``:2685-2740``)."""
    assert {"prompt_ids", "completion_ids", "logprobs"} <= out.keys()  # :2146-2149
    extra_fields = {k: v for k, v in out.items() if k not in {"prompt_ids", "completion_ids", "logprobs"}}  # :2150
    tool_mask = extra_fields.pop("env_mask", None)  # :2201
    completion_ids_list = out["completion_ids"]
    is_truncated = [ids[-1] not in [2, 0] for ids in completion_ids_list]  # :2224 (raises on an empty row)
    inputs = [{"prompt": p} for p in prompts]
    exec(_merge_block(), {"extra_fields": extra_fields, "inputs": inputs})  # noqa: S102 - the upstream source lines, verbatim

    def crucible_reward(prompts, completions, crucible_reward, **_):
        return list(crucible_reward)

    calc = trl.grpo.GRPOTrainer._calculate_rewards
    calc.__globals__["gather"] = lambda t: t  # one process: accelerate's gather is the identity
    calc.__globals__["profiling_context"] = lambda *a, **k: nullcontext()
    fake = SimpleNamespace(accelerator=SimpleNamespace(device="cpu"), reward_funcs=[crucible_reward], reward_processing_classes=[None],
                           reward_func_names=["crucible_reward"], state=None, _log_completion_extra=None, _log_metric=None, environments=None)
    completions = ["x"] * len(prompts)
    rewards_per_func = calc(fake, inputs, prompts, completions, completion_ids_list)
    ns = {"self": SimpleNamespace(multi_objective_aggregation="sum_then_normalize", scale_rewards="group", reward_weights=torch.ones(1)),
          "rewards_per_func": rewards_per_func, "num_generations": num_generations, "device": "cpu", "torch": torch, "nanstd": trl.utils.nanstd}
    exec(_block("unscorable_mask = torch.isnan(rewards_per_func).all(dim=1)", "advantages = torch.nan_to_num(advantages, nan=0.0)"), ns)  # noqa: S102
    return SimpleNamespace(inputs=inputs, tool_mask=tool_mask, rewards_per_func=rewards_per_func, advantages=ns["advantages"], is_truncated=is_truncated)


def _merge_block() -> str:
    """``grpo_trainer.py:2670-2677``: the lines from the merge comment up to (not including) the reward calculation."""
    return "\n".join(_block("# Merge extra_fields from rollout_func into inputs for reward functions",
                            "# Calculate rewards for each reward function").splitlines()[:-1])


# ---- tests ----------------------------------------------------------------------------------------------------------------

def test_V22_trl_gen_takes_the_sampled_token_logprob_from_upstream_generate(trl):
    from crucible.train.bridges.trl_bridge import trl_gen

    llm = FakeLLM(lambda p: [40, 41, 2])
    res = asyncio.run(trl_gen(colocate(trl, llm))([1, 2, 3], 16, 1.0))
    # TRL's generate returns (B, T, num_logprobs) sorted by rank; the sampled token is rank 2 here
    assert res.completion_ids == [40, 41, 2] and res.logprobs == [FakeLLM.lp(40), FakeLLM.lp(41), FakeLLM.lp(2)]
    assert res.finish == "stop" and res.version == 7


def test_V28_trl_gen_honours_max_tokens_temperature_and_reports_length(trl):
    from crucible.train.bridges.trl_bridge import trl_gen

    llm = FakeLLM(lambda p: list(range(10, 30)))
    tr = colocate(trl, llm, max_completion_length=64)
    res = asyncio.run(trl_gen(tr)([1, 2], 5, 1.0))
    assert llm.calls[-1]["max_tokens"] == 5 and tr.vllm_generation.max_completion_length == 64  # the cap reached vLLM, then restored
    assert res.completion_ids == [10, 11, 12, 13, 14] and res.finish == "length"
    with pytest.raises(ValueError, match="temperature"):  # TRL scores π_θ at its own temperature: another one is refused
        asyncio.run(trl_gen(tr)([1, 2], 5, 0.7))


def test_V23_one_episode_per_trl_row_with_group_seeds(trl):
    from crucible.train.bridges.trl_bridge import make_rollout_func

    log: list = []
    rf = make_rollout_func(scripted_rollout(lambda p: 1, lambda prompt, seed: float(seed), log=log), num_generations=2)
    prompts = ["p0", "p0", "p1", "p1"]  # TRL's RepeatSampler already repeated each prompt num_generations=2 times
    out = rf(prompts, colocate(trl, FakeLLM(lambda p: [50, 2])))
    assert len(log) == 4 and [s for _, _, s in log] == [0, 1, 0, 1]  # one episode per row, seed = copy index in the group
    assert len(out["completion_ids"]) == len(out["crucible_reward"]) == 4
    scored = run_trl_scoring(trl, out, prompts, num_generations=2)
    assert scored.rewards_per_func[:, 0].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert scored.advantages.tolist() == pytest.approx([-0.7070, 0.7070, -0.7070, 0.7070], abs=1e-3)


def test_V23_seed_uses_the_global_row_index_across_processes(trl):
    from crucible.train.bridges.trl_bridge import make_rollout_func

    log: list = []
    rf = make_rollout_func(scripted_rollout(lambda p: 0, log=log), num_generations=4)
    tr = colocate(trl, FakeLLM(lambda p: [2]))
    tr.accelerator = SimpleNamespace(process_index=1, num_processes=2)  # process 1 holds copies 2 and 3 of the group
    rf(["p0", "p0"], tr)
    assert [s for _, _, s in log] == [2, 3]


def test_V24_masked_episode_keeps_its_row_and_is_unscorable_in_trl(trl):
    from crucible.train.bridges.trl_bridge import make_rollout_func
    from crucible.train.engine import GenResult

    rewards = {"pA": 1.0, "pB": 0.0, "pC": 0.5, "pD": 1.0}

    async def rollout(rid, prompt, seed, gen, *, temperature=1.0):  # records as a World returns them (no engine needed)
        if prompt == "pB":
            return None  # an infra fault masked this episode
        return chained_record(rid, [([1, 2], GenResult([60, 61, 2], [-0.5, -0.5, -0.5], 7, "stop"))], rewards[prompt])

    rf = make_rollout_func(rollout, num_generations=1)
    prompts = ["pA", "pB", "pC", "pD"]
    out = rf(prompts, colocate(trl, FakeLLM(lambda p: [60, 61, 2])))
    assert len(out["completion_ids"]) == len(out["env_mask"]) == len(out["logprobs"]) == 4
    assert out["crucible_reward"][1] is None and out["env_mask"][1] == [0] and out["crucible_masked"][1]
    assert out["env_mask"][2] == [1, 1, 1] and out["crucible_reward"] == [1.0, None, 0.5, 1.0]  # every row keeps its own reward
    scored = run_trl_scoring(trl, out, prompts, num_generations=1)
    assert torch.isnan(scored.rewards_per_func[1, 0]) and scored.advantages[1] == 0.0  # unscorable, never 0 reward


def test_V24_untrainable_record_is_a_placeholder_not_zero_logprobs(trl):
    from crucible.train.bridges.trl_bridge import make_rollout_func

    async def rollout(rid, prompt, seed, gen, *, temperature=1.0):
        res = await gen([1, 2], 4, temperature)
        rec = chained_record(rid, [([1, 2], res)], 1.0)
        if prompt == "bad":  # behaviour logprobs lost: it must not train as π_rollout = 1
            rec.turns[0].logprobs = None
        return rec

    out = make_rollout_func(rollout, num_generations=1)(["ok", "bad"], colocate(trl, FakeLLM(lambda p: [70, 2])))
    assert out["env_mask"] == [[1, 1], [0]] and out["crucible_reward"] == [1.0, None] and "logprobs" in out["crucible_masked"][1]


def test_D4_rounds_batch_one_generate_per_round_on_one_thread(trl):
    from crucible.train.bridges.trl_bridge import make_rollout_func

    llm = FakeLLM(lambda p: [80, 2], delay=0.02)
    rf = make_rollout_func(scripted_rollout(lambda p: {"a": 3, "b": 1, "c": 2, "d": 3}[p]), num_generations=1)
    out = rf(["a", "b", "c", "d"], colocate(trl, llm))
    assert llm.max_in_flight == 1 and len({c["thread"] for c in llm.calls}) == 1  # never concurrent, never from worker threads
    assert [c["n"] for c in llm.calls] == [4, 3, 2]  # one generate per round, batching every live episode
    assert out["crucible_reward"] == [1.0] * 4


class ThreadCollective:
    """accelerate's collectives across ranks simulated by threads (a broken barrier, not a hang, if call counts differ)."""

    def __init__(self, n: int):
        self.n, self.barrier, self.slots, self.local = n, threading.Barrier(n, timeout=10), [None] * n, threading.local()

    def all_gather(self, obj):
        r = self.local.rank
        self.slots[r] = obj
        self.barrier.wait()
        out = list(self.slots)
        self.barrier.wait()
        return out


class FakeServerClient:
    """The vLLM server behind TRL's ``vllm_client.generate`` (``n`` completions per prompt, nested logprobs)."""

    def __init__(self):
        self.batches: list[int] = []

    def generate(self, prompts, features=None, n=1, max_tokens=16, logprobs=0, **_):
        self.batches.append(len(prompts))
        comps = [[100 + len(p) % 7, 2][:max_tokens] for p in prompts for _ in range(n)]
        return {"prompt_ids": [list(p) for p in prompts], "completion_ids": comps,
                "logprobs": [[[-0.25]] * len(c) for c in comps], "logprob_token_ids": [[[t] for t in c] for c in comps]}


def test_D4_server_mode_ranks_run_generate_in_lockstep(trl):
    from crucible.train.bridges.trl_bridge import make_rollout_func, trl_rounds

    coll, client = ThreadCollective(2), FakeServerClient()
    gen_mod = trl.vg.VLLMGeneration.generate.__globals__
    gen_mod["gather_object"] = lambda obj: [x for part in coll.all_gather(obj) for x in part]

    def bcast(obj_list, from_process=0):
        obj_list[0] = coll.all_gather(obj_list[0])[from_process]
    gen_mod["broadcast_object_list"] = bcast
    turns = {"r0a": 3, "r0b": 2, "r1a": 1, "r1b": 1}  # unequal per-rank call counts: rank 1 finishes after one round
    results: dict = {}
    trainers = []

    def run(rank):
        coll.local.rank = rank
        tr = colocate(trl, None)
        tr.vllm_generation.__dict__.update(mode="server", vllm_client=client, accelerator=SimpleNamespace(process_index=rank, num_processes=2,
                                                                                                          is_main_process=rank == 0))
        tr.accelerator = tr.vllm_generation.accelerator
        trainers.append(tr)
        rf = make_rollout_func(scripted_rollout(lambda p: turns[p]), num_generations=1,
                               rounds_factory=lambda t: trl_rounds(t, gather=coll.all_gather))
        try:
            results[rank] = rf([f"r{rank}a", f"r{rank}b"], tr)
        except BaseException as e:  # noqa: BLE001
            results[rank] = e

    ts = [threading.Thread(target=run, args=(r,), daemon=True) for r in (0, 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert not any(t.is_alive() for t in ts), "deadlock"
    assert all(isinstance(results[r], dict) for r in (0, 1)), results
    assert client.batches == [4, 4, 2]  # three lockstep rounds, equal per-rank batches (rank 1 pads with a dummy once done)
    assert results[0]["crucible_reward"] == [1.0, 1.0] and results[1]["crucible_reward"] == [1.0, 1.0]
    assert results[1]["completion_ids"][0][-1] == 2  # rank 1 got its own slice back, not rank 0's


def test_D4_trl_rollout_func_runs_world_episodes_through_upstream_generate(trl):
    """The toy World end to end: TRL's colocate ``generate`` (real), one episode per row, rewards via TRL's scoring."""
    from crucible_pack_toy.runtime import HelpdeskTask
    from crucible_pack_toy.verify import GOLD

    from crucible.core.schema.actions import openai_name
    from crucible.train.bridges.trl_bridge import make_rollout_func
    from crucible.train.chat import ByteChatCodec
    from crucible.train.world_env import world_rollout

    codec = ByteChatCodec()

    def script(prompt):
        k = bytes(t for t in prompt if t < 256).count(b"<tool_call>")  # stateless: position from the context, like a real model
        if k < len(GOLD):
            call = {"name": openai_name(GOLD[k].function_name), "arguments": GOLD[k].arguments}
            return list(f"<tool_call>{json.dumps(call)}</tool_call>".encode()) + [codec.EOS]
        return list(b"done") + [codec.EOS]

    llm = FakeLLM(script)
    tr = colocate(trl, llm, max_completion_length=4096)
    tr.eos_token_ids = [codec.EOS]
    rf = make_rollout_func(world_rollout(lambda p: HelpdeskTask(), codec, max_steps=len(GOLD) + 3, max_tokens=2048), num_generations=2)
    prompts = ["p0", "p0"]
    out = rf(prompts, tr)
    assert len(out["completion_ids"]) == 2 and all(len(c) == len(m) == len(lp) for c, m, lp in zip(out["completion_ids"], out["env_mask"], out["logprobs"], strict=True))
    assert out["crucible_reward"] == [1.0, 1.0] and out["crucible_masked"] == [None, None]
    sampled = [lp for c, m, lps in zip(out["completion_ids"], out["env_mask"], out["logprobs"], strict=True) for t, mm, lp in zip(c, m, lps, strict=True) if mm]
    assert sampled and all(lp <= -0.5 for lp in sampled)  # the sampled tokens' logprobs, never rank 1's -0.01
    assert llm.max_in_flight == 1 and all(c["n"] == 2 for c in llm.calls) and all(c["max_tokens"] == 2048 for c in llm.calls)
    scored = run_trl_scoring(trl, out, prompts, num_generations=2)
    assert scored.rewards_per_func[:, 0].tolist() == [1.0, 1.0] and scored.advantages.tolist() == [0.0, 0.0]


# ------------------------------------------------------------------------------------------------------- SkyRL and rLLM

SKYRL = ["skyrl.train.trainer", "skyrl.train.utils.trainer_utils", "skyrl.train.generators.base", "skyrl.train.generators.utils",
         "skyrl.backends.skyrl_train.utils.ppo_utils"]
RLLM = ["rllm.types", "rllm.engine.rollout.rollout_engine", "rllm.trainer.verl.dataclass", "rllm.trainer.verl.transform",
        "rllm.trainer.algorithms.config", "rllm.trainer.algorithms.rl_algo", "rllm.trainer.algorithms.advantage"]


def rec(rid: str, reward: float | None, *, group: str = "p0", segs: int = 1, truncated: bool = False, finish: str | None = "stop",
        masked: bool = False, advantage: float | None = None) -> RolloutRecord:
    turns = []
    for k in range(segs):
        turns.append(Turn(call_id=f"{rid}:c{2 * k + 1}", actor="agent", segment=k, prompt_token_ids=[1, 2, 40 + k], completion_token_ids=[5, 6],
                          logprobs=[-0.1, -0.2], policy_version=3 + k, finish="stop", advantage=advantage))
        turns.append(Turn(call_id=f"{rid}:c{2 * k + 2}", actor="agent", segment=k, prompt_is_delta=True, prompt_token_ids=[9, 9],
                          completion_token_ids=[7], logprobs=[-0.3], policy_version=3 + k, finish=finish if k == segs - 1 else "stop"))
    return RolloutRecord(rollout_id=rid, sample=rid, task="t", group=group, harness="h", policy="p", reward=reward, turns=turns,
                         truncated=truncated, mask_sample=masked, failure_kind="infra" if masked else None)


def _cfg(step_wise: bool, std: bool = False):
    algo = SimpleNamespace(advantage_estimator="grpo", gamma=1.0, lambd=1.0, grpo_norm_by_std=std, zero_variance_filter=False,
                           zero_variance_filter_tol=0.0)
    gen = SimpleNamespace(step_wise_trajectories=step_wise, merge_stepwise_output=False, n_samples_per_prompt=3,
                          inference_engine=SimpleNamespace(enable_return_routed_experts=False))
    return SimpleNamespace(generator=gen, trainer=SimpleNamespace(algorithm=algo))


class _Batch(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.metadata: dict = {}

    def to(self, _):
        return self


def _skyrl_advantages(ns, go, uids, *, step_wise: bool, std: bool = False):
    """SkyRL's own path from a GeneratorOutput to advantages: ``RayPPOTrainer.postprocess_generator_output`` then
    ``compute_advantages_and_returns`` (trainer.py:1083-1260), on a stand-in ``self`` holding only the config."""
    me = SimpleNamespace(cfg=_cfg(step_wise, std), all_metrics={})
    T = ns.trainer.RayPPOTrainer
    go, uids = T.postprocess_generator_output(me, go, uids)
    L = max(len(r) for r in go["response_ids"])
    pad = lambda xs, v=0: [list(x) + [v] * (L - len(x)) for x in xs]
    data = _Batch(rewards=torch.tensor(pad(go["rewards"], 0.0)), response_mask=torch.tensor(pad([[1] * len(r) for r in go["response_ids"]])),
                  values=None)
    data.metadata = {"uids": uids, "avg_response_length": 1.0, **({"is_last_step": go["is_last_step"]} if step_wise else {})}
    data = T.compute_advantages_and_returns(me, data)
    return data["advantages"], go


def _gen_input(ns, n_prompts: int, n: int):
    prompts = [{"prompt": [{"role": "user", "content": f"q{i}"}], "env_class": None, "env_extras": {}, "uid": f"p{i}"} for i in range(n_prompts)]
    return ns.utils.prepare_generator_input(prompts, n, {}, "crucible", "train", 0)


def test_V25_skyrl_output_passes_skyrl_generate():
    """Every GeneratorOutput key (rollout_metrics is indexed at trainer.py:1069 before validation), one row per input with
    a masked, a compacted (two segments) and a truncated rollout in the batch."""
    from crucible.train.bridges.adapters import skyrl_generator_output

    with upstream("skyrl", SKYRL) as ns:
        gi, _ = _gen_input(ns, 2, 2)
        recs = [rec("a", 1.0, group="p0"), rec("b", None, group="p0", masked=True), rec("c", 1.0, group="p1", segs=2), rec("d", 0.0, group="p1", truncated=True)]
        go = skyrl_generator_output(recs, trajectory_ids=gi["trajectory_ids"])
        assert set(go) == set(ns.base.GeneratorOutput.__annotations__)
        me = SimpleNamespace(generator=SimpleNamespace(generate=lambda b: asyncio.sleep(0, go)), all_metrics={}, cfg=_cfg(False))
        out = asyncio.run(ns.trainer.RayPPOTrainer.generate(me, gi))  # validate_generator_output inside
        assert len(out["response_ids"]) == 4 and me.all_metrics["crucible/masked"] == 1 and me.all_metrics["crucible/segments_dropped"] == 1
        assert out["loss_masks"][1] == [0] and sum(out["loss_masks"][0]) == 3


def test_V25_skyrl_masked_rollout_never_trains_its_group_as_zero():
    """A masked member of [1.0, 0.0, masked]: SkyRL averages every row of a uid (ppo_utils.py:1367-1390), so a 0.0
    placeholder would shift the baseline. One row per input: the live mean is preserved exactly (Dr.GRPO). Step-wise: the
    masked rollout has no row at all, so even GRPO with std equals GRPO over the live rollouts."""
    from crucible.train.bridges.skyrl_bridge import generator_output

    with upstream("skyrl", SKYRL) as ns:
        gi, uids = _gen_input(ns, 1, 3)
        recs = [rec("a", 1.0), rec("b", 0.0), None]
        go = generator_output(recs, trajectory_ids=gi["trajectory_ids"])
        ns.trainer_utils.validate_generator_output(3, go)
        adv, go2 = _skyrl_advantages(ns, go, list(uids), step_wise=False)
        live = [float(adv[0][go2["loss_masks"][0].index(1)]), float(adv[1][go2["loss_masks"][1].index(1)])]
        assert live == pytest.approx([0.5, -0.5]) and sum(go2["loss_masks"][2]) == 0  # the placeholder trains nothing
        go = generator_output(recs, trajectory_ids=gi["trajectory_ids"], step_wise=True)
        ns.trainer_utils.validate_generator_output(3, go, step_wise=True)
        assert len(go["response_ids"]) == 2
        adv, _ = _skyrl_advantages(ns, go, [t.instance_id for t in go["trajectory_ids"]], step_wise=True, std=True)
        ref, _ = ns.ppo_utils.compute_grpo_outcome_advantage(torch.tensor([[1.0], [0.0]]), torch.ones(2, 1), ["p0", "p0"], grpo_norm_by_std=True)
        assert float(adv[0][0]) == pytest.approx(float(ref[0][0])) and float(adv[1][0]) == pytest.approx(float(ref[1][0]))


def test_V25_skyrl_step_wise_split_rollout_is_one_trajectory():
    """A compacted rollout is two causal sequences: step-wise rows with the input's trajectory id, contiguous, is_last_step
    on the second; SkyRL's cumsum broadcast (trainer.py:1236-1247) gives both the trajectory's advantage."""
    from crucible.train.bridges.skyrl_bridge import generator_output

    with upstream("skyrl", SKYRL) as ns:
        gi, _ = _gen_input(ns, 1, 2)
        go = generator_output([rec("c", 1.0, segs=2), rec("d", 0.0)], trajectory_ids=gi["trajectory_ids"], step_wise=True)
        ns.trainer_utils.validate_generator_output(2, go, step_wise=True)
        assert go["is_last_step"] == [False, True, True] and go["rewards"] == [0.0, 1.0, 0.0]
        adv, _ = _skyrl_advantages(ns, go, [t.instance_id for t in go["trajectory_ids"]], step_wise=True)
        assert [float(a[0]) for a in adv] == pytest.approx([0.5, 0.5, -0.5])


def test_V28_skyrl_stop_reasons_follow_truncation_and_finish():
    from crucible.train.bridges.skyrl_bridge import generator_output

    go = generator_output([rec("a", 1.0), rec("b", 0.0, truncated=True), rec("c", 0.0, finish="length"), rec("d", 1.0, masked=True)])
    assert go["stop_reasons"] == ["stop", "length", "length", "abort"]


def _groups(ns, recs, backend, **kw):
    from crucible.train.bridges.rllm_bridge import trajectory_groups

    return trajectory_groups(recs, backend=backend, **kw)


def test_V26_rllm_verl_backend_trains_every_live_trajectory():
    """rLLM's verl transform drops steps without ``model_output`` (transform.py:293-299) and reads one scalar advantage per
    trajectory (:592-597). The bridge's steps reach a row each (a compacted rollout: two rows, one uid), rLLM computes the
    advantage from Trajectory.reward, and the masked rollout is out of the group."""
    with upstream("rllm", RLLM) as ns:
        tf = sys.modules["rllm.trainer.verl.transform"]
        groups = _groups(ns, [rec("a", 1.0), rec("b", 0.0, segs=2), rec("m", None, masked=True)], "verl")
        assert len(groups) == 1 and groups[0].group_id == "p0:agent" and len(groups[0].trajectories) == 2
        ns.advantage.collect_reward_and_advantage_from_trajectory_groups(groups, ns.config.AlgorithmConfig(norm_adv_by_std_in_grpo=False))
        acc = ns.dataclass.AccumulatedData()
        rows = tf._process_trajectory_group(groups[0], groups[0].task_id, acc)
        assert rows == 3 and acc.step_ids == ["a:agent", "b:agent", "b:agent"]
        mask = torch.nn.utils.rnn.pad_sequence(acc.traj_mask, batch_first=True)
        batch = SimpleNamespace(batch={"response_mask": mask, "traj_rewards": torch.zeros_like(mask, dtype=torch.float)},
                                non_tensor_batch={"trajectory_ids": acc.trajectory_ids, "step_ids": acc.step_ids})
        tf.update_dataproto_with_advantages(batch, groups)
        per_row = [float(batch.batch["advantages"][i][mask[i].bool()][0]) for i in range(3)]
        assert per_row == pytest.approx([0.5, -0.5, -0.5])


def test_V26_rllm_tinker_backend_gets_per_token_advantages():
    with upstream("rllm", RLLM) as ns:
        groups = _groups(ns, [rec("a", 1.0, advantage=0.25), rec("b", 0.0)], "tinker", seq_adv=lambda r: 0.5 if r.reward else -0.5, turn_weight=1.0)
        ns.advantage.collect_reward_and_advantage_from_trajectory_groups(groups, ns.config.AlgorithmConfig(use_precomputed_advantage=True))
        st = groups[0].trajectories[0].steps
        assert [s.advantage for s in st] == [[0.75, 0.75], [0.5]]
        assert all(len(s.advantage) == len(s.response_ids) for t in groups[0].trajectories for s in t.steps)


def test_H5_rllm_outcome_once_versions_and_model_output():
    with upstream("rllm", RLLM) as ns:
        from crucible.train.bridges.rllm_bridge import trajectories

        (tr,) = trajectories(rec("c", 1.0, segs=2))
        assert isinstance(tr, ns.types.Trajectory) and tr.reward == 1.0
        assert [s.reward for s in tr.steps] == [0.0, 0.0, 0.0, 1.0] and [s.done for s in tr.steps] == [False, False, False, True]
        assert [s.weight_version for s in tr.steps] == [3, 3, 4, 4]
        assert all(isinstance(s.model_output, ns.rollout_engine.ModelOutput) for s in tr.steps)
        assert tr.steps[1].model_output.prompt_ids == [1, 2, 40, 5, 6, 9, 9]  # the call's full prompt (prefix merge needs it)
        assert trajectories(rec("m", 1.0, masked=True)) == []


def test_V26_rllm_steps_are_not_skipped_by_the_verl_transform():
    """The reproducer (H4 repro_rllm [3]): steps built from ``adapters.rllm_steps`` and run through rLLM's verl
    ``_process_trajectory`` emitted 0 rows (no ``model_output``); they must emit the trajectory's row."""
    from crucible.train.bridges.adapters import rllm_steps
    from crucible.train.bridges.common import flatten

    with upstream("rllm", RLLM) as ns:
        tf = sys.modules["rllm.trainer.verl.transform"]
        (s,) = flatten(rec("a", 1.0))
        sts = [x if isinstance(x, ns.types.Step) else ns.types.Step(**x) for x in rllm_steps(s)]
        acc = ns.dataclass.AccumulatedData()
        assert tf._process_trajectory(ns.types.Trajectory(steps=sts, reward=1.0), "t", acc) == 1
        assert acc.responses[0].tolist() == [5, 6, 9, 9, 7] and acc.traj_mask[0].tolist() == [1, 1, 0, 0, 1]
        assert all(st.advantage is None or isinstance(st.advantage, float) for st in sts)  # verl backend: float(adv) (transform.py:597)


# -------------------------------------------------------------------------------- verifiers v1 import and interception

_DUMP = r"""
import json, sys
from verifiers.v1.trace import WireTrace
t = WireTrace.model_validate(json.loads(sys.stdin.read()))
print(json.dumps({"record": t.to_record(), "is_truncated": t.is_truncated}))
"""


def _trace(*, sampling=None, finish="stop", stop="agent_completed", agent=None, ok=True, is_timeout=False, **kw) -> dict:
    return {"task": {"type": "T", "data": {"q": 1}}, "agent": agent or {"config": {}, "name": "solver", "trainable": True},
            "nodes": [{"message": {"role": "user", "content": "hi"}, "token_ids": [1, 2], "mask": [False, False]},
                      {"parent": 0, "message": {"role": "assistant", "content": "x"}, "sampled": True, "token_ids": [3, 4],
                       "mask": [True, True], "logprobs": [-0.1, -0.2]}],
            "calls": [{"node": 1, "sampling": sampling or {"temperature": 0.7, "max_tokens": 64}, "finish_reason": finish}],
            "rewards": {"solved": {"score": 1.0, "weight": 1.0}}, "ok": ok, "is_timeout": is_timeout, "stop_condition": stop, **kw}


def _dump(trace: dict) -> tuple[dict, bool]:
    """verifiers' own serialization of ``trace`` and its ``Trace.is_truncated`` (a property: absent from the dump)."""
    py = Path(__file__).resolve().parents[2] / ".envs" / "verifiers" / "bin" / "python"  # this checkout's verifiers env
    if not py.exists():
        pytest.skip("needs crucible/.envs/verifiers (verifiers v1)")
    out = subprocess.run([str(py), "-c", _DUMP], input=json.dumps(trace), capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    d = json.loads(out.stdout.strip().splitlines()[-1])
    assert "is_truncated" not in d["record"]  # the import must recompute it (V/trace.py:620-632)
    return d["record"], d["is_truncated"]


# ------------------------------------------------------------------------------------------- G4 H6: verifiers v1 import


def test_g4h6_length_finish_is_truncation_and_turn_records_finish_and_sampling():
    rec_d, vf_trunc = _dump(_trace(finish="length"))
    assert vf_trunc is True
    rec = trace_to_record(rec_d, "r")
    assert rec.truncated is True and not rec.mask_sample  # a limit stop is scored (LimitPolicy default), and flagged
    t = rec.turns[0]
    assert t.finish == "length" and t.sampling.temperature == 0.7 and t.sampling.max_tokens == 64
    rec_d, vf_trunc = _dump(_trace(finish="tool_calls"))
    assert vf_trunc is False and trace_to_record(rec_d, "r").truncated is False and trace_to_record(rec_d, "r").turns[0].finish == "stop"


@pytest.mark.parametrize("stop", ["max_output_tokens", "max_input_tokens"])
def test_g4h6_verifiers_token_limit_stops_are_truncation(stop):
    rec_d, vf_trunc = _dump(_trace(stop=stop))
    assert vf_trunc is True and trace_to_record(rec_d, "r").truncated is True


def test_g4h6_actor_is_the_agent_name_and_an_untrainable_agent_never_trains():
    rec_d, _ = _dump(_trace())
    rec = trace_to_record(rec_d, "r")
    assert rec.turns[0].actor == "solver" and rec.actor == "solver"  # not str(AgentInfo dict)
    judge, _ = _dump(_trace(agent={"config": {}, "name": "judge", "trainable": False}))
    rj = trace_to_record(judge, "j")
    assert rj.turns[0].actor == "judge" and rj.mask_sample and rj.reward is None and rj.failure_kind == "vf:untrainable_agent"


def test_g4h6_truncated_sampling_is_refused_not_silently_changed():
    rec_d, _ = _dump(_trace(sampling={"temperature": 1.0, "top_p": 0.9, "max_tokens": 64}))
    rec = trace_to_record(rec_d, "r")
    assert rec.mask_sample and rec.failure_kind == "crucible:harness_unsupported" and rec.turns[0].sampling.top_p == 0.9
    rec_d, _ = _dump(_trace(sampling={"temperature": 1.0, "top_k": 20}))
    assert trace_to_record(rec_d, "r").failure_kind == "crucible:harness_unsupported"
    rec_d, _ = _dump(_trace(sampling={"temperature": 1.0, "top_p": 1.0, "top_k": -1}))
    assert not trace_to_record(rec_d, "r").mask_sample  # full sampling trains


def test_g4h6_timeouts_follow_crucible_limit_semantics():
    from crucible.runtime.infra import LimitPolicy

    agent_to, _ = _dump(_trace(stop="agent_timeout", is_timeout=True))  # verifiers: a stop, not a failure (ok stays True)
    rec = trace_to_record(agent_to, "r")
    assert rec.truncated and not rec.mask_sample and rec.reward == 1.0 and rec.stop == "agent_timeout"
    assert trace_to_record(agent_to, "r", limit_policy=LimitPolicy(on_timeout="mask")).mask_sample
    assert trace_to_record(agent_to, "r", limit_policy=LimitPolicy(on_timeout="penalty:-0.5")).reward == -0.5
    for stop, kind in (("setup_timeout", "crucible:backend_error"), ("finalize_timeout", "crucible:backend_error"),
                       ("scoring_timeout", "verifier_timeout")):
        d, _ = _dump(_trace(stop=stop, is_timeout=True))
        r = trace_to_record(d, "r")
        assert r.mask_sample and r.reward is None and r.failure_kind == kind, stop


def test_g4h6_a_failed_episode_masks_its_ok_trace():
    rec_d, _ = _dump(_trace())
    r = trace_to_record(rec_d, "r", episode={"ok": False, "errors": [{"type": "SandboxError", "message": "x"}]})
    assert r.mask_sample and r.failure_kind == "crucible:backend_error"


def test_g4h6_generate_server_records_served_sampling_and_matches_shared_routes():
    from fastapi.testclient import TestClient

    from crucible.interop.verifiers.v1_import import RolloutGenerateServer
    from crucible.train.engine import GenResult

    srv = RolloutGenerateServer("m")
    outs = iter([[7, 7], [3, 4]])

    async def gen(ids, max_tokens, temperature):
        c = next(outs)
        return GenResult(c, [-0.5] * len(c), 2, "stop")

    srv.gens["k"] = gen
    cl = TestClient(srv.app())
    for _ in range(2):  # the judge's call (same route) then the solver's
        r = cl.post("/r/k/inference/v1/generate", json={"token_ids": [1, 2], "sampling_params": {"max_tokens": 8, "temperature": 0.5, "top_p": 0.8}})
        assert r.status_code == 200
    assert srv.sampling["k"][0]["top_p"] == 1.0 and srv.sampling["k"][0]["requested"] == {"top_p": 0.8}
    rec_d, _ = _dump(_trace(sampling={"temperature": 0.5, "max_tokens": 8}))
    full = dict(rec_d, _crucible_versions=srv.versions["k"], _crucible_completions=srv.completions["k"], _crucible_sampling=srv.sampling["k"])
    rec = trace_to_record(full, "r")
    # the solver's completion is the server's 2nd: matched by content (not call index), with that call's exact logprobs
    assert rec.turns[0].logprobs == [-0.5, -0.5] and rec.turns[0].policy_version == 2
    assert rec.mask_sample and rec.failure_kind == "crucible:harness_unsupported"  # it asked for top_p 0.8, which was not served
    ok_srv = [dict(s, requested={}) for s in srv.sampling["k"]]
    rec = trace_to_record(dict(full, _crucible_sampling=ok_srv), "r")
    assert not rec.mask_sample and rec.turns[0].sampling.top_p == 1.0  # Turn.sampling = what was served


async def test_g4h6_vf_rollout_trains_the_solver_of_a_multi_agent_episode(monkeypatch):
    import crucible.interop.verifiers.v1_import as v1

    solver, _ = _dump(_trace())
    judge, _ = _dump(_trace(agent={"config": {}, "name": "judge", "trainable": False}))

    fn = v1.vf_rollout("env", "m")

    async def fake_eval(env_id, base_url, model, **k):  # both agents called the model through the rollout's route
        key = base_url.rstrip("/").split("/r/", 1)[1].split("/", 1)[0]
        for ids in ([3, 4], [3, 4]):
            fn.server.versions.setdefault(key, []).append(1)
            fn.server.completions.setdefault(key, []).append((ids, [-0.1, -0.2]))
            fn.server.sampling.setdefault(key, []).append({"temperature": 0.7, "max_tokens": 64, "top_p": 1.0, "top_k": -1, "requested": {}})
        return [{"ok": True, "errors": [], "traces": [judge, solver]}]

    monkeypatch.setattr(v1, "run_vf_eval", fake_eval)
    try:
        rec = await fn("rid", "0", 0, None)
        assert rec is not None and rec.actor == "solver" and not rec.mask_sample  # the judge's trace is context
    finally:
        await fn.server.stop()


# ------------------------------------------------------------------------------------ G1 H7: the git back-link (Qwen)

BACKLINKS = [
    "git remote add origin https://github.com/o/r && git fetch origin",
    "git remote add up https://github.com/o/r; git pull up main",
    "git remote add up git@github.com:o/r.git",  # the split form: the fetch comes in a later call
    "git remote set-url origin https://github.com/o/r",
    "git -C repo remote add up https://gitlab.com/o/r",
    "git config remote.origin.url https://github.com/o/r",
    "git ls-remote https://github.com/o/r",
    "pip install git+https://github.com/o/r",
    "wget example.org/x.tar.gz",
]


@pytest.mark.parametrize("cmd", BACKLINKS)
def test_g1h7_git_backlink_is_intercepted(cmd):
    hit = RuleInterceptor().check("agent", "ws.shell", {"command": cmd})
    assert hit is not None and hit.rule == "remote_code_fetch", cmd


def test_g1h7_remote_written_into_git_config_is_intercepted_and_allowlisted_hosts_pass():
    ic = RuleInterceptor(allow_hosts=("git.internal.example",))
    assert ic.check("a", "ws.write_file", {"path": ".git/config", "content": '[remote "up"]\n\turl = https://github.com/o/r\n'}) is not None
    assert ic.check("a", "ws.shell", {"command": "git remote add up https://git.internal.example/t/r && git fetch up"}) is None


LOCAL = [
    "git status --short && echo ok", "git log --oneline", "git diff HEAD~1", 'git commit -am "fix https://github.com/o/r/issues/1"',
    "git remote -v", "git remote add backup ../mirror && git fetch backup", "git pull . feature", "git fetch origin",
    "cat README.md  # see https://github.com/o/r",
    "curl http://localhost:8000/health", "curl -s http://127.0.0.1:8080/api", "wget http://[::1]:9000/x",  # the world's own services
]


def test_g1h7_local_git_use_and_loopback_are_not_intercepted():
    ic = RuleInterceptor()
    passed = [cmd for cmd in LOCAL if ic.check("agent", "ws.shell", {"command": cmd}) is None]
    assert passed == LOCAL  # the M11 rule refused curl/wget to the world's own loopback services
