"""M12 audit regression tests: the rollout plane, token fidelity and the weight plane (H3 V1-V23, sweep G3 D6/D8/D9/D11,
G2 H5, H1 V7, G5 weightsync P2s). Each test fails on the M11 commit (3b522cb).

Rules under test: an infrastructure failure is a masked rollout with a named failure kind; the trained tokens, log-probs
and versions are exactly what was sampled; ``capabilities()`` never claims more than the engine does."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch

from crucible.train.chat import ByteChatCodec
from crucible.train.engine import GenRequest, GenResult, TinyPolicy, TorchEngine
from crucible.train.world_env import EngineLLM

codec = ByteChatCodec()


def _text(ids):
    return bytes(t for t in ids if t < 256)


# ---- V1 (P0): EngineLLM extends a conversation only through the reply it produced for it ----------------------------------
async def test_v1_engine_llm_never_splices_another_conversations_reply():
    replies = iter([b'<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>', b'<tool_call>{"name": "f", "arguments": {"x": 2}}</tool_call>',
                    b"done B", b"done A"])

    async def gen(prompt, n, t):
        ids = list(next(replies)) + [codec.EOS]
        return GenResult(ids, [-0.1] * len(ids), 0, "stop")

    for keyed in (True, False):  # keyed by the episode (sample_key) or, without one, by the reply itself
        replies = iter([b'<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>', b'<tool_call>{"name": "f", "arguments": {"x": 2}}</tool_call>',
                        b"done B", b"done A"])
        llm = EngineLLM(codec, gen)
        opening = [{"role": "system", "content": "s"}, {"role": "user", "content": "same task"}]  # a GRPO group: one opening
        conv, first = {"A": list(opening), "B": list(opening)}, {}
        for k in ("A", "B"):
            r = await llm.complete(conv[k], sample_key=k if keyed else None)
            first[k] = r.tokens["prompt_token_ids"] + r.tokens["completion_token_ids"]
            c = r.tool_calls[0]
            conv[k] += [{**r.raw_message, "role": "assistant", "content": r.text,  # as LLMToolPolicy replays it
                         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}]},
                        {"role": "tool", "tool_call_id": "c1", "content": f"result for {k}"}]
        for k in ("B", "A"):
            r = await llm.complete(conv[k], sample_key=k if keyed else None)
            p = r.tokens["prompt_token_ids"]
            assert p[: len(first[k])] == first[k], (keyed, k, _text(p)[-80:])  # its own sampled tokens, token for token
            assert f"result for {k}".encode() in _text(p) and not r.tokens.get("new_segment")
        assert llm.prefix_breaks == 0


# ---- V2 (P0): a vLLM publish under running calls never stamps tokens of two weight sets with one version ------------------
async def test_v2_vllm_publish_under_a_running_call_is_aborted_and_reissued_whole():
    from crucible.train.vllm_http import VLLMServerEngine

    state = {"weights": 0, "paused": False}
    log = []

    async def handler(req):
        log.append(req.url.path)
        body = json.loads(req.content) if req.content else {}
        if req.url.path == "/v1/completions":
            toks = []
            for _ in range(body["max_tokens"]):  # one decode step per 10 ms on whatever weights are loaded now
                if state["paused"]:  # /pause?mode=abort: running requests come back aborted
                    return httpx.Response(200, json={"choices": [{"token_ids": toks, "finish_reason": "abort", "logprobs": {"token_logprobs": [-0.1] * len(toks)}}]})
                await asyncio.sleep(0.01)
                toks.append(100 + 100 * state["weights"])
            return httpx.Response(200, json={"choices": [{"token_ids": toks, "finish_reason": "length", "logprobs": {"token_logprobs": [-0.1] * len(toks)}}]})
        if req.url.path == "/pause":
            state["paused"] = True
        elif req.url.path == "/resume":
            state["paused"] = False
        elif req.url.path == "/collective_rpc":
            await asyncio.sleep(0.02)
            state["weights"] = 1
        return httpx.Response(200, json={})

    eng = VLLMServerEngine("http://vllm", "m", placement="split")
    eng.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def call():  # the runtime's re-issue loop (call-level switching)
        while True:
            r = await eng.generate(GenRequest(rid="r", call_id="c1", prompt_ids=[1], max_tokens=10))
            if r.finish != "abort":
                return r

    task = asyncio.create_task(call())
    await asyncio.sleep(0.045)
    await eng.load_weights("/ckpt/v1", 1)  # disaggregated_stream publishes while the call runs
    r = await task
    assert r.version == 1 and set(r.completion_ids) == {200}  # every token from the stamped version's weights
    assert log.index("/pause") < log.index("/collective_rpc") < log.index("/resume")


# ---- V4 / G2 H5 / V13: SGLang R3 on slime's row contract, sampling_seed -----------------------------------------------------
async def test_v4_sglang_routed_experts_base64_rows_align_with_completion_tokens():
    from crucible.train.sglang_http import SGLangServerEngine

    bodies = []

    async def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        n = body["sampling_params"]["max_new_tokens"]
        start = body.get("routed_experts_start_len", 0)
        rows = np.array([[p * 10 + j for j in range(6)] for p in range(start, len(body["input_ids"]) + n - 1)], dtype="<i4")  # 3 layers x top-2
        meta = {"output_token_logprobs": [[-0.5, 50 + i, None] for i in range(n)], "finish_reason": {"type": "length"},
                "routed_experts": base64.b64encode(rows.tobytes()).decode()}
        return httpx.Response(200, json={"meta_info": meta})

    eng = SGLangServerEngine("http://sglang", routed_experts=True)  # rows of 6 = 3 layers x top-2, read from the reply
    eng.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prompt = list(range(1, 11))  # 10 prompt tokens
    r = await eng.generate(GenRequest(rid="r", call_id="c", prompt_ids=prompt, max_tokens=4, seed=3))
    assert bodies[0]["sampling_params"]["sampling_seed"] == 3 and "seed" not in bodies[0]["sampling_params"]  # V13
    assert len(r.routing) == len(r.completion_ids) == 4
    assert [row[0] for row in r.routing] == [90, 100, 110, 120]  # completion token t: server row len(prompt) - 1 + t
    r2 = await eng.generate(GenRequest(rid="r", call_id="c", prompt_ids=prompt, max_tokens=6, resume_completion=r.completion_ids,
                                       resume_logprobs=r.logprobs, resume_routing=r.routing, resume_spans=[(0, 0)]))
    assert bodies[1]["routed_experts_start_len"] == 13 and len(r2.routing) == len(r2.completion_ids) == 6
    assert [row[0] for row in r2.routing] == [90, 100, 110, 120, 130, 140]


# ---- V5: a refused HF load leaves the serving model untouched -----------------------------------------------------------------
async def test_v5_hf_engine_refused_load_changes_no_tensor():
    transformers = pytest.importorskip("transformers")
    from crucible.train.hf_engine import HFGenerateEngine

    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    eng = HFGenerateEngine(transformers.GPT2LMHeadModel(cfg), device="cpu")
    before = {k: v.clone() for k, v in eng.model.state_dict().items()}
    new = transformers.GPT2LMHeadModel(cfg).state_dict()
    bad = {**{k: v for k, v in new.items() if not k.startswith("transformer.h.1.")}, "extra.weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="do not match"):
        await eng.load_weights(bad, 1)
    assert eng.version == 0 and all(torch.equal(v, before[k]) for k, v in eng.model.state_dict().items())


# ---- V6, V7, V8, V9: the gateway --------------------------------------------------------------------------------------------
async def test_v6_gateway_publish_reaches_the_held_elastic_group():
    from crucible.train.gateway import GatewayPool

    engines = [TorchEngine(TinyPolicy(seed=0)) for _ in range(3)]
    gw = GatewayPool(engines)
    await gw.hold([2])
    await gw.load_weights(TinyPolicy(seed=1).state_dict(), 1)
    await gw.release()
    assert [e.version for e in engines] == [1, 1, 1]


class _Restartable(TorchEngine):
    async def health(self):
        return True


async def test_v7_gateway_catches_up_an_engine_restarted_between_probes():
    from crucible.train.gateway import GatewayPool

    engines = [_Restartable(TinyPolicy(seed=0), decode_delay_s=0.0) for _ in range(2)]
    gw = GatewayPool(engines)
    await gw.load_weights(TinyPolicy(seed=5).state_dict(), 7)
    engines[1].__init__(TinyPolicy(seed=0), decode_delay_s=0.0)  # the launcher restarted it: fresh weights, version 0
    await gw.check_health()
    assert [e.version for e in engines] == [7, 7] and all(gw.alive)
    rs = await asyncio.gather(*(gw.generate(GenRequest(rid=f"r{i}", call_id="c", prompt_ids=[1, 2], max_tokens=2)) for i in range(6)))
    assert {r.version for r in rs} == {7}


async def test_v8_gateway_revival_replays_the_published_snapshot_not_live_weights():
    from crucible.train.gateway import GatewayPool

    trainer = TinyPolicy(seed=0)
    engines = [TorchEngine(TinyPolicy(seed=0)) for _ in range(2)]
    gw = GatewayPool(engines)
    await gw.load_weights(trainer.state_dict(), 1)  # live references, as HFPolicy.state_dict() returns
    snap = {k: v.clone() for k, v in trainer.state_dict().items()}
    with torch.no_grad():
        for p in trainer.parameters():
            p.add_(0.5)  # the trainer keeps training after the publish
    gw.evict(1)
    engines[1].version = 0
    assert await gw.revive(1)
    served = engines[1]._serve.state_dict()
    assert engines[1].version == 1 and all(torch.equal(served[k].float(), snap[k].float()) for k in snap)


async def test_v9_gateway_deterministic_error_is_a_named_failure_not_endless_aborts():
    from crucible.train.gateway import GatewayPool

    class Broken(TorchEngine):
        calls = 0

        async def generate(self, req):
            Broken.calls += 1
            raise ValueError("SGLang routed_experts: 7 ids do not split into 3 rows")  # a reply that never parses

    engines = [Broken(TinyPolicy(seed=0)) for _ in range(2)]
    gw = GatewayPool(engines, failure_threshold=3)
    req = GenRequest(rid="r", call_id="c", prompt_ids=[1], max_tokens=2)
    with pytest.raises(Exception) as ei:
        for _ in range(50):  # the runtime's re-issue loop
            res = await gw.generate(req)
            assert res.finish == "abort"
    assert getattr(ei.value, "failure_kind", None) == "provider_unavailable" and Broken.calls <= 3
    assert all(gw.alive) and gw.stats["evictions"] == 0  # the engines answered: none is evicted for a bad request


# ---- V10: the engine never samples past the end of the turn into the trained record -----------------------------------------
async def test_v10_engine_llm_completion_ends_at_the_first_end_of_turn():
    async def gen(prompt, n, t):  # an engine with no stop ids (RuntimeConfig.stop_ids defaults to ()) samples on
        ids = list(b"Hello") + [codec.EOS] + [codec.BOS] + list(b"user\nwhat next") + [codec.EOS]
        return GenResult(ids, [-0.1 * (i + 1) for i in range(len(ids))], 0, "length")

    llm = EngineLLM(codec, gen)
    r = await llm.complete([{"role": "user", "content": "hi"}])
    comp = r.tokens["completion_token_ids"]
    assert comp == list(b"Hello") + [codec.EOS] and r.text == "Hello" and r.raw_finish == "stop"
    assert r.tokens["logprobs"] == pytest.approx([-0.1 * (i + 1) for i in range(len(comp))])  # its own behaviour log-probs


# ---- V12: the seed travels per call (tito) ------------------------------------------------------------------------------------
async def test_v12_tito_sends_each_calls_seed(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from crucible.actors.llm import LLMClient
    from crucible.core.config import Config, LLMProvider
    from crucible.train.vllm_http import VLLMServerEngine

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except OSError:
        pytest.skip("tokenizer not cached")
    seeds = []

    async def fake_generate(self, req):
        seeds.append(req.seed)
        ids = tok.encode("ok", add_special_tokens=False) + [tok.eos_token_id]
        return GenResult(ids, [-0.5] * len(ids), 0, "stop", None, len(req.prompt_ids))

    monkeypatch.setattr(VLLMServerEngine, "generate", fake_generate)
    cfg = Config(llm_providers={"lab": LLMProvider(base_url="http://127.0.0.1:1/v1", tito=True)}, models={"p": "lab:Qwen/Qwen2.5-0.5B-Instruct"},
                 data_dir=str(tmp_path))
    llm = LLMClient("p", config=cfg, cache_mode="off")
    msgs = [{"role": "user", "content": "x"}]
    await llm.complete(msgs, temperature=0.0, max_tokens=8, seed=0, sample_key="greedy")
    for i in range(3):
        await llm.complete(msgs, temperature=1.0, max_tokens=8, seed=None, sample_key=f"m{i}")
    assert seeds == [0, None, None, None]


# ---- V14 / G3 D6: a resume aborted before its first token opens no span -------------------------------------------------------
async def test_v14_resume_with_no_new_tokens_keeps_spans_valid():
    from crucible.core.schema.rollout import Turn
    from crucible.train.sglang_http import SGLangServerEngine

    plan = iter([(4, "abort"), (0, "abort"), (3, "stop")])

    async def handler(req):
        n, fin = next(plan)
        return httpx.Response(200, json={"meta_info": {"output_token_logprobs": [[-0.1, 60 + i, None] for i in range(n)], "finish_reason": {"type": fin}}})

    eng = SGLangServerEngine("http://sglang")
    eng.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    req = GenRequest(rid="r", call_id="c", prompt_ids=[1, 2], max_tokens=10)
    for v in range(3):  # the runtime's token-level resume, one version per attempt
        eng.version = v
        res = await eng.generate(req)
        req.resume_completion, req.resume_logprobs = res.completion_ids, res.logprobs
        req.resume_spans = list(res.version_spans or [(0, res.version)])
    assert res.finish == "stop" and res.version_spans == [(0, 0), (4, 2)]
    t = Turn(call_id="c", actor="a", completion_token_ids=res.completion_ids, logprobs=res.logprobs, policy_version=res.version,
             version_spans=res.version_spans)
    assert t.spans_valid() and t.versions() == [0, 2]


# ---- V15: a nudge after an empty reply is a new segment, never a prefix break -------------------------------------------------
async def test_v15_rerender_after_a_served_call_is_flagged_new_segment():
    outs = iter([[codec.EOS], list(b"fine") + [codec.EOS]])

    async def gen(prompt, n, t):
        ids = next(outs)
        return GenResult(ids, [-0.1] * len(ids), 0, "stop")

    llm = EngineLLM(codec, gen)
    m = [{"role": "user", "content": "go"}]
    await llm.complete(m, sample_key="e")
    r = await llm.complete(m + [{"role": "user", "content": "Please continue."}], sample_key="e")  # LLMToolPolicy's NUDGE
    assert r.tokens.get("new_segment") is True


# ---- V16: a reaped attempt's model call dies with its lease -------------------------------------------------------------------
async def test_v16_reap_cancels_the_dead_attempts_model_calls():
    from crucible.train.workers import LeaseController

    tag = {"versions": [], "length": 0}

    async def runtime_gen(ids, max_tokens, temperature):
        await asyncio.sleep(0.3)
        tag["versions"].append(4)
        tag["length"] += 7
        return GenResult([5] * 7, [-0.1] * 7, 4, "stop")

    lc = LeaseController(lease_s=0.1)
    lc.on_reset = lambda rid: tag.update(versions=[], length=0)
    fut = asyncio.create_task(lc.rollout_fn()("r1", "p", 0, runtime_gen))
    await asyncio.sleep(0)
    job = lc.lease("w1", 1)[0]
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=lc.app()), base_url="http://c")
    zombie = asyncio.create_task(client.post("/gen", json={"worker": "w1", "rid": "r1", "attempt": job["attempt"], "ids": [1], "max_tokens": 4}))
    await asyncio.sleep(0.15)
    lc.reap()
    r = await zombie
    assert r.status_code == 409 and tag == {"versions": [], "length": 0}  # the live attempt's tag is its own
    fut.cancel()
    await asyncio.gather(fut, return_exceptions=True)


# ---- V17, V18: chat codecs --------------------------------------------------------------------------------------------------
def test_v17_tool_call_arguments_are_always_a_dict():
    from crucible.train.chat import parse_tool_calls

    _, calls = parse_tool_calls('<tool_call>{"name": "f", "arguments": "{\\"x\\": 1}"}</tool_call><tool_call>{"name": "g", "arguments": [1, 2]}</tool_call>')
    assert calls[0]["arguments"] == {"x": 1}  # the OpenAI string encoding is decoded
    assert set(calls[1]["arguments"]) == {"__unparseable__"}  # a non-object is the policy's format error, not a crash


def test_v18_gemma_codec_frames_turns_with_end_of_turn_and_folds_the_system_message():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from crucible.train.chat import HFChatCodec

    try:
        tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    except OSError:
        pytest.skip("gemma tokenizer not cached")
    c = HFChatCodec(tok)
    assert c.eos_ids[0] == tok.convert_tokens_to_ids("<end_of_turn>") and tok.eos_token_id in c.eos_ids
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    p = c.render(msgs, None)  # the template refuses a system turn: folded into the first user message
    comp = tok.encode("hello", add_special_tokens=False) + [c.eos_ids[0]]
    suf = c.suffix([{"role": "user", "content": "again"}], None)
    assert p + comp + suf == c.render(msgs + [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "again"}], None)


# ---- V19 / G3 D8: honest capabilities, token resume on vLLM, /abort_requests optional ---------------------------------------
async def test_v19_vllm_capabilities_and_token_resume_sends_the_kept_tokens():
    from crucible.train.gateway import HTTPEngine
    from crucible.train.vllm_http import VLLMServerEngine

    bodies = []

    async def handler(req):
        if req.url.path == "/abort_requests":
            return httpx.Response(404, json={"detail": "Not Found"})
        body = json.loads(req.content)
        bodies.append(body)
        return httpx.Response(200, json={"choices": [{"token_ids": [9] * body["max_tokens"], "finish_reason": "length",
                                                      "logprobs": {"token_logprobs": [-0.2] * body["max_tokens"]}}]})

    eng = VLLMServerEngine("http://vllm", "m")
    eng.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    caps = eng.capabilities()
    assert caps["pause"] == ["abort"] and caps["token_resume"] is True
    eng.version = 1
    r = await eng.generate(GenRequest(rid="r", call_id="c", prompt_ids=[1, 2], max_tokens=5, resume_completion=[7, 8], resume_logprobs=[-1.0, -1.0],
                                      resume_spans=[(0, 0)]))
    assert bodies[0]["prompt"] == [1, 2, 7, 8] and bodies[0]["max_tokens"] == 3
    assert r.completion_ids == [7, 8, 9, 9, 9] and r.logprobs[:2] == [-1.0, -1.0] and r.version_spans == [(0, 0), (2, 1)]
    assert await eng.abort_all() == 0  # a vLLM build without /abort_requests is not an engine failure
    assert HTTPEngine("http://e").capabilities()["top_p_candidates"] is False  # unknown until the server says so


# ---- G3 D11: SGLang weight_version ----------------------------------------------------------------------------------------------
async def test_d11_sglang_stamps_the_servers_weight_version_and_sends_it_on_update():
    from crucible.train.sglang_http import SGLangServerEngine

    seen = []

    async def handler(req):
        body = json.loads(req.content) if req.content else {}
        seen.append((req.url.path, body))
        if req.url.path == "/generate":
            return httpx.Response(200, json={"meta_info": {"output_token_logprobs": [[-0.1, 5, None]], "finish_reason": {"type": "stop"}, "weight_version": "4"}})
        return httpx.Response(200, json={"success": True})

    eng = SGLangServerEngine("http://sglang", placement="split")
    eng.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await eng.load_weights("/ckpt/v4", 4)
    assert ("/update_weights_from_disk", {"model_path": "/ckpt/v4", "weight_version": "4"}) in seen
    eng.version = 3  # a client counter behind the server: the server's stamp wins
    r = await eng.generate(GenRequest(rid="r", call_id="c", prompt_ids=[1], max_tokens=2))
    assert r.version == 4


# ---- V20: the publisher never burns a version on a failed send ---------------------------------------------------------------
async def test_v20_failed_send_keeps_the_publishers_version():
    from crucible.train.weightsync import DistributedPublisher

    class Sync:
        plan_ = object()

        def update(self, version, state):
            raise RuntimeError("NCCL timeout")

    class T:
        def state_dict(self):
            return {"w": torch.zeros(2)}

    pub = DistributedPublisher(T(), Sync())
    with pytest.raises(RuntimeError):
        await pub.publish()
    assert pub.version == 0


# ---- V22: generate_app returns candidates and never a partial abort as "length" ----------------------------------------------
async def test_v22_generate_app_candidates_and_abort_limit():
    pytest.importorskip("fastapi")
    from crucible.train.gateway import generate_app

    class Pool:
        def __init__(self, fin):
            self.fin = fin

        async def generate(self, g):
            return GenResult([3, 4], [-0.1, -0.2], 2, self.fin, None, len(g.prompt_ids), [[3, 5], [4]])

    body = {"token_ids": [1, 2], "sampling_params": {"max_tokens": 2, "top_p": 0.9}}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=generate_app(Pool("length"), model="m")), base_url="http://g") as c:
        d = (await c.post("/inference/v1/generate", json=body)).json()
    assert d["choices"][0]["candidates"] == [[3, 5], [4]]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=generate_app(Pool("abort"), model="m", max_aborts=3)), base_url="http://g") as c:
        r = await c.post("/inference/v1/generate", json=body)
    assert r.status_code >= 500  # not a partial result labelled "length"


# ---- V23: per-call abort marks only running calls; kill reaps the whole group ----------------------------------------------------
async def test_v23_hf_abort_of_an_idle_call_does_not_kill_its_reissue():
    transformers = pytest.importorskip("transformers")
    from crucible.train.hf_engine import HFGenerateEngine

    cfg = transformers.GPT2Config(vocab_size=64, n_positions=64, n_embd=16, n_layer=1, n_head=2)
    eng = HFGenerateEngine(transformers.GPT2LMHeadModel(cfg), device="cpu")
    await eng.abort("c1")  # the call is not running (it just finished, or is about to be re-issued)
    r = await eng.generate(GenRequest(rid="r", call_id="c1", prompt_ids=[1, 2], max_tokens=3, seed=0))
    assert r.finish != "abort" and len(r.completion_ids) == 3


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_v23_child_kill_reaps_grandchildren_after_the_leader_exited(tmp_path):
    from crucible.train.launch import Child

    pidfile = tmp_path / "gc.pid"
    ch = Child("engine", "e", ["/bin/sh", "-c", f"sleep 60 & echo $! > {pidfile}; exit 0"])
    ch.start(tmp_path)
    ch.proc.wait()  # the leader exited; its grandchild still holds the group
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        time.sleep(0.02)
    gc = int(pidfile.read_text())
    ch.kill()
    for _ in range(100):
        try:
            os.kill(gc, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        os.kill(gc, signal.SIGKILL)
        pytest.fail("the grandchild survived Child.kill()")


# ---- H1 V7: teachers score the student's distribution (temperature, top-p nucleus) ----------------------------------------
async def test_h1v7_teacher_equal_to_the_policy_scores_the_behaviour_logprobs_exactly():
    from crucible.core.schema.rollout import RolloutRecord, Turn
    from crucible.train.teachers import TeacherPool, TorchTeacher

    pol = TinyPolicy(32, 24, seed=3)
    prompt, comp = [1, 2, 3], [4, 5, 6, 7]
    for temp, cands in ((0.6, None), (1.0, [[4, 9], [5, 1, 2], [6], [7, 8, 3]]), (0.7, [[4, 9], [5, 1], [6, 0], [7]])):
        with torch.no_grad():
            beh = pol.token_logprobs(prompt, comp, temp, candidates=cands).tolist()  # what the engine reports
        turn = Turn(call_id="c", actor="a", prompt_token_ids=prompt, completion_token_ids=comp, logprobs=beh, policy_version=0,
                    sampling={"temperature": temp, **({"top_p": 0.8} if cands else {})}, candidates=cands)
        rec = RolloutRecord(rollout_id="r", sample="r", task="t", harness="h", policy="p", reward=1.0, turns=[turn])
        pool = TeacherPool([TorchTeacher(pol, id="self", vocab_id="v")], default="self", reference=TorchTeacher(pol, id="ref", vocab_id="v"))
        await pool.teach(rec, None)
        gap = [a - b for a, b in zip(rec.turns[0].teacher_logprobs, beh, strict=True)]
        assert max(abs(g) for g in gap) < 1e-5, (temp, cands, gap)  # OPD signal 0 when the teacher is the policy
        assert rec.turns[0].ref_logprobs == pytest.approx(beh, abs=1e-5)  # k3 KL 0 at the reference


async def test_h1v7_vllm_teacher_renormalizes_over_the_nucleus_at_the_turns_temperature():
    from crucible.train.teachers import VLLMTeacher

    raw = {4: math.log(0.5), 9: math.log(0.3), 11: math.log(0.2)}

    def handler(req):
        body = json.loads(req.content)
        assert body["prompt_logprobs"] >= 2
        pl = [None] * 3 + [{str(t): {"logprob": lp} for t, lp in raw.items()}]
        return httpx.Response(200, json={"choices": [{"prompt_logprobs": pl}]})

    t = VLLMTeacher("http://t/v1", "m", id="t", vocab_id="v", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    T = 0.5
    rows = await t.score([([1, 2, 3], [4])], temperatures=[T], candidates=[[[4, 9]]])
    want = raw[4] / T - math.log(math.exp(raw[4] / T) + math.exp(raw[9] / T))
    assert rows[0] == pytest.approx([want])
    with pytest.raises(NotImplementedError):  # full vocabulary at T != 1 cannot be rebuilt from top-k: refused, not guessed
        await t.score([([1, 2, 3], [4])], temperatures=[0.5], candidates=[None])


# ---- V11, V20, G5 P2-f, P2-g: weight sync over gloo -----------------------------------------------------------------------------
WORKER = Path(__file__).with_name("_weightsync_worker.py")


def _ws(scenario: str, tmp_path: Path, world: int, timeout_s: float = 20.0) -> list[dict]:
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")
    env = {**os.environ, "WORLD_SIZE": str(world), "CRUCIBLE_WS_TIMEOUT_S": str(timeout_s),
           "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")}
    procs = [subprocess.Popen([sys.executable, str(WORKER), scenario, str(tmp_path)], env={**env, "RANK": str(r)}, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for r in range(world)]
    errs = []
    for p in procs:
        try:
            _, err = p.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
            _, err = p.communicate()
            err = "TIMEOUT\n" + err
        if p.returncode:
            errs.append(err[-2000:])
    assert not errs, errs
    return [json.loads((tmp_path / f"r{r}.json").read_text()) for r in range(world)]


@pytest.mark.parametrize("scenario", ["m12_owner_fails", "m12_dtype_drift", "m12_stage_raises"])
def test_v11_owner_failure_raises_weightsyncerror_on_every_rank_promptly(tmp_path, scenario):
    rs = _ws(scenario, tmp_path, world=3)
    for r in rs:
        assert r["raised"] and not r["raised"].startswith("OTHER"), r  # WeightSyncError, not a backend timeout
        assert r["raised"].startswith("rank 0") and r["seconds"] < 10, r  # every rank names the owner, well inside the timeout
        assert "committed" not in r


def test_v20_weightsync_refuses_a_missing_version_and_moves_fp8(tmp_path):
    rs = _ws("m12_no_version", tmp_path, world=2)
    assert all(r["raised"] and "no version" in r["raised"] and "committed" not in r for r in rs)
    rs = _ws("m12_fp8", tmp_path / "fp8", world=2) if (tmp_path / "fp8").mkdir() is None else None
    assert rs[0]["raised"] is None and rs[1]["got"] == 1 and rs[0]["d"] == rs[1]["d"]


def test_g5_weight_group_without_global_rank_0_and_one_cached_subgroup(tmp_path):
    rs = _ws("m12_group_ranks", tmp_path, world=3)
    assert rs[1]["raised"] is None and rs[2]["got"] == 3 and rs[1]["d"] == rs[2]["d"] and rs[1]["owners"] == [0]
    (tmp_path / "sub").mkdir()
    rs = _ws("m12_subgroup_cached", tmp_path / "sub", world=3)
    assert all(r["new_groups"] <= 1 for r in rs), rs  # one communicator for five updates (it was one per update)
