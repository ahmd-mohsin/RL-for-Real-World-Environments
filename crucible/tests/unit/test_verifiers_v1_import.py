"""m13 E (audit 03 G6, pick 6): verifiers v1 environments run their own episodes (harness, toolsets, rubric) with
Crucible supplying the model over the vLLM /inference/v1/generate protocol; traces become train-valid
crucible.rollout/v1 records with exact tokens; the M12 runtime trains on a verifiers environment."""

from __future__ import annotations

import asyncio
import json
import re
import socket
import tempfile
from pathlib import Path

import pytest

from crucible.interop.verifiers.v1_import import VF_PY, run_vf_eval, trace_to_record, vf_rollout
from crucible.train.engine import EnginePool, ScriptedEngine
from crucible.train.rollout import full_prompts

pytestmark = pytest.mark.skipif(not (VF_PY / "vf-eval").exists(), reason="needs crucible/.envs/verifiers with the glossary env installed")


def _tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except OSError:
        pytest.skip("Qwen3 tokenizer not cached")


def _oracle(tok):
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    def script(rid, k, prompt):  # looks the entity up, then answers with what the tool returned
        text = tok.decode(prompt)
        if "<tool_response>" not in text:
            m = re.search(r'look up "([^"]+)"', text)
            call = {"name": "facts_lookup", "arguments": {"name": m.group(1) if m else "x"}}
            return tok.encode("<tool_call>\n" + json.dumps(call) + "\n</tool_call>", add_special_tokens=False) + [eos]
        fact = text.rsplit("<tool_response>", 1)[1].split("</tool_response>", 1)[0].strip()
        return tok.encode(f"<answer>{fact}</answer>", add_special_tokens=False) + [eos]

    return script


async def test_verifiers_env_on_crucible_engines_gives_exact_records():
    import uvicorn

    from crucible.train.gateway import generate_app

    tok = _tok()
    pool = EnginePool([ScriptedEngine(_oracle(tok), capacity=64)])
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(generate_app(pool, model="Qwen/Qwen3-0.6B"), host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    try:
        rows = await run_vf_eval("glossary", f"http://127.0.0.1:{port}/v1", "Qwen/Qwen3-0.6B", out_dir=Path(tempfile.mkdtemp()))
    finally:
        server.should_exit = True
        await task
    traces = [t for r in rows for t in r["traces"]]
    assert len(traces) == 5
    for i, t in enumerate(traces):
        rec = trace_to_record(t, f"g{i}", policy_version=0)
        ok, why = rec.valid_for_training()
        assert ok, why
        assert rec.reward == 1.0 and len(rec.turns) == 2  # verifiers' own rubric: the looked-up fact is in the answer
        nodes = [n for n in t["nodes"]]
        last = nodes[-1]
        first_true = last["mask"].index(True)
        want = [x for n in nodes[:-1] for x in n["token_ids"]] + last["token_ids"][:first_true]
        assert full_prompts(rec)[rec.turns[-1].call_id] == want  # the trainer sees exactly verifiers' token graph


async def test_m12_runtime_trains_on_a_verifiers_environment():
    from crucible.train.engine import TinyPolicy
    from crucible.train.mismatch import Correction
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.trainer import TorchTrainer, TrainerConfig
    from crucible.train.weights import InprocPublisher

    tok = _tok()
    eng = EnginePool([ScriptedEngine(_oracle(tok), capacity=64)])
    tr = TorchTrainer(TinyPolicy(len(tok), 8, seed=0), TrainerConfig(lr=1e-3, correction=Correction("dis")))
    fn = vf_rollout("glossary", "Qwen/Qwen3-0.6B")
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=2, max_turn_tokens=256, warmup_batches=0, max_wall_s=300),
                 SchedulerConfig(n=2, B=2, max_staleness=2), eng, tr, InprocPublisher(tr, eng), None, {"glossary": [str(i) for i in range(5)] * 4},
                 rollout_fn=fn)
    try:
        rep = await rt.run()
    finally:
        await fn.server.stop()
    assert len(rep.steps) == 2 and rep.masked_rollouts == 0 and len(rep.trained_records) == 8  # 2 steps x B=2 groups x n=2
    for r in rep.trained_records:
        ok, why = r.valid_for_training()
        assert ok, why
        assert r.reward == 1.0 and all(t.policy_version is not None for t in r.turns)  # stamped by the runtime's own engine calls
