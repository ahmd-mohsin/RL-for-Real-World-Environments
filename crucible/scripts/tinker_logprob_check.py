"""Sampler-vs-trainer logprob check (about USD 0.01): one base-model rollout of the smoke, its datum sent through a fresh
LoRA training client's forward pass (LoRA starts at zero, so the learner is the base model the sampler used), and the
per-token differences of the policy's own tokens. Usage: tinker_logprob_check.py <env-file> <rollout.json> [model]."""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tinker_rl

from crucible.core.schema.rollout import RolloutRecord
from crucible.train.bridges.tinker.datums import build_datums
from crucible.train.bridges.tinker.models import resolve
from crucible.train.bridges.tinker.spend import SpendMeter

tinker_rl.load_env_key(Path(sys.argv[1]))
ro = RolloutRecord.model_validate_json(Path(sys.argv[2]).read_text())
model = resolve(sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3.6-35B-A3B")
ro = ro.model_copy(update={"reward": 1.0})
other = ro.model_copy(update={"rollout_id": ro.rollout_id + "-b", "reward": 0.0})
d = build_datums([ro, other], ["g", "g"], turn_eta=0.0)[0]
OUT = Path(__file__).resolve().parents[1] / "data" / "tinker" / "kl-check"
OUT.mkdir(parents=True, exist_ok=True)
meter = SpendMeter(0.05, path=OUT / "spend.jsonl")


async def go():
    import tinker

    svc = tinker.ServiceClient()
    tc = await svc.create_lora_training_client_async(base_model=model.name, rank=32)
    datum = tinker.Datum(model_input=tinker.ModelInput.from_ints(d.input_ids),
                         # the sampling logprobs are sent as zeros: an output equal to them would be an echo of our input
                         loss_fn_inputs={"target_tokens": d.target_ids, "logprobs": [0.0] * len(d.logprobs), "advantages": [0.0] * len(d.advantages)})
    with meter.hold(model.prices.usd(train=len(d.input_ids)), "forward"):
        fut = await tc.forward_async([datum], "importance_sampling")
        out = await fut.result_async()
        meter.book("train", model, train=len(d.input_ids), op="forward")
    lt = out.loss_fn_outputs[0]["logprobs"].tolist()
    diffs = [s - t for s, t, m in zip(d.logprobs, lt, d.mask, strict=True) if m]
    return {"tokens": len(diffs), "mean_diff": sum(diffs) / len(diffs), "max_abs_diff": max(abs(x) for x in diffs),
            "identical": sum(1 for x in diffs if x == 0.0), "usd": meter.spent, "keys": list(out.loss_fn_outputs[0].keys())}


res = asyncio.run(go())
(OUT / "result.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
os.environ.pop("TINKER_API_KEY", None)
