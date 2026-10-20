"""8-GPU training smoke on one Marlowe node: Crucible's FSDP2 trainer (``train.fsdp.FSDPTrainer``, HSDP mesh by default)
on Qwen2.5-7B-Instruct with scripted multi-turn rollouts on the token credit path (``Turn.advantage`` per call, weighted by
``TrainerConfig.turn_weight``), a sharded checkpoint, a forced stop and a restore in a new process group.

    srun --ntasks-per-node=8 python train_smoke.py --phase a --out $RUN/train     # steps 1..4; checkpoint after step 2;
                                                                                  # then waits for SIGTERM and checkpoints
    srun --ntasks-per-node=8 python train_smoke.py --phase b --out $RUN/train     # restore step 2, redo steps 3..4, compare

Phase a writes ``ref.json`` (metrics of steps 3..4 and per-rank shard checksums at step 2) and, on SIGTERM, a step-4
checkpoint plus ``term.json`` (seconds from the signal to a finished checkpoint: must fit the 900 s grace). Phase b
(new processes, as after a requeue on any node) restores step 2 (checksums must match bitwise), re-runs steps 3..4 on the
same batches (losses and gradient norms within ``--tol``), then compares its weights with the step-4 checkpoint and
writes ``verdict.json``. On a real preemption the same handler checkpoints whatever step was reached.

``--model tiny`` with ``CRUCIBLE_CPU_ONLY=1`` and fake ``SLURM_*`` variables runs the logic on gloo (how it was checked
before any GPU time).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

STOP = {"sig": None, "t": None}


def _on_term(signum, _frame) -> None:
    if STOP["sig"] is None:
        STOP["sig"], STOP["t"] = signal.Signals(signum).name, time.time()


def _flag_any(dev) -> bool:
    """Every rank agrees whether any rank was signalled (Slurm signals ranks at slightly different times)."""
    import torch
    import torch.distributed as dist

    t = torch.tensor([1.0 if STOP["sig"] else 0.0], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


SCRIPTED = [  # (user, [(assistant, tool result), ...]): the toy coding task's turns, as a policy would act them
    ("Fix calc.add so that add(2, 3) == 5 and make mean([]) return 0.0. Use the tools.",
     [('{"tool": "repo.read_file", "path": "calc.py"}', "1 def add(a, b):\n2     return a - b\n3 def mean(xs):\n4     return sum(xs) / len(xs)"),
      ('{"tool": "repo.edit_file", "path": "calc.py", "old_string": "return a - b", "new_string": "return a + b"}', "edited calc.py"),
      ('{"tool": "repo.shell", "command": "python -m pytest -q"}', "2 passed in 0.03s")]),
    ("The test suite fails on an empty list. Investigate and fix it without changing the tests.",
     [('{"tool": "repo.search", "pattern": "len\\\\(xs\\\\)"}', "calc.py:4:    return sum(xs) / len(xs)"),
      ('{"tool": "repo.edit_file", "path": "calc.py", "old_string": "sum(xs) / len(xs)", "new_string": "sum(xs) / len(xs) if xs else 0.0"}', "edited calc.py"),
      ("Done: mean([]) now returns 0.0 and the suite passes.", "")]),
]


def records(tok, n: int, seed: int = 0):
    """``n`` rollouts in groups of 4 over the scripted conversations: each assistant message is one trained call whose
    prompt is the chat-templated conversation so far, with a per-call credit ``Turn.advantage`` (the checkpoint sidecar's
    ΔΦ in a real run) and an outcome reward that differs within each group."""
    import random

    from crucible.core.schema.rollout import RolloutRecord, SamplingParams, Turn

    rng = random.Random(seed)
    recs, groups = [], []
    for i in range(n):
        user, steps = SCRIPTED[i % len(SCRIPTED)]
        msgs = [{"role": "system", "content": "You are a coding agent. Reply with one JSON tool call per turn."}, {"role": "user", "content": user}]
        turns = []
        for k, (assistant, tool_out) in enumerate(steps):
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            prompt = [int(x) for x in (prompt["input_ids"] if hasattr(prompt, "keys") else prompt)]  # transformers 5: a BatchEncoding
            comp = tok.encode(assistant + (" " * (i % 3)), add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
            turns.append(Turn(call_id=f"r{i}:c{k}", actor="agent", prompt_token_ids=prompt, completion_token_ids=comp, logprobs=[-1.0] * len(comp),
                              policy_version=0, sampling=SamplingParams(temperature=1.0), advantage=round(rng.uniform(-0.5, 0.5), 3)))
            msgs.append({"role": "assistant", "content": assistant})
            if tool_out:
                msgs.append({"role": "user", "content": f"[tool result]\n{tool_out}"})
        recs.append(RolloutRecord(rollout_id=f"r{i}", sample=f"s{i}", task="marlowe/toy-calc", harness="scripted@1", policy="qwen2.5-7b@0",
                                  group=f"g{i // 4}", reward=[1.0, 0.0, 0.5, 0.25][i % 4], turns=turns))
        groups.append(f"g{i // 4}")
    return recs, groups


def tiny_tokenizer():
    class Tok:  # enough of a tokenizer for the CPU check of the logic
        def apply_chat_template(self, msgs, **_):
            return [3 + (ord(c) % 200) for m in msgs for c in (m["role"][:2] + m["content"][:12])] + [1]

        def encode(self, s, **_):
            return [3 + (ord(c) % 200) for c in s[:24]]

        def convert_tokens_to_ids(self, _):
            return 2

    return Tok()


def shard_checksums(tr) -> dict[str, float]:
    """Sums of this rank's local parameter and moment shards in float64: equal bitwise after a correct restore."""
    import torch

    p = m = 0.0
    for prm in tr.model.parameters():
        loc = prm.to_local() if hasattr(prm, "to_local") else prm
        p += float(loc.detach().double().abs().sum())
        for k, v in tr.opt.state.get(prm, {}).items():
            if torch.is_tensor(v) and k != "step":
                vv = v.to_local() if hasattr(v, "to_local") else v
                m += float(vv.detach().double().abs().sum())
    return {"params": p, "moments": m}


def build(a, rank: int, dev):
    """Meta-device build on every rank, the full fp32 state on rank 0 only (scattered shard by shard), behaviour log-probs
    from the initial weights on rank 0 (a real ratio signal)."""
    import torch
    import torch.distributed as dist
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

    from crucible.train.fsdp import FSDPTrainer, meta_params, transformer_blocks
    from crucible.train.hf import HFPolicy
    from crucible.train.mismatch import Correction
    from crucible.train.trainer import TrainerConfig

    tiny = a.model == "tiny"
    if tiny:
        cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2)
        tok = tiny_tokenizer()
    else:
        cfg = AutoConfig.from_pretrained(a.model)
        tok = AutoTokenizer.from_pretrained(a.model)
    full = None
    t0 = time.time()
    if rank == 0:
        torch.manual_seed(0)
        full = LlamaForCausalLM(cfg).float() if tiny else AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    load_s = time.time() - t0
    recs, groups = records(tok, a.items)
    if rank == 0 and a.phase == "a":
        pol = HFPolicy(full.to(dev), lm_head_fp32=True)
        with torch.no_grad():
            recs = [r.model_copy(update={"turns": [t.model_copy(update={"logprobs": [
                float(x) - 0.02 for x in pol.token_logprobs(t.prompt_token_ids, t.completion_token_ids).float().cpu()]}) for t in r.turns]}) for r in recs]
        full.to("cpu")
        del pol
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    # every rank trains the same batches in both phases: phase a's behaviour log-probs are saved with the reference
    box = [recs if rank == 0 else None]
    dist.broadcast_object_list(box, src=0)
    recs = box[0]
    amp = torch.bfloat16 if dev.type == "cuda" else None
    with meta_params():
        empty = HFPolicy(AutoModelForCausalLM.from_config(cfg, dtype=torch.float32) if not tiny else LlamaForCausalLM(cfg).float(), lm_head_fp32=True,
                         gradient_checkpointing=True, autocast_dtype=amp)
    rep, shard = (int(x) for x in a.mesh.split("x"))
    t1 = time.time()
    tr = FSDPTrainer(empty, TrainerConfig(lr=a.lr, correction=Correction("dis"), token_budget=a.token_budget, dp=rep * shard, turn_weight=a.turn_weight,
                                          temperature=1.0),
                     mesh_shape=(rep, shard), reshard="layer", units=transformer_blocks, device=dev, broadcast_batch=True,
                     full_state=full.state_dict() if full is not None else None, mixed_precision="bf16" if dev.type == "cuda" else None)
    del full
    return tr, recs, groups, {"load_s": round(load_s, 1), "shard_s": round(time.time() - t1, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a", "b"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--mesh", default="2x4", help="replicate x shard (HSDP); 1x8 = plain FSDP")
    ap.add_argument("--items", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--ckpt-at", dest="ckpt_at", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--turn-weight", dest="turn_weight", type=float, default=0.5)
    ap.add_argument("--token-budget", dest="token_budget", type=int, default=8192)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--wait-term-s", dest="wait_term_s", type=float, default=600.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for s in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(s, _on_term)

    import torch
    import torch.distributed as dist

    from crucible.train.cluster import init_rank, slurm_env

    if "SLURM_PROCID" in os.environ:
        os.environ.update(slurm_env())
    rank, world, dev = init_rank()
    out = Path(a.out)
    ck = out / "ckpt"
    if rank == 0:
        ck.mkdir(parents=True, exist_ok=True)
    tr, recs, groups, timing = build(a, rank, dev)
    dist.barrier()
    feed = (recs, groups) if rank == 0 else ([], [])
    res: dict = {"rank": rank, "world": world, "host": socket.gethostname(), "mesh": a.mesh, **timing}

    def step(k: int) -> dict:
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        t = time.time()
        m = tr.step(*feed)
        return {"step": k, "loss": m.get("train/loss"), "grad_norm": m.get("train/grad_norm"), "seconds": round(time.time() - t, 2),
                "peak_gb": round(torch.cuda.max_memory_allocated(dev) / 1e9, 2) if dev.type == "cuda" else None,
                "metrics": {k2: v for k2, v in m.items() if isinstance(v, int | float)}}

    if a.phase == "a":
        hist = []
        for k in range(1, a.steps + 1):
            hist.append(step(k))
            if k == a.ckpt_at:
                t = time.time()
                tr.save_sharded(str(ck / f"step-{k}"))
                res["save_s"] = round(time.time() - t, 1)
                res["checksums_at_ckpt"] = shard_checksums(tr)
            if _flag_any(dev):  # a real preemption during the reference steps: checkpoint now and leave
                break
        allr = [None] * world
        dist.all_gather_object(allr, {**res, "hist": hist})
        if rank == 0:
            (out / "ref.json").write_text(json.dumps({"ranks": allr, "records": [r.model_dump(mode="json") for r in recs]}, default=str))
            print(json.dumps({"ref": [{k: h[k] for k in ("step", "loss", "grad_norm", "seconds", "peak_gb")} for h in hist], **timing}), flush=True)
        # the forced stop: wait for SIGTERM (stage driver / Slurm), then checkpoint within the grace period
        t0 = time.time()
        while not _flag_any(dev):
            if time.time() - t0 > a.wait_term_s:
                STOP["sig"], STOP["t"] = "SELF", time.time()
                break
            time.sleep(1.0)
        t = time.time()
        tr.save_sharded(str(ck / f"step-{len(hist)}-term"))
        dist.barrier()
        if rank == 0:
            (out / "term.json").write_text(json.dumps({"signal": STOP["sig"], "step": len(hist), "signal_to_saved_s": round(time.time() - (STOP["t"] or t), 1),
                                                       "save_s": round(time.time() - t, 1), "host": socket.gethostname(), "grace_s": 900}))
            print("term checkpoint written", flush=True)
    else:
        ref = json.loads((out / "ref.json").read_text())
        from crucible.core.schema.rollout import RolloutRecord

        recs = [RolloutRecord.model_validate(r) for r in ref["records"]]  # the same batches, behaviour log-probs included
        feed = (recs, groups) if rank == 0 else ([], [])
        mine = ref["ranks"][rank]
        t = time.time()
        tr.load_sharded(str(ck / f"step-{a.ckpt_at}"))
        res["load_ckpt_s"] = round(time.time() - t, 1)
        got = shard_checksums(tr)
        res["restore_exact"] = got == mine["checksums_at_ckpt"]
        res["updates_after_restore"] = tr.updates
        hist = [step(k) for k in range(a.ckpt_at + 1, a.steps + 1)]
        refh = {h["step"]: h for h in mine["hist"]}
        rel = []
        for h in hist:
            r0 = refh.get(h["step"])
            if r0:
                for key in ("loss", "grad_norm"):
                    x, y = h[key], r0[key]
                    if x is not None and y is not None:
                        rel.append(abs(x - y) / max(abs(y), 1e-8))
        res["max_rel_vs_ref"] = max(rel, default=None)
        # weights after the redone steps vs the checkpoint phase a wrote on SIGTERM (same step count)
        term = json.loads((out / "term.json").read_text())
        if term["step"] == a.steps:
            mine_now = {n: (p.to_local() if hasattr(p, "to_local") else p).detach().float().clone() for n, p in tr.model.named_parameters()}
            tr.load_sharded(str(ck / f"step-{term['step']}-term"))
            worst = 0.0
            for n, p in tr.model.named_parameters():
                loc = (p.to_local() if hasattr(p, "to_local") else p).detach().float()
                worst = max(worst, float((loc - mine_now[n]).norm() / (loc.norm() + 1e-12)))
            res["max_rel_weight_vs_term_ckpt"] = worst
        allr = [None] * world
        dist.all_gather_object(allr, {**res, "hist": hist})
        if rank == 0:
            ok_restore = all(r["restore_exact"] for r in allr)
            rels = [r["max_rel_vs_ref"] for r in allr if r["max_rel_vs_ref"] is not None]
            wrel = [r.get("max_rel_weight_vs_term_ckpt") for r in allr if r.get("max_rel_weight_vs_term_ckpt") is not None]
            verdict = {"restore_exact_all_ranks": ok_restore, "max_rel_metrics_vs_ref": max(rels, default=None), "max_rel_weight_vs_term_ckpt": max(wrel, default=None),
                       "term": term, "steps_redone": [h["step"] for h in hist], "losses": [h["loss"] for h in hist], "peak_gb_per_rank": [max(h["peak_gb"] or 0 for h in r["hist"]) for r in allr],
                       "world": world, "mesh": a.mesh, "host": socket.gethostname(), "tol": a.tol}
            verdict["pass"] = bool(ok_restore and rels and max(rels) <= a.tol and term["signal_to_saved_s"] < 900 and (not wrel or max(wrel) <= a.tol))
            (out / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
            print(json.dumps(verdict, indent=1, default=str), flush=True)
            if not verdict["pass"]:
                dist.destroy_process_group()
                sys.exit(1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
