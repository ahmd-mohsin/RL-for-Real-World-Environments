"""Worker for ``test_m12_audit_math.py``'s data-parallel cases (M12 audit H4 V1 / V2 / V12, G5 D1, G4 #2): one process per
rank on a gloo group; rank 0 also runs the single-process reference and writes the comparison."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.train.engine import TinyPolicy
from crucible.train.trainer import DistributedTorchTrainer, TorchTrainer, TrainerConfig


class SideHeadPolicy(TinyPolicy):
    """A policy with a parameter only some rows reach (an adapter / vision tower / idle expert): prompts that start with
    token 31 add a side bias."""

    def __init__(self) -> None:
        super().__init__(32, 8, seed=3)
        self.side = torch.nn.Parameter(torch.full((32,), 0.1))

    def token_logprobs(self, prompt, completion, temperature=1.0, candidates=None):
        ids = torch.tensor([prompt + completion[:-1]])
        logits = self(ids)[0, len(prompt) - 1:] / temperature
        if prompt[0] == 31:
            logits = logits + self.side
        lp = torch.log_softmax(logits, -1)
        return lp[torch.arange(len(completion)), torch.tensor(completion)]


def rec(rid: str, prompt: list[int], comp: list[int], reward: float) -> RolloutRecord:
    t = Turn(call_id="c0", actor="agent", prompt_token_ids=prompt, completion_token_ids=comp, logprobs=[-2.0] * len(comp), policy_version=0)
    return RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", reward=reward, turns=[t])


def batch(kind: str) -> tuple[list[RolloutRecord], list[str]]:
    if kind == "one_item":  # fewer rows than ranks: rank 1 gets nothing
        return [rec("a", [1, 2], [3, 4, 5], 1.0), rec("b", [1, 2], [3, 4, 6], 0.0)][:2], ["g", "g"]
    if kind == "side":  # only the first row reaches the side parameter
        return [rec("a", [31, 2], [3, 4, 5], 1.0), rec("b", [1, 2], [3, 4, 6], 0.0), rec("c", [1, 2], [7, 8, 9, 10], 0.5),
                rec("d", [1, 2], [7], 0.0)], ["g", "g", "h", "h"]
    raise ValueError(kind)


def main() -> None:
    scenario, out = sys.argv[1], Path(sys.argv[2])
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("M12_PORT", "29581"))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    res: dict = {}
    for name, kind, cfg_kw, model_fn in (
        ("empty_rank", "one_item", {"loss_agg": "token_mean_global"}, lambda: TinyPolicy(32, 8, seed=3)),
        ("side_param", "side", {}, SideHeadPolicy),
        ("seq_mean", "side", {"loss_agg": "seq_mean_token_mean"}, SideHeadPolicy),
        ("prompt_mean_ppo", "side", {"loss_agg": "prompt_mean", "correction": None}, SideHeadPolicy),
    ):
        if scenario == "subgroup":
            break
        recs, groups = batch(kind)
        if name == "empty_rank":  # one scored row only: the other record is masked
            recs = [recs[0], recs[1].model_copy(update={"reward": None, "mask_sample": True})]
        kw = dict(cfg_kw)
        if kw.get("correction", 1) is None:
            from crucible.train.mismatch import Correction

            kw["correction"] = Correction("ppo", old_logprob_source="rollout")
        torch.manual_seed(0)
        ref_model, model = model_fn(), model_fn()
        model.load_state_dict(ref_model.state_dict())
        cfg = TrainerConfig(lr=0.05, token_budget=4, dp=world, **kw)
        tr = DistributedTorchTrainer(model, cfg, broadcast_batch=True)
        m = tr.step(recs if rank == 0 else [], groups if rank == 0 else [])
        if rank == 0:
            ref = TorchTrainer(ref_model, TrainerConfig(lr=0.05, token_budget=10_000, dp=1, **kw))
            ref.step(recs, groups)
            diff = max(float((a - b).abs().max()) for a, b in zip(model.state_dict().values(), ref_model.state_dict().values(), strict=True))
            res[name] = {"max_abs_diff": diff, "loss": m.get("train/loss")}
    if scenario == "subgroup":  # a trainer group [1, 2] whose source is group rank 0 = global rank 1 (H4 V12)
        g = dist.new_group([1, 2])
        if rank in (1, 2):
            recs, groups = batch("side")
            model = SideHeadPolicy()
            tr = DistributedTorchTrainer(model, TrainerConfig(lr=0.05, dp=2), group=g, broadcast_batch=True, src=0)
            m = tr.step(recs if rank == 1 else [], groups if rank == 1 else [])
            res["subgroup"] = {"loss": m.get("train/loss"), "w0": float(next(model.parameters()).flatten()[0])}
        dist.barrier()
    (out / f"r{rank}.json").write_text(json.dumps(res))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
