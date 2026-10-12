"""FSDP2 sharded data parallelism (``train/fsdp.py``, M12-infra C) over real processes on a gloo group
(``_fsdp_worker.py``): one sharded update equals the single-process full-batch update, in both resharding modes, under
HSDP, with padded micro batches, and on a real HF decoder with its blocks as units."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

WORKER = Path(__file__).with_name("_fsdp_worker.py")


def run(scenario: str, tmp_path: Path, world: int) -> list[dict]:
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")
    env = {**os.environ, "WORLD_SIZE": str(world), "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")}
    procs = [subprocess.Popen([sys.executable, str(WORKER), scenario, str(tmp_path)], env={**env, "RANK": str(r)}, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for r in range(world)]
    errs = []
    for p in procs:
        try:
            _, err = p.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            p.kill()
            _, err = p.communicate()
            err = "TIMEOUT\n" + err
        if p.returncode:
            errs.append(err[-3000:])
    assert not errs, errs
    return [json.loads((tmp_path / f"r{r}.json").read_text()) for r in range(world)]


def test_fsdp_step_and_layer_modes_equal_the_full_batch_update(tmp_path):
    """Two updates (AdamW moments carried in shards), dis and ppo (π_θold recomputed per rank), batch sent by rank 0.
    The gathered state equals ``TorchTrainer`` on the whole batch; the pre-clip gradient norm is the global one; a rank
    holds half the parameters and half the moments at rest; the skewed batch makes rank 0 pad with probe micro batches."""
    rs = run("fsdp", tmp_path, world=2)
    for key in ("step/dis", "step/ppo", "layer/dis", "layer/ppo", "layer/ppo/skew"):
        a = rs[0][key]
        assert a["max_abs_diff"] < 5e-6, (key, a)
        assert abs(a["grad_norm"] - a["ref_grad_norm"]) < 1e-5 and a["grad_norm"] > 0, (key, a)
        assert a["params_frac"] == pytest.approx(0.5, abs=0.02) and a["moments_frac"] == pytest.approx(0.5, abs=0.02)
        assert rs[0][key]["loss"] == pytest.approx(rs[1][key]["loss"])  # every rank reports the global loss
    assert rs[0]["step/dis"]["all_gathers"] == 2 and rs[0]["step/dis"]["probe_micro_batches"] == 2  # one gather + one reduce-scatter per update
    assert rs[0]["layer/dis"]["probe_micro_batches"] == 0  # balanced: no padding
    assert rs[0]["layer/ppo/skew"]["probe_micro_batches"] + rs[1]["layer/ppo/skew"]["probe_micro_batches"] > 0  # skewed: lockstep padding ran
    assert rs[0]["refused_unbatched_layer"]
    assert rs[0]["rnn_flatten_disabled"]  # cuDNN would re-point GRU weights away from FSDP's gathered storage (seen on NCCL only)


def test_hsdp_two_replicas_of_two_shards(tmp_path):
    """A 2×2 mesh (replicate, shard): shards within a replica, gradients summed across replicas; same update."""
    rs = run("hsdp", tmp_path, world=4)
    for key in ("step/dis", "step/ppo"):
        assert rs[0][key]["max_abs_diff"] < 5e-6, rs[0][key]
        assert all(r[key]["params_frac"] == pytest.approx(0.5, abs=0.02) for r in rs)  # sharded over 2, not 4


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers not installed")
def test_hf_decoder_layer_mode(tmp_path):
    """A random-init Llama (3 blocks) through ``HFPolicy``: blocks as FSDP units, per-block gather in the forward, HF names
    in the published state, equal to the single-process update."""
    rs = run("hf_layer", tmp_path, world=2)
    assert rs[0]["units"] == 3 and rs[0]["max_abs_diff"] < 5e-6, rs[0]
    assert rs[0]["names"][0] == "lm_head.weight"


def test_single_controller_runtime_over_hsdp_ranks_with_sharded_resume(tmp_path):
    """Rank 0 runs the real Runtime (disaggregated_stream, 16 steps) through ``SPMDDriver``; ranks 1–3 ``follow``. Every
    command reaches every rank; the policy learns; staleness holds; the run checkpoint writes DCP shards; a fresh
    runtime resumes at step 8 with exactly the weights published at version 8 and continues 9…12."""
    rs = run("spmd_run", tmp_path, world=4)
    d = rs[0]
    assert d["steps"] == 16 and d["publishes"] == 16 and d["last"] > d["first"] and d["max_staleness"] <= 2
    assert d["checkpoints"] == ["step-000008", "step-000016"] and d["calls"]["save_sharded"] == 2
    assert d["resumed_at"] == 8 and d["resumed_version"] == 8 and d["resumed_updates"] == 8 and d["restore_diff"] == 0.0
    assert d["steps2"] == 4 and d["first_step2"] == 9
    for r in rs[1:]:
        assert r["first"]["step"] == 16 and r["first"]["state_dict"] == d["calls"]["state_dict"] and r["first"]["save_sharded"] == 2
        assert r["second"]["load_sharded"] == 1 and r["second"]["step"] == 4 and r["second"]["stop"] == 1
