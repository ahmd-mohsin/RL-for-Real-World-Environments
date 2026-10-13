"""M12 audit regressions for the multi-GPU / scale-out path (``docs/audit/2026-10-12``: H4 V2-V10, V13-V15, V17, V18;
G5 D2, D3, D5, D6, D8, D10, P2-a; H2 V12, V20). The distributed cases run real processes on a gloo group
(``_m12_scale_worker.py``), with a short process-group timeout so a hang fails fast instead of stalling the suite."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).parent
WORKER = HERE / "_m12_scale_worker.py"
IFNAME = "lo0" if sys.platform == "darwin" else "lo"


def run(scenario: str, tmp_path: Path, world: int, args: dict | None = None, *, wall_s: float = 50.0) -> tuple[list[dict | None], list]:
    """Start ``world`` ranks; returns (per-rank results, per-rank exit codes or 'TIMEOUT')."""
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")
    a = {"pg_timeout": 20, **(args or {})}
    env = {**os.environ, "WORLD_SIZE": str(world), "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", IFNAME), "OMP_NUM_THREADS": "1"}
    procs = [subprocess.Popen([sys.executable, str(WORKER), scenario, str(tmp_path), json.dumps(a)], env={**env, "RANK": str(r)},
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for r in range(world)]
    deadline, codes = time.monotonic() + wall_s, []
    for p in procs:
        try:
            p.communicate(timeout=max(1.0, deadline - time.monotonic()))
            codes.append(p.returncode)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            codes.append("TIMEOUT")
    out = []
    for r in range(world):
        f = tmp_path / f"r{r}.json"
        out.append(json.loads(f.read_text()) if f.exists() else None)
    return out, codes


def ok(rs: list[dict | None], codes: list) -> list[dict]:
    assert codes == [0] * len(codes), (codes, rs)
    assert all(r is not None and "error" not in r for r in rs), rs
    return rs  # type: ignore[return-value]


# ---- V2: a parameter one rank never reaches ---------------------------------------------------------------------------
@pytest.mark.parametrize("expert_unit", [True, False])
def test_V2_fsdp_step_unreached_parameter_equals_single_process(tmp_path, expert_unit):
    """An expert only rank 0's item routes through (and the probe misses): step mode zero-fills the unsharded gradient
    on rank 1 before the reduce-scatter, so the update equals ``TorchTrainer`` (old: gloo abort / hang)."""
    rs = ok(*run("unused", tmp_path, 2, {"mode": "step", "expert_unit": expert_unit}))
    assert rs[0]["max_abs_diff"] < 1e-5 and rs[0]["expert_moved"] > 1e-3, rs[0]
    assert rs[1]["zero_filled"] > 0


def test_V2_V16_fsdp_step_parameter_no_rank_reached_keeps_no_gradient(tmp_path):
    """No item routes; the probe's zero-weighted backward reaches the expert: it must stay without a gradient, as in
    ``TorchTrainer`` (old: a zero gradient, so AdamW decayed it: 2.5e-5 per step)."""
    rs = ok(*run("unused", tmp_path, 2, {"mode": "step", "expert_unit": True, "route": False, "probe_no_expert": False}))
    assert rs[0]["max_abs_diff"] < 1e-6 and rs[0]["expert_moved"] == 0.0, rs[0]


def test_V2_fsdp_layer_data_dependent_unit_raises_on_every_rank(tmp_path):
    """Layer mode with the expert as its own unit: rank 1 never runs it, so the all-gathers cannot pair up. Every rank
    raises the same clear error instead of hanging (old: hang until the process-group timeout / abort)."""
    rs, codes = run("unused", tmp_path, 2, {"mode": "layer", "expert_unit": True})
    assert codes == [0, 0], (codes, rs)
    assert all(r and "ran different units" in r.get("error", "") for r in rs), rs


def test_V2_fsdp_layer_unreached_parameter_inside_a_running_unit_is_correct(tmp_path):
    """Layer mode, the expert inside the root unit (which always runs): the unreached parameter reduce-scatters a zero on
    rank 1 and the update equals the single-process one."""
    rs = ok(*run("unused", tmp_path, 2, {"mode": "layer", "expert_unit": False}))
    assert rs[0]["max_abs_diff"] < 1e-5 and rs[0]["expert_moved"] > 1e-3, rs[0]


# ---- V4 / D2: no full fp32 model per GPU ---------------------------------------------------------------------------------
def test_V4_meta_device_model_loads_shards_from_rank0(tmp_path):
    """Every rank builds the model on the meta device; rank 0 alone holds the full state; each rank keeps half the
    parameters; the loaded state and the first update equal the single-process ones."""
    rs = ok(*run("meta_load", tmp_path, 2))
    assert all(r["built_on_meta"] and r["local_frac"] == pytest.approx(0.5, abs=0.02) for r in rs)
    assert rs[0]["load_diff"] == 0.0 and rs[0]["max_abs_diff"] < 5e-6, rs[0]
    assert rs[0]["init_broadcasts"] == rs[0]["n_params"]  # one per tensor while loading, no second replica broadcast


@pytest.mark.parametrize("tied", [False, True])
def test_V4_meta_device_hf_decoder_keeps_buffers_and_tied_head(tmp_path, tied):
    pytest.importorskip("transformers")
    rs = ok(*run("meta_hf", tmp_path, 2, {"tied": tied}))
    assert rs[0]["keys_equal"] and rs[0]["max_abs_diff"] < 5e-6, rs[0]


def test_V4_planner_reports_the_load_and_refuses_what_the_host_cannot_hold():
    from crucible.train.placement import ClusterSpec, ModelSpec, PlacementError, Workload, plan

    m30 = ModelSpec(30.5, 48, 4, 128, active_b=3.3)
    c = ClusterSpec.preset("repacss", 2)
    p = plan(c, m30, Workload(256, 1024, 2048))
    assert p.init == "meta_rank0" and p.load_peak["gpu"] < 0.92 * c.gpu_mem_gb and p.load_peak["host"] == pytest.approx(122.0)
    with pytest.raises(PlacementError):  # the source node cannot hold the fp32 copy
        plan(ClusterSpec.preset("repacss", 2, host_mem_gb=64.0), m30, Workload(256, 1024, 2048))


# ---- V7 / D3: offload between steps -------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["step", "layer"])
def test_V7_fsdp_offload_roundtrip_keeps_the_update_exact(tmp_path, mode):
    rs = ok(*run("offload", tmp_path, 2, {"mode": mode}))
    assert all(r["offloaded_device"] == ["cpu"] and r["grads_dropped"] and r["roundtrip_exact"] and r["same_param_objects"] for r in rs)
    assert rs[0]["max_abs_diff"] < 5e-6, rs[0]


def test_V7_colocated_plans_offload_the_trainer_and_fit_the_host():
    from crucible.train.placement import ClusterSpec, ModelSpec, Workload, candidates, plan

    m8 = ModelSpec(8.2, 36, 8, 128)
    w = Workload(256, 1024, 8192, turns=8, env_s_per_turn=5)
    p = plan(ClusterSpec(1), m8, w, "colocated_partial")
    assert p.runtime_overrides()["train_state_on_rollout"] == "cpu"
    assert not candidates(ClusterSpec(1, host_mem_gb=100.0), m8, w, "colocated_partial")  # 8 GPUs of offloaded shards + sleeping weights


# ---- V14: the planner's trainer kwargs build the trainer ---------------------------------------------------------------
def test_V14_plan_kwargs_build_the_hsdp_trainer(tmp_path):
    rs = ok(*run("plan_kwargs", tmp_path, 4))
    assert rs[0]["mesh"] == [2, 2] and rs[0]["kwargs"] == {"mesh_shape": [2, 2], "reshard": "layer"} and rs[0]["mesh_ndim"] == 2
    assert all(r["params_frac"] == pytest.approx(0.5, abs=0.02) for r in rs) and rs[0]["max_abs_diff"] < 5e-6


# ---- G5 P2-a / H4 V17: FSDP2 mixed precision --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["layer", "step"])
def test_P2a_mixed_precision_bf16_blocks_fp32_head_and_master(tmp_path, mode):
    pytest.importorskip("transformers")
    rs = ok(*run("mixed_precision", tmp_path, 2, {"mode": mode}))
    for r in rs:
        assert r["dtypes"] == {"block": "torch.bfloat16", "head": "torch.float32"} and r["master_dtype"] == ["torch.float32"], r
    assert rs[0]["max_abs_diff"] < 3e-3, rs[0]  # lr 1e-3: bf16 forwards move a few AdamW signs, nothing more


# ---- V3: resume through SPMDDriver -------------------------------------------------------------------------------------
def test_V3_spmd_resume_loads_every_replica_and_counts_on_the_trainer(tmp_path):
    rs = ok(*run("resume_ddp", tmp_path, 2, wall_s=55))
    d = rs[0]
    assert d["spread_after_run1"] == 0.0 and d["spread_after_resume_run"] == 0.0, d  # old: 0.074 after two resumed steps
    assert d["updates_after_resume"] == [4, 4] and d["driver_updates_attr"] == d["trainer_updates_attr"] == 6, d
    assert rs[1]["second"]["load_state"] == 1 and rs[1]["follower_updates"] == 6


# ---- V5 / V6: SPMD start-up and liveness --------------------------------------------------------------------------------
def test_V5_followers_outlive_a_driver_slower_than_the_pg_timeout(tmp_path):
    """Followers wait 15 s for the driver with an 8 s process-group timeout (old: they died at the timeout)."""
    rs = ok(*run("late_driver", tmp_path, 2, {"pg_timeout": 8, "delay": 15}, wall_s=55))
    assert rs[0]["driver"] == "ok" and rs[1]["follower"]["step"] == 1 and rs[1]["follower"]["stop"] == 1


def test_V6_followers_stop_when_the_driver_dies(tmp_path):
    rs, codes = run("driver_dies", tmp_path, 2, {"period": 0.2, "stale": 2.0, "pg_timeout": 60}, wall_s=30)
    assert codes[1] == 0 and "LostPeer" in rs[1]["follower"] and rs[1]["wait_s"] < 10, (codes, rs)  # old: blocked for the PG timeout


def test_V6_driver_exits_when_a_follower_dies_mid_step(tmp_path):
    t0 = time.monotonic()
    rs, codes = run("follower_dies", tmp_path, 2, {"period": 0.2, "stale": 2.0, "pg_timeout": 60}, wall_s=40)
    assert codes[0] == 75 and time.monotonic() - t0 < 30, (codes, rs)  # old: blocked in the step for the PG timeout


def test_V6_local_launcher_stops_the_job_when_one_process_fails():
    from crucible.train.cluster import LocalLauncher, ProcSpec

    specs = [ProcSpec("trainer", 0, [g], rank=g, env={"RANK": str(g)}) for g in range(3)]
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="rank 0 failed"):
        LocalLauncher(extra_env={"PYTHONPATH": os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")])}, timeout_s=50,
                      grace_s=1.0).run(specs, "_m12_scale_worker:fail_or_sleep")
    assert time.monotonic() - t0 < 25  # old: waited for the sleepers (60 s)


# ---- V13: offload path ------------------------------------------------------------------------------------------------
def test_V13_offload_path_reaches_every_rank(tmp_path):
    rs = ok(*run("offload_path", tmp_path, 2))
    assert all(r["calls"] == [["offload", "nvme", "/shared/offload"], ["onload"]] for r in rs), rs


# ---- V8 / V9 / V10 / V15: launch ------------------------------------------------------------------------------------------
def _repacss_plans():
    from crucible.train.placement import ClusterSpec, ModelSpec, Workload, plan

    c = ClusterSpec.preset("repacss", 2)
    return [plan(c, ModelSpec(8.2, 36, 8, 128), Workload(256, 1024, 4096), t) for t in ("disaggregated_stream", "colocated_partial")]


def test_V8_local_rank_indexes_the_visible_devices():
    from crucible.train.cluster import process_specs

    for p in _repacss_plans():
        for s in process_specs(p):
            if s.rank is not None:
                assert int(s.env["LOCAL_RANK"]) < len(s.env["CUDA_VISIBLE_DEVICES"].split(",")), s.env  # old: LOCAL_RANK = physical GPU
                assert s.env["CRUCIBLE_GPUS"] == s.env["CUDA_VISIBLE_DEVICES"] and s.env["NODE_RANK"] == str(s.node)


def test_V8_init_rank_sets_the_device_first_and_an_explicit_timeout(monkeypatch):
    import torch.distributed as dist

    from crucible.train import cluster

    seen: dict = {}
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", lambda backend, **kw: seen.update(backend=backend, **kw))
    monkeypatch.setattr(dist, "get_rank", lambda *a: 3)
    monkeypatch.setattr(dist, "get_world_size", lambda *a: 8)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("CRUCIBLE_PG_TIMEOUT_S", "900")
    monkeypatch.delenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", raising=False)
    rank, world, dev = cluster.init_rank("gloo")
    assert (rank, world, dev.type) == (3, 8, "cpu") and seen["timeout"].total_seconds() == 900 and seen["init_method"] == "env://"
    assert os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"


def test_V9_slurm_environment_maps_to_torchrun_variables():
    from crucible.train.cluster import expand_nodelist, node_identity, slurm_env

    assert expand_nodelist("rpc-91-[7-8]") == ["rpc-91-7", "rpc-91-8"]
    assert expand_nodelist("gpu[001-002,010],login1") == ["gpu001", "gpu002", "gpu010", "login1"]
    base = {"SLURM_JOB_NODELIST": "rpc-91-[7-8]", "SLURM_NTASKS": "8", "SLURM_JOB_NUM_NODES": "2"}
    envs = [slurm_env({**base, "SLURM_PROCID": str(r), "SLURM_LOCALID": str(r % 4), "SLURM_NODEID": str(r // 4)}) for r in range(8)]
    assert [e["RANK"] for e in envs] == [str(r) for r in range(8)] and {e["MASTER_ADDR"] for e in envs} == {"rpc-91-7"}
    assert [e["LOCAL_RANK"] for e in envs] == ["0", "1", "2", "3"] * 2 and [e["NODE_RANK"] for e in envs] == ["0"] * 4 + ["1"] * 4
    assert {e["WORLD_SIZE"] for e in envs} == {"8"} and {e["LOCAL_WORLD_SIZE"] for e in envs} == {"4"}
    assert node_identity({**base, "SLURM_NODEID": "1"}) == (1, "rpc-91-7")


def test_V9_local_launcher_runs_one_node_and_refuses_all_nodes_on_one_host():
    from crucible.train.cluster import LocalLauncher, process_specs, run_node

    p = _repacss_plans()[0]
    specs = process_specs(p, master_addr="rpc-91-7")
    assert {s.node for s in specs} == {0, 1}
    with pytest.raises(ValueError, match="span nodes"):
        LocalLauncher().run(specs, "crucible.train.cluster:smoke")
    assert {s.node for s in LocalLauncher(node=1).select(specs)} == {1}
    with pytest.raises(ValueError, match="MASTER_ADDR"):  # a multi-node run never rendezvous on loopback by default
        run_node(p, "crucible.train.cluster:smoke", environ={})


def test_V10_multi_gpu_engines_get_all_their_gpus():
    from crucible.train.cluster import process_specs
    from crucible.train.placement import ClusterSpec, ModelSpec, Workload, plan

    p = plan(ClusterSpec(1), ModelSpec(25, 48, 8, 128), Workload(64, 512, 512), "colocated_partial")  # serving needs TP2
    assert p.engine_gpus == 2
    specs = process_specs(p)
    hosts = [s for s in specs if s.env.get("ENGINE_HOST") == "1"]
    assert len(hosts) == p.engines and all(len(s.env["CUDA_VISIBLE_DEVICES"].split(",")) == 2 == int(s.env["ENGINE_GPUS"]) for s in hosts)
    assert all(s.env["ENGINE_HOST"] == "0" for s in specs if s.rank is not None)


def test_V15_env_workers_are_one_nodes_slots():
    from crucible.train.placement import ClusterSpec, ModelSpec, Workload, plan

    p = plan(ClusterSpec(2), ModelSpec(7, 32, 8, 128), Workload(2048, 1024, 4096, env_slots_per_seq=4.0), "disaggregated_stream")
    assert p.env_workers <= ClusterSpec(2).cpu_slots_per_node


# ---- D10: node-local shard groups, communication term, no elastic for vLLM -------------------------------------------------
def test_D10_shard_groups_stay_inside_a_node_and_cost_their_traffic():
    from crucible.train.placement import ClusterSpec, ModelSpec, Workload, plan, shard_local

    c = ClusterSpec.preset("repacss", 2)
    for m in (ModelSpec(4.0, 36, 8, 128), ModelSpec(8.2, 36, 8, 128), ModelSpec(30.5, 48, 4, 128, active_b=3.3)):
        p = plan(c, m, Workload(256, 1024, 4096))
        world = p.mesh[0] * p.mesh[1]
        assert shard_local(c, world, p.mesh[1]), (m, p.explain())  # old: (1, 7) across both nodes
        assert p.times["train_comm"] > 0 or p.mesh[1] == 1
    assert not shard_local(c, 7, 7) and shard_local(c, 8, 4) and shard_local(c, 8, 8)


def test_D10_vllm_engines_never_get_an_elastic_plan():
    from crucible.train.placement import ClusterSpec, ModelSpec, PlacementError, Workload, plan

    c, m, w = ClusterSpec.preset("repacss", 2), ModelSpec(8.2, 36, 8, 128), Workload(512, 1024, 4096)
    assert plan(c, m, w).topology != "elastic" and "elastic" not in plan(c, m, w).alternatives
    with pytest.raises(PlacementError, match="hold"):
        plan(c, m, w, "elastic")
    assert plan(c, m, w, "elastic", engine="torch").topology == "elastic"


# ---- D8: checkpoint publisher garbage collection --------------------------------------------------------------------------
class _Trainer:
    def __init__(self):
        self.updates = 0

    def state_dict(self):
        return {"w": torch.full((4,), float(self.updates))}


class _Engines:
    def __init__(self, fail_at: set[int] | None = None):
        self.fail_at, self.loaded = fail_at or set(), []

    async def load_weights(self, path, version):
        await asyncio.sleep(0)
        if version in self.fail_at:
            raise RuntimeError(f"engine failed to load v{version}")
        assert Path(path, "model.safetensors").exists()
        self.loaded.append(version)


async def test_D8_checkpoint_publisher_keeps_the_last_versions(tmp_path):
    from crucible.train.weights import CheckpointPublisher

    tr = _Trainer()
    pub = CheckpointPublisher(tr, _Engines(), tmp_path)  # default keep=2
    for _ in range(5):
        tr.updates += 1
        await pub.publish()
    assert sorted(d.name for d in tmp_path.iterdir()) == ["v4", "v5"] and pub.deleted == ["v1", "v2", "v3"]
    assert pub.published_updates == 5


async def test_D8_checkpoint_publisher_never_deletes_a_version_an_engine_may_be_loading(tmp_path):
    from crucible.train.weights import CheckpointPublisher

    tr = _Trainer()
    pub = CheckpointPublisher(tr, _Engines(fail_at={2}), tmp_path, keep=1)
    await pub.publish()
    with pytest.raises(RuntimeError):
        await pub.publish()  # v2: an engine failed mid-load; v1 and v2 stay
    assert sorted(d.name for d in tmp_path.iterdir()) == ["v1", "v2"]
    await pub.publish()  # v3 confirmed: v1 goes, v2 (a load that never confirmed) stays
    assert sorted(d.name for d in tmp_path.iterdir()) == ["v2", "v3"]


# ---- H2 V12 / V20: state plane -------------------------------------------------------------------------------------------
def _runtime(topology: str, steps: int, ckdir=None, every: int = 0, delay: float = 0.0):
    from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
    from crucible.train.envs import CopyTask
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.trainer import TorchTrainer, TrainerConfig
    from crucible.train.weights import InprocPublisher

    model = TinyPolicy(32, 24, seed=3)
    eng = EnginePool([TorchEngine(model, capacity=64, decode_delay_s=delay)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2))
    return Runtime(RuntimeConfig(topology=topology, steps=steps, max_turn_tokens=3, warmup_batches=0, checkpoint_every=every,
                                 checkpoint_dir=str(ckdir) if ckdir else None),
                   SchedulerConfig(n=2, B=2, max_staleness=2), eng, tr, InprocPublisher(tr, eng),
                   lambda: CopyTask(turns=2, symbols=8, latency_s=delay, long_tail=0.3 if delay else 0.0), {"copy": [f"p{i}" for i in range(400)]})


async def test_V12_checkpoint_refuses_weights_newer_than_the_published_version():
    from crucible.train import state

    rt = _runtime("colocated_sync", 2)
    await rt.run()
    state.capture(rt, 2)  # right after a publish: fine
    rt.trainer.updates += 1  # a step that trained but did not publish
    with pytest.raises(ValueError, match="published"):
        state.capture(rt, 3)


async def test_V20_game_seats_travel_with_the_checkpoint(tmp_path):
    from crucible.train import state

    a = _runtime("colocated_partial", 3, tmp_path / "a", every=3, delay=0.003)
    await a.run()
    mid = _runtime("colocated_partial", 6, delay=0.003)
    state.restore(mid, state.load(state.latest(tmp_path / "a")))  # the run at its step-3 barrier
    from crucible.core.schema.rollout import RolloutRecord

    rid = next(iter(mid.pool.tags))
    rec = RolloutRecord(rollout_id=rid, sample="s", task="t", harness="h", policy="p", reward=1.0, turns=[])
    mid.pool.tags[rid].status = "finished"  # a finished two-seat game rollout, not yet trained
    mid.pool.values[rid] = rec
    mid._seats[rid] = [rec, rec.model_copy(update={"reward": 0.0})]
    path = state.save(state.capture(mid, 3), tmp_path / "b")
    b = _runtime("colocated_partial", 6, delay=0.003)
    state.restore(b, state.load(path))
    assert list(b._seats) == [rid] and len(b._seats[rid]) == 2  # old: dropped, so only seat 0 trained after resume


# ---- V18: the REPACSS recipe ---------------------------------------------------------------------------------------------
def test_V18_repacss_sbatch_recipe_is_complete():
    root = HERE.parents[2]
    sb = root / "scripts" / "repacss_train.sbatch"
    text = sb.read_text()
    assert subprocess.run(["bash", "-n", str(sb)], capture_output=True, check=False).returncode == 0
    for need in ("#SBATCH --nodes=2", "#SBATCH --ntasks-per-node=", "nvidia_h100_nvl:4", "--kill-on-bad-exit=1", "scontrol show hostnames",
                 "NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-", "TORCH_NCCL_ASYNC_ERROR_HANDLING", "CRUCIBLE_PG_TIMEOUT_S", "CRUCIBLE_SPMD_STALE_S",
                 "CKPT_DIR", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6"):
        assert need in text, need
    from crucible.train.placement import ClusterSpec

    c = ClusterSpec.preset("repacss", 2)
    assert (c.gpus, c.gpus_per_node, c.gpu_mem_gb, c.nvlink_domain) == (8, 4, 94.0, 2)
