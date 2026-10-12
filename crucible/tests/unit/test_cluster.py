"""Placement planner (``train/placement.py``) and launchers (``train/cluster.py``): the cost model's choices, the layout
invariants on clusters from 1 to 128 nodes, and the same placement launched as local processes and as Ray actors."""

from __future__ import annotations

import importlib.util
import itertools
import os
import sys
from collections import Counter

import pytest

from crucible.train.cluster import LocalLauncher, process_specs, ray_bundles
from crucible.train.placement import TOPOLOGIES, ClusterSpec, ModelSpec, PlacementError, Workload, engine_shape, plan, rollout_s, train_mesh, train_s

M7, M70 = ModelSpec(7, 32, 8, 128), ModelSpec(70, 80, 8, 128)
MOE = ModelSpec(1000, 61, 8, 128, active_b=32)
CASES = [(ClusterSpec(1), M7, Workload(512, 1024, 4096)), (ClusterSpec(2), M7, Workload(512, 1024, 4096)),
         (ClusterSpec(16), M70, Workload(2048, 2048, 8192)), (ClusterSpec(128), MOE, Workload(8192, 4096, 16384, turns=8, env_s_per_turn=5)),
         (ClusterSpec(4), ModelSpec(8, 32, 8, 128), Workload(4096, 8192, 1024, decode_tok_s=80))]


def test_engine_and_trainer_shapes():
    c = ClusterSpec(16)
    assert engine_shape(c, M7, Workload(64))[0] == 1 and engine_shape(c, M70, Workload(64))[0] == 4  # 140 GB over 4×80 leaves 40% for KV
    assert engine_shape(ClusterSpec(128), MOE, Workload(64))[0] % 8 == 0  # 2 TB of weights: whole nodes per engine (EP / PP across nodes)
    with pytest.raises(PlacementError, match="does not fit"):
        engine_shape(ClusterSpec(1), MOE, Workload(64))
    assert train_mesh(c, M70, Workload(64), 128) == (4, 32)  # 1.12 TB of state: 32-way shards (≤ 56 GB each), 4 HSDP replicas
    assert train_mesh(c, M7, Workload(64), 8) == (4, 2)
    assert train_s(c, MOE, Workload(64), 64) < train_s(c, ModelSpec(1000, 61, 8, 128), Workload(64), 64) / 30  # MoE: active params pay


@pytest.mark.parametrize("c,m,w", CASES)
def test_layout_invariants(c, m, w):
    for topo in TOPOLOGIES:
        try:
            p = plan(c, m, w, topo)
        except PlacementError:
            continue
        roles = [r for row in p.layout for r in row]
        assert len(roles) == c.gpus and all(roles)  # every GPU has exactly one role
        specs = process_specs(p)
        owned = Counter((s.node, g) for s in specs for g in s.gpus)
        assert set(owned) == {(n, g) for n in range(c.nodes) for g in range(c.gpus_per_node)} and set(owned.values()) == {1}
        ranks = sorted(s.rank for s in specs if s.rank is not None)
        assert ranks == list(range(len(ranks))) and len(ranks) == p.mesh[0] * p.mesh[1]  # the trainer world is the FSDP/HSDP mesh
        eng: dict[int, list] = {}
        for s in specs:
            if s.engine is not None:
                eng.setdefault(s.engine, []).append(s)
        assert len(eng) == p.engines
        for parts in eng.values():
            gpus = sum(len(s.gpus) for s in parts)
            assert gpus == p.engine_gpus  # an engine owns exactly its GPUs
            if p.engine_gpus <= c.gpus_per_node:
                assert len({s.node for s in parts}) == 1  # a TP group never crosses a node
            else:
                assert all(len(s.gpus) == c.gpus_per_node for s in parts if s.role == "engine")  # a multi-node engine owns whole nodes


def test_disaggregated_split_is_optimal_and_topology_choice_reacts():
    c, w = ClusterSpec(4), Workload(2048, 2048, 4096)
    best = plan(c, M7, w, "disaggregated_stream")
    brute = min(max(rollout_s(c, M7, w, c.gpus - k * 8, tail=False), train_s(c, M7, w, k * 8) + best.times["publish"]) for k in range(1, 4))
    assert best.step_s == pytest.approx(brute)
    assert plan(c, M7, w).step_s <= min(plan(c, M7, w, t).step_s for t in TOPOLOGIES) + 1e-9
    assert plan(c, M7, w, "colocated_sync").step_s > plan(c, M7, w, "colocated_partial").step_s  # the tail costs synchronous rounds
    slow = plan(ClusterSpec(4, net_gbps=0.05), M7, w, "disaggregated_stream")
    assert slow.times["publish"] > 100 * best.times["publish"] and slow.step_s >= best.step_s  # weight broadcast over a thin network
    rt = best.runtime_overrides()
    assert rt["topology"] == "disaggregated_stream" and rt["env_workers"] > 0
    el = plan(ClusterSpec(2), M7, Workload(512, 1024, 4096), "elastic")
    assert el.runtime_overrides()["elastic_engines"] == el.elastic_gpus // el.engine_gpus > 0
    assert 0.0 <= el.times["elastic_roll_share"] <= 1.0


def test_plan_feeds_the_simulator():
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.sim import default_shapes, simulate

    w = Workload(64, 64, 32, decode_tok_s=50.0)
    p = plan(ClusterSpec(2, gpus_per_node=4, gpu_mem_gb=24), ModelSpec(0.5, 24, 2, 64), w, "disaggregated_stream")
    cfg = p.sim_config(w, steps=6)
    assert cfg.slots == p.engines * p.seqs_per_engine and cfg.topology == "disaggregated_stream"
    res = simulate(SchedulerConfig(n=4, B=4, max_staleness=2), cfg, {"a": [f"a{i}" for i in range(2000)]}, default_shapes(0))
    assert res.steps == 6 and res.max_staleness() <= 2
    el = plan(ClusterSpec(2), M7, Workload(512, 1024, 4096), "elastic")
    ec = el.sim_config(Workload(512, 1024, 4096))
    assert ec.elastic_train_speedup == (el.train_gpus + el.elastic_gpus) / el.train_gpus > 1
    from crucible.train.sim import SimConfig

    heavy = {"topology": "elastic", "slots": 64, "steps": 6, "train_s_base": 400.0, "elastic_frac": 0.25}  # train-bound: the elastic gain shows
    slow = simulate(SchedulerConfig(n=4, B=4, max_staleness=2), SimConfig(**heavy), {"a": [f"a{i}" for i in range(2000)]}, default_shapes(0))
    fast = simulate(SchedulerConfig(n=4, B=4, max_staleness=2), SimConfig(**heavy, elastic_train_speedup=2.0), {"a": [f"a{i}" for i in range(2000)]}, default_shapes(0))
    assert fast.time < 0.75 * slow.time


def _local(p):
    from crucible.train.cluster import free_port

    return process_specs(p, master_port=free_port())


def test_local_launcher_runs_a_two_node_placement():
    p = plan(ClusterSpec(2, gpus_per_node=2, gpu_mem_gb=24), ModelSpec(0.5, 24, 2, 64), Workload(64, 64, 32), "disaggregated_stream")
    specs = _local(p)
    out = LocalLauncher(extra_env={"CRUCIBLE_CPU_ONLY": "1", "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo"}).run(specs, "crucible.train.cluster:smoke")
    trainers = [o for o in out if o["rank"] is not None]
    W = len(trainers)
    assert W == p.train_gpus and all(o["world"] == W and o["rank_sum"] == W * (W - 1) / 2 for o in trainers)
    assert {o["node"] for o in out} == {0, 1} and sum(o["role"] == "engine" for o in out) == p.engines


@pytest.mark.skipif(importlib.util.find_spec("ray") is None, reason="ray not installed")
def test_ray_launcher_places_per_node_groups():
    import ray

    from crucible.train.cluster import RayLauncher

    p = plan(ClusterSpec(2, gpus_per_node=2, gpu_mem_gb=24), ModelSpec(0.5, 24, 2, 64), Workload(64, 64, 32), "colocated_partial")
    assert ray_bundles(p, gpu_resource="CPU") == [[{"CPU": 1.0}] * 2] * 2
    specs = _local(p)
    for s in specs:
        s.env.update(CRUCIBLE_CPU_ONLY="1", GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo")
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False, runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join(sys.path)}})
    try:
        out = RayLauncher(gpu_resource="CPU", timeout_s=120).run(p, specs, "crucible.train.cluster:smoke")
    finally:
        ray.shutdown()
    assert len(out) == 4 and all(o["role"] == "both" and o["world"] == 4 and o["rank_sum"] == 6 for o in out)
    assert sorted((o["node"], o["gpus"][0]) for o in out) == list(itertools.product((0, 1), (0, 1)))


def test_activation_memory_term_matches_the_live_oom():
    """Modal A10G (24 GB): Qwen3-1.7B under FSDP2 layer mode over 2 ranks ran out of memory at an 8192-token micro batch
    (17.5 GB allocated against 13.6 GB of state). With width and vocab known the planner counts activations and asks for 4."""
    from crucible.train.placement import activation_bytes

    m = ModelSpec(1.7, 28, 8, 128, hidden=2048, vocab=151936)
    c = ClusterSpec(1, gpus_per_node=4, gpu_mem_gb=23.7)
    assert 3.5e9 < activation_bytes(m, Workload(32, micro_tokens=8192)) < 5e9
    assert train_mesh(c, m, Workload(32, micro_tokens=8192), 4) == (1, 4)
    assert train_mesh(c, ModelSpec(1.7, 28, 8, 128), Workload(32), 4) == (2, 2)  # state only: the old, optimistic answer


def test_decode_curve_follows_the_engine_load():
    from crucible.train.placement import CALIBRATED

    cal = CALIBRATED["a10g/qwen3-1.7b"]
    w = Workload(512, 512, 1024, **cal["workload"])
    assert w.decode_at(1) == 110.8 and w.decode_at(128) == 53.2 and w.decode_at(1000) == 53.2
    assert 86.0 < w.decode_at(16) < 106.1  # interpolated between the measured points
    c = ClusterSpec(1, gpus_per_node=4, **cal["cluster"])
    m = ModelSpec(1.7, 28, 8, 128, hidden=2048, vocab=151936)
    light, heavy = Workload(8, 512, 1024, **cal["workload"]), Workload(2048, 512, 1024, **cal["workload"])
    one_light = rollout_s(c, m, light, 4, tail=False)
    one_heavy = rollout_s(c, m, heavy, 4, tail=False) * 8 / 2048  # per sequence-equivalent
    assert one_light < 1024 / 100  # a lightly loaded engine decodes near its batch-1 speed
    assert rollout_s(c, m, heavy, 4, tail=False) > 2048 * 1024 / (4 * 256 * 53.3) * 0.9 and one_heavy < one_light * 2


def test_memory_model_brackets_both_measured_points():
    """The fitted overhead puts the planner's per-rank estimate within 10% of both measured FSDP2 peaks."""
    from crucible.train.placement import activation_bytes

    for m, w, shard, measured in ((ModelSpec(1.72, 28, 8, 128, hidden=2048, vocab=151936), Workload(24, micro_tokens=4096), 3, 14.4e9),
                                  (ModelSpec(8.19, 36, 8, 128, hidden=4096, vocab=151936), Workload(48, micro_tokens=8192), 3, 65.0e9)):
        est = (16 * m.params_b * 1e9 / shard + activation_bytes(m, w)) * w.mem_overhead
        assert abs(est - measured) / measured < 0.10, (m.params_b, est / 1e9, measured / 1e9)
