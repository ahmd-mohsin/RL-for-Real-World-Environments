"""Distributed weight sync (``train/weightsync.py``): the bucket plan and size rule in-process, and every protocol
property over real processes on a gloo group (``_weightsync_worker.py``)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from crucible.train.weightsync import ALIGN, BucketPlan, choose_bucket_bytes, plan_buckets

WORKER = Path(__file__).with_name("_weightsync_worker.py")


def run(scenario: str, tmp_path: Path, world: int = 3) -> list[dict]:
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")
    env = {**os.environ, "WORLD_SIZE": str(world), "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")}
    procs = [subprocess.Popen([sys.executable, str(WORKER), scenario, str(tmp_path)], env={**env, "RANK": str(r)}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for r in range(world)]
    errs = []
    for p in procs:
        try:
            _, err = p.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
            _, err = p.communicate()
            err = "TIMEOUT\n" + err
        if p.returncode:
            errs.append(err[-3000:])
    assert not errs, errs
    return [json.loads((tmp_path / f"r{r}.json").read_text()) for r in range(world)]


# ---- in-process: plan and bucket size -----------------------------------------------------------------------------
def test_plan_is_aligned_single_owner_deterministic_and_portable():
    shapes = {"b": ("float32", (10, 10)), "a": ("bfloat16", (300,)), "c": ("int64", (3,)), "d": ("float32", (50,))}
    owners = {"a": 1, "b": 0, "c": 1, "d": 0}
    p = plan_buckets(shapes, 1024, owners)
    assert all(m.offset % ALIGN == 0 for m in p.metas)
    assert all(len({m.owner for m in b}) == 1 for b in p.buckets())  # the owner is the broadcast source
    assert [m.name for m in p.metas] == ["b", "d", "a", "c"]  # sorted by (owner, name)
    assert all(b[-1].offset + b[-1].nbytes <= 1024 for b in p.buckets())
    assert [(m.name, m.bucket, m.offset) for m in p.metas] == [("b", 0, 0), ("d", 0, 512), ("a", 1, 0), ("c", 1, 768)]  # 400 B → next at 512
    assert plan_buckets(dict(reversed(list(shapes.items()))), 1024, owners).to_json() == p.to_json()
    assert BucketPlan.from_json(p.to_json()).crc() == p.crc()
    with pytest.raises(ValueError, match="exceeds the bucket"):
        plan_buckets({"big": ("float32", (1000,))}, 512)


def test_bucket_size_rule():
    G = 1 << 30
    assert choose_bucket_bytes(30 * G, 1 * G, max_bucket=8 * G) == 8 * G  # capped by max_bucket
    third = int(9 * G * 0.9) // 3
    assert choose_bucket_bytes(9 * G, 1 * G, max_bucket=8 * G) == (third + ALIGN - 1) // ALIGN * ALIGN  # a third: room for a staging buffer
    b = choose_bucket_bytes(9 * G, 3 * G, max_bucket=8 * G)  # the largest tensor does not fit a third: half
    assert 3 * G <= b <= int(9 * G * 0.9) // 2 + ALIGN
    with pytest.raises(ValueError, match="not enough free memory"):
        choose_bucket_bytes(4 * G, 3 * G)


# ---- multi-process over gloo --------------------------------------------------------------------------------------
def test_broadcast_two_versions_bitwise_through_many_buckets(tmp_path):
    rs = run("broadcast", tmp_path)
    assert rs[0]["buckets"] >= 3 and rs[0]["aligned"]  # bucket 2 reuses half 0: the double buffer is exercised
    for r in rs[1:]:
        assert r["v1"] == 1 and r["v2"] == 2  # the version travels in the header
        assert r["d1"] == rs[0]["d1"] and r["d2"] == rs[0]["d2"]  # bitwise, every dtype (fp32, bf16, fp16, int64, bool, 0-d)
    assert rs[0]["d1"] != rs[0]["d2"]


def test_pipeline_with_a_slow_reload_never_corrupts(tmp_path):
    rs = run("slow_reload", tmp_path)
    assert rs[1]["d"] == rs[0]["d"] == rs[2]["d"]


def test_sharded_owners_colocated_receivers(tmp_path):
    rs = run("sharded", tmp_path)
    assert rs[0]["owners"] == [0, 1] and rs[0]["single_owner_buckets"]
    assert rs[0]["d"] == rs[1]["d"] == rs[2]["d"] == rs[0]["full"]  # every rank ends with the full state


def test_a_failed_reload_aborts_every_rank_and_nobody_commits(tmp_path):
    rs = run("fail", tmp_path)
    for r in rs:
        assert r["raised"].startswith("rank 2 failed to reload bucket 2 of version 2")  # every rank names the same culprit
    assert "injected reload failure" in rs[2]["raised"]  # the failing rank adds its own cause
    for r in rs[1:]:
        assert r["version_after"] == 1 and r["unchanged"]  # the old weights and version keep serving


def test_subgroup_update_and_elastic_plan(tmp_path):
    rs = run("subgroup", tmp_path)
    assert all(r["same_crc"] for r in rs)
    assert rs[1]["got"] is None and rs[2]["got"] == 1  # rank 1 not in the subgroup: returned at once
    assert rs[2]["d"] == rs[0]["full"]


def test_distributed_publisher_and_engine_receiver(tmp_path):
    rs = run("publisher", tmp_path, world=2)
    assert rs[0]["versions"] == [1, 2] == rs[1]["versions"] and rs[1]["engine_version"] == 2
    assert rs[1]["d"] == rs[0]["d"]  # the engine serves exactly the trainer's weights at version 2


@pytest.mark.parametrize("scenario", ["dp_exact", "dp_broadcast"])
def test_data_parallel_trainer_equals_the_full_batch_step(tmp_path, scenario):
    """DistributedTorchTrainer (DP=2, token-budget micro batches, π_θold per rank for ppo): after one step both replicas
    are identical and equal the single-process full-batch update; ``dp_broadcast``: rank 1 receives the batch."""
    rs = run(scenario, tmp_path, world=2)
    for corr in ("dis", "ppo"):
        a, b = rs[0][corr], rs[1][corr]
        assert a["d"] == b["d"] and a["changed"]  # replicas bitwise identical after the step, and the step did something
        assert a["max_abs_diff_vs_full_batch"] < 1e-6, a
        assert a["micro_batches"] > 2 and a["imbalance"] < 1.2  # placement over forward tokens, balanced
    assert rs[0]["accepts_seq_mean"]  # DP takes every loss_agg (exact per-rollout weights; test_m12_audit_math)
