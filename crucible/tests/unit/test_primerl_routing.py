"""MoE routing replay on the DeltaAI bridge (``train.bridges.primerl.routing``): routes recorded at sampling, stitched per
training sequence, sent all or none, switched on for the MoE student and off for the dense smoke (swarm-pipeline/14)."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from crucible.train.bridges.primerl import deploy, smoke, wire
from crucible.train.bridges.primerl.backend import PrimeRLBackend
from crucible.train.bridges.primerl.routing import RoutingStore, decode, stitch, topk_overlap
from crucible.train.bridges.tinker.trainer import TrainConfig

MODEL = "Qwen/Qwen3.6-35B-A3B"


def _field(rows, start):
    a = np.asarray(rows, dtype=np.uint8)
    return {"data": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape), "dtype": "uint8", "start": start}


def test_routes_stitch_first_forward_wins_and_report_coverage() -> None:
    store = RoutingStore()
    full = list(range(10))  # call 1 ends at 7 (prompt 0-3, completion 4-6); call 2 ends at 10
    r1 = decode(_field([[[1, 2]]] * 6, start=0))  # positions 0..5 (6, its last sampled token, is not forwarded by it)
    r2 = decode(_field([[[9, 9]], [[3, 4]], [[3, 4]], [[3, 4]]], start=5))  # 5..8: position 5 was call 1's, which wins
    store.put(full[:7], r1)
    store.put(full[:10], r2)
    arr, cov = stitch(full, [7, 10], store)
    assert arr.shape == (10, 1, 2) and cov == 1.0  # every position but the never-forwarded last one
    assert arr[5, 0].tolist() == [1, 2] and arr[6, 0].tolist() == [3, 4] and arr[9, 0].tolist() == [0, 0]
    gap = RoutingStore()
    gap.put(full[:10], r2)  # call 1's routes missing
    arr2, cov2 = stitch(full, [7, 10], gap)
    assert cov2 == pytest.approx(4 / 9) and arr2[0, 0].tolist() == [0, 0]
    assert stitch(full, [10], RoutingStore()) == (None, 0.0)
    a = np.array([[[1, 2, 3, 4]]])
    assert topk_overlap(a, a) == 1.0 and topk_overlap(a, np.array([[[4, 3, 9, 8]]])) == 0.5 and topk_overlap(a, a + 10) == 0.0
    with pytest.raises(ValueError):
        store.put([1], decode(_field([[[1, 2, 3]]], 0)))  # another (layers, top_k) than the run's


class GenHttp:
    def __init__(self):
        self.posts = []

    async def post(self, path, body, *, timeout_s):
        self.posts.append((path, body))
        if path == "/inference/v1/generate":
            n = len(body["token_ids"]) + 2
            return {"choices": [{"token_ids": [50, 51], "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]}, "finish_reason": "stop",
                                 "routed_experts": _field([[[7, 8]]] * (n - 1), start=0)}]}
        return {}


class FakeIO:
    def __init__(self, root: Path):
        self.root, self.sent = root, []

    async def send(self, step, rows):
        self.sent.append((step, [dict(r) for r in rows]))

    async def wait_broadcast(self, step, *, timeout_s):
        d = self.root / "broadcasts" / f"step_{step}"
        d.mkdir(parents=True, exist_ok=True)
        (self.root / "checkpoints" / f"step_{step}").mkdir(parents=True, exist_ok=True)
        return d

    def metrics(self, step):
        return {}

    def checkpoint(self, step):
        return self.root / "checkpoints" / f"step_{step}"


def test_router_replay_samples_over_generate_keeps_routes_and_sends_them(tmp_path: Path) -> None:
    from crucible.train.bridges.tinker.sampler import _EPISODE

    store = RoutingStore()
    b = PrimeRLBackend(io=FakeIO(tmp_path), http=GenHttp(), base_model=MODEL, learning_rate=4e-5, routing=store)

    async def go():
        tr = await b.trainer(MODEL, rank=32)
        s = await b.sampler(MODEL, None)
        tok = _EPISODE.set("ep-1")
        try:
            out = await s.sample([1, 2, 3], max_tokens=8, temperature=1.0, stop=None, seed=1)
        finally:
            _EPISODE.reset(tok)
        path, body = b.http.posts[-1]
        assert path == "/inference/v1/generate" and body["token_ids"] == [1, 2, 3] and body["cache_salt"] == "crucible-ep-1"
        assert out.tokens == [50, 51] and out.logprobs == [-0.1, -0.2] and store.get([1, 2, 3, 50, 51]) is not None
        d = smoke.datum([1, 2, 3], out.tokens, out.logprobs, 1.0, rollout="r", segment=0)
        fb = await tr.forward_backward([d], loss_fn="importance_sampling", config=None)
        row = b.io.sent[0][1][0]
        assert row["routed_experts"]["shape"] == [5, 1, 2] and row["routing_coverage"] == 1.0
        assert fb.metrics["routing/coverage_mean"] == 1.0 and fb.metrics["routing/replayed_share"] == 1.0
        assert store.get([1, 2, 3, 50, 51]) is None  # trained: dropped

    asyncio.run(go())


def test_a_step_replays_all_its_sequences_or_none_and_the_json_wire_carries_base64() -> None:
    assert wire.replay_all_or_none([{"routed_experts": {"x": 1}}, {"routed_experts": None}]) == [None, None]
    assert wire.replay_all_or_none([{"routed_experts": {"x": 1}, "routing_coverage": 1.0}]) == [{"x": 1}]
    assert wire.replay_all_or_none([{"routed_experts": {"x": 1}, "routing_coverage": 0.9}]) == [None]  # a gap: refused, as R3 does
    grid = wire.simple_pack([{"routed_experts": {"data": b"\x01\x02", "shape": [1, 1, 2], "dtype": "uint8"}}], ranks=1)
    assert isinstance(grid[0][0]["routed_experts"]["data"], str)


def test_replay_is_on_for_the_moe_run_and_off_when_asked() -> None:
    cfg = TrainConfig(model=MODEL, require_band=False)
    assert deploy.trainer_toml(cfg, deploy.DeltaConfig(), Path("/w"))["enable_router_replay"] is True
    assert deploy.inference_toml(cfg, deploy.DeltaConfig())["vllm"]["enable_return_routed_experts"] is True
    off = deploy.DeltaConfig(router_replay=False)
    assert deploy.trainer_toml(cfg, off, Path("/w"))["enable_router_replay"] is False
    assert deploy.inference_toml(cfg, off)["vllm"]["enable_return_routed_experts"] is False
