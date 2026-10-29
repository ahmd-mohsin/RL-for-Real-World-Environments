"""Part 1 §3.2 #1 (2026-10-30, docs/research/env-synthesis/part1-framework-improvements.md §3.3): on 2 trainer GPUs the
DeltaAI run stores the LoRA weights in bf16, where an AdamW step of about the learning rate can round back to the same
value. The bridge now compares each step's adapter broadcast with the previous one (``backend.lora_change``) and logs
``lora/changed_frac`` (overall, A and B), so the first GPU run shows whether updates survive. No GPU: fake broadcasts."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file

from crucible.train.bridges.primerl.backend import ADAPTER_FILE, PrimeRLBackend, lora_change
from crucible.train.bridges.primerl.smoke import datum
from tests.unit.test_primerl_bridge import MODEL, FakeHttp, FakeIO

LR = 4e-5  # the DeltaAI run's learning rate: about one AdamW step's size per weight


def _adapter(d: Path, a: float, b: float, dtype) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    save_file({"model.layers.0.self_attn.q_proj.lora_A.weight": torch.full((4, 64), a).to(dtype),
               "model.layers.0.self_attn.q_proj.lora_B.weight": torch.full((64, 4), b).to(dtype)}, str(d / ADAPTER_FILE))
    return d


@pytest.mark.parametrize(("dtype", "a_share"), [(torch.bfloat16, 0.0), (torch.float32, 1.0)])
def test_a_step_of_the_learning_rate_is_rounded_away_on_bf16_a_and_kept_on_b(tmp_path: Path, dtype, a_share: float) -> None:
    prev = _adapter(tmp_path / "step_0", 0.02, 0.0, dtype)  # A at a kaiming-sized value, B at its zero init
    cur = _adapter(tmp_path / "step_1", 0.02 + LR, LR, dtype)
    m = lora_change(prev, cur)
    assert m["lora/changed_frac_A"] == a_share  # bf16 holds 0.02 as 0.02001953 and 0.02004 rounds back to it; before: not measured
    assert m["lora/changed_frac_B"] == 1.0 and m["lora/changed_frac"] == (a_share + 1.0) / 2  # B near 0 moves in either dtype
    assert m["lora/norm_B"] == pytest.approx(LR * 16, rel=1e-2)
    assert lora_change(prev, tmp_path / "no_such_step") == {}


class AdapterIO(FakeIO):
    """``FakeIO`` whose broadcasts hold an adapter: the startup one at init, then steps that only move B (bf16)."""

    async def wait_broadcast(self, step, *, timeout_s):
        d = await super().wait_broadcast(step, timeout_s=timeout_s)
        _adapter(d, 0.02 + LR * step, LR * step, torch.bfloat16)
        return d


class TornIO(AdapterIO):
    async def wait_broadcast(self, step, *, timeout_s):
        d = await super().wait_broadcast(step, timeout_s=timeout_s)
        if step == 0:
            (d / ADAPTER_FILE).write_bytes(b"torn")  # an unreadable startup adapter
        return d


def test_the_share_reaches_the_steps_metrics_and_a_bad_file_never_stops_the_step(tmp_path: Path, caplog) -> None:
    async def step(io):
        b = PrimeRLBackend(io=io, http=FakeHttp(), base_model=MODEL, learning_rate=LR)
        tr = await b.trainer(MODEL, rank=32)
        return (await tr.forward_backward([datum([1, 2], [3, 4], [-1.0, -1.0], 1.0, rollout="r0", segment=0)],
                                          loss_fn="importance_sampling", config=None)).metrics

    m = asyncio.run(step(AdapterIO(tmp_path / "a")))
    assert m["lora/changed_frac_A"] == 0.0 and m["lora/changed_frac_B"] == 1.0 and m["lora/check_s"] >= 0.0
    assert "lora/changed_frac" not in asyncio.run(step(FakeIO(tmp_path / "b")))  # no adapter files (e.g. another transport)
    with caplog.at_level(logging.WARNING):
        m = asyncio.run(step(TornIO(tmp_path / "c")))
    assert "lora/changed_frac" not in m and m["primerl/step"] == 1.0 and "not measured" in caplog.text  # the step went on
