"""m13 C-1: metric / record sinks. Step metrics, trained records, failures and the summary reach every sink; a sink that
raises is disabled and the run finishes (a metrics backend never stops training)."""

from __future__ import annotations

import json

import pytest

from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask
from crucible.train.mismatch import Correction
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import SchedulerConfig
from crucible.train.sinks import JsonlSink, Tracking, from_spec
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher


class Broken:
    def __init__(self):
        self.calls = 0

    def log(self, metrics, step):
        self.calls += 1
        raise BrokenPipeError("wandb went away")

    def record(self, kind, data, step):
        pass

    def summary(self, data):
        pass

    def finish(self, status):
        pass


async def test_runtime_streams_to_sinks_and_survives_a_broken_one(tmp_path):
    model = TinyPolicy(32, 48, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64, numerics="bf16", decode_delay_s=0.001)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2, correction=Correction("dis")))
    broken = Broken()
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=6, max_turn_tokens=3, sinks=[JsonlSink(tmp_path / "m"), broken], max_masked_share=0.9),
                 SchedulerConfig(n=4, B=2, max_staleness=2), eng, tr, InprocPublisher(tr, eng),
                 lambda: CopyTask(turns=2, symbols=8, latency_s=0.001, infra_fail=0.1), {"copy": [f"p{i}" for i in range(4000)]})
    rep = await rt.run()
    assert len(rep.steps) == 6 and broken.calls == 1  # disabled after its first failure; the run went on
    lines = [json.loads(x) for x in (tmp_path / "m" / "metrics.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines] == [1, 2, 3, 4, 5, 6] and all("reward/mean" in x for x in lines)
    recs = [json.loads(x) for x in (tmp_path / "m" / "records.jsonl").read_text().splitlines()]
    rolls = [r for r in recs if r["kind"] == "rollout"]
    assert len(rolls) == len(rep.trained_records) and rolls[0]["data"]["turns"]  # full crucible.rollout/v1 bodies
    assert sum(r["kind"] == "failure" for r in recs) == sum(rep.failures.values())  # every failure, not only the examples
    summ = json.loads((tmp_path / "m" / "summary.json").read_text())
    assert summ["status"] == "steps" and summ["steps"] == 6 and "Broken" in summ["sinks_failed"]


def test_spec_and_isolation(tmp_path):
    sinks = from_spec(f"jsonl:{tmp_path / 'j'},console")
    t = Tracking(sinks + [Broken()])
    t.log({"a": 1.0, "nan": float("nan"), "s": "text"}, 1)
    t.finish("steps")
    row = json.loads((tmp_path / "j" / "metrics.jsonl").read_text())
    assert row["a"] == 1.0 and "nan" not in row and "s" not in row and list(t.failed) == ["Broken"]
    with pytest.raises(ValueError):
        from_spec("mlflow:x")
