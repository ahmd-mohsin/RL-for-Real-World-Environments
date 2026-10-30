"""Part 1 A7 (2026-10-30, docs/research/env-synthesis/part1-framework-improvements.md §2(a)): framework_health was an
operator tool the trainer never read (``scripts/train_monitor.py health`` exits 3 on an open CRITICAL defect). Now the
trainer checks after every step and stops before the next one while such a defect is open; the stop saves state, and a
resume trains on once the fix is recorded. Plus the run monitor §2(a) asks for: the share of the gradient that comes from
shaping only (``shaping_metrics``)."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from crucible.core.schema.rollout import RolloutRecord
from crucible.factory import framework_health as fh
from crucible.factory import tinker as ft
from crucible.factory.tinker import framework_pause, train
from crucible.train.bridges.tinker import trainer as trainer_mod
from crucible.train.bridges.tinker.backend import TokenDatum
from crucible.train.bridges.tinker.fake import FakeTinker
from crucible.train.bridges.tinker.trainer import Group, TrainConfig, shaping_metrics, step_metrics
from crucible.train.chat import ByteChatCodec
from tests.unit.test_open_risks_loop import MODEL, ScriptedTrainer, _Sink

BAD = ["fd-0123456789 trainer_numeric (trainer): loss is NaN or inf"]


def _trainer(tmp_path, health, *, store=None, sinks=()) -> ScriptedTrainer:
    env = SimpleNamespace(task=SimpleNamespace(task_id="t1", split="train", verifier_version="v1"), policy_kwargs=dict)
    cfg = TrainConfig(model=MODEL, group_size=2, steps=3, require_band=False, turn_eta=0.0, quarantine_q=0, cap_usd=100.0, save_every=10)
    return ScriptedTrainer([env], config=cfg, backend=FakeTinker(store=store if store is not None else {}), run_dir=tmp_path, codec=ByteChatCodec(),
                           objective_id="obj", sinks=list(sinks), plan=[0.0] * 6, health=health)


def _steps(tmp_path) -> list[int]:
    return [json.loads(x)["step"] for x in (tmp_path / "steps.jsonl").read_text().splitlines()]


def test_an_open_critical_defect_stops_the_run_before_the_next_step_and_a_resume_trains_on_once_it_is_fixed(tmp_path) -> None:
    store, sink, seen = {}, _Sink(), []
    t = _trainer(tmp_path, lambda d: seen.append(d) or BAD, store=store, sinks=[sink])
    out = asyncio.run(t.run())
    assert out["stop"] == "framework_health" and out["steps_done"] == 1 and seen == [tmp_path]  # before: no such check, 3 steps
    assert out["health_stop"] == {"step": 1, "defects": BAD} and ("stop", {"stop": "framework_health", **out["health_stop"]}) in sink.records
    assert json.loads((tmp_path / "state.json").read_text())["resume"]["step"] == 1  # saved at the stop (save_every 10)
    out = asyncio.run(_trainer(tmp_path, lambda d: BAD, store=store).run())  # resumed, the defect still open: the check runs first
    assert out["stop"] == "framework_health" and _steps(tmp_path) == [1]
    out = asyncio.run(_trainer(tmp_path, lambda d: [], store=store).run())  # fixed and recorded: the check comes back empty
    assert out["stop"] == "done" and out["steps_done"] == 3 and _steps(tmp_path) == [1, 2, 3]


def test_a_check_that_fails_is_a_warning_and_the_run_trains_on(tmp_path, caplog) -> None:
    def broken(run_dir):
        raise OSError("a stored episode could not be read")

    with caplog.at_level(logging.WARNING):
        out = asyncio.run(_trainer(tmp_path, broken).run())
    assert out["stop"] == "done" and out["steps_done"] == 3 and out["health_stop"] is None
    assert "framework_health not checked (OSError" in caplog.text


def test_the_factory_check_reads_the_runs_own_defects_and_its_fix_journal(tmp_path) -> None:
    (tmp_path / "steps.jsonl").write_text(json.dumps({"step": 1, "loss": float("nan"), "rewards": []}) + "\n")
    defects = framework_pause(tmp_path)  # a planted defect: the NaN loss is CRITICAL (trainer_numeric)
    assert len(defects) == 1 and "trainer_numeric" in defects[0]
    fh.record_fix(tmp_path, defect=defects[0].split(" ")[0], component="trainer", version="t2", commit="abc1234", after_step=1)
    assert framework_pause(tmp_path) == []


def test_train_hands_the_check_to_the_trainer_unless_told_not_to(tmp_path, monkeypatch) -> None:
    class Built(Exception):
        pass

    def built(*a, **kw):
        raise Built(kw)

    monkeypatch.setattr(trainer_mod, "TinkerTrainer", built)
    monkeypatch.setattr(ft, "make_env", lambda t, **kw: SimpleNamespace(task=t))
    task, cfg = SimpleNamespace(task_id="t1", verifier_version="v1"), TrainConfig(model=MODEL, require_band=False)
    for flag, want in ((True, framework_pause), (False, None)):
        with pytest.raises(Built) as e:
            asyncio.run(train([task], config=cfg, run_dir=tmp_path, backend=FakeTinker(), codec=ByteChatCodec(), observe=False, health_stop=flag))
        assert e.value.args[0]["health"] is want


# ---- the shaping-only share ----------------------------------------------------------------------------------------------------
def _rec(rid: str, reward: float) -> RolloutRecord:
    return RolloutRecord.model_validate({"rollout_id": rid, "sample": rid, "task": "t", "harness": "h", "policy": "p", "reward": reward})


def _datum(rid: str, adv: list[float], scale: float = 1.0) -> TokenDatum:
    n = len(adv) + 1  # one prompt token first: mask 0, its advantage never counts
    return TokenDatum(input_ids=[1] * n, target_ids=[1] * n, logprobs=[-1.0] * n, advantages=[9.0, *adv], mask=[0] + [1] * len(adv),
                      meta={"rollout": rid, "scale": scale})


def test_the_shaping_shares_split_each_tokens_advantage_into_outcome_and_shaping() -> None:
    # g1 has outcome spread (o = +0.5, -0.5); g2 is flat (o = 0) and trains only through honesty
    batch = [Group(0, None, "t", "v1", "g1", [_rec("a", 1.0), _rec("b", 0.0)], 0, 0.0),
             Group(1, None, "t", "v1", "g2", [_rec("c", 0.0), _rec("d", 0.0)], 0, 0.0)]
    datums = [_datum("a", [0.25, 0.35], scale=0.5),  # 0.5 and 0.7 before the token-mean scale: turn credit added 0.2
              _datum("b", [-0.5]), _datum("c", [-0.3]), _datum("d", [0.0])]
    m = shaping_metrics(batch, datums)
    assert m["credit/shaping_share"] == pytest.approx(0.5 / (1.5 + 0.5))  # |A - o|: 0.2 (turn credit) + 0.3 (honesty)
    assert m["credit/shaping_only_share"] == pytest.approx(0.3 / 2.0)  # the flat group's tokens: no outcome asked for them
    assert shaping_metrics([], []) == {}
    row = step_metrics(batch, datums, SimpleNamespace(metrics={}), [], version=0, sampler_version=0, drops={}, usd_step=0.0, usd_cum=0.0, cap_usd=1.0,
                       tokens={}, train_s=0.0)
    assert row["credit/shaping_only_share"] == pytest.approx(0.15)  # before: no such key on the dashboard row
