"""Part 1 §3.2 #2, first half (2026-10-30, docs/research/env-synthesis/part1-framework-improvements.md; notes/05 §6): the
student path's dashboard row (``TinkerTrainer.step_metrics``, Tinker and DeltaAI) logged none of the leading indicators.
``behaviour_metrics`` adds them from what the episodes already record: the entropy estimate, lengths and turns,
stop-reason shares and truncation, format errors, flagged and give-up shares, and the within-group reward std."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from crucible.core.schema.rollout import RolloutRecord
from crucible.train.bridges.tinker.backend import TokenDatum
from crucible.train.bridges.tinker.trainer import Group, behaviour_metrics, step_metrics


def _rec(rid: str, reward: float | None, *, calls: tuple[int, ...], stop: str = "agent_done", truncated: bool = False, errs: int = 0,
         flags: tuple[str, ...] = (), give_up: str | None = None, context: int | None = None) -> RolloutRecord:
    turns = [{"call_id": f"{rid}:c{i}", "actor": "agent", "prompt_token_ids": [1], "completion_token_ids": [2] * n, "logprobs": [-0.5] * n,
              "policy_version": 0, "format_errors": errs if i == 0 else 0} for i, n in enumerate(calls)]
    return RolloutRecord.model_validate({"rollout_id": rid, "sample": rid, "task": "t", "harness": "h", "policy": "p", "reward": reward,
                                         "mask_sample": reward is None, "turns": turns, "stop": stop, "truncated": truncated,
                                         "flags": [{"rule": f, "evidence": "x"} for f in flags], "progress": {"give_up": give_up},
                                         "trajectory_tokens": context})


def _batch() -> list[Group]:
    g1 = [_rec("a", 1.0, calls=(2, 4), errs=1, flags=("writes_test_file",), context=100),
          _rec("b", 0.0, calls=(3, 3, 3), stop="max_steps", truncated=True, context=300)]
    g2 = [_rec("c", 0.0, calls=(5,), give_up="the task needs network access"), _rec("d", 0.5, calls=(1,)),
          _rec("e", None, calls=(6,), stop="timeout", truncated=True)]  # masked: still how an episode ended
    return [Group(0, None, "t", "v1", "g1", g1, 0, 0.0), Group(1, None, "t", "v1", "g2", g2, 0, 0.0)]


DATUMS = [TokenDatum(input_ids=[1, 2, 3], target_ids=[2, 3, 4], logprobs=[-9.0, -0.5, -1.5], advantages=[0.0, 1.0, 1.0], mask=[0, 1, 1])]


def test_the_leading_indicators_are_read_from_the_episodes_and_the_trained_tokens() -> None:
    m = behaviour_metrics(_batch(), DATUMS)
    assert m["entropy/sampled"] == pytest.approx(1.0)  # -mean log p of the two trained tokens; the prompt's -9 is not one
    assert m["length/turns/max"] == 3.0 and m["length/turns/mean"] == pytest.approx(8 / 5)
    assert m["length/call_tokens/max"] == 6.0 and m["length/call_tokens/mean"] == pytest.approx(27 / 8)
    assert m["length/context_tokens/mean"] == pytest.approx(200.0)  # only the episodes that recorded it
    assert m["stop/agent_done"] == pytest.approx(3 / 5) and m["stop/max_steps"] == m["stop/timeout"] == pytest.approx(1 / 5)
    assert m["stop/context_overflow"] == m["stop/too_many_errors"] == 0.0  # always present: a chart without gaps
    assert m["stop/truncated"] == pytest.approx(2 / 5)
    assert m["format_errors/per_episode"] == m["format_errors/episode_share"] == pytest.approx(1 / 5)
    assert m["hack/flagged_share"] == m["give_up/share"] == pytest.approx(1 / 5)
    assert m["reward/group_std_mean"] == pytest.approx((0.5 + 0.25) / 2)  # the masked episode has no reward to spread
    assert behaviour_metrics([], []) == {}


def test_the_dashboard_row_carries_them() -> None:
    m = step_metrics(_batch(), DATUMS, SimpleNamespace(metrics={}), [], version=0, sampler_version=0, drops={}, usd_step=0.0, usd_cum=0.0,
                     cap_usd=1.0, tokens={}, train_s=0.0)
    assert m["stop/truncated"] == pytest.approx(2 / 5) and m["entropy/sampled"] == pytest.approx(1.0)  # before: neither key
