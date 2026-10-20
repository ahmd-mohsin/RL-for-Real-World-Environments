"""M11 lab audit R-1: a masked verl session must not crash verl's v1 trainer, and its placeholder must not reach a stock
estimator's group mean.

verl v1 tags every stored row with ``extra_fields["min_global_steps"]``/``["max_global_steps"]`` (verl 5eb4826
``verl/trainer/ppo/v1/agent_loop_tq.py:216-221``) and its metrics step converts them with ``np.array(..., dtype=int)``
after the actor update (``v1/trainer_base.py:1901-1902``): a ``None`` (the placeholder, or a frozen-actor sequence) raised
TypeError and lost the step. A placeholder's reward 0.0 is right only under ``crucible_delta``; a stock estimator
(``core_algos.py:267-331``) would count it in the group mean.

The routing and the step stamping are torch-free (``crucible.train.bridges.adapters``) and run everywhere; the tests
through ``CrucibleAgentLoop`` run the pinned verl source (``_upstream``) and need torch."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.train.bridges.adapters import stamp_verl_steps, verl_agent_loop_output, verl_mask_route
from crucible.train.bridges.common import flatten


class AttrDict(dict):
    __getattr__ = dict.get


def _cfg(estimator=None, use_v1=None) -> AttrDict:
    cfg = AttrDict(actor_rollout_ref=AttrDict(rollout=AttrDict()))
    if estimator is not None:
        cfg["algorithm"] = AttrDict(adv_estimator=estimator)
    if use_v1 is not None:
        cfg["trainer"] = AttrDict(use_v1=use_v1)
    return cfg


def _judged_record(rid: str = "r") -> RolloutRecord:
    """A live rollout with a frozen judge call (context only) before the agent's two chained calls."""
    return RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", reward=1.0, turns=[
        Turn(call_id=f"{rid}:j", actor="judge", prompt_token_ids=[3], completion_token_ids=[4], logprobs=None, policy_version=None),
        Turn(call_id=f"{rid}:c1", actor="agent", prompt_token_ids=[1, 2], completion_token_ids=[5, 6], logprobs=[-0.1, -0.2], policy_version=3),
        Turn(call_id=f"{rid}:c2", actor="agent", prompt_is_delta=True, prompt_token_ids=[9, 9], completion_token_ids=[7, 8],
             logprobs=[-0.3, -0.4], policy_version=4)])


def _v1_metrics_tags(extra_fields: list[dict]) -> None:
    """What v1 does with the tags: ``agent_loop_tq.py:216-221`` copies them from ``extra_fields`` into the row tags and
    ``trainer_base.py:1901-1902`` converts them with ``np.array(..., dtype=int)`` (without numpy: ``int``, which raises the
    same TypeError on a None)."""
    tags = [{"min_global_steps": e.get("min_global_steps"), "max_global_steps": e.get("max_global_steps")} for e in extra_fields]
    try:
        import numpy as np
    except ImportError:
        np = None
    for key in ("min_global_steps", "max_global_steps"):
        if np is not None:
            np.array([t[key] for t in tags], dtype=int)
        else:
            [int(t[key]) for t in tags]


# ------------------------------------------------------------------------------------------------- routing (no torch)

def test_the_route_follows_the_trainer_and_the_estimator():
    assert verl_mask_route(_cfg("crucible_delta")) == "v1_placeholder"
    assert verl_mask_route(_cfg("crucible_delta", use_v1=True)) == "v1_placeholder"
    for stock in ("grpo", "rloo", "gae"):
        assert verl_mask_route(_cfg(stock)) == "v1_drop"
    assert verl_mask_route(_cfg("crucible_delta", use_v1=False)) == "legacy_placeholder"


def test_the_legacy_trainer_under_a_stock_estimator_is_refused():
    for stock in ("grpo", "rloo", None):
        with pytest.raises(ValueError, match="legacy trainer"):
            verl_mask_route(_cfg(stock, use_v1=False))


def test_a_stock_estimator_with_the_group_filter_is_refused():
    cfg = _cfg("grpo")
    cfg["algorithm"]["filter_groups"] = AttrDict(enable=True, metric="reward")
    with pytest.raises(ValueError, match="filter_groups"):
        verl_mask_route(cfg)  # an all-masked uid would have no row, which verl's group filter raises on
    cfg["algorithm"]["filter_groups"]["enable"] = False
    assert verl_mask_route(cfg) == "v1_drop"
    cfg = _cfg("crucible_delta")
    cfg["algorithm"]["filter_groups"] = AttrDict(enable=True, metric="reward")
    assert verl_mask_route(cfg) == "v1_placeholder"  # a placeholder row keeps every uid present


def test_a_config_with_no_algorithm_key_still_routes():
    assert verl_mask_route(_cfg()) == "v1_drop"  # an unknown estimator is a stock one: no placeholder reaches it
    assert verl_mask_route(AttrDict()) == "v1_drop"
    assert verl_mask_route(None) == "v1_drop"


# ------------------------------------------------------------------------------------------- step stamping (no torch)

def test_a_frozen_actor_sequence_takes_its_siblings_steps():
    seqs = flatten(_judged_record(), frozen_actors=["judge"])
    assert [s.actor for s in seqs] == ["judge", "agent"]
    extra = [verl_agent_loop_output(s)["extra_fields"] for s in seqs]
    assert extra[0]["min_global_steps"] is None  # no sampled token of a trained actor: no version of its own
    with pytest.raises(TypeError):
        _v1_metrics_tags(extra)
    stamp_verl_steps(extra, 9)
    assert (extra[0]["min_global_steps"], extra[0]["max_global_steps"]) == (3, 4)  # the session's own span, not the current step
    assert (extra[1]["min_global_steps"], extra[1]["max_global_steps"]) == (3, 4)  # a live row keeps its own tags
    _v1_metrics_tags(extra)


def test_a_row_with_no_sibling_takes_the_current_step():
    extra = [{"min_global_steps": None, "max_global_steps": None}]
    stamp_verl_steps(extra, 12)
    assert extra == [{"min_global_steps": 12, "max_global_steps": 12}]
    _v1_metrics_tags(extra)


def test_with_no_step_and_no_sibling_the_row_is_left_for_the_legacy_worker():
    extra = [{"min_global_steps": None, "max_global_steps": None}]
    stamp_verl_steps(extra, None)
    assert extra == [{"min_global_steps": None, "max_global_steps": None}]


# ------------------------------------------------------------------- CrucibleAgentLoop on the pinned verl (needs torch)

@pytest.fixture
def verl(monkeypatch):
    pytest.importorskip("torch")
    sys.path.insert(0, str(Path(__file__).parent))
    import test_m12_audit_bridges as m12

    m12._forget_crucible_verl()
    with m12.upstream("verl", m12.VERL_MODULES) as ns:
        m12._verl_plugin_discovery(monkeypatch)
        yield ns
    m12._forget_crucible_verl()


def _loop(verl, rollout, *, algorithm=None, trainer=None):
    import test_m12_audit_bridges as m12

    class Server:  # never called: the rollouts below are canned
        async def generate(self, *a, **k):
            raise AssertionError("no generation expected")

    return m12._agent_loop(verl, Server(), lambda _loop: rollout, algorithm=algorithm, trainer=trainer, frozen_actors=("judge",), turn_weight=0.0)


def _sessions(loop, n: int, **kw) -> list:
    async def go():
        return await asyncio.gather(*[loop.run({"temperature": 1.0}, uid="u0", session_id=i, raw_prompt=[{"role": "user", "content": "x"}],
                                               global_steps=5, **kw) for i in range(n)])
    return asyncio.run(go())


async def _masked_or_live(rid, prompt, seed, gen, *, temperature=1.0):
    return None if seed == 1 else _judged_record(f"s{seed}")


def test_a_masked_session_and_a_frozen_actor_sequence_pass_v1_metrics_under_crucible_delta(verl):
    outs = _sessions(_loop(verl, _masked_or_live), 2)
    live, masked = outs
    assert [o.extra_fields["crucible_actor"] for o in live] == ["judge", "agent"]
    assert masked[0].extra_fields["crucible_mask_route"] == "v1_placeholder" and masked[0].response_mask == [0]
    assert {o.extra_fields["crucible_mask_route"] for o in live} == {"v1_placeholder"}  # every row records the route
    assert (masked[0].extra_fields["min_global_steps"], masked[0].extra_fields["max_global_steps"]) == (5, 5)
    assert (live[0].extra_fields["min_global_steps"], live[0].extra_fields["max_global_steps"]) == (3, 4)
    for o in [*live, *masked]:
        o.as_dict()  # verl's own as_dict (agent_loop_tq.py:207)
    _v1_metrics_tags([o.extra_fields for o in [*live, *masked]])


def test_under_a_stock_estimator_v1_receives_no_row_for_a_masked_session(verl):
    outs = _sessions(_loop(verl, _masked_or_live, algorithm=AttrDict(adv_estimator="grpo")), 2)
    assert outs[1] == []  # v1 skips an empty output (agent_loop_tq.py:153-157); the uid group stays finished
    assert len(outs[0]) == 2 and {o.extra_fields["crucible_mask_route"] for o in outs[0]} == {"v1_drop"}
    _v1_metrics_tags([o.extra_fields for o in outs[0]])


def test_the_legacy_worker_under_use_v1_true_and_a_stock_estimator_is_refused(verl):
    """``verl/experimental/one_step_off_policy`` runs the legacy worker (no ``session_id``) under ``use_v1: true``: the route
    says v1_drop, which the legacy worker cannot honour (it has no skip channel)."""
    loop = _loop(verl, _masked_or_live, algorithm=AttrDict(adv_estimator="grpo"))
    with pytest.raises(ValueError, match="legacy agent-loop worker"):
        asyncio.run(loop.run({"temperature": 1.0}, uid="u0", raw_prompt=[{"role": "user", "content": "x"}]))
    ok = _loop(verl, _masked_or_live)  # crucible_delta: the legacy worker gets a placeholder for the masked session
    out = asyncio.run(ok.run({"temperature": 1.0}, uid="u0", raw_prompt=[{"role": "user", "content": "x"}]))
    assert out.extra_fields["crucible_actor"] == "agent"  # session 0 is live: its final sequence


def test_the_legacy_trainer_under_a_stock_estimator_refuses_at_construction(verl):
    with pytest.raises(ValueError, match="legacy trainer"):
        _loop(verl, _masked_or_live, algorithm=AttrDict(adv_estimator="grpo"), trainer=AttrDict(use_v1=False))
    loop = _loop(verl, _masked_or_live, trainer=AttrDict(use_v1=False))  # crucible_delta: allowed
    assert loop.mask_route == "legacy_placeholder"


def test_a_config_with_no_algorithm_key_constructs(verl):
    loop = _loop(verl, _masked_or_live, algorithm=AttrDict())
    assert loop.mask_route == "v1_drop"


