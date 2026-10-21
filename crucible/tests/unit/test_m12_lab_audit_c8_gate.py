"""M12 lab audit C-8: the trainer's own validity gate (``TorchTrainer._gate``) missed checks ``Turn.missing`` makes.

The Runtime's capture gate masks a call with no ``policy_version`` or a ``prefix_break`` before the trainer sees it, but a
caller that reaches ``step`` directly (offline records, the rLLM tinker route) trained them. Sampling penalties were
recorded and never read, so a penalised call trained against the unpenalised π_θ; a log-prob above 0 passed the finite
check."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crucible.core.schema.rollout import RolloutRecord, Turn  # noqa: E402
from crucible.train.engine import TinyPolicy  # noqa: E402
from crucible.train.trainer import TorchTrainer, TrainerConfig  # noqa: E402


def _rec(rid: str = "r", **turn) -> RolloutRecord:
    t = {"call_id": f"{rid}:c1", "actor": "agent", "prompt_token_ids": [1, 2, 3], "completion_token_ids": [4, 5], "logprobs": [-0.5, -0.7],
         "policy_version": 0, "sampling": {"temperature": 1.0}, **turn}
    return RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", reward=1.0, turns=[Turn(**t)])


def _gate(*recs: RolloutRecord) -> tuple[list[RolloutRecord], int]:
    return TorchTrainer(TinyPolicy(32, 8), TrainerConfig())._gate(list(recs))


def test_a_complete_record_passes():
    out, bad = _gate(_rec())
    assert bad == 0 and not out[0].mask_sample and out[0].reward == 1.0


@pytest.mark.parametrize("turn", [{"policy_version": None}, {"prefix_break": True}, {"logprobs": [-0.5, 0.2]}])
def test_missing_version_prefix_break_and_positive_logprob_are_masked(turn):
    out, bad = _gate(_rec("bad", **turn), _rec("ok"))
    assert bad == 1
    assert out[0].mask_sample and out[0].reward is None and out[0].failure_kind == "crucible:capture_incomplete"
    assert not out[1].mask_sample  # its sibling still trains


def test_rounding_at_zero_is_not_a_positive_logprob():
    out, bad = _gate(_rec(logprobs=[0.0, 1e-6]))
    assert bad == 0 and not out[0].mask_sample


def test_an_untrainable_call_is_exempt():
    out, bad = _gate(_rec(policy_version=None, prefix_break=True, trainable=False, model="judge"))
    assert bad == 0 and not out[0].mask_sample


@pytest.mark.parametrize("pen", [{"repetition_penalty": 1.1}, {"presence_penalty": 0.5}, {"frequency_penalty": -0.2}])
def test_a_sampling_penalty_is_refused(pen):
    with pytest.raises(ValueError, match=f"sampled with {next(iter(pen))}="):
        _gate(_rec(sampling={"temperature": 1.0, **pen}))


def test_neutral_penalties_pass():
    out, bad = _gate(_rec(sampling={"temperature": 1.0, "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}))
    assert bad == 0 and not out[0].mask_sample


def test_step_trains_only_the_valid_record():
    tr = TorchTrainer(TinyPolicy(32, 8, seed=0), TrainerConfig())
    recs = [_rec("a"), _rec("b", policy_version=None), _rec("c"), _rec("d", prefix_break=True)]
    recs[2].reward = 0.0
    m = tr.step(recs, ["g", "g", "g", "g"])
    assert m["train/records_invalid"] == 2.0
    assert m["train/items"] == 2.0 and m["train/tokens"] == 4.0  # a and c trained, their 2 tokens each


def test_just_above_the_tolerance_is_masked():
    out, bad = _gate(_rec(logprobs=[-0.5, 2e-3]))
    assert bad == 1 and out[0].mask_sample


def test_a_penalised_record_already_masked_or_unscored_is_not_refused():
    masked = _rec("m", sampling={"temperature": 1.0, "presence_penalty": 0.5})
    masked.mask_sample = True
    unscored = _rec("u", sampling={"temperature": 1.0, "presence_penalty": 0.5})
    unscored.reward = None
    out, bad = _gate(masked, unscored)
    assert bad == 0 and len(out) == 2


def test_a_nan_penalty_is_not_neutral():
    """The schema refuses a NaN penalty (finite numbers only); a record built without validation is still not neutral."""
    from types import SimpleNamespace

    from crucible.train.trainer import _penalties

    with pytest.raises(ValueError, match="finite"):
        _rec(sampling={"temperature": 1.0, "repetition_penalty": float("nan")})
    assert _penalties(SimpleNamespace(repetition_penalty=float("nan"))) == "repetition_penalty=nan"
    assert _penalties(SimpleNamespace(repetition_penalty=1.0, presence_penalty=None)) == ""


def test_the_verifiers_import_masks_a_requested_penalty_it_did_not_serve():
    """The generate server samples unpenalised: a harness that asked for a penalty is masked by name (harness_unsupported),
    instead of reaching the trainer with a recorded penalty it would refuse."""
    from crucible.interop.verifiers.v1_import import unsupported_sampling

    assert unsupported_sampling({"temperature": 1.0, "frequency_penalty": 0.3}) == "frequency_penalty=0.3"
    assert unsupported_sampling({"repetition_penalty": 1.05}) == "repetition_penalty=1.05"
    assert unsupported_sampling({"temperature": 1.0, "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0}) is None
