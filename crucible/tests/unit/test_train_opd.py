"""m13 C-5: teacher pool, on-policy distillation (token and full-vocabulary) and the k3 KL to a reference."""

from __future__ import annotations

import asyncio
import copy
import math

import pytest
import torch

from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask
from crucible.train.mismatch import Correction
from crucible.train.rollout import RolloutRecord, Turn
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import SchedulerConfig
from crucible.train.teachers import TeacherPool, TorchTeacher, VLLMTeacher
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher


def _recs(policy, *, teacher=None, ref=None, source="copy"):
    out = []
    for k, (rew, comp) in enumerate([(1.0, [3, 4, 5]), (0.0, [6, 7, 8])]):
        prompt = [1, 2]
        with torch.no_grad():
            beh = policy.token_logprobs(prompt, comp).tolist()
            tl = teacher.token_logprobs(prompt, comp).tolist() if teacher is not None else None
            rl = ref.token_logprobs(prompt, comp).tolist() if ref is not None else None
        out.append(RolloutRecord(rollout_id=f"r{k}", sample=f"r{k}", task="t", harness="h", policy="p", reward=rew, source=source,
                                 turns=[Turn(call_id="agent:c1", actor="agent", prompt_token_ids=prompt, completion_token_ids=comp, logprobs=beh,
                                             policy_version=0, teacher_logprobs=tl, ref_logprobs=rl)]))
    return out


def _params(m):
    return [p.detach().clone() for p in m.parameters()]


@pytest.mark.parametrize("unit", ["call", "segment"])
def test_zero_weight_reproduces_the_rl_step_bitwise(unit):
    base = TinyPolicy(16, 12, seed=0)
    recs = _recs(base, teacher=TinyPolicy(16, 12, seed=5), ref=TinyPolicy(16, 12, seed=6))
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    TorchTrainer(a, TrainerConfig(item_unit=unit)).step(copy.deepcopy(recs), ["g", "g"])
    TorchTrainer(b, TrainerConfig(item_unit=unit, opd_coef=0.0, rl_coef=1.0, kl_coef=0.0)).step(copy.deepcopy(recs), ["g", "g"])
    assert all(torch.equal(x, y) for x, y in zip(_params(a), _params(b), strict=True))


def test_teacher_equal_to_the_policy_gives_zero_gradient():
    base = TinyPolicy(16, 12, seed=0)
    recs = _recs(base, teacher=base)  # π_T == π_behaviour: the dense signal is exactly 0
    tr = TorchTrainer(copy.deepcopy(base), TrainerConfig(item_unit="call", opd_coef=1.0, rl_coef=0.0))
    tr.step(recs, ["g", "g"])  # the gradient is the criterion (AdamW's weight decay moves parameters regardless)
    assert all(p.grad is None or float(p.grad.abs().max()) == 0.0 for p in tr.model.parameters())
    tr2 = TorchTrainer(copy.deepcopy(base), TrainerConfig(item_unit="call", opd_coef=1.0, rl_coef=0.0))
    tr2.step(_recs(base, teacher=TinyPolicy(16, 12, seed=5)), ["g", "g"])  # a different teacher: a real gradient
    assert any(p.grad is not None and float(p.grad.abs().max()) > 0.0 for p in tr2.model.parameters())


def test_token_mode_advantage_is_rl_plus_teacher_gap():
    base, teacher = TinyPolicy(16, 12, seed=0), TinyPolicy(16, 12, seed=5)
    recs = _recs(base, teacher=teacher)
    items = TorchTrainer(base, TrainerConfig(item_unit="call", opd_coef=0.3, rl_coef=0.5))._items(recs, ["g", "g"])
    for it, r, a in zip(items, recs, (0.5, -0.5), strict=True):
        t = r.turns[0]
        want = [0.5 * a + 0.3 * (x - y) for x, y in zip(t.teacher_logprobs, t.logprobs, strict=True)]
        assert it.adv == pytest.approx(want, abs=1e-12)


def test_full_vocab_kl_matches_a_hand_computation():
    base, teacher = TinyPolicy(16, 12, seed=0), TinyPolicy(16, 12, seed=5)
    recs = _recs(base)
    tr = TorchTrainer(base, TrainerConfig(item_unit="call", opd_coef=0.7, opd_mode="full_vocab"), teacher=teacher)
    items = tr._items(recs, ["g", "g"])
    lp, _roll, _adv, mask, _ = tr._padded(items, [0, 1], None)
    got = tr._extra_terms(items, [0, 1], lp, mask)
    for k, it in enumerate(items):
        ids = torch.tensor([it.prompt + it.completion[:-1]])
        ps = torch.softmax(base(ids)[0, len(it.prompt) - 1:].double(), -1)
        pt = torch.softmax(teacher(ids)[0, len(it.prompt) - 1:].double(), -1)
        hand = 0.7 * (ps * (ps.log() - pt.log())).sum(-1)
        assert torch.allclose(got[k, : len(it.completion)].double(), hand, atol=1e-6)
    with pytest.raises(ValueError):
        TorchTrainer(base, TrainerConfig(opd_coef=1.0, opd_mode="full_vocab"))  # needs a teacher


def test_k3_kl_golden_zero_at_reference_and_per_source():
    base = TinyPolicy(16, 12, seed=0)
    tr = TorchTrainer(base, TrainerConfig(item_unit="call", kl_coef=2.0, kl_coef_by_source={"math": 0.0}))
    recs = _recs(base, ref=base)  # π_ref == π_θ: k3 == 0
    items = tr._items(recs, ["g", "g"])
    lp, *_ , mask, _ = tr._padded(items, [0, 1], None)
    assert float(tr._extra_terms(items, [0, 1], lp, mask).detach().abs().max()) < 1e-12
    it = items[0]
    it.ref = [math.log(0.25)] * len(it.completion)
    lp1 = torch.full((1, len(it.completion)), math.log(0.5), dtype=torch.float64)
    k3 = tr._extra_terms([it], [0], lp1, torch.ones_like(lp1))
    assert float(k3[0, 0]) == pytest.approx(2.0 * (0.5 - math.log(0.5) - 1.0), abs=1e-12)
    assert all(i.kl == 0.0 and i.ref is None for i in tr._items(_recs(base, ref=base, source="math"), ["g", "g"]))  # per-source 0


def _runtime(pool, **cfg):
    model = TinyPolicy(32, 48, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64, numerics="bf16", decode_delay_s=0.001)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2, correction=Correction("dis"), opd_coef=0.5))
    return Runtime(RuntimeConfig(topology="colocated_partial", steps=4, max_turn_tokens=3, **cfg), SchedulerConfig(n=4, B=2, max_staleness=2),
                   eng, tr, InprocPublisher(tr, eng), lambda: CopyTask(turns=2, symbols=8, latency_s=0.001),
                   {"copy": [f"p{i}" for i in range(4000)]}, teacher_pool=pool)


async def test_runtime_scores_every_token_with_the_routed_teacher():
    t1 = TorchTeacher(TinyPolicy(32, 48, seed=7), id="coder", vocab_id="v1")
    t2 = TorchTeacher(TinyPolicy(32, 48, seed=8), id="math", vocab_id="v1")
    pool = TeacherPool([t1, t2], route={"copy": "coder", "gsm": "math"})
    rep = await _runtime(pool, student_vocab_id="v1").run()
    assert len(rep.steps) == 4 and sum(rep.series("teacher/scored")) >= len(rep.trained_records) and sum(rep.series("teacher/fallback")) == 0
    for r in rep.trained_records:
        assert r.teacher_id == "coder" and r.source == "copy"
        assert all(t.teacher_logprobs is not None and len(t.teacher_logprobs) == len(t.completion_token_ids) for t in r.turns if t.completion_token_ids)
    with pytest.raises(ValueError, match="vocabularies differ"):
        _runtime(pool, student_vocab_id="v2")  # refused before any rollout


async def test_late_teacher_falls_back_or_masks():
    class Slow:
        id, vocab_id = "slow", "v1"

        async def score(self, pairs):
            await asyncio.sleep(5)

    rep = await _runtime(TeacherPool([Slow()], default="slow"), teacher_deadline_s=0.01).run()
    assert len(rep.steps) == 4 and all(r.opd_fallback and r.teacher_id is None for r in rep.trained_records)
    rep = await _runtime(TeacherPool([Slow()], default="slow"), teacher_deadline_s=0.01, teacher_on_miss="mask", max_masked_share=1.0, breaker_min=10**6,
                         max_wall_s=2.0).run()
    assert rep.masked_rollouts > 0 and not rep.trained_records and rep.stop_reason == "wall_clock"  # nothing trainable: never trained as 0


async def test_vllm_teacher_reads_prompt_logprobs():
    import httpx

    def handler(req):
        body = __import__("json").loads(req.content)
        ids = body["prompt"]
        pl = [None] + [{str(t): {"logprob": -0.1 * (i + 1), "rank": 1}} for i, t in enumerate(ids[1:])]
        return httpx.Response(200, json={"choices": [{"text": "", "prompt_logprobs": pl}]})

    t = VLLMTeacher("http://teacher:8000/v1", "m", id="t", vocab_id="v", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    rows = await t.score([([1, 2, 3], [4, 5])])
    assert rows == [pytest.approx([-0.3, -0.4])]  # positions 3 and 4 of prompt + completion
