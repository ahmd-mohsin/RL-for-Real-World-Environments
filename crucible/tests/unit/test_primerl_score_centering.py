"""Score centering composed with masked importance sampling on the DeltaAI bridge (swarm-pipeline/14 §7): the loss math
against full centering and the reference code (martin-marek/score-centering, MIT), the drift it removes, the top-k path
from vLLM's response to prime-rl's wire, and the trainer-side LM-head hook. CPU, small vocabularies."""

from __future__ import annotations

import asyncio
import base64
import importlib
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from crucible.train.bridges.primerl import deploy, sc_head, smoke, topk, wire
from crucible.train.bridges.primerl.backend import PrimeRLBackend
from crucible.train.bridges.primerl.loss import Head, crucible_loss, mass, token_objective
from crucible.train.bridges.primerl.topk_server import pack_topk
from crucible.train.bridges.tinker.trainer import TrainConfig

MODEL = "Qwen/Qwen3.6-35B-A3B"
F64 = torch.float64
WEIGHT_OF = {"sc": "one", "sc+tis": "tis", "sc+mis": "mis"}


def _w(r: torch.Tensor, weight: str, low: float = 0.5, high: float = 5.0) -> torch.Tensor:
    return {"one": torch.ones_like(r), "is": r, "tis": r.clamp(max=2.0), "mis": torch.where((r >= low) & (r <= high), r, torch.zeros_like(r))}[weight]


def reference_sc_loss(train_logp, samp_logp, topk_ids, sampled_token, samp_token_logp, advantage, weight_fn=torch.ones_like, eps=1e-6):
    """The reference implementation's PyTorch ``score_centering_loss`` (martin-marek/score-centering README, MIT,
    64842967), verbatim but for the name: the oracle the bridge's loss is checked against."""
    head_logp = train_logp.gather(-1, topk_ids)
    token_logp = train_logp.gather(-1, sampled_token[..., None])[..., 0]
    p_head, q_head = head_logp.exp(), samp_logp.exp()
    p_tail = (1 - p_head.sum(-1)).clamp_min(eps)
    q_tail = (1 - q_head.sum(-1)).clamp_min(eps)
    rho = q_tail / p_tail
    alpha = rho * weight_fn(1 / rho)
    head_weight = weight_fn((head_logp - samp_logp).exp())
    residual = q_head * head_weight - alpha[..., None] * p_head
    correction = (residual.detach() * head_logp).sum(-1)
    token_weight = weight_fn((token_logp - samp_token_logp).exp())
    return -advantage.detach() * (token_weight.detach() * token_logp - correction)


def _case(T=5, V=7, seed=0, noise=0.4):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(T, V, generator=g, dtype=F64)
    lq_full = torch.log_softmax(z + noise * torch.randn(T, V, generator=g, dtype=F64), -1)
    y = torch.multinomial(lq_full.exp(), 1, generator=g).squeeze(-1)
    adv = torch.randn(T, generator=g, dtype=F64)
    return z, lq_full, y, adv


def _ours(z, lq_full, y, adv, ids, pg, **kw):
    z = z.clone().requires_grad_(True)
    lp_full = torch.log_softmax(z, -1)
    head = Head(lp=lp_full.gather(-1, ids), lq=lq_full.gather(-1, ids), valid=torch.ones_like(ids, dtype=torch.bool),
                present=torch.ones(z.shape[0], dtype=torch.bool))
    loss, m = token_objective(lp_full.gather(-1, y[:, None])[:, 0], lq_full.gather(-1, y[:, None])[:, 0], adv, torch.ones_like(adv, dtype=torch.bool),
                              pg_correction=pg, head=head, **kw)
    (g,) = torch.autograd.grad(loss, z)
    return g, loss, m


def _head_ids(lq_full, k):
    return lq_full.topk(k, dim=-1).indices


# ---- the loss: exactness ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pg", ["sc", "sc+tis", "sc+mis"])
def test_with_k_equal_to_the_vocabulary_the_gradient_is_full_score_centering(pg) -> None:
    z, lq_full, y, adv = _case()
    g, _, m = _ours(z, lq_full, y, adv, _head_ids(lq_full, z.shape[1]), pg)
    zf = z.clone().requires_grad_(True)
    lp = torch.log_softmax(zf, -1)
    weight = WEIGHT_OF[pg]
    lp_y, lq_y = lp.gather(-1, y[:, None])[:, 0], lq_full.gather(-1, y[:, None])[:, 0]
    w_y = _w(torch.exp(lp_y - lq_y).detach(), weight)
    centering_mass = lq_full.exp() * _w(torch.exp(lp - lq_full).detach(), weight)  # q_v·w(p_v/q_v) over the whole vocabulary
    full = -(adv * (w_y * lp_y - (centering_mass.detach() * lp).sum(-1))).sum()
    (gf,) = torch.autograd.grad(full, zf)
    assert torch.allclose(g, gf, atol=1e-9) and m["sc/head_mass"].item() == pytest.approx(1.0)


@pytest.mark.parametrize("pg", ["sc", "sc+tis", "sc+mis"])
def test_the_gradient_matches_the_reference_implementation_with_a_top_k_head(pg) -> None:
    z, lq_full, y, adv = _case(T=6, V=9, seed=3)
    ids = _head_ids(lq_full, 3)
    g, _, _ = _ours(z, lq_full, y, adv, ids, pg)
    zr = z.clone().requires_grad_(True)
    weight = WEIGHT_OF[pg]
    ref = reference_sc_loss(torch.log_softmax(zr, -1), lq_full.gather(-1, ids), ids, y, lq_full.gather(-1, y[:, None])[:, 0], adv,
                            weight_fn=lambda r: _w(r, weight)).sum()
    (gr,) = torch.autograd.grad(ref, zr)
    assert torch.allclose(g, gr, atol=1e-10)


def test_exact_importance_weights_make_the_centering_term_vanish() -> None:
    z, lq_full, _, _ = _case(T=4, V=8, seed=1, noise=0.2)
    ids = _head_ids(lq_full, 3)
    lp = torch.log_softmax(z.clone().requires_grad_(True), -1)
    from crucible.train.bridges.primerl.loss import centering

    head = Head(lp=lp.gather(-1, ids), lq=lq_full.gather(-1, ids), valid=torch.ones_like(ids, dtype=torch.bool), present=torch.ones(4, dtype=torch.bool))
    term, _ = centering(head, torch.ones(4, dtype=torch.bool), "is")  # w = r: q·w = p on the head, α = ρ·(1/ρ) = 1
    (g,) = torch.autograd.grad(term.sum(), head.lp)
    assert torch.count_nonzero(g) == 0
    # MIS is w = r inside its band: with every ratio and the tail inside it, the term vanishes as well
    term2, m = centering(head, torch.ones(4, dtype=torch.bool), "mis")
    (g2,) = torch.autograd.grad(term2.sum(), head.lp)
    assert torch.count_nonzero(g2) == 0 and m["sc/tail_dropped"].item() == 0.0


@pytest.mark.parametrize("pg", ["sc", "sc+mis", "sc+tis"])
def test_on_policy_score_centering_is_plain_policy_gradient(pg) -> None:
    z, _, y, adv = _case(T=5, V=10, seed=2)
    lq_full = torch.log_softmax(z, -1)  # q = p
    g, loss, m = _ours(z, lq_full, y, adv, _head_ids(lq_full, 4), pg)
    zp = z.clone().requires_grad_(True)
    pg_loss = -(adv * torch.log_softmax(zp, -1).gather(-1, y[:, None])[:, 0]).sum()
    (gp,) = torch.autograd.grad(pg_loss, zp)
    assert torch.allclose(g, gp, atol=1e-12) and loss.item() == pytest.approx(float(-adv.sum()))  # value: the IS surrogate's, −A
    assert m["sc/rho"].item() == pytest.approx(1.0) and m["sc/score_mean_l1"].item() == pytest.approx(0.0, abs=1e-12)


# ---- the drift ------------------------------------------------------------------------------------------------------------------------
def _expected_grad(logits, lq_full, pg, k, **band):
    """E_{y~q}[∇ loss(y)] for one position with a constant advantage A = 1, by enumerating the vocabulary."""
    V = logits.numel()
    ids = _head_ids(lq_full[None], k)
    total = torch.zeros(V, dtype=F64)
    for y in range(V):
        z = logits.clone()[None].requires_grad_(True)
        lp = torch.log_softmax(z, -1)
        head = Head(lp=lp.gather(-1, ids), lq=lq_full[None].gather(-1, ids), valid=torch.ones_like(ids, dtype=torch.bool), present=torch.ones(1, dtype=torch.bool))
        loss, _ = token_objective(lp[:, y], lq_full[None, y], torch.ones(1, dtype=F64), torch.ones(1, dtype=torch.bool), pg_correction=pg,
                                  head=head if pg.startswith("sc") else None, **band)
        (g,) = torch.autograd.grad(loss, z)
        total += lq_full[y].exp() * g[0]
    return total


def test_with_a_mismatched_sampler_the_centered_expected_gradient_is_zero_and_plain_pg_drifts() -> None:
    g = torch.Generator().manual_seed(7)
    logits = torch.randn(6, generator=g, dtype=F64)
    lq_full = torch.log_softmax(logits + 0.5 * torch.randn(6, generator=g, dtype=F64), -1)
    for pg in ("sc", "sc+tis", "sc+mis"):
        for k in (6, 5):  # the whole vocabulary, and a one-token tail (where ρ·p is exact)
            assert _expected_grad(logits, lq_full, pg, k).abs().max() < 1e-9, (pg, k)
    # plain PG (w = 1, no centering) drifts: E_q[∇log p] = q − p in logits; an IS band that drops tokens drifts too
    plain = -(lq_full.exp() - torch.softmax(logits, -1))
    assert plain.abs().max() > 1e-2 and _expected_grad(logits, lq_full, "mis", 6, mis_low=0.9, mis_high=1.1).abs().max() > 1e-3
    # an approximate tail (k = 2 of 6): not exactly zero, but far below the plain drift
    assert _expected_grad(logits, lq_full, "sc", 2).abs().max() < 0.5 * plain.abs().max()
    # the k1 anchor is a deliberate pull toward q: with A = 0 its expected gradient is β·∇KL(q‖p) = β(p − q) in logits
    z = logits.clone().requires_grad_(True)
    exp_anchor = torch.zeros(6, dtype=F64)
    for y in range(6):
        lp = torch.log_softmax(z, -1)
        loss, _ = token_objective(lp[y:y + 1], lq_full[y:y + 1], torch.zeros(1, dtype=F64), torch.ones(1, dtype=torch.bool), kl_coeff=0.1, pg_correction="mis")
        (gy,) = torch.autograd.grad(loss, z)
        exp_anchor += lq_full[y].exp() * gy
    assert torch.allclose(exp_anchor, 0.1 * (torch.softmax(logits, -1) - lq_full.exp()), atol=1e-12)


# a port of the reference repository's tests/test_loss.py (MIT): the same logits, sampler and corrections
REF_TRAINER_LOGITS = torch.tensor([0.5, -0.2, 1.1], dtype=F64)
REF_SAMPLER_LOGP = torch.log_softmax(REF_TRAINER_LOGITS + torch.tensor([0.1, 0.3, -0.2], dtype=F64), -1)
REF_CORRECTIONS = {"pg": ("sc", {}), "tis": ("sc+tis", {"tis_cap": 2.0}), "mis": ("sc+mis", {"mis_low": 0.8, "mis_high": 1.1})}  # this band drops tokens


@pytest.mark.parametrize("center", ["full", "topk"])
@pytest.mark.parametrize("name", list(REF_CORRECTIONS))
def test_reference_score_centering_cancels_drift(name, center) -> None:
    pg, band = REF_CORRECTIONS[name]
    k = 3 if center == "full" else 2  # k = V - 1: the tail is one token, so modeling it as ρ·p is exact
    assert _expected_grad(REF_TRAINER_LOGITS, REF_SAMPLER_LOGP, pg, k, **band).abs().max() < 1e-6


def test_reference_drift_without_centering() -> None:  # the test can fail: plain REINFORCE does drift
    lp = torch.log_softmax(REF_TRAINER_LOGITS, -1)
    total = torch.zeros(3, dtype=F64)
    for y in range(3):
        z = REF_TRAINER_LOGITS.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(-torch.log_softmax(z, -1)[y], z)
        total += REF_SAMPLER_LOGP[y].exp() * g
    assert total.abs().max() > 1e-3 and not torch.allclose(lp.exp(), REF_SAMPLER_LOGP.exp())


# ---- head, tail, MIS edges ------------------------------------------------------------------------------------------------------------
def test_the_sampled_token_outside_the_head_keeps_its_own_logprob_and_only_the_tail_mass_matters() -> None:
    V, k = 8, 3
    g = torch.Generator().manual_seed(11)
    z = torch.randn(1, V, generator=g, dtype=F64)
    lq_full = torch.log_softmax(z + 0.3 * torch.randn(1, V, generator=g, dtype=F64), -1)
    ids = _head_ids(lq_full, k)
    y = torch.tensor([int(lq_full[0].argsort()[0])])  # the least likely token: outside the head
    assert int(y) not in ids[0].tolist()
    adv = torch.tensor([1.5], dtype=F64)
    gr, _, m = _ours(z, lq_full, y, adv, ids, "sc+mis")
    zr = z.clone().requires_grad_(True)
    ref = reference_sc_loss(torch.log_softmax(zr, -1), lq_full.gather(-1, ids), ids, y, lq_full[:, int(y)], adv, weight_fn=lambda r: _w(r, "mis")).sum()
    assert torch.allclose(gr, torch.autograd.grad(ref, zr)[0], atol=1e-12)
    lp_y, lq_y = torch.log_softmax(z, -1)[0, y], lq_full[0, y]
    if 0.5 <= float(torch.exp(lp_y - lq_y)) <= 5.0:
        assert gr[0, y].item() < 0  # the sampled token is raised (A > 0), although it is in no head
    # moving sampler mass between tail tokens (head and tail mass unchanged) does not change the loss
    q = lq_full.exp().clone()
    tail = [v for v in range(V) if v not in ids[0].tolist()]
    q[0, tail] = q[0, tail].sum() / len(tail)
    zz = z.clone().requires_grad_(True)
    lp = torch.log_softmax(zz, -1)
    head = Head(lp=lp.gather(-1, ids), lq=lq_full.gather(-1, ids), valid=torch.ones_like(ids, dtype=torch.bool), present=torch.ones(1, dtype=torch.bool))
    loss_a, _ = token_objective(lp[:, int(y)], lq_full[:, int(y)], adv, torch.ones(1, dtype=torch.bool), pg_correction="sc", head=head)
    ga = torch.autograd.grad(loss_a, zz, retain_graph=True)[0]
    head_b = Head(lp=lp.gather(-1, ids), lq=q.log().gather(-1, ids), valid=head.valid, present=head.present)
    loss_b, _ = token_objective(lp[:, int(y)], lq_full[:, int(y)], adv, torch.ones(1, dtype=torch.bool), pg_correction="sc", head=head_b)
    assert torch.allclose(ga, torch.autograd.grad(loss_b, zz)[0], atol=1e-14) and m["sc/head_mass"].item() < 1.0


def test_the_mis_band_keeps_its_edges_and_drops_just_outside() -> None:
    f32 = torch.float32
    lo, hi = torch.tensor(0.5, dtype=f32).log(), torch.tensor(5.0, dtype=f32).log()
    lp = torch.stack([lo, hi, torch.nextafter(lo, torch.tensor(-1.0)), torch.nextafter(hi, torch.tensor(9.0)), torch.tensor(0.0)]).requires_grad_(True)
    lq = torch.zeros(5, dtype=f32)  # log r = log p exactly
    w = mass(lp.detach(), lq, "mis")
    assert w[0].item() == pytest.approx(0.5) and w[1].item() == pytest.approx(5.0) and w[2].item() == 0.0 and w[3].item() == 0.0 and w[4].item() == 1.0
    loss, m = token_objective(lp, lq, torch.ones(5, dtype=f32), torch.ones(5, dtype=torch.bool), pg_correction="mis")
    loss.backward()
    assert lp.grad.tolist() == pytest.approx([-0.5, -5.0, 0.0, 0.0, -1.0]) and m["is/masked_frac"].item() == pytest.approx(0.4)
    assert mass(torch.tensor([3.0]).log(), torch.zeros(1), "tis").item() == pytest.approx(2.0)


# ---- missing heads: all or none, and the fallback ---------------------------------------------------------------------------------
def test_without_heads_a_centered_step_falls_back_to_mis_and_a_partial_head_is_refused() -> None:
    lq = torch.tensor([-1.0, -1.0, -2.0], dtype=F64)
    lp = torch.tensor([-0.2, -1.0, -2.1], dtype=F64, requires_grad=True)  # r = e^0.8 = 2.2, 1, 0.9
    adv, mask = torch.tensor([1.0, 2.0, -1.0], dtype=F64), torch.ones(3, dtype=torch.bool)
    inp = SimpleNamespace(trainer_logprobs=lp, inference_logprobs=lq, advantages=adv, loss_mask=mask, loss_weights=None, ref_logprobs=None)
    out = crucible_loss(inp, kind="importance_sampling")  # default sc+mis; no hook here, so no heads
    mis, _ = token_objective(lp, lq, adv, mask, pg_correction="mis")
    assert out.metrics["sc/fallback"].item() == 1.0 and out.loss.item() == pytest.approx(mis.item())
    tis_out = crucible_loss(inp, kind="importance_sampling", pg_correction="sc+tis")
    tis, _ = token_objective(lp, lq, adv, mask, pg_correction="tis")
    assert tis_out.loss.item() == pytest.approx(tis.item()) and abs(tis.item() - mis.item()) > 0.1  # r = 2.2: TIS caps it, MIS keeps it
    partial = Head(lp=torch.zeros(3, 2, dtype=F64), lq=torch.zeros(3, 2, dtype=F64), valid=torch.ones(3, 2, dtype=torch.bool),
                   present=torch.tensor([True, False, True]))
    with pytest.raises(ValueError, match="all or none"):
        token_objective(lp, lq, adv, mask, pg_correction="sc+mis", head=partial)
    with pytest.raises(ValueError, match="no PPO clip"):
        token_objective(lp, lq, adv, mask, kind="cispo", pg_correction="sc+mis")
    # the bridge's rule: one sequence short of full coverage sends no heads for any
    rows = [{"sc_heads": {"ids": b"", "counts": b""}, "sc_coverage": 1.0}, {"sc_heads": {"ids": b"", "counts": b""}, "sc_coverage": 0.5}]
    assert topk.all_or_none(rows) == [None, None] and topk.all_or_none(rows[:1]) == [rows[0]["sc_heads"]]
    assert topk.all_or_none([{"sc_coverage": 1.0}]) == [None]


# ---- the top-k path: vLLM response -> store -> stitched wire payload ----------------------------------------------------------------
def _arr(a):
    a = np.ascontiguousarray(a)
    return {"data": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape), "dtype": a.dtype.name}


def test_the_server_packs_entries_one_to_k_of_flat_logprobs() -> None:
    # position 0: sampled token 7 outside the head [3, 4]; position 1: sampled 3 inside it (vLLM repeats it); position 2: one entry short
    flat = SimpleNamespace(start_indices=[0, 3, 6], end_indices=[3, 6, 8], token_ids=[7, 3, 4, 3, 3, 4, 5, 6],
                           logprobs=[-3.0, -0.5, -1.0, -0.4, -0.4, -1.2, -0.1, float("-inf")])
    p = pack_topk(flat, 2)
    assert p["ids"].tolist() == [[3, 4], [3, 4], [6, -1]] and p["logprobs"][0].tolist() == [-0.5, -1.0] and p["logprobs"][2, 0] == -9999.0
    regular = SimpleNamespace(start_indices=[0, 3], end_indices=[3, 6], token_ids=[9, 1, 2, 1, 1, 2], logprobs=[-2, -0.1, -0.3, -0.1, -0.1, -0.3])
    assert pack_topk(regular, 2)["ids"].tolist() == [[1, 2], [1, 2]] and pack_topk(regular, 0) is None


def test_heads_stitch_per_sequence_and_round_trip_through_prime_rls_sampling_mask_bytes() -> None:
    store = topk.TopKStore(k=2)
    full = list(range(20, 30))  # call 1: prompt 20-22, completion 23-25 (ends 6); call 2: prompt 0-7, completion 28-29 (ends 10)
    store.put(full[:6], topk.Heads(ids=np.array([[1, 2], [3, 4], [5, 6]], np.int32), logprobs=np.full((3, 2), -0.5, np.float32), start=3))
    store.put(full[:10], topk.Heads(ids=np.array([[7, 8], [9, 10]], np.int32), logprobs=np.array([[-0.1, -2.0], [-0.2, -3.0]], np.float32), start=8))
    mask = [False] * 3 + [True] * 3 + [False] * 2 + [True] * 2
    ids, lps, present, cov = topk.stitch(full, [6, 10], mask, store)
    assert cov == 1.0 and present.tolist() == mask and ids[8].tolist() == [7, 8] and ids[0].tolist() == [-1, -1]
    pay = topk.encode_payload(ids, lps, present)
    i2, l2, p2 = topk.decode_payload(pay["ids"], pay["counts"])
    assert (i2 == np.where(present[:, None], ids, -1)).all() and np.allclose(l2[present], lps[present]) and (p2 == present).all()
    _, _, _, cov2 = topk.stitch(full, [6, 10], mask, topk.TopKStore(k=2))
    assert cov2 == 0.0
    with pytest.raises(ValueError):
        store.put([1], topk.Heads(ids=np.zeros((1, 3), np.int32), logprobs=np.zeros((1, 3), np.float32), start=0))


class TopKHttp:
    def __init__(self, k=2, missing_after=None):
        self.posts, self.k, self.n, self.missing_after = [], k, 0, missing_after

    async def post(self, path, body, *, timeout_s):
        self.posts.append((path, body))
        if path != "/inference/v1/generate":
            return {}
        self.n += 1
        ch = {"token_ids": [50, 51], "completion_logprobs": _arr(np.array([-0.1, -0.2], np.float32)), "finish_reason": "stop"}
        if self.missing_after is None or self.n <= self.missing_after:
            ch |= {"topk_ids": _arr(np.array([[50, 9], [51, 9]], np.int32)), "topk_logprobs": _arr(np.array([[-0.1, -3.0], [-0.2, -3.0]], np.float32))}
        return {"choices": [ch]}


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

    def metrics(self, step):  # the two rows prime-rl's trainer logs per step (backend.TRAINER_METRIC_KEYS)
        return {"primerl/loss/mean": 0.0, "primerl/optim/grad_norm": 0.0}

    def checkpoint(self, step):
        return self.root / "checkpoints" / f"step_{step}"


@pytest.mark.parametrize("missing", [False, True])
def test_the_sampler_asks_for_k_logprobs_keeps_the_heads_and_a_step_sends_all_or_none(tmp_path: Path, missing: bool) -> None:
    store = topk.TopKStore(k=2)
    b = PrimeRLBackend(io=FakeIO(tmp_path), http=TopKHttp(missing_after=1 if missing else None), base_model=MODEL, learning_rate=4e-5, topk=store)

    async def go():
        tr = await b.trainer(MODEL, rank=32)
        s = await b.sampler(MODEL, None)
        outs = [await s.sample(p, max_tokens=8, temperature=1.0, stop=None, seed=1) for p in ([1, 2, 3], [4, 5])]
        path, body = b.http.posts[-1]
        assert path == "/inference/v1/generate" and body["sampling_params"]["logprobs"] == 2 and outs[0].logprobs == pytest.approx([-0.1, -0.2])
        ds = [smoke.datum(p, o.tokens, o.logprobs, 1.0, rollout=f"r{i}", segment=0) for i, (p, o) in enumerate(zip(([1, 2, 3], [4, 5]), outs, strict=True))]
        fb = await tr.forward_backward(ds, loss_fn="importance_sampling", config=None)
        rows = b.io.sent[0][1]
        sent = topk.all_or_none(rows)
        if missing:
            assert rows[1]["sc_coverage"] == 0.0 and sent == [None, None] and fb.metrics["sc/fallback"] == 1.0 and fb.metrics["sc/centered_share"] == 0.0
        else:
            assert fb.metrics["sc/coverage_min"] == 1.0 and fb.metrics["sc/fallback"] == 0.0 and fb.metrics["sc/fallback_steps_share"] == 0.0
            ids, lps, present = topk.decode_payload(sent[0]["ids"], sent[0]["counts"])
            assert present.tolist() == [False, False, False, True, True] and ids[3].tolist() == [50, 9] and lps[4].tolist() == pytest.approx([-0.2, -3.0])
        assert store.get([1, 2, 3, 50, 51]) is None and b.last[1].name == "step_1"  # trained: dropped; the broadcast path is the sampler's
        grid = wire.simple_pack(rows, ranks=1)
        assert isinstance(grid[0][0]["sc_heads"]["ids"], str)

    asyncio.run(go())


# ---- the trainer-side hook: prime-rl's chunked LM head, emulated on the CPU -------------------------------------------------------
def test_the_chunked_head_function_matches_a_full_softmax_in_value_and_gradient() -> None:
    g = torch.Generator().manual_seed(5)
    N, H, V, k = 7, 6, 11, 3
    hidden = torch.randn(N, H, generator=g, dtype=F64, requires_grad=True)
    weight = torch.randn(V, H, generator=g, dtype=F64, requires_grad=True)
    labels = torch.randint(0, V, (N,), generator=g)
    inv_t = 1.0 / (0.7 + torch.rand(N, generator=g, dtype=F64))
    ids = torch.randint(0, V, (N, k), generator=g)
    ids[2] = -1  # a position without a head
    ids[4, 1] = -1
    lp, ent, hl = sc_head.HeadLogProbFn.apply(hidden, weight, labels, inv_t, 3, ids, 4)  # 3 tokens x 4 vocab entries per chunk
    g_lp, g_hl = torch.randn(N, generator=g, dtype=F64), torch.randn(N, k, generator=g, dtype=F64)
    gh, gw = torch.autograd.grad((lp * g_lp).sum() + (hl * g_hl).sum(), (hidden, weight))
    h2, w2 = hidden.detach().clone().requires_grad_(True), weight.detach().clone().requires_grad_(True)
    full = torch.log_softmax((h2 @ w2.t()) * inv_t[:, None], -1)
    lp2 = full.gather(-1, labels[:, None])[:, 0]
    hl2 = torch.where(ids >= 0, full.gather(-1, ids.clamp(min=0)), torch.zeros(N, k, dtype=F64))
    gh2, gw2 = torch.autograd.grad((lp2 * g_lp).sum() + (hl2 * g_hl).sum(), (h2, w2))
    ent2 = -(full.exp() * full).sum(-1)
    assert torch.allclose(lp.double(), lp2, atol=1e-5) and torch.allclose(hl.double(), hl2, atol=1e-5) and torch.allclose(ent.double(), ent2, atol=1e-5)
    assert torch.allclose(gh, gh2, atol=1e-5) and torch.allclose(gw, gw2, atol=1e-5)


def _shift_left(t, pad):
    return torch.cat([t[:, 1:], torch.full_like(t[:, :1], pad)], dim=1)


def _shift_right(t, pad):
    return torch.cat([torch.full_like(t[:, :1], pad), t[:, :-1]], dim=1)


def test_the_hook_delivers_each_sequences_head_to_the_loss_as_prime_rls_train_loop_does() -> None:
    """prime-rl's micro-batch path, emulated: the packed payload as its data loader pads it, shifted to the labels, the
    fused head's forward, logprobs shifted back and split per sequence, ``compute_loss`` calling the custom loss per
    sequence. The hook's loss equals the loss computed from a full softmax with the same heads."""
    g = torch.Generator().manual_seed(9)
    H, V, k = 5, 13, 3
    weight = torch.randn(V, H, generator=g, dtype=torch.float32, requires_grad=True)
    lm_head = SimpleNamespace(weight=weight, chunk_size=4)
    # two sequences packed into one micro batch: lengths 6 and 5; trained (sampled) positions 3-5 and 8-10
    lengths, T = [6, 5], 11
    input_ids = torch.randint(0, V, (1, T), generator=g)
    mask = torch.zeros(T, dtype=torch.bool)
    mask[[3, 4, 5, 8, 9, 10]] = True
    hidden = torch.randn(1, T, H, generator=g, requires_grad=True)
    q = torch.log_softmax(torch.randn(T, V, generator=g), -1)  # the sampler at each position (row j sampled token j from row j-1's state)
    heads_ids = np.full((T, k), -1, np.int32)
    heads_lp = np.full((T, k), -np.inf, np.float32)
    for j in mask.nonzero()[:, 0].tolist():
        top = q[j - 1].topk(k)
        heads_ids[j], heads_lp[j] = top.indices.numpy(), top.values.numpy()
    pay = topk.encode_payload(heads_ids, heads_lp, mask.numpy())
    counts = np.frombuffer(pay["counts"], np.int32)
    padded = np.full((T, int(counts.max())), -1, np.int32)
    padded[np.arange(padded.shape[1])[None, :] < counts[:, None]] = np.frombuffer(pay["ids"], np.int32)  # prime-rl data.py
    sm = _shift_left(torch.from_numpy(padded).unsqueeze(0), -1)
    labels = _shift_left(input_ids, 0)
    temperature = torch.ones(1, T)
    payload = sc_head.decode_payload(sm)
    out = sc_head.head_forward(lm_head, hidden, labels, temperature, payload)
    lp_tok = _shift_right(out["logprobs"], math.log(1.0 / V))
    lq_tok = torch.tensor([float(q[j - 1, input_ids[0, j]]) if j else 0.0 for j in range(T)])
    adv = torch.randn(T, generator=g)
    seen = []

    def prime_compute_loss(*, trainer_logprobs, inference_logprobs, advantages, loss_mask, **_):
        total = trainer_logprobs[0].sum() * 0.0
        for t, i, a, m in zip(trainer_logprobs, inference_logprobs, advantages, loss_mask, strict=True):
            r = crucible_loss(SimpleNamespace(trainer_logprobs=t, inference_logprobs=i, advantages=a, loss_mask=m, loss_weights=None, ref_logprobs=None),
                              kind="importance_sampling", pg_correction="sc+mis")
            seen.append(r.metrics)
            total = total + r.loss
        return total, {}

    wrapped = sc_head.compute_loss_wrapper(prime_compute_loss)
    loss, _ = wrapped(trainer_logprobs=lp_tok[0].split(lengths), inference_logprobs=lq_tok.split(lengths), advantages=adv.split(lengths),
                      loss_mask=mask.split(lengths))
    gw, ghid = torch.autograd.grad(loss, (weight, hidden))
    assert all(s["sc/fallback"].item() == 0.0 for s in seen) and seen[0]["sc/align_err"].item() < 1e-5 and sc_head.CTX.by_sample == {}
    # the same loss from a full softmax and the stitched heads, per sequence
    w2, h2 = weight.detach().clone().requires_grad_(True), hidden.detach().clone().requires_grad_(True)
    full = torch.log_softmax(h2[0] @ w2.t(), -1)  # row i predicts token i + 1
    full_tok = torch.cat([torch.zeros(1, V), full[:-1]])  # row j: the distribution of token j
    ids_t = torch.from_numpy(heads_ids).long()
    total = 0.0
    for lo, hi in ((0, 6), (6, 11)):
        sl = slice(lo, hi)
        head = Head(lp=torch.where(ids_t[sl] >= 0, full_tok[sl].gather(-1, ids_t[sl].clamp(min=0)), torch.zeros(hi - lo, k)),
                    lq=torch.from_numpy(heads_lp[sl]), valid=ids_t[sl] >= 0, present=mask[sl])
        l2, _ = token_objective(full_tok[sl].gather(-1, input_ids[0, sl, None])[:, 0], lq_tok[sl], adv[sl], mask[sl], pg_correction="sc+mis", head=head)
        total = total + l2
    gw2, gh2 = torch.autograd.grad(total, (w2, h2))
    assert torch.allclose(loss, total, atol=1e-5) and torch.allclose(gw, gw2, atol=1e-4) and torch.allclose(ghid, gh2, atol=1e-4)
    # logprobs are the full-vocabulary ones: never renormalized over the head
    assert torch.allclose(out["logprobs"][0], full.gather(-1, labels[0, :, None])[:, 0].detach(), atol=1e-5)


def test_a_genuine_sampling_mask_is_not_read_as_heads_and_a_mixed_one_is_refused() -> None:
    assert sc_head.decode_payload(torch.tensor([[[5, 6, -1], [-1, -1, -1]]], dtype=torch.int32)) is None
    with pytest.raises(ValueError, match="mixes"):
        sc_head.decode_payload(torch.tensor([[[topk.SC_MAGIC, 1, 0], [5, 6, -1]]], dtype=torch.int32))


# ---- deployment and CLI ---------------------------------------------------------------------------------------------------------------
def test_score_centering_with_mis_is_the_default_and_vllm_serves_the_head() -> None:
    cfg = TrainConfig(model=MODEL, require_band=False)
    d = deploy.DeltaConfig()
    assert d.pg_correction == "sc+mis" and d.sc_topk == 32 and d.centered
    t = deploy.trainer_toml(cfg, d, Path("/w"))
    assert t["loss"]["kwargs"]["pg_correction"] == "sc+mis" and t["model"]["fused_lm_head_token_chunk_size"] == deploy.LM_HEAD_CHUNK
    inf = deploy.inference_toml(cfg, d)["vllm"]
    assert inf["max_logprobs"] == 32 and inf["middleware"] == [deploy.MIDDLEWARE]
    mod, _, cls = deploy.MIDDLEWARE.rpartition(".")
    assert hasattr(importlib.import_module(mod), cls)
    off = deploy.DeltaConfig(pg_correction="mis")
    assert "middleware" not in deploy.inference_toml(cfg, off)["vllm"] and deploy.trainer_toml(cfg, off, Path("/w"))["loss"]["kwargs"]["pg_correction"] == "mis"
    with pytest.raises(ValueError, match="no PPO clip"):
        deploy.trainer_toml(TrainConfig(model=MODEL, require_band=False, loss_fn="cispo"), d, Path("/w"))
    assert deploy.trainer_toml(TrainConfig(model=MODEL, require_band=False, loss_fn="cispo"), deploy.DeltaConfig(pg_correction="none"), Path("/w"))
    with pytest.raises(ValueError):
        deploy.DeltaConfig(pg_correction="ppo+sc")


# ---- the patches themselves, against stand-ins with prime-rl's v0.10.0 names -----------------------------------------------------
def _module(name, **attrs):
    import types

    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


def test_install_patches_the_fused_head_and_the_train_modules_loss_and_refuses_the_vanilla_head(monkeypatch) -> None:
    import sys

    from crucible.train.bridges.primerl import loss as cl

    class Fused:  # prime_rl.trainer.models.layers.lm_head.FusedOutputLinear's interface
        def __init__(self, weight):
            self.weight, self.chunk_size = weight, 4

        def forward(self, hidden_states, labels=None, temperature=None, sampling_mask=None):
            return {"logprobs": "stock", "entropy": None}

    class Vanilla:
        def forward(self, hidden_states, labels=None, temperature=None, sampling_mask=None):
            return {"logits": "stock"}

    calls = []

    def compute_loss(**kw):
        calls.append({id(t): sc_head.lookup(t) for t in kw["trainer_logprobs"]})
        return "loss", {}

    lmh = _module("prime_rl.trainer.models.layers.lm_head", FusedOutputLinear=Fused, VanillaOutputLinear=Vanilla, PrimeLmOutput=dict)
    prl_loss = _module("prime_rl.trainer.rl.loss", compute_loss=compute_loss)
    train = _module("prime_rl.trainer.rl.train", compute_loss=compute_loss, gather_for_cp=lambda t, g: t, setup_rl_loss_fn=lambda c: None)
    for name, mod in (("prime_rl", _module("prime_rl")), ("prime_rl.trainer", _module("prime_rl.trainer")), ("prime_rl.trainer.models", _module("prime_rl.trainer.models")),
                      ("prime_rl.trainer.models.layers", _module("prime_rl.trainer.models.layers", lm_head=lmh)), ("prime_rl.trainer.models.layers.lm_head", lmh),
                      ("prime_rl.trainer.rl", _module("prime_rl.trainer.rl", loss=prl_loss)), ("prime_rl.trainer.rl.loss", prl_loss), ("prime_rl.trainer.rl.train", train)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(sc_head.CTX, "installed", False)
    cl._install_trainer_hook()  # what importing the loss in prime-rl's trainer does
    assert sc_head.CTX.installed and train.compute_loss is not compute_loss and getattr(train.compute_loss, "__crucible_sc__", False)
    g = torch.Generator().manual_seed(0)
    head = Fused(torch.randn(9, 4, generator=g))
    hidden, labels, temp = torch.randn(1, 3, 4, generator=g), torch.tensor([[1, 2, 3]]), torch.ones(1, 3)
    assert head.forward(hidden, labels, temperature=temp, sampling_mask=None)["logprobs"] == "stock"  # no payload: prime-rl's own path
    assert head.forward(hidden, labels, temperature=temp, sampling_mask=torch.tensor([[[4, -1], [-1, -1], [-1, -1]]], dtype=torch.int32))["logprobs"] == "stock"
    pay = topk.encode_payload(np.array([[1, 5], [2, 6], [-1, -1]], np.int32), np.full((3, 2), -1.0, np.float32), np.array([True, True, False]))
    counts = np.frombuffer(pay["counts"], np.int32)
    padded = np.full((3, int(counts.max())), -1, np.int32)
    padded[np.arange(padded.shape[1])[None, :] < counts[:, None]] = np.frombuffer(pay["ids"], np.int32)
    sm = torch.from_numpy(padded).unsqueeze(0)
    out = head.forward(hidden, labels, temperature=temp, sampling_mask=sm)
    assert torch.is_tensor(out["logprobs"]) and sc_head.CTX.pending is not None
    with pytest.raises(RuntimeError, match="fused LM head"):
        Vanilla().forward(hidden, labels, temperature=temp, sampling_mask=sm)
    lps = list(torch.zeros(3).split([2, 1]))
    assert train.compute_loss(trainer_logprobs=lps) == ("loss", {})
    assert [h is not None for h in calls[0].values()] == [True, True] and sc_head.CTX.pending is None and sc_head.CTX.by_sample == {}


def test_the_server_patch_adds_the_head_to_prime_rls_packed_generate_response(monkeypatch) -> None:
    import sys

    pydantic = pytest.importorskip("pydantic")
    from crucible.train.bridges.primerl import topk_server

    class FlatLogprobs(SimpleNamespace):
        pass

    class Choice(pydantic.BaseModel):  # prime-rl's PrimeRlGenerateResponseChoice
        index: int
        token_ids: list[int] | None = None
        completion_logprobs: dict | None = None

    class Response(pydantic.BaseModel):
        choices: list[Choice]

    st = _module("prime_rl.inference.vllm.serving_tokens", PrimeRlGenerateResponseChoice=Choice, PrimeRlGenerateResponse=Response,
                 encode_array=lambda a: {"data": base64.b64encode(np.ascontiguousarray(a).tobytes()).decode(), "shape": list(a.shape), "dtype": a.dtype.name})

    class Packed:  # prime-rl's _PackedOutputs: takes the logprobs off each output, post_process builds the module's models by name
        def __init__(self, generator, request):
            self._generator, self._request, self.fields = generator, request, {}

        async def __aiter__(self):
            async for ro in self._generator:
                for o in ro.outputs:
                    self.fields.setdefault(o.index, {})["completion_logprobs"] = {"n": len(o.logprobs.start_indices)}
                    o.logprobs = None
                yield ro

        def post_process(self, response):
            return st.PrimeRlGenerateResponse(choices=[st.PrimeRlGenerateResponseChoice(**c.model_dump(exclude={"completion_logprobs"}), **self.fields.get(c.index, {}))
                                                       for c in response.choices])

    st._PackedOutputs = Packed
    for name, mod in (("prime_rl", _module("prime_rl")), ("prime_rl.inference", _module("prime_rl.inference")), ("prime_rl.inference.vllm", _module("prime_rl.inference.vllm", serving_tokens=st)),
                      ("prime_rl.inference.vllm.serving_tokens", st), ("vllm", _module("vllm")), ("vllm.logprobs", _module("vllm.logprobs", FlatLogprobs=FlatLogprobs))):
        monkeypatch.setitem(sys.modules, name, mod)
    topk_server.install()
    topk_server.install()  # idempotent
    flat = FlatLogprobs(start_indices=[0, 3], end_indices=[3, 6], token_ids=[9, 1, 2, 1, 1, 2], logprobs=[-2.0, -0.1, -0.3, -0.1, -0.1, -0.3])
    out = SimpleNamespace(index=0, logprobs=flat)

    async def gen():
        yield SimpleNamespace(outputs=[out])

    async def go():
        p = st._PackedOutputs(gen(), SimpleNamespace(sampling_params=SimpleNamespace(logprobs=2)))
        async for _ in p:
            pass
        return p.post_process(Response(choices=[Choice(index=0, token_ids=[9, 1])])).model_dump()

    resp = asyncio.run(go())
    ch = resp["choices"][0]
    heads = topk.decode(ch["topk_ids"], ch["topk_logprobs"], start=5)
    assert heads.ids.tolist() == [[1, 2], [1, 2]] and heads.logprobs[0].tolist() == pytest.approx([-0.1, -0.3]) and ch["completion_logprobs"] == {"n": 2}


def test_the_cost_bench_times_each_k_against_the_server() -> None:
    from crucible.train.bridges.primerl.backend import topk_cost

    rows = asyncio.run(topk_cost(TopKHttp(), MODEL, [1, 2], ks=[0, 2], n=2, max_tokens=2))
    assert [r["k"] for r in rows] == [0.0, 2.0] and all(r["tokens"] == 4.0 for r in rows) and rows[1]["response_bytes_per_token"] > 0


# ---- the audit's fixes (swarm-pipeline/14 §7.7) -------------------------------------------------------------------------------------
def test_calls_that_returned_the_same_tokens_keep_their_own_heads_and_drop_one_at_a_time() -> None:
    """Two calls with equal prompt + completion (the same task in two groups, sampled from versions 3 and 4): each
    training sequence gets the heads of its own call, picked by its sampled logprobs, and training one leaves the other.
    One entry per key handed the first sequence the newer call's heads and lost the second's head when the first trained."""
    store = topk.TopKStore(k=2)
    full = [7, 8, 9, 50, 51]
    v3 = topk.Heads(ids=np.array([[50, 1], [51, 1]], np.int32), logprobs=np.array([[-0.1, -3.0], [-0.2, -3.0]], np.float32), start=3,
                    sampled=np.array([-0.1, -0.2], np.float32))
    v4 = topk.Heads(ids=np.array([[50, 2], [51, 2]], np.int32), logprobs=np.array([[-0.4, -1.5], [-0.6, -1.1]], np.float32), start=3,
                    sampled=np.array([-0.4, -0.6], np.float32))
    store.put(full, v3)
    store.put(full, v4)
    mask = [False, False, False, True, True]
    lp3, lp4 = [0.0, 0.0, 0.0, -0.1, -0.2], [0.0, 0.0, 0.0, -0.4, -0.6]
    ids3, lps3, _, cov3 = topk.stitch(full, [5], mask, store, logprobs=lp3)
    ids4, _, _, cov4 = topk.stitch(full, [5], mask, store, logprobs=lp4)
    assert cov3 == cov4 == 1.0 and ids3[3].tolist() == [50, 1] and lps3[4].tolist() == pytest.approx([-0.2, -3.0]) and ids4[3].tolist() == [50, 2]
    store.drop(full, full_logprobs=topk.trained_logprobs(lp3, mask))  # the version-3 group trained
    assert store.calls == 1 and store.get(full).ids[0].tolist() == [50, 2]
    ids4b, _, _, cov4b = topk.stitch(full, [5], mask, store, logprobs=lp4)
    assert cov4b == 1.0 and ids4b[3].tolist() == [50, 2]
    # untrained positions carry logprob 0 on the wire: they are ignored when calls are compared
    store.put(full, v3)
    lp3_partial = [0.0, 0.0, 0.0, 0.0, -0.2]
    assert topk.stitch(full, [5], [False, False, False, False, True], store, logprobs=lp3_partial)[0][4].tolist() == [51, 1]
    # the count bound evicts whole keys, oldest first
    small = topk.TopKStore(k=2, max_calls=2)
    small.put(full, v3)
    small.put(full, v4)
    small.put([1, 50, 51], replace(v3, start=1))
    assert small.calls == 1 and small.get(full) is None and small.get([1, 50, 51]) is not None


class VersionedHttp(TopKHttp):
    """The generate endpoint returning the same tokens for every call, with logprobs and heads that depend on the
    adapter (``model``): two policy versions that sample the same completion."""

    async def post(self, path, body, *, timeout_s):
        self.posts.append((path, body))
        if path != "/inference/v1/generate":
            return {}
        v = 0.1 if body["model"].endswith("-base") else 0.3
        return {"choices": [{"token_ids": [50, 51], "completion_logprobs": _arr(np.array([-v, -2 * v], np.float32)), "finish_reason": "stop",
                             "topk_ids": _arr(np.array([[50, 9], [51, 9]], np.int32)),
                             "topk_logprobs": _arr(np.array([[-v, -3.0], [-2 * v, -3.0]], np.float32))}]}


def test_an_async_group_keeps_its_heads_when_an_older_group_with_the_same_tokens_trains_first(tmp_path: Path) -> None:
    store = topk.TopKStore(k=2)
    b = PrimeRLBackend(io=FakeIO(tmp_path), http=VersionedHttp(), base_model=MODEL, learning_rate=4e-5, topk=store)
    prompt = [1, 2, 3]

    async def go():
        tr = await b.trainer(MODEL, rank=32)
        old = await (await b.sampler(MODEL, None)).sample(prompt, max_tokens=8, temperature=1.0, stop=None, seed=1)  # version 0
        b.adapters[str(tmp_path / "v1")] = "crucible-s000001"
        new = await (await b.sampler(MODEL, str(tmp_path / "v1"))).sample(prompt, max_tokens=8, temperature=1.0, stop=None, seed=2)  # version 1
        assert old.tokens == new.tokens and old.logprobs != new.logprobs
        d_old = smoke.datum(prompt, old.tokens, old.logprobs, 1.0, rollout="a", segment=0)
        d_new = smoke.datum(prompt, new.tokens, new.logprobs, 1.0, rollout="b", segment=0)
        fb1 = await tr.forward_backward([d_old], loss_fn="importance_sampling", config=None)
        _, lps, _ = topk.decode_payload(b.io.sent[0][1][0]["sc_heads"]["ids"], b.io.sent[0][1][0]["sc_heads"]["counts"])
        assert fb1.metrics["sc/fallback"] == 0.0 and lps[3].tolist() == pytest.approx([-0.1, -3.0])  # its own (version-0) heads
        fb2 = await tr.forward_backward([d_new], loss_fn="importance_sampling", config=None)
        _, lps2, _ = topk.decode_payload(b.io.sent[1][1][0]["sc_heads"]["ids"], b.io.sent[1][1][0]["sc_heads"]["counts"])
        assert fb2.metrics["sc/fallback"] == 0.0 and lps2[3].tolist() == pytest.approx([-0.3, -3.0]) and not store.by_key

    asyncio.run(go())


def test_the_sampler_asks_for_the_full_softmax_and_keeps_heads_only_at_the_training_temperature(tmp_path: Path) -> None:
    """vLLM's /v1/completions fills unset sampling fields from the model's generation_config.json (the student ships
    top_k = 20, top_p = 0.95), which the trainer cannot score; every call sends neutral values and the server is told to
    use vLLM's defaults. A call at another temperature (the smoke's greedy probe) keeps no heads: its logprobs are
    softmax(z / T_call), not the distribution the trainer scores."""
    from crucible.train.bridges.primerl.backend import NEUTRAL_SAMPLING

    store = topk.TopKStore(k=2)
    b = PrimeRLBackend(io=FakeIO(tmp_path), http=TopKHttp(), base_model=MODEL, learning_rate=4e-5, topk=store, temperature=0.7)

    async def go():
        s = await b.sampler(MODEL, None)
        await s.sample([1, 2, 3], max_tokens=8, temperature=0.0, stop=None, seed=1)
        assert store.calls == 0 and b.stats["sc_heads_other_temperature"] == 1
        await s.sample([1, 2, 3], max_tokens=8, temperature=0.7, stop=None, seed=1)
        assert store.calls == 1 and store.get([1, 2, 3, 50, 51]).sampled.tolist() == pytest.approx([-0.1, -0.2])
        sp = b.http.posts[-1][1]["sampling_params"]
        assert all(sp[k] == v for k, v in NEUTRAL_SAMPLING.items()) and sp["temperature"] == 0.7
        plain = PrimeRLBackend(io=FakeIO(tmp_path), http=TopKHttp(), base_model=MODEL, learning_rate=4e-5)
        await (await plain.sampler(MODEL, None)).sample([1, 2], max_tokens=4, temperature=1.0, stop=None, seed=None)  # the fake answers {}
        path, body = plain.http.posts[-1]
        assert path == "/v1/completions" and all(body[k] == v for k, v in NEUTRAL_SAMPLING.items())

    asyncio.run(go())
    assert deploy.inference_toml(TrainConfig(model=MODEL, require_band=False), deploy.DeltaConfig())["vllm"]["generation_config"] == "vllm"
