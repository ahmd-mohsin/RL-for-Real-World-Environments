"""Score centering (arXiv 2609.20807) checked equation by equation against brute-force full-vocabulary computations and
against the authors' training loss (martin-marek/score-centering ``losses.py``, MIT, fdca22b; ported below as a test
oracle). CPU, float64, small categorical distributions. The audit is swarm-pipeline/14 §7.7.

- App. A.1 Eq. 7-8: the top-k head plus the ρ·p tail model equals full centering under the reconstructed q̂.
- App. A.2 Eq. 9-10: composed with TIS / MIS, the head-only form with α = ρ·f(1/ρ) equals the full-vocabulary
  expectation of the weighted score under q̂ (ŵ = f(p/q̂)).
- §4 Eq. 5: the expected centered update is Cov_q(R, w·s): no drift, whatever the rewards.
- §4: the centering term is deterministic given the prefix, so at a constant advantage it leaves the update's variance
  unchanged, while naive importance sampling (also drift-free) inflates it.
- Edge cases: an empty tail, ρ → 0 and ρ → ∞, a head without the sampled token, head entries the sampler gave no mass.
- Aggregation: per-token sums over prime-rl's sequences, divided by its token count after ``wire``'s scaling, equal the
  reference README's token-mean loss; zero advantages and untrained (environment) tokens add nothing.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch")

from crucible.train.bridges.primerl import wire
from crucible.train.bridges.primerl.loss import Head, crucible_loss, token_objective
from crucible.train.bridges.tinker.backend import TokenDatum
from crucible.train.bridges.tinker.datums import scale

F64 = torch.float64
INF = math.inf


# ---- the oracle: the reference repository's training loss, ported from JAX to torch ----------------------------------------------
# martin-marek/score-centering, losses.py at fdca22b (2026-10-27; the loss is unchanged since 6484296), MIT License,
# Copyright (c) 2026 Martin Marek (LICENSES/MIT-score-centering.txt). Ported: ``ISCorrection`` (token level, stat=ratio,
# sign=both: the corrections this bridge composes with), ``log_mass``, ``is_weight`` and the per-token "score" terms of
# ``loss_fn_rl`` (the top-k head branch and the full-vocabulary branch). JAX -> torch, sg -> detach; nothing else changed.
class ISCorrection(NamedTuple):
    low: float = 0.0
    high: float = INF
    outside: str = "drop"  # clamp | drop


REF_CORRECTIONS = {"sc": None, "sc+tis": ISCorrection(high=2.0, outside="clamp"), "sc+mis": ISCorrection(low=0.5, high=5.0)}


def ref_log_mass(log_p, log_q, c):
    """log(q * w(p/q)) (reference ``log_mass`` with sign=both, stat=ratio)."""
    log_p, log_q = torch.broadcast_tensors(torch.as_tensor(log_p, dtype=F64), torch.as_tensor(log_q, dtype=F64))
    if c is None:
        return log_q
    lo = log_q + math.log(c.low) if c.low > 0 else torch.full_like(log_q, -INF)
    hi = log_q + math.log(c.high) if c.high < INF else torch.full_like(log_q, INF)
    in_band = (log_p >= lo) & (log_p <= hi)
    clipped = torch.minimum(torch.maximum(log_p, lo), hi)
    return torch.where(in_band, log_p, clipped if c.outside == "clamp" else torch.full_like(log_p, -INF))


def ref_is_weight(log_ratio, c):
    """Detached IS weight w(r), r = exp(log_ratio) clipped to e^±20 (reference ``is_weight``)."""
    if c is None:
        return torch.ones_like(log_ratio)
    return torch.exp(ref_log_mass(log_ratio.clamp(-20.0, 20.0), torch.zeros_like(log_ratio), c)).detach()


def ref_score(logp_vocab, targets, logq, c, *, center_ids=None, center_logprobs=None, full_logq=None):
    """The per-token ``terms["score"]`` of the reference ``loss_fn_rl`` (score centering on); the loss is
    −Σ A·score·mask / n_tokens."""
    logp = logp_vocab.gather(-1, targets[..., None])[..., 0]
    w = ref_is_weight(logp - logq, c)
    plogp = (torch.exp(logp_vocab) * logp_vocab).sum(-1).detach()  # E_p[log p] has zero gradient
    score = w * logp
    if full_logq is not None:
        mass = torch.exp(ref_log_mass(logp_vocab, full_logq, c))
        score = score - (mass.detach() * logp_vocab).sum(-1)
    if center_ids is not None:
        logp_head, logq_head = logp_vocab.gather(-1, center_ids), center_logprobs
        p_head, q_head = torch.exp(logp_head).detach(), torch.exp(logq_head)
        q_tail = 1.0 - q_head.sum(-1)
        rho = q_tail.clamp(min=0.0) / (1.0 - p_head.sum(-1)).clamp(min=1e-6)
        alpha = torch.exp(ref_log_mass(torch.zeros_like(rho), torch.log(rho), c))
        mass = torch.exp(ref_log_mass(logp_head, logq_head, c))
        residual = mass - alpha[..., None] * p_head
        score = score - ((residual.detach() * logp_head).sum(-1) + (alpha * plogp).detach())
    return score


# ---- helpers ------------------------------------------------------------------------------------------------------------------------------
def _w(r, pg):
    """The paper's weight functions f: 1 (SC), TIS min(r, 2), MIS r·1[0.5 <= r <= 5]."""
    if pg == "sc":
        return torch.ones_like(r)
    if pg == "sc+tis":
        return r.clamp(max=2.0)
    return torch.where((r >= 0.5) & (r <= 5.0), r, torch.zeros_like(r))


def _case(T=6, V=10, seed=0, noise=0.5):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(T, V, generator=g, dtype=F64)
    lq = torch.log_softmax(z + noise * torch.randn(T, V, generator=g, dtype=F64), -1)
    y = torch.multinomial(lq.exp(), 1, generator=g).squeeze(-1)
    adv = torch.randn(T, generator=g, dtype=F64)
    return z, lq, y, adv


def _ours(z, lq_full, y, adv, ids, pg, *, lq_head=None, mask=None, valid=None):
    """Our gradient d loss / d logits with the head ``ids`` (and their sampler logprobs ``lq_head``, default read off
    ``lq_full``)."""
    z = z.clone().requires_grad_(True)
    lp = torch.log_softmax(z, -1)
    lq_head = lq_full.gather(-1, ids) if lq_head is None else lq_head
    m = torch.ones(z.shape[0], dtype=torch.bool) if mask is None else mask
    head = Head(lp=lp.gather(-1, ids), lq=lq_head, valid=torch.ones_like(ids, dtype=torch.bool) if valid is None else valid,
                present=torch.ones(z.shape[0], dtype=torch.bool))
    loss, metrics = token_objective(lp.gather(-1, y[:, None])[:, 0], lq_full.gather(-1, y[:, None])[:, 0], adv, m, pg_correction=pg, head=head)
    (g,) = torch.autograd.grad(loss, z)
    return g, metrics


def _ref(z, lq_full, y, adv, ids, pg, *, lq_head=None, mask=None):
    """The ported reference loss's gradient (summed rather than divided by n_tokens, as ours is before prime-rl's scale)."""
    z = z.clone().requires_grad_(True)
    lp = torch.log_softmax(z, -1)
    lq_head = lq_full.gather(-1, ids) if lq_head is None else lq_head
    score = ref_score(lp, y, lq_full.gather(-1, y[:, None])[:, 0], REF_CORRECTIONS[pg], center_ids=ids, center_logprobs=lq_head)
    m = torch.ones(z.shape[0], dtype=F64) if mask is None else mask.to(F64)
    (g,) = torch.autograd.grad(-(adv * score * m).sum(), z)
    return g


PGS = ("sc", "sc+tis", "sc+mis")


# ---- equivalence with the authors' training loss ------------------------------------------------------------------------------------
@pytest.mark.parametrize("pg", PGS)
@pytest.mark.parametrize("k", [1, 3, 9])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_gradient_equals_the_reference_training_loss(pg, k, seed) -> None:
    z, lq, y, adv = _case(T=7, V=12, seed=seed, noise=0.8)
    ids = lq.topk(k, dim=-1).indices
    mask = torch.tensor([True, True, False, True, True, False, True])  # untrained (environment) positions in between
    assert torch.allclose(_ours(z, lq, y, adv, ids, pg, mask=mask)[0], _ref(z, lq, y, adv, ids, pg, mask=mask), atol=1e-12)


@pytest.mark.parametrize("pg", PGS)
def test_the_full_vocabulary_branch_of_the_reference_is_k_equal_to_v(pg) -> None:
    z, lq, y, adv = _case(T=5, V=8, seed=4, noise=1.0)
    g_ours, m = _ours(z, lq, y, adv, lq.topk(8, dim=-1).indices, pg)
    zr = z.clone().requires_grad_(True)
    lp = torch.log_softmax(zr, -1)
    score = ref_score(lp, y, lq.gather(-1, y[:, None])[:, 0], REF_CORRECTIONS[pg], full_logq=lq)
    (g_ref,) = torch.autograd.grad(-(adv * score).sum(), zr)
    assert torch.allclose(g_ours, g_ref, atol=1e-10) and m["sc/head_mass"].item() == pytest.approx(1.0)


# ---- App. A.1-A.2: the top-k head with the ρ·p tail is full centering under q̂ --------------------------------------------------------
def _q_hat(lp_full, lq_full, ids):
    """Eq. 7: q̂_v = q_v on the head H, ρ·p_v on the tail T, ρ = (1 − Σ_H q)/(1 − Σ_H p) (no floors: the tests keep
    both tails well above them)."""
    p, q = lp_full.exp(), lq_full.exp()
    in_h = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, ids, True)
    rho = (1 - (q * in_h).sum(-1)) / (1 - (p * in_h).sum(-1))
    return torch.where(in_h, q, rho[:, None] * p), rho


@pytest.mark.parametrize("pg", PGS)
@pytest.mark.parametrize("k", [1, 2, 4])
def test_eq_8_and_10_the_head_only_term_is_the_full_expectation_under_q_hat(pg, k) -> None:
    """Brute force over the whole vocabulary: −A·(sg(w_y)·log p_y − Σ_v sg(q̂_v·f(p_v/q̂_v))·log p_v) with q̂ of Eq. 7. Our
    head-only gradient (α = ρ·f(1/ρ)) must equal it exactly, for SC, SC+TIS and SC+MIS."""
    z, lq, y, adv = _case(T=6, V=9, seed=10 + k, noise=0.7)
    ids = lq.topk(k, dim=-1).indices
    g_ours, m = _ours(z, lq, y, adv, ids, pg)
    zb = z.clone().requires_grad_(True)
    lp = torch.log_softmax(zb, -1)
    q_hat, rho = _q_hat(lp.detach(), lq, ids)
    w_hat = _w(lp.detach().exp() / q_hat, pg)  # ŵ_v = f(p_v / q̂_v): 1/ρ on the tail
    lp_y, lq_y = lp.gather(-1, y[:, None])[:, 0], lq.gather(-1, y[:, None])[:, 0]
    w_y = _w(torch.exp(lp_y.detach() - lq_y), pg)
    brute = -(adv * (w_y * lp_y - ((q_hat * w_hat).detach() * lp).sum(-1))).sum()
    (g_brute,) = torch.autograd.grad(brute, zb)
    assert torch.allclose(g_ours, g_brute, atol=1e-12)
    assert m["sc/rho"].item() == pytest.approx(float(rho.mean()), rel=1e-12)


def test_when_the_sampler_tail_is_proportional_to_the_trainer_tail_top_k_is_exact() -> None:
    """Eq. 7 is the truth when q_T = ρ·p_T: then a k = 2 head over V = 8 (a six-token tail) cancels the drift exactly."""
    g = torch.Generator().manual_seed(21)
    V, k = 8, 2
    logits = torch.randn(V, generator=g, dtype=F64)
    p = torch.softmax(logits, -1)
    ids = p.topk(k).indices
    q = p.clone()
    q[ids] = torch.tensor([0.5, 0.3], dtype=F64)  # the sampler moves head mass; its tail keeps p's shape
    tail = torch.ones(V, dtype=torch.bool)
    tail[ids] = False
    q[tail] = p[tail] * (1 - q[ids].sum()) / p[tail].sum()
    lq = q.log()
    for pg in PGS:
        assert _expected_grad(logits, lq, pg, ids, rewards=torch.ones(V, dtype=F64)).abs().max() < 1e-12, pg


# ---- §4 Eq. 5: the expected update is the covariance under q ------------------------------------------------------------------------
def _expected_grad(logits, lq_full, pg, ids, *, rewards):
    """E_{y~q}[∇_z loss(y)] at one position with advantage ``rewards[y]``, by enumerating the vocabulary."""
    V = logits.numel()
    total = torch.zeros(V, dtype=F64)
    for y in range(V):
        z = logits.clone()[None].requires_grad_(True)
        lp = torch.log_softmax(z, -1)
        head = None
        if pg.startswith("sc"):
            head = Head(lp=lp.gather(-1, ids[None]), lq=lq_full[None].gather(-1, ids[None]), valid=torch.ones(1, len(ids), dtype=torch.bool),
                        present=torch.ones(1, dtype=torch.bool))
        loss, _ = token_objective(lp[:, y], lq_full[None, y], rewards[y:y + 1], torch.ones(1, dtype=torch.bool), pg_correction=pg, head=head)
        (gz,) = torch.autograd.grad(loss, z)
        total += lq_full[y].exp() * gz[0]
    return total


@pytest.mark.parametrize("pg", PGS)
def test_eq_5_the_expected_centered_update_is_the_covariance_of_reward_and_weighted_score(pg) -> None:
    g = torch.Generator().manual_seed(5)
    V = 7
    logits = torch.randn(V, generator=g, dtype=F64)
    lq = torch.log_softmax(logits + 0.6 * torch.randn(V, generator=g, dtype=F64), -1)
    rewards = torch.randn(V, generator=g, dtype=F64) + 2.0  # mostly positive: the regime where drift hurts most (Fig. 2)
    got = _expected_grad(logits, lq, pg, torch.arange(V), rewards=rewards)
    q, p = lq.exp(), torch.softmax(logits, -1)
    w = _w(p / q, pg)
    scores = torch.eye(V, dtype=F64) - p  # ∇_z log p_y = e_y − p
    r_bar = (q * rewards).sum()
    cov = (q[:, None] * (rewards - r_bar)[:, None] * w[:, None] * scores).sum(0)  # Cov_q(R, w·s)
    assert torch.allclose(got, -cov, atol=1e-12)
    # plain policy gradient adds the drift E_q[R]·s̄ on top of the signal
    plain = -(q[:, None] * rewards[:, None] * scores).sum(0)
    drift = -r_bar * (q[:, None] * scores).sum(0)
    assert torch.allclose(plain, drift - (q[:, None] * (rewards - r_bar)[:, None] * scores).sum(0), atol=1e-12) and drift.abs().max() > 0.05


# ---- §4: deterministic correction, unchanged variance; IS inflates it -------------------------------------------------------------
def test_centering_shifts_the_update_by_a_constant_while_naive_is_inflates_its_variance() -> None:
    g = torch.Generator().manual_seed(3)
    V = 8
    logits = torch.randn(V, generator=g, dtype=F64)
    lq = torch.log_softmax(logits + 1.2 * torch.randn(V, generator=g, dtype=F64), -1)  # a severe mismatch
    q = lq.exp()

    def per_token(pg, kind="importance_sampling"):
        rows = []
        for y in range(V):
            z = logits.clone()[None].requires_grad_(True)
            lp = torch.log_softmax(z, -1)
            head = Head(lp=lp, lq=lq[None], valid=torch.ones(1, V, dtype=torch.bool), present=torch.ones(1, dtype=torch.bool)) if pg.startswith("sc") else None
            loss, _ = token_objective(lp[:, y], lq[None, y], torch.ones(1, dtype=F64), torch.ones(1, dtype=torch.bool), kind=kind, pg_correction=pg,
                                      head=head, max_importance_ratio=1e9)
            rows.append(torch.autograd.grad(loss, z)[0][0])
        return torch.stack(rows)

    def mean_cov(gs):
        mu = (q[:, None] * gs).sum(0)
        d = gs - mu
        return mu, (q[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0)

    pg_g = -(torch.eye(V, dtype=F64) - torch.softmax(logits, -1))  # plain PG at A = 1: −∇log p_y
    mu_pg, cov_pg = mean_cov(pg_g)
    mu_sc, cov_sc = mean_cov(per_token("sc"))
    mu_is, cov_is = mean_cov(per_token("none"))  # naive token IS: −r·A
    assert mu_pg.abs().max() > 1e-2  # PG drifts
    assert mu_sc.abs().max() < 1e-12 and mu_is.abs().max() < 1e-12  # both corrections cancel it
    assert torch.allclose(cov_sc, cov_pg, atol=1e-12)  # the centering term is the same vector for every sampled y
    assert torch.trace(cov_is) > 1.5 * torch.trace(cov_sc)  # the random multiplicative weight inflates the variance


# ---- edge cases -------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pg", PGS)
def test_an_empty_tail_is_full_centering_whatever_the_floors_give_rho(pg) -> None:
    z, lq, y, adv = _case(T=4, V=6, seed=8)
    g_ours, m = _ours(z, lq, y, adv, lq.topk(6, dim=-1).indices, pg)  # both tails are ~1e-16: floored, ρ = 1
    zf = z.clone().requires_grad_(True)
    lp = torch.log_softmax(zf, -1)
    lp_y, lq_y = lp.gather(-1, y[:, None])[:, 0], lq.gather(-1, y[:, None])[:, 0]
    full = -(adv * (_w(torch.exp(lp_y.detach() - lq_y), pg) * lp_y - ((lq.exp() * _w(lp.detach().exp() / lq.exp(), pg)) * lp).sum(-1))).sum()
    assert torch.allclose(g_ours, torch.autograd.grad(full, zf)[0], atol=1e-10) and m["sc/rho"].item() == pytest.approx(1.0)


@pytest.mark.parametrize("pg", PGS)
def test_extreme_tail_ratios_stay_finite_and_match_the_reference(pg) -> None:
    """ρ → 0 (the sampler's head holds all its mass, the trainer's does not) and ρ → ∞ (the reverse). The reference floors
    the trainer's tail at 1e-6; the README (and Crucible) floor the sampler's too, which moves α by ≤ 1e-6/p_tail."""
    ids = torch.tensor([[0, 1, 2]])
    y = torch.tensor([1])
    adv = torch.tensor([1.3], dtype=F64)
    # ρ → 0: q puts 1 − 1e-12 on the head, p puts 0.5
    z0 = torch.log(torch.tensor([[0.2, 0.2, 0.1, 0.2, 0.2, 0.1]], dtype=F64))
    lq0 = torch.log(torch.tensor([[0.5, 0.3, 0.2 - 1e-12, 4e-13, 3e-13, 3e-13]], dtype=F64))
    # ρ → ∞: q leaves 0.3 in its tail, p leaves 1e-9 (floored at 1e-6, so ρ = 3e5)
    z1 = torch.log(torch.tensor([[0.6, 0.3, 0.1 - 1e-9, 4e-10, 3e-10, 3e-10]], dtype=F64))
    lq1 = torch.log(torch.tensor([[0.3, 0.3, 0.1, 0.1, 0.1, 0.1]], dtype=F64))
    for z, lq, tol in ((z0, lq0, 1e-6), (z1, lq1, 1e-9)):
        g_ours, m = _ours(z, lq, y, adv, ids, pg)
        assert torch.isfinite(g_ours).all() and all(torch.isfinite(v).all() for v in m.values())
        assert torch.allclose(g_ours, _ref(z, lq, y, adv, ids, pg), atol=tol)
    _, m1 = _ours(z1, lq1, y, adv, ids, pg)
    assert m1["sc/rho"].item() == pytest.approx(0.3 / 1e-6, rel=1e-6)
    # MIS drops a tail with 1/ρ outside [0.5, 5] (α = 0); TIS keeps α = min(1, 2ρ) = 1; vanilla SC uses α = ρ itself
    assert m1["sc/tail_dropped"].item() == (1.0 if pg == "sc+mis" else 0.0)


def test_head_entries_the_sampler_gave_no_mass_are_head_tokens_with_q_zero() -> None:
    """A head entry at -9999 (vLLM's clamp of -inf, e.g. k above the sampler's support) is q̂_v = 0 on the head (Eq. 7),
    not tail: it only moves α·p_v, and it matches the reference given the same head."""
    z, lq, y, adv = _case(T=3, V=8, seed=12)
    ids = lq.topk(3, dim=-1).indices
    lq_head = lq.gather(-1, ids).clone()
    lq_head[:, 2] = -9999.0
    for pg in PGS:
        g_ours, _ = _ours(z, lq, y, adv, ids, pg, lq_head=lq_head)
        assert torch.allclose(g_ours, _ref(z, lq, y, adv, ids, pg, lq_head=lq_head), atol=1e-12), pg


def test_the_sampled_token_outside_the_head_is_trained_with_the_samplers_own_logprob() -> None:
    z, lq, _, adv = _case(T=5, V=10, seed=13, noise=0.3)
    ids = lq.topk(2, dim=-1).indices
    y = lq.argsort(-1)[:, 0]  # each position's least likely token: never in the head
    assert not (ids == y[:, None]).any()
    for pg in PGS:
        assert torch.allclose(_ours(z, lq, y, adv, ids, pg)[0], _ref(z, lq, y, adv, ids, pg), atol=1e-12), pg


def test_the_sign_a_positive_advantage_raises_the_sampled_token_and_a_negative_one_lowers_it() -> None:
    z, lq, y, _ = _case(T=1, V=6, seed=14, noise=0.1)
    ids = lq.topk(6, dim=-1).indices
    for a in (1.0, -1.0):
        g, _ = _ours(z, lq, y, torch.tensor([a], dtype=F64), ids, "sc+mis")
        assert (-g[0, int(y)]).item() * a > 0  # a descent step moves the sampled logit with the advantage's sign


# ---- aggregation through prime-rl's normalization ---------------------------------------------------------------------------------
def _datum(rng, full_len, trained, V):
    full = torch.randint(0, V, (full_len,), generator=rng).tolist()
    mask = [1 if i + 1 in trained else 0 for i in range(full_len - 1)]  # target coordinates
    return full, mask


def test_token_mean_through_wire_and_prime_rls_divide_equals_the_reference_token_mean_loss() -> None:
    """Three sequences, sampled spans with an environment span between them, ``loss_scale = token_mean``. The bridge
    sends A/N·N (``wire.undo_token_mean``), prime-rl sums ``crucible_loss`` per sequence and divides by its N trainable
    tokens: the gradient must be the README loss ``(token_loss·mask).sum() / mask.sum()`` with the unscaled A."""
    rng = torch.Generator().manual_seed(17)
    V, k = 11, 3
    specs = [(9, {3, 4, 5, 7, 8}), (6, {2, 3, 4, 5}), (7, {5, 6})]  # positions 6 (seq 0) and 1 (seq 1) are tool output
    datums, logits, lqs, ys = [], [], [], []
    for n, trained in specs:
        full, mask = _datum(rng, n, trained, V)
        z = torch.randn(n, V, generator=rng, dtype=F64)  # row j: the trainer's logits for token j
        lq = torch.log_softmax(z + 0.7 * torch.randn(n, V, generator=rng, dtype=F64), -1)
        lp_tok = [float(lq[j, full[j]]) for j in range(1, n)]
        adv_seq = float(torch.randn(1, generator=rng, dtype=F64))
        datums.append(TokenDatum(input_ids=full[:-1], target_ids=full[1:], logprobs=[x if m else 0.0 for x, m in zip(lp_tok, mask, strict=True)],
                                 advantages=[adv_seq] * (n - 1), mask=mask))
        logits.append(z)
        lqs.append(lq)
        ys.append(torch.tensor(full))
    scaled = scale(datums, "token_mean")
    rows = wire.samples(scaled, temperature=1.0)
    n_train = sum(d.n_train for d in datums)
    zs = [z.clone().requires_grad_(True) for z in logits]
    total = 0.0
    for z, lq, y, row in zip(zs, lqs, ys, rows, strict=True):
        lp = torch.log_softmax(z, -1)
        m = torch.tensor(row["mask"])
        ids = lq.topk(k, dim=-1).indices
        head = Head(lp=lp.gather(-1, ids), lq=lq.gather(-1, ids), valid=torch.ones_like(ids, dtype=torch.bool), present=m.clone())
        inp = SimpleNamespace(trainer_logprobs=lp.gather(-1, y[:, None])[:, 0], inference_logprobs=torch.tensor(row["logprobs"], dtype=F64),
                              advantages=torch.tensor(row["advantages"], dtype=F64), loss_mask=m, loss_weights=None, ref_logprobs=None)
        loss, _ = token_objective(inp.trainer_logprobs, inp.inference_logprobs, inp.advantages, m, pg_correction="sc+mis", head=head)
        total = total + loss
    total = total / n_train  # prime-rl's rl_scale: the step's trainable-token count
    grads = torch.autograd.grad(total, zs)
    # the reference: README token_loss on every trained token with the datum's own A, token-mean over the batch
    zr = [z.clone().requires_grad_(True) for z in logits]
    ref_total = 0.0
    for z, lq, y, d in zip(zr, lqs, ys, datums, strict=True):
        lp = torch.log_softmax(z, -1)
        ids = lq.topk(k, dim=-1).indices
        m = torch.tensor([False] + [bool(x) for x in d.mask])
        a = torch.tensor([0.0] + d.advantages, dtype=F64)
        score = ref_score(lp, y, lq.gather(-1, y[:, None])[:, 0], REF_CORRECTIONS["sc+mis"], center_ids=ids, center_logprobs=lq.gather(-1, ids))
        ref_total = ref_total - (a * score * m.to(F64)).sum()
    ref_grads = torch.autograd.grad(ref_total / n_train, zr)
    for g_ours, g_ref in zip(grads, ref_grads, strict=True):
        assert torch.allclose(g_ours, g_ref, atol=1e-12)
    # untrained positions (token 0 and the tool output) get no gradient from their own row
    assert grads[0][6].abs().max() == 0.0 and grads[1][1].abs().max() == 0.0 and grads[0][0].abs().max() == 0.0


def test_a_zero_advantage_group_contributes_nothing_even_under_mismatch() -> None:
    z, lq, y, _ = _case(T=6, V=9, seed=19, noise=1.5)
    ids = lq.topk(3, dim=-1).indices
    for pg in PGS:
        g, _ = _ours(z, lq, y, torch.zeros(6, dtype=F64), ids, pg)
        assert g.abs().max() == 0.0, pg
    # through the prime-rl entry point as well, with the per-token rl weights it may send
    zz = z.clone().requires_grad_(True)
    lp = torch.log_softmax(zz, -1)
    inp = SimpleNamespace(trainer_logprobs=lp.gather(-1, y[:, None])[:, 0], inference_logprobs=lq.gather(-1, y[:, None])[:, 0],
                          advantages=torch.zeros(6, dtype=F64), loss_mask=torch.ones(6, dtype=torch.bool), loss_weights=torch.full((6,), 2.0, dtype=F64),
                          ref_logprobs=None)
    out = crucible_loss(inp, kind="importance_sampling")
    assert torch.autograd.grad(out.loss, zz, allow_unused=True)[0].abs().max() == 0.0
