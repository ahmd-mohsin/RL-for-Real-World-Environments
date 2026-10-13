"""Group-level credit operators for the M12 group grading stage (m13 C-3; article §3.6.3.2b-c, MiMo GRS / GAR).

**GAR** (group advantage redistribution). Within a GRPO group:

1. confirmed shortcut solutions (``hack``) are zeroed and the group statistics recomputed (a hack is a failure);
2. raw advantages ``A_i = R_i - mean`` (optionally ``/ std``);
3. among the passing members ``P`` of a *mixed* group, the positive advantage mass ``M = sum_P A_i`` is redistributed by
   quality factors ``f_i in (0, 1]`` (a groupwise grader: approach, precision, minimality, no side effects,
   craftsmanship). Default (``mode="redistribute"``, the article's MiMo form): ``λ = M / Σ_P f_i A_i`` capped at
   ``lambda_max``, ``A'_i = λ f_i A_i``. ``mode="interpolate"``: ``target_i = M * f_i / sum_P f`` and
   ``A'_i = A_i + lam * (target_i - A_i)``. Both conserve the mass exactly; ``train.group_ops.gar`` is this same operator;
4. failing members keep their advantage; all-pass and all-fail groups (after hack zeroing) are untouched, since quality
   then has no signal to separate from the outcome.

Since the passing mass is conserved and failing members are unchanged, ``sum A' = sum A = 0``: the group stays centred.

**GRS** (multiplicative rubric reward): ``R = R_test * S_sol * S_beh`` with rubric scores in ``[0, 1]``: a failure stays
0, and passing solutions are ranked by solution and behaviour quality. The rubric must be grounded in the task, not in
the rollouts it ranks: ``check_rubric_grounding`` rejects items that cite a rollout id.

``gar_group_fn`` adapts GAR to ``Runtime(group_fn=...)``: ``quality(passing_members) -> [f_i]`` (sync or async, e.g. an
LLM groupwise grader), ``hack(member) -> bool`` (default: ``reward_components["hack/confirmed"] > 0``, produced by the
two-level detector).
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Callable, Sequence
from typing import Any


def gar(rewards: Sequence[float | None], quality: Sequence[float | None], hack: Sequence[bool] | None = None, *, mode: str | None = None,
        lam: float | None = None, lambda_max: float = 3.0, pass_threshold: float = 1.0, std_normalize: bool = False,
        eps: float = 1e-6) -> tuple[list[float | None], list[float | None]]:
    """Returns ``(advantages, rewards_after_hack_zeroing)``; masked members (``None``) stay ``None``. ``quality[i]`` is
    read only for passing members (``None`` there means ``1.0``). The one GAR of Crucible (M12 audit: two operators shared
    the name and disagreed by 2× on the same group):

    - ``mode="redistribute"`` (default; MiMo-V2.6 / article §3.6.3.2c): λ = Σ_P A_j / Σ_P f_j A_j, capped at
      ``lambda_max``; A'_i = λ f_i A_i on the passing set P; when the cap binds the group is re-centred to zero mean.
    - ``mode="interpolate"`` (the default when ``lam`` is given): A'_i = A_i + lam·(M f_i / Σf − A_i), mass-conserving
      for any lam ∈ [0, 1], at most lam of the way to the quality-proportional split."""
    mode = mode or ("interpolate" if lam is not None else "redistribute")
    if mode not in ("redistribute", "interpolate"):
        raise ValueError("GAR mode: redistribute | interpolate")
    lam = 0.5 if lam is None else lam
    if mode == "interpolate" and not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1] (got {lam})")
    n = len(rewards)
    hack = list(hack or [False] * n)
    r = [None if x is None else (0.0 if hack[i] else float(x)) for i, x in enumerate(rewards)]
    idx = [i for i in range(n) if r[i] is not None]
    if not idx:
        return [None] * n, r
    mu = sum(r[i] for i in idx) / len(idx)
    sd = math.sqrt(sum((r[i] - mu) ** 2 for i in idx) / len(idx)) if std_normalize else 1.0
    adv: list[float | None] = [None] * n
    for i in idx:
        adv[i] = (r[i] - mu) / (sd + eps) if std_normalize else r[i] - mu
    passing = [i for i in idx if r[i] >= pass_threshold]
    if not passing or len(passing) == len(idx) or (mode == "interpolate" and lam == 0.0):
        return adv, r
    f = []
    for i in passing:
        q = 1.0 if quality[i] is None else float(quality[i])
        if not 0.0 < q <= 1.0:
            raise ValueError(f"quality factors must be in (0, 1] (member {i}: {q})")
        f.append(q)
    if mode == "interpolate":
        mass = sum(adv[i] for i in passing)
        fs = sum(f)
        for i, q in zip(passing, f, strict=True):
            adv[i] = adv[i] + lam * (mass * q / fs - adv[i])
        return adv, r
    den = sum(q * adv[i] for i, q in zip(passing, f, strict=True))
    lam_r = sum(adv[i] for i in passing) / den if den > 0 else 1.0
    capped = lam_r > lambda_max
    lam_r = min(lam_r, lambda_max)
    for i, q in zip(passing, f, strict=True):
        adv[i] = lam_r * q * adv[i]
    if capped:
        c = sum(adv[i] for i in idx) / len(idx)
        for i in idx:
            adv[i] = adv[i] - c
    return adv, r


def grs(r_test: float | None, s_sol: float, s_beh: float) -> float | None:
    """MiMo GRS: ``R_test * S_sol * S_beh``; scores in [0, 1]; a masked outcome stays masked."""
    for name, v in (("s_sol", s_sol), ("s_beh", s_beh)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0, 1] (got {v})")
    return None if r_test is None else float(r_test) * s_sol * s_beh


def check_rubric_grounding(items: Sequence[str], rollout_ids: Sequence[str]) -> list[str]:
    """Rubric items that cite a rollout id (or 'rollout N' / 'sample N'): grounded in the samples, not the task. Empty
    list = grounded."""
    bad = []
    pat = re.compile(r"\b(rollout|sample|attempt|candidate)\s*#?\d+\b", re.IGNORECASE)
    for it in items:
        if any(rid and rid in it for rid in rollout_ids) or pat.search(it):
            bad.append(it)
    return bad


def _default_hack(rec: Any) -> bool:
    return float((rec.reward_components or {}).get("hack/confirmed", 0.0)) > 0.0


def gar_group_fn(quality: Callable[[list[Any]], Any], *, hack: Callable[[Any], bool] = _default_hack, mode: str | None = None, lam: float | None = None,
                 lambda_max: float = 3.0, pass_threshold: float = 1.0, std_normalize: bool = False) -> Callable:
    """``Runtime(group_fn=gar_group_fn(quality))``: sets ``reward`` (hacks zeroed), ``advantage`` and
    ``reward_components["gar/quality"]`` on each member. Members with ``mask_sample`` are left out (never scored)."""

    async def fn(gid: str, members: list[Any]) -> list[Any]:
        live = [m for m in members if not getattr(m, "mask_sample", False)]
        hacks = [bool(hack(m)) for m in live]
        zeroed = [None if m.reward is None else (0.0 if h else m.reward) for m, h in zip(live, hacks, strict=True)]
        passing = [m for m, z in zip(live, zeroed, strict=True) if z is not None and z >= pass_threshold]
        q: list[float | None] = [None] * len(live)
        if passing and len(passing) < sum(z is not None for z in zeroed):  # a mixed group: ask the grader
            got = quality(passing)
            if asyncio.iscoroutine(got):
                got = await got
            got = list(got)
            if len(got) != len(passing):
                raise ValueError(f"quality returned {len(got)} factors for {len(passing)} passing members")
            it = iter(got)
            q = [next(it) if (z is not None and z >= pass_threshold) else None for z in zeroed]
        adv, rew = gar([m.reward for m in live], q, hacks, mode=mode, lam=lam, lambda_max=lambda_max, pass_threshold=pass_threshold,
                       std_normalize=std_normalize)
        for m, a, rr, qq, h in zip(live, adv, rew, q, hacks, strict=True):
            m.reward, m.advantage = rr, a
            if qq is not None:
                m.reward_components["gar/quality"] = float(qq)
            if h:
                m.reward_components["gar/hack_zeroed"] = 1.0
        return members

    return fn
