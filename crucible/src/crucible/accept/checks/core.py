"""A1-A8 on the contract runner. They keep the M8 subset's logic (``accept.contract``) and add what DESIGN §9 asks for:
perf headroom and fresh-seed no-crash grades (A1), every reward channel at its floor (A2), generated defect-class
mutants with validity and equivalence filters (A3), full-verdict stability, fresh-gate flake draws and reference
self-consistency (A4), canary planting and the agent image (A5)."""

from __future__ import annotations

from typing import Any

from crucible.accept.checks.base import AcceptContext, InfraUnavailable, MissingCapability, Out, need, register
from crucible.accept.model import Need, Status
from crucible.verify import calibrate
from crucible.world.leaks import leak_tokens


def _caps(t: Any, name: str) -> Any:
    fn = getattr(t, name, None)
    return fn() if callable(fn) else None


@register("A1", 2, cost_class="gpu", detectors=("B.12-kinfu-headroom", "B.10-fresh-cases-crash"))
async def a1_oracle(t: Any, ctx: AcceptContext) -> Out:
    """Oracle: reward 1 on >= 3 grades in the declared environment, replay-identical, gold check, perf headroom, fresh no-crash."""
    p = ctx.policy
    reasons: list[str] = []
    grades = [await ctx.grade(t, "oracle", t.oracle, run=k) for k in range(p.a1_grades)]
    rewards = [r.reward for r, _ in grades]
    fps = [fp for _, fp in grades]
    if any(r != 1.0 for r in rewards):
        reasons.append(f"oracle rewards {rewards} (failing {[r.failing_gates() for r, _ in grades]})")
    if len(set(fps)) != 1:
        reasons.append("oracle replays end in different states")
    start = await need(t, "start_fingerprint")()
    if fps and start == fps[0]:
        reasons.append("the oracle does not change the state (gold check)")
    noop, _ = await ctx.grade(t, "noop", t.noop(), run=0)
    outcome = sorted({c.gate for c in getattr(t, "checks", []) if getattr(c, "gate_kind", "gate") in ("outcome", "score")})
    false_at_start = [g for g in outcome if noop.gates.get(g) != "pass"]
    if outcome and not false_at_start:
        reasons.append("no outcome gate is false at the start state (gold check)")
    m: dict = {"rewards": rewards, "replay_identical": len(set(fps)) == 1, "outcome_gates_false_at_start": false_at_start,
               "declared_env": _caps(t, "declared_env"), "grade_seconds": [r.elapsed_s for r, _ in grades]}
    bars = _caps(t, "perf_bars") or {}
    if bars:
        ratios: dict[str, float] = {}
        for r, _ in grades:
            for k, v in need(t, "oracle_ratios")(r).items():
                ratios[k] = max(v, ratios.get(k, v))  # the oracle's worst grade
        h = calibrate.perf_headroom(ratios, bars, required=p.a1_headroom)
        m["perf_headroom"] = h
        if not h["ok"]:
            reasons.append(f"perf headroom {h['min']} < {p.a1_headroom} (or bars unmeasured: {h['missing']})")
    fresh = _caps(t, "fresh_gates") or []
    if fresh:
        grade_fresh = need(t, "grade_fresh")
        bad = []
        for i in range(p.a1_fresh_grades):
            seed = 7000 + i
            try:
                r, _ = await ctx.grade(t, "oracle", t.oracle, run=seed, variant="fresh", call=lambda s, seed=seed: grade_fresh(t.oracle, seed=seed))
            except InfraUnavailable as e:  # the same seed crashes the grader on every retry: the task's grader, not the node
                bad.append({"seed": seed, "grader_crash": str(e)[:200]})
                continue
            if r.reward != 1.0:
                bad.append({"seed": seed, "reward": r.reward, "failing": r.failing_gates()})
        m["fresh"] = {"grades": p.a1_fresh_grades, "failures": bad}
        if bad:
            reasons.append(f"{len(bad)}/{p.a1_fresh_grades} fresh-seed oracle grades failed (grader crash or wrong reference): {bad[:3]}")
    m["reasons"] = reasons
    return Out.verdict(reasons, m)


@register("A2", 2, cost_class="cpu", depends_on=())
async def a2_floors(t: Any, ctx: AcceptContext) -> Out:
    """Floors: no-op, unmodified original and a naive solution score 0 on every channel, with a clean (non-infra) grade."""
    rows, bad = {}, {}
    sols = [("noop", t.noop())] + [(n.id, need(t, "solution_of")(n)) for n in t.negatives if n.kind == "floor"]
    for sid, sol in sols:
        r, _ = await ctx.grade(t, sid, sol, run=0)
        rows[sid] = {"reward": r.reward, "outcome": r.outcome}
        if r.reward != 0.0 or (ctx.policy.a2_exact_zero and r.outcome not in (0.0, None)):
            bad[sid] = rows[sid]
    naive = [n.id for n in t.negatives if n.kind == "floor" and "naive" in n.id]
    reasons = ([f"floors above 0: {bad}"] if bad else []) + ([] if naive else ["no naive floor"])
    return Out.verdict(reasons, {"floors": rows})


@register("A3", 2, cost_class="gpu", depends_on=("A1",), detectors=("B.4-weak-negatives", "B.13-family-coverage"))
async def a3_mutation(t: Any, ctx: AcceptContext) -> Out:
    """Negatives and mutation score: authored negatives fail their gate, valid alternatives pass, every gate family is covered,
    generated defect-class mutants are killed (score over valid, non-equivalent mutants >= 0.9)."""
    from crucible.accept.mutation import actions

    p = ctx.policy
    reasons, rows = [], {}
    gates_hit: set[str] = set()
    for n in t.negatives:
        r, _ = await ctx.grade(t, n.id, need(t, "solution_of")(n), run=0)
        failing = r.failing_gates()
        if n.expected == "pass":
            ok = r.reward == 1.0
            if not ok:
                reasons.append(f"valid alternative {n.id} fails {failing}: a check grades the oracle's internals, not the property")
        else:
            ok = r.reward == 0.0 and (n.expected_gate is None or n.expected_gate in failing)
            if r.reward == 0.0 and n.expected_gate:
                gates_hit.add(n.expected_gate)
            if not ok:
                reasons.append(f"negative {n.id} ({n.kind}) not killed at {n.expected_gate}: reward {r.reward}, failing {failing}")
        rows[n.id] = {"kind": n.kind, "family": getattr(n, "family", None) or n.kind, "expected_gate": n.expected_gate, "reward": r.reward, "failing": failing, "ok": ok}
    families = sorted({c.gate for c in getattr(t, "checks", [])})
    uncovered = [g for g in families if g not in gates_hit]
    if uncovered:
        reasons.append(f"check families with no negative failing them: {uncovered}")
    need_n = p.a3_min_negatives_h3 if getattr(t, "horizon", "H1") in ("H3", "H4", "H5") else 1
    n_neg = sum(1 for n in t.negatives if n.expected == "fail")
    if n_neg < need_n:
        reasons.append(f"{n_neg} negatives, need >= {need_n} for {t.horizon}")
    if not any(n.kind == "valid_alternative" for n in t.negatives):
        reasons.append("no valid alternative (a different correct solution that must pass)")
    if hasattr(t, "mutants"):
        mutants = t.mutants()
    elif actions.is_actions(t.oracle):
        mutants = actions.generate(t.oracle)
    else:
        mutants = None
        reasons.append("no mutant source for this solution kind (register a mutation operator)")
    table, killed, survivors, equivalent = {}, 0, [], []
    _, oracle_fp = await ctx.grade(t, "oracle", t.oracle, run=0)
    for mu in mutants or []:
        r, fp = await ctx.grade(t, f"mutant:{mu.id}", mu.solution, run=0)
        if r.reward == 0.0:
            killed += 1
            st = "killed"
        elif fp == oracle_fp:
            equivalent.append(mu.id)  # same end state as the oracle: the operator changed nothing the task is about
            st = "equivalent"
        else:
            survivors.append({"id": mu.id, "family": mu.family, "what": mu.description, "reward": r.reward})
            st = "survived"
        table[mu.id] = {"family": mu.family, "status": st, "reward": r.reward}
    scored = killed + len(survivors)
    score = killed / scored if scored else None
    if mutants is not None and scored and score < p.a3_mutation_min:
        reasons.append(f"mutation score {score:.2f} < {p.a3_mutation_min}; survivors (weak checks): {survivors[:5]}")
    per_family: dict[str, dict] = {}
    for v in table.values():
        f = per_family.setdefault(v["family"], {"killed": 0, "survived": 0, "equivalent": 0})
        f[v["status"]] += 1
    return Out.verdict(reasons, {"negatives": rows, "families": families, "mutation": {"score": score, "killed": killed, "survivors": survivors,
                                                                                         "equivalent": equivalent, "per_family": per_family, "table": table}})


@register("A4", 2, cost_class="gpu", depends_on=("A1",), detectors=("B.2-kinfu-bifurcation", "B.2-haar-reference-nondeterminism"))
async def a4_flake(t: Any, ctx: AcceptContext) -> Out:
    """Flakiness: identical full verdicts on >= 8 oracle grades, zero false negatives over fresh-gate draws (+ induced), and
    reference self-consistency over >= 3 reference runs."""
    p = ctx.policy
    reasons, m = [], {}
    vecs = []
    for k in range(p.a4_stable_vectors):
        r, fp = await ctx.grade(t, "oracle", t.oracle, run=k)
        vecs.append((r.reward, tuple(sorted(r.gates.items())), fp))
    distinct = len({v[:2] for v in vecs})
    m["stability"] = {"grades": len(vecs), "distinct_verdicts": distinct, "distinct_end_states": len({v[2] for v in vecs})}
    if distinct != 1:
        reasons.append(f"{distinct} distinct full verdicts over {len(vecs)} oracle grades")
    fresh = _caps(t, "fresh_gates") or []
    if fresh:
        grade_fresh = need(t, "grade_fresh")
        natural = []
        for i in range(p.a4_draws):
            seed = 9000 + i
            r, _ = await ctx.grade(t, "oracle", t.oracle, run=seed, variant="draw",
                                   call=lambda s, seed=seed: grade_fresh(t.oracle, seed=seed, gates=fresh))
            natural.append(all(r.gates.get(g) == "pass" for g in fresh))
        induced = []
        for name, val in (_caps(t, "perturbations") or []):
            r, _ = await ctx.grade(t, "oracle", t.oracle, run=0, variant=f"perturb:{name}={val}",
                                   call=lambda s, name=name, val=val: need(t, "grade_perturbed")(t.oracle, name, val))
            induced.append(all(r.gates.get(g) == "pass" for g in fresh))
        est = calibrate.flake_rate(natural, induced)
        m["flake"] = est.to_dict()
        if not est.ok:
            reasons.append(f"{est.false_negatives} false negatives over {est.draws} draws (upper 95% {est.upper_95})")
    if callable(getattr(t, "reference_fingerprints", None)):
        fps = [await t.reference_fingerprints(i) for i in range(p.a4_reference_runs)]
        sc = calibrate.self_consistency(fps)
        m["reference_self_consistency"] = sc
        if not sc["ok"]:
            reasons.append(f"reference runs disagree: {sc.get('differing') or sc.get('reason')}")
    return Out.verdict(reasons, m)


@register("A5", 2, cost_class="cpu", detectors=("B.13-sibling-overlap",))
async def a5_leak(t: Any, ctx: AcceptContext) -> Out:
    """Leak audit: no hidden string or canary in anything the agent can read (text, workspace, agent image)."""
    visible = need(t, "agent_visible")()
    hidden = need(t, "hidden_strings")()
    hits = [{"string": s[:60], "in": name} for s in hidden if len(s) >= 4 for name, text in visible.items() if s in text]
    tokens = leak_tokens(t.instruction)
    reasons = ([f"hidden material readable by the agent: {hits[:5]}"] if hits else []) + ([f"instruction leak tokens: {tokens}"] if tokens else [])
    m: dict = {"scanned": sorted(visible), "hits": hits[:20], "hidden_strings": len(hidden)}
    canary = getattr(t, "canary", None)
    if canary is not None and canary not in hidden:
        reasons.append("the task's canary is not among the hidden strings (it would never be detected)")
    if hasattr(t, "sibling_overlap"):
        ov = t.sibling_overlap()
        m["sibling_overlap"] = ov
        if ov > 0.2:
            reasons.append(f"sibling-fork overlap {ov:.0%} > 20%")
    imgs = (_caps(t, "images") or {})
    if imgs.get("agent"):
        if Need.DOCKER not in ctx.needs:
            return Out(Status.PENDING, m, ["the agent image audit needs docker"])
        from crucible.verify.leak_audit import audit_image

        rep = audit_image(imgs["agent"], canaries=[canary] if canary else [], strings=hidden)
        m["image"] = {"hits": len(rep.hits), "first": rep.hits[:5]} if hasattr(rep, "hits") else str(rep)[:500]
        if getattr(rep, "hits", None):
            reasons.append(f"hidden material in the agent image: {rep.hits[:3]}")
    return Out.verdict(reasons, m)


@register("A7", 1, cost_class="static")
async def a7_coverage(t: Any, ctx: AcceptContext) -> Out:
    """Instruction <-> check coverage: every graded property is stated, every stated property is graded."""
    if not isinstance(getattr(t, "properties", None), list):
        raise MissingCapability("the task adapter has no stated properties (A7 needs instruction spans <-> check ids)")
    props = {p.id for p in t.properties}
    checks = getattr(t, "checks", [])
    graded = {s for c in checks for s in c.states}
    unstated = [c.id for c in checks if not c.states or not set(c.states) <= props]
    ungraded = sorted(props - graded)
    missing = [p.id for p in t.properties if p.text.strip() and p.text.strip()[:40] not in t.instruction]
    reasons = ([f"checks grading nothing the instruction states: {unstated}"] if unstated else []) + \
              ([f"stated properties no check grades: {ungraded}"] if ungraded else []) + \
              ([f"property sentences missing from the instruction: {missing}"] if missing else [])
    return Out.verdict(reasons, {"properties": sorted(props), "graded": sorted(graded)})


@register("A8", 1, cost_class="static")
async def a8_grounding(t: Any, ctx: AcceptContext) -> Out:
    """Grounding: >= 80% of checks have provenance that resolves; a trap cites its signal and has a trap-reversion negative."""
    from crucible.accept.contract import a8_grounding as legacy

    o = legacy(t, ctx.resolve, min_share=ctx.policy.a8_min_share)
    return Out(Status(o.status), {k: v for k, v in o.detail.items() if k != "reasons"}, o.detail.get("reasons", []))
