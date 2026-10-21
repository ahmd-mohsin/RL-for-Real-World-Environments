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

EQUIVALENT_FAMILIES = frozenset({"drop_read"})  # operators that can leave a task's meaning untouched


def _caps(t: Any, name: str) -> Any:
    fn = getattr(t, name, None)
    return fn() if callable(fn) else None


@register("A1", 3, cost_class="gpu", detectors=("B.12-kinfu-headroom", "B.10-fresh-cases-crash"))
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
    # M7 R6: the grade declares each gate's kind, and that is what credit reads; the check list (``CheckSpec.gate_kind``)
    # is a description of it, so a disagreement is the task's defect, not a reason to skip the gold check
    try:
        kinds = noop.gate_kinds()
    except ValueError as e:
        kinds = {}
        reasons.append(f"the grade does not declare its gates' kinds: {e}")
    stated = {c.gate: c.gate_kind for c in getattr(t, "checks", []) if getattr(c, "gate_kind", None) is not None}
    # outcome vs score is the same for credit (neither is a hard rule); hard vs not is what per-turn credit reads (M6 R2)
    disagree = {g: {"checks": k, "grade": kinds[g]} for g, k in stated.items() if g in kinds and (k == "gate") != (kinds[g] == "gate")}
    if disagree:
        reasons.append(f"the check list's gate kinds disagree with the grade's: {disagree}")
    outcome = sorted(g for g, k in kinds.items() if k in ("outcome", "score"))
    false_at_start = [g for g in outcome if noop.gates.get(g) != "pass"]
    if outcome and not false_at_start:
        reasons.append("no outcome gate is false at the start state (gold check)")
    m: dict = {"rewards": rewards, "replay_identical": len(set(fps)) == 1, "outcome_gates_false_at_start": false_at_start, "gate_kinds": kinds,
               "declared_env": _caps(t, "declared_env"), "grade_seconds": [r.elapsed_s for r, _ in grades]}
    facts = _caps(t, "facts") or {}
    for fact, cap in (("perf_bars", "perf_bars"), ("fresh_gates", "fresh_gates")):
        if facts.get(fact) and not callable(getattr(t, cap, None)):
            raise MissingCapability(f"the task has {fact} but no {cap} capability (M9 audit V6)")
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
            except InfraUnavailable as e:
                if (e.infra or {}).get("failure_kind") != "grader_exception":
                    raise  # a node problem (contention, OOM) on every retry: unmeasured, never the task's failure (M9 audit V14)
                bad.append({"seed": seed, "grader_crash": str(e)[:200]})  # the same seed crashes the task's grader every time
                continue
            if r.reward != 1.0:
                bad.append({"seed": seed, "reward": r.reward, "failing": r.failing_gates()})
        m["fresh"] = {"grades": p.a1_fresh_grades, "failures": bad}
        if bad:
            reasons.append(f"{len(bad)}/{p.a1_fresh_grades} fresh-seed oracle grades failed (grader crash or wrong reference): {bad[:3]}")
    m["reasons"] = reasons
    return Out.verdict(reasons, m)


@register("A2", 3, cost_class="cpu", depends_on=())
async def a2_floors(t: Any, ctx: AcceptContext) -> Out:
    """Floors: no-op, unmodified original and a naive solution score 0 on every channel, with a clean (non-infra) grade."""
    rows, bad = {}, {}
    sols = [("noop", t.noop())] + [(n.id, need(t, "solution_of")(n)) for n in t.negatives if n.kind == "floor"]
    import math

    for sid, sol in sols:
        r, _ = await ctx.grade(t, sid, sol, run=0)
        rows[sid] = {"reward": r.reward, "outcome": r.outcome}
        # a floor's reward must be a finite 0 (NeMo-Gym verifier fixtures: NaN and out-of-range rewards pass a `!= 0` test
        # on some paths and train as signal)
        if not isinstance(r.reward, (int, float)) or not math.isfinite(r.reward) or r.reward != 0.0 \
                or (ctx.policy.a2_exact_zero and r.outcome not in (0.0, None)):
            bad[sid] = rows[sid]
    naive = [n.id for n in t.negatives if n.kind == "floor" and "naive" in n.id]
    reasons = ([f"floors above 0: {bad}"] if bad else []) + ([] if naive else ["no naive floor"])
    return Out.verdict(reasons, {"floors": rows})


@register("A3", 3, cost_class="gpu", depends_on=("A1",), detectors=("B.4-weak-negatives", "B.13-family-coverage"))
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
    gen = getattr(t, "generated_alternatives", None)
    for gid, sol in (gen() if callable(gen) else []):  # built by the contract, not the author (M8 audit V8)
        r, _ = await ctx.grade(t, gid, sol, run=0)
        ok = r.reward == 1.0
        rows[gid] = {"kind": "valid_alternative", "generated": True, "reward": r.reward, "failing": r.failing_gates(), "ok": ok}
        if not ok:
            reasons.append(f"generated alternative {gid} fails {r.failing_gates()}: a check grades wording the instruction does not state")
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
        elif fp == oracle_fp and mu.family in EQUIVALENT_FAMILIES:
            # same end state and an operator that can only be inert (dropping a read): changed nothing the task is about.
            # A dropped message or a reordering also leaves the state alone but breaks a process rule (B.15 forged
            # approval), so those families are never "equivalent" (M9 audit V3)
            equivalent.append(mu.id)
            st = "equivalent"
        else:
            survivors.append({"id": mu.id, "family": mu.family, "what": mu.description, "reward": r.reward})
            st = "survived"
        table[mu.id] = {"family": mu.family, "status": st, "reward": r.reward}
    scored = killed + len(survivors)
    score = killed / scored if scored else None
    if mutants is not None and not scored:  # zero scored mutants is no evidence, never a pass (M9 audit V3)
        reasons.append(f"no scored mutant ({len(mutants)} generated, {len(equivalent)} equivalent): the mutation score is unmeasured")
    if mutants is not None and scored and score < p.a3_mutation_min:
        reasons.append(f"mutation score {score:.2f} < {p.a3_mutation_min}; survivors (weak checks): {survivors[:5]}")
    per_family: dict[str, dict] = {}
    for v in table.values():
        f = per_family.setdefault(v["family"], {"killed": 0, "survived": 0, "equivalent": 0})
        f[v["status"]] += 1
    return Out.verdict(reasons, {"negatives": rows, "families": families, "mutation": {"score": score, "killed": killed, "survivors": survivors,
                                                                                         "equivalent": equivalent, "equivalent_share": round(len(equivalent) / len(table), 3) if table else None,
                                                                                         "per_family": per_family, "table": table}})


@register("A4", 3, cost_class="gpu", depends_on=("A1",), detectors=("B.2-kinfu-bifurcation", "B.2-haar-reference-nondeterminism"))
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
    if (_caps(t, "facts") or {}).get("reference_build") and not callable(getattr(t, "reference_fingerprints", None)):
        # the task has a reference build, so its self-consistency must be measured (M9 audit V6: the Haar detector
        # failed open on every real adapter)
        raise MissingCapability("the task has a reference build but no reference_fingerprints (A4 self-consistency)")
    if callable(getattr(t, "reference_fingerprints", None)):
        fps = [await t.reference_fingerprints(i) for i in range(p.a4_reference_runs)]
        sc = calibrate.self_consistency(fps)
        m["reference_self_consistency"] = sc
        if not sc["ok"]:
            reasons.append(f"reference runs disagree: {sc.get('differing') or sc.get('reason')}")
    return Out.verdict(reasons, m)


@register("A5", 3, cost_class="cpu", detectors=("B.13-sibling-overlap",))
async def a5_leak(t: Any, ctx: AcceptContext) -> Out:
    """Leak audit: no hidden string or canary in anything the agent can read (text, workspace, agent image)."""
    visible = need(t, "agent_visible")()
    hidden = need(t, "hidden_strings")()
    flat = {name: " ".join(text.split()) for name, text in visible.items()}
    hits = [{"string": s[:60], "in": name} for s in hidden if len(s) >= 4 for name, text in visible.items()
            if s in text or " ".join(s.split()) in flat[name]]
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
        # ``LeakReport.findings`` (M9 audit V1: the check read a ``hits`` attribute that does not exist, so it never failed)
        m["image"] = {"findings": len(rep.findings), "first": rep.findings[:5], "scanned_files": rep.scanned_files}
        if rep.findings:
            reasons.append(f"hidden material in the agent image: {rep.findings[:3]}")
    return Out.verdict(reasons, m)


@register("A7", 2, cost_class="static")
async def a7_coverage(t: Any, ctx: AcceptContext) -> Out:
    """Instruction <-> check coverage: every graded property is stated, every stated property is graded."""
    if not isinstance(getattr(t, "properties", None), list):
        raise MissingCapability("the task adapter has no stated properties (A7 needs instruction spans <-> check ids)")
    props = {p.id for p in t.properties}
    checks = getattr(t, "checks", [])
    graded = {s for c in checks for s in c.states}
    unstated = [c.id for c in checks if not c.states or not set(c.states) <= props]
    ungraded = sorted(props - graded)
    flat = " ".join(t.instruction.split())
    missing = [p.id for p in t.properties if p.text.strip() and " ".join(p.text.split()) not in flat]  # the whole sentence (M8 audit V8)
    reasons = ([f"checks grading nothing the instruction states: {unstated}"] if unstated else []) + \
              ([f"stated properties no check grades: {ungraded}"] if ungraded else []) + \
              ([f"property sentences missing from the instruction: {missing}"] if missing else [])
    return Out.verdict(reasons, {"properties": sorted(props), "graded": sorted(graded)})


@register("A8", 2, cost_class="static")
async def a8_grounding(t: Any, ctx: AcceptContext) -> Out:
    """Grounding: >= 80% of checks have provenance that resolves; a trap cites its signal and has a trap-reversion negative."""
    from crucible.accept.contract import a8_grounding as legacy

    o = legacy(t, ctx.resolve, min_share=ctx.policy.a8_min_share)
    return Out(Status(o.status), {k: v for k, v in o.detail.items() if k != "reasons"}, o.detail.get("reasons", []))
