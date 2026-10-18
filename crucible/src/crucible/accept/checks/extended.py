"""A6, A9-A13 and S1 on the contract runner (DESIGN §9; research m9-m10/01 §14).

Adapter capabilities used (all optional; a check whose capability is missing fails, see ``base``):

- A6  ``tamper_attempts() -> [Attack]`` (each must be *blocked*: reward 0 and its effect absent) and ``isolation_audit()``.
- A9  ``images()`` resolved to digests and equal to ``pinned_images()``; ``recorded_fingerprints()`` vs a fresh
      ``reference_fingerprints(0)``; ``static_lint()``; World tasks: two fresh starts give one fingerprint.
- A10 ``calibration_runs()`` (recorded M10 records) or ``calibrate(budget_usd)`` (paid); ``split``.
- A11 ``runtime_task()``: an M10 ``RunTask``; the B.5 injectors run against it through the M10 runner.
- A12 ``verifier_timeout_s``; ``plan_tiers()`` and ``grade_tier(solution, tier)``.
- A13 ``attacks() -> [Attack]`` (each must *take effect* — the control — and still score 0 at its gate).
- S1  ``signoff.yaml`` bound to the acceptance key, for H4+.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from crucible.accept.checks.base import AcceptContext, Out, need, register
from crucible.accept.model import Need, Status
from crucible.forge.schema import Attack  # noqa: F401 - moved to forge.schema (M0 audit D3); re-exported here


def _caps(t: Any, name: str) -> Any:
    fn = getattr(t, name, None)
    return fn() if callable(fn) else None


@register("A6", 2, cost_class="docker", depends_on=("A1",), detectors=("B.9-policy-root", "B.8-verifier-staging"))
async def a6_isolation(t: Any, ctx: AcceptContext) -> Out:
    """Isolation: the containment audit passes in the declared environment and every tamper attempt is blocked."""
    reasons, m = [], {}
    audit = await need(t, "isolation_audit")()
    m["audit"] = audit
    bad = {k: v for k, v in audit.items() if v not in ("pass", "skip")}
    if bad:
        reasons.append(f"containment audit failures: {bad}")
    if not any(v == "pass" for v in audit.values()):  # an audit of skips proved nothing (M9 audit V9)
        reasons.append(f"the containment audit passed no item ({len(audit)} items, all skipped)")
    rows = {}
    for a in need(t, "tamper_attempts")():
        r, _ = await ctx.grade(t, f"tamper:{a.id}", a.solution, run=0)
        took = await a.effect() if a.effect else None
        rows[a.id] = {"family": a.family, "reward": r.reward, "took_effect": took}
        if a.effect is None:  # without a control a blocked tamper and a tamper that never ran look the same (V9)
            reasons.append(f"tamper {a.id} has no effect probe: it cannot show it was blocked")
        elif r.reward != 0.0 or took:
            reasons.append(f"tamper {a.id} not blocked (reward {r.reward}, effect {took})")
    if not rows:
        reasons.append("no tamper attempt was run (A6 needs at least one: write grader state, plant a reward)")
    m["tamper"] = rows
    return Out.verdict(reasons, m)


@register("A9", 2, cost_class="static", detectors=("B.5-stale-verifier-image", "B.7-harbor-taskconfig"))
async def a9_repro(t: Any, ctx: AcceptContext) -> Out:
    """Reproducibility: images pinned by digest, reference fingerprints reproduce the recorded ones, static task lint clean."""
    reasons, m = [], {}
    images = ctx.key.images
    unresolved = sorted(r for r, d in images.items() if d.startswith("unresolved:"))
    if unresolved:
        reasons.append(f"image references that do not resolve to a digest: {unresolved}")
    pinned = _caps(t, "pinned_images") or {}
    unpinned = sorted(r for r in images if r not in pinned)
    if unpinned:  # an image with no pinned digest cannot show it is not stale (M9 audit V6: the detector failed open)
        reasons.append(f"images with no pinned digest (stale-image check impossible): {unpinned}")
    drift = {r: {"pinned": d, "actual": images.get(r)} for r, d in pinned.items() if images.get(r) != d}
    if drift:
        reasons.append(f"images differ from the pinned digests (stale image): {drift}")
    m["images"], m["pinned"] = images, pinned
    if callable(getattr(t, "recorded_fingerprints", None)):
        rec, now = t.recorded_fingerprints(), await need(t, "reference_fingerprints")(0)
        diff = sorted(k for k in set(rec) | set(now) if rec.get(k) != now.get(k))
        m["fingerprints"] = {"keys": len(rec), "differing": diff[:20]}
        if diff:
            reasons.append(f"reference fingerprints differ from the recorded ones: {diff[:5]}")
    lint = _caps(t, "static_lint") or []
    m["lint"] = lint
    if lint:
        reasons.append(f"static task lint: {lint[:5]}")
    if not images:  # a World task: the start state must be reproducible from the spec alone
        a, b = await need(t, "start_fingerprint")(), await t.start_fingerprint()
        m["start_fingerprints_identical"] = a == b
        if a != b:
            reasons.append("two fresh starts give different start states")
    return Out.verdict(reasons, m)


@register("A10", 2, cost_class="paid", depends_on=("A1",))
async def a10_calibration(t: Any, ctx: AcceptContext) -> Out:
    """Calibration: >= 2 frontier runs with a failure taxonomy; the pass rate is inside the band of the task's split."""
    from crucible.forge.learnability import BANDS, estimate

    runs = list(_caps(t, "calibration_runs") or [])
    if len([r for r in runs if not r.get("masked")]) < ctx.policy.a10_min_runs and callable(getattr(t, "calibrate", None)):
        if Need.PAID not in ctx.needs or ctx.budget_usd <= 0:
            return Out(Status.PENDING, {"recorded": len(runs)}, ["needs paid frontier runs (no budget in this run)"])
        runs += await t.calibrate(ctx.budget_usd)
    scored = [r for r in runs if not r.get("masked")]
    reasons, m = [], {"runs": runs, "masked_excluded": len(runs) - len(scored)}
    if len(scored) < ctx.policy.a10_min_runs:
        return Out(Status.PENDING, m, [f"{len(scored)} scored frontier runs, need {ctx.policy.a10_min_runs}"])
    untaxed = [r.get("policy") for r in scored if r.get("reward", 0) < 1.0 and not r.get("failing")]
    if untaxed:
        reasons.append(f"failed runs without a failure taxonomy (failing gates): {untaxed}")
    split = getattr(t, "split", "train")
    band = BANDS["bench" if split == "bench" else "train"]  # one band definition with the ladder (sweep G4 D15)

    def rate_of(rs: list[dict]):
        # success is the task's own: a record-relative score above 1 is a success (sweep G2 D11: 1.3 counted as a failure)
        return estimate([(r.get("reward") or 0) >= 1.0 for r in rs], model=",".join(sorted({r.get("policy", "?") for r in rs})), harness="mixed")

    # The sample size is fixed in advance (``a10_max_runs`` when the task can sample): stopping as soon as the estimate
    # entered the band biased the check toward acceptance (M9 audit V15). Without paid sampling it decides on what exists.
    rate = rate_of(scored)
    can_sample = callable(getattr(t, "calibrate", None)) and Need.PAID in ctx.needs and ctx.budget_usd > 0 \
        and "k" in inspect.signature(t.calibrate).parameters
    rounds = 0
    while can_sample and len(scored) < ctx.policy.a10_max_runs:
        target = min(ctx.policy.a10_max_runs, 2 * len(scored))
        more = [r for r in await t.calibrate(ctx.budget_usd, k=target) if not r.get("masked")]
        rounds += 1
        if len(more) <= len(scored):  # no new evidence (budget gone, runs masked): decide on what there is
            break
        scored, rate = more, rate_of(more)
    m.update({"split": split, "band": [band.lo, band.hi, band.on], "p": rate.p, "interval": [rate.lo, rate.hi], "n": rate.n, "topup_rounds": rounds})
    lad = getattr(t, "ladder", None)  # m13 G-4: a solver-ladder report; an inversion (a weaker rung clearly above a stronger) blocks
    if callable(lad):
        rep = lad()
        rep = await rep if inspect.isawaitable(rep) else rep
        if rep is not None:
            m["ladder"] = {k: {"p": v.p, "interval": [v.lo, v.hi], "n": v.n} for k, v in rep.rates.items()}
            reasons += [f"solver ladder inversion: {x}" for x in rep.inversions]
    if band.on == "hi":  # bench: the frontier's upper bound must be below the band (0/2 has an upper bound of 0.66)
        if rate.hi > (band.hi or 1.0):
            reasons.append(f"frontier pass-rate upper bound {rate.hi:.2f} > {band.hi} (n={rate.n}): not shown hard enough for bench")
    elif not (band.lo or 0.0) <= rate.p <= (band.hi or 1.0):
        clear = rate.hi < (band.lo or 0.0) or rate.lo > (band.hi or 1.0)
        how = "interval clear of the band" if clear else f"interval still overlaps it at the sampling cap (n={rate.n})"
        reasons.append(f"pass rate {rate.p:.2f} outside the {split} band [{band.lo}, {band.hi}] ({how})")
    return Out.verdict(reasons, m)


@register("A11", 2, cost_class="cpu", depends_on=("A1",), detectors=("B.5-empty-patch", "B.5-blank-usage", "B.5-credits", "B.5-zombie"))
async def a11_harness(t: Any, ctx: AcceptContext) -> Out:
    """Harness hygiene: the B.5 failure classes injected into this task's runtime path are classified and masked, never
    scored; the controls (oracle, no-op) are scored 1 and 0."""
    from crucible.runtime.harness.native import NativeHarness, PolicyRef
    from crucible.runtime.inject import PROVIDER_FAILURES, FakeLLM, FlakyGradeTask
    from crucible.runtime.run import Runner, RunSpec

    rt = need(t, "runtime_task")()
    root = Path(ctx.workdir or ".") / "a11"
    runs: dict[str, Any] = {}

    def llm_ref(name: str, fail: str | None) -> PolicyRef:
        from crucible.actors.policy import LLMToolPolicy

        def make(a):
            llm = FakeLLM(_oracle_script(t.oracle), fail_at=(0, PROVIDER_FAILURES[fail]) if fail and a.n == 1 else None)
            return {"agent": LLMToolPolicy("agent", llm)}

        return PolicyRef(id=f"a11:{name}", make=make, estimate_usd=0.0)

    def scripted(name: str, actions: list) -> PolicyRef:
        from crucible.actors.policy import ScriptedPolicy
        from crucible.core.schema.actions import Message, ToolCall

        acts = [ToolCall(tool_call_id=a.get("tool_call_id") or f"o{i}", function_name=a["function_name"], arguments=a.get("arguments") or {})
                if a.get("function_name") else Message(to=a["to"], text=a.get("text", "")) for i, a in enumerate(actions) if a.get("function_name") or a.get("to")]
        return PolicyRef(id=f"a11:{name}", make=lambda a: {"agent": ScriptedPolicy("agent", list(acts))})

    # the oracle must fit the episode: a step cap below its length would score the control 0 (M14 audit: finance_ap
    # oracles take 37–85 steps against the default 30, so A11's "oracle is scored 1" control failed)
    import dataclasses as _dc

    base = NativeHarness()
    harness = NativeHarness(limits=_dc.replace(base.limits, max_steps=max(int(base.limits.max_steps or 30), len(list(t.oracle or [])) + 10)))

    async def one(name, task, pol):
        s = await Runner(root).run(RunSpec(run_id=name, tasks=[task], policies=[pol], harnesses=[harness], k=1))
        import json

        f = root / name / "failures.jsonl"
        fails = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
        j = root / name / "journal.jsonl"
        ev = [json.loads(x) for x in j.read_text().splitlines() if x.strip()] if j.exists() else []
        runs[name] = {"stopped": s.stopped, "scored": s.scored, "mean": s.mean_reward, "attempts": s.attempts,
                      "failure_kinds": [x["failure"]["kind"] for x in fails],
                      # M10 lab audit R-3: how often the policy was sampled (a grade-only retry is dispatched too, runs no agent)
                      "agent_dispatches": sum(1 for e in ev if e.get("event") == "dispatched" and not e.get("grade_only"))}
        return s, fails

    def graded_again(name: str, s: Any) -> bool:
        """M10 lab audit R-3: a failed grade was graded again on the same work and the policy was sampled once. A separate
        verifier persists its hand-off and a grade-only retry re-grades it (the oracle's 1.0); graded in place, the World is
        gone after the attempt, so the grade is retried inside it (``verifier/grades.json``; the injection under ``/a1/``
        masks that try too) and the attempt ends terminal-masked. Before R-3 the case passed only through an a2 that re-ran
        the agent, which R-3 removes (every in-place task failed A11)."""
        if runs[name]["agent_dispatches"] != 1:
            return False
        if s.mean_reward == 1.0:
            return True
        adir = next((root / name / "samples").glob("*/a1"), None)
        return s.scored == 0 and s.masked_terminal == 1 and s.attempts == 1 and adir is not None and (adir / "verifier" / "grades.json").exists()

    expect = []
    s, f = await one("blank_usage", rt, llm_ref("blank", "usage_blank"))
    expect.append(("blank usage masked then retried", bool(f) and f[0]["failure"]["kind"] == "crucible:usage_blank" and s.scored == 1))
    s, f = await one("credits", rt, llm_ref("credits", "credits"))
    expect.append(("credit exhaustion stops the run, resumable, nothing scored", bool(s.stopped) and "credits" in s.stopped and s.scored == 0))
    # M10 lab audit R-3: a masked grade is graded again on the same work, never by sampling the policy again (graded_again)
    s, f = await one("verifier_contention", FlakyGradeTask(rt, "perf rounds contaminated (spread 0.9)"), scripted("oracle", t.oracle))
    expect.append(("contaminated grade masked as node_contention, re-graded on the same work, policy not re-sampled",
                   bool(f) and f[0]["failure"]["kind"] == "crucible:node_contention" and graded_again("verifier_contention", s)))
    s, f = await one("grader_crash", FlakyGradeTask(rt, "grader setup crashed", crash_setup=True), scripted("oracle2", t.oracle))
    expect.append(("grader crash masked as verifier_error, re-graded on the same work, policy not re-sampled",
                   bool(f) and f[0]["failure"]["kind"] == "verifier_error" and graded_again("grader_crash", s)))
    s, f = await one("control_noop", rt, scripted("noop", []))
    expect.append(("control: no-op is scored 0, not masked", s.scored == 1 and s.mean_reward == 0.0 and not f))
    s, f = await one("control_oracle", rt, scripted("oracle3", t.oracle))
    expect.append(("control: oracle is scored 1", s.scored == 1 and s.mean_reward == 1.0))
    reasons = [name for name, ok in expect if not ok]
    return Out.verdict([f"misclassified: {r}" for r in reasons], {"cases": dict(expect), "runs": runs})


def _oracle_script(oracle: list[dict]):
    from crucible.actors.llm import LLMResponse

    out = []
    for i, a in enumerate(oracle):
        if a.get("function_name"):
            out.append(LLMResponse(text="", tool_calls=[{"id": f"a11c{i}", "name": a["function_name"].replace(".", "__", 1), "arguments": a.get("arguments") or {}}],
                                   usage={"prompt_tokens": 10, "completion_tokens": 5}))
        elif a.get("to"):
            out.append(LLMResponse(text="", tool_calls=[{"id": f"a11c{i}", "name": "send_message", "arguments": {"to": a["to"], "text": a.get("text", "")}}],
                                   usage={"prompt_tokens": 10, "completion_tokens": 5}))
    out.append(LLMResponse(text="Done.", usage={"prompt_tokens": 10, "completion_tokens": 2}))
    return out


@register("A12", 2, cost_class="cpu", depends_on=("A1", "A3"))
async def a12_cost_tiers(t: Any, ctx: AcceptContext) -> Out:
    """Cost and tiers: the slowest full grade fits the verifier timeout with margin; the fast tier agrees with the full tier
    (Spearman >= 0.8) on a graded calibration set and kills every negative the full tier kills."""
    from crucible.verify.tiers import agreement

    reasons, m = [], {}
    # grade through the store itself: with A1/A3 reused from the cache ``ctx.grades`` is empty (M9 audit V11)
    await ctx.grade(t, "oracle", t.oracle, run=0)
    for n in t.negatives:
        await ctx.grade(t, n.id, need(t, "solution_of")(n), run=0)
    secs = [r.elapsed_s for (sid, _run, var), (r, _) in ctx.grades.items() if var == "" and sid == "oracle"]
    timeout = float(getattr(t, "verifier_timeout_s", 3600.0))
    m["full_grade_s"] = {"max": max(secs) if secs else None, "timeout": timeout, "margin": ctx.policy.a12_timeout_margin}
    if not secs:
        reasons.append("no full oracle grade timing available")
    elif max(secs) > ctx.policy.a12_timeout_margin * timeout:
        reasons.append(f"slowest full grade {max(secs):.0f}s exceeds {ctx.policy.a12_timeout_margin:.0%} of the {timeout:.0f}s timeout")
    tiers = _caps(t, "plan_tiers") or {}
    if not tiers.get("fast") or tiers.get("fast") == tiers.get("full"):
        if reasons:
            return Out.verdict(reasons, m)
        return Out(Status.NA, m, [], na_reason="single_tier")
    grade_tier = need(t, "grade_tier")
    fast, full, fpff = [], [], []
    for (sid, run, var), (r, _) in list(ctx.grades.items()):
        if var or run != 0:
            continue
        sol = t.oracle if sid == "oracle" else next((need(t, "solution_of")(n) for n in t.negatives if n.id == sid), None)
        if sol is None:
            continue
        fr = await grade_tier(sol, "fast")
        fast.append(fr.reward)
        full.append(r.reward)
        if fr.reward > 0 and r.reward == 0:
            fpff.append(sid)
    ag = agreement(fast, full, required=ctx.policy.a12_spearman)
    m["tiers"] = {**ag, "n": len(fast), "fast_pass_full_fail": fpff}
    if not ag.get("ok"):
        reasons.append(f"fast/full Spearman {ag.get('spearman')} < {ctx.policy.a12_spearman}")
    if fpff:
        reasons.append(f"negatives that pass the fast tier but fail the full tier (a training-time hack): {fpff}")
    return Out.verdict(reasons, m)


@register("A13", 2, cost_class="docker", depends_on=("A1",), detectors=("B.10-doctest-disable", "B.10-os-exit", "B.12-racecheck-first-summary"))
async def a13_attacks(t: Any, ctx: AcceptContext) -> Out:
    """Grader attack suite: every attack the task's grading path is exposed to takes effect (its control) and still scores 0
    at its gate, with a clean (non-masked) grade."""
    attacks = need(t, "attacks")()
    if not attacks:
        kind = (_caps(t, "declared_env") or {}).get("kind")
        # a World or container task always has an agent surface, whatever the adapter asserts (M9 audit V10)
        if getattr(t, "attack_surface", None) == "none" and kind not in ("world", "container", "harbor"):
            return Out(Status.NA, {}, [], na_reason="no_agent_surface")
        return Out(Status.FAIL, {}, ["no attacks instantiated for this task"])
    rows, reasons = {}, []
    for a in attacks:
        r, _ = await ctx.grade(t, f"attack:{a.id}", a.solution, run=0)
        took = await a.effect() if a.effect else None
        failing = r.failing_gates()
        rows[a.id] = {"family": a.family, "reward": r.reward, "took_effect": took, "failing": failing, "expect_gate": a.expect_gate}
        if a.effect is None:  # no control: a wrong solution looks like a caught attack (M9 audit V9)
            reasons.append(f"attack {a.id} has no control (effect probe): it cannot show the attack took effect")
        elif took is False:
            reasons.append(f"attack {a.id} did not take effect: the test proves nothing (fix the attack)")
        elif r.reward != 0.0 or (a.expect_gate and a.expect_gate not in failing):
            reasons.append(f"attack {a.id} ({a.family}) not caught at {a.expect_gate}: reward {r.reward}, failing {failing}")
    confirmed = sum(1 for v in rows.values() if v["reward"] > 0 and v["took_effect"] is not False)
    return Out.verdict(reasons, {"attacks": rows, "run": len(rows), "confirmed_hacks": confirmed})


def _s1_applies(t: Any) -> str | None:
    return None if getattr(t, "horizon", "H1") in ("H4", "H5") else "horizon<H4"


@register("S1", 1, cost_class="human", applies=_s1_applies)
async def s1_signoff(t: Any, ctx: AcceptContext) -> Out:
    """Human sign-off for H4+: ``signoff.yaml`` exists, approves, and is bound to the current acceptance key."""
    import yaml

    p = Path(ctx.signoff_root or ".") / t.task_id.replace("/", "__") / "signoff.yaml"
    if not p.exists():
        return Out(Status.PENDING, {"path": str(p)}, ["no sign-off yet (crucible accept --signoff-packet)"])
    d = yaml.safe_load(p.read_text()) or {}
    m = {k: d.get(k) for k in ("reviewer", "date", "key_digest", "decision")}
    if d.get("key_digest") != ctx.key_digest:
        return Out(Status.PENDING, m, [f"sign-off is for key {d.get('key_digest')}, current key is {ctx.key_digest}"])
    return Out.verdict([] if d.get("decision") == "accept" else [f"reviewer decision: {d.get('decision')}"], m)
