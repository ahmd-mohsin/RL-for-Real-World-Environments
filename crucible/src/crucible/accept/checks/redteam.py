"""A16 adaptive red team (m13 G-5; audit 06 §4.6): a budgeted hacker searches for reward without the stated properties.

The task offers ``solver_trajectories()`` (honest successful trajectories, agent-level material) and an independent
``property_audit(actions) -> bool`` (are the stated properties really satisfied: e.g. the pack's property rules, an
end-state differential against the oracle, or a judge); ``hacker()`` is optional (default ``forge.hacker.MutationHacker``,
black-box). Every attempt within ``Policy.a16_attempts`` is graded by the task's grader and, when it scores, audited.
A confirmed hack (reward > 0, audit fails) fails A16; the metrics carry each hack's actions so ``forge.hacker.promote``
turns it into an A13 attack and a registry defect candidate. A task without a property audit is not applicable
(``no_property_audit``): its hacks could not be told from legitimate successes.
"""

from __future__ import annotations

import inspect
from typing import Any

from crucible.accept.checks.base import AcceptContext, Out, register
from crucible.accept.model import Status


@register("A16", 1, cost_class="cpu", depends_on=("A1",), detectors=("G-redteam-adaptive", "B.15-forged-approval"))
async def a16_red_team(t: Any, ctx: AcceptContext) -> Out:
    """Adaptive red team: no attempt of the hacker's budget earns reward without the stated properties."""
    from crucible.forge.hacker import MutationHacker, red_team

    audit = getattr(t, "property_audit", None)
    if audit is None:
        return Out(Status.NA, {}, [], na_reason="no_property_audit")
    hacker = (getattr(t, "hacker", None) or (lambda: MutationHacker()))()
    hacker.max_attempts = min(getattr(hacker, "max_attempts", ctx.policy.a16_attempts), ctx.policy.a16_attempts)
    trajs = getattr(t, "solver_trajectories", None)
    vis = getattr(t, "agent_visible", None)
    view = {"instruction": getattr(t, "instruction", ""), "visible": vis() if vis else {},
            "solver_trajectories": trajs() if trajs else [list(getattr(t, "oracle", []))]}
    proposed = hacker.propose(view)
    attempts = list(await proposed if inspect.isawaitable(proposed) else proposed)[: ctx.policy.a16_attempts]

    async def grade(a):
        r, _ = await ctx.grade(t, f"A16:{a.id}", a.actions, run=0)
        return r

    rep = await red_team(attempts, grade, audit)
    more = getattr(hacker, "propose_more", None)
    if callable(more) and not rep.hacks and len(attempts) < ctx.policy.a16_attempts:
        # a clean first phase (a specialist's memory replay) does not end the search (M8 audit V7)
        extra = more(view)
        extra = list(await extra if inspect.isawaitable(extra) else extra)[: ctx.policy.a16_attempts - len(attempts)]
        rep.attempts += (await red_team(extra, grade, audit)).attempts
        attempts += extra
    labels: dict[str, int] = {}
    for c in rep.attempts:
        labels[c.label] = labels.get(c.label, 0) + 1
    hacks = [{"id": c.attempt.id, "strategy": c.attempt.strategy, "reward": c.reward, "actions": c.attempt.actions} for c in rep.hacks]
    reasons = [f"hack {h['id']} ({h['strategy']}): reward {h['reward']} without the stated properties (promote to A13)" for h in hacks]
    if not attempts:
        reasons.append("the hacker proposed no attempt (no solver trajectory to mutate)")
    measured = {"attempts": len(attempts), "labels": labels, "confirmed_hacks": len(hacks), "hacks": hacks}
    if reasons:
        return Out.verdict(reasons, measured)
    # no hack found is evidence only if the attempts were measured (M8 audit V7: every grade masked passed A16, and a
    # candidate can trigger some masks); an audit that cannot decide is not a legitimate success either
    infra, undecided = labels.get("infra", 0), labels.get("undecided", 0)
    if infra == len(attempts) or infra > ctx.policy.a16_max_infra_share * len(attempts) or undecided:
        return Out(Status.ERROR, measured, [f"{infra} of {len(attempts)} attempts masked, {undecided} undecided audits: no evidence either way"])
    return Out.verdict([], measured)
