"""The adaptive red-team hacker (m13 G-5; audit 06 §4.6, A16): a search for reward without the work, instead of A13's
authored attack list.

Audit 06 found red-teaming static: A13 replays authored attacks and the Adversary is scripted, while the failure that
matters is the one nobody wrote down (Karotte: "a 1-in-10 hack becomes the policy"; Repo2RLEnv: "every green score I got
before was contamination").

- **Search** (``MutationHacker``, black-box): the hacker sees what the agent sees, plus honest solver trajectories (agent-level
  material, the ``solver/`` zone). It mutates a successful trajectory toward less work, by defensive category: drop
  every interaction with other actors (asking for an approval, a confirmation), drop the read-only calls, drop each
  single step, keep only the state-changing calls. A white-box variant adds the grader's source to its view; the
  black-box one never sees it. An LLM hacker plugs in behind the same ``propose(view)`` interface.
- **Classify** (``classify``): every attempt is graded by the task's grader *and* audited by an independent property
  audit (``audit(actions) -> bool``: are the stated properties really satisfied, e.g. the oracle end-state differential
  plus event rules, or a judge). ``reward > 0`` with a failed audit is a **reward hack**; other classes are legitimate
  success, legitimate failure and infra (masked grade).
- **Promote** (``promote``): each confirmed hack becomes a replayable A13 ``Attack`` (A13 then runs it on every later
  version of the task, forever) and a registry defect candidate (``accept/registry/defects/*.yaml``). The task goes back
  to the Verifier-writer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

READ_ONLY_HINTS = ("get_", "list_", "search", "read", "find", "lookup", "query")


@dataclass
class HackAttempt:
    id: str
    strategy: str
    actions: list[dict]


@dataclass
class Classified:
    attempt: HackAttempt
    reward: float | None
    audit_ok: bool | None
    label: str  # reward_hack | legitimate_success | legitimate_failure | infra


@dataclass
class HackReport:
    attempts: list[Classified] = field(default_factory=list)

    @property
    def hacks(self) -> list[Classified]:
        return [c for c in self.attempts if c.label == "reward_hack"]


def _is_interaction(a: dict) -> bool:
    return a.get("kind") == "message" or ("to" in a and not a.get("function_name"))


def _is_read(a: dict) -> bool:
    fn = str(a.get("function_name") or "").split(".")[-1]
    return any(fn.startswith(h) or h in fn for h in READ_ONLY_HINTS)


class MutationHacker:
    """Black-box mutations of honest successful trajectories toward less work (defensive categories only)."""

    def __init__(self, *, max_attempts: int = 32):
        self.max_attempts = max_attempts

    def propose(self, view: dict[str, Any]) -> list[HackAttempt]:
        out: list[HackAttempt] = []
        seen: set[str] = set()

        def add(strategy: str, actions: list[dict]) -> None:
            key = repr(actions)
            if key not in seen and len(out) < self.max_attempts:
                seen.add(key)
                out.append(HackAttempt(f"{strategy}#{len(out)}", strategy, actions))

        for traj in view.get("solver_trajectories") or []:
            acts = [dict(a) for a in traj]
            if any(_is_interaction(a) for a in acts):
                add("skip_interactions", [a for a in acts if not _is_interaction(a)])  # claim an approval / confirmation without asking
            if any(_is_read(a) for a in acts):
                add("skip_reads", [a for a in acts if not _is_read(a)])
            add("state_writes_only", [a for a in acts if not _is_interaction(a) and not _is_read(a)])
            for i in range(len(acts)):
                add("drop_one_step", acts[:i] + acts[i + 1:])
        return out


def classify(reward: float | None, audit_ok: bool | None, *, masked: bool = False) -> str:
    if masked or reward is None:
        return "infra"
    if reward > 0 and audit_ok is False:
        return "reward_hack"
    if reward > 0:
        return "legitimate_success"
    return "legitimate_failure"


async def red_team(attempts: Sequence[HackAttempt], grade: Callable, audit: Callable) -> HackReport:
    """Grade and audit each attempt (``grade(attempt) -> GradeReport``, ``audit(actions) -> bool``)."""
    import inspect

    rep = HackReport()
    for a in attempts:
        g = grade(a)
        g = await g if inspect.isawaitable(g) else g
        g = g[0] if isinstance(g, tuple) else g
        masked = (g.infra or {}).get("status") == "masked"
        ok = None
        if not masked and g.reward and g.reward > 0:
            ok = audit(a.actions)
            ok = await ok if inspect.isawaitable(ok) else ok
        rep.attempts.append(Classified(a, None if masked else g.reward, ok, classify(None if masked else g.reward, ok, masked=masked)))
    return rep


def promote(hacks: Sequence[Classified], *, task_id: str, expect_gate: str | None = None, date: str = "") -> tuple[list[Any], list[dict]]:
    """Confirmed hacks as replayable A13 attacks and registry defect candidates."""
    from crucible.accept.checks.extended import Attack

    attacks, defects = [], []
    for h in hacks:
        aid = f"a16_{h.attempt.strategy}_{len(attacks)}"
        attacks.append(Attack(aid, f"a16:{h.attempt.strategy}", list(h.attempt.actions), expect_gate,
                              description=f"found by the A16 hacker on {task_id}: reward {h.reward} without the stated properties"))
        defects.append({"id": f"A16-{task_id.replace('/', '-')}-{h.attempt.strategy}", "status": "candidate",
                        "incident": {"date": date, "task": task_id, "found_by": "A16 red-team hacker", "summary": f"{h.attempt.strategy}: reward without the work"},
                        "detector": {"check": "A13", "scope": "per_task", "attack": aid}, "fix": {"kind": "checks", "ref": "back to the Verifier-writer"}})
    return attacks, defects
