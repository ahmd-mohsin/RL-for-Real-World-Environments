"""Policy as data: the Part 2 decisions as versioned rules under ordered principles, and the lawful-remedy procedure that
resolves blocking conditions (DESIGN §7.7 item 24; runs/policy-1.md is the paper's §9.7).

- ``model``: principles, rules, conditions, remedies, escalations; the closed vocabularies (actions, validity
  parameters, escalation grounds).
- ``genesis``: the principles and the decisions already taken, as data.
- ``ledger``: the hash-chained ledger (``ledger.jsonl`` beside this file), its guards, the escalation queue, ``current()``.
- ``resolve``: ``resolve(condition) -> Remedy`` and the principles' tests.

The accessors below are what the code reads instead of constants, so a rule changes in one place (a ledger entry) and
every run records the version it used (``tag()``).
"""

from __future__ import annotations

from typing import Any

from crucible.synth.policy.ledger import EscalationQueue, Policy, PolicyLedger, current, replay
from crucible.synth.policy.model import (
    ACTIONS,
    ESCALATION_KINDS,
    RUNG_ORDER,
    VALIDITY,
    Condition,
    Escalation,
    ParamChange,
    PolicyViolation,
    Principle,
    Remedy,
    Rule,
    Source,
    Step,
    relaxes,
)
from crucible.synth.policy.resolve import CONDITIONS, check, grounds, holds, resolve


def tag() -> str:
    """The current policy version (``policy-v<n>@<sha>``): recorded in certificates and run reports."""
    return current().tag


def param(rule: str, name: str) -> Any:
    return current().param(rule, name)


def admission_params(scope: str = "tool_world") -> dict[str, Any]:
    """R-admission-fixed's parameters, with R-patch-scope's recorded exemption applied for ``scope="repository"``."""
    pol = current()
    p = dict(pol.rule("R-admission-fixed").params)
    if scope != "tool_world":
        for r in pol.rules.values():
            if r.params.get("scope") == scope and r.id != "R-admission-fixed":
                p.update({k: v for k, v in r.params.items() if k != "scope"})
    return p


def admission_deviations(policy: Any, scope: str = "tool_world") -> list[str]:
    """Where an ``AdmissionPolicy`` (or its dict) is looser than the policy's admission rule for its scope."""
    d = policy.model_dump(mode="json") if hasattr(policy, "model_dump") else dict(policy)
    p = admission_params(scope)
    rid = "R-admission-fixed"
    out = []
    u = (d.get("u_max") or {}).get("MUT")
    if u is None or relaxes(rid, "u_max_mut", p["u_max_mut"], u):
        out.append(f"u_max.MUT {u} (policy {p['u_max_mut']})")
    if relaxes(rid, "max_confirmed", p["max_confirmed"], d.get("max_confirmed", 0)):
        out.append(f"max_confirmed {d.get('max_confirmed')}")
    if relaxes(rid, "atk_max_confirmed", p["atk_max_confirmed"], d.get("atk_max_confirmed", 0)):
        out.append(f"atk_max_confirmed {d.get('atk_max_confirmed')}")
    missing = sorted(set(p["zero_confirmed_classes"]) - set(d.get("deterministic") or []))
    if missing:
        out.append(f"classes without the zero-confirmed rule: {missing}")
    missing = sorted(set(p["require_applicable"]) - set(d.get("require_applicable") or []))
    if missing:
        out.append(f"classes not required to apply: {missing}")
    if p["require_alt_pass"] and not d.get("require_alt_pass", True):
        out.append("require_alt_pass off")
    if d.get("allow_open_suspects") and not p["allow_open_suspects"]:
        out.append("open suspects allowed")
    return out


def panel_calibration() -> dict[str, Any]:
    """R-panel as ``falsify.agents.calibrate`` keywords (the strict rule: an uncalibrated panel abstains)."""
    r = current().rule("R-panel").params
    return {"min_decided": int(r["min_decided"]), "require_both": bool(r["require_both"]), "refuse_uncalibrated": bool(r["refuse_uncalibrated"]),
            "target": float(r["target"]), "floor": float(r["floor"])}


def holdout() -> float:
    """R-holdout's h (fixed; never lowered)."""
    return float(current().param("R-holdout", "h"))


def stage_cap(stage: str) -> float:
    return float(current().param("R-budget", "stage_caps_usd")[stage])


def part2_total() -> float:
    return float(current().param("R-budget", "part2_total_usd"))


def scale_target() -> tuple[int, int]:
    """R-scale: (environments, tasks per environment)."""
    p = current().rule("R-scale").params
    return int(p["environments"]), int(p["tasks_per_environment"])


__all__ = [
    "ACTIONS", "CONDITIONS", "ESCALATION_KINDS", "RUNG_ORDER", "VALIDITY", "Condition", "Escalation", "EscalationQueue", "ParamChange",
    "Policy", "PolicyLedger", "PolicyViolation", "Principle", "Remedy", "Rule", "Source", "Step", "admission_deviations",
    "admission_params", "check", "current", "grounds", "holdout", "holds", "panel_calibration", "param", "part2_total", "relaxes",
    "replay", "resolve", "scale_target", "stage_cap", "tag",
]
