"""Re-path stored episodes (swarm-pipeline/07 §6.1, 00 §5.2): the path checks recomputed under the current rules from what
an attempt left behind, with no model call and no test run.

A path rule reads only the trail (``trace.json``) and the collected diff (``artifacts/model.patch``); the end state (the
hidden tests' milestone vector, Phi, the end-state full pass) does not depend on it. So a path-rule change is re-applied to
a stored episode exactly: the window's own rules (``windows.path_rules_of``, from the stored ``window.json`` and its start
tree) run ``pathcheck.check_paths`` on the stored trail and diff, inside the window's language scope (a non-Python profile
names its own plumbing), and ``windows.path_reward`` recomputes the reward from the stored end state. A changed verdict is
a ``flip``: the report lists every one with the old and new install, network and history evidence.

A rule that needs the tests to run again (a protected-path restore, a new hidden test) is not a path rule:
``factory.tinker.regrade`` re-grades in the World.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.pathcheck import LearnedRule, PathRules, check_paths, rules_version, trail_from_trace

TRAIL_CHECKS = ("path:install", "path:network", "path:history")


@dataclass
class Repath:
    attempt: str
    task: str
    arm: str
    old: dict[str, Any]
    new: dict[str, Any]
    changed: list[str] = field(default_factory=list)  # check ids whose status changed

    @property
    def flipped(self) -> bool:
        return self.old.get("reward") != self.new.get("reward") or bool(self.changed)

    def to_json(self) -> dict[str, Any]:
        return {"attempt": self.attempt, "task": self.task, "arm": self.arm, "old": self.old, "new": self.new, "changed": self.changed,
                "flipped": self.flipped}


def _load(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def task_of(attempt: Path) -> str:
    """``.../arms/<arm>/<task>/rollouts/...``: the task id an attempt belongs to."""
    parts = attempt.parts
    return parts[parts.index("rollouts") - 1] if "rollouts" in parts else ""


def arm_of(attempt: Path) -> str:
    parts = attempt.parts
    return parts[parts.index("rollouts") - 2] if "rollouts" in parts else ""


def rules_for(window: Mapping[str, Any], base_dir: Path, **over: Any) -> PathRules:
    from dataclasses import replace

    from crucible.factory.lanes.windows import path_rules_of

    r = path_rules_of(window, base_dir)
    return replace(r, **over) if over else r


def repath(attempt: Path, window: Mapping[str, Any], base_dir: Path, *, rules: PathRules | None = None, phi_weight: float = 0.5,
           learned: Sequence[LearnedRule] | None = None) -> Repath | None:
    """The attempt's path checks and reward under ``rules`` (default: the window's current rules) and the learned rules
    (default: the committed ones). None when the attempt has no graded end state (masked, or no report) or no trail."""
    from crucible.factory.generators.debug import scope
    from crucible.factory.lanes.windows import path_reward

    rep = _load(attempt / "verifier" / "report.json") or {}
    meta = rep.get("meta") or {}
    trace = _load(attempt / "trace.json")
    if (rep.get("infra") or {}).get("status") != "ok" or "phi" not in meta or not trace:
        return None
    patch_p = attempt / "artifacts" / "model.patch"
    patch = patch_p.read_text(errors="replace") if patch_p.exists() else ""
    with scope(str(window.get("lang") or "python")):
        r = rules or rules_for(window, base_dir)
        pr = check_paths(r, patch=patch, trail=trail_from_trace(trace), trail_source="trace.json", learned=learned)
        pw = path_reward(meta, pr, float(meta.get("phi_weight") or phi_weight))
    old_gates = {k: v for k, v in (rep.get("gates") or {}).items() if k.startswith("path:")}
    new_gates = pr.gates()
    # a check the old report did not run counts only when it now fails (a passing new check changed no verdict)
    changed = sorted(k for k in set(old_gates) | set(new_gates)
                     if old_gates.get(k) != new_gates.get(k) and not (k not in old_gates and new_gates.get(k) == "pass"))

    def ev(checks: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
        return {c["id"]: list(c.get("evidence") or [])[:3] for c in checks
                if (c.get("id") in TRAIL_CHECKS or str(c.get("id")).startswith("path:learned:")) and c.get("evidence")}

    old_checks = ((meta.get("path") or {}).get("checks")) or []
    return Repath(str(attempt), task_of(attempt), arm_of(attempt),
                  {"reward": rep.get("reward"), "full_pass": meta.get("full_pass"), "phi": meta.get("phi"), "gates": old_gates, "evidence": ev(old_checks),
                   "rules_version": (meta.get("path") or {}).get("rules_version")},
                  {"reward": pw["outcome_reward"], "full_pass": pw["full_pass"], "phi": meta.get("phi"), "gates": new_gates,
                   "evidence": ev(pr.to_json()["checks"])}, changed)


def attempts(roots: Iterable[Path]) -> list[Path]:
    """Every graded attempt under ``roots`` (``.../rollouts/<run>/samples/<sample>/a<n>`` with a verifier report)."""
    out: list[Path] = []
    for root in roots:
        out += sorted(p.parent.parent for p in Path(root).rglob("verifier/report.json") if "rollouts" in p.parts)
    return out


def scan(roots: Iterable[Path], store: Path, *, rules_over: Mapping[str, Any] | None = None, learned: Sequence[LearnedRule] | None = None) -> dict[str, Any]:
    """Re-path every stored attempt under ``roots`` against the window records in ``store`` (``<store>/<split>/tasks/<id>``:
    ``window.json`` and its start tree ``base/``), with ``learned`` rules (default: the committed ones). Returns the flips
    (reward or a path verdict changed) and a summary."""
    from crucible.factory.pathcheck import LEARNED

    version = rules_version(LEARNED if learned is None else learned)
    windows: dict[str, tuple[dict[str, Any], Path] | None] = {}
    flips: list[dict[str, Any]] = []
    seen = skipped = stale = 0
    for a in attempts(roots):
        tid = task_of(a)
        if tid not in windows:
            hit = next((d for d in Path(store).glob(f"*/tasks/{tid}") if (d / "window.json").exists()), None)
            windows[tid] = (json.loads((hit / "window.json").read_text()), hit / "base") if hit else None
        w = windows[tid]
        if w is None:
            skipped += 1
            continue
        r = repath(a, w[0], w[1], rules=rules_for(w[0], w[1], **rules_over) if rules_over else None, learned=learned)
        if r is None:
            skipped += 1
            continue
        seen += 1
        stale += r.old.get("rules_version") != version  # graded under older path rules (or before the stamp)
        if r.flipped:
            flips.append(r.to_json())
    up = [f for f in flips if (f["new"]["reward"] or 0) > (f["old"]["reward"] or 0)]
    down = [f for f in flips if (f["new"]["reward"] or 0) < (f["old"]["reward"] or 0)]
    return {"attempts": seen, "skipped": skipped, "stale_verdicts": stale, "rules_version": version, "flipped": len(flips), "reward_up": len(up), "reward_down": len(down),
            "full_pass_gained": sum(1 for f in flips if f["new"]["full_pass"] and not f["old"]["full_pass"]),
            "by_task": {t: sum(1 for f in flips if f["task"] == t) for t in sorted({f["task"] for f in flips})}, "flips": flips,
            "rules_over": dict(rules_over or {})}
