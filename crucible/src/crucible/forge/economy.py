"""Forge economics (m13 G-6; audit 06 §4.7): yield and cost per generator / operator, stop rules, the diversity quota.

The forge ``Ledger`` caps spend; it says nothing about what the spend buys. Repo2RLEnv's economics are the reference: the
number that matters is dollars per *accepted* task, per operator, and a generator that stops producing acceptable tasks
must stop spending.

- **Book** (``YieldBook``): one row per forge attempt (a plan run: generator, operator, seed, family, construction state,
  role spend), later resolved by M9 (accepted or not, confirmed A16 hacks). Optionally appended to ``events.jsonl`` and
  re-loaded, so a resumed forge keeps its counts.
- **Yield** counts *decided* attempts only: rejected at construction, or accepted / rejected by M9. Constructed tasks still
  waiting for M9 are pending, and abandoned ones (budget or infra exhaustion) are not the generator's fault; neither
  counts against it.
- **Stop rules** (``StopRules``): a generator is paused when its yield is below ``min_yield`` (10%) after
  ``min_attempts`` (20) decided attempts, or when its dollars per accepted task exceed its family's cap (with nothing
  accepted, as soon as its spend exceeds the cap). Each pause files one issue (``issues``, plus ``issues.jsonl`` when the
  book has a path). The orchestrator's ``run_plans`` skips plans of paused generators.
- **Diversity quota** (``admit``): no operator and no seed above ``max_share`` (30%) of a family's accepted tasks
  (GitSwarm herding: one approach went from 17% to 90% of the accepted work). A task that would break the quota is
  deferred, not rejected; ``release`` admits deferred tasks once the family has diversified. Below ``quota_floor``
  accepted tasks a share is noise, and the quota is not enforced.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Attempt:
    task_id: str
    generator: str
    operator: str = ""
    seed: str = ""
    family: str = ""
    construction: str = "constructed"  # constructed | rejected | abandoned
    cost_usd: float = 0.0
    accepted: bool | None = None  # M9 verdict (None: not decided yet)
    hacks: int = 0  # confirmed A16 hacks found on this task
    admitted: bool | None = None  # diversity quota: True admitted, False deferred


@dataclass
class StopRules:
    min_attempts: int = 20
    min_yield: float = 0.10
    usd_per_accepted_cap: dict[str, float] = field(default_factory=dict)  # family -> cap ($ per accepted task)
    max_share: float = 0.30
    quota_floor: int = 10


class YieldBook:
    def __init__(self, rules: StopRules | None = None, *, path: Path | None = None):
        self.rules = rules or StopRules()
        self.path = Path(path) if path else None
        self.rows: dict[str, Attempt] = {}
        self.issues: list[dict] = []
        self._paused: dict[str, str] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                ev = json.loads(line)
                if ev.get("event") == "attempt":
                    self.rows[ev["task_id"]] = Attempt(**ev["row"])
                elif ev.get("event") == "pause":
                    self._paused[ev["generator"]] = ev["reason"]
                    self.issues.append(ev)
                elif ev.get("event") == "resume":  # replayed like the live call (M8 audit V15: a reload re-paused the generator)
                    self._paused.pop(ev["generator"], None)
                    for k in [k for k, r in self.rows.items() if r.generator == ev["generator"]]:
                        del self.rows[k]

    def _log(self, ev: dict) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(ev, default=str) + "\n")

    def _save(self, row: Attempt) -> None:
        self._log({"event": "attempt", "task_id": row.task_id, "row": asdict(row)})
        self._check(row.generator)

    # ---- recording -------------------------------------------------------------------------------------------------
    def record_construction(self, task_id: str, generator: str, *, operator: str = "", seed: str = "", family: str = "",
                            state: str = "constructed", cost_usd: float = 0.0) -> Attempt:
        prev = self.rows.get(task_id)
        row = Attempt(task_id, generator, operator or generator, seed, family, state, (prev.cost_usd if prev else 0.0) + cost_usd)
        if prev is not None and state in ("audited", "accepted"):
            # an unchanged task (the orchestrator kept its judged state): keep the M9 verdict, its hacks and its quota
            # decision (M8 audit V15: a re-run wiped them, and yield counted an accepted task as pending)
            row.accepted, row.hacks, row.admitted = prev.accepted, prev.hacks, prev.admitted
        self.rows[task_id] = row
        self._save(row)
        return row

    def record_acceptance(self, task_id: str, accepted: bool, *, hacks: int = 0, cost_usd: float = 0.0) -> None:
        row = self.rows[task_id]
        row.accepted, row.hacks, row.cost_usd = accepted, hacks, row.cost_usd + cost_usd
        self._save(row)

    # ---- yield and stop rules --------------------------------------------------------------------------------------
    def _decided(self, rows: list[Attempt]) -> list[Attempt]:
        return [r for r in rows if r.construction == "rejected" or r.accepted is not None]

    def stats(self, rows: list[Attempt]) -> dict[str, Any]:
        dec = self._decided(rows)
        acc = sum(1 for r in rows if r.accepted)
        usd = sum(r.cost_usd for r in rows)
        hacked = [r for r in rows if r.accepted is not None]
        return {"attempts": len(rows), "decided": len(dec), "constructed": sum(1 for r in rows if r.construction == "constructed"),
                "accepted": acc, "yield": (acc / len(dec)) if dec else None, "usd": round(usd, 6),
                "usd_per_accepted": round(usd / acc, 6) if acc else (math.inf if usd > 0 else None),
                "hack_found_rate": (sum(1 for r in hacked if r.hacks) / len(hacked)) if hacked else None}

    def _check(self, generator: str) -> None:
        if generator in self._paused:
            return
        rows = [r for r in self.rows.values() if r.generator == generator]
        s = self.stats(rows)
        reason = None
        if s["decided"] >= self.rules.min_attempts and (s["yield"] or 0.0) < self.rules.min_yield:
            reason = f"yield {s['yield'] or 0.0:.0%} < {self.rules.min_yield:.0%} after {s['decided']} decided attempts"
        else:
            for fam in {r.family for r in rows}:
                cap = self.rules.usd_per_accepted_cap.get(fam)
                f = self.stats([r for r in rows if r.family == fam])
                if cap is None:
                    continue
                if f["accepted"] and f["usd_per_accepted"] > cap:
                    reason = f"{f['usd_per_accepted']:.4f} USD per accepted task in {fam!r} exceeds the family cap {cap:.4f}"
                elif not f["accepted"] and f["usd"] > cap:
                    reason = f"{f['usd']:.4f} USD spent in {fam!r} with nothing accepted exceeds the family cap {cap:.4f}"
                if reason:
                    break
        if reason:
            self._paused[generator] = reason
            issue = {"event": "pause", "generator": generator, "reason": reason, "stats": s}
            self.issues.append(issue)
            self._log(issue)

    def paused(self, generator: str) -> str | None:
        return self._paused.get(generator)

    def resume(self, generator: str) -> None:
        """A person resumed the generator (after fixing it); its counts start over."""
        self._paused.pop(generator, None)
        for k in [k for k, r in self.rows.items() if r.generator == generator]:
            del self.rows[k]
        self._log({"event": "resume", "generator": generator})

    # ---- diversity quota -------------------------------------------------------------------------------------------
    def _would_break(self, row: Attempt) -> list[str]:
        fam = [r for r in self.rows.values() if r.family == row.family and r.accepted and r.admitted and r.task_id != row.task_id]
        n = len(fam) + 1
        if n < self.rules.quota_floor:
            return []
        out = []
        for key, val in (("operator", row.operator), ("seed", row.seed)):
            if not val:
                continue
            share = (sum(1 for r in fam if getattr(r, key) == val) + 1) / n
            if share > self.rules.max_share + 1e-12:
                out.append(f"{key} {val!r} would hold {share:.0%} of {row.family!r} (max {self.rules.max_share:.0%})")
        return out

    def admit(self, task_id: str) -> tuple[bool, list[str]]:
        """Admit an accepted task into its family's pool, or defer it (the quota)."""
        row = self.rows[task_id]
        if not row.accepted:
            return False, ["not accepted"]
        why = self._would_break(row)
        row.admitted = not why
        self._log({"event": "attempt", "task_id": row.task_id, "row": asdict(row)})
        return row.admitted, why

    def release(self) -> list[str]:
        """Deferred tasks the quota now allows, admitted in order."""
        out = []
        for r in list(self.rows.values()):
            if r.accepted and r.admitted is False and not self._would_break(r):
                r.admitted = True
                self._log({"event": "attempt", "task_id": r.task_id, "row": asdict(r)})
                out.append(r.task_id)
        return out

    # ---- report ----------------------------------------------------------------------------------------------------
    def report(self) -> dict[str, Any]:
        rows = list(self.rows.values())

        def by(key: str) -> dict[str, Any]:
            return {v: self.stats([r for r in rows if getattr(r, key) == v]) for v in sorted({getattr(r, key) for r in rows})}

        stages = {"attempted": len(rows), "constructed": sum(1 for r in rows if r.construction == "constructed"),
                  "accepted": sum(1 for r in rows if r.accepted), "admitted": sum(1 for r in rows if r.admitted)}
        per_stage = {"construct": (stages["constructed"] / stages["attempted"]) if stages["attempted"] else None,
                     "accept": (stages["accepted"] / stages["constructed"]) if stages["constructed"] else None}
        return {"total": self.stats(rows), "by_operator": by("operator"), "by_generator": by("generator"), "by_family": by("family"),
                "stages": stages, "stage_yield": per_stage, "paused": dict(self._paused), "deferred": [r.task_id for r in rows if r.admitted is False]}
