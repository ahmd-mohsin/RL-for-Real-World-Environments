"""Family acceptance (05 §8): a family enters training only with enough accepted tasks on enough surfaces, a measured
verifier (the VerifierGoldSet, per class), orthogonal tasks, and a transfer proxy that improves held-out probes.

This module assembles the evidence from what the per-task machines and the lane recorded and scores it with agent R's
modules; it decides nothing on its own beyond the thresholds below.

- **Size** (§8.1, *v1*): ≥ 20 accepted tasks across ≥ 3 surfaces; a held-out probe set of ≥ 10 tasks on a surface the
  family never uses (``transfer.disjoint``).
- **Gold set** (§8.2, ``goldset.score``): items are real solver passes and fails, requirement mutants, valid alternatives,
  blocked and confirmed hacks. Truth for constructed items is the construction; for real items only a human label
  (``labels.LabelStore``) counts. The *v1* gate is per class: with zero errors the Wilson upper bound is ≤ 10%, which needs
  ≥ 36 truly-failing **and** ≥ 36 truly-passing measured items (``GOLD_V1``). The family's verifier version is the
  recipe (the commit's own tests as checks), so items from every accepted task score one family verifier.
- **Orthogonality** (§8.3, ``ortho.orthogonality``): prompt, start/end state, check-set and behaviour layers, the duplicate
  rule, and the surface skew (top surface ≤ 40%).
- **Transfer proxy** (§8.4, ``transfer.transfer_proxy``): base policy vs the same policy with K solved family traces in
  context, paired per probe, the gain surviving on the low-overlap half.

``accept_family`` returns every number with its denominator and the reasons it fails.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.goldset import GoldConfig, GoldItem, score
from crucible.factory.labels import item_sha256
from crucible.factory.ortho import OrthoConfig, TaskProfile, orthogonality

MIN_TASKS = 20  # 05 §8.1 v1
MIN_SURFACES = 3
MIN_PROBES = 10
GOLD_V1 = GoldConfig(min_items=72, fp_upper_max=0.10, fn_upper_max=0.10)  # 05 §8.2 v1: <= 10% per class, >= 36 per class
GOLD_LARGE_SCALE = GoldConfig(min_items=146, fp_upper_max=0.05, fn_upper_max=0.05)  # before large-scale training: >= 73 per class


@dataclass
class GoldRecord:
    """One graded end state of one accepted task, as the lane recorded it."""

    task: str
    source: str  # real_pass | real_fail | mutant | alternative | hack
    solution: Any
    reward: float | None  # the family verifier's verdict (None: masked)
    checks: dict[str, bool] = field(default_factory=dict)
    proven_by: str = ""
    model_label: str | None = None  # the auditor's judgement on a real item ("pass" / "fail"), for kappa only

    @property
    def item_id(self) -> str:
        return f"{self.task}::{self.source}::{item_sha256(self.solution)[:12]}"


def gold_items(records: Sequence[GoldRecord]) -> tuple[list[GoldItem], dict[str, Any]]:
    """GoldItems (deduplicated by task and content) and the verifier's recorded verdicts."""
    items, verdicts, seen = [], {}, set()
    for r in records:
        iid = r.item_id
        if iid in seen:
            continue
        seen.add(iid)
        items.append(GoldItem(iid, r.source, r.solution, item_sha256(r.solution), r.proven_by))  # type: ignore[arg-type]
        verdicts[iid] = {"reward": r.reward, "checks": r.checks}
    return items, verdicts


def gold_report(records: Sequence[GoldRecord], labels: Any = (), *, config: GoldConfig = GOLD_V1) -> dict[str, Any]:
    """The gold-set card per class, plus what the constructed items alone show (the part with no human dependency)."""
    items, verdicts = gold_items(records)
    full = score(items, verdicts, labels, config=config)
    constructed = [it for it in items if it.source in ("mutant", "alternative", "hack")]
    cons = score(constructed, verdicts, labels, config=config)
    per_class = {"truly_failing": full.fp.n, "truly_passing": full.fn.n, "needed_per_class": _per_class_need(config)}
    return {"full": full.to_dict(), "constructed_only": cons.to_dict(), "per_class": per_class,
            "sources": dict(Counter(it.source for it in items)), "ok": full.ok}


def _per_class_need(cfg: GoldConfig) -> int:
    """The smallest n whose zero-error Wilson upper bound is within the config's bound."""
    from crucible.factory.labels import wilson

    n = 1
    while wilson(0, n)[1] > min(cfg.fp_upper_max, cfg.fn_upper_max):
        n += 1
    return n


def profiles(cards: Sequence[Mapping[str, Any]], extra: Mapping[str, Mapping[str, Any]]) -> list[TaskProfile]:
    """TaskProfiles of accepted TaskCards. ``extra[task id]`` gives the lane facts the card does not hold: start and end
    fingerprints, footprint (files and functions the fix touches), check signatures (test ids), the solver pass panel
    and the cause counts."""
    out = []
    for c in cards:
        e = extra.get(c["id"], {})
        out.append(TaskProfile(id=c["id"], intent=c.get("intent", ""), prompt=c.get("prompt", ""), surface=dict(e.get("surface") or c.get("surface") or {}),
                               start_fp=e.get("start_fp"), end_fp=e.get("end_fp"), footprint=list(e.get("footprint") or []),
                               checks=list(e.get("checks") or []), passes=dict(e.get("passes") or {}), causes=dict(e.get("causes") or {}),
                               family=c.get("family", "")))
    return out


def accept_family(*, cards: Sequence[Mapping[str, Any]], extra: Mapping[str, Mapping[str, Any]], gold: Sequence[GoldRecord], labels: Any = (),
                  transfer: Mapping[str, Any] | None = None, probes: Sequence[Any] = (), probe_problems: Sequence[str] = (),
                  surface_axis: str = "repo", provisional_intent: bool = False, gold_config: GoldConfig = GOLD_V1) -> dict[str, Any]:
    """The 05 §8 verdict for one family with every number and its denominator."""
    reasons: list[str] = []
    surfaces = Counter(str((extra.get(c["id"], {}).get("surface") or {}).get(surface_axis, "?")) for c in cards)
    size = {"accepted": len(cards), "surfaces": dict(surfaces), "need_tasks": MIN_TASKS, "need_surfaces": MIN_SURFACES,
            "probes": len(probes), "need_probes": MIN_PROBES}
    if len(cards) < MIN_TASKS:
        reasons.append(f"{len(cards)} accepted tasks, need {MIN_TASKS}")
    if len(surfaces) < MIN_SURFACES:
        reasons.append(f"{len(surfaces)} surfaces, need {MIN_SURFACES}")
    if len(probes) < MIN_PROBES:
        reasons.append(f"{len(probes)} held-out probes, need {MIN_PROBES}")
    reasons += [f"probe set: {p}" for p in probe_problems]
    g = gold_report(gold, labels, config=gold_config)
    if not g["ok"]:
        reasons += [f"gold set: {r}" for r in g["full"]["reasons"]]
    o = orthogonality(profiles(cards, extra), config=OrthoConfig()) if cards else {"ok": False, "reasons": ["no accepted task"], "kept": [], "dropped": {}}
    if not o["ok"]:
        reasons += [f"orthogonality: {r}" for r in o["reasons"]]
    if o.get("dropped"):
        reasons.append(f"orthogonality: {len(o['dropped'])} duplicate task(s) removed: {sorted(o['dropped'])[:5]}")
    if transfer is None:
        reasons.append("transfer proxy: not run")
    elif not transfer.get("ok"):
        reasons += [f"transfer proxy: {r}" for r in transfer.get("reasons") or []]
    if provisional_intent:
        reasons.append("the intent card is a candidate awaiting human approval (provisional_intent=True): the family cannot enter training")
    return {"ok": not reasons, "reasons": reasons, "size": size, "gold": g, "orthogonality": {k: o.get(k) for k in ("ok", "reasons", "kept", "dropped", "skew",
                                                                                                                  "behaviourally_redundant")},
            "pairs": len(o.get("pairs") or {}), "transfer": transfer, "provisional_intent": provisional_intent}


def load_gold(path: Path) -> list[GoldRecord]:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                out.append(GoldRecord(**{k: d[k] for k in GoldRecord.__dataclass_fields__ if k in d}))
    return out
