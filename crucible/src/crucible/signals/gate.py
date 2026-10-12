"""The M4 exit gate (ARCHITECTURE M4): extracted signals against hand labels, per pack.

Labels file (JSON): ``{"pack", "labeled_by", "method", "events": [{"trace_id", "event", "type", "grades", "note"}],
"outcomes": [{"trace_id", "label", "note"}]}``. ``event`` names the labelled event by its stable key
(``core.schema.trace.event_key``: ``row:<id>`` for sources that keep row ids); ``grades`` lists the keys of the events
the signal is about, or null when it grades no step of the trace (e.g. a comment about code that predates the PR).
Older files give ``seq`` and integer grades instead; both forms are read.

- type accuracy: the set of signal types at each labelled event (``source_event``) equals the label (``{}`` for
  "none") -- threshold 0.9;
- step-attribution precision: over labelled events where the prediction grades steps, the share of predicted graded
  steps that the label also names (a label of "grades nothing" makes every predicted step a miss) -- threshold 0.9;
  recall is reported beside it (a prediction that grades nothing where the label names a step is a miss);
- outcome agreement: the trace's outcome label equals the hand label -- threshold 0.95.

The gate passes only with at least ``MIN_EVENTS`` labelled events and ``MIN_OUTCOMES`` outcomes (ARCHITECTURE: 50 per
pack). Each metric carries a 95% Wilson interval; ``confident`` says whether the lower bounds clear the thresholds too.

M4 audit (2026-10-12): labels were keyed by seq, and the M3 connector's reordering silently pointed them at other
events (the committed coding report described traces that no longer existed); a label for a missing trace raised; only
the first signal at an event was seen; a step graded where the label says "none" was never penalised; one label passed
the gate; outcome agreement mixed "unknown" with decisions; no uncertainty was reported.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.schema.trace import Trace, event_key

THRESHOLDS = {"type_accuracy": 0.9, "attribution_precision": 0.9, "outcome_agreement": 0.95}
MIN_EVENTS, MIN_OUTCOMES = 50, 10


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


@dataclass
class GateReport:
    pack: str
    type_accuracy: float
    attribution_precision: float
    attribution_recall: float
    outcome_agreement: float
    n_events: int
    n_attributed: int
    n_outcomes: int
    errors: list[dict[str, Any]] = field(default_factory=list)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    per_type_recall: dict[str, float] = field(default_factory=dict)
    macro_recall: float = 0.0
    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    outcome_coverage: float = 0.0  # share of labelled traces the labeler decided (good / bad)
    unlabelled_signals: int = 0  # signals on labelled traces at events nobody labelled: unscored, reported
    label_errors: int = 0  # labels naming a trace or event that does not exist (each counted as a miss)
    by_actor_class: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (self.n_events >= MIN_EVENTS and self.n_outcomes >= MIN_OUTCOMES
                and self.type_accuracy >= THRESHOLDS["type_accuracy"] and self.attribution_precision >= THRESHOLDS["attribution_precision"]
                and self.outcome_agreement >= THRESHOLDS["outcome_agreement"])

    @property
    def confident(self) -> bool:
        return self.passed and all(self.intervals.get(k, (0.0, 0.0))[0] >= v for k, v in THRESHOLDS.items())

    def to_dict(self) -> dict[str, Any]:
        return {**dict(self.__dict__.items()), "passed": self.passed, "confident": self.confident, "thresholds": THRESHOLDS,
                "minimums": {"events": MIN_EVENTS, "outcomes": MIN_OUTCOMES}}


def _resolve(t: Trace, ref: Any) -> int | None:
    if isinstance(ref, int) and not isinstance(ref, bool):
        return ref if any(e.seq == ref for e in t.events) else None
    return next((e.seq for e in t.events if event_key(e) == ref), None)


def evaluate(traces: dict[str, Trace], labels: dict[str, Any]) -> GateReport:
    correct = 0
    errors: list[dict[str, Any]] = []
    confusion: dict[str, dict[str, int]] = {}
    precisions, recalls = [], []
    label_errors = 0
    seen_events: dict[str, set[int]] = {}
    cls_tally: dict[str, list[int]] = {}
    for lab in labels["events"]:
        ref = lab.get("event", lab.get("seq"))
        t = traces.get(lab["trace_id"])
        seq = _resolve(t, ref) if t is not None else None
        if seq is None:  # fail closed: a label that names nothing is a miss, reported
            label_errors += 1
            errors.append({"trace_id": lab["trace_id"], "event": ref, "label": lab["type"], "error": "no such trace" if t is None else "no such event"})
            confusion.setdefault(lab["type"], {}).setdefault("?", 0)
            confusion[lab["type"]]["?"] += 1
            continue
        seen_events.setdefault(lab["trace_id"], set()).add(seq)
        preds = [s for s in t.signals if s.source_event == seq]
        types = sorted({s.type for s in preds})
        ptype = "+".join(types) if types else "none"
        confusion.setdefault(lab["type"], {}).setdefault(ptype, 0)
        confusion[lab["type"]][ptype] += 1
        ok = ptype == lab["type"]
        correct += ok
        for s in preds:
            c = cls_tally.setdefault("automated" if s.meta.get("automated") else "human", [0, 0])
            c[0] += ok
            c[1] += 1
        if not ok:
            errors.append({"trace_id": lab["trace_id"], "event": ref, "seq": seq, "label": lab["type"], "predicted": ptype, "note": lab.get("note")})
        pred_grades = sorted({g for s in preds for g in s.grades})
        lab_grades = None if lab.get("grades") is None else [_resolve(t, g) for g in lab["grades"]]
        if lab_grades is not None and None in lab_grades:
            label_errors += 1
            errors.append({"trace_id": lab["trace_id"], "event": ref, "error": "label grades an event that does not exist", "grades": lab["grades"]})
            lab_grades = [g for g in lab_grades if g is not None]
        if lab["type"] == "none" and not preds:
            continue
        hit = len(set(pred_grades) & set(lab_grades or []))
        if pred_grades:
            precisions.append(hit / len(pred_grades))
        if lab_grades:
            recalls.append(hit / len(lab_grades))
        if (pred_grades or lab_grades) and set(pred_grades) != set(lab_grades or []):
            errors.append({"trace_id": lab["trace_id"], "event": ref, "attribution": {"label": lab_grades, "predicted": pred_grades}})
    outs = labels.get("outcomes", [])
    agree = 0
    decided = 0
    for o in outs:
        t = traces.get(o["trace_id"])
        got = t.outcome.label if t is not None else None
        decided += got in ("good", "bad")
        agree += got == o["label"]
        if got != o["label"]:
            errors.append({"trace_id": o["trace_id"], "outcome": {"label": o["label"], "predicted": got}})
    unlabelled = sum(1 for tid, done in seen_events.items() for s in traces[tid].signals if s.source_event not in done)
    by_label: dict[str, list[int]] = {}
    for lab_t, row in confusion.items():
        by_label[lab_t] = [row.get(lab_t, 0), sum(row.values())]
    per_type = {k: round(v[0] / v[1], 4) for k, v in sorted(by_label.items()) if v[1]}
    n = len(labels["events"])
    p_sum, r_sum = sum(precisions), sum(recalls)
    return GateReport(pack=labels.get("pack", "?"), type_accuracy=correct / n if n else 0.0,
                      attribution_precision=p_sum / len(precisions) if precisions else 0.0,
                      attribution_recall=r_sum / len(recalls) if recalls else 0.0,
                      outcome_agreement=agree / len(outs) if outs else 0.0, n_events=n, n_attributed=len(precisions), n_outcomes=len(outs),
                      errors=errors, confusion=confusion, per_type_recall=per_type, macro_recall=round(sum(per_type.values()) / len(per_type), 4) if per_type else 0.0,
                      intervals={"type_accuracy": wilson(correct, n), "attribution_precision": wilson(p_sum, len(precisions)),
                                 "outcome_agreement": wilson(agree, len(outs))},
                      outcome_coverage=decided / len(outs) if outs else 0.0, unlabelled_signals=unlabelled, label_errors=label_errors,
                      by_actor_class={k: {"events": v[1], "type_accuracy": round(v[0] / v[1], 4)} for k, v in sorted(cls_tally.items())})


def load_labels(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    data.setdefault("sha256", hashlib.sha256(Path(path).read_bytes()).hexdigest())  # the report names the exact label set
    return data
