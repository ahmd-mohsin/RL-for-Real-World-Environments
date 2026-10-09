"""The M4 exit gate (ARCHITECTURE M4): extracted signals against hand labels, per pack.

Labels file (JSON): ``{"pack", "labeled_by", "method", "events": [{"trace_id", "seq", "type", "grades", "note"}],
"outcomes": [{"trace_id", "label", "note"}]}``. ``type`` is a taxonomy type or ``"none"`` (the event carries no
signal); ``grades`` is the list of event seqs the signal is about, or null when it grades no step of the trace
(e.g. a comment about code that predates the PR).

- type accuracy: predicted type of each labeled event (the signal whose ``source_event`` is that event, else
  "none") equals the label -- threshold 0.9;
- step-attribution precision: over labeled events where both the prediction and the label grade steps, the share
  of predicted graded steps that the label also names -- threshold 0.9;
- outcome agreement: the trace's outcome label equals the hand label -- threshold 0.95.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.schema.trace import Trace

THRESHOLDS = {"type_accuracy": 0.9, "attribution_precision": 0.9, "outcome_agreement": 0.95}


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

    @property
    def passed(self) -> bool:
        return (self.type_accuracy >= THRESHOLDS["type_accuracy"] and self.attribution_precision >= THRESHOLDS["attribution_precision"]
                and self.outcome_agreement >= THRESHOLDS["outcome_agreement"])

    def to_dict(self) -> dict[str, Any]:
        return {**{k: v for k, v in self.__dict__.items()}, "passed": self.passed, "thresholds": THRESHOLDS}


def evaluate(traces: dict[str, Trace], labels: dict[str, Any]) -> GateReport:
    correct = 0
    errors: list[dict[str, Any]] = []
    confusion: dict[str, dict[str, int]] = {}
    precisions, recalls = [], []
    for lab in labels["events"]:
        t = traces[lab["trace_id"]]
        pred = next((s for s in t.signals if s.source_event == lab["seq"]), None)
        ptype = pred.type if pred else "none"
        confusion.setdefault(lab["type"], {}).setdefault(ptype, 0)
        confusion[lab["type"]][ptype] += 1
        ok = ptype == lab["type"]
        correct += ok
        if not ok:
            errors.append({"trace_id": lab["trace_id"], "seq": lab["seq"], "label": lab["type"], "predicted": ptype, "note": lab.get("note")})
        if pred and pred.grades and lab.get("grades"):
            hit = len(set(pred.grades) & set(lab["grades"]))
            precisions.append(hit / len(pred.grades))
            recalls.append(hit / len(lab["grades"]))
            if hit < len(pred.grades):
                errors.append({"trace_id": lab["trace_id"], "seq": lab["seq"], "attribution": {"label": lab["grades"], "predicted": pred.grades}})
    outs = labels.get("outcomes", [])
    agree = sum(traces[o["trace_id"]].outcome.label == o["label"] for o in outs)
    for o in outs:
        got = traces[o["trace_id"]].outcome.label
        if got != o["label"]:
            errors.append({"trace_id": o["trace_id"], "outcome": {"label": o["label"], "predicted": got}})
    n = len(labels["events"])
    return GateReport(pack=labels.get("pack", "?"), type_accuracy=correct / n if n else 0.0,
                      attribution_precision=sum(precisions) / len(precisions) if precisions else 0.0,
                      attribution_recall=sum(recalls) / len(recalls) if recalls else 0.0,
                      outcome_agreement=agree / len(outs) if outs else 0.0, n_events=n, n_attributed=len(precisions), n_outcomes=len(outs),
                      errors=errors, confusion=confusion)


def load_labels(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
