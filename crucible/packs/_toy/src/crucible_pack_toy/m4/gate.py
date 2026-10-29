"""M4 gate for the toy pack: the 24 ticket logs through the toy mapper and outcome labeler, scored against hand labels
written before the mapper (``labels_test.json``: 84 events, 12 outcomes, test cases T1013-T1024). Reliability is
estimated on the dev cases (T1001-T1012) and applied to the labelled ones, never fitted on what it is scored on.

M4 audit (2026-10-12): the labels existed but nothing scored them; they named the M5 family
(``toy/helpdesk_ticket``) that no mapper serves, and positions rather than log rows."""

from __future__ import annotations

import json
import sys

from crucible.connectors.file_drop import FieldMap, FileDropConnector
from crucible.core.schema.trace import Trace
from crucible.signals.extract import extract
from crucible.signals.gate import code, evaluate, inputs, load_labels
from crucible.signals.reliability import apply, estimate
from crucible_pack_toy import PACK_ROOT

HERE = PACK_ROOT / "src" / "crucible_pack_toy" / "m4"
LOGS = PACK_ROOT / "data" / "ticket_logs" / "tickets.jsonl"
REPORT = PACK_ROOT.parent.parent / "docs" / "modules" / "reports"
# every column the mapper and labeler read (the M3 ingest kept only "team", so CSAT scores were lost)
TICKET_FIELDS = FieldMap(family="toy/support_ticket", extra=["status", "team", "score", "amount", "rule", "sla", "plan"])


def traces() -> dict[str, Trace]:
    c = FileDropConnector(TICKET_FIELDS)
    out = {}
    for ref in c.discover({"glob": str(LOGS)}):
        t = c.normalize(c.fetch(ref))
        out[t.trace_id] = t
    return out


def run(split: str = "test") -> dict:
    labels = load_labels(HERE / f"labels_{split}.json")
    labelled = {e["trace_id"] for e in labels["events"]} | {o["trace_id"] for o in labels.get("outcomes", [])}
    raw = traces()
    all_t = {tid: extract(t) for tid, t in raw.items()}
    rel = estimate(t for tid, t in all_t.items() if tid not in labelled)
    scored = {tid: apply(t, rel) for tid, t in all_t.items() if tid in labelled}
    rep = evaluate(scored, labels)
    # M4 audit R-4: the scored traces' content and the code, beside the labels' hash
    return {**rep.to_dict(), "split": split, "labels_sha256": labels["sha256"], "inputs": inputs(t for tid, t in raw.items() if tid in labelled),
            "code": code(HERE.parent),
            "reliability": {"/".join(k): {"confirmed": r.confirmed, "contradicted": r.contradicted, "unscored": r.unscored, "mean": round(r.mean, 3),
                                          "interval90": [round(x, 3) for x in r.interval()]} for k, r in sorted(rel.items())}}


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "test"
    out = run(split)
    print(json.dumps({k: v for k, v in out.items() if k != "errors"}, indent=1))
    for e in out["errors"]:
        print("ERR", e)
    if "--report" in sys.argv:
        REPORT.mkdir(parents=True, exist_ok=True)
        (REPORT / f"m4_gate_toy_{split}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
