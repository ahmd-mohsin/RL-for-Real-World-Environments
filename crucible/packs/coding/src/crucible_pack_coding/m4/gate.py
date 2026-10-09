"""M4 gate for the coding pack: rebuild the labeled traces, extract signals, score them against hand labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from crucible.connectors.trace_store import TraceStore
from crucible.core.store import Store
from crucible.signals.extract import extract
from crucible.signals.gate import evaluate, load_labels
from crucible.signals.reliability import apply, estimate
from crucible_pack_coding import PACK_ROOT
from crucible_pack_coding.m4.build_traces import DATA, build

HERE = Path(__file__).resolve().parent
REPORT = PACK_ROOT.parent.parent / "docs" / "modules" / "reports"


def run(split: str = "dev", rebuild: bool = True) -> dict:
    ts = TraceStore(Store(DATA / "m4"))
    if rebuild:
        build("PointCloudLibrary/pcl", {"from_review_pages": 3, "min_comments": 3}, ts)
    labels = load_labels(HERE / f"labels_{split}.json")
    ids = sorted({e["trace_id"] for e in labels["events"]} | {o["trace_id"] for o in labels.get("outcomes", [])})
    traces = {t: extract(ts.get(t)) for t in ids}
    rel = estimate(traces.values())
    traces = {t: apply(tr, rel) for t, tr in traces.items()}
    rep = evaluate(traces, labels)
    out = {**rep.to_dict(), "split": split, "reliability": {k: {"confirmed": r.confirmed, "contradicted": r.contradicted, "mean": round(r.mean, 3)} for k, r in sorted(rel.items())}}
    return out


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    out = run(split, rebuild="--no-rebuild" not in sys.argv)
    print(json.dumps({k: v for k, v in out.items() if k not in ("errors",)}, indent=1))
    for e in out["errors"]:
        print("ERR", e)
    if "--report" in sys.argv:
        REPORT.mkdir(parents=True, exist_ok=True)
        (REPORT / f"m4_gate_coding_{split}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
