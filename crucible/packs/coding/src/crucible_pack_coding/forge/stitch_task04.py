"""Stitch the four task04 span tasks back into the H4 task (m13 G-10) and compare with task04 v3.

    python -m crucible_pack_coding.forge.stitch_task04

The spans (people, ncv, kinfu_family; ``derive.py``) are stitched with ``forge.stitch`` (residual shared
helpers back into the oracle). The stitched task reproduces v3 when its files equal v3's oracle file for file and its
gates equal the seed grader's (``verify/task04.py:GATES``): the same end state under the same grader (scope ``full``)
gives the same verdicts, so the golden runner's v3 verdicts carry over without a GPU grade. Writes
``data/forge/coding/task04_stitched`` and ``docs/modules/reports/m13_stitch.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crucible.forge.stitch import Span, stitch
from crucible_pack_coding.forge.derive import CRUCIBLE, OUT, oracle_patch, split_patch

SPANS = OUT / "task04_spans"
SCOPES = ("people", "ncv", "kinfu_family")
REPORT = CRUCIBLE / "docs" / "modules" / "reports" / "m13_stitch.json"


def load_spans(root: Path = SPANS) -> list[Span]:
    out = []
    for s in SCOPES:
        d = root / s
        checks = json.loads((d / "crucible" / "checks.json").read_text())
        gates = [c["gate"] for c in (checks.get("checks") if isinstance(checks, dict) else checks)]
        out.append(Span(s, split_patch((d / "start.patch").read_text()), split_patch((d / "solution" / "solution.patch").read_text()), gates))
    return out


def run(write: bool = True) -> dict[str, Any]:
    from crucible_pack_coding.verify.task04 import GATES

    st = stitch(load_spans(), residual="oracle", gate_order=GATES)
    v3 = split_patch(oracle_patch("task04_pcl_gpu_cuda12"))
    files = st.files()
    rep = {"spans": list(SCOPES), "nested": st.nested, "residual_into_oracle": st.residual, "start_files": len(st.start), "oracle_files": len(st.oracle),
           "v3_files": len(v3), "missing_vs_v3": sorted(set(v3) - set(files)), "extra_vs_v3": sorted(set(files) - set(v3)),
           "differing_vs_v3": sorted(p for p in set(v3) & set(files) if v3[p] != files[p]), "gates": st.gates, "v3_gates": list(GATES)}
    rep["reproduces_v3"] = not (rep["missing_vs_v3"] or rep["extra_vs_v3"] or rep["differing_vs_v3"]) and not st.start and st.gates == list(GATES)
    if write:
        d = OUT / "task04_stitched"
        (d / "solution").mkdir(parents=True, exist_ok=True)
        (d / "start.patch").write_text("".join(st.start[p] for p in sorted(st.start)))
        (d / "solution" / "solution.patch").write_text("".join(st.oracle[p] for p in sorted(st.oracle)))
        (d / "stitch.json").write_text(json.dumps({**rep, "owners": st.owners}, indent=1) + "\n")
        REPORT.write_text(json.dumps(rep, indent=1) + "\n")
    return rep


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
