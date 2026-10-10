"""M7 golden gate, CPU part: the kit's decision logic re-scores the seed tasks' *recorded* history.

For every historical task04 grade (pulled from the lab into ``data/m7/history``):

- **static**: the agent's patch is applied to the pinned PCL tree (``gpu/``, ``cuda/``) and scanned with the kit's
  ``static.scan`` and the task04 rule set; the verdict must equal the recorded one;
- **perf**: the recorded per-round timings are replayed through the kit's ``PerfBars`` (best-of-k, re-measurement,
  spread check) with the bars recorded in that grade; the verdict must equal the recorded one, and no recorded grade
  may be flagged contaminated (the spread check must not turn a historical verdict into infra);
- **sanitizer**: the recorded summaries and exit codes go through the kit's summary rule; the verdict must match.

The GPU part (``golden_gpu.py``) re-runs the full kit grader in the verifier image on the lab 5090.
``python -m crucible_pack_coding.verify.golden_cpu`` writes ``docs/modules/reports/m7_golden_cpu.json``.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from crucible.verify.runtime.patterns.perf import PerfBars
from crucible.verify.runtime.patterns.static import scan
from crucible_pack_coding import PACK_ROOT
from crucible_pack_coding.verify import task04

CRUCIBLE = PACK_ROOT.parent.parent
HIST = CRUCIBLE / "data" / "m7" / "history"
PCL = CRUCIBLE / "data" / "cache" / "git" / "pcl.git"
BASE = "c47198828379841dbbeae6991e0757011a552cec"
REPORT = CRUCIBLE / "docs" / "modules" / "reports" / "m7_golden_cpu.json"
# Historical verdicts the kit deliberately does not reproduce: the seed grader scored a measurement the kit rejects.
# Each must be classified exactly as stated here, and its re-grade must be reproduced.
EXPECTED_DIVERGENCE = {
    ("astra_v3", "perf"): {"kit": "infra", "regrade": "astra_v3b",
                           "why": "another user's GPU job contaminated the rounds (KinFu 1.6 -> 60 ms/frame inside one grade); the seed grader "
                                  "scored the noise as a perf failure; the kit masks it and asks for a re-grade (Appendix B.5)"},
}


def task04_tree(patch: Path, dest: Path) -> tuple[bool, str]:
    dest.mkdir(parents=True)
    arch = subprocess.run(["git", f"--git-dir={PCL}", "archive", BASE, "gpu", "cuda"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "xf", "-"], input=arch, cwd=dest, check=True)
    if patch.stat().st_size == 0:
        return True, ""
    subprocess.run(["git", "init", "-q"], cwd=dest, check=True)
    r = subprocess.run(["git", "apply", "--whitespace=nowarn", "--include=gpu/*", "--include=cuda/*", str(patch)], cwd=dest, capture_output=True, text=True,
                       check=False)
    return r.returncode == 0, r.stderr[-300:]


def replay_perf(rec: dict) -> dict:
    attempts = [rec["first_attempt"], rec] if "first_attempt" in rec else [rec]
    bars = {m: v["max_ratio"] for m, v in rec["metrics"].items()}
    it = iter(attempts)
    state = {"cur": None, "k": 0}

    def round_fn(k):
        if k == 0:
            state["cur"] = next(it)
        rounds = state["cur"]["rounds"]
        return {side: {m: vs[k] for m, vs in rounds[side].items() if k < len(vs)} for side in ("original", "candidate")}

    n = len(next(iter(rec["rounds"]["original"].values())))
    v = PerfBars(bars, round_fn, rounds=n, remeasure=len(attempts) - 1, spread_max=0.25)()
    return {"pass": v.passed, "infra": v.infra, "contaminated": v.detail.get("contaminated", [])}


def task04_golden() -> dict:
    rows, ok = {}, True
    for rep in sorted((HIST / "task04" / "glogs").glob("*/verifier/report.json")):
        name = rep.parts[-3]
        r = json.loads(rep.read_text())
        g = r.get("gates", {})
        row: dict = {"recorded_reward": r.get("reward")}
        patch = HIST / "task04" / "patches" / f"{name}.patch"
        if patch.exists() and "static" in g:
            with tempfile.TemporaryDirectory() as d:
                applied, err = task04_tree(patch, Path(d) / "t")
                if applied:
                    s = scan(Path(d) / "t", task04.RULES)
                    row["static"] = {"recorded": g["static"]["pass"], "kit": s["pass"], "violations": s["count"]}
                    row["static"]["match"] = row["static"]["recorded"] == s["pass"]
                else:
                    row["static"] = {"skipped": f"patch does not apply to gpu/ cuda/: {err}"}
        if isinstance(g.get("perf"), dict) and "metrics" in g["perf"]:
            p = replay_perf(g["perf"])
            row["perf"] = {"recorded": g["perf"]["pass"], "kit": p["pass"], "infra": p["infra"], "match": p["pass"] == g["perf"]["pass"] and not p["infra"]}
            exp = EXPECTED_DIVERGENCE.get((name, "perf"))
            if exp:
                row["perf"].update({"expected_divergence": exp, "match": bool(p["infra"]) and exp["kit"] == "infra"})
        san = g.get("sanitizer")
        if isinstance(san, dict) and any(isinstance(v, dict) and "summary" in v for v in san.values()):
            kit_ok = True
            for key, v in san.items():
                if isinstance(v, dict) and "summary" in v:
                    clean, _ = task04.sanitizer_clean(key.split("/")[0], v.get("returncode", 0), "\n".join(v["summary"]))
                    kit_ok &= clean  # recorded runs all produced result.json unless the run failed (rc != 0, caught by clean)
            row["sanitizer"] = {"recorded": san["pass"], "kit": kit_ok, "match": kit_ok == san["pass"]}
        for k in ("static", "perf", "sanitizer"):
            if isinstance(row.get(k), dict) and "match" in row[k]:
                ok &= row[k]["match"]
        rows[name] = row
    for (name, _), exp in EXPECTED_DIVERGENCE.items():  # the re-grade must be reproduced exactly
        rg = rows.get(exp["regrade"], {}).get("perf", {})
        ok &= bool(rg.get("match")) and not rg.get("expected_divergence")
    counts = {k: sum(1 for r in rows.values() if isinstance(r.get(k), dict) and r[k].get("match")) for k in ("static", "perf", "sanitizer")}
    totals = {k: sum(1 for r in rows.values() if isinstance(r.get(k), dict) and "match" in r[k]) for k in ("static", "perf", "sanitizer")}
    return {"pass": ok, "matched": {k: f"{counts[k]}/{totals[k]}" for k in counts}, "grades": rows}


def main() -> None:
    rep = {"gate": "M7 golden, CPU part (recorded history re-scored by the kit)", "task04": task04_golden()}
    rep["pass"] = rep["task04"]["pass"]
    REPORT.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"pass": rep["pass"], "task04": rep["task04"]["matched"]}, indent=1))
    for n, r in rep["task04"]["grades"].items():
        print(n, {k: (v.get("recorded"), v.get("kit")) if isinstance(v, dict) and "kit" in v else v for k, v in r.items()})


if __name__ == "__main__":
    main()
