"""M5 and M6 exit gates for the coding pack, assembled from the artifacts the analyses wrote (nothing recomputed):

- M5 S1 replay: SWE-rebench edit replay over every attempt of the sampled issues (agent-state and strict rates);
- M5 rediscovery: ``m5_rediscovery.json`` (audited);
- M5 tracker κ: measured on the toy pack (``m56_toy.json``); coding stage labels are not written yet;
- M6 S2: Φ at the midpoint vs the LLM judge on later issues (and the end-of-attempt comparison);
- M6 R1-R6: ``m6_properties_coding.json`` (recorded verifier outcomes);
- M6 corr(C3, ΔΦ) on task04: needs K continuations per branch point on the lab GPU with a paid model — pending.

``python -m crucible_pack_coding.m56.gate`` writes ``docs/modules/reports/m56_gate_coding.json``.
"""

from __future__ import annotations

import json
from typing import Any

from crucible_pack_coding import PACK_ROOT
from crucible_pack_coding.m56.rebench_s2 import analyse, load

CRUCIBLE = PACK_ROOT.parent.parent
REPORTS = CRUCIBLE / "docs" / "modules" / "reports"
DATA = CRUCIBLE / "data" / "m56"


def run(tag: str = "n80_s7") -> dict[str, Any]:
    rows = load(DATA / f"rebench_phi_{tag}.jsonl")
    jm = {j["trajectory_id"]: j["p_success"] for j in load(DATA / f"rebench_judge_{tag}.jsonl") if j.get("p_success") is not None}
    je_path = DATA / f"rebench_judge_end_{tag}.jsonl"
    je = {j["trajectory_id"]: j["p_success"] for j in load(je_path) if j.get("p_success") is not None} if je_path.exists() else None
    s2 = analyse(rows, jm, judge_end=je)
    n = len(rows)
    replay = {"traces": n, "agent_state_replayed": sum(r["replay_agent_state"] for r in rows), "strict_replayed": sum(r["replay_strict"] for r in rows)}
    replay["agent_state_rate"] = round(replay["agent_state_replayed"] / n, 4)
    replay["strict_rate"] = round(replay["strict_replayed"] / n, 4)
    from crucible.workflow.replay import replay_version

    versions = sorted({str(r.get("replay_version")) for r in rows})
    replay["row_versions"] = versions  # "None": rows written before rows were stamped
    replay["current_version"] = replay_version()
    replay["rows_from_current_code"] = versions == [replay["current_version"]]
    rerun = REPORTS / "m5_s1_replay.json"  # the replay alone, re-run with the current code (M5 audit)
    if rerun.exists():
        rr = json.loads(rerun.read_text())
        replay["rerun"] = {k: rr[k] for k in ("version", "n", "agent_state_rate", "strict_rate", "unreplayable_calls", "stale")}
        if rr["version"] == replay["current_version"]:
            replay.update(agent_state_rate=rr["agent_state_rate"], strict_rate=rr["strict_rate"], agent_state_replayed=rr["agent_state"],
                          strict_replayed=rr["strict"], traces=rr["n"], source="m5_s1_replay.json (current code)")
            n = rr["n"]
    redisc = json.loads((REPORTS / "m5_rediscovery.json").read_text())
    props = json.loads((REPORTS / "m6_properties_coding.json").read_text())
    toy = json.loads((REPORTS / "m56_toy.json").read_text()) if (REPORTS / "m56_toy.json").exists() else {}
    mid = s2["metrics"]["phi_mid_validity_weighted"]
    gates = {
        "M5.S1_replay>=0.9_on>=30": replay["agent_state_rate"] >= 0.9 and n >= 30,
        "M5.rediscovery_all_families": redisc["pass"],
        "M5.tracker_kappa>=0.7(toy)": bool(toy and toy["tracker"]["kappa"] >= 0.7),
        "M6.S2_phi_mid_auc>=0.75_and>judge": bool(mid["auc"] and mid["auc"] >= 0.75 and s2.get("judge") and s2["judge"]["phi_minus_judge"]["lo"] > 0),
        "M6.R1-R6_coding": props["pass"],
        "M6.corr_C3_dPhi_task04_reported": False,
    }
    out = {"gate": "M5+M6 coding", "gates": gates, "replay": replay, "s2": s2, "rediscovery": {"pass": redisc["pass"], "families": redisc["families_rediscovered"], "audit": redisc["audit"]},
           "properties": {t: {"ok": v["ok"], "milestones": v["milestones"], "runs": v["runs"]} for t, v in props["tasks"].items()},
           "pending": {"M6.corr_C3_dPhi_task04": "needs K continuations per branch point of 3 task04 trajectories on the lab RTX 5090 with a paid model (~$130 estimated); harness and operator are built and tested"}}
    (REPORTS / "m56_gate_coding.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps({"gates": r["gates"], "replay": r["replay"]}, indent=1))
