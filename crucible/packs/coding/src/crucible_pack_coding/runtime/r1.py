"""M10 exit gate R1 (research m9-m10/02 §6.1): re-grade the historical GPT 5.6 sol run on task02 through the runtime.

The run (mini-swe-agent 2.4.6, 88 calls, $2.29) enters as an imported attempt; the task is pinned to commit
``5e53057`` (the grader it was graded with; later commits tightened the suite) and to the task image digest; grading
runs on the lab RTX 5090 over ``DOCKER_HOST=ssh://`` inside a ``PerfGuard`` slot (the lab is shared: it waits for an
idle GPU and for no other crucible container). Pass: same gate profile as the recorded grade (``ops`` with the same 6
failing test ids, ``hard`` failing), reward 0.0, infra ok, exactly one attempt, not masked.

    python -m crucible_pack_coding.runtime.r1          # runs (waits politely), then writes docs/modules/reports/m10_r1.json
"""

from __future__ import annotations

import asyncio
import json
import re

from crucible.interop.harbor.verify_dir import HarborDirTask, export_ref
from crucible.runtime.gpu import GpuHost
from crucible.runtime.harness.imported import ImportedHarness
from crucible.runtime.harness.native import PolicyRef
from crucible.runtime.perf_guard import PerfGuard
from crucible.runtime.run import Runner, RunSpec
from crucible_pack_coding import PACK_ROOT

CRUCIBLE = PACK_ROOT.parent.parent
REPO = CRUCIBLE.parent
TASK = REPO / "migration_tasks" / "task02_pcops_triton"
RUN = TASK / "runs" / "gpt-5.6-sol-run1"
REF = "5e53057"
IMAGE, DIGEST = "task02-pcops-triton:v1", "sha256:c0eb6115f55eb7fe88edf0485752da2de2fb41a7aea433fdfcf37638cd42d2c7"
LAB = GpuHost(name="lab5090", docker_host="ssh://ahmed@100.100.101.36", ssh="ahmed@100.100.101.36", gpu_arch="12.0", shared=True)
REPORT = CRUCIBLE / "docs" / "modules" / "reports" / "m10_r1.json"


def failed_ids(report: dict) -> list[str]:
    return sorted(re.sub(r" - .*$", "", x).replace("FAILED ", "") for x in ((report.get("gates") or {}).get("ops") or {}).get("failed", []))


def spec() -> RunSpec:
    task = HarborDirTask(export_ref(REPO, TASK, REF, CRUCIBLE / "data" / "runtime" / "tasks"), image=IMAGE, expected_image=DIGEST, host=LAB, ref=REF, perf=True)
    harness = ImportedHarness({"import:gpt-5.6-sol-run1": RUN / "model.patch"}, perf_guard=PerfGuard(probe=LAB.exec, wait_s=12 * 3600, poll_s=120), host=LAB.name)
    # M10 lab audit R-4: the imported run is the policy (its patch content is in the harness's identity); the ref names which recording
    return RunSpec(run_id="m10-r1-task02-sol", tasks=[task], policies=[PolicyRef(id="import:gpt-5.6-sol-run1", make=lambda a: {}, config={"recorded": "gpt-5.6-sol-run1"})],
                   harnesses=[harness], k=1)


async def main() -> dict:
    s = await Runner(CRUCIBLE / "data" / "runs").run(spec())
    rdir = CRUCIBLE / "data" / "runs" / "m10-r1-task02-sol"
    rec = [json.loads(x) for x in (rdir / "results.jsonl").read_text().splitlines()][-1]
    got = json.loads(next(rdir.glob("samples/*/a*/verifier/report.json")).read_text())
    want = json.loads((RUN / "grade_report.json").read_text())
    got_report = (got.get("meta") or {}).get("report") or {}
    profile = {g: v for g, v in got["gates"].items()}
    want_profile = {g: ("pass" if v.get("pass") else "fail") for g, v in want["gates"].items() if isinstance(v, dict) and "pass" in v}
    out = {"run": s.__dict__, "record": {k: rec.get(k) for k in ("attempt", "reward", "mask_sample", "failure")}, "gates": profile,
           "recorded_gates": want_profile, "ops_failed": failed_ids(got_report), "recorded_ops_failed": failed_ids(want),
           "reward": got["reward"], "recorded_reward": want.get("reward")}
    out["pass"] = (out["reward"] == 0.0 and not rec["mask_sample"] and s.attempts == 1 and out["ops_failed"] == out["recorded_ops_failed"]
                   and all(profile.get(g) == v for g, v in want_profile.items() if g in profile))
    REPORT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("pass", "reward", "gates", "recorded_gates")}, indent=1))
    return out


if __name__ == "__main__":
    asyncio.run(main())
