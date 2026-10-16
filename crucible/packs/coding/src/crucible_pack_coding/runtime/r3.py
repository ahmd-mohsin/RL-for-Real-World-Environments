"""M10 exit gate R3 (M10 lab audit R-1, 2026-10-15): mini-swe-agent 2.4.6, installed by Harbor, runs task02 live through
``RunSpec`` and the separate-verifier patch seam (``interop/harbor/patch_task.HarborPatchTask`` +
``interop/harbor/agent_harness.HarborAgentHarness``), with its model traffic through Crucible's capture proxy.

The agent works in a Harbor World on the task image on the lab RTX 5090 (``DOCKER_HOST=ssh://``), its agent phase
restricted to the proxy's host; its work leaves through task02's collect hook as ``/logs/artifacts/model.patch`` and is
graded at task02@5e53057 with the image digest pinned, in a fresh container, inside a ``PerfGuard`` slot (as R1).

    CRUCIBLE_R3_PROXY_HOST=<this machine as the lab container reaches it, e.g. its tailnet IP> \\
    OPENROUTER_API_KEY=... python -m crucible_pack_coding.runtime.r3      # writes docs/modules/reports/m10_r3.json

Environment:
- ``CRUCIBLE_R3_PROXY_HOST`` (required), ``CRUCIBLE_R3_PROXY_PORT`` (default 8788): the proxy listens on 0.0.0.0 here;
- ``OPENROUTER_API_KEY`` (required): the upstream key, held by the proxy only; the agent sees a per-attempt capture key;
- ``CRUCIBLE_R3_UPSTREAM`` (default ``https://openrouter.ai/api/v1``: the recorded run used OpenRouter);
- ``CRUCIBLE_R3_BUDGET_USD`` (default 15).
The forwarded model name (``FORWARDED``, the recorded ``openai/gpt-5.6-sol``) must be priced in crucible.toml ``[prices]``
under exactly that name, or the spend is booked uncertain and the gate's "cost settled above $0" cannot hold: the script
refuses to start.

Pass (M10 lab audit R-1 Acceptance): the recorded gate profile reproduced (``profile_equal``: the same pass/fail per gate
and the same failing ops test ids as the recorded GPT 5.6 run), with the attempt journaled, not masked, one attempt, its
cost settled from the capture above $0, the grade's infra ok, the reconcile not FATAL and the Harbor version recorded.
``pass_relaxed`` (everything but the profile) and ``unmet_acceptance`` are reported because a live sample is stochastic;
closing R3 on ``pass_relaxed`` needs the audit's R-1 Acceptance amended. Every way this run differs from the recorded one
is listed under ``deviations``. Not executed in this change (no lab access from the authoring machine).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from crucible.connectors.capture_proxy import CaptureProxy, Upstream
from crucible.interop.harbor.agent_harness import AgentLimits, HarborAgentHarness
from crucible.interop.harbor.patch_task import HarborPatchTask
from crucible.interop.harbor.verify_dir import export_ref
from crucible.runtime.harness.native import PolicyRef
from crucible.runtime.infra import RetryRule
from crucible.runtime.perf_guard import PerfGuard
from crucible.runtime.run import Runner, RunSpec
from crucible_pack_coding.runtime.r1 import CRUCIBLE, DIGEST, IMAGE, LAB, REF, REPO, RUN, TASK, failed_ids

REPORT = CRUCIBLE / "docs" / "modules" / "reports" / "m10_r3.json"
RUNS = CRUCIBLE / "data" / "runs"
RUN_ID = "m10-r3-task02-live"
UPSTREAM_KEY = "OPENROUTER_API_KEY"
FORWARDED = "openai/gpt-5.6-sol"  # the recorded info.config.model.model_name: what OpenRouter must receive
# litellm strips its provider prefix before calling an OpenAI-compatible base URL, so the agent is given the provider
# "openai" twice to forward the recorded name unchanged [unverified: litellm's prefix handling behind a custom base URL]
HARBOR_MODEL = f"openai/{FORWARDED}"
# the recorded agent was mini-swe-agent's InteractiveAgent in yolo mode, writing to the recorder's own disk: these keys
# are that class's or that machine's, not the agent's behaviour
AGENT_DROP = ("output_path", "mode", "whitelist_actions", "confirm_exit")
DEVIATIONS = [
    "agent class: recorded minisweagent.agents.interactive.InteractiveAgent (mode yolo); here Harbor's mini-swe-agent install run as "
    "`mini-swe-agent --yolo --exit-immediately` with the recorded agent config (templates, step_limit, cost_limit, format-error limit)",
    "model class: recorded mswea.ext.LongOpenRouterModel; here mini-swe-agent's default litellm model behind the capture proxy, with the "
    "recorded model_kwargs (max_tokens 32000, reasoning.effort high); request_timeout and set_cache_control are not carried",
    "environment: recorded mswea.ext.SSHEnvironment into a docker container on the lab; here mini-swe-agent runs inside a Harbor World "
    "container of the task image on the lab, with the recorded environment env, cwd and timeout",
    f"upstream: recorded OpenRouter called directly; here OpenRouter through Crucible's capture proxy (per-attempt key), the agent given "
    f"{HARBOR_MODEL!r} so the forwarded name is {FORWARDED!r} [unverified]",
    "limits: the proxy caps the attempt at 2 x step_limit model calls (a safety net: the cap counts failed and retried calls too); "
    "Harbor passes --cost-limit 0 by default, which may override the recorded cost_limit [unverified]; budget_usd caps the spend",
    "network: the agent phase is task02's no-network phase plus the proxy host (Harbor extra_allowed_hosts); the recorded run had "
    "network_guard false",
]


def recorded() -> dict[str, Any]:
    """The recorded run's ``info`` (configuration, mini-swe-agent version, exit status)."""
    return json.loads((RUN / "trajectory.json").read_text())["info"]


def agent_config(info: dict[str, Any]) -> dict[str, Any]:
    """The recorded mini-swe-agent configuration, as Harbor's ``config=`` (written to a YAML layered over ``mini``)."""
    cfg = info["config"]
    agent = {k: v for k, v in (cfg.get("agent") or {}).items() if k not in AGENT_DROP}
    model = {"model_kwargs": dict((cfg.get("model") or {}).get("model_kwargs") or {})}
    env_cfg = cfg.get("environment") or {}
    environment = {k: env_cfg[k] for k in ("env", "cwd", "timeout") if k in env_cfg}  # never the recorder's ssh command or key path
    return {"agent": agent, "model": model, "environment": environment}


def spec(proxy: CaptureProxy, proxy_url: str) -> RunSpec:
    info = recorded()
    host = urlparse(proxy_url).hostname
    task = HarborPatchTask(export_ref(REPO, TASK, REF, CRUCIBLE / "data" / "runtime" / "tasks"), image=IMAGE, expected_image=DIGEST, host=LAB, ref=REF,
                           perf=True, docker_host=LAB.docker_host, docker_image=IMAGE, agent_allowed_hosts=[host] if host else [])
    harness = HarborAgentHarness("mini-swe-agent", str(info.get("mini_version") or "2.4.6"), model=HARBOR_MODEL, proxy=proxy, proxy_url=proxy_url,
                                 agent_kwargs={"config": agent_config(info)},
                                 limits=AgentLimits(max_model_calls=2 * int(info["config"]["agent"]["step_limit"])),
                                 perf_guard=PerfGuard(probe=LAB.exec, wait_s=12 * 3600, poll_s=120), host=LAB.name)
    # one attempt: a masked agent phase is not paid for twice. A grade masked for the lab (contention) is re-graded later
    # from the stored patch with Runner(RUNS).regrade(spec), never by re-running the agent
    return RunSpec(run_id=RUN_ID, tasks=[task], policies=[PolicyRef(id=f"harbor:{FORWARDED}", make=lambda a: {})], harnesses=[harness], k=1,
                   concurrency=1, retry=RetryRule(max_attempts=1), budget_usd=float(os.environ.get("CRUCIBLE_R3_BUDGET_USD", "15")))


def preflight() -> list[str]:
    """Why the gate cannot start here (empty: it can)."""
    from crucible.core.config import get_config

    problems = [f"{v} is not set" for v in ("CRUCIBLE_R3_PROXY_HOST", UPSTREAM_KEY) if not os.environ.get(v)]
    if FORWARDED not in (get_config().prices or {}):
        problems.append(f"crucible.toml [prices] has no entry for {FORWARDED!r}: the gate needs its spend settled, not booked uncertain")
    return problems


def _read(p: Path) -> dict[str, Any]:
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def report(summary: Any) -> dict[str, Any]:
    """The gate report from the run's journal and the attempt's records."""
    rdir = RUNS / RUN_ID
    res = rdir / "results.jsonl"
    rows = [json.loads(x) for x in res.read_text().splitlines() if x.strip()] if res.exists() else []
    rec = rows[-1] if rows else {}
    adir = None
    if rec.get("attempt"):
        sid, n = str(rec["attempt"]).rsplit("-a", 1)
        adir = rdir / "samples" / sid / f"a{n}"
    got = _read(adir / "verifier" / "report.json") if adir else {}
    run = _read(adir / "agent_run.json") if adir else {}
    rc = _read(adir / "reconcile.json") if adir else {}
    want = _read(RUN / "grade_report.json")
    profile = dict(got.get("gates") or {})
    want_profile = {g: ("pass" if v.get("pass") else "fail") for g, v in (want.get("gates") or {}).items() if isinstance(v, dict) and "pass" in v}
    ops, want_ops = failed_ids((got.get("meta") or {}).get("report") or {}), failed_ids(want)
    out: dict[str, Any] = {
        "run": getattr(summary, "__dict__", summary),
        "record": {k: rec.get(k) for k in ("attempt", "reward", "mask_sample", "failure", "cost_usd", "cost_uncertain_usd")},
        "agent": {k: run.get(k) for k in ("agent", "version", "harness", "config_digest", "harbor_version", "model", "proxy_session", "exit_status",
                                          "outcome", "limit_stops", "processes")},
        "reconcile": {k: rc.get(k) for k in ("level", "code")},
        "infra": got.get("infra"), "gates": profile, "recorded_gates": want_profile, "ops_failed": ops, "recorded_ops_failed": want_ops,
        "reward": got.get("reward"), "recorded_reward": want.get("reward"), "journaled": bool(rec), "deviations": DEVIATIONS,
    }
    out["profile_equal"] = profile == want_profile and ops == want_ops
    # M10 lab audit R-1 Acceptance: gate R3 "reproduces the recorded gate profile". pass_relaxed is the run being valid
    # evidence (journaled, unmasked, one attempt, cost settled from the capture above $0, infra ok, reconcile not FATAL,
    # the same gate set, the Harbor version recorded); pass also needs profile_equal. A live sample is stochastic, so a
    # pass_relaxed run with another profile is reported with its unmet item for a reviewer, never as gate R3 closed
    checks = {"journaled": out["journaled"], "not_masked": not rec.get("mask_sample"), "one_attempt": getattr(summary, "attempts", None) == 1,
              "cost_settled_above_0": float(rec.get("cost_usd") or 0.0) > 0.0, "infra_ok": (got.get("infra") or {}).get("status") == "ok",
              "reconcile_not_fatal": rc.get("level") not in (None, "fatal"), "same_gate_set": set(profile) == set(want_profile),
              "harbor_version_recorded": bool(run.get("harbor_version")), "profile_equal": out["profile_equal"]}
    out["unmet_acceptance"] = [k for k, ok in checks.items() if not ok]
    out["pass_relaxed"] = all(ok for k, ok in checks.items() if k != "profile_equal")
    out["pass"] = not out["unmet_acceptance"]
    return out


async def main() -> dict[str, Any]:
    problems = preflight()
    if problems:
        raise RuntimeError("gate R3 cannot start: " + "; ".join(problems))
    import uvicorn

    host, port = os.environ["CRUCIBLE_R3_PROXY_HOST"], int(os.environ.get("CRUCIBLE_R3_PROXY_PORT", "8788"))
    proxy = CaptureProxy(CRUCIBLE / "data" / "runtime" / "capture" / RUN_ID,
                         openai=Upstream(base=os.environ.get("CRUCIBLE_R3_UPSTREAM", "https://openrouter.ai/api/v1"), key=os.environ[UPSTREAM_KEY]))
    server = uvicorn.Server(uvicorn.Config(proxy.app(), host="0.0.0.0", port=port, log_level="warning", lifespan="on"))  # noqa: S104 -- the lab container reaches it
    serving = asyncio.create_task(server.serve())
    while not server.started:
        if serving.done():
            raise RuntimeError(f"the capture proxy did not start on port {port}: {serving.exception()!r}")
        await asyncio.sleep(0.05)
    try:
        summary = await Runner(RUNS).run(spec(proxy, f"http://{host}:{port}"))
    finally:
        server.should_exit = True
        await serving
    out = report(summary)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("pass", "pass_relaxed", "unmet_acceptance", "profile_equal", "reward", "gates", "recorded_gates", "record",
                                          "reconcile")}, indent=1, default=str))
    return out


if __name__ == "__main__":
    asyncio.run(main())
