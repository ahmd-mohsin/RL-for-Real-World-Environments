"""Any agentenv A2A agent as a Crucible harness (m13 C7; audit 02). DEPEND on ``agent_env.a2a_agent.protocol`` (message
send / poll, terminal parsing) and the agentenv-protocol extensions the agent advertises.

For each attempt the harness opens the task's World (it must be served over MCP: the agentenv backend's gateway), then:

1. **mcp-config** (``urn:agentenv:mcp-config/v1``): registers the World's MCP URL with the actor's role header
   (``AgentEnv-Role``), so the agent sees exactly the role-filtered tools the native harness would.
2. **agent-config** (``urn:agentenv:agent-config/v1``): the agent's own config fields (model, ``max_turns``, ...).
   ``${capture.base_url}`` and ``${capture.key}`` in it are replaced by Crucible's capture proxy and a per-attempt
   session key, so the agent's model calls go through the proxy (the upstream key never reaches the agent).
3. **prompt**: the task instruction as an A2A message. The terminal status is the agent's own report; blame comes from
   evidence (``runtime.agent_evidence``, M10 lab audit R-1, 2026-10-15). The agent is the policy, so ``failed`` with
   ``agent_error`` is its outcome (blame ``policy``, scored) unless the proxy log's last upstream call was a 5xx, 408
   or 429, a connect error or a dropped stream (``provider_unavailable``, masked); an upstream 401/403/404 is
   ``crucible:harness_unsupported`` and a 402 (or a quota answer) ``crucible:credits_exhausted``, each only when Crucible's
   own probe of the configured model on the configured dialect (``api``; default the proxy's only upstream;
   ``CaptureProxy.probe``) answers the same: the agent holds the capture key and writes its own requests, their model and
   their route, so an unconfirmed verdict is ``unknown`` (a forged 402 can no longer stop the run). An ``infra_error``,
   ``canceled`` or ``rejected`` the proxy does not confirm is ``unknown`` blame (masked, and counted by the 5%
   unknown-blame stop): it used to be masked as infrastructure on the agent's word alone. An agent that stops answering
   polls (agentenv's ``AgentUnreachableError``, raised only when ``poll_a2a_task`` gets a ``sandbox_id``; an older
   agentenv without it falls back to the timeout) is masked ``transport_unreachable`` within about a minute, where it
   was scored as a timeout after ``timeout_s``. Crucible's own deadline (``timeout_s``) is a ``timeout`` limit stop,
   unless the proxy shows the provider failing, which makes it the provider's. The agent's error code and text go into
   the failure, scrubbed.
4. **trajectory/v1**: the agent's native trajectory is saved next to the World trace (``agent_trajectory.json``).
5. **calls**: the capture log becomes the actor's model calls (with each call's upstream ``status``, ``error`` and
   ``complete``); a call has a token record only when the upstream
   returned prompt and completion token ids with logprobs (vLLM-style), stamped with ``policy_version`` (the checkpoint
   the agent's model server serves). In ``rollout_mode="train"`` a rollout
   without complete token records is masked ``crucible:capture_incomplete`` by the record builder (no capture, no
   training data: the trainer never guesses logprobs).

M10 lab audit R-1 (2026-10-15), besides item 3:

- The capture key is revoked as soon as the agent phase ends (before the trajectory fetch and grading): a request with it
  during verification gets 401. It stayed live until cleanup, so an agent still running could call the model while its
  work was graded.
- Spend is settled from the session's capture log on every exit path, masked attempts included
  (``agent_evidence.capture_spend``, the ``LLMClient._prices`` rule: ``prices`` or crucible.toml ``[prices]``; a
  self-hosted upstream costs 0, decided per upstream by ``self_hosted_apis``): every forwarded row, retries included, and
  a usage-less or cut call booked as an uncertain bound. Every attempt settled $0, so ``budget_usd`` never tripped on an
  A2A run.
- Limit stops follow ``limit_policy`` (``score`` by default, ``mask`` or ``penalty:<c>``), as in the native harness;
  only an evidenced stop (Crucible's deadline) can be masked or penalized.

Grading is the task's own (``task.grade(world, dir)``), graded in place exactly as for the native harness (M10 lab audit
R-3): a failed grade is graded again inside the attempt before the World closes (``runtime.grading.grade_in_place``),
the attempt then ends terminal-masked (never an agent re-run), and a repeated answer-sensitive failure (a hang, a checker
outage, a non-finite reward) is decided by the task's control (``runtime.grading.settle``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib
import inspect
import json
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from crucible.core.schema.run import Failure
from crucible.runtime.agent_evidence import (
    DRAIN_S,
    PROXY_APIS,
    AgentEnd,
    UpstreamEvidence,
    capture_spend,
    classify_agent_end,
    confirm_upstream,
    read_upstream_evidence,
    self_hosted_apis,
)
from crucible.runtime.grading import grade_in_place, grade_summary, grading_window, liveness, settle, write_reports
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.harness.native import LIMIT_STOPS, _keep_tries
from crucible.runtime.infra import LimitPolicy, StageFailure, boundary

FAILED_STATES = {"failed", "canceled", "rejected"}  # the agent's own failure reports (``_outcome`` classifies each by evidence)
PROTOCOL = "agent_env.a2a_agent.protocol"


def _protocol() -> Any:
    """agentenv's A2A client helpers, imported when an attempt runs (the module imports without agentenv)."""
    return importlib.import_module(PROTOCOL)


def _has_liveness(proto: Any) -> bool:
    """The version seam: agentenv's ``poll_a2a_task`` takes ``sandbox_id`` (and raises ``AgentUnreachableError``) only
    from the revision that added agent liveness; an older one is polled without it and falls back to the timeout."""
    try:
        return "sandbox_id" in inspect.signature(proto.poll_a2a_task).parameters
    except (TypeError, ValueError):
        return False


def token_record(resp: dict) -> dict | None:
    """Prompt/completion token ids and logprobs from an OpenAI-compatible response (vLLM ``return_token_ids`` /
    ``return_tokens_as_token_ids``); None when any piece is missing."""
    ch = (resp.get("choices") or [{}])[0]
    prompt = resp.get("prompt_token_ids") or ch.get("prompt_token_ids")
    completion = ch.get("token_ids")
    content = ((ch.get("logprobs") or {}).get("content")) or []
    if completion is None and content and all(str(c.get("token", "")).startswith("token_id:") for c in content):
        completion = [int(str(c["token"]).split(":", 1)[1]) for c in content]
    lps = [c.get("logprob") for c in content]
    if not prompt or not completion or len(lps) != len(completion) or any(x is None for x in lps):
        return None
    return {"prompt_token_ids": list(prompt), "completion_token_ids": list(completion), "logprobs": lps}


def calls_from_capture(path: Path | None, actor: str, policy_version: int | None = None) -> list[dict[str, Any]]:
    """The capture log as model calls, with ``CaptureLogConnector``'s retry rule: a repeated request collapses onto its
    earlier attempt only when it is marked as a retry (or the earlier attempt failed); two identical genuine calls
    (best-of-n, parallel subagents) stay two calls. M6 audit: every repeated digest collapsed, which shifted every later
    ``call_id`` and misplaced credit.

    M10 lab audit R-1: each call keeps the upstream ``status``, its ``error`` (scrubbed: a provider can echo a key) and
    ``complete: false`` for a cut or errored stream, plus ``latency_s``. They were dropped (harness.py:71-79), so the
    calls a harness recorded could not show a provider outage. The proxy's own limit-stop rows are never calls."""
    if path is None or not Path(path).exists():
        return []
    from crucible.connectors.base import SourceRef
    from crucible.connectors.capture_proxy import CaptureLogConnector
    from crucible.core.secrets import scrub_secrets

    rows = CaptureLogConnector().fetch(SourceRef(connector="capture_log", key=Path(path).stem, meta={"path": str(path)})).payload["exchanges"]
    out = []
    for i, r in enumerate(rows, 1):
        resp = r.get("response") or {}
        if not isinstance(resp, dict):
            resp = {}
        ch = (resp.get("choices") or [{}])[0]
        status = r.get("status")
        call: dict[str, Any] = {"call_id": f"{actor}:cap{i}", "model": (r.get("request") or {}).get("model"), "usage": resp.get("usage") or {},
                                "finish": ch.get("finish_reason"), "kind": "turn", "status": status}
        # the same rule as CaptureLogConnector.normalize: the proxy's error, the response's, else a failed status
        err = r.get("error") or resp.get("error") or (f"HTTP {status}" if isinstance(status, int) and status >= 400 else None)
        if err:
            call["error"] = scrub_secrets(str(err))[:500]
        if r.get("complete") is False:
            call["complete"] = False
        if r.get("latency_s") is not None:
            call["latency_s"] = r["latency_s"]
        tok = token_record(resp)
        if tok is not None:
            call["tokens"] = {**tok, **({"policy_version": policy_version} if policy_version is not None else {})}
        out.append(call)
    return out


class A2AHarness:
    def __init__(self, name: str = "a2a", version: str = "1", *, agent: Callable[[], Any] | None = None, agent_url: str | None = None,
                 actor: str = "agent", agent_config: dict[str, Any] | None = None, capture: Any = None, capture_base_url: str | None = None,
                 timeout_s: int = 600, policy_version: int | None = None, limit_policy: LimitPolicy | None = None, sandbox_id: str | None = None,
                 prices: dict[str, list[float]] | None = None, self_hosted: bool | None = None, api: str | None = None,
                 grade_retries: int = 1, scoring_budget_s: float | None = None, control_timeout_s: float | None = 1800.0):
        """``agent``: a factory of an ``AgentEnvAgent`` served in-process per attempt; or ``agent_url`` of a running one.
        ``capture``: a ``connectors.capture_proxy.CaptureProxy`` reachable at ``capture_base_url``.
        ``limit_policy``: how a limit stop is treated (module doc). ``sandbox_id``: what agentenv's liveness check names the
        agent by (default: its URL). ``prices``: USD per 1M [input, output] tokens by model (default: crucible.toml
        ``[prices]``); ``self_hosted``: the upstream costs nothing when unpriced (default: ``policy_version`` is set, or
        every proxy upstream is local). ``api``: the dialect the agent's model calls are configured for ("openai",
        "anthropic" or "responses"), the one Crucible's probe confirms an upstream verdict on (M10 lab audit R-1; default:
        the proxy's only upstream; with several and none given, an upstream verdict stays unconfirmed).
        ``grade_retries`` / ``scoring_budget_s`` / ``control_timeout_s`` (M10 lab audit R-3, as ``NativeHarness``): how often
        a failed grade is graded again inside the attempt, one budget for all its tries, and a control grade's budget."""
        if (agent is None) == (agent_url is None):
            raise ValueError("give exactly one of agent (factory) or agent_url")
        if api is not None and api not in PROXY_APIS:
            raise ValueError(f"api must be one of {PROXY_APIS}, got {api!r}")
        self.api = api
        self.id = f"{name}@{version}"
        self.agent, self.agent_url, self.actor, self.agent_config = agent, agent_url, actor, dict(agent_config or {})
        # the checkpoint version the agent's model server serves (a train rollout needs it on every token record)
        self.capture, self.capture_base_url, self.timeout_s, self.policy_version = capture, capture_base_url, timeout_s, policy_version
        self.limit_policy = limit_policy or LimitPolicy()
        self.sandbox_id, self.prices, self.self_hosted = sandbox_id, prices, self_hosted
        self.grade_retries, self.scoring_budget_s, self.control_timeout_s = grade_retries, scoring_budget_s, control_timeout_s

    def _config(self, key: str | None) -> dict[str, Any]:
        text = json.dumps(self.agent_config)
        if key is not None:
            text = text.replace("${capture.base_url}", (self.capture_base_url or "").rstrip("/")).replace("${capture.key}", key)
        return json.loads(text)

    async def _ask(self, url: str, text: str, sandbox_id: str | None = None) -> tuple[str, str, Any, str | None]:
        """Send the task and wait for a terminal state: (task id, state, terminal response, why). The agent's own overrun
        (or a park in ``input-required``) is ``"timeout"`` and the remote task is cancelled (M11 audit G4 D8: the poll's
        TimeoutError was a transport fault, masked and retried). M10 lab audit R-1: with ``sandbox_id`` (and an agentenv
        that has liveness) an agent that stops answering polls is ``"unreachable"``, with agentenv's reason as ``why``.
        ``AgentUnreachableError`` subclasses TimeoutError, so it is told apart first; ``poll_a2a_task`` already asked the
        agent to cancel (protocol.py:197-198), so it is not cancelled twice."""
        proto = _protocol()
        unreachable = getattr(proto, "AgentUnreachableError", None)
        kw = {"sandbox_id": sandbox_id} if sandbox_id is not None and _has_liveness(proto) else {}
        task_id, _ = await proto.send_a2a_message(url, [{"kind": "text", "text": text}], uuid.uuid4().hex, None, self.timeout_s)
        try:
            task = await proto.poll_a2a_task(url, task_id, self.timeout_s, 1, **kw)
        except TimeoutError as ex:
            if unreachable is not None and isinstance(ex, unreachable):
                return task_id, "unreachable", None, str(ex) or type(ex).__name__
            try:
                await proto.cancel_a2a_task(url, task_id)
            except Exception:  # noqa: BLE001, S110 -- best effort; the agent may already be gone
                pass
            return task_id, "timeout", None, None
        status = task.get("status") or {}
        return task_id, str(status.get("state") or ""), proto.TerminalResponse.from_message(status.get("message") or {}), None

    def _liveness_supported(self) -> bool:
        """Whether the installed agentenv can tell a dead agent from a slow one (recorded on every attempt)."""
        try:
            return _has_liveness(_protocol())
        except ImportError:
            return False

    def _outcome(self, state: str, term: Any, why: str | None, up: UpstreamEvidence) -> AgentEnd:
        """The A2A terminal state through the evidence table (``agent_evidence.classify_agent_end``; module doc item 3).
        The agent's error code and text prefix every failure reason (scrubbed, then cut: a provider can echo a key)."""
        from crucible.core.secrets import scrub_secrets

        ctx = f"A2A task {state}"
        if term is not None:
            said = scrub_secrets(f"{term.error_code or term.error_type or ''} {term.error_message or term.response_text or ''}".strip())[:300]
            ctx += f": {said}" if said else ""
        if state == "unreachable":
            return classify_agent_end(upstream=up, unreachable=why or "the agent stopped answering", context=ctx)
        if state == "timeout":  # Crucible's deadline (timeout_s): a limit stop, unless the proxy shows the provider failing
            return classify_agent_end(upstream=up, deadline=True, context=ctx)
        if state == "failed":
            return classify_agent_end(upstream=up, self_report=f"failed:{(term.error_type if term is not None else None) or 'agent_error'}", context=ctx)
        if state in ("canceled", "rejected"):
            return classify_agent_end(upstream=up, self_report=state, context=ctx)
        return classify_agent_end(upstream=up, context=ctx)

    def _settle_spend(self, res: HarnessResult, path: Path) -> None:
        """M10 lab audit R-1: cost was never set, so every attempt settled $0 (base.py:41, run.py:183). Every row of the
        session's capture log is priced by ``LLMClient._prices``' rule (``capture_spend``: the retry-collapsed calls dropped
        billed retries); a masked attempt paid too."""
        if self.prices is not None:
            prices = self.prices
        else:
            try:
                from crucible.core.config import get_config

                prices = get_config().prices
            except Exception:  # noqa: BLE001 -- an unreadable config prices nothing: hosted calls are booked uncertain
                prices = {}
        free = self_hosted_apis(getattr(self.capture, "upstreams", {}), policy_version=self.policy_version, self_hosted=self.self_hosted)
        res.cost_usd, res.cost_uncertain_usd = capture_spend(path, prices, self_hosted=free)

    async def run(self, attempt: Attempt) -> HarnessResult:
        from crucible.core.secrets import scrub_json, scrub_secrets

        t: dict[str, float] = {}
        res = HarnessResult()
        world = served = key = None
        path: Path | None = None  # the session's capture log, known before its key exists (spend settles on every exit)
        end: AgentEnd | None = None
        try:
            t0 = time.monotonic()
            with boundary("seed"):
                world = await attempt.task.open(attempt.dir / "work")
                world.stream_trace(attempt.dir / "trace.jsonl")
            if not hasattr(world.backend, "agent_mcp_url"):
                raise StageFailure(Failure(kind="crucible:harness_unsupported", stage="seed", reason="the A2A harness needs a World served over MCP (agentenv backend)"))
            # the role-pinned proxy, never the gateway root (M11 audit H2 V6: the agent re-enabled a revoked tool there)
            mcp_url = await world.backend.agent_mcp_url(self.actor)
            t["seed"] = time.monotonic() - t0
            t0 = time.monotonic()
            with boundary("agent"):
                if self.agent is not None:
                    from crucible.interop.agentenv.a2a import ServedAgent  # only to serve one in-process (needs agentenv_protocol)

                    served = await ServedAgent(self.agent()).start()
                url = served.url if served else self.agent_url
                # one capture log per attempt *run*: two runs of one task share attempt ids, and their exchanges were
                # appended to one log (M3 audit: the proxy now refuses a second live key for a session)
                tag = hashlib.sha256(str(Path(attempt.dir).resolve()).encode()).hexdigest()[:8]
                session = (re.sub(r"[^A-Za-z0-9._-]", "_", attempt.id).lstrip("._-") or "attempt") + f"-{tag}"
                if self.capture is not None:
                    # M10 lab audit R-1: set with the session, before the key exists, so a failure before the prompt (an
                    # agent-config refused after a model call) still settles what the proxy forwarded
                    path = Path(self.capture.capture_dir) / f"{session}.jsonl"
                key = self.capture.register(session) if self.capture is not None else None
                async with httpx.AsyncClient(timeout=60) as c:
                    # one server per attempt (H2 V7: the fixed name "world" got 409 on a reused agent, which kept the first
                    # attempt's World). agentenv cannot remove a registration, so an agent holding another World is refused
                    r = await c.post(f"{url}/ext/mcp-config", json={"url": mcp_url, "headers": {"AgentEnv-Role": self.actor}, "name": f"world-{tag}"})
                    if r.status_code != 200:
                        raise StageFailure(Failure(kind="agent_run_error", stage="agent", reason=f"mcp-config refused: {r.status_code} {r.text[:200]}", blame="harness"))
                    listed = await c.get(f"{url}/ext/mcp-config")
                    others = [n for n in ((listed.json() or {}).get("mcp_servers") or {}) if n != f"world-{tag}"] if listed.status_code == 200 else []
                    if others:
                        raise StageFailure(Failure(kind="crucible:harness_unsupported", stage="agent", retryable=False,
                                                   reason=f"the agent already holds other MCP servers {others[:3]}: serve a fresh agent per attempt"))
                    cfg = self._config(key)
                    if cfg:
                        r = await c.post(f"{url}/ext/agent-config", json=cfg)
                        if r.status_code != 200:
                            raise StageFailure(Failure(kind="agent_run_error", stage="agent", reason=f"agent-config refused: {r.status_code} {r.text[:200]}", blame="harness"))
                sid = self.sandbox_id or url
                try:
                    task_id, state, term, why = await self._ask(url, getattr(attempt.task, "instruction", ""), sid)
                finally:
                    # M10 lab audit R-1: the key is revoked when the agent phase ends; it stayed live through grading
                    # (harness.py:204-206), so an agent still running (a timeout, a lost poll) could call the model while
                    # its work was graded. The outer finally's unregister stays (idempotent)
                    if self.capture is not None and key is not None:
                        self.capture.unregister(key)
                res.artifacts["a2a_reply"] = "a2a_reply.txt"
                # scrubbed like the native harness's records (H2 V14: a provider-echoed key reached these files)
                (attempt.dir / "a2a_reply.txt").write_text(scrub_secrets(term.response_text or "") if term is not None else "")
                if state != "unreachable":  # M10 lab audit R-1: a dead agent serves no trajectory; asking it raised a ConnectError
                    try:
                        async with httpx.AsyncClient(timeout=60) as c:
                            tr = await c.post(f"{url}/ext/trajectory", json={"task_id": task_id})
                    except httpx.HTTPError:  # best effort: the agent's own record is evidence, not the attempt's outcome
                        tr = None
                    if tr is not None and tr.status_code == 200:
                        try:
                            (attempt.dir / "agent_trajectory.json").write_text(scrub_json(tr.text))
                        except ValueError:
                            (attempt.dir / "agent_trajectory.json").write_text(scrub_secrets(tr.text))
                        res.artifacts["agent_trajectory"] = "agent_trajectory.json"
                # M10 lab audit R-1 review round 3: the session's calls still in flight upstream (an agent that was stopped
                # does not cancel the proxy's forward) are waited for, so they are read and billed; what is left counts as the
                # session's own load and keeps Crucible's probe off (agent_evidence._self_loaded)
                drain = getattr(self.capture, "drain", None) if self.capture is not None else None
                inflight = await drain(session, DRAIN_S) if drain is not None else 0
                if self.capture is not None:
                    res.calls = {self.actor: calls_from_capture(path, self.actor, self.policy_version)}
                    if any(res.calls.values()):
                        (attempt.dir / "calls.json").write_text(scrub_json(json.dumps(res.calls, default=str)))
                        res.artifacts["calls"] = "calls.json"
                # M10 lab audit R-1: blame from evidence (module doc item 3), not from the agent's own terminal status
                up = dataclasses.replace(read_upstream_evidence(path), inflight=inflight)
                if state not in ("completed", "unreachable") and self.capture is not None:
                    # M10 lab audit R-1: an upstream verdict counts only when Crucible's own probe agrees. The model and the
                    # dialect are the configured ones, never the log's first request: an A2A agent is the policy and writes
                    # that request (its model and its route) too. No configured model, or no configured dialect and several
                    # upstreams: unconfirmed, unknown blame
                    model = self.agent_config.get("model")
                    ups = list(getattr(self.capture, "upstreams", None) or {})
                    api = self.api or (ups[0] if len(ups) == 1 else None)
                    up = await confirm_upstream(up, getattr(self.capture, "probe", None), model if isinstance(model, str) and api else None, api=api)
                end = self._outcome(state, term, why, up)
                res.grade["agent"] = {"state": state, "stop": end.stop, **({"why": end.why} if end.why else {}), "upstream": up.to_dict(),
                                      "liveness": "sandbox_id" if self._liveness_supported() else "unsupported"}
                if end.failure is not None:
                    raise StageFailure(end.failure)
                res.stop, res.truncated = end.stop, end.stop in LIMIT_STOPS
                if end.stop == "agent_done" and term is not None:
                    res.steps = term.tool_call_count or 0
                if res.stop in LIMIT_STOPS and end.rule(self.limit_policy) == "mask":  # as native.py:354-355
                    raise StageFailure(Failure(kind="agent_timeout", stage="agent", reason=f"{res.stop} ({end.why})", mask=True, blame="harness"))
            t["agent"] = time.monotonic() - t0
            (attempt.dir / "trace.json").write_text(scrub_json(world.to_trace().model_dump_json()))
            res.artifacts["trace"] = "trace.json"
            t0 = time.monotonic()
            # M10 lab audit R-3: the World is gone after the attempt (res.regradable stays False), so a failed grade is graded
            # again here, before it closes, under one scoring budget (grading.grade_in_place; a grader's TimeoutError is
            # verifier_timeout). A non-finite reward is verifier_error / non_finite_reward: it raised HarnessResult's
            # allow_inf_nan=False, escaped run(), and the Runner re-ran the agent as agent_run_error@agent
            attempt.phase = "verification"  # an attempt clock firing from here is the grade's (run.py)
            # review round 2: the grading's own window on the Runner's clock (grading.grading_window), so a grade that hangs ends
            # as verifier_timeout and is settled with the control, never cut short by a clock sized for the agent
            budget = grading_window(attempt, attempt.task, budget_s=self.scoring_budget_s, control_s=self.control_timeout_s)
            reps: list[Any] = []
            try:
                with boundary("verification"):
                    _, f = await grade_in_place(lambda d: attempt.task.grade(world, d), attempt.dir / "grade", retries=self.grade_retries,
                                                budget_s=budget, out=reps, attempt=attempt)
            except StageFailure:
                _keep_tries(attempt.dir / "verifier", reps, res)
                raise
            t["verification"] = time.monotonic() - t0
            write_reports(attempt.dir / "verifier", reps)
            # merged, never replaced: grade["agent"] (the evidence, the liveness) must survive into the record
            res.grade = {**res.grade, **grade_summary(reps, f)}
            if f is not None:
                # R-3 item 2: probe while the World runs, close it (nothing of the policy's runs beside the control), then
                # grading.settle decides: scored 0, a task-blamed mask, or no_control
                probe = await liveness(attempt.task, world)
                closing, world = world, None
                with contextlib.suppress(Exception):  # cleanup never overrides a result
                    await closing.close()
                raise StageFailure(await settle(attempt.task, f, workdir=attempt.dir / "control", timeout_s=self.control_timeout_s, probe=probe,
                                                attempt=attempt))
            res.reward, res.gates = reps[-1].reward, reps[-1].gate_bits()
            pen = end.penalty(self.limit_policy) if end is not None and res.stop in LIMIT_STOPS else None
            if pen is not None:  # m13 C-8 (A2-24), as native.py:368-371: the configured penalty, not the partial grade
                res.grade["limit_penalty"] = pen
                res.reward = pen
        except StageFailure as sf:
            res.failure = sf.failure
        finally:
            if self.capture is not None and key is not None:
                self.capture.unregister(key)
            if path is not None:
                try:
                    self._settle_spend(res, path)
                except Exception:  # noqa: BLE001, S110 -- cost bookkeeping never overrides a result
                    pass
            for closer in (world.close if world is not None else None, served.stop if served is not None else None):
                if closer is not None:
                    try:
                        await closer()
                    except Exception:  # noqa: BLE001, S110 -- cleanup never overrides a result
                        pass
            res.timings = {k: round(v, 3) for k, v in t.items()}
        return res
