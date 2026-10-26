"""Evidence-based blame for agents Crucible does not drive itself (Harbor installed agents, A2A agents): the
classification table that turns what the capture proxy and the host observed, plus how the agent ended, into a stop or a
typed ``Failure``; and the spend of an attempt settled from its capture log (M10 lab audit R-1, 2026-10-15).

The rule (DESIGN P16, both ways): a non-policy blame needs evidence from the proxy log (the session's last recorded
upstream exchange) or from the host (the container is gone, the A2A agent stopped answering). What the agent says about
itself never masks its 0 on its own: Harbor's regex error classes (``ApiRateLimitError`` is a regex over the agent's own
output, harbor ``agents/installed/base.py:34-170``), mini-swe-agent's exception exit statuses and A2A ``infra_error`` /
``canceled`` / ``rejected`` self-reports do not count. A non-zero exit with a healthy proxy is the policy's outcome
(scored); an unconfirmed claim of an infrastructure failure is ``unknown`` blame (masked, and counted by the 5%
unknown-blame stop, so a policy that learns to print "rate limit" stops the run instead of hiding its zeros). An
installed agent (``agent_is_harness``) that ends abnormally or at the deadline before the proxy recorded any exchange is
``unknown`` too: the policy (the model behind the proxy) never answered. An A2A agent is the policy, so its own failure
before calling its model stays its outcome.

The proxy log itself is writable by the agent's commands: they hold the capture key, so a policy can send its own
request (a bogus model, an unfundable ``max_tokens``), pick what the upstream answers and end the agent so that row is
last. A non-policy upstream verdict (infra, credits, auth, not_found) therefore counts only when Crucible's own probe of
the same upstream answers the same (``confirm_upstream``, ``CaptureProxy.probe``); an unconfirmed one is ``unknown``
(``detail="unconfirmed:<verdict>"``). The probe asks the configured model on the configured dialect (an A2A agent is the
policy and writes the session's first request too, so its route never picks what Crucible probes). It shares the
session's upstream key, so a failure the session may have caused itself is never confirmed: an infra verdict after more
than ``SELF_LOAD_CALLS`` forwarded calls running in the last ``SELF_LOAD_WINDOW_S``, or while any of the session's calls
is still in flight upstream (the harnesses wait ``DRAIN_S`` for them first), is not probed, and a 429 is probed only
after its Retry-After. A connection the upstream refused (``LOAD_FREE_ERRORS``) is probed at any rate: no bucket or
queue the session fills refuses a connection.

- PATTERN: AReaL ``areal/experimental/openai/proxy/workflow.py:303-342`` (what the proxy saw overrides what the agent
  reported). AReaL rejects a sample on *any* failed call of the session; Crucible deliberately relaxes that to the
  *last* call: the upstreams are third-party providers, and an agent may legitimately retry through a transient 5xx and
  then fail on its own, which is its outcome.
- PATTERN: agentenv ``src/agent_env/a2a_agent/protocol.py:21-32`` (``AgentUnreachableError``: host evidence of a dead
  agent server), mini-swe-agent ``src/minisweagent/agents/default.py:74-145`` (typed exit statuses).

Limit stops carry ``evidenced`` (P23): only Crucible's own deadline (``timeout``) and the proxy's call cap
(``max_steps``) are evidenced, and only an evidenced stop may be masked or penalized by a ``LimitPolicy``
(``AgentEnd.rule``). Both first yield to the proxy's failure evidence (a provider outage that kept the agent retrying
until the deadline, or burned the cap with failed calls, is the provider's). A stop the agent reports (mini-swe-agent
``LimitsExceeded`` / ``TimeExceeded`` / ``RepeatedFormatError`` read from a log file the policy can write, an overflow
class name parsed from its output) is recorded with ``why="agent-reported"`` and always scored: a mask policy must not
let a policy turn its own failure into a mask. An upstream context-overflow answer is recorded (``why`` names it) but
not evidenced: the policy can provoke one with an oversized request of its own.

M10 lab audit R-5 item 3: a ``provider_unavailable`` the proxy's evidence proves carries the last row's Retry-After
(``UpstreamEvidence.retry_after``) as ``Failure.retry_after``, so the Runner's retry waits at least that long.

No framework is imported here: Harbor classes are matched by name (``error_names``).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import os
from collections import deque
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from crucible.actors.llm import CONTEXT_OVERFLOW_MARKERS, UNPRICED_USD_PER_M
from crucible.connectors.capture_proxy import CLIENT_LEFT, LIMIT_STOP_KIND, client_left, limit_stop_count
from crucible.core.config import looks_like_secret_key
from crucible.core.schema.run import Failure
from crucible.core.secrets import scrub_secrets
from crucible.runtime.infra import LimitPolicy

VERDICTS = ("none", "ok", "client_left", "infra", "credits", "auth", "not_found", "context_overflow", "client_error")
# the verdicts that blame someone other than the policy: each needs Crucible's own probe to agree (module doc)
CONFIRMABLE = frozenset({"infra", "credits", "auth", "not_found"})
# provider wordings of an exhausted balance or quota (OpenAI 429 ``insufficient_quota``, Anthropic 400 "credit balance is
# too low"): the provider's own answer, so proxy evidence, not the agent's claim
CREDIT_MARKERS = ("insufficient_quota", "exceeded your current quota", "credit balance is too low", "billing_hard_limit")

# mini-swe-agent's typed exits (default.py:74-145): its own stops, scored. Any other exception gives exit_status=<class name>
CLEAN_EXITS = {"Submitted": "agent_done", "LimitsExceeded": "max_steps", "TimeExceeded": "timeout", "RepeatedFormatError": "too_many_errors"}
OVERFLOW_NAMES = frozenset({"ContextWindowExceededError", "OutputTokenExceededError", "ContextOverflow"})
NON_ZERO_EXIT = "NonZeroAgentExitCodeError"  # harbor agents/installed/base.py:34; every Harbor Api*Error subclasses it
AGENT_REPORTED = "agent-reported"
PROXY_APIS = ("openai", "anthropic", "responses")  # capture_proxy dialects; "responses" goes to the openai upstream
PROBE_TIMEOUT_S = 90
# M10 lab audit R-1: the probe shares the session's upstream key, rate-limit bucket and server queue, which the policy can
# saturate through the capture key. A session that had more than SELF_LOAD_CALLS forwarded calls running in the
# SELF_LOAD_WINDOW_S before its last row's end (an honest sequential agent makes a handful a minute), or that still has
# calls in flight upstream, may have caused the failure itself, so its infra verdict is never probed (unconfirmed:
# unknown). A 429 is probed only after its Retry-After (at most RETRY_AFTER_CAP_S)
SELF_LOAD_WINDOW_S = 60.0
SELF_LOAD_CALLS = 30
RETRY_AFTER_CAP_S = 60.0
# M10 lab audit R-1 review round 3: how long a harness waits for the session's calls still in flight upstream (a killed
# client does not cancel the proxy's forward) before it reads the log; what is left counts as load (``inflight``)
DRAIN_S = 30.0
# M10 lab audit R-1 review round 3: the proxy's own record of a connection the upstream refused or could not resolve (httpx
# ConnectError): no rate-limit bucket or server queue the session fills causes it, so it is probed at any rate (a fast
# honest agent made more than SELF_LOAD_CALLS calls a minute against a fast self-hosted engine, and a real outage stayed
# unconfirmed). Timeouts (ReadTimeout, PoolTimeout, ConnectTimeout), 429, 5xx and stream drops keep the rule
LOAD_FREE_ERRORS = ("ConnectError",)
_RECENT_ROWS = 256  # forwarded rows kept to count the window (past SELF_LOAD_CALLS the exact count does not matter)


@dataclass(frozen=True)
class UpstreamEvidence:
    """What the capture proxy observed for one session: the verdict of its last recorded upstream exchange."""

    calls: int = 0  # recorded exchanges (the proxy's limit-stop rows excluded)
    limit_stops: int = 0  # times the proxy answered its call cap's stop itself
    last_status: int | None = None
    last_error: str | None = None
    verdict: str = "none"  # one of VERDICTS; "none" = no exchange recorded
    detail: str = ""
    # M10 lab audit R-1: the session's first exchange (an installed agent makes it before the policy has written anything),
    # what Crucible's probe of it answered (None: not probed), and how many exchanges the upstream failed (verdict infra;
    # the proxy's call cap counts them too)
    api: str | None = None
    model: str | None = None
    probe: str | None = None
    failed: int = 0
    # M10 lab audit R-1: forwarded calls running within SELF_LOAD_WINDOW_S of the last row's end (review round 3: any that
    # ended in the window, wherever it started: a long generation loads the upstream for its whole span; the proxy's own
    # timestamps), the last row's Retry-After in seconds, and the "<dialect>:<model>" Crucible's probe asked
    recent: int = 0
    retry_after: float | None = None
    probed: str | None = None
    # M10 lab audit R-1 review round 3: the session's forwarded calls still unanswered upstream when the log was read (after
    # the harness's DRAIN_S wait): they have no row yet and still load the upstream the probe would share
    inflight: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verdict_of(row: Mapping[str, Any]) -> tuple[str, str]:
    """(verdict, detail) of one capture row; the first matching rule wins."""
    status = row.get("status") if isinstance(row.get("status"), int) and not isinstance(row.get("status"), bool) else None
    err = scrub_secrets(str(row.get("error") or ""))  # before any cut: the detail is stored in records
    resp = row.get("response")
    if client_left(row):  # the agent's own client left before the answer (it was killed, or gave up; M3 audit R-1, R-2)
        return "client_left", CLIENT_LEFT
    if status == 502 and err and "response" not in row:  # the proxy's own record of an httpx error (capture_proxy forward())
        return "infra", f"connect: {err}"[:200]
    if row.get("complete") is False:  # stream_drop, stream_error, or a connection cut mid-stream
        return "infra", (err or "incomplete stream")[:200]
    body = json.dumps(resp, default=str).casefold() if resp is not None else ""
    if status == 402 or (status in (400, 403, 429) and any(m in body for m in CREDIT_MARKERS)):
        return "credits", f"HTTP {status}"
    if status in (401, 403):
        return "auth", f"HTTP {status}"
    if status == 404:
        return "not_found", "HTTP 404"
    if status in (408, 429) or (status is not None and status >= 500):  # LLMToolAgent's own infra statuses (agents.py:88-93)
        return "infra", f"HTTP {status}"
    if status in (400, 413, 422) and any(m in body for m in CONTEXT_OVERFLOW_MARKERS):
        return "context_overflow", f"HTTP {status}"
    if status is not None and 400 <= status < 500:  # the request's own content: the agent's
        return "client_error", f"HTTP {status}"
    if isinstance(resp, dict) and resp.get("error") and not any(k in resp for k in ("choices", "content", "output")):
        return "infra", "error_body"  # a success status carrying only an error (or a body that was not JSON)
    return "ok", f"HTTP {status}" if status is not None else "ok"


def _log_rows(path: Path | str | None) -> Iterator[dict[str, Any]]:
    """The session log's raw rows (forwarded exchanges and the proxy's limit-stop rows), streamed: the log can be large
    (M10 lab audit R-1). A blank, torn (the proxy is still writing it) or non-object line is skipped."""
    if path is None or not Path(path).exists():
        return
    with Path(path).open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def read_upstream_evidence(path: Path | str | None) -> UpstreamEvidence:
    """The session log's raw rows, not the retry-collapsed exchanges: the last physical row is what the upstream did last.
    A torn line (the proxy is still writing it) is skipped; limit-stop rows are counted, never judged. One streaming pass:
    only the counters, the first row's api and model, the last row and the end (else start) times of the last
    ``_RECENT_ROWS`` forwarded rows (``recent``) are kept. M10 lab audit R-1 review round 3: a row counts when its call ran
    in the window (it ended in it), not only when it started there: a flood of long generations started before the window
    and still loading the upstream was invisible. ``inflight`` (calls with no row yet) is the harness's to set."""
    first: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    calls = stops = failed = 0
    ends: deque[float | None] = deque(maxlen=_RECENT_ROWS)
    for row in _log_rows(path):
        if row.get("kind") == LIMIT_STOP_KIND:
            stops += limit_stop_count(row)
            continue
        calls += 1
        first = row if first is None else first
        last = row
        t_end = _ts(row.get("time"))
        ends.append(t_end if t_end is not None else _ts(row.get("request_time")))
        if verdict_of(row)[0] == "infra":
            failed += 1
    if first is None or last is None:
        return UpstreamEvidence(limit_stops=stops)
    verdict, detail = verdict_of(last)
    status = last.get("status") if isinstance(last.get("status"), int) else None
    err = scrub_secrets(str(last["error"]))[:300] if last.get("error") else None
    req = first.get("request") if isinstance(first.get("request"), Mapping) else {}
    model = req.get("model") if isinstance(req.get("model"), str) else None
    api = str(first["api"]) if first.get("api") in PROXY_APIS else None
    end = _ts(last.get("time"))
    recent = 0 if end is None else sum(1 for e in ends if e is not None and e >= end - SELF_LOAD_WINDOW_S)
    return UpstreamEvidence(calls=calls, limit_stops=stops, last_status=status, last_error=err, verdict=verdict, detail=detail, api=api,
                            model=model, failed=failed, recent=recent, retry_after=_seconds(last.get("retry_after")))


def _ts(x: Any) -> float | None:
    """A capture row's ISO timestamp (``capture_proxy._now``) as epoch seconds, or None."""
    if not isinstance(x, str):
        return None
    try:
        return datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _seconds(x: Any) -> float | None:
    """A Retry-After header's delay in seconds (the HTTP-date form is not read), or None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v >= 0 else None


def _spent_on_failures(up: UpstreamEvidence) -> bool:
    """The proxy's cap was reached and failed upstream calls filled at least half of what it forwarded."""
    return up.limit_stops > 0 and up.failed > 0 and 2 * up.failed >= up.calls


def _self_loaded(up: UpstreamEvidence) -> bool:
    """M10 lab audit R-1: the session blames the upstream's failure (infra, or a cap spent on failed calls) after running
    more than ``SELF_LOAD_CALLS`` forwarded calls in the ``SELF_LOAD_WINDOW_S`` before its last, or (review round 3) with
    calls still in flight upstream: the failure may be its own (a rate-limit bucket or a server queue the policy filled
    through the capture key), and the probe would queue behind them. A last call whose connection the upstream refused
    (``LOAD_FREE_ERRORS``, the proxy's own 502 record) is probed at any rate once nothing is in flight."""
    if not (up.verdict == "infra" or _spent_on_failures(up)):
        return False
    if up.inflight > 0:
        return True
    if up.last_status == 502 and (up.last_error or "").startswith(LOAD_FREE_ERRORS):
        return False
    return up.recent > SELF_LOAD_CALLS


def _probe_note(up: UpstreamEvidence) -> str:
    """What Crucible's probe asked and answered, for a failure reason."""
    if up.probe is None and _self_loaded(up):
        if up.inflight > 0:
            return f"not probed: {up.inflight} of the session's calls were still in flight upstream, so the upstream failure may be its own"
        return (f"not probed: the session ran {up.recent} calls in the {SELF_LOAD_WINDOW_S:.0f}s before its last, so the "
                "upstream failure may be its own")
    return f"Crucible's probe of {up.probed or repr(up.model)} answered {up.probe or 'nothing'}"


async def confirm_upstream(up: UpstreamEvidence, probe: Callable[[str, str], Awaitable[Mapping[str, Any]]] | None,
                           model: str | None, *, api: str | None = None) -> UpstreamEvidence:
    """M10 lab audit R-1: ``up`` with ``probe`` set to the verdict of Crucible's own request (``CaptureProxy.probe``) of
    ``model`` on the dialect ``api``, when ``up`` would blame the upstream (a verdict in ``CONFIRMABLE``, or a call cap
    spent on failed calls); otherwise ``up`` unchanged. ``model``: what the harness configured (Harbor: the session's first
    request, made before the policy wrote anything; A2A: the agent-config model, never a request the agent wrote).
    ``api``: the dialect probed (Harbor: None, so the session's first request's dialect; A2A: the harness's configured
    dialect, never a request the agent wrote: the dialect picks the upstream and the route probed). No probe, no model or
    any error leaves ``probe=None``, which ``classify_agent_end`` treats as unconfirmed. An infra verdict from a session
    under its own load (``_self_loaded``) is never probed (the probe shares its rate-limit bucket and queue); a 429 is
    probed after its Retry-After (at most ``RETRY_AFTER_CAP_S``), when a limit the session filled itself has cleared."""
    if not (up.verdict in CONFIRMABLE or _spent_on_failures(up)) or probe is None or not model:
        return up
    if _self_loaded(up):
        return dataclasses.replace(up, probe=None)
    dialect = api or up.api or "openai"
    if up.last_status == 429 and up.retry_after:
        await asyncio.sleep(min(up.retry_after, RETRY_AFTER_CAP_S))
    try:
        row = await asyncio.wait_for(probe(dialect, model), PROBE_TIMEOUT_S)
    except Exception:  # noqa: BLE001 -- no answer is no confirmation
        return dataclasses.replace(up, probe=None, probed=f"{dialect}:{model}")
    return dataclasses.replace(up, probe=verdict_of(row)[0] if row else None, probed=f"{dialect}:{model}")


@dataclass
class AgentEnd:
    """How an agent's phase ended: a stop (graded) or a typed failure (raised by the harness)."""

    stop: str = "agent_done"
    failure: Failure | None = None
    why: str = ""
    evidenced: bool = False  # Crucible's deadline or the proxy's cap; never the agent's word

    def rule(self, policy: LimitPolicy) -> str:
        """The ``LimitPolicy`` rule for this stop: "score", "mask" or "penalty:<c>". An agent-reported stop is always
        "score" (module doc, P23)."""
        return policy.for_stop(self.stop) if self.evidenced else "score"

    def penalty(self, policy: LimitPolicy) -> float | None:
        return policy.penalty(self.stop) if self.evidenced else None


def error_names(ex: BaseException | None) -> list[str]:
    """The class names of an exception's MRO: Harbor classes are matched by name, with no Harbor import."""
    return [c.__name__ for c in type(ex).__mro__] if ex is not None else []


def _failure(kind: str, reason: str, context: str, **kw: Any) -> AgentEnd:
    # built once through the validator (never model_copy(update=), which skips it): the agent's own text is scrubbed before
    # the 500-character cap, so the cap cannot cut a key into a fragment the patterns no longer match
    text = f"{context} | {reason}" if context else reason
    return AgentEnd(stop="agent_error", failure=Failure(kind=kind, stage="agent", reason=scrub_secrets(text)[:500], **kw), why=kind)


def _upstream_failure(up: UpstreamEvidence, what: str, context: str) -> AgentEnd | None:
    """The failure the proxy's evidence proves, if any. M10 lab audit R-1: a verdict that blames the upstream counts only
    when Crucible's probe answered the same (module doc); otherwise it is ``unknown``, masked and counted by the 5% stop."""
    if up.verdict in CONFIRMABLE and up.probe != up.verdict:
        return _failure("agent_run_error", f"{what}; the last upstream call answered {up.verdict} ({up.detail}) but {_probe_note(up)}: "
                        "the session log is writable by the agent's commands", context, blame="unknown", detail=f"unconfirmed:{up.verdict}")
    if up.verdict == "infra":
        # M10 lab audit R-5 item 3: the last row's Retry-After (a 429 or 503) rides on the failure: ``RetryRule.backoff`` waits at
        # least that long and the provider breaker's cool-down is never shorter. The cap-spent branch below has no such row
        kw = {} if up.retry_after is None else {"retry_after": up.retry_after}
        return _failure("provider_unavailable", f"{what}; the proxy's last upstream call failed: {up.detail}", context, detail=up.detail[:200] or None, **kw)
    if up.verdict == "credits":
        return _failure("crucible:credits_exhausted", f"{what}; the upstream refused for credits or quota ({up.detail})", context, detail="credits")
    if up.verdict in ("auth", "not_found"):
        return _failure("crucible:harness_unsupported", f"{what}; the upstream answered {up.verdict} ({up.detail}): the model name or the proxy's "
                        "upstream key is misconfigured", context, retryable=False, detail=up.verdict)
    return None


def _never_answered(up: UpstreamEvidence, what: str, context: str) -> AgentEnd | None:
    """M10 lab audit R-1: an installed agent ended before the proxy recorded any upstream exchange (and no cap stop): the
    policy (the model behind the proxy) never answered, so the end is not its outcome (P16: a harness crash is never scored
    0). The agent could not reach the proxy (agent-phase allowlist, DNS, a dropped route to the proxy host, a request refused
    before it was recorded: unknown key, 413, non-JSON) or failed before its first call (a config key it rejects, a missing
    binary); nothing tells those apart, so it is ``unknown`` (Recommendation 5, "anything else"): masked, and counted by the
    5% unknown-blame stop, so a run whose agents cannot reach the model stops instead of benching 0s."""
    if up.calls or up.limit_stops:
        return None
    return _failure("agent_run_error", f"{what}; the capture proxy recorded no upstream call: the policy never answered", context,
                    blame="unknown", detail="no_upstream_call")


def classify_agent_end(*, upstream: UpstreamEvidence, error: BaseException | None = None, exit_status: str | None = None,
                       self_report: str | None = None, container_lost: str | None = None, unreachable: str | None = None,
                       deadline: bool = False, agent_is_harness: bool = False, context: str = "") -> AgentEnd:
    """The classification table (first match wins). ``error``: what ``agent.run`` raised (None when it returned);
    ``exit_status``: mini-swe-agent's ``info.exit_status``; ``self_report``: an A2A terminal state ("completed",
    "failed:<error_type>", "canceled", "rejected"); ``container_lost``: the host's reason when the container is gone;
    ``unreachable``: the host's reason when the A2A agent stopped answering; ``deadline``: Crucible's own deadline expired
    (``asyncio.timeout(...).expired()``; a TimeoutError raised inside the agent is an ``error`` like any other).
    ``agent_is_harness``: the agent program is the harness and the policy the model behind the proxy (a Harbor installed
    agent): an abnormal or deadline end with no recorded upstream exchange is ``unknown`` (``_never_answered``). False for
    an A2A agent, which is the policy (test_a2a_harness.py:125).
    ``context`` prefixes every failure reason (the agent's own error text; scrubbed and capped with it)."""
    up = upstream
    if unreachable:
        return _failure("transport_unreachable", f"the agent stopped answering: {unreachable}", context, detail="agent_unreachable")
    if container_lost:
        return _failure("crucible:env_container_lost", f"the task container is gone: {container_lost}", context)
    if deadline:
        # a provider outage that kept the agent retrying until the deadline is the provider's, not the policy's timeout
        found = _upstream_failure(up, "Crucible's deadline expired", context)
        if found is None and agent_is_harness:
            found = _never_answered(up, "Crucible's deadline expired", context)
        return found or AgentEnd(stop="timeout", why="Crucible's deadline", evidenced=True)
    if up.limit_stops > 0:
        # M10 lab audit R-1: the proxy's cap counts every forwarded call, failed ones included (capture_proxy), and an agent
        # retries a failed call without spending a step (mini-swe-agent: up to 10 attempts per step, models/utils/retry.py:21;
        # its step_limit counts steps, agents/default.py:132-147), so a provider outage can burn the cap. As at the deadline,
        # the proxy's evidence goes first: the last forwarded call failed, or failed calls filled at least half of it (the
        # latter confirmed by Crucible's probe too, else unknown)
        found = _upstream_failure(up, "the proxy's call cap was reached", context)
        if found is None and _spent_on_failures(up):
            what = f"the proxy's call cap was reached with {up.failed} of {up.calls} forwarded calls failed upstream"
            if up.probe == "infra":
                found = _failure("provider_unavailable", what, context, detail="cap_spent_on_upstream_failures")
            else:
                found = _failure("agent_run_error", f"{what}, unconfirmed ({_probe_note(up)})", context,
                                 blame="unknown", detail="unconfirmed:cap_spent_on_upstream_failures")
        return found or AgentEnd(stop="max_steps", why="proxy call cap", evidenced=True)
    names = error_names(error)
    abnormal = error is not None or bool(self_report and self_report != "completed") or bool(exit_status and exit_status not in CLEAN_EXITS)
    if not abnormal:
        stop = CLEAN_EXITS.get(exit_status or "Submitted", "agent_done")
        return AgentEnd(stop=stop, why=AGENT_REPORTED if stop != "agent_done" else "")
    claim = names[0] if names else (exit_status if exit_status and exit_status not in CLEAN_EXITS else self_report) or "unknown"
    said = f": {str(error)[:300]}" if error is not None and str(error) else ""
    found = _upstream_failure(up, f"the agent ended abnormally ({claim})", context)
    if found is None and agent_is_harness:
        found = _never_answered(up, f"the agent ended abnormally ({claim}{said})", context)
    if found is not None:
        return found
    if up.verdict == "context_overflow":  # recorded, never evidenced: the policy can provoke it with a request of its own
        return AgentEnd(stop="context_overflow", why=f"upstream context overflow ({up.detail}), not evidenced")
    if OVERFLOW_NAMES & {*names, exit_status or ""}:
        return AgentEnd(stop="context_overflow", why=AGENT_REPORTED)
    last = f"last upstream: {up.verdict}" + (f" {up.detail}" if up.detail else "")
    if NON_ZERO_EXIT in names or self_report == "failed:agent_error":
        return _failure("agent_run_error", f"the agent's exit is its own outcome (unconfirmed claim {claim}{said}; {last})", context,
                        blame="policy", mask=False)
    return _failure("agent_run_error", f"unclassified agent end {claim}{said}; no proxy or host evidence ({last})", context,
                    blame="unknown", detail=f"unmapped:{claim}")


# --------------------------------------------------------------------------------------------- spend


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _tokens(usage: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """(input, output) tokens of an OpenAI or Anthropic usage block. Anthropic's cache writes and reads are input too,
    priced at the input rate (an approximation: writes cost more, reads less)."""
    tin = _num(usage.get("prompt_tokens"))
    if tin is None:
        parts = [_num(usage.get(k)) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")]
        tin = sum(p for p in parts if p is not None) if any(p is not None for p in parts) else None
    tout = _num(usage.get("completion_tokens"))
    if tout is None:
        tout = _num(usage.get("output_tokens"))
    return tin, tout


def _rate(model: Any, prices: Mapping[str, Sequence[float]], *, self_hosted: bool) -> tuple[Sequence[float] | None, bool]:
    """(USD per 1M [input, output], is_fallback) by ``LLMClient._prices``' rule (actors/llm.py:236-244): a priced model is
    certain; an unpriced one costs nothing on a self-hosted upstream, else ``prices["*"]`` or ``UNPRICED_USD_PER_M``."""
    p = prices.get(model) if isinstance(model, str) else None
    if p:
        return p, False
    if self_hosted:
        return None, False
    return prices.get("*") or UNPRICED_USD_PER_M, True


def capture_cost(calls: Sequence[dict[str, Any]], prices: Mapping[str, Sequence[float]], *, self_hosted: bool) -> tuple[float, float]:
    """(certain USD, uncertain USD) of captured calls, by ``LLMClient._prices``' rule (actors/llm.py:236-255): a model
    in ``prices`` is certain; a self-hosted upstream costs 0; a hosted model with no price is booked at ``prices["*"]``
    or ``UNPRICED_USD_PER_M`` as uncertain, so a hard budget still trips. The two are disjoint (the ledger settles their
    sum, budget.py:56-58). Each call gets ``cost_usd`` (priced) or ``cost_uncertain_usd`` (fallback). A call without usage
    costs nothing here. The harnesses settle an attempt's spend with ``capture_spend`` (every forwarded row), not this."""
    certain = uncertain = 0.0
    for c in calls:
        usage = c.get("usage")
        if not isinstance(usage, Mapping):
            continue
        tin, tout = _tokens(usage)
        if tin is None and tout is None:
            continue
        rate, fallback = _rate(c.get("model"), prices, self_hosted=self_hosted)
        if not rate:
            continue
        cost = round((tin or 0.0) / 1e6 * float(rate[0]) + (tout or 0.0) / 1e6 * float(rate[1]), 9)
        if fallback:
            uncertain += cost
            c["cost_uncertain_usd"] = cost
        else:
            certain += cost
            c["cost_usd"] = cost
    return round(certain, 9), round(uncertain, 9)


def _max_output(req: Mapping[str, Any], resp: Any) -> int:
    """The most output tokens a request may have been billed for: its token ceiling (times ``n``), else what was relayed
    (~4 characters a token; a cut stream closes the upstream)."""
    for k in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        v = req.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            n = req.get("n")
            return v * (n if isinstance(n, int) and not isinstance(n, bool) and n > 1 else 1)
    return len(json.dumps(resp, default=str)) // 4


def capture_spend(path: Path | str | None, prices: Mapping[str, Sequence[float]], *, self_hosted: bool | Collection[str]) -> tuple[float, float]:
    """(certain USD, uncertain USD) the upstream billed for one session (M10 lab audit R-1): every forwarded row, never the
    retry-collapsed exchanges (``CaptureLogConnector.fetch`` folds an identical or retry-marked body onto the earlier one,
    but both were forwarded and billed). A complete row with usage is priced from it. A row the upstream answered (int
    status < 400) that has no usage or was cut (``complete`` False: partial usage) is booked as uncertain at
    ``LLMClient``'s blank-usage bound (llm.py:357-361): its max output tokens (else the relayed size) plus the request's
    size (~4 characters a token). A refused or unreachable call (status >= 400, the proxy's 502) costs nothing.
    ``self_hosted``: a bool, or the dialects whose upstream is self-hosted (``self_hosted_apis``)."""
    certain = uncertain = 0.0
    for row in _log_rows(path):
        if row.get("kind") == LIMIT_STOP_KIND:
            continue
        api = str(row.get("api") or "openai")
        free = self_hosted if isinstance(self_hosted, bool) else api in self_hosted
        req = row.get("request") if isinstance(row.get("request"), Mapping) else {}
        resp = row.get("response") if isinstance(row.get("response"), Mapping) else {}
        rate, fallback = _rate(req.get("model"), prices, self_hosted=free)
        if not rate:
            continue
        usage = resp.get("usage")
        tin, tout = _tokens(usage) if isinstance(usage, Mapping) else (None, None)
        status = row.get("status")
        if (tin is not None or tout is not None) and row.get("complete") is not False:
            cost = (tin or 0.0) / 1e6 * float(rate[0]) + (tout or 0.0) / 1e6 * float(rate[1])
            if fallback:
                uncertain += cost
            else:
                certain += cost
        elif isinstance(status, int) and not isinstance(status, bool) and status < 400 \
                and not (resp.get("error") and not any(k in resp for k in ("choices", "content", "output"))):
            uncertain += len(json.dumps(req, default=str)) // 4 / 1e6 * float(rate[0]) + _max_output(req, resp) / 1e6 * float(rate[1])
    return round(certain, 9), round(uncertain, 9)


def is_local_base(url: str | None) -> bool:
    """A self-hosted endpoint by ``LLMClient._self_hosted``' host rule (actors/llm.py:230-234)."""
    host = (urlparse(url or "").hostname or "").casefold()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.endswith(".local")


def self_hosted_apis(upstreams: Mapping[str, Any], *, policy_version: int | None = None, self_hosted: bool | None = None) -> frozenset[str]:
    """The dialects whose upstream is a self-hosted engine, decided per upstream as ``LLMClient._self_hosted`` is per
    provider (M10 lab audit R-1). An explicit ``self_hosted`` wins. Otherwise: a local base, or, with ``policy_version``
    set, the OpenAI upstream (the policy's server: token records are read from OpenAI-shaped responses only). It was one
    global flag, so with ``policy_version`` set a hosted Anthropic upstream on the same proxy was free."""
    if self_hosted is not None:
        return frozenset(PROXY_APIS) if self_hosted else frozenset()
    out = {api for api, up in upstreams.items() if is_local_base(getattr(up, "base", None))}
    if policy_version is not None:
        out.add("openai")
    if "openai" in out:
        out.add("responses")
    return frozenset(out)


def _host_secret(k: Any, v: Any) -> bool:
    return looks_like_secret_key(k) and isinstance(v, str) and len(v) >= 8


def host_secret_names(environ: Mapping[str, str] | None = None) -> list[str]:
    """The sorted names of this host's credential variables (the rule ``leaked_host_secrets`` uses: a secret-shaped name,
    ``core.config.looks_like_secret_key``, and a value of at least 8 characters). M10 lab audit R-1: an installed agent's
    commands get each of them blank, so an agent that reads ``os.environ`` at run time forwards nothing."""
    host = os.environ if environ is None else environ
    return sorted(k for k, v in host.items() if _host_secret(k, v))


def leaked_host_secrets(env: Mapping[str, str], *, allow: Collection[str] = (), environ: Mapping[str, str] | None = None) -> list[str]:
    """The names in ``env`` (what an agent will see) whose value is a credential of this host: the value of a host variable
    with a secret-shaped name (``core.config.looks_like_secret_key``), at least 8 characters, and not in ``allow`` (the
    capture key). An agent's environment must never carry a host credential (M10 lab audit R-1; Harbor's
    ``model_connection.env`` falls back to ``os.environ``)."""
    host = os.environ if environ is None else environ
    allowed = set(allow)
    secret = {v for k, v in host.items() if _host_secret(k, v) and v not in allowed}
    return [k for k, v in env.items() if isinstance(v, str) and v in secret]
