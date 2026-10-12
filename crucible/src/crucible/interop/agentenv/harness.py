"""Any agentenv A2A agent as a Crucible harness (m13 C7; audit 02). DEPEND on ``agent_env.a2a_agent.protocol`` (message
send / poll, terminal parsing) and the agentenv-protocol extensions the agent advertises.

For each attempt the harness opens the task's World (it must be served over MCP: the agentenv backend's gateway), then:

1. **mcp-config** (``urn:agentenv:mcp-config/v1``): registers the World's MCP URL with the actor's role header
   (``AgentEnv-Role``), so the agent sees exactly the role-filtered tools the native harness would.
2. **agent-config** (``urn:agentenv:agent-config/v1``): the agent's own config fields (model, ``max_turns``, ...).
   ``${capture.base_url}`` and ``${capture.key}`` in it are replaced by Crucible's capture proxy and a per-attempt
   session key, so the agent's model calls go through the proxy (the upstream key never reaches the agent).
3. **prompt**: the task instruction as an A2A message; the terminal status decides. The agent is the policy, so its
   own ``agent_error`` is an outcome (blame ``policy``, scored); an ``infra_error``, ``canceled`` or ``rejected`` is
   infrastructure (blame ``infra``, masked). The agent's error code and text go into the failure.
4. **trajectory/v1**: the agent's native trajectory is saved next to the World trace (``agent_trajectory.json``).
5. **calls**: the capture log becomes the actor's model calls; a call has a token record only when the upstream
   returned prompt and completion token ids with logprobs (vLLM-style), stamped with ``policy_version`` (the checkpoint
   the agent's model server serves). In ``rollout_mode="train"`` a rollout
   without complete token records is masked ``crucible:capture_incomplete`` by the record builder (no capture, no
   training data: the trainer never guesses logprobs).

Grading is the task's own (``task.grade(world, dir)``), exactly as for the native harness.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from crucible.core.schema.run import Failure
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.infra import StageFailure, boundary, classify_verifier

FAILED_STATES = {"failed", "canceled", "rejected"}


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


def calls_from_capture(path: Path, actor: str, policy_version: int | None = None) -> list[dict[str, Any]]:
    """The capture log as model calls, with ``CaptureLogConnector``'s retry rule: a repeated request collapses onto its
    earlier attempt only when it is marked as a retry (or the earlier attempt failed); two identical genuine calls
    (best-of-n, parallel subagents) stay two calls. M6 audit: every repeated digest collapsed, which shifted every later
    ``call_id`` and misplaced credit."""
    if not Path(path).exists():
        return []
    from crucible.connectors.base import SourceRef
    from crucible.connectors.capture_proxy import CaptureLogConnector

    rows = CaptureLogConnector().fetch(SourceRef(connector="capture_log", key=Path(path).stem, meta={"path": str(path)})).payload["exchanges"]
    out = []
    for i, r in enumerate(rows, 1):
        resp = r.get("response") or {}
        ch = (resp.get("choices") or [{}])[0]
        call: dict[str, Any] = {"call_id": f"{actor}:cap{i}", "model": (r.get("request") or {}).get("model"), "usage": resp.get("usage") or {},
                                "finish": ch.get("finish_reason"), "kind": "turn"}
        tok = token_record(resp)
        if tok is not None:
            call["tokens"] = {**tok, **({"policy_version": policy_version} if policy_version is not None else {})}
        out.append(call)
    return out


class A2AHarness:
    def __init__(self, name: str = "a2a", version: str = "1", *, agent: Callable[[], Any] | None = None, agent_url: str | None = None,
                 actor: str = "agent", agent_config: dict[str, Any] | None = None, capture: Any = None, capture_base_url: str | None = None,
                 timeout_s: int = 600, policy_version: int | None = None):
        """``agent``: a factory of an ``AgentEnvAgent`` served in-process per attempt; or ``agent_url`` of a running one.
        ``capture``: a ``connectors.capture_proxy.CaptureProxy`` reachable at ``capture_base_url``."""
        if (agent is None) == (agent_url is None):
            raise ValueError("give exactly one of agent (factory) or agent_url")
        self.id = f"{name}@{version}"
        self.agent, self.agent_url, self.actor, self.agent_config = agent, agent_url, actor, dict(agent_config or {})
        # the checkpoint version the agent's model server serves (a train rollout needs it on every token record)
        self.capture, self.capture_base_url, self.timeout_s, self.policy_version = capture, capture_base_url, timeout_s, policy_version

    def _config(self, key: str | None) -> dict[str, Any]:
        text = json.dumps(self.agent_config)
        if key is not None:
            text = text.replace("${capture.base_url}", (self.capture_base_url or "").rstrip("/")).replace("${capture.key}", key)
        return json.loads(text)

    async def _ask(self, url: str, text: str) -> tuple[str, str, Any]:
        from agent_env.a2a_agent.protocol import TerminalResponse, poll_a2a_task, send_a2a_message

        task_id, _ = await send_a2a_message(url, [{"kind": "text", "text": text}], uuid.uuid4().hex, None, self.timeout_s)
        task = await poll_a2a_task(url, task_id, self.timeout_s, 1)
        status = task.get("status") or {}
        return task_id, str(status.get("state") or ""), TerminalResponse.from_message(status.get("message") or {})

    async def run(self, attempt: Attempt) -> HarnessResult:
        from crucible.interop.agentenv.a2a import ServedAgent

        t: dict[str, float] = {}
        res = HarnessResult()
        world = served = key = None
        try:
            t0 = time.monotonic()
            with boundary("seed"):
                world = await attempt.task.open(attempt.dir / "work")
                world.stream_trace(attempt.dir / "trace.jsonl")
            mcp_url = getattr(world.backend, "mcp_url", None)
            if not mcp_url:
                raise StageFailure(Failure(kind="crucible:harness_unsupported", stage="seed", reason="the A2A harness needs a World served over MCP (agentenv backend)"))
            t["seed"] = time.monotonic() - t0
            t0 = time.monotonic()
            with boundary("agent"):
                if self.agent is not None:
                    served = await ServedAgent(self.agent()).start()
                url = served.url if served else self.agent_url
                # one capture log per attempt *run*: two runs of one task share attempt ids, and their exchanges were
                # appended to one log (M3 audit: the proxy now refuses a second live key for a session)
                tag = hashlib.sha256(str(Path(attempt.dir).resolve()).encode()).hexdigest()[:8]
                session = (re.sub(r"[^A-Za-z0-9._-]", "_", attempt.id).lstrip("._-") or "attempt") + f"-{tag}"
                key = self.capture.register(session) if self.capture is not None else None
                async with httpx.AsyncClient(timeout=60) as c:
                    r = await c.post(f"{url}/ext/mcp-config", json={"url": mcp_url, "headers": {"AgentEnv-Role": self.actor}, "name": "world"})
                    if r.status_code not in (200, 409):
                        raise StageFailure(Failure(kind="agent_run_error", stage="agent", reason=f"mcp-config refused: {r.status_code} {r.text[:200]}", blame="harness"))
                    cfg = self._config(key)
                    if cfg:
                        r = await c.post(f"{url}/ext/agent-config", json=cfg)
                        if r.status_code != 200:
                            raise StageFailure(Failure(kind="agent_run_error", stage="agent", reason=f"agent-config refused: {r.status_code} {r.text[:200]}", blame="harness"))
                task_id, state, term = await self._ask(url, getattr(attempt.task, "instruction", ""))
                res.artifacts["a2a_reply"] = "a2a_reply.txt"
                (attempt.dir / "a2a_reply.txt").write_text(term.response_text or "")
                async with httpx.AsyncClient(timeout=60) as c:
                    tr = await c.post(f"{url}/ext/trajectory", json={"task_id": task_id})
                    if tr.status_code == 200:
                        (attempt.dir / "agent_trajectory.json").write_text(tr.text)
                        res.artifacts["agent_trajectory"] = "agent_trajectory.json"
                if self.capture is not None:
                    res.calls = {self.actor: calls_from_capture(Path(self.capture.capture_dir) / f"{session}.jsonl", self.actor, self.policy_version)}
                    if any(res.calls.values()):
                        (attempt.dir / "calls.json").write_text(json.dumps(res.calls, default=str))
                        res.artifacts["calls"] = "calls.json"
                if state in FAILED_STATES:
                    # the agent is the policy: its own agent_error is an outcome (scored); an infra_error, a cancel or a
                    # rejection is the harness's infrastructure (masked)
                    own = state == "failed" and (term.error_type or "agent_error") == "agent_error"
                    raise StageFailure(Failure(kind="agent_run_error", stage="agent", blame="policy" if own else "infra",
                                               reason=f"A2A task {state}: {term.error_code or term.error_type or ''} {term.error_message or term.response_text or ''}"[:400]))
                res.steps = term.tool_call_count or 0
                res.stop = "agent_done"
            t["agent"] = time.monotonic() - t0
            (attempt.dir / "trace.json").write_text(world.to_trace().model_dump_json())
            res.artifacts["trace"] = "trace.json"
            t0 = time.monotonic()
            with boundary("verification"):
                rep = await attempt.task.grade(world, attempt.dir / "grade")
            t["verification"] = time.monotonic() - t0
            res.grade = {"reward": rep.reward, "gates": rep.gate_bits(), "infra": rep.infra}
            if rep.infra.get("status") == "masked":
                raise StageFailure(classify_verifier(rep.infra))
            res.reward, res.gates = rep.reward, rep.gate_bits()
        except StageFailure as sf:
            res.failure = sf.failure
        finally:
            if self.capture is not None and key is not None:
                self.capture.unregister(key)
            for closer in (world.close if world is not None else None, served.stop if served is not None else None):
                if closer is not None:
                    try:
                        await closer()
                    except Exception:  # noqa: BLE001, S110 -- cleanup never overrides a result
                        pass
            res.timings = {k: round(v, 3) for k, v in t.items()}
        return res
