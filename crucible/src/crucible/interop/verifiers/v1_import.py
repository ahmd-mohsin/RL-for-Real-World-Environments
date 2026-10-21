"""verifiers v1 environments trained and evaluated by Crucible (import; m13 E / audit 03 G6, pick 6).

verifiers runs its own episode, with its harness, toolsets, user simulator and ``@reward`` rubric, unchanged (DEPEND),
in its own Python (``crucible/.envs/verifiers``). Crucible supplies only the model: verifiers' ``TrainClient`` renders
prompts to token ids itself and calls the vLLM ``/inference/v1/generate`` protocol. Crucible serves that protocol from its
engines, so verifiers' traces carry exact token records, and they become ``crucible.rollout/v1``.

- ``trace_to_record(trace, rid)``: a v1 trace is a token graph. Each node has ``token_ids``, a trainable ``mask`` (the
  sampled tokens) and ``logprobs``, and its prompt is the concatenation of its ancestors. Every sampled node becomes a
  ``Turn``: the prompt delta since the previous completion, the masked tokens as the completion and their logprobs (the
  generate server's exact values: ``traces.jsonl`` rounds them to 4 decimals). The reward is the trace's weighted rubric
  (``sum(score * weight)``); a trace that is not ``ok``, has an unscored reward, several trainable branches or an
  incomplete token record is masked. M12 audit G4 H6 / pick 6 (verifiers ae819de, ``V`` = ``verifiers/v1``):

  - **truncation** is verifiers' own ``Trace.is_truncated`` (``V/trace.py:620-632``), recomputed because it is a plain
    property and never reaches the dump: the stop conditions ``max_turns``/``max_input_tokens``/``max_output_tokens``/
    ``max_total_tokens``/``compaction_failed`` (``V/session.py:100-112`` names them) or the last error-free call's
    ``finish_reason == "length"`` (``ModelCall.finish_reason``, ``V/trace.py:153-171``), plus ``agent_timeout``;
  - ``Turn.finish`` is the node's call's ``finish_reason`` (``ModelCall.node`` links a call to its node, ``:156``):
    ``length`` stays ``length``, ``stop``/``tool_calls`` are ``stop``; ``Turn.sampling`` is what the call actually sent
    (``ModelCall.sampling``, ``V/types.py:268-292``: temperature, top_p, max_tokens and pass-through keys such as top_k),
    overridden by what Crucible's generate server actually served when the trace ran on it;
  - **truncated sampling is refused, not silently changed**: Crucible's ``gen(ids, max_tokens, temperature)`` samples the
    full distribution and returns no candidate sets, so a call that asked for ``top_p < 1``, ``top_k > 0``,
    ``min_p > 0`` or a repetition / presence / frequency penalty is masked ``crucible:harness_unsupported`` (configure the
    environment with full, unpenalised sampling);
  - the actor is the agent's name (``AgentInfo.name``, ``V/trace.py:105-115``; the dump is a dict, ``:389``), and an
    agent with ``trainable=False`` (a judge, a user simulator: ``:114``) yields a record that never trains:
    masked ``vf:untrainable_agent``, as prime-rl skips it (``PR/orchestrator/algo/base.py:11-18``);
  - **timeouts** (``Trace.record_timeout``, ``V/trace.py:735-740``: a stop, not a failure, ``V/rollout.py:211-215``)
    follow Crucible's M10 limit semantics instead of prime-rl's blanket drop (``PR/orchestrator/dispatcher.py:669-674``,
    which drops every timed-out training trace): ``agent_timeout`` is the policy's own limit stop (truncated, scored
    under ``LimitPolicy.on_timeout``: score / mask / penalty), while ``setup_timeout`` and ``finalize_timeout`` are the
    environment's (masked ``crucible:backend_error``) and ``scoring_timeout`` the verifier's (masked
    ``verifier_timeout``). verifiers' own turn/token limits follow ``on_steps``/``on_context_window`` the same way;
  - the **episode** must be ``ok`` too (``Episode.ok = all(t.ok ...)``, ``V/env.py:298``; prime-rl skips a trace of a
    failed episode, ``algo/base.py:15``).
- ``RolloutGenerateServer``: one HTTP server per runtime, routing ``/r/<rid>/inference/v1/generate`` to that rollout's
  ``gen`` (``Runtime._call``: versions, sample-barrier aborts, staleness). Each completion is stamped with its policy
  version, which ``trace_to_record`` puts on the turn.
- ``vf_rollout(env_id, model)``: the runtime's ``rollout_fn``. A prompt names a task index (``"<idx>"``), and the rollout
  runs ``vf-eval <env> --select.include.idx <idx> -r 1`` against its own route, then converts the trace. A multi-agent
  episode trains its one trainable trace (``agent.trainable``); untrainable traces (judge, user simulator) are context.
  Zero or several trainable traces: no record (one record per trace needs an actor map).
- ``run_vf_eval(...)``: a plain verifiers evaluation against any Crucible engine pool (M10-style), returning the traces.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from crucible.core.schema.rollout import RolloutRecord, Turn

VF_PY = Path(__file__).resolve().parents[4] / ".envs" / "verifiers" / "bin"
# sampling penalties at their neutral values; the generate server applies none (``unsupported_sampling``)
PENALTIES = {"repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def _leaves(nodes: list[dict]) -> list[int]:
    parents = {str(n.get("parent")) for n in nodes if n.get("parent") is not None}
    return [i for i in range(len(nodes)) if str(i) not in parents]


def _chain(nodes: list[dict], leaf: int | None = None) -> list[dict]:
    """The single branch root -> leaf (``parent`` links)."""
    if not nodes:
        return []
    by = {str(i): n for i, n in enumerate(nodes)}
    out, cur = [], nodes[-1 if leaf is None else leaf]
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        out.append(cur)
        p = cur.get("parent")
        cur = by.get(str(p)) if p is not None else None
    return list(reversed(out))


# verifiers error classes -> Crucible failure kinds (M11 audit G2 D15: the raw class name of the oldest error was kept)
VF_ERRORS = {"ProviderError": "provider_unavailable", "SandboxError": "crucible:backend_error", "HarnessError": "agent_run_error",
             "ToolsetError": "crucible:tool_crashed", "InterceptionError": "transport_peer_drop", "TaskError": "verifier_error"}


def _trainable(n: dict) -> bool:
    return bool(n.get("sampled")) and any(n.get("mask") or [])


# verifiers' limit stops (``Trace.is_truncated``, V/trace.py:620-632) -> Crucible's limit-stop names (runtime/infra.LimitPolicy)
VF_LIMIT_STOPS = {"max_turns": "max_steps", "max_input_tokens": "context_overflow", "max_output_tokens": "context_overflow",
                  "max_total_tokens": "context_overflow", "compaction_failed": "context_overflow", "agent_timeout": "timeout"}
# stage deadlines that are not the policy's (V/rollout.py:389, :556, :572): the environment's or the verifier's
VF_INFRA_TIMEOUTS = {"setup_timeout": "crucible:backend_error", "finalize_timeout": "crucible:backend_error", "scoring_timeout": "verifier_timeout"}


def agent_info(trace: dict) -> tuple[str, bool]:
    """(name, trainable) of the trace's agent (``AgentInfo``, V/trace.py:105-115: dumped as a dict)."""
    a = trace.get("agent")
    if isinstance(a, dict):
        return str(a.get("name") or "agent"), bool(a.get("trainable", True))
    return (str(a) if a else "agent"), True


def vf_truncated(trace: dict) -> bool:
    """verifiers' ``Trace.is_truncated`` (V/trace.py:620-632) on a dump (a plain property: not serialized)."""
    if str(trace.get("stop_condition") or "") in ("max_turns", "max_input_tokens", "max_output_tokens", "max_total_tokens", "compaction_failed"):
        return True
    last = next((c for c in reversed(trace.get("calls") or []) if isinstance(c, dict) and c.get("error") is None), None)
    return bool(last and last.get("finish_reason") == "length")


def unsupported_sampling(sampling: dict) -> str | None:
    """A truncated-sampling knob or a sampling penalty Crucible's generate path cannot serve (full-distribution sampling,
    no candidate sets, no penalties). M12 lab audit C-8: a requested penalty was recorded on the turn although the server
    sampled without it, so the trainer, which refuses penalised calls, raised on log-probs that were in fact unbiased."""
    def num(k):
        v = sampling.get(k)
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None
    if (v := num("top_p")) is not None and v < 1.0:
        return f"top_p={v:g}"
    if (v := num("top_k")) is not None and v > 0:
        return f"top_k={v:g}"
    if (v := num("min_p")) is not None and v > 0:
        return f"min_p={v:g}"
    for k, neutral in PENALTIES.items():
        if (v := num(k)) is not None and not abs(v - neutral) <= 1e-9:
            return f"{k}={v:g}"
    return None


def _finish(reason: str | None) -> str | None:
    return None if reason is None else ("length" if reason == "length" else "abort" if reason == "abort" else "stop")


def trace_to_record(trace: dict, rid: str, *, policy_version: int | None = None, task: str = "", harness: str = "verifiers:v1@1",
                    episode: dict | None = None, limit_policy: Any = None) -> RolloutRecord:
    """One record per trace. Masked, never guessed (M11 audit G2 D4-D6, H2 V13), when: the trace (or its episode) is not
    ``ok`` (verifiers' success stamp: earlier attempts' errors stay in ``errors`` after a successful retry); any reward is
    ``None`` (scoring did not run); the trace has more than one trainable branch (subagents, parallel samples); a sampled
    node has no token record; a completion's tokens differ from what Crucible's generate server returned; a call asked for
    truncated sampling; an environment or verifier stage timed out; or the agent is not trainable (module docstring)."""
    from crucible.runtime.infra import LimitPolicy

    limits = limit_policy or LimitPolicy()
    actor, trainable_agent = agent_info(trace)
    turns: list[Turn] = []
    pending: list[int] = []  # tokens since the previous completion (the next prompt delta)
    versions = list(trace.get("_crucible_versions") or [])
    served = trace.get("_crucible_completions")  # [(ids, logprobs)] in call order: exact, unlike traces.jsonl's 4 decimals (D3)
    served_sampling = list(trace.get("_crucible_sampling") or [])  # per served call: what the generate server actually used
    nodes = trace.get("nodes") or []
    calls_by_node = {int(c["node"]): c for c in (trace.get("calls") or []) if isinstance(c, dict) and c.get("node") is not None}
    by_index = {id(n): i for i, n in enumerate(nodes)}
    problem: str | None = None
    unsupported: str | None = None
    branches = [i for i in _leaves(nodes) if any(_trainable(n) for n in _chain(nodes, i))]
    if len(branches) > 1:
        problem = f"{len(branches)} trainable branches (one record per trace)"
    k = 0
    j = 0  # next served completion: matched by content, in order (other agents of the episode share the route)
    for n in _chain(nodes, branches[0] if branches else None):
        ids = list(n.get("token_ids") or [])
        mask = list(n.get("mask") or [False] * len(ids))
        if n.get("sampled") and not any(mask):
            problem = problem or "a sampled node without a token record"
        if not _trainable(n):
            pending += ids
            continue
        first = mask.index(True)
        last = len(mask) - 1 - mask[::-1].index(True)
        if not all(mask[first:last + 1]):
            raise ValueError(f"{rid}: sampled node with a non-contiguous trainable span")
        comp = ids[first:last + 1]
        lps = list(n.get("logprobs") or [])
        lps = lps[first:last + 1] if len(lps) == len(ids) else lps
        call = calls_by_node.get(by_index[id(n)]) or {}
        sampling = {kk: v for kk, v in (call.get("sampling") or {}).items() if v is not None}
        ver = None
        if served is not None:
            hit = next((m for m in range(j, len(served)) if list(served[m][0]) == comp), None)
            if hit is not None:
                lps = list(served[hit][1])
                ver = versions[hit] if hit < len(versions) else None
                if hit < len(served_sampling):
                    sampling = {**sampling, **{kk: v for kk, v in served_sampling[hit].items() if kk != "requested"}}
                    req = served_sampling[hit].get("requested") or {}
                    unsupported = unsupported or unsupported_sampling(req)
                j = hit + 1
            else:
                problem = problem or f"completion {k + 1} differs from what the generate server returned"
        unsupported = unsupported or unsupported_sampling({kk: v for kk, v in (call.get("sampling") or {}).items()})
        if len(lps) != len(comp):
            raise ValueError(f"{rid}: {len(lps)} logprobs for {len(comp)} sampled tokens")
        k += 1
        if ver is None:
            ver = versions[k - 1] if served is None and k - 1 < len(versions) else policy_version
        turns.append(Turn(call_id=f"{rid}:c{k}", actor=actor, prompt_is_delta=bool(turns), prompt_token_ids=pending + ids[:first],
                          completion_token_ids=comp, logprobs=[float(x) for x in lps], policy_version=ver,
                          finish=_finish(call.get("finish_reason")), sampling=sampling))
        pending = ids[last + 1:]
    errors = list(trace.get("errors") or [])
    rewards = trace.get("rewards") or {}
    unscored = [k for k, v in rewards.items() if v is None or not isinstance(v, dict) or v.get("score") is None]
    ok = bool(trace["ok"]) if "ok" in trace else not errors
    if episode is not None and ok and not bool(episode.get("ok", True)):  # V/env.py:298: Episode.ok = all(t.ok)
        ok = False
        errors = errors + list(episode.get("errors") or [])
    reward = sum(float(v["score"]) * float(v.get("weight", 1.0)) for v in rewards.values()) if rewards and not unscored else None
    stop = trace.get("stop_condition")
    truncated = vf_truncated(trace) or stop == "agent_timeout"
    limit_stop = VF_LIMIT_STOPS.get(str(stop or ""))
    if limit_stop is None and truncated:
        limit_stop = "max_steps"  # a length-finished last response: the policy's own output budget
    kind = None
    if not ok:
        last_err = errors[-1] if errors else {}
        t = str(last_err.get("type") or "")
        kind = VF_ERRORS.get(t, f"vf:{t}" if t else "agent_run_error")
    elif str(stop or "") in VF_INFRA_TIMEOUTS:
        kind = VF_INFRA_TIMEOUTS[str(stop)]
    elif unscored or reward is None:
        kind = "verifier_error"
    elif problem:
        kind = "crucible:capture_incomplete"
    elif unsupported:
        kind = "crucible:harness_unsupported"
        problem = f"truncated sampling requested ({unsupported}): Crucible's generate path samples the full distribution"
    elif not trainable_agent:
        kind = "vf:untrainable_agent"
    elif truncated and limit_stop is not None:
        rule = limits.for_stop(limit_stop)
        if rule == "mask":
            kind = f"vf:{stop or 'length'}"
        elif rule.startswith("penalty:"):
            reward = float(rule.split(":", 1)[1])
    masked = kind is not None
    return RolloutRecord(rollout_id=rid, sample=rid, task=task, harness=harness, policy="crucible", turns=turns, reward=None if masked else reward,
                         reward_components={f"vf/{k}": float(v["score"]) for k, v in rewards.items() if isinstance(v, dict) and v.get("score") is not None},
                         mask_sample=masked, failure_kind=kind, stop=stop, truncated=truncated, actor=actor,
                         gates={"vf/trainable": trainable_agent})


class RolloutGenerateServer:
    """``/r/<rid>/inference/v1/generate`` -> that rollout's ``gen``; ``/r/<rid>/v1/models`` for the client's length check."""

    def __init__(self, model: str, *, max_model_len: int = 32768):
        self.model, self.max_model_len = model, max_model_len
        self.gens: dict[str, Callable] = {}
        self.versions: dict[str, list[int]] = {}
        self.completions: dict[str, list[tuple[list[int], list[float]]]] = {}  # exact ids and logprobs per route, in call order
        self.sampling: dict[str, list[dict]] = {}  # per route, in call order: what was served (+ ``requested``: what was asked)
        self._server = None
        self._task: asyncio.Task | None = None
        self.url: str | None = None

    def app(self):
        from fastapi import FastAPI, HTTPException

        app = FastAPI()

        @app.get("/r/{rid}/v1/models")
        async def models(rid: str) -> dict:
            return {"object": "list", "data": [{"id": self.model, "object": "model", "max_model_len": self.max_model_len}]}

        @app.post("/r/{rid}/inference/v1/generate")
        async def generate(rid: str, body: dict) -> dict:
            gen = self.gens.get(rid)
            if gen is None:
                raise HTTPException(status_code=404, detail=f"no live rollout {rid}")
            sp = body.get("sampling_params") or {}
            max_tokens, temperature = int(sp.get("max_tokens") or 512), float(sp.get("temperature", 1.0))
            res = await gen(list(body["token_ids"]), max_tokens, temperature)
            self.versions.setdefault(rid, []).append(int(res.version))
            ids, lps = list(res.completion_ids), list(res.logprobs)
            self.completions.setdefault(rid, []).append((ids, [float(x) for x in lps]))
            # gen samples the full distribution at ``temperature``: top_p/top_k and penalties are not served (trace_to_record
            # masks a call that asked for them: crucible:harness_unsupported)
            self.sampling.setdefault(rid, []).append({"temperature": temperature, "max_tokens": max_tokens, "top_p": 1.0, "top_k": -1,
                                                      "requested": {k: sp.get(k) for k in ("top_p", "top_k", "min_p", *PENALTIES) if sp.get(k) is not None}})
            stops = set(sp.get("stop_token_ids") or ())
            finish = "stop" if (res.finish == "stop" or (ids and ids[-1] in stops)) else "length"
            return {"request_id": rid, "prompt_token_ids": list(body["token_ids"]),
                    "choices": [{"index": 0, "token_ids": ids, "finish_reason": finish, "policy_version": res.version,
                                 "logprobs": {"content": [{"token": f"token_id:{t}", "logprob": float(lp)} for t, lp in zip(ids, lps, strict=True)]}}],
                    "usage": {"prompt_tokens": len(body["token_ids"]), "completion_tokens": len(ids), "total_tokens": len(body["token_ids"]) + len(ids)}}

        return app

    async def start(self) -> str:
        import socket

        import uvicorn

        if self.url is not None:
            return self.url
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self._server = uvicorn.Server(uvicorn.Config(self.app(), host="127.0.0.1", port=port, log_level="warning", lifespan="off", timeout_graceful_shutdown=2))
        self._task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            await asyncio.sleep(0.02)
        self.url = f"http://127.0.0.1:{port}"
        return self.url

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            await asyncio.gather(self._task, return_exceptions=True)


async def run_vf_eval(env_id: str, base_url: str, model: str, *, out_dir: Path, select_idx: str | None = None, rollouts: int = 1, limit: int | None = None,
                      extra: list[str] | None = None, vf_bin: Path = VF_PY, timeout_s: float = 3600) -> list[dict]:
    """Run verifiers' ``vf-eval`` (local subprocess runtime) against a generate endpoint; returns the trace rows."""
    cmd = [str(vf_bin / "vf-eval"), env_id, "-m", model, "-r", str(rollouts), "--no-push", "-o", str(out_dir),
           "--client.type", "train", "--client.base-url", base_url, "--client.api-key-var", "CRUCIBLE_VF_KEY", "--client.renderer-model-name", model,
           "--env.agent.runtime.type", "subprocess", *(["--select.include.idx", select_idx] if select_idx is not None else []),
           *(["--select.limit", str(limit)] if limit is not None else []), *(extra or [])]
    p = await asyncio.create_subprocess_exec(*cmd, env={**os.environ, "CRUCIBLE_VF_KEY": "crucible"}, stdout=asyncio.subprocess.PIPE,
                                             stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(p.communicate(), timeout=timeout_s)
    except TimeoutError:  # never leave the child running (G2 D14)
        p.kill()
        await p.wait()
        raise
    files = list(Path(out_dir).rglob("traces.jsonl"))
    if p.returncode != 0 or not files:
        raise RuntimeError(f"vf-eval {env_id} failed (rc {p.returncode}): {out.decode(errors='replace')[-2000:]}")
    rows = []
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def vf_rollout(env_id: str, model: str, *, server: RolloutGenerateServer | None = None, extra: list[str] | None = None,
               limit_policy: Any = None) -> Callable:
    """The runtime's ``rollout_fn``: the prompt is a task index of ``env_id``; one verifiers episode per rollout."""
    srv = server or RolloutGenerateServer(model)

    async def rollout(rid: str, prompt: str, seed: int, gen: Callable, *, temperature: float = 1.0, harness: str = "verifiers:v1@1"):
        import hashlib

        base = await srv.start()
        key = hashlib.sha1(rid.encode()).hexdigest()[:20]  # runtime ids hold '/', '@', ':': a route key, not the raw id
        srv.gens[key] = gen
        try:
            with tempfile.TemporaryDirectory(prefix="vf-") as d:
                rows = await run_vf_eval(env_id, f"{base}/r/{key}/v1", model, out_dir=Path(d), select_idx=str(prompt), extra=extra)
            episode = rows[0]
            traces = [t for t in episode["traces"] if agent_info(t)[1]]  # untrainable agents (judge, user sim) are context
            if len(traces) != 1:  # zero or several trainable traces: one record per trace needs an actor map (G2 D6)
                return None
            trace = traces[0]
            trace["_crucible_versions"] = srv.versions.pop(key, [])
            trace["_crucible_completions"] = srv.completions.pop(key, [])
            trace["_crucible_sampling"] = srv.sampling.pop(key, [])
            rec = trace_to_record(trace, rid, task=f"{env_id}:{prompt}", harness=harness, episode=episode, limit_policy=limit_policy)
            return None if rec.mask_sample else rec
        finally:
            srv.gens.pop(key, None)
            srv.versions.pop(key, None)
            srv.completions.pop(key, None)
            srv.sampling.pop(key, None)

    rollout.server = srv  # type: ignore[attr-defined]
    return rollout
