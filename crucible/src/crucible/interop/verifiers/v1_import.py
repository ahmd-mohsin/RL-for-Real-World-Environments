"""verifiers v1 environments trained and evaluated by Crucible (import; m13 E / audit 03 G6, pick 6).

verifiers runs its own episode, with its harness, toolsets, user simulator and ``@reward`` rubric, unchanged (DEPEND),
in its own Python (``crucible/.envs/verifiers``). Crucible supplies only the model: verifiers' ``TrainClient`` renders
prompts to token ids itself and calls the vLLM ``/inference/v1/generate`` protocol. Crucible serves that protocol from its
engines, so verifiers' traces carry exact token records, and they become ``crucible.rollout/v1``.

- ``trace_to_record(trace, rid)``: a v1 trace is a token graph. Each node has ``token_ids``, a trainable ``mask`` (the
  sampled tokens) and ``logprobs``, and its prompt is the concatenation of its ancestors. Every sampled node becomes a
  ``Turn``: the prompt delta since the previous completion, the masked tokens as the completion and their logprobs. The
  reward is the trace's weighted rubric (``sum(score * weight)``); a trace with errors is masked.
- ``RolloutGenerateServer``: one HTTP server per runtime, routing ``/r/<rid>/inference/v1/generate`` to that rollout's
  ``gen`` (``Runtime._call``: versions, sample-barrier aborts, staleness). Each completion is stamped with its policy
  version, which ``trace_to_record`` puts on the turn.
- ``vf_rollout(env_id, model)``: the runtime's ``rollout_fn``. A prompt names a task index (``"<idx>"``), and the rollout
  runs ``vf-eval <env> --select.include.idx <idx> -r 1`` against its own route, then converts the trace.
- ``run_vf_eval(...)``: a plain verifiers evaluation against any Crucible engine pool (M10-style), returning the traces.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from crucible.core.schema.rollout import RolloutRecord, Turn

VF_PY = Path(__file__).resolve().parents[4] / ".envs" / "verifiers" / "bin"


def _chain(nodes: list[dict]) -> list[dict]:
    """The single branch root -> leaf (``parent`` links)."""
    if not nodes:
        return []
    by = {str(i): n for i, n in enumerate(nodes)}
    out, cur = [], nodes[-1]
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        out.append(cur)
        p = cur.get("parent")
        cur = by.get(str(p)) if p is not None else None
    return list(reversed(out))


def trace_to_record(trace: dict, rid: str, *, policy_version: int | None = None, task: str = "", harness: str = "verifiers:v1@1") -> RolloutRecord:
    turns: list[Turn] = []
    pending: list[int] = []  # tokens since the previous completion (the next prompt delta)
    versions = list(trace.get("_crucible_versions") or [])
    k = 0
    for n in _chain(trace.get("nodes") or []):
        ids = list(n.get("token_ids") or [])
        mask = list(n.get("mask") or [False] * len(ids))
        if not n.get("sampled") or not any(mask):
            pending += ids
            continue
        first = mask.index(True)
        last = len(mask) - 1 - mask[::-1].index(True)
        if not all(mask[first:last + 1]):
            raise ValueError(f"{rid}: sampled node with a non-contiguous trainable span")
        comp = ids[first:last + 1]
        lps = list(n.get("logprobs") or [])
        lps = lps[first:last + 1] if len(lps) == len(ids) else lps
        if len(lps) != len(comp):
            raise ValueError(f"{rid}: {len(lps)} logprobs for {len(comp)} sampled tokens")
        k += 1
        ver = versions[k - 1] if k - 1 < len(versions) else policy_version
        turns.append(Turn(call_id=f"{rid}:c{k}", actor=str(trace.get("agent") or "agent"), prompt_is_delta=bool(turns),
                          prompt_token_ids=pending + ids[:first], completion_token_ids=comp, logprobs=[float(x) for x in lps], policy_version=ver))
        pending = ids[last + 1:]
    errors = trace.get("errors") or []
    rewards = trace.get("rewards") or {}
    reward = sum(float(v.get("score", 0.0)) * float(v.get("weight", 1.0)) for v in rewards.values()) if rewards else None
    masked = bool(errors) or reward is None
    return RolloutRecord(rollout_id=rid, sample=rid, task=task, harness=harness, policy="crucible", turns=turns, reward=None if masked else reward,
                         reward_components={f"vf/{k}": float(v.get("score", 0.0)) for k, v in rewards.items()}, mask_sample=masked,
                         failure_kind=(errors[0].get("type") if errors else None), stop=trace.get("stop_condition"))


class RolloutGenerateServer:
    """``/r/<rid>/inference/v1/generate`` -> that rollout's ``gen``; ``/r/<rid>/v1/models`` for the client's length check."""

    def __init__(self, model: str, *, max_model_len: int = 32768):
        self.model, self.max_model_len = model, max_model_len
        self.gens: dict[str, Callable] = {}
        self.versions: dict[str, list[int]] = {}
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
            res = await gen(list(body["token_ids"]), int(sp.get("max_tokens") or 512), float(sp.get("temperature", 1.0)))
            self.versions.setdefault(rid, []).append(int(res.version))
            ids, lps = list(res.completion_ids), list(res.logprobs)
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
    out, _ = await asyncio.wait_for(p.communicate(), timeout=timeout_s)
    files = list(Path(out_dir).rglob("traces.jsonl"))
    if p.returncode != 0 or not files:
        raise RuntimeError(f"vf-eval {env_id} failed (rc {p.returncode}): {out.decode(errors='replace')[-2000:]}")
    rows = []
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def vf_rollout(env_id: str, model: str, *, server: RolloutGenerateServer | None = None, extra: list[str] | None = None) -> Callable:
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
            trace = rows[0]["traces"][0]
            trace["_crucible_versions"] = srv.versions.pop(key, [])
            rec = trace_to_record(trace, rid, task=f"{env_id}:{prompt}", harness=harness)
            return None if rec.mask_sample else rec
        finally:
            srv.gens.pop(key, None)
            srv.versions.pop(key, None)

    rollout.server = srv  # type: ignore[attr-defined]
    return rollout
