"""slime's custom generate function for Crucible coding tasks (swarm-pipeline/00 §11 item 3, §11.3 items 4-5; 03 §5 item 3).

``--custom-generate-function-path crucible.train.bridges.slime.generate.generate``; slime calls
``generate(args, sample, sampling_params, evaluation=False)`` (``slime/rollout/sglang_rollout.py:336-343``) and accepts a
``list[Sample]`` (one trajectory may be several segments).

**One episode** (the task side comes from the env provider, ``factory.envs.slime_provider`` by default):
1. resolve ``sample.metadata`` (``crucible_task_ref``) to a task env; refuse any row that is not ``split="train"`` (V9);
2. open the World; its ``Sandbox`` (``runtime.harness.sandbox``) is all the agent harness touches;
3. the checkpoint observer starts; the adapter session's ``before_model_call`` hook reports the previous turn's actions
   to it, so every checkpoint snapshot is taken before a model call, on a quiescent workspace (03 §4.3);
4. the harness runs under the **agent timeout**: expiring it is a limit stop (``TRUNCATED``), and the work is graded;
5. the end snapshot (the patch against ``base``), then the World is closed and the patch is graded **in a fresh
   container** with the trail of the tool calls the model issued (path checks; V3, V4);
6. ``R = phi_weight·Φ_final + (1 − phi_weight)·[every gate]`` (0.5 / 0.5; 0 on a hack-severity path violation): the task's
   own rule, recomputed from the grade report and cross-checked against the report's reward;
7. the sample carries ``metadata.crucible``: ``turn_credit`` (ΔΦ per turn with its token span, mapped exactly as the
   native record builder maps progress rows to calls: ``runtime.record.credit_from_progress``), STATUS claims and the
   honesty term, give-up, gates, Φ, the objective id and the verifier version.

**Masking (V8).** The whole episode runs under the **rollout timeout**; expiring it, a dead sandbox
(``SandboxTerminatedError``, or the World found dead at the end), a World that would not open, a grade the verifier could
not measure, or any other exception returns one **removed sample** (``remove_sample=True``, ``loss_mask=[0]``; slime's
``_abort_result`` shape, ``examples/coding_agent_rl/generate.py:446-463``). slime's own example trains a dead sandbox as
reward 0 (:369-399); this one never does, and ``filters.group_filter`` drops the whole GRPO group.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

from crucible.runtime.harness.sandbox import EXIT_TIME_BUDGET_EXCEEDED, SandboxTerminatedError
from crucible.runtime.record import credit_from_progress
from crucible.train.bridges.slime.episode import AdapterSession, AgentHarness, TurnModel
from crucible.train.bridges.slime.types import sample_types

DEFAULT_PROVIDER = "crucible.factory.envs:slime_provider"
AGENT_TIMEOUT_S = 3600.0
ROLLOUT_TIMEOUT_S = 7200.0


class TaskEnv(Protocol):
    """What ``generate`` needs from a task (``factory.envs.SlimeTaskEnv`` implements it for every ``LongTask``)."""

    task_id: str
    prompt: str
    objective_id: str
    verifier_version: str
    phi_weight: float

    async def open(self, workdir: Path) -> Any: ...  # a handle with ``.sandbox`` (the Sandbox) and ``.world``

    def observer(self, handle: Any, out_dir: Path) -> Any: ...  # the checkpoint sidecar (on_start / on_step / on_end), or None

    async def snapshot(self, handle: Any) -> str: ...

    async def alive(self, handle: Any) -> bool: ...

    async def close(self, handle: Any) -> None: ...

    async def grade(self, patch: str, out_dir: Path, *, trail: list[dict[str, Any]]) -> Any: ...  # a GradeReport

    def harness(self, agent: str) -> AgentHarness: ...

    def model(self, metadata: dict[str, Any], sampling_params: dict[str, Any]) -> TurnModel: ...


def load_pointer(spec: str) -> Any:
    """``pkg.mod:attr`` or slime's dotted ``pkg.mod.attr`` (``slime/utils/misc.py:51-59``)."""
    mod, _, attr = spec.partition(":") if ":" in spec else spec.rpartition(".")
    return getattr(importlib.import_module(mod), attr)


def provider_of(args: Any) -> Any:
    spec = getattr(args, "crucible_env_provider", None) or os.environ.get("CRUCIBLE_SLIME_ENV") or DEFAULT_PROVIDER
    if hasattr(spec, "resolve"):  # an in-process provider object (tests, scripts)
        return spec
    p = load_pointer(spec)
    return p(args) if callable(p) and not hasattr(p, "resolve") else p


def _knob(args: Any, md: dict[str, Any], name: str, default: float) -> float:
    v = md.get(name) or getattr(args, f"crucible_{name}", None) or os.environ.get(f"CRUCIBLE_SLIME_{name.upper()}")
    return float(v) if v else default


def removed_sample(sample: Any, reason: str, *, extra: dict[str, Any] | None = None) -> Any:
    """slime's ``_abort_result`` shape (``generate.py:446-463``): no trainable token, ``remove_sample``; the reward field is
    a placeholder that no statistic reads (``credit`` and ``filters`` skip removed samples; V8)."""
    S, _ = sample_types()
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.remove_sample = True
    sample.status = S.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "crucible_masked": True, "abort_reason": reason, **(extra or {})}
    return sample


def outcome_reward(rep: Any, phi_weight: float) -> tuple[float, str]:
    """``phi_weight·Φ + (1 − phi_weight)·[every gate]``, 0 on a hack-severity path violation or a patch that did not apply
    (``WindowTask.grade_patch``'s rule); a task without Φ in its report is graded by its own reward."""
    m = rep.meta or {}
    if "phi" not in m or m.get("phi") is None:
        return float(rep.reward), "task reward"
    if m.get("hack_suspect") or m.get("apply_errors"):
        return 0.0, "0 (hack_suspect or unapplied)"
    full = bool(m.get("full_pass"))
    return round(phi_weight * float(m["phi"]) + (1.0 - phi_weight) * float(full), 6), f"{phi_weight:g}*phi + {1 - phi_weight:g}*[all gates]"


def _actions(turn: Any) -> list[Any]:
    """A turn's actions as Crucible actions (what the native World step would see): tool calls, else ``Done``."""
    from crucible.core.schema.actions import Done, ToolCall

    if turn.actions:
        return [ToolCall(tool_call_id=str(a.get("id") or f"t{turn.index}"), function_name=f"repo.{a['tool']}", arguments=dict(a.get("arguments") or {}))
                for a in turn.actions]
    return [Done(reason=turn.text[:2000], answer=turn.text)]


async def generate(args: Any, sample: Any, sampling_params: dict[str, Any], evaluation: bool = False) -> list[Any]:
    md = dict(sample.metadata or {})
    if not evaluation and md.get("split") != "train":
        return [removed_sample(sample, f"refused: split {md.get('split')!r} is not train")]
    try:
        env = provider_of(args).resolve(md)
    except Exception as e:  # noqa: BLE001 -- an unresolvable row is infrastructure, never a 0
        return [removed_sample(sample, f"resolve:{type(e).__name__}: {e}"[:300])]
    rollout_s = _knob(args, md, "rollout_timeout_s", ROLLOUT_TIMEOUT_S)
    t0 = time.monotonic()
    try:
        async with asyncio.timeout(rollout_s):
            out = await _episode(args, env, sample, md, sampling_params, evaluation)
    except TimeoutError:
        return [removed_sample(sample, "rollout_timeout", extra={"rollout_timeout_s": rollout_s})]
    except SandboxTerminatedError as e:
        return [removed_sample(sample, "sandbox_terminated", extra={"error": str(e)[:300]})]
    except Exception as e:  # noqa: BLE001 -- anything else the episode raised is not the policy's outcome
        return [removed_sample(sample, f"exception:{type(e).__name__}", extra={"error": str(e)[:300]})]
    for s in out:
        s.non_generation_time = getattr(s, "non_generation_time", 0.0) or 0.0
        s.metadata.setdefault("crucible", {})["seconds"] = round(time.monotonic() - t0, 3)
    return out


async def _episode(args: Any, env: Any, sample: Any, md: dict[str, Any], sampling_params: dict[str, Any], evaluation: bool) -> list[Any]:
    S, _ = sample_types()
    agent_s = _knob(args, md, "agent_timeout_s", AGENT_TIMEOUT_S)
    root = Path(tempfile.mkdtemp(prefix="crucible-slime-", dir=getattr(args, "crucible_workdir", None) or os.environ.get("CRUCIBLE_SLIME_WORKDIR") or None))
    out_dir = root / "episode"
    out_dir.mkdir(parents=True, exist_ok=True)
    harness = env.harness(str(md.get("agent") or "scripted"))
    model = env.model(md, sampling_params)
    handle = await env.open(root / "work")
    calls: list[dict[str, Any]] = []
    try:
        world = getattr(handle, "world", None)
        obs = env.observer(handle, out_dir)
        if obs is not None:
            await obs.on_start(world)
        step = 0
        reported = 0

        async def flush(session: AdapterSession) -> None:
            """Report the turns not yet reported to the observer, then check the World is alive (the native ``InfraWatch``
            checks after every World step: a World that died is infrastructure, wherever it died)."""
            nonlocal step, reported
            fresh = reported < len(session.turns)
            while reported < len(session.turns):
                t = session.turns[reported]
                reported += 1
                acts = _actions(t)
                calls.append({"call_id": f"agent:c{t.index}", "actions": [a.model_dump() for a in acts]})
                for a in acts:
                    step += 1
                    if obs is not None:
                        await obs.on_step(world, step, "agent", a, None)
            if fresh and not await env.alive(handle):
                raise SandboxTerminatedError(f"the World died (after step {step})")

        sid = sample.session_id or f"crucible-{env.task_id}-{sample.index}-{sample.group_index}"
        session = AdapterSession(sid, model, sampling_params=sampling_params, before_model_call=flush)
        exit_code = await harness.run(handle.sandbox, session, env.prompt, time_budget_s=agent_s)
        await flush(session)
        progress = await obs.on_end(world) if obs is not None else {}
        if not await env.alive(handle):
            raise SandboxTerminatedError("the World died during the episode")
        patch = await env.snapshot(handle)
    finally:
        try:
            await env.close(handle)
        except Exception:  # noqa: BLE001, S110 -- cleanup never overrides the result
            pass
    trail = list(getattr(harness, "trail", []) or [])
    rep = await env.grade(patch, out_dir / "grade", trail=trail)
    if (rep.infra or {}).get("status") != "ok":
        return [removed_sample(sample, "grade_unmeasured", extra={"why": str((rep.infra or {}).get("why"))[:300]})]
    reward, rule = outcome_reward(rep, float(env.phi_weight))
    m = rep.meta or {}
    rows = [json.loads(x) for x in (out_dir / "progress.jsonl").read_text().splitlines() if x.strip()] if (out_dir / "progress.jsonl").exists() else []
    per_call, drift = credit_from_progress(calls, rows, "agent") if rows else ({}, 0.0)
    prompt_ids, resp, mask, lps, spans = session.linearize()
    turn_credit = [{"turn": t.index, "call_id": f"agent:c{t.index}", "delta_phi": round(per_call.get(f"agent:c{t.index}", 0.0), 9), "span": list(sp)}
                   for t, sp in zip(session.turns, spans, strict=True)]
    hon = progress.get("honesty") or {}
    truncated = exit_code == EXIT_TIME_BUDGET_EXCEEDED
    sample.tokens = prompt_ids + resp
    sample.response_length = len(resp)
    sample.response = "\n".join(t.text for t in session.turns if t.text)
    sample.loss_mask = mask
    sample.rollout_log_probs = lps
    sample.reward = reward
    sample.remove_sample = False
    sample.weight_versions = [v for t in session.turns for v in t.weight_versions]
    sample.status = S.Status.TRUNCATED if truncated else S.Status.COMPLETED
    if sample.rollout_id is None:
        sample.rollout_id = sample.index
    sample.metadata = {**md, "raw_reward": reward, "crucible": {
        "task_id": env.task_id, "objective_id": env.objective_id, "verifier_version": env.verifier_version, "reward": reward, "reward_rule": rule,
        "report_reward": rep.reward, "phi": m.get("phi"), "phi_final": progress.get("phi_final"), "full_pass": m.get("full_pass"),
        "milestones": m.get("milestones"), "gates": rep.gate_bits() if hasattr(rep, "gate_bits") else {}, "hack_suspect": bool(m.get("hack_suspect")),
        "hack_confirmed": 0.0, "path_violations": (m.get("path") or {}).get("violations", []), "turn_credit": turn_credit, "credit_drift": round(drift, 9),
        "status_claims": {k: v[0] for k, v in (hon.get("scored") or {}).items()}, "honesty": hon, "honesty_term": float(hon.get("term") or 0.0),
        "give_up": progress.get("give_up"), "exit_code": exit_code, "truncated": truncated, "stop": "timeout" if truncated else "agent_done",
        "turns": len(session.turns), "evidence_dir": str(out_dir), "progress": {k: v for k, v in progress.items() if k != "series"}}}
    if evaluation:  # slime's eval path reads only the reward (``_eval_result``, generate.py:466-491)
        sample.remove_sample = True
        sample.loss_mask = [0] * sample.response_length
    return [sample]
