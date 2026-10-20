"""``CrucibleAgentLoop``: a verl agent loop that runs one Crucible World episode per verl session (M12 audit G2 pick 8,
H4 V19-V21). Imports verl; reached through the ``verl.plugins`` entry point (``verl_plugin.py``) or the
``agent_loop_config_path`` YAML (``verl_plugin.AGENT_LOOP_YAML``). Upstream: verl 5eb4826, paths relative to
``RL-Environments/06-frameworks-and-env-code/infra-volcengine-verl``.

Per session (verl v1 runs a prompt's ``rollout.n`` sessions as tasks of one worker, ``v1/agent_loop_tq.py:117-130``, and
passes ``uid`` and ``session_id`` to ``run``):

- the engine call is verl's ``LLMServerClient.generate(request_id, prompt_ids=…, sampling_params=…)``
  (``verl/workers/rollout/llm_server.py:99-164``) → ``TokenOutput`` (``verl/workers/rollout/replica.py:34-46``): token ids,
  ``log_probs`` (requested with ``logprobs=True``, ``vllm_async_server.py:629,743``), the policy version as
  ``extra_fields["global_steps"]`` (``vllm_async_server.py:713``; None before the first sync → 0), and ``stop_reason``
  (``aborted`` → ``abort``; vLLM reports stop and length both as ``completed``, ``:750-756``, so a completion that filled
  ``max_tokens`` is ``length``);
- the episode is Crucible's ``rollout(rid, prompt, seed=session_id, gen, temperature=…)`` (``train.world_env.world_rollout``),
  built once per process by the factory named in the loop config (``rollout: "pkg.mod:factory"``, or
  ``CRUCIBLE_VERL_ROLLOUT``), called as ``factory(agent_loop)``; the task prompt is ``extra_info["crucible_prompt"]``, else
  the last user message of ``raw_prompt``;
- with ``turn_weight`` > 0 the sessions of one ``uid`` meet once (an in-process rendezvous; ``group_timeout_s``, default
  600 s, after which a session falls back to outcome-only credit; the group size is ``rollout.n``: a per-prompt
  ``__rollout_n__`` is popped before ``run`` sees it, ``agent_loop_tq.py:120``) and M6's group rule (``train.credit.turn_credit_map``)
  gives each call its credit, which ``CrucibleAgentLoopOutput.as_dict`` writes into ``rm_scores`` (``verl_bridge``). A
  group none of whose live rollouts recorded per-turn progress trains outcome-only and says so: every output's
  ``extra_fields["crucible_credit_fallback"]`` is ``"no_turn_credit"`` (``"timeout"`` after the rendezvous timed out;
  ``"legacy_worker"`` on ``use_v1: false``, which trains the scalar reward only), with one warning per process (M6 R3; a
  raise here would mask every sample of the step instead);
- the record becomes one ``AgentLoopOutput`` per (actor, segment) sequence (``bridges.common.flatten``); a masked or
  token-incomplete rollout becomes one placeholder row with no trainable token (``response_mask=[0]``): verl keeps the
  session (no failure status, so the uid group is not evicted, ``agent_loop_tq.py:131-150``) and ``crucible_delta`` leaves
  it out of the group mean. The legacy worker (no ``session_id``) gets the final sequence only.

Truncated sampling is refused: with ``rollout.top_p < 1`` or ``top_k > 0`` verl returns no candidate sets, and Crucible's
trainer-side checks (``Turn.candidates_valid``) need them.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

from crucible.core import env as knobs
from crucible.core.schema.rollout import RolloutRecord, full_prompts
from crucible.train.bridges.adapters import verl_agent_loop_output
from crucible.train.bridges.common import flatten, untrainable
from crucible.train.bridges.verl_bridge import token_rewards
from crucible.train.engine import GenResult

logger = logging.getLogger(__name__)


class CrucibleAgentLoopOutput(AgentLoopOutput):
    """``AgentLoopOutput`` whose ``as_dict`` (``agent_loop.py:116-179``, called by ``agent_loop_tq.py:207``) writes the
    per-token ``rm_scores`` encoding (row sum = ``reward_score``) instead of the scalar at the last position (``:143-148``)."""

    crucible_token_rewards: list[float] | None = None

    def as_dict(self) -> dict[str, Any]:
        out = super().as_dict()
        tr = out.pop("crucible_token_rewards", None)
        if tr is not None:
            if len(tr) != out["responses"].size(0):
                raise ValueError(f"{len(tr)} token rewards for a response of {out['responses'].size(0)} tokens")
            out["rm_scores"] = torch.tensor(tr, dtype=torch.float32)
        return out


class _Group:
    def __init__(self, n: int):
        self.n, self.left = n, n
        self.records: dict[int, list[RolloutRecord]] = {}
        self.done = asyncio.Event()
        self.credit: dict[tuple[str, str], float] = {}
        self.reason: str | None = None  # why the group trains outcome-only (M6 R3), None when it carries turn credit


_GROUPS: dict[str, _Group] = {}
_WARNED: set[str] = set()


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(msg, *args)
_ROLLOUTS: dict[Any, Callable] = {}


async def group_credit(uid: str, session: int, n: int, records: list[RolloutRecord], *, eta: float, frozen_actors=(),
                       timeout_s: float = 600.0) -> tuple[dict[tuple[str, str], float] | None, str | None]:
    """Rendezvous of a uid's ``n`` sessions; returns ((rollout id, call id) -> credit, None), or (None, why) when the group
    trains outcome-only: ``"timeout"`` (not every session arrived) or ``"no_turn_credit"`` (no live rollout of the group
    recorded per-turn progress, M6 R3)."""
    from crucible.train.credit import has_turn_credit, turn_credit_map

    g = _GROUPS.setdefault(uid, _Group(n))
    g.records[session] = records
    if len(g.records) >= g.n and not g.done.is_set():
        recs = [r for rs in g.records.values() for r in rs]
        live = [r for r in recs if not r.mask_sample and r.reward is not None]
        if live and not has_turn_credit(live, frozen_actors):
            g.reason = "no_turn_credit"
            _warn_once("no_turn_credit", "crucible_agent: turn_weight=%s but no rollout of uid %s recorded per-turn progress (Turn.advantage is "
                       "None): outcome-only credit, marked crucible_credit_fallback='no_turn_credit'. Train on tasks with subgoals, or set "
                       "turn_weight=0.", eta, uid)
        elif recs:
            g.credit = turn_credit_map(recs, [f"{uid}#{r.actor or ''}" for r in recs], eta=eta, frozen_actors=frozen_actors)
        g.done.set()
    try:
        await asyncio.wait_for(g.done.wait(), timeout_s)
        return (None, g.reason) if g.reason else (g.credit, None)
    except TimeoutError:
        logger.warning("crucible_agent: uid %s: %d/%d sessions after %.0f s; outcome-only credit", uid, len(g.records), g.n, timeout_s)
        return None, "timeout"
    finally:
        g.left -= 1
        if g.left <= 0 and _GROUPS.get(uid) is g:
            _GROUPS.pop(uid, None)


def _resolve(spec: Any) -> Callable:
    if callable(spec):
        return spec
    mod, _, attr = str(spec).partition(":")
    if not attr:
        raise ValueError(f"crucible_agent rollout {spec!r}: expected 'package.module:factory'")
    return getattr(importlib.import_module(mod), attr)


class CrucibleAgentLoop(AgentLoopBase):
    def __init__(self, *args: Any, rollout: Any = None, turn_weight: float | None = None, frozen_actors: Any = (),
                 group_timeout_s: float | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.rollout_spec = rollout or knobs.get("CRUCIBLE_VERL_ROLLOUT")
        if not self.rollout_spec:
            raise ValueError("crucible_agent needs a rollout factory: agent loop config 'rollout: pkg.mod:factory' or CRUCIBLE_VERL_ROLLOUT")
        self.turn_weight = float(turn_weight if turn_weight is not None else knobs.get("CRUCIBLE_VERL_TURN_WEIGHT"))
        self.frozen_actors = tuple(frozen_actors or ())
        self.group_timeout_s = float(group_timeout_s or 600.0)
        rc = self.rollout_config
        top_p, top_k = float(_get(rc, "top_p", 1.0) or 1.0), _get(rc, "top_k", -1)
        if top_p < 1.0 or (top_k is not None and int(top_k) > 0):
            raise ValueError(f"crucible_agent: rollout.top_p={top_p}, top_k={top_k}: truncated sampling needs per-token candidate sets, "
                             "which verl's TokenOutput does not return; train with top_p=1, top_k=-1")

    def _rollout(self) -> Callable:
        key = self.rollout_spec
        if key not in _ROLLOUTS:
            _ROLLOUTS[key] = _resolve(key)(self)
        return _ROLLOUTS[key]

    @staticmethod
    def task_prompt(kwargs: dict[str, Any]) -> str:
        extra = kwargs.get("extra_info") or {}
        if isinstance(extra, dict) and extra.get("crucible_prompt") is not None:
            return str(extra["crucible_prompt"])
        if kwargs.get("crucible_prompt") is not None:
            return str(kwargs["crucible_prompt"])
        msgs = list(kwargs.get("raw_prompt") or [])
        users = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
        return str((users or msgs or [{"content": ""}])[-1].get("content", ""))

    def gen(self, rid: str, sampling_params: dict[str, Any]) -> Callable:
        async def gen(ids: list[int], max_tokens: int, temperature: float) -> GenResult:
            sp = {**sampling_params, "temperature": float(temperature), "max_tokens": int(max_tokens), "logprobs": True}
            out = await self.server_manager.generate(request_id=rid, prompt_ids=list(ids), sampling_params=sp)
            toks = [int(t) for t in out.token_ids]
            lps = [float(x) for x in out.log_probs] if out.log_probs is not None else []
            ver = (out.extra_fields or {}).get("global_steps")
            finish = "abort" if out.stop_reason == "aborted" else ("length" if len(toks) >= int(max_tokens) else "stop")
            return GenResult(toks, lps, int(ver) if ver is not None else 0, finish, None, len(ids))
        return gen

    def _pad(self) -> int:
        for k in ("pad_token_id", "eos_token_id"):
            v = getattr(self.tokenizer, k, None)
            if isinstance(v, int):
                return v
        return 0

    def placeholder(self, records: list[RolloutRecord], why: str) -> CrucibleAgentLoopOutput:
        prompt: list[int] = []
        for r in records:
            if r.turns:
                prompt = full_prompts(r).get(r.turns[0].call_id) or []
                break
        pad = self._pad()
        return CrucibleAgentLoopOutput(prompt_ids=prompt or [pad], response_ids=[pad], response_mask=[0], response_logprobs=[0.0],
                                       reward_score=0.0, num_turns=0, metrics={}, crucible_token_rewards=[0.0],
                                       extra_fields={"min_global_steps": None, "max_global_steps": None, "reward_extra_info": {},
                                                     "turn_scores": [], "tool_rewards": [], "crucible_masked": why})

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput | list[AgentLoopOutput]:
        uid = str(kwargs.get("uid", uuid4().hex))
        v1 = "session_id" in kwargs
        session = int(kwargs.get("session_id", 0))
        rid = f"verl:{uid}:{session}:{uuid4().hex[:8]}"
        why = "masked"
        try:
            got = await self._rollout()(rid, self.task_prompt(kwargs), session, self.gen(rid, dict(sampling_params)),
                                        temperature=float(sampling_params.get("temperature", 1.0)))
        except Exception as e:  # one session's fault masks that session only (its siblings still train)
            logger.exception("crucible_agent %s: rollout failed", rid)
            got, why = None, f"rollout error: {type(e).__name__}: {e}"
        records = [r for r in (got if isinstance(got, list) else [got]) if r is not None]
        credit: dict[tuple[str, str], float] | None = {}
        fallback: str | None = None
        if self.turn_weight and v1:
            credit, fallback = await group_credit(uid, session, int(_get(self.rollout_config, "n", 1) or 1), records, eta=self.turn_weight,
                                                  frozen_actors=self.frozen_actors, timeout_s=self.group_timeout_s)
        elif self.turn_weight:  # the legacy worker pads from the scalar reward: no per-token channel, so no turn credit (M6 R3)
            fallback = "legacy_worker"
            _warn_once("legacy_worker", "crucible_agent: turn_weight=%s on verl's legacy agent-loop worker (trainer.use_v1: false): it trains the "
                       "scalar reward only, so turn credit is dropped; marked crucible_credit_fallback='legacy_worker'", self.turn_weight)
        L = int(_get(self.rollout_config, "response_length", 0) or 0) or None
        outs: list[AgentLoopOutput] = []
        for r in records:
            reason = untrainable(r, self.frozen_actors)
            if reason is not None:
                why = reason
                continue
            per_call = {cid: c for (rr, cid), c in (credit or {}).items() if rr == r.rollout_id}
            for s in flatten(r, frozen_actors=self.frozen_actors):
                d = verl_agent_loop_output(s, response_length=L)
                if fallback is not None:
                    d["extra_fields"]["crucible_credit_fallback"] = fallback
                outs.append(CrucibleAgentLoopOutput(**d, crucible_token_rewards=token_rewards(s, turn_credit=per_call, length=L)))
        if not v1 and outs:
            plen = int(_get(self.rollout_config, "prompt_length", 0) or 0)
            if plen and len(outs[-1].prompt_ids) > plen:  # the legacy worker pads prompts to prompt_length (agent_loop.py:774-779)
                return self.placeholder(records, f"prompt of {len(outs[-1].prompt_ids)} tokens > rollout.prompt_length {plen}")
            return outs[-1]
        if not outs:
            outs = [self.placeholder(records, why)]
        return outs if v1 else outs[-1]


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:  # noqa: BLE001, S110 - a structured config without the key: fall back to the attribute
            pass
    return getattr(cfg, key, default)
