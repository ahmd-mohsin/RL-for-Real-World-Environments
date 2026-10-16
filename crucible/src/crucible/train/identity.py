"""What a training run is, for its checkpoints (M0 audit 2026-10-14 R-1 step 4).

``state.restore`` installed a checkpoint into "a freshly constructed runtime (same configuration)" without checking the
configuration, so a resume with another group size, loss correction, temperature, prompt set or reward function
continued from the saved weights and one training curve mixed two experiments. The identity projects every field of
``RuntimeConfig``, ``SchedulerConfig`` and the trainer's (and critic's) config, the trainer class, the vocabulary id, the
model's parameter names and shapes (a digest, not the values), each prompt source (count and digest) and the rollout,
reward and environment functions (by qualified name only: listed in ``opaque``), minus the operational settings below,
which change how a run executes, not what it trains. Every checkpoint carries it; ``check`` on resume raises
``TrainResumeMismatch`` naming up to 20 changed settings unless each matches a glob in ``RuntimeConfig.allow_change``.
A checkpoint from before this change has no identity: it resumes once, recorded as ``adopted``.

PATTERN: NeMo-RL ``rollout_checkpoint.py`` (fingerprint of the full config minus a denylist, refusal naming changed
paths) and slime ``training_recovery.py`` (every argument frozen unless declared mutable), Apache-2.0; the identity
machinery is ``core.fingerprint``, shared with the run lock (R-1 step 1).
"""

from __future__ import annotations

import hashlib
from typing import Any

from crucible.core.fingerprint import SCHEME, describe, diff, excluded, fingerprint, project
from crucible.core.prov import utc_now
from crucible.runtime.lock import jsonable

# Settings a resume may change freely: when the run stops (steps, wall clock), polling and worker counts, the checkpoint
# cadence and place, sinks and kept records, the breaker's thresholds, where state and KV live while the other side
# holds the GPU, call-reissue pacing, the heartbeat, data parallelism and micro-batching (both exact), prompt-prefix
# sharing (the same loss, encoded once), and the allowance itself. Everything else counts.
OPERATIONAL = ("runtime.steps", "runtime.poll_s", "runtime.max_wall_s", "runtime.env_workers", "runtime.checkpoint_every",
               "runtime.checkpoint_dir", "runtime.keep_records", "runtime.sinks", "runtime.record_rollouts", "runtime.per_source_metrics",
               "runtime.max_masked_share", "runtime.max_harness_share", "runtime.breaker_window", "runtime.breaker_min",
               "runtime.train_state_on_rollout", "runtime.kv_on_train", "runtime.max_call_reissues", "runtime.reissue_backoff_s",
               "runtime.heartbeat_s", "runtime.allow_change", "trainer.dp", "trainer.token_budget", "trainer.prefix_sharing")
FUNCTIONS = ("harness", "env_factory", "rollout_fn", "reward_fn", "aux_fn", "group_fn", "audit_fn", "on_alarm", "teacher_pool")


class TrainResumeMismatch(ValueError):
    """A resume whose training settings differ from the checkpoint's."""


def _digest(value: Any) -> str:
    from crucible.core.canonical import dumps

    return "sha256:" + hashlib.sha256(dumps(jsonable(value)).encode()).hexdigest()


def _name(fn: Any) -> Any:
    """A function or component by qualified name (its code is not hashed); a per-source map by source."""
    if fn is None or isinstance(fn, (str, int, float, bool)):
        return fn
    if isinstance(fn, dict):
        return {str(k): _name(v) for k, v in fn.items()}
    target = fn if callable(fn) and hasattr(fn, "__qualname__") else type(fn)
    return f"{getattr(target, '__module__', '?')}:{getattr(target, '__qualname__', type(fn).__name__)}"


def sources_doc(sources: dict[str, list[Any]]) -> dict[str, Any]:
    """Each prompt source as its prompt count and the digest of its prompts, in order."""
    return {str(s): {"prompts": len(ps), "digest": _digest(list(ps))} for s, ps in sources.items()}


def identity_doc(rt: Any) -> dict[str, Any]:
    tr = rt.trainer
    model = getattr(tr, "model", None)
    shapes = [[n, list(p.shape)] for n, p in model.named_parameters()] if hasattr(model, "named_parameters") else None
    doc: dict[str, Any] = {
        "runtime": jsonable(rt.cfg), "scheduler": jsonable(rt.sched_cfg), "trainer": jsonable(getattr(tr, "cfg", None)),
        "model": {"trainer": _name(getattr(tr, "trainer", tr)), "vocab_id": getattr(tr, "vocab_id", None),
                  "parameters": _digest(shapes) if shapes is not None else None},
        "sources": rt._source_identity, "functions": {k: _name(getattr(rt, k, None)) for k in FUNCTIONS},
    }
    if getattr(tr, "critic", None) is not None:
        doc["critic"] = jsonable(getattr(tr, "critic_cfg", None))
    return doc


def identity(rt: Any) -> dict[str, Any]:
    """The runtime's training identity: scheme, fingerprint, projected settings, exclusions, the functions that count by
    name only, and the history of allowed changes."""
    settings = project(identity_doc(rt), OPERATIONAL)
    opaque = [k for k in FUNCTIONS if getattr(rt, k, None) is not None and not isinstance(getattr(rt, k), str)]
    return {"scheme": SCHEME, "fingerprint": fingerprint(settings), "settings": settings, "excluded": list(OPERATIONAL),
            "opaque": opaque, "changes": list(getattr(rt, "_identity_changes", []))}


def check(rt: Any, blob: dict) -> dict[str, Any]:
    """Compare a checkpoint's identity with the runtime about to resume it, before anything is installed. Returns what
    happened (``same`` / ``changed`` / ``adopted``); raises TrainResumeMismatch for a change ``allow_change`` does not
    name. The runtime keeps the change history, so its next checkpoints carry it."""
    old = blob.get("identity")
    now = identity(rt)
    step = int(blob.get("step", 0))
    if not isinstance(old, dict):  # a checkpoint from before R-1 step 4: nothing to compare, recorded once
        event = {"status": "adopted", "step": step, "at": utc_now(), "after": now["fingerprint"]}
        rt._identity_changes = [event]
        return event
    history = list(old.get("changes") or [])
    # settings the current operational list excludes do not count, even if an older checkpoint recorded them
    before = {k: v for k, v in (old.get("settings") or {}).items() if not excluded(tuple(k.split(".")), OPERATIONAL)}
    changes = diff(before, now["settings"])
    if not changes:
        rt._identity_changes = history
        return {"status": "same", "step": step}
    allow = tuple(getattr(rt.cfg, "allow_change", ()) or ())
    refused = {k: c for k, c in changes.items() if not excluded(tuple(k.split(".")), allow)}
    if refused:
        raise TrainResumeMismatch(
            f"checkpoint {blob.get('_path', '?')} (step {step}) was trained with other settings; refusing to resume it. Changed:\n"
            f"{describe(refused)}\nResume with the checkpoint's settings, start a new checkpoint_dir, or set "
            f"RuntimeConfig(allow_change=[...]) naming these settings (globs such as 'trainer.lr') to record the change.")
    event = {"status": "changed", "step": step, "at": utc_now(), "changed": sorted(changes), "before": old.get("fingerprint"), "after": now["fingerprint"]}
    rt._identity_changes = [*history, event]  # only once accepted: a refused resume leaves the runtime as it was
    return event
