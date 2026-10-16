"""Publish accepted task cards to agentenv (M17; study 04 §3.1, §4 rec 5).

The ``TaskCard`` stays the truth in Crucible. Publishing writes a read-only mirror so ``agent-env eval run --k``, the
``Runner`` and the explorer's run groups work on it:

- one agentenv ``Task`` per accepted card: steps ``crucible_world`` (the card's World), ``deploy_agent``,
  ``prompt_agent`` (the card's prompt) and ``crucible_verify`` (Crucible's single reward producer). Measured stats stay
  on the card, not the ``Task``;
- one ``crucible_ref`` artifact (``kind="task_card"``) that names the card by content hash and records the card's refs in
  ``derived_from``, so the explorer's version list shows which card produced which Task version;
- one ``Eval`` per split (``<family>.<split>``), each pinning exact ``(task_id, task_version)`` pairs. A card may sit in
  one split only: a held-out probe that also trains is contamination, so it is refused here.

Refused: a card whose status is not ``accepted``; a ``world_ref`` that is not a loadable ``module:Class`` spec;
a split that is empty, names an unknown card, mixes families, or repeats a card; a card in two splits.
agentenv is imported lazily (only ``crucible.interop`` and the studio may import it at module level), so this module
imports without it and fails with a clear message when it is absent.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from crucible.factory.cards import TaskCard, check_id

_SPEC = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
ENV_ID = "world"


class PublishError(ValueError):
    """The card or split layout cannot be published."""


@dataclass(frozen=True)
class PublishedTask:
    card_id: str
    family: str
    task_id: str
    task_version: int
    artifact_id: str
    artifact_version: int
    card_sha256: str


@dataclass(frozen=True)
class PublishedEval:
    split: str
    eval_id: str
    eval_version: int
    pins: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class FamilyPublication:
    tasks: dict[str, PublishedTask]
    evals: dict[str, PublishedEval]


def _agentenv() -> None:
    try:
        import agent_env  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise PublishError("publishing needs agentenv (pip install the agentenv extra)") from e


def publish_task(card: TaskCard, *, world_spec: str | None = None, task_kwargs: dict[str, Any] | None = None,
                 a2a_agent_id: str | None = None, a2a_agent_version: int | None = None,
                 store_key: str | None = None) -> PublishedTask:
    """Write the ``Task`` and the ``crucible_ref`` artifact for one accepted card.

    ``world_spec`` is the ``module:Class`` the ``crucible_world`` step loads (default: ``card.world_ref``, which must
    then be such a spec). ``store_key`` names where the card lives in Crucible's store (default
    ``task_card:<id>@sha256:<hash>``)."""
    _agentenv()
    if card.status != "accepted":
        raise PublishError(f"card {card.id!r} is {card.status!r}: only an accepted card is published")
    spec = world_spec or card.world_ref
    if not _SPEC.match(spec):
        raise PublishError(f"card {card.id!r}: {spec!r} is not a loadable 'module:Class' World spec (pass world_spec=)")
    from agent_env.task.task import Task
    from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
    from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

    from crucible.interop.agentenv.plugin.artifacts import CrucibleRefArtifact
    from crucible.interop.agentenv.plugin.steps import CrucibleVerifyStep, CrucibleWorldStep

    steps = [
        CrucibleWorldStep("world", None, env_id=ENV_ID, task=spec, task_kwargs=task_kwargs),
        DeployAgentTaskStep("deploy_agent", None, env_ids=[ENV_ID], env_step_id=None, a2a_agent_id=a2a_agent_id,
                            a2a_agent_version=a2a_agent_version, depends_on=["world"]),
        PromptAgentTaskStep("prompt_agent", None, prompt=card.prompt, depends_on=["deploy_agent"]),
        CrucibleVerifyStep("verify", None, env_id=ENV_ID, depends_on=["prompt_agent"]),
    ]
    digest = card.content_hash()
    task = Task.put(id=card.id, steps=steps)
    artifact = CrucibleRefArtifact.put(
        id=f"{card.id}--card", store_key=store_key or f"task_card:{card.id}@{digest}", kind="task_card",
        derived_from=[card.world_ref, card.checkset_ref, card.oracle_ref, f"task:{card.id}@{task.version}"])
    return PublishedTask(card.id, card.family, task.id, int(task.version), artifact.id, int(artifact.version), digest)


def _split_name(split: str) -> str:
    split = check_id(split)
    if "/" in split or "." in split:
        raise PublishError(f"split name {split!r} must not contain '/' or '.'")
    return split


def publish_eval(family: str, split: str, tasks: Sequence[PublishedTask]) -> PublishedEval:
    """One ``Eval`` version for ``split``, pinning exactly the given task versions (each is checked to exist)."""
    _agentenv()
    family, split = check_id(family), _split_name(split)
    if not tasks:
        raise PublishError(f"split {split!r} is empty: an Eval needs at least one task")
    seen: set[str] = set()
    for t in tasks:
        if t.family != family:
            raise PublishError(f"split {split!r} of family {family!r} holds {t.card_id!r} of family {t.family!r}")
        if t.task_id in seen:
            raise PublishError(f"split {split!r} lists {t.task_id!r} twice")
        seen.add(t.task_id)
    from agent_env.eval.eval import Eval, EvalTask
    from agent_env.task.task import Task

    for t in tasks:
        Task.get(t.task_id, t.task_version)  # a pin to a version that does not exist raises NotFoundError
    pins = tuple((t.task_id, t.task_version) for t in tasks)
    ev = Eval.put(id=f"{family}.{split}", tasks=[EvalTask(task_id=i, task_version=v) for i, v in pins])
    return PublishedEval(split, ev.id, int(ev.version), pins)


def publish_family(cards: Sequence[TaskCard], splits: Mapping[str, Sequence[str]], **kwargs: Any) -> FamilyPublication:
    """Publish ``cards`` and one pinned Eval per split. ``splits`` maps a split name (``train``, ``heldout``) to card ids.
    Every layout rule is checked before anything is written. Extra keyword arguments go to ``publish_task``."""
    by_id = {c.id: c for c in cards}
    if len(by_id) != len(cards):
        raise PublishError("two cards share an id")
    families = {c.family for c in cards}
    if len(families) != 1:
        raise PublishError(f"a publication covers one family, got {sorted(families)}")
    (family,) = families
    owner: dict[str, str] = {}
    for split, ids in splits.items():
        _split_name(split)
        if not ids:
            raise PublishError(f"split {split!r} is empty")
        if len(set(ids)) != len(ids):
            raise PublishError(f"split {split!r} lists a card twice")
        for cid in ids:
            if cid not in by_id:
                raise PublishError(f"split {split!r} names {cid!r}, which is not among the cards")
            if cid in owner and owner[cid] != split:
                raise PublishError(f"card {cid!r} is in both {owner[cid]!r} and {split!r}: splits must not share a task")
            owner[cid] = split
    for c in cards:
        if c.status != "accepted":
            raise PublishError(f"card {c.id!r} is {c.status!r}: only an accepted card is published")
    published = {c.id: publish_task(c, **kwargs) for c in cards}
    evals = {split: publish_eval(family, split, [published[i] for i in ids]) for split, ids in splits.items()}
    return FamilyPublication(published, evals)
