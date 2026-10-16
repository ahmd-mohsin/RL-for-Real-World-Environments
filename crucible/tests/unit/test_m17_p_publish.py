"""M17 P: publishing accepted cards to agentenv (Task + crucible_ref artifact + one pinned Eval per split), against
agentenv's own models on a temporary SQLite document store."""

from __future__ import annotations

import pytest

pytest.importorskip("agent_env")

from tests.unit.m17_p_fixtures import PROMPT, card


@pytest.fixture(autouse=True)
def agentenv_store(tmp_path):
    from agent_env.config import runtime
    from agent_env.store.document_store.sqlite_document_store import LocalSqliteDocumentStore

    runtime.reset_config()
    runtime.configure(document_store=LocalSqliteDocumentStore(str(tmp_path / "ae.db")))
    yield
    runtime.reset_config()


def test_accepted_card_becomes_a_task_with_the_four_steps():
    from agent_env.task.task import Task

    from crucible.factory.publish import publish_task

    p = publish_task(card("t1"))
    task = Task.get(p.task_id, p.task_version)
    assert [s.type for s in task.steps] == ["crucible_world", "deploy_agent", "prompt_agent", "crucible_verify"]
    world, deploy, prompt, verify = task.steps
    assert world.task == "crucible_pack_toy.runtime:HelpdeskTask" and prompt.prompt == PROMPT
    assert deploy.env_ids == [world.env_id] == [verify.env_id]
    assert [d.task_step_id for d in prompt.depends_on] == ["deploy_agent"]
    assert (p.task_id, p.task_version, p.family) == ("t1", 1, "fam")
    assert task.preflight() is not None  # the Task's own preflight runs (store lookups need the agent registry)


def test_the_crucible_ref_artifact_names_the_card_by_hash():
    from crucible.factory.publish import publish_task
    from crucible.interop.agentenv.plugin.artifacts import CrucibleRefArtifact

    c = card("t1")
    p = publish_task(c)
    a = CrucibleRefArtifact.get(p.artifact_id, p.artifact_version)
    assert a.kind == "task_card" and a.type == "crucible_ref" and a.store_key == f"task_card:t1@{c.content_hash()}" == f"task_card:t1@{p.card_sha256}"
    assert a.derived_from[:3] == ["crucible_pack_toy.runtime:HelpdeskTask", "checks/t1", "oracle/t1"] and a.derived_from[3] == "task:t1@1"
    assert publish_task(c, store_key="my/key").artifact_version == 2
    assert CrucibleRefArtifact.get(p.artifact_id).store_key == "my/key"  # latest version


@pytest.mark.parametrize("status", ["draft", "rejected", "quarantined", "retired"])
def test_non_accepted_cards_are_refused(status):
    from agent_env.store.base import NotFoundError
    from agent_env.task.task import Task

    from crucible.factory.publish import PublishError, publish_task

    with pytest.raises(PublishError, match="only an accepted card"):
        publish_task(card("t1", status=status))
    with pytest.raises(NotFoundError):  # nothing was written
        Task.get("t1")


def test_world_ref_must_be_a_loadable_spec():
    from crucible.factory.publish import PublishError, publish_task

    with pytest.raises(PublishError, match="module:Class"):
        publish_task(card("t1", world_ref="worlds/abc123"))
    with pytest.raises(PublishError, match="module:Class"):
        publish_task(card("t1"), world_spec="os.path; rm -rf")
    assert publish_task(card("t1", world_ref="worlds/abc123"), world_spec="pkg.mod:Cls").task_version == 1


def test_republishing_makes_a_new_version_not_an_edit():
    from agent_env.task.task import Task

    from crucible.factory.publish import publish_task

    v1 = publish_task(card("t1"))
    v2 = publish_task(card("t1", prompt=PROMPT))
    assert (v1.task_version, v2.task_version) == (1, 2)
    assert Task.get("t1", 1).steps[2].prompt == PROMPT


def test_family_publishes_one_pinned_eval_per_split():
    from agent_env.eval.eval import Eval

    from crucible.factory.publish import publish_family

    cards = [card(f"t{n}") for n in range(4)]
    pub = publish_family(cards, {"train": ["t0", "t1", "t2"], "heldout": ["t3"]})
    assert set(pub.evals) == {"train", "heldout"} and set(pub.tasks) == {"t0", "t1", "t2", "t3"}
    train = Eval.get("fam.train", pub.evals["train"].eval_version)
    held = Eval.get("fam.heldout")
    assert [(t.task_id, t.task_version) for t in train.tasks] == [("t0", 1), ("t1", 1), ("t2", 1)]
    assert [(t.task_id, t.task_version) for t in held.tasks] == [("t3", 1)]
    assert all(t.task_version is not None for t in train.tasks)  # pinned, never "latest"
    # a re-publish after the card changed pins the new version in a new Eval version; the old Eval version is unchanged
    pub2 = publish_family(cards, {"train": ["t0", "t1", "t2"], "heldout": ["t3"]})
    assert pub2.evals["train"].eval_version == 2
    assert [(t.task_id, t.task_version) for t in Eval.get("fam.train", 1).tasks] == [("t0", 1), ("t1", 1), ("t2", 1)]
    assert [(t.task_id, t.task_version) for t in Eval.get("fam.train", 2).tasks] == [("t0", 2), ("t1", 2), ("t2", 2)]


def test_family_layout_rules_checked_before_anything_is_written():
    from agent_env.store.base import NotFoundError
    from agent_env.task.task import Task

    from crucible.factory.publish import PublishError, publish_family

    cards = [card("t0"), card("t1")]
    for splits, why in [({"train": ["t0"], "heldout": ["t0", "t1"]}, "must not share"),
                        ({"train": []}, "empty"),
                        ({"train": ["zzz"]}, "not among the cards"),
                        ({"train": ["t0", "t0"]}, "twice"),
                        ({"bad/name": ["t0"]}, "must not contain")]:
        with pytest.raises((PublishError, ValueError), match=why):
            publish_family(cards, splits)
    with pytest.raises(PublishError, match="only an accepted card"):
        publish_family([card("t0"), card("t1", status="draft")], {"train": ["t0"]})
    with pytest.raises(PublishError, match="one family"):
        publish_family([card("t0"), card("t1", family="other")], {"train": ["t0"]})
    with pytest.raises(PublishError, match="share an id"):
        publish_family([card("t0"), card("t0")], {"train": ["t0"]})
    with pytest.raises(NotFoundError):
        Task.get("t0")  # every refusal happened before the first write


def test_eval_pins_must_exist_and_belong_to_the_family():
    from dataclasses import replace

    from agent_env.store.base import NotFoundError

    from crucible.factory.publish import PublishError, publish_eval, publish_task

    p = publish_task(card("t0"))
    with pytest.raises(NotFoundError):
        publish_eval("fam", "train", [replace(p, task_version=9)])
    with pytest.raises(PublishError, match="of family"):
        publish_eval("other", "train", [p])
    with pytest.raises(PublishError, match="empty"):
        publish_eval("fam", "train", [])
    with pytest.raises(PublishError, match="twice"):
        publish_eval("fam", "train", [p, p])
    with pytest.raises(PublishError, match="must not contain"):
        publish_eval("fam", "a.b", [p])
