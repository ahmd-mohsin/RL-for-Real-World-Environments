"""M10 lab audit R-4, slice 2: the run identity v2 in the runtime: a resolved document per run, fail-closed, compared leaf by
leaf on a resume (dotpaths named), an operational denylist, accept-drift epochs, code versions that open an epoch and never
refuse, and harness ids that carry a digest of their configuration.

Nothing here runs an episode: ``Quick`` is a real ``NativeHarness`` (so it inherits the declaration and the id) whose ``run``
returns a reward at once, and the journal-level tests open ``RunJournal`` with ``RunSpec.resolved_doc`` directly.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crucible.actors.policy import EpisodeLimits
from crucible.core.schema.base import CrucibleModel
from crucible.core.schema.run import Failure
from crucible.runtime import identity as ident
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.harness.imported import ImportedHarness
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.harness.rlm import RLMConfig, RLMHarness
from crucible.runtime.harness.workspace import WorkspaceHarness
from crucible.runtime.identity import IdentityDrift, Undeclared, describe
from crucible.runtime.infra import LimitPolicy
from crucible.runtime.journal import RunBusy, RunJournal
from crucible.runtime.run import Runner, RunSpec


class Doc(CrucibleModel):
    a: int = 1
    b: str | None = None


class Plain:
    """Stateless: its behaviour is its code, so it is described as its type."""


class Quick(NativeHarness):
    """A NativeHarness that runs no episode: it inherits ``id`` and ``identity_config``, so the identity is the real one."""

    async def run(self, attempt: Attempt) -> HarnessResult:
        return HarnessResult(reward=1.0)


class Credits:
    """Its first call fails with ``crucible:credits_exhausted`` (the sample stays owed: resumable), later calls score."""

    id = "credits@1"

    def __init__(self) -> None:
        self.calls = 0

    def identity_config(self) -> dict:
        return {}

    async def run(self, attempt: Attempt) -> HarnessResult:
        self.calls += 1
        if self.calls == 1:
            return HarnessResult(failure=Failure(kind="crucible:credits_exhausted", stage="agent", reason="x"))
        return HarnessResult(reward=1.0)


def pol(**cfg) -> PolicyRef:
    return PolicyRef(id="llm:m", make=lambda a: {}, config={"model": "m", **cfg})


def spec(**kw) -> RunSpec:
    base = {"run_id": "r", "tasks": [SimpleNamespace(id="t")], "policies": [pol(temperature=0.7)], "harnesses": [Quick("native:quick")], "k": 1}
    return RunSpec(**{**base, **kw})


def open_journal(root: Path, s: RunSpec, **kw) -> RunJournal:
    """What ``Runner._open`` does, so a test can hold the journal (and close it) without running anything."""
    doc = s.resolved_doc()
    return RunJournal(root, s.run_id, ident.fingerprint(doc), s.identity_doc(), resolved=ident.normalize(doc), code=kw.pop("code", {}),
                      accept_drift=kw.pop("accept_drift", s.accept_drift), **kw)


def run_json(root: Path, run_id: str = "r") -> dict:
    return json.loads((root / run_id / "run.json").read_text())


def rows(root: Path, run_id: str = "r") -> list[dict]:
    p = root / run_id / "results.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def events(root: Path, run_id: str = "r") -> list[dict]:
    return [json.loads(x) for x in (root / run_id / "journal.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------- describing objects


def test_describe_inspects_plain_data_and_refuses_what_it_cannot(tmp_path):
    assert describe(SimpleNamespace(a=1, p=Path("x")), "w") == {"a": 1, "p": "x"}
    assert describe(EpisodeLimits(max_steps=7), "w")["max_steps"] == 7
    assert describe(Doc(a=3), "w") == {"a": 3}  # a pydantic model: its dump without the unset
    assert describe(Plain(), "w") == {"type": f"{Plain.__module__}.Plain"}  # stateless: its behaviour is its code
    with pytest.raises(Undeclared):
        describe(object(), "w")  # no __dict__ at all: nothing says it is stateless
    assert describe({"a": (1, 2), "s": {3, 1, 2}, "f": float("inf"), "n": None, "b": b"x"}, "w") == {
        "a": [1, 2], "s": [1, 2, 3], "f": "inf", "n": None, "b": describe(b"x", "w")}
    assert describe(b"x", "w").startswith("sha256:")

    class Stateful:
        def __init__(self) -> None:
            self.calls = 0

    with pytest.raises(Undeclared, match="tasks.t"):
        describe(Stateful(), "tasks.t")
    with pytest.raises(Undeclared) as e:
        describe({"f": lambda: 0}, "w")
    assert "w.f" in str(e.value) and "identity_config" in str(e.value)
    long = describe("x" * 2000, "w")
    assert long.startswith("sha256:") and len(long) == len("sha256:") + 16
    assert describe("x" * 1024, "w") == "x" * 1024
    big = describe(list(range(300)), "w")
    assert big["len"] == 300 and len(big["sha256"]) == 64
    assert describe(list(range(256)), "w") == list(range(256))


def test_a_declared_identity_config_is_used_and_must_be_a_mapping():
    class Declared:
        def __init__(self) -> None:
            self.counter = 5  # state the declaration does not mention: not part of the identity

        def identity_config(self) -> dict:
            return {"x": 1, "p": Path("a/b")}

    class Bad:
        def identity_config(self) -> list:
            return [1]

    assert describe(Declared(), "w") == {"x": 1, "p": "a/b"}
    with pytest.raises(Undeclared, match="must return a mapping"):
        describe(Bad(), "w")


def test_a_mock_and_a_delegating_wrapper_are_not_inspected_as_declared():
    """``identity_config`` is looked up on the type: a Mock (which answers every attribute) and a wrapper that delegates
    ``__getattr__`` (FlakyGradeTask's pattern) never report another object's document as their own."""
    with pytest.raises(Undeclared):
        describe(MagicMock(), "m")

    class Inner:
        def identity_config(self) -> dict:
            return {"inner": True}

    class Wrap:
        def __init__(self) -> None:
            self.inner = Inner()

        def __getattr__(self, k):
            return getattr(self.inner, k)

    with pytest.raises(Undeclared, match="w"):
        describe(Wrap(), "w")


def test_a_policy_ref_without_config_is_refused_with_its_dotpath():
    ref = PolicyRef(id="p", make=lambda a: {})
    with pytest.raises(Undeclared, match="policies.p"):
        ref.identity_config()
    assert describe(PolicyRef(id="p", make=lambda a: {}, config={"model": "m"}), "policies.p") == {"id": "p", "config": {"model": "m"}}


# ---------------------------------------------------------------- the comparison


def open_doc(root: Path, doc: dict, run_id: str = "r", **kw) -> RunJournal:
    """Open a run directory on a resolved document directly (what ``Runner._open`` does with ``RunSpec.resolved_doc``)."""
    return RunJournal(root, run_id, ident.fingerprint(doc), {"run_id": run_id}, resolved=ident.normalize(doc), code=kw.pop("code", {}), **kw)


def test_the_denylist_drops_operational_leaves_and_keeps_content():
    doc = {"concurrency": 3, "budget_usd": 1.0, "k": 2,
           "tasks": {"t": {"timeout": 5, "workdir": "/x", "image": "img@sha256:aa", "files": {"token": "abc", "root": "d", "dir": "e"}}},
           "policies": {"p": {"config": {"api_key": "k", "eos_token": "</s>", "temperature": 0.7, "base_url": "http://h"}, "estimate_usd": 0.1}},
           "harnesses": {"h@1": {"limits": {"timeout_s": 60, "max_steps": 30}, "scoring_budget_s": 5, "host": "local", "docker_host": "ssh://x",
                                 "patches": {"import:x": "d1"}}}}
    assert ident.compared(ident.normalize(doc)) == {
        "k": 2,
        "tasks": {"t": {"image": "img@sha256:aa", "files": {"token": "abc", "root": "d", "dir": "e"}}},  # a file named like a credential is still content
        # a credential is stored as a placeholder (never compared by value, never written); an endpoint is compared: it names what is measured
        "policies": {"p": {"config": {"api_key": "<secret>", "eos_token": "</s>", "temperature": 0.7, "base_url": "http://h"}}},
        "harnesses": {"h@1": {"limits": {"timeout_s": 60, "max_steps": 30}, "host": "local", "patches": {"import:x": "d1"}}},  # the episode's own timeout is policy-facing
    }


def test_an_id_never_decides_whether_its_configuration_is_compared():
    """The denylist is matched below the id: a task named ``cleanup_build_dir``, a dotted id, a policy ``llm:reset_password`` keep their
    leaves (M10 lab audit R-4 review)."""
    doc = {"tasks": {"cleanup_build_dir": {"files": {"tests": "d1"}, "command": "bash t.sh", "timeout": 5, "work_dir": "/x"}, "a.dir": {"image": "i"}},
           "policies": {"llm:reset_password": {"config": {"model": "m", "work_dir": "/y"}}, "scripted:fix_root": {"config": {"script": "s"}}},
           "harnesses": {"native:x_url@1": {"limits": {"max_steps": 3}}}}
    assert ident.compared(ident.normalize(doc)) == {
        "tasks": {"cleanup_build_dir": {"files": {"tests": "d1"}, "command": "bash t.sh"}, "a.dir": {"image": "i"}},  # timeout and work_dir are below the id: dropped
        "policies": {"llm:reset_password": {"config": {"model": "m"}}, "scripted:fix_root": {"config": {"script": "s"}}},
        "harnesses": {"native:x_url@1": {"limits": {"max_steps": 3}}}}


def test_editing_a_task_whose_id_ends_like_a_denied_path_is_refused(tmp_path):
    def doc(sh: str, model: str) -> dict:
        return {"tasks": {"cleanup_build_dir": {"files": {"tests": sh}}}, "policies": {"llm:reset_password": {"config": {"model": model}}}}

    open_doc(tmp_path, doc("a", "m")).close()
    with pytest.raises(IdentityDrift) as e:
        open_doc(tmp_path, doc("b", "n"))
    assert len(e.value.changes) == 2
    text = str(e.value)
    assert "tasks.cleanup_build_dir.files.tests: 'a' -> 'b'" in text and "policies.llm:reset_password.config.model: 'm' -> 'n'" in text


def test_credential_keys_are_never_stored_or_compared_and_credential_text_is_keyed():
    doc = {"tasks": {"t": {"verifier_env": {"JUDGE_API_KEY": "internal-key-1", "HF_TOKEN": "tok-9f3a", "MODE": "strict"}, "prompt": "log in with password=hunter2x now"}},
           "policies": {"p": {"config": {"api_key": "internal-vllm-key-1234", "eos_token": "</s>", "max_tokens": 5, "token": ""}}}}
    n = ident.normalize(doc)
    text = json.dumps(n)
    assert not any(v in text for v in ("internal-key-1", "tok-9f3a", "internal-vllm-key-1234", "hunter2x"))  # nothing credential-shaped is stored
    assert n["tasks"]["t"]["verifier_env"] == {"JUDGE_API_KEY": "<secret>", "HF_TOKEN": "<secret>", "MODE": "strict"}
    assert n["policies"]["p"]["config"] == {"api_key": "<secret>", "eos_token": "</s>", "max_tokens": 5, "token": ""}  # a tokenizer's token is no credential
    assert re.search(r"password=<secret#[0-9a-f]{12}>", n["tasks"]["t"]["prompt"])

    rotated = copy.deepcopy(doc)
    rotated["tasks"]["t"]["verifier_env"]["JUDGE_API_KEY"] = "rotated-key"
    rotated["policies"]["p"]["config"]["api_key"] = "another"
    assert ident.fingerprint(rotated) == ident.fingerprint(doc)  # a rotation is not drift

    edited = copy.deepcopy(doc)
    edited["tasks"]["t"]["prompt"] = "log in with password=hunter3x now"  # a credential-shaped prompt that changed is a changed task
    assert ident.fingerprint(edited) != ident.fingerprint(doc)
    assert [p for p, _ in ident.diff(ident.compared(n), ident.compared(ident.normalize(edited)))] == ["tasks.t.prompt"]
    assert "hunter3x" not in json.dumps(ident.normalize(edited))
    renamed = copy.deepcopy(doc)
    renamed["tasks"]["t"]["verifier_env"]["OTHER_API_KEY"] = "x-key"
    assert ident.fingerprint(renamed) != ident.fingerprint(doc)  # the name of a credential variable is still part of the configuration


def test_a_credential_in_a_run_json_is_never_written(tmp_path):
    doc = {"tasks": {"t": {"verifier_env": {"JUDGE_API_KEY": "internal-key-1"}}}}
    open_doc(tmp_path, doc).close()
    assert "internal-key-1" not in (tmp_path / "r" / "run.json").read_text()
    rotated = {"tasks": {"t": {"verifier_env": {"JUDGE_API_KEY": "rotated"}}}}
    open_doc(tmp_path, rotated).close()  # the rotated key resumes
    assert len(run_json(tmp_path)["epochs"]) == 1


def test_a_harness_digest_is_over_the_leaves_the_comparison_sees():
    a = ident.harness_digest({"model": "m", "work_dir": "/a", "api_key": "k1", "timeout": 5, "temperature": 0.5})
    assert re.fullmatch(r"[0-9a-f]{12}", a)
    assert a == ident.harness_digest({"model": "m", "work_dir": "/b", "api_key": "k2", "timeout": 9, "temperature": 0.5})
    assert a != ident.harness_digest({"model": "m", "work_dir": "/a", "api_key": "k1", "timeout": 5, "temperature": 0.6})
    assert ident.harness_digest({"endpoint": "http://gpu1"}) != ident.harness_digest({"endpoint": "http://gpu2"})  # an endpoint is compared


def test_diff_names_dotpaths_and_accepts_matches_them():
    old = {"a": {"b": 1, "c": [1, 2], "gone": 1}, "same": 1}
    new = {"a": {"b": 2, "c": [1, 3, 4], "new": {"x": 1}}, "same": 1}
    got = dict(ident.diff(old, new))
    assert set(got) == {"a.b", "a.c.1", "a.c.2", "a.gone", "a.new"}
    assert got["a.b"] == "a.b: 1 -> 2" and got["a.gone"] == "a.gone: 1 -> <missing>" and got["a.c.2"].endswith("<missing> -> 4")
    assert ident.changes(old, old) == []
    assert ident.accepts("policies.llm:m.config.temperature", ["policies.*.config.temperature"])
    assert ident.accepts("policies.llm:m.config.temperature", ["policies.llm:m"])  # a subtree, at a dot boundary
    assert not ident.accepts("policies.llm:mm.config.temperature", ["policies.llm:m"])
    assert ident.accepts("anything.at.all", ["*"])
    assert not ident.accepts("a.b", [])


def test_the_fingerprint_ignores_operational_fields_and_sees_the_rest():
    a = ident.fingerprint(spec(concurrency=1, budget_usd=1.0).resolved_doc())
    assert a == ident.fingerprint(spec(concurrency=8, budget_usd=9.0, wall_clock_s=5.0, retry_backoff_s=0.0).resolved_doc())
    assert a != ident.fingerprint(spec(sampling={"temperature": 1.0}).resolved_doc())
    assert a != ident.fingerprint(spec(rollout_mode="train").resolved_doc())
    assert a != ident.fingerprint(spec(retry=dataclasses.replace(spec().retry, max_attempts=5)).resolved_doc())


def test_tree_digests_name_the_part_that_changed(tmp_path):
    d = tmp_path / "task"
    (d / "tests").mkdir(parents=True)
    (d / "tests" / "test.sh").write_text("exit 0\n")
    (d / "task.toml").write_text("[task]\n")
    (d / ".git").mkdir()
    (d / ".git" / "HEAD").write_text("x")
    one = ident.tree_digests(d)
    assert set(one) == {"tests", "task.toml"} and all(len(v) == 64 for v in one.values())  # a dotdir is not content
    (d / "tests" / "test.sh").write_text("exit 1  # a different length, so the stat signature differs\n")
    two = ident.tree_digests(d)
    assert two["tests"] != one["tests"] and two["task.toml"] == one["task.toml"]
    assert ident.tree_digest(d / "missing") is None and ident.file_digest(d / "missing") is None and ident.tree_digests(d / "missing") == {}


def test_task_trees_count_hidden_files_and_pack_trees_do_not(tmp_path):
    """Harbor's task hash covers every file but VCS and junk: ``environment/.dockerignore`` decides what is built and
    ``tests/.expected_output.json`` what is graded (a pack tree keeps ``studio.runs.pack_hash``'s rule: dotfiles are not content)."""
    d = tmp_path / "task"
    (d / "environment").mkdir(parents=True)
    (d / "tests" / "__pycache__").mkdir(parents=True)
    (d / ".git").mkdir()
    (d / "environment" / ".dockerignore").write_text("fixtures\n")
    (d / "tests" / ".expected_output.json").write_text('{"a": 1}')
    (d / "tests" / "test.sh").write_text("exit 0\n")
    (d / ".gitignore").write_text("x\n")
    (d / ".git" / "HEAD").write_text("ref")
    (d / "tests" / "__pycache__" / "m.pyc").write_text("a")
    (d / "tests" / "test.sh.swp").write_text("a")
    one = ident.tree_digests(d)
    assert set(one) == {"environment", "tests", ".gitignore"}  # a top-level .gitignore decides what Harbor hashes: it is named
    pack_one = ident.tree_digest(d)

    (d / "environment" / ".dockerignore").write_text("fixtures\nand a second, longer line\n")
    two = ident.tree_digests(d)
    assert two["environment"] != one["environment"] and two["tests"] == one["tests"]
    (d / "tests" / ".expected_output.json").write_text('{"a": 2, "b": 3}')
    three = ident.tree_digests(d)
    assert three["tests"] != two["tests"] and three["environment"] == two["environment"]
    (d / ".gitignore").write_text("x\nand a second line\n")
    four = ident.tree_digests(d)
    assert four[".gitignore"] != three[".gitignore"]

    (d / ".git" / "HEAD").write_text("ref: another, longer")  # version control, bytecode and swap files are not content
    (d / "tests" / "__pycache__" / "m.pyc").write_text("bytecode, longer")
    assert ident.tree_digests(d) == four
    assert ident.tree_digest(d) == pack_one  # a pack tree never counted the dotfiles
    assert ident.tree_digest(d, task=True) != pack_one
    (d / "tests" / "test.sh.swp").write_text("swap, longer")  # an editor's swap file is junk in a task tree (Harbor's DEFAULT_IGNORES)
    assert ident.tree_digests(d) == four


def test_code_versions_has_the_recorded_fields():
    v = ident.code_versions()
    assert set(v) == {"crucible", "git_sha", "git_dirty", "harbor", "verifiers"}
    assert v is ident.code_versions()  # cached for the process


# ---------------------------------------------------------------- harness ids


def test_harness_ids_carry_a_config_digest(tmp_path):
    h = NativeHarness()
    assert re.fullmatch(r"native:llm-tool@1#[0-9a-f]{12}", h.id)
    assert h.id == NativeHarness().id  # the same configuration: the same id
    assert NativeHarness(limits=EpisodeLimits(max_steps=30)).id != NativeHarness(limits=EpisodeLimits(max_steps=100)).id
    assert h.id != NativeHarness(limit_policy=LimitPolicy(on_steps="mask")).id
    assert h.id != NativeHarness(checkpoint_every=5).id
    # the grading policy and the progress cadence are not in the id (R-3: they say how the same work is graded)
    assert h.id == NativeHarness(grade_retries=0).id == NativeHarness(scoring_budget_s=5.0).id == NativeHarness(control_timeout_s=10.0).id
    assert h.id == NativeHarness(progress_full_every=10).id

    assert WorkspaceHarness(collect_mode="legacy").id != WorkspaceHarness().id
    assert WorkspaceHarness().id.startswith("native:workspace@1#") and WorkspaceHarness(host="other").id == WorkspaceHarness().id

    p = tmp_path / "x.patch"
    p.write_text("one")
    first = ImportedHarness({"import:x": p}).id
    assert first == ImportedHarness({"import:x": p}).id and first.startswith("import:patch@1#")
    p.write_text("two, longer")
    assert ImportedHarness({"import:x": p}).id != first  # the recorded run's content is the harness

    r = RLMHarness(RLMConfig(), launcher=lambda a, actor: None)
    assert r.id.startswith("rlm:") and "#" in r.id
    assert r.id != RLMHarness(RLMConfig(), launcher=lambda a, actor: None, checkpoint_every=3).id

    sized = copy.copy(h)  # bench's _sized: a copy with another step limit
    sized.limits = dataclasses.replace(h.limits, max_steps=h.limits.max_steps + 5)
    assert sized.id != h.id and h.id == NativeHarness().id


def test_harness_keys_drop_the_digest_and_fall_back_on_collision():
    a, b = NativeHarness("native:a"), NativeHarness("native:b")
    assert ident.harness_keys([a, b]) == ["native:a@1", "native:b@1"]
    assert ident.harness_keys([NativeHarness("native:a"), NativeHarness("native:a", checkpoint_every=2)]) == [
        NativeHarness("native:a").id, NativeHarness("native:a", checkpoint_every=2).id]
    assert ident.harness_keys([SimpleNamespace(id="x@1#abc"), SimpleNamespace(id="y@2")]) == ["x@1", "y@2"]
    assert ident.harness_keys([RLMHarness(RLMConfig(temperature=0.1), launcher=lambda a, x: None)]) == ["rlm@1"]


def test_the_resolved_document_names_every_part(tmp_path):
    doc = spec(sampling={"temperature": 0.7}).resolved_doc()
    assert doc["v"] == 2 and doc["rollout_mode"] == "eval" and doc["sampling"] == {"temperature": 0.7}
    assert doc["tasks"] == {"t": {"id": "t"}}
    assert doc["policies"]["llm:m"] == {"id": "llm:m", "config": {"model": "m", "temperature": 0.7}}
    h = doc["harnesses"]["native:quick@1"]
    assert h["limits"]["max_steps"] == 30 and h["limit_policy"]["on_steps"] == "score" and h["checkpoint_every"] == 0 and h["grade_retries"] == 1
    assert doc["retry"]["max_attempts"] == 3 and doc["concurrency"] == 4


# ---------------------------------------------------------------- the resume


async def test_a_policy_without_config_is_refused_before_the_run_dir_exists(tmp_path):
    s = spec(policies=[PolicyRef(id="p", make=lambda a: {})])
    with pytest.raises(Undeclared) as e:
        await Runner(tmp_path).run(s)
    assert "policies.p" in str(e.value) and "config" in str(e.value)
    assert not (tmp_path / "r").exists()


async def test_max_steps_change_refuses_the_resume_naming_the_dotpath(tmp_path):
    def s(n: int) -> RunSpec:
        return spec(harnesses=[Quick("native:quick", limits=EpisodeLimits(max_steps=n))])

    assert (await Runner(tmp_path).run(s(30))).terminal == 1
    before = {n: (tmp_path / "r" / n).read_bytes() for n in ("results.jsonl", "run.json")}
    with pytest.raises(IdentityDrift) as e:
        await Runner(tmp_path).run(s(100))
    assert "harnesses.native:quick@1.limits.max_steps" in str(e.value) and "30 -> 100" in str(e.value)
    assert {n: (tmp_path / "r" / n).read_bytes() for n in ("results.jsonl", "run.json")} == before
    await Runner(tmp_path).run(s(30))  # the refusal left the run unlocked and untouched: the same configuration resumes


async def test_temperature_change_refuses_the_resume(tmp_path):
    await Runner(tmp_path).run(spec(policies=[pol(temperature=0.7)]))
    with pytest.raises(IdentityDrift) as e:
        await Runner(tmp_path).run(spec(policies=[pol(temperature=0.9)]))
    assert "policies.llm:m.config.temperature" in str(e.value)

    await Runner(tmp_path).run(spec(run_id="r2", sampling={"temperature": 0.7}))
    with pytest.raises(IdentityDrift) as e2:
        await Runner(tmp_path).run(spec(run_id="r2", sampling={"temperature": 1.0}))
    assert "sampling.temperature" in str(e2.value)


async def test_grader_file_change_refuses_the_resume(tmp_path):
    d = tmp_path / "task"
    (d / "tests").mkdir(parents=True)
    (d / "task.toml").write_text("[task]\n")
    (d / "tests" / "test.sh").write_text("exit 0\n")

    class GradedTask:
        id = "t/graded@1"

        def identity_config(self) -> dict:
            return {"files": ident.tree_digests(d)}

    def s() -> RunSpec:
        return spec(tasks=[GradedTask()])

    await Runner(tmp_path).run(s())
    (d / "tests" / "test.sh").write_text("exit 1  # edited: a different length\n")
    with pytest.raises(IdentityDrift) as e:
        await Runner(tmp_path).run(s())
    assert "tasks.t/graded@1.files.tests" in str(e.value) and "files.task.toml" not in str(e.value)


def test_patch_map_change_refuses_the_resume(tmp_path):
    p = tmp_path / "x.patch"
    p.write_text("--- a\n")

    def s() -> RunSpec:
        return RunSpec(run_id="r", tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p")], harnesses=[ImportedHarness({"import:x": p})])

    open_journal(tmp_path, s()).close()
    open_journal(tmp_path, s()).close()  # the same patch: resumes
    p.write_text("--- b\n+++ c, rewritten and longer\n")
    with pytest.raises(IdentityDrift, match=r"patches\.import:x"):
        open_journal(tmp_path, s())


def test_an_rlm_config_change_is_named_by_dotpath(tmp_path):
    def s(temp: float) -> RunSpec:
        return RunSpec(run_id="r", tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p")],
                       harnesses=[RLMHarness(RLMConfig(temperature=temp), launcher=lambda a, actor: None)])

    open_journal(tmp_path, s(0.5)).close()
    with pytest.raises(IdentityDrift) as e:
        open_journal(tmp_path, s(1.0))
    assert "harnesses.rlm@1.config.temperature: 0.5 -> 1.0" in str(e.value)


async def test_operational_changes_resume_without_drift(tmp_path):
    await Runner(tmp_path).run(spec(concurrency=2, budget_usd=5.0))
    s2 = await Runner(tmp_path).run(spec(concurrency=1, budget_usd=9.0, wall_clock_s=100.0, attempt_timeout_s=50.0, retry_backoff_s=0.0, max_mask_fraction=0.5,
                                         harnesses=[Quick("native:quick", scoring_budget_s=10.0, control_timeout_s=5.0, progress_full_every=7)]))
    assert s2.attempts == 0 and s2.terminal == 1
    assert len(run_json(tmp_path)["epochs"]) == 1 and run_json(tmp_path)["epochs"][0]["epoch"] == 0


async def test_a_grading_policy_change_that_is_not_operational_is_refused(tmp_path):
    await Runner(tmp_path).run(spec(harnesses=[Quick("native:quick", grade_retries=1)]))
    with pytest.raises(IdentityDrift, match=r"harnesses\.native:quick@1\.grade_retries: 1 -> 0"):
        await Runner(tmp_path).run(spec(harnesses=[Quick("native:quick", grade_retries=0)]))


async def test_accept_drift_opens_an_epoch_and_rows_carry_it(tmp_path):
    h = Credits()
    first = await Runner(tmp_path).run(spec(policies=[pol(temperature=0.7)], harnesses=[h]))
    assert first.terminal == 0 and rows(tmp_path) == []  # owed: the provider ran out of credits

    with pytest.raises(IdentityDrift):  # unaccepted: refused
        await Runner(tmp_path).run(spec(policies=[pol(temperature=0.9)], harnesses=[h]))
    s2 = await Runner(tmp_path).run(spec(policies=[pol(temperature=0.9)], harnesses=[h], accept_drift=("policies.*.config.temperature",)))
    assert s2.terminal == 1
    assert rows(tmp_path)[-1]["meta"]["epoch"] == 1
    j = run_json(tmp_path)
    assert [e["epoch"] for e in j["epochs"]] == [0, 1] and j["epochs"][1]["accepted"] == ["policies.llm:m.config.temperature"]
    assert j["resolved"]["policies"]["llm:m"]["config"]["temperature"] == 0.9 and j["identity_version"] == 2
    ev = events(tmp_path)
    assert any(e["event"] == "epoch" and e["epoch"] == 1 and e["accepted"] == ["policies.llm:m.config.temperature"] for e in ev)
    assert any(e["event"] in ("planned", "resume") and e.get("epoch") == 1 for e in ev)

    s3 = await Runner(tmp_path).run(spec(policies=[pol(temperature=0.9)], harnesses=[h]))  # the new configuration is now the run's
    assert s3.attempts == 0 and len(run_json(tmp_path)["epochs"]) == 2


def test_accept_drift_must_cover_every_changed_path(tmp_path):
    open_journal(tmp_path, spec(policies=[pol(temperature=0.7)])).close()
    s = spec(policies=[pol(temperature=0.9, model="other")], accept_drift=("policies.llm:m.config.temperature",))
    with pytest.raises(IdentityDrift) as e:
        open_journal(tmp_path, s)
    assert len(e.value.changes) == 1 and e.value.changes[0].startswith("policies.llm:m.config.model: ")  # the accepted path is not listed
    assert len(run_json(tmp_path)["epochs"]) == 1  # nothing was written


async def test_a_code_version_change_opens_an_epoch_and_is_not_refused(tmp_path, monkeypatch):
    h = Credits()
    monkeypatch.setattr(ident, "code_versions", lambda: {"crucible": "1", "git_sha": "aaa", "git_dirty": False})
    assert (await Runner(tmp_path).run(spec(harnesses=[h]))).terminal == 0
    monkeypatch.setattr(ident, "code_versions", lambda: {"crucible": "1", "git_sha": "bbb", "git_dirty": False})
    s2 = await Runner(tmp_path).run(spec(harnesses=[h]))  # not refused
    assert s2.terminal == 1
    j = run_json(tmp_path)
    assert j["epochs"][-1]["code_changed"] == ["git_sha"] and j["epochs"][-1]["accepted"] == [] and j["code"]["git_sha"] == "bbb"
    assert rows(tmp_path)[-1]["meta"]["epoch"] == 1
    ev = events(tmp_path)
    assert any(e["event"] == "epoch" and e["code_changed"] == ["git_sha"] for e in ev)


async def test_a_finished_run_reopened_under_new_code_dispatches_nothing_and_records_an_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(ident, "code_versions", lambda: {"git_sha": "aaa"})
    await Runner(tmp_path).run(spec())
    monkeypatch.setattr(ident, "code_versions", lambda: {"git_sha": "bbb"})
    again = await Runner(tmp_path).run(spec())
    assert again.attempts == 0 and again.terminal == 1
    assert [e["epoch"] for e in run_json(tmp_path)["epochs"]] == [0, 1]
    assert "epoch" not in rows(tmp_path)[0]["meta"]  # epoch 0 rows are as they were before R-4


async def test_a_run_written_before_identity_v2_is_refused_unless_accepted(tmp_path):
    RunJournal(tmp_path, "r", "old", {"run_id": "r"}).close()  # the v1 shape: ids only
    with pytest.raises(IdentityDrift, match="identity v1"):
        await Runner(tmp_path).run(spec())
    # a listed pseudo-path or a real dotpath cannot accept it: the message says that only ``*`` does, and what it costs
    with pytest.raises(IdentityDrift) as e:
        await Runner(tmp_path).run(spec(accept_drift=("tasks",)))
    assert "'*'" in str(e.value) and "every sample again" in str(e.value) and "does not cover" not in str(e.value) and "<dotpath>" not in str(e.value)
    s = await Runner(tmp_path).run(spec(accept_drift=("*",)))
    assert s.terminal == 1
    j = run_json(tmp_path)
    assert j["epochs"][-1]["epoch"] == 1 and j["epochs"][0]["resolved"] is None and "resolved" in j
    assert rows(tmp_path)[-1]["meta"]["epoch"] == 1
    assert (await Runner(tmp_path).run(spec())).attempts == 0  # and the run is v2 now: the same configuration resumes


async def test_rows_under_an_old_epochs_keys_are_not_counted_in_the_summary(tmp_path):
    """An accepted harness-configuration drift gives every sample a new key: the earlier epoch's rows stay in results.jsonl but are not
    this plan's results (``terminal`` never passes ``planned``, the means are not an average of two configurations)."""
    def s(n: int) -> RunSpec:
        return spec(harnesses=[Quick("native:quick", limits=EpisodeLimits(max_steps=n))], accept_drift=("harnesses.*.limits.max_steps",))

    first = await Runner(tmp_path).run(s(30))
    assert (first.planned, first.terminal, first.superseded) == (1, 1, 0)
    second = await Runner(tmp_path).run(s(100))
    assert (second.planned, second.terminal, second.attempts, second.superseded) == (1, 1, 1, 1)
    assert second.coverage <= 1.0 and len(rows(tmp_path)) == 2  # both epochs' rows are on disk
    assert rows(tmp_path)[-1]["meta"]["epoch"] == 1
    again = await Runner(tmp_path).run(s(100))
    assert again.attempts == 0 and again.terminal == 1 and again.superseded == 1  # the same plan resumes: nothing is owed


def test_a_journal_without_a_resolved_document_keeps_the_old_check(tmp_path):
    RunJournal(tmp_path, "r", "id-1", {}).close()
    assert "resolved" not in run_json(tmp_path) and "epochs" not in run_json(tmp_path)
    with pytest.raises(ValueError, match="different identity"):
        RunJournal(tmp_path, "r", "id-2", {})


def test_refusals_are_user_errors():
    assert issubclass(Undeclared, ValueError) and issubclass(IdentityDrift, ValueError) and issubclass(RunBusy, ValueError)
    e = IdentityDrift("r", [f"a.{i}: 1 -> 2" for i in range(25)])
    assert "(+5 more)" in str(e) and "a.19: 1 -> 2" in str(e) and "a.20" not in str(e) and "--accept-drift" in str(e) and "--out" in str(e)
