"""M10 lab audit R-4, slice 3: every in-tree adapter above the runtime declares its configuration and content (the run's
identity v2 compares them on a resume), the A2A id carries a configuration digest, ``bench run --accept-drift`` reaches the
``RunSpec``, and a bench report splits by identity epoch only when results span more than one.

No Docker, no Harbor, no network: the Harbor task is a directory of plain files (``HarborDirTask`` needs no Harbor), the
harnesses are real ``NativeHarness`` subclasses that run no episode or fakes with a declaration, and ``run_bench`` is replaced
where the test is about the CLI's wiring.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.core.schema.run import Failure
from crucible.runtime import identity as ident
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.identity import IdentityDrift, describe
from crucible.runtime.inject import FakeLLM
from crucible.runtime.journal import RunJournal
from crucible.runtime.run import Runner, RunSpec


class Quick(NativeHarness):
    """A real ``NativeHarness`` that runs no episode: it inherits ``id`` and ``identity_config``."""

    async def run(self, attempt: Attempt) -> HarnessResult:
        return HarnessResult(reward=1.0)


class Split:
    """Sample 1 hits ``crucible:credits_exhausted`` (the run stops, the sample stays owed) until ``fixed``; sample 0 always scores."""

    id = "split@1"

    def __init__(self) -> None:
        self.fixed = False

    def identity_config(self) -> dict:
        return {}

    async def run(self, attempt: Attempt) -> HarnessResult:
        if attempt.sample.sample_idx == 1 and not self.fixed:
            return HarnessResult(failure=Failure(kind="crucible:credits_exhausted", stage="agent", reason="x"))
        return HarnessResult(reward=1.0)


def pol(**cfg) -> PolicyRef:
    return PolicyRef(id="llm:m", make=lambda a: {}, config={"model": "m", **cfg})


def open_journal(root: Path, run_id: str, doc: dict, **kw) -> RunJournal:
    """What ``Runner._open`` does with a resolved document, so a test can hold the journal and close it."""
    return RunJournal(root, run_id, ident.fingerprint(doc), {"run_id": run_id}, resolved=ident.normalize(doc), code={}, **kw)


def run_json(root: Path, run_id: str = "r") -> dict:
    return json.loads((root / run_id / "run.json").read_text())


# ---------------------------------------------------------------- Harbor tasks


def _task_dir(tmp_path: Path) -> Path:
    d = tmp_path / "demo"
    (d / "tests").mkdir(parents=True)
    (d / "solution").mkdir()
    (d / "task.toml").write_text('[task]\nname = "demo"\n[verifier]\ntimeout_sec = 60\n')
    (d / "instruction.md").write_text("Make the thing.\n")
    (d / "tests" / "test.sh").write_text("echo ok\n")
    (d / "solution" / "model.patch").write_text("diff --git a/x b/x\n")
    return d


def test_harbor_dir_task_declares_its_tree_and_image(tmp_path):
    from crucible.interop.harbor.verify_dir import HarborDirTask

    d = _task_dir(tmp_path)
    image = "img@sha256:aa"
    d1 = describe(HarborDirTask(d, image=image, expected_image=image), "tasks.demo")
    assert d1["image"] == image and d1["expected_image"] == image
    assert set(d1["files"]) == {"task.toml", "instruction.md", "tests", "solution"} and d1["files"]["tests"]  # one digest per part
    assert d1["oracle_patch"] == ident.file_digest(d / "solution" / "model.patch")
    assert d1["host"] is None and "timeout" not in d1  # a non-perf task's host is placement, a timeout operational: neither is its identity

    open_journal(tmp_path / "runs", "r", {"tasks": {"demo": d1}}).close()
    (d / "tests" / "test.sh").write_text("echo changed, and longer so the digest cannot be memoized away\n")
    d2 = describe(HarborDirTask(d, image=image, expected_image=image), "tasks.demo")
    with pytest.raises(IdentityDrift, match=r"tasks\.demo\.files\.tests") as e:
        open_journal(tmp_path / "runs", "r", {"tasks": {"demo": d2}})
    assert len(e.value.changes) == 1  # the other parts of the tree did not change
    # the stored tree still resumes: the refused open wrote nothing
    open_journal(tmp_path / "runs", "r", {"tasks": {"demo": d1}}).close()


def test_a_perf_tasks_host_is_part_of_what_it_measures(tmp_path):
    """A speedup is measured on the host's hardware: moving a perf task to another GPU box is refused, a non-perf task's host is not compared."""
    from crucible.interop.harbor.verify_dir import HarborDirTask
    from crucible.runtime.gpu import GpuHost

    d = _task_dir(tmp_path)
    lab = describe(HarborDirTask(d, image="i", perf=True, host=GpuHost(name="LAB")), "tasks.demo")
    other = describe(HarborDirTask(d, image="i", perf=True, host=GpuHost(name="OTHER")), "tasks.demo")
    assert lab["host"] == "LAB" and other["host"] == "OTHER"
    open_journal(tmp_path / "runs", "r", {"tasks": {"demo": lab}}).close()
    with pytest.raises(IdentityDrift, match=r"tasks\.demo\.host"):
        open_journal(tmp_path / "runs", "r", {"tasks": {"demo": other}})
    plain = [describe(HarborDirTask(d, image="i", host=GpuHost(name=n)), "tasks.demo") for n in ("LAB", "OTHER")]
    assert plain[0] == plain[1]  # not perf: the host is where it runs, not what it measures


def test_a_hidden_task_file_is_content(tmp_path):
    """``environment/.dockerignore`` decides what is built: editing it is a different task, as Harbor's own task hash has it."""
    from crucible.interop.harbor.verify_dir import HarborDirTask

    d = _task_dir(tmp_path)
    (d / "environment").mkdir()
    (d / "environment" / ".dockerignore").write_text("fixtures\n")
    t1 = describe(HarborDirTask(d, image="i"), "tasks.demo")
    (d / "environment" / ".dockerignore").write_text("fixtures\nand a second, longer line\n")
    t2 = describe(HarborDirTask(d, image="i"), "tasks.demo")
    assert t1["files"]["environment"] != t2["files"]["environment"] and t1["files"]["tests"] == t2["files"]["tests"]
    open_journal(tmp_path / "runs", "r", {"tasks": {"demo": t1}}).close()
    with pytest.raises(IdentityDrift, match=r"tasks\.demo\.files\.environment"):
        open_journal(tmp_path / "runs", "r", {"tasks": {"demo": t2}})


def test_harbor_dir_task_image_change_is_refused(tmp_path):
    from crucible.interop.harbor.verify_dir import HarborDirTask

    d = _task_dir(tmp_path)
    t1 = describe(HarborDirTask(d, image="img:v1", expected_image="sha256:aa"), "tasks.demo")
    t2 = describe(HarborDirTask(d, image="img:v1", expected_image="sha256:bb"), "tasks.demo")
    open_journal(tmp_path / "runs", "r", {"tasks": {"demo": t1}}).close()
    with pytest.raises(IdentityDrift, match=r"tasks\.demo\.expected_image"):
        open_journal(tmp_path / "runs", "r", {"tasks": {"demo": t2}})


def test_a_task_file_named_like_a_credential_is_still_content(tmp_path):
    """A tree digest map is never filtered by the denylist: a file named ``token`` or ``root`` is task content."""
    from crucible.interop.harbor.verify_dir import HarborDirTask

    d = _task_dir(tmp_path)
    (d / "token").write_text("one\n")
    t1 = describe(HarborDirTask(d, image="i", expected_image=None), "tasks.demo")
    (d / "token").write_text("another, longer\n")
    t2 = describe(HarborDirTask(d, image="i", expected_image=None), "tasks.demo")
    assert t1["files"]["token"] != t2["files"]["token"]
    kept = ident.compared(ident.normalize({"tasks": {"demo": t2}}))
    assert "token" in kept["tasks"]["demo"]["files"]


def test_harbor_agent_harness_declares_its_config(tmp_path):
    from crucible.connectors.capture_proxy import CaptureProxy
    from crucible.interop.harbor.agent_harness import AgentLimits, HarborAgentHarness

    proxy = CaptureProxy(tmp_path / "cap")
    fac = lambda **kw: None  # noqa: E731 -- never called: the declaration needs no Harbor

    def h(**kw) -> HarborAgentHarness:
        kw.setdefault("harbor_version", "0.24.0")
        return HarborAgentHarness("fake-agent", "1", model="openai/m", proxy=proxy, proxy_url="http://proxy", agent_factory=fac,
                                  endpoint_env={"OPENAI_API_KEY": "{key}", "OPENAI_BASE_URL": "{proxy}/v1"}, **kw)

    base = h(agent_kwargs={"config": {"agent": {"step_limit": 10}}}, limits=AgentLimits(max_model_calls=7))
    doc = describe(base, "harnesses.x")
    assert doc["agent_kwargs"] == {"config": {"agent": {"step_limit": 10}}} and doc["limits"]["max_model_calls"] == 7
    assert doc["model"] == "openai/m" and doc["agent"] == "fake-agent" and doc["grade_retries"] == 1
    assert doc["agent_factory"] == ident.code_ref(fac)
    assert doc["harbor_version"] == "0.24.0"  # in the id's digest, so a compared leaf too (the id and the comparison see the same things)
    assert base.id == f"harbor:fake-agent@1#{base.digest}"  # R-1's id is unchanged by the declaration

    # a Harbor upgrade gives every sample a new key (R-1's digest): resuming under it would run the whole plan again, paid, so it is
    # refused naming the leaf, and accepted only on purpose (a new identity epoch)
    up = h(agent_kwargs={"config": {"agent": {"step_limit": 10}}}, limits=AgentLimits(max_model_calls=7), harbor_version="0.25.0")
    assert up.id != base.id and ident.harness_keys([up]) == ident.harness_keys([base]) == ["harbor:fake-agent@1"]
    key = ident.harness_keys([base])[0]
    open_journal(tmp_path / "runs", "r", {"harnesses": {key: doc}}).close()
    with pytest.raises(IdentityDrift, match=r"harnesses\.harbor:fake-agent@1\.harbor_version: '0\.24\.0' -> '0\.25\.0'"):
        open_journal(tmp_path / "runs", "r", {"harnesses": {key: describe(up, "harnesses.x")}})
    open_journal(tmp_path / "runs", "r", {"harnesses": {key: describe(up, "harnesses.x")}}, accept_drift=("harnesses.*.harbor_version",)).close()
    assert len(run_json(tmp_path / "runs")["epochs"]) == 2 and run_json(tmp_path / "runs")["epochs"][1]["accepted"] == ["harnesses.harbor:fake-agent@1.harbor_version"]

    # the id digests what the comparison sees: a path or a credential in agent_kwargs is neither compared nor in the digest
    assert h(agent_kwargs={"work_dir": "/a", "api_key": "k1", "x": 1}).id == h(agent_kwargs={"work_dir": "/b", "api_key": "k2", "x": 1}).id
    assert h(agent_kwargs={"x": 1}).id != h(agent_kwargs={"x": 2}).id

    # while a real configuration change is named by dotpath
    key = ident.harness_keys([up])[0]
    open_journal(tmp_path / "runs", "r2", {"harnesses": {key: describe(base, "harnesses.x")}}).close()
    with pytest.raises(IdentityDrift, match=r"harnesses\.harbor:fake-agent@1\.agent_kwargs\.config\.agent\.step_limit"):
        open_journal(tmp_path / "runs", "r2", {"harnesses": {key: describe(h(agent_kwargs={"config": {"agent": {"step_limit": 20}}},
                                                                            limits=AgentLimits(max_model_calls=7)), "harnesses.x")}})


# ---------------------------------------------------------------- the A2A harness


def test_a2a_harness_id_carries_a_config_digest():
    from crucible.interop.agentenv.harness import A2AHarness

    a = A2AHarness(agent_url="http://h:1", agent_config={"a": 1})
    assert a.id.startswith("a2a@1#") and len(a.id.split("#")[1]) == 12
    assert A2AHarness(agent_url="http://h:1", agent_config={"a": 1}).id == a.id  # the same configuration: the same SampleKeys
    assert A2AHarness(agent_url="http://h:1", agent_config={"a": 2}).id != a.id
    assert A2AHarness(agent_url="http://other:9", agent_config={"a": 1}).id == a.id  # where the agent runs is placement
    assert A2AHarness(agent_url="http://h:1", agent_config={"a": 1}, timeout_s=5).id != a.id
    assert A2AHarness(agent_url="http://h:1", agent_config={"a": 1}, grade_retries=0, scoring_budget_s=9.0).id == a.id  # grading policy (R-3) is not in it
    assert A2AHarness(name="a2b", agent_url="http://h:1", agent_config={"a": 1}).id.startswith("a2b@1#")

    doc = describe(a, "harnesses.a2a")
    assert doc["agent_config"] == {"a": 1} and doc["agent_url"] == "http://h:1" and doc["grade_retries"] == 1
    assert a.identity_key == "a2a@1" and ident.harness_keys([a]) == ["a2a@1"]
    kept = ident.compared(ident.normalize({"harnesses": {"a2a@1": doc}}))["harnesses"]["a2a@1"]
    assert kept["agent_url"] == "http://h:1" and "scoring_budget_s" not in kept and kept["agent_config"] == {"a": 1}  # the agent measured is compared


def test_a2a_agent_url_and_endpoint_leaves_are_compared(tmp_path):
    """Moving a run to another remote agent (or another model server in its config) is a different measurement: refused by name."""
    from crucible.interop.agentenv.harness import A2AHarness

    a1, a2 = A2AHarness(agent_url="http://h:1", agent_config={"llm_base_url": "http://gpu1:8000/v1"}), A2AHarness(
        agent_url="http://h:2", agent_config={"llm_base_url": "http://gpu1:8000/v1"})
    open_journal(tmp_path, "r", {"harnesses": {a1.identity_key: describe(a1, "h")}}).close()
    with pytest.raises(IdentityDrift, match=r"harnesses\.a2a@1\.agent_url: 'http://h:1' -> 'http://h:2'"):
        open_journal(tmp_path, "r", {"harnesses": {a2.identity_key: describe(a2, "h")}})
    assert a1.id == a2.id  # where the agent runs is not in the id: accepting the drift keeps the SampleKeys, and the rows carry the epoch
    a3 = A2AHarness(agent_url="http://h:1", agent_config={"llm_base_url": "http://gpu2:8000/v1"})
    assert a3.id != a1.id  # the model server is the agent's configuration: it is in the id, and (an endpoint is compared) in the identity
    with pytest.raises(IdentityDrift, match=r"harnesses\.a2a@1\.agent_config\.llm_base_url"):
        open_journal(tmp_path, "r", {"harnesses": {a3.identity_key: describe(a3, "h")}})


def test_a2a_id_digests_what_the_identity_compares():
    """A leaf the comparison drops (a path, a credential, a timeout) is not in the id: changing it cannot re-key every sample unseen."""
    from crucible.interop.agentenv.harness import A2AHarness

    def a(**cfg) -> A2AHarness:
        return A2AHarness(agent_url="http://h:1", agent_config={"model": "m", **cfg})

    assert a(work_dir="/a", api_key="k1").id == a(work_dir="/b", api_key="k2").id
    assert a(model_dir="/x").id == a(model_dir="/y").id  # a path key is dropped from both views
    assert a(temperature=0.5).id != a(temperature=0.6).id
    assert a().id !=A2AHarness(agent_url="http://h:1", agent_config={"model": "m2"}).id


def test_a2a_agent_factory_is_named_or_described_never_a_closure():
    """A lambda or a closure holds configuration nobody can inspect (code_ref named every one ``<locals>.<lambda>``); a partial's bound
    arguments are the agent's configuration."""
    import functools

    from crucible.interop.agentenv.harness import A2AHarness
    from crucible.runtime.identity import Undeclared

    class Agent:
        def __init__(self, model: str = "x") -> None:
            self.model = model

    def make(model: str):
        return lambda: Agent(model)

    ids = {A2AHarness(agent=functools.partial(Agent, model=m)).id for m in ("a", "b")}
    assert len(ids) == 2  # partial(Agent, model="a") is not partial(Agent, model="b")
    assert A2AHarness(agent=functools.partial(Agent, model="a")).id == A2AHarness(agent=functools.partial(Agent, model="a")).id
    assert A2AHarness(agent=Agent).id.startswith("a2a@1#")  # a class is named
    with pytest.raises(Undeclared, match="lambda or a closure"):
        A2AHarness(agent=make("a"))
    with pytest.raises(Undeclared, match="lambda or a closure"):
        A2AHarness(agent=lambda: Agent())
    with pytest.raises(Undeclared, match="agent.keywords.model"):
        A2AHarness(agent=functools.partial(Agent, model=lambda: 0))  # a bound argument that cannot be inspected


def test_a2a_agent_config_change_is_refused_by_name(tmp_path):
    from crucible.interop.agentenv.harness import A2AHarness

    a1, a2 = A2AHarness(agent_url="http://h:1", agent_config={"model": "m1"}), A2AHarness(agent_url="http://h:1", agent_config={"model": "m2"})
    open_journal(tmp_path, "r", {"harnesses": {a1.identity_key: describe(a1, "h")}}).close()
    with pytest.raises(IdentityDrift, match=r"harnesses\.a2a@1\.agent_config\.model"):
        open_journal(tmp_path, "r", {"harnesses": {a2.identity_key: describe(a2, "h")}})


# ---------------------------------------------------------------- other interop tasks


def test_nemo_gym_and_openenv_tasks_declare_their_content():
    from crucible.interop.nemo_gym.import_server import NemoGymTask
    from crucible.interop.openenv.import_env import OpenEnvTask

    row = {"responses_create_params": {"input": "hi"}, "ground_truth": [{"name": "f", "arguments": "{}"}]}
    a = NemoGymTask("nemo_gym/x", "http://127.0.0.1:5001", row)
    b = NemoGymTask("nemo_gym/x", "http://localhost:5002", dict(row), tool_timeout_s=5.0)
    c = NemoGymTask("nemo_gym/x", "http://127.0.0.1:5001", {**row, "ground_truth": []})
    fold = lambda t: ident.compared(ident.normalize({"tasks": {"x": describe(t, "tasks.x")}}))  # noqa: E731
    assert fold(a) == fold(b)  # a loopback server is the one this process started (an ephemeral port); the tool timeout is operational
    assert fold(a) != fold(c)  # the row decides the grade
    remote, other = NemoGymTask("nemo_gym/x", "http://resource:8000", row), NemoGymTask("nemo_gym/x", "http://elsewhere:8000", row)
    assert fold(remote) != fold(other) and fold(remote) != fold(a)  # the server whose /verify computes the reward is what is measured

    def render(obs):
        return "rules"

    o1 = OpenEnvTask("openenv/x", "pkg.mod:make", instruction=render, progress={"p": lambda j: True}, reward_mode="last")
    d1 = describe(o1, "tasks.x")
    assert d1["instruction"] == ident.code_ref(render) and d1["progress"] == {"p": ident.code_ref(o1.progress["p"])}
    assert d1["params"]["factory"] == "pkg.mod:make" and d1["reward_mode"] == "last"
    o2 = OpenEnvTask("openenv/x", "pkg.mod:make", instruction=render, progress={"p": lambda j: True}, reward_mode="sum", tool_timeout_s=1.0)
    changed = [p for p, _ in ident.diff(fold(o1), fold(o2))]
    assert changed == ["tasks.x.reward_mode"]  # the reward rule decides the grade; the tool timeout is operational (and both lambdas share a name)


# ---------------------------------------------------------------- the toy pack


async def test_toy_llm_policy_declares_temperature_and_a_change_refuses(tmp_path):
    from crucible_pack_toy.runtime import HelpdeskTask, llm_policy

    p = llm_policy(FakeLLM([], model="m"), temperature=0.7)
    assert p.config == {"model": "m", "temperature": 0.7}

    def spec(temperature: float) -> RunSpec:
        return RunSpec(run_id="r", tasks=[HelpdeskTask()], policies=[llm_policy(FakeLLM([], model="m"), temperature=temperature)],
                       harnesses=[Quick("native:quick")], k=1)

    first = await Runner(tmp_path).run(spec(0.7))
    assert first.terminal == 1
    with pytest.raises(IdentityDrift, match=r"policies\.llm:m\.config\.temperature"):
        await Runner(tmp_path).run(spec(0.9))
    assert (await Runner(tmp_path).run(spec(0.7))).attempts == 0  # the same configuration resumes


def test_pack_tasks_declare_their_pack_content():
    from crucible_pack_toy import PACK_ROOT
    from crucible_pack_toy.runtime import FibTask, FibWorldTask, HelpdeskTask, WordFileTask, crashing_tool_world

    base = HelpdeskTask().identity_config()
    assert base["pack"] == ident.tree_digest(PACK_ROOT) and base["pack"]
    crash = HelpdeskTask(world_factory=crashing_tool_world(PACK_ROOT, tool="list_tickets")).identity_config()
    assert {k for k in base if base[k] != crash[k]} == {"world_factory"}  # differs only in the world factory
    other = HelpdeskTask(world_factory=crashing_tool_world(PACK_ROOT, tool="get_ticket")).identity_config()
    assert other["world_factory"] != crash["world_factory"]  # the injected bug is part of what it is
    later = HelpdeskTask(world_factory=crashing_tool_world(PACK_ROOT, tool="get_ticket", on_attempts=("a2",))).identity_config()
    assert later == other  # the attempts it fires on are the injection's schedule, not its identity

    for t in (HelpdeskTask(), HelpdeskTask(world_factory=crashing_tool_world(PACK_ROOT)), FibTask(), FibWorldTask(), FibWorldTask("local"), WordFileTask(seed=1)):
        doc = describe(t, "tasks.t")
        assert doc["pack"] == ident.tree_digest(PACK_ROOT)
    assert describe(WordFileTask(seed=1), "t") != describe(WordFileTask(seed=2), "t")
    assert describe(FibWorldTask("agentenv"), "t") != describe(FibWorldTask("local"), "t")


def test_a_held_out_items_seed_never_reaches_the_resolved_document():
    """``run.json`` is written under the bench's out directory, which is shared; a dynamic item's seed is an HMAC of the secret bench key
    and the pack's generator is public, so the document carries a digest of the seed, never the seed."""
    from crucible_pack_toy.longhorizon.task import BacklogTask
    from crucible_pack_toy.runtime import WordFileTask

    seed, other = 12345678901234567890, 12345678901234567891
    for make in (lambda s: BacklogTask(seed=s, n=12, reward="all"), lambda s: WordFileTask(seed=s)):
        t, u = make(seed), make(other)
        text = json.dumps(ident.normalize({"tasks": {"t": describe(t, "tasks.t")}}))
        assert str(seed) not in text and ident.seed_digest(seed) in text
        assert describe(t, "t") != describe(u, "t")  # a changed seed is still a changed document
        assert ident.seed_digest(seed) == ident.seed_digest(seed) != ident.seed_digest(other)


def test_a_callable_instruction_is_named_not_refused():
    """A task that renders its instruction from the reset observation holds a callable: it is named, never described."""
    from crucible_pack_toy.runtime import HelpdeskTask

    class Rendered(HelpdeskTask):
        def __init__(self) -> None:
            super().__init__()
            self.instruction = lambda obs: "rendered"

    doc = describe(Rendered(), "tasks.toy/helpdesk")
    assert isinstance(doc["instruction"], str) and "lambda" in doc["instruction"]


def test_pack_edit_is_refused_naming_the_pack(tmp_path):
    """A pack's content digest changes with a file under it (a grader, a seed): the resume is refused at ``tasks.<id>.pack``."""
    root = tmp_path / "pack"
    (root / "data").mkdir(parents=True)
    (root / "pack.yaml").write_text("name: x\n")
    (root / "data" / "grader.py").write_text("def grade():\n    return 1\n")
    d1 = {"pack": ident.tree_digest(root)}
    open_journal(tmp_path / "runs", "r", {"tasks": {"t": d1}}).close()
    (root / "data" / "grader.py").write_text("def grade():\n    return 1  # edited: a different length\n")
    with pytest.raises(IdentityDrift, match=r"tasks\.t\.pack"):
        open_journal(tmp_path / "runs", "r", {"tasks": {"t": {"pack": ident.tree_digest(root)}}})


# ---------------------------------------------------------------- policies


def test_in_tree_policies_declare_a_config():
    from crucible.bench.run import control_policies
    from crucible.runtime.single_turn import answer_policy, single_turn_policy
    from crucible.studio.pack import policy_refs

    refs = [*control_policies(), *policy_refs([lambda t: []]), single_turn_policy(FakeLLM([], model="m")), answer_policy(str)]
    assert len(refs) == 5
    for ref in refs:
        assert ref.config is not None, ref.id
        doc = describe(ref, "policies.p")
        assert doc["id"] == ref.id and doc["config"] == ref.config

    from crucible_pack_toy.runtime import NOOP, ORACLE

    assert describe(ORACLE, "p")["config"]["actions"] and describe(NOOP, "p")["config"]["actions"] == []
    assert policy_refs([pol()])[0].config == {"model": "m"}  # a PolicyRef the pack wrote passes through as it is


def test_a_pack_policy_without_a_config_is_refused_at_run_start(tmp_path):
    from crucible.runtime.identity import Undeclared

    spec = RunSpec(run_id="r", tasks=[SimpleNamespace(id="t")], policies=[PolicyRef(id="mine", make=lambda a: {})], harnesses=[Quick("native:quick")], k=1)
    with pytest.raises(Undeclared, match=r"policies\.mine"):
        spec.resolved_doc()


# ---------------------------------------------------------------- bench reports split by epoch


def _bench_spec(**kw) -> RunSpec:
    base = {"run_id": "r", "tasks": [SimpleNamespace(id="t")], "policies": [pol(temperature=0.7)], "harnesses": [Split()], "k": 2, "concurrency": 1}
    return RunSpec(**{**base, **kw})


async def test_bench_reports_split_by_epoch(tmp_path, monkeypatch):
    from crucible.bench.metrics import episodes_json
    from crucible.bench.run import summarize_run

    h = Split()
    monkeypatch.setattr(ident, "code_versions", lambda: {"crucible": "1", "git_sha": "aaa", "git_dirty": False})
    s1 = await Runner(tmp_path).run(_bench_spec(harnesses=[h]))
    assert s1.terminal == 1 and s1.stopped and "credits" in s1.stopped  # sample 0 scored, sample 1 is owed

    single, _ = summarize_run(tmp_path / "r")
    assert "epochs" not in single["run"] and not [w for w in single["warnings"] if "identity epochs" in w]  # one epoch: the report is as it was

    h.fixed = True
    monkeypatch.setattr(ident, "code_versions", lambda: {"crucible": "1", "git_sha": "bbb", "git_dirty": False})
    s2 = await Runner(tmp_path).run(_bench_spec(harnesses=[h]))
    assert s2.terminal == 2

    summary, eps = summarize_run(tmp_path / "r")
    epochs = summary["run"]["epochs"]
    assert [e["epoch"] for e in epochs] == [0, 1]
    assert [e["terminal"] for e in epochs] == [1, 1] and [e["measured"] for e in epochs] == [1, 1] and [e["mean_reward"] for e in epochs] == [1.0, 1.0]
    assert epochs[0]["code_changed"] == [] and epochs[1]["code_changed"] == ["git_sha"] and epochs[1]["accepted"] == []
    assert any("identity epochs" in w and "2" in w for w in summary["warnings"])
    assert [e.epoch for e in eps] == [0, 1]
    assert ["epoch" in e for e in episodes_json(eps)] == [False, True]  # epoch 0 is exported as before


async def test_a_finished_run_reopened_under_new_code_has_no_epoch_block(tmp_path, monkeypatch):
    """Re-opening a finished run after a commit opens an epoch that holds no results: the report must not call itself mixed."""
    from crucible.bench.run import summarize_run

    monkeypatch.setattr(ident, "code_versions", lambda: {"git_sha": "aaa"})
    spec = RunSpec(run_id="r", tasks=[SimpleNamespace(id="t")], policies=[pol(temperature=0.7)], harnesses=[Quick("native:quick")], k=1)
    await Runner(tmp_path).run(spec)
    monkeypatch.setattr(ident, "code_versions", lambda: {"git_sha": "bbb"})
    again = await Runner(tmp_path).run(spec)
    assert again.attempts == 0
    assert [e["epoch"] for e in run_json(tmp_path)["epochs"]] == [0, 1]  # the epoch is recorded
    summary, _ = summarize_run(tmp_path / "r")
    assert "epochs" not in summary["run"]
    assert not [w for w in summary["warnings"] if "identity epochs" in w]


# ---------------------------------------------------------------- bench run --accept-drift


_MOD = '''
from types import SimpleNamespace
from crucible.runtime.harness.native import PolicyRef


def tasks():
    return [SimpleNamespace(id="t")]


def policies():
    return [PolicyRef(id="p", make=lambda a: {}, config={"script": "none"})]
'''


def test_bench_run_accept_drift_reaches_the_run_spec(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from crucible.cli.main import cli
    from crucible.cli.studio import bench_run

    assert "accept_drift" in [p.name for p in bench_run.params]
    (tmp_path / "r4mod.py").write_text(_MOD)
    monkeypatch.syspath_prepend(str(tmp_path))
    seen: list[dict] = []

    async def fake_run_bench(name, **kw):
        seen.append({"name": name, **kw})
        return {"summary": {"models": [], "warnings": []}, "paths": {"report": "r.html"}}

    monkeypatch.setattr("crucible.bench.run.run_bench", fake_run_bench)
    args = ["bench", "run", "--tasks", "r4mod:tasks", "--policies", "r4mod:policies", "--no-controls", "--k", "1", "--json"]
    res = CliRunner().invoke(cli, [*args, "--accept-drift", "policies.*", "--accept-drift", "harnesses.*.limits.max_steps", "--out", str(tmp_path / "a")],
                             catch_exceptions=False)
    assert res.exit_code == 0, res.output
    assert seen[0]["spec"] == {"accept_drift": ("policies.*", "harnesses.*.limits.max_steps")}

    res = CliRunner().invoke(cli, [*args, "--out", str(tmp_path / "b")], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    assert seen[1]["spec"] is None  # nothing accepted: the default RunSpec


def test_a_second_bench_run_is_refused_before_the_cli_writes_beside_the_first(tmp_path, monkeypatch):
    """The preflight (journal.assert_free): the provenance record, the item key and the validation episodes are all written before
    ``Runner.run`` takes the lease, so a second ``bench run`` on the same name is refused first."""
    from click.testing import CliRunner

    from crucible.cli.main import cli

    (tmp_path / "r4mod.py").write_text(_MOD)
    monkeypatch.syspath_prepend(str(tmp_path))
    out = tmp_path / "o"
    held = RunJournal(out, "bench-r4mod", "id", {})  # another process's open run of the same name
    try:
        before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.name != "run.lock")
        res = CliRunner().invoke(cli, ["bench", "run", "--tasks", "r4mod:tasks", "--policies", "r4mod:policies", "--no-controls", "--k", "1",
                                       "--out", str(out)])
        after = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.name != "run.lock")
    finally:
        held.close()
    assert res.exit_code != 0
    assert "is in use by" in res.output + str(res.exception)
    assert after == before and not (out / "configs").exists()  # nothing was written beside the first run


@pytest.mark.parametrize(("held", "controls"), [("bench-toy-bench", False), ("bench-toy-bench", True), ("bench-toy-bench-controls", True)])
def test_a_second_pack_bench_is_refused_before_it_writes_anything(tmp_path, held, controls):
    """The pack path (``_bench_pack``): the run dir is derived from the pack's name (``bench-<pack>-bench``) and, with controls, the
    ``-controls`` run too; the provenance record, the item key and the validation episodes are written before ``Runner.run``, so a
    second ``bench run toy`` is refused first (``_assert_bench_free``): nothing is written beside the first run."""
    from click.testing import CliRunner

    from crucible.cli.main import cli

    out = tmp_path / "o"
    first = RunJournal(out, held, "id", {})  # another process's open run of the same pack
    try:
        before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))
        args = ["bench", "run", "toy", "--fresh", "0", "--k", "1", "--out", str(out)]
        res = CliRunner().invoke(cli, args if controls else [*args, "--no-controls"])
        after = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))
    finally:
        first.close()
    assert res.exit_code != 0
    assert "is in use by" in res.output + str(res.exception)
    assert after == before and not (out / "configs").exists() and not (out / "bench.keys.jsonl").exists()
