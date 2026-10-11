"""m14: the RLM harness for RL trajectories. A scripted token engine (byte codec, exact TITO) drives an RLM through the
long-horizon backlog with a context budget small enough to compact several times. The rollout must be trainable:

- every root, sub and compaction call carries a token record;
- within a segment each prompt extends the last one (stored as a delta), and every compaction starts a fresh sequence;
- sub-calls are tree nodes, and turns record their World step.

The compacted turns stay reachable in the kernel's ``HISTORY`` (the model reads them back after the summary replaced them).
The harness id is the config's digest, and the kernel journal is saved as evidence. The same RLM runs as an M12
``rollout_fn``. An episode stopped by a step limit closes its kernel."""

from __future__ import annotations

import json

from crucible_pack_toy.longhorizon.task import BacklogTask

from crucible.train.chat import ByteChatCodec
from crucible.train.engine import GenResult

CODEC = ByteChatCodec()


def scripted_engine(task: BacklogTask, chunk: int = 6):
    acts = task.oracle_actions()
    cells = [f"ACTS = {acts!r}\nprint(len(ACTS))", "s = await rlm.query('rlm-sub: what kind of ticket is this?', context=str(ACTS[0]))\nprint(s)"]
    for i in range(0, len(acts), chunk):
        cells.append(f"for a in ACTS[{i}:{i + chunk}]:\n"
                     "    r = await (world.call(a['function_name'], **a['arguments']) if a['kind'] == 'tool_call' else world.send(a['to'], a['text']))\n"
                     f"print('chunk {i}', 'x' * 1200)")
    cells += ["print('HIST', len(HISTORY), HISTORY[0]['turns'][0]['code'][:7])", "rlm.finish('backlog done')"]
    seen = {"root": 0, "sub": 0, "compact": 0, "last_prompt": ""}

    def body(text: str) -> list[int]:
        return list(text.encode()) + [CODEC.EOS]

    async def gen(prompt: list[int], max_tokens: int, temperature: float) -> GenResult:
        text = bytes(t for t in prompt if t < 256).decode(errors="replace")
        if "Your context is nearly full" in text:
            seen["compact"] += 1
            out = body(f"Summary {seen['compact']}: ACTS holds the plan; chunks so far are done.")
        elif prompt[:5] == [CODEC.BOS, *b"user"]:  # a sub-LLM call: a single user message, no system prompt or tools
            seen["sub"] += 1
            out = body("a refund ticket")
        else:
            seen["last_prompt"] = text
            code = cells[min(seen["root"], len(cells) - 1)]
            seen["root"] += 1
            out = body("<tool_call>" + json.dumps({"name": "python", "arguments": {"code": code}}) + "</tool_call>")
        return GenResult(completion_ids=out, logprobs=[-0.1] * len(out), version=3, finish="stop")

    return gen, seen


def _config():
    from crucible.runtime.harness.rlm import RLMConfig

    return RLMConfig(max_tokens=16_000, temperature=1.0, budget_tokens=2_500, trigger=0.8, keep_recent_turns=1, summary_max_tokens=200, kernel="local")


async def test_rlm_harness_long_horizon_train_rollout(tmp_path):
    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.harness.rlm import RLMConfig, RLMHarness
    from crucible.runtime.run import Runner, RunSpec
    from crucible.train.rollout import RolloutRecord
    from crucible.train.world_env import EngineLLM

    task = BacklogTask(seed=0, n=12)
    gen, seen = scripted_engine(task)
    h = RLMHarness(_config())
    assert h.id.startswith("rlm:") and h.id != RLMHarness(RLMConfig()).id  # the config is part of the harness id (one per GRPO group)
    try:
        ref = PolicyRef(id="engine:scripted", make=lambda a: {"agent": EngineLLM(CODEC, gen)})  # the harness wraps the model in the RLM
        await Runner(tmp_path).run(RunSpec(run_id="r", tasks=[task], policies=[ref], harnesses=[h], k=1, rollout_mode="train"))
    finally:
        task.cleanup()
    adir = min((tmp_path / "r" / "samples").glob("*/a*"))
    ro = RolloutRecord.model_validate_json((adir / "rollout.json").read_text())
    assert not ro.mask_sample and ro.reward == 1.0, (ro.failure_kind, ro.stop)
    assert ro.valid_for_training()[0] and ro.harness == h.id
    kinds = [t.kind for t in ro.turns]
    assert kinds.count("compaction") >= 2 and kinds.count("sub") == 1 and seen["compact"] == kinds.count("compaction"), kinds
    assert len(ro.segments) == kinds.count("compaction") and max(t.segment for t in ro.turns) == len(ro.segments)
    for prev, t in zip(ro.turns, ro.turns[1:]):
        if t.kind != "sub" and prev.kind != "sub" and t.segment == prev.segment:
            assert t.prompt_is_delta and not t.prefix_break, t.call_id  # one exact-prefix sequence per segment
    assert all(t.policy_version == 3 and t.logprobs for t in ro.turns)
    assert [t.world_step for t in ro.turns if t.kind == "turn"][-1] > 0
    # the compacted turns stayed in the kernel: the model read the first archived cell back after two summaries
    assert "HIST " in seen["last_prompt"] and "ACTS = " in seen["last_prompt"].split("HIST ")[-1][:40]
    doc = json.loads((adir / "rlm.agent.json").read_text())
    assert doc["harness"] == h.id and doc["counters"]["sub_calls"] == 1 and doc["stop"] == "agent_done"
    assert sum(e["code"].startswith("_archive(") for e in doc["kernel"]["journal"]) == len(ro.segments)


async def test_rlm_world_rollout_for_m12(tmp_path):
    from crucible.train.world_env import rlm_world_rollout

    task = BacklogTask(seed=1, n=12)
    gen, _ = scripted_engine(task)
    try:
        ro = await rlm_world_rollout(lambda p: task, CODEC, _config(), workroot=tmp_path)("rid-1", "p", 0, gen, temperature=1.0)
    finally:
        task.cleanup()
    assert ro is not None and ro.reward == 1.0 and ro.valid_for_training()[0], ro.stop
    assert ro.harness == f"rlm:{_config().digest()}@1" and any(t.kind == "compaction" for t in ro.turns)


async def test_step_limit_closes_the_kernel(tmp_path):
    from crucible.actors.policy import EpisodeLimits
    from crucible.actors.rlm.policy import RLMPolicy
    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.harness.rlm import RLMHarness
    from crucible.runtime.run import Runner, RunSpec
    from crucible.train.world_env import EngineLLM

    task = BacklogTask(seed=2, n=12)
    gen, _ = scripted_engine(task)
    made = []

    def make(a):
        made.append(RLMPolicy("agent", EngineLLM(CODEC, gen), max_tokens=16_000))
        return {"agent": made[-1]}

    try:
        await Runner(tmp_path).run(RunSpec(run_id="r", tasks=[task], policies=[PolicyRef(id="rlm", make=make)],
                                           harnesses=[RLMHarness(_config(), limits=EpisodeLimits(max_steps=3))], k=1))
    finally:
        task.cleanup()
    p = made[0]
    assert p.kernel is not None and (p.kernel.proc is None or p.kernel.proc.returncode is not None)
    assert p._task is not None and p._task.done()


async def test_tool_index_and_docker_sandboxed_kernel(tmp_path):
    """The default kernel is a container with no network, a read-only root and an unprivileged user (the m14 live model
    probed the host's working directory). The task message carries a one-line tool index. The container is removed at
    the end of the episode."""
    import asyncio
    import shutil
    import subprocess

    import pytest

    def sh(*argv):
        return asyncio.to_thread(subprocess.run, list(argv), capture_output=True, text=True, check=False)

    if not shutil.which("docker") or (await sh("docker", "info")).returncode != 0:
        pytest.skip("docker is not available")
    from dataclasses import replace

    from crucible.actors.policy import EpisodeLimits
    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.harness.rlm import RLMHarness
    from crucible.runtime.run import Runner, RunSpec
    from crucible.train.world_env import EngineLLM

    task = BacklogTask(seed=3, n=12)
    probe = ("import os, socket\ntry:\n    socket.create_connection(('1.1.1.1', 80), timeout=2); net = 'open'\nexcept OSError:\n    net = 'blocked'\n"
             "try:\n    open('/etc/x', 'w'); ro = 'writable'\nexcept OSError:\n    ro = 'read-only'\n"
             "r = await world.call(sorted(TOOLS)[0])\nprint('PROBE', net, ro, os.getuid(), os.getcwd(), sorted(os.listdir('.')), r['ok'] is not None)")
    seen: dict = {}

    async def gen(prompt, max_tokens, temperature):
        text = bytes(t for t in prompt if t < 256).decode(errors="replace")
        seen.setdefault("first", text)
        code = probe if "PROBE" not in text else "rlm.finish('probed')"
        seen["last"] = text
        out = list(("<tool_call>" + json.dumps({"name": "python", "arguments": {"code": code}}) + "</tool_call>").encode()) + [CODEC.EOS]
        return GenResult(completion_ids=out, logprobs=[-0.1] * len(out), version=0, finish="stop")

    h = RLMHarness(replace(_config(), kernel="docker"), limits=EpisodeLimits(max_steps=20))
    try:
        await Runner(tmp_path).run(RunSpec(run_id="r", tasks=[task], policies=[PolicyRef(id="e", make=lambda a: {"agent": EngineLLM(CODEC, gen)})],
                                           harnesses=[h], k=1))
    finally:
        task.cleanup()
    assert "[tools] Everything in this workplace" in seen["first"] and "- tickets." in seen["first"]
    line = seen["last"].split("PROBE ")[-1].splitlines()[0]
    assert line.startswith("blocked read-only 65534 /tmp ['crucible_rlm_repl.py'] True"), line  # no network, read-only root, nobody, nothing of the host
    left = (await sh("docker", "ps", "-aq", "--filter", "label=crucible.rlm=1")).stdout.split()
    assert not left, left
