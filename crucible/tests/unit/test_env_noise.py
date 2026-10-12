"""M12-infra D: the flaky-task admission filter (``train/admission.py``), World noise at the tool and instruction choke
points (``world/noise.py``), and the noise curriculum (``train/noise.py``) alone and inside the runtime."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.train.admission import screen
from crucible.train.noise import NoiseCurriculum, TokenNoise
from crucible.world.noise import NoiseConfig, WorldNoise
from crucible.world.world import World


# ---- flaky-task admission -----------------------------------------------------------------------------------------
async def test_screen_keeps_only_stable_tasks():
    flips = random.Random(0)
    calls: dict[str, int] = {}

    def math(src, prompt, i):
        calls[prompt] = calls.get(prompt, 0) + 1
        return {"ok": True, "never": False, "flaky": flips.random() < 0.7, "crash": None}.get(prompt.split("-")[0], True)

    async def code(src, prompt, i):  # full verdicts (reward, gates), as A4 compares them
        if prompt == "gate-flip":
            return (1.0, ("build", "pass"), ("perf", "pass" if i % 3 else "fail"))
        if prompt == "boom":
            raise RuntimeError("sandbox lost")
        return (1.0, ("build", "pass"), ("perf", "pass"))

    rep = await screen({"math": ["ok-1", "never-1", "flaky-1", "crash-1"], "code": ["steady", "gate-flip", "boom"]}, {"math": math, "*": code}, k=8)
    v = {(s.source, s.prompt): s.verdict for s in rep.screens}
    assert v == {("math", "ok-1"): "stable", ("math", "never-1"): "broken", ("math", "flaky-1"): "flaky", ("math", "crash-1"): "infra",
                 ("code", "steady"): "stable", ("code", "gate-flip"): "flaky", ("code", "boom"): "infra"}
    assert rep.kept == {"math": ["ok-1"], "code": ["steady"]}
    assert calls["ok-1"] == 8 and calls["flaky-1"] < 8 and calls["crash-1"] == 2  # early stop: flaky at the first disagreement; infra past max_infra=1
    assert rep.counts()["math"] == {"stable": 1, "flaky": 1, "broken": 1, "infra": 1}
    assert rep.flake_upper_95() == pytest.approx(1 - 0.05 ** (1 / 8))  # 0 disagreements in 8 runs
    full = await screen({"math": ["flaky-2"]}, math, k=8, early_stop=False)
    assert full.screens[0].runs == 8 and full.screens[0].verdict == "flaky"
    with pytest.raises(ValueError):
        await screen({"a": ["x"]}, math, k=1)


# ---- World noise --------------------------------------------------------------------------------------------------
def _spec() -> WorldSpec:
    return WorldSpec.model_validate({
        "world": "test/noise",
        "apps": [{"id": "kb", "template": "common/doc_library", "state": {"adapter": "fs", "params": {"root": "kb"}}, "params": {"seed_dir": "kb"}},
                 {"id": "mail", "template": "common/mailbox", "state": {"adapter": "sql", "params": {"path": "mail.db"}}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["kb.*", "mail.*"]}],
    })


INSTR = "Read the refund policy document carefully and then email the supervisor a short summary of the approval threshold for refunds."


async def _world(tmp_path: Path, noise) -> World:
    root = tmp_path / "pack"
    (root / "kb").mkdir(parents=True, exist_ok=True)
    (root / "kb" / "policy.md").write_text("\n".join(f"Line {i}: refunds above {i * 10} USD need supervisor approval before payment." for i in range(12)))
    w = World(_spec(), instructions={"agent": INSTR}, store=Store(tmp_path / "store"), pack_root=root, workdir=tmp_path / f"w{id(noise)}", noise=noise)
    await w.start()
    return w


async def test_world_noise_perturbs_the_view_not_the_state(tmp_path):
    clean = await _world(tmp_path, None)
    o = (await clean.reset())["agent"]
    r0 = (await clean.step("agent", ToolCall(tool_call_id="1", function_name="kb.read_doc", arguments={"path": "policy.md"}))).observation.tool_results[0]
    assert o.instruction == INSTR and not [e for e in clean.log if e.type == "noise"]
    off = await _world(tmp_path, WorldNoise(NoiseConfig(intensity=0.0), seed="x"))  # intensity 0: identical to no noise
    assert (await off.reset())["agent"].instruction == INSTR

    cfg = NoiseConfig(intensity=1.0, tool_before={"transient_error": 1.0}, max_transient_per_tool=1,
                      tool_after={"shuffle_lines": 1.0, "truncate": 1.0, "typo": 0.0, "empty": 0.0, "drop_json_field": 0.0})
    w = await _world(tmp_path, WorldNoise(cfg, seed="ep1"))
    o = (await w.reset())["agent"]
    assert o.instruction != INSTR and w.instructions["agent"] == INSTR  # the stored instruction stays clean
    r1 = (await w.step("agent", ToolCall(tool_call_id="2", function_name="kb.read_doc", arguments={"path": "policy.md"}))).observation.tool_results[0]
    assert not r1.ok and r1.error == "transient_unavailable" and r1.fault == "tool"  # retryable, scored (not masked)
    r2 = (await w.step("agent", ToolCall(tool_call_id="3", function_name="kb.read_doc", arguments={"path": "policy.md"}))).observation.tool_results[0]
    assert r2.ok and r2.content != r0.content and "truncated by the service" in r2.content  # the retry runs; its result is perturbed
    sent = (await w.step("agent", ToolCall(tool_call_id="4", function_name="mail.send_email",
                                           arguments={"to": ["sup@world.local"], "subject": "s", "body": "b"}))).observation.tool_results[0]
    assert not sent.ok  # first call to a new tool: transient
    sent = (await w.step("agent", ToolCall(tool_call_id="5", function_name="mail.send_email",
                                           arguments={"to": ["sup@world.local"], "subject": "s", "body": "b"}))).observation.tool_results[0]
    assert sent.ok and not w.failures  # tool noise never masks the episode
    ops = [op for e in w.log if e.type == "noise" for op in e.data["ops"]]
    assert ops.count("tool_transient_error") == 2 and "tool_shuffle_lines" in ops and any(op.startswith("instr_") for op in ops)
    again = await _world(tmp_path, WorldNoise(cfg, seed="ep1"))  # the same seed: the same perturbation
    assert (await again.reset())["agent"].instruction == o.instruction
    with pytest.raises(ValueError):
        WorldNoise(NoiseConfig(intensity=1.5))


# ---- curriculum ---------------------------------------------------------------------------------------------------
def test_curriculum_raises_when_robust_and_backs_off_when_fragile():
    cur = NoiseCurriculum(window=32, step=0.1, clean_frac=0.25, seed=1)
    keys = [f"k{i}" for i in range(4000)]
    assert abs(sum(cur.is_clean(k) for k in keys) / len(keys) - 0.25) < 0.03  # the control group share
    rng = random.Random(0)
    for k in keys:  # a robust policy: noise does not hurt
        lvl = cur.assign(k)
        cur.observe(float(rng.random() < 0.6), clean=cur.is_clean(k), level=lvl)
    assert cur.intensity == pytest.approx(1.0) and cur.history[0]["previous"] == 0.0

    frag = NoiseCurriculum(window=32, step=0.1, seed=2, raise_below=0.05, lower_above=0.2)
    for i in range(20000):  # success falls with the noise level: the curriculum settles where the gap sits between the thresholds
        k = f"f{i}"
        lvl = frag.assign(k)
        frag.observe(float(rng.random() < 0.8 * (1 - 0.8 * lvl)), clean=frag.is_clean(k), level=lvl)
    tail = [h["intensity"] for h in frag.history[-20:]]
    assert 0.0 < min(tail) and max(tail) < 0.6  # held off the maximum: the noise costs more than the policy can absorb
    assert any(h["intensity"] < h["previous"] for h in frag.history)  # it backed off at least once


async def test_curriculum_in_the_runtime():
    from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
    from crucible.train.envs import CopyTask
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.trainer import TorchTrainer, TrainerConfig
    from crucible.train.weights import InprocPublisher

    model = TinyPolicy(32, 48, seed=1)
    pool = EnginePool([TorchEngine(model, capacity=32, numerics="bf16", decode_delay_s=0.0005)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2))
    cur = NoiseCurriculum(window=8, step=0.25, clean_frac=0.3, raise_below=0.5, start=0.25)  # a loose threshold: it must move
    factory = cur.token_env(lambda: CopyTask(turns=2, symbols=8), TokenNoise(insert=0.5, protect=frozenset({1, 2, 3})))
    rt = Runtime(RuntimeConfig(topology="disaggregated_stream", steps=10, max_turn_tokens=3, sinks=[cur]), SchedulerConfig(n=4, B=4, max_staleness=2),
                 pool, tr, InprocPublisher(tr, pool), factory, {"copy": [f"p{i}" for i in range(4000)]})
    rep = await rt.run()
    comps = [r.reward_components for r in rep.trained_records]
    assert all("noise/intensity" in c and "noise/clean" in c for c in comps)
    assert {c["noise/clean"] for c in comps} == {0.0, 1.0}  # both arms trained
    assert any(c["noise/ops"] > 0 for c in comps if c["noise/clean"] == 0.0) and all(c["noise/ops"] == 0 for c in comps if c["noise/clean"] == 1.0)
    assert cur.history and cur.intensity != 0.25  # the sink saw the trained records and moved the knob
