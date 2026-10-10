"""Environment workers and bubble accounting (M12 checklist #8, #9): a blocking environment never stalls the engines
or the sample barrier when environments run on the ``EnvWorkerPool``; environment steps complete while the engine is
paused; World episodes run on workers with their model calls bridged to the main loop; per-step bubbles are logged."""

from __future__ import annotations

import asyncio
import time

from crucible.train.engine import EnginePool, GenRequest, TinyPolicy, TorchEngine
from crucible.train.envpool import EnvWorkerPool
from crucible.train.envs import CopyTask, Step
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import SchedulerConfig
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher


class BlockingCopy(CopyTask):
    """A synchronous (blocking) environment step: ``time.sleep``, as a pure-Python simulator or a blocking SDK does."""

    def __init__(self, block_s: float, only: str | None = None, log: list | None = None):
        super().__init__(turns=2, symbols=8)
        self.block_s, self.only, self.log, self.prompt = block_s, only, log, ""

    async def reset(self, prompt: str, seed: int):
        self.prompt = prompt
        return await super().reset(prompt, seed)

    async def step(self, completion) -> Step:
        if self.only is None or self.prompt == self.only:
            t0 = time.monotonic()
            time.sleep(self.block_s)  # noqa: ASYNC251 -- the point: an environment that blocks its thread
            if self.log is not None:
                self.log.append((t0, time.monotonic()))
        return await super().step(completion)


class CountingPool(EnginePool):
    def __init__(self, engines):
        super().__init__(engines)
        self.done_at: list[float] = []

    async def generate(self, req: GenRequest):
        r = await super().generate(req)
        self.done_at.append(time.monotonic())
        return r


def _rt(env_factory, *, workers: int, steps: int = 2, topology: str = "colocated_partial", trainer_delay: float = 0.0, prompts=None):
    model = TinyPolicy(32, 16, seed=1)
    eng = CountingPool([TorchEngine(model, capacity=64, decode_delay_s=0.001)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-3))
    if trainer_delay:
        step = tr.step

        def slow(recs, groups):
            time.sleep(trainer_delay)
            return step(recs, groups)

        tr.step = slow
    rt = Runtime(RuntimeConfig(topology=topology, steps=steps, max_turn_tokens=2, warmup_batches=0, env_workers=workers),
                 SchedulerConfig(n=2, B=2, max_staleness=2), eng, tr, InprocPublisher(tr, eng), env_factory,
                 {"copy": prompts or [f"p{i}" for i in range(400)]})
    return rt, eng


async def test_blocking_environments_run_in_parallel_on_workers():
    t0 = time.monotonic()
    await _rt(lambda: BlockingCopy(0.05), workers=0)[0].run()
    inline = time.monotonic() - t0
    t0 = time.monotonic()
    await _rt(lambda: BlockingCopy(0.05), workers=8)[0].run()
    pooled = time.monotonic() - t0
    assert pooled < 0.5 * inline, (inline, pooled)  # inline, every blocking step serializes the whole loop


async def test_engines_keep_serving_while_an_environment_blocks():
    async def window(workers: int) -> int:
        log: list = []
        prompts = ["slow"] + [f"p{i}" for i in range(400)]
        rt, eng = _rt(lambda: BlockingCopy(0.6, only="slow", log=log), workers=workers, steps=3, prompts=prompts)
        await rt.run()
        start, end = log[0]
        return sum(1 for t in eng.done_at if start < t < end)

    assert await window(0) == 0  # inline: the blocked environment froze every model call
    assert await window(4) > 5  # on workers: other rollouts' calls complete during the block


async def test_environment_steps_complete_while_the_engine_is_paused():
    pool = EnvWorkerPool(2)
    eng = TorchEngine(TinyPolicy(32, 16), capacity=4)
    done = {"env": False}

    async def env_step():
        await asyncio.sleep(0.05)
        done["env"] = True

    try:
        await eng.pause()
        gen = asyncio.create_task(eng.generate(GenRequest(rid="r", call_id="c", prompt_ids=[1, 2], max_tokens=2, temperature=1.0)))
        envt = asyncio.create_task(pool.run(env_step))
        await asyncio.sleep(0.2)
        assert done["env"] and envt.done() and not gen.done()  # the issued environment step finished; the model call waits
        await eng.resume()
        await asyncio.wait_for(gen, 5)
        assert pool.busy_s >= 0.04 and pool.submitted == 1
    finally:
        pool.close()


async def test_bubbles_are_logged_and_reflect_the_topology():
    rt, _ = _rt(lambda: CopyTask(turns=2, symbols=8, latency_s=0.002), workers=2, steps=4, trainer_delay=0.08)
    rep = await rt.run()
    for s in rep.steps:
        m = s.metrics
        assert all(k in m and m[k] >= 0 for k in ("idle/rollout_s", "idle/train_s", "idle/env_wait_s", "env/busy_s", "step/wall_s", "engine/utilization"))
        assert m["time/train_s"] >= 0.07 and m["idle/rollout_s"] >= 0.8 * m["time/train_s"]  # colocated: engines paused while the trainer trains
        assert 0.0 <= m["engine/utilization"] <= 1.0
    rd, _ = _rt(lambda: CopyTask(turns=2, symbols=8, latency_s=0.002), workers=2, steps=4, trainer_delay=0.08, topology="disaggregated_stream")
    rep_d = await rd.run()
    util = lambda r: sum(s.metrics["engine/utilization"] for s in r.steps[1:]) / len(r.steps[1:])
    assert util(rep_d) > util(rep)  # disaggregated keeps generating while the trainer trains


async def test_world_episodes_on_env_workers():
    """The rollout_fn path on workers: World episodes run on worker loops; their model calls are bridged back to the
    engine pool on the main loop (the exact-TITO oracle episode still solves the task)."""
    import json as _json

    from crucible_pack_toy.runtime import HelpdeskTask
    from crucible_pack_toy.verify import GOLD

    from crucible.core.schema.actions import openai_name
    from crucible.train.chat import ByteChatCodec
    from crucible.train.engine import ScriptedEngine
    from crucible.train.world_env import world_rollout

    codec = ByteChatCodec()

    def script(rid, k, prompt):
        if k < len(GOLD):
            a = GOLD[k]
            return list(f"<tool_call>{_json.dumps({'name': openai_name(a.function_name), 'arguments': a.arguments})}</tool_call>".encode()) + [codec.EOS]
        return list(b"done") + [codec.EOS]

    eng = EnginePool([ScriptedEngine(script, stop_ids=[codec.EOS], delay_s=0.001)])
    tr = TorchTrainer(TinyPolicy(codec.vocab, 16, seed=0), TrainerConfig(lr=1e-3))
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=2, max_turn_tokens=2048, warmup_batches=0, env_workers=3), SchedulerConfig(n=2, B=2, max_staleness=2),
                 eng, tr, InprocPublisher(tr, eng), None, {"helpdesk": [f"h{i}" for i in range(50)]},
                 rollout_fn=world_rollout(lambda p: HelpdeskTask(), codec, max_steps=len(GOLD) + 3, max_tokens=2048))
    rep = await rt.run()
    assert len(rep.steps) == 2 and rep.masked_rollouts == 0 and all(r.reward == 1.0 for r in rep.trained_records)
    assert rt.envs is not None and rt.envs.submitted >= 4 and max(w.max_inflight for w in rt.envs.workers) >= 1
