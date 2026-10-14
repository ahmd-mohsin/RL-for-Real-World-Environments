"""M14 audit regression tests for the RLM agent (docs/audit/2026-10-13/M14-study-defects-H3.md V1-V13 and
M14-sweep-G1.md D1-D15). Each test fails on f2d9e18 and passes after the fix. Credential canaries are fake values."""

from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from crucible.actors.llm import LLMError, LLMResponse
from crucible.actors.rlm import kernel as K
from crucible.actors.rlm.kernel import Kernel
from crucible.runtime.inject import FakeLLM
from crucible.train.chat import ByteChatCodec
from crucible.train.engine import GenResult

CODEC = ByteChatCodec()
REPO = str(Path(__file__).resolve().parents[3])


def host_launcher():
    """The host launcher, opted into explicitly (on f2d9e18 it had no opt-in)."""
    try:
        return K.LocalLauncher(allow_host=True)
    except TypeError:
        return K.LocalLauncher()


def rlm_config(**kw):
    from crucible.runtime.harness.rlm import RLMConfig

    base = {"max_tokens": 16_000, "temperature": 1.0, "budget_tokens": 0, "kernel": "local", "cell_timeout_s": 10, "allow_host": True}
    base.update(kw)
    try:
        return RLMConfig(**base)
    except TypeError:
        base.pop("allow_host")
        return RLMConfig(**base)


def code(src: str, i: int = 0, **kw) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[{"id": f"py{i}", "name": "python", "arguments": {"code": src}}], usage={"completion_tokens": 10}, **kw)


def engine(cells, *, sub_reply="sub answer", compact_reply="Summary: state is in variables.", think=False):
    seen = {"root": 0, "sub": 0, "compact": 0}

    def body(text: str) -> list[int]:
        return list(text.encode()) + [CODEC.EOS]

    async def gen(prompt, max_tokens, temperature):
        text = bytes(t for t in prompt if t < 256).decode(errors="replace")
        if "Your context is nearly full" in text:
            seen["compact"] += 1
            out = body(("<think>summarizing</think>" if think else "") + compact_reply)
        elif prompt[:5] == [CODEC.BOS, *b"user"]:
            seen["sub"] += 1
            out = body(sub_reply)
        else:
            c = cells[min(seen["root"], len(cells) - 1)]
            seen["root"] += 1
            out = body(("<think>plan</think>" if think else "") + "<tool_call>" + json.dumps({"name": "python", "arguments": {"code": c}}) + "</tool_call>")
        return GenResult(completion_ids=out, logprobs=[-0.1] * len(out), version=3, finish="stop")

    return gen, seen


async def sh(*argv: str) -> str:
    r = await asyncio.to_thread(subprocess.run, list(argv), capture_output=True, text=True, check=False)
    return r.stdout


async def echo(d):
    return {"echo": d}


def helpdesk_episode_parts():
    from crucible_pack_toy.runtime import HelpdeskTask

    return HelpdeskTask()


async def run_rlm(tmp_path, script, *, sub=None, **kw):
    from crucible.actors.policy import EpisodeLimits, run_episode
    from crucible.actors.rlm.policy import RLMPolicy

    task = helpdesk_episode_parts()
    w = await task.open(tmp_path / "work")
    kw.setdefault("launcher", host_launcher())
    p = RLMPolicy("agent", FakeLLM(script), sub_llm=sub, **kw)
    try:
        ep = await run_episode(w, {"agent": p}, limits=EpisodeLimits(max_steps=60))
        return ep, p, w
    finally:
        await w.close()
        await p.aclose()


# ---- V1 / D8: model code never runs on the host by default ----------------------------------------------------------
def test_V1_default_launcher_is_isolated_or_refused(monkeypatch):
    from crucible.accept.repair import RLMAgent
    from crucible.actors.rlm.policy import RLMPolicy

    monkeypatch.setattr("shutil.which", lambda name, *a, **k: "/usr/bin/docker" if name == "docker" else None)
    assert type(Kernel().launcher).__name__ == "DockerLauncher"
    assert type(RLMPolicy("a", FakeLLM([])).launcher).__name__ == "DockerLauncher"
    monkeypatch.setattr("shutil.which", lambda *a, **k: None)
    with pytest.raises(K.KernelError):
        Kernel()
    with pytest.raises(K.KernelError):
        RLMPolicy("a", FakeLLM([]))

    async def run_agent():
        return await RLMAgent("m", FakeLLM([code("rlm.finish('x')")])).run("t")

    with pytest.raises(K.KernelError):
        asyncio.run(run_agent())


async def test_V1_local_launcher_refuses_without_opt_in():
    with pytest.raises(K.KernelError, match="allow_host"):
        await K.LocalLauncher().start(0)
    with pytest.raises(ValueError, match="allow_host"):
        rlm_config(allow_host=False).launcher()


async def test_D8_host_kernel_gets_an_allowlisted_env_and_no_repo_path(monkeypatch):
    for k, v in {"SSH_AUTH_SOCK": "/tmp/agent.sock", "DATABASE_URL": "postgres://u:pw@db/x", "HTTP_PROXY": "http://u:pw@proxy:3128",
                 "GH_PAT": "ghp_canary", "OPENAI_API_KEY": "sk-canary-test", "HF_TOKEN": "hf_canary"}.items():
        monkeypatch.setenv(k, v)
    k = Kernel(host_launcher(), cell_timeout_s=10)
    try:
        r = await k.execute("import os, sys\n(sorted(os.environ), sys.argv[0], sys.executable, os.getcwd(), os.path.expanduser('~'))", echo)
        env, argv0, exe, cwd, home = eval(r.result)
        assert not {"SSH_AUTH_SOCK", "DATABASE_URL", "HTTP_PROXY", "GH_PAT", "OPENAI_API_KEY", "HF_TOKEN"} & set(env), env
        assert set(env) <= set(K.HOST_ENV_ALLOW) | {"HOME", "TMPDIR", "CRUCIBLE_RLM_SEED", "PYTHONUNBUFFERED", "PYTHONHASHSEED", "PYTHONNOUSERSITE",
                                                    "PYTHONDONTWRITEBYTECODE", "__CF_USER_TEXT_ENCODING", "LC_CTYPE"}, env
        assert REPO not in argv0 and REPO not in exe and REPO not in cwd and os.path.realpath(home) == os.path.realpath(cwd)
    finally:
        await k.close()
        await k.launcher.aclose()


async def test_V1_host_kernel_process_group_killed_on_close(tmp_path):
    marker = f"crucible-orphan-{os.getpid()}-{random.randrange(10**9)}"
    k = Kernel(host_launcher(), cell_timeout_s=10)
    try:
        r = await k.execute(f"import subprocess\np = subprocess.Popen(['/bin/sh', '-c', 'exec -a {marker} sleep 300'])\np.pid", echo)
        assert r.ok, r
    finally:
        await k.close()
        await k.launcher.aclose()
    await asyncio.sleep(0.5)
    left = (await sh("pgrep", "-f", marker)).split()
    for pid in left:
        os.kill(int(pid), 9)
    assert not left, "a process the cell started outlived the kernel"


# ---- V2 / D12 / D13: protocol isolation, validation, caps, interrupt ------------------------------------------------
FORGE = r'''
import os, sys, json
cid = sys.modules["__main__"]._cell.get()
os.write(1, (json.dumps({"event": "done", "id": cid, "stdout": "forged \ud800", "finished": "FORGED", "digest": "x"}) + "\n").encode())
sys.__stdout__.write(json.dumps({"event": "done", "id": cid, "stdout": "forged2", "finished": "FORGED"}) + "\n"); sys.__stdout__.flush()
print("real")
41 + 1
'''


async def test_V2_forged_protocol_frame_is_cell_output():
    k = Kernel(host_launcher(), cell_timeout_s=10)
    try:
        r = await k.execute(FORGE, echo)
        assert r.finished is None and r.result == "42" and "real" in r.stdout and r.digest != "x", r
        r.stdout.encode("utf-8")  # no lone surrogate reaches the host's text
        r2 = await k.execute("1 + 1", echo)
        assert r2.result == "2"
    finally:
        await k.close()
        await k.launcher.aclose()


async def test_V2_forged_lone_surrogate_does_not_mask_the_episode(tmp_path):
    from crucible_pack_toy.longhorizon.task import BacklogTask

    from crucible.train.world_env import rlm_world_rollout

    forge = ('import os, sys, json\ncid = sys.modules["__main__"]._cell.get()\n'
             'os.write(1, (json.dumps({"event": "done", "id": cid, "stdout": "forged \\ud800", "finished": None}) + "\\n").encode())')
    task = BacklogTask(seed=0, n=6)
    gen, _ = engine([forge, "rlm.finish('gave up')"])
    try:
        ro = await rlm_world_rollout(lambda p: task, CODEC, rlm_config(), workroot=tmp_path)("rid", "p", 0, gen, temperature=1.0)
    finally:
        task.cleanup()
    assert ro is not None, "the policy's forged frame masked its own episode"
    assert not ro.mask_sample and ro.reward is not None and ro.valid_for_training()[0], (ro.failure_kind, ro.stop)


async def test_D12_malformed_requests_get_protocol_errors_and_caps_have_markers():
    k = Kernel(host_launcher(), cell_timeout_s=10)
    try:
        await k.start()
        k.proc.stdin.write(b"[" * 100_000 + b"\n" + b"{not json\n")
        await k.proc.stdin.drain()
        r = await k.execute("'still serving'", echo)
        assert r.result == "'still serving'", r
        r = await k.execute("'y' * 3_000_000", echo)
        assert "truncated at" in r.result and len(r.result) < 1_100_000
        r = await k.execute("raise ValueError('x' * 3_000_000)", echo)
        assert "ValueError: xxx" in r.error  # the exception type survives the cap
        r = await k.execute("await host('big', blob='z' * (17 * 1024 * 1024))", echo)
        assert r.error and "cap" in r.error and r.host_calls == 0
        r = await k.execute("import os\nos.close(1)\nos.close(2)\n'closed'", echo)
        assert r.result == "'closed'" and not r.killed  # stdio abuse no longer kills the protocol
    finally:
        await k.close()
        await k.launcher.aclose()


async def test_D13_interrupt_keeps_the_namespace():
    k = Kernel(host_launcher(), cell_timeout_s=1, kill_grace_s=3, interrupt_grace_s=0.5)
    try:
        await k.execute("keep = 41", echo)
        r = await k.execute("import time\nwhile True:\n    time.sleep(0.05)", echo)
        assert r.timed_out and r.interrupted and not r.killed, r
        r = await k.execute("keep + 1", echo)
        assert r.result == "42"  # the same kernel: state kept, no kill + replay
    finally:
        await k.close()
        await k.launcher.aclose()


async def test_D13_deadline_excludes_host_time():
    async def slow(d):
        await asyncio.sleep(1.2)
        return "ok"

    k = Kernel(host_launcher(), cell_timeout_s=1)
    try:
        r = await k.execute("a = await host('w')\nb = await host('w')\n(a, b)", slow)
        assert r.ok and r.result == "('ok', 'ok')", r
    finally:
        await k.close()
        await k.launcher.aclose()
    from crucible.actors.rlm.policy import RLMPolicy

    p = RLMPolicy("a", FakeLLM([]), launcher=host_launcher(), cell_timeout_s=7)
    assert "7 s deadline" in p.system  # the model is told its deadline (RLM blog)


# ---- V3 / D7: kill and close stop the kernel inside the container ----------------------------------------------------
COUNT = ("import os\nn = 0\nfor p in os.listdir('/proc'):\n    if p.isdigit() and int(p) != os.getpid():\n        try:\n"
         "            n += 'crucible_rlm_repl' in open(f'/proc/{p}/cmdline').read()\n        except OSError:\n            pass\nprint(n)")


@pytest.mark.docker
async def test_V3_wedged_and_closed_kernels_do_not_survive_in_the_container():
    launcher = K.DockerLauncher()
    k = Kernel(launcher, cell_timeout_s=1, kill_grace_s=2)
    try:
        await k.execute("x = 1", echo)
        wedge = "import time\nwhile True:\n    try:\n        time.sleep(0.05)\n    except BaseException:\n        pass"
        r = await k.execute(wedge, echo)
        assert r.killed
        await k.replay(k.journal[:-2] if len(k.journal) > 1 else [])
        out = (await sh("docker", "exec", launcher.name, "python3", "-c", COUNT)).strip()
        assert out == "1", f"kernel processes in the container: {out}"
        r = await k.execute("import threading, time\ndef w():\n    n = 0\n    while True:\n        n += 1\n"
                            "        open('/tmp/still', 'w').write(str(n))\n        time.sleep(0.1)\nthreading.Thread(target=w).start()", echo)
        assert r.ok, r
        await k.close()
        await asyncio.sleep(0.5)
        a = (await sh("docker", "exec", launcher.name, "cat", "/tmp/still"))
        await asyncio.sleep(1.0)
        b = (await sh("docker", "exec", launcher.name, "cat", "/tmp/still"))
        assert a == b, "a thread the cell started kept writing after close()"
        out = (await sh("docker", "exec", launcher.name, "python3", "-c", COUNT)).strip()
        assert out == "0"
    finally:
        await k.close()
        await launcher.aclose()


# ---- V4 / D10: sub-call budget ----------------------------------------------------------------------------------------
class CountingLLM:
    model = "policy"

    def __init__(self, delay: float = 0.0):
        self.calls, self.max_tokens, self.live, self.peak, self.delay = 0, [], 0, 0, delay

    async def complete(self, messages, *, tools=None, max_tokens=512, temperature=0.0, **kw):
        self.calls += 1
        self.max_tokens.append(max_tokens)
        self.live += 1
        self.peak = max(self.peak, self.live)
        await asyncio.sleep(self.delay)
        self.live -= 1
        return LLMResponse(text="ok", usage={"prompt_tokens": 3, "completion_tokens": 1})


async def test_V4_sub_calls_are_budgeted_and_clamped(tmp_path):
    sub = CountingLLM(delay=0.05)
    script = [code("try:\n    await rlm.batch(['x'] * 300, max_tokens=10**6)\nexcept Exception as e:\n    print('refused', e)", 1),
              code("await rlm.batch(['q'] * 10)\nawait rlm.query('big', max_tokens=10**9)", 2), code("rlm.finish('ok')", 3)]
    _ep, p, _w = await run_rlm(tmp_path, script, sub=sub, sub_max_tokens=1000)
    assert sub.calls <= 64 and all(mt <= 1000 for mt in sub.max_tokens), (sub.calls, max(sub.max_tokens))
    assert sub.peak <= 4, sub.peak
    tool_msgs = [m["content"] for m in p.messages if m["role"] == "tool"]
    assert "refused" in tool_msgs[0]


# ---- V5 / D2: a sub-call provider fault masks -------------------------------------------------------------------------
async def test_V5_sub_call_provider_fault_masks_with_a_record(tmp_path):
    sub = FakeLLM([], fail_at=(0, LLMError("provider 503")))
    script = [code("s = await rlm.query('summarize')\nprint(s)", 1), code("rlm.finish('x')", 2)]
    ep, p, _w = await run_rlm(tmp_path, script, sub=sub)
    assert ep.masked and any(f["kind"] == "provider_unavailable" for f in ep.failures), ep.failures
    assert any(c["kind"] == "sub" and c.get("failed") for c in p.model_calls)


# ---- V6 / D6: sandbox and kernel start faults mask ----------------------------------------------------------------------
class DeadDaemon:
    async def start(self, seed):
        raise RuntimeError("docker run failed: Cannot connect to the Docker daemon at unix:///var/run/docker.sock")


async def test_V6_kernel_start_fault_is_masked_infra(tmp_path):
    ep, _p, _w = await run_rlm(tmp_path, [code("rlm.finish('x')", 1)], launcher=DeadDaemon())
    assert ep.masked, ep.failures
    assert all(f["kind"] != "crucible:policy_error" for f in ep.failures), ep.failures
    assert any(f["kind"] == "crucible:env_container_lost" for f in ep.failures), ep.failures


# ---- V7 / D11: the A/B gate -------------------------------------------------------------------------------------------
async def test_V7_harness_ab_gate_needs_paired_complete_evidence():
    from crucible.actors.rlm.harness_state import HarnessError, HarnessState

    edit = [{"action": "create", "kind": "note", "id": "n1", "title": "t", "content": "c"}]

    async def guard_vanishes(state, seed):
        return {"primary": 1.0 if "n1" in state else 0.5, **({} if "n1" in state else {"g": 1.0})}

    ev = await HarnessState().evaluate(edit, guard_vanishes, seeds=[0, 1, 2], primary="primary", guards=["g"])
    assert not ev["accepted"]  # (a) a guard the candidate stops reporting is a regression

    async def unpaired(state, seed):
        if "n1" in state:
            return {"primary": 1.0 if seed == 0 else float("nan")}
        return {"primary": 0.0 if seed == 0 else 1.0}

    ev = await HarnessState().evaluate(edit, unpaired, seeds=[0, 1, 2], primary="primary")
    assert not ev["accepted"]  # (b) NaN runs are not dropped per arm

    async def deterministic_gain(state, seed):
        return {"primary": 0.6 if "n1" in state else 0.5}

    ev = await HarnessState().evaluate(edit, deterministic_gain, seeds=[0], primary="primary")
    assert not ev["accepted"] and any("seed" in r for r in ev["reasons"])  # one seed is not evidence
    order: list[tuple[str, int]] = []

    async def logged(state, seed):
        order.append(("cand" if "n1" in state else "base", seed))
        return {"primary": 0.6 if "n1" in state else 0.5}

    ev = await HarnessState().evaluate(edit, logged, seeds=[0, 1, 2], primary="primary")
    assert ev["accepted"] and order[:4] == [("base", 0), ("cand", 0), ("cand", 1), ("base", 1)]  # interleaved, alternating
    rng = random.Random(7)

    async def noise(state, seed):  # (c) the edit has no effect: fn is noise
        return {"primary": rng.random()}

    kept = 0
    for _ in range(60):
        kept += (await HarnessState().evaluate(edit, noise, seeds=[0, 1, 2], primary="primary"))["accepted"]
    assert kept <= 6, kept
    h = HarnessState()
    h.apply_unmeasured([{"action": "create", "kind": "skill", "id": "s1", "title": "s", "content": "x = 1"}], reason="seed")
    with pytest.raises(HarnessError):  # a note-kind update of a skill skipped the compile check and stayed a skill
        h.propose([{"action": "update", "kind": "note", "id": "s1", "content": "this is not python((("}])


# ---- V8: commit needs a full verification at the current version --------------------------------------------------------
async def test_V8_commit_refuses_a_subset_verification(monkeypatch, tmp_path):
    import crucible.accept.run as accept_run
    from crucible.accept.repair import ForgeWorkspace

    async def fake_accept(cand, *, policy, cache_root, only, reuse):
        res = {k: SimpleNamespace(status="fail" if k == "A16" else "pass", reasons=["hack found"] if k == "A16" else [], measured={})
               for k in only}
        return SimpleNamespace(results=res)

    monkeypatch.setattr(accept_run, "accept_task", fake_accept)
    t = SimpleNamespace(cs=SimpleNamespace(checks=[]), spec=SimpleNamespace(title="t"), instruction="i", oracle=[])
    ws = ForgeWorkspace({"t2": t}, rebuild=lambda task, checks: task, checks=("A1", "A2", "A16"), cache_root=tmp_path)
    assert not (await ws.verify("t2", 0))["passed"]
    assert (await ws.verify("t2", 0, checks=["A2"]))["passed"]
    with pytest.raises(PermissionError):
        await ws.commit("t2", 0)


# ---- V9 / D3: compaction under a re-rendering template ----------------------------------------------------------------
class Interleaved(ByteChatCodec):
    def thinking_behaviour(self):
        return "interleaved"

    def render(self, messages, tools):
        import re

        strip = [{**m, "content": re.sub(r"<think>.*?</think>", "", m.get("content") or "", flags=re.DOTALL)} if m["role"] == "assistant" else m
                 for m in messages]
        return super().render(strip, tools)


async def test_V9_compaction_under_a_rerendering_template_is_trainable(tmp_path):
    from crucible_pack_toy.longhorizon.task import BacklogTask

    from crucible.train.world_env import rlm_world_rollout

    task = BacklogTask(seed=0, n=12)
    acts = task.oracle_actions()
    cells = [f"ACTS = {acts!r}"]
    for i in range(0, len(acts), 4):
        cells.append(f"for a in ACTS[{i}:{i + 4}]:\n    r = await (world.call(a['function_name'], **a['arguments']) if a['kind'] == 'tool_call' "
                     f"else world.send(a['to'], a['text']))\nprint('chunk', 'x' * 1200)")
    cells.append("rlm.finish('done')")
    gen, seen = engine(cells, think=True)
    # M16 B1: the window is in the engine's real (byte) tokens now, compared with usage-anchored counts; 2,500 was chars/4 units
    cfg = rlm_config(budget_tokens=24_000, trigger=0.8, keep_recent_turns=1, summary_max_tokens=200)
    try:
        ro = await rlm_world_rollout(lambda p: task, Interleaved(), cfg, workroot=tmp_path)("rid", "p", 0, gen, temperature=1.0)
    finally:
        task.cleanup()
    assert seen["compact"] >= 1 and ro.reward == 1.0
    assert not [t.call_id for t in ro.turns if t.prefix_break] and ro.valid_for_training()[0], ro.stop


# ---- V10 / D5: a nondeterministic cell does not end the episode at a restore -------------------------------------------
async def test_V10_restore_after_a_nondeterministic_cell_continues(tmp_path):
    script = [code("import time\nt0 = time.time()\nkeep = 41", 1),
              code("import time\nwhile True:\n    try:\n        time.sleep(0.05)\n    except BaseException:\n        pass", 2),
              code("print('alive', keep + 1)\nrlm.finish('ok')", 3)]
    ep, p, _w = await run_rlm(tmp_path, script, cell_timeout_s=1)
    assert not any(f["kind"] == "crucible:policy_error" for f in ep.failures), ep.failures
    outs = [m["content"] for m in p.messages if m["role"] == "tool"]
    assert ep.stop_condition == "agent_done" and outs[-1].startswith("alive 42"), outs


# ---- V11: spawn breadth ------------------------------------------------------------------------------------------------
async def test_V11_spawn_budget_is_shared_by_the_tree():
    from crucible.accept import repair

    child = FakeLLM([code("rlm.finish('child done')", i) for i in range(10)])
    cell = "out = []\nfor i in range(5):\n    try:\n        out.append(await rlm.spawn('worker', 'go'))\n    except Exception as e:\n        out.append('ERR ' + str(e))\nrlm.finish(repr(out))"
    agent = repair.RLMAgent("m", FakeLLM([code(cell, 0)]), roles={"worker": {"llm": child}}, launcher=host_launcher(),
                            budget=repair.TreeBudget(max_spawns=3))
    run = await agent.run("spawn")
    res = eval(run.answer)
    assert res[:3] == ["child done"] * 3 and all(r.startswith("ERR") and "spawn budget" in r for r in res[3:]), res


# ---- V12 / D9: per-rollout seed and launcher ---------------------------------------------------------------------------
async def test_V12_rollouts_get_their_own_launcher_and_seed(tmp_path):
    from crucible_pack_toy.longhorizon.task import BacklogTask

    from crucible.train.world_env import rlm_world_rollout

    seeds: list[int] = []

    class Recording(K.LocalLauncher):
        async def start(self, seed):
            seeds.append(seed)
            return await super().start(seed)

    made: list[Any] = []

    def factory():
        made.append(Recording(allow_host=True))
        return made[-1]

    with pytest.raises(TypeError):
        rlm_world_rollout(lambda p: None, CODEC, rlm_config(), launcher=host_launcher())
    fn = rlm_world_rollout(lambda p: task, CODEC, rlm_config(), workroot=tmp_path, launcher=factory)
    for rid in ("rid-1", "rid-2"):
        task = BacklogTask(seed=0, n=4)
        gen, _ = engine(["rlm.finish('x')"])
        try:
            await fn(rid, "p", 0, gen, temperature=1.0)
        finally:
            task.cleanup()
    assert len(made) == 2 and made[0] is not made[1]
    assert len(set(seeds)) == 2 and 0 not in seeds, seeds


# ---- V13: bridge inputs ------------------------------------------------------------------------------------------------
async def test_V13_bridge_checks_inputs_and_unattributed_world_calls(tmp_path):
    from crucible.actors.policy import EpisodeLimits, run_episode
    from crucible.actors.rlm.policy import RLMPolicy

    script = [code("await host('rlm.note', text='n' * 5000)\n"
                   "async def _deliver(msgs):\n    return await world.call('tickets.list_tickets', status='open')\n", 1),
              code("rlm.finish('ok')", 2)]
    task = helpdesk_episode_parts()
    w = await task.open(tmp_path / "work")
    p = RLMPolicy("agent", FakeLLM(script), launcher=host_launcher())
    try:
        await run_episode(w, {"agent": p}, limits=EpisodeLimits(max_steps=60))
        assert len(p.notes[0]) <= 512  # the host bounds a note, not only the kernel-side wrapper
        await p._ensure_kernel()
        await p.kernel.execute("async def _deliver(msgs):\n    return await world.call('tickets.list_tickets', status='open')", p._host)
        await asyncio.wait_for(p._deliver_inbox([{"from": "x", "text": "hi", "at": 0}]), timeout=10)  # model code redefined _deliver
        assert p._out.empty() and p._pending_action is None  # no World action attributed to no model call
    finally:
        await w.close()
        await p.aclose()


# ---- D1: sub-calls answered by another model are not trained ------------------------------------------------------------
async def test_D1_other_model_sub_calls_are_recorded_not_trained(tmp_path):
    from crucible.core.schema.run import RunRecord
    from crucible.runtime.record import build
    from crucible.train.credit import token_credit

    tok = {"prompt_token_ids": [1, 2], "completion_token_ids": [5, 6], "logprobs": [-0.1, -0.2], "policy_version": 1}
    sub = FakeLLM([LLMResponse(text="helper says", model="frozen-helper", tokens=dict(tok))])
    script = [code("s = await rlm.query('q')\nrlm.finish(s)", 1, tokens=dict(tok))]
    _ep, p, _w = await run_rlm(tmp_path, script, sub=sub)
    rec = RunRecord(run_id="r", attempt="r-a1", task="t", world="w", policy="p", outcome="completed", reward=1.0, started_at="2026-10-13T00:00:00Z")
    ro = build(rec, calls={"agent": p.model_calls}, mode="train")
    s = next(t for t in ro.turns if t.kind == "sub")
    assert s.trainable is False and s.model == "frozen-helper" and not s.completion_token_ids
    assert ro.valid_for_training()[0]
    assert all(sum(c.loss_mask) == 0 for c in token_credit(ro, 1.0) if c.call_id == s.call_id)


# ---- D4: sub turns share the parent's credit without diluting it -------------------------------------------------------
def test_D4_sub_turns_do_not_dilute_root_credit():
    from crucible.core.schema.rollout import RolloutRecord, Turn
    from crucible.train.credit import turn_credit_map

    def rec(rid: str, reward: float, n_subs: int) -> RolloutRecord:
        turns = [Turn(call_id="c1", actor="a", completion_token_ids=[1], advantage=0.25)]
        turns += [Turn(call_id=f"s{i}", actor="a", kind="sub", completion_token_ids=[2], advantage=0.25, depends_on=["c1"]) for i in range(n_subs)]
        turns += [Turn(call_id="c2", actor="a", completion_token_ids=[3], advantage=0.0)]
        return RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", turns=turns, reward=reward)

    roots = []
    for n in (0, 3, 10):
        recs = [rec("x", 1.0, n), rec("y", 0.0, 0)]
        m = turn_credit_map(recs, ["g", "g"], eta=1.0)
        roots.append(m[("x", "c1")])
        assert all(m[("x", f"s{i}")] == m[("x", "c1")] for i in range(n))
    assert max(roots) - min(roots) < 1e-9, roots


# ---- D14: sub turns keep version spans; a second python call is answered -----------------------------------------------
async def test_D14_sub_version_spans_and_second_python_call(tmp_path):
    from crucible.core.schema.run import RunRecord
    from crucible.runtime.record import build

    calls = [{"call_id": "a:c1", "kind": "turn", "segment": 0, "tokens": {"prompt_token_ids": [1], "completion_token_ids": [2, 3], "logprobs": [0, 0],
                                                                           "policy_version": 1}},
             {"call_id": "a:c2", "kind": "sub", "parent": "a:c1", "segment": 0,
              "tokens": {"prompt_token_ids": [9], "completion_token_ids": [4, 5, 6], "logprobs": [0, 0, 0], "policy_version": 2,
                         "version_spans": [[0, 1], [2, 2]]}}]
    rec = RunRecord(run_id="r", attempt="r-a1", task="t", world="w", policy="p", outcome="completed", reward=1.0, started_at="2026-10-13T00:00:00Z")
    ro = build(rec, calls={"a": calls})
    assert next(t for t in ro.turns if t.kind == "sub").version_spans == [(0, 1), (2, 2)]
    two = LLMResponse(text="", tool_calls=[{"id": "p1", "name": "python", "arguments": {"code": "x = 1"}},
                                           {"id": "p2", "name": "python", "arguments": {"code": "y = 2"}}])
    _ep, p, _w = await run_rlm(tmp_path, [two, code("rlm.finish('ok')", 3)])
    answered = {m.get("tool_call_id") for m in p.messages if m["role"] == "tool"}
    assert {"p1", "p2"} <= answered, answered


# ---- D15: INBOX announcements stay under the output cap ----------------------------------------------------------------
async def test_D15_inbox_announcement_is_capped():
    from crucible.actors.rlm.policy import RLMPolicy

    p = RLMPolicy("agent", FakeLLM([]), launcher=host_launcher(), max_output_chars=200)
    real = p._host

    async def host(data):
        if data.get("type") == "rlm.note":
            p._cell_msgs += [{"from": f"user{i}", "text": "m" * 300, "at": 0} for i in range(50)]
        return await real(data)

    p._host = host
    try:
        await p._ensure_kernel()
        await p._run_cell({"call_id": "agent:c1", "actions": []}, "t1", "await rlm.note('x')\nprint('o' * 1000)")
        out = p.messages[-1]["content"]
        assert len(out) <= 260 and "new message(s) appended to INBOX" in out, len(out)
    finally:
        await p.aclose()


# ---- trainer side of G1 D1 / D14 (lead) -----------------------------------------------------------------------------

def _sub_rec(sub_trainable: bool):
    from crucible.core.schema.rollout import RolloutRecord, Turn

    root = Turn(call_id="c0", actor="agent", prompt_token_ids=[1, 2], completion_token_ids=[3, 4, 5], logprobs=[-1.0] * 3, policy_version=0)
    sub = Turn(call_id="c0.s1", actor="agent", kind="sub", prompt_token_ids=[7, 8], completion_token_ids=[9, 10], logprobs=[-1.0] * 2,
               policy_version=0, trainable=sub_trainable)
    root2 = Turn(call_id="c1", actor="agent", prompt_is_delta=True, prompt_token_ids=[6], completion_token_ids=[11, 12], logprobs=[-1.0] * 2,
                 policy_version=0)
    return RolloutRecord(rollout_id="r", sample="r", task="t", harness="h", policy="p", reward=1.0, turns=[root, sub, root2])


def test_G1_D1_trainer_never_trains_another_models_sub_call():
    from crucible.train.engine import TinyPolicy
    from crucible.train.trainer import TorchTrainer, TrainerConfig

    tr = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(item_unit="call"))
    items = tr._items([_sub_rec(False), _sub_rec(False).model_copy(update={"rollout_id": "q", "reward": 0.0})], ["g", "g"])
    assert all(it.completion != [9, 10] for it in items)  # the other model's tokens carry no loss
    items = tr._items([_sub_rec(True), _sub_rec(True).model_copy(update={"rollout_id": "q", "reward": 0.0})], ["g", "g"])
    assert any(it.completion == [9, 10] for it in items)


def test_G1_D14_critic_keeps_sub_calls_out_of_the_root_chain():
    import torch

    from crucible.train.critic import CriticConfig, TinyValue
    from crucible.train.engine import TinyPolicy
    from crucible.train.trainer import TorchTrainer, TrainerConfig

    critic = TinyValue(32, 8, seed=1)
    with torch.no_grad():
        critic.head.weight.normal_(0, 0.5)
    cfg = CriticConfig(lam_policy=0.9)
    with_sub = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(item_unit="call"), critic=critic, critic_cfg=cfg)._gae_advantages([_sub_rec(True)])
    rec = _sub_rec(True)
    rec.turns = [t for t in rec.turns if t.kind != "sub"]
    without = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(item_unit="call"), critic=critic, critic_cfg=cfg)._gae_advantages([rec])
    for k in (("r", "c0"), ("r", "c1")):  # the root chain is the same whether or not the RLM called a sub-LLM
        assert with_sub[k] == without[k]
    assert len(with_sub[("r", "c0.s1")]) == 2
