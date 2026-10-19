"""M14 audit: the validation campaign's core-side findings (docs/audit/2026-10-13/M14-campaign.md, B1 B3 B4 B6 B9 B10 B12
B13 B14). One test per fix; each fails on the M13 commit (f2d9e18) where the behaviour existed there. B10 (the
authenticated kernel protocol) guards a path this audit introduced: its test runs against the current kernel."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

import crucible
from crucible.core.schema.actions import Done, Message, ToolCall

SRC = Path(crucible.__file__).resolve().parents[1]  # the crucible under test (a child process imports the same one)


def _tc(i, name, **args):
    return ToolCall(tool_call_id=f"x{i}", function_name=name, arguments=args)


# ---- B1: a lone surrogate in a tool argument is the policy's scored error -------------------------------------------------
async def test_b1_surrogate_argument_is_scored_through_the_runner(tmp_path):
    """approve-all-like (no-op) + a surrogate argument scores 0 and oracle + the same call scores 1, never masked, and the
    run's health rule is not tripped (on f2d9e18 every such episode was agent_run_error, unknown blame, retried)."""
    from crucible_pack_toy.runtime import HelpdeskTask, scripted
    from crucible_pack_toy.verify import GOLD

    from crucible.core.schema.run import RunRecord
    from crucible.runtime.harness.native import NativeHarness
    from crucible.runtime.infra import Health
    from crucible.runtime.run import Runner, RunSpec

    bad = [_tc(91, "tickets.get_ticket", ticket_id="\udcff"), _tc(92, "kb.search_docs", query="\ud800"),
           _tc(93, "tickets.get_\udcffticket", ticket_id=1), Message(to="supervisor", text="hello \udcff")]
    pols = [scripted("noop+surrogate", bad), scripted("oracle+surrogate", list(GOLD) + bad)]
    s = await Runner(tmp_path).run(RunSpec(run_id="b1", tasks=[HelpdeskTask()], policies=pols, harnesses=[NativeHarness(name="native", version="1")],
                                           k=2, min_terminal_for_health=1))
    recs = [RunRecord.model_validate_json(x) for x in (tmp_path / "b1" / "results.jsonl").read_text().splitlines()]
    by = {}
    for r in recs:
        by.setdefault(r.policy, []).append(r)
    assert s.stopped is None and s.attempts == 4, s
    assert [r.reward for r in by["scripted:noop+surrogate"]] == [0.0, 0.0] and [r.reward for r in by["scripted:oracle+surrogate"]] == [1.0, 1.0]
    assert not any(r.mask_sample for r in recs)
    h = Health(planned=4)
    for r in recs:
        h.add(r)
    assert h.stop_reason(min_terminal=1) is None and h.unknown == 0
    trace = (next((tmp_path / "b1" / "samples").glob("*/a1")) / "trace.jsonl").read_text()
    trace.encode("utf-8")  # every line of the streamed trace is writable UTF-8 (the surrogate kept as escaped text)


async def test_b1_world_step_refuses_unencodable_text_and_keeps_the_trace_writable(tmp_path):
    from crucible_pack_toy.runtime import HelpdeskTask

    from crucible.core.schema.trace import Trace

    w = await HelpdeskTask().open(tmp_path / "w")
    try:
        w.stream_trace(tmp_path / "trace.jsonl")
        for call in (_tc(1, "tickets.get_ticket", ticket_id="\udcff"), _tc(2, "tickets.get_ticket", **{"\ud800": 1}),
                     _tc(3, "tickets.list_\udcfftickets"), ToolCall(tool_call_id="i\udcff", function_name="tickets.get_ticket", arguments={"ticket_id": 1})):
            res = await w.step("agent", call)
            (r,) = res.observation.tool_results
            assert not r.ok and r.error == "invalid_arguments" and r.fault == "policy" and not res.done, r
            r.content.encode("utf-8")
        assert not w.failures
        await w.step("agent", Message(to="supervisor", text="approve \udcff please"))
        res = await w.step("agent", Done(reason="bye \ud800", answer="v\udcff"))
        assert res.done
        Trace.model_validate_json(w.to_trace().model_dump_json())  # the whole trace serializes
        lines = (tmp_path / "trace.jsonl").read_text().splitlines()
        assert any("\\udcff" in x for x in lines)  # escaped, not dropped
    finally:
        await w.close()


async def test_b1_a_models_json_escaped_surrogate_is_scored(tmp_path):
    """The campaign's form: a model writes ``"\\udcff"`` (plain ASCII) inside its tool-call JSON; the parsed argument holds
    a lone surrogate. The oracle's calls plus that call still score 1 through the LLM tool policy, with every evidence
    file written (trace, calls, trajectory)."""
    from crucible_pack_toy.runtime import HelpdeskTask
    from crucible_pack_toy.verify import GOLD

    from crucible.actors.policy import EpisodeLimits, LLMToolPolicy
    from crucible.runtime.harness.native import NativeHarness, PolicyRef
    from crucible.runtime.run import Runner, RunSpec
    from crucible.train.chat import ByteChatCodec
    from crucible.train.engine import GenResult
    from crucible.train.world_env import EngineLLM

    codec = ByteChatCodec()
    calls = [{"name": a.function_name, "arguments": a.arguments} for a in GOLD if isinstance(a, ToolCall)]
    calls.append({"name": "tickets.get_ticket", "arguments": {"ticket_id": "\udcff"}})
    assert "\\udcff" in json.dumps(calls[-1])  # ASCII on the wire

    def make(a):
        st = {"i": 0}

        async def gen(prompt, max_tokens, temperature):
            i = st["i"]
            st["i"] += 1
            text = ("<tool_call>" + json.dumps(calls[i]) + "</tool_call>") if i < len(calls) else "Done."
            ids = list(text.encode()) + [codec.EOS]
            return GenResult(completion_ids=ids, logprobs=[-0.1] * len(ids), version=1, finish="stop")

        return {"agent": LLMToolPolicy("agent", EngineLLM(codec, gen, model="byte"), max_tokens=4000, temperature=1.0)}

    await Runner(tmp_path).run(RunSpec(run_id="b1llm", tasks=[HelpdeskTask()], policies=[PolicyRef(id="llm", make=make, config={})],
                                       harnesses=[NativeHarness(limits=EpisodeLimits(max_steps=60))], k=1))
    (r,) = [json.loads(x) for x in (tmp_path / "b1llm" / "results.jsonl").read_text().splitlines()]
    assert r["reward"] == 1.0 and not r["mask_sample"], r.get("failure")


def test_b1_residual_serialization_error_is_the_policys():
    from crucible.runtime.infra import StageFailure, boundary

    class PydanticSerializationError(ValueError):
        pass

    for ex in (UnicodeEncodeError("utf-8", "\udcff", 0, 1, "surrogates not allowed"),
               PydanticSerializationError("Error serializing to JSON: UnicodeEncodeError: 'utf-8' codec can't encode character '\\udcff': surrogates not allowed")):
        with pytest.raises(StageFailure) as e, boundary("agent"):
            raise ex
        f = e.value.failure
        assert f.kind == "crucible:policy_misbehavior" and f.who == "policy" and f.mask is False, f


# ---- B3: one seed tarball per World ------------------------------------------------------------------------------------
def _ws_spec(root: Path, seed: Path):
    from crucible.core.schema.world import WorldSpec

    return WorldSpec.model_validate({"world": "b3/ws", "apps": [{"id": "ws", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": str(root)}},
                                                                   "params": {"seed_dir": str(seed)}}],
                                     "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*"]}]})


def _seed(tmp_path: Path, n: int = 300) -> Path:
    seed = tmp_path / "seed"
    for i in range(n):
        (seed / f"d{i % 7}").mkdir(parents=True, exist_ok=True)
        (seed / f"d{i % 7}" / f"f{i}.txt").write_text(f"file {i}\n" * 200)
    return seed


async def test_b3_seed_tarball_is_unique_per_world_and_removed(tmp_path, monkeypatch):
    from crucible.core.store import Store
    from crucible.world.exec import LocalExec
    from crucible.world.world import World

    seed = _seed(tmp_path, 20)
    cmds: list[str] = []
    orig = LocalExec.run

    async def spy(self, cmd, **kw):
        cmds.append(cmd)
        return await orig(self, cmd, **kw)

    monkeypatch.setattr(LocalExec, "run", spy)
    for i in range(2):
        w = World(_ws_spec(tmp_path / f"root{i}", seed), store=Store(tmp_path / f"s{i}"), workdir=tmp_path / f"w{i}")
        await w.start()
        assert (tmp_path / f"root{i}" / "d3" / "f3.txt").read_text() == "file 3\n" * 200
        await w.close()
    tars = [c.split("tar -xf ", 1)[1].split()[0].strip("'") for c in cmds if "tar -xf " in c]
    assert len(tars) == 2 and tars[0] != tars[1], tars  # f2d9e18: /tmp/.crucible_seed_ws.tar for every World on the host
    assert not any(Path(t).exists() for t in tars)


async def test_b3_concurrent_worlds_never_corrupt_each_other(tmp_path):
    from crucible.core.store import Store
    from crucible.world.world import World

    seed = _seed(tmp_path)

    async def one(i: int) -> int:
        w = World(_ws_spec(tmp_path / f"root{i}", seed), store=Store(tmp_path / f"s{i}"), workdir=tmp_path / f"w{i}")
        await w.start()
        try:
            return sum(1 for _ in (tmp_path / f"root{i}").rglob("*.txt"))
        finally:
            await w.close()

    assert await asyncio.gather(*(one(i) for i in range(8))) == [300] * 8


async def test_b3_a_failed_seed_copy_is_infra(tmp_path, monkeypatch):
    from crucible.core.store import Store
    from crucible.runtime.infra import StageFailure, boundary
    from crucible.world.exec import ExecOut, LocalExec
    from crucible.world.world import World

    orig = LocalExec.run

    async def broken(self, cmd, **kw):
        if "tar -xf" in cmd:
            return ExecOut(rc=2, stdout="", stderr="tar: Unexpected EOF in archive")
        return await orig(self, cmd, **kw)

    monkeypatch.setattr(LocalExec, "run", broken)
    w = World(_ws_spec(tmp_path / "root", _seed(tmp_path, 3)), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    with pytest.raises(StageFailure) as e, boundary("seed"):
        await w.start()
    assert e.value.failure.kind == "crucible:backend_error" and e.value.failure.who == "infra", e.value.failure


# ---- B4: RLM containers never outlive their rollout or their process ---------------------------------------------------
IMAGE = "python:3.12-slim"


def _have_image() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=False).returncode == 0


def _exists(name: str) -> bool:
    out = subprocess.run(["docker", "ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"], capture_output=True, text=True, check=False)
    return name in out.stdout.split()


def _gone_within(name: str, s: float) -> bool:
    import time

    end = time.monotonic() + s
    while time.monotonic() < end:
        if not _exists(name):
            return True
        time.sleep(0.25)
    return not _exists(name)


needs_image = pytest.mark.skipif(not _have_image(), reason=f"needs Docker and {IMAGE}")


@pytest.mark.docker
@needs_image
async def test_b4_a_rollout_cancelled_mid_docker_run_leaves_no_container(monkeypatch):
    from crucible.actors.rlm import kernel as K

    spawned = asyncio.Event()
    names: list[str] = []
    orig = K._run_subprocess_raw  # M10 lab audit R-5 item 4: the launcher's docker calls go through world.exec, not asyncio's child watcher

    async def spy(argv, **kw):
        if len(argv) > 2 and argv[1] == "run" and "--name" in argv:  # a slow daemon: the container appears a second later
            names.append(argv[argv.index("--name") + 1])
            spawned.set()
            return await orig(["sh", "-c", 'sleep 1; exec "$@"', "sh", *argv], **kw)
        return await orig(argv, **kw)

    monkeypatch.setattr(K, "_run_subprocess_raw", spy)
    launcher = K.DockerLauncher(IMAGE)
    t = asyncio.ensure_future(launcher.start(0))
    try:
        await asyncio.wait_for(spawned.wait(), 30)
        t.cancel()  # the trainer stops while ``docker run`` is in flight
        await asyncio.gather(t, return_exceptions=True)
        await launcher.aclose()  # what the rollout's cleanup does (f2d9e18: ``rm -f`` before the container existed)
        await asyncio.sleep(2.5)  # past the moment the slow ``docker run`` creates its container
        assert names and _gone_within(names[0], 8), f"{names[0]} leaked"
    finally:
        for n in names:
            subprocess.run(["docker", "rm", "-f", n], capture_output=True, check=False)  # noqa: ASYNC221


def _child(code: str, *args: str) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.Popen([sys.executable, "-c", code, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


CHILD = f"""
import asyncio, sys
from crucible.actors.rlm.kernel import DockerLauncher

async def main():
    l = DockerLauncher({IMAGE!r})
    await l.start(0)
    print(l.name, flush=True)
    if sys.argv[1] == "wait":
        await asyncio.sleep(120)

asyncio.run(main())
"""


@pytest.mark.docker
@needs_image
@pytest.mark.parametrize("how", ["exit", "sigterm"])
def test_b4_a_process_that_ends_with_containers_removes_them(how):
    p = _child(CHILD, "wait" if how == "sigterm" else "exit")
    name = ""
    try:
        name = p.stdout.readline().strip()
        assert name.startswith("crucible-rlm-"), p.stderr.read()
        if how == "sigterm":
            p.send_signal(signal.SIGTERM)
        p.wait(timeout=60)
        assert _gone_within(name, 8), f"{name} outlived its process ({how})"
    finally:
        if p.poll() is None:
            p.kill()
        if name:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


@pytest.mark.docker
@needs_image
def test_b4_stale_containers_of_dead_owners_are_reaped_and_lifetime_is_bounded():
    import socket
    import time
    import uuid

    from crucible.actors.rlm import kernel as K

    dead = subprocess.Popen(["true"])
    dead.wait()
    host = socket.gethostname()
    mk = lambda owner, expires: (f"crucible-rlm-b4-{uuid.uuid4().hex[:8]}", owner, expires)
    cases = [mk(f"{host}:{dead.pid}", int(time.time()) + 3600), mk(f"{host}:{os.getpid()}", int(time.time()) + 3600),
             mk(f"other-host:{dead.pid}", int(time.time()) - 5)]
    try:
        for name, owner, exp in cases:
            subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "--label", "crucible.rlm=1", "--label", f"{K.OWNER_LABEL}={owner}",
                            "--label", f"{K.EXPIRES_LABEL}={exp}", IMAGE, "sleep", "300"], capture_output=True, check=True)
        reaped = K.reap_stale()
        assert cases[0][0] in reaped and cases[2][0] in reaped and cases[1][0] not in reaped  # dead owner, expired; a live owner is kept
        assert _exists(cases[1][0])
    finally:
        for name, _, _ in cases:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)

    async def started() -> list:
        launcher = K.DockerLauncher(IMAGE, lifetime_s=900)
        await launcher.start(0)
        try:
            out = subprocess.run(["docker", "inspect", launcher.name, "--format", "{{json .Config.Cmd}} {{json .Config.Labels}}"],  # noqa: ASYNC221
                                 capture_output=True, text=True, check=True).stdout
        finally:
            await launcher.aclose()
        return out.split(" ", 1)

    cmd, labels = asyncio.run(started())
    assert json.loads(cmd) == ["sleep", "900"]  # never ``sleep infinity``
    lab = json.loads(labels)
    assert lab[K.OWNER_LABEL] == K.owner_id() and lab["crucible.rlm"] == "1" and int(lab[K.EXPIRES_LABEL]) > time.time()


# ---- B6: bench run awaits async hooks, honours requires, needs no policies hook ----------------------------------------
def _demo_pack(tmp_path: Path, *, requires: str | None = None) -> Path:
    from click.testing import CliRunner

    from crucible.cli.main import cli

    d = tmp_path / "b6demo"
    r = CliRunner().invoke(cli, ["pack", "new", "b6demo", "--dir", str(d)], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    studio = next(d.glob("src/*/studio.py"))
    text = studio.read_text().replace("def tasks() -> list:", "async def tasks() -> list:")
    text = text.replace("def generate(seed: int, item_id: str | None = None)", "async def generate(seed: int, item_id: str | None = None)")
    studio.write_text(text)
    y = (d / "pack.yaml").read_text()
    y = "\n".join(ln for ln in y.splitlines() if not ln.strip().startswith("policies:")) + "\n"
    if requires:
        y = y.replace("determinism: seeded", f"determinism: seeded\nrequires: [{requires}]")
    (d / "pack.yaml").write_text(y)
    return d


def test_b6_bench_run_awaits_async_hooks_and_benches_a_pack_without_policies(tmp_path):
    from click.testing import CliRunner

    from crucible.cli.main import cli

    d = _demo_pack(tmp_path)
    r = CliRunner().invoke(cli, ["bench", "run", str(d), "--out", str(tmp_path / "runs"), "--fresh", "1", "--k", "1", "--json"])
    assert r.exit_code == 0, r.output  # f2d9e18: TypeError: 'coroutine' object is not iterable
    doc = json.loads(r.output)
    assert any("no hooks.policies" in n for n in doc["notes"])
    means = {m["policy"]: m["overall"]["deterministic"]["mean_reward"] for m in doc["summary"]["models"]}
    assert means["pack:gold"] == 1.0 and means["pack:noop"] == 0.0, means
    assert doc["summary"]["config"]["tasks"] == 3  # 2 anchors + 1 fresh item from the async generator


def test_b6_bench_run_refuses_a_missing_requirement_unless_allowed(tmp_path):
    from click.testing import CliRunner

    from crucible.cli.main import cli

    d = _demo_pack(tmp_path, requires="paid")
    r = CliRunner().invoke(cli, ["bench", "run", str(d), "--out", str(tmp_path / "runs"), "--fresh", "0", "--k", "1", "--no-controls"])
    assert r.exit_code != 0 and "requires paid" in r.output and "--allow paid" in r.output, r.output
    r = CliRunner().invoke(cli, ["bench", "run", str(d), "--out", str(tmp_path / "runs2"), "--fresh", "0", "--k", "1", "--no-controls", "--allow", "paid"])
    assert r.exit_code == 0, r.output


async def test_b6_dynamic_build_awaits_an_async_generator():
    from crucible.bench import dynamic

    class T:
        def __init__(self, seed):
            self.id, self.seed = "tmp", seed

    async def gen(seed, item_id=None):
        return T(seed)

    bm = await dynamic.build("b6", "v1", key=dynamic.new_key(), anchors=[], generators={"g": gen}, n_fresh=2, allow_unvalidated=True)
    assert [type(i.task) for i in bm.items] == [T, T]


# ---- B9: a host kernel without allow_host is refused when the harness is built --------------------------------------------
def test_b9_rlm_harness_refuses_a_host_kernel_at_construction():
    from crucible.runtime.harness.rlm import RLMConfig, RLMHarness

    with pytest.raises(ValueError, match="allow_host"):
        RLMHarness(RLMConfig(kernel="local"))
    RLMHarness(RLMConfig(kernel="local", allow_host=True))  # opted in: built
    RLMHarness(RLMConfig(kernel="local"), launcher=lambda a, actor: None)  # an explicit launcher decides placement itself


# ---- B10: forged frames on the protocol descriptor are dropped ----------------------------------------------------------
FORGE_FD = r'''
import os, sys, json
m = sys.modules["__main__"]
cid = (getattr(m, "_active", None) or {}).get("cid") or m._cell.get()
frame = {"event": "done", "id": cid, "stdout": "forged", "stderr": "", "result": "666", "error": None, "timed_out": False,
         "finished": "FORGED", "digest": "deadbeef"}
hits = 0
for fd in range(3, 256):
    for f in (frame, {**frame, "mac": "0" * 64}):
        try:
            os.write(fd, (json.dumps(f) + "\n").encode())
            hits += 1
        except OSError:
            pass
assert not hasattr(m, "_protocol_fd") and not hasattr(m, "_send")
print("wrote", hits > 0)
41 + 1
'''


async def test_b10_forged_done_frames_on_any_descriptor_are_dropped():
    from crucible.actors.rlm import kernel as K

    async def echo(data):
        return data

    k = K.Kernel(K.LocalLauncher(allow_host=True), cell_timeout_s=20)
    try:
        r = await k.execute(FORGE_FD, echo)
        assert r.result == "42" and r.finished is None and r.digest != "deadbeef" and "wrote True" in r.stdout, r
        assert r.protocol_errors >= 1 and k.journal[-1].digest == r.digest
        r2 = await k.execute("x = 5\nx", echo)
        assert r2.result == "5"
    finally:
        await k.close()
        await k.launcher.aclose()


async def test_b10_a_frame_signed_with_another_key_is_dropped():
    import hashlib
    import hmac

    from crucible.actors.rlm import kernel as K

    k = K.Kernel(K.LocalLauncher(allow_host=True))
    k._mac = hmac.new(b"k" * 32, digestmod=hashlib.sha256)
    ev = {"event": "done", "id": "c", "digest": "d"}
    mac = hmac.new(b"x" * 32, json.dumps(ev, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256).hexdigest()
    assert k._unauthentic({**ev, "mac": mac}) == "bad frame MAC"
    good = hmac.new(b"k" * 32, json.dumps(ev, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256).hexdigest()
    assert k._unauthentic({**ev, "mac": good}) is None and k._unauthentic(dict(ev)) == "unauthenticated frame"


# ---- B12: the key registry names the clash -------------------------------------------------------------------------------
def test_b12_one_key_for_two_benchmarks_names_the_other_benchmark(tmp_path):
    from crucible.bench import dynamic

    key = dynamic.new_key()
    reg = tmp_path / "bench.keys.jsonl"
    dynamic._register(reg, key, "demo-bench", "v1")
    with pytest.raises(ValueError, match="another benchmark, demo-bench@v1") as e:
        dynamic._register(reg, key, "toy-bench", "v1")
    assert "new version" not in str(e.value)
    with pytest.raises(ValueError, match="a new version needs a new key"):
        dynamic._register(reg, key, "demo-bench", "v2")


# ---- B13: the policy's bad arguments are the policy's error ----------------------------------------------------------------
TOOLS = '''
def get_invoice(invoice_id: int, verbose: bool = False) -> dict:
    """One invoice."""
    if invoice_id != 1:
        raise ValueError(f"no invoice {invoice_id}")
    return {"id": invoice_id}


def set_status(invoice_id: int, status: str) -> dict:
    """Set a status."""
    if status not in ("approved", "held"):
        raise KeyError(f"unknown status {status!r}")
    return {"id": invoice_id, "status": status}
'''


@pytest.mark.parametrize("name, args, why", [
    ("get_invoice", {"invoice_id": 99999}, "no invoice 99999"),
    ("set_status", {"invoice_id": 1, "status": "*"}, "unknown status"),
    ("get_invoice", {"invoice_id": 1, "vendor_id": 7}, "unknown argument(s) vendor_id"),
    ("get_invoice", {"invoice_id": 1.0}, "expected an integer"),
    ("get_invoice", {"invoice_id": True}, "expected an integer"),
])
async def test_b13_tool_layer_attributes_bad_arguments_to_the_policy(tmp_path, name, args, why):
    from crucible.world.packaging.mcp_tools import InProcessToolModule

    f = tmp_path / "erp_tools.py"
    f.write_text(TOOLS)
    m = InProcessToolModule("erp", f)
    try:
        r = await m.call(name, args, call_id="c")
        assert not r.ok and r.error == "invalid_arguments" and r.fault == "policy" and why in r.content, r
        ok = await m.call("get_invoice", {"invoice_id": 1}, call_id="c2")
        assert ok.ok
    finally:
        m.unload()


async def test_b13_a_path_outside_the_workspace_is_the_policys_error(tmp_path):
    from crucible.core.store import Store
    from crucible.world.world import World

    w = World(_ws_spec(tmp_path / "root", _seed(tmp_path, 2)), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()
        res = await w.step("agent", _tc(1, "ws.read_file", path="../../../etc/passwd"))
        (r,) = res.observation.tool_results
        assert not r.ok and r.error == "invalid_arguments" and r.fault == "policy" and "outside the workspace" in r.content, r
    finally:
        await w.close()


# ---- B14: a step limit below the oracle's length is reported, not a silent broken oracle --------------------------------
async def test_b14_controls_report_oracle_truncated_and_size_their_step_limit(tmp_path):
    from crucible_pack_toy.runtime import HelpdeskTask
    from crucible_pack_toy.verify import GOLD

    from crucible.actors.policy import EpisodeLimits
    from crucible.bench.report import warnings
    from crucible.bench.run import run_controls
    from crucible.runtime.harness.native import NativeHarness

    t = HelpdeskTask()
    t.bench_gold = list(GOLD)
    short = NativeHarness(name="native", version="1", limits=EpisodeLimits(max_steps=3))
    man = await run_controls(tmp_path, "b14", [t], [short], k=1)
    e = man[t.id]
    assert e["valid"] is False and e["reasons"] == ["oracle_truncated"], e  # f2d9e18: ["broken_oracle"]
    assert e["oracle"]["passed"] == 1 and e["oracle_steps"] == len(GOLD) and e["max_steps"] == 3  # the oracle itself works
    assert short.limits.max_steps == 3  # the run's own harness is untouched
    w = warnings({"invalid_tasks": {t.id: e}})
    assert any("oracle_truncated" in x for x in w)
    ok = await run_controls(tmp_path, "b14ok", [t], [NativeHarness(name="native", version="1", limits=EpisodeLimits(max_steps=len(GOLD) + 1))], k=1)
    assert ok[t.id]["valid"] is True and ok[t.id]["reasons"] == []
