"""m14: the RLM kernel. Cells with top-level await and a persistent namespace; the host bridge; a per-cell deadline; a
wedged cell killed; exact restore by journal replay (digest-checked, seeded random included) and divergence refused; a
credential-free environment; and the same kernel inside a no-network Docker container through ``docker exec -i``."""

from __future__ import annotations

import subprocess
import time

import pytest

from crucible.actors.rlm.kernel import ExecLauncher, Kernel, KernelDivergence, LocalLauncher

HOST = LocalLauncher(allow_host=True)  # trusted test code only: model code never runs on the host by default


async def echo(data):
    return {"echo": data}


async def test_cells_bridge_deadline_and_replay(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-not-leak")
    k = Kernel(HOST, cell_timeout_s=5)
    try:
        r = await k.execute("x = 1\nprint('hi', x)\nx + 1", echo)
        assert r.ok and r.stdout == "hi 1\n" and r.result == "2"
        r = await k.execute("y = await host('w.call', tool='t', a=3)\nimport random\nz = random.random()\ny['echo']['a']", echo)
        assert r.result == "3" and r.host_calls == 1
        r = await k.execute("while True:\n    await asyncio.sleep(0.05)", echo, timeout_s=1)
        assert r.timed_out and "exceeded 1" in r.error
        r = await k.execute("1/0", echo)
        assert "ZeroDivisionError" in r.error
        r = await k.execute("import os\n[k for k in os.environ if 'KEY' in k or 'TOKEN' in k]", echo)
        assert r.result == "[]"  # the kernel never sees the operator's credentials
        d = await k.digest()
        k2 = Kernel(HOST)
        await k2.replay(k.state()["journal"])
        assert await k2.digest() == d  # replayed with recorded host replies, the seeded random included
        bad = k.state()["journal"]
        bad[1]["replies"][0]["request"]["type"] = "other"
        with pytest.raises(KernelDivergence):
            await Kernel(HOST).replay(bad)
        await k2.close()
    finally:
        await k.close()


async def test_wedged_cell_is_killed():
    k = Kernel(HOST, cell_timeout_s=1, kill_grace_s=1)
    try:
        t0 = time.monotonic()
        r = await k.execute("import time\nwhile True: time.sleep(0.05)", echo)  # blocks the kernel's event loop: interrupted (else killed)
        assert r.timed_out and (r.interrupted or r.killed) and time.monotonic() - t0 < 10
        r = await k.execute("import time\nwhile True:\n    try:\n        time.sleep(0.05)\n    except BaseException:\n        pass", echo)
        assert r.killed and r.timed_out  # a cell that swallows the interrupt is killed
        r = await k.execute("1 + 1", echo)  # a fresh kernel starts on demand
        assert r.result == "2"
    finally:
        await k.close()


@pytest.mark.docker
async def test_kernel_inside_a_no_network_container(tmp_path):
    name = f"crucible-rlm-test-{int(time.time() * 1000)}"
    import asyncio

    await asyncio.to_thread(subprocess.run, ["docker", "run", "-d", "--rm", "--name", name, "--network", "none", "python:3.13-slim", "sleep", "infinity"],
                            check=True, capture_output=True)
    try:
        async def push(src, dest):
            await asyncio.to_thread(subprocess.run, ["docker", "cp", str(src), f"{name}:{dest}"], check=True, capture_output=True)

        k = Kernel(ExecLauncher(["docker", "exec", "-i", name], push=push), cell_timeout_s=10)
        r = await k.execute("import socket, platform\ntry:\n    socket.create_connection(('1.1.1.1', 80), timeout=2); net = 'open'\n"
                            "except OSError:\n    net = 'blocked'\nv = await host('ping')\n(net, v['echo']['type'], platform.python_version()[:4])", echo)
        assert r.ok and r.result.startswith("('blocked', 'ping', '3.13"), r
        k2 = Kernel(ExecLauncher(["docker", "exec", "-i", name], push=push))
        await k2.replay(k.state()["journal"])
        assert await k2.digest() == await k.digest()
        await k.close()
        await k2.close()
    finally:
        await asyncio.to_thread(subprocess.run, ["docker", "rm", "-f", name], capture_output=True, check=False)
