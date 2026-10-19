"""A command's output is read as it arrives and only a head and a tail are kept (M1 audit 2026-10-14 R-2): a policy's
flood never grows the orchestrator, a background job never holds the call open, and Harbor's exec (which keeps every
byte it receives) gets its output cut inside the container."""

from __future__ import annotations

import asyncio
import resource
import shutil
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

import pytest

from crucible.world import exec as exec_mod
from crucible.world.exec import DockerExec, LocalExec


def marker(n: int) -> str:
    return f"\n...[{n} bytes dropped]...\n"


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(exec_mod, "OUT_HEAD", 1000)
    monkeypatch.setattr(exec_mod, "OUT_TAIL", 4000)


async def test_a_flood_keeps_its_head_and_tail_per_stream(small):
    out = await LocalExec().run("printf START; head -c 10000000 /dev/zero; printf END; head -c 100000 /dev/zero >&2")
    assert out.ok and out.stdout.startswith("START") and out.stdout.endswith("END")
    assert out.stdout[1000:].startswith(marker(10_000_008 - 5000)) and len(out.stdout) == 5000 + len(marker(10_000_008 - 5000))
    assert out.stderr[1000:].startswith(marker(100_000 - 5000))


async def test_output_that_fits_passes_byte_for_byte(small):
    assert (await LocalExec().run("head -c 4999 /dev/zero | tr '\\0' y")).stdout == "y" * 4999  # longer than the head alone
    assert await LocalExec().run_python("def main(a):\n    return 'y' * 3000", {}) == "y" * 3000  # its result sits at the end


async def test_the_real_caps_bound_memory():
    def peak() -> int:  # bytes (macOS reports bytes, Linux KiB)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)

    before = peak()
    out = await LocalExec().run("head -c 300000000 /dev/zero; echo END")
    dropped = 300_000_004 - exec_mod.OUT_HEAD - exec_mod.OUT_TAIL
    assert out.stdout.endswith("END\n") and len(out.stdout) == exec_mod.OUT_HEAD + exec_mod.OUT_TAIL + len(marker(dropped))
    assert peak() - before < 100 << 20  # 300 MB through the pipe; before, all of it was held


async def test_a_background_job_does_not_hold_the_call():
    ex = LocalExec()
    t0 = time.monotonic()
    out = await ex.run("sleep 30 & echo x")
    assert out.stdout == "x\n" and out.rc == 0 and time.monotonic() - t0 < 5  # was 30 s: communicate waited for the pipes
    ex.kill_sessions()


async def test_stdin_and_timeouts_still_work():
    assert len((await LocalExec().run("cat", stdin=b"z" * 3_000_000)).stdout) == 3_000_000  # fed while it is read
    assert (await LocalExec().run("true", stdin=b"z" * 3_000_000)).ok  # never read: the feeder stops, the call returns
    t0 = time.monotonic()
    out = await LocalExec().run("echo a; sleep 20", timeout_s=1)
    assert out.timed_out and out.rc == -9 and out.stdout == "a\n" and time.monotonic() - t0 < 5


class BashEnv:
    """Harbor's Docker exec, run here: ``bash -c <command>``, stderr merged into stdout, every byte kept."""

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        p = await asyncio.create_subprocess_exec("bash", "-c", command, cwd=cwd, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.STDOUT)
        out, _ = await p.communicate()
        return SimpleNamespace(return_code=p.returncode, stdout=out.decode(errors="replace"), stderr=None)

    async def upload_file(self, src, dest):
        shutil.copy(src, dest)


async def test_harbor_output_is_cut_inside_the_container(small, tmp_path):
    pytest.importorskip("harbor")
    from crucible.interop.harbor.world_backend import HarborExec

    ex = HarborExec(BashEnv())
    ex.staging_dir = str(tmp_path)
    out = await ex.run("printf START; head -c 100000 /dev/zero; echo oops >&2; printf END; exit 7")
    assert out.rc == 7 and len(out.stdout) == 5000 and out.stdout.startswith("START") and out.stdout.endswith("oops\nEND")
    assert (await ex.run("printf hello")).stdout == "hello"  # nothing marks a cut, so what fits is exact
    assert (await ex.run("cat; exit 3", stdin=b"abc")).stdout == "abc"  # staged stdin and its exit code compose with it


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=120, check=False)


@pytest.mark.docker
async def test_a_flood_through_docker_exec():
    name = f"crucible-r2-{uuid.uuid4().hex[:8]}"
    assert _docker("run", "-d", "--name", name, "python:3.12-slim", "sleep", "300").returncode == 0
    try:
        out = await DockerExec(name).run("head -c 50000000 /dev/zero; echo END")
        dropped = 50_000_004 - exec_mod.OUT_HEAD - exec_mod.OUT_TAIL
        assert out.ok and out.stdout.endswith("END\n") and len(out.stdout) == exec_mod.OUT_HEAD + exec_mod.OUT_TAIL + len(marker(dropped))
    finally:
        _docker("rm", "-f", name)
