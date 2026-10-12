"""VM App: a full virtual machine under QEMU/KVM, with pre-booted pools and RAM snapshots (M1 open item; the OSWorld
pattern, a qcow2 guest under QEMU/KVM inside Docker with ``/dev/kvm``, as Harbor's osworld template runs it).

The guest is a Debian 12 cloud image (``genericcloud``, cloud-init injects a root SSH key from a seed ISO). It runs
in its own container (``--device /dev/kvm``, ``--network none``): QEMU's user-mode network gives the guest a NIC with
no route out and forwards the container's ``127.0.0.1:2222`` to the guest's SSH. The App talks to the guest over SSH
and to QEMU over its QMP socket.

**Pre-booted, RAM-snapshotted starts.** A cold boot (firmware, kernel, cloud-init) takes tens of seconds. The first VM on a
host boots cold once and saves an internal qcow2 snapshot named ``ready`` (RAM + devices + disk) into a **golden
overlay** on a shared Docker volume. Every later VM copies the golden overlay and starts with ``-loadvm ready``: it
resumes a booted, logged-in guest in about a second. ``VMPool`` keeps N such VMs warm; an episode acquires one and a
release resets it with ``loadvm ready`` in the background.

**Tools** (policy): ``exec`` (a shell command in the guest, with timeout; stdout, stderr, exit code), ``read_file``,
``write_file``. Graders read the guest through ``inspect(cmd)``.

**State and resets.** The journal (``json_doc`` at ``path``) records every action. ``before_snapshot`` saves a QMP
``savevm`` under the snapshot's tag (RAM and disk, exact); ``after_restore`` loads it with ``loadvm``. That is exact
mid-episode branching while the VM's container lives. A restore in a fresh container (another process) re-provisions
from the golden overlay and replays the journal's commands (``restore: replay``; outputs may differ, e.g. clocks),
or refuses mid-episode restores (``restore: start``, default).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import posixpath
import shlex
import time
import uuid
from typing import Any, ClassVar

from crucible.core.schema.actions import ToolResult, ToolSpec
from crucible.world.apps.base import App, ToolError
from crucible.world.exec import DockerExec, _run_subprocess
from crucible.world.state.base import SnapshotMismatch

GUEST_URL = "https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-genericcloud-amd64.qcow2"
DOCKERFILE = f"""FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends qemu-system-x86 qemu-utils socat openssh-client cloud-image-utils curl ca-certificates \\
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /vm && curl -fsSL -o /vm/base.qcow2 {GUEST_URL} \\
    && ssh-keygen -q -t ed25519 -N '' -f /vm/key \\
    && printf '#cloud-config\\ndisable_root: false\\nssh_pwauth: false\\nusers:\\n  - name: root\\n    ssh_authorized_keys:\\n      - %s\\npackage_update: false\\npackage_upgrade: false\\n' "$(cat /vm/key.pub)" > /vm/user-data \\
    && printf 'instance-id: crucible-vm\\nlocal-hostname: crucible-vm\\n' > /vm/meta-data \\
    && cloud-localds /vm/seed.iso /vm/user-data /vm/meta-data
"""
IMAGE = "crucible-vm:" + hashlib.sha256(DOCKERFILE.encode()).hexdigest()[:12]
GOLDEN_VOLUME = "crucible-vm-golden"
SSH = "ssh -i /vm/key -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=5 root@127.0.0.1"


async def ensure_image(docker_host: str | None = None, image: str = IMAGE) -> str:
    env = {"DOCKER_HOST": docker_host} if docker_host else None
    if (await _run_subprocess(["docker", "image", "inspect", image], cwd=None, env=env, timeout_s=60, stdin=None)).ok:
        return image
    out = await _run_subprocess(["docker", "build", "-t", image, "-"], cwd=None, env=env, timeout_s=3600, stdin=DOCKERFILE.encode())
    if not out.ok:
        raise RuntimeError(f"building {image} failed: {(out.stderr or out.stdout)[-800:]}")
    return image


class VM:
    """One QEMU/KVM guest in its own container. ``boot()`` resumes the golden ``ready`` state (building it on first use)."""

    def __init__(self, *, mem_mb: int = 1024, cpus: int = 2, docker_host: str | None = None, image: str | None = None, golden: str = GOLDEN_VOLUME):
        self.mem_mb, self.cpus, self.docker_host, self.image, self.golden = mem_mb, cpus, docker_host, image, golden
        self.name: str | None = None
        self.dexec: DockerExec | None = None
        self.timings: dict[str, float] = {}

    def _env(self) -> dict[str, str] | None:
        return {"DOCKER_HOST": self.docker_host} if self.docker_host else None

    def _qemu(self, *, loadvm: bool) -> str:
        return (f"qemu-system-x86_64 -enable-kvm -cpu host -m {self.mem_mb} -smp {self.cpus} -drive file=/vm/disk.qcow2,if=virtio,cache=unsafe "
                "-drive file=/vm/seed.iso,if=virtio,format=raw,readonly=on "
                "-netdev user,id=n0,restrict=on,hostfwd=tcp:127.0.0.1:2222-:22 -device virtio-net-pci,netdev=n0 "
                "-display none -serial file:/tmp/serial.log -qmp unix:/tmp/qmp.sock,server=on,wait=off -daemonize -pidfile /tmp/qemu.pid"
                + (" -loadvm ready" if loadvm else ""))

    async def run(self, cmd: str, timeout_s: float = 120):
        assert self.dexec is not None
        return await self.dexec.run(cmd, timeout_s=timeout_s)

    async def qmp(self, *commands: dict[str, Any]) -> list[dict[str, Any]]:
        """QMP commands in one session (capabilities negotiated first); returns the replies."""
        lines = [json.dumps({"execute": "qmp_capabilities"})] + [json.dumps(c) for c in commands]
        out = await self.run(f"printf '%s\\n' {' '.join(shlex.quote(x) for x in lines)} | socat -t 600 - UNIX-CONNECT:/tmp/qmp.sock", timeout_s=900)
        replies = [json.loads(x) for x in out.stdout.splitlines() if x.strip().startswith("{")]
        return [r for r in replies if "return" in r or "error" in r][1:]

    async def hmp(self, command: str) -> str:
        r = await self.qmp({"execute": "human-monitor-command", "arguments": {"command-line": command}})
        if not r or "error" in r[0]:
            raise RuntimeError(f"QMP {command!r} failed: {r}")
        out = str(r[0].get("return", ""))
        if "Error" in out or "error" in out.lower():
            raise RuntimeError(f"{command}: {out.strip()}")
        return out

    async def wait_ssh(self, timeout_s: float = 300) -> float:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if (await self.run(f"{SSH} true", timeout_s=15)).ok:
                return time.monotonic() - t0
            await asyncio.sleep(0.5)
        log = (await self.run("tail -c 2000 /tmp/serial.log")).stdout
        raise RuntimeError(f"the guest's SSH never came up ({timeout_s}s); serial tail: {log}")

    async def start_container(self) -> None:
        image = self.image or await ensure_image(self.docker_host)
        self.name = f"crucible-vm-{uuid.uuid4().hex[:10]}"
        out = await _run_subprocess(["docker", "run", "-d", "--rm", "--init", "--network", "none", "--device", "/dev/kvm", "-v", f"{self.golden}:/golden",
                                     "--name", self.name, "--entrypoint", "", image, "sleep", "infinity"], cwd=None, env=self._env(), timeout_s=300, stdin=None)
        if not out.ok:
            raise RuntimeError(f"VM container failed to start: {(out.stderr or out.stdout)[-500:]}")
        self.dexec = DockerExec(self.name, docker_host=self.docker_host)

    async def build_golden(self) -> None:
        """Cold boot, wait for cloud-init and SSH, save ``ready`` (RAM + disk), publish the overlay to the golden volume."""
        t0 = time.monotonic()
        await self.run("qemu-img create -q -f qcow2 -b /vm/base.qcow2 -F qcow2 /vm/disk.qcow2 8G")
        out = await self.run(self._qemu(loadvm=False))
        if not out.ok:
            raise RuntimeError(f"qemu failed: {out.stderr[-800:]}")
        await self.wait_ssh(600)
        await self.run(f"{SSH} 'cloud-init status --wait >/dev/null 2>&1; sync'", timeout_s=600)
        self.timings["cold_boot_s"] = time.monotonic() - t0
        t1 = time.monotonic()
        await self.hmp("savevm ready")
        self.timings["savevm_s"] = time.monotonic() - t1
        await self.run("cp /vm/disk.qcow2 /golden/ready.qcow2.tmp && mv /golden/ready.qcow2.tmp /golden/ready.qcow2")

    async def boot(self) -> None:
        await self.start_container()
        t0 = time.monotonic()
        lock = await self.run("mkdir /golden/.building 2>/dev/null && echo mine || echo other")
        if not (await self.run("test -f /golden/ready.qcow2")).ok:
            if lock.stdout.strip() == "mine":
                try:
                    await self.build_golden()
                finally:
                    await self.run("rmdir /golden/.building")
                self.timings["boot_s"] = time.monotonic() - t0
                return
            for _ in range(1200):  # another VM is building the golden overlay
                if (await self.run("test -f /golden/ready.qcow2")).ok:
                    break
                await asyncio.sleep(1)
        elif lock.stdout.strip() == "mine":
            await self.run("rmdir /golden/.building")
        await self.run("cp /golden/ready.qcow2 /vm/disk.qcow2")
        out = await self.run(self._qemu(loadvm=True))
        if not out.ok:
            raise RuntimeError(f"qemu -loadvm failed: {out.stderr[-800:]}")
        self.timings["ssh_after_loadvm_s"] = await self.wait_ssh(120)
        self.timings["boot_s"] = time.monotonic() - t0

    async def reset(self) -> float:
        """Back to ``ready`` (RAM + disk) in this container: a QMP loadvm."""
        t0 = time.monotonic()
        await self.hmp("loadvm ready")
        await self.wait_ssh(60)
        return time.monotonic() - t0

    async def exec(self, cmd: str, timeout_s: float = 60) -> dict[str, Any]:
        out = await self.run(f"timeout {int(timeout_s)} {SSH} {shlex.quote(cmd)}", timeout_s=timeout_s + 30)
        code = out.rc
        if code == 255 and not out.stdout:  # SSH itself failed (the guest is down): infrastructure, not the command
            raise RuntimeError(f"guest unreachable: {out.stderr[-300:]}")
        return {"exit_code": code, "stdout": out.stdout, "stderr": out.stderr}

    async def shutdown(self) -> None:
        if self.name:
            await _run_subprocess(["docker", "rm", "-f", self.name], cwd=None, env=self._env(), timeout_s=120, stdin=None)
            self.name, self.dexec = None, None


class VMPool:
    """N pre-booted VMs at ``ready``. ``acquire()`` hands out a warm VM; ``release(vm)`` resets it in the background."""

    _pools: ClassVar[dict[str, VMPool]] = {}

    def __init__(self, size: int, **vm_kw: Any):
        self.size, self.vm_kw = size, vm_kw
        self.free: asyncio.Queue[VM] = asyncio.Queue()
        self.vms: list[VM] = []
        self.stats = {"acquired": 0, "resets": 0, "reset_s": [], "boot_s": []}
        self._bg: set[asyncio.Task] = set()

    @classmethod
    def named(cls, name: str, size: int = 2, **kw: Any) -> VMPool:
        if name not in cls._pools:
            cls._pools[name] = cls(size, **kw)
        return cls._pools[name]

    async def start(self) -> None:
        first = VM(**self.vm_kw)  # builds the golden overlay if the host has none; the rest resume it concurrently
        await first.boot()
        rest = [VM(**self.vm_kw) for _ in range(self.size - 1)]
        await asyncio.gather(*(v.boot() for v in rest))
        for v in [first, *rest]:
            self.vms.append(v)
            self.stats["boot_s"].append(round(v.timings["boot_s"], 2))
            self.free.put_nowait(v)

    async def acquire(self) -> VM:
        if not self.vms:
            await self.start()
        vm = await self.free.get()
        self.stats["acquired"] += 1
        return vm

    def release(self, vm: VM) -> None:
        async def reset() -> None:
            try:
                dt = await vm.reset()
            except Exception as e:  # noqa: BLE001 -- a VM that cannot reset is replaced, never handed out dirty
                self.stats.setdefault("replaced", []).append(f"{type(e).__name__}: {e}"[:200])
                await vm.shutdown()
                fresh = VM(**self.vm_kw)
                await fresh.boot()
                self.vms = [x for x in self.vms if x is not vm] + [fresh]
                self.free.put_nowait(fresh)
                return
            self.stats["resets"] += 1
            self.stats["reset_s"].append(round(dt, 2))
            self.free.put_nowait(vm)

        def done(t: asyncio.Task) -> None:
            self._bg.discard(t)
            if not t.cancelled() and t.exception() is not None:  # never silent: a lost VM shrinks the pool
                self.stats.setdefault("lost", []).append(repr(t.exception())[:200])

        t = asyncio.get_running_loop().create_task(reset())
        self._bg.add(t)
        t.add_done_callback(done)

    async def close(self) -> None:
        await asyncio.gather(*self._bg, return_exceptions=True)
        await asyncio.gather(*(v.shutdown() for v in self.vms))
        self.vms = []


class VMApp(App):
    template = "common/vm"
    PATH_PARAMS = ("path",)

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        p = spec.params
        self.vm: VM | None = None
        self.pool = VMPool.named(p["pool"], int(p.get("pool_size", 2)), mem_mb=int(p.get("mem_mb", 1024)), cpus=int(p.get("cpus", 2)),
                                 docker_host=p.get("docker_host")) if p.get("pool") else None
        self.journal: dict[str, Any] = {}
        self.snapshots: set[str] = set()

    async def _fresh(self) -> None:
        if self.vm is not None:
            if self.pool is not None:
                self.pool.release(self.vm)
            else:
                await self.vm.shutdown()
        p = self.spec.params
        if self.pool is not None:
            self.vm = await self.pool.acquire()
        else:
            self.vm = VM(mem_mb=int(p.get("mem_mb", 1024)), cpus=int(p.get("cpus", 2)), docker_host=p.get("docker_host"))
            await self.vm.boot()
        self.snapshots = set()
        for cmd in p.get("setup") or []:
            out = await self.vm.exec(cmd, timeout_s=float(p.get("setup_timeout_s", 300)))
            if out["exit_code"] != 0:
                raise RuntimeError(f"{self.id}: setup command failed ({out['exit_code']}): {cmd}: {out['stderr'][-300:]}")

    async def start(self) -> None:
        await self._fresh()
        self.journal = {"restore": self.spec.params.get("restore", "start"), "steps": []}
        await self._write()

    async def stop(self) -> None:
        if self.vm is not None:
            if self.pool is not None:
                self.pool.release(self.vm)
            else:
                await self.vm.shutdown()
            self.vm = None

    async def _write(self) -> None:
        path = self.state_params["path"]
        await self.ctx.exec.run(f"mkdir -p {shlex.quote(posixpath.dirname(path))}")
        await self.ctx.exec.write_text(path, json.dumps(self.journal, sort_keys=True, separators=(",", ":")))

    def _tag(self) -> str:
        return "s" + hashlib.sha256(json.dumps(self.journal["steps"], sort_keys=True).encode()).hexdigest()[:12]

    async def before_snapshot(self) -> None:
        """RAM + disk under a tag derived from the journal: the same history always maps to the same saved state."""
        assert self.vm is not None
        tag = self._tag()
        if tag not in self.snapshots:
            await self.vm.hmp(f"savevm {tag}")
            self.snapshots.add(tag)
        self.journal["vm_tag"] = tag
        await self._write()

    async def after_restore(self) -> None:
        doc = json.loads(await self.ctx.exec.read_text(self.state_params["path"]))
        tag = doc.get("vm_tag")
        if self.vm is not None and tag in self.snapshots:  # same container: exact RAM + disk
            await self.vm.hmp(f"loadvm {tag}")
            await self.vm.wait_ssh(60)
            self.journal = doc
            return
        if not doc["steps"]:  # the episode start: a ready VM from the pool (or a fresh boot)
            await self._fresh()
            self.journal = doc
            return
        if doc.get("restore", "start") != "replay":
            raise SnapshotMismatch(f"{self.id}: the VM's saved state is gone (another container); restore: replay rebuilds it by re-running the commands")
        await self._fresh()
        for s in doc["steps"]:
            await self._do(s["tool"], s["args"])
        self.journal = doc

    async def inspect(self, cmd: str, timeout_s: float = 60) -> dict[str, Any]:
        """A root shell in the guest, for graders (never a policy tool)."""
        assert self.vm is not None
        return await self.vm.exec(cmd, timeout_s=timeout_s)

    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(app=self.id, name="exec", description="Run a shell command in the virtual machine (as root). Returns exit code, stdout and stderr.",
                     input_schema={"type": "object", "properties": {"command": {"type": "string"}, "timeout_s": {"type": "number", "minimum": 1, "maximum": 600}},
                                   "required": ["command"], "additionalProperties": False}),
            ToolSpec(app=self.id, name="read_file", read_only=True, description="Read a text file in the virtual machine.",
                     input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}),
            ToolSpec(app=self.id, name="write_file", description="Write a text file in the virtual machine (creating directories).",
                     input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"],
                                   "additionalProperties": False}),
        ]

    async def _do(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        assert self.vm is not None
        cap = int(self.spec.params.get("max_output_chars", 20000))
        if tool == "exec":
            out = await self.vm.exec(str(args["command"]), timeout_s=float(args.get("timeout_s", 60)))
        elif tool == "read_file":
            out = await self.vm.exec(f"cat -- {shlex.quote(str(args['path']))}")
        elif tool == "write_file":
            b64 = base64.b64encode(str(args["content"]).encode()).decode()
            p = shlex.quote(str(args["path"]))
            out = await self.vm.exec(f"mkdir -p \"$(dirname {p})\" && echo {b64} | base64 -d > {p}")
        else:
            raise ToolError(f"unknown tool {tool}")
        return {**out, "stdout": out["stdout"][-cap:], "stderr": out["stderr"][-cap:]}

    async def call(self, name: str, args: dict[str, Any], *, actor: str, call_id: str) -> ToolResult:
        q = f"{self.id}.{name}"
        try:
            out = await self._do(name, args)
        except ToolError as e:
            return ToolResult(tool_call_id=call_id, function_name=q, ok=False, content=str(e), error="unknown_tool", fault="policy")
        except RuntimeError as e:  # the guest is unreachable: infrastructure, the episode is masked
            return ToolResult(tool_call_id=call_id, function_name=q, ok=False, content=f"internal error: {e}", error="transport", fault="infra")
        self.journal["steps"].append({"tool": name, "args": args, "exit_code": out["exit_code"],
                                      "digest": hashlib.sha256((out["stdout"] + "\0" + out["stderr"]).encode()).hexdigest()[:16]})
        await self._write()
        if name == "read_file" and out["exit_code"] == 0:
            content = out["stdout"]
        else:
            content = json.dumps({"exit_code": out["exit_code"], "stdout": out["stdout"], "stderr": out["stderr"]})
        ok = out["exit_code"] == 0 or name == "exec"  # a failing command is a normal observation; a failing read/write is an error
        return ToolResult(tool_call_id=call_id, function_name=q, ok=ok, content=content, data=out, error=None if ok else "tool_error", fault=None if ok else "tool")
