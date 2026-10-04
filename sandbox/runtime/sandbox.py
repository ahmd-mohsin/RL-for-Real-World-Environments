"""Part 2: Runtime. One fresh, isolated container per rollout.

Reset is free: every rollout starts a new container from the same image and throws it away
afterwards, so no state leaks between attempts. Only the Docker CLI is used, no extra packages.
"""

from __future__ import annotations

import subprocess
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path

SANDBOX_DIR = Path(__file__).resolve().parent.parent
MAX_OUTPUT_CHARS = 20_000


def load_task(path: Path = SANDBOX_DIR / "task.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


@dataclass
class ExecResult:
    returncode: int
    output: str  # stdout and stderr merged, truncated from the middle if too long


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} characters truncated] ...\n{text[-half:]}"


class Sandbox:
    def __init__(self, task: dict | None = None):
        self.task = task or load_task()
        self.image = self.task["image"]["tag"]
        self.workdir = self.task["image"]["workdir"]
        self.name = f"rlsandbox-{uuid.uuid4().hex[:12]}"
        self.started = False

    # --- lifecycle --------------------------------------------------------------

    def start(self) -> None:
        rt = self.task["runtime"]
        subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "--name", self.name,
                "--network", rt["network"],
                "--cpus", str(rt["cpus"]),
                "--memory", f"{rt['memory_mb']}m",
                "--pids-limit", "512",
                "--workdir", self.workdir,
                self.image,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.started = True

    def stop(self) -> None:
        if self.started:
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
            self.started = False

    def __enter__(self) -> Sandbox:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # --- primitives every other part is built on -----------------------------------

    def exec(self, command: str, timeout: int | None = None, cwd: str | None = None) -> ExecResult:
        """Run a bash command inside the container. The timeout is enforced in the container
        so a runaway process is killed there, not just abandoned by the host."""
        timeout = timeout or self.task["runtime"]["command_timeout_sec"]
        wrapped = f"timeout --kill-after=5 {timeout} bash -lc {_shell_quote(command)}"
        try:
            proc = subprocess.run(
                ["docker", "exec", "-w", cwd or self.workdir, self.name, "bash", "-c", wrapped],
                capture_output=True,
                text=True,
                timeout=timeout + 30,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(124, f"command timed out after {timeout}s")
        output = proc.stdout + proc.stderr
        if proc.returncode == 124:
            output += f"\n[command timed out after {timeout}s]"
        return ExecResult(proc.returncode, _truncate(output))

    def write_file(self, container_path: str, content: str) -> None:
        """Copy text into the container byte-exactly (no shell quoting, no length limit)."""
        subprocess.run(
            ["docker", "exec", "-i", self.name, "bash", "-c",
             f"mkdir -p \"$(dirname {_shell_quote(container_path)})\" && cat > {_shell_quote(container_path)}"],
            input=content,
            text=True,
            check=True,
            capture_output=True,
        )

    def read_file(self, container_path: str) -> str:
        proc = subprocess.run(
            ["docker", "exec", self.name, "cat", container_path],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise FileNotFoundError(proc.stderr.strip() or container_path)
        return proc.stdout


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"
