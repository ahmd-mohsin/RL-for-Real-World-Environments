"""The separate-verifier patch seam for an installed agent (M10 lab audit R-1, 2026-10-15; DEPEND: Harbor's task model and
environments, as ``inplace``).

A Harbor task whose verifier runs in its own environment (``[verifier].environment_mode = "separate"``, e.g. task02)
cannot be graded in the agent's container (``HarborInPlaceTask`` refuses it: a plant would score and a real solve could
fail, M11 audit H1 V2). Harbor itself hands the work over (trial.py:1443-1510): after the agent phase it runs the task's
``[[verifier.collect]]`` hooks in the agent's container, downloads the declared artifacts, and grades them in a fresh
verifier container. ``HarborPatchTask`` is that hand-off as a Crucible seam, the ``WorkspaceHarness`` shape:

- ``open``: a World on the task environment (``HarborInPlaceTask.open``; the agent phase admits ``agent_allowed_hosts``).
  No grader surface is captured (``grader_integrity="off"``): nothing is graded in this container;
- ``collect(world, dest)``: the task's main-service collect hooks, then ``/logs/artifacts/model.patch`` read through the
  container as root (base64 over the exec pipe), never through a host path the container controls: ``/logs/artifacts``
  is mode 777 (Harbor environments/base.py:604-605), ``docker cp`` copies a symlink as a symlink and the host's read
  followed it (a host file read into the record, or a mode-000 file the host cannot read turned into a masked retry that
  re-ran the agent; M10 lab audit R-1). Only a regular file within ``MAX_PATCH_BYTES`` is the patch. The read is framed
  and classified by the verdict the in-container shell prints (Harbor's Docker exec merges stderr into stdout and
  returns a docker or ssh failure as a non-zero exit, so neither the exit code nor the whole output is the read's); each
  hook is framed the same way (a start marker and its own ``rc`` verdict). A hook whose verdict is non-zero or that
  outlives its deadline, a read that outlives its deadline or frames something that is not the patch, or no patch, in a
  live container is the agent's outcome (its repository, its ``base`` tag, its shell and tools); a symlink or special
  file at the patch path is ``unsafe`` (``crucible:policy_misbehavior``); the harness asks the host first
  (``crucible:env_container_lost``). M10 lab audit R-1 review round 3: a root policy controls every binary in the
  container (Harbor runs each exec as the image's ``bash -c``, found through its PATH) and can forge any text the exec
  returns, so a missing report is the agent's only on evidence that can only blame it: the shell was killed by a signal
  after its marker, the container's runtime refused to start it (``bash`` or the working directory gone), or a fresh
  root shell in ``/`` still answers. An exec that raises fast (host side) is ``transport``, masked; anything else without
  the shell's report is ``shell``, masked with unknown blame (counted by the 5% stop). Neither is retried, since a retry
  would re-run the agent (``CollectError.kind``). The read runs in ``/``: the policy cannot remove it;
- ``grade_patch(patch, gdir)``: ``HarborDirTask.grade_patch`` in a fresh container of the task image (verify_dir.py:133),
  on any Docker host, inside the harness's ``PerfGuard`` slot when ``perf``. A masked grade is retried as a re-grade of
  the stored ``artifacts/model.patch``, never by re-running the agent; a hang or checker outage that repeats on it is
  decided by ``grade_control`` (``HarborDirTask``'s ``solution/model.patch``, M10 lab audit R-3).

``grade(world, ...)`` is refused (masked ``crucible:harness_unsupported``): a separate-verifier task is graded only from its
collected patch. Multi-step tasks and sidecar (non-main) collect hooks are refused at construction. ``QUIESCE = False``
(M10 lab audit R-2): no grade runs in the agent's container, so no process baseline is taken and a non-Docker provider
may host the agent; the collect is guarded by R-1's own stop of the agent's processes and the framed, root-run read.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import math
import re
import secrets
import shlex
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from crucible.interop.harbor.inplace import HarborInPlaceTask
from crucible.interop.harbor.verify_dir import HarborDirTask
from crucible.runtime.gpu import GpuHost

PATCH = "/logs/artifacts/model.patch"
MAX_PATCH_BYTES = 64 << 20  # M10 lab audit R-1: the patch crosses the exec pipe and is held in memory
_READ_TRIES = 3
_READ_BACKOFF_S = 1.0  # doubled per retry of the patch read
_READ_TIMEOUT_S = 300  # M10 lab audit R-1: one read's deadline (Harbor's exec raises at it)
_UNSAFE_RC, _MISSING_RC = 4, 3
# M10 lab audit R-1: the read's in-band frame. Harbor's Docker exec merges stderr into stdout and does not raise on a
# docker or ssh error (``stderr=STDOUT``, ``check=False``: docker.py:742-743, 1441-1446), so a compose warning (an
# obsolete ``version:``, an unset variable), a bash setlocale notice or the CLI's own error can come before the payload,
# and the exit code cannot tell the docker CLI's failure from the container's. Only the verdict line the in-container shell
# prints decides; ``printf`` is a shell builtin, so the policy cannot replace it through PATH
_BEGIN, _END, _SIZE = "CRUCIBLE-PATCH-BEGIN", "CRUCIBLE-PATCH-END", "CRUCIBLE-PATCH-SIZE"
_HOOK_BEGIN, _HOOK_END = "CRUCIBLE-HOOK-BEGIN", "CRUCIBLE-HOOK-END"
_PROBE = "CRUCIBLE-PROBE"
_PROBE_TIMEOUT_S = 60
# M10 lab audit R-1 review round 3: runc's refusal to start a process in the container's present state (``bash`` removed
# from the image, the working directory removed: "OCI runtime exec failed: ... executable file not found", "chdir to cwd
# ... no such file or directory"). Read only to blame the policy, never to excuse it: a policy that prints it itself
# blames itself
_NO_START = re.compile(r"executable file not found|chdir to cwd", re.IGNORECASE)


class CollectError(RuntimeError):
    """The agent's work could not be collected. ``kind``: ``"hook"`` (a collect hook's verdict was non-zero, it outlived
    its deadline, or its shell was killed or could not start in the container's state; the patch read outlived its
    deadline or its shell was stopped before its verdict while a fresh root shell answers: the container's shell and tools
    are the policy's), ``"missing"`` (no regular patch file within ``MAX_PATCH_BYTES`` after the hooks, the read's own
    verdict was non-zero, or its framed output was not the patch), ``"unsafe"`` (a symlink or special file at the patch
    path: refused, never followed), ``"transport"`` (the exec raised fast, a host-side failure: the read on every try, or
    a hook), ``"shell"`` (M10 lab audit R-1 review round 3: an exec returned without the in-container shell's report and
    nothing shows whose: a docker or ssh failure and a root policy's planted or replaced shell look the same in band).
    M10 lab audit R-1: ``"download"`` is gone; nothing is fetched through a host path."""

    def __init__(self, message: str, *, kind: str = "hook"):
        super().__init__(message)
        self.kind = kind


async def _exec_answers(env: Any, *, cwd: str | None = None, user: str | int | None = None) -> bool:
    """M10 lab audit R-1 review round 3: whether a fresh shell in the container (``cwd`` / ``user``: None is the exec's
    default, the agent's own context) prints a fresh nonce with the builtin ``printf``: the exec channel and the
    container's shell work now. An answer can only blame the policy (the failure was in what it controls), so a policy
    that fakes one blames itself; no answer is no evidence."""
    nonce = f"{_PROBE}-{secrets.token_hex(8)}"
    try:
        r = await asyncio.wait_for(env.exec(f"printf '%s\\n' {nonce}", cwd=cwd, timeout_sec=_PROBE_TIMEOUT_S, user=user), _PROBE_TIMEOUT_S + 30)
    except Exception:  # noqa: BLE001 -- no answer is no evidence
        return False
    return nonce in [s.strip() for s in ((getattr(r, "stdout", "") or "") + (getattr(r, "stderr", "") or "")).splitlines()]


def _hook_cmd(command: str) -> str:
    """M10 lab audit R-1 review round 3: the hook in a subshell between a ``_HOOK_BEGIN`` line and a ``_HOOK_END rc=<n>``
    verdict the outer shell prints after it (``printf`` is a builtin; the newlines keep a trailing comment or heredoc in
    the hook from eating the ``)``); the shell exits with the hook's code."""
    return f"printf '%s\\n' {_HOOK_BEGIN}\n(\n{command}\n)\nrc=$?; printf '\\n%s rc=%s\\n' {_HOOK_END} $rc; exit $rc"


def _hook_verdict(out: str) -> int | None:
    """M10 lab audit R-1 review round 3: the hook's exit code from its own verdict line (the last one after
    ``_HOOK_BEGIN``), or None when the outer shell never reported (a docker or ssh failure mid-hook carried the streamed
    marker and the CLI's own exit code, which was read as the hook's)."""
    lines = [s.strip() for s in out.splitlines()]
    try:
        i = lines.index(_HOOK_BEGIN)
    except ValueError:
        return None
    for j in range(len(lines) - 1, i, -1):
        m = re.fullmatch(rf"{_HOOK_END} rc=(\d+)", lines[j])
        if m:
            return int(m.group(1))
    return None


def _patch_read_cmd(path: str = PATCH, cap: int = MAX_PATCH_BYTES) -> str:
    """M10 lab audit R-1: the shell that prints the patch, framed: a ``_BEGIN`` line first, then (for a regular,
    non-symlink file of at most ``cap`` bytes) a ``_SIZE <bytes>`` line and the file as base64, and always last a
    ``_END rc=<n>`` verdict: 4 for a symlink or a special file (refused, never followed), 3 when there is none or it is
    over the cap, else base64's own exit. The shell exits with the same code. ``wc -c`` and ``base64`` are coreutils and
    busybox (``$n`` is unquoted: busybox pads it)."""
    q = shlex.quote(path)
    end = f"printf '\\n%s rc=%s\\n' {_END}"
    return (f"printf '\\n%s\\n' {_BEGIN}; "
            f"if [ -L {q} ] || {{ [ -e {q} ] && [ ! -f {q} ]; }}; then {end} {_UNSAFE_RC}; exit {_UNSAFE_RC}; fi; "
            f"if ! {{ [ -f {q} ] && n=$(wc -c < {q}) && [ $n -le {int(cap)} ]; }} 2>/dev/null; then {end} {_MISSING_RC}; exit {_MISSING_RC}; fi; "
            f"printf '%s %s\\n' {_SIZE} $n; base64 < {q}; rc=$?; {end} $rc; exit $rc")


def _read_verdict(out: str) -> tuple[int, list[str]] | None:
    """M10 lab audit R-1: (the verdict, the lines between the frame's markers) of a ``_patch_read_cmd`` output, or None
    when its first ``_BEGIN`` line or a ``_END rc=<n>`` line after it is missing (the in-container shell never reported:
    the docker CLI, the daemon or ssh failed, or the shell was killed or replaced). Anything before the frame (merged
    stderr) is ignored; the last verdict line is the shell's own, printed after base64 exited."""
    lines = [s.strip() for s in out.splitlines()]  # strip: a TTY's \r too
    try:
        i = lines.index(_BEGIN)
    except ValueError:
        return None
    for j in range(len(lines) - 1, i, -1):
        m = re.fullmatch(rf"{_END} rc=(\d+)", lines[j])
        if m:
            return int(m.group(1)), lines[i + 1:j]
    return None


def _payload(body: Sequence[str]) -> bytes:
    """M10 lab audit R-1: the patch bytes of a framed read whose verdict was 0: a ``_SIZE <n>`` line, then the base64
    lines. ValueError (``binascii.Error`` included) when the size line is missing, the base64 is not valid
    (``validate=True``: no character is silently dropped) or the decoded length is not ``n``."""
    parts = [s for s in body if s]
    m = re.fullmatch(rf"{_SIZE} (\d+)", parts[0]) if parts else None
    if m is None:
        raise ValueError(f"no {_SIZE} line in the framed read")
    data = base64.b64decode("".join(parts[1:]), validate=True)
    if len(data) != int(m.group(1)):
        raise ValueError(f"the read decoded {len(data)} bytes; the container counted {m.group(1)}")
    return data


class HarborPatchTask(HarborInPlaceTask):
    SEPARATE_OK = True
    QUIESCE = False  # M10 lab audit R-2: graded only in a fresh container from the collected patch (module docstring)

    def __init__(self, task_dir: Path, *, image: str, host: GpuHost | None = None, expected_image: str | None = None, ref: str | None = None,
                 perf: bool = False, timeout: float | None = None, docker_image: str | None = None, agent_allowed_hosts: Sequence[str] = (),
                 docker_host: str | None = None, gpu_host: str | None = None, environment_type: str = "docker", environment_kwargs: dict | None = None,
                 user: str | None = None, keep_containers: bool = False):
        """``image`` / ``host`` / ``expected_image`` / ``ref`` / ``perf`` / ``timeout``: the grading container
        (``HarborDirTask``). ``docker_image``: a prebuilt image for the agent's World (Harbor's prebuilt path, no rebuild).
        ``agent_allowed_hosts``: hosts the agent phase must admit (the capture proxy)."""
        from harbor.models.task.verifier_mode import VerifierEnvironmentMode, resolve_task_verifier_mode

        # grading never happens in the agent's container, so its grader surface is never needed (and a non-Docker provider
        # can host the agent)
        super().__init__(task_dir, docker_host=docker_host, gpu_host=gpu_host, environment_type=environment_type, environment_kwargs=environment_kwargs,
                         user=user, keep_containers=keep_containers, grader_integrity="off", agent_allowed_hosts=agent_allowed_hosts)
        cfg = self.task.config
        if resolve_task_verifier_mode(cfg) != VerifierEnvironmentMode.SEPARATE:
            raise ValueError(f"{self.dir}: the verifier shares the agent's environment; grade it in place (HarborInPlaceTask)")
        if cfg.steps:
            raise ValueError(f"{self.dir}: multi-step tasks are not supported on the patch seam")
        declared = [a if isinstance(a, str) else getattr(a, "source", None) for a in (cfg.artifacts or [])]
        if PATCH not in declared:
            raise ValueError(f"{self.dir}: the task declares no {PATCH} artifact (declared: {declared}); the patch seam grades only that patch")
        sidecar = [h.service for h in getattr(cfg.verifier, "collect", None) or [] if h.service != "main"]
        if sidecar:
            raise ValueError(f"{self.dir}: collect hooks in sidecar services {sidecar} are not supported (only the main container is collected)")
        if docker_image:
            self.backend_params["docker_image"] = docker_image
        self.grader = HarborDirTask(self.dir, image=image, host=host, expected_image=expected_image, ref=ref, perf=perf, timeout=timeout)
        self.id, self.perf = self.grader.id, perf

    def _declared_phases(self) -> tuple[Any, ...]:
        # the separate verifier's phase belongs to the grading container: this one switches only to the agent's
        return (self.plan.agent_phase,)

    async def grade(self, world, workdir: Path):
        return self._masked(self.id, time.monotonic(), "crucible:harness_unsupported",
                            "a separate-verifier task is graded only from its collected patch in a fresh container (grade_patch)")

    async def collect(self, world, dest: Path) -> str:
        """Harbor's main-service collect pass (trial.py:1443-1507), strict: a hook's failure is evidence, not a warning.
        Returns the patch text; the hook outputs go to ``dest/collect.log``."""
        env = world.backend.env
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        log: list[str] = []
        try:
            for hook in getattr(self.task.config.verifier, "collect", None) or []:
                deadline = float(hook.timeout_sec)
                t0 = time.monotonic()
                try:
                    # M10 lab audit R-1: framed (``_hook_cmd``): the marker shows the hook's shell started and the verdict
                    # is the hook's own exit (Harbor's exec returns a docker or ssh failure as a non-zero exit, not an
                    # exception, with the marker already streamed)
                    r = await env.exec(_hook_cmd(hook.command), timeout_sec=max(1, math.ceil(deadline)), user=hook.user)
                except Exception as e:  # noqa: BLE001 -- Harbor's Docker exec raises RuntimeError at the deadline (docker.py:785-787)
                    log.append(f"$ {hook.command}\n[{type(e).__name__}] {e}"[:2000])
                    # M10 lab audit R-1: it outlived its own deadline in the agent's repository (a git clean filter that
                    # sleeps): the agent's, as a non-zero exit is; it was an unknown-blame mask. Review round 3: decided by
                    # the time only: a provider's fast TimeoutError (aiohttp, a socket) is host side, not the hook's stall
                    if time.monotonic() - t0 >= deadline:
                        raise CollectError(f"collect hook outlived its {deadline:.0f}s deadline: {type(e).__name__}: {e}", kind="hook") from e
                    raise CollectError(f"collect hook could not be run: {type(e).__name__}: {e}", kind="transport") from e
                full = (r.stdout or "") + (r.stderr or "")
                out = full[-2000:]
                log.append(f"$ {hook.command}\n[rc {r.return_code}]\n{out}")
                rc = _hook_verdict(full)
                if rc is not None:  # the hook's own exit, whatever the exec's
                    if rc != 0:
                        raise CollectError(f"collect hook exited {rc}: {out[-600:]}", kind="hook")
                    continue
                raise await self._unreported_hook(env, hook, r.return_code, full, out)
            return await self._read_patch(env)
        finally:
            from crucible.core.secrets import scrub_secrets

            (dest / "collect.log").write_text(scrub_secrets("\n".join(log)))

    @staticmethod
    async def _unreported_hook(env: Any, hook: Any, exec_rc: Any, full: str, out: str) -> CollectError:
        """M10 lab audit R-1 review round 3: a hook exec that returned without the hook's verdict. The agent's (``hook``)
        only on evidence that can only blame it: the shell printed its marker and was killed by a signal inside the
        container (docker exec passes 128+n through; a host-side CLI or ssh failure exits 1, 255 or negative); runc refused
        to start it in the container's state (``_NO_START``: ``bash`` or the working directory removed); or a shell in the
        hook's own context still fails while a root shell in ``/`` answers (its working directory, user or shell). Else
        ``shell``: a docker or ssh failure (it was read as the hook's exit, so an ssh blip scored the policy 0) and a root
        policy's replaced shell look the same in band."""
        started = _HOOK_BEGIN in [s.strip() for s in full.splitlines()]
        if started and isinstance(exec_rc, int) and 128 < exec_rc <= 192:
            return CollectError(f"collect hook's shell was killed by signal {exec_rc - 128} in the agent's container: {out[-600:]}", kind="hook")
        if not started and _NO_START.search(full):
            return CollectError(f"collect hook's shell could not start in the agent's container (rc {exec_rc}): {out[-600:]}", kind="hook")
        if not started and not await _exec_answers(env, user=hook.user) and await _exec_answers(env, cwd="/", user="root"):
            return CollectError(f"collect hook's shell could not start in the agent's context (its working directory, user or shell; rc {exec_rc}) "
                                f"while a root shell answers: {out[-600:]}", kind="hook")
        return CollectError(f"collect hook returned no verdict (exec rc {exec_rc}): {out[-600:]}", kind="shell")

    @staticmethod
    async def _read_patch(env) -> str:
        """M10 lab audit R-1: the patch is read through the container as root, never through a host path the container
        controls (/logs/artifacts is mode 777, environments/base.py:604-605: docker cp copies a symlink as a symlink and the
        host's read follows it; a mode-000 file is unreadable to the host user). Only a regular, non-symlink file of at most
        MAX_PATCH_BYTES is the patch: a symlink or special file is ``unsafe``, anything else the agent's (``missing``).
        Classified by the read's own in-band verdict (``_read_verdict``), never by the exec's exit code or the whole of its
        output: Harbor's Docker exec merges stderr into stdout and returns a docker or ssh failure as a non-zero exit. An
        exec that raises fast, or that returns without the verdict, is retried (idempotent and cheap while the container is
        up, so a transient blip never loses the work); after ``_READ_TRIES`` it is ``transport`` when every try raised
        fast (host side). A read that outlives its deadline, or a frame whose verdict is 0 but whose body is not the patch,
        is the agent's: the container's shell and tools (``wc``, ``base64``) were the policy's to replace (P23), so neither
        may turn its 0 into an infra mask. M10 lab audit R-1 review round 3: nor may a missing verdict. A read whose shell
        printed its ``_BEGIN`` every time and was stopped before its verdict (a planted ``base64`` that kills its shell)
        while a fresh root shell answers, or whose shell runc could not start (``_NO_START``: ``bash`` removed), is the
        agent's (``hook``); anything else without a verdict is ``shell`` (unknown blame). The read runs in ``/``, which the
        policy cannot remove (Harbor's exec chdirs into the task's working directory, which it can), and its stall is
        decided by the time alone (a provider's fast TimeoutError is retried like any fast failure)."""
        last: BaseException | None = None
        tries: list[str] = []  # per try: "raised" (fast, host side), "started" (_BEGIN printed, no verdict), "unstarted"
        out = ""
        for i in range(_READ_TRIES):
            t0 = time.monotonic()
            try:
                r = await env.exec(_patch_read_cmd(), cwd="/", timeout_sec=_READ_TIMEOUT_S, user="root")
            except Exception as e:  # noqa: BLE001 -- retried; classified after the last try
                if time.monotonic() - t0 >= _READ_TIMEOUT_S:
                    raise CollectError(f"reading {PATCH} outlived its {_READ_TIMEOUT_S}s deadline in the agent's container (its shell or "
                                       f"tools stalled): {type(e).__name__}: {e}", kind="hook") from e
                last = e
                tries.append("raised")
            else:
                out = r.stdout or ""
                v = _read_verdict(out)
                if v is not None:
                    rc, body = v
                    if rc == _UNSAFE_RC:
                        raise CollectError(f"{PATCH} is a symlink or a special file: refused, never followed", kind="unsafe")
                    if rc != 0:  # 3: no regular patch within the cap; else the container's own tools (P23): the agent's
                        raise CollectError(f"no regular, non-symlink {PATCH} of at most {MAX_PATCH_BYTES} bytes after the collect hooks "
                                           f"(rc {rc}: {out[-300:]})", kind="missing")
                    try:
                        return _payload(body).decode("utf-8", errors="replace")
                    except (ValueError, binascii.Error) as e:
                        raise CollectError(f"{PATCH} read returned verdict 0 but a body that is not the framed base64 of the patch "
                                           f"(the container's own tools, P23): {e}", kind="missing") from e
                last = RuntimeError(f"exit {r.return_code} without the read's verdict: {out[-300:]}")
                tries.append("started" if _BEGIN in [s.strip() for s in out.splitlines()] else "unstarted")
            if i + 1 < _READ_TRIES:
                await asyncio.sleep(_READ_BACKOFF_S * 2 ** i)
        what = f"reading {PATCH} through the container failed {_READ_TRIES} times: {type(last).__name__}: {last}"
        if all(x == "raised" for x in tries):
            raise CollectError(what, kind="transport")
        if all(x == "started" for x in tries) and await _exec_answers(env, cwd="/", user="root"):
            raise CollectError(f"{what} (its shell started every time and was stopped before its verdict while a root shell answers: the "
                               "container's tools, P23)", kind="hook")
        if tries[-1] == "unstarted" and _NO_START.search(out):
            raise CollectError(f"{what} (the container's runtime could not start its shell)", kind="hook")
        raise CollectError(what, kind="shell")

    async def grade_patch(self, patch: str, gdir: Path):
        return await self.grader.grade_patch(patch, gdir)

    async def grade_control(self, workdir: Path):
        """M10 lab audit R-3: the grading container's control (``HarborDirTask.grade_control``: the task's
        ``solution/model.patch`` graded like the collected patch). Never the in-place control it would inherit: its
        ``grade`` is refused, so that control could never run cleanly."""
        return await self.grader.grade_control(workdir)

    @property
    def control_timeout_s(self):
        """M10 lab audit R-3: the grading container's control budget (``HarborDirTask.control_timeout_s``), never the
        in-place one it would inherit."""
        return getattr(self.grader, "control_timeout_s", None)
