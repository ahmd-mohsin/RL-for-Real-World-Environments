"""Harbor installed agents (mini-swe-agent, claude-code, codex, ...) as a Crucible harness (M10 lab audit R-1, 2026-10-15;
closes exit gate R3). DEPEND: Harbor's agent classes and environments (Apache-2.0), imported only here and lazily, inside
the adapter functions: ``import crucible`` and this module work without Harbor (tests drive it with fakes).

Never Harbor ``Trial`` / ``Job`` / ``TrialQueue`` or Harbor's retries: ``Trial`` turns any TimeoutError raised inside
``agent.run`` into ``AgentTimeoutError`` (trial.py:651-662, a provider's socket timeout would read as the policy's
timeout: P16); ``Job`` runs every trial in one TaskGroup, so one ``Trial.create`` error cancels its siblings
(job.py:1234); a retry rmtrees the trial (queue.py:222), the evidence a journal keeps. Crucible drives the agent class
itself inside the World its task adapter opened, as it already drives Harbor's ``OracleAgent`` (``inplace.oracle``), and
its own ``RetryRule`` alone decides retries. The cost: setup and run are sequenced here, and Harbor's live ATIF sync
(trial/sync_trajectory.py) is not used (the ATIF is read once, after the run).

Per attempt:

1. **checks**: the task must be ``HarborInPlaceTask`` / ``HarborPatchTask`` (``admits_agent_host``) on a single-step
   task.toml (``steps`` / ``multi_step``: a ``[[steps]]`` task applies its agent phases per step, so even a plain adapter
   on one is refused), and its agent phase must admit the capture proxy's host (``agent_allowed_hosts``): a refusal is
   ``crucible:harness_unsupported``, never retried, never run with an open network.
2. **key**: a per-attempt session on the capture proxy (``register(session, max_calls=limits.max_model_calls)``: OpenEnv
   sessions.py:158-178 found that no agent obeys its own step setting). The agent's endpoint variables are built from
   scratch (``endpoint_env`` or Harbor's ``MODEL_CONNECTION`` + ``PROVIDERS``): every key variable is the capture key and
   every base-URL variable the proxy as the container reaches it. They go into the agent's ``extra_env`` and into
   ``scoped_exec_env`` for setup and run only (the 0.24 API, environments/base.py:471): never into verification. A host
   credential visible in what the agent will see (``extra_env``, Harbor's ``model_connection.env``, which falls back to
   ``os.environ``) refuses the attempt (``crucible:harness_unsupported``) before any setup, and every other host credential
   variable is blanked in the agent's commands (``_agent_scope``: agents that read ``os.environ`` at run time). On a
   public agent phase the proxy-only rule holds through key scoping alone, not through the network (``agent_network`` is
   recorded): the capture key is useless at a provider and no host key reaches the container.
3. **setup** (harness-owned): ``agent.setup`` (the install) through ``task.open(before_agent=...)``, under its own deadline
   (``limits.setup_timeout_s``), under the World's baseline network, before the grader surface is captured and before the
   restricted agent phase. Expiry is ``crucible:backend_error`` (detail ``AgentSetupTimeoutError``, Harbor
   trial/errors.py:4), a crash ``crucible:backend_error`` or, when the host shows the container gone,
   ``crucible:env_container_lost``: never the policy's.
4. **agent phase**: ``agent.run`` under Crucible's deadline (``limits.timeout_s``, else the task's ``[agent].timeout_sec``,
   else 3600 s). Only that deadline's own expiry (``asyncio.timeout(...).expired()``) is a ``timeout`` stop; a
   TimeoutError raised inside the agent is classified by evidence like any other exception. The key is revoked the moment
   the phase ends (a request with it during verification gets 401), then the agent's processes are stopped
   (``kill_command``): asyncio cancellation stops only the host-side await, not mini-swe-agent and its commands in the
   container, which would otherwise keep working while the patch is collected. On the patch seam (the container is
   only collected, never graded) the default stop is, for a non-root agent user, that user's processes outside the
   container's init session (the keep-alive may run as that user), else the agent phase's process cohort by start time
   (marked just before the phase; a named user's uid is resolved then too), else the agent's own tree. The user and
   cohort stops run as root in ``/`` and read ``/proc`` with shell builtins (no procps), skip zombies, and kill and look
   again up to ``_STOP_ROUNDS`` times a second apart; the container's init session and every process older than the
   phase are spared, but the children such a service forks during the phase are killed every round and reported as
   ``respawned`` (a postmaster or an nginx master replacing the workers the stop killed), never blamed on the policy;
   a survivor in uninterruptible sleep (state D, SIGKILL pending) is waited for, then reported as ``uninterruptible``;
   a stop that cannot list the processes exits non-zero. M10 lab audit R-1 review round 3: the stop still runs under
   the image's ``bash`` (Harbor's ``bash -c``) and calls ``sleep`` (and ``id`` without a resolved uid), all found through
   a PATH a root policy can write, so it is trusted only when it prints its own report: one that never reports is run
   once more, then is the policy's (``broken``) when its shell started and was stopped while a fresh root shell answers,
   or when runc could not start it (``bash`` removed), else unknown. The collect is refused when processes of the phase
   survive the stop (``crucible:policy_misbehavior``), when only uninterruptible ones do (masked, never retried) and,
   for a run that did not end on its own, when the stop outlived its deadline or broke in a live container
   (``crucible:agent_broke_env``: the policy controlled its shell and tools) or failed otherwise (masked
   ``crucible:backend_error``, unknown blame when it never reported): never a patch from a live agent. In place the
   grade runs in this container, so only the agent program's own tree is stopped here (``AGENT_KILL``, never the user
   or cohort stop), before its evidence is read; the guarded grade then kills every other process started after
   ``open`` from the host and confirms it with the host's listing (M10 lab audit R-2, ``HarborInPlaceTask._verify``,
   receipt ``verifier/quiesce.json``). A user or cohort stop here would only repeat that kill with in-container
   evidence a root policy can forge. R-2's accepted trade-off: a service the agent left running for the tests is
   killed too, so a task whose tests need it fails in place. ``agent_run.json["processes"]`` records which stop ran
   (``scope``).
5. **evidence**: the agent's logs (downloaded when not bound) are first copied to a host-only ``agent_logs`` that follows
   nothing (``copy_untrusted_logs``: the policy writes ``/logs``; a link it planted was followed by the host's reads and by
   Harbor's converter's writes). Harbor's ``populate_context_post_run`` (its ATIF converter) runs on that copy, then
   mini-swe-agent's ``exit_status``, the proxy log (after the session's calls still in flight upstream are waited for,
   ``DRAIN_S``: ``calls_from_capture``, ``read_upstream_evidence``, confirmed by Crucible's own probe of the upstream when
   it would blame the upstream, never while the session's own calls are still in flight) and, when the agent ended abnormally, the host
   (``container_lost``) go through ``runtime.agent_evidence.classify_agent_end``: a non-policy blame needs proxy or host
   evidence; the agent's own claims never mask its 0; an abnormal or deadline end before any recorded upstream exchange
   is ``unknown`` (the policy never answered).
6. **records** (scrubbed): ``agent_run.json`` (agent, version, harness id, config digest, the installed Harbor version,
   model, proxy session, exit status, exception, upstream and host evidence, the process stop, the blanked variable names,
   the agent phase's network mode, outcome), ``calls.json``, ``agent_trajectory.atif.json`` (Harbor's converter output;
   never a ``trajectory.*`` artifact key, which the Runner parses as Crucible's own ATIF) and ``reconcile.json``
   (``interop.harbor.reconcile``: auxiliary calls are marked ``trainable=False``). ``grade["agent"]`` reaches every
   record's ``meta`` (the identity from the first line of ``run``, the evidence once the agent phase ends);
   ``grade["reconcile"]`` every record whose agent phase ran.
7. **grading**, always a guarded Crucible grader: a separate verifier (task02) is the patch seam (``task.collect`` in the
   agent's container, the World closed, then ``HarborDirTask.grade_patch`` on a pristine tree, through a composed
   ``WorkspaceHarness``, so a masked grade is retried as a re-grade of the stored ``artifacts/model.patch``, never by
   re-running the agent); a shared verifier is the guarded ``HarborInPlaceTask.grade``, which first quiesces the
   container host-side itself (M10 lab audit R-2: a process the policy started still listed after the bounded kill
   rounds is ``crucible:policy_misbehavior``, scored 0, flag ``policy_process_survived``; nothing is reset or verified
   before a clean receipt), so no caller can grade in place without it. Limit stops follow ``limit_policy`` (``score``
   by default, ``mask`` or ``penalty:<c>``), and only evidenced ones (Crucible's deadline, the proxy's cap) can be
   masked or penalized.
8. **spend**: settled from the session's capture log at the very end of every exit path, masked attempts included
   (``agent_evidence.capture_spend``, ``LLMClient._prices``' rule): every forwarded row, retries included, and a usage-less
   or cut call booked as an uncertain bound; self-hosted is decided per upstream (``self_hosted_apis``). ``budget_usd``
   trips on a Harbor run.

The id is ``harbor:<agent>@<version>#<12 hex>``: a digest of the agent name (the full import path of a ``module:Class``
agent), agent kwargs, limits, limit policy, model, Harbor version and explicit endpoint variables, so any configuration
change makes new SampleKeys and new GRPO groups.

Harbor API: coded against 0.24.0 (``external-repos/harbor``) while the lock pins 0.23.0. The 0.24-only surfaces
(``TrialAgentConfig.extra_allowed_hosts``, ``BaseEnvironment.scoped_exec_env``, ``MODEL_CONNECTION`` / ``PROVIDERS``,
``AgentFactory.get_agent_class_from_config``) sit behind the adapters below and fail closed with the installed version
named; the installed version is recorded on every attempt.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import importlib
import json
import os
import re
import shlex
import shutil
import stat
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from crucible.core.schema.run import Failure
from crucible.interop.agentenv.harness import calls_from_capture
from crucible.interop.harbor.patch_task import _NO_START, CollectError, _exec_answers
from crucible.interop.harbor.reconcile import reconcile
from crucible.runtime.agent_evidence import (
    CLEAN_EXITS,
    DRAIN_S,
    AgentEnd,
    capture_spend,
    classify_agent_end,
    confirm_upstream,
    error_names,
    host_secret_names,
    leaked_host_secrets,
    read_upstream_evidence,
    self_hosted_apis,
)
from crucible.runtime.collect import Collected
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.harness.native import LIMIT_STOPS, _take_grade_findings
from crucible.runtime.harness.workspace import WorkspaceHarness
from crucible.runtime.infra import LimitPolicy, StageFailure, boundary, classify_verifier

DEFAULT_TIMEOUT_S = 3600.0
_STOP_TIMEOUT_S = 60  # M10 lab audit R-1: the in-container stop's own deadline (Harbor's exec raises at it)
_STOP_ROUNDS = 5  # M10 lab audit R-1: kill-and-look-again rounds, a second apart, before a process still found survives
# M10 lab audit R-1 review round 3: further rounds while every process still found is in uninterruptible sleep (state D:
# SIGKILL pending, a driver or I/O teardown such as a CUDA context's release); within _STOP_TIMEOUT_S with the scans
_STOP_D_WAIT = 30
_STOP_BEGIN = "CRUCIBLE-STOP-BEGIN"  # M10 lab audit R-1 review round 3: printed first by the user and cohort stops (builtin)
_STOP_REASONS = ("no /proc", "no uid", "no pgrep")  # a stop's own exit-3 reasons
# the agent's whole process tree, found before anything is killed (a killed parent's children are reparented and would
# escape a second look). The bracket keeps the pattern from matching the shell that runs it (B.5 row 8: a self-matching
# ``pkill -f`` killed its own control channel). M10 lab audit R-1: exit 3 without pgrep (procps, absent from slim
# images), never a stop that found nothing reported as run
_TREE_KILL = ("command -v pgrep >/dev/null 2>&1 || {{ echo 'no pgrep'; exit 3; }}; "
              "pids=$(pgrep -f {pattern} 2>/dev/null); all=\"$pids\"; "
              "while [ -n \"$pids\" ]; do pids=$(for p in $pids; do pgrep -P \"$p\" 2>/dev/null; done); all=\"$all $pids\"; done; "
              "[ -z \"$(echo $all)\" ] || kill -KILL $all 2>/dev/null; echo \"killed:$(echo $all)\"")
AGENT_KILL = {"mini-swe-agent": _TREE_KILL.format(pattern=shlex.quote("[m]ini-swe-agent"))}
# M10 lab audit R-1: the /proc reads of the user and cohort stops use shell builtins only (``read``, ``printf``, ``case``,
# ``kill``: POSIX sh, dash, bash, busybox): no procps (absent from slim images, where the user stop found nothing and
# still reported a stop that ran) and no ``cat`` looked up through a PATH a root policy can write (a planted ``cat`` that
# hangs stalled the stop). Review round 3: that covers the /proc reads only. The stop itself runs under the image's
# ``bash`` (Harbor's ``bash -c``), ``_ROUNDS`` calls ``sleep`` and ``_USER_KILL`` calls ``id`` (without a uid resolved
# before the phase), all found through that PATH: the stop is trusted only when it prints its own report
# (``_stop_reported``), and one that stalls, or never reports in a container where a fresh root shell answers, is the
# policy's (``_stop_agent``). Every line of /proc/<pid>/stat is joined (a comm may hold a newline) and the fields are read
# past the last ') ' (a comm may hold spaces or parentheses): ``$1`` the state (Z, X, x: a zombie or a dead process, never
# a survivor; D: uninterruptible sleep), ``$2`` the parent, ``$4`` the session, ``$20`` the start time (field 22). No
# readable /proc: exit 3.
_PROC = "printf '%s\\n' " + _STOP_BEGIN + "; [ -r /proc/self/stat ] || {{ echo 'no /proc'; exit 3; }}; "
_STAT = ("s=$(exec 2>/dev/null; while IFS= read -r l || [ -n \"$l\" ]; do printf '%s ' \"$l\"; done < $d/stat) || continue; "
         "s=${{s##*) }}; set -- $s; case $1 in Z|X|x) continue;; esac; ")
# M10 lab audit R-1: kill what the scan found (``o``), then look again a second later (forks in flight, a SIGKILL not yet
# delivered), up to _STOP_ROUNDS times, then up to _STOP_D_WAIT more while only uninterruptible processes remain; what
# is still found is reported on one last line, ``left:`` last, and the stop exits 1. Review round 3: the scan sorts what
# it found into ``r`` (respawned: forked by a service older than the phase), ``w`` (uninterruptible: state D) and ``l``
# (left: the phase's own), so only ``left`` is blamed on the policy
_ROUNDS = ("scan; k=; i=0; while [ -n \"$o\" ] && [ $i -lt " + str(_STOP_ROUNDS) + " ]; do k=\"$k$o\"; kill -KILL $o 2>/dev/null; sleep 1; scan; "
           "i=$((i+1)); done; i=0; while [ -n \"$o\" ] && [ -z \"$l$r\" ] && [ $i -lt " + str(_STOP_D_WAIT) + " ]; do kill -KILL $o 2>/dev/null; "
           "sleep 1; scan; i=$((i+1)); done; echo \"killed:$k respawned:$r uninterruptible:$w left:$l\"; [ -z \"$o\" ]")
# M10 lab audit R-1: the agent user's processes outside the container's init session (sid 1). Harbor's keep-alive
# (``sh -c "sleep infinity"``, a child of docker-init under the overlay's ``init: true``) runs as the image's USER, which
# can be the agent user (``USER agent`` plus ``[agent] user = "agent"``). A bare ``pkill -u`` killed it, PID 1 exited and
# the stopped container was graded as the policy's ``agent_broke_env``. Every ``docker exec`` starts its own session (runc
# setsid) and no process can join session 1, so the agent's commands, detached ones included, are all outside it. A
# process is the user's when the real or effective uid on its /proc/<pid>/status ``Uid:`` line is the user's (``id -u``
# for a name; exit 3 when it does not resolve; review round 3: the harness passes the uid it resolved as root before the
# phase, so no ``id`` the policy planted runs). Every match is killed and reported as ``left`` (``uninterruptible`` in
# state D); none is ``respawned`` (a service older than the phase does not run as the agent user outside session 1).
_USER_KILL = (_PROC + "case {user} in *[!0-9]*) u=$(id -u {user} 2>/dev/null) || {{ echo 'no uid'; exit 3; }};; *) u={user};; esac; "
              "scan() {{ o= r= w= l=; for d in /proc/[0-9]*; do p=${{d#/proc/}}; case $p in 1|$$|$PPID) continue;; esac; "
              "e=$(exec 2>/dev/null; while read -r a b c r; do case $a in (Uid:) echo \"$b $c\";; esac; done < $d/status) || continue; "
              "case \" $e \" in *\" $u \"*) ;; *) continue;; esac; " + _STAT + "[ \"$4\" = 1 ] && continue; o=\"$o $p\"; "
              "case $1 in D) w=\"$w $p\";; *) l=\"$l $p\";; esac; done; }}; " + _ROUNDS)
# M10 lab audit R-1: the agent phase's start mark, the start time (clock ticks, /proc field 22) of this probe itself, taken
# as root just before the phase; its comm is ``cut``, which has no space, so the field can be cut directly
_PHASE_MARK = "cut -d' ' -f22 /proc/self/stat"
# M10 lab audit R-1: a root (or user-less) agent's stop: every process that started at or after the mark, except PID 1 and
# this shell, found through /proc only (no procps), killed and looked for again (``_ROUNDS``). Never ``kill -KILL -1``:
# processes older than the phase (PID 1's keep-alive child, boot services) are spared, because a container stopped from
# inside reads as the policy's in ``container_lost`` (world_backend.py:399), which would blame the policy for the
# harness's own kill (P16). M10 lab audit R-1 review round 3: every process started since the mark is still killed, but
# its blame follows its parent (``q``, field 4, read with the same builtin reader): a parent other than PID 1 that started
# before the mark is a service older than the phase (a postmaster's crash restart, an nginx master or Apache prefork
# replacing workers), so the child is ``respawned``, never the policy's; a parent in the cohort, PID 1 (an orphan, a
# double fork) or an unreadable one leaves it ``left``; state D is ``uninterruptible``
_COHORT_KILL = (_PROC + "scan() {{ o= r= w= l=; for d in /proc/[0-9]*; do p=${{d#/proc/}}; case $p in 1|$$|$PPID) continue;; esac; " + _STAT +
                "[ \"${{20:-0}}\" -ge {mark} ] 2>/dev/null || continue; o=\"$o $p\"; case $1 in D) w=\"$w $p\"; continue;; esac; q=$2; "
                "if [ \"$q\" != 1 ] && t=$(exec 2>/dev/null; while IFS= read -r x || [ -n \"$x\" ]; do printf '%s ' \"$x\"; done < /proc/$q/stat) "
                "&& t=${{t##*) }} && set -- $t && [ \"${{20:-{mark}}}\" -lt {mark} ] 2>/dev/null; then r=\"$r $p\"; else l=\"$l $p\"; fi; "
                "done; }}; " + _ROUNDS)
_LOGS_CAP = 256 * 1024**2


@dataclass(frozen=True)
class AgentLimits:
    """``max_model_calls``: the proxy's per-attempt call cap (0 = none); past it the proxy answers a stop itself and the
    attempt ends as an evidenced ``max_steps``, unless the proxy log shows the upstream failing (``provider_unavailable``,
    confirmed by Crucible's probe; M10 lab audit R-1). ``timeout_s``: Crucible's agent deadline (None: the task's
    ``[agent].timeout_sec``, else 3600 s). ``setup_timeout_s``: the install's own deadline."""

    max_model_calls: int = 0
    timeout_s: float | None = None
    setup_timeout_s: float = 1800.0


# ------------------------------------------------------------------------------------------- Harbor adapters (lazy)


def installed_harbor_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("harbor")
    except PackageNotFoundError:
        return "absent"


def harbor_agent_class(agent: str) -> type:
    """The agent class for a Harbor agent name (``mini-swe-agent``) or an import path (``pkg.mod:Class``).
    ``AgentFactory.get_agent_class_from_config`` takes ``harbor.models.trial.config.AgentConfig`` (0.24, factory.py:112);
    an older Harbor falls back to ``get_agent_class(AgentName(...))`` or a direct import."""
    from harbor.agents.factory import AgentFactory

    get = getattr(AgentFactory, "get_agent_class_from_config", None)
    if get is not None:
        from harbor.models.trial.config import AgentConfig as TrialAgentConfig

        return get(TrialAgentConfig(import_path=agent) if ":" in agent else TrialAgentConfig(name=agent))
    if ":" in agent:
        mod, _, name = agent.partition(":")
        return getattr(importlib.import_module(mod), name)
    from harbor.models.agent.name import AgentName

    return AgentFactory.get_agent_class(AgentName(agent))


def default_endpoint_env(cls: type, model: str) -> dict[str, str]:
    """Every key and base-URL variable the agent class and its provider read (Harbor ``MODEL_CONNECTION`` +
    ``PROVIDERS``, model_connection.py:180-260), as templates: ``{key}`` (the capture key) and ``{proxy}`` (the proxy as the
    container reaches it; ``/v1`` appended except for Anthropic, whose SDK appends it). Setting every one leaves Harbor's
    resolver nothing to take from ``os.environ``. For mini-swe-agent with ``openai/...``: ``MSWEA_API_KEY`` and
    ``OPENAI_API_KEY``, ``OPENAI_BASE_URL`` and ``OPENAI_API_BASE``."""
    try:
        from harbor.agents.model_connection import PROVIDERS
    except ImportError as e:
        raise ValueError(f"this Harbor ({installed_harbor_version()}) has no agents.model_connection (the 0.24 API): pass endpoint_env=...") from e
    try:
        from harbor.agents.model_connection import _PROVIDER_ALIASES as aliases
    except ImportError:
        aliases = {}
    spec = getattr(cls, "MODEL_CONNECTION", None)
    provider = getattr(spec, "default_provider", None) or (model.split("/", 1)[0] if "/" in model else None)
    provider = aliases.get(provider or "", provider)
    access = PROVIDERS.get(provider or "")
    keys = tuple(dict.fromkeys((*getattr(spec, "api_key_envs", ()), *(access.api_key_envs if access else ()))))
    urls = tuple(dict.fromkeys((*getattr(spec, "base_url_envs", ()), *(access.base_url_envs if access else ()))))
    if not keys or not urls:
        raise ValueError(f"cannot tell which variables {getattr(cls, '__name__', cls)} reads for model {model!r} (provider {provider!r}: key "
                         f"variables {list(keys)}, base-URL variables {list(urls)}): pass endpoint_env=...")
    url = "{proxy}" if provider == "anthropic" else "{proxy}/v1"
    return {**{k: "{key}" for k in keys}, **{u: url for u in urls}}


def _agent_context() -> Any:
    from harbor.models.agent.context import AgentContext

    return AgentContext()


def config_digest(**parts: Any) -> str:
    """The first 12 hex of sha256 over the canonical JSON of the parts (dataclasses as dicts)."""
    norm = {k: dataclasses.asdict(v) if dataclasses.is_dataclass(v) and not isinstance(v, type) else v for k, v in parts.items()}
    return hashlib.sha256(json.dumps(norm, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


def copy_untrusted_logs(src: Path, dest: Path, *, max_bytes: int = _LOGS_CAP, max_entries: int = 10_000, max_depth: int = 16) -> list[str]:
    """M10 lab audit R-1: a host-only copy of the agent's ``/logs/agent`` that follows nothing, so the host never opens a
    path the policy controls. ``/logs/{agent,artifacts}`` are mode 777 in the container (Harbor environments/base.py:581,
    605) and the agent's commands run as root; they are bind-mounted from the trial dir on a local daemon
    (world_backend.py:245-252) or fetched with ``docker compose cp`` (docker_unix.py:280-304), which copies a symlink as a
    symlink. Either way a link the policy planted resolved on the host: Harbor's converter writes ``logs_dir/trajectory.json``
    with ``Path.write_text`` (mini_swe_agent.py:462,720; claude_code.py:1669), overwriting the link's host target (the
    run's journal, a Docker config), and the harness's reads copied any host file into the records; a FIFO blocked them.
    Only directories and regular files are copied (opened ``O_NOFOLLOW`` relative to their parent's descriptor, re-checked
    with ``fstat``), within ``max_bytes`` in all, ``max_entries`` and ``max_depth``; every symlink, FIFO, socket, device,
    anything over a cap and any ``OSError`` is listed in the returned ``"<rel>: <why>"`` list and never followed. A
    platform without descriptor-relative calls copies nothing (fail closed)."""
    dest = Path(dest)
    if os.path.lexists(dest):
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    dest.mkdir(mode=0o700, parents=True)
    dropped: list[str] = []
    if not ({os.open, os.stat} <= os.supports_dir_fd and os.listdir in os.supports_fd):
        return [".: this platform cannot copy without following links: nothing copied"]
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    dflags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow
    fflags = os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOCTTY", 0)
    budget = [max_bytes, max_entries]

    def walk(dfd: int, out: Path, rel: str, depth: int) -> None:
        for name in sorted(os.listdir(dfd)):
            r = f"{rel}{name}"
            if budget[1] <= 0:
                dropped.append(f"{r}: over {max_entries} entries")
                continue
            budget[1] -= 1
            try:
                st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
                if stat.S_ISDIR(st.st_mode):
                    if depth >= max_depth:
                        dropped.append(f"{r}/: deeper than {max_depth}")
                        continue
                    sub = os.open(name, dflags, dir_fd=dfd)
                    try:
                        (out / name).mkdir(mode=0o700)
                        walk(sub, out / name, f"{r}/", depth + 1)
                    finally:
                        os.close(sub)
                elif stat.S_ISREG(st.st_mode):
                    fd = os.open(name, fflags, dir_fd=dfd)
                    try:
                        fst = os.fstat(fd)
                        if not stat.S_ISREG(fst.st_mode):  # swapped between the stat and the open
                            dropped.append(f"{r}: not a regular file when opened")
                            continue
                        if fst.st_size > budget[0]:
                            dropped.append(f"{r}: over the {max_bytes}-byte cap")
                            continue
                        with os.fdopen(os.dup(fd), "rb") as f:
                            data = f.read(budget[0] + 1)
                        if len(data) > budget[0]:
                            dropped.append(f"{r}: over the {max_bytes}-byte cap")
                            continue
                        budget[0] -= len(data)
                        w = os.open(out / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o600)
                        with os.fdopen(w, "wb") as f:
                            f.write(data)
                    finally:
                        os.close(fd)
                else:
                    dropped.append(f"{r}: {'symlink' if stat.S_ISLNK(st.st_mode) else 'special file'}, never followed")
            except OSError as e:  # ELOOP, ENOENT, EACCES: never followed
                dropped.append(f"{r}: {type(e).__name__}: {e.strerror or e}")

    try:
        top = os.open(src, dflags)
    except OSError as e:
        return [f".: {e.strerror or e}"]
    try:
        walk(top, dest, "", 0)
    finally:
        os.close(top)
    return dropped


def mini_swe_exit_status(logs: Path) -> str | None:
    """mini-swe-agent's ``info.exit_status`` (default.py:174; Harbor mini_swe_agent.py:675-677), or None. The file is in
    ``/logs/agent``, which the policy's shell can write: an agent-reported stop, never evidence (``agent_evidence``)."""
    doc = _read_json(Path(logs) / "mini-swe-agent.trajectory.json")
    info = doc.get("info") if isinstance(doc, dict) else None
    status = info.get("exit_status") if isinstance(info, dict) else None
    return status if isinstance(status, str) and status else None


@contextlib.contextmanager
def _agent_scope(env: Any, env_vars: Mapping[str, str], user: Any) -> Iterator[list[str]]:
    """The endpoint variables and the agent user, for setup and run only (Harbor trial.py:625-631, 1718); yields the names
    of the host credential variables blanked in the agent's commands.

    M10 lab audit R-1 (provider keys scrubbed): Harbor agents read the host's ``os.environ`` inside ``run`` and put the
    values into a command's own env, after the pre-setup refusal looked: cursor_cli.py:895-896 (``CURSOR_API_KEY``),
    eve.py:430-435 (every inferred auto-env key), claude_code.py:1789 (``CLAUDE_CODE_OAUTH_TOKEN`` through ``_get_env``'s
    ``os.environ`` fallback, agents/base.py:251-264). The scoped overlay wins over a command's own env
    (environments/base.py:452-467: persistent < per-exec < scoped), so every host credential name the template does not set
    is overlaid blank (the template wins on its own names). Not closed here: a host value copied under another name or
    embedded in the command string, host files an agent uploads (codex ``CODEX_FORCE_AUTH_JSON`` uploads
    ``~/.codex/auth.json``), and a blank that overrides a credential the task's own persistent env declares under the
    same name."""
    scoped = getattr(env, "scoped_exec_env", None)
    if scoped is None:
        raise StageFailure(Failure(kind="crucible:harness_unsupported", stage="seed", retryable=False,
                                   reason=f"this Harbor environment ({type(env).__name__}; Harbor {installed_harbor_version()}) has no scoped_exec_env "
                                          "(the 0.24 API, environments/base.py:471): the endpoint variables could not be kept off verification"))
    blank = {n: "" for n in host_secret_names() if n not in env_vars}
    with scoped({**blank, **env_vars}), env.with_default_user(user):
        yield sorted(blank)


def _reported_pids(out: str, field: str) -> list[int]:
    """M10 lab audit R-1: the PIDs a stop reported under ``field`` on its last report line (the last line ending in
    ``left:<pids>``; Harbor's exec merges stderr into the same stream, so only that line is read)."""
    for line in reversed(out.splitlines()):
        if re.search(r"\bleft:[0-9 ]*\s*$", line):
            m = re.search(rf"\b{field}:([0-9 ]*)", line)
            return [int(x) for x in m.group(1).split()] if m else []
    return []


def _survivors(out: str) -> list[int]:
    """M10 lab audit R-1: the PIDs of the agent's phase a stop reported as still running after its last round (``left:``)."""
    return _reported_pids(out, "left")


def _respawned(out: str) -> list[int]:
    """M10 lab audit R-1 review round 3: survivors forked by a service older than the phase (``respawned:``): never blamed
    on the policy."""
    return _reported_pids(out, "respawned")


def _uninterruptible(out: str) -> list[int]:
    """M10 lab audit R-1 review round 3: survivors still in uninterruptible sleep (``uninterruptible:``, state D, SIGKILL
    pending): exiting, not resisting."""
    return _reported_pids(out, "uninterruptible")


def _stop_reported(out: str, rc: Any) -> bool:
    """M10 lab audit R-1 review round 3: whether a non-zero stop printed its own report: one of its exit-3 reasons with
    exit 3, or its last report line (``killed:``) with exit 1 and a survivor in it (the user and cohort stops exit 1 only
    with one). A planted program that prints a report and kills the shell gives neither exit."""
    if rc == 3:
        return any(x in out for x in _STOP_REASONS)
    return rc == 1 and "killed:" in out and bool(_survivors(out) or _respawned(out) or _uninterruptible(out))


async def _container_lost(world: Any) -> str | None:
    """The host's reason when the task container is gone (``HarborBackend.container_lost``), else None; best effort."""
    lost = getattr(getattr(world, "backend", None), "container_lost", None)
    if lost is None:
        return None
    try:
        return await asyncio.wait_for(lost(), 120)
    except Exception:  # noqa: BLE001 -- asking the host is best effort; no answer is no evidence
        return None


def _agent_env_views(agent: Any) -> dict[str, str]:
    """What the agent will put in its commands' environment, one entry per source and name (``<source>:<NAME>``, so one
    source never hides another's value): its ``extra_env``, Harbor's resolved model connection (which falls back to
    ``os.environ``, base.py:251-289) and its env-var options."""
    out: dict[str, str] = {}
    for source, get in (("extra_env", lambda: getattr(agent, "extra_env", None)), ("model_connection", lambda: getattr(agent, "model_connection", None).env),
                        ("env_options", lambda: getattr(agent, "_resolved_env_vars", None))):
        try:
            got = get()
        except Exception:  # noqa: BLE001, S112 -- an agent without the attribute (or whose resolver raises) adds nothing
            continue
        if isinstance(got, Mapping):
            out.update({f"{source}:{k}": v for k, v in got.items() if isinstance(v, str)})
    return out


def _agent_summary(doc: Mapping[str, Any]) -> dict[str, Any]:
    """``grade["agent"]`` (the record's ``meta.agent``) from an ``agent_run.json`` document."""
    up = doc.get("upstream") or {}
    out = doc.get("outcome") or {}
    return {"agent": f"{doc.get('agent')}@{doc.get('version')}", "harness": doc.get("harness"), "config_digest": doc.get("config_digest"),
            "harbor_version": doc.get("harbor_version"), "model": doc.get("model"), "proxy_session": doc.get("proxy_session"),
            "exit_status": doc.get("exit_status"), "stop": out.get("stop"), "why": out.get("why"),
            "upstream": up.get("verdict") if isinstance(up, Mapping) else None, "limit_stops": doc.get("limit_stops", 0)}


# ------------------------------------------------------------------------------------------------------- the harness


class HarborAgentHarness:
    def __init__(self, agent: str, version: str, *, model: str, proxy: Any, proxy_url: str, agent_kwargs: dict[str, Any] | None = None,
                 limits: AgentLimits | None = None, limit_policy: LimitPolicy | None = None, actor: str = "agent",
                 endpoint_env: dict[str, str] | None = None, prices: dict[str, list[float]] | None = None, self_hosted: bool | None = None,
                 policy_version: int | None = None, perf_guard: Any = None, host: str = "local", harbor_version: str | None = None,
                 kill_command: str | None = None, agent_factory: Callable[..., Any] | None = None,
                 context_factory: Callable[[], Any] | None = None):
        """``agent``: a Harbor agent name or ``module:Class``; ``version``: the agent's version (installed as such).
        ``model``: the model name the agent is given (``provider/model``). ``proxy``: a ``CaptureProxy``; ``proxy_url``: that
        proxy as the task container reaches it (its host must be in the task's ``agent_allowed_hosts``).
        ``endpoint_env``: the agent's endpoint variables as ``{key}`` / ``{proxy}`` templates (default: derived from
        Harbor's model connection). ``prices`` / ``self_hosted`` / ``policy_version``: as ``A2AHarness``. ``perf_guard`` /
        ``host``: the patch seam's grading slot. ``kill_command``: how the agent's processes are stopped after its phase
        (default on the patch seam: a non-root agent user's processes outside the container's init session (the
        keep-alive may run as that user), else the agent phase's process cohort by start time, else the agent's own tree,
        else nothing; in place: the agent's own tree, and the grade's host-side quiesce (M10 lab audit R-2) kills the
        rest; ``""``: never). It must exit 0 only when it stopped them; it runs as root in ``/`` (M10 lab audit R-1
        review round 3: never in a working directory the policy can remove).
        ``harbor_version``,
        ``agent_factory(logs_dir=, model_name=, version=, extra_env=, **agent_kwargs)``, ``context_factory()``: for tests."""
        if not urlparse(proxy_url).hostname:
            raise ValueError(f"proxy_url {proxy_url!r} has no host: it must be the capture proxy as the task container reaches it")
        self.agent_name, self.version, self.model = agent, version, model
        self.proxy, self.proxy_url = proxy, proxy_url.rstrip("/")
        self.agent_kwargs = dict(agent_kwargs or {})
        self.limits = limits or AgentLimits()
        self.limit_policy = limit_policy or LimitPolicy()
        self.actor, self.endpoint_env = actor, dict(endpoint_env) if endpoint_env is not None else None
        self.prices, self.self_hosted, self.policy_version = prices, self_hosted, policy_version
        self.kill_command = kill_command
        self.agent_factory, self.context_factory = agent_factory, context_factory
        self.harbor_version = harbor_version or installed_harbor_version()
        # M10 lab audit R-1: the full agent name is digested, because the id keeps only the class of a ``module:Class`` agent
        # and two modules' ``MyAgent`` must not share SampleKeys or GRPO groups
        self.digest = config_digest(agent=agent, agent_kwargs=self.agent_kwargs, limits=self.limits, limit_policy=self.limit_policy, model=model,
                                    harbor=self.harbor_version, endpoint_env=self.endpoint_env)
        self.id = f"harbor:{agent.rsplit(':', 1)[-1]}@{version}#{self.digest}"
        # the separate-verifier seam reuses the workspace harness's grade, re-grade and prior-attempt rules unchanged
        self._patch = WorkspaceHarness("harbor-patch", "1", perf_guard=perf_guard, host=host)

    # ---- construction --------------------------------------------------------------------------------------------
    def _endpoint_template(self) -> dict[str, str]:
        if self.endpoint_env is not None:
            return dict(self.endpoint_env)
        if self.agent_factory is not None and not isinstance(self.agent_factory, type):
            raise ValueError("an agent_factory that is not an agent class needs endpoint_env=...")
        return default_endpoint_env(self.agent_factory or harbor_agent_class(self.agent_name), self.model)

    def _make_agent(self, logs: Path, env_vars: Mapping[str, str]) -> Any:
        make = self.agent_factory or harbor_agent_class(self.agent_name)
        return make(logs_dir=Path(logs), model_name=self.model, version=self.version, extra_env=dict(env_vars), **self.agent_kwargs)

    def _kill_command(self, task: Any, mark: int | None = None, *, in_place: bool = False, uid: int | None = None) -> str | None:
        """The stop after the agent phase (M10 lab audit R-1): an explicit ``kill_command`` (``""``: none). In place
        (``in_place``: the grade runs in this container) only the agent program's own tree (``AGENT_KILL``), stopped
        before its evidence is read; the grade's host-side quiesce (M10 lab audit R-2, ``HarborInPlaceTask._verify``)
        then kills every other process started after ``open``, a service the agent left running included, and decides
        from the host's listing. A user or cohort stop here would only duplicate it with in-container evidence a root
        policy can forge (review round 3). On the patch seam (the container is only
        collected): a non-root agent user's processes outside the container's init session (``_USER_KILL``: never the
        keep-alive; ``uid``: the user's uid resolved before the phase, review round 3); for a root or user-less agent with
        a phase ``mark``, every process started since (``_COHORT_KILL``, any agent, detached ones included); else the
        agent's own tree (``AGENT_KILL``); else nothing."""
        if self.kill_command is not None:
            return self.kill_command or None
        if in_place:
            return AGENT_KILL.get(self.agent_name.rsplit(":", 1)[-1])
        user = getattr(task, "agent_user", None)
        if user not in (None, "", "root", 0, "0"):  # every process of the agent's own user outside the container's init session
            return _USER_KILL.format(user=shlex.quote(str(uid if isinstance(uid, int) and not isinstance(uid, bool) else user)))
        if isinstance(mark, int) and not isinstance(mark, bool):
            return _COHORT_KILL.format(mark=mark)
        return AGENT_KILL.get(self.agent_name.rsplit(":", 1)[-1])

    async def _phase_mark(self, world: Any) -> int | None:
        """M10 lab audit R-1: the start time (clock ticks) of a probe run as root just before the agent phase: every process
        the phase starts is at least as late. None when it cannot be read (no /proc, a failed exec). The last all-digit
        line is the mark: Harbor's Docker exec merges stderr into stdout (docker.py:742-743), so a compose or shell warning
        can come first. Run in ``/`` (review round 3: the harness's own execs never depend on a directory the policy can
        remove)."""
        try:
            r = await asyncio.wait_for(world.backend.env.exec(_PHASE_MARK, cwd="/", timeout_sec=60, user="root"), 90)
            digits = [s.strip() for s in str(getattr(r, "stdout", "") or "").splitlines() if s.strip().isdigit()]
            return int(digits[-1]) if getattr(r, "return_code", 1) == 0 and digits else None
        except Exception:  # noqa: BLE001 -- no mark: the stop falls back to the agent's own tree
            return None

    async def _agent_uid(self, world: Any, task: Any) -> int | None:
        """M10 lab audit R-1 review round 3: a named agent user's uid, resolved as root in ``/`` just before the agent phase,
        before any policy command could plant an ``id`` (one that exits non-zero turned the user stop into an infra mask);
        None for a root, numeric or absent user, with an explicit ``kill_command``, or when it does not resolve (the stop
        then resolves it itself)."""
        user = getattr(task, "agent_user", None)
        if self.kill_command is not None or user in (None, "", "root", 0, "0") or str(user).isdigit():
            return None
        try:
            r = await asyncio.wait_for(world.backend.env.exec(f"id -u {shlex.quote(str(user))}", cwd="/", timeout_sec=60, user="root"), 90)
            digits = [s.strip() for s in str(getattr(r, "stdout", "") or "").splitlines() if s.strip().isdigit()]
            return int(digits[-1]) if getattr(r, "return_code", 1) == 0 and digits else None
        except Exception:  # noqa: BLE001 -- unresolved: the stop resolves it itself
            return None

    async def _stop_agent(self, world: Any, task: Any, mark: int | None = None, *, in_place: bool = False, uid: int | None = None) -> dict[str, Any]:
        """The agent's processes in the container, after its phase and before collect or grade (``_kill_command``), run as
        root in ``/``. M10 lab audit R-1: ``status`` is "ran" only when the stop exited 0; "failed" when its exec failed
        fast or it exited non-zero (it could not list the processes, or some survived it: ``left``, ``respawned``,
        ``uninterruptible``); "stalled" when its exec outlived ``_STOP_TIMEOUT_S`` (the container's shell or tools hung;
        decided by the time alone, review round 3: a provider's fast TimeoutError is a failed stop); "skipped" when there
        is no stop. ``scope``: "phase" (patch seam) or "agent_tree" (in place).

        Review round 3: a built-in stop that exited non-zero without its own report (``_stop_reported``) is run once more
        (it is idempotent; a docker or ssh blip passes). Still unreported, it is "broken" (the policy's: its shell printed
        ``_STOP_BEGIN`` both times and was stopped while a fresh root shell in ``/`` answers, a planted ``sleep`` or ``id``
        that kills it; or runc could not start it, ``bash`` removed), else "failed" with ``unreported`` (nothing shows
        whose: a channel failure and a replaced shell look the same in band)."""
        scope = "agent_tree" if in_place else "phase"
        cmd = self._kill_command(task, mark, in_place=in_place, uid=uid)
        if cmd is None:
            why = ("in place only the agent program's own tree is stopped, and this agent has none and no kill command" if in_place
                   else "no agent user, no phase mark and no kill command for this agent")
            return {"status": "skipped", "scope": scope, "mark": mark, "why": why}
        env = world.backend.env
        builtin = self.kill_command is None  # an explicit kill command's non-zero exit keeps its own meaning ("failed")
        outs: list[str] = []
        for _ in range(2 if builtin else 1):
            t0 = time.monotonic()
            try:
                r = await asyncio.wait_for(env.exec(cmd, cwd="/", timeout_sec=_STOP_TIMEOUT_S, user="root"), _STOP_TIMEOUT_S + 30)
            except Exception as e:  # noqa: BLE001 -- recorded; a container that cannot run it is checked by the host evidence
                stalled = time.monotonic() - t0 >= _STOP_TIMEOUT_S  # Harbor raises RuntimeError at its deadline; the outer wait_for later
                return {"status": "stalled" if stalled else "failed", "scope": scope, "mark": mark, "command": cmd, "error": f"{type(e).__name__}: {e}"[:300],
                        "left": [], "respawned": [], "uninterruptible": []}
            rc = getattr(r, "return_code", None)
            out = str(getattr(r, "stdout", "") or "")
            outs.append(out)
            if rc == 0 or not builtin or _stop_reported(out, rc):
                break
        doc = {"status": "ran" if rc == 0 else "failed", "scope": scope, "mark": mark, "command": cmd, "rc": rc, "output": out[-300:],
               "left": _survivors(out), "respawned": _respawned(out), "uninterruptible": _uninterruptible(out)}
        if rc != 0:
            doc["error"] = f"the stop exited {rc}: {out[-200:]}"
            if builtin and not _stop_reported(out, rc):
                began = [_STOP_BEGIN in [s.strip() for s in o.splitlines()] for o in outs]
                if (not began[-1] and _NO_START.search(out)) or (all(began) and await _exec_answers(env, cwd="/", user="root")):
                    doc["status"] = "broken"
                else:
                    doc["unreported"] = True
        return doc

    def _settle_spend(self, res: HarnessResult, path: Path) -> None:
        """M10 lab audit R-1: the session's spend from every row of its capture log (``capture_spend``, ``LLMClient._prices``'
        rule, self-hosted per upstream), as ``A2AHarness``: the retry-collapsed calls dropped billed retries."""
        if self.prices is not None:
            prices = self.prices
        else:
            try:
                from crucible.core.config import get_config

                prices = get_config().prices
            except Exception:  # noqa: BLE001 -- an unreadable config prices nothing: hosted calls are booked uncertain
                prices = {}
        free = self_hosted_apis(getattr(self.proxy, "upstreams", {}), policy_version=self.policy_version, self_hosted=self.self_hosted)
        res.cost_usd, res.cost_uncertain_usd = capture_spend(path, prices, self_hosted=free)

    def _identity(self, session: str | None = None) -> dict[str, Any]:
        """M10 lab audit R-1 (record per attempt): what every record of this harness carries in meta.agent, a refusal or a
        setup failure included (a 0.23 environment refused at _agent_scope is the case a by-version query must not lose):
        agent@version, harness id, config digest, Harbor version, model and, once registered, the proxy session.
        _agent_summary replaces it with the evidence when the agent phase ends."""
        return {"agent": f"{self.agent_name}@{self.version}", "harness": self.id, "config_digest": self.digest,
                "harbor_version": self.harbor_version, "model": self.model, "proxy_session": session}

    # ---- re-grade (patch seam) -------------------------------------------------------------------------------------
    async def regrade(self, attempt: Attempt, prior: Any) -> HarnessResult:
        """Grade the patch an earlier attempt stored (``Runner.regrade`` and grade-only retries). The prior attempt's
        reconcile report and agent summary are re-attached (and copied with its calls), so ``meta.reconcile`` and
        ``meta.agent`` survive the re-grade and a FATAL reconcile still masks a train rollout (``Runner._reconcile_mask``, on a
        grade-only retry and on ``Runner.regrade``)."""
        prior = Path(str(prior))
        res = await self._patch.regrade(attempt, prior)
        rc = _read_json(prior / "reconcile.json")
        if isinstance(rc, dict):
            res.grade["reconcile"] = rc
        doc = _read_json(prior / "agent_run.json")
        res.grade["agent"] = {**(_agent_summary(doc) if isinstance(doc, dict) else self._identity()), "regraded_from": prior.name}
        for name in ("reconcile.json", "agent_run.json", "calls.json"):  # a later re-grade of this attempt finds them here
            src, dst = prior / name, Path(attempt.dir) / name
            if src.exists() and not dst.exists():
                with contextlib.suppress(OSError):
                    shutil.copyfile(src, dst)
        return res

    # ---- one attempt ----------------------------------------------------------------------------------------------
    async def run(self, attempt: Attempt) -> HarnessResult:
        from crucible.core.secrets import scrub_json, scrub_secrets

        task = attempt.task
        patch_seam = hasattr(task, "collect") and hasattr(task, "grade_patch")
        if patch_seam:
            prior = self._patch._prior(attempt)
            if prior is not None and prior[0] == "regrade":  # a masked grade: never re-run the agent whose work was measured
                return await self.regrade(attempt, prior[1])
        t: dict[str, float] = {}
        res = HarnessResult()
        res.grade["agent"] = self._identity()  # M10 lab audit R-1: on every record, an attempt refused before setup included
        world: Any = None
        key: str | None = None
        session: str | None = None
        path: Path | None = None
        calls: list[dict[str, Any]] | None = None
        holder: dict[str, Any] = {}
        end: AgentEnd | None = None
        try:
            if not hasattr(task, "admits_agent_host"):
                raise StageFailure(Failure(kind="crucible:harness_unsupported", stage="seed", retryable=False,
                                           reason="the Harbor agent harness needs a HarborInPlaceTask or HarborPatchTask (an agent phase it can open to the proxy)"))
            # HarborMultiStepTask (``steps``) or any adapter on a [[steps]] task.toml (``multi_step``): each step opens its own
            # network phase and grade, which this harness does not drive, and ``HarborInPlaceTask.open`` applies no agent
            # phase to such a dir, so a plain adapter on one would run the agent under the baseline network (M10 lab audit R-1)
            if getattr(task, "steps", None) or getattr(task, "multi_step", False):
                raise StageFailure(Failure(kind="crucible:harness_unsupported", stage="seed", retryable=False,
                                           reason="multi-step Harbor tasks are not supported with installed agents"))
            host = urlparse(self.proxy_url).hostname
            if not task.admits_agent_host(host):
                raise StageFailure(Failure(kind="crucible:harness_unsupported", stage="seed", retryable=False,
                                           reason=f"the task's agent phase does not admit the capture proxy host {host!r}; construct the task with "
                                                  f"agent_allowed_hosts=[{host!r}]"))
            # one capture log per attempt run (the A2A naming rule); its path is known before the key exists, so spend
            # settles on every exit, an install that called the model and then timed out included
            tag = hashlib.sha256(str(Path(attempt.dir).resolve()).encode()).hexdigest()[:8]
            session = (re.sub(r"[^A-Za-z0-9._-]", "_", attempt.id).lstrip("._-") or "attempt") + f"-{tag}"
            path = Path(self.proxy.capture_dir) / f"{session}.jsonl"
            key = self.proxy.register(session, max_calls=self.limits.max_model_calls)
            res.grade["agent"]["proxy_session"] = session

            async def setup(w: Any) -> None:
                t0 = time.monotonic()
                try:
                    logs = await task.agent_logs(w)  # the host's trial dirs (a mkdir): an OSError there stays the boundary's
                    try:
                        env_vars = {k: str(v).replace("{key}", str(key)).replace("{proxy}", self.proxy_url) for k, v in self._endpoint_template().items()}
                        agent = self._make_agent(logs, env_vars)
                    # configuration: no retry fixes it. M10 lab audit R-1: Harbor's constructors read the host files their kwargs
                    # name (mini-swe-agent's config_file / litellm_model_registry_file, mini_swe_agent.py:556,579; an installed
                    # agent's config path, installed/base.py:692); a missing one reached boundary("seed") as
                    # transport_unreachable (infra, retried, a new container per attempt)
                    except (ImportError, ValueError, TypeError, OSError) as e:
                        raise StageFailure(Failure(kind="crucible:harness_unsupported", stage="seed", retryable=False,
                                                   reason=f"cannot build agent {self.agent_name}@{self.version} (Harbor {self.harbor_version}): "
                                                          f"{type(e).__name__}: {e}")) from e
                    leaked = leaked_host_secrets({**{f"endpoint_env:{k}": v for k, v in env_vars.items()}, **_agent_env_views(agent)}, allow={str(key)})
                    if leaked:
                        raise StageFailure(Failure(kind="crucible:harness_unsupported", stage="seed", retryable=False,
                                                   reason=f"the agent's environment would carry a host credential in {sorted(leaked)}: refused before "
                                                          "setup (set every key variable through endpoint_env)"))
                    if hasattr(agent, "session_id"):
                        agent.session_id = session
                    env = w.backend.env
                    cm = None
                    try:
                        async with asyncio.timeout(self.limits.setup_timeout_s) as cm:
                            with _agent_scope(env, env_vars, task.agent_user) as scrubbed:
                                holder["scrubbed"] = scrubbed  # names only, never values
                                await agent.setup(env)
                    except StageFailure:
                        raise
                    except Exception as e:  # noqa: BLE001 -- the install is the harness's: never the policy's outcome
                        if isinstance(e, TimeoutError) and cm is not None and cm.expired():
                            raise StageFailure(Failure(kind="crucible:backend_error", stage="seed", detail="AgentSetupTimeoutError",
                                                       reason=f"agent setup exceeded {self.limits.setup_timeout_s:.0f}s")) from e
                        lost = await _container_lost(w)
                        if lost:
                            raise StageFailure(Failure(kind="crucible:env_container_lost", stage="seed",
                                                       reason=f"agent setup failed and the task container is gone: {lost}")) from e
                        raise StageFailure(Failure(kind="crucible:backend_error", stage="seed", detail=f"agent_setup:{type(e).__name__}",
                                                   reason=f"agent setup failed: {type(e).__name__}: {scrub_secrets(str(e))[-400:]}")) from e
                    holder.update(agent=agent, env_vars=env_vars, logs=logs)
                finally:
                    t["setup"] = time.monotonic() - t0

            t0 = time.monotonic()
            with boundary("seed"):
                world = await task.open(attempt.dir / "work", before_agent=setup)
            t["seed"] = time.monotonic() - t0
            if "agent" not in holder:
                raise StageFailure(Failure(kind="crucible:harness_unsupported", stage="seed", retryable=False,
                                           reason="the task's open() never ran the agent setup (before_agent)"))
            agent, env_vars = holder["agent"], holder["env_vars"]
            env = world.backend.env

            # ---- agent phase: Crucible's deadline, the key revoked when it ends
            timeout = self.limits.timeout_s or getattr(task, "agent_timeout_s", None) or DEFAULT_TIMEOUT_S
            ctx = self.context_factory() if self.context_factory is not None else _agent_context()
            err: BaseException | None = None
            deadline = False
            cm = None
            mark = await self._phase_mark(world)  # M10 lab audit R-1: every process the phase starts is at least this late
            uid = await self._agent_uid(world, task) if patch_seam else None  # review round 3: before any policy command
            t0 = time.monotonic()
            try:
                async with asyncio.timeout(timeout) as cm:
                    with _agent_scope(env, env_vars, task.agent_user) as scrubbed:
                        holder["scrubbed"] = sorted({*holder.get("scrubbed", []), *scrubbed})
                        await agent.run(task.instruction, env, ctx)
            except StageFailure:
                raise
            except TimeoutError as ex:
                deadline = cm is not None and cm.expired()
                err = None if deadline else ex
            except Exception as ex:  # noqa: BLE001 -- classified by evidence below, never by its name alone
                err = ex
            finally:
                # M10 lab audit R-1: revoked when the agent phase ends, before anything reads or grades its work
                if key is not None:
                    self.proxy.unregister(key)
                    key = None
                t["agent"] = time.monotonic() - t0
            # M10 lab audit R-1: in place, only the agent's own tree here; M10 lab audit R-2: the grade's host-side quiesce
            # then kills every other process started after open (a service the agent left running too) before it grades
            killed = await self._stop_agent(world, task, mark, in_place=not patch_seam, uid=uid)

            # ---- evidence
            t0 = time.monotonic()
            logs: Path = holder["logs"]
            notes: list[str] = []
            try:
                logs = await task.agent_logs(world, fetch=True)
            except Exception as ex:  # noqa: BLE001 -- best effort: missing logs are no evidence, never a verdict
                notes.append(f"agent logs: {type(ex).__name__}: {ex}"[:300])
            # M10 lab audit R-1: the host never opens a path under the policy-writable /logs: everything below reads (and
            # Harbor's converter writes) only a host-only copy that followed nothing (copy_untrusted_logs)
            safe = attempt.dir / "agent_logs"
            try:
                dropped = copy_untrusted_logs(logs, safe)
            except OSError as ex:
                dropped = [f"<copy>: {type(ex).__name__}: {ex}"]
            if dropped:
                notes.append(f"agent logs not copied (link, special file or over the cap; never followed): {dropped[:20]}"[:600])
            logs = safe
            try:
                agent.logs_dir = safe  # Harbor's converter reads and writes logs_dir: only the host-only copy
                agent.populate_context_post_run(ctx)  # Harbor's ATIF converter writes logs/trajectory.json
            except Exception as ex:  # noqa: BLE001 -- the converter is best effort (Harbor's own trial swallows it too)
                notes.append(f"populate_context_post_run: {type(ex).__name__}: {ex}"[:300])
            exit_status = mini_swe_exit_status(logs)
            # M10 lab audit R-1 review round 3: a forwarded call whose client the stop killed still runs upstream (the
            # proxy's handler is not cancelled) and leaves no row until it ends, so the self-load guard did not see it and
            # the probe queued behind it. Waited for (they are read and billed too); what is left is recorded and never
            # probed past (``agent_evidence._self_loaded``)
            drain = getattr(self.proxy, "drain", None)
            inflight = await drain(session, DRAIN_S) if drain is not None and session is not None else 0
            calls = calls_from_capture(path, self.actor, self.policy_version)
            up = dataclasses.replace(read_upstream_evidence(path), inflight=inflight)
            abnormal = err is not None or deadline or (exit_status is not None and exit_status not in CLEAN_EXITS)
            if abnormal or up.limit_stops:  # M10 lab audit R-1: an upstream verdict counts only when Crucible's own probe agrees
                up = await confirm_upstream(up, getattr(self.proxy, "probe", None), up.model)
            lost = await _container_lost(world) if abnormal else None
            end = classify_agent_end(upstream=up, error=err, exit_status=exit_status, container_lost=lost, deadline=deadline,
                                     agent_is_harness=True, context=f"harbor agent {self.agent_name}@{self.version}")

            # ---- records
            atif = _read_json(logs / "trajectory.json")
            atif = atif if isinstance(atif, dict) else None
            if atif is not None:
                (attempt.dir / "agent_trajectory.atif.json").write_text(scrub_json(json.dumps(atif, default=str)))
                res.artifacts["agent_atif"] = "agent_trajectory.atif.json"
            rep = reconcile(atif, calls, limit_stops=up.limit_stops)
            aux = set(rep.aux_call_ids)
            for c in calls:
                if c.get("call_id") in aux:
                    c["trainable"] = False  # a call the agent did not log as a step (a summarizer, a title): never trained
            (attempt.dir / "reconcile.json").write_text(json.dumps(rep.to_dict(), indent=1, default=str))
            res.artifacts["reconcile"] = "reconcile.json"
            res.grade["reconcile"] = rep.to_dict()
            res.calls = {self.actor: calls}
            if calls:
                (attempt.dir / "calls.json").write_text(scrub_json(json.dumps(res.calls, default=str)))
                res.artifacts["calls"] = "calls.json"
            res.steps = len(calls)
            doc = {"agent": self.agent_name, "version": self.version, "harness": self.id, "config_digest": self.digest,
                   "harbor_version": self.harbor_version, "model": self.model, "proxy_session": session, "exit_status": exit_status,
                   "exception": {"type": type(err).__name__, "chain": error_names(err), "message": scrub_secrets(str(err))[:2000]} if err is not None else None,
                   "deadline": deadline, "timeout_s": timeout, "upstream": up.to_dict(), "container_lost": lost, "processes": killed,
                   "outcome": {"stop": end.stop, "failure": end.failure.kind if end.failure is not None else None, "why": end.why,
                               "evidenced": end.evidenced},
                   "limit_stops": up.limit_stops, "scrubbed": holder.get("scrubbed", []), "agent_network": _agent_network(task), "notes": notes}
            (attempt.dir / "agent_run.json").write_text(scrub_json(json.dumps(doc, default=str), indent=1))
            res.artifacts["agent_run"] = "agent_run.json"
            res.grade["agent"] = _agent_summary(doc)
            t["evidence"] = time.monotonic() - t0

            # ---- outcome
            if end.failure is not None:
                raise StageFailure(end.failure)
            res.stop, res.truncated = end.stop, end.stop in LIMIT_STOPS
            if res.stop in LIMIT_STOPS and end.rule(self.limit_policy) == "mask":  # as native.py:354-355; evidenced stops only
                raise StageFailure(Failure(kind="agent_timeout", stage="agent", reason=f"{res.stop} ({end.why})", mask=True, blame="harness"))

            # ---- grading
            if patch_seam:
                if killed.get("left"):
                    # M10 lab audit R-1: processes of the agent's phase (its user's outside the init session, or started since
                    # the mark) still running after _STOP_ROUNDS rounds of SIGKILL as root, zombies excluded (a fork chain):
                    # the policy's, and the collect would race them (P23)
                    raise StageFailure(Failure(kind="crucible:policy_misbehavior", stage="collect", blame="policy", mask=False,
                                               reason=f"the agent's processes survived the stop after its phase ({killed['left'][:20]}): "
                                                      "nothing collected (P23)"))
                if killed.get("uninterruptible"):
                    # M10 lab audit R-1 review round 3: only processes in uninterruptible sleep (state D, SIGKILL pending: a
                    # driver or I/O teardown) were left: exiting, not resisting, so never the policy's; nothing is collected
                    # while they may still write, and a retry would re-run the agent
                    raise StageFailure(Failure(kind="crucible:backend_error", stage="collect", detail="agent_stop_uninterruptible", retryable=False,
                                               reason="the agent's processes were still exiting in uninterruptible sleep (state D, SIGKILL pending) "
                                                      f"after the stop ({killed['uninterruptible'][:20]}): nothing collected (P23)"))
                if (deadline or err is not None) and killed.get("status") != "ran":
                    # M10 lab audit R-1: agent.run did not return on its own and its processes could not be stopped: the
                    # collect would race a live agent (P23); not retried into a re-run of the agent. Review round 3: survivors
                    # a service older than the phase respawned (``respawned``) land here too, never as the policy's
                    lost = await _container_lost(world)
                    if lost:
                        raise StageFailure(Failure(kind="crucible:env_container_lost", stage="collect",
                                                   reason=f"the agent's processes could not be stopped and the task container is gone: {lost}"))
                    if killed.get("status") == "stalled":  # the policy controlled the shell and tools the stop ran with
                        raise StageFailure(Failure(kind="crucible:agent_broke_env", stage="collect", blame="policy", mask=False,
                                                   reason="the harness's root stop of the agent's processes outlived its deadline in a live "
                                                          f"container the policy controlled (its shell or tools stalled: {killed.get('error')}): "
                                                          "nothing collected"))
                    if killed.get("status") == "broken":  # review round 3: a planted sleep or id killed it, or bash is gone
                        raise StageFailure(Failure(kind="crucible:agent_broke_env", stage="collect", blame="policy", mask=False,
                                                   reason="the harness's root stop of the agent's processes never printed its report in a live "
                                                          "container the policy controlled (its shell was stopped while a fresh root shell answers, "
                                                          f"or could not start: {killed.get('error')}): nothing collected"))
                    # review round 3: a stop that never reported, with nothing showing whose (a docker or ssh failure, or a
                    # shell the policy replaced), is unknown: masked and counted by the 5% unknown-blame stop
                    raise StageFailure(Failure(kind="crucible:backend_error", stage="collect", detail="agent_stop", retryable=False,
                                               blame="unknown" if killed.get("unreported") else None,
                                               reason=f"the agent's processes could not be stopped ({killed.get('why') or killed.get('error')}); "
                                                      "collecting would race a live agent (P23)"))
                t0 = time.monotonic()
                with boundary("collect"):
                    try:
                        patch = await task.collect(world, attempt.dir / "collect")
                    except CollectError as ex:
                        raise StageFailure(await self._collect_failure(world, ex)) from ex
                t["collect"] = time.monotonic() - t0
                # the agent's World is closed before grading: a perf grade must not count its container as foreign load
                closing, world = world, None
                with contextlib.suppress(Exception):
                    await closing.close()
                await self._patch._grade(attempt, Collected(patch=patch, mode="harbor-collect", tree_changed=bool(patch.strip())), res, t)
            else:
                t0 = time.monotonic()
                # M10 lab audit R-2: no separate quiesce seam here. The guarded grade quiesces the container host-side
                # itself (HarborInPlaceTask._verify: kill, confirm by docker top, receipt verifier/quiesce.json) before
                # any reset or verifier, so no caller can grade in place without it; its survivors report reaches
                # _take_grade_findings below (crucible:policy_misbehavior), its respawned / uninterruptible StageFailure
                # passes through boundary("verification") unretried
                with boundary("verification"):
                    rep_g = await task.grade(world, attempt.dir / "grade")
                t["verification"] = time.monotonic() - t0
                (attempt.dir / "verifier").mkdir(exist_ok=True)
                (attempt.dir / "verifier" / "report.json").write_text(json.dumps(rep_g.to_dict(), default=str, indent=1))
                res.grade = {**res.grade, "reward": rep_g.reward, "gates": rep_g.gate_bits(), "infra": rep_g.infra}
                _take_grade_findings(rep_g, res)
                if rep_g.infra.get("status") == "masked":
                    raise StageFailure(classify_verifier(rep_g.infra))
                res.reward, res.gates = rep_g.reward, rep_g.gate_bits()
            pen = end.penalty(self.limit_policy) if res.stop in LIMIT_STOPS else None
            if pen is not None:  # m13 C-8 (A2-24), as native.py:368-371: the configured penalty, not the partial grade
                res.grade["limit_penalty"] = pen
                res.reward = pen
        except StageFailure as sf:
            res.failure = sf.failure
        finally:
            if key is not None:
                self.proxy.unregister(key)
            if session is not None and path is not None:
                try:  # at the very end: a row the proxy wrote after the evidence read (a stream still in flight) is billed too
                    self._settle_spend(res, path)
                except Exception:  # noqa: BLE001, S110 -- cost bookkeeping never overrides a result
                    pass
            if world is not None:
                with contextlib.suppress(Exception):
                    await world.close()
            res.timings = {k: round(v, 3) for k, v in t.items()}
        return res

    async def _collect_failure(self, world: Any, ex: CollectError) -> Failure:
        """Critic amendment (P23): a collect that fails in a live container is the agent's (task02's hook diffs against a
        ``base`` tag in the agent's own repository: breaking it must not turn a failure into a mask; a hook past its
        deadline too); a container the host shows gone is the infrastructure's (host evidence the policy cannot fake: the
        work is gone, so a retry may re-run the agent). M10 lab audit R-1: a symlink or special file at the patch path is
        ``crucible:policy_misbehavior`` (custody's rule, native.py:202); a read that kept failing or never returned its
        verdict in a live container masks ``crucible:backend_error`` without a retry, since a retry re-runs the agent and
        replaces the solution being measured (R-1 rec. 2). It was a retryable mask, so a policy that broke its own patch file
        was re-sampled. A read that outlived its deadline, or whose framed output was not the patch, is the agent's (the
        container's shell and tools were the policy's to change). Review round 3: only an exec that raised fast (host side,
        ``transport``) masks as infrastructure; one that returned without the in-container shell's report with nothing
        showing whose (``shell``: a docker or ssh failure and a root policy's planted or replaced shell look the same in
        band, and the policy can forge any in-band sign) masks ``crucible:backend_error`` with unknown blame without a
        retry, counted by the 5% unknown-blame stop; one with evidence that can only blame the policy is ``hook``."""
        if ex.kind == "unsafe":
            return Failure(kind="crucible:policy_misbehavior", stage="collect", blame="policy", mask=False, reason=str(ex))
        lost = await _container_lost(world)
        if lost:
            return Failure(kind="crucible:env_container_lost", stage="collect", reason=f"collect failed and the task container is gone: {lost} ({ex})")
        if ex.kind == "transport":  # no host evidence and the patch unread: never retried
            return Failure(kind="crucible:backend_error", stage="collect", detail="collect_transport", retryable=False, reason=str(ex))
        if ex.kind == "shell":  # review round 3: neither infra nor the policy's on the evidence: unknown, never retried
            return Failure(kind="crucible:backend_error", stage="collect", detail="collect_unreported", blame="unknown", retryable=False,
                           reason=str(ex))
        return Failure(kind="crucible:agent_broke_env", stage="collect", blame="policy", mask=False,
                       reason=f"the patch could not be collected from the agent's container: {ex}")


def _agent_network(task: Any) -> str | None:
    """The task's agent-phase network mode (recorded: on a public phase the proxy-only rule rests on key scoping alone)."""
    try:
        mode = task.plan.agent_phase.network_mode
    except AttributeError:
        return None
    return str(getattr(mode, "value", mode))


__all__ = ["AgentLimits", "HarborAgentHarness", "config_digest", "copy_untrusted_logs", "default_endpoint_env", "harbor_agent_class",
           "installed_harbor_version", "mini_swe_exit_status"]
