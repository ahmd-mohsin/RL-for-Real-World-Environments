"""Every ``CRUCIBLE_*`` environment variable, in one typed table (R-15; PATTERN: rLLM ``rllm/env.py``, Apache-2.0).

Read a variable with ``get(name)``: unset or empty gives the declared default, and a malformed value raises ``EnvError``
naming the variable, the value and what it accepts. A typo in an operational knob must fail loudly, not fall back:
``CRUCIBLE_LLM_CACHE=reply`` used to turn the LLM cache off and pay for every call again. The first ``get`` also warns
once about any ``CRUCIBLE_*`` variable in the environment that is not declared here (``CRUCIBLE_LLM_CAHCE``), naming
the closest declared one. ``tests/unit/test_core_audit.py`` fails when code under ``src/`` or ``packs/`` names a
``CRUCIBLE_*`` variable missing from ``KNOBS``.

Kinds: ``user`` knobs are set by whoever runs Crucible; ``internal`` ones are set by Crucible for a process it starts
(two of them, ``CRUCIBLE_RLM_SEED`` and ``CRUCIBLE_VERIFY_FD``, are read inside sandboxes that cannot import Crucible,
so those readers use ``os.environ`` directly); ``test`` ones switch on tests and scripts that need a GPU, Modal or a lab
host. A ``secret`` value is registered with the scrubber when read (``core.secrets.register_secret``).
"""

from __future__ import annotations

import difflib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

Type = Literal["str", "int", "float", "bool", "path", "paths", "choice"]
_TRUE, _FALSE = frozenset({"1", "true", "yes", "on"}), frozenset({"0", "false", "no", "off"})


class EnvError(ValueError):
    """A ``CRUCIBLE_*`` variable holds a value its declaration does not accept."""


@dataclass(frozen=True)
class Knob:
    name: str
    type: Type
    default: Any
    help: str
    kind: Literal["user", "internal", "test"] = "user"
    choices: tuple[str, ...] = ()
    secret: bool = False


KNOBS: dict[str, Knob] = {k.name: k for k in [
    # ---- user: set by whoever runs Crucible
    Knob("CRUCIBLE_CONFIG", "path", None, "the one crucible.toml to use instead of the user file and the project file (core.config layers)"),
    Knob("CRUCIBLE_DATA", "path", None, "studio explorer's data root when crucible.toml sets none (default ./data)"),
    Knob("CRUCIBLE_DATA_ROOTS", "paths", (), "extra directories a pack may read data from (os.pathsep-separated)"),
    Knob("CRUCIBLE_LLM_CACHE", "choice", "record", "LLM response cache: record (read and write), replay (a miss is an error), off",
         choices=("record", "replay", "off")),
    Knob("CRUCIBLE_LLM_TIMEOUT_S", "float", 600.0, "read timeout per LLM call, in seconds"),
    Knob("CRUCIBLE_LOCK_DIR", "path", None, "where host-level locks live (default ~/.crucible/locks)"),
    Knob("CRUCIBLE_DOCKER_SSH_TUNNEL", "bool", True, "reach a remote Docker daemon through an SSH tunnel"),
    Knob("CRUCIBLE_REDACTION_SALT", "str", None, "tenant salt for stable pseudonyms in redacted traces", secret=True),
    Knob("CRUCIBLE_BENCH_KEY", "str", None, "dynamic-bench HMAC key, hex (prefer CRUCIBLE_BENCH_KEY_FILE)", secret=True),
    Knob("CRUCIBLE_BENCH_KEY_FILE", "path", None, "file holding the dynamic-bench HMAC key"),
    Knob("CRUCIBLE_CPU_ONLY", "bool", False, "train on CPU (gloo) even when CUDA is available"),
    Knob("CRUCIBLE_PG_TIMEOUT_S", "float", 1800.0, "torch.distributed process-group timeout, in seconds"),
    Knob("CRUCIBLE_SPMD_PERIOD_S", "float", 2.0, "SPMD heartbeat period, in seconds"),
    Knob("CRUCIBLE_SPMD_STALE_S", "float", 120.0, "SPMD: a rank silent this long is stale, in seconds"),
    Knob("CRUCIBLE_SPMD_JOIN_S", "float", 3600.0, "SPMD: how long to wait for every rank to join, in seconds"),
    Knob("CRUCIBLE_VERL_ROLLOUT", "str", None, "verl agent loop: the rollout factory, pkg.mod:factory"),
    Knob("CRUCIBLE_VERL_TURN_WEIGHT", "float", 0.0, "verl agent loop: per-turn reward weight"),
    # slime bridge (train/bridges/slime): each is the fallback after slime's own args.crucible_* option
    Knob("CRUCIBLE_SLIME_ENV", "str", "crucible.factory.envs:slime_provider", "slime bridge: the environment provider, pkg.mod:attr"),
    Knob("CRUCIBLE_SLIME_WORKDIR", "path", None, "slime bridge: where episode workspaces are made (default the system temp dir)"),
    Knob("CRUCIBLE_SLIME_ROLLOUT_TIMEOUT_S", "float", 7200.0, "slime bridge: one rollout's time limit, in seconds"),
    Knob("CRUCIBLE_SLIME_AGENT_TIMEOUT_S", "float", 3600.0, "slime bridge: the agent's time limit in one rollout, in seconds"),
    Knob("CRUCIBLE_SLIME_TURN_ETA", "float", 0.1, "slime bridge: the turn-credit weight added to the outcome advantage"),
    Knob("CRUCIBLE_SLIME_MAX_STALENESS", "int", 2, "slime bridge: a group more weight updates behind than this is dropped"),
    Knob("CRUCIBLE_SLIME_WEIGHT_VERSION", "int", None, "slime bridge: the current weight version when slime passes none (unset: no staleness cap)"),
    Knob("CRUCIBLE_R3_PROXY_HOST", "str", None, "coding pack R3 gate: this machine as the lab container reaches it (the capture proxy's host)"),
    Knob("CRUCIBLE_R3_PROXY_PORT", "int", 8788, "coding pack R3 gate: the capture proxy's port"),
    Knob("CRUCIBLE_R3_UPSTREAM", "str", "https://openrouter.ai/api/v1", "coding pack R3 gate: the model API the proxy forwards to"),
    Knob("CRUCIBLE_R3_BUDGET_USD", "float", 15.0, "coding pack R3 gate: the run's budget, in USD"),
    Knob("CRUCIBLE_ENGINE_NCCL_CUMEM_ENABLE", "choice", None, "toy M12 gate: NCCL_CUMEM_ENABLE passed to the engines", choices=("0", "1")),
    Knob("CRUCIBLE_VLLM_LOG", "path", Path("/work/out/vllm.log"), "toy M12 gate: where the vLLM server log is written"),
    # ---- internal: set by Crucible for a process it starts
    Knob("CRUCIBLE_SPEC", "str", None, "train.cluster: the process spec (JSON) of a launched rank", kind="internal"),
    Knob("CRUCIBLE_GPUS", "str", None, "train.cluster: the physical GPU indices a rank owns", kind="internal"),
    Knob("CRUCIBLE_ROLE", "str", None, "train.cluster: a launched process's role", kind="internal"),
    Knob("CRUCIBLE_ENV_URLS", "str", None, "train.launch: comma-separated environment server URLs", kind="internal"),
    Knob("CRUCIBLE_GATE_ARGS", "str", None, "toy M12 gate: the gate's arguments (JSON) for its ranks", kind="internal"),
    Knob("CRUCIBLE_RLM_SEED", "int", 0, "RLM kernel: the seed of a cell's random module (read in the sandbox)", kind="internal"),
    Knob("CRUCIBLE_VERIFY_FD", "int", -1, "hidden-test runner: the result pipe (read in the candidate's pytest)", kind="internal"),
    Knob("CRUCIBLE_VF_KEY", "str", None, "verifiers import: the API-key variable handed to vf-eval", kind="internal"),
    # ---- test: switches for tests and scripts that need hardware or a service
    Knob("CRUCIBLE_GPU", "bool", False, "run tests/gpu on a GPU host", kind="test"),
    Knob("CRUCIBLE_SEEDS", "str", None, "tests/gpu: comma-separated seed tasks to run", kind="test"),
    Knob("CRUCIBLE_MODAL", "bool", False, "run the Modal GPU-sandbox tests", kind="test"),
    Knob("CRUCIBLE_MODAL_GPU", "str", "L4", "Modal GPU type for the sandbox tests", kind="test"),
    Knob("CRUCIBLE_MODAL_IMAGE", "str", None, "Modal image for the sandbox tests", kind="test"),
    Knob("CRUCIBLE_VM_DOCKER_HOST", "str", None, "Docker host with /dev/kvm for the VM app test (e.g. ssh://user@lab)", kind="test"),
    Knob("CRUCIBLE_MIMO_JUDGE", "bool", False, "run the MiMo judge interop test against a paid endpoint", kind="test"),
    Knob("CRUCIBLE_FRAMEWORKS_DIR", "path", None, "where tests find the upstream framework clones", kind="test"),
    Knob("CRUCIBLE_SITE_DIR", "path", None, "the built site the studio tests read", kind="test"),
    Knob("CRUCIBLE_DIST_BACKEND", "choice", "gloo", "torch.distributed backend of the FSDP probe", kind="test", choices=("gloo", "nccl")),
    Knob("CRUCIBLE_RENDEZVOUS", "str", None, "FSDP worker rendezvous: env reads MASTER_ADDR / MASTER_PORT", kind="test"),
    Knob("CRUCIBLE_WS_TIMEOUT_S", "float", 60.0, "rollout test: world-server timeout handed to workers, in seconds", kind="test"),
    Knob("CRUCIBLE_CAL_GPU", "str", "A10G:4", "scripts/modal_calibrate.py: Modal GPU spec", kind="test"),
    Knob("CRUCIBLE_GATE_GPU", "str", "L4:4", "scripts/modal_gate_dist.py: Modal GPU spec", kind="test"),
    Knob("CRUCIBLE_INFRA_GPU", "str", "L4:4", "scripts/modal_infra_check.py: Modal GPU spec", kind="test"),
    Knob("CRUCIBLE_VLLM_MODEL", "str", "Qwen/Qwen3-4B-Instruct-2507", "scripts/modal_vllm.py: model to serve", kind="test"),
    Knob("CRUCIBLE_CLAUDE_BIN", "path", None, "factory builder: the claude executable for headless Claude Code (default ~/.local/bin/claude, then PATH)"),
    Knob("CRUCIBLE_TINKER_CAP_USD", "float", 1.0, "Tinker bridge: hard USD cap of the shared spend meter for factory roles sampled on Tinker"),
    Knob("CRUCIBLE_TINKER_SPEND", "path", None, "Tinker bridge: spend.jsonl the shared meter appends receipts to (default: none)"),
    Knob("CRUCIBLE_CUDA_HOME", "path", "/usr/local/cuda", "kernel lane grader: the CUDA toolkit inside the grading image", kind="internal"),
    Knob("CRUCIBLE_CUDA_ARCH", "str", None, "kernel lane grader: the native sm_XX the grader builds for (set by the GPU stage)", kind="internal"),
    Knob("CRUCIBLE_OBS", "path", None, "kernel lane grader: directory the launch-observer shim writes to (unset = observer off)", kind="internal"),
    Knob("CRUCIBLE_SEED", "int", 0, "kernel lane grader: grading-time seed mixed into the candidate's srand", kind="internal"),
    Knob("CRUCIBLE_LAUNCH_LOG", "path", None, "kernel lane grader: file the CUPTI kernel-launch counter flushes to", kind="internal"),
    Knob("CRUCIBLE_VLLM_GPU", "str", "L4", "scripts/modal_vllm.py: Modal GPU type", kind="test"),
    Knob("CRUCIBLE_VLLM_MAX_LEN", "int", 32768, "scripts/modal_vllm.py: max model length", kind="test"),
    Knob("CRUCIBLE_VLLM_KEY", "str", None, "API key of the Modal vLLM server", kind="test", secret=True),
]}

_checked = False


def get(name: str) -> Any:
    """The typed value of ``name`` from the environment, or its declared default when unset or empty."""
    _check_environ_once()
    return parse(name, os.environ.get(name))


def parse(name: str, raw: str | None) -> Any:
    """``raw`` (an environment value, or None for unset) as declared for ``name``; EnvError when it does not fit."""
    knob = KNOBS.get(name)
    if knob is None:
        raise KeyError(f"{name} is not a declared CRUCIBLE_* variable: add it to crucible.core.env.KNOBS")
    if raw is None or raw.strip() == "":
        return knob.default
    if knob.secret:
        from crucible.core.secrets import register_secret

        register_secret(raw)
    value = raw.strip()
    try:
        if knob.type == "int":
            return int(value)
        if knob.type == "float":
            return float(value)
        if knob.type == "bool":
            if value.lower() in _TRUE:
                return True
            if value.lower() in _FALSE:
                return False
            raise ValueError("expected one of 1/0, true/false, yes/no, on/off")
        if knob.type == "choice":
            if value not in knob.choices:
                raise ValueError(f"expected one of {', '.join(knob.choices)}")
            return value
        if knob.type == "path":
            return Path(value).expanduser()
        if knob.type == "paths":
            return tuple(Path(p.strip()).expanduser() for p in value.split(os.pathsep) if p.strip())
        return raw
    except ValueError as e:
        shown = "<secret>" if knob.secret else repr(raw)
        raise EnvError(f"{name}={shown}: {e} ({knob.help})") from None


def require(name: str) -> Any:
    """Like ``get``, for a variable that must be set (an internal hand-off): unset raises EnvError."""
    value = get(name)
    if value is None:
        raise EnvError(f"{name} is not set ({KNOBS[name].help})")
    return value


def undeclared(environ: dict[str, str] | None = None) -> dict[str, str | None]:
    """Each ``CRUCIBLE_*`` name in ``environ`` that ``KNOBS`` does not declare, with the closest declared name or None."""
    env = os.environ if environ is None else environ
    return {n: next(iter(difflib.get_close_matches(n, KNOBS, n=1, cutoff=0.8)), None) for n in sorted(env)
            if n.startswith("CRUCIBLE_") and n not in KNOBS}


def _check_environ_once() -> None:
    global _checked
    if _checked:
        return
    _checked = True
    for name, near in undeclared().items():
        logger.warning("%s is set but Crucible reads no such variable%s", name, f" (did you mean {near}?)" if near else "")
