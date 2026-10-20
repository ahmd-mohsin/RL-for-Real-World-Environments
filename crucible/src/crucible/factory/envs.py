"""One task -> a trainable environment (swarm-pipeline/00 §11, §11.3; P6).

**Native** (``make_env``): a ``LongTask`` becomes ``NativeEnv(harness, sidecar, honesty)``:
- the harness: ``WorkspaceHarness`` (or the M16 ``RLMHarness``) with ``EpisodeLimits(max_steps, timeout_s=agent timeout)``;
- the sidecar: the checkpoint sidecar (ΔΦ as ``Turn.advantage`` via ``turn_credit_map``), wrapped in ``InfraWatch``, which
  marks a World found dead after any step as an infrastructure fault (``world._infra``): ``run_episode`` then masks the
  attempt (V8). Without it a dead container left the host-mounted workspace collectable and the attempt was graded;
- honesty: STATUS.json Brier and GIVE_UP from the sidecar, composed into the advantage by ``honesty_group_fn`` (never into
  the reward);
- the tool limit: ``policy_kwargs()`` gives ``LLMToolPolicy`` the limit, the warning and the final STATUS turn
  (``tool_limit``, ``limit_warning``), and ``max_steps`` = limit + 2.

**The two timeouts** (§11.3 item 4, ``Timeouts``): the agent timeout ends the agent phase as a limit stop and the work is
graded (``EpisodeLimits.timeout_s``; slime: ``EXIT_TIME_BUDGET_EXCEEDED`` -> ``TRUNCATED``); the rollout timeout bounds
everything, and expiring it is infrastructure, masked (``RunSpec.attempt_timeout_s``; slime: a removed sample).

**Groups** (§11.3 item 5): ``drop_infra_groups`` drops every GRPO group with an infrastructure-masked member and returns
how many rollouts to refill; ``train.bridges.slime.filters.group_filter`` is the same rule inside slime.

**Objective** (§11.3 item 6): ``objective_of`` pins task -> verifier version (``versioning.objective_id``); every row and
every slime sample carries it.

**Portable rows** (§11.3 item 1): ``task_row`` is the unit published, synced and exported: ``(env, template, args, slug,
split, runtime_config)`` with the ``LongTaskRef`` as the bound arguments; ``write_task_rows`` / ``read_task_rows`` are the
JSONL codec. ``slime_rows`` emits slime's input rows from the same tasks (held-out never written: V9).

**slime** (``slime_provider``, the default ``--crucible-env-provider``): ``SlimeTaskEnv`` is the task side of
``train.bridges.slime.generate``: the World opened by the task, its ``ExecSandbox`` (``DockerExec`` into the World's
``--network none`` container, or ``LocalExec`` for a host World), checkpoint snapshots and the final collection run
*through the sandbox* (the private-index diff the native sidecar uses; the tolerant collect hook ``verify.collect.COLLECT_SH``
the native harness uses), grading by the task's own ``grade_patch`` in a fresh container with the model's tool calls as the
trail. Tasks resolve from an in-process registry (``register``) or, for the window lane, from their store and an
authored prompt dir, with the ref's prompt and check-set hashes verified.

**Export** (§11.3 item 7): ``export_harbor`` writes a Harbor task folder (separate verifier, the patch collected by
``COLLECT_SH``, a stdlib grader computing the same R from the hidden milestone tests) and validates it with Harbor's own
loader (``interop.harbor.task_io.load_harbor_task``); ``export_agentenv`` publishes accepted cards through
``factory.publish`` with the portable row as the World step's arguments.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from crucible.factory.checkpoint import CheckpointConfig, CheckpointSidecar, MilestoneTask
from crucible.factory.longtask import LongTaskRef
from crucible.factory.versioning import objective_id

ENV_NAME = "crucible/longtask"


# ---- timeouts ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Timeouts:
    """``agent_s`` ends the agent phase (a limit stop, graded); ``rollout_s`` bounds the whole attempt (masked)."""

    agent_s: float = 3600.0
    rollout_s: float = 7200.0

    def __post_init__(self) -> None:
        if not 0 < self.agent_s < self.rollout_s:
            raise ValueError("0 < agent timeout < rollout timeout (grading must fit between them)")


# ---- liveness and the infra watch -------------------------------------------------------------------------------------------
async def container_running(name: str) -> bool:
    p = await asyncio.create_subprocess_exec("docker", "inspect", "-f", "{{.State.Running}}", name, stdout=asyncio.subprocess.PIPE,
                                             stderr=asyncio.subprocess.PIPE)
    out, _ = await p.communicate()
    return p.returncode == 0 and out.decode().strip() == "true"


async def world_alive(world: Any) -> bool:
    """A World is alive when its container runs (Docker World) and no host probe says otherwise (``_crucible_alive``)."""
    probe = getattr(world, "_crucible_alive", None)
    if callable(probe):
        r = probe()
        if asyncio.iscoroutine(r):
            r = await r
        if not r:
            return False
    name = getattr(world, "_crucible_container", None)
    return await container_running(name) if name else True


class InfraWatch:
    """An episode sidecar around another (or none): after every World step and at the end, a dead World is recorded as an
    infrastructure fault, which masks the attempt; everything else is delegated unchanged."""

    def __init__(self, inner: Any = None, *, alive: Callable[[Any], Any] = world_alive):
        self.inner, self.alive = inner, alive
        self.dead: str | None = None

    async def _check(self, world: Any, where: str) -> None:
        if self.dead is None and not await self.alive(world):
            self.dead = where
            if hasattr(world, "_infra"):
                world._infra("crucible:env_container_lost", f"the World's sandbox died ({where})")

    async def on_start(self, world: Any) -> None:
        if self.inner is not None:
            await self.inner.on_start(world)

    async def before_action(self, world: Any, actor: str, policy: Any, action: Any) -> None:
        if self.inner is not None and hasattr(self.inner, "before_action"):
            await self.inner.before_action(world, actor, policy, action)

    async def on_step(self, world: Any, step: int, actor: str, action: Any, result: Any) -> Any:
        await self._check(world, f"after step {step}")
        if self.inner is not None:
            return await self.inner.on_step(world, step, actor, action, result)
        return None

    async def on_end(self, world: Any) -> dict[str, Any]:
        await self._check(world, "at the end")
        out = await self.inner.on_end(world) if self.inner is not None else {}
        if self.dead:
            out = {**out, "infra_dead": self.dead}
        return out


# ---- native ------------------------------------------------------------------------------------------------------------------
def objective_of(tasks: Iterable[Any]) -> str:
    return objective_id({str(t.task_id): str(getattr(t, "verifier_version", "") or "") for t in tasks})


def task_ref(task: Any) -> LongTaskRef:
    if callable(getattr(task, "ref", None)):
        return task.ref()
    return LongTaskRef(lane=str(getattr(task, "lane", "custom")), task_id=str(task.task_id), source_id=str(getattr(task, "source_id", task.task_id)),
                       store="", split=str(getattr(task, "split", "train")), prompt_sha256=hashlib.sha256(str(task.prompt).encode()).hexdigest(),
                       checkset_sha256="", verifier_version=str(getattr(task, "verifier_version", "") or ""))


@dataclass
class NativeEnv:
    task: Any
    harness: Any
    sidecar: Callable[[Any], Any]
    honesty_group_fn: Callable
    limits: Any
    timeouts: Timeouts
    tool_limit: int | None
    limit_warning: int
    checkpoint: CheckpointConfig | None
    objective_id: str
    ref: LongTaskRef

    def policy_kwargs(self) -> dict[str, Any]:
        """``LLMToolPolicy`` keyword arguments for the tool limit and its final STATUS turn (empty without a limit)."""
        return {} if self.tool_limit is None else {"tool_limit": self.tool_limit, "limit_warning": self.limit_warning}

    def run_spec(self, *, run_id: str, policies: Sequence[Any], k: int = 1, **kw: Any) -> Any:
        from crucible.runtime.run import RunSpec

        return RunSpec(run_id=run_id, tasks=[self.task], policies=list(policies), harnesses=[self.harness], k=k, attempt_timeout_s=self.timeouts.rollout_s,
                       **kw)


def make_env(task: Any, *, mode: str = "workspace", checkpoint: CheckpointConfig | None = None, honesty_lambda: float | None = None,
             tool_limit: int | None = None, limit_warning: int = 5, max_steps: int = 150, timeouts: Timeouts | None = None, infra_watch: bool = True,
             rlm_config: Any = None) -> NativeEnv:
    """The native environment of ``task`` (see the module doc). ``checkpoint`` None = ``CheckpointConfig()`` (K = 10, cap
    20); pass ``checkpoint=False``-like ``None`` via ``CheckpointConfig(every=0, on_commit=False)`` to keep only the end."""
    from crucible.actors.policy import EpisodeLimits
    from crucible.factory.honesty import honesty_group_fn

    timeouts = timeouts or Timeouts()
    cfg = checkpoint or CheckpointConfig()
    if honesty_lambda is not None:
        cfg = replace(cfg, honesty_lambda=honesty_lambda)
    cache: dict[str, Any] = {}

    def sidecar(attempt: Any) -> Any:
        inner = CheckpointSidecar(attempt.task, attempt.dir, cfg, baseline=cache) if isinstance(attempt.task, MilestoneTask) else None
        return InfraWatch(inner) if infra_watch else inner

    steps = max_steps if tool_limit is None else tool_limit + 2  # the policy stops itself; +1 final STATUS write, +1 its Done
    limits = EpisodeLimits(max_steps=steps, timeout_s=timeouts.agent_s)
    if mode == "workspace":
        from crucible.runtime.harness.workspace import WorkspaceHarness

        harness = WorkspaceHarness(limits=limits, sidecar=sidecar)
    elif mode == "rlm":
        from crucible.runtime.harness.rlm import RLMHarness

        harness = RLMHarness(rlm_config, limits=limits, sidecar=sidecar)
    else:
        raise ValueError(f"mode is workspace or rlm, got {mode!r}")
    return NativeEnv(task=task, harness=harness, sidecar=sidecar, honesty_group_fn=honesty_group_fn(), limits=limits, timeouts=timeouts,
                     tool_limit=tool_limit, limit_warning=limit_warning, checkpoint=cfg, objective_id=objective_of([task]), ref=task_ref(task))


def infra_masked(rec: Any) -> bool:
    """A rollout masked for infrastructure (never the policy's outcome)."""
    return bool(getattr(rec, "mask_sample", False)) or getattr(rec, "reward", 0) is None


def drop_infra_groups(records: Sequence[Any], groups: Sequence[str]) -> tuple[list[Any], list[str], dict[str, int]]:
    """§11.3 item 5: keep only groups with no infrastructure-masked member; returns (kept records, their groups,
    {dropped group: rollouts to refill}). A partial group is never trained."""
    by: dict[str, list[Any]] = {}
    for r, g in zip(records, groups, strict=True):
        by.setdefault(g, []).append(r)
    bad = {g: len(rs) for g, rs in by.items() if any(infra_masked(r) for r in rs)}
    kept = [(r, g) for r, g in zip(records, groups, strict=True) if g not in bad]
    return [r for r, _ in kept], [g for _, g in kept], bad


# ---- portable rows ---------------------------------------------------------------------------------------------------------
def task_row(task: Any, *, template: str | None = None, slug: str | None = None, runtime_config: Mapping[str, Any] | None = None,
             env: str = ENV_NAME) -> dict[str, Any]:
    ref = task_ref(task)
    return {"env": env, "template": template or ref.lane, "args": {"ref": ref.to_json()}, "slug": slug or ref.task_id, "split": ref.split,
            "runtime_config": dict(runtime_config or {}), "objective_id": objective_of([task]), "verifier_version": ref.verifier_version}


def write_task_rows(rows: Iterable[Mapping[str, Any]], path: Path, *, allow_heldout: bool = False) -> dict[str, Any]:
    rows = list(rows)
    held = [r["slug"] for r in rows if r.get("split") != "train"]
    if held and not allow_heldout:
        raise ValueError(f"held-out rows go only to the sealed benchmark (allow_heldout=True there): {held[:5]}")
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return {"path": str(path), "rows": len(rows), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def read_task_rows(path: Path) -> list[dict[str, Any]]:
    out = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    for r in out:
        LongTaskRef.from_json(r["args"]["ref"])  # every row carries a well-formed ref
    return out


def slime_rows(tasks: Iterable[Any], *, agent: str = "claude_code", workdir: str = "/workspace", epilogue: str = "", sealed: Iterable[str] = (),
               extra: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """slime input rows (``train.bridges.slime.rows``) for training tasks; a held-out task raises (V9)."""
    from crucible.train.bridges.slime.rows import check_row, slime_row

    seal = set(sealed)
    out = []
    for t in tasks:
        ref = task_ref(t)
        w = getattr(t, "world", {}) or {}
        row = slime_row(instance_id=ref.task_id, prompt=str(t.prompt), image=str(w.get("image") or ""), workdir=workdir, agent=agent,
                        crucible_task_ref=ref.to_json(), epilogue=epilogue, split=ref.split,
                        extra={"objective_id": objective_of([t]), "verifier_version": ref.verifier_version, **dict(extra or {})})
        if bad := check_row(row, seal):
            raise ValueError(f"{ref.task_id} may not train: {bad}")
        out.append(row)
    return out


# ---- slime: the task side of generate ----------------------------------------------------------------------------------------
_REGISTRY: dict[str, Any] = {}
_SCRIPTS: dict[str, Any] = {}


def register(task: Any, *, script: Any = None, key: str | None = None) -> str:
    """Make ``task`` resolvable by ``generate`` in this process (``crucible_task_ref.task_id``); ``script`` registers the
    scripted model a row selects with ``metadata.script`` (the equivalence run)."""
    _REGISTRY[str(task.task_id)] = task
    if script is not None:
        _SCRIPTS[key or str(task.task_id)] = script
    return str(task.task_id)


def register_script(key: str, script: Any) -> None:
    _SCRIPTS[key] = script


def resolve_ref(ref: Mapping[str, Any], *, prompts: Path | None = None) -> Any:
    """The task a ref names: the registry, else (window lane) its store plus the authored prompt
    ``<prompts>/<source_id>/prompt.json``; the ref's prompt and check-set hashes must match."""
    r = LongTaskRef.from_json(ref)
    if r.task_id in _REGISTRY:
        return _REGISTRY[r.task_id]
    if r.lane != "window" or prompts is None:
        raise KeyError(f"task {r.task_id} is not registered and cannot be resolved (lane {r.lane}, prompts {prompts})")
    from crucible.factory.lanes.windows import WindowLane, WindowStore, WindowTask, mined_of, window_checkset

    store = WindowStore(Path(r.store), split="train")
    rec = store.window(r.source_id)
    doc = json.loads((Path(prompts) / r.source_id / "prompt.json").read_text())
    reqs = list(doc.get("requirements") or [])
    t = WindowTask(WindowLane(store), mined_of(rec, store.dir(r.source_id) / "base"), rec, task_id=r.task_id, prompt=doc["prompt"], requirements=reqs,
                   spans=dict(doc.get("spans") or {}), checkset=dict(doc.get("checkset") or window_checkset(rec, reqs)))
    got = t.ref()
    if (got.prompt_sha256, got.checkset_sha256) != (r.prompt_sha256, r.checkset_sha256):
        raise ValueError(f"{r.task_id}: the resolved prompt or check set differs from the ref (a different task)")
    return t


def collect_script(base: str, out: str) -> str:
    """``verify.collect.COLLECT_SH`` for the sandbox's own workspace: never touching a global git config (on a host sandbox
    that would be the user's), ``safe.directory`` given through the environment instead."""
    from crucible.verify.collect import COLLECT_SH

    body = "\n".join(ln for ln in COLLECT_SH.splitlines() if "git config --global" not in ln)
    body = body.replace('"${1:-.}"', ".").replace('"${2:-HEAD}"', base).replace('"${3:-/logs/artifacts/model.patch}"', out)
    return "export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'\n" + body


@dataclass
class SlimeTaskEnv:
    """``train.bridges.slime.generate.TaskEnv`` for one ``LongTask``."""

    task: Any
    checkpoint: CheckpointConfig | None = field(default_factory=CheckpointConfig)
    adapter_url: str | None = None
    _baseline: dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return str(self.task.task_id)

    @property
    def prompt(self) -> str:
        return str(self.task.prompt)

    @property
    def objective_id(self) -> str:
        return objective_of([self.task])

    @property
    def verifier_version(self) -> str:
        return str(getattr(self.task, "verifier_version", "") or "")

    @property
    def phi_weight(self) -> float:
        return float(getattr(self.task, "phi_weight", 0.5))

    async def open(self, workdir: Path) -> Any:
        from crucible.runtime.harness.sandbox import ExecSandbox
        from crucible.world.exec import DockerExec, LocalExec

        world = await self.task.open(Path(workdir))
        ws = Path(self.task.workspace(world))
        name = getattr(world, "_crucible_container", None)
        exec_ = DockerExec(name) if name else LocalExec()
        sb = ExecSandbox(exec_, root=str(ws), sandbox_id=name or f"crucible-host-{uuid.uuid4().hex[:8]}", adapter_url=self.adapter_url,
                         alive=lambda: world_alive(world))
        _, out, _ = await sb.exec("git rev-parse --verify -q 'base^{commit}' || git rev-parse --verify -q 'HEAD^{commit}'", timeout=60)
        base = out.strip() or "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
        return SimpleNamespace(world=world, sandbox=sb, ws=ws, base=base, container=name)

    def observer(self, handle: Any, out_dir: Path) -> Any:
        if self.checkpoint is None or not isinstance(self.task, MilestoneTask):
            return None
        from crucible.train.bridges.slime.episode import SNAPSHOT_CMD

        async def snap(_world: Any) -> str:
            rc, out, err = await handle.sandbox.exec(SNAPSHOT_CMD.format(base="base"), timeout=300)
            if rc != 0:
                raise RuntimeError(f"snapshot failed: {err[-300:]}")
            return out

        return CheckpointSidecar(self.task, out_dir, replace(self.checkpoint, snapshot=snap), baseline=self._baseline)

    async def snapshot(self, handle: Any) -> str:
        """The final patch: the tolerant collect hook (``verify.collect.COLLECT_SH``, what the native harness runs on the
        host) executed inside the sandbox against the base recorded at open."""
        out = f"/tmp/.crucible-model-{uuid.uuid4().hex[:8]}.patch"
        await handle.sandbox.exec(collect_script(handle.base, out), timeout=600)
        text = await handle.sandbox.read_file(out)
        await handle.sandbox.exec(f"rm -f {out}; git reset -q 2>/dev/null; true", timeout=60)
        return text

    async def alive(self, handle: Any) -> bool:
        return await world_alive(handle.world)

    async def close(self, handle: Any) -> None:
        await handle.world.close()

    async def grade(self, patch: str, out_dir: Path, *, trail: list[dict[str, Any]]) -> Any:
        import inspect

        params = inspect.signature(self.task.grade_patch).parameters
        if "trail" in params:
            return await self.task.grade_patch(patch, Path(out_dir), trail=trail)
        return await self.task.grade_patch(patch, Path(out_dir))

    def harness(self, agent: str) -> Any:
        if agent == "scripted":
            from crucible.train.bridges.slime.episode import ScriptedCLIHarness

            return ScriptedCLIHarness()
        raise NotImplementedError(f"agent {agent!r}: the CLI harnesses (claude_code, codex) and crucible_rlm are not ported to the slime path yet "
                                  "(swarm-pipeline/00 §11 'Port from slime'); P6 runs the scripted harness")

    def model(self, metadata: Mapping[str, Any], sampling_params: Mapping[str, Any]) -> Any:
        key = str(metadata.get("script") or self.task_id)
        if key in _SCRIPTS:
            s = _SCRIPTS[key]
            return s() if callable(s) and not hasattr(s, "steps") else s
        raise NotImplementedError("no model for this row: the SGLang token adapter (slime adapters/common.py) is not ported in P6; register a script")


@dataclass
class SlimeProvider:
    prompts: Path | None = None
    checkpoint: CheckpointConfig | None = field(default_factory=CheckpointConfig)
    envs: dict[str, SlimeTaskEnv] = field(default_factory=dict)

    def resolve(self, metadata: Mapping[str, Any]) -> SlimeTaskEnv:
        if metadata.get("split") not in (None, "train"):
            raise ValueError("held-out rows never train (V9)")
        task = resolve_ref(metadata["crucible_task_ref"], prompts=self.prompts)
        if task.task_id not in self.envs:
            self.envs[task.task_id] = SlimeTaskEnv(task, checkpoint=self.checkpoint)
        return self.envs[task.task_id]


_PROVIDER: SlimeProvider | None = None


def slime_provider(args: Any = None) -> SlimeProvider:
    """``--crucible-env-provider crucible.factory.envs:slime_provider`` (one provider per process; ``args.crucible_prompts``
    names the authored prompt dir for window refs)."""
    global _PROVIDER
    if _PROVIDER is None:
        p = getattr(args, "crucible_prompts", None)
        _PROVIDER = SlimeProvider(prompts=Path(p) if p else None)
    return _PROVIDER


def reset_provider(provider: SlimeProvider | None = None) -> None:
    global _PROVIDER
    _PROVIDER = provider


# ---- export ------------------------------------------------------------------------------------------------------------------
HARBOR_GRADER = r'''"""Crucible window grader for Harbor (stdlib + pytest): the hidden milestone tests on A + the agent's patch.
R = phi_weight * Phi + (1 - phi_weight) * [every check passes]; Phi counts a milestone only with its requires passing.
Path checks (protected paths, network, history reads) need the agent's trail, which Harbor does not hand the verifier:
listed as not_verified in reward.json."""
import json, os, subprocess, sys, xml.etree.ElementTree as ET
spec = json.load(open(os.path.join(os.path.dirname(__file__), "milestones.json")))
os.chdir(spec["workdir"])
patch = sys.argv[1] if len(sys.argv) > 1 else "/logs/artifacts/model.patch"
out = {"reward": 0.0, "not_verified": ["path checks (no trail in Harbor)"]}
os.makedirs("/logs/verifier", exist_ok=True)
if os.path.exists(patch) and os.path.getsize(patch):
    ok = any(subprocess.run(c, shell=True).returncode == 0 for c in (f"git apply --whitespace=nowarn {patch}", f"git apply --3way {patch}", f"patch -p1 < {patch}"))
    if not ok:
        out["apply_error"] = True
for rel, text in spec["tests"].items():
    os.makedirs(os.path.dirname(rel) or ".", exist_ok=True)
    open(rel, "w").write(text)
for rel in spec.get("deleted", []):
    if os.path.exists(rel):
        os.remove(rel)
junit = "/tmp/crucible-junit.xml"
targets = sorted({t.split("::", 1)[0] for m in spec["milestones"] for t in m["tests"]})
subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--continue-on-collection-errors", f"--junitxml={junit}", *targets])
passed = set()
if os.path.exists(junit):
    for tc in ET.parse(junit).getroot().iter("testcase"):
        if not any(ch.tag in ("failure", "error", "skipped") for ch in tc):
            cls, name = tc.get("classname", ""), tc.get("name", "")
            passed.add(f"{tc.get('file') or cls.replace('.', '/') + '.py'}::{name}")
            passed.add(name)
def ok(ts):
    return all(t in passed or t.split("::")[-1] in passed for t in ts)
st = {m["id"]: ok(m["tests"]) for m in spec["milestones"]}
w = spec["weights"]
phi = sum(w[m["id"]] for m in spec["milestones"] if st[m["id"]] and all(st.get(r) for r in m.get("requires", [])))
full = all(st.values()) and not out.get("apply_error")
out.update(milestones=st, phi=round(phi, 6), full_pass=full, reward=0.0 if out.get("apply_error") else round(spec["phi_weight"] * phi + (1 - spec["phi_weight"]) * full, 6))
json.dump(out, open("/logs/verifier/reward.json", "w"))
open("/logs/verifier/reward.txt", "w").write(str(out["reward"]))
'''


def export_harbor(task: Any, out_dir: Path, *, timeouts: Timeouts | None = None, workdir: str = "/app", cpus: int = 2, memory_mb: int = 2048) -> dict[str, Any]:
    """A Harbor task folder for a window task (see the module doc); validated with Harbor's own loader."""
    from crucible.interop.harbor.task_io import load_harbor_task
    from crucible.verify.collect import COLLECT_SH

    timeouts = timeouts or Timeouts()
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "environment").mkdir(parents=True)
    (out_dir / "tests").mkdir()
    ref = task_ref(task)
    w = task.world
    shutil.copytree(Path(w["base_dir"]), out_dir / "environment" / "repo", symlinks=True)
    image = str(w.get("image"))
    digest = w.get("digest")
    (out_dir / "environment" / "collect.sh").write_text(COLLECT_SH)
    (out_dir / "environment" / "Dockerfile").write_text(
        f"FROM {image}\nCOPY repo/ {workdir}/\nCOPY collect.sh /opt/crucible/collect.sh\nWORKDIR {workdir}\n"
        "RUN git init -q -b main . && git add -A && git -c user.email=crucible@local -c user.name=crucible commit -qm base --allow-empty "
        "&& git tag -f base && git rev-parse HEAD > /opt/crucible/base_sha\n")
    (out_dir / "instruction.md").write_text(str(task.prompt).strip() + "\n")
    ms = [{"id": m["id"], "tests": list(m["tests"]), "requires": list(m.get("requires") or [])} for m in task.milestones]
    weights = dict(getattr(task, "weights", None) or {m["id"]: 1.0 / len(ms) for m in ms})
    spec = {"workdir": workdir, "milestones": ms, "weights": weights, "phi_weight": float(getattr(task, "phi_weight", 0.5)),
            "tests": dict(getattr(task, "grader_tests", {}) or {}), "deleted": list((getattr(task, "window", {}) or {}).get("tests_deleted") or [])}
    (out_dir / "tests" / "milestones.json").write_text(json.dumps(spec, indent=1, sort_keys=True))
    (out_dir / "tests" / "crucible_grade.py").write_text(HARBOR_GRADER)
    (out_dir / "tests" / "test.sh").write_text("#!/bin/sh\npython3 /tests/crucible_grade.py /logs/artifacts/model.patch\n")
    meta = {"crucible_ref": json.dumps(ref.to_json(), sort_keys=True), "objective_id": objective_of([task]), "image_digest": str(digest or "")}
    toml = [
        'schema_version = "1.1"', "", "[task]", f'name = "crucible/{re.sub(r"[^A-Za-z0-9_.-]", "-", ref.task_id)}"', f'description = "Crucible long-horizon task {ref.task_id}"', "",
        "[metadata]", *[f"{k} = {json.dumps(v)}" for k, v in meta.items()], "",
        "[agent]", f"timeout_sec = {float(timeouts.agent_s)}", "",
        "[environment]", f'workdir = "{workdir}"', f"cpus = {cpus}", f"memory_mb = {memory_mb}", 'network_mode = "no-network"', "",
        "[verifier]", f"timeout_sec = {float(timeouts.rollout_s - timeouts.agent_s)}", 'environment_mode = "separate"', 'network_mode = "no-network"', "",
        "[[verifier.collect]]", f"command = {json.dumps(f'mkdir -p /logs/artifacts && sh /opt/crucible/collect.sh {workdir} \"$(cat /opt/crucible/base_sha)\" /logs/artifacts/model.patch')}",
        "timeout_sec = 300.0", "", "[[artifacts]]", 'source = "/logs/artifacts/model.patch"', ""]
    (out_dir / "task.toml").write_text("\n".join(toml))
    loaded = load_harbor_task(out_dir)
    return {"path": str(out_dir), "task_hash": loaded.summary.task_hash, "verifier_mode": loaded.summary.verifier_mode,
            "network_mode": loaded.summary.network_mode, "ignored_keys": loaded.summary.ignored_keys, "warnings": loaded.summary.metadata.get("harbor_warnings")}


class RowTask:
    """The ``module:Class`` an agentenv ``crucible_world`` step loads for a published row: the row's task, resolved."""

    def __init__(self, *, backend: str = "local", row: Mapping[str, Any] | None = None, prompts: str | None = None):
        if backend != "local":
            raise NotImplementedError(f"backend {backend!r}: window Worlds run on Docker exec, not the agentenv gateway yet")
        self.task = resolve_ref((row or {})["args"]["ref"], prompts=Path(prompts) if prompts else None)
        self.world_id = getattr(self.task, "world_id", "")

    async def open(self, workdir: Path) -> Any:
        return await self.task.open(workdir)


def export_agentenv(cards: Sequence[Any], rows: Mapping[str, Mapping[str, Any]], splits: Mapping[str, Sequence[str]]) -> Any:
    """Publish accepted cards (``factory.publish.publish_family``) with each card's portable row as its World step's
    arguments (``crucible.factory.envs:RowTask``). ``rows``: card id -> ``task_row``."""
    from crucible.factory.publish import PublishError, publish_eval, publish_task

    owner: dict[str, str] = {}
    for s, ids in splits.items():
        for i in ids:
            if owner.setdefault(i, s) != s:
                raise PublishError(f"card {i!r} is in both {owner[i]!r} and {s!r}: splits must not share a task")
    out = {}
    for c in cards:
        out[c.id] = publish_task(c, world_spec="crucible.factory.envs:RowTask", task_kwargs={"row": dict(rows[c.id])})
    return {"tasks": out, "evals": {s: publish_eval(cards[0].family, s, [out[i] for i in ids]) for s, ids in splits.items()}}


# ---- scripted solvers (the P6 equivalence run; $0) ---------------------------------------------------------------------------
class ScriptedLLM:
    """The native twin of ``train.bridges.slime.episode.ScriptedModel``: the same steps (``ScriptedSteps``) as LLM replies
    with tool calls ``<app>__shell`` / ``<app>__write_file``, the same token ids (``tokens``), chaos run before the reply."""

    def __init__(self, steps: Sequence[tuple], *, app: str = "repo", model: str = "scripted", delay_s: float = 0.0):
        from crucible.runtime.inject import FakeSpend
        from crucible.train.bridges.slime.episode import ScriptedSteps

        self.cursor, self.app, self.model, self.delay_s = ScriptedSteps(steps), app, model, delay_s
        self.spend = FakeSpend()

    async def complete(self, messages: Any, **kw: Any) -> Any:
        from crucible.actors.llm import LLMResponse

        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        self.spend.calls += 1
        t = await self.cursor.next()
        calls = [{"id": a["id"], "name": f"{self.app}__{a['tool']}", "arguments": dict(a["arguments"])} for a in t.actions]
        tokens = {"prompt_token_ids": [1], "completion_token_ids": list(t.output_ids), "logprobs": [0.0] * len(t.output_ids), "policy_version": 0}
        return LLMResponse(text=t.text, tool_calls=calls, tokens=tokens, model=self.model)


def scripted_policy(steps: Sequence[tuple], *, app: str = "repo", pid: str = "scripted", delay_s: float = 0.0, **policy_kw: Any) -> Any:
    """A ``PolicyRef`` whose agent replays ``steps`` (a fresh cursor per attempt)."""
    from crucible.actors.policy import LLMToolPolicy
    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.lock import definition_digest, jsonable

    # config: what the run identity compares (M10 lab audit R-4; ``make`` is a closure): the replayed steps by digest
    return PolicyRef(id=f"{pid}", make=lambda _a: {"agent": LLMToolPolicy("agent", ScriptedLLM(steps, app=app, delay_s=delay_s), **policy_kw)},
                     config={"script": pid, "app": app, "steps": definition_digest(list(steps)), "policy": jsonable(policy_kw)})
