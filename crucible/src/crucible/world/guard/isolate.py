"""Backend-independent isolation: everything a container backend does to turn a running World into a hardened one.

Used by every backend that gives the World a container with a root Exec (Harbor, agentenv's sandbox providers).
The sequence, after the Apps have seeded their files as root and before the policy's first step:

1. install the guard (root 0600) and create a fresh policy account (never an image account), refusing images
   whose root PATH or interpreter the policy could alter;
2. scrub git history where asked, and keep a root-only copy of every repo's ``.git``: verifier-side git reads
   only that copy (``--git-dir``), so hooks, filters, fsmonitor or excludes the policy plants are never used;
3. hand the policy-writable roots to the policy uid; App state the policy does not own (SQLite, mailbox, JSON
   documents) must not sit inside them;
4. pin the verifier interpreter (``python -I -S``) and stage payloads in a root-only directory;
5. route every policy tool through the guard as the demoted uid;
6. prove there is no egress (no-network / allowlist worlds).

Between episodes (``before_restore`` / ``after_restore``) the policy's processes are killed, restored files are
written as the policy uid, its ``.git`` returns to the trusted copy, and ``/tmp``, ``/var/tmp``, ``/dev/shm``,
its home and its SysV IPC objects are emptied.
"""

from __future__ import annotations

import logging
import posixpath
import shlex
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from crucible.world.exec import Exec
from crucible.world.guard import Guard, GuardError, PolicyExec

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from crucible.world.world import World

STATE_DIR = "/var/lib/crucible"
NON_POLICY_TEMPLATES = ("common/toolchain", "coding/gpu_toolchain")


class IsolationError(RuntimeError):
    pass


@dataclass
class Isolation:
    guard: Guard
    uid: int
    policy_exec: PolicyExec
    writable: list[str]
    trusted_git: dict[str, tuple[str, str]] = field(default_factory=dict)  # app id -> (worktree root, trusted git dir)
    private: list[str] = field(default_factory=list)  # App-owned state files: root 0600 (re-applied after restores)
    egress: dict[str, Any] | None = None
    root_exec: Exec | None = None
    firewall: dict[str, Any] | None = None


def _abs(root: str, base: str) -> str:
    return root if root.startswith("/") else posixpath.normpath(posixpath.join(base, root))


async def isolate_world(world: World, root_exec: Exec, *, base_path: str = "/", root_only: list[str] | None = None,
                        egress_targets: list[list[Any]] | None = None) -> Isolation:
    spec, iso = world.spec, world.spec.isolation
    guard = Guard(root_exec)
    await guard.install()
    writable = [_abs(a.state.params["root"], base_path) for a in spec.apps
                if a.state.adapter in ("fs", "fs_git") and "root" in a.state.params and a.params.get("policy_writable", True)
                and a.template not in NON_POLICY_TEMPLATES]
    root_path = (await root_exec.run("printenv PATH", timeout_s=30)).stdout.strip()
    try:
        info = await guard.setup(user=iso.policy_user, uid=iso.policy_uid, writable=writable, root_path=root_path,
                                 root_only=[*(root_only or []), STATE_DIR])
    except GuardError as e:
        raise IsolationError(str(e)) from e
    uid = int(info["uid"])
    # App state the policy must not own: refuse it inside a policy-writable tree (it could swap it for a symlink), and
    # keep it unreadable (a mailbox or ERP database holds other people's records; only the App's tools may read it).
    private = []
    for a in spec.apps:
        path = a.state.params.get("path")
        if path and a.state.adapter in ("sql", "json_doc"):
            full = _abs(path, base_path)
            inside = [w for w in writable if full == w or full.startswith(w.rstrip("/") + "/")]
            if inside:
                raise IsolationError(f"app {a.id!r}: its state {full} lies inside the policy-writable root {inside[0]}")
            private.append(full)
    trusted: dict[str, tuple[str, str]] = {}
    for a in spec.apps:
        if a.state.adapter != "fs_git" or a.template in NON_POLICY_TEMPLATES:
            continue
        root = _abs(world.apps[a.id].state.params["root"], base_path)
        if not (await root_exec.run(f"test -d {shlex.quote(root)}/.git")).ok:
            continue
        if a.params.get("scrub_history"):
            report = await guard.call("git_scrub", {"root": root, "keep": a.params.get("history_keep", "HEAD"),
                                                    "forbidden": a.params.get("forbidden_commits", [])})
            if not report["ok"]:
                raise IsolationError(f"{a.id}: git history still leaks after scrub: {report}")
        dest = f"{STATE_DIR}/git/{a.id}.git"
        await guard.call("git_trust", {"root": root, "dest": dest})
        trusted[a.id] = (root, dest)
        world.apps[a.id].state.trusted_git_dir = dest
    await guard.call("own", {"uid": uid, "paths": writable})
    for a in spec.apps:
        if a.state.adapter in ("fs", "fs_git") and _abs(world.apps[a.id].state.params.get("root", "/"), base_path) in writable:
            world.apps[a.id].state.owner_uid = uid
    # Verifier side: the interpreter resolved at setup, isolated mode without site (no PYTHONPATH, user site, .pth
    # files or cwd imports), and a root-only staging dir, so snippets and snapshot payloads never pass through /tmp.
    await root_exec.run(f"mkdir -p {STATE_DIR}/stage && chmod 700 {STATE_DIR}/stage")
    root_exec.python = f"{shlex.quote(guard.python)} -I -S"
    root_exec.staging_dir = f"{STATE_DIR}/stage"
    nproc = max(16, iso.pids_limit - 64) if iso.pids_limit else None  # headroom for the harness under the container's limit
    policy_exec = PolicyExec(guard, uid, path=root_path or None, nproc=nproc)
    for app in world.apps.values():
        app.ctx.extra["policy_exec"] = policy_exec
    out = Isolation(guard=guard, uid=uid, policy_exec=policy_exec, writable=writable, trusted_git=trusted, private=private, root_exec=root_exec)
    await _seal(out)
    if base_path != "/":
        await root_exec.run(f"chmod 711 {shlex.quote(base_path)}")  # traversable to reach App roots, not listable
    out.firewall = await _firewall(world, guard, uid)
    if iso.egress_probe and spec.network in ("no-network", "allowlist"):
        allow = list(spec.allowed_hosts) if spec.network == "allowlist" else []
        probe = await guard.call("egress", {"uid": uid, "extra_targets": egress_targets or [], "allowed": allow})
        out.egress = probe
        if not probe["isolated"]:
            raise IsolationError(f"refusing to run: the policy user can reach {probe['reachable']} in a {spec.network} world")
        if probe.get("lanes_ok") is False:  # the allowlist is enforced but too narrow: the agent would fail for the environment's reasons
            log.warning("allowlisted hosts unreachable from the policy user: %s", {k: v for k, v in probe["allowed_reachable"].items() if not v})
    return out


def app_ports(world: World) -> list[int]:
    """Localhost ports of the World's App servers (websites, MCP sidecars): the policy reaches Apps through tools only."""
    from urllib.parse import urlsplit

    ports = set()
    for app in world.apps.values():
        for attr in ("url", "mcp_url"):
            try:
                u = getattr(app, attr, None)
            except Exception:  # noqa: BLE001, S112 -- a property that needs a started server
                continue
            if isinstance(u, str) and u:
                parts = urlsplit(u)
                if parts.hostname in ("127.0.0.1", "localhost", "::1", "0.0.0.0") and parts.port:
                    ports.add(parts.port)
    return sorted(ports)


async def _firewall(world: World, guard: Guard, uid: int) -> dict[str, Any] | None:
    """Per-uid firewall (``IsolationSpec.firewall``; guard op ``firewall``, ported from Karotte)."""
    spec, iso = world.spec, world.spec.isolation
    want = iso.firewall or "auto"
    if want == "off":
        return None
    ports = sorted(set(iso.blocked_ports or []) | set(app_ports(world)))
    mode = "allowlist" if spec.network == "allowlist" else "ports"
    if mode == "ports" and not ports:
        return None
    rep = await guard.call("firewall", {"uid": uid, "mode": mode, "allowed": list(spec.allowed_hosts), "blocked_ports": ports})
    if not rep.get("took"):
        if want == "required":
            raise IsolationError(f"firewall required but not enforced: {rep.get('error') or ('gVisor ignores iptables' if rep.get('gvisor') else rep)}")
        log.warning("per-uid firewall not applied (%s); relying on the provider's network policy and the egress probe", rep.get("error"))
    if rep.get("unresolved"):
        log.warning("allowlisted names the firewall could not resolve to IPs (wildcards stay closed to the policy): %s", rep["unresolved"])
    return rep


async def before_restore(iso: Isolation) -> None:
    rep = await iso.guard.call("quiesce", {"uid": iso.uid})
    if not rep.get("ok"):
        raise IsolationError(f"policy processes survived quiesce before restore: {rep}")


async def _seal(iso: Isolation) -> None:
    if iso.private and iso.root_exec is not None:
        files = " ".join(shlex.quote(p + s) for p in iso.private for s in ("", "-journal", "-wal", "-shm"))
        await iso.root_exec.run(f"for f in {files}; do [ -e \"$f\" ] && chown 0:0 \"$f\" && chmod 600 \"$f\"; done; true")


async def after_restore(iso: Isolation) -> dict[str, Any]:
    for root, git_dir in iso.trusted_git.values():
        await iso.guard.call("git_reset", {"root": root, "src": git_dir, "uid": iso.uid})
    await _seal(iso)
    return await iso.guard.call("reclaim", {"uid": iso.uid})
