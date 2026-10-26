#!/usr/bin/env python3
"""A ``docker`` command for Apptainer hosts (DeltaAI GH200; doc 14-deltaai §3). Standard library only.

The lanes' Worlds and graders speak Docker (``docker run``, ``exec``, ``rm -f``, ``image inspect``, ``ps``, ``cp``). On a
host without Docker this script, installed as ``docker`` first on ``PATH``, runs the same commands as Apptainer
instances, so no lane code changes and the measured Worlds (the very images the Mac built, imported as SIFs) are the ones
the episodes run in.

**Lockdown, on every container, whatever the caller asked:**
- ``--containall --cleanenv --no-home``: private PID, IPC and /dev, no host home, no host environment (the caller's
  ``-e`` values arrive through an env file);
- ``--net --network none``: loopback only. A run that asks for any other network is refused (exit 125: the callers
  treat it as infrastructure);
- binds only where the caller asked (``-v``/``--mount``), and only under the allowed roots (``CRUCIBLE_SHIM_ALLOW``,
  colon-separated: the job's scratch and work directories). A bind of the user's home, ``~/.ssh``, the HF token, the
  project and work trees outside those roots, or ``/sw`` is refused;
- a per-container writable layer: ``--overlay <node-local dir>`` (fuse-overlayfs; the site's writable tmpfs is capped at
  64 MiB), removed with the container;
- ``--fakeroot`` (a root-mapped user namespace: root inside, the invoking uid outside; no subuid range is needed) unless
  the caller ran as a numeric user;
- cgroup limits (``--memory``, ``--cpus``, ``--pids-limit``) when the host accepts them (``CRUCIBLE_SHIM_LIMITS=1``);
  the Slurm job's cgroup bounds the rest;
- GPUs only when the caller asked (``--gpus`` / ``--runtime=nvidia``): ``--nv``.

What this cannot give (one uid on the host, no subuid range): code inside runs as the invoking user. Anything that user
can reach *through a bind* it can reach; the allowed roots must hold nothing secret. The lockdown probe
(``lockdown_probe.py``) checks the rest from inside a container before any model episode runs.

**Images.** ``<repo>:<tag>`` maps to ``$CRUCIBLE_SIF_DIR/<repo>_<tag>.sif`` (``/`` and ``:`` become ``_``); a side file
``<sif>.json`` holds what Docker kept in the image config (``WorkingDir``, ``User``, ``Entrypoint``, ``Cmd``), written
when the image was imported (``lh_deltaai.py images``). ``image inspect --format {{.Id}}`` is the SIF's sha256.

**State.** Instances are Apptainer's (``APPTAINER_CONFIGDIR``: ``CRUCIBLE_SHIM_CONFIGDIR``); the shim's own record per
container (its overlay, its SIF, its labels) is ``$CRUCIBLE_SHIM_STATE/<name>.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

APPTAINER = os.environ.get("CRUCIBLE_APPTAINER", "apptainer")
LOCKDOWN = ["--containall", "--cleanenv", "--no-home", "--net", "--network", "none"]
KEEPALIVE = (("sleep", "infinity"), ("tail", "-f", "/dev/null"), ("sleep", "inf"))
VALUE_FLAGS = {"--name", "--label", "-l", "--network", "--net", "--tmpfs", "--memory", "-m", "--cpus", "--pids-limit", "-v", "--volume", "--mount",
               "-w", "--workdir", "-e", "--env", "--env-file", "-u", "--user", "--entrypoint", "--runtime", "--gpus", "--cap-drop", "--cap-add",
               "--security-opt", "--ulimit", "--shm-size", "--hostname", "-h", "--add-host", "--platform", "--stop-timeout", "--memory-swap",
               "--cpuset-cpus", "--ipc", "--restart", "--log-driver", "--log-opt", "--init-path", "--device", "--stop-signal"}
BOOL_FLAGS = {"--rm", "-d", "--detach", "--init", "--read-only", "-i", "-t", "-it", "-ti", "--interactive", "--tty", "--no-healthcheck", "-q",
              "--quiet"}
REFUSED_FLAGS = {"--privileged", "--pid", "--userns", "--uts"}


class ShimError(Exception):
    code = 125


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default)


def state_dir() -> Path:
    p = env_path("CRUCIBLE_SHIM_STATE", str(Path(tempfile.gettempdir()) / "crucible-shim"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def scratch_dir() -> Path:
    p = env_path("CRUCIBLE_SHIM_SCRATCH", str(Path(tempfile.gettempdir()) / "crucible-shim-ovl"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def host_env() -> dict[str, str]:
    e = dict(os.environ)
    if os.environ.get("CRUCIBLE_SHIM_CONFIGDIR"):
        e["APPTAINER_CONFIGDIR"] = os.environ["CRUCIBLE_SHIM_CONFIGDIR"]
    return e


# ---- images -------------------------------------------------------------------------------------------------------------------
def sif_of(image: str) -> Path:
    if image.endswith(".sif") or image.startswith("/"):
        return Path(image)
    name = re.sub(r"[/:@]", "_", image if ":" in image or "@" in image else f"{image}:latest")
    return env_path("CRUCIBLE_SIF_DIR", "sif") / f"{name}.sif"


def image_config(sif: Path) -> dict:
    side = Path(str(sif) + ".json")
    return json.loads(side.read_text()) if side.exists() else {}


def image_id(sif: Path) -> str:
    cache = Path(str(sif) + ".id")
    st = sif.stat()
    key = f"{st.st_size}:{int(st.st_mtime)}"
    if cache.exists():
        k, _, v = cache.read_text().partition(" ")
        if k == key:
            return v.strip()
    h = hashlib.sha256()
    with sif.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    v = "sha256:" + h.hexdigest()
    try:
        cache.write_text(f"{key} {v}")
    except OSError:
        pass
    return v


# ---- binds -------------------------------------------------------------------------------------------------------------------
def allowed_roots() -> list[Path]:
    roots = [r for r in (os.environ.get("CRUCIBLE_SHIM_ALLOW") or "").split(":") if r]
    if not roots:
        roots = [tempfile.gettempdir()]
    return [Path(r).resolve() for r in roots]


def forbidden_roots() -> list[Path]:
    home = Path(os.path.expanduser("~")).resolve()
    out = [home, home / ".ssh", home / ".hf_token", Path("/sw"), Path("/projects"), Path("/u"), Path("/work"), Path("/etc"), Path("/var"),
           Path("/run"), Path("/proc"), Path("/sys"), Path("/dev"), Path("/root"), Path("/")]
    return out


def check_bind(src: str) -> Path:
    p = Path(src).resolve()
    for r in allowed_roots():
        if p == r or r in p.parents:
            return p
    raise ShimError(f"bind of {src} refused: not under an allowed root ({':'.join(map(str, allowed_roots()))})")


def parse_volume(v: str) -> tuple[str, str, bool]:
    parts = v.split(":")
    if len(parts) == 1:
        raise ShimError(f"anonymous volume {v!r} is not supported")
    src, dst = parts[0], parts[1]
    opts = parts[2].split(",") if len(parts) > 2 else []
    return src, dst, "ro" in opts


def parse_mount(m: str) -> tuple[str, str, bool]:
    kv = dict(x.split("=", 1) if "=" in x else (x, "true") for x in m.split(","))
    if kv.get("type", "bind") != "bind":
        raise ShimError(f"mount type {kv.get('type')!r} is not supported")
    src = kv.get("source") or kv.get("src")
    dst = kv.get("target") or kv.get("destination") or kv.get("dst")
    if not src or not dst:
        raise ShimError(f"mount {m!r} needs a source and a target")
    return src, dst, kv.get("readonly", kv.get("ro", "false")) in ("true", "1", "")


# ---- docker run --------------------------------------------------------------------------------------------------------------
def parse_run(argv: list[str]) -> dict:
    o: dict = {"rm": False, "detach": False, "name": None, "binds": [], "env": [], "env_files": [], "workdir": None, "user": None, "gpus": False,
               "network": None, "memory": None, "cpus": None, "pids": None, "entrypoint": None, "labels": [], "interactive": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith("-") or a == "-":
            break
        if "=" in a and a.startswith("--"):
            a, val = a.split("=", 1)
            inline = True
        else:
            val, inline = None, False
        if a in REFUSED_FLAGS:
            raise ShimError(f"{a} is refused by the Apptainer shim")
        if a in BOOL_FLAGS:
            o["rm"] |= a == "--rm"
            o["detach"] |= a in ("-d", "--detach")
            o["interactive"] |= a in ("-i", "-it", "-ti", "--interactive")
            i += 1
            continue
        if a not in VALUE_FLAGS:
            raise ShimError(f"docker run flag {a} is not supported by the Apptainer shim")
        if not inline:
            i += 1
            if i >= len(argv):
                raise ShimError(f"{a} needs a value")
            val = argv[i]
        i += 1
        if a == "--name":
            o["name"] = val
        elif a in ("--label", "-l"):
            o["labels"].append(val)
        elif a in ("--network", "--net"):
            o["network"] = val
        elif a in ("-v", "--volume"):
            o["binds"].append(parse_volume(val))
        elif a == "--mount":
            o["binds"].append(parse_mount(val))
        elif a in ("-w", "--workdir"):
            o["workdir"] = val
        elif a in ("-e", "--env"):
            o["env"].append(val)
        elif a == "--env-file":
            o["env_files"].append(val)
        elif a in ("-u", "--user"):
            o["user"] = val
        elif a in ("--memory", "-m"):
            o["memory"] = val
        elif a == "--cpus":
            o["cpus"] = val
        elif a == "--pids-limit":
            o["pids"] = val
        elif a == "--entrypoint":
            o["entrypoint"] = val
        elif a == "--runtime":
            o["gpus"] |= val == "nvidia"
        elif a == "--gpus":
            o["gpus"] = True
        elif a == "--device" and "nvidia" not in val:
            raise ShimError(f"--device {val} is refused")
        # --tmpfs, --read-only, --cap-*, --security-opt, --ulimit, --shm-size, ...: the lockdown covers them or they are moot
    if i >= len(argv):
        raise ShimError("docker run needs an image")
    o["image"], o["cmd"] = argv[i], argv[i + 1:]
    if o["network"] not in (None, "none"):
        raise ShimError(f"network {o['network']!r} refused: every container is offline (--network none)")
    return o


def write_env_file(pairs: list[str], files: list[str], cfg_env: list[str] | None = None) -> str | None:
    lines = []
    for f in files:
        lines += [x for x in Path(f).read_text().splitlines() if x and not x.startswith("#")]
    for p in pairs:
        if "=" in p:
            lines.append(p)
        elif p in os.environ:
            lines.append(f"{p}={os.environ[p]}")
    if not lines:
        return None
    fd, path = tempfile.mkstemp(prefix="env-", dir=state_dir())
    with os.fdopen(fd, "w") as fh:
        for ln in lines:
            k, _, v = ln.partition("=")
            fh.write(f"{k}={json.dumps(v)}\n")  # quoted: values may hold commas, spaces and '='
    return path


def common_flags(o: dict, *, overlay: str | None) -> list[str]:
    """``overlay``: the container's own directory (``new_overlay``): ``ovl/`` is its writable layer, ``tmp/`` its /tmp and
    /var/tmp (``--containall`` would put /tmp in the session tmpfs, capped at 64 MiB on DeltaAI)."""
    f = list(LOCKDOWN)
    user = o.get("user")
    if not user or str(user).split(":")[0] in ("root", "0"):
        # the root-mapped user namespace only: without a subuid range Apptainer would also preload the host's libfakeroot,
        # which needs glibc 2.38 and breaks every image older than the host (Debian bookworm: 2.36; job 3457639)
        f += ["--fakeroot", "--ignore-fakeroot-command"]
    if overlay:
        f += ["--overlay", str(Path(overlay) / "ovl"), "-B", f"{Path(overlay) / 'tmp'}:/tmp", "-B", f"{Path(overlay) / 'vartmp'}:/var/tmp"]
    if o.get("gpus"):
        f.append("--nv")
    for src, dst, ro in o["binds"]:
        check_bind(src)
        f += ["-B", f"{Path(src).resolve()}:{dst}" + (":ro" if ro else "")]
    if os.environ.get("CRUCIBLE_SHIM_LIMITS") == "1":
        if o.get("memory"):
            f += ["--memory", o["memory"]]
        if o.get("cpus"):
            f += ["--cpus", o["cpus"]]
        if o.get("pids"):
            f += ["--pids-limit", str(o["pids"])]
    return f


def run_argv(o: dict, *, overlay: str | None, env_file: str | None, instance: str | None = None) -> list[str]:
    """The ``apptainer`` command for a parsed ``docker run`` (``instance``: ``instance start`` of that name)."""
    sif = sif_of(o["image"])
    cfg = image_config(sif) if sif.exists() else {}
    flags = common_flags(o, overlay=overlay)
    if env_file:
        flags += ["--env-file", env_file]
    wd = o.get("workdir") or cfg.get("WorkingDir")
    if instance:
        return [APPTAINER, "instance", "start", *flags, str(sif), instance]
    if wd:
        flags += ["--pwd", wd]
    cmd = list(o["cmd"])
    if o.get("entrypoint") is not None:
        cmd = [o["entrypoint"], *cmd] if o["entrypoint"] else cmd
    elif not cmd:
        cmd = [*(cfg.get("Entrypoint") or []), *(cfg.get("Cmd") or [])]
    if not cmd:
        return [APPTAINER, "run", *flags, str(sif)]
    return [APPTAINER, "exec", *flags, str(sif), *cmd]


def exec_argv(name: str, cmd: list[str], *, workdir: str | None, env_file: str | None, record: dict) -> list[str]:
    a = [APPTAINER, "exec", "--cleanenv"]
    if env_file:
        a += ["--env-file", env_file]
    wd = workdir or record.get("workdir")
    if wd:
        a += ["--pwd", wd]
    return [*a, f"instance://{name}", *cmd]


def save(name: str, rec: dict) -> None:
    (state_dir() / f"{name}.json").write_text(json.dumps(rec))


def load(name: str) -> dict | None:
    p = state_dir() / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def instances() -> list[str]:
    r = subprocess.run([APPTAINER, "instance", "list", "--json"], capture_output=True, text=True, env=host_env(), check=False)
    try:
        return [i["instance"] for i in json.loads(r.stdout or "{}").get("instances") or []]
    except json.JSONDecodeError:
        return []


def stop(name: str) -> bool:
    r = subprocess.run([APPTAINER, "instance", "stop", "-F", name], capture_output=True, text=True, env=host_env(), check=False)
    rec = load(name)
    if rec:
        if rec.get("overlay"):
            shutil.rmtree(rec["overlay"], ignore_errors=True)
        if rec.get("env_file"):
            Path(rec["env_file"]).unlink(missing_ok=True)
        (state_dir() / f"{name}.json").unlink(missing_ok=True)
    return r.returncode == 0 or rec is not None


def new_overlay(name: str) -> str:
    d = scratch_dir() / f"{name}-{uuid.uuid4().hex[:8]}"
    for sub in ("ovl", "tmp", "vartmp"):
        (d / sub).mkdir(parents=True)
    return str(d)


def cmd_run(argv: list[str]) -> int:
    o = parse_run(argv)
    sif = sif_of(o["image"])
    if not sif.exists():
        raise ShimError(f"no SIF for image {o['image']} ({sif}): import it first (lh_deltaai.py images)")
    env_file = write_env_file(o["env"], o["env_files"])
    name = o["name"] or f"crucible-shim-{uuid.uuid4().hex[:10]}"
    cfg = image_config(sif)
    overlay = new_overlay(name)
    rec = {"name": name, "sif": str(sif), "image": o["image"], "overlay": overlay, "env_file": env_file, "labels": o["labels"],
           "workdir": o.get("workdir") or cfg.get("WorkingDir"), "gpus": o["gpus"]}
    start = run_argv(o, overlay=overlay, env_file=env_file, instance=name)
    r = subprocess.run(start, capture_output=True, text=True, env=host_env(), check=False)
    if r.returncode != 0:
        shutil.rmtree(overlay, ignore_errors=True)
        raise ShimError(f"apptainer instance start failed: {(r.stderr or r.stdout)[-800:]}")
    save(name, rec)
    if rec["workdir"]:  # Docker creates a missing -w directory; Apptainer would refuse to chdir into it (kernel World, smoke 3459051)
        subprocess.run([APPTAINER, "exec", "--cleanenv", "--pwd", "/", f"instance://{name}", "mkdir", "-p", rec["workdir"]], capture_output=True,
                       env=host_env(), check=False)
    cmd = list(o["cmd"]) or [*(cfg.get("Entrypoint") or []), *(cfg.get("Cmd") or [])]
    if o["entrypoint"]:
        cmd = [o["entrypoint"], *o["cmd"]]
    if o["detach"]:
        if cmd and tuple(cmd) not in KEEPALIVE:
            log = state_dir() / f"{name}.log"
            with log.open("ab") as fh:
                subprocess.Popen(exec_argv(name, cmd, workdir=None, env_file=env_file, record=rec), stdout=fh, stderr=fh, stdin=subprocess.DEVNULL,
                                 env=host_env(), start_new_session=True)
        print(hashlib.sha256(name.encode()).hexdigest())  # docker prints the container id
        return 0
    # foreground: run in the instance (so ``docker kill <name>`` works), then remove it
    def _term(*_a: object) -> None:
        stop(name)
        sys.exit(137)

    signal.signal(signal.SIGTERM, _term)
    try:
        p = subprocess.run(exec_argv(name, cmd or ["true"], workdir=None, env_file=env_file, record=rec), env=host_env(), check=False,
                           stdin=None if o["interactive"] else subprocess.DEVNULL)
        return p.returncode
    finally:
        stop(name)


def cmd_exec(argv: list[str]) -> int:
    workdir, env, i, interactive = None, [], 0, False
    while i < len(argv) and argv[i].startswith("-"):
        a = argv[i]
        if a in ("-i", "-t", "-it", "-ti", "--interactive", "--tty", "-d"):
            interactive |= "i" in a
            i += 1
        elif a in ("-u", "--user", "-w", "--workdir", "-e", "--env"):
            if a in ("-w", "--workdir"):
                workdir = argv[i + 1]
            elif a in ("-e", "--env"):
                env.append(argv[i + 1])
            i += 2
        elif a.startswith(("--user=", "--workdir=", "--env=")):
            k, _, v = a.partition("=")
            if k == "--workdir":
                workdir = v
            elif k == "--env":
                env.append(v)
            i += 1
        else:
            raise ShimError(f"docker exec flag {a} is not supported")
    name, cmd = argv[i], argv[i + 1:]
    rec = load(name)
    if rec is None and name not in instances():
        print(f"Error response from daemon: No such container: {name}", file=sys.stderr)
        return 1
    env_file = write_env_file(env, [])
    try:
        p = subprocess.run(exec_argv(name, cmd, workdir=workdir, env_file=env_file, record=rec or {}), env=host_env(), check=False,
                           stdin=None if interactive else subprocess.DEVNULL)
        return p.returncode
    finally:
        if env_file:
            Path(env_file).unlink(missing_ok=True)


def cmd_rm(argv: list[str]) -> int:
    names = [a for a in argv if not a.startswith("-")]
    rc = 0
    for n in names:
        if not stop(n):
            print(f"Error: No such container: {n}", file=sys.stderr)
            rc = 1
    return rc if "-f" not in argv and "--force" not in argv else 0


def fmt_value(template: str, values: dict) -> str:
    def one(m: re.Match) -> str:
        expr = m.group(1).strip()
        if expr.startswith("json "):
            return json.dumps(values.get(expr[5:].strip()))
        v = values.get(expr)
        return "" if v is None else (str(v).lower() if isinstance(v, bool) else str(v))
    return re.sub(r"\{\{\s*([^}]+?)\s*\}\}", one, template)


def cmd_inspect(argv: list[str], *, image: bool = False) -> int:
    fmt, names, i = None, [], 0
    while i < len(argv):
        a = argv[i]
        if a in ("-f", "--format"):
            fmt = argv[i + 1]
            i += 2
            continue
        if a.startswith("--format="):
            fmt = a.split("=", 1)[1]
        elif a.startswith("--type="):
            image = image or a.endswith("=image")
        elif not a.startswith("-"):
            names.append(a)
        i += 1
    out, rc = [], 0
    live = None if image else instances()
    for n in names:
        if image or (live is not None and n not in live and load(n) is None and sif_of(n).exists()):
            sif = sif_of(n)
            if not sif.exists():
                print(f"Error: No such image: {n}", file=sys.stderr)
                rc = 1
                continue
            cfg = image_config(sif)
            vals = {".Id": image_id(sif), ".Architecture": "arm64", ".Os": "linux", ".Config.WorkingDir": cfg.get("WorkingDir", ""),
                    ".Config.User": cfg.get("User", "")}
        else:
            running = n in (live or [])
            if not running and load(n) is None:
                print(f"Error: No such object: {n}", file=sys.stderr)
                rc = 1
                continue
            state = {"Running": running, "Status": "running" if running else "exited", "ExitCode": 0, "OOMKilled": False, "Dead": False,
                     "Paused": False, "Restarting": False}
            vals = {".State.Running": running, ".State.Status": state["Status"], ".Id": hashlib.sha256(n.encode()).hexdigest(),
                    ".Name": "/" + n, ".State.ExitCode": 0, ".State.OOMKilled": False, ".State": state}
        out.append(fmt_value(fmt, vals) if fmt else json.dumps([{k.lstrip("."): v for k, v in vals.items()}]))
    if out:
        print("\n".join(out))
    return rc


def cmd_ps(argv: list[str]) -> int:
    filt, fmt, quiet, i = [], None, False, 0
    while i < len(argv):
        a = argv[i]
        if a in ("--filter", "-f"):
            filt.append(argv[i + 1])
            i += 2
            continue
        if a == "--format":
            fmt = argv[i + 1]
            i += 2
            continue
        if a.startswith("--filter="):
            filt.append(a.split("=", 1)[1])
        elif a.startswith("--format="):
            fmt = a.split("=", 1)[1]
        quiet |= a in ("-q", "--quiet") or (a.startswith("-") and not a.startswith("--") and "q" in a)
        i += 1
    names = instances()
    for f in filt:
        k, _, v = f.partition("=")
        if k == "name":
            names = [n for n in names if v.lstrip("^").rstrip("$") in n]
        elif k == "label":
            names = [n for n in names if v in ((load(n) or {}).get("labels") or [])]
    for n in names:
        print(hashlib.sha256(n.encode()).hexdigest()[:12] if quiet else (fmt_value(fmt, {".Names": n, ".Status": "Up", ".ID": n}) if fmt else n))
    return 0


def cmd_cp(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("-")]
    if len(args) != 2:
        raise ShimError("docker cp needs SRC and DST")
    src, dst = args
    if ":" in src and not src.startswith("/"):
        name, path = src.split(":", 1)
        p = Path(path)
        tar = subprocess.Popen([APPTAINER, "exec", f"instance://{name}", "tar", "-C", str(p.parent), "-cf", "-", p.name], stdout=subprocess.PIPE,
                               env=host_env())
        Path(dst).mkdir(parents=True, exist_ok=True) if not Path(dst).exists() and not Path(dst).parent.exists() else None
        target = Path(dst)
        if target.is_dir():
            r = subprocess.run(["tar", "-C", str(target), "-xf", "-"], stdin=tar.stdout, check=False)
        else:
            tmp = Path(tempfile.mkdtemp(dir=state_dir()))
            r = subprocess.run(["tar", "-C", str(tmp), "-xf", "-"], stdin=tar.stdout, check=False)
            shutil.move(str(tmp / p.name), str(target))
            shutil.rmtree(tmp, ignore_errors=True)
        tar.wait()
        return r.returncode or tar.returncode
    name, path = dst.split(":", 1)
    s = Path(src)
    tar = subprocess.Popen(["tar", "-C", str(s.parent), "-cf", "-", s.name], stdout=subprocess.PIPE)
    r = subprocess.run([APPTAINER, "exec", f"instance://{name}", "sh", "-c", f'mkdir -p "$(dirname {json.dumps(path)})" && tar -C "$(dirname {json.dumps(path)})" -xf -'],
                       stdin=tar.stdout, env=host_env(), check=False)
    tar.wait()
    return r.returncode


def cmd_images(argv: list[str]) -> int:
    quiet = "-q" in argv
    refs = [a for a in argv if not a.startswith("-")]
    for ref in refs:
        sif = sif_of(ref)
        if sif.exists():
            print(image_id(sif)[7:19] if quiet else f"{ref} {sif}")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print("docker (Apptainer shim): run exec rm kill stop inspect image ps cp images info version", file=sys.stderr)
        return 1
    cmd, rest = argv[0], argv[1:]
    try:
        if cmd == "run":
            return cmd_run(rest)
        if cmd == "exec":
            return cmd_exec(rest)
        if cmd in ("rm", "kill", "stop"):
            return cmd_rm(rest if cmd == "rm" else ["-f", *rest])
        if cmd == "inspect":
            return cmd_inspect(rest)
        if cmd == "image" and rest[:1] == ["inspect"]:
            return cmd_inspect(rest[1:], image=True)
        if cmd == "ps" or (cmd == "container" and rest[:1] in (["ls"], ["list"])):
            return cmd_ps(rest if cmd == "ps" else rest[1:])
        if cmd == "cp":
            return cmd_cp(rest)
        if cmd == "images":
            return cmd_images(rest)
        if cmd == "events":  # no daemon event stream: nothing happened that Docker would report (OOM kills are the job cgroup's)
            return 0
        if cmd in ("info", "version"):
            print("Apptainer shim (crucible scripts/deltaai/docker_shim.py); containers are Apptainer instances, offline")
            return 0
        raise ShimError(f"docker {cmd} is not supported by the Apptainer shim (build images with lh_deltaai.py images)")
    except ShimError as e:
        print(f"docker: {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
