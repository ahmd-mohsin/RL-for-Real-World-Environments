"""Crucible sandbox guard. STANDALONE (stdlib only); installed root-only in container worlds and run as root.

Defenses ported from Preference Model's Karotte (MIT, https://github.com/preferencemodel/karotte @ 31c693e;
see THIRD_PARTY.md and LICENSES/MIT-karotte.txt): a demoted policy user, process-cohort killing before any verification,
an egress self-test, non-following file access. Git-history scrubbing follows Hugging Face Repo2RLEnv
(Apache-2.0) and adds what it lacks: every ref is removed and unreachability is *verified*.

Usage: ``python3 -I guard.py <op> < args.json``; prints one JSON line prefixed ``__GUARD__``.
Ops: setup, run, quiesce, egress, audit, git_scrub, git_audit.
"""

import errno
import json
import os
import pwd
import select
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time

SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
STATE_DIR = "/var/lib/crucible"
OUT_CAP = 1 << 20  # bytes kept per stream (head); the tail of the last 8 KiB is always kept


def _trusted(binary):
    for d in SAFE_PATH.split(":"):
        p = os.path.join(d, binary)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            st = os.stat(p)
            if st.st_uid == 0 and not st.st_mode & 0o022:
                return p
    raise FileNotFoundError(f"no trusted {binary}")


def _run_root(argv, timeout=120):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env={"PATH": SAFE_PATH, "LC_ALL": "C"}, check=False)


# ----------------------------------------------------------------------------------------- setup
def setup(a):
    user, uid = a.get("user", "agent"), int(a.get("uid", 1000))
    os.makedirs(STATE_DIR, exist_ok=True)
    os.chmod(STATE_DIR, 0o700)
    created = False
    try:
        pw = pwd.getpwnam(user)
        if pw.pw_uid != uid:
            raise SystemExit(f"user {user} exists with uid {pw.pw_uid} != {uid}")
    except KeyError:
        home = a.get("home", "/home/" + user)
        r = _run_root([_trusted("useradd"), "-M", "-u", str(uid), "-U", "-d", home, "-s", "/bin/bash", user]) if _which("useradd") else None
        if r is None or r.returncode != 0:  # minimal images: edit the databases directly
            with open("/etc/group", "a") as g:
                g.write(f"{user}:x:{uid}:\n")
            with open("/etc/passwd", "a") as p:
                p.write(f"{user}:x:{uid}:{uid}::{home}:/bin/bash\n")
        os.makedirs(home, exist_ok=True)
        os.chown(home, uid, uid)
        created = True
    for f in ("/etc/subuid", "/etc/subgid"):  # no user-namespace remapping around uid-based rules (Karotte)
        if os.path.exists(f):
            with open(f) as fh:
                lines = [ln for ln in fh if not ln.startswith((user + ":", f"{uid}:"))]
            with open(f, "w") as fh:
                fh.writelines(lines)
    for path in a.get("writable", []):
        if os.path.exists(path):
            _chown_tree(path, uid)
    for path in a.get("root_only", []):
        if os.path.exists(path):
            os.chown(path, 0, 0)
            os.chmod(path, 0o700)
    py = os.path.realpath(sys.executable)
    info = {"user": user, "uid": uid, "created": created, "python": py, "python_sha256": _sha256(py)}
    with open(os.path.join(STATE_DIR, "guard.json"), "w") as f:
        json.dump(info, f)
    return info


def _which(b):
    try:
        return _trusted(b)
    except FileNotFoundError:
        return None


def _sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _chown_tree(path, uid):
    for dirpath, dirnames, filenames in os.walk(path):
        os.lchown(dirpath, uid, uid)
        for n in filenames + dirnames:
            try:
                os.lchown(os.path.join(dirpath, n), uid, uid)
            except FileNotFoundError:
                pass


# ----------------------------------------------------------------------------------------- run (demoted)
def run(a):
    uid = int(a["uid"])
    pw = pwd.getpwuid(uid)
    env = {"PATH": a.get("path", SAFE_PATH), "HOME": pw.pw_dir, "USER": pw.pw_name, "LOGNAME": pw.pw_name, "LANG": "C.UTF-8",
           "TERM": "dumb", **a.get("env", {})}
    stdin_data = a.get("stdin")

    def demote():  # Karotte subprocess.py _demote(): new session, OOM victim first, drop every privilege
        os.setsid()
        try:
            with open("/proc/self/oom_score_adj", "w") as f:
                f.write("1000")
        except OSError:
            pass
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)

    t0 = time.monotonic()
    # preexec_fn is safe here: the guard is a single-threaded short-lived process (Karotte does the same)
    p = subprocess.Popen(["/bin/bash", "-c", a["cmd"]], cwd=a.get("cwd") or pw.pw_dir, env=env, preexec_fn=demote,  # noqa: PLW1509
                         stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if stdin_data is not None:
        try:
            p.stdin.write(stdin_data.encode())
            p.stdin.close()
        except BrokenPipeError:
            pass
    out, err = bytearray(), bytearray()
    tails = {p.stdout.fileno(): bytearray(), p.stderr.fileno(): bytearray()}
    heads = {p.stdout.fileno(): out, p.stderr.fileno(): err}
    open_fds = [p.stdout.fileno(), p.stderr.fileno()]
    deadline = t0 + float(a.get("timeout", 600))
    timed_out = False
    while open_fds:
        left = deadline - time.monotonic()
        if left <= 0:
            timed_out = True
            break
        ready, _, _ = select.select(open_fds, [], [], min(left, 1.0))
        for fd in ready:
            chunk = os.read(fd, 65536)
            if not chunk:
                open_fds.remove(fd)
                continue
            head = heads[fd]
            if len(head) < OUT_CAP:
                head += chunk[: OUT_CAP - len(head)]
            tails[fd] = (tails[fd] + chunk)[-8192:]
    if timed_out:
        try:
            os.killpg(p.pid, signal.SIGKILL)  # the whole foreground session; leftovers die at quiesce
        except ProcessLookupError:
            pass
    rc = p.wait()

    def text(fd):
        h = bytes(heads[fd])
        if len(h) >= OUT_CAP:
            h = h + b"\n...[truncated]...\n" + bytes(tails[fd])
        return h.decode("utf-8", "replace")

    return {"rc": -9 if timed_out else rc, "stdout": text(p.stdout.fileno()), "stderr": text(p.stderr.fileno()),
            "timed_out": timed_out, "seconds": round(time.monotonic() - t0, 3)}


# ----------------------------------------------------------------------------------------- quiesce (Karotte process_utils)
def _become_uid_and_kill_all(uid, done_fd):
    os.setgroups([])
    os.setgid(uid)
    os.setuid(uid)
    for _ in range(100):  # kill(-1) as the uid reaches every process of the uid, even reparented ones
        try:
            os.kill(-1, signal.SIGKILL)
        except ProcessLookupError:
            break
        except OSError:
            break
    os.write(done_fd, b"\x01")
    os._exit(0)


def _uid_pids(uid):
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/status") as f:
                st = f.read()
        except OSError:
            continue
        real_uid = next((ln.split()[1] for ln in st.splitlines() if ln.startswith("Uid:")), None)
        state = next((ln.split()[1] for ln in st.splitlines() if ln.startswith("State:")), "")
        if real_uid == str(uid) and state != "Z":
            pids.append(int(name))
    return pids


def quiesce(a):
    uid = int(a["uid"])
    if uid == 0:
        raise SystemExit("refusing to quiesce uid 0")
    deadline = time.monotonic() + float(a.get("timeout", 30))
    rounds, killed = 0, 0
    while time.monotonic() < deadline:
        rounds += 1
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            _become_uid_and_kill_all(uid, w)
        os.close(w)
        confirmed = os.read(r, 1) == b"\x01"
        os.close(r)
        os.waitpid(pid, 0)
        survivors = _uid_pids(uid)
        for p in survivors:
            try:
                os.kill(p, signal.SIGKILL)
                killed += 1
            except ProcessLookupError:
                pass
        if confirmed and not _uid_pids(uid):
            return {"ok": True, "rounds": rounds, "killed_in_sweep": killed}
        time.sleep(0.05)
    return {"ok": False, "rounds": rounds, "survivors": _uid_pids(uid)}


# ----------------------------------------------------------------------------------------- egress self-test (Karotte reachable_as)
def egress(a):
    uid = int(a["uid"])
    targets = a.get("targets") or [["1.1.1.1", 80], ["8.8.8.8", 53], ["pypi.org", 443], ["github.com", 443]]
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)
        reachable = []
        for host, port in targets:
            try:
                with socket.create_connection((host, int(port)), timeout=float(a.get("timeout", 3))):
                    reachable.append(f"{host}:{port}")
            except OSError:
                pass
        os.write(w, json.dumps(reachable).encode())
        os._exit(0)
    os.close(w)
    data = b""
    while chunk := os.read(r, 65536):
        data += chunk
    os.close(r)
    os.waitpid(pid, 0)
    reach = json.loads(data or b"[]")
    return {"reachable": reach, "isolated": not reach, "tested": [f"{h}:{p}" for h, p in targets]}


# ----------------------------------------------------------------------------------------- audit
def audit(a):
    """Anomalies a policy can plant: special files in its roots, files of its uid outside them, setuid files."""
    uid = int(a["uid"])
    roots = [os.path.realpath(r) for r in a.get("roots", [])]
    skip = ("/proc", "/sys", "/dev", "/run")
    specials, outside, setuid = [], [], []
    for r in roots:
        for dirpath, dirnames, filenames in os.walk(r):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for n in filenames:
                p = os.path.join(dirpath, n)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                if not (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)):
                    specials.append(p)
    for dirpath, dirnames, filenames in os.walk("/"):
        if dirpath.startswith(skip):
            dirnames[:] = []
            continue
        if any(dirpath == r or dirpath.startswith(r + "/") for r in roots) or dirpath.startswith(("/tmp", "/var/tmp")):
            dirnames[:] = []
            continue
        for n in filenames + dirnames:
            p = os.path.join(dirpath, n)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if st.st_uid == uid and not p.startswith(pwd.getpwuid(uid).pw_dir):
                outside.append(p)
            if stat.S_ISREG(st.st_mode) and st.st_mode & (stat.S_ISUID | stat.S_ISGID) and st.st_uid == uid:
                setuid.append(p)
        if len(outside) > 200:
            break
    return {"specials": specials[:100], "owned_outside_roots": outside[:100], "setuid_by_uid": setuid}


# ----------------------------------------------------------------------------------------- git history
def _git(root, *args, check=True):
    r = subprocess.run([_trusted("git"), "-C", root, *args], capture_output=True, text=True, env={"PATH": SAFE_PATH, "LC_ALL": "C",
                       "GIT_CONFIG_NOSYSTEM": "1", "HOME": "/root"}, timeout=600, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[-400:]}")
    return r.stdout


def git_scrub(a):
    """Reduce a repo to the history reachable from ``keep`` (default: HEAD) and prove nothing else is left."""
    root, keep = a["root"], a.get("keep", "HEAD")
    _git(root, "config", "--global", "--add", "safe.directory", "*", check=False)
    base = _git(root, "rev-parse", keep).strip()
    branch = a.get("branch", "main")
    _git(root, "checkout", "-q", "-B", branch, base)
    for rem in _git(root, "remote").split():
        _git(root, "remote", "remove", rem)
    keep_refs = {f"refs/heads/{branch}"} | ({f"refs/tags/{t}" for t in a.get("keep_tags", ["base"])})
    for ref in _git(root, "for-each-ref", "--format=%(refname)").split():
        if ref not in keep_refs:
            _git(root, "update-ref", "-d", ref)
    for t in a.get("keep_tags", ["base"]):
        _git(root, "tag", "-f", t, base)
    gitdir = _git(root, "rev-parse", "--absolute-git-dir").strip()
    for f in ("FETCH_HEAD", "ORIG_HEAD", "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "logs", "refs/original", "packed-refs"):
        p = os.path.join(gitdir, f)
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.unlink(p)
    _git(root, "reflog", "expire", "--expire=now", "--all", check=False)
    _git(root, "gc", "-q", "--prune=now", check=False)
    _git(root, "repack", "-a", "-d", "-q", check=False)
    _git(root, "prune", "--expire=now", check=False)
    return git_audit({"root": root, "base": base, "forbidden": a.get("forbidden", [])})


def git_audit(a):
    root = a["root"]
    base = a.get("base") or _git(root, "rev-parse", "HEAD").strip()
    refs = _git(root, "for-each-ref", "--format=%(refname)").split()
    all_commits = set(_git(root, "rev-list", "--all").split())
    from_base = set(_git(root, "rev-list", base).split())
    unreachable = [ln for ln in _git(root, "fsck", "--unreachable", "--no-reflogs", "--no-progress", check=False).splitlines() if ln.startswith("unreachable")]
    present = []
    for sha in a.get("forbidden", []):
        r = subprocess.run([_trusted("git"), "-C", root, "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True, env={"PATH": SAFE_PATH, "HOME": "/root"}, check=False)
        if r.returncode == 0:
            present.append(sha)
    remotes = _git(root, "remote").split()
    ok = all_commits <= from_base and not unreachable and not present and not remotes
    return {"ok": ok, "base": base, "refs": refs, "commits": len(all_commits), "beyond_base": sorted(all_commits - from_base)[:20],
            "unreachable_objects": len(unreachable), "forbidden_present": present, "remotes": remotes}


OPS = {"setup": setup, "run": run, "quiesce": quiesce, "egress": egress, "audit": audit, "git_scrub": git_scrub, "git_audit": git_audit}


def main():
    if os.geteuid() != 0:
        raise SystemExit("guard must run as root")
    op = sys.argv[1]
    args = json.loads(sys.stdin.read() or "{}")
    try:
        res = OPS[op](args)
        print("__GUARD__" + json.dumps({"ok": True, "result": res}))
    except BaseException as e:  # noqa: BLE001
        print("__GUARD__" + json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "errno": getattr(e, "errno", None) and errno.errorcode.get(e.errno)}))


if __name__ == "__main__":
    main()
