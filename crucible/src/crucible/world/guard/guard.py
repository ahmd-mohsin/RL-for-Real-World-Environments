"""Crucible sandbox guard. STANDALONE (stdlib only); installed root-only in container worlds and run as root.

Defenses ported from Preference Model's Karotte (MIT, https://github.com/preferencemodel/karotte @ 31c693e;
see THIRD_PARTY.md and LICENSES/MIT-karotte.txt): a demoted policy user, process-cohort killing before any verification,
an egress self-test, non-following file access. Git-history scrubbing follows Hugging Face Repo2RLEnv
(Apache-2.0) and adds what it lacks: every ref is removed and unreachability is *verified*.

Usage: ``python3 -I guard.py <op> < args.json``; prints one JSON line prefixed ``__GUARD__``.
Ops: setup, run, quiesce, egress, firewall, firewall_lift, audit, git_scrub, git_audit, git_trust, git_reset, own, reclaim.
"""

import errno
import grp
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
def _owner_name(uid):
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return grp.getgrgid(uid).gr_name


def _uid_taken(uid):
    try:
        pwd.getpwuid(uid)
        return True
    except KeyError:
        pass
    try:
        grp.getgrgid(uid)
        return True
    except KeyError:
        return False


def setup(a):
    """Create a fresh, unprivileged policy account. An account that already exists in the image is never reused:
    images ship users (e.g. ``ubuntu`` at uid 1000) that may own files, belong to groups or be in sudoers."""
    user = a.get("user", "agent")
    want = a.get("uid")
    os.makedirs(STATE_DIR, exist_ok=True)
    os.chmod(STATE_DIR, 0o700)
    try:
        pw = pwd.getpwnam(user)
        info_path = os.path.join(STATE_DIR, "guard.json")
        prior = {}
        if os.path.exists(info_path):
            with open(info_path) as f:
                prior = json.load(f)
        if not (prior.get("created") and prior.get("uid") == pw.pw_uid):  # only our own account from an earlier setup
            raise SystemExit(f"user {user!r} already exists in the image (uid {pw.pw_uid}); pick another policy_user")
        uid, created = pw.pw_uid, True
    except KeyError:
        if want is not None:
            uid = int(want)
            if _uid_taken(uid):
                raise SystemExit(f"uid {uid} is already used by the image ({_owner_name(uid)}); leave policy_uid unset to allocate one") from None
        else:
            uid = next(u for u in range(1000, 60000) if not _uid_taken(u))
        home = a.get("home", "/home/" + user)
        if _which("useradd"):
            r = _run_root([_trusted("useradd"), "-M", "-u", str(uid), "-U", "-d", home, "-s", "/bin/bash", user])
            if r.returncode != 0:
                raise SystemExit(f"useradd failed: {r.stderr.strip()[-300:]}")
        else:  # minimal images without shadow-utils: edit the databases directly (uid and name verified free above)
            with open("/etc/group", "a") as g:
                g.write(f"{user}:x:{uid}:\n")
            with open("/etc/passwd", "a") as p:
                p.write(f"{user}:x:{uid}:{uid}::{home}:/bin/bash\n")
        os.makedirs(home, exist_ok=True)
        os.chown(home, uid, uid)
        created = True
    member_of = [g.gr_name for g in grp.getgrall() if user in g.gr_mem]
    if member_of:
        raise SystemExit(f"policy user {user!r} is a member of {member_of}")
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
    unsafe = {p: why for p in [py, *[d for d in a.get("root_path", "").split(":") if d]] if (why := _policy_can_alter(p, uid))}
    if unsafe:  # a policy-writable PATH dir or interpreter would let it plant binaries the verifier runs as root
        raise SystemExit(f"root's PATH/interpreter can be altered by the policy: {unsafe}")
    info = {"user": user, "uid": uid, "created": created, "python": py, "python_sha256": _sha256(py)}
    with open(os.path.join(STATE_DIR, "guard.json"), "w") as f:
        json.dump(info, f)
    return info


def _policy_can_alter(path, uid):
    """Why ``uid`` could replace ``path`` (a file or dir), or None: any component owned by it, or writable by
    group/other without the sticky bit (Karotte check_paths / trusted_bin)."""
    path = os.path.abspath(path)
    parts = path.split("/")
    for i in range(1, len(parts) + 1):
        comp = "/".join(parts[:i]) or "/"
        try:
            st = os.lstat(comp)
        except FileNotFoundError:
            return None  # a missing PATH entry cannot be created unless its parent is writable (checked above)
        if st.st_uid == uid:
            return f"{comp} is owned by the policy"
        if st.st_uid != 0:
            return f"{comp} is owned by uid {st.st_uid}"
        if stat.S_ISLNK(st.st_mode):  # a link's own mode bits mean nothing; what it points to (and the rest) must hold
            rest = "/".join(parts[i:])
            target = os.path.join(os.path.dirname(comp), os.readlink(comp))
            return _policy_can_alter(os.path.join(target, rest) if rest else target, uid)
        if st.st_mode & 0o022 and not (stat.S_ISDIR(st.st_mode) and st.st_mode & stat.S_ISVTX):
            return f"{comp} is group/other-writable"
    return None


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
PR_SET_NO_NEW_PRIVS = 38


def _no_new_privs():
    """prctl(PR_SET_NO_NEW_PRIVS): inherited by every descendant and irreversible, so no setuid binary, file
    capability or sudo rule can raise the policy's privileges. Per process tree, so it needs no container flag
    (``--security-opt no-new-privileges`` breaks ``--init`` on some hosts)."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")


def run(a):
    uid = int(a["uid"])
    pw = pwd.getpwuid(uid)
    env = {"PATH": a.get("path", SAFE_PATH), "HOME": pw.pw_dir, "USER": pw.pw_name, "LOGNAME": pw.pw_name, "LANG": "C.UTF-8",
           "TERM": "dumb", **a.get("env", {})}
    stdin_data = a.get("stdin")

    nproc = a.get("nproc")

    def demote():  # Karotte subprocess.py _demote(): new session, OOM victim first, drop every privilege
        os.setsid()
        _no_new_privs()
        if nproc:  # below the container's pids limit: a fork bomb can never starve the harness of the pids to stop it
            import resource

            resource.setrlimit(resource.RLIMIT_NPROC, (int(nproc), int(nproc)))
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
            p.stdin.write(stdin_data.encode("utf-8", "surrogateescape"))
            p.stdin.close()
        except BrokenPipeError:
            pass
    out, err = bytearray(), bytearray()
    tails = {p.stdout.fileno(): bytearray(), p.stderr.fileno(): bytearray()}
    heads = {p.stdout.fileno(): out, p.stderr.fileno(): err}
    open_fds = [p.stdout.fileno(), p.stderr.fileno()]
    deadline = t0 + float(a.get("timeout", 600))
    timed_out = False
    drain_until = None  # once the command itself exits, background jobs holding our pipes get a short grace, not the timeout
    while open_fds:
        now = time.monotonic()
        if drain_until is None and p.poll() is not None:
            drain_until = now + float(a.get("drain", 1.0))
        if drain_until is not None and now >= drain_until:
            break
        left = deadline - now
        if left <= 0:
            timed_out = True
            break
        ready, _, _ = select.select(open_fds, [], [], min(left, 0.2))
        for fd in ready:
            chunk = os.read(fd, 65536)
            if not chunk:
                open_fds.remove(fd)
                continue
            head = heads[fd]
            if len(head) < OUT_CAP:
                head += chunk[: OUT_CAP - len(head)]
            tails[fd] = (tails[fd] + chunk)[-8192:]
    # The call is over: its process group goes with it (plain `cmd &` jobs, fork bombs). Deliberately detached work
    # (setsid / nohup setsid) has left the group and lives until the next quiesce.
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    rc = p.wait()
    detached = bool(open_fds) and not timed_out  # background jobs still hold the pipes; they live on until quiesce
    p.stdout.close()
    p.stderr.close()

    def text(fd):
        h = bytes(heads[fd])
        if len(h) >= OUT_CAP:
            h = h + b"\n...[truncated]...\n" + bytes(tails[fd])
        return h.decode("utf-8", "replace")

    out_fd, err_fd = list(heads)
    return {"rc": -9 if timed_out else rc, "stdout": text(out_fd), "stderr": text(err_fd),
            "timed_out": timed_out, "detached": detached, "seconds": round(time.monotonic() - t0, 3)}


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
# Where a published fix lives (Repo2RLEnv: GitHub, PyPI wheels and sdists, mirrors, archives): none of these may be
# reachable from a no-network or allowlist world unless the world allows it by name.
ANSWER_SOURCES = ["github.com", "codeload.github.com", "raw.githubusercontent.com", "objects.githubusercontent.com", "gitlab.com",
                  "bitbucket.org", "pypi.org", "files.pythonhosted.org", "registry.npmjs.org", "crates.io", "static.crates.io",
                  "proxy.golang.org", "conda.anaconda.org", "huggingface.co", "web.archive.org", "sourceforge.net", "git.kernel.org"]


def allowed(host, allow):
    """``allow`` entries are host names, ``*.suffix`` wildcards or IPs (Harbor / agentenv allowlist syntax)."""
    host = str(host).lower().rstrip(".")
    for a in allow or []:
        a = str(a).lower().rstrip(".")
        if host == a or (a.startswith("*.") and (host == a[2:] or host.endswith(a[1:]))):
            return True
    return False


def _reach(host, port, timeout):
    """True if the policy user gets a working connection. Port 443 needs a completed TLS handshake with the host's SNI:
    a transparent egress proxy (Harbor's docker sidecar) accepts every TCP connection and drops denied names at SNI."""
    import ssl

    try:
        with socket.create_connection((host, int(port)), timeout=timeout) as sock:
            if int(port) != 443:
                return True
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE  # reachability, not authenticity: a MITM proxy that answers counts as reachable
            sock.settimeout(timeout)
            name = None if all(c.isdigit() or c in ".:" for c in str(host)) else host
            with ctx.wrap_socket(sock, server_hostname=name):
                return True
    except (OSError, ValueError):
        return False


def egress(a):
    uid = int(a["uid"])
    allow = a.get("allowed") or []
    targets = a.get("targets") or [["1.1.1.1", 80], ["8.8.8.8", 53], ["169.254.169.254", 80], ["2606:4700:4700::1111", 80],  # cloud metadata, IPv6
                                   *[[h, 443] for h in ANSWER_SOURCES], ["pypi.org", 80], ["github.com", 22]]
    gw = _default_gateway()
    if gw:
        targets = [*targets, [gw, 80], [gw, 53], [gw, 22]]
    targets = [*targets, *a.get("extra_targets", [])]  # e.g. host services the backend runs (gateway, App servers)
    denied = [t for t in targets if not allowed(t[0], allow)]
    lanes = [[h, 443] for h in allow if not str(h).startswith("*.")]
    timeout = float(a.get("timeout", 3))
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)
        reachable = [f"{h}:{p}" for h, p in denied if _reach(h, p, timeout)]
        if _udp_dns_answers("8.8.8.8", timeout) and not allowed("8.8.8.8", allow):
            reachable.append("udp:8.8.8.8:53")
        lane = {f"{h}:{p}": _reach(h, p, timeout) for h, p in lanes}
        os.write(w, json.dumps({"reachable": reachable, "lanes": lane}).encode())
        os._exit(0)
    os.close(w)
    data = b""
    while chunk := os.read(r, 65536):
        data += chunk
    os.close(r)
    os.waitpid(pid, 0)
    out = json.loads(data or b"{}")
    reach, lanes_ok = out.get("reachable", []), out.get("lanes", {})
    return {"reachable": reach, "isolated": not reach, "tested": [f"{h}:{p}" for h, p in denied],
            "allowed": list(allow), "allowed_reachable": lanes_ok, "lanes_ok": all(lanes_ok.values()) if lanes_ok else None}


def _default_gateway():
    try:
        with open("/proc/net/route") as f:
            for line in f.readlines()[1:]:
                fields = line.split()
                if fields[1] == "00000000":
                    return socket.inet_ntoa(int(fields[2], 16).to_bytes(4, "little"))
    except (OSError, IndexError, ValueError):
        pass
    return None


def _udp_dns_answers(server, timeout):
    query = b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(timeout)
            s.sendto(query, (server, 53))
            return bool(s.recv(512))
    except OSError:
        return False


# ----------------------------------------------------------------------------------------- per-uid firewall (Karotte confinement)
GVISOR_SENTRY_PROC = "/proc/sentry-meminfo"  # Karotte container.is_gvisor


def _own_addresses():
    """The sandbox's own non-loopback, non-link-local addresses (Karotte ``_own_addresses``, read from /proc, no psutil)."""
    import ipaddress

    v4, v6 = set(), set()
    try:
        with open("/proc/net/fib_trie") as f:
            last = None
            for line in f:
                t = line.strip()
                if t.startswith(("|--", "+--")):
                    last = t.split()[-1]
                elif t.startswith("/32 host LOCAL") and last:
                    v4.add(last)
    except OSError:
        pass
    try:
        with open("/proc/net/if_inet6") as f:
            for line in f:
                h = line.split()[0]
                v6.add(str(ipaddress.ip_address(":".join(h[i:i + 4] for i in range(0, 32, 4)))))
    except OSError:
        pass

    def keep(a):
        ip = ipaddress.ip_address(a)
        return not (ip.is_loopback or ip.is_link_local)

    return sorted(a for a in v4 if keep(a)), sorted(a for a in v6 if keep(a))


def _resolve(hosts):
    """IPs of allowlisted names (resolved once, at setup, as root; ``*.suffix`` wildcards cannot be resolved and are
    reported), plus literal IPs and CIDRs as given."""
    import ipaddress

    v4, v6, unresolved = set(), set(), []
    for h in hosts or []:
        h = str(h).strip()
        try:
            net = ipaddress.ip_network(h, strict=False)
            (v4 if net.version == 4 else v6).add(str(net) if "/" in h else str(net.network_address))
            continue
        except ValueError:
            pass
        if h.startswith("*."):
            unresolved.append(h)
            continue
        try:
            for fam, _, _, _, sa in socket.getaddrinfo(h, None):
                (v4 if fam == socket.AF_INET else v6).add(sa[0].split("%", 1)[0])
        except OSError:
            unresolved.append(h)
    return sorted(v4), sorted(v6), unresolved


def _fw_bins():
    if os.path.exists(GVISOR_SENTRY_PROC):  # gVisor: iptables-legacy, no RST/ICMP; its rules are accepted and not enforced
        return "iptables-legacy", "ip6tables-legacy", "DROP", True
    return "iptables", "ip6tables", "REJECT", False


def _fw_apply(binary, rules):
    for rule in rules:
        r = _run_root([binary, *rule], timeout=30)
        if r.returncode != 0:
            return f"{binary} {' '.join(rule)}: {(r.stderr or r.stdout).strip()[-300:]}"
    return None


def firewall(a):
    """Per-uid egress rules for the policy (Karotte ``restrict_to_internal_network`` / ``deny_all_network``):
    ``blocked_ports`` on localhost are dropped first (App servers the policy must reach only through tools); then, for
    ``mode`` ``allowlist``, the uid may reach localhost, the sandbox's own addresses and the allowlisted IPs and nothing
    else; ``deny`` keeps it off every address but localhost; ``ports`` only blocks the ports. Rules take effect for new
    connections and need CAP_NET_ADMIN. ``took`` is False where they cannot be enforced (gVisor accepts and ignores them)."""
    uid = str(int(a["uid"]))
    mode = a.get("mode", "allowlist")
    if mode not in ("allowlist", "deny", "ports"):
        raise ValueError(f"unknown firewall mode {mode!r}")
    ipt, ip6t, reject, gvisor = _fw_bins()
    try:
        ipt, ip6t = _trusted(ipt), _trusted(ip6t)
    except FileNotFoundError as e:
        return {"took": False, "error": str(e), "rules": 0}
    own4, own6 = _own_addresses()
    al4, al6, unresolved = _resolve(a.get("allowed") if mode == "allowlist" else [])
    who = ["-A", "OUTPUT", "-m", "owner", "--uid-owner", uid]
    ports = [int(p) for p in a.get("blocked_ports") or []]
    v4 = [[*who, "-p", proto, "--dport", str(p), "-d", "127.0.0.0/8", "-j", "DROP"] for p in ports for proto in ("tcp", "udp")]
    v6 = [[*who, "-p", proto, "--dport", str(p), "-d", "::1", "-j", "DROP"] for p in ports for proto in ("tcp", "udp")]
    v4 += [[*who, "-p", proto, "--dport", str(p), "-d", ip, "-j", "DROP"] for p in ports for proto in ("tcp", "udp") for ip in own4]
    if mode != "ports":
        v4 += [[*who, "-d", "127.0.0.0/8", "-j", "ACCEPT"], *[[*who, "-d", ip, "-j", "ACCEPT"] for ip in [*own4, *al4]], [*who, "-j", reject]]
        v6 += [[*who, "-d", "::1", "-j", "ACCEPT"], *[[*who, "-d", ip, "-j", "ACCEPT"] for ip in [*own6, *al6]], [*who, "-j", reject]]
    err = _fw_apply(ipt, v4)
    err6 = None if err else _fw_apply(ip6t, v6)
    if err6 and "ip6tables" in err6 and ("not supported" in err6 or "Address family" in err6 or "can't initialize" in err6.lower()):
        err6 = None  # no IPv6 stack in this sandbox: nothing to filter
    took = err is None and err6 is None and not gvisor
    return {"took": took, "error": err or err6, "gvisor": gvisor, "rules": len(v4) + len(v6), "own": own4 + own6, "allowed_ips": al4 + al6,
            "unresolved": unresolved, "blocked_ports": ports, "mode": mode}


def firewall_lift(a):
    """Delete every OUTPUT rule owned by ``uid`` (Karotte ``lift_network_rules``); a reused sandbox starts clean."""
    import itertools
    import shlex

    uid = str(int(a["uid"]))
    ipt, ip6t, _, _ = _fw_bins()
    left = 0
    for name in (ipt, ip6t):
        try:
            binary = _trusted(name)
        except FileNotFoundError:
            continue
        listing = _run_root([binary, "-S", "OUTPUT"], timeout=30)
        if listing.returncode != 0:
            continue
        for line in listing.stdout.splitlines():
            words = shlex.split(line)
            owned = words[:2] == ["-A", "OUTPUT"] and any(w == "--uid-owner" and v == uid for w, v in itertools.pairwise(words))
            if owned and _run_root([binary, "-D", *words[1:]], timeout=30).returncode != 0:
                left += 1
    return {"clean": left == 0, "left": left}


# ----------------------------------------------------------------------------------------- reset hygiene (Karotte reclaim)
def own(a):
    """Re-own the policy's roots after a restore (directories and files written by root), never following links."""
    uid = int(a["uid"])
    for path in a.get("paths", []):
        if os.path.lexists(path) and not os.path.islink(path):
            _chown_tree(path, uid)
    return {"owned": a.get("paths", [])}


def _sweep_uid(top, uid, keep_top=True):
    removed = 0
    if not os.path.isdir(top) or os.path.islink(top):
        return 0
    for dirpath, dirnames, filenames in os.walk(top, topdown=False, followlinks=False):
        for n in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            p = os.path.join(dirpath, n)
            try:
                if os.lstat(p).st_uid == uid:
                    os.unlink(p)
                    removed += 1
            except OSError:
                pass
        for d in dirnames:
            p = os.path.join(dirpath, d)
            try:
                st = os.lstat(p)
                if stat.S_ISDIR(st.st_mode) and st.st_uid == uid:
                    os.rmdir(p)
                    removed += 1
            except OSError:
                pass
    return removed


def _ipc_objects(uid):
    out = []
    for kind in ("shm", "msg", "sem"):
        try:
            with open(f"/proc/sysvipc/{kind}") as f:
                rows = f.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            cols = row.split()
            if int(cols[7] if kind != "sem" else cols[4]) == uid:
                out.append((kind, int(cols[1])))
    return out


def _ipc_clean(uid):
    """Remove SysV shared memory, message queues and semaphores owned by the uid (they outlive every process).
    Container root has no CAP_IPC_OWNER, so the removal runs in a child that has become the owner."""
    objs = _ipc_objects(uid)
    if not objs:
        return 0
    pid = os.fork()
    if pid == 0:
        try:
            import ctypes

            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)
            libc = ctypes.CDLL(None, use_errno=True)
            for kind, ident in objs:  # IPC_RMID = 0
                if kind == "shm":
                    libc.shmctl(ident, 0, None)
                elif kind == "msg":
                    libc.msgctl(ident, 0, None)
                else:
                    libc.semctl(ident, 0, 0)
        finally:
            os._exit(0)
    os.waitpid(pid, 0)
    left = _ipc_objects(uid)
    if left:
        raise RuntimeError(f"SysV IPC objects of uid {uid} survived removal: {left}")
    return len(objs)


def reclaim(a):
    """Between episodes: nothing the policy left outside its restored roots survives (Karotte reclaim.delete_files)."""
    uid = int(a["uid"])
    home = pwd.getpwuid(uid).pw_dir
    removed = {t: _sweep_uid(t, uid) for t in ("/tmp", "/var/tmp", "/dev/shm", "/run/user", *a.get("extra", []))}
    if os.path.isdir(home) and not os.path.islink(home):  # the home starts empty every episode
        for e in os.scandir(home):
            p = os.path.join(home, e.name)
            if e.is_dir(follow_symlinks=False):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.unlink(p)
        os.chown(home, uid, uid)
        os.chmod(home, 0o755)
    removed["sysv_ipc"] = _ipc_clean(uid)
    return {"removed": removed}


def git_trust(a):
    """Copy a repo's .git to root-only storage before the policy runs; verifier-side git uses only this copy."""
    root, dest = a["root"], a["dest"]
    os.makedirs(os.path.dirname(dest), mode=0o700, exist_ok=True)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(os.path.join(root, ".git"), dest, symlinks=True)
    _chown_tree(dest, 0)
    os.chmod(dest, 0o700)
    return {"git_dir": dest}


def git_reset(a):
    """After a restore: the policy's .git becomes the trusted copy again (its commits, refs, hooks and config go)."""
    root, src, uid = a["root"], a["src"], int(a["uid"])
    gd = os.path.join(root, ".git")
    if os.path.lexists(gd):
        if os.path.islink(gd) or not os.path.isdir(gd):
            os.unlink(gd)
        else:
            shutil.rmtree(gd)
    shutil.copytree(src, gd, symlinks=True)
    _chown_tree(gd, uid)
    return {"reset": gd}


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
GIT_ENV = {"PATH": SAFE_PATH, "LC_ALL": "C", "HOME": "/nonexistent", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
GIT_FLAGS = ["-c", "safe.directory=*", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null"]  # per call, never global


def _git(root, *args, check=True):
    r = subprocess.run([_trusted("git"), *GIT_FLAGS, "-C", root, *args], capture_output=True, text=True, env=GIT_ENV, timeout=600, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[-400:]}")
    return r.stdout


def git_scrub(a):
    """Reduce a repo to the history reachable from ``keep`` (default: HEAD) and prove nothing else is left."""
    root, keep = a["root"], a.get("keep", "HEAD")
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
        r = subprocess.run([_trusted("git"), *GIT_FLAGS, "-C", root, "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True, env=GIT_ENV, check=False)
        if r.returncode == 0:
            present.append(sha)
    remotes = _git(root, "remote").split()
    ok = all_commits <= from_base and not unreachable and not present and not remotes
    return {"ok": ok, "base": base, "refs": refs, "commits": len(all_commits), "beyond_base": sorted(all_commits - from_base)[:20],
            "unreachable_objects": len(unreachable), "forbidden_present": present, "remotes": remotes}


OPS = {"setup": setup, "run": run, "quiesce": quiesce, "egress": egress, "audit": audit, "git_scrub": git_scrub, "git_audit": git_audit,
       "own": own, "reclaim": reclaim, "git_trust": git_trust, "git_reset": git_reset, "firewall": firewall, "firewall_lift": firewall_lift}


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
