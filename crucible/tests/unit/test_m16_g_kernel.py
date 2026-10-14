"""M16 agent G: the RLM kernel's snapshots, bounded digests, replayable entropy, describe, lost-between-cells and lifetime.

Covers 04-design.md §3.4 and §5 B4 / B5 / B9 / B10: dill snapshots streamed back in signed chunks (the host never
unpickles), restore = bootstrap re-run + blob + digest check + journal suffix, with fallbacks to full and lenient replay
reported in ``last_restore``; unpicklable or oversized names reported ``lost``; forged snapshot frames rejected; a
bounded per-cell fingerprint and a complete full digest (no 100k-character blind spot); ``time.time`` / ``uuid4``
replayed exactly; ``describe`` for the python-state row; ``lost_between_cells``; per-episode container lifetime and
``renew``; ``JournalReply`` records rebuilt by a resolver on replay. Host-launcher tests (``allow_host=True``, trusted
test code) plus Docker tests (skipped without a daemon).
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from crucible.actors.rlm import kernel as K
from crucible.actors.rlm.kernel import JournalEntry, JournalReply, Kernel, KernelDivergence, KernelError, LocalLauncher, Snapshot
from crucible.world.state.base import SnapshotMismatch

BOOT = '''
OBS = []
class _Result(dict):
    __getattr__ = dict.get
class _World:
    async def call(self, tool, **args):
        r = await host("world.call", tool=tool, args=args)
        res = _Result(r)
        OBS.append({"tool": tool, "args": args, **res})
        return res
world = _World()
def _deliver(msgs):
    OBS.extend(msgs)
'''


async def echo(data):
    if data.get("type") == "world.call":
        i = (data.get("args") or {}).get("i", 0)
        return {"ok": True, "content": f"result {i} " * 20}
    return {"echo": data}


def kern(**kw) -> Kernel:
    return Kernel(LocalLauncher(allow_host=True), cell_timeout_s=kw.pop("cell_timeout_s", 20), **kw)


async def booted(**kw) -> Kernel:
    k = kern(**kw)
    r = await k.execute(BOOT, echo, bootstrap=True)
    assert r.ok, r
    return k


async def done(*ks: Kernel) -> None:
    for k in ks:
        await k.close()
        await k.launcher.aclose()


WORK = [
    "import time, uuid, random\nt0 = time.time(); n0 = time.time_ns(); u = str(uuid.uuid4()); rr = random.random()",
    "r = await world.call('tickets.get', i=1)\nby_id = {'a': OBS[-1]}\ntickets = [OBS[-1]]",
    ("class Ticket:\n    def __init__(self, x):\n        self.x = x\n    def score(self):\n        return self.x + BONUS\nBONUS = 1\nt = Ticket(2)\n"
     "def total():\n    return t.score() + BONUS"),
    "r = await world.call('tickets.get', i=2)\nhandlers = {'total': total}\nseen = {1, 2, 3}",
]
SUFFIX = ["BONUS = 10\nz = total()\nw = random.random()\nr = await world.call('tickets.get', i=3)", "z2 = handlers['total']() + len(OBS)"]


# ---- snapshot / restore -----------------------------------------------------------------------------------------------
async def test_snapshot_restores_exactly_and_replays_only_the_suffix(tmp_path):
    k = await booted()
    k2 = kern()
    try:
        for c in WORK:
            assert (await k.execute(c, echo)).ok
        snap = await k.snapshot(tmp_path / "s.bin")
        assert snap.journal_len == 1 + len(WORK) and not snap.lost and snap.bytes == (tmp_path / "s.bin").stat().st_size
        assert len(snap.bootstrap) == 1 and snap.bootstrap[0]["bootstrap"] is True
        for c in SUFFIX:
            assert (await k.execute(c, echo)).ok
        want = await k.digest()
        await k2.replay(k.journal, snapshot=snap)
        assert k2.last_restore["mode"] == "snapshot" and k2.last_restore["cells"] == len(SUFFIX), k2.last_restore
        assert await k2.digest() == want
        r = await k2.execute("(z, z2, total(), isinstance(t, Ticket), by_id['a'] is tickets[0] is OBS[0], type(r).__name__, "
                             "type(world).__name__, w)", echo)
        r0 = await k.execute("(z, z2, total(), isinstance(t, Ticket), by_id['a'] is tickets[0] is OBS[0], type(r).__name__, "
                             "type(world).__name__, w)", echo)
        assert r.result == r0.result and "True, True" in r.result, (r.result, r0.result)  # live globals, identity, random state
        r = await k2.execute("r = await world.call('tickets.get', i=9)\n(r.ok, len(OBS))", echo)  # the bootstrap's world, live host
        assert r.ok and r.result == "(True, 4)", r
    finally:
        await done(k, k2)


async def test_a_suffix_only_journal_restores_through_the_snapshot_and_reports_lost_names(tmp_path):
    k = await booted()
    k2 = kern()
    try:
        await k.execute("import socket, threading\na, b = socket.socketpair()\ngen = (i for i in range(3))\nlock = threading.Lock()\n"
                        "big = 'x' * (20 * 1024 * 1024)\nkeep = [1, 2, 3]", echo)
        snap = await k.snapshot(tmp_path / "s.bin")
        assert {"gen", "big", "a", "b"} <= set(snap.lost) and "keep" not in snap.lost, snap.lost_reasons
        assert "per-variable" in snap.lost_reasons["big"] and "generator" in snap.lost_reasons["gen"]
        await k.execute("keep.append(4)", echo)
        suffix = [e.to_dict() for e in k.journal[snap.journal_len:]]  # a B5 checkpoint: the snapshot + the cells since
        sd = json.loads(json.dumps(snap.to_dict()))
        await k2.replay(suffix, snapshot=Snapshot.from_dict(sd), offset=sd["journal_len"])
        assert k2.last_restore["from"] == "snapshot" and set(k2.last_restore["lost"]) >= {"gen", "big"}
        r = await k2.execute("(keep, 'big' in globals(), 'gen' in globals(), type(lock).__name__, OBS)", echo)
        assert r.result == "([1, 2, 3, 4], False, False, 'lock', [])", r.result  # lost names are gone, never stale
        assert k2.journal_offset == snap.journal_len and len(k2.journal) == 2
        # this kernel's own suffix journal (journal_offset > 0) restores through its snapshot by itself
        await k2.replay(k2.journal[:-1])
        assert k2.last_restore["from"] == "snapshot" and (await k2.execute("keep", echo)).result == "[1, 2, 3, 4]"
        with pytest.raises(KernelError, match="snapshot"):
            await kern().replay(suffix, offset=sd["journal_len"])
    finally:
        await done(k, k2)


async def test_a_lost_bootstrap_name_keeps_its_fresh_value_and_the_harness_works(tmp_path):
    k = await booted()
    k2 = kern()
    try:
        await k.execute("r = await world.call('t.get', i=1)\nimport threading\nOBS.append(threading.Thread(target=print))", echo)
        snap = await k.snapshot(tmp_path / "s.bin", max_var_bytes=1 << 20)
        assert "OBS" in snap.lost, snap.lost_reasons  # a thread does not pickle
        await k2.replay([], snapshot=snap, offset=snap.journal_len)
        r = await k2.execute("r = await world.call('t.get', i=2)\nlen(OBS)", echo)
        assert r.ok and r.result == "1", r  # OBS is the bootstrap's fresh list: world.call still works
    finally:
        await done(k, k2)


async def test_bootstrap_data_is_bounded_by_the_total_cap_only_and_the_total_cap_holds(tmp_path):
    k = await booted()
    try:
        await k.execute("OBS.extend('o%07d' % i * 250 for i in range(10_000))\nmine = 'y' * (17 * 1024 * 1024)", echo)  # ~20 MB of OBS
        snap = await k.snapshot(tmp_path / "s.bin")
        assert "OBS" not in snap.lost and snap.lost == ["mine"] and snap.bytes > 20_000_000, snap.lost_reasons
        small = await k.snapshot(tmp_path / "t.bin", max_bytes=2 << 20)
        assert "OBS" in small.lost and "total" in small.lost_reasons["OBS"] and small.bytes <= 2 << 20
    finally:
        await done(k)


async def test_snapshot_fallbacks_full_then_lenient_are_reported(tmp_path):
    k = await booted()
    k2 = kern()
    try:
        await k.execute("x = [1, 2]", echo)
        snap = await k.snapshot(tmp_path / "s.bin")
        await k.execute("x.append(3)", echo)
        blob = (tmp_path / "s.bin").read_bytes()
        (tmp_path / "s.bin").write_bytes(blob[:-10] + b"0123456789")  # changed on disk: refused before the kernel loads it
        await k2.replay(k.journal, snapshot=snap)
        assert k2.last_restore["mode"] == "full" and "sha256" in k2.last_restore["fallback"][0], k2.last_restore
        assert (await k2.execute("x", echo)).result == "[1, 2, 3]"
        await k.execute("from datetime import datetime\nwhen = datetime.now()", echo)  # the C clock: not replayable
        await k2.replay(k.journal, snapshot=snap)
        assert k2.last_restore["mode"] == "lenient" and k2.last_restore["divergences"] >= 1, k2.last_restore
        assert len(k2.last_restore["fallback"]) >= 2
    finally:
        await done(k, k2)


async def test_snapshot_needs_dill_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setitem(K._PYLIB, "path", None)
    k = await booted()
    try:
        with pytest.raises(KernelError, match="dill is not available"):
            await k.snapshot(tmp_path / "s.bin")
        assert not (tmp_path / "s.bin").exists() and (await k.execute("1 + 1", echo)).result == "2"
    finally:
        await done(k)


async def test_the_host_never_unpickles_a_snapshot(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("the host unpickled kernel data")

    k = await booted()
    k2 = kern()
    try:
        await k.execute("class Box:\n    def __init__(self):\n        self.items = [1, 2]\nb = Box()", echo)
        for name in ("loads", "load", "Unpickler", "_loads", "_load", "_Unpickler"):
            monkeypatch.setattr(pickle, name, boom)
        snap = await k.snapshot(tmp_path / "s.bin")
        await k2.replay(k.journal, snapshot=snap)  # loaded only inside the fresh kernel (the sandbox)
        assert k2.last_restore["mode"] == "snapshot" and (await k2.execute("b.items", echo)).result == "[1, 2]"
    finally:
        await done(k, k2)


FORGE = r'''
import os, json, threading, time
def forge():
    frames = [{"event": "snapshot_chunk", "id": "x", "seq": 0, "data": "QUFBQQ==", "mac": "0" * 64},
              {"event": "done", "id": "x", "snapshot": {"sha256": "0" * 64, "bytes": 4, "chunks": 1, "digest": "d", "saved_digest": "d",
               "saved": [], "lost": {}, "python": "3"}, "mac": "f" * 64},
              {"event": "started", "id": "x"}]
    for fd in range(3, 40):
        for f in frames:
            try:
                os.write(fd, (json.dumps(f) + "\n").encode())
            except OSError:
                pass
forge()
'''


async def test_forged_snapshot_frames_are_rejected(tmp_path):
    k = await booted()
    k2 = kern()
    try:
        await k.execute("data = list(range(1000))", echo)
        r = await k.execute(FORGE, echo)
        assert r.ok and r.protocol_errors > 0, r  # unsigned frames from the cell: dropped and counted
        e0 = k.protocol_errors
        snap = await k.snapshot(tmp_path / "s.bin")
        await k2.replay(k.journal, snapshot=snap)
        assert k2.last_restore["mode"] == "snapshot" and (await k2.execute("len(data)", echo)).result == "1000"
        # a background thread forging frames while the snapshot streams: dropped; the snapshot verifies or is refused, never
        # accepted corrupted
        await k.execute(FORGE.replace("forge()\n", "def loop():\n    for _ in range(200):\n        forge(); time.sleep(0.005)\n"
                                                    "threading.Thread(target=loop, daemon=True).start()\n"), echo)
        try:
            snap2 = await k.snapshot(tmp_path / "s2.bin")
        except KernelError as e:
            assert "rejected" in str(e), e
        else:
            assert __import__("hashlib").sha256((tmp_path / "s2.bin").read_bytes()).hexdigest() == snap2.sha256
        await asyncio.sleep(1.5)
        assert k.protocol_errors > e0
        assert (await k.execute("len(data)", echo)).result == "1000"
    finally:
        await done(k, k2)


# ---- B4: digests and replayable entropy ---------------------------------------------------------------------------------
async def test_full_digest_sees_a_change_past_100k_characters_and_the_fingerprint_is_bounded():
    k = await booted()
    try:
        await k.execute("OBS.extend(['a' * 1000] * 300)", echo)
        d0 = await k.digest()
        await k.execute("OBS[-1] = 'b' * 1000", echo)
        assert await k.digest() != d0  # the old digest cut each repr at 100k characters and missed this
        await k.execute("OBS.extend('o%07d' % i * 250 for i in range(10_000))", echo)  # 20M characters
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            await k.execute("x = 1", echo)
            ts.append(time.perf_counter() - t0)
        assert sorted(ts)[5] < 0.01, ts  # median trivial cell under 10 ms (it was ~28 ms: a repr of OBS every cell)
    finally:
        await done(k)


async def test_clock_and_uuid_replay_exactly_datetime_now_takes_the_lenient_path():
    k = await booted()
    k2 = kern()
    try:
        await k.execute("import time, uuid, os, secrets, tempfile\nt = [time.time() for _ in range(5)]\nns = time.time_ns()\n"
                        "u = uuid.uuid4().hex\ns = secrets.token_hex(8)\nb = os.urandom(4)", echo)
        await asyncio.sleep(0.05)
        e = k.journal[-1]
        assert e.entropy and len(e.entropy["t"]) == 5 and e.entropy["n"] and len(e.entropy["u"]) >= 3, e.entropy
        want = await k.digest()
        await k2.replay([x.to_dict() for x in k.journal])  # strict: exact
        assert k2.last_restore["mode"] == "full" and await k2.digest() == want
        await k.execute("import datetime\nnow = datetime.datetime.now()", echo)
        with pytest.raises(SnapshotMismatch):
            await k2.replay(k.journal)
        await k2.replay(k.journal, strict=False)
        assert k2.last_restore["mode"] == "lenient" and k2.replay_divergences >= 1
    finally:
        await done(k, k2)


async def test_old_journal_dicts_still_load_and_entries_round_trip():
    e = JournalEntry.from_dict({"code": "x = 1", "timeout_s": 5, "replies": [], "digest": "abc", "done": True})
    assert e.entropy is None and not e.bootstrap and e.full_digest is None
    e2 = JournalEntry("y", 1.0, [], "d", True, {"t": [1.5]}, "fd", True)
    assert JournalEntry.from_dict(json.loads(json.dumps(e2.to_dict()))) == e2


# ---- JournalReply: records instead of large replies (B5) -----------------------------------------------------------------
async def test_journal_reply_records_are_resolved_on_replay(tmp_path):
    store = {"m1": {"text": "ledger event " * 5000}}

    async def handler(data):
        if data["type"] == "history.read":
            return JournalReply(store["m1"], {"ledger": "m1"})
        return await echo(data)

    async def resolve(req, record):
        assert req["type"] == "history.read"
        return store[record["ledger"]]

    k = await booted()
    k2 = kern()
    try:
        await k.execute("h = await host('history.read', i=3)\nn = len(h['text'])", handler)
        rec = k.journal[-1].replies[0]
        assert rec == {"request": {"type": "history.read", "i": 3}, "record": {"ledger": "m1"}}  # the small record only
        snap = await k.snapshot(tmp_path / "s.bin")
        await k.execute("h2 = await host('history.read', i=4)", handler)
        want = await k.digest()
        await k2.replay(k.journal, resolve=resolve)
        assert k2.last_restore["mode"] == "full" and await k2.digest() == want
        await k2.replay(k.journal, snapshot=snap, resolve=lambda req, record: store[record["ledger"]])  # sync resolver too
        assert k2.last_restore["mode"] == "snapshot" and await k2.digest() == want
        with pytest.raises(KernelDivergence, match="rebuild"):
            await k2.replay(k.journal)  # no resolver: a divergence
        await k2.replay(k.journal, strict=False)
        assert k2.last_restore["mode"] == "lenient" and k2.replay_divergences >= 1
    finally:
        await done(k, k2)


# ---- describe -----------------------------------------------------------------------------------------------------------
async def test_describe_lists_the_models_names_with_sizes():
    k = await booted()
    try:
        await k.execute("import json\nrows = [1, 2, 3]\ntext = 'abcd'\n_private = 1\nclass Box:\n    pass\nb = Box()\n"
                        "OBS.append(1)\nmany = None\nfor i in range(600):\n    globals()[f'v{i:03d}'] = i", echo)
        ents = await k.describe()
        by = {e["name"]: e for e in ents}
        assert len(ents) == 500 and k.describe_total == 600 + 8
        assert by["rows"] == {"name": "rows", "type": "list", "size": 3, "unit": "items", "boot": False}
        assert by["text"]["size"] == 4 and by["text"]["unit"] == "chars" and by["OBS"]["boot"] and by["OBS"]["size"] == 1
        assert by["b"]["type"] == "Box" and by["json"]["type"] == "module"
        assert not {"world", "_World", "_Result", "_private", "host", "asyncio"} & set(by)
    finally:
        await done(k)


# ---- B10: a kernel lost between cells ------------------------------------------------------------------------------------
async def test_a_kernel_that_died_between_cells_is_reported_as_such():
    k = await booted(cell_timeout_s=3)
    try:
        await k.execute("x = 1", echo)
        os.killpg(k.pgid, 9)
        await asyncio.sleep(0.3)
        assert not k.alive()
        r = await k.execute("x = 2\nprint('ran')", echo)
        assert r.killed and r.lost_between_cells and "did not run" in r.error and not r.timed_out, r
        r = await k.execute("import os\nos._exit(3)", echo)  # died while running: not "between cells"
        assert r.killed and not r.lost_between_cells, r
    finally:
        await done(k)


async def test_a_kernel_wedged_by_background_work_before_the_cell_is_reported_as_such():
    k = await booted(cell_timeout_s=1, kill_grace_s=1)
    try:
        await k.execute("import asyncio, time\nasync def hog():\n    await asyncio.sleep(0.2)\n    while True:\n        time.sleep(1)\n"
                        "task = asyncio.ensure_future(hog())", echo)
        await asyncio.sleep(0.5)
        r = await k.execute("y = 1", echo)
        assert r.killed and r.lost_between_cells and "unresponsive" in r.error, r
    finally:
        await done(k)


# ---- B9: lifetime ------------------------------------------------------------------------------------------------------
async def test_lifetime_for_episode_and_fixed_containers_refuse_renew():
    assert K.lifetime_for_episode(None) == K.DEFAULT_LIFETIME_S
    assert K.lifetime_for_episode(3600) == 4500 and K.lifetime_for_episode(10, margin_s=0) == 600
    with pytest.raises(KernelError, match="renewable"):
        await K.DockerLauncher().renew()


def _docker_ok() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0


@pytest.mark.docker
async def test_docker_snapshot_and_restore_in_the_container(tmp_path):
    if not _docker_ok():
        pytest.skip("docker is not available")
    launcher = K.DockerLauncher()
    k = Kernel(launcher, cell_timeout_s=30)
    k2 = Kernel(launcher, cell_timeout_s=30)  # the same container (a restore after a wedged cell)
    launcher2 = K.DockerLauncher()
    k3 = Kernel(launcher2, cell_timeout_s=30)  # a new container (the old one was lost)
    try:
        assert (await k.execute(BOOT, echo, bootstrap=True)).ok
        for c in WORK:
            r = await k.execute(c, echo)
            assert r.ok, r
        r = await k.execute("import os\nsorted(os.listdir('/tmp'))", echo)
        assert r.result == "['crucible_rlm_repl.py']", r.result  # dill was loaded into memory and its zip removed
        snap = await k.snapshot(tmp_path / "s.bin")
        assert not snap.lost and snap.python.startswith("3.12"), snap
        for c in SUFFIX:
            assert (await k.execute(c, echo)).ok
        want = await k.digest()
        await k2.replay(k.journal, snapshot=snap)
        assert k2.last_restore["mode"] == "snapshot" and await k2.digest() == want, k2.last_restore
        await k3.replay([e.to_dict() for e in k.journal[snap.journal_len:]], snapshot=snap, offset=snap.journal_len)
        assert k3.last_restore["mode"] == "snapshot" and await k3.digest() == want, k3.last_restore
        r = await k3.execute("r = await world.call('t.get', i=5)\n(z, r.ok, len(OBS))", echo)
        assert r.ok and r.result == "(22, True, 4)", r
    finally:
        for kk in (k, k2, k3):
            await kk.close()
        await launcher.aclose()
        await launcher2.aclose()


@pytest.mark.docker
async def test_docker_renewable_lifetime_is_extended_by_the_host_only():
    if not _docker_ok():
        pytest.skip("docker is not available")
    launcher = K.DockerLauncher(lifetime_s=8, renewable=True)
    k = Kernel(launcher, cell_timeout_s=20)
    try:
        r = await k.execute("try:\n    open('/crucible-ctl/deadline', 'w').write('9999999999'); w = 'writable'\nexcept OSError:\n    w = 'refused'\nw", echo)
        assert r.result == "'refused'", r  # the kernel cannot extend its own life
        d0 = launcher.deadline
        d1 = await launcher.renew(60)
        assert d1 > d0
        await asyncio.sleep(12)  # past the start deadline: still alive
        r = await k.execute("1 + 1", echo)
        assert r.ok and r.result == "2", r
        out = subprocess.run(["docker", "inspect", launcher.name, "--format", "{{json .Config.Cmd}}"], capture_output=True, text=True, check=True).stdout  # noqa: ASYNC221
        assert json.loads(out)[:2] == ["sh", "-c"]
    finally:
        await k.close()
        await launcher.aclose()


@pytest.mark.docker
def test_docker_reap_stale_keeps_an_expired_container_of_a_live_owner_here():
    if not _docker_ok():
        pytest.skip("docker is not available")
    import uuid

    name = f"crucible-rlm-g16-{uuid.uuid4().hex[:8]}"
    try:
        subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "--label", "crucible.rlm=1", "--label", f"{K.OWNER_LABEL}={K.owner_id()}",
                        "--label", f"{K.EXPIRES_LABEL}={int(time.time()) - 5}", "python:3.12-slim", "sleep", "60"], capture_output=True, check=True)
        assert name not in K.reap_stale()  # renewed past its label: the live owner keeps it
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def test_pylib_zip_holds_dill_and_its_licence():
    import zipfile

    p = K.pylib_zip()
    assert p is not None and p.is_file()
    names = zipfile.ZipFile(p).namelist()
    assert "dill/__init__.py" in names and "dill/_dill.py" in names and "dill/LICENSE" in names
    assert not any(n.startswith("dill/tests/") for n in names)


def test_snapshot_dict_round_trip(tmp_path):
    s = Snapshot(tmp_path / "x", 5, "d", ["a"], 10, "0" * 64, "sd", {"a": "why"}, [{"code": "c", "timeout_s": 1}], 3, "3.12.9", 0.5)
    assert Snapshot.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    assert isinstance(Path(s.to_dict()["path"]), Path)
