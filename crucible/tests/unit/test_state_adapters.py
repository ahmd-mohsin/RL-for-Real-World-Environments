"""M1 state adapters: canonical fingerprints, snapshot/restore round-trips, diffs, queries.

The docker-marked test proves the core claim of the design: the same state has the same
fingerprint locally (macOS, in-process snippet) and inside a Linux container (shipped snippet).
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import uuid
from pathlib import Path

import pytest

from crucible.core.ids import tree_manifest_hash
from crucible.core.schema.state import Query
from crucible.core.store import Store
from crucible.world.exec import DockerExec, LocalExec
from crucible.world.state.base import SnapshotMismatch
from crucible.world.state.fs import FsAdapter, FsGitAdapter
from crucible.world.state.json_doc import JsonDocAdapter
from crucible.world.state.process import ProcessAdapter
from crucible.world.state.snippets.fs import match
from crucible.world.state.sql import SqliteAdapter


def _tree(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.cu").write_text("__global__ void k() {}\n")
    (root / "run.sh").write_text("#!/bin/sh\necho hi\n")
    os.chmod(root / "run.sh", 0o755)
    (root / "README.md").write_text("doc\n")
    os.symlink("src/a.cu", root / "link.cu")
    (root / "build").mkdir()
    (root / "build" / "out.o").write_bytes(b"\x00\x01")


def _git(root: Path) -> None:
    (root / ".gitignore").write_text("build/\n")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"], cwd=root, check=True)
    subprocess.run(["git", "tag", "base"], cwd=root, check=True)


@pytest.mark.parametrize(
    "rel, pat, want",
    [
        ("build/x.o", "build/**", True), ("a/build/x.o", "**/build/**", True), ("a/build", "**/build*", True),
        ("src/a.cu", "*.cu", False), ("src/a.cu", "**/*.cu", True), ("x/y/z", "x", True), ("xy", "x", False),
    ],
)
def test_glob_match(rel, pat, want):
    assert match(rel, pat) is want


async def test_fs_roundtrip_modes_symlinks_and_excludes(tmp_path):
    root = tmp_path / "w"
    _tree(root)
    store = Store(tmp_path / "store")
    ad = FsAdapter("ws", LocalExec(), {"root": str(root), "exclude": ["build/**"]}, store)
    m = await ad.q_manifest()
    paths = {e[0]: e[1] for e in m["entries"]}
    assert paths == {"src/a.cu": "100644", "run.sh": "100755", "README.md": "100644", "link.cu": "120000"}
    assert m["fingerprint"] == tree_manifest_hash([tuple(e) for e in m["entries"]])
    ref0 = await ad.snapshot()
    # mutate: edit, chmod, delete, add, retarget symlink; excluded build file changes must not count
    (root / "src" / "a.cu").write_text("changed\n")
    os.chmod(root / "README.md", 0o755)
    (root / "run.sh").unlink()
    (root / "new.txt").write_text("n")
    os.unlink(root / "link.cu")
    os.symlink("README.md", root / "link.cu")
    (root / "build" / "out.o").write_bytes(b"zzz")
    ref1 = await ad.snapshot()
    assert ref1.fingerprint != ref0.fingerprint
    diff = await ad.diff(ref0, ref1)
    kinds = {c.path: c.change for c in diff.files}
    assert kinds == {"src/a.cu": "modified", "README.md": "mode", "run.sh": "removed", "new.txt": "added", "link.cu": "modified"}
    await ad.restore(ref0)
    assert await ad.fingerprint() == ref0.fingerprint
    assert (root / "build" / "out.o").read_bytes() == b"zzz"  # unmanaged files are left alone
    assert os.access(root / "run.sh", os.X_OK) and os.readlink(root / "link.cu") == "src/a.cu"
    # deterministic tarballs: same state -> same blob
    assert (await ad.snapshot()).blob == ref0.blob


async def test_fs_git_managed_set_and_git_diff(tmp_path):
    root = tmp_path / "repo"
    _tree(root)
    _git(root)
    store = Store(tmp_path / "store")
    ad = FsGitAdapter("repo", LocalExec(), {"root": str(root)}, store)
    m = await ad.q_manifest()
    assert "build/out.o" not in {e[0] for e in m["entries"]} and ".gitignore" in {e[0] for e in m["entries"]}
    assert m["head"]
    ref0 = await ad.snapshot()
    (root / "src" / "a.cu").write_text("ported\n")
    (root / "untracked.cu").write_text("new\n")
    d = await ad.query(Query(op="git_diff", args={"ref": "base"}))
    assert d.ok and "untracked.cu" in d.stdout and "ported" in d.stdout
    # the real index is untouched by git_diff
    st = (await LocalExec().run("git status --porcelain", cwd=str(root))).stdout
    assert "?? untracked.cu" in st
    await ad.restore(ref0)
    assert not (root / "untracked.cu").exists() and (root / "src/a.cu").read_text() == "__global__ void k() {}\n"


async def test_restore_detects_mismatch(tmp_path):
    root = tmp_path / "w"
    _tree(root)
    store = Store(tmp_path / "store")
    ad = FsAdapter("ws", LocalExec(), {"root": str(root)}, store)
    ref = await ad.snapshot()
    bad = ref.model_copy(update={"fingerprint": "0" * 64})
    with pytest.raises(SnapshotMismatch):
        await ad.restore(bad)


def _db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT, amt REAL, blob BLOB);"
        "CREATE INDEX t_name ON t(name);"
        "INSERT INTO t(name, amt, blob) VALUES ('a', 1.5, x'00ff'), ('b', NULL, NULL);"
    )
    con.commit()
    con.close()


async def test_sqlite_canonical_snapshot_restore_diff_query(tmp_path):
    db = tmp_path / "s.db"
    _db(db)
    store = Store(tmp_path / "store")
    ad = SqliteAdapter("erp", LocalExec(), {"path": str(db)}, store)
    ref0 = await ad.snapshot()
    # A live connection (like an App server) stays open across restore and must see restored data.
    live = sqlite3.connect(db)
    live.execute("UPDATE t SET amt = 9.0 WHERE name='a'")
    live.execute("INSERT INTO t(name) VALUES ('c')")
    live.commit()
    ref1 = await ad.snapshot()
    diff = await ad.diff(ref0, ref1)
    assert diff.tables_changed == ["t"] and {r.change for r in diff.rows} == {"inserted", "deleted"}
    await ad.restore(ref0)
    assert live.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2
    assert await ad.fingerprint() == ref0.fingerprint
    # row order / physical layout do not matter; values and schema do
    db2 = tmp_path / "s2.db"
    con = sqlite3.connect(db2)
    con.executescript(
        "CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT, amt REAL, blob BLOB);CREATE INDEX t_name ON t(name);"
        "INSERT INTO t(id, name, amt, blob) VALUES (2, 'b', NULL, NULL), (1, 'a', 1.5, x'00ff');"
    )
    con.commit()
    con.close()
    assert await SqliteAdapter("x", LocalExec(), {"path": str(db2)}, store).fingerprint() == ref0.fingerprint
    r = await ad.query(Query(op="sql", args={"sql": "SELECT name FROM t ORDER BY name"}))
    assert r.ok and r.value["rows"] == [["a"], ["b"]]
    w = await ad.query(Query(op="sql", args={"sql": "DELETE FROM t"}))
    assert not w.ok and "readonly" in w.error.lower()  # verifier queries can never write
    assert (await ad.query(Query(op="count", args={"table": "t", "where": "amt IS NULL"}))).value == 1


async def test_json_doc_and_process(tmp_path):
    p = tmp_path / "api.json"
    p.write_text('{"b": [1, 2], "a": {"x": 1}}')
    store = Store(tmp_path / "store")
    ad = JsonDocAdapter("api", LocalExec(), {"path": str(p)}, store)
    ref0 = await ad.snapshot()
    p.write_text('{"a": {"x": 2}, "b": [1, 2]}')
    ref1 = await ad.snapshot()
    assert (await ad.diff(ref0, ref1)).detail["changed_paths"] == ["/a/x"]
    await ad.restore(ref0)
    assert (await ad.query(Query(op="get", args={"path": "/a/x"}))).value == 1
    proc = ProcessAdapter("tool", LocalExec(), {"cwd": str(tmp_path)}, store)
    r = await proc.query(Query(op="run", args={"cmd": "echo ok && exit 3"}))
    assert r.exit_code == 3 and "ok" in r.stdout and not r.ok
    r = await proc.query(Query(op="run", args={"cmd": "sleep 5", "timeout_s": 0.5}))
    assert r.value["timed_out"]


@pytest.mark.docker
async def test_fingerprints_identical_local_vs_linux_container(tmp_path):
    """Same tree + same db: macOS in-process fingerprint == Linux container fingerprint."""
    root = tmp_path / "w"
    _tree(root)
    _git(root)
    db = tmp_path / "s.db"
    _db(db)
    store = Store(tmp_path / "store")
    name = f"crucible-test-{uuid.uuid4().hex[:8]}"
    host = LocalExec()
    assert (await host.run(f"docker run -d --rm --name {name} python:3.12-slim sleep 600")).ok
    try:
        dx = DockerExec(name)
        r = await dx.run("apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y git >/dev/null 2>&1; git --version", timeout_s=300)
        assert r.ok, r.stdout
        assert (await host.run(f"docker cp {root} {name}:/work && docker cp {db} {name}:/s.db")).ok
        await dx.run("git config --global --add safe.directory '*'")
        local_fs = FsGitAdapter("repo", LocalExec(), {"root": str(root)}, store)
        ctr_fs = FsGitAdapter("repo", dx, {"root": "/work"}, store)
        assert await local_fs.fingerprint() == await ctr_fs.fingerprint()
        local_db = SqliteAdapter("db", LocalExec(), {"path": str(db)}, store)
        ctr_db = SqliteAdapter("db", dx, {"path": "/s.db"}, store)
        assert await local_db.fingerprint() == await ctr_db.fingerprint()
        # cross-backend restore: a snapshot taken locally restores inside the container
        ref = await local_fs.snapshot()
        await dx.run("echo junk > /work/src/a.cu && rm /work/run.sh")
        await ctr_fs.restore(ref)
        assert await ctr_fs.fingerprint() == ref.fingerprint
        dref = await ctr_db.snapshot()
        assert dref.fingerprint == (await local_db.snapshot()).fingerprint
    finally:
        await host.run(f"docker rm -f {name}")
