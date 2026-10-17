"""Task content digests (M0 audit 2026-10-14 R-2): core.ids.tree_digest and the runner's checks."""

from __future__ import annotations

import json
import os

import pytest
from crucible_pack_toy.runtime import ORACLE, HelpdeskTask

from crucible.core.ids import tree_digest
from crucible.runtime.harness.native import NativeHarness
from crucible.runtime.lock import RunLockMismatch
from crucible.runtime.run import Runner, RunSpec


def _tree(root):
    (root / "tests").mkdir(parents=True)
    (root / "task.toml").write_text("[task]\nname = 't'\n")
    (root / "tests" / "test.sh").write_text("exit 0\n")
    return root


def test_tree_digest_follows_content_and_names_only(tmp_path):
    root = _tree(tmp_path / "t")
    d = tree_digest(root)
    assert d.startswith("sha256:") and tree_digest(root) == d
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\0")
    (root / ".DS_Store").write_bytes(b"\0")
    assert tree_digest(root) == d
    (root / "tests" / "test.sh").write_text("exit 1\n")
    assert tree_digest(root) != d
    (root / "tests" / "test.sh").write_text("exit 0\n")
    os.rename(root / "tests" / "test.sh", root / "tests" / "check.sh")
    assert tree_digest(root) != d


def test_tree_digest_refuses_a_symlink(tmp_path):
    root = _tree(tmp_path / "t")
    (root / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symbolic link"):
        tree_digest(root)


def test_a_harbor_task_digest_is_harbors_own_hash():
    pytest.importorskip("harbor")
    from pathlib import Path
    from types import SimpleNamespace

    from crucible.interop.harbor.inplace import HarborInPlaceTask
    from crucible.interop.harbor.task_io import load_harbor_task

    seed = Path(__file__).resolve().parents[3] / "migration_tasks" / "task04_pcl_gpu_cuda12"
    if not (seed / "task.toml").exists():
        pytest.skip("seed task absent")
    # task04 grades in a separate container, so it cannot open in place; the digest needs only the task directory
    assert HarborInPlaceTask.digest(SimpleNamespace(dir=seed)) == "sha256:" + load_harbor_task(seed).summary.task_hash


class DirTask:
    """The toy helpdesk task with a task directory: its digest is the directory's; ``edit`` rewrites a file at grading."""

    def __init__(self, root, *, edit=False, digests=None):
        self.inner, self.root, self.edit, self.digests = HelpdeskTask(), root, edit, digests
        self.id = self.inner.id

    def digest(self):
        return self.digests.pop(0) if self.digests else tree_digest(self.root)

    async def grade(self, world, workdir):
        if self.edit:
            (self.root / "tests" / "test.sh").write_text("exit 2\n")
        return await self.inner.grade(world, workdir)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _spec(task, **kw):
    return RunSpec(run_id="r", tasks=[task], policies=[ORACLE], harnesses=[NativeHarness("native:scripted")], k=1, **kw)


def _record(tmp_path):
    (path,) = (tmp_path / "r" / "samples").glob("*/a1/record.json")
    return json.loads(path.read_text())


async def test_an_unchanged_task_records_its_digest(tmp_path):
    task = DirTask(_tree(tmp_path / "task"))
    s = await Runner(tmp_path / "runs").run(_spec(task))
    rec = _record(tmp_path / "runs")
    assert s.scored == 1 and rec["task_digest"] == tree_digest(task.root)
    assert json.loads((tmp_path / "runs" / "r" / "lock.json").read_text())["settings"]["tasks.0.digest"] == tree_digest(task.root)


async def test_a_task_edited_during_the_attempt_is_masked_and_not_retried(tmp_path):
    task = DirTask(_tree(tmp_path / "task"), edit=True)
    s = await Runner(tmp_path / "runs").run(_spec(task))
    rec = _record(tmp_path / "runs")
    assert (s.scored, s.attempts) == (0, 1)
    assert rec["failure"]["kind"] == "crucible:task_changed" and rec["mask_sample"] and "during the attempt" in rec["failure"]["reason"]


async def test_a_task_edited_after_the_lock_is_never_dispatched(tmp_path):
    task = DirTask(_tree(tmp_path / "task"), digests=["sha256:" + "a" * 64, "sha256:" + "b" * 64])
    await Runner(tmp_path / "runs").run(_spec(task))
    rec = _record(tmp_path / "runs")
    events = [json.loads(x)["event"] for x in (tmp_path / "runs" / "r" / "journal.jsonl").read_text().splitlines()]
    assert rec["failure"]["kind"] == "crucible:task_changed" and "since the run's lock" in rec["failure"]["reason"]
    assert "dispatched" not in events


async def test_resuming_with_an_edited_task_is_refused_naming_it(tmp_path):
    task = DirTask(_tree(tmp_path / "task"))
    await Runner(tmp_path / "runs").run(_spec(task))
    (task.root / "tests" / "test.sh").write_text("exit 3\n")
    with pytest.raises(RunLockMismatch, match=r"tasks\.0\.digest"):
        await Runner(tmp_path / "runs").run(_spec(task))
