"""Window lane (09): ``WindowTask`` grading (protected paths restored, B's tests written over the solution, the hidden
tests run apart from the solution), the milestone vector and Phi, ``milestone_status`` for the checkpoint sidecar, the
oracle as unit steps for the horizon replay, and the held-out store refusal. Tests run on the host interpreter."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from test_m17_w_miner import SB, built_record

from crucible.factory.lanes.windows import (
    HELD_OUT_REPOS,
    HeldOutError,
    WindowLane,
    WindowStore,
    mined_of,
    open_store,
    probe_dependencies,
    revert_mutants,
    solution_files,
    split_patch,
    window_checkset,
)

REQS = [{"id": "R1", "milestone": "m01", "text": "alpha() returns 1.", "kind": "behaviour"},
        {"id": "R2", "milestone": "m02", "text": "beta() returns 2.", "kind": "behaviour"},
        {"id": "R3", "milestone": "m03", "text": "gamma() returns one more than beta().", "kind": "behaviour"}]


def make_task(tmp: Path, *, weights: dict | None = None):
    rec, repo, base = built_record(tmp)
    asyncio.run(probe_dependencies(rec, repo, SB()))
    rec.update(asyncio.run(revert_mutants(rec, repo, base, SB())))
    rec["world"] = {"image": "local", "digest": "sha256:" + "0" * 64}
    store = WindowStore(tmp / "store")
    d = store.dir(rec["id"])
    d.mkdir(parents=True)
    subprocess.run(["cp", "-R", str(base), str(d / "base")], check=True)
    store.save_window(rec)
    store.save(mined_of(rec, d / "base"))
    lane = WindowLane(store, repo_dirs={"toyrepo": str(repo.dir)}, sandboxes=lambda image: SB(), weights={rec["id"]: weights} if weights else None)
    prompt = "Release request.\n" + "\n".join(f"- [{r['id']}] {r['text']}" for r in REQS)
    task = lane.materialize({"world": {"task": rec["id"]}, "prompt": prompt, "spec": {"requirements": REQS}, "spans": {},
                             "checkset": window_checkset(rec, REQS)})
    return task, rec, repo


def diff(repo, a: str, b: str, *paths: str) -> str:
    return subprocess.run(["git", "-C", str(repo.dir), "diff", "--binary", a, b, "--", *paths], capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return make_task(tmp_path_factory.mktemp("wtask"))


def test_the_oracle_patch_passes_with_full_phi(built) -> None:
    task, rec, repo = built
    rep = asyncio.run(task.grade_patch(diff(repo, rec["a"], rec["b"], "pkg")))
    assert rep.reward == 1.0 and rep.infra["status"] == "ok"
    assert rep.meta["milestones"] == {"m01": True, "m02": True, "m03": True} and rep.meta["phi"] == 1.0


def test_no_op_and_partial_work_score_their_milestones(built) -> None:
    task, rec, repo = built
    rep = asyncio.run(task.grade_patch(""))
    assert rep.reward == 0.0 and rep.meta["phi"] == 0.0 and not any(rep.meta["milestones"].values())
    u1, u2 = rec["units"][0]["sha"], rec["units"][1]["sha"]
    partial = diff(repo, rec["a"], u2, "pkg")  # alpha and beta, no gamma
    status = asyncio.run(task.milestone_status(partial))
    assert status == {"m01": True, "m02": True, "m03": False}
    assert task.phi(status) == pytest.approx(2 / 3)
    assert asyncio.run(task.milestone_status(diff(repo, rec["a"], u1, "pkg"))) == {"m01": True, "m02": False, "m03": False}


def test_phi_counts_a_milestone_only_with_its_prerequisites(built) -> None:
    task, _, _ = built
    assert [m["requires"] for m in task.milestones] == [[], [], ["m02"]]
    assert task.phi({"m01": False, "m02": False, "m03": True}) == 0.0  # m03 without m02 adds nothing (R5)
    assert task.phi({"m01": True, "m02": True, "m03": True}) == 1.0


def test_injected_weights_are_normalised(tmp_path: Path) -> None:
    task, _, _ = make_task(tmp_path, weights={"m01": 1.0, "m02": 1.0, "m03": 2.0})
    assert task.weights == {"m01": 0.25, "m02": 0.25, "m03": 0.5}
    assert task.phi({"m01": True, "m02": True, "m03": True}) == 1.0


def test_protected_paths_are_restored_and_tests_come_from_b(built) -> None:
    task, rec, repo = built
    hack = ("diff --git a/tests/test_core.py b/tests/test_core.py\n--- a/tests/test_core.py\n+++ b/tests/test_core.py\n@@ -1,5 +1,5 @@\n"
            "-from pkg.core import base\n+from pkg.core import base as alpha\n \n \n def test_base():\n     assert base() == 0\n"
            "diff --git a/conftest.py b/conftest.py\nnew file mode 100644\n--- /dev/null\n+++ b/conftest.py\n@@ -0,0 +1,2 @@\n"
            "+import pytest\n+pytest.skip('everything', allow_module_level=True)\n")
    files, protected, errors = solution_files(task.base_dir, hack)
    assert protected == ["conftest.py", "tests/test_core.py"] and files == {} and errors == []
    rep = asyncio.run(task.grade_patch(hack))
    assert rep.reward == 0.0 and sorted(rep.meta["protected_writes"]) == ["conftest.py", "tests/test_core.py"]
    # the oracle plus a test edit: the edit is dropped, the real tests still pass
    rep = asyncio.run(task.grade_patch(diff(repo, rec["a"], rec["b"], "pkg") + hack))
    assert rep.reward == 1.0 and rep.meta["protected_writes"]


def test_a_patch_that_does_not_apply_scores_zero(built) -> None:
    task, _, _ = built
    bad = "diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n+++ b/pkg/core.py\n@@ -1,2 +1,2 @@\n-def nothing_like_this():\n+def x():\n     return 0\n"
    rep = asyncio.run(task.grade_patch(bad))
    assert rep.reward == 0.0 and rep.meta["apply_errors"] and rep.infra["status"] == "ok"


def test_oracle_steps_replay_milestones_in_history_order(built) -> None:
    task, _, _ = built
    assert [s["function_name"] for s in task.oracle] == ["apply_unit"] * 4
    series = [asyncio.run(task.milestones_of(task.oracle[:k])) for k in range(len(task.oracle) + 1)]
    assert [sorted(k for k, v in s.items() if v) for s in series] == [[], ["m01"], ["m01", "m02"], ["m01", "m02", "m03"], ["m01", "m02", "m03"]]
    g, _ = asyncio.run(task.grade(task.oracle))
    assert g.reward == 1.0


def test_lane_task_contract(built) -> None:
    task, rec, _ = built
    assert {m.id for m in task.mutants()} == {"revert_m01", "revert_m02", "revert_m03"}
    for m in task.mutants():
        g, _ = asyncio.run(task.grade(m.solution))
        assert g.reward == 0.0
    assert task.noop() == {"edits": []} and task.floor_kwargs()["empty"] == {"edits": [], "files": {}}
    assert task.requirement_ids == ["R1", "R2", "R3"] and task.requirement_milestones == {"R1": ["m01"], "R2": ["m02"], "R3": ["m03"]}
    assert task.facts()["dependency_edges"] == 1 and task.pinned_images()["grader"].startswith("sha256:")
    vis = task.agent_visible()
    assert "repo/pkg/core.py" in vis and "repo/pkg/gamma.py" not in vis  # the agent sees A
    _, _, repo = built
    assert asyncio.run(task.property_audit(diff(repo, rec["a"], rec["b"], "pkg")))
    assert task.episode_sidecar(object()) is None  # checkpoint credit is opt-in


def test_split_patch_names_the_path_after_the_change() -> None:
    p = ("diff --git a/x.py b/x.py\ndeleted file mode 100644\n--- a/x.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n"
         "diff --git a/y.py b/z.py\nsimilarity index 90%\nrename from y.py\nrename to z.py\n")
    assert [p for p, _ in split_patch(p)] == ["x.py", "z.py"]


def test_held_out_repositories_never_enter_a_training_store(tmp_path: Path) -> None:
    train = WindowStore(tmp_path / "train", split="train")
    held = next(iter(HELD_OUT_REPOS))
    with pytest.raises(HeldOutError):
        train.save_window({"id": "w", "repo": held})
    with pytest.raises(HeldOutError):
        WindowStore(tmp_path / "train", split="heldout")  # a store keeps its split
    h = WindowStore(tmp_path / "heldout", split="heldout")
    h.save_window({"id": "w", "repo": held})
    with pytest.raises(HeldOutError):
        h.save_window({"id": "w2", "repo": "boltons"})
    assert open_store(tmp_path / "s", held).split == "heldout" and open_store(tmp_path / "s", "boltons").split == "train"
    assert json.loads((tmp_path / "s" / "heldout" / "split.json").read_text())["split"] == "heldout"
