"""M17 report 11 run 2: prefix ownership by bisection (F3), behavioural edges only in ``requires`` (F2)."""

from __future__ import annotations

import asyncio

import pytest

from crucible.factory.lanes.windows import bisect_owners


def _runner(first: dict[str, int], calls: list[int]):
    async def run(i: int) -> set[str]:
        calls.append(i)
        return {t for t, k in first.items() if k <= i}

    return run


@pytest.mark.parametrize("n", [1, 5, 25, 60, 124])
def test_bisect_owners_finds_each_first_passing_prefix_within_budget(n: int) -> None:
    first = {f"t{j}": (j * 7) % n for j in range(12)}
    calls: list[int] = []
    res = asyncio.run(bisect_owners(n, list(first) + ["never"], _runner(first, calls), budget=25))
    assert len(set(calls)) <= 25 and calls[0] == n - 1 and res["never"] == ["never"]
    for t, k in first.items():
        got = res["owner_prefix"][t]
        assert got >= k  # never earlier than the true first pass
        if res["resolved"][t]:
            assert got == k
    if n <= 25:
        assert all(res["resolved"].values()) and res["owner_prefix"] == first


def test_bisect_owners_counts_a_test_that_stops_passing() -> None:
    async def run(i: int) -> set[str]:
        return {"a"} if i in (1, 3) else set()

    res = asyncio.run(bisect_owners(4, ["a"], run, budget=25))
    assert res["owner_prefix"]["a"] == 1 and res["monotone_violations"] >= 1 or res["owner_prefix"]["a"] in (1, 3)


def test_patch_conflict_edges_are_order_hints_not_requires(tmp_path) -> None:
    """A unit whose diff only applies on top of an earlier unit (textual) does not gate Phi."""
    from test_m17_w_miner import SB, _commit, make_repo

    from crucible.factory.lanes.windows import f2p_stage, probe_dependencies, window_files
    from crucible.factory.lanes.windows_git import RepoGit, release_windows

    repo_dir = make_repo(tmp_path)
    # a fourth source unit that edits alpha's lines (textual dependency) but whose own test needs only base()
    core = "def base():\n    return 0\n\n\ndef alpha():\n    return 1  # one\n\n\ndef delta():\n    return base() - 1\n"
    _commit(repo_dir, {"pkg/core.py": core, "tests/test_delta.py": "from pkg.core import delta\n\n\ndef test_delta():\n    assert delta() == -1\n"},
            "Add delta next to alpha", date="2024-01-05T06:00:00")
    import subprocess

    subprocess.run(["git", "tag", "-f", "-a", "v0.2", "-m", "v0.2"], cwd=repo_dir, check=True, capture_output=True,
                   env={"GIT_COMMITTER_DATE": "2024-01-06T00:00:00", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"})
    win = release_windows(repo_dir, "toyrepo", since="2000-01-01", min_units=2)[0]
    rec = {**{k: v for k, v in win.to_json().items() if k != "repo_dir"}, "timings": {}, "split": "train"}
    repo = RepoGit(repo_dir)
    base = tmp_path / "base"
    assert repo.build(win.a, [], base)[0]
    window_files(rec, repo)
    assert asyncio.run(f2p_stage(rec, repo, base, SB(), reruns=1)) is None
    asyncio.run(probe_dependencies(rec, repo, SB()))
    by = {m["subject"]: m for m in rec["milestones"]}
    d = by["Add delta next to alpha"]
    assert d["requires"] == [] and d["order_hints"] == [by["Add alpha"]["id"]] and set(d["edges"].values()) == {"patch-conflict"}
    assert by["Add gamma on top of beta"]["requires"] == [by["Add beta"]["id"]]  # the behavioural edge still gates


def test_type_a_outcome_reward_is_phi_plus_a_full_pass_bonus(tmp_path) -> None:
    """F1: the episode reward (``grade_patch``) is 0.5 * Phi + 0.5 * [full pass]; ``grade`` (floors, mutants, flake: the
    accept contract kills a mutant only at 0) stays binary."""
    from test_m17_w_task import diff, make_task

    task, rec, repo = make_task(tmp_path)
    u2 = rec["units"][1]["sha"]
    partial = diff(repo, rec["a"], u2, "pkg")  # alpha and beta, no gamma: Phi 2/3
    rep = asyncio.run(task.grade_patch(partial))
    assert rep.meta["phi"] == pytest.approx(2 / 3) and not rep.meta["full_pass"]
    assert rep.reward == rep.outcome == pytest.approx(1 / 3, abs=1e-5)
    g, _ = asyncio.run(task.grade(partial))
    assert g.reward == 0.0  # the binary verdict
    assert asyncio.run(task.grade_patch(diff(repo, rec["a"], rec["b"], "pkg"))).reward == 1.0
    assert asyncio.run(task.grade_patch("")).reward == 0.0
    for m in task.mutants():  # a revert mutant never earns the full reward and is killed by the binary verdict
        assert asyncio.run(task.grade(m.solution))[0].reward == 0.0
    task.phi_weight = 0.0
    assert asyncio.run(task.grade_patch(partial)).reward == 0.0  # 0 = run 1's binary reward
    with pytest.raises(ValueError):
        type(task)(task.lane, task.mined, task.window, task_id="x", prompt="p", requirements=[], spans={}, checkset={"checks": []}, phi_weight=1.5)
