"""The checkpoint sidecar and the final grader agree on the end state (factory.checkpoint, "One verdict on the end state").

2026-10-25 (framework health, ``credit_grade_mismatch``, CRITICAL): on one held-out debug task the sidecar graded the
episode's final patch with one milestone failed and the final grader passed it on the same bytes; Φ and per-turn credit
never counted what the outcome reward counted. The sidecar's single grading ran during the episode, whose own ``go test``
was killed in the World (the host's load); the killed run's missing tests read as failures. Two fixes, both tested here:
the final grade's vector is the end state's one verdict (``CheckpointSidecar.reconcile``, called by both harnesses), and a
run killed from outside is unmeasured, never a failure of the tests it did not reach (``executable.killed``, the Go
profile's package-level ``signal: killed``)."""

from __future__ import annotations

import json
import random

import pytest
from test_m17_c_checkpoint import FakeTask, cps, drive, edit, rows
from test_m17_c_harness import IDS, FibWindow, _status, llm, run_one

from crucible.factory import checkpoint as ck
from crucible.factory.lanes.executable import killed
from crucible.factory.langs.golang import PROFILE as GO
from crucible.runtime.harness.workspace import WorkspaceHarness


class LoadedSidecarWindow(FibWindow):
    """The sidecar's grading of the output milestone ran under load and lost it (as the excelize grade did); the final
    grade, after the World closed, passes it. Its report carries the milestone vector, as WindowTask's does."""

    async def milestone_status(self, patch: str) -> dict[str, bool]:
        st = await super().milestone_status(patch)
        return {**st, IDS[1]: False}

    async def grade_patch(self, patch, gdir):
        rep = await super().grade_patch(patch, gdir)
        rep.meta = {**(rep.meta or {}), "milestones": _status(patch if isinstance(patch, str) else patch.decode())}
        return rep


async def test_the_end_checkpoint_takes_the_final_grades_vector(tmp_path):
    task = LoadedSidecarWindow()
    adir, rec, ro = await run_one(tmp_path, task, llm(), WorkspaceHarness(sidecar=ck.checkpoint_sidecar(ck.CheckpointConfig(every=3))))
    assert rec["reward"] == 1.0 and not rec["mask_sample"]
    end = [json.loads(x) for x in (adir / "checkpoints.jsonl").read_text().splitlines()][-1]
    final = json.loads((adir / "verifier" / "report.json").read_text()) if (adir / "verifier" / "report.json").exists() else None
    assert end["triggers"] == ["end"] and end["source"] == "final_grade"
    assert end["milestones"] == {IDS[0]: True, IDS[1]: True, IDS[2]: True}  # the final grade's, not the loaded sidecar's
    prog = rec["meta"]["progress"]
    assert prog["phi_final"] == pytest.approx(1.0)  # without reconciling: 2/3 (the output milestone and its credit missing)
    assert prog["reconciled"] and prog["reconciled"][0]["from"][IDS[1]] is False and prog["reconciled"][0]["to"][IDS[1]] is True
    # the per-turn credit sums to the end state's Φ: the gain the final grade counts is booked on a turn
    assert sum(t.advantage for t in ro.turns if t.advantage is not None) == pytest.approx(1.0)
    assert final is None or final.get("reward") == 1.0


async def test_a_masked_final_grade_changes_nothing(tmp_path):
    task = FakeTask()
    side, _ = await drive(task, [(edit(1), ["a"], False), (edit(2), ["a", "c"], False)], tmp_path, every=1)
    before = cps(tmp_path)
    end_patch = json.dumps(["a", "c"])
    assert side.reconcile(end_patch, None) is None  # a masked grade has no vector: never read as all-fail
    assert side.reconcile(end_patch, {"a": True}) is None  # a vector without a verdict for every milestone: unmeasured
    assert side.reconcile(json.dumps(["b"]), {"a": True, "b": True, "c": True}) is None  # no checkpoint holds this patch
    assert side.reconcile(end_patch, {"a": True, "b": False, "c": True}) is None  # already agree
    assert cps(tmp_path) == before


def test_reconcile_before_the_end_does_nothing(tmp_path):
    side = ck.CheckpointSidecar(FakeTask(), tmp_path, ck.CheckpointConfig(snapshot=lambda w: None))
    assert side.reconcile("", {"a": True, "b": True, "c": True}) is None


async def test_invariant_end_vector_equals_the_final_vector(tmp_path):
    """For any episode and any unmasked final vector of its end patch: after reconcile the end checkpoint's vector is the
    final one, Φ_end = Φ(final), and the rows' ΔΦ sum to Φ_end − Φ_start."""
    rng = random.Random(7)
    ids = ["a", "b", "c"]
    for trial in range(25):
        d = tmp_path / f"t{trial}"
        d.mkdir()
        task = FakeTask()
        script = [(edit(i), sorted(rng.sample(ids, rng.randint(0, 3))), False) for i in range(1, rng.randint(2, 7))]
        side, _ = await drive(task, script, d, every=1)
        end_patch = json.dumps(script[-1][1]) if script[-1][1] else ""
        final = {i: rng.random() < 0.5 for i in ids}
        got = side.reconcile(end_patch, final)
        end = cps(d)[-1]
        assert end["milestones"] == final, (trial, end, final)
        phi = ck.potential(final, task.milestones, ck.uniform_weights(task.milestones))
        summary = got if got is not None else side.summary
        assert summary["phi_final"] == pytest.approx(phi)
        assert sum(r["delta_phi"] for r in rows(d)) == pytest.approx(phi - summary["phi0"], abs=1e-6)


async def test_native_harness_reconciles_the_end_checkpoint(tmp_path):
    """``patch=None``: the harness graded the World's end state itself (NativeHarness), the end checkpoint's."""
    task = FakeTask()
    side, _ = await drive(task, [(edit(1), ["a"], False), (edit(2), ["a", "b"], False)], tmp_path, every=1)
    got = side.reconcile(None, {"a": True, "b": True, "c": True})
    assert got is not None and cps(tmp_path)[-1]["milestones"] == {"a": True, "b": True, "c": True} and got["phi_final"] == pytest.approx(1.0)


def test_a_killed_run_is_recognised():
    assert killed("ok  \tgithub.com/x/y\nsignal: killed\nFAIL\tgithub.com/x/y\t151.3s\n") == "signal: killed"
    assert killed("go test exit 137\n") and killed("Killed\n") and killed("fatal error: runtime: out of memory")
    assert not killed("ok  \tgithub.com/x/y\t66.4s\ngo test exit 1\n") and not killed("TestKillProcess passed") and not killed("")


def test_the_go_profile_reports_a_killed_test_binary(tmp_path):
    (tmp_path / "go.mod").write_text("module example.com/m\n")
    (tmp_path / "a_test.go").write_text("package m\n\nimport \"testing\"\n\nfunc TestA(t *testing.T) {}\nfunc TestB(t *testing.T) {}\n")
    ev = [{"Action": "run", "Package": "example.com/m", "Test": "TestA"}, {"Action": "pass", "Package": "example.com/m", "Test": "TestA"},
          {"Action": "run", "Package": "example.com/m", "Test": "TestB"}, {"Action": "output", "Package": "example.com/m", "Output": "signal: killed\n"},
          {"Action": "fail", "Package": "example.com/m", "Elapsed": 151.3}]
    res, notes = GO.parse_report("\n".join(json.dumps(e) for e in ev), tmp_path, ["a_test.go"])
    assert any(n.startswith("killed:") for n in notes)  # LangSandbox.run turns this into a masked run
    assert res.get("a_test.go::TestA") == "passed" and "a_test.go::TestB" not in res
    # a test's own log line saying "signal: killed" is not the binary's kill
    ev2 = [*ev[:3], {"Action": "output", "Package": "example.com/m", "Test": "TestB", "Output": "signal: killed\n"},
           {"Action": "pass", "Package": "example.com/m", "Test": "TestB"}]
    _, notes2 = GO.parse_report("\n".join(json.dumps(e) for e in ev2), tmp_path, ["a_test.go"])
    assert not any(n.startswith("killed:") for n in notes2)
