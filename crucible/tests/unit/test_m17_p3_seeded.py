"""P3 exit criterion (swarm-pipeline/00 §15): seeded traces (a lucky pass, an unstated public name, a broken World tool)
become accepted, versioned fixes with executed evidence, and a fake proposal (a check that also fails the oracle) is
rejected. Every grade and World command is executed by pytest (locally here: trusted seeded code; in ``crucible-p3-*``
containers in the Docker variant and in ``scripts/lh_audit.py seeded``)."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from crucible.factory.traceaudit import AcceptConfig, LoopConfig, PytestRunner, load_bundle, run_seeded
from crucible.factory.versioning import VersionStore

CFG = LoopConfig(intent={"id": "I-seeded", "behaviour": "Durations parse and render"}, accept=AcceptConfig(flake_reruns=30))


def _check(rep: dict, root) -> None:
    cases = rep["seeded"]
    assert rep["seeded_ok"], json.dumps(cases, indent=1, default=str)
    assert {k: v["got"] for k, v in cases.items()} == {"verifier_gap(lucky pass)": "accepted", "spec_defect(unstated name)": "accepted",
                                                       "world_bug(broken tool)": "accepted", "fake proposal(check fails the oracle)": "rejected"}
    assert "fails the oracle" in cases["fake proposal(check fails the oracle)"]["reasons"][0]
    rows = {r["id"]: r for r in map(json.loads, (root / "loop" / "proposals.jsonl").read_text().splitlines())}
    gap = next(r for r in rows.values() if r["kind"] == "verifier_gap" and r["source"] == "pass_audit")
    ev = gap["evidence"]
    assert ev["current_on_demo"] is True and ev["new_check_on_demo"] is False and ev["oracle"]["new_check"] is True
    assert ev["alternatives"] == {"alt_own_scanner": True, "alt_p2_regex": True} and set(ev["mutants"].values()) == {False}
    assert set(ev["floors"].values()) == {False} and ev["flake"]["n"] == 60 and ev["flake"]["upper"] <= 0.05
    spec = next(r for r in rows.values() if r["kind"] == "spec_defect")
    assert spec["evidence"]["grounded"] == {"format_duration": ["tests/test_format.py::test_format", "tests/test_format.py::test_round_trip"]}
    assert spec["evidence"]["counterfactual_rerun_needed"] and "`format_duration`" in spec["change"]["prompt"]
    world = next(r for r in rows.values() if r["kind"] == "world_bug")
    assert world["evidence"]["before_fix_on_oracle"]["failed"] and not world["evidence"]["after_fix_on_oracle"]["failed"]
    assert world["evidence"]["s3"]["ok"] and world["evidence"]["s3"]["checks"]["existing_tests_clean"] == "5/5"
    store = VersionStore(root / "versions.jsonl")
    assert store.verify() == []
    for k in ("verifier_gap(lucky pass)", "spec_defect(unstated name)", "world_bug(broken tool)"):
        v = store.get(cases[k]["version"])
        assert v.parent is not None and v.proposal_id in rows and v.evidence_hash  # lineage: parent, proposal, evidence
    w, sp = cases["world_bug(broken tool)"]["version"], cases["spec_defect(unstated name)"]["version"]
    assert w in [v.id for v in store.lineage(sp)] or sp in [v.id for v in store.lineage(w)]  # the prompt edit and the World fix: one task lineage
    fws = {json.loads(x)["id"].split("/")[1]: json.loads(x)["first_wrong_step"]["kind"] for x in (root / "loop" / "bundles.jsonl").read_text().splitlines()}
    assert fws == {"lucky": "none", "good_a": "none", "good_b": "none", "unstated": "plateau", "broken_tool": "plateau", "lost": "lost"}


@pytest.mark.slow
async def test_seeded_exit_criterion_local(tmp_path):
    rep = await run_seeded(tmp_path, runner=PytestRunner(), config=CFG)
    _check(rep, tmp_path)
    b = load_bundle(tmp_path / "episodes" / "runs" / "seeded" / "samples" / "broken_tool" / "a1", modules=["durations"])
    assert "No module named 'durations'" in b.steps[1].result  # the tool's failure was executed, not scripted


def _have_image() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(["docker", "image", "inspect", "crucible-lh-base:py312"], capture_output=True, check=False).returncode == 0


def _ours() -> list[str]:
    return subprocess.run(["docker", "ps", "-a", "--filter", "name=crucible-p3-", "--format", "{{.Names}}"], capture_output=True, text=True, check=False).stdout.split()


@pytest.mark.docker
@pytest.mark.slow
async def test_seeded_exit_criterion_in_containers(tmp_path):
    if not _have_image():
        pytest.skip("needs Docker and crucible-lh-base:py312")
    rep = await run_seeded(tmp_path, runner=PytestRunner(image="crucible-lh-base:py312"), config=CFG)
    _check(rep, tmp_path)
    left = _ours()
    assert [n for n in left if n.startswith(("crucible-p3-grade-", "crucible-p3-world-"))] == []  # every grading and World container removed
