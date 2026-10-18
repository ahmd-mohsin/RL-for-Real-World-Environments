"""P2 exit criterion on the toy window (swarm-pipeline/00 §15 P2): three seeded verifier gaps are caught.

(i) a hidden test that checks the implementation's text (a variable-name check, HUD's attention-fix grader): the
fairness lint refuses it, and the executed admission refuses it too (a valid alternative fails it);
(ii) a check missing its requirement's behaviour (``callable(gamma)`` instead of ``gamma() == beta() + 1``): a behaviour
mutant passes it, the shadow test fails it, and the differential assigns a confirmed ``verifier_gap`` to verifier 1;
(iii) a check that fails a valid alternative (a monkeypatch that couples the test to one implementation): the lint
does not see it, the ten-competent-people test and ``admit_check`` refuse it.

The helpers here (``toy``, ``shadow_suite``, ``alternatives_of``, ``seeded_task``) are shared by the other P2 tests.
Tests run on the host interpreter (``LocalSandbox``)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from test_m17_w_task import REQS, make_task

from crucible.factory.fairness import admit_check, lint_checks, recheck_false_fail, ten_competent_people
from crucible.factory.lanes.windows import WindowTask, window_checkset
from crucible.factory.verifier2 import EndStateRec, ShadowSuite, WindowRunner, qualify, run_differential, shadow_path, truth_for

SHADOW = {
    "R1": "def test_r1_alpha():\n    from pkg.core import alpha\n    assert alpha() == 1\n",
    "R2": "def test_r2_beta():\n    from pkg.beta import beta\n    assert beta() == 2\n",
    "R3": ("def test_r3_gamma_is_beta_plus_one():\n    from pkg.beta import beta\n    from pkg.gamma import gamma\n    assert gamma() == beta() + 1\n\n\n"
           "def test_r3_gamma_value_guess():\n    from pkg.gamma import gamma\n    assert gamma() == 4\n"),  # unsound: fails the oracle
}
ALT_FUNCTIONAL = {"pkg/core.py": "def base():\n    return 0\n\n\ndef alpha():\n    return int('1')\n", "pkg/beta.py": "def beta():\n    return 1 + 1\n",
                  "pkg/gamma.py": "from pkg.beta import beta\n\n\ndef gamma():\n    return sum([beta(), 1])\n"}
ALT_MODULE = {"pkg/core.py": "def base():\n    return 0\n\n\ndef alpha():\n    one = 1\n    return one\n", "pkg/beta.py": "def beta():\n    two = 2\n    return two\n",
              "pkg/gamma.py": "import pkg.beta as _b\n\n\ndef gamma():\n    return _b.beta() + 1\n"}
WRONG_GAMMA = {"pkg/core.py": "def base():\n    return 0\n\n\ndef alpha():\n    return 1\n", "pkg/beta.py": "def beta():\n    return 2\n",
               "pkg/gamma.py": "def gamma():\n    return 0\n"}


def toy(tmp: Path):
    task, rec, repo = make_task(tmp)
    return task, rec, repo


def shadow_suite() -> ShadowSuite:
    return ShadowSuite({shadow_path(r): t for r, t in SHADOW.items()}, {r: shadow_path(r) for r in SHADOW}, model="scripted/deepseek", family="deepseek")


def with_oracle(task, files: dict) -> str:
    return task.patch_of({**task.window["oracle_files"], **files})


def alternatives_of(task) -> dict[str, str]:
    return {"alt_functional": with_oracle(task, ALT_FUNCTIONAL), "alt_module": with_oracle(task, ALT_MODULE)}


def seeded_task(task, rec, *, test_name: str, body: str, milestone: str = "m03", replace: bool = False) -> WindowTask:
    """``task`` with one more hidden test in ``tests/test_gamma.py`` added to ``milestone`` (or replacing its tests)."""
    tests_b = dict(rec["tests_b"])
    tests_b["tests/test_gamma.py"] = tests_b["tests/test_gamma.py"] + "\n\n" + body
    tid = f"tests/test_gamma.py::{test_name}"
    ms = [{**m, "tests": [tid] if replace else [*m["tests"], tid]} if m["id"] == milestone else m for m in rec["milestones"]]
    rec2 = {**rec, "tests_b": tests_b, "milestones": ms}
    return WindowTask(task.lane, task.mined, rec2, task_id=f"seed-{test_name}", prompt=task.instruction, requirements=REQS, spans=task.spans,
                      checkset=window_checkset(rec2, REQS))


async def end_states(task, alts: dict[str, str], extra: list[EndStateRec] = ()) -> list[EndStateRec]:
    rids = [r["id"] for r in REQS]
    ms_of = {r["id"]: r["milestone"] for r in REQS}
    out = [EndStateRec("oracle", "oracle", task.oracle_patch(), truth_for("oracle", rids, milestone_of=ms_of)),
           EndStateRec("start", "start", "", truth_for("start", rids, milestone_of=ms_of))]
    out += [EndStateRec(n, "alternative", p, truth_for("alternative", rids, milestone_of=ms_of)) for n, p in alts.items()]
    recs = {m["id"]: m for m in task.window.get("mutants") or []}
    out += [EndStateRec(str(m.id), "mutant", m.solution, truth_for("mutant", rids, milestone_of=ms_of, mutant=recs.get(str(m.id)))) for m in task.mutants()]
    return [*out, *extra]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return toy(tmp_path_factory.mktemp("p2seed"))


def test_seed_i_a_variable_name_check_is_refused_by_the_lint_and_by_execution(built) -> None:
    task, rec, _ = built
    body = "def test_gamma_impl():\n    import inspect\n    from pkg.gamma import gamma\n    assert 'beta() + 1' in inspect.getsource(gamma)\n"
    seeded = seeded_task(task, rec, test_name="test_gamma_impl", body=body)
    rep = lint_checks(seeded.checkset["checks"], seeded.window["tests_b"], seeded.prompt)
    assert rep.refused_checks() == ["c_m03"] and [f.kind for f in rep.findings] == ["source_text"]
    assert rep.findings[0].test == "tests/test_gamma.py::test_gamma_impl" and "getsource" in rep.findings[0].evidence
    assert lint_checks(task.checkset["checks"], task.window["tests_b"], task.prompt).findings == []  # the clean task has none
    runner = WindowRunner(seeded)

    async def run(sol, checks):
        v = await runner.hidden(sol, checks=checks)
        return None if v is None else v["statuses"]

    adm = asyncio.run(admit_check("c_m03", seeded.check_tests["c_m03"], run, oracle=seeded.oracle_patch(), alternatives=alternatives_of(seeded)))
    assert not adm["admitted"] and adm["rows"]["oracle"] is True and adm["rows"]["alt_functional"] is False


def test_seed_ii_a_check_missing_its_behaviour_is_caught_by_the_shadow_differential(built) -> None:
    task, rec, _ = built
    weak = seeded_task(task, rec, test_name="test_gamma_exists", body="def test_gamma_exists():\n    from pkg import gamma as g\n    assert callable(g.gamma)\n",
                       replace=True)
    runner = WindowRunner(weak)
    suite = shadow_suite()
    alts = alternatives_of(weak)
    wrong = EndStateRec("mutant_wrong_gamma", "mutant", with_oracle(weak, WRONG_GAMMA),
                        truth_for("mutant", ["R1", "R2", "R3"], milestone_of={"R1": "m01", "R2": "m02", "R3": "m03"},
                                  mutant={"milestone": "m03", "expected": {"m01": True, "m02": True, "m03": False}}))

    async def go():
        sh = {n: await runner.shadow(suite, p) for n, p in alts.items()}
        muts = {str(m.id): (str(m.milestone), await runner.shadow(suite, m.solution)) for m in weak.mutants()}
        qual = qualify(suite, oracle=await runner.shadow(suite, weak.oracle_patch()), alternatives=sh, start=await runner.shadow(suite, ""), mutants=muts,
                       req_milestone={"R1": "m01", "R2": "m02", "R3": "m03"})
        v1 = await runner.hidden(wrong.solution)
        diff = await run_differential(runner, suite, qual, await end_states(weak, alts, [wrong]), requirements=REQS)
        return qual, v1, diff

    qual, v1, diff = asyncio.run(go())
    assert v1["milestones"]["m03"] is True and v1["reward"] == 1.0  # the weak check passes a gamma that returns 0
    assert qual.tests[f"{shadow_path('R3')}::test_r3_gamma_value_guess"] == "fails_oracle"  # the unsound shadow test never counts
    assert "R3" in qual.gating() and qual.requirements["R3"]["sound"] == [f"{shadow_path('R3')}::test_r3_gamma_is_beta_plus_one"]
    caught = [d for d in diff.defects if d.end_state == "mutant_wrong_gamma"]
    assert len(caught) == 1 and caught[0].owner == "verifier1" and caught[0].defect == "verifier_gap" and caught[0].confirmed
    assert caught[0].requirement == "R3" and caught[0].v1 is True and caught[0].v2 is False
    assert not [d for d in diff.defects if d.end_state != "mutant_wrong_gamma"]  # nothing else disagrees


def test_seed_iii_a_check_that_fails_a_valid_alternative_is_refused(built) -> None:
    task, rec, _ = built
    body = ("def test_gamma_uses_beta(monkeypatch):\n    import pkg.gamma as g\n    monkeypatch.setattr(g, 'beta', lambda: 10)\n"
            "    assert g.gamma() == 11\n")
    seeded = seeded_task(task, rec, test_name="test_gamma_uses_beta", body=body)
    assert lint_checks(seeded.checkset["checks"], seeded.window["tests_b"], seeded.prompt).findings == []  # behaviour-shaped: the lint passes it
    runner = WindowRunner(seeded)
    alts = alternatives_of(seeded)

    async def go():
        pos = {"oracle": (await runner.hidden(seeded.oracle_patch()))["statuses"]}
        pos |= {n: (await runner.hidden(p))["statuses"] for n, p in alts.items()}
        return pos

    pos = asyncio.run(go())
    ff = ten_competent_people(seeded.check_tests, pos, gating=seeded.gating)
    assert [f.check for f in ff] == ["c_m03"] and "alt_module" in ff[0].evidence and "alt_functional" not in ff[0].evidence

    async def run(sol, checks):
        v = await runner.hidden(sol, checks=checks)
        return None if v is None else v["statuses"]

    adm = asyncio.run(admit_check("c_m03", seeded.check_tests["c_m03"], run, oracle=seeded.oracle_patch(), alternatives=alts))
    assert not adm["admitted"] and adm["rows"] == {"oracle": True, "alt_functional": True, "alt_module": False}
    # a false-fail candidate from an audit is re-run on the alternatives before the check is kept
    rc = asyncio.run(recheck_false_fail({"c_m03": seeded.check_tests["c_m03"], "c_m02": seeded.check_tests["c_m02"]}, run, alts))
    assert rc["refused"] == ["c_m03"] and rc["kept"] == ["c_m02"]
    # the original verifier admits its own check: oracle and both alternatives pass, the revert mutant of m03 fails it
    base_runner = WindowRunner(task)

    async def run0(sol, checks):
        v = await base_runner.hidden(sol, checks=checks)
        return None if v is None else v["statuses"]

    muts = {str(m.id): m.solution for m in task.mutants()}
    ok = asyncio.run(admit_check("c_m03", task.check_tests["c_m03"], run0, oracle=task.oracle_patch(), alternatives=alternatives_of(task), mutants=muts,
                                 must_fail=[n for n, m in zip(muts, task.mutants(), strict=True) if m.milestone == "m03"]))
    assert ok["admitted"], json.dumps(ok)
