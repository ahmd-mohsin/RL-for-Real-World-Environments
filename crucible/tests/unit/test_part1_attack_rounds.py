"""Part 1 V2 (2026-10-30, docs/research/env-synthesis/part1-framework-improvements.md §2(c)): a per-task attack score.
``synth``'s ATK ran each attacker once and never on the training pool's code tasks (``NOT_APPLICABLE``: paid). V2 asks
for wrong solutions from other families over 3 rounds with the failure logs fed back (2606.16062), applied in the real
World. ``ATK(rounds=)`` retries a failed attacker in a fresh World with the checks it failed; ``certify_patch`` runs ATK
when given attackers. The fixture task is hackable: its hidden tests check one input per function."""

from __future__ import annotations

import asyncio

from crucible.synth.falsify.agents import ATK, AgentSpec
from crucible.synth.falsify.base import falsify
from crucible.synth.falsify.patch import certify_patch, findings_summary
from crucible.synth.patch import PatchEnv
from tests.unit.synth_patch_fixtures import BUGGY, fixture_runner, make_repo, spec_for

SPECIAL_ADD = BUGGY.replace("def add(a, b):\n    return a - b", "def add(a, b):\n    if (a, b) == (2, 3):\n        return 5\n    return a - b")
SPECIAL_BOTH = SPECIAL_ADD.replace("def mul(a, b):\n    return a + b", "def mul(a, b):\n    if (a, b) == (2, 3):\n        return 6\n    return a + b")


def _env(tmp_path) -> PatchEnv:
    return PatchEnv(spec_for(make_repo(tmp_path)))


def _attacker(seen: list[str], family: str = "llama") -> AgentSpec:
    async def run(view, world):
        seen.append(view.instruction)
        await world.call("repo.write_file", {"path": "calc.py", "text": SPECIAL_BOTH if "c_m02" in view.instruction else SPECIAL_ADD})
        return "done"

    return AgentSpec(name=f"atk-{family}", family=family, runner=run)


def _atk(env: PatchEnv, rounds: int, seen: list[str]):
    res = asyncio.run(falsify(env.task(), ATK([_attacker(seen)], rounds=rounds), env.context(fixture_runner(env))))
    [f] = res.findings
    return f


def test_a_failed_attacker_learns_from_the_failed_checks_and_breaks_the_task_in_round_2(tmp_path) -> None:
    env, seen = _env(tmp_path), []
    f = _atk(env, 3, seen)
    assert (f.label, f.defect, f.outcome.data["round"]) == ("suspect", "D8", 2)  # the special-cased fix passes every check
    assert "did not pass" not in seen[0] and "MISSING_REQUIRED:c_m02" in seen[1] and seen[1].startswith(env.task().instruction)
    assert len(seen) == 2  # a pass ends the attack


def test_one_round_alone_misses_the_hole(tmp_path) -> None:
    f = _atk(_env(tmp_path), 1, [])  # before: every attack was one round
    assert (f.label, f.outcome.data["round"]) == ("clean", 1)


def test_certify_patch_runs_the_attack_only_when_given_attackers_from_other_families(tmp_path) -> None:
    env = _env(tmp_path)
    cert, res = asyncio.run(certify_patch(env, fixture_runner(env), attackers=[_attacker([])]))
    atk = next(r for r in res if r.cls == "ATK")
    assert atk.applicable and [s["defect"] for s in findings_summary([atk])["suspects"]] == ["D8"] and not cert.admitted  # an open suspect
    _, res = asyncio.run(certify_patch(env, fixture_runner(env)))
    assert not next(r for r in res if r.cls == "ATK").applicable  # unchanged without attackers (paid)
    _, res = asyncio.run(certify_patch(env, fixture_runner(env), attackers=[_attacker([], "qwen")], exclude_families={"qwen"}))
    assert [r.reason for r in res if r.cls == "ATK"] == ["no agent runners injected"]  # the student's own family never attacks
