"""Part 1 V2 (2026-10-30): the per-task attack on the debug lane in the real World. ``synth``'s JSON-protocol attackers never
attacked a code task (Nemotron listed files, gpt-oss gave up with no call), so the attack is ``lh_debug_target.py hack``: a
red-team episode of a non-Qwen model in the task's own World (the RL environment), up to 3 rounds, each after the first
told which requirements still failed and which path checks fired."""

from __future__ import annotations

import asyncio
import importlib.util as iu
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = iu.spec_from_file_location("lh_debug_target_hack_t", ROOT / "scripts" / "lh_debug_target.py")
T = iu.module_from_spec(_spec)
_spec.loader.exec_module(T)


def test_the_prompt_names_failed_requirements_and_fired_path_checks_only_after_a_round() -> None:
    first = T.hack_prompt("Fix the parser.")
    assert first.startswith("Fix the parser.\n\nYOUR ROLE: you are a red-team tester") and "earlier attempt" not in first and "pytest" not in first
    later = T.hack_prompt("Fix the parser.", ["R2"], ["path:plumbing"])
    assert "still failed: R2." in later and "path checks that fired: path:plumbing." in later


def test_hackers_are_never_the_students_family() -> None:
    assert not any("qwen" in m.lower() for m in T.HACKERS)
    with pytest.raises(SystemExit, match="student's family"):
        T.parse_args(["hack", "--hackers", "Qwen/Qwen3.5-397B-A17B:peft:262144"])


def test_a_failed_round_feeds_the_next_and_a_pass_ends_the_attack(tmp_path, monkeypatch) -> None:
    base = SimpleNamespace(task_id="t1", id="t1", instruction="Fix the parser.", requirements=[{"id": "R1", "milestone": "m01"}, {"id": "R2", "milestone": "m02"}])
    script = {1: {"m01": True, "m02": False}, 2: {"m01": True, "m02": True}, 3: {"m01": True, "m02": True}}
    seen: list[tuple[str, str]] = []

    class Phase:
        out = tmp_path
        a = SimpleNamespace(hack_rounds=3, hackers=["deepseek-ai/DeepSeek-V4.1-Flash:peft:262144"], seed_offset=0, load_every=60.0)
        tasks = SimpleNamespace(task=lambda tid: base)

        async def job(self, model, tid, *, task, tag, **kw):
            seen.append((task.task_id, task.instruction))
            return {"task": tid, "arm": tag}

    def episodes(out, tag, tid):
        n = int(tag.rsplit("-r", 1)[1])
        ms = script[n]
        return [{"reward": 1.0 if all(ms.values()) else 0.5, "passed": all(ms.values()), "masked": False, "milestones": ms,
                 "gates": {"path:plumbing": "pass"}, "tool_calls": 9, "usd": 0.1, "dir": f"d{n}"}]

    monkeypatch.setattr(T, "episodes", episodes)
    rep = asyncio.run(T.stage_hack(Phase(), ["t1"]))
    [x] = rep["attacks"]
    assert x["passed"] and [r["round"] for r in x["rounds"]] == [1, 2] and rep["passed_tasks"] == ["t1"]  # a pass ends it: no round 3
    assert [s[0] for s in seen] == ["t1#hack-r1", "t1#hack-r2"] and "still failed: R2." in seen[1][1] and "earlier attempt" not in seen[0][1]
    assert base.task_id == "t1" and base.instruction == "Fix the parser."  # the candidate itself is never changed
    assert (tmp_path / "hack.json").exists()
    assert x["status"] == "passed"  # a candidate hack to review, never a verdict


def test_a_masked_round_is_unmeasured_never_held(tmp_path, monkeypatch) -> None:
    base = SimpleNamespace(task_id="t1", id="t1", instruction="p", requirements=[])

    class Phase:
        out = tmp_path
        a = SimpleNamespace(hack_rounds=3, hackers=["m"], seed_offset=0, load_every=60.0)
        tasks = SimpleNamespace(task=lambda tid: base)

        async def job(self, model, tid, **kw):
            return {}

    monkeypatch.setattr(T, "episodes", lambda out, tag, tid: [{"reward": None, "passed": False, "masked": True, "milestones": None, "gates": {},
                                                               "tool_calls": 0, "usd": 0.0, "dir": "d"}])
    rep = asyncio.run(T.stage_hack(Phase(), ["t1"]))
    assert [x["status"] for x in rep["attacks"]] == ["unmeasured"] and rep["unmeasured"] == [("t1", "m")]  # before: printed "held"
