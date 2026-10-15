"""The run lock (M0 audit 2026-10-14 R-1): core.fingerprint and runtime.lock."""

from __future__ import annotations

import dataclasses
import json

import pytest
from crucible_pack_toy.runtime import NOOP, ORACLE, HelpdeskTask

from crucible.core.fingerprint import SCHEME, describe, diff, excluded, fingerprint, project
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.lock import OPERATIONAL, RunLockMismatch, build
from crucible.runtime.run import Runner, RunSpec


# ------------------------------------------------------------------------------------------------ core.fingerprint
def test_globs_match_whole_segments_and_prefixes():
    assert excluded(("policies", "0", "settings", "api_key"), ["**.api_key"])
    assert excluded(("retry_backoff_s",), ["retry_backoff_s"])
    assert excluded(("cluster", "nodes", "0"), ["cluster"])  # a prefix excludes what is under it
    assert excluded(("policies", "3", "settings", "temperature"), ["policies.*.settings.temperature"])
    assert not excluded(("policies", "3", "settings", "temperature_x"), ["policies.*.settings.temperature"])
    assert not excluded(("max_tokens",), ["**.token", "**.secret"])  # no substring matching


def test_projection_counts_every_leaf_except_operational_and_secret_ones():
    doc = {"k": 2, "concurrency": 8, "llm": {"model": "m", "api_key": "sk-x", "headers": {"Authorization": "env:T"}}, "tags": [], "s": ["a", "b"]}
    assert project(doc, ["concurrency"]) == {"k": 2, "llm.model": "m", "s.0": "a", "s.1": "b", "tags": []}
    a = project({"t": 0.0})
    assert fingerprint(a) == fingerprint(dict(a)) != fingerprint(project({"t": 1.0}))
    assert fingerprint(a).startswith("sha256:") and SCHEME == "crucible.compat_identity/v1"


def test_diff_names_added_removed_and_changed_settings():
    d = diff({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4})
    assert d == {"b": (2, 3), "c": ("<absent>", 4)}
    assert "b: 2 -> 3" in describe(d) and "... and 1 more" in describe(d, limit=1)


# ------------------------------------------------------------------------------------------------ runtime.lock
def _spec(**kw):
    base = {"run_id": "r", "tasks": [HelpdeskTask()], "policies": [ORACLE], "harnesses": [NativeHarness("native:scripted")], "k": 1}
    return RunSpec(**{**base, **kw})


def test_every_runspec_field_is_locked_or_declared_operational():
    settings = build(_spec()).settings
    for f in dataclasses.fields(RunSpec):
        assert f.name in OPERATIONAL or any(k == f.name or k.startswith(f.name + ".") for k in settings), f.name


async def test_a_resume_with_other_settings_is_refused_naming_them(tmp_path):
    await Runner(tmp_path).run(_spec(sampling={"temperature": 0.0}))
    lock = json.loads((tmp_path / "r" / "lock.json").read_text())
    assert lock["schema_version"] == "crucible.run_lock/v1" and lock["settings"]["sampling.temperature"] == 0.0
    h = NativeHarness("native:scripted", limits=dataclasses.replace(NativeHarness().limits, max_steps=300))
    with pytest.raises(RunLockMismatch) as e:
        await Runner(tmp_path).run(_spec(sampling={"temperature": 1.0}, harnesses=[h]))
    assert "sampling.temperature: 0.0 -> 1.0" in str(e.value) and "harnesses.0.settings.limits.max_steps: 30 -> 300" in str(e.value)


async def test_operational_changes_resume_freely(tmp_path):
    await Runner(tmp_path).run(_spec())
    s = await Runner(tmp_path).run(_spec(concurrency=1, budget_usd=5.0, retry_backoff_s=0.0))
    assert s.terminal == 1


async def test_an_allowed_change_is_recorded_in_the_lock(tmp_path):
    await Runner(tmp_path).run(_spec(sampling={"temperature": 0.0}))
    await Runner(tmp_path).run(_spec(sampling={"temperature": 0.5}, allow_change=["sampling.*"]))
    lock = json.loads((tmp_path / "r" / "lock.json").read_text())
    assert lock["settings"]["sampling.temperature"] == 0.5 and lock["changes"][0]["changed"] == ["sampling.temperature"]
    events = [json.loads(x) for x in (tmp_path / "r" / "journal.jsonl").read_text().splitlines()]
    assert any(e["event"] == "run_lock" and e["status"] == "changed" for e in events)


async def test_a_run_from_before_the_lock_is_adopted_once(tmp_path):
    await Runner(tmp_path).run(_spec())
    (tmp_path / "r" / "lock.json").unlink()  # as a run directory written before this change
    await Runner(tmp_path).run(_spec())
    events = [json.loads(x) for x in (tmp_path / "r" / "journal.jsonl").read_text().splitlines()]
    assert [e["status"] for e in events if e["event"] == "run_lock"] == ["created", "adopted"]
    with pytest.raises(RunLockMismatch):
        await Runner(tmp_path).run(_spec(sampling={"temperature": 1.0}))


def test_policies_without_settings_are_listed_as_opaque_and_declared_ones_count():
    lock = build(_spec(policies=[ORACLE, NOOP]))
    assert lock.opaque == [f"policies.0 ({ORACLE.id})", f"policies.1 ({NOOP.id})"]
    llm = PolicyRef(id="single_turn:m", make=lambda a: {}, settings={"temperature": 0.7, "llm": {"model": "m", "base_url": "http://h"}})
    lock = build(_spec(policies=[llm]))
    assert lock.settings["policies.0.settings.temperature"] == 0.7 and not [o for o in lock.opaque if o.startswith("policies")]


# ------------------------------------------------------------------------------------------------ launches (step 2)
async def test_each_invocation_is_recorded_as_a_launch(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["crucible", "bench", "run", "--key", "sk-or-v1-" + "a1" * 16])
    await Runner(tmp_path).run(_spec())
    await Runner(tmp_path).run(_spec(concurrency=1, budget_usd=2.0))
    first, second = (json.loads((tmp_path / "r" / "launches" / f"{n}.json").read_text()) for n in (1, 2))
    assert (first["n"], first["lock_status"], second["n"], second["lock_status"]) == (1, "created", 2, "same")
    assert first["operational"]["concurrency"] == 4 and second["operational"] == {**second["operational"], "concurrency": 1, "budget_usd": 2.0}
    assert first["lock"] == second["lock"] and len(first["build"]["source_sha256"]) == 64
    assert "<secret>" in first["argv"] and not any("sk-or-v1" in a for a in first["argv"])
    events = [json.loads(x) for x in (tmp_path / "r" / "journal.jsonl").read_text().splitlines()]
    assert [e["n"] for e in events if e["event"] == "launch"] == [1, 2]
