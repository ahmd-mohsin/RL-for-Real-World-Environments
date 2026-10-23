"""The controller <-> swarm reviser bridge (swarm-pipeline/12 §3, 13 §5): one knob vocabulary, the controller as a
``difficulty_hooks.Controller``, the checkpoint level it picks, and the debug lane's reviser and gates on fakes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible.factory.difficulty import KnobAction as ControlAction
from crucible.factory.difficulty import control
from crucible.factory.difficulty_bridge import ControlController, choose_checkpoint, k1_target, rate_of, stats_from_targeting
from crucible.factory.difficulty_hooks import EASE_KNOBS, HARDEN_KNOBS, Controller, HeldOutRefused, KnobAction, from_control, revise_and_gate
from crucible.factory.revise_debug import SUPPORTED, DebugGates, DebugReviser, hidden_hash, tree_hash, variant_id

MS = [{"id": "m01", "requires": []}, {"id": "m02", "requires": []}, {"id": "m03", "requires": []}, {"id": "m04", "requires": ["m03"]}]
COND = {"milestones": {"m01": {"p_given_prereqs": 0.625, "requires": []}, "m02": {"p_given_prereqs": 0.0, "requires": []},
                       "m03": {"p_given_prereqs": 0.75, "requires": []}, "m04": {"p_given_prereqs": 0.667, "requires": ["m03"]}}}


def _row(passes: int = 0, n: int = 8, **kw: Any) -> dict[str, Any]:
    return {"task": "t", "split": "train", "n": n, "passes": passes, "cause": {"labels": kw.pop("causes", {"skill_gap": n - passes})},
            "modes": kw.pop("modes", {"stalled/interacting_faults": 4}), "frontier": kw.pop("frontier", {"x": [True, False]}),
            "per_milestone": {"m01": 0.625, "m02": 0.0, "m03": 0.75, "m04": 0.5}, **kw}


def test_one_knob_vocabulary() -> None:
    assert set(EASE_KNOBS) == {"K1", "K2", "K3", "K4", "K5", "K6", "C1"}
    assert set(HARDEN_KNOBS) == {"H1", "H2", "H3"}
    assert "distractor" not in json.dumps(HARDEN_KNOBS)  # V5: nothing the tests do not check is asked for
    with pytest.raises(ValueError):
        KnobAction("harden", "H4", "gone")


def test_from_control_maps_the_first_supported_knob() -> None:
    a = ControlAction("t", "ease", ("C1", "K2", "K4"), "p=0", 0.0, 0.0, 0.3, 8, 0.0)
    assert from_control(a).knob == "C1"
    assert from_control(a, supported={"K4", "K2"}).knob == "K2"
    assert from_control(a, supported={"K5"}) is None
    for action in ("keep", "sample_more", "fix_task"):
        assert from_control(ControlAction("t", action, (), "r", 0.3, 0.1, 0.6, 8, 0.6)) is None
    b = from_control(ControlAction("t", "budget_bound", ("C1", "K1"), "budget", 0.0, 0.0, 0.3, 8, 0.0), supported={"K1"})
    assert b.direction == "ease" and b.knob == "K1"


def test_budget_bound_shortens_the_horizon_never_the_budget() -> None:
    a = control("t", rate_of(0, 8), causes={"budget": 6, "skill_gap": 2})
    assert a.action == "budget_bound" and a.knobs == ("C1", "K1")
    assert control("t", rate_of(0, 8), causes={"budget": 6, "skill_gap": 2}, applied=("C1",)).knobs == ("K1",)


def test_a_root_milestone_rate_is_no_checkpoint_signal() -> None:
    cond = {"milestones": {"m01": {"p_given_prereqs": 0.875, "requires": []}, "m03": {"p_given_prereqs": 0.0, "requires": []},
                           "m04": {"p_given_prereqs": None, "requires": ["m03"]}}}
    st = stats_from_targeting(_row(), cond)
    assert st["conditional_rates"] == {"m04": None}
    assert "checkpoint start first" not in control("t", st["rate"], causes=st["causes"], modes=st["modes"], conditional_rates=st["conditional_rates"]).reason
    # and a conditional rate above the band is not "in band" either
    assert "checkpoint start first" not in control("t", rate_of(0, 8), causes={"skill_gap": 8}, conditional_rates={"m4": 0.95}).reason


def test_choose_checkpoint_closest_to_half_and_never_on_unmeasured() -> None:
    cp = choose_checkpoint(COND, MS)
    assert cp == {"level": 2, "pre": ["m01", "m02"], "remaining": ["m03", "m04"], "expected": pytest.approx(0.5, abs=0.01)}
    unmeasured = {"milestones": {**COND["milestones"], "m04": {"p_given_prereqs": None, "requires": ["m03"]}}}
    assert choose_checkpoint(unmeasured, MS) is None


def test_k1_target_is_the_hardest_free_milestone() -> None:
    assert k1_target(MS, {"m01": 0.6, "m02": 0.0, "m03": 0.75}) == "m02"
    assert k1_target(MS, {"m01": 1.0, "m02": 0.875, "m03": 0.1}) is None  # every free fault is always fixed: nothing to drop
    assert k1_target(MS, {}) is None  # unmeasured is never a target
    assert k1_target(MS[:2], {}) is None  # never down to one milestone
    linked = [{"id": "a", "requires": []}, {"id": "b", "requires": ["a"]}, {"id": "c", "requires": ["b"]}]
    assert k1_target(linked, {}) is None  # every fault interacts: K1 would remove an interaction


def test_controller_adapter_is_a_hook_controller(tmp_path: Path) -> None:
    from crucible.factory.difficulty import Journal

    cc = ControlController(supported=SUPPORTED, journal=Journal(tmp_path / "a.jsonl"))
    assert isinstance(cc, Controller)
    k = cc.propose("t", stats_from_targeting(_row(), COND, milestones=MS))
    assert k.knob == "C1" and k.params["checkpoint"]["pre"] == ["m01", "m02"]
    no_cp = cc.propose("t", stats_from_targeting(_row(), {"milestones": {}}, milestones=MS))
    assert no_cp.knob != "C1" and "checkpoint" not in no_cp.params
    k6 = ControlController(supported={"K6"}).propose("t", stats_from_targeting(_row(), COND, milestones=MS))
    assert k6.knob == "K6" and k6.milestone == "m02"
    rows = Journal(tmp_path / "a.jsonl").rows()
    assert [r["chosen"] for r in rows] == ["C1", no_cp.knob]
    with pytest.raises(HeldOutRefused):
        cc.propose("t", {**stats_from_targeting(_row(), COND, milestones=MS), "split": "heldout"})


def test_repathed_frontier_overrides_the_stored_verdict() -> None:
    st = stats_from_targeting(_row(frontier={"ds": [False, False]}), COND, frontier_override={"ds": True})
    assert st["frontier_passes"] is True


def test_variant_ids_are_container_safe_and_stable() -> None:
    v = variant_id("db-go-x-1", "C1", ["m02", "m01"])
    assert v == variant_id("db-go-x-1", "C1", ["m01", "m02"]) and v.startswith("db-go-x-1-c1-") and "~" not in v


# ---- the reviser and the gates on fakes --------------------------------------------------------------------------------
class Store:
    def __init__(self, root: Path, split: str = "train"):
        self.root, self.split = root, split

    def dir(self, wid: str) -> Path:
        return self.root / "tasks" / wid

    def window(self, wid: str) -> dict[str, Any]:
        return json.loads((self.dir(wid) / "window.json").read_text())

    def save_window(self, rec: dict[str, Any]) -> None:
        self.dir(rec["id"]).mkdir(parents=True, exist_ok=True)
        (self.dir(rec["id"]) / "window.json").write_text(json.dumps(rec))

    def save(self, mined: Any) -> None:
        (self.dir(mined["id"]) / "record.json").write_text(json.dumps(mined))


def _parent(tmp_path: Path, split: str = "train") -> tuple[Store, Path]:
    st = Store(tmp_path / "store", split)
    rec = {"id": "p", "repo": "r", "lang": "go", "stage": "mutants_valid", "tests_b": {"a_test.go": "x"}, "p2p": [],
           "milestones": [{"id": "m01", "tests": ["a_test.go::TestA"], "requires": []}, {"id": "m02", "tests": ["a_test.go::TestB"], "requires": []}]}
    st.save_window(rec)
    (st.dir("p") / "base").mkdir()
    (st.dir("p") / "base" / "x.go").write_text("package x\n")
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "p.json").write_text(json.dumps({"opening": "o", "requirements": {"m01": "a", "m02": "b"}, "closing": ""}))
    return st, prompts


async def test_a_report_knob_variant_shares_the_world_and_inherits_gates(tmp_path: Path) -> None:
    st, prompts = _parent(tmp_path)
    briefs: list[dict[str, Any]] = []

    async def write(vid: str, rec: dict, brief: dict) -> dict:
        briefs.append(brief)
        (prompts / f"{vid}.json").write_text(json.dumps({"opening": "two problems", "requirements": {"m01": "a", "m02": "b"}}))
        return {"status": "done", "usd": 0.12}

    rv = DebugReviser(st, prompts, task_of=lambda pid: None, mined_of=lambda rec, base: {"id": rec["id"]}, write_report=write, scratch=tmp_path / "s")
    gd = tmp_path / "gates"
    gd.mkdir()
    (gd / "p.json").write_text(json.dumps({"ok": True, "passed": ["S5"]}))

    async def never(*a: Any) -> Any:
        raise AssertionError("a report knob runs no World gate")

    gates = DebugGates(st, gd, prompts, author=lambda vid: {"findings": []}, world_gates=never, start_status=never, repro=never)
    act = KnobAction("ease", "K2", "p=0")
    res = await revise_and_gate("p", act, reviser=rv, gates=gates)
    vid = res["revision"].task_id
    assert res["ok"] and vid.startswith("p-k2-") and "2" in briefs[0]["instruction"] and briefs[0]["current_report"]["requirements"] == {"m01": "a", "m02": "b"}
    v = st.window(vid)
    assert v["revision"]["parent"] == "p" and v["revision"]["applied"] == ["K2"] and tree_hash(st.dir(vid) / "base") == tree_hash(st.dir("p") / "base")
    assert json.loads((gd / f"{vid}.json").read_text())["inherited_from"] == "p" and hidden_hash(v) == hidden_hash(st.window("p"))
    assert rv.spent == pytest.approx(0.12)
    bad = DebugGates(st, gd, prompts, author=lambda vid: {"findings": ["m01: names a function"]}, world_gates=never, start_status=never, repro=never)
    assert not (await revise_and_gate("p", act, reviser=rv, gates=bad))["ok"]


async def test_held_out_and_unsupported_are_refused(tmp_path: Path) -> None:
    st, prompts = _parent(tmp_path, split="heldout")

    async def write(*a: Any) -> dict:
        raise AssertionError("nothing is written for a held-out window")

    rv = DebugReviser(st, prompts, task_of=lambda pid: None, mined_of=lambda rec, base: {"id": rec["id"]}, write_report=write, scratch=tmp_path / "s")
    with pytest.raises(HeldOutRefused):
        await rv.revise("p", KnobAction("ease", "K2", "r"))
    with pytest.raises(ValueError):
        await rv.revise("p", KnobAction("ease", "K5", "r"))


async def test_a_world_knob_must_pass_the_start_check(tmp_path: Path) -> None:
    st, prompts = _parent(tmp_path)
    vid = "p-c1-x"
    st.save_window({**st.window("p"), "id": vid, "milestones": st.window("p")["milestones"][1:],
                    "revision": {"parent": "p", "knob": "C1", "pre_applied": ["m01"], "applied": ["C1"]}})
    (prompts / f"{vid}.json").write_text("{}")
    from crucible.factory.difficulty_hooks import Revision

    rev = Revision(vid, "p", KnobAction("ease", "C1", "r"), task={})
    calls: list[str] = []

    async def world(v: str) -> dict:
        calls.append(v)
        return {"ok": True, "passed": ["S6", "S5", "S8", "S3", "S10"]}

    async def broken_start(v: str, pre: Any) -> dict:
        return {"m01": False, "m02": False}  # the pre-applied fix does not hold

    async def good_start(v: str, pre: Any) -> dict:
        return {"m01": True, "m02": False}

    async def no_repro(v: str) -> dict:
        return {}

    g = DebugGates(st, tmp_path, prompts, author=lambda v: {"findings": []}, world_gates=world, start_status=broken_start, repro=no_repro)
    r = await g.run(rev)
    assert not r.ok and any("pre-applied but not credited" in f for f in r.findings) and not calls
    g.start_status = good_start
    assert (await g.run(rev)).ok and calls == [vid]


def test_a_run_cap_raises_the_shared_meter_never_lowers_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from crucible.train.bridges.tinker import sampler

    monkeypatch.delenv("CRUCIBLE_TINKER_CAP_USD", raising=False)
    monkeypatch.setattr(sampler, "_SHARED", {})
    assert sampler.ensure_cap(8.0) == 8.0 and sampler.shared_meter().cap_usd == 8.0
    monkeypatch.setattr(sampler, "_SHARED", {})
    monkeypatch.setenv("CRUCIBLE_TINKER_CAP_USD", "1.0")
    m = sampler.shared_meter()
    assert m.cap_usd == 1.0 and sampler.ensure_cap(10.0) == 10.0 and m.cap_usd == 10.0
    monkeypatch.setenv("CRUCIBLE_TINKER_CAP_USD", "50")
    assert sampler.ensure_cap(10.0) == 50.0  # the operator's larger cap stands


def test_requirement_findings_group_by_what_they_say() -> None:
    from crucible.factory.swarm_journal import finding_kind

    a = finding_kind("m03: requirement leaks the fix by naming internal functions (assignString, tryX) and describing the change")
    b = finding_kind("m04: requirement leaks the fix by naming internal functions (appendValue) and describing the change")
    c = finding_kind("m01: requirement does not describe an observable symptom")
    assert a == b != c and a != "m#"


async def test_an_infrastructure_stop_is_not_blamed_on_the_reviser(tmp_path: Path) -> None:
    from crucible.factory.difficulty_hooks import GateResult, Revision
    from crucible.factory.swarm_journal import SwarmJournal, tables

    class R:
        async def revise(self, task: Any, action: KnobAction) -> Revision:
            return Revision("v", task, action, task={})

    class G:
        async def run(self, rev: Revision) -> GateResult:
            return GateResult(False, {"infra": True}, ["S7: no report written (infrastructure: TinkerCapReached)"])

    j = SwarmJournal(tmp_path / "j")
    res = await revise_and_gate("t", KnobAction("ease", "K2", "r"), reviser=R(), gates=G(), journal=j)
    assert not res["ok"] and tables(tmp_path / "j")["roles"].get("reviser", {}).get("blamed", 0) == 0


def test_a_frame_knob_never_rewrites_a_requirement(tmp_path: Path) -> None:
    from crucible.factory.revise_debug import restore_requirements

    parent = {"m03": "decoding ignores UnmarshalText ... and for date and time values", "m04": "encoding writes the raw string"}
    f = tmp_path / "v.json"
    f.write_text(json.dumps({"opening": "two problems", "requirements": {"m03": "decoding ignores UnmarshalText behind pointers", "m04": parent["m04"]}}))
    assert restore_requirements(f, parent, KnobAction("ease", "C1", "r")) == ["m03"]
    d = json.loads(f.read_text())
    assert d["requirements"] == parent and d["swarm_requirements"] == {"m03": "decoding ignores UnmarshalText behind pointers"} and d["opening"] == "two problems"
    f.write_text(json.dumps({"requirements": {"m03": "sharper m03", "m04": "changed m04"}}))
    assert restore_requirements(f, parent, KnobAction("ease", "K6", "r", milestone="m03")) == ["m04"]
    assert json.loads(f.read_text())["requirements"] == {"m03": "sharper m03", "m04": parent["m04"]}
    f.write_text(json.dumps({"requirements": {"m03": "rewritten", "m04": parent["m04"]}, "areas": {"m03": "the decoder", "m04": "the encoder."}}))
    assert restore_requirements(f, parent, KnobAction("ease", "K3", "r")) == ["m03"]  # K3 appends an area to the verbatim text
    assert json.loads(f.read_text())["requirements"] == {"m03": parent["m03"] + " This is in the decoder.", "m04": parent["m04"] + " This is in the encoder."}


async def test_a_sharpened_requirement_needs_its_executed_reproduction(tmp_path: Path) -> None:
    from crucible.factory.difficulty_hooks import Revision

    st, prompts = _parent(tmp_path)
    gd = tmp_path / "gates"
    gd.mkdir()
    (gd / "p.json").write_text(json.dumps({"ok": True}))

    async def write(vid: str, rec: dict, brief: dict) -> dict:
        assert brief["format"]["repro"].keys() == {"m02"} and "really prints" in brief["instruction"]
        (prompts / f"{vid}.json").write_text(json.dumps({"opening": "o", "requirements": {"m01": "a", "m02": "b sharper"},
                                                          "repro": {"m02": {"command": "x"}}}))
        return {"status": "done", "usd": 0.1}

    async def never(*a: Any) -> Any:
        raise AssertionError("no World gate for a report knob")

    rv = DebugReviser(st, prompts, task_of=lambda pid: None, mined_of=lambda rec, base: {"id": rec["id"]}, write_report=write, scratch=tmp_path / "s")
    rev = await rv.revise("p", KnobAction("ease", "K6", "stuck", milestone="m02"))
    assert isinstance(rev, Revision)

    async def wrong_claim(v: str) -> dict:
        return {"ok": False, "status": "failed", "per_requirement": {"m02": {"status": "fails_at_oracle"}}}

    async def verified(v: str) -> dict:
        return {"ok": True, "status": "partial", "per_requirement": {"m02": {"status": "verified"}}}

    g = DebugGates(st, gd, prompts, author=lambda v: {"findings": []}, world_gates=never, start_status=never, repro=wrong_claim)
    assert not (await g.run(rev)).ok
    g.repro = verified
    assert (await g.run(rev)).ok
