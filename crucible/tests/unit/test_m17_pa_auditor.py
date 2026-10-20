"""Phase A: the RLM auditor's forced final-verdict turn (``traceaudit.RLMAuditor.final_verdict``), a harness switch enabled
only through ``AuditorHarness.refine`` (``HarnessState.evaluate``: paired seeds, measured)."""

from __future__ import annotations

from typing import Any

from crucible.actors.llm import LLMResponse
from crucible.actors.rlm.agent import RLMConfig
from crucible.actors.rlm.kernel import LocalLauncher
from crucible.factory.audit import SeededCase, Trace, audit
from crucible.factory.traceaudit import FINAL_VERDICT_EDIT, FINAL_VERDICT_SWITCH, AuditorHarness, RLMAuditor, switch_on

TRACE = ("[episode] r/s/a1 task=w1~x\n[request]\nAdd format_duration.\n[tool 1] repo.write_file {\"path\": \"d.py\"}\n"
         "[result 1] (ok) wrote d.py\n[grader tail]\nE   ImportError: cannot import name 'format_duration' from 'durations'\n"
         "[facts] none\n[first wrong step] flat step=0")
VERDICT = {"cause": "spec_defect", "solution_correct": False, "evidence": "cannot import name 'format_duration'", "rationale": "the name is unstated"}


class Wanderer:
    """A scripted auditor model that keeps searching and never calls ``rlm.finish``; given only the verdict tool, it answers."""

    model = "script"

    def __init__(self, verdict: dict[str, Any] | None = VERDICT, *, via_text: bool = False):
        self.verdict, self.via_text, self.n, self.final_calls = verdict, via_text, 0, []

    async def complete(self, messages: list[dict[str, Any]], *, tools: Any = None, **kw: Any) -> LLMResponse:
        names = [t["function"]["name"] for t in tools or []]
        if names == ["verdict"]:
            self.final_calls.append(messages)
            if self.verdict is None:
                return LLMResponse(text="I cannot tell.", tool_calls=[], usage={})
            if self.via_text:
                import json

                return LLMResponse(text=json.dumps(self.verdict), tool_calls=[], usage={})
            return LLMResponse(text="", tool_calls=[{"id": "v1", "name": "verdict", "arguments": dict(self.verdict)}], usage={}, cost_usd=0.001)
        self.n += 1
        return LLMResponse(text="", tool_calls=[{"id": f"py{self.n}", "name": "python", "arguments": {"code": f"print({self.n})"}}],
                           usage={"prompt_tokens": 50, "completion_tokens": 5})


def auditor(tmp_path: Any, llm: Any, **kw: Any) -> RLMAuditor:
    cfg = RLMConfig(temperature=0.0, kernel="local", allow_host=True, tree_max_depth=0, max_turns=3, window_tokens=12_000, max_tokens=2000)
    return RLMAuditor(llm, model="deepseek/deepseek-v4-flash", workdir=tmp_path / "aud", launcher=lambda: LocalLauncher(allow_host=True), rlm_config=cfg,
                      max_steps=10, **kw)


async def test_at_the_cap_one_final_turn_with_only_the_verdict_tool_gives_a_validated_label(tmp_path: Any) -> None:
    off = auditor(tmp_path / "off", Wanderer())
    lab = await off.label(Trace("r/s/a1", TRACE, False), prompt="")
    assert lab.status == "rejected" and off.sessions[0]["stop"].startswith("budget:turns") and "final_turn" not in off.sessions[0]
    llm = Wanderer()
    on = auditor(tmp_path / "on", llm, final_verdict_turn=True)
    lab = await on.label(Trace("r/s/a1", TRACE, False), prompt="")
    assert lab.status == "labelled" and lab.cause == "spec_defect"  # audit.validate checked the closed taxonomy and the verbatim quote
    s = on.sessions[0]
    assert s["final_turn"] == {"forced": True, "usd": 0.001, "via": "tool", "verdict": True} and not s["finished"]
    # the final turn sees the session's own messages plus the cap notice, and no python tool
    last = llm.final_calls[0]
    assert "turn limit" in last[-1]["content"] and any(m.get("role") == "tool" for m in last)


async def test_the_final_turn_keeps_the_label_rules(tmp_path: Any) -> None:
    lazy = auditor(tmp_path / "a", Wanderer({**VERDICT, "evidence": "something the attempt never said"}), final_verdict_turn=True)
    assert (await lazy.label(Trace("r/s/a1", TRACE, False), prompt="")).status == "rejected"  # not verbatim
    mute = auditor(tmp_path / "b", Wanderer(None), final_verdict_turn=True)
    assert (await mute.label(Trace("r/s/a1", TRACE, False), prompt="")).status == "rejected" and mute.sessions[0]["final_turn"]["via"] == "none"
    text = auditor(tmp_path / "c", Wanderer(via_text=True), final_verdict_turn=True)
    assert (await text.label(Trace("r/s/a1", TRACE, False), prompt="")).status == "labelled"
    abst = auditor(tmp_path / "d", Wanderer({**VERDICT, "cause": "abstain"}), final_verdict_turn=True)
    assert (await abst.label(Trace("r/s/a1", TRACE, False), prompt="")).cause is None


async def test_the_switch_is_a_harness_entry_enabled_only_through_refine(tmp_path: Any) -> None:
    h = AuditorHarness(tmp_path / "h.json")
    assert not h.switched() and not auditor(tmp_path, Wanderer(), harness=h).final_verdict_turn
    control = SeededCase(Trace("ctl", TRACE, False, policy="control"), "spec_defect", False)

    async def fn(entries: dict[str, Any], seed: int) -> dict[str, float]:
        a = auditor(tmp_path / f"s{seed}-{len(entries)}-{switch_on(entries)}", Wanderer(), harness=h, entries=entries)
        rep = await audit([Trace(f"r/s/a{seed}", TRACE, False, policy="solver:qwen/qwen3.7-flash")], a, prompt="", seeded=[control], salt=f"t{seed}")
        labelled = sum(1 for lab in rep.labels.values() if lab.status == "labelled")
        return {"label_rate": labelled / len(rep.labels), "seeded_accuracy": sum(r["ok"] for r in rep.seeded) / len(rep.seeded)}

    ev = await h.refine([FINAL_VERDICT_EDIT], fn, [1, 2, 3], primary="label_rate", guards=("seeded_accuracy",))
    assert ev["accepted"] and ev["metrics"]["label_rate"]["base"] == 0.0 and ev["metrics"]["label_rate"]["candidate"] == 1.0
    assert h.switched() and FINAL_VERDICT_SWITCH in h.entries
    a = auditor(tmp_path / "after", Wanderer(), harness=h)
    assert a.final_verdict_turn and h.digest(TRACE) == ""  # a switch is behaviour, never prompt text
    assert not auditor(tmp_path / "forced-off", Wanderer(), harness=h, final_verdict_turn=False).final_verdict_turn
