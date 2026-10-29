"""The falsifier population (part2-paper-core §5). ``cheap()`` is the deterministic set the CEGIS loop runs every round,
in the order of §7 (REP, RED, TAINT, NULL, MUT, TWIN, PERT); ``run_classes`` schedules them (REP first, gating MUT, TWIN
and PERT). ALT and ATK take injected agent runners (M4 wires LLMs)."""

from __future__ import annotations

from crucible.synth.falsify.agents import ALT, ATK, AgentRunner, AgentSpec, AgentView
from crucible.synth.falsify.base import (
    CLASSES,
    DEFECTS,
    DETERMINISTIC,
    GATED_BY_REP,
    BaseFalsifier,
    ClassResult,
    Counterexample,
    Falsifier,
    FalsifyContext,
    Finding,
    Outcome,
    Probe,
    effect_signature,
    equivalent,
    falsify,
    run_classes,
)
from crucible.synth.falsify.mut import MUT
from crucible.synth.falsify.null import NULL
from crucible.synth.falsify.online import OnlineMonitor
from crucible.synth.falsify.pert import PERT
from crucible.synth.falsify.red import RED
from crucible.synth.falsify.rep import REP
from crucible.synth.falsify.taint import TAINT
from crucible.synth.falsify.twin import TWIN


def cheap() -> list[BaseFalsifier]:
    return [REP(), RED(), TAINT(), NULL(), MUT(), TWIN(), PERT()]


__all__ = ["ALT", "ATK", "CLASSES", "DEFECTS", "DETERMINISTIC", "GATED_BY_REP", "MUT", "NULL", "PERT", "RED", "REP", "TAINT", "TWIN", "AgentRunner", "AgentSpec",
           "AgentView", "BaseFalsifier", "ClassResult", "Counterexample", "Falsifier", "FalsifyContext", "Finding", "OnlineMonitor", "Outcome", "Probe",
           "cheap", "effect_signature", "equivalent", "falsify", "run_classes"]
