"""The falsifier population (part2-paper-core §5). ``cheap()`` is the deterministic set the CEGIS loop runs every round,
in the order of §7 (REP, RED, TAINT, NULL, MUT, TWIN, PERT); ``run_classes`` schedules them (REP first, gating MUT, TWIN
and PERT). ALT and ATK take injected agent runners (M4 wires LLMs). ``population(h)`` is ``cheap()`` with MUT showing only
the mutants outside the held-out share ``h`` (Part 1 V5; ``MUTHeldOut`` runs the rest at certification)."""

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
from crucible.synth.falsify.leak import LAYERS, LeakMatcher, LeakPanel, LeakProbe, SynonymMap
from crucible.synth.falsify.mut import HOLDOUT, MUT, MUTHeldOut
from crucible.synth.falsify.null import NULL
from crucible.synth.falsify.online import OnlineMonitor
from crucible.synth.falsify.pert import PERT
from crucible.synth.falsify.red import RED
from crucible.synth.falsify.rep import REP
from crucible.synth.falsify.taint import TAINT
from crucible.synth.falsify.twin import TWIN


def cheap(*, synonyms: SynonymMap | None = None) -> list[BaseFalsifier]:
    """The deterministic population: MUT with the unstated-trace-constraint family, TAINT with every structural leak
    layer (``falsify.leak``) and the given learned synonym map."""
    return [REP(), RED(), TAINT(synonyms=synonyms), NULL(), MUT(), TWIN(), PERT()]


def population(holdout: float = 0.0, *, synonyms: SynonymMap | None = None) -> list[BaseFalsifier]:
    """What a repair loop runs: ``cheap()``, with MUT on its shown mutants only when ``holdout`` > 0."""
    if holdout <= 0:
        return cheap(synonyms=synonyms)
    return [MUT(split="shown", holdout=holdout) if isinstance(f, MUT) else f for f in cheap(synonyms=synonyms)]


def cheap_m3() -> list[BaseFalsifier]:
    """The population the RSI-1/2 numbers were measured with (before DESIGN §7.7 item 20): no read mutants in MUT, the
    exact-token leak matcher in TAINT. Kept so before/after matrices are measured the same way."""
    return [REP(), RED(), TAINT(layers=("exact",)), NULL(), MUT(trace=False), TWIN(), PERT()]


__all__ = [
    "ALT",
    "ATK",
    "CLASSES",
    "DEFECTS",
    "DETERMINISTIC",
    "GATED_BY_REP",
    "HOLDOUT",
    "LAYERS",
    "MUT",
    "NULL",
    "PERT",
    "RED",
    "REP",
    "TAINT",
    "TWIN",
    "AgentRunner",
    "AgentSpec",
    "AgentView",
    "BaseFalsifier",
    "ClassResult",
    "Counterexample",
    "Falsifier",
    "FalsifyContext",
    "Finding",
    "LeakMatcher",
    "LeakPanel",
    "LeakProbe",
    "MUTHeldOut",
    "OnlineMonitor",
    "Outcome",
    "Probe",
    "SynonymMap",
    "cheap",
    "cheap_m3",
    "effect_signature",
    "equivalent",
    "falsify",
    "population",
    "run_classes",
]
