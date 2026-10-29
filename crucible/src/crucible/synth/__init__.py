"""crucible.synth: counterexample-guided synthesis of RL environments (Part 2 paper core; part2-paper-core.md).

M1: ``canon`` (typed state and the exact diff), ``iface`` (the ``SynthWorld`` protocol, ``PyWorld``, the Crucible World
adapter), ``program`` (the straight-line oracle DSL with taint), ``contract`` (contracts as data and the trusted grader),
``infer`` (the skeleton contract), ``task``.
M2: ``falsify`` (MUT, NULL, RED, TAINT, TWIN, PERT, REP; ALT/ATK/ONL protocols), ``certificate``, ``archive``, ``cegis``.

Layering: beside ``crucible.forge``; imports runtime, verify, world and core, never factory. LLM roles are injected.
"""
