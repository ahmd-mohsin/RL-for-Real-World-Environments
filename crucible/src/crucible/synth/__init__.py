"""crucible.synth: counterexample-guided synthesis of RL environments (Part 2 paper core; part2-paper-core.md).

M1: ``canon`` (typed state and the exact diff), ``iface`` (the ``SynthWorld`` protocol, ``PyWorld``, the Crucible World
adapter), ``program`` (the straight-line oracle DSL with taint), ``contract`` (contracts as data and the trusted grader),
``infer`` (the skeleton contract), ``task``.
M2: ``falsify`` (MUT, NULL, RED, TAINT, TWIN, PERT, REP; ALT/ATK/ONL protocols), ``certificate``, ``archive``, ``cegis``.
M3: MUT scaled with the World and deduplicated by effect, REP-gated scheduling, certified vs provisional admission;
``meta`` (the planter, recall and residual estimates, the regularised update rule R1–R10, the operator DSL and birth gate,
the lemma bank).
M4: ``roles`` (role prompts, schemas, visibility and family rules as data; the ``RoleClient`` protocol), LLM ALT/ATK
runners and the adjudication panel (``falsify.agents``), ``worldgen`` + ``sandbox`` (F1 World synthesis with blind frozen
tests; model code only in a restricted child process), ``pipeline`` (one end-to-end run over a domain).
M5: ``patch`` (the repository substrate: a file-tree World, the fix commit as the oracle, the lane's grader as a contract
with exec checks run in custody through an injected runner, migration of stored grades), ``falsify.patch`` (MUT, NULL,
RED, TAINT, REP on repositories), ``realdefects`` (the real-defect set, E8), ``falsify.online`` (ONL on eval and training
rollouts).

Layering: beside ``crucible.forge``; imports runtime, verify, world and core, never factory. LLM roles are injected.
"""
