"""M5 workflow model (DESIGN §5.3, ARCHITECTURE M5): segment traces into steps, materialize and replay the state at
step boundaries, mine and filter milestones, induce stages, and track where an episode is in the workflow.

``actions``: what a tool call does (investigate / edit / build / test / run / ...); ``steps``: segmentation;
``states``: state materialization; ``replay``: the replay test (S1); ``milestones``: mining, filters, evaluation;
``stages``: precedence graph and stage induction; ``graph``: the family's ``crucible.workflow/v1`` document;
``tracker``: the online Workflow State Tracker; ``fidelity``: replay rate, tracker agreement, rediscovery.
"""
