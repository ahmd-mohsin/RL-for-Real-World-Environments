"""Observability: every result, graph and agent trace in Weights & Biases (docs/research/swarm-pipeline/11-observability.md).

The one module that talks to W&B and Weave (an import-linter contract forbids ``wandb`` and ``weave`` everywhere else):

- ``run.ObsRun``: a W&B run (offline by default; ``wandb sync`` later) with a redacted local mirror (``ops.jsonl``); it is
  also a ``train.sinks.Sink``. ``run.WandbSink`` is the M12 trainer's ``wandb:`` sink;
- ``naming``: the projects (``crucible-factory``, ``crucible-train``, ``crucible-bench``), run names, groups and tags;
- ``redact.Scrubber``: the M3 redaction plus the process's credential values, on every payload;
- ``traces``: episodes as Weave-shaped spans (episode -> turns -> tool calls, the trace audit's verdict), written
  locally; ``weave_sync`` uploads them once a key exists, keeping the call ids;
- ``train_views``, ``bench``, ``factory_views``: the training dashboards, the benchmark and comparison reports, the
  factory's funnel, targeting, cards, audit and spend; ``report``: the W&B Report template; ``charts``: bars with CIs;
- ``backfill``: existing local results (run JSONs, Tinker run directories, benchmark directories, funnel logs) into
  offline runs and traces.

Layering: below ``workflow`` and above ``connectors | signals``, so the runtime's callers, ``train``, ``bench``, the
factory and the CLI all reach it; it imports only ``core`` and ``connectors`` (the redaction).
"""

from crucible.observe.run import ObserveConfig, ObsRun, WandbSink, has_key

__all__ = ["ObsRun", "ObserveConfig", "WandbSink", "has_key"]
