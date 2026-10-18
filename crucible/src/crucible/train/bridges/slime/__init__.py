"""The slime bridge (swarm-pipeline/00 §11 "slime bridge", §11.3; 03 §4.3, §5; P6).

THUDM slime (Apache-2.0, ``LICENSES/Apache-2.0.txt``; studied at 859938d,
``RL-Environments/06-frameworks-and-env-code/thudm-slime``) owns ``train.py``; Crucible plugs in through its hooks. slime
is **not installed** here (it pulls Megatron and SGLang): every module imports it lazily and falls back to a mirror of
the few interfaces it needs, each mirror citing the slime file and lines it copies.

| module | slime hook (``arguments.py`` line) | what it does |
|---|---|---|
| ``rows`` | ``--input-key prompt --metadata-key metadata`` | one JSONL row per training task; held-out rows are refused (V9) |
| ``generate`` | ``--custom-generate-function-path`` (:527) | one episode: the World as a ``Sandbox``, the agent harness, checkpoint snapshots before each model call, grading in a fresh container, R = 0.5·Φ + 0.5·[gates], turn credit, STATUS claims, give-up; masked failures are removed samples (V8) |
| ``credit`` | ``--custom-reward-post-process-path`` (:1497), ``--custom-convert-samples-to-train-data-path`` (:1505), ``--custom-advantage-function-path`` (:1091) | outcome advantage over unique rollouts with the honesty term, per-turn ΔΦ credit (M6's group rule) as per-token ``turn_advantages`` |
| ``filters`` | ``--dynamic-sampling-filter-path`` (:493), ``--rollout-sample-filter-path`` (:1555) | whole-group drop on any infrastructure failure (slime then refills by over-sampling), a staleness cap, nonzero reward spread |
| ``episode`` | the adapter's ``_run_turn`` (``slime/agent/adapters/common.py:387-476``) | the turn recorder with the pre-model-call snapshot hook, the scripted model and CLI harness used for the P6 equivalence run |
| ``types`` | ``slime/utils/types.py:107-169``, ``slime/rollout/filter_hub/base_types.py:5-14`` | ``Sample`` and ``DynamicFilterOutput``: slime's own when importable, else a field-for-field mirror |

The one source change slime needs is ``batch_builder_turn_advantages.patch`` (the data-parallel split forwards only a
fixed field list, ``slime/data/batch_builder.py:467-487``); it is shipped as a patch file, never applied to their tree.

Layering: ``crucible.train`` never imports ``crucible.factory``. The task side (a ``LongTask`` opened as a World, its
checkpoint sidecar and grader) reaches ``generate`` through a registry pointer (``--crucible-env-provider`` /
``CRUCIBLE_SLIME_ENV``, default ``crucible.factory.envs:slime_provider``).
"""
