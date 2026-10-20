"""The Tinker bridge (swarm-pipeline/09): Tinker (Thinking Machines) trains the student and samples every non-Claude model.

Tinker is a hosted LoRA trainer and sampler: ``ServiceClient`` -> ``TrainingClient`` (``forward_backward`` with an RL
loss, ``optim_step``, ``save_weights_for_sampler`` / ``save_state``) and ``SamplingClient`` (token-in/token-out with
logprobs). Crucible keeps the environment, the renderer, the credit and the accounting; Tinker gets token ids, sampling
logprobs and finished per-token advantages. The SDK (``tinker``, Apache-2.0) is the optional extra ``crucible[tinker]``
and is imported lazily.

| module | what it does |
|---|---|
| ``models`` | the model registry: Tinker id, family, context (64K / ``:peft:262144`` variants), tool-call format, the pinned price table (2026-10-20) |
| ``spend`` | ``SpendMeter``: per-call token booking (cached / uncached prefill, sample, train, checkpoints), the hard USD cap, ``spend.jsonl`` receipts |
| ``backend`` | the narrow service interface (``RealTinker`` over the SDK), the bridge's own types, infra-vs-task error split with backoff |
| ``fake`` | ``FakeTinker``: the same interface in memory (scripted sampling, the documented losses, checkpoints): tests and dry runs |
| ``codec`` | ``TinkerChatCodec``: the model's HF chat template with exact TITO, tool-call parsers per family (ported from tinker-cookbook), ``enable_thinking`` explicit |
| ``sampler`` | ``TinkerSampler`` (the ``EngineLLM`` generate function: context check, cap hold, booking, version per call, refresh), ``SamplerLLM`` (the policy's client), ``TinkerChatClient`` (a factory role: complete / complete_json with schema retry) |
| ``datums`` | records -> ``TokenDatum``: outcome advantage over the group, honesty composed, M6 turn credit, one datum per TITO sequence |
| ``trainer`` | ``TinkerTrainer``: groups from ``make_env`` environments, the slime filters (objective, infra, staleness, spread), the update, resume, the cap |

The task side (``make_env``, targeting records, the benchmark metrics) is joined in ``crucible.factory.tinker`` (the
factory sits above both ``train`` and ``bench``); ``scripts/tinker_rl.py`` is the command line.
"""
