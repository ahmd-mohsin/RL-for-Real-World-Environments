"""The DeltaAI trainer bridge: Crucible's RL loop on prime-rl's GPU trainer and vLLM server (swarm-pipeline/14).

prime-rl (PrimeIntellect-ai/prime-rl v0.10.0, Apache-2.0; DEPEND on the cluster, imported lazily, never on the Mac) is
chosen over slime, verl, SkyRL and NeMo-RL for this student (14 §1): it trains the student's own architecture
(``qwen3_5_moe``: gated delta-net + full attention, 256 experts) natively, LoRA on the routed experts included with vLLM's
per-expert adapter format, expert and ring context parallel for 100K+ sequences, aarch64 wheels and a multi-arch image for
GH200, a Slurm launcher, filesystem rollout and weight transports, and an import-path custom loss.

Crucible keeps its loop (``bridges/tinker/trainer.TinkerTrainer``: its own runtime runs the groups, so the native harness,
checkpoint sidecar, path gates, honesty, compaction, filters, quarantine, the difficulty sampler, M6 credit and resume are
unchanged); prime-rl replaces the two remote calls:

| module | what |
|---|---|
| ``backend`` | ``PrimeRLBackend``: the Tinker bridge's ``Backend`` protocol over vLLM ``/v1/completions`` (token ids in and out) and prime-rl's batch / broadcast handshake; ``local_model`` (zero prices) |
| ``wire`` | ``TokenDatum`` -> ``TrainingSample`` (full-length per-token streams), the token-count factor that keeps ``loss_scale`` exact, prime-rl's packer |
| ``loss`` | ``crucible_loss``: prime-rl custom loss with the Tinker objectives (``importance_sampling`` / ``ppo`` / ``cispo``) plus the stable_reinforce-style sampler anchor β·Σ(log q − log p_θ) |
| ``deploy`` | ``DeltaConfig`` (account ``bjhg-dtai-gh`` only), the trainer / inference TOMLs, the Slurm job, ``plan`` (GPU-hours) |

The CLI is ``scripts/delta_rl.py`` (the flags of ``tinker_rl.py train``, plus the deployment's). This package imports no
framework at module level (the trainer process imports ``loss`` alone).
"""
