# M12 `train` audit (2026-10-13)

Scope: `src/crucible/train/` (runtime, scheduler, pool, mismatch, objectives, trainer, critic, engines, gateway,
workers, weight plane, FSDP / SPMD / cluster / placement, bridges) plus what it calls (`credit/gar.py`,
`interop/verifiers/v1_import.py`, `world/intercept.py`). Base: `3b522cb` (M11). Study: `M12-study-sweep.md` (five
source sweeps G1–G5, four defect hunts H1–H4, about 100 findings). Every fix below has a regression test in
`tests/unit/test_m12_audit_{math,control,rollout,bridges,scale}.py`; each was run against a pristine checkout of
`3b522cb` and fails there (counts per file in §4).

## 1. Requirement ledger

| # | Requirement (source) | Before | After |
|---|---|---|---|
| L1 | Bad numerics never reach the weights (SkyRL, AReaL, slime) | one behaviour logprob of −9999 or −800 NaN'd every parameter; no non-finite check | log ratios clamped at ±20, gated tokens multiply a finite zero weight, non-finite behaviour logprobs drop their token (`nonfinite_tokens`), a non-finite grad norm skips the step (`train/skipped_nonfinite`) |
| L2 | The loss unit is the reward unit (GSPO; verl / MiMo one row per trajectory) | sequence / prompt / const-max-len means counted calls and segments as sequences, ran the whole batch in one forward, were refused under DP; GSPO and HisPO wrong under the default | `objectives.row_weights`: every aggregation is an exact per-rollout weight, micro-batched and split across ranks; `loss_agg="auto"` (GSPO → sequence mean, HisPO → per-group G·max\|y\|); GSPO / MIS-PO / HisPO statistics per rollout (`seq_ids`) |
| L3 | Train only on what was sampled, as it was sampled | the trainer trained calls without logprobs (as π = 1), never compared temperatures, sent no images to vision policies; teacher / reference logprobs at T = 1 over the full vocabulary | trainer gate masks token-incomplete records (`crucible:capture_incomplete`), refuses a temperature mismatch, passes `Turn.media`; teachers score at the turn's temperature over its nucleus; runtime masks incomplete records before grading |
| L4 | Exact token fidelity end to end (TITO) | one `EngineLLM` could splice another episode's reply into a prompt; engines decoded past EOS (invented turns trained); one seed for a whole GRPO group | conversations keyed per sample with a reply nonce; stop ids default to the codec's end-of-turn token and records cut at the first EOS; seed per call |
| L5 | A version stamp is the truth | vLLM loaded weights under running calls; the gateway kept published weights by reference (revived engines served unpublished weights); held engines never got weights; empty resume spans masked rollouts | drain-then-load with an epoch that aborts overlapping calls; detached weight snapshots; publishes reach held engines; spans only for versions that emitted a token; SGLang `weight_version` read back |
| L6 | Staleness bounds are hard and nothing deadlocks | `max_offpolicy_frac` trained at staleness 8 with τ = 1–3 and deadlocked disaggregated runs; `windowed_fifo` deadlocked at default settings and trained past τ; `allow` was unbounded; refills and zero-variance resampling unbounded | at-bound groups go first and bypass the window and the cap; the window counts untrained groups; `allow` ≤ τ + 1; `max_refills`, a zero-variance round cap, refused impossible configs; `drop` cancels doomed in-flight groups |
| L7 | Infra failures mask, never vanish | any non-`EnvInfraError` exception killed the rollout task silently and stalled the run; NaN rewards, engine errors, `mask_sample` records from a `rollout_fn` were invisible to the breaker; unbounded abort re-issue | every exception masks under a named kind (`RunReport.masked_kinds`), re-issue capped (`crucible:reissue_limit`), `EngineCallFailed` masks as `provider_unavailable` |
| L8 | Group grading composes | GRS / BT / tournament → GAR turned passing solutions into failures; two GAR operators disagreed 2×; ops wrote rewards onto masked members; refilled members skipped the stage; the stage crashed on every World rollout | one GAR (`credit.gar.gar`, MiMo's redistribution by default), pass set from the raw outcome, previous stage's score as quality; masked members untouched; stage keyed by member set and written back by position |
| L9 | Critic mode trains what group mode trains | a truncated rollout's target was R + V(s_T) (1.5 for R = 1); shaping and step rewards ignored | truncation replaces R with V(s_T) of the state after the last token (`CriticConfig.truncation`); shaped outcome and step rewards on the critic's chain; incompatible combinations refused |
| L10 | Data parallelism is exact and never hangs | a rank with no rows hung every trainer; a parameter one rank never reached hung DDP and mis-updated FSDP step mode; `src` mixed group and global ranks; no critic / teacher under DP | metric keys agreed across ranks, used-parameter sets agreed and zero-filled, `group_src`, critic replicated, FSDP layer mode refuses data-dependent units loudly |
| L11 | Multi-node runs start, fit and stop cleanly | no multi-node launch; every GPU loaded the full fp32 model (30B-A3B: 122 GB > 94 GB); colocated FSDP could not offload; `LOCAL_RANK` was a physical index; a dead peer blocked others for the full timeout; SPMD resume diverged replicas | Slurm launch (`slurm_env`, `run_node`, `scripts/repacss_train.sbatch`), meta-device init with rank-0 broadcast, FSDP CPU offload, node-local shard groups with a communication term, store-backed SPMD channel with heartbeats, `load_state` on every rank |
| L12 | Bridges produce what the current upstream consumes | verl, TRL, SkyRL, rLLM bridges all failed against their pinned upstreams; turn credit never reached verl's actor | tests run the real upstream code; a `verl.plugins` entry point; credit in `rm_scores`; TRL one episode per row with placeholder rows for masks; SkyRL / rLLM full contracts |
| L13 | Engine-specific formats are the engine's | vLLM top-p nuclei rebuilt from renormalized probabilities (too small); SGLang routed experts parsed per completion; seed key wrong | nucleus = finite processed entries, raw mode refused; slime's R3 row contract; `sampling_seed` |
| L14 | Learning-math parity with the frameworks | matched upstream already (≤ 1e-8) — kept; missing: prime-rl `ipo`, sequence-level IS, CISPO mask, DAPO overlong wiring, Bessel std, REINFORCE++ baseline, reverse-KL gradient, logit softcap | all added; k3 KL now has the reverse-KL gradient (`kl_grad="reverse"`) and clamps |
| L15 | Weights published as trained | QDQ wrote the MXFP4 grid into the fp32 master (sub-grid updates vanished) and never matched HF names; ties rounded down | fp32 master accumulates, trainer and engine share the grid; HF names; ties to even; checkpoint dirs garbage-collected (keep 2, never one being loaded) |

## 2. Defects fixed, by area (ids from the study documents)

| Area | Findings | Severity |
|---|---|---|
| Numerics | H1 V2, G2, G3 D7, G4 #1 | P0 |
| Aggregation and sequence unit | H1 V4, V13, G1 #3, G2 H4, H8, G4 #7, G5 D11 | P1 |
| Trainer gates (validity, temperature, vision, OPD temperature) | G2 H9, G4 #7, H1 V3, V6, V7, H3 V21 | P1 |
| GAR / group stage | H1 V1, V16, G1 #8, G2 H1, H2 V2, V7 | P0 |
| Critic | G1 P0, H1 V11, V12 | P0 |
| QDQ, MXFP4, HF head | G2 H2, H1 V5, V14, V15 | P0 / P2 |
| KL, speed gate, teacher vocab | G4 #3, H1 V8, V9 | P1 |
| Scheduler bounds and progress | G1 #2, #6, G3 D1–D5, H2 V3–V6, V9–V11 | P0 / P1 |
| Rollout failure handling | H2 V1, V8, V13–V20, H3 V9 | P0 / P1 |
| TITO and engines | H3 V1, V2, V4–V8, V10–V20, V22, V23, G3 D6, D8, D9, D11, G2 H3, H5 | P0 / P1 |
| Data parallelism | H4 V1, V2, V11, V12, V16, G4 #2, G5 D1 | P0 |
| Scale-out | H4 V3–V10, V13–V15, V17 (FSDP part), V18, G5 D2, D3, D5, D6, D8, D10, H2 V12, V20 | P0 / P1 |
| Bridges and imports | H4 V19–V28, G2 H6, H7, G4 #4–6, G5 D4, G1 #7 | P1 |
| Vacuous tests | G1 #6, G3 D4, H2 V11, H4 V27, H1 V17 | P1 |

## 3. Corrections to earlier claims

- M12-train.md §4 said `allow` lets staleness reach τ + 1; it was unbounded (now bounded). §9 items 25–28 were "done"
  on vacuous tests; the windowed-FIFO lag example attributed to the article is not in the Chinese text.
- τ = 4 was labelled "MiMo's value"; MiMo counts versions spanned, so its T = 4 is Crucible τ = 3 (τ = T − 1).
  AReaL's η is Crucible S − 1, and AReaL's manager bounds capacity, not per-sample staleness.
- The doc's verl turn-credit path never reached verl's trainer actor; bridges "built" meant "shaped like the API".
- REPACSS is 4 × H100 NVL 94 GB per node: an 8-GPU run is two nodes.

## 4. Gates and numbers

- Regression tests (functions; parametrized cases in parentheses as reported by the batches): math 30 (44), control
  37, rollout 26 (28), bridges 37 (46), scale 30 (34). Each fails on `3b522cb` (math: 42 of 44 cases; the 2 that pass on
  old are MIS-PO and CISPO, whose gates already rejected the extreme token).
- Full suite on the fixed tree: see the commit message (filled at commit time).
- lint-imports: 4 contracts kept; ruff clean.
- Not run (needs approval): the REPACSS stages 0–6 in `scripts/repacss_train.sbatch` (≈ 50–55 GPU-hours), the vLLM
  NCCL / CUDA-IPC weight path, a real verl / TRL / SkyRL training step.

## 5. Open

- checkpoint-engine bucketed broadcast + CUDA IPC into vLLM (G5 D7, pick 17); DDP bucketed all-reduce (H4 V17).
- verl legacy trainer trains outcome only; SkyRL sums rows (no turn credit); TRL masks multi-segment rollouts.
- top-p / top-k on the verifiers and verl paths need candidate capture; varlen packing; AReaL-style OpenAI proxy.
- FSDP layer mode: an unreached parameter inside a unit that ran gets a zero gradient (AdamW decay); step mode is exact.
- REPACSS preset numbers (25 Gb/s, 512 GB host, 128 CPU slots) are placeholders until stage 0 measures them.
- Ling focus, StepFun asymmetric rubric mapping, PARL annealing, Kimi verbosity-as-loss: deferred until a pack needs them.
