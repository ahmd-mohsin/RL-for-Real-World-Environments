# M5/M6/harness research, slice 01: the articles

Date: 2026-10-09. Scope: every text file in `RL-Environments/07-articles/`, plus the FineEnvs article sources (embeds and data files) and two load-bearing lines of the companion training code. Read-only; no network.

**Path aliases.** All paths are relative to `/Users/muahmed/Desktop/Projects/RL-for-Real-World-Environments/`.

| Alias | Path |
|---|---|
| GS | `RL-Environments/07-articles/gitswarm-2610.04862.txt` (full GitSwarm paper, arXiv 2610.04862) |
| GSP | `RL-Environments/07-articles/git-swarm.md` (project page) |
| MH | `RL-Environments/07-articles/hf-space-fineenvs-multi-harness-rl.md` ("The ultimate guide to multi-harness RL") |
| MHsrc | `RL-Environments/06-frameworks-and-env-code/adithya-s-k-fineenvs/content/articles/multi-harness-rl/app/src/content/` |
| MHcode | `RL-Environments/06-frameworks-and-env-code/adithya-s-k-fineenvs/05-multi-harness-rl/` |
| EG | `RL-Environments/07-articles/hf-space-rl-environments-guide.md` (framework comparison) |
| EGsrc | `RL-Environments/06-frameworks-and-env-code/adithya-s-k-fineenvs/content/articles/rl-environments-guide/app/src/content/embeds/` |
| HB | `RL-Environments/07-articles/hf-blog-rl-environments.md` |
| VF | `RL-Environments/07-articles/pi-verifiers-v1.md` |
| SW | `RL-Environments/07-articles/pi-on-the-nature-of-the-swarm.md` |
| PMB | `RL-Environments/07-articles/pm-building-rl-environments-for-superintelligence.md` |
| KI | `RL-Environments/07-articles/pm-introducing-karotte.md` |
| KD | `RL-Environments/07-articles/pm-karotte-technical-deep-dive.md` |
| R2E | `RL-Environments/07-articles/hf-repo2rlenv-how-not-to-build-coding-environments.md` |
| DES / ARCH | `crucible/DESIGN.md` / `crucible/ARCHITECTURE.md` |

**Already absorbed.** Commit `538627f` cross-checked M0–M4 against the Preference Model, Repo2RLEnv and Prime Intellect swarm articles. That work produced P20–P28 (DES:103-111) and `docs/reviews/2026-10-09-articles-crosscheck.md`. This report repeats those sources only where they bear on M5, M6 or harnesses. The new material is GitSwarm, the two FineEnvs guides and verifiers v1.

---

## 1. TL;DR: the 18 most useful takeaways

1. **Keep two kinds of edge: the physical parent and the semantic sources.** In GitSwarm, every commit has one Git parent (the state the worker started from) and a mandatory `informed_by=` set U_i, with p_i ∈ U_i. The graph is E = {(p_i,v_i)} ∪ {(u,v_i): u ∈ U_i \ {p_i}} (GS:159-179, prompt GS:3324-3326, 3601-3614). On ProgramBench, Git parentage covers only about 25% of the winner's ancestry; the `informed_by` edges connect most of the rest (GS:2262-2265).
   **Crucible should** give `crucible.trace/v1` events two fields: `state_parent` (the state_ref the step started from) and `informed_by: [seq|sha]`. That makes the M5 step graph a DAG, not a chain. For expert traces, derive `informed_by` from `reply_to` (review → correction) and CI-run → fix edges; for multi-agent Worlds (P28, DES:111), require it on every write.

2. **Ancestry is not causation. Measure reuse, consumption and ancestry separately, and never use them as reward.** GitSwarm reports four separate numbers:
   - declared reuse: the share of eligible episodes that cite earlier work (99.9% / 96.2%);
   - built-upon rate: the share of contributions cited later (94.7% / 44.4%);
   - winner transitive-ancestry coverage (81.6–92.6% / 21.3–33.0%);
   - an artifact funnel: created → accessed → actively reused → winner-linked (GS:495-557, 2247-2270, 2333-2340, 2485-2490).

   The authors say outright that the declared dependencies are "not independently established as causal influence" (GS:175-179, 744-755).
   **Crucible should** add these as M5 trace diagnostics and benchmark metrics (§11.1). A policy can inflate any of them by citing everything, so none of them is reward.

3. **GitSwarm's proposed causal test is Crucible's C3.** "A direct intervention would resume matched runs from the same candidate, with and without access to its preceding research history, under equal remaining compute" (GS:747-752).
   **Crucible should** extend C3 from V̂(s_t) to a *contribution ablation*: ΔV̂(c) = V̂(s_t) − V̂(s_t ∖ c), where c is a step's artifacts or a cited dependency, using the same K and the same harness. Use it to validate `informed_by` edges and milestones (does ΔΦ track ΔV̂?), not as an online reward.

4. **Every node must be a candidate: separate the publication gate from correctness.** Every GitSwarm graph-expansion commit "must preserve the task-specific structure … to stand as a candidate answer" (GS:3330-3333). ProgramBench enforces this with `compile.sh → ./executable` on every node (GS:3650-3657). Validation "enforces task-specific publication requirements, not solution correctness" (GS:187-189).
   **Crucible should** define a cheap per-family *candidate invariant* (for GPU migration: the configure and build steps run). The Workflow State Tracker evaluates milestones only on states that satisfy it, and records `not_candidate` otherwise. This is the G-versus-O split of DES:409-415, applied at step granularity.

5. **Use the bounded-contribution scope as the unit of segmentation and intent.** Before editing, each GitSwarm worker publishes "Gap | Deliverable | Success | Abort" plus target commits (GS:3423-3447). It ends with `MY_THOUGHTS.md` (sources and how they were used, evidence, uncertainty, TODOs) and a candid `EVALUATION.md` (GS:3481-3500).
   **Crucible should** use this schema for (a) M5 step and stage segmentation of agent sessions (one scoped contribution is one step), (b) the handoff note given to C3 continuations, and (c) the M2 instruction template for multi-agent Worlds. Claims in these notes are never reward (R6, DES:448).

6. **Seed stage induction with GitSwarm's bottom-up activity taxonomy.** For programming GitSwarm labels six activities: bootstrap, capability expansion, candidate synthesis, fidelity repair, empirical validation, candidate selection (GS:2563-2589). Labels come from the diff against the *physical parent only*, so inherited files are not attributed (GS:2591-2601). The time profile is: a short bootstrap wave, sustained expansion and repair, then a late selection pulse (GS:2603-2615, 2805-2817).
   **Crucible should** use these as the seed label set for the κ ≥ 0.7 manual stage-label protocol, and as a sanity check on the ≥90% precedence graph (DES:350).

7. **Cheap evaluations misrank, and noise is the size of real progress.** Examples from the research runs:
   - 128-example evaluations disagreed with the full set on 5 of 12 model pairs, and with 512-example evaluations on 15 of 50 (GS:2059-2064).
   - Proxy-to-full promotion helped in 15 of 17 cases early in the run and 1 of 8 late (GS:1948-1950).
   - Gains measured in isolation vanished when combined with the leader (GS:1951-1952).
   - Seed noise of 0.010 matched the size of real steps (GS:1953-1954).

   **Crucible should** gate the fast and full verifier tiers (DES:616-623) on *pairwise rank agreement*, not just on whether pass/fail agrees. It should adopt a 128 → 512 → full funnel for expensive milestones, and should estimate Φ weights conditioned on the current leader state, because milestone effects may not add up.

8. **Negative results get reused heavily, so don't penalize evidence-only steps.** 70.1% of negative-labelled NanoChat research commits were cited later (76.4% over the whole run) (GS:2123-2129). In the Looped run, 74% of negative architectural results were cited (GS:466-467). In RMT, the most consequential reuse was a failure that another worker revisited 49.5 hours later (GS:1837-1840).
   **Crucible should** expect ΔΦ = 0 on investigate, probe and refute steps (consistent with P27, DES:110) and should credit such steps only through C3 counterfactual value. When reporting corr(C3, ΔΦ) on task04, it should split state-changing steps from investigate steps.

9. **Votes and self-assessments are coordination state, not reward.** GitSwarm's prompt says the nomination tally is "selection state, not behavioral evidence … agreement can be correlated or redundant" (GS:3404-3408, 2865-2867). On the dev set, a central selector beat decentralized nomination by 2.34 pp; a central *aggregator* lost 3.87 pp and took 16% longer (GS:3225-3251). Votes attach to exact SHAs, so every repair splits support (GS:2815-2817).
   **Crucible should** never read agent votes, `EVALUATION.md` claims or judge-free agreement as reward. Readout for multi-agent Worlds is the verifier.

10. **Use the IMO judge protocol as the LLM-judge baseline that Φ must beat.** GitSwarm graded proofs with an LLM judge on a 0/1/6/7 scale (a wrong answer presented as correct scores 0) and reported the mean and min–max over 3 judge passes (GS:3708-3717, 1719-1728).
    **Crucible should** build the M6 gate baseline the same way: a judge reads the midpoint state and trajectory prefix and outputs P(success), averaged over 3 passes, with its variance reported. Φ(midpoint) must beat it at AUC ≥ 0.75 on held-out traces.

11. **Capture exact tokens at the model endpoint; never re-tokenize.** The OpenEnv capture-proxy contract (MH:480-584):
    - the session id is the API key, and an unregistered key gets a 401;
    - the dialect is detected from the path, then the headers, then the body shape;
    - the call to the engine is non-streaming, replayed to the harness as a stream if it asked for one;
    - the proxy records prompt ids, sampled ids and per-token logprobs, sampling with `top_p=1, top_k=-1` (this moved the importance ratio from 0.985–0.993 to 0.9984–0.9999);
    - vLLM runs with `--return-tokens-as-token-ids --logprobs-mode processed_logprobs`;
    - the proxy probes each engine and grades its capture level (only "tokens" is trainable; hosted APIs are evaluation-only);
    - the rollout graph links each call by longest exact token prefix, and every root-to-leaf path is one sequence;
    - loss mask: context 0, sampled 1; turns with misaligned logprobs are kept as context only;
    - each rollout is cross-checked call by call against Harbor's ATIF.

    **Crucible should** upgrade the built M3 `capture_proxy` (OpenAI + Anthropic, streaming; ARCH:40) to this contract. It also needs the OpenAI Responses dialect, which Codex requires (MHsrc `assets/data/harness-ecosystem.json`; VF:68).

12. **Key dense credit to model-call nodes, because compaction and subagents branch the trace.** Verifiers v1 stores a message DAG whose size is linear in turns, not quadratic; each root-to-leaf branch is one trainable sample, and a compaction or subagent starts a new branch (VF:95-117). Because Claude Code rewrites its history, each of its rollouts became about 8 training rows; Claude Code produced 77% of the rows from 25% of the rollouts (MH:1012-1014, 1196-1197).
    **Crucible should** add the proxy's `call_id` to each `steps.jsonl` line, so the turn-level advantage A_t (DES:426) can be attached to the sampled tokens in whichever branch contains that call. Loss should be normalized per rollout, not per row, so harnesses that rewrite history don't dominate.

13. **Validate Crucible's reward bounds with the multi-harness reward lessons.**
    - A "+0.2 for submitting anything" term made the policy submit immediately: held-out accuracy fell 0.740 → 0.178 while training reward looked healthy (MH:704-711). This validates R2 and κ < 1.
    - Correctness-only reward let tool calls grow from 13 to 41 per rollout (MH:884-886), and 35–58% of optimizer steps had zero contrast (MH:1194-1195).
    - The bonus actually used is `r = correct · (1 + 0.1·15/(15+calls))`, with None (infra) passed through (MHcode `train/multi_harness.py:78-85`). It was the only contrast in 17.5–22.6% of groups (MH:912-916).

    **Crucible should** keep β·e inside O (DES:409-415), and log the zero-contrast-group share, tool calls per rollout, the submit rate and the output-cap truncation rate as collapse warnings.

14. **Match the output caps in training and evaluation.** The Qwen multi-harness run trained with 16,384-token responses and was evaluated at 4,096. Answers grew until truncated cells went from 9 to 556 of 1,000. The OpenCode-only run's submit rate fell from 69% to 41% (MH:1163-1176). Fixing the caps was one of three changes in the LFM runs that did not decline (MH:1205-1210).
    **Crucible should** make `max_tokens` per call and the episode turn and token budgets part of the task card. The same values must be used for C3 continuations, training and evaluation.

15. **The harness is a first-order variable, so condition every value estimate on it.** The same weights solved 62% of tasks under mini-swe-agent and 33% under Claude Code (MH:871-874). A harness swap moved a score by 13 pp, while a model swap moved it 2.5–5 pp (MH:269-270). OpenForgeRL found that 3-harness training beat 1-harness training even on the single harness's home turf, 48.5 vs 46.0 (MH:375-377).
    **Crucible should** record `harness_id@version` in every C3 estimate and calibration record, and keep one harness per GRPO group (MH:805-807). It should calibrate on at least two harnesses, with mini-swe-agent as the minimal reference.

16. **Mask harness-specific failure modes as infra (C6).**
    - Gemini issued a whole workflow in one response, predicting outputs and guessing a SHA. The fix was a harness-side `set -e`, skipping the remaining blocks, and rejecting pre-branch mutations (GS:3746-3757).
    - An empty `tools` array got a 400 that looked like a well-formed short rollout (MH:723-725).
    - An empty turn is not a stop; the loop should nudge (KD:78-81).
    - A call that hits the model-call budget has its final reply dropped (MH:583-584).
    - The proxy saturates at about 200 concurrent sessions and crashes at 320 (MH:582-583).
    - 3.1% of charged GitSwarm episodes produced no accepted action: 515 failed publication validation, 79 failed nomination validation, 21 failed on API, turn-limit or protocol errors (GS:2677-2680).

    **Crucible should** add these to the DESIGN Appendix B.5 failure classes and the C6 mask.

17. **Make abstention a first-class action.** GitSwarm's `gs_abstain` does not consume budget. An abstention calibration review found 94.9% justified (GS:194, 2908-2922, 3571-3581). Preference Model argues that hard tasks need "an honest way to give up" (PMB:39).
    **Crucible should** give Worlds an `abstain(rationale)` terminal tool, distinct from failure in traces and in C3 statistics (see Open Question 3 on its reward).

18. **Continue C3 from state with a fresh context, and also from the prefix.** GitSwarm workers each start with a fresh context and retrieve what they need from a repository "that is never summarized or pruned" (GS:1166-1168). Prime Intellect argues that compaction "is a bet" about what will matter later (SW:21-23).
    **Crucible should** run C3 in two modes: (a) World snapshot + fresh context + handoff note, which measures *state* value and is consistent with P12 (DES:86); (b) Harbor `trajectory.json` prefix seeding (DES:440). Report the gap between the two as the "context value" of the conversation prefix.

---

## 2. Per-source mechanisms

### 2.1 GitSwarm: Decentralized Compounding Inference (GS, read in full including appendices A–H)

**Substrate and protocol (M5: state representation).**
- *Design properties.* The design rests on three properties: persistence, branchability and attribution. The harness "coordinates execution but does not assign intellectual roles or prescribe a sequence of problem-solving stages" (GS:150-157).
- *Graph.* The task q and its artifacts are the root commit v0. Accepted contributions form G_t = (V_t, E_t), and each v_i is an atomic commit with physical parent p_i (GS:159-165). The semantic sources U_i ⊆ V_{i−1} must include p_i (GS:166-172). In the prompt: "the Git parent records physical ancestry; provenance records semantic influence" (GS:3324-3326). Abbreviated SHAs are valid if unique (GS:3608-3609); mistyped hashes in the Looped analysis were recovered by unique prefix (GS:2040-2041).
- *Episode actions.* Every episode ends in exactly one of: Contribute (one validated atomic commit), Nominate (a separate record with a rationale), or Abstain (GS:183-196).
- *Budget.* B counts charged episodes; voluntary abstentions are replaced for free, though they still cost tokens; concurrency is C = 5 (GS:183-185, 194-196, 284-285).
- *Shared state.* Each worker gets a spawn-time dashboard and can call `gs_dashboard`. In research mode the dashboard shows GPU, CPU and job usage, a low-utilization warning, and the best validation score with its commit (GS:198-205). `gs_inflight` is a public activity log (scope announcement, progress updates, uncommitted diff), readable but not modifiable by peers (GS:206-210, 233-263). In-flight content "is unfinished coordination evidence, not durable provenance" (GS:3390-3392).
- *Readout.* v* = argmax_v Σ_i 1[c_i = v], ties broken by lexicographic full SHA, falling back to the root if no nomination is valid (GS:212-224, 3335-3338). For research tasks, nomination is removed and selection is by validation score (GS:399-400, 3302-3307).
- *Nomination hygiene.* Nominators announce `--stage nominating` without disclosing a preference, so judgments stay independent until ballots are published (GS:3512-3516, 3537-3538). "Pass counts from different candidate-authored suites are not directly comparable" (GS:3525-3527). The scheduler validates the declaration, the exact parent, the provenance, a tree unchanged except for `BEST_SOLUTION.md`, and the rationale (GS:3628-3631).

**How work is segmented.**
- *Scope first.* A worker publishes Gap / Deliverable / Success / Abort plus `--target`. The success criterion applies to the scoped contribution, not the whole query. "A negative result may be committed only when it directly satisfies the original scoped deliverable" (GS:3423-3447).
- *Bounded steps.* "Progress on a long or ambitious approach should be distributed across successive nodes" (GS:3457-3458). Exactly one `gs_fork` and exactly one commit are allowed (GS:3462-3474, 3601).
- *Commit format.* `[<type-words>|informed_by=<sha1>,…] <summary>` (GS:3601-3614). The type words are free-form hyphenated labels, effectively self-declared activity labels.
- *Post-hoc taxonomy.*
  - Programming: Bootstrap / Capability expansion / Candidate synthesis / Fidelity repair / Empirical validation / Candidate selection.
  - Proofs: Initial proof / Proof exploration / Proof synthesis / Proof repair / Proof validation / Candidate selection (GS:2563-2589).
  - Method: review the declared scope, tool trajectory, terminal action, commit metadata and changed files (the diff against the physical parent only); summarize; group by hand; one primary label per episode (GS:2591-2601).
- *Workflow archetypes inferred from the graph.* Late integration (Delta: branches converge at 80% of the run, winner at 92%), divide-and-conquer (Pandoc), and search-and-pivot (Cheat: CLI work for the first 35%, then four renderer experiments within 1 point of progress) (GS:2516-2545, 639-663).

**Accumulation metrics (M5/M6 diagnostics, with exact definitions).**
- *Declared reuse.* 99.9% (ProgramBench) and 96.2% (proof) of eligible episodes cite earlier work. Built-upon: 94.7% and 44.4% of published contributions are later named as a Git parent or `informed_by` source. Pooled over 200 and 150 runs (GS:495-498, 538-539).
- *Selected-solution ancestry.* Recursively follow both edge kinds from v*, excluding the root and nomination records. Coverage is the share of the "substantive commit graph": 81.6–92.6% on ProgramBench and 21.3–33.0% on proofs. Coverage does not grow consistently with budget, and Git parentage alone is about a quarter of the graph (GS:540-549, 2251-2265). Interpretation: "Local reuse and selected ancestry must therefore be measured separately" (GS:2267-2270).
- *Artifact consumption.*
  - Accessed: a later trace explicitly reads, searches, parses, compares or opens the artifact.
  - Actively reused: the artifact informs a later contribution; an executable must be executed, imported, adapted or supplied to another computation, and the consumer must cite the producer.
  - Winner-linked: the consumer is in v*'s combined ancestry.
  - "Passive inheritance and directory listings do not count" (GS:2333-2340, 2485-2490).
  - Results: 79.9–82.5% of eligible voluntary bundles were actively reused. On proofs, 583 of 759 probe bundles were opened, 172 were run or adapted with a citation, and 52 were winner-linked. `MY_THOUGHTS.md` and `EVALUATION.md` were accessed in 98–99% of revisions (GS:550-557, 2446-2477).
  - Files created together for one role are counted as one *logical bundle* (GS:2313-2317).
- *Timing.*
  - ProgramBench: the winner is first committed at 93.1–98.0% median progress; the first vote comes at 68.4% (B100) down to 16.1% (B500) because a roughly fixed amount of work precedes the first submittable candidate.
  - Proofs: the winner appears at 8.6–17.5%, and the selection share rises from 44.2% to 69.7% of episodes (GS:2805-2826).
- *Inspection.* Workers open about 6 candidates on ProgramBench and about 3 on proofs, regardless of graph size, so the history fraction reached falls (IMO: 81.9% at B20 → 53.8% at B160) (GS:3179-3196).
- *Redundancy.* About 1 in 6 ProgramBench workers abstains as redundant, mostly citing in-flight work; 97.1% of the colliding commits landed after the abstainer started. Proof abstainers mostly cite committed work, under 2.7% (GS:2879-2901). Abstainers use 6–7% of tokens (GS:2903-2906). The score-blind calibration labels are Justified / Premature-recovered / Premature-unrecovered / Spurious / Ambiguous; 94.9% were justified (GS:2908-2922).

**Controlled substrate tests (Appendix C).** Each one isolates a property and gives a metric we can reuse.
- *Persistence: Hidden Rulebook.* Coverage = distinct relevant records queried / Q, with Q = P(D+1). Shared memory keeps coverage high; independent workers with 4Q/3 queries reach only 70–75% (B=2) or 20–23% (k=4) (GS:1392-1444).
- *Branchability: Hidden Maze.* Metric: the *repeated-move fraction* (steps whose directed room–exit pair was already traversed). Branch-native checkpoint trees beat flat logs. "The harness validates every restored checkpoint and commits the final state … so workers cannot fabricate states" (GS:1536-1560). Caveat: the prompt also changed (GS:1591-1592).
- *Attribution: Retracted Evidence.* "An artifact is valid only if its producer and every artifact in its ancestry remain valid" (GS:1600). Metric: the fraction of the archive read before the first exact submission. Attribution metadata reduces it, and the gap grows with lineage depth (GS:1594-1672).

**Verification of intermediate results (M5/M6).**
- *Evaluation funnel.* A worker measured that the 128-example evaluation misranked pairs, and the swarm then adopted a 128 → 512 → 4,873 funnel. Workers also overturned each other's verdicts, and four workers independently ran parameter-matched controls (GS:2059-2069).
- *Shared protocols emerged.* A fixed proxy-context set was used by 257 commits, and commits that built measurement tools and controls were among the most cited (GS:2070-2072).
- *Replication.* Repeating an identical source gave 0.911027 vs 0.911918, which turned a claimed gain into "an unresolved near-tie" (GS:2150-2157). A predeclared reverse-order replication checked whether experiment order could explain a gap (GS:2159-2170). Ten measurements of byte-identical source became a reusable noise baseline (GS:2172-2179).
- *Rejection and audit.* The extension's best observation, 0.898671, had an exact-source repeat of 0.899864, and its publication was rejected because it added a file outside the allowed paths. In addition, 701 harness records exceeded the original 10-minute timeout (GS:2213-2226).
- *Isolation.* Isolated worktrees and environments with no internet; candidates scored by fixed evaluators on protected-dev and held-out sets (GS:807-809). The RMT prompt says "never invent metrics" and "every submitted job must become terminal before this worker exits" (GS:3724-3740).

**Credit-relevant claims (M6).**
- The authors' RL agenda: "rather than rewarding only the outcome of an individual episode, such training could incentivize contributions whose value emerges through subsequent work" (GS:780-784). No reward is specified, and causal attribution is left open (GS:744-755).
- Over-concentration: the share of commits building on the adapter rose from 17% to about 90%, and a worker flagged it (GS:2073-2076). The RMT prompt has a "watch for search collapse" clause (GS:3735-3739).
- The NanoChat run assigned experimenter and critic roles at 3:1, so its specialization claims are qualified (GS:2196-2202).

**Harness facts.**
- *Interfaces.* Codex workers used a guarded MCP tool, `pb.container_bash`, through the Codex CLI. Gemini workers had no tool definition; the harness executed their fenced bash blocks. GPT-5.5 workers used a mix (GS:3278-3283, 3804-3821).
- *Baselines.* Single agents under bash, mini-SWE-agent or Codex, "Forced" with `Keep working...` at every natural stop, with intermediate checkpoints graded for an anytime curve (GS:280-286, 1320-1323).
- *Compute accounting.* Input tokens count the full prompt of every request, including accumulated context. Output tokens count new tokens, including hidden reasoning (GS:287-294).

### 2.2 GitSwarm project page (GSP)
This restates §2.1: "no planner, no assigned roles and no privileged final judge" (GSP:3), and following both links back from a commit gives its ancestry (GSP:9). It adds no new mechanism.

### 2.3 The ultimate guide to multi-harness RL (MH, every chapter; MHsrc embeds and data)

**Definitions and the black-box contract.**
- *Harness.* "The software layer between the model and the task that constructs the context …, mediates its tool calls, validates its outputs, and decides when to retry, escalate, or stop" (MH:71-75).
- *Ownership of the loop.* White-box means the trainer owns the loop; black-box means the harness owns it and the trainer sees only request/response pairs (MH:133-155). KAT-Coder uses "white/black-box" for context handling instead; MH treats that as a separate axis (MH:157-163).
- *Why text and score are not enough.* An on-policy gradient needs the sampled token ids and their behavior logprobs. Re-tokenizing can change the ids, harnesses edit or repair text, and recomputing logprobs after an update makes the IS ratio 1 (MH:192-211). Keep the sampled ids and append the template suffix by id concatenation; 18 of 19 chat templates extend token for token (MH:220-225).
- *Alternatives.* PI's `renderers` keeps tokens trainer-side but breaks on harnesses that repair or compact (MH:227-232). OpenForgeRL rebuilds samples from prompt/response text and still reports the best multi-harness results, so the question is "not settled" (MH:234-237).

**Why multi-harness.**
- Benchmarks now name a harness (MH:248-253).
- Scaffold reversal on 100 SWE-bench Verified tasks across three harness configurations (Minimal / Improved / Full): a harness swap moved GLM-5.1 by 13 pp, a model swap only 2.5–5 pp (MH:255-271).
- Orchard: OpenSWE-32B lost 58.8 pp under Kimi-CLI. The paper distinguishes "degraded resolve rate" from "catastrophic format failure" (MH:277-301).
- KAT's three overfitting modes: action **format**, **context structure**, and **control flow** (retries and stop rules) (MH:305-312). Concretely: JSON tool calls sent to Aider's prose SEARCH/REPLACE format are scored as non-answers; a harness that compacts history causes repeated work; a harness without auto-retry causes repeated failing commands (MHsrc `embeds/d3-overfit-modes.html`).
- Tool-name and argument mismatches are rejected before the tool runs (MH:340-342).
- Labs already train across harnesses (MH:354-377): Poolside put 1.3B tokens of external-harness SFT data in, but ran RL only in `pool`; Kimi K3 composes harnesses from modules; Qwen3-Coder-Next uses 6 harnesses; Liquid picks a random harness per task.

**Harness / API matrix** (MHsrc `assets/data/harness-ecosystem.json`):

| Harness | API | Tool calls |
|---|---|---|
| Claude Code | Anthropic Messages only | yes |
| Codex | OpenAI Responses only | yes |
| OpenCode | all four | yes |
| Mini-SWE-Agent | chat completions / Responses / plain text | one tool |
| Terminus 2 | chat completions | none (plain text) |
| Aider | chat completions | none (prose edits) |
| OpenHands SDK | chat completions | yes |
| SWE-agent | chat completions | yes (tools are shell scripts) |
| Qwen Code | Google by default | yes |
| Goose | Anthropic / chat / Google | yes |
| Kimi CLI | chat completions | yes |

**Wiring seams.** The ten harnesses validated in OpenEnv 0.7.0 (MHsrc `embeds/harness-seams.html:40-49,130`) get the proxy URL and session id as follows:

| Harness | Seam |
|---|---|
| Claude Code | `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`, also in the server process |
| Gemini CLI | `GOOGLE_GEMINI_BASE_URL` / `GEMINI_API_KEY`, also in the server process |
| Codex | `OPENAI_BASE_URL` / `OPENAI_API_KEY` |
| Mini-SWE-Agent | `OPENAI_BASE_URL` / `MSWEA_API_KEY` |
| Qwen Code | `OPENAI_BASE_URL` / `OPENAI_API_KEY` |
| Vibe | `VIBE_API_BASE` / `OPENAI_API_KEY` |
| OpenHands SDK | `LLM_BASE_URL` / `LLM_API_KEY` |
| OpenCode | provider block in its config file |
| Pi | `~/.pi/agent/models.json` |
| Terminus 2 | host-side `api_base` / `api_key` arguments |

Another 19 harnesses are listed but not validated (2 unstable, 3 unsupported, 3 blocked, 11 untested). Variables are set for the agent's run only, so the session id never reaches a verifier that needs a real key (MH:696-700).

**Capture proxy and rollout graph.** The contract is in TL;DR #11 (MH:480-584). Additional details:
- Capture has three levels: tokens (prompt ids, sampled ids, aligned logprobs; trainable), logprobs without ids, and text (evaluation only) (MHsrc `embeds/d3-capture-levels.html`).
- The proxy is a single process (MH:582-583).
- Every call is a node, linked to the earlier call whose prompt plus completion is the longest exact token prefix of its prompt. A retry is a sibling that never continued; a subagent or a compaction starts a new root (MH:541-547).
- Validation runs as each call arrives, not at export (MH:556-559).
- Any program can be captured: `POST /sessions`, then set the base URL and use the session id as the key, then `GET /sessions/{id}/rollout` (MH:568-580).

**Serving and failure isolation.**
- The trainer talks to the environment over HTTP. An earlier in-process version let an exception from inside someone else's agent crash a rank and hang NCCL. Now a failed rollout returns `ok=False, reward=None` (MH:635-639).
- `rollout` has no default `--llm-url`, because a stale endpoint produces rollouts that look fine but carry no token ids (MH:657-660).
- `max_concurrent_envs` defaults to 4, below a GRPO group of 8; past the limit, `serve` returns `CAPACITY_REACHED` (MH:739-741).

**Reward selection.**
- The integration uses the only score, or the one named `reward`, and otherwise asks the caller. "It never combines scores itself, because whatever weighting it picked would become the training objective."
- The +0.2 submit-anything collapse (TL;DR #13).
- If the verifier never runs, the reward stays `None` all the way to the trainer (MH:704-711).
- Independent ATIF cross-check, call by call: number of turns, completion tokens per call, and which calls are agent steps (MH:712-725).

**Training recipe (LFM2.5-2.6B).**
- Data: 1,000 SmolDataEnvs tasks (400 medium, 600 hard) in a fixed order; 250 held-out tasks; 4 harnesses × 250 = 1,000 evaluation cells every 100 steps (MH:832-842).
- Algorithm: async GRPO in TRL for 1,000 steps; rollouts are usually 2 optimizer steps stale, never more than 4. Two H100s, one E2B sandbox per rollout (1 CPU, 4 GB); 32 h single-harness and 46 h multi-harness (MH:842-849).
- Code hyperparameters (MHcode `train/multi_harness.py:105-130`): `learning_rate=3e-6` constant, `num_generations=8`, per-device batch 4 × grad-accum 4, `temperature=0.8`, `top_p=1.0`, `top_k=-1`, `max_staleness=4`, `token_budget=40960`, `max_completion_length=16384`, `paged_adamw_8bit`, bf16. The article's worker example uses `temperature=1.0` and `max_tokens=4096` (MH:787-794), so treat the code as authoritative for the runs.
- Results:
  - Multi-harness RL went from 42% to 54.2% averaged over harnesses; OpenCode-only RL reached 52.3%. The gap between them is within noise (MH:955-959).
  - Per harness, OpenCode-only won under OpenCode (58 vs 50) and multi-harness won under Claude Code (49 vs 42) and Codex (54 vs 43) (MH:932-937).
  - Tool calls on tasks both models solved fell 31% for multi-harness and 11% for OpenCode-only; OpenCode-only generated up to 62% more tokens under Claude Code (MH:984-1000).
  - Exposure was unequal: 555 vs 626 tasks, and 162M vs 451M training tokens (MH:1004-1014).
- SFT comparison:
  - Teacher Qwen3.8-27B, first correct of up to 3 attempts per harness; 3,189 rollouts. Every assistant turn is an example, with loss on the next assistant response only, tokenized with the *student's* template. lr 3e-6, effective batch 8, 2 epochs (MH:1025-1042).
  - Results: RL 54.6% vs OpenCode SFT 47.5% vs multi-harness SFT 43.1%. Multi-harness SFT dropped Mini-SWE-Agent from 62.1 to 45.2, and the authors have not explained why (MH:1077-1091). SFT cut tool calls by 24% with no efficiency term (MH:1101-1106).
- Qwen failure analysis (MH:1153-1210):
  - the multi-harness run hit its output budget;
  - the OpenCode-only run kept working without finishing;
  - a resume bug replayed old tasks;
  - harder data did not rescue the decline: two thirds of steps had no contrast.

### 2.4 The ultimate guide to RL environments: framework comparison (EG, all chapters + EGsrc matrices)

**The four questions that split frameworks:** what "environment" covers, where it runs, how much trainer comes with it, and when the reward fires (EG:37-42).

**Five-stage spine:** tasks → harness → reward → rollout collector → trainer (EG:105-113).

**Eleven components** (EGsrc `taxonomy-of-environment.html:237-247`): Tasks/Dataset, Initial State Management, Prompt Template, Tool Definition, Observation Format, Execution Backend, State Management, Reward/Rubric, Done/Termination, Episode Control, Transport/Protocol.

**Ten dimensions** (EG:154-439): building, communication & deployment, tool & action model, reward architecture, episode control, tasks & datasets, ecosystem, ease of adding environments, local & cluster setup, scaling.

**Global matrix rows** (EGsrc `global-comparison-matrix.html:238-365`):
- Identity: creator, type, package, Python.
- API surface: tool syntax, observation, reward, done signal.
- Deployment: deployment, Docker needed, scales independently, heavy deps.
- Tooling: CLI scaffold, built-in envs, own trainer.

**Reward architecture (M6-relevant)** (EG:217-255):
1. External: SkyRL Gym, GEM.
2. Server-embedded per tool call: ORS.
3. Post-episode `/verify`: NeMo Gym.
4. Environment-embedded composable `Rubric` (`WeightedSum`, `Sequential`, `Gate`, `LLMJudge`, trajectory-level): OpenEnv, Verifiers.

Rewards are procedural (RLVR) or LLM-judge. Judges invite hacking that a stricter gold judge would penalize; the mitigations are rubric decomposition and "thinking judges" (EG:251). "Dense rewards train faster and more stably; sparse rewards are simpler to define and harder to game", and the embedded Rubric is the natural place for dense rewards (EG:253). In practice the reward is a procedural part plus a judge part, weighted into one scalar (EG:255). "Reward timing matters more than reward content" (EG:510).

**Episode control.** In every tested framework the trainer drives the loop. Harness-controlled rollouts give "benchmark-grade fidelity" but make data collection "harness-specific work each time" (EG:263-279). That work is what the capture proxy of §2.3 removes.

**Tasks.** "A task is more than a row." SETA ships `task.yaml`, data, `tests/test_outputs.py` and a `weights.json` mapping test outcomes to a scalar. SWE-bench grades with FAIL_TO_PASS / PASS_TO_PASS (EG:295-299). Coupled task specs (Verifiers, ORS, NeMo Gym) cost a one-time mapping and make tasks portable (EG:301-316).

**Scaling.**
- In-process environment creation takes under 1 ms (EG:384-390).
- HTTP limits at ≥95% success (EG:403-411): 16,384 concurrent sessions on two SLURM nodes behind Envoy, 2,048 on a local 8-core machine, 128 on a free HF Space.
- "The server is rarely the bottleneck … the execution backend dominates." Horizontal scaling is a load-balancer problem; sticky WebSocket sessions are harder to balance than a stateless session id (EG:429-435).

**Gotchas** (EG:524-531): NeMo Gym returns 422 for unexpected trajectory shapes; Verifiers couples the dataset at init; SkyRL `step()` returns a dict.

**Where Crucible stands on these components and dimensions.**

| Component / dimension | Crucible (built = B, designed = D) | Gap or action |
|---|---|---|
| Tasks | Harbor superset task directory + `crucible/` (DES:163-224) — D/B | Export to verifiers, NeMo Gym and OpenEnv (DES §10.1) — D |
| Initial state | World snapshot/restore `WorldRef` with verified fingerprints (M1) — B | C3 continuation from snapshot — D (M6) |
| Tools / observation | Apps over MCP; single choke point `World._call_tool` (DES P25) — B | Black-box harnesses bypass the choke point; see Open Question 1 |
| Execution backend | local, Harbor (GPU overlay), agentenv (R2E:13-16) — B | — |
| State management | State Adapters (fs, fs_git, sql, postgres, json_doc, process, browser) (R2E:17) — B | — |
| Reward | Separate-image verifier (pattern 3) + multi-key `reward.json` + dense `steps.jsonl` (pattern-4-like) (DES:183-201) — D | Choose the scalar explicitly (MH:704-707); add `call_id` to steps (TL;DR #12) |
| Done / termination | Typed stop conditions in M2 (ARCH:41) — B | Add `abstain`; nudge on empty turns |
| Episode control | Trainer-owned `LLMToolPolicy.run_episode` (M2) — B; harness-owned via Harbor agents (M10) — D | Need the capture proxy to train through black-box harnesses |
| Transport | In-process + MCP stdio/HTTP + backend RPC — B | Sessions keyed by id, not sticky WS, for scale (EG:435) |
| Own trainer | No, by design: export adapters (DES:599-614) | Same choice as OpenEnv |
| Scaling | Not measured | Expect the sandbox, not the server, to dominate (EG:434) |

### 2.5 "Welcome RL Environments to the hub" (HB)
- *Shape.* An environment is a taskset plus a runtime; this release covers tasksets. An environment on the Hub is a dataset repo tagged `rl-environment` plus framework tags; there is no registry (HB:11-13, 23, 137-148).
- *Run commands.* Harbor runs a reference solution with `--agent oracle` to check the task before any model runs (HB:55-69). Verifiers runs `eval harbor` with `--env.agent.harness.id bash` (HB:74-88). OpenEnv runs `openenv harbor rollout --harness opencode`; a `reward: None` means no verifier reward was produced, so check `error` (HB:94-119).
- *Prompt/grader mismatch example.* NeMo Gym's structured-output verifier checks schema adherence, not factual content (HB:127).
- *Shared maintenance.* One repo means one discussion tab for broken tasks across frameworks (HB:129).
- *For Crucible:* an oracle-first run is the R1 smoke test, and `None` versus 0 is C6.

### 2.6 verifiers v1 (VF)
- *Decomposition.* Taskset (data, tools, scoring) × Harness (ReAct loop, Codex, Terminus 2, custom) × Runtime (subprocess, docker, prime, modal) (VF:7-14).
- *Runtime ABC.* `start`, `stop`, `cleanup` (sync, idempotent), `run(argv, env)`, `read(path)`, `write(path, data)` (VF:28-42).
- *Interception server* (VF:50-58). It records traces on the fly, can set sampling parameters a harness does not expose, and "can intercept and rewrite server-side tool responses, e.g. to mitigate reward hacks". It is multiplexed at 32 rollouts per server with an elastic pool.
- *Dialects.* Chat (`/v1/chat/completions`), Responses (`/v1/responses`), Messages (`/v1/messages`). Each dialect class has `routes` plus `parse_request` and `parse_response` into canonical `vf.Messages` (VF:60-80). Clients: `EvalClient` is a blind proxy; `TrainClient` renders client-side and calls a token-in generate endpoint for exact `token_ids` and `logprobs` (VF:82-91).
- *Trace.* A typed Pydantic `Trace` holding a message DAG whose size is linear in turns (v0 was quadratic). Each root-to-leaf branch is one trainable sample, which makes compaction and subagents trainable (VF:95-117).
- *Rewards.* `@vf.reward async def f(self, trace) -> float` over the whole message graph (VF:139-141). Harbor tasksets inherit the Harbor reward (VF:157-174).
- *Supported harnesses:* Codex, Kimi Code, Terminus 2, Mini-SWE-Agent, plus custom ones (VF:178).
- *Eval config:* a TOML naming model, sampling, `taskset.id` and `harness.id@version` (VF:180-195).
- *Training.* `prime-rl` consumes branches directly from the trace. Their example: GLM-4.5-Air in a custom harness on ScaleSWE, 6 H200 nodes, 2 days (VF:196-203).
- *For Crucible:* this matches DES:428-431, which uses `MessageNode.advantages` / `loss_weights` for A_t. Pin the harness version in every record.

### 2.7 On the Nature of the Swarm (SW)
- *Compaction is a bet.* It "raises total context capacity" but has to decide now what will matter later (SW:17-23). A 1M-token working context typically compresses to about 16k tokens (SW:109, fn 3).
- *Disk offload* turns "what to remember" into "what do I need to know exists", but understanding is lost and must be rebuilt (SW:25-31).
- *Sub-agents* are compaction written *after* the question is known, but they forget everything when they return (SW:37-47). *Persistent sub-agents* are parallel over context, so capacity is about one window per agent (SW:51-63).
- *No free lunch.* Communication is lossy and costs tokens (SW:65-73).
- *Trees fail at scale; meshes let specialists be reused* (SW:75-93). Seed the swarm with human tools (a git repo, an issue tracker, Slack) and let the agents restructure it (SW:95-101).
- *Already absorbed as P28* (DES:111). For M5, a handoff note (MY_THOUGHTS-style) is the compaction artifact, and stage state should live in the World, not in a summary.

### 2.8 Preference Model (PMB, KI, KD)
- *Grading.* Grade outcomes, not process: "If the grader can't check it, it cannot go into the prompt" (PMB:13-21). Accept environment awareness (PMB:23-31). Make tasks deep, so that the grader "rewards doing better rather than clearing a fixed bar", and give an honest way to give up (PMB:33-39). All of this is absorbed as P20, P22 and P24 and the record-relative e (DES:417-422).
- *Evals vs training.* In training, a hack found in 10% of runs becomes the policy's default (KI:17; KD:13-19). Prior evidence: METR caught o3 reward hacking in about 30% of RE-Bench runs (KI:29).
- *Karotte mechanisms* (KD:50-111):
  - root-only answers and firewall egress self-test;
  - a persistent student shell with a timeout and output truncation;
  - file tools that act as the student;
  - retries honoring `retry-after`, and a nudge when a turn comes back empty;
  - a model catalog that maps "min" / "max" reasoning effort to each provider;
  - VM limits with 1 GiB reserved for the harness and maximum OOM score for the student;
  - `collect_submission`: kill and confirm, copy to root-only, refuse special files and oversize trees;
  - scrub student-writable paths from `PATH` / `LD_*`; score with `sys.executable`.
- *Red-team pattern* (KD:214-235). A fake-model mode replays scripted attack tool calls; the step must score 0, with `metadata["misbehavior"]` set. This is an R3 negative expressed as a rollout.
- *Framework comparison* (KD:237-255). Harbor 0.23 was not protected against grading tampering (a pre-written `reward.json`, a leftover process, a fake `python3`). Verifiers' staged grading script in `/tmp/vf-scripts/` was overwritten by the agent. Inspect custom scorers were fooled by a replaced `python3`.
- *For harness rollouts:* run every black-box harness rollout behind Crucible's guard and quiesce, and never trust the verifier output of a third-party runner.

### 2.9 Repo2RLEnv (R2E)
**Not found:** the article text itself. The file holds a session-compaction summary from an earlier Crucible session; only its first line paraphrases the article's lessons. Those lessons are: verifier trust first, strip the answer from the prompt, scrub git, pre-bake dependencies and run offline, allowlist rather than denylist, keep gold and hidden tests out of the container, "the environment enforces, the prompt never asks" (R2E:1). They also cover gold = 1 / nop = 0 validation and targeted F2P/P2P tests (R2E:33). All are absorbed (P21, P23, P27). Relevant to M5: per-step F2P mining ports SWE-smith-style logic (DES:342). The original article's text should be re-pasted if exact quotes are needed.

---

## 3. Mechanism table

| Mechanism | Source path:line | Use in Crucible (M5/M6/harness) | DEPEND / PORT / PATTERN |
|---|---|---|---|
| Physical parent ≠ semantic sources (`informed_by`, parent ∈ U_i) | GS:159-179, 3324-3326 | M5: `state_parent` + `informed_by` on trace events; step DAG | PATTERN |
| Commit grammar `[type\|informed_by=…] summary`, unique-prefix SHA resolution | GS:3601-3614, 2040-2041 | M3/M5: parse agent commits into typed steps with edges | PORT (parser) |
| Bounded scope: Gap/Deliverable/Success/Abort + targets | GS:3423-3447 | M5 segmentation unit; C3 handoff; M2 multi-agent prompt | PATTERN |
| `MY_THOUGHTS.md` / `EVALUATION.md` / `BEST_SOLUTION.md` protocol files | GS:3481-3500, 3559-3569 | M5 handoff-note schema; never reward (R6) | PATTERN |
| Candidate-node invariant; validation = publication rules, not correctness | GS:187-189, 3330-3333, 3650-3657 | M5 tracker evaluates Φ only on candidate states; gate vs outcome | PATTERN |
| Abstain action (budget-free, rationale; 94.9% justified) | GS:194, 2908-2922, 3571-3581 | World terminal tool `abstain`; separate stat in C3 | PATTERN |
| Decentralized plurality readout with SHA tie-break; root fallback | GS:212-224, 3335-3338 | Not for reward; optional multi-agent readout baseline | PATTERN (contrast) |
| Central selector +2.34 pp; aggregator −3.87 pp; async ×1.30 faster | GS:3225-3251 | Readout = verifier; async scheduling for multi-agent Worlds | PATTERN |
| Dashboard / in-flight log (scope, updates, uncommitted diff, read-only) | GS:198-210, 3384-3392 | P28 Worlds: a read-only `inflight` App; coordination evidence ≠ provenance | PATTERN |
| Winner transitive ancestry over both edge kinds (excl. root, nominations) | GS:540-549, 2251-2265 | M5 diagnostic; benchmark §11.1 | PORT (metric) |
| Declared-reuse and built-upon rates | GS:495-498, 538-539 | M5 diagnostic for multi-agent traces | PORT (metric) |
| Artifact funnel: created / accessed / actively reused (cited) / winner-linked; logical bundles | GS:2313-2317, 2333-2340, 2485-2490 | M4/M5: consumption evidence from tool traces; credit audit | PORT (metric) |
| Activity taxonomy from diff vs physical parent only | GS:2563-2601 | M5 stage-label seed; κ ≥ 0.7 protocol | PATTERN |
| Workflow archetypes (late integration, divide-and-conquer, search-and-pivot) | GS:2516-2545 | M5 family-graph descriptors | PATTERN |
| Matched-resumption causal test | GS:744-755 | C3 contribution ablation ΔV̂(c) | PATTERN |
| Retraction propagates through ancestry | GS:1594-1600 | C2/C4: invalidate derived milestones, rubric items, labels via provenance | PATTERN |
| Repeated-move fraction; relevant-record coverage; archive-read fraction | GS:1436-1438, 1558-1560, 1623-1625 | Benchmark redundancy and memory metrics for multi-agent Worlds | PORT (metric) |
| Harness validates restored checkpoints; workers can't fabricate states | GS:1536-1540 | C3: verify snapshot fingerprint before continuations (already in M1) | PATTERN |
| Eval funnel 128 → 512 → full; misranking 5/12; seed noise ≈ step size | GS:427-431, 1946-1954, 2059-2072 | §10.2 tier agreement measured as pairwise rank agreement; noise bands | PATTERN |
| Exact-source replication; ten-repeat noise baseline | GS:2150-2179 | M5 replay: noise band per recorded metric; P15 | PATTERN |
| Allowed-path publication validation | GS:2213-2217 | Gate: writes outside allowed paths fail | PATTERN |
| Search-collapse / over-concentration watch | GS:2073-2076, 3735-3739 | Training monitor: milestone diversity across a group | PATTERN |
| IMO judge 0/1/6/7 with 3 passes, min–max | GS:3708-3717, 1719-1728 | M6 LLM-judge baseline protocol | PATTERN |
| Forced continuation (`Keep working...`) with checkpoint grading | GS:1320-1323 | Calibration anytime curve; natural-stop boundaries | PATTERN |
| Token accounting (input = full prompt per request; output incl. reasoning) | GS:287-294 | Benchmark cost metric definition | PATTERN |
| Gemini sequential-shell fix (`set -e`, skip remaining blocks, reject pre-fork commands) | GS:3746-3800 | M10 harness adapter for text-parsing harnesses | PATTERN |
| Capture proxy: session id = key, dialect detection, non-streaming engine | MH:480-508 | M3 `capture_proxy` upgrade; M12 | DEPEND (OpenEnv) or PORT |
| Token capture: ids + logprobs; `top_p=1`; vLLM flags; engine probe; capture levels | MH:512-532 | M12 training path; mask non-token rollouts | DEPEND |
| Rollout graph by longest exact token prefix; per-branch sequences; loss mask | MH:541-559 | Map `steps.jsonl` → call nodes → branches | PORT |
| ATIF cross-check (turns, completion tokens/call); drop empty tools arrays | MH:712-725 | M3/M10 capture validation | PORT |
| `ok=False` / `reward=None`; no exceptions across the trainer boundary | MH:635-639, 704-711 | C6 mask path | PATTERN |
| Never combine verifier scores implicitly; +0.2 submit collapse | MH:704-711 | M6 compose: explicit scalar + R2 | PATTERN |
| Efficiency bonus `r=c·(1+0.1·15/(15+calls))`; None passthrough | MHcode train/multi_harness.py:78-85; MH:878-900 | e-channel candidate (tool-call efficiency) inside O | PORT |
| Zero-contrast audit (mixed / all_wrong / all_correct_contrast / no_contrast) | MHsrc assets/data/reward-group-audit.json; MH:912-916 | M6 training diagnostics | PORT |
| Harness seams table (env vars / config / host args) | MHsrc embeds/harness-seams.html:40-49 | M10 harness launch adapters | DEPEND (Harbor + OpenEnv) |
| One harness per GRPO group; async staleness ≤ 4 | MH:805-807, 842-845 | M12 recipe | PATTERN |
| Output-cap parity between train and eval; Qwen failure modes | MH:1163-1176, 1205-1210 | Task card fields; collapse monitors | PATTERN |
| SFT: per assistant turn, student template, lr 3e-6, bs 8, 2 epochs | MH:1025-1042 | §10.3 SFT exports | PATTERN |
| Three overfitting modes (format, context, control flow) | MH:305-312; MHsrc embeds/d3-overfit-modes.html | M10 OOD harness splits (§11.2) | PATTERN |
| verifiers v1 Runtime / Dialect / Trace message-graph / branches | VF:28-117 | Export target; A_t via MessageNode | DEPEND |
| Interception rewrites tool responses (anti-hack) | VF:56 | Choke point for black-box harnesses (Open Question 1) | PATTERN |
| Reward architecture patterns 1–4; dense-in-Rubric | EG:217-255 | Crucible = pattern 3 + dense channel | PATTERN |
| Scaling: sandbox dominates; avoid sticky sessions | EG:429-435 | M10/M12 infra | PATTERN |
| Karotte fake-model attack replay with `misbehavior` metadata | KD:214-235 | R3 negatives as scripted rollouts | PATTERN (MIT, ported in M1) |
| Empty-turn nudge; retry-after backoff; effort catalog | KD:73-83 | `LLMToolPolicy` and C3 continuation loop | PORT |
| Compaction as a bet; persistent sub-agents; mesh | SW:17-93 | C3 fresh-context mode; P28 | PATTERN |

---

## 4. Open questions and risks

1. **P25's single choke point does not exist inside a black-box harness.** Claude Code, Codex and OpenCode execute tools inside the sandbox. Crucible sees only the model calls (MH:137-151), unless the harness's tools are replaced by Crucible MCP Apps (as GitSwarm did with `pb.container_bash`, GS:3804-3821). Options:
   - (a) route harness tools through a Crucible MCP server wherever the harness allows it (Codex supports MCP);
   - (b) for the rest, rely on environment enforcement (P21) and state-only grading (R6);
   - (c) rewrite tool results at the proxy (VF:56). This changes what the harness sees, so it must be logged as an intervention.

   Decide per harness and record it in the harness capability contract.

2. **What do `informed_by` edges mean for credit?** They are declared, not causal (GS:175-179). If Crucible ever propagates reward along them (credit to the step whose artifact a later success used), a policy can farm credit by citing everything, or by being cited. The only safe propagation is counterfactual (C3 ablation). The cost is K × (number of contributions) rollouts. Mitigations are to sample contributions or to ablate only edges on the winner's ancestry.

3. **How should abstention be rewarded?** At 0 it equals failure, and GRPO then gives no incentive to abstain correctly. With any positive value it becomes a give-up hack: compare the +0.2 submit collapse (MH:704-711). Proposal: abstention scores 0, is excluded from the R2 no-op test, and only on tasks marked possibly-unsolvable (PMB:39) does the verifier grade the abstention's *justification* against state. This is unresolved.

4. **Fast and full tier misranking.** GitSwarm's numbers (5/12 pairs, 15/50 pairs) are from loss evaluations. For GPU migration, nobody has measured the misranking rate of the fast tier's perf round. The M6 gate needs the A12 agreement criterion (DES:623) restated as a pairwise rank-agreement threshold, for example ≥ 90% on oracle versus negatives and on policy pairs.

5. **Milestone effects may not add.** "Gains measured in isolation often vanished when combined with the leader" (GS:1951-1952). A linear Φ with fixed weights (DES:380) can then over-credit milestones that only matter on weak states. To check: fit weights by time-split AUC on leader-conditioned states (§6.5) and compare with the unconditioned fit.

6. **Replay noise.** Exact-source re-evaluations moved scores by more than the claimed gains (GS:2150-2157, 2215-2217). The M5 replay gate ("reproduce recorded CI outcomes") needs a per-metric tolerance band (P15). Without one, perf-milestone replays will fail spuriously.

7. **Token-faithful versus text reconstruction.** OpenForgeRL trains from text and still leads (MH:234-237). Crucible's "mask rollouts missing token records" (DES:626) discards all hosted-API harness runs for training. That is acceptable for RL, but SFT and PRM data (C5) can use text-level traces. Keep the two data paths separate.

8. **Harness data imbalance.** Claude Code's history rewrites gave it 77% of rows (MH:1196-1197). With dense per-turn advantages, harnesses that compact more get more gradient. Normalizing per rollout (TL;DR #12) is a guess, not something the article tested.

9. **Multi-harness SFT damaged mini-swe-agent** (62.1 → 45.2, unexplained; MH:1086-1091). Lifted expert traces (DES:629) must be checked against every target harness, not pooled blindly.

10. **Proxy scale.** One proxy process saturates at about 200 sessions (MH:582-583). C3 with K continuations × steps × tasks will exceed that. Use the multiplexed pool pattern (32 per server, elastic; VF:58).

11. **Source gaps.** The Repo2RLEnv article text is absent (§2.9). The HTML embeds hold data that the extracted `.md` files lack (the scaffold-reversal numbers, the per-checkpoint results JSON). Only `harness-ecosystem.json`, `reward-group-audit.json`, the harness-seams table and the guide's matrices were read for this report.
