# M5/M6 research, slice 03b: reward design across prime-environments

Source: `RL-Environments/06-frameworks-and-env-code/primeintellect-ai-prime-environments/environments/` (paths below are
relative to it). Read-only. The `verifiers` library internals (`JudgeRubric.judge()` formatting, `RubricGroup`
aggregation, `ToolRubric` metrics) are not in that repo; anything depending on them is marked *inferred*.

## Takeaways for Crucible (M6)
1. **Ungated shaped terms dominate correctness in many envs — the failure P13/§6.4 gating exists to prevent.**
   rust_cargo: 5 of 7 weight points need no correctness and `cargo test` runs only the model's own tests
   (`rust_cargo/rust_cargo.py:76-130`); uq_project rewards discourse-marker counts and length (`:326-393`, reasoning 0.45
   > factual judge 0.4); gptworld `min_moves/moves` is uncapped and ungated (`gptworld.py:90-94`); agent_dojo pays 0.5 for
   doing nothing (`:155-158`); ktor/art_e pay tool-economy bonuses on wrong answers. **Crucible:** every shaped/efficiency
   term sits inside O or behind gates; R2 (no-op = 0) catches agent_dojo-style payouts.
2. **Efficiency gated on success works** (wiki_race `:178-223`, nyt_connections `:191-229`, datasets_codebase_search
   `:177-224` via `info["correct"]`). Matches DESIGN β·e inside O.
3. **Success read from text the policy can write** (gpu_puzzles `"SUCCESS"`, gptworld stdout brackets, hanabi JSON from any
   message, humaneval/scicode exit 0 with model code before the asserts). Fixes seen in-repo: random sentinel printed
   after tests (autodiff `secrets.token_hex` `:136-139`, transformerpuzzles, llm_training_puzzles), host-side comparison
   (polars_env `assert_frame_equal` `:218-271`, scorer errors kept separate in `scoring_error`), action replay (fruit_box
   `:410-439` recomputes the score by replaying actions on a fresh game). Confirms P26.
4. **"Compiles" ≠ "proved":** minif2f has no `sorry` filter (`:198-227`); stepfun_prover rejects sorries but not which
   theorem was proved (`containerized_lean_state.py:113-145`) and shares sessions keyed by `sha1(theorem)` across rollouts.
5. **Judge parsing that fails open:** `"yes" in text` (patents_ar_env:220, congressional_records:383, gutenberg_env:227,
   search_r1_ish:185, github_mcp:165); wmdp first `(A|B)` matches the "A" in "Answer"; browsecomp_plus first-number
   fallback turns a year into 1.0. **Fail-closed patterns to copy:** anchored regex with 0 default (datasets_cb
   `correct:\s*(yes|no)` `:188-202`; torch_ao `\b(0\.\d+|1\.0|0|1)\b` `:244-258`), structured output (art_e
   `{thinking, accept}` via `response_format`), constrained XML verdict with stop token (nextjs
   `<evaluation>correct|partially_correct|incorrect</evaluation>`, stop `</evaluation>`, temp 0, 256 tokens → 1/0.7/0),
   explicit raise (vllm_code_search `:274-286`), max deduction on parse failure (agency_bench).
6. **Rubric structures worth reusing for C4 / the judge baseline:**
   - per-item weighted boolean criteria → Σ weight·met clipped [0,1] (misguided_attn `:64-175`);
   - deduction codes `{"A":10,"B":4,...}`, score `max(10−Σ,0)/10`, parse failure = max deduction (agency_bench);
   - anchored checklist 1.0/0.75/0.5/0.25/0.0 "return only a single decimal number" (torch_ao `:304-338`, ktor `:65-83`);
   - multi-judge panel, Hölder mean p=0.5, behind a hard eligibility gate (creative_writing `:89-271`);
   - chunked transcript judging (spiral_bench `utils.py:163-276`).
7. **Judge wiring bugs that silently blind the judge** (bixbench prompt has no placeholders `:440-447`; misguided_attn reads
   `kwargs["question"]` `:79`; uq_project reads `title`/`body` kwargs; writing_bench criteria misalign after filtering;
   spiral_bench `final_judgment` vs `final_judgement`). **Crucible:** every judge item ships with a known-good and a
   known-bad fixture that must score 1 and 0 (unit test), and its prompt is rendered and snapshot-tested.
8. **Cross-rollout state leakage:** arc_agi_tool keeps `arc_reward`/`episode_done` on the env instance (`:437-447, 612,
   808`), backend_bench `self.feedback_info`, stepfun_prover shared sessions. **Crucible:** per-episode state only in
   World/WorldRef; a property test runs two concurrent episodes and checks independence.
9. **Sign/scale errors:** browsecomp_plus adds calibration *error* with positive weight (`:37-45, 131-142`); hanabi 0–25,
   spiral_bench 0–100, writing_bench 1–10, fh_aviary unbounded sums. **Crucible:** every channel normalized to [0,1]
   with a declared direction (RewardJson validation).
10. **Shaped signals that work:** skyrl_sql {−1 format, 0 wrong, +1 right} with strict format; art_e abstain 0 vs
    hallucinate −1 (bears on the abstain open question); scicode sub-step `pass_rate` kept at weight 0 as a metric next to
    a binary all-pass reward; search_r1_ish small (0.1) retrieval reward for the gold string in tool output; vrcli
    perplexity-gain reward (buckets too generous: 0.1% gain → 0.9).

## Other facts
- **Multi-turn state:** fh_aviary sums step rewards (README wrongly says 0–1); fruit_box per-move reward = cells cleared
  but final reward re-derived by replay; backend_bench overwrites per turn (last counts) and hard-codes
  `have_perf_test=False` (`:376-379`); scicode stores code per sub-step; tau_bench records `actions_taken`, scores only at
  episode end.
- **No pass-rate or curriculum selection anywhere** (only static difficulty filters: enigmata, longbench_v2, gptworld,
  dabstep, med_agent_bench).
- **Code grading:** only mini_swe_agent_bench uses F2P/P2P; aider_polyglot runs the full test command on exit 0 and lets the
  model overwrite test files (guarded only by prompt text, `:21`); mbpp `exec`s in the grader process without sandbox
  or timeout; humaneval maps infra errors to exit 124 and raises rather than scoring 0 (good).
- **Repo guidance** (`AGENTS.md:99-105`): one `Rubric` with `funcs=[...]`, `weights`, "0-weighted reward functions for
  purely informational metrics"; validate with `vf-eval -s -n 5 -r 3`; PR template only asks for "rewards greater than 0
  at least some of the time". No guidance on gating, judge robustness or hackability.
