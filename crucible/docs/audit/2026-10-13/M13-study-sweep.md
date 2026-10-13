# M13 study: every local source, read for benchmarking, a campaign over the local environments, and the picks

Two source sweeps, two defect hunts and one benchmark campaign audited M13 `bench` (≈1.2k lines; 4 tests before):

| Group | What | Document |
|---|---|---|
| G1 | how every local framework evaluates (Harbor, verifiers, NeMo-Gym, OpenEnv, agentenv, rLLM, SkyRL, verl, NeMo-RL, TRL, AReaL, slime, mini-swe-agent, SWE-smith): estimators ported verbatim and compared on identical inputs; CI coverage by simulation | `M13-sweep-G1.md` |
| G2 | the articles and the dataset cards (canaries, exclusion lists, splits, difficulty), contamination tools and IRT on the real local data | `M13-sweep-G2.md` |
| H1 | metrics, compare, report, run (fuzzed against 1,000 synthetic M10 run dirs written by the real writers) | `M13-study-defects-H1.md` |
| H2 | splits, IRT, contamination, dynamic items | `M13-study-defects-H2.md` |
| campaign | the benchmark run end to end over 8 local sources (TB2.1, Terminal-Lego ×2, termgrade in Docker; NeMo-Gym workplace and calendar; EnterpriseOps-Gym; OpenEnv PortSim; the toy pack) with oracle, no-op, adversarial and Qwen2.5-0.5B policies, n = 3: ≈1,750 episodes | `M13-bench-campaign.md` |

## What held

- Per-task estimators (pass@k, pass^k, max@k) and exact McNemar match every reference to ≈5e-10; Crucible's max@k is
  better than verl's best@k (resampling with replacement) and NeMo-Gym's continuous pass@k (`max(vals[:k])`).
- "Masked samples are never samples" is the strictest and right rule (NeMo-Gym default, verifiers, prime-rl effective
  subset, OpenEnv); Harbor, rLLM, SWE-smith, EnterpriseOps, agentenv count infra errors as 0.
- Recomputing every campaign metric from raw `results.jsonl` gave 0 diffs over 16 runs; reports are byte-identical on
  re-render; NeMo server deaths mask; four of five adversary families scored 0.

## What the groups found (most certain first)

- **A hack outranks an honest model** two ways: exposed-answer episodes are masked, so their failures vanish (G2-1:
  pass@1 1.0 vs 0.5), and in Harbor in-place mode a policy that stubs the grader's own tools scores 1.0 on every Harbor
  set with no flag (campaign B1).
- **Infra faults scored 0** in Harbor in place (campaign B2, 9 of 9).
- **`compare` mixes policies** (G1 D1, H1 V7, campaign B3) and **McNemar treats a task's k samples as independent**
  (G2-8, campaign B11: p = 1.2e-7 where a per-task test gives 0.0078).
- **Each k uses its own task set; the headline is an episode mean; percentile bootstrap undercovers** (G1 D2/D3/D5, H1
  V8/V9, G2-6/7, campaign B13): 12.6% coverage in one realistic case.
- **Spend of masked retries never reaches cost; stopped runs read as the whole benchmark** (G1 D4, H1 V4/V5, campaign
  B6).
- **Progress metrics read rows in the wrong order and ignore the start state** (H1 V1–V3).
- **No validity floor**: a no-op passes 13.4% of workplace rows and a broken-oracle task counts as a hard item
  (campaign B4/B5).
- **Splits leak lineage; dynamic item ids derive from seeds under public keys; IRT does not converge, shrinks new models
  toward 0 with 26–79% CI coverage, and drift quarantines clean anchors 17–89% of the time** (H2 V1–V7, G2-2/3).
- **Contamination tools miss the real canaries in the local data, flag template boilerplate (12–18% precision), and
  are wired nowhere** (G2, H2 V13/V15/V16).

## Picks

| # | Pick | Source | Kind, effort |
|---|---|---|---|
| 1 | Exposed / tampered episodes scored 0 with a hack flag in bench views; models compared on common support | own (G2-1) | S |
| 2 | Grader toolchain manifest checked at grade time; tamper scored 0 and flagged | own (campaign B1) | M |
| 3 | Harbor in-place infra faults mask by evidence | M11 rule (campaign B2) | S |
| 4 | One task set for all k; task-mean headline; t / cluster CI with n_tasks; per-task paired tests | Harbor `pass_at_k.py`, NeMo-Gym `reward_profile.py`, `comparison/diff.py` | PATTERN, S |
| 5 | Validity manifest (oracle, no-op floor, flakiness) gating headline, IRT and compare | M9 A1/A2/A4; M11 I1–I3 | PATTERN, M |
| 6 | Lineage closure in splits; full-label seal; secret-keyed, seed-free dynamic ids | own (H2) | S |
| 7 | IRT: line-searched MAP, anchor-centred priors, joint SE, robust drift with multiplicity control | own (H2 V5/V6, G2-2/3) | M |
| 8 | Canary registry, template subtraction, containment score, fixed leak regex; wire `scan` into the curator | own (G2 d) | M |
| 9 | Spend from every attempt; planned vs run; masked-by-kind breakdown; exports Harbor / NeMo-Gym / verifiers can read | Harbor JobStats, NeMo-Gym `_aggregate_metrics.json` | PATTERN, M |
| 10 | Row → runnable task adapters for the SWE row sets and Reverse-Text (I0–I10) | M11 G5 §4.2 | M–L, queued |
