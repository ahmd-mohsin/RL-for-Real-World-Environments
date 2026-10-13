# M15 audit: the studio (2026-10-13)

Scope: `crucible label` and the dataset builders (`studio/label.py`, `studio/datasets.py`), `crucible build`
(`studio/build.py`), exports (`studio/export.py`), the explorer plugin (`studio/explorer.py`), run identity
(`studio/runs.py`, new), the CLI (`cli/*`), the runner's ATIF writer (tool cards), and the public site (`site/`). Base
`076ab65` (M14).

Study: two sweeps (G1: how TRL, prime-rl, verl, NeMo-RL, OpenRLHF, verifiers, rLLM and SkyRL consume SFT, preference and KTO
data, with their preprocessing code run on Crucible's outputs and the real Qwen2.5, Qwen3 and gemma-2 templates; G2: CLI,
pipeline, run-directory and viewer conventions in agentenv, Harbor, verifiers, NeMo-Gym, rLLM, SkyRL, OpenEnv, FineEnvs,
repo2rlenv, SWE-smith, with the README path run end to end and the explorer mounted in agentenv's app), two hunts (H1
label and datasets; H2 build, exports, explorer, CLI, site) and a campaign on the fixed tree (`M15-campaign.md`).
Regression tests: `tests/unit/test_m15_audit_label.py` (33 functions) and `test_m15_audit_studio.py` (36); 29 of the label
tests and all 36 studio tests fail on `076ab65`, the rest cover defects the campaign found in code written during this
audit.

## 1. Requirement ledger

| # | Requirement (source) | Before | After |
|---|---|---|---|
| L1 | A dataset row is what the policy produced, as it saw it (G1, H1 V3–V5, V18, D3, D4, D10) | rows rebuilt from ATIF and the bench reward: RLM episodes rewritten into chats the model never produced, the pack path invented tool calls from recorder fields (108k tokens on one coding row), DPO / KTO completions held the rest of the episode (61% of the tokens TRL would train were World output), JSON-string arguments double-encoded by Qwen's template | the runner's training view (`rollout.json`); one policy turn per completion; real tool names and dict arguments; system prompt, tool cards (now written by the runner into ATIF) and `reasoning_content` kept; trained tokens checked equal to the policy's with TRL's own preprocessing |
| L2 | Positives are the model's own clean successes (H1 V1, V2, V7, G1 D2, D5–D7, campaign B19) | oracle, no-op and adversary episodes were SFT positives and DPO chosen (every toy pair was oracle vs no-op); flagged, exposed and invalid-task episodes kept; SFT and KTO used contradictory rules | controls, scripted and adversary policies excluded (scripted demos only with `--allow-scripted`, in a separate file); flagged, masked, invalid and training-view-less episodes never positives; one rule (outcome and reward agree, reward ≥ threshold) for SFT, DPO and KTO |
| L3 | Held-out data never becomes training data (G1 D1, H1 V9) | build's `bench` split and the toy credit hook's test tickets were labelled | label reads the sealed `splits.json` and the packs' splits; bench items excluded unless `--include-bench` |
| L4 | Preference data states a real preference (H1 V10–V13, campaign B20–B22) | best-vs-worst without requiring a fail, at any differing turn; pivots across state shifts; the same turn labelled both ways in KTO | chosen passes, rejected fails, margin > 0, identical prompt and tools, evidence at the cut (graded wrong or higher dense credit); pivots only at one state; conflicting KTO turns dropped |
| L5 | Every consumer loads and trains it correctly (G1 D8, D9, D11, D12) | no SFT file loaded with HF `datasets`; KTO labels were strings; TRL and verl ignored `message_loss_mask`; gemma-2 rejected every row | numeric fields, bool labels; one row per kept turn (the `messages` + mask format is an option for prime-rl); `--template` adapts strict templates; all 43 campaign files load |
| L6 | Build and export fail loudly (H2 V1, V4–V8, V10, V12, V20, G2 D7, D8) | an oracle crash still benched the task and exited 0; an all-rejected forge or empty ingest exited 0; train-split items were benched; no contamination step | non-zero exits by stage (with explicit allow flags), staged rebuilds, the range's best end as the oracle bar, only the bench split benched, the seal verified, contamination in build, a non-empty bench split sealed reproducibly |
| L7 | Exports never leak or destroy (H2 V2, V3, V9, G2 D1) | exports copied the pack's `.env` and raw logs; an `--out` that contained the pack deleted it; task ids could escape the export dir | an allowlist bundle with a post-copy secret scan, a marker-checked atomic swap, refused overlaps, sanitized and collision-free task dirs |
| L8 | The explorer stays inside its root and runs no foreign script (H2 V13, V14, G2 D2–D5) | a symlinked report served a file outside the root; a planted report ran script on the explorer origin (it could start task runs); pack checks ran in the server process; duplicate run ids shadowed each other | every read resolved and contained; reports under a sandbox CSP; pack checks by known name in a subprocess; unique relative run ids |
| L9 | Runs are reproducible identities (G2 D6, D12, H2 V17) | a rerun with changed flags rewrote outputs before refusing; only two commands wrote a resolved config; nothing recorded versions | `runs.record` before any write (exit 4 on a changed rerun), every command's resolved config with provenance (crucible version, git sha, python, pack content hash) |
| L10 | The CLI tells the truth (H2 V15, V16, V18, V19, G2 D9, D10) | `--json` printed nothing on error; one exit code for unknown pack and failed check; an invalid bench key was silently replaced; hidden answers on argv | JSON error objects, exit codes 0–4, key files 0600 and invalid keys refused, `--forbidden` from a file, unsupported train specs refused |
| L11 | Public material follows the user's rules (H2 V22–V25) | allusions to other work ("the standard tool protocol", "group-relative advantages"), module-code anchors (`#m4`), "we", colons in labels, overlong sentences, three different attack counts | rewritten in the formal, impersonal register; plain-word anchors; a test scans every visible string |

## 2. Corrections to earlier claims

- `docs/modules/M15-studio.md`: the toy gate reads "5 provisional" (A10 pending), not "5 accepted"; only `label` and
  `build` wrote resolved configs.
- `README.md`: build runs the full M9 contract, not "model-free checks".
- The H2 auditor read the formal register of the site as a rule violation; the user asked for a formal, impersonal
  register together with ASD STE100, so the register stays.

## 3. Gates and numbers

- Campaign on the fixed tree: toy build exit 0 and byte-identical across reruns; 9,883 label rows checked against the
  rollouts with no masked, flagged, adversary, control, invalid or held-out episode; SFT and DPO trainings of
  Qwen2.5-0.5B on MPS with exactly zero gradient outside the policy's tokens (SFT loss 0.396 → 0.185 on 32 fixed rows;
  DPO 0.693 → 1e-5). The SFT model did not improve the toy benchmark at this scale (0.0 vs 0.0), reported as is.
- Full suite, lint and import contracts: in the commit message.

## 4. Open

- `build` refuses packs without `hooks.forge` (finance_ap, coding, a scaffold) and its bench runs no declared adversaries
  (campaign B10); `bench run` covers them.
- Toy builds bench one to three items, and a fresh key can seal a bench whose items all match the training corpus, which
  build then refuses; the test fixture uses a fixed key.
- Redaction can rewrite a policy turn's tokens (an e-mail in a trained target, campaign B24); `--mask-negative` is a no-op on
  run rows (B14); `--threshold` outside the range is accepted (B13); timing keys in the canonical export (B17); DPO keeps one
  worst negative per task (B22); scripted-demo rows lack the system prompt (B23).
- Pack rows group by trace family for `--max-per-task`; no length policy for overlong rows; contamination compares
  instruction text only.
