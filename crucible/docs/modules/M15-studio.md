# M15 `studio`: `crucible label`, `crucible build`, packs, and the CLI

**Study:** `docs/research/bench-studio/01-framework-study.md` §3.

**Code:**
- `src/crucible/studio/{pack,label,build}.py`
- `src/crucible/studio/datasets.py`
- `src/crucible/core/schema/pack.py`
- `src/crucible/cli/{main,studio}.py`
- `packs/_template/`
- `packs/_toy/pack.yaml` + `crucible_pack_toy/studio.py`

## CLI

click, as Crucible and agentenv already use it. Every command does one module's job. Every command that writes
outputs (`--out` / `--work`) writes `configs/resolved/<cmd>.json` first (`studio/runs.py`: scrubbed of credentials,
stamped with the crucible version, git sha and dirty flag, Python, agentenv and the pack's name, version and content
hash). A re-run into the same `--out` with different identity flags is refused before anything is written ("use a new
--out"; the verifiers / Repo2RLEnv resume receipt). `build` and the export manifests carry the same provenance.
- **Lazy subcommands:** a command table imported on use (rLLM `_LazyGroup`, NeMo-Gym `COMMANDS`).
- **Exit codes:** 0 ok; 1 the command ran and something failed (a check, a rejected task, an export error, nothing
  exported); 2 click usage (and `pack check`'s skipped level); 3 internal error (traceback printed); 4 user error.
- **User errors:** one line; `--verbose` shows the traceback (agentenv). An exception from crucible's own code
  (`KeyError`, `TypeError`, a JSON decode error, ...) is an internal error, never a one-line user error.
- **Output:** `--json` on reporting commands (on failure a JSON `{"error": ...}` object on stdout), `--dry-run` on
  `label` and `build`, and the next commands printed with paths (`./x` or absolute), never bare names.

**Commands:**
- `label`, `build`;
- `pack new|check`;
- `bench run|report|compare`;
- `ingest`, `signals`, `forge`, `accept`, `train`;
- plus `registry`, `task`, `seeds`, `world`.

## `crucible label`

**Audit (2026-10-13; `docs/audit/2026-10-13/M15-studio.md`).** A row is what the policy produced, as it saw it:
- **The run path reads the runner's training view** (`rollout.json`: trainable turns, kinds, segments, `mask_sample`,
  flags), plus `validity.json` and `splits.json` (sealed). It no longer rebuilds rows from ATIF and the bench reward, which
  had made every DPO pair in the toy build oracle-vs-no-op and kept RLM, flagged and held-out episodes.
- **Positives** come only from model-produced episodes: controls, scripted, oracle, no-op and adversary policies need
  `--allow-scripted` and go to a separate `<name>.chat.jsonl`. Flagged (`hack/*`, `answer_exposed`, `grader_tampered`),
  masked, invalid and bench-split episodes are never positives; one rule (outcome and reward agree, reward ≥ the pack
  threshold) serves SFT, DPO and KTO; skipped episodes are counted per reason.
- **One policy turn per completion.** SFT defaults to one row per kept assistant turn (prompt = everything before it), so
  TRL, verl and prime-rl all train exactly the policy's tokens (the multi-turn `messages` + `message_loss_mask` format
  stays as `--sft-format messages` for prime-rl); DPO chosen / rejected and the KTO completion hold only the policy's
  turn, with identical prompts and tools; KTO `label` is a bool.
- **Formats consumers accept:** dict tool arguments, the World's tool cards in `tools` (ATIF `agent.tool_definitions`,
  now written by the runner), `reasoning_content` kept, no consecutive same-role turns (`--template` adapts for strict
  templates such as gemma-2), numeric weights so every file loads with HF `datasets`.
- **Runs are identities:** `studio/runs.py` records the resolved config with provenance before any write and refuses a
  rerun with different output-changing flags (exit 4, "use a new --out").

**From a pack:**
1. M3 ingest of `pack.yaml` sources (idempotent, redacted).
2. M4 signals and pivots.
3. The pack's `credit` hook: M5 milestones mined from good dev logs; M6 Φ per event, a gated reward, and the outcome.
4. **The validity gate** (`credit.channels`, 2026-10-15, M5 R1): Φ is used two ways, and each is a channel with its own
   verdict on the hook's held-out traces (split ≠ `train`). The **step** channel (per-turn weights) needs Φ at the
   midpoint event to predict the outcome; the **terminal** channel (partial credit κ·(1−O)·Φ) needs Φ at the end state
   to. A channel is released when the held-out AUC is ≥ 0.6 and its 95 % lower bound is > 0.5. An unreleased step
   channel writes weight 0.0 with `credit: false`; an unreleased terminal channel pays the reward recomposed with Φ = 0
   (`reward_with_progress` keeps the hook's value). Verdicts are in `summary.json` `credit_channels`, in each
   `rewards.jsonl` row (`channels`) and in the CLI output. Toy: both unreleased (step AUC 0.50; terminal 0.67 with lower
   bound 0.50), same rows as before, failed tickets T1011 / T1019 no longer earn 0.133.
5. `credit.datasets`.

**From a run:** M10 episodes; masked episodes are skipped.

**Formats:** TRL-canonical JSONL.
- **SFT:** `messages` + `tools` + prime-rl `message_loss_mask` + per-turn Φ `weights` + provenance (task, run, episode,
  model, reward, resolved, `traj_id` content hash).
- **DPO:** `prompt` / `chosen` / `rejected`. Pairs come from **best vs worst of k** (same prompt; chosen passes; margin),
  or from **correction pivots** (a policy action graded wrong and then fixed by the policy).
- **KTO:** `prompt` / `completion` / `label`.

Curation uses a closed filter grammar over per-task stats (`avg best worst n n_correct solved`), with `--select
correct|best|all` and `--max-per-task`.

## `crucible build`

0. Preflight: the pack's `requires` and `reward.higher_is_better`, as `bench run` (`--allow`); the run identity
   (`configs/resolved/build.json`) is compared before anything is written.
1. M3 ingest.
2. The pack's `forge` hook (toy: world_synth with scripted roles, no paid calls).
3. The full M9 contract per task: accepted, provisional (nothing failed; checks that need Docker or paid calls are
   pending without `--docker` / `--paid`) or rejected.
4. M11 canonical export: oracle replayed into `reward.json` + `steps.jsonl`, the outputs every runtime adapter must
   reproduce. A World task is not a Harbor-native directory; Harbor reaches it through ATIF replay. An anchor whose
   oracle did not replay to the top of `reward.range` is invalid in the benchmark.
5. M13 splits (lineage-closed, sealed; when the fractional draw leaves the bench empty, the lineage ranked lowest under
   the key is sealed to bench, so a re-run with the same key reproduces the seal) and the contamination scan of the bench split against the training corpus (the
   train split and the ingested traces; the template is learnt over the whole benchmark, so a family preamble is not a
   hit, and an exact copy always is); a contaminated bench item is invalid.
6. An M13 benchmark of the bench split only (anchors and fresh generated items), with scripted policies; the seal rides
   in the run's report and is verified.

No traces, no forged task or no accepted task stops the build at that stage (exit 1, stage named; `--allow-empty` goes
on to a fresh-items-only bench). Rejected tasks, oracle replay errors or owed samples exit 1 after the bench
(`--allow-rejected`); so does a bench split with no valid item. Outputs are written to a staging dir and swapped in only when the build completes, so a failed
rebuild leaves the previous build whole.

## Exports and the explorer

- **`export`:** the NeMo-Gym / OpenEnv bundle is an allowlist (`pack.yaml`, README, LICENSE, `pyproject.toml`,
  `src/<package>`, `data` / `worlds` / `families`) without dotfiles, env and key files, ingest-source files or the
  `redaction_env`, scanned for credentials (a hit refuses the export); an `out/pack` it did not write, or the source pack
  itself, is never replaced. Async `hooks.tasks` are awaited (also in the launchers). OpenEnv's manifest names an ASGI
  app (`openenv_env:app`, OpenEnv's `create_app`, port 8000). Task ids become safe, distinct directory names. ATIF
  export refuses a dir without trajectories or a missing validator (`--no-validate`), and writes a `.result.json`
  (task, policy, reward, run identity) beside each trajectory.
- **Explorer plugin:** pack checks by index name only, in a subprocess; run ids are paths under the data root; every
  file read is resolved and contained; reports are served under `Content-Security-Policy: sandbox` (rendered in memory
  when absent; a GET never writes).

## Packs (ARCHITECTURE M14)

- **`pack.yaml`:** a pydantic `PackManifest` with a cross-field contract. Hooks are `module:attr`: `credit`, `tasks`,
  `generator`, `stages`, `forge`, `policies`.
- **`crucible pack new`:** NeMo-Gym's conflict-safe writer; identical files are kept, any conflict writes nothing, and
  symlinks are refused.
- **`crucible pack check`:**
  - static: manifest, placeholders, license, layout, README, entry point, hooks, ingest globs;
  - semantic: model-free, gold reaches the top of `reward.range` and the untouched task stays below it;
  - severity policy v1, and exit code 1 on failure.
- **The template** is a real pack (a seeded workspace task) that passes both levels with no Docker or model.

## Gate

`tests/unit/test_studio.py` (3), `test_packs.py` (4), `test_explorer_plugin.py` (1). The toy part of the exit gate runs through the CLI exactly as the README quickstart does:

**`pack new demo` → `pack check --level semantic`:**
- passes;
- a license set to `unknown` fails with exit 1;
- an unknown pack is a one-line error (exit 4).

**`label --pack toy`:**
- 24 traces, all credited;
- outcomes 16 good / 6 bad / 2 unknown;
- 34 signals;
- SFT rows with loss masks on assistant turns only, and Φ weights;
- 22 KTO rows.

**`build toy`:**
- 5 tasks forged; 0 accepted and 5 provisional without `--docker` / `--paid` (A10 pending), none rejected;
- 5 canonical exports with oracle reward 1.0;
- 5 anchors + 2 fresh items split into train and bench (sealed); the bench split is benchmarked (oracle 1.0, no-op 0).

**`label --run`:** one best-vs-worst DPO pair per benchmarked task, with OpenAI-legal tool names.

**`bench compare`:** every benchmarked task regresses from oracle to no-op.

The M15 audit's regression tests are `tests/unit/test_m15_audit_studio.py`.

Also: the scaffold conflict rules, the filter grammar refusing anything outside its whitelist, the manifest contract,
and the row shapes.

## Findings and open items

- **The toy logs contain 5 correction pivots, and in every one the fix was made by someone other than the agent.** So
  correction-DPO is empty on toy, and DPO comes from runs (best vs worst).
- **ATIF tool names are dotted** (`tickets.add_note`). Dataset rows use `openai_name` so OpenAI-format trainers accept
  them.
- **Run-sourced rows carry no `tools` schemas,** because the run directory does not hold the World's tool list. Datasets
  from runs should record the tool cards.
- **Built 2026-10-11:**
  - `crucible model <pack>`: the M5 workflow model (`workflow.json`; toy: 3 milestones, 3 stages);
  - `crucible export --to canonical|atif|nemo-gym|openenv`:
    - canonical replays (finance_ap gold 1.0);
    - ATIF trajectories validated by Harbor's validator;
    - a NeMo-Gym server launcher (the app is built to prove the tasks load);
    - an OpenEnv environment module (it builds and resets in `.envs/openenv`);
  - the agentenv explorer plugin (`crucible.studio.explorer:CrucibleExplorerPlugin`, entry point
    `agent_env.explorer_plugins`), serving packs, pack checks, runs, M13 summaries and reports, episodes and label
    datasets under `/api/v1/crucible`. Run ids are paths relative to the data root, and every file read is resolved
    and contained in it;
  - `pack check --level fixture` (see `M14-packs.md`).
- **`crucible label --pack coding`:** the 11 recorded frontier runs and 55 PCL PRs give 66 traces, 11 credited (end-state
  Φ, recorded outcome), 62 KTO rows. **The `build` part of the task02 gate needs the lab GPU** (the coding forge's tasks
  are graded in GPU verifier images).
