# M15 `studio`: `crucible label`, `crucible build`, packs, and the CLI

**Study:** `docs/research/bench-studio/01-framework-study.md` §3.

**Code:**
- `src/crucible/studio/{pack,label,build}.py`
- `src/crucible/credit/datasets.py`
- `src/crucible/core/schema/pack.py`
- `src/crucible/cli/{main,studio}.py`
- `packs/_template/`
- `packs/_toy/pack.yaml` + `crucible_pack_toy/studio.py`

## CLI

click, as Crucible and agentenv already use it. Every command does one module's job and writes a run directory
(`configs/resolved/<cmd>.json`, outputs, `summary.json`; the verifiers convention).
- **Lazy subcommands:** a command table imported on use (rLLM `_LazyGroup`, NeMo-Gym `COMMANDS`).
- **User errors:** one line; `--verbose` shows the traceback (agentenv).
- **Output:** `--json` on reporting commands, `--dry-run` on `label` and `build`, and the next commands printed.

**Commands:**
- `label`, `build`;
- `pack new|check`;
- `bench run|report|compare`;
- `ingest`, `signals`, `forge`, `accept`, `train`;
- plus `registry`, `task`, `seeds`, `world`.

## `crucible label`

**From a pack:**
1. M3 ingest of `pack.yaml` sources (idempotent, redacted).
2. M4 signals and pivots.
3. The pack's `credit` hook: M5 milestones mined from good dev logs; M6 Φ per event, a gated reward, and the outcome.
4. `credit.datasets`.

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

1. M3 ingest.
2. The pack's `forge` hook (toy: world_synth with scripted roles, no paid calls).
3. M9 model-free contract checks: A1, A2, A3, A4-lite, A5, A7, A8.
4. M11 canonical export: oracle replayed into `reward.json` + `steps.jsonl`, the outputs every runtime adapter must
   reproduce. A World task is not a Harbor-native directory; Harbor reaches it through ATIF replay.
5. An M13 benchmark: accepted tasks as anchors, plus fresh generated items, with scripted policies.

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
- an unknown pack is a one-line error.

**`label --pack toy`:**
- 24 traces, all credited;
- outcomes 16 good / 6 bad / 2 unknown;
- 34 signals;
- SFT rows with loss masks on assistant turns only, and Φ weights;
- 22 KTO rows.

**`build toy`:**
- 5 tasks forged, 5 accepted;
- 5 canonical exports with oracle reward 1.0;
- a 5-anchor + 2-fresh benchmark (oracle 1.0, no-op 0).

**`label --run`:** 7 best-vs-worst DPO pairs with OpenAI-legal tool names.

**`bench compare`:** 7 regressions, McNemar p < 0.05.

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
  - the agentenv explorer plugin (`crucible.interop.agentenv.explorer:CrucibleExplorerPlugin`, entry point
    `agent_env.explorer_plugins`), serving packs, pack checks, runs, M13 summaries and reports, episodes and label
    datasets under `/api/v1/crucible`. Run ids are validated and resolved inside the data root;
  - `pack check --level fixture` (see `M14-packs.md`).
- **`crucible label --pack coding`:** the 11 recorded frontier runs and 55 PCL PRs give 66 traces, 11 credited (end-state
  Φ, recorded outcome), 62 KTO rows. **The `build` part of the task02 gate needs the lab GPU** (the coding forge's tasks
  are graded in GPU verifier images).
