# Crucible

Workflow traces with expert signal → dense-reward RL environments, frontier-hard tasks, step-level benchmarks.
`DESIGN.md` covers what and why; `ARCHITECTURE.md` covers how (modules M0–M15). `docs/modules/` holds each built module's spec and gate report; `THIRD_PARTY.md` lists what we build on.

**Status:** every module, M0–M15, has code. Not every exit gate passes in full: `ARCHITECTURE.md` lists each gate, and `docs/modules/` holds the reports.

## Install
```bash
uv sync --python 3.12 --extra mcp --extra harbor --extra agentenv --extra llm --extra dev --extra openenv-awm --group packs
```
`verifiers` (the prime-rl path) pins `mcp==2.0.0` and cannot share this environment with agentenv; it gets its own environment.

## Quickstart: no model, no Docker, no API key

Each command prints the next one. Everything below runs on CPU in a couple of minutes.

```bash
# 1. Make a domain pack and validate it: static checks, then gold passes and the untouched task fails
crucible pack new demo --dir ./demo
crucible pack check ./demo --level semantic

# 2. Benchmark it with scripted policies: every step metric, plus an HTML report
crucible bench run ./demo --out runs
open runs/bench-demo-bench/bench/report.html

# 3. label: workflow logs -> traces with dense rewards + SFT / DPO / KTO datasets (TRL formats)
crucible label --pack toy --out label        # the toy pack's 24 helpdesk ticket logs
head -c 600 label/sft.jsonl                  # messages, message_loss_mask, per-turn Φ weights, provenance

# 4. build: workflow logs -> RL tasks (forge, accept, export) + a benchmark of anchors and fresh items
crucible build toy --out build
open build/runs/bench-toy-build/bench/report.html

# 5. A real domain: accounts payable (3-way match, fraud controls)
crucible pack check finance_ap --level semantic
crucible bench run finance_ap --out runs

# 6. Datasets and comparisons from any run
crucible label --run build/runs/bench-toy-build --out label_run          # best-vs-worst DPO pairs per task
crucible bench compare build/runs/bench-toy-build build/runs/bench-toy-build \
    --base-policy scripted:oracle --cand-policy scripted:half              # flips, exact McNemar, paired CI
```

What each command does:

| Command | Modules | Output |
|---|---|---|
| `crucible label --pack P` | M3 ingest → M4 signals → M5 milestones → M6 credit → validity gate per Φ channel | `traces.jsonl`, `rewards.jsonl` (Φ per event, channel verdicts), `sft.jsonl`, `dpo.jsonl`, `kto.jsonl`, `summary.json` |
| `crucible label --run DIR` | M10 run → datasets | SFT from solved episodes; DPO from best vs worst of k |
| `crucible build P` | M3 → M8 forge → M9 accept (the full contract: accepted, provisional while checks that need Docker or paid calls are pending, or rejected) → M11 canonical export → M13 splits, contamination and bench of the bench split | task dirs, `accept.json`, `export/`, `splits.json`, `contamination.json`, `report.html`; exit 1 on rejected tasks or export errors (`--allow-rejected`) |
| `crucible bench run / report / compare` | M13 | pass@k, pass^k, gate profile, progress AUC, stage reach, regressions, cost, IRT, flips |
| `crucible pack new / check` | M14 | a pack from `packs/_template`; a check report (exit 1 on failure) |
| `crucible ingest / signals / model / forge / accept / export / train` | one module each | `export --to canonical|atif|nemo-gym|openenv`; `train` runs the toy launcher (TinyPolicy) only |

The packs are `toy`, `coding` (its live grading needs the lab GPU) and `finance_ap`. `crucible pack check --all` validates
them all. Options are listed under `crucible <command> --help`. `--json` prints the full result. `--dry-run` (`label`, `build`)
shows what would run. To make a pack your own, edit `src/<package>/studio.py` and `pack.yaml` (its hooks), then rerun
`crucible pack check`.

**Browsing results.** The agentenv explorer shows Crucible's runs, reports and datasets through the `crucible` plugin
(`/api/v1/crucible`). It reads one data root: `[plugins.crucible] data_root` in agentenv's config, else
`$CRUCIBLE_DATA`, else `./data`. Point it at the directory that holds your `--out` directories (the defaults are under
`data/`). Run ids are paths under that root; reports are served sandboxed. Exit codes: 0 ok, 1 something failed, 2
usage, 3 internal error, 4 bad input; `--json` prints an `{"error": ...}` object on failure.

## Library: a World in 20 lines
```python
import asyncio
from crucible.actors.policy import ScriptedPolicy, run_episode
from crucible.core.schema.actions import Message, ToolCall
from crucible.world.world import World
from crucible_pack_toy import INSTRUCTION, PACK_ROOT, helpdesk_spec, responders

async def main():
    w = World(helpdesk_spec(), instructions={"agent": INSTRUCTION}, pack_root=PACK_ROOT)  # backend="agentenv" works too
    for r in responders():            # grounded customers + supervisor bot run inside the world
        w.attach(r)
    await w.start()
    try:
        policy = ScriptedPolicy("agent", [ToolCall(tool_call_id="1", function_name="tickets.list_tickets", arguments={}),
                                          Message(to="customer_lee", text="How much were you charged for O-5002?")])
        res = await run_episode(w, {"agent": policy})
        print(next(e.text for e in res.trace.events if e.actor == "customer_lee"))  # the customer's grounded answer
        ref = await w.snapshot(); await w.restore(ref)         # exact, verified by fingerprints
        print(await w.verifier_view().query("tickets", {"op": "sql", "args": {"sql": "SELECT id, status FROM tickets"}}))
    finally:
        await w.close()

asyncio.run(main())
```

## Library tour
- **Seed coding worlds** (tasks 01–04) on a GPU host through Harbor: `packs/coding/families/legacy_gpu_migration/seeds/*/crucible/world.yaml`. Gate: `CRUCIBLE_GPU=1 pytest tests/gpu -m gpu`.
- **Import any MiMo env or AgentWorldModel-1K scenario as a World:** `crucible.interop.mimo.importer.import_mimo_env`, `crucible.interop.openenv.awm.awm_world`. Sweeps: `python -m crucible.interop.mimo.check <envs_dir>`, `python -m crucible.interop.openenv.awm_check 100`.
- **GPU sandboxes:** the lab GPU host through Harbor (`backend_params: {gpu_host: lab5090}`) or agentenv (`sandbox: {provider: docker, image, gpu_host: lab5090}`); cloud GPUs through Harbor (`environment_type: modal`) or agentenv (`sandbox: {provider: modal, image}` with `resources.gpu_type: H100`). Gates: `CRUCIBLE_GPU=1 pytest tests/gpu -m gpu`; `CRUCIBLE_MODAL=1` for Modal.
- **Harden any container World:** `isolation: {policy_user: agent}` (guard, trusted git, episode hygiene, egress self-test); attack suite: `pytest tests/redteam -m docker`.
- **Audit a World before using it:** `crucible world validate <world.yaml> [--gold actions.json] [--forbidden FILE]` (hidden answers, one per line, from a file or `-` for stdin; never on the command line).
- **Serve a World to any MCP harness:** with the agentenv backend, `world.backend.mcp_url` plus the header `AgentEnv-Role: <actor>`.
- **Export an episode as an ATIF trajectory:** `crucible.interop.harbor.atif.episode_to_atif`.
- **Grounded actors:** `GroundedResponder` (verbatim / quote / llm), and the coding pack's `ReviewerActor`, which replays real review comments. Fidelity: `python -m crucible_pack_coding.reviewer_fidelity`.

## Website
`site/` is a single-page tour of every module as live 3D figures (three.js, no build step). `page.html` is the page body, `assets/*.js` the engine and scenes. Preview: `python site/build.py && python -m http.server -d site` (writes the standalone `site/index.html`). Publish: `site/publish.sh "what changed"` pushes it to https://ahmd-mohsin.github.io/crucible/.

## Tests
`pytest tests/unit tests/toy_e2e` (CPU, no paid calls). End-to-end CPU smokes of each major path: `scripts/smoke/`
(`--help` on each; DESIGN §14). `-m docker` needs Docker; GPU gates need `CRUCIBLE_GPU=1` and a `[gpu_hosts]` entry in `crucible.toml`. `lint-imports` checks the layering contracts.
