# Module audits against the lab RL repos: plan (2026-10-20)

Each module of Crucible is audited, one at a time and in its own session, against every reference repo cloned inside
this repository's git-ignored folders (`agentenv-framework/` and `RL-Environments/{01-lab-releases,06-frameworks-and-env-code,07-articles,08-hf-rl-env-datasets}/`). The question for every
module: what can Crucible adopt, add, replace or drop, given how the same responsibilities are built elsewhere; and which
earlier recommendations (`../00-index.md` and its studies, dated 2026-10-11; the module audits in `docs/audit/`) are done.

## Output

One document per module in this folder, `<Mxx>-<name>.md` (e.g. `M00-core.md`), with sections:
0 Summary · 1 What the module does · 2 Status ledger · 3 Comparison questions · 4 Repo triage · 5 Findings per repo ·
6 Gap matrix · 7 Recommendations, ranked (R-n: kind, tier, status, origin, problem, reference, recommendation,
reasoning, established by, impact / effort / risk, mode, acceptance; later items as L-n rows) · 8 What we already do
better · 9 Open questions · 10 Handoff to other modules · 11 Sources.

Recommendations are designs, not code edits. Each one is verified against both codebases before it is listed, and
must name a realistic failure today and who it hurts: must (data loss, a secret leak, a silently wrong result, a licence
obligation, a regression, a broken consumer), should (rarer failures, cheap tests against likely breaks), later (no
consumer yet: one line, with what would make it due). Matching every reference repo is not a goal.
Each document is at most 650 lines, shorter when there is less to say: the top recommendations in full, the rest in a
compact form with every field; the agents' notes and probes hold the detail.

## Method

Run per module with the `module-audit` skill (workflow `.claude/workflows/module-audit.js` in the workspace root):

| Phase | Agents | Work |
|---|---|---|
| Scout | 1 | reads the module's spec, audits, sweeps and earlier studies and its code; sets 6-12 comparison questions (each with an axis), the ledger items, the commits earlier studies read; builds a probe environment |
| Sweep | 10 + 1 | one agent per repo cluster triages and reads its repos against the questions; one agent establishes the status of every ledger item |
| Verify | 2-4 | adversarial: each candidate is checked on both codebases at HEAD, reproduced when cheap, and rejected if its problem cannot be shown |
| Write | 1 | assembles the document from the template |
| Check | 1 | completeness critic: sections, every repo, every ledger item, citation spot-checks |

Rules: every repo is read-only (git log/show/diff/blame only); claims cite `path:line` and the repo's commit; licences
decide the mode (Apache-2.0, MIT, BSD: PORT with notice; GPL family or no licence: PATTERN only). Agents' notes and
probes are kept outside the repository, in the workspace's `.module-audit/2026-10-20/<Mxx>-<name>/`.

## Snapshot

Crucible at `d282251` on branch `uzair`, equal to `main` and `dev` (after `git pull --ff-only`). Reference repos are read at the commits below; they are not
pulled during the series, so every module is compared against the same snapshot.

| Repo | Commit | Date | Licence |
|---|---|---|---|
| `08-hf-rl-env-datasets` | not a git repo | — | none |
| `adithya-s-k-fineenvs` | `5e0b46e` | 2026-10-08 | Apache-2.0 |
| `agentenv-framework` | `041afd7` | 2026-10-08 | Apache-2.0 |
| `harbor-framework-harbor` | `a2fafbcf` | 2026-10-03 | Apache-2.0 |
| `hf-space-fineenvs-multi-harness-rl` | not a git repo | — | CC-BY-4.0 |
| `hf-space-fineenvs-simulation-rl-environments` | not a git repo | — | CC-BY-4.0 |
| `hf-space-rl-environments-guide` | not a git repo | — | CC-BY-4.0 |
| `huggingface-repo2rlenv` | `27d04ee` | 2026-10-11 | Apache-2.0 |
| `infra-alibaba-roll` | `0f37fe13` | 2026-10-12 | Apache-2.0 |
| `infra-huggingface-trl` | `d9b9f39b` | 2026-10-12 | Apache-2.0 |
| `infra-inclusionai-areal` | `aedc21ec` | 2026-10-12 | Apache-2.0 |
| `infra-moonshotai-checkpoint-engine` | `84631a2` | 2026-09-15 | MIT |
| `infra-novasky-ai-skyrl` | `a1e34dbf` | 2026-10-12 | Apache-2.0 |
| `infra-nvidia-nemo-rl` | `925f70d55` | 2026-10-12 | Apache-2.0 |
| `infra-openrlhf-openrlhf` | `dc2a7ad` | 2026-09-17 | Apache-2.0 |
| `infra-primeintellect-ai-prime-rl` | `1e5318108` | 2026-10-11 | Apache-2.0 |
| `infra-rllm-org-rllm` | `4e828c81` | 2026-10-08 | Apache-2.0 |
| `infra-thudm-slime` | `05e3b017` | 2026-10-12 | Apache-2.0 |
| `infra-volcengine-verl` | `2035f58d` | 2026-10-12 | Apache-2.0 |
| `karpathy-autoresearch` | `228791f` | 2026-03-25 | none |
| `meta-pytorch-openenv` | `8d3455ab` | 2026-10-09 | BSD-3-Clause |
| `nvidia-nemo-gym` | `7ece89985` | 2026-10-12 | Apache-2.0 |
| `preferencemodel-karotte` | `31c693e` | 2026-10-08 | MIT |
| `primeintellect-ai-prime-agent` | `3e0893260` | 2026-10-11 | MIT |
| `primeintellect-ai-prime-environments` | `cf220ee` | 2026-05-07 | Apache-2.0 |
| `primeintellect-ai-research-environments` | `f63f07c8a` | 2026-10-11 | Apache-2.0 |
| `primeintellect-ai-verifiers` | `e22533430` | 2026-10-12 | MIT |
| `servicenow-enterpriseops-gym` | `271f2c3` | 2026-08-19 | Apache-2.0 |
| `swe-agent-mini-swe-agent` | `04d809ce` | 2026-09-03 | MIT |
| `swe-bench-swe-smith` | `9b74ac0` | 2026-03-22 | MIT |
| `veds12-git-swarm` | `9f66af0` | 2026-10-09 | none |
| `xiaomi-mimo-v2.6-rl-oss` | not a git repo | — | none |
| `xiaomimimo-verl` | `a2ad9f61` | 2026-09-26 | Apache-2.0 |

Two commits differ from the 2026-10-16 snapshot on purpose: `agentenv-framework` and `harbor-framework-harbor` are
Crucible's path dependencies, and on 2026-10-16 their clones were checked out at main's pins (agentenv `041afd7`,
0.9.1301, as THIRD_PARTY.md records; Harbor `a2fafbc`, 0.23.0, as CI checks out), replacing `015c89b` and `dc933be7`.

## Schedule

| Module | Document | Status | Recommendations |
|---|---|---|---|
| M0 core | `M00-core.md` | not started | |
| M1 world | `M01-world.md` | not started | |
| M2 actors | `M02-actors.md` | not started | |
| M3 connectors | [`M03-connectors.md`](M03-connectors.md) | done (2026-10-20) | 0/4/23 |
| M4 signals | `M04-signals.md` | not started | |
| M5 workflow | `M05-workflow.md` | not started | |
| M6 credit | `M06-credit.md` | not started | |
| M7 verify | `M07-verify.md` | not started | |
| M8 forge | `M08-forge.md` | not started | |
| M9 accept | `M09-accept.md` | not started | |
| M10 runtime | `M10-runtime.md` | not started | |
| M11 interop | `M11-interop.md` | not started | |
| M12 train | `M12-train.md` | not started | |
| M13 bench | `M13-bench.md` | not started | |
| M14 packs (and the RLM agent) | `M14-packs.md` | not started | |
| M15 studio | `M15-studio.md` | not started | |
| M16 long-horizon RLM | `M16-long-horizon-rlm.md` | not started | |
