# M14 packs: domain knowledge as plugins (ARCHITECTURE §4 M14)

This is not `M14-rlm.md`. The repository also uses "M14" for the RLM agent, which is a separate module.

**Code:**
- `src/crucible/core/schema/pack.py` (`PackManifest`);
- `src/crucible/studio/pack.py` (`pack new`, `pack check`);
- `packs/_template`, `packs/_toy`, `packs/coding`, `packs/finance_ap`.

**`pack.yaml`** declares:
- identity, license, kind;
- `reward.range` and `determinism`;
- `requires` (what live tasks need: docker, gpu, network, paid);
- `ingest` sources: any registered connector, constructed with `init`, discovered with `scope` / `discover`, paths
  relative to the pack, and `redaction_env`;
- `label` (policy roles, threshold);
- hooks: `credit`, `tasks`, `generator`, `stages`, `forge`, `policies`, `fixtures`, `model`.

**`crucible pack check` levels** are cumulative:
- **static:** manifest contract, placeholders, license, layout, README, entry point, hooks, ingest sources.
- **fixture:** designed cases grade exactly as expected. A case is a World episode (`task` + `actions`) or, for packs
  graded on artifacts, an async `grade` callable.
- **semantic:** gold reaches the top of the reward range and the untouched task does not. It is skipped, with the reason,
  when the pack's `requires` is not met here (`--allow gpu` where it is).

## Packs

| Pack | What | Check |
|---|---|---|
| `_template` | A seeded workspace task with 5 designed fixture cases | static + fixture + semantic |
| `toy` | Helpdesk: ticket-log ingest, M5/M6 credit, the forge, 7 designed grader negatives as fixtures, the workflow model | static + fixture + semantic |
| `coding` | Seeds 01–04 (GPU verifier images), the recorded frontier runs and the 55 PCL PRs as ingest sources (any-connector ingest), credit for recorded runs, CPU fixtures over the dense reward (oracle 1 / base 0 on all four seeds) | static + fixture; semantic needs `--allow gpu` on the lab |
| `finance_ap` | **Accounts payable**, the thesis MVP domain. Seeded invoice worlds under a 3-way-match policy with fraud controls (duplicates, blocked vendors, unverified bank changes, a per-world price tolerance stated only in the policy, missing receipts). Graded from state: a `controls` gate (approving what a rule should stop scores 0) plus the share decided correctly with reason codes; Φ per invoice | static + fixture (gold, 5 control breaches, 2 near misses with exact partial credit, no-op) + semantic |

## Gate

`tests/unit/test_packs.py`, `tests/unit/test_studio.py`.
- `crucible pack new demo` passes `crucible pack check` at every level.
- `crucible pack check --all --level fixture` passes for toy, coding and finance_ap.
- The finance_ap benchmark: oracle 1.0; approve-everything 0.0 (controls pass rate 0); no-op 0.

**The other half of the M14 gate** ("coding and `_toy` pass every module gate built so far") is not met while module gates
stay partial. See `docs/research/improvements/00-index.md` and the ARCHITECTURE status table.

**Findings:**
- Label ingest was hard-wired to `file_drop` and filtered by an id prefix that coding connectors don't use. Ingest now
  takes any registered connector and returns the traces by stored key.
- The coding PR title filter has to be the gate's exact regex: a partial copy ingested 91 PRs instead of 55.
- On finance_ap, approve-everything earns progress AUC 0.15 while scoring 0. That is live evidence for the study's
  "dense credit ignores gates" finding (M6 V3).
