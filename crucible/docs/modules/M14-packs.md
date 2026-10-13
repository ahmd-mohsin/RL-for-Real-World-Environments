# M14 packs: domain knowledge as plugins (ARCHITECTURE §4 M14)

This is not `M14-rlm.md`. The repository also uses "M14" for the RLM agent, which is a separate module.

**Code:**
- `src/crucible/core/schema/pack.py` (`PackManifest`);
- `src/crucible/studio/pack.py` (`pack new`, `pack check`);
- `packs/_template`, `packs/_toy`, `packs/coding`, `packs/finance_ap`.

**`pack.yaml`** declares (every field is honoured somewhere; M14 audit V12):
- identity (`name` matches `^[a-z][a-z0-9_]{1,40}$`, `version` is PEP 440), an SPDX licence, kind;
- `reward.range` (every graded reward must lie in it), `higher_is_better` (orients gold, no-op and adversary bounds;
  `bench run` refuses a lower-is-better pack rather than report it upside down) and `determinism` (`seeded` packs are
  re-graded on fresh Worlds and their tasks rebuilt);
- `requires` (what live tasks need: docker and gpu are detected, network and paid vouched for with `--allow`);
- `ingest` sources: any registered connector, constructed with `init`, discovered with `scope` / `discover`. Paths are
  pack-relative (absolute ones are refused) and must resolve, symlinks followed, inside the pack, the checkout or a
  directory in `$CRUCIBLE_DATA_ROOTS`. A declared `redaction_env` must exist unless `redaction_env_optional: true`;
  otherwise `pack check` fails and `label` refuses before storing anything;
- `label` (policy roles, a `threshold` inside the range: what `label` and `bench run` count as solved);
- hooks: `credit`, `tasks`, `generator`, `stages`, `forge`, `policies`, `fixtures`, `model`, all inside the pack's own
  package. `fixtures` and `policies` may carry adversaries: `{"name", "policy", "expect_max"}`, where `policy` is a
  `PolicyRef`, a policy id from `policies`, or a callable `task -> actions`.

**`crucible pack check` levels** are cumulative and fail closed (`studio/pack.py` docstring has the full contract):
- **static:** manifest contract; placeholders anywhere in pack.yaml, README, pyproject and `src/`; SPDX licence;
  pyproject agrees with pack.yaml (name, version, licence, entry point, crucible requirement); `src/` holds only the
  package; `register()` registers this pack at this directory; hooks inside the package; ingest confined. Template text,
  no authors, no README and a wheel without `pack.yaml` are warnings, failures with `--publish`.
- **fixture:** the hook is required; cases grade exactly, inside the range, and finitely; `fails_gate` names a gate the
  grader reports as failed; one case at each end of the range; seeded packs are graded 3 times on fresh Worlds. A
  self-graded `grade` callable is only for packs without live tasks.
- **semantic:** the `tasks` hook is required and non-empty; gold reaches the best end (3 fresh Worlds), the untouched task
  the worst end; tasks and generated items rebuild identically; generated items have an oracle (what `bench run` needs);
  every adversary stays within `expect_max` on every task and generated item.
- **accept:** the M9 Acceptance Contract on fixture tasks, anchors and a generated item (`studio/accept_adapter.py`);
  `crucible accept <pack>` uses the same adapter when the pack has no `forge` hook.

Unmet `requires` make the live levels loud skips (exit 2 without `--allow-skip`). A raising hook is a
`<level>.error` failure. Exit codes: 0 pass or warn, 1 fail, 2 incomplete, 3 internal error. Each check imports the pack
from its own `src/` and restores `sys.path` and `sys.modules`; `--all` runs every pack in its own process.

## Packs

| Pack | What | Check |
|---|---|---|
| `_template` | A seeded workspace task with 5 designed fixture cases, `oracle_actions` for generated items, an example adversary, the M9 task capabilities (properties, checks, negatives, mutants, attacks, tamper and isolation audits); installable (pack.yaml / data force-included under `_pack/`) | static + fixture + semantic + accept (provisional: A10 pending paid calibration) |
| `toy` 0.3.0 | Helpdesk: ticket-log ingest, M5/M6 credit, the forge, designed grader negatives as fixtures, the workflow model. **Audit:** approvals must be requested through `tickets.request_approval` and match the supervisor exactly (a typed name scored 1.0); the 50 USD threshold applies to the refunded total per order and ticket (3 × 43.33 dodged it); notes must say why; a premature close is not erased by a re-close; adversaries `forged_approval`, `split_refund` bounded at 0. **Campaign:** the backlog (`@3`) requires a note that says why per ticket and an approval requested for the named ticket or order, consumed by one refund (any note scored 0.19 and amount-only approvals 1.0), waiting tickets stay untouched until the reply, and benchmark items are all-or-nothing; helpdesk and backlog tasks carry the M9 capabilities | static + fixture + semantic + accept (provisional: A10 pending paid calibration) |
| `coding` 0.3.0 | Seeds task01v2–04 (GPU verifier images), the recorded frontier runs (now carrying `task_version`; a run graded under an earlier version is `unknown`) and the 55 PCL PRs as ingest sources, credit for recorded runs. **Audit:** dense Φ is measured on code only (`codeonly.py`): comments removed, string and char literal contents blanked, preprocessor-dead, constant-false and unreachable code and unused new functions dropped, unparseable Python / TOML has no code (campaign B2: string-literal and `#if 0` stuffing reached Φ 1.0 on every seed); token milestones earn credit on their own and created / deleted / removed / located ones only by the token share (comment stuffing reached Φ 0.92–1.00); CPU `dense_replay` tasks (`@cpu2`) give the pack a tasks hook; adversaries `comment_stuffing`, `destructive_only`, `string_literal_stuffing`, `dead_code_stuffing` bounded at 0; every hook runs on CPU, so `requires` is empty (B11); the dotenv ingest is optional | static + fixture + semantic (CPU) |
| `finance_ap` 0.3.0 | **Accounts payable**, the thesis MVP domain. **Generator v2 (audit):** 6–14 invoices, 45–70% clean; a per-world tolerance (0.5–10%, inclusive) and a per-world order of the three hold rules stated only in the policy; a forced compound case, a duplicate variant an exact match misses, price probes at and one cent over the bar, near misses (partial receipts, an earlier rejected duplicate, another vendor, a similar number) and an infeasible `po_mismatch`; labels are AP-3 applied to the data (checked over 1,000 seeds against an independent reading of the policy text). A memorized 2% solves 15% of worlds (73% before). **Generator v3 (campaign B7):** every world holds two compound cases on adjacent pairs of its hold order, a restated-amount duplicate, a one-cent-over price whole units pass, and a stopped payee (in 85% of worlds an alias: a new vendor id sharing a blocked vendor's bank account, found with `erp.list_vendors`); over 2,000 worlds a fixed hold order solves ≤ 18% (57.8% before), whole-unit prices and same-amount duplicates 0% (68%, 54.5%), ignoring aliases 15%. **Reward:** all-or-nothing over the `controls` and `outcome` gates (partial credit let every grader mutant survive M9 A3); Φ per invoice, zeroed by a control breach. Adversaries `approve_all`, `hold_all`, `exact_match_duplicates`, `wrong_tolerance`, `reverse_rule_order`, `price_round_units`, `dup_needs_same_amount`, `alias_vendor_args` bounded at 0 | static + fixture + semantic + accept (A1–A9, A13, A16 pass; A10 pending paid calibration) |

## Gate

`tests/unit/test_packs.py`, `tests/unit/test_studio.py`, `tests/unit/test_m14_audit_framework.py` (the H1 and G2 mutant
campaigns: 25 broken scaffolded packs and 23 broken finance_ap copies all fail `pack check` with a named reason; an honest
lower-is-better pack and the unchanged finance_ap pass).
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
