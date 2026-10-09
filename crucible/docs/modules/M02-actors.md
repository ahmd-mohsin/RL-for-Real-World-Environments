# M2 `actors` — everyone who isn't the policy

**Status: built (2026-10-09).** Code: `src/crucible/actors/`, `src/crucible/interop/agentenv/a2a.py`, `packs/coding/src/crucible_pack_coding/{github_reviews,reviewer,reviewer_fidelity,grounding_gate}.py`, `packs/_toy/src/crucible_pack_toy/m2_gate.py`. Tests: `tests/unit/test_actors*.py`, `tests/unit/test_reviewer_actor.py`, `tests/toy_e2e/`.

## What it is
Simulated people, bots, humans and peer agents run *inside* the World as environment dynamics: they act when they receive a message or an event, under their own tool permissions, and their memory is part of the `WorldRef`. Policies are wrapped here too, with the episode loop.

| Piece | What | Built on |
|---|---|---|
| `records.py` | `RecordStore`: BM25 over what people actually wrote (CJK-aware). **Time rule:** a record is invisible before its timestamp (instants are compared, not strings), so replayed actors cannot leak the future. **Scope rule:** filter by source / role / tag | — |
| `grounding.py` | Citation checker: every factual sentence must cite an existing, visible record; numbers / amounts / dates / ids must appear verbatim; content overlap. Questions and acknowledgements need no citation | — |
| `grounded.py` | `GroundedResponder` in three modes: **verbatim** (best record, no LLM), **quote** (an LLM *chooses* records, which are returned verbatim with an optional fact-free lead-in), **llm** (paraphrase with citations, checked, regenerated, unsupported sentences dropped). Otherwise "I have no record of that." | — |
| `simulated.py` | `SimulatedActor` driven by a scenario (persona, reason, known / unknown info, instructions, out-of-scope), with an optional per-reply **consistency judge** from a different model family, bounded regeneration, tool use under the actor's ACL, and `crucible:sim_failure` masking | verifiers v1 `UserSimEnv` persona / done marker; NeMo-Gym scenario fields + judge + masking; tau2 "never hallucinate / progressive disclosure / out-of-scope" |
| `scripted.py` | `RelayActor` (rules → reply or tool call, e.g. a clerk bot that looks orders up); `CheckRunnerActor` (CI bot: runs checks through the *verifier* view and reports only pass/fail per check name) | — |
| `human.py`, `transports.py` | `HumanActor` / `PeerActor` over transports: callable, **file queue** (any UI or a person with an editor answers), **A2A** (`interop/agentenv/a2a.py`: agentenv's `send_a2a_message` / `poll_a2a_task` / `TerminalResponse`; agents served with agentenv-protocol's `create_app`) | agentenv A2A protocol + `deploy_human_agent` model |
| `policy.py` | `ScriptedPolicy`, `LLMToolPolicy` (OpenAI function calling over the World's tools + `send_message`), `run_episode` (round-robin; reports simulator failures so the sample is masked) | — |
| `llm.py` | OpenAI-compatible client: **hard USD budget** (`BudgetExceeded` → `crucible:credits_exhausted`, masked), **record / replay cache** (deterministic, free reruns), provider cost or configured prices | OpenRouter |
| `fidelity.py` | Grounding audit (deterministic + entailment judge), consistency summary, location precision / recall | — |
| coding pack `reviewer.py` | **Grounded reviewer**: replays a recorded review comment exactly when its diff-hunk anchor is present in the reviewed file, at the matching line (nearest the recorded line when repeated); silent once the code changed. No LLM, so it cannot fabricate | GitHub review history |
| coding pack `github_reviews.py` | Review comments → `Record`s (path, anchor, original commit) without auth: repo-wide comments endpoint + raw.githubusercontent.com + `git ls-remote refs/pull/*/head`; bot reviewers excluded | GitHub REST |

## Exit gate

| Requirement (ARCHITECTURE M2) | Result |
|---|---|
| Reviewer simulator on held-out review rounds: **defect-location precision ≥ 0.8** | ✅ **0.956 (PCL, 90 comments), 0.977 (CCCL, 131 comments)** on real human review comments (bots excluded), ±3 lines. Silent on 100% of comments whose code had changed by the PR head; 66% of comments led to a code change at the commented spot. `reports/m2_reviewer_fidelity.json` |
| **Fabrication rate ≤ 1% on 200 audited replies** (coding) | GATE_CODING |
| … (toy) | GATE_TOY |
| Every simulated reply carries citations | GATE_CITES |

FINDINGS
