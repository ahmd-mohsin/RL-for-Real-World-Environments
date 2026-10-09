# M2 `actors` — everyone who isn't the policy

**Status: built (2026-10-09).** Code: `src/crucible/actors/`, `src/crucible/interop/agentenv/a2a.py`, `packs/coding/src/crucible_pack_coding/{github_reviews,reviewer,reviewer_fidelity,grounding_gate}.py`, `packs/_toy/src/crucible_pack_toy/m2_gate.py`. Tests: `tests/unit/test_actors*.py`, `tests/unit/test_reviewer_actor.py`, `tests/unit/test_episode_loop.py`, `tests/toy_e2e/`.

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

**What the policy sees of an actor (2026-10-09).** Replies to a policy are leak-scrubbed (`world/leaks.py`: PR and commit links, fixed-version pointers, package pins), and the unscrubbed text stays in the trace. Citations stay in the trace for audit but are removed from the policy's observation (`isolation.hide_citations`, default on): a citation names a record, and records can name the answer.

## Episode loop and LLM client (hardened 2026-10-09, from the verifiers / NeMo-Gym / OpenEnv / MiMo survey)

- **Parallel tool calls.** `LLMToolPolicy` runs a batch in order and appends all of its results before any message that arrived meanwhile. A reply to a batched `send_message` used to split the batch and made the next request invalid for OpenAI-style APIs, which silently masked the episode.
- **What is replayed.** Provider reasoning fields (`reasoning`, `reasoning_details`, thinking signatures) are replayed verbatim. Tool names are made safe for `^[a-zA-Z0-9_-]{1,64}$` through a reversible map (`core.schema.actions.openai_name`). Output is bounded once, by the World.
- **Errors by status code** (`actors/llm.py`):
  - 402 or quota → `crucible:credits_exhausted` (masked);
  - a context overflow (verifiers' provider markers) → the episode ends **truncated and scored**, so long-horizon failures stay in the data;
  - any other 4xx → `agent_request_failed`;
  - 429/5xx → retried with jittered backoff and `Retry-After`;
  - an empty `choices` list → retried;
  - a missing `usage` block → `crucible:usage_blank`. It used to be priced at $0, so the budget could never trip.
  - Double-encoded JSON arguments are accepted; non-object arguments become a typed `invalid_arguments` result.
- **Record/replay cache.** Policies key the cache by episode (`sample_key`). N samples of one task no longer replay one completion with zero variance. Simulators and judges stay content-keyed.
- **Episode limits and stop conditions.** `EpisodeLimits` has `max_steps`, `max_tool_errors`, `timeout_s`, `end_on` and `snapshot_every`. Every `EpisodeResult` carries a `stop_condition`: `agent_done`, `all_done`, `max_steps`, `too_many_errors`, `timeout`, `context_overflow`, `user_stop`, `out_of_scope`, or a failure kind. It also carries `truncated`, per-call model records (usage incl. reasoning and cached tokens, latency, finish reason, cache hit) and an optional fingerprint-deduplicated snapshot series.
- **Failures and masking.** Every failure carries its own mask. Infrastructure is masked; the policy's own outcomes (`agent_timeout`, `crucible:policy_misbehavior`) are scored. The unregistered `policy_error` kind is gone.
- **Simulated actors end the conversation.** `###DONE###` and out-of-scope make a simulated actor emit `Done`, which the World records (`world.finished`). The marker never reaches the policy. A DONE as the opening reply is regenerated (NeMo-Gym: the first user message cannot be STOP). `run_episode(end_on=...)` stops on it.
- **Mailbox identity.** The acting identity is held in a `ContextVar`: a shared attribute let concurrent calls send mail as the wrong actor.

## Exit gate

| Requirement (ARCHITECTURE M2) | Result |
|---|---|
| Reviewer simulator on held-out review rounds: **defect-location precision ≥ 0.8** | ✅ **0.956 (PCL, 90 comments), 0.977 (CCCL, 131 comments)** on real human review comments (bots excluded), ±3 lines. Silent on 100% of comments whose code had changed by the PR head; 66% of comments led to a code change at the commented spot. `reports/m2_reviewer_fidelity.json` |
| **Fabrication rate ≤ 1% on 200 audited replies** (coding) | ✅ **0 / 225 factual sentences** with `GroundedResponder(mode="quote")` over 223 real PCL/CCCL review records. Of 170 answerable questions, 166 were answered with citations; 30/30 unanswerable got "no record". Budget: $0.024 responder (`openai/gpt-6-luna`) + $0.003 judge (`anthropic/claude-haiku-5.5`). Judging every sentence strictly gives 3/225 (1.3%), all the same verbatim record "Same for copy_point.hpp", which the judge rejected as context-dependent. A verbatim copy of a cited record cannot add meaning, so the audit now treats verbatim sentences as supported (`m2_grounding_gate_quote.json`, strict run in `…_quote_strict.json`) |
| … (toy) | ✅ **0%** over 200 replies of the grounded customers (all non-"no record" replies cited). The scenario-simulated customer (LLM, per-reply consistency judge) in a live helpdesk episode with an LLM agent: consistent, not masked; the agent confirms the duplicate charge with the customer, refunds 39.50 USD, notes and resolves ticket 2, which is the expert end state (`m2_toy_gate.json`) |
| Every simulated reply carries citations | ✅ 0 uncited non-"no record" replies in both packs |

## Findings
1. **Paraphrase drift is real, and citation checks don't catch it.** With `mode="llm"`, gpt-6-luna turned tentative notes into claims ("as far as I see" disappeared), turned a statement into a question, and attributed locations the record never stated, while every sentence still cited a valid, visible record with matching numbers. Only the different-family entailment judge caught these. Measured on the same 200 questions: **paraphrase 10/147 factual sentences rejected (6.8%) vs quote 0/225** (`m2_grounding_gate_paraphrase.json`; paraphrase recall is also lower, 148/170 vs 166/170). **Quote mode** (the LLM selects, the record speaks) removes the failure class by construction and is the default for grounded actors.
2. **Judges and selectors on reasoning models need large token caps.** `gpt-6-luna` spends hidden reasoning tokens: under a 200-token cap it returned empty completions (`finish=length`), which silently became "no record" (recall 92/170 → 166/170 after the fix) or masked episodes. Verdicts are now also salvaged from truncated JSON.
3. **Time visibility must compare instants, not strings.** `"…56Z" > "…56.5Z"` lexically, which hid records from their own timestamp onward.
4. **System actions must not inherit an actor's tool access.** Found by the backend-independence test: under agentenv, event-driven tool calls ran with the event actor's (empty) role and were refused. They now use a privileged system role.
5. **Verbatim mode answers with related truths.** 8/40 unanswerable toy questions got a related, true record (e.g. "which bank issued your card?" → the card-statement note). That's a relevance gap, not fabrication; quote mode's LLM selection handles relevance.
