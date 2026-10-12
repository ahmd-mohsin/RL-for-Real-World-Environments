# M3 study: redaction, privacy and trace storage

Read-only comparison for the M3 audit. Crucible code read at HEAD `dc29c0f`. The working tree (WT) already carries an
in-flight fix to `redact.py` with tests `tests/unit/test_m3_audit.py` A1–A10. Each candidate below says whether it
is **open in WT** or **fixed in WT**. Every "open" candidate was re-run against the WT and still reproduces.
Line numbers are HEAD unless marked WT.

Probe preamble for every snippet:

```python
import os; os.environ["CRUCIBLE_REDACTION_SALT"] = "probe"   # WT needs a salt; HEAD ignores it
from crucible.connectors.redact import Redactor, RedactionPolicy, redact
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceSource
R = Redactor(RedactionPolicy())
```

## 1. Mechanism table

| Framework | What it redacts | How | Where |
|---|---|---|---|
| **Crucible** | Provider key formats (21), PEM blocks, Bearer, `password=`-style pairs, known `.env` values, e-mails, private IPs, internal-TLD hosts, human actor ids and handles, policy terms | Regex `sub` over every string of a dumped trace. Salted SHA-256 pseudonyms at HEAD, HMAC with a tenant key in WT. Runs in `ingest` before `put` | `src/crucible/connectors/redact.py:21-139`, `base.py:52` |
| Crucible core config | Values under secret-shaped keys | Key-substring match (`token`, `key`, `cookie`, `authorization` ...), value masked | `src/crucible/core/config.py:89-108` (PORT of NeMo-Gym). **Not used by trace redaction** |
| NeMo-Gym | Config leaves under secret-shaped keys; CLI overrides | Key substring list, mask `****` | `nemo_gym/secret_utils.py:25-56` |
| NeMo-Gym | Transport credentials in recorded model I/O | Exact key set (`authorization`, `cookie`, `set_cookie`, `proxy_authorization`, `x_api_key` ...) | `responses_api_agents/osworld_agent/client.py:910-952` |
| NeMo-Gym | URL query strings in span attributes | Drop everything after `?` | `nemo_gym/server_utils.py:545-551` |
| NeMo-Gym | `--env KEY=VALUE` in argv logs | Mask the value | `nemo_gym/sandbox/providers/docker/provider.py:102-112` |
| NeMo-Gym | Grader env values in captured stdout | Exact replace, values >= 8 chars, named after the key | `resources_servers/vibench/app.py:360-372` |
| Harbor | Sensitive env on persist/resume | Key regex `KEY|SECRET|TOKEN(?!S)|PASSWORD|CREDENTIAL|AUTH`; template `${KEY}` or partial mask | `src/harbor/utils/env.py:5-7, 48-92` |
| Harbor | Every file in a trial dir after the run | Exact replace of all sensitive env values, longest first, text files only | `src/harbor/trial/trial.py:1107-1160` |
| Harbor | Secret CLI option values in error messages | Exact replace | `src/harbor/agents/installed/base.py:898-910` |
| Harbor (strands) | Observed model config | Drops keys whose parts are `credential/password/secret/token` or contain `access_key/api_key/private_key` | `src/harbor/agents/installed/strands_runner.py:445-462` |
| repo2rlenv | Build logs | Env values by key name; URL userinfo; `Authorization: Bearer|Basic`. Redact **before** excerpt and size cap; record `redacted: True` | `src/repo2rlenv/tasksmith/build_logs.py:38-80` |
| repo2rlenv | Token in clone `.git/config` | Reset remote URL after clone | `src/repo2rlenv/bootstrap/runner.py:54-56, 230-251` |
| OpenEnv | `user:pass@` in endpoints | Split authority, mask userinfo | `src/openenv/core/llm_client.py:39-45` |
| OpenEnv | Injected env values in server logs | Replace **then** truncate; logs withheld by default | `src/openenv/core/containers/runtime/modal_provider.py:607-618` |
| agentenv | Model key in commands and output | Replace raw, shell-quoted and single-quote-escaped forms; refuse keys < 16 chars | `src/agent_env/providers/sandbox_providers/sail_vm/model_key.py:43-51, 196-208` |
| agentenv | Request URLs in httpx logs | Logging filter strips userinfo, path and query | `packages/agentenv-protocol/src/agentenv_protocol/transfers.py:210-241` |
| NeMo-RL | K8s manifests in logs | `Secret`/`ConfigMap` data and secret-named env values | `infra/nrl_k8s/src/nrl_k8s/_logging.py:28-70` |
| FineEnvs explorer | Owner name and endpoint in public trajectories | Exact replace of known identity strings across every nested value | `tooling/mimo-rl-explorer/app/main.py:346-363`, `tooling/hf-rl-explorer/app/main.py:530-536` |
| verifiers, prime-agent, SWE-smith, EnterpriseOps-Gym, autoresearch | No privacy redaction found. prime-agent only strips ANSI and binary bytes (`crates/pa-daemon/src/user_bash.rs:624-640`). verifiers patches a third-party transcript sanitiser (`verifiers/v1/harnesses/openclaw/harness.py:26`) | — | — |

No reference does PII detection (names, phones, cards, national ids). No reference pseudonymises. All of them
mask known values or secret-shaped keys.

## 2. Defect candidates

### D1. High. Secrets under secret-shaped keys in `data`/`meta` survive (open in WT)
`Redactor.value` (`redact.py:101-108`; WT adds key text only) never looks at the key name. A bare value has no
pattern to match. Connectors put such dicts in `data`: OTel span attributes (`otel_import.py:137`), file_drop extra
columns (`file_drop.py:94`), tool arguments parsed as JSON. Crucible already has the fix in its own core
(`core/config.py:89-108`) but `redact.py` does not call it.
```python
t = Trace(trace_id="t", source=TraceSource(connector="c", ref="r"), events=[TraceEvent(seq=0, kind="action", type="x",
    data={"password": "hunter2hunter2", "headers": {"Authorization": "Basic YWRtaW46aHVudGVyMg==", "Cookie": "sid=abcdef0123456789"},
          "env": {"DB_PASS": "Tr0ub4dor&3"}})])
print(redact(t).events[0].data)   # all four values unchanged
```
Better: NeMo-Gym `secret_utils.py:25-42`, `osworld_agent/client.py:910-952`; Harbor `strands_runner.py:445-462`.
Owner: `redact.py` `Redactor.value`, reuse `core.config.looks_like_secret_key` (but keep `max_tokens`, `author`).

### D2. High. Truncation runs before redaction, and an unterminated PEM is never matched (open in WT)
Connectors cut text to 4000 or 200 chars in `normalize` (`capture_proxy.py:264, 268, 273-274, 318, 327`;
`chat_messages.py:60-65`; `otel_import.py:68, 121`; `interop/harbor/atif_import.py:68-88`). `redact` sees the cut
text. The PEM rule needs both markers (`redact.py:37`). A key that crosses the cut, or a `head` of a key file, is
stored. Verified end to end: 3000 chars of output then a 1.2 KB RSA key in a capture log stores the BEGIN line and
~1000 chars of key body. A Bearer or `sk-` key cut under 20 chars also leaks its first 19 chars.
```python
print(R.text("-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW"))  # unchanged
```
Better: OpenEnv `modal_provider.py:607-618` and repo2rlenv `build_logs.py:59-61` redact first, then truncate.
Owner: every connector's `normalize`. Either redact the raw payload before normalise, or make the PEM rule
`BEGIN ... PRIVATE KEY-----[\s\S]*?(?:-----END ...-----|\Z)`.

### D3. High. Raw text stays at rest with no purge path (open in WT)
- `CaptureProxy._record` appends raw requests and responses, base64 images included, to `<capture_dir>/<session>.jsonl`
  with default umask (`capture_proxy.py:68-70`). The frozen session copy (`claude_code_sessions.py:52-55`) and the
  GitHub cache are also raw. The M3 doc lists the session copy as "Not done" only.
- Store blobs are `0o444` (`core/store/__init__.py:173`) and `index.sqlite` is `0o644` (umask, measured): readable by
  every local user.
- The store has no delete. A trace stored before a pattern fix keeps its leaked version. `TraceStore.get` returns the
  latest (`trace_store.py:21-25`) but `list_docs`/`get_bytes` still serve the old blob.
```python
# after ingest with an old policy, then re-ingest with a better one:
rows = ts.store.list_docs("trace", name=tid); assert len(rows) == 2   # the leaked version is still addressable
```
Better: Harbor `trial.py:1107-1160` rewrites every output file in place after the run; repo2rlenv
`runner.py:230-251` removes the token from `.git/config`. Owner: `capture_proxy.py`, `core/store` (tombstone or
`purge(key)`), `trace_store.py`.

### D4. Med-High. `Authorization: Basic`, cookies and session ids pass (open in WT)
Only `bearer` is matched (`redact.py:38`). `token=` catches `csrftoken=` by luck.
```python
print(R.text("Authorization: Basic YWRtaW46aHVudGVyMmh1bnRlcjI="))       # unchanged
print(R.text("Cookie: sessionid=9f8e7d6c5b4a39281706f5e4d3c2b1a0"))      # unchanged
```
Better: repo2rlenv `build_logs.py:51` (`bearer|basic`); NeMo-Gym `osworld_agent/client.py:910-919` (`cookie`,
`set_cookie`). Owner: `redact.py` `SECRET_PATTERNS`.

### D5. Med. Env-style `*_KEY=` / `*_PASS=` assignments and escaped quotes pass (open in WT)
The assignment rule needs the word right before `=`: `SECRET_KEY=` fails on `_KEY`, `DB_PASS=` has no listed word.
`['\"]?` cannot step over a backslash, so a JSON-escaped value in a tool argument string fails. URL-encoded `%3D` fails.
WT's `{4,}` cuts a quoted password at its first space.
```python
for s in ["SECRET_KEY=django-insecure-8f7d6s5a4q3w2e1r", 'export PASSWORD=\\"hunter2hunter\\"',
          "password%3Dhunter2hunter2", 'password: "correct horse battery"']:
    print(R.text(s))   # first three unchanged; last keeps "horse battery" in WT
```
Better: Harbor `utils/env.py:5-7, 78-92` matches the **key name** (`KEY|SECRET|TOKEN(?!S)|PASSWORD|CREDENTIAL|AUTH`);
agentenv `model_key.py:43-49` parses quoted and escaped shell words. Owner: `redact.py`.

### D6. Med. Credential formats not covered (open in WT)
Bare AWS secret keys, GCP OAuth `ya29.`, Vault `hvs.`, Slack webhooks, Azure AD client secrets without a key name,
Twilio, Shopify, Atlassian, Sentry DSN (WT treats the key as an e-mail). Secrets split by a newline, base64 or
URL-encoded are out of reach for any regex.
```python
for s in ["wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "ya29.a0AfH6SMBx3kLmNoPqRsTuVwXyZ0123456789abcdefghijkl",
          "hvs.CAESIJlWh0BU7e7sVqEXAMPLEtokenXYZ12345", "https://hooks.slack.com/" + "services/T…/B…/<24 chars>"]:
    print(R.text(s))   # all unchanged
```
Better: no reference has a broader list; this is a gap against detect-secrets / gitleaks rule sets. Owner: `redact.py`.

### D7. Med. `with_env_file` misses values and keeps comments (open in WT)
`redact.py:65-69`: the key regex skips `DATABASE_URL`, `*_DSN`, `*CREDENTIAL*`, `*COOKIE*`, `*AUTH*`, `*_PASS`.
An inline comment stays in the value, so the real secret never matches.
```python
open("/tmp/e", "w").write("DATABASE_URL=postgres://u:LongPassw0rd@db/x\nOPENAI_API_KEY=sk-abc123def456 # rotated\n")
print(RedactionPolicy.with_env_file("/tmp/e").secret_values)   # ['sk-abc123def456 # rotated']
```
Better: Harbor `utils/env.py:5-7` adds `CREDENTIAL|AUTH`; NeMo-RL `_logging.py:28` adds `CRED`. Owner: `redact.py`.

### D8. Med. Quadratic e-mail regex (open in WT)
`EMAIL` has no left anchor (`redact.py:41`, WT line ~57). On a long run of local-part characters each start position
rescans the run. Measured: 20k chars 0.18 s, 40k chars 0.68 s. A 1 MB base64 attachment in `data` (chat_messages keeps
full arguments, `chat_messages.py:74`; OTel attributes, `otel_import.py:137`) would take hours and stall ingest.
```python
import time; s = "A" * 40000; t = time.time(); R.text(s); print(time.time() - t)
```
Better: repo2rlenv caps retained streams at 4 MB (`build_logs.py:12, 63-68`). Owner: `redact.py` (anchor with
`(?<![A-Za-z0-9._%+\-])` and cap string length), connectors (drop or hash binary payloads).

### D9. Med. Names and mentions identify people after pseudonymisation (open in WT)
Ids and handles become pseudonyms but text is not rewritten. `github_pr` stores the real git author name in
`data.git_author` (`github_pr.py:195`). Commit text keeps `Signed-off-by: Jane Q. Doe`. Review text keeps `@octocat`.
For the AP-invoice pack, IBANs, bank account numbers, VAT ids and SSNs are not covered at all.
```python
print(R.text("Signed-off-by: Jane Q. Doe <jane@acme.com>"), R.text("@octocat please rebase"), R.text("SSN 123-45-6789"))
```
Better: FineEnvs `mimo-rl-explorer/app/main.py:346-363` replaces each known identity string in every nested value.
Owner: `redact.py` (rewrite known handles, logins and git names in text, from the actors list), `github_pr.py:195`.

### D10. Med. Over-redaction destroys code signal (partly open, WT adds two)
Coding traces are code. Still mangled in WT:
```python
for s in ["token = tokenizer.encode(prompt)", "password = getpass.getpass()", "pwd = os.getcwd()",   # all -> <secret>
          "y = x@self.weights", "git clone git@github.com:acme/repo.git",                          # e-mail pseudonyms
          "range +1 2024-05-01"]:                                                                   # WT phone pseudonym
    print(R.text(s))
```
`pwd = os.getcwd()` and the date-as-phone are new in WT (`pwd` rule, `PHONE`). Hex SHAs, UUIDs and base64 test data
are kept (good). Better: Harbor masks by key name only, never free code. Owner: `redact.py` (require a literal value,
not an identifier followed by `(` or `.`).

### D11. Med. No record of which policy redacted a trace (open in WT)
`Redactor.hits` is counted (`redact.py:85-88`) and thrown away. The trace carries no policy version or pattern-set
hash. After a pattern fix nobody can list which stored traces need re-redaction.
Better: repo2rlenv writes `{"redacted": True, "redacted_bytes": ...}` beside each log (`build_logs.py:70-80`).
Owner: `redact.py` (return hits), `trace_store.py` (put provenance `redaction@<hash>`).

### D12. Med. The gate cannot see the holes above (open in WT)
`m3/gate.py:68-81, 196-206` plants canaries only as free text, only in formats the redactor knows (the doc's own
Finding 1), and checks for the **whole** value as a substring of the dump. It plants nothing in dict keys or values
under secret keys, `source.url`, targets, actor ids, or across the truncation boundary. A partial leak (PEM body,
`glpat-` tail, D2) passes. Owner: `m3/gate.py` (add structured canaries, check 12-char fragments).

### D13. Low. Agent and system handles are pseudonymised (open in WT)
`redact.py:127-128` hashes every handle, so the agent's model id (`capture_proxy.py:335`, `atif_import.py:50`) becomes
`<user:...>`. The model stays in `data.model`, so the loss is small. Owner: `redact.py`.

### D14. Low. Handle pseudonyms are case-sensitive (open in WT)
E-mails are lowercased before hashing, handles and ids are not. `Alice` and `alice` get two pseudonyms and one person
splits into two actors.

### D15. Low. Host coverage (open in WT)
Only `.internal/.corp/.local/.lan/.intra/.ts.net`. Real company hosts (`jenkins.eng.acme.com`), IPv6 ULA (`fd12::1`)
and link-local stay. A policy list of tenant domains is missing.

Fixed in WT (verified at HEAD, re-checked in WT): `trace_id`, `source.url`, targets, refs and dict keys were not
redacted (HEAD `redact.py:111-139`); handles were re-hashed on a second pass (non-idempotent); the public default salt
`"crucible"` (`redact.py:53`) made e-mail pseudonyms reversible by dictionary and `salt` was serialised; URL userinfo
(`postgres://admin:S3cretPass@...` became an e-mail pseudonym, `mongodb+srv://svc:Abc$123xyz@...` leaked `Abc$`);
Azure `AccountKey=`; cards; phones; `threading.local()` and `self.internal` mangled to hosts.

## 3. Features Crucible lacks (ranked)

1. **Key-aware structured redaction** for `data`, `meta` and `outcome.details`. Ref: NeMo-Gym `secret_utils.py:25-42`,
   `osworld_agent/client.py:910-952`. Owner: `redact.py` reusing `core/config.py:96-108`.
2. **Redact before truncate**, or redact the raw payload. Ref: OpenEnv `modal_provider.py:607-618`, repo2rlenv
   `build_logs.py:59-61`. Owner: `base.ingest` / each connector `normalize`.
3. **Raw-at-rest hygiene**: post-run scrub of capture logs and caches, 0600 files, a store purge. Ref: Harbor
   `trial.py:1107-1160`, repo2rlenv `runner.py:230-251`. Owner: `capture_proxy.py`, `core/store`.
4. **Redaction provenance** on each stored trace (policy hash, hit counts). Ref: repo2rlenv `build_logs.py:70-80`.
   Owner: `redact.py`, `trace_store.py`.
5. **Multi-encoding known-value scrub** (shell-quoted, JSON-escaped, URL-encoded, base64 of each `.env` value).
   Ref: agentenv `model_key.py:196-208`. Owner: `redact.py` `Redactor.__init__`.
6. **Key-name `.env` and assignment detection** (`CREDENTIAL`, `AUTH`, `DSN`, `_URL` with userinfo). Ref: Harbor
   `utils/env.py:5-7`. Owner: `redact.py` `with_env_file` and assignment rule.
7. **Known-identity rewrite in text** (logins, git names, display names from the actor list). Ref: FineEnvs
   `mimo-rl-explorer/app/main.py:346-363`. Owner: `redact.py`.
8. **Size bound** on each redacted string. Ref: repo2rlenv `build_logs.py:12, 63-68`. Owner: `redact.py`.
9. **Query-string stripping** for URLs in OTel attributes. Ref: NeMo-Gym `server_utils.py:545-551`. Owner:
   `otel_import.py`.

## 4. Where Crucible is stronger

- **One choke point.** `ingest` always redacts before `put` (`base.py:52`). References scrub at each log site and miss
  new ones.
- **Format patterns, not only known values.** References mask only values they injected. Crucible catches a secret the
  tenant never listed. The gate proves this against the real `.env` (`m3/gate.py:150-152, 209-212`).
- **Linkable pseudonyms.** E-mails, hosts and human actors keep thread structure. No reference pseudonymises; they
  delete. WT uses an HMAC under a 0600 tenant key, as `bench/splits.py:40-41` already does.
- **Whole-document coverage** (WT): every string field, dict keys, actor references in events and signals.
- **The capture proxy keeps the upstream key off the agent and never records headers** (`capture_proxy.py:90-107`).
- **`secret_values` cannot be serialised** (`redact.py:57`), and WT adds the same for `salt`.
- **A canary gate with a real-secret check** exists at all. No reference has a planted-canary redaction test.
- **Some PII in WT** (Luhn-checked cards, phones). No reference touches PII.
