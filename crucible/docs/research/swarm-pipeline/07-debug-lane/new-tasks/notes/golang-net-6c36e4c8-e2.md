# golang/net task: a new task, kept for training (after three rebuilds)

**Task:** `db-go-golang-net-6c36e4c8-e2` (Uzair, 2026-10-28)
**Repository:** golang/net (new: never used for a debug task before)
**Result:** kept for training. The bugs are not linked: the user relaxed the link rule for this task on 2026-10-28.

## What the task is

The agent gets **golang.org/x/net** (Go's extra networking packages: HTTP/2, QUIC, DNS messages, HTML parsing, XSRF tokens)
with five real bug fixes undone, all from May-October 2026, after the student was released. It also gets a bug report
written like a message from a user. It has **300 tool calls** to find and fix every bug. Hidden tests check each bug
separately. The five bugs are in five different packages.

## The bugs

| # | Bug, as the user sees it | Package | Student fixed it |
|---|---|---|---|
| m01 | QUIC stream returns corrupted data when the peer resends data that overlaps what was already received | quic | 7/8 |
| m02 | `http2.ConfigureServer` no longer offers HTTP/2 ("h2") on the server's TLS settings | http2 | 8/8 |
| m03 | DNS SVCB/HTTPS records with parameters out of order or repeated are accepted and written | dns/dnsmessage | 5/8 |
| m04 | XSRF tokens collide when the user or action starts or ends with a colon | xsrftoken | 7/8 |
| m05 | `html.Render` turns escaped text inside some foster-parented MathML/SVG content back into a live `<img onerror=…>` | html | 7/8 |

## How it got here

| Version | Bugs | What happened | Change |
|---|---|---|---|
| v1 `61a78cba` | http2 trailer limits + ALPN + transport (ALPN and transport linked) | student 0/3 | the trailer bug's milestone ran through `TestLegacy` (the whole http2 suite again under another build tag), which flaked under load (S10), and every model wrote `"Trailer"` with a capital T |
| v3 `6c36e4c8` | dns SVCB + ALPN + transport (linked) | student 0/6, Qwen 0/2, DeepSeek-1017 0/2 | the transport bug is a trap: every model copies the old "add h2 to NextProtos" recipe from the legacy file in the same package, which breaks Go 1.27's net/http. Only DeepSeek-1017 found the real fix once, and that attempt tried to download Go's source with curl (network gate) |
| e1 | dns + ALPN + html render, transport bug left fixed (no link) | student 3/3: too easy | add two bugs |
| **e2** | **e1 + quic overlapping data + xsrftoken collisions** | **kept** | |

## Results at 300 tool calls

| Quick look | |
|---|---|
| Student | 1/3 |
| Qwen3.5-397B | 1/1 |

| Full run | |
|---|---|
| Student full passes | **4/8 (50%)**, inside the 10-60% training band |
| Average progress (0 = nothing fixed, 1 = all fixed) | 0.85 |
| Attempts that ran out of tool calls | 0 (all 8 stopped on their own, 198 calls on average) |
| Failures on the skill the task trains | **4/4** (all "skill gap": declared done with one or more bugs left) |
| Qwen3.5-397B | 2/2 full passes |
| Verdict | **train (kept)** |
| Cost | quick look $7.87, full run $24.48 (earlier versions: about $28) |

## Checks before measuring

Lint 0 findings; gates S6, S5, S8, S3 and S10 all pass; no hidden test or issue number in the start code; no changelog
(golang/net has none); the old version of the html test that stays in the start code is a pre-existing case that passes
with the fix.

## Where things are

- Task: `crucible/data/debug-uzair/gen/store/train/tasks/db-go-golang-net-6c36e4c8-e2/`
- Bug report: `crucible/data/debug-uzair/prompts/db-go-golang-net-6c36e4c8-e2.json`
- Gates: `crucible/data/debug-uzair/gates/db-go-golang-net-6c36e4c8-e2.json`
- Results: `crucible/data/targeting/uzair-gnet-e2-quick-*`, `crucible/data/targeting/uzair-gnet-e2-full/report.json`
