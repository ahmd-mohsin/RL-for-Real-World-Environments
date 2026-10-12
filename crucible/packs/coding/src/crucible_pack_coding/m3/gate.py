"""M3 exit gate (ARCHITECTURE M3), run without any API call:

  * >= 30 public GPU-migration PR traces (from git history: ``discover_git``), our task 01-04 construction transcript,
    and every recorded frontier run are ingested; so are the runs' CI reports, the policy documents and the toy
    ticket logs; 100% of stored traces are schema-valid;
  * redaction removes 100% of planted canaries (every API-key format, e-mails, internal hosts, private IPs) planted
    in raw records of each connector kind, and no value from the workspace ``.env`` survives into any stored trace;
  * re-ingest is idempotent: a second ingest of every source yields the same content hashes and stores nothing.

``python -m crucible_pack_coding.m3.gate`` writes ``docs/modules/reports/m3_gate.json``.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from crucible.connectors.base import SourceRef, ingest
from crucible.connectors.capture_proxy import CaptureLogConnector
from crucible.connectors.file_drop import FieldMap, FileDropConnector
from crucible.connectors.redact import RedactionPolicy, Redactor
from crucible.connectors.trace_store import TraceStore
from crucible.core.schema.trace import Trace
from crucible.core.store import Store
from crucible_pack_coding import PACK_ROOT
from crucible_pack_coding.connectors.agent_runs import AgentRunConnector
from crucible_pack_coding.connectors.ci_logs import CILogConnector
from crucible_pack_coding.connectors.claude_code_sessions import ClaudeCodeSessionConnector
from crucible_pack_coding.connectors.github_pr import GitHubPRConnector
from crucible_pack_coding.connectors.policy_docs import PolicyDocConnector

CRUCIBLE = PACK_ROOT.parent.parent
DATA = CRUCIBLE / "data"
TASKS = CRUCIBLE.parent / "migration_tasks"
ENV_FILE = CRUCIBLE.parent / ".env"
REPORT = CRUCIBLE / "docs" / "modules" / "reports" / "m3_gate.json"
PCL = "PointCloudLibrary/pcl"
NOW = "2026-10-09T12:00:00+00:00"  # outcome windows are computed against this instant
# PR titles of GPU-migration work (toolkit / architecture / API moves), matched on merges that touch gpu/ or cuda/
GPU_MIGRATION = (r"\bcuda\b|thrust|\bcub\b|nvcc|\bsm_\d|compute capabilit|texture|\bwarp|__shfl|deprecat|\bc\+\+1[47]\b|toolkit"
                 r"|kepler|maxwell|pascal|volta|turing|ampere|hopper|blackwell|\bport(?:s|ed|ing)?\b|moderni[sz]")
PCL_POLICY = ["CONTRIBUTING.md", "doc/advanced/content/pcl_style_guide.rst", ".clang-format", ".clang-tidy", ".github/CODEOWNERS", "CODEOWNERS"]
TICKET_FIELDS = FieldMap(family="toy/support_ticket", extra=["status", "team", "score", "amount", "rule", "sla", "plan"])  # what the M4 toy mapper reads (CSAT scores were dropped)


def sources() -> list[tuple[str, Any, list[SourceRef]]]:
    gh = GitHubPRConnector(DATA / "cache" / "github", DATA / "cache" / "git", now=NOW, offline=True)
    prs = list(gh.discover_git({"repo": PCL, "paths": ["gpu", "cuda"], "title_re": GPU_MIGRATION}))
    sess = ClaudeCodeSessionConnector(DATA / "cache" / "sessions")
    frozen = sorted(p for p in (DATA / "cache" / "sessions").glob("*") if p.is_dir())
    sess_refs = [SourceRef(connector=sess.name, key=p.name, meta={"dir": str(p), "session": p.name}) for p in frozen]
    runs = AgentRunConnector(TASKS)
    ci = CILogConnector(TASKS)
    pol = PolicyDocConnector()
    pol_refs = list(pol.discover({"git_dir": DATA / "cache" / "git" / "pcl.git", "rev": "master", "repo": PCL, "paths": PCL_POLICY, "privacy": "public"}))
    pol_refs += list(pol.discover({"root": TASKS, "glob": "*/MIGRATION_LOG.md"})) + list(pol.discover({"root": TASKS, "glob": "*/DESIGN_NOTES.md"}))
    toy = FileDropConnector(TICKET_FIELDS)
    toy_refs = list(toy.discover({"glob": str(CRUCIBLE / "packs" / "_toy" / "data" / "ticket_logs" / "*.jsonl")}))
    return [("github_pr", gh, prs), ("claude_code_sessions", sess, sess_refs), ("agent_runs", runs, list(runs.discover())),
            ("ci_logs", ci, list(ci.discover({"glob": "*/runs/*/grade_report.json"}))), ("policy_docs", pol, pol_refs), ("file_drop", toy, toy_refs)]


# ---------------------------------------------------------------- redaction canaries
def canaries() -> dict[str, str]:
    """One value per format the redactor must remove, generated so no real-looking credential sits in the source."""
    return {
        "openai": "sk-proj-" + "Qx7" * 14, "openrouter": "sk-or-v1-" + "9f3a" * 16, "anthropic": "sk-ant-api03-" + "Zr2" * 15,
        "github": "ghp_" + "Gh5" * 12, "github_pat": "github_pat_" + "Pt4" * 18, "huggingface": "hf_" + "Hf8" * 12,
        "aws": "AKIA" + "Q2W3E4R5T6Y7U8I9", "slack": "xoxb-" + "4815162342-abcdEFGHij", "gitlab": "glpat-" + "Gl3" * 6 + "ab",
        "google": "AIza" + "Go0" * 11 + "gl", "jwt": "eyJ" + "hd" * 10 + ".eyJ" + "pl" * 10 + "." + "sg" * 10,
        "dockerhub": "dckr_pat_" + "Dk4" * 9, "runpod": "rpa_" + "Rp7" * 14, "bearer": "Bearer " + "Br3." * 8 + "x",
        "password": "password=" + "Corr3ctH0rse!", "email": "canary.person@example-enterprise.com",
        "host": "ci-runner-7.build.corp", "tailnet": "gpu-box.tail9f9f.ts.net", "ip": "10.44.55.66", "ip_tailscale": "100.101.102.103",
        # M3 audit (2026-10-12): formats the redactor missed
        "url_userinfo": "https://deploy:Ur1P4ssw0rdX@git.example.com/r.git", "dsn": "postgres://svc:Ds5nPassQ@db.example.com/prod",
        "short_password": "pwd=Hn2x9", "azure": "AccountKey=" + "Az9" * 20 + "==", "card": "4111 1111 1111 1111", "phone": "+1 415 555 0132",
        "basic_auth": "Authorization: Basic " + "QmFzaWM6" * 4, "cookie": "Cookie: sessionid=" + "c0ok1e" * 5,
        # longer than any connector's text cap, so a cut lands inside the body (the END line never reaches the redactor)
        "private_key": "-----BEGIN RSA PRIVATE KEY-----\n" + "Pk9Rs4" * 1200 + "\n-----END RSA PRIVATE KEY-----",
        # structured: values under credential-named keys, with no recognizable format of their own
        "field_password": "Wq3rTy7uIo", "field_cookie": "zz91aa82bb73cc64",
    }


_STRUCTURED = {"field_password", "field_cookie"}  # planted as column values, not in the free-text blob
_SECRET_PART = {"basic_auth": lambda v: v.split(" ", 2)[2], "cookie": lambda v: v.split("=", 1)[1],
                "private_key": lambda v: v.split("\n")[1],"password": lambda v: v.split("=", 1)[1], "short_password": lambda v: v.split("=", 1)[1], "azure": lambda v: v.split("=", 1)[1],
                "bearer": lambda v: v.split(" ", 1)[1], "url_userinfo": lambda v: v.split(":")[2].split("@")[0],
                "dsn": lambda v: v.split(":")[2].split("@")[0], "phone": lambda v: v[-8:]}


def _value(k: str, v: str) -> str:
    """The part of a canary that must be gone (the host of a URL may stay)."""
    return _SECRET_PART.get(k, lambda x: x)(v)


def _leaked(secret: str, dumped: str, width: int = 12) -> bool:
    """Any 12-character piece of a secret counts: a key cut by a text cap, or a token half matched, still leaks."""
    if len(secret) <= width:
        return secret in dumped
    return any(secret[i:i + width] in dumped for i in range(0, len(secret) - width + 1, 3))


def plant(work: Path) -> list[tuple[str, Any, list[SourceRef]]]:
    """Raw records of each connector kind with every canary planted where real data would carry it."""
    c = canaries()
    blob = " ".join(v for k, v in c.items() if k not in _STRUCTURED)
    work.mkdir(parents=True, exist_ok=True)
    out: list[tuple[str, Any, list[SourceRef]]] = []
    # file_drop: a ticket log
    (work / "tickets.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"case": "C1", "id": "1", "time": "2026-01-01T00:00:00Z", "actor": c["email"], "type": "opened", "text": f"creds: {blob}"},
        {"case": "C1", "id": "2", "time": "2026-01-01T01:00:00Z", "actor": "ops", "type": "note", "text": "rotated", "team": c["host"], "reply_to": "1",
         "db_password": c["field_password"], "headers": {"Cookie": c["field_cookie"]}}]))
    fd = FileDropConnector(FieldMap(extra=["team", "db_password", "headers"]))
    out.append(("file_drop", fd, list(fd.discover({"glob": str(work / "tickets.jsonl")}))))
    # agent run: a real trajectory with canaries in the task text and in a tool output
    src = next(iter(sorted(TASKS.glob("*/runs/*/trajectory.json"))), None)
    if src is not None:
        run_dir = work / "taskX" / "runs" / "canary-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        traj = json.loads(src.read_text())
        msgs = traj.get("messages") or []
        for role in ("user", "tool"):  # the task text and a tool output
            m = next((m for m in msgs if m.get("role") == role), None)
            if m is not None and isinstance(m.get("content"), str):
                m["content"] = f"{m['content']}\n{blob}"
            elif m is not None and isinstance(m.get("content"), list):
                m["content"].append({"type": "text", "text": blob})
        (run_dir / "trajectory.json").write_text(json.dumps(traj))
        (run_dir / "model.patch").write_text(f"diff --git a/x b/x\n+token = '{c['github']}'\n")
        ar = AgentRunConnector(work)
        out.append(("agent_runs", ar, list(ar.discover())))
    # capture proxy: one exchange
    cap = work / "capture"
    cap.mkdir(exist_ok=True)
    (cap / "s1.jsonl").write_text(json.dumps({"api": "openai", "digest": "d", "request": {"messages": [{"role": "user", "content": blob}]},
                                              "response": {"choices": [{"message": {"content": f"I used {c['openai']}", "tool_calls": [
                                                  {"id": "t", "function": {"name": "bash", "arguments": json.dumps({"cmd": f"curl -H 'Authorization: Bearer {c['anthropic']}' http://{c['host']}"})}}]}}]}}) + "\n")
    cl = CaptureLogConnector()
    out.append(("capture_proxy", cl, list(cl.discover({"dir": cap}))))
    # CI log
    (work / "ci").mkdir(exist_ok=True)
    (work / "ci" / "pytest.log").write_text(f"tests/t.py::test_x FAILED\nFAILED tests/t.py::test_x - ConnectionError {c['ip']} {c['jwt']}\n=== 1 failed in 1s ===\n")
    ci = CILogConnector(work / "ci")
    out.append(("ci_logs", ci, list(ci.discover({"glob": "*.log"}))))
    # policy doc
    (work / "policy").mkdir(exist_ok=True)
    (work / "policy" / "RUNBOOK.md").write_text(f"# Runbook\n\nYou must never paste {c['huggingface']} or {c['password']} into tickets. Ask {c['email']}.\n")
    pol = PolicyDocConnector()
    out.append(("policy_docs", pol, list(pol.discover({"root": work / "policy", "glob": "*.md"}))))
    return out


def env_secrets() -> dict[str, str]:
    """name -> value of every credential in the workspace .env (values are compared, never printed or stored)."""
    known = set(RedactionPolicy.with_env_file(ENV_FILE).secret_values)
    out = {}
    for line in ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []:
        k, _, v = line.removeprefix("export ").partition("=")
        v = v.strip().strip("'\"")
        if v in known:
            out[k.strip()] = v
    return out


def pattern_misses(secrets: dict[str, str]) -> list[str]:
    """Credentials the format patterns alone miss as bare values (the known-value list still removes them; this says
    whether a tenant that forgot to list a secret would leak it)."""
    r = Redactor(RedactionPolicy())
    return sorted(k for k, v in secrets.items() if v in r.text(f" {v} "))


# ---------------------------------------------------------------- the gate
def run() -> dict[str, Any]:
    t_start = time.time()
    root = DATA / "m3"
    if root.exists():
        shutil.rmtree(root)
    ts = TraceStore(Store(root / "store"))
    policy = RedactionPolicy.with_env_file(ENV_FILE)  # what a tenant runs: format patterns + its known secret values
    per: dict[str, dict[str, Any]] = {}
    keys: dict[str, list[str]] = {}
    for name, conn, refs in sources():
        t0 = time.time()
        keys[name] = ingest(conn, refs, ts, policy=policy)
        per[name] = {"refs": len(refs), "traces": len(set(keys[name])), "seconds": round(time.time() - t0, 1)}
    rows = ts.store.list_docs("trace")
    invalid = []
    traces: list[Trace] = []
    for r in rows:
        try:
            traces.append(Trace.model_validate_json(ts.store.get_bytes(r.key)))
        except Exception as e:  # noqa: BLE001 -- every failure is reported, none stops the gate
            invalid.append({"key": r.key, "name": r.name, "error": str(e)[:300]})
    by_conn: dict[str, list[Trace]] = {}
    for t in traces:
        by_conn.setdefault(t.source.connector, []).append(t)

    # coverage
    prs = by_conn.get("github_pr", [])
    sess = by_conn.get("claude_code_sessions", [])
    covered = sorted({k for t in sess for k in (t.outcome.details.get("events_per_task") or {})})
    run_dirs = sorted(f"{p.parent.parent.parent.name}/{p.parent.name}" for p in TASKS.glob("*/runs/*/trajectory.json"))
    runs = by_conn.get("agent_runs", [])
    for name, stats in per.items():
        stats["events"] = sum(len(t.events) for t in by_conn.get(name, []))
    per["github_pr"]["outcomes"] = _labels(prs)
    per["github_pr"]["post_merge"] = {
        "reverted": sum(1 for t in prs if any(e.type == "post_merge_commit" and e.data.get("is_revert") for e in t.events)),
        "fix_touching_pr_lines": sum(1 for t in prs if any(e.type == "post_merge_commit" and e.data.get("fixlike") and e.data.get("touches_pr_commits") for e in t.events)),
        "origin": _count(t.outcome.details.get("origin") for t in prs)}
    per["agent_runs"]["outcomes"] = _labels(runs)
    per["ci_logs"]["outcomes"] = _labels(by_conn.get("ci_logs", []))
    per["policy_docs"]["rules"] = sum(1 for t in by_conn.get("policy_docs", []) for e in t.events if e.type in ("policy_rule", "config_rule"))
    per["claude_code_sessions"]["events_per_task"] = {k: v for t in sess for k, v in (t.outcome.details.get("events_per_task") or {}).items()}

    # redaction: planted canaries (separate store) and the workspace's real secrets
    work = root / "canary"
    cts = TraceStore(Store(root / "canary_store"))
    planted: dict[str, dict[str, Any]] = {}
    for name, conn, refs in plant(work / "raw"):
        ks = ingest(conn, refs, cts, policy=RedactionPolicy())  # canaries must fall to the format patterns alone
        dumped = "\n".join(cts.store.get_bytes(k).decode() for k in ks)
        present = sorted(k for k, v in canaries().items() if _value(k, v) in _raw(conn, refs))
        leaked = sorted(k for k, v in canaries().items() if _value(k, v) in _raw(conn, refs) and _leaked(_value(k, v), dumped))
        planted[name] = {"traces": len(ks), "planted": present, "leaked": leaked}
    secrets = env_secrets()
    all_dump = [ts.store.get_bytes(r.key).decode() for r in rows]
    env_leaks = sorted({k for k, v in secrets.items() for d in all_dump if v in d})
    frozen = [p.read_text(errors="replace") for p in sorted((DATA / "cache" / "sessions").glob("*/*.jsonl"))]
    env_in_raw = sorted({k for k, v in secrets.items() for text in frozen if v in text})
    misses = pattern_misses(secrets)

    # idempotency: ingest every source again into the same store
    before = len(ts.store.list_docs("trace"))
    t0 = time.time()
    again = {name: ingest(conn, refs, ts, policy=policy) for name, conn, refs in sources()}
    after = len(ts.store.list_docs("trace"))
    same = all(again[n] == keys[n] for n in keys)
    # the harder cases (M3 audit): a connector with its own clock (the fetch time was hashed), and exports written in
    # another row order (equal times hashed by export order)
    fresh = GitHubPRConnector(DATA / "cache" / "github", DATA / "cache" / "git", offline=True)
    gh_refs = next(r for n, _c, r in sources() if n == "github_pr")[:5]
    fresh_clock = ingest(fresh, gh_refs, ts, policy=policy) == keys["github_pr"][:5]
    shuffled = root / "reordered"
    shuffled.mkdir(parents=True, exist_ok=True)
    for f in sorted((CRUCIBLE / "packs" / "_toy" / "data" / "ticket_logs").glob("*.jsonl")):
        lines = [x for x in f.read_text().splitlines() if x.strip()]
        (shuffled / f.name).write_text("\n".join(reversed(lines)) + "\n")
    toy = FileDropConnector(TICKET_FIELDS)
    reordered = ingest(toy, list(toy.discover({"glob": str(shuffled / "*.jsonl")})), ts, policy=policy) == keys["file_drop"]
    after = len(ts.store.list_docs("trace"))

    checks = {
        "pr_traces>=30": len(prs) >= 30,
        "construction_transcript_tasks_01_04": all(any(k.startswith(f"task0{i}") for k in covered) for i in (1, 2, 3, 4)),
        "every_frontier_run": sorted(t.outcome.details.get("task", "") + "/" + t.outcome.details.get("run", "") for t in runs) == run_dirs,
        "schema_valid_100pct": not invalid and len(traces) == len(rows),
        "planted_canaries_removed_100pct": all(not v["leaked"] and v["planted"] for v in planted.values())
                                           and set().union(*(v["planted"] for v in planted.values())) == set(canaries()),
        "env_secrets_absent": not env_leaks,
        "env_secrets_caught_by_patterns_alone": not misses,
        "reingest_idempotent": same and before == after,
        "reingest_idempotent_fresh_clock_and_reordered": fresh_clock and reordered,
    }
    return {
        "gate": "M3", "pass": all(checks.values()), "checks": checks, "now": NOW, "github_api_calls": 0,
        "traces": len(traces), "invalid": invalid, "sources": per, "frontier_runs": run_dirs,
        "redaction": {"canary_formats": sorted(canaries()), "planted": planted, "env_secrets": sorted(secrets),
                      "env_secrets_in_raw_transcript": env_in_raw, "env_secrets_in_store": env_leaks, "env_secrets_missed_by_patterns": misses},
        "idempotency": {"docs_before": before, "docs_after": after, "same_keys": same, "fresh_clock": fresh_clock, "reordered_export": reordered,
                        "seconds": round(time.time() - t0, 1)},
        "seconds": round(time.time() - t_start, 1),
    }


def _labels(ts: list[Trace]) -> dict[str, int]:
    return _count(t.outcome.label for t in ts)


def _count(xs) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return dict(sorted(out.items()))


def _raw(conn: Any, refs: list[SourceRef]) -> str:
    """Everything the connector fetched, serialized: the canaries must be in here for the check to mean anything."""
    return "\n".join(json.dumps(conn.fetch(r).payload, default=str) for r in refs)


def main() -> None:
    out = run()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("pass", "checks", "traces", "sources", "redaction", "idempotency", "seconds")}, indent=1))
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
