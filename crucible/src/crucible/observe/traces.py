"""Agent traces as Weave calls, written locally first (swarm-pipeline/11 §5).

**Why a local span file.** Weave (0.53) has no offline mode: ``weave.init`` logs in to W&B and asks the trace server for
its version before it creates a client, and ``WEAVE_DISABLED`` drops the calls instead of keeping them. Its write-ahead
log (``WEAVE_ENABLE_WAL``) only exists behind an authenticated client. So every episode is written as **Weave-shaped
spans** in ``<observe root>/weave/<project>/spans.jsonl`` (one JSON line per call: ``id``, ``trace_id``, ``parent_id``,
``op``, ``display_name``, ``inputs``, ``output``, ``attributes``, ``started_at``, ``ended_at``), and
``observe.weave_sync`` replays them through ``WeaveClient.create_call`` / ``finish_call`` once a key exists, with the
same call ids (``_call_id_override``) and timestamps, so a link written today (``naming.call_url``) opens the trace
after the sync.

**One episode** (an attempt directory as the runtime wrote it: ``record.json``, ``calls.json``,
``trajectory.agent.json``, ``trace.json``, ``progress.jsonl``, ``rollout.json``, ``artifacts/model.patch``) is:

- the root call ``crucible_episode``: inputs (task, prompt, model, arm, seed), output (reward, passed, Φ final, stop,
  masked and its infrastructure reason, the patch, the final message), attributes ``crucible.*`` (task, model, arm,
  run, reward, gates, stop reason, masked/infra, Φ, milestones lost, cost, tokens, harness): filterable in Weave;
- one ``crucible_turn`` child per model call: inputs = a summary of the prompt messages (how many, the latest tool
  results or the request, prompt and cached tokens), output = the model text and its tool calls, plus ``phi``: Φ after
  the turn, ΔΦ, the milestones gained and lost, and the turn's credit (the rollout's per-turn advantage, M6);
- one ``tool_<function>`` grandchild per tool call: the arguments and the result (truncated to ``max_chars``), ok/error,
  and Φ at that step;
- ``crucible_trace_audit`` (``add_audit``): the trace audit's verdict on the episode (first wrong step, cause label,
  verbatim quote, mode, routed owner), added when the audit runs, under the same root.

The full text (every message, result and the patch, token ids stripped) is ``full/<episode>.json``, logged as a W&B
artifact by the run that exported the episode; truncated fields say so. Everything is redacted
(``observe.redact.Scrubber``) before it is written. Timestamps inside an episode are reconstructed from the record's
start and the model call latencies (``attributes.crucible.timing = "reconstructed"``).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html as _html
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crucible.observe.naming import call_url, slug
from crucible.observe.redact import Scrubber, default_scrubber, truncate

SCHEMA = "crucible.weave_span/v1"
STRIP_KEYS = {"prompt_token_ids", "completion_token_ids", "logprobs", "token_ids", "tokens"}


def uuid7(ts_ms: int, key: str) -> str:
    """A UUIDv7 (Weave's call id format) with ``ts_ms`` as its time and 74 bits from ``sha256(key)``: deterministic, so
    a re-export gives the same ids and a link stays valid."""
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:10], "big")
    rand_a, rand_b = (h >> 62) & 0xFFF, h & ((1 << 62) - 1)
    n = ((ts_ms & ((1 << 48) - 1)) << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    x = f"{n:032x}"
    return f"{x[:8]}-{x[8:12]}-{x[12:16]}-{x[16:20]}-{x[20:]}"


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_time(s: Any) -> dt.datetime | None:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(str(s))
    except ValueError:
        return None


def _read(p: Path, default: Any = None) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def _jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(errors="replace").splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    return out


def strip_tokens(x: Any) -> Any:
    """Token-id and logprob lists removed (they are training data, not something a person reads)."""
    if isinstance(x, dict):
        return {k: strip_tokens(v) for k, v in x.items() if k not in STRIP_KEYS}
    if isinstance(x, list):
        return [strip_tokens(v) for v in x]
    return x


def op_name(s: str) -> str:
    """A Weave op name (Weave refs allow ``[A-Za-z0-9_-]``)."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")[:120] or "op"


@dataclass
class Span:
    id: str
    trace_id: str
    parent_id: str | None
    op: str
    display_name: str
    inputs: dict[str, Any]
    output: Any
    attributes: dict[str, Any]
    started_at: str
    ended_at: str
    project: str = ""
    episode: str = ""
    exception: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {"schema": SCHEMA, **self.__dict__}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Span:
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[arg-type]


def attempt_dirs(root: Path | str) -> list[Path]:
    """Every scored attempt under a runtime run root (``<run>/samples/<sample>/a<n>/record.json``); of a retried sample
    only the last attempt (the one that counts)."""
    out = []
    for rec in sorted(Path(root).glob("**/samples/*/a*/record.json")):
        d = rec.parent
        if d.name[1:].isdigit() and (d.parent / f"a{int(d.name[1:]) + 1}" / "record.json").exists():
            continue
        out.append(d)
    return out


def trail_dir(adir: Path, rec: Mapping[str, Any] | None = None) -> Path:
    """The attempt that holds the agent's trail: ``adir``, or for a re-graded attempt (``Runner.regrade``: stop
    ``regraded``, ``meta.artifacts.regraded_from``) its source attempt (``factory.traceaudit.regrade_source``, the same
    rule: this module imports no factory code)."""
    meta = (rec if rec is not None else (_read(adir / "record.json", {}) or {})).get("meta") or {}
    name = (meta.get("artifacts") or {}).get("regraded_from")
    if meta.get("stop") == "regraded" and name and str(name) != adir.name and (adir.parent / str(name)).is_dir():
        return adir.parent / str(name)
    return adir


def episode_key(adir: Path, rec: Mapping[str, Any] | None = None) -> str:
    """``<run id>/<sample>/<attempt>``: the trace audit's bundle id for the same attempt (``traceaudit.discover``)."""
    run = (rec or {}).get("run_id") or adir.parents[2].name
    return f"{run}/{adir.parent.name}/{adir.name}"


def _gates(rec: Mapping[str, Any]) -> dict[str, Any]:
    meta = rec.get("meta") or {}
    g = {k: bool(v) for k, v in (meta.get("gates") or {}).items()}
    kinds = meta.get("gate_kinds") or {}
    gate_only = {k: v for k, v in g.items() if kinds.get(k) == "gate"}
    return {"all": g, "gates_passed": all(gate_only.values()) if gate_only else None, "gate_pass_rate": (sum(gate_only.values()) / len(gate_only)) if gate_only
            else None, "outcome_milestones": {k: v for k, v in g.items() if kinds.get(k) == "outcome"}}


def episode_spans(adir: Path | str, *, project: str, arm: str = "", run: str = "", model: str | None = None, attrs: Mapping[str, Any] | None = None,
                  scrubber: Scrubber | None = None, max_chars: int = 2000) -> tuple[list[Span], dict[str, Any], dict[str, Any]]:
    """``(spans, index row, full document)`` of one attempt directory (module doc)."""
    s = scrubber or default_scrubber()
    d = Path(adir)
    rec = _read(d / "record.json", {}) or {}
    meta = rec.get("meta") or {}
    src = trail_dir(d, rec)  # a re-graded attempt: the grade is here, the agent's trail in its source attempt
    prog = meta.get("progress") or (((_read(src / "record.json", {}) or {}).get("meta") or {}).get("progress") if src != d else None) or {}
    key = episode_key(d, rec)
    calls_doc = _read(src / "calls.json", {}) or {}
    calls = [c for v in (calls_doc.values() if isinstance(calls_doc, dict) else []) for c in (v if isinstance(v, list) else []) if isinstance(c, dict)]
    traj = _read(src / "trajectory.agent.json", {}) or {}
    tsteps = traj.get("steps") or []
    trace = _read(src / "trace.json", {}) or {}
    rows = _jsonl(src / "progress.jsonl")
    rollout = _read(src / "rollout.json", {}) or {}
    patch_p = d / "artifacts" / "model.patch"
    patch = patch_p.read_text(errors="replace") if patch_p.exists() else ""
    model = model or traj.get("agent", {}).get("model_name") or (calls[0].get("model") if calls else None) or rec.get("policy")
    prompt = next((str(x.get("message") or "") for x in tsteps if x.get("source") == "user"), "")

    # tool calls with full results (trace.json), joined to their turn (calls.json actions) and to Φ (progress.jsonl)
    tools: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for e in trace.get("events") or []:
        dd = e.get("data") or {}
        if e.get("type") == "tool_call" and (e.get("actor") or "agent") != "system":
            tid = str(dd.get("id") or f"call{len(order) + 1}")
            tools[tid] = {"id": tid, "i": len(order) + 1, "function": str(dd.get("function") or ""), "arguments": dd.get("arguments"), "result": "", "ok": True,
                          "error": None}
            order.append(tid)
        elif e.get("type") == "tool_result" and str(dd.get("id")) in tools:
            tools[str(dd.get("id"))].update(result=str(e.get("text") or ""), ok=bool(dd.get("ok", True)), error=dd.get("error"), fault=dd.get("fault"))
    phi_of = {str(r.get("tool_call_id")): r for r in rows if r.get("tool_call_id")}
    adv = {str(t.get("call_id")): t.get("advantage") for t in rollout.get("turns") or [] if t.get("call_id")}
    msg_of: dict[str, list[str]] = {}
    for x in tsteps:
        cid = (x.get("extra") or {}).get("call_id")
        if x.get("source") == "agent" and cid and str(x.get("message") or "").strip():
            msg_of.setdefault(str(cid), []).append(str(x["message"]))
    final_message = next((str(x.get("message") or "") for x in reversed(tsteps) if x.get("source") == "agent" and str(x.get("message") or "").strip()), "")

    # timing: the record's start, then each model call's latency and an even share of the remaining agent time per tool
    t0 = _parse_time(rec.get("started_at")) or dt.datetime.fromtimestamp((d / "record.json").stat().st_mtime if (d / "record.json").exists() else 0, dt.UTC)
    agent_s = float((rec.get("timing") or {}).get("agent") or 0.0)
    lat = sum(float(c.get("latency_s") or 0.0) for c in calls)
    n_tools = max(1, len(order))
    tool_s = max(0.0, agent_s - lat) / n_tools if agent_s else 0.05
    t_end_all = _parse_time(rec.get("ended_at"))
    ts_ms = int(t0.timestamp() * 1000)
    root_id = uuid7(ts_ms, f"{project}|{key}|root")
    masked = bool(rec.get("mask_sample")) or rec.get("reward") is None
    reward = rec.get("reward")
    passed = (not masked) and reward is not None and float(reward) >= 1.0 - 1e-9
    gates = _gates(rec)
    usage = rec.get("usage") or {}
    attrs_c = {"task": rec.get("task"), "model": model, "policy": rec.get("policy"), "arm": arm, "run": run or rec.get("run_id"), "episode": key,
               "reward": reward, "passed": passed, "gates": gates["all"], "gates_passed": gates["gates_passed"], "stop": meta.get("stop"),
               "masked": masked, "infra": rec.get("failure") if masked else None, "failure": rec.get("failure"), "phi0": prog.get("phi0"),
               "phi_final": prog.get("phi_final"), "milestones_lost": prog.get("milestones_lost"), "cost_usd": rec.get("cost_usd"),
               "tokens": {"input": usage.get("input_tokens"), "cached": usage.get("cache_read_tokens"), "output": usage.get("output_tokens")},
               "turns": len(calls), "tool_calls": len(order), "harness": meta.get("harness") or (rec.get("sample") or {}).get("harness"),
               "seed": rec.get("seed"), "timing": "reconstructed", "record_dir": str(d), **dict(attrs or {})}
    full_name = f"full/{slug(key.replace('/', '__'), 160)}.json"
    spans: list[Span] = []
    t = t0
    phi_now = prog.get("phi0")
    prev_results: list[dict[str, Any]] = []
    n_msgs = 1
    for k, c in enumerate(calls):
        cid = str(c.get("call_id") or f"agent:c{k + 1}")
        tids = [a[5:] for a in c.get("actions") or [] if isinstance(a, str) and a.startswith("tool:") and a[5:] in tools]
        t_start = t
        t_model_end = t_start + dt.timedelta(seconds=float(c.get("latency_s") or 0.0))
        turn_id = uuid7(int(t_start.timestamp() * 1000), f"{project}|{key}|turn|{k}")
        last_in = ("; ".join(f"{r['function']}: {r['result']}" for r in prev_results) if prev_results else prompt)
        u = c.get("usage") or {}
        tool_spans: list[Span] = []
        gained: list[str] = []
        lost: list[str] = []
        dphi = 0.0
        tt = t_model_end
        for tid in tids:
            tl = tools[tid]
            pr = phi_of.get(tid) or {}
            if pr.get("phi") is not None:
                phi_now = pr["phi"]
            dphi += float(pr.get("delta_phi") or 0.0)
            gained += [g for g in pr.get("gained") or [] if g not in gained]
            lost += [x for x in pr.get("lost") or [] if x not in lost]
            res, cut = truncate(s.text(tl["result"]), max_chars)
            args = s.value(tl["arguments"])
            args_txt = json.dumps(args, default=str)
            if len(args_txt) > max_chars:
                args = {"truncated": truncate(args_txt, max_chars)[0]}
            te = tt + dt.timedelta(seconds=tool_s)
            tool_spans.append(Span(uuid7(int(tt.timestamp() * 1000), f"{project}|{key}|tool|{tid}"), root_id, turn_id, op_name(f"tool_{tl['function']}"),
                                   tl["function"] or "tool", {"name": tl["function"], "arguments": args, "tool_call_id": tid},
                                   {"content": res, "ok": tl["ok"], "error": s.text(str(tl["error"])) if tl["error"] else None, "truncated": cut,
                                    "chars": len(tl["result"]), "phi_after": pr.get("phi"), "delta_phi": pr.get("delta_phi"), "gained": pr.get("gained") or [],
                                    "lost": pr.get("lost") or [], "checkpoint": pr.get("checkpoint"), "full_text": full_name if cut else None},
                                   {"crucible": {"step": tl["i"], "turn": k, "episode": key, "measured": pr.get("measured")}}, _iso(tt), _iso(te), project, key))
            tt = te
        text, cut_t = truncate(s.text("\n".join(msg_of.get(cid, []))), max_chars)
        turn_out = {"text": text, "truncated": cut_t, "finish": c.get("finish"),
                    "tool_calls": [{"name": tools[i]["function"], "id": i} for i in tids],
                    "phi": {"after": phi_now, "delta": round(dphi, 9), "gained": gained, "lost": lost, "credit": adv.get(cid)},
                    "completion_tokens": u.get("output_tokens")}
        inp_last, _ = truncate(s.text(last_in), min(600, max_chars))
        turn_in = {"turn": k, "messages": {"count": n_msgs, "latest": inp_last}, "prompt_tokens": u.get("input_tokens"),
                   "cached_tokens": u.get("cache_read_tokens"), "context_tokens": c.get("context_tokens"), "model": c.get("model")}
        spans.append(Span(turn_id, root_id, root_id, "crucible_turn", f"turn {k + 1}" + (f" ({len(tids)} tools)" if tids else ""), turn_in, turn_out,
                          {"crucible": {"turn": k, "call_id": cid, "episode": key, "phi_after": phi_now, "delta_phi": round(dphi, 9), "credit": adv.get(cid),
                                        "cost_usd": c.get("cost_usd"), "latency_s": c.get("latency_s")}},
                          _iso(t_start), _iso(tt), project, key))
        spans += tool_spans
        prev_results = [tools[i] for i in tids]
        n_msgs += 1 + len(tids)
        t = tt
    t_end = max(t, t_end_all) if t_end_all else t
    patch_v, patch_cut = truncate(s.text(patch), max_chars)
    final_v, _ = truncate(s.text(final_message), max_chars)
    prompt_v, prompt_cut = truncate(s.text(prompt), max_chars)
    status = "INFRA" if masked else ("PASS" if passed else "FAIL")
    root = Span(root_id, root_id, None, "crucible_episode", f"{rec.get('task') or key} | {arm or 'run'} | {status}",
                {"task": rec.get("task"), "prompt": prompt_v, "prompt_truncated": prompt_cut, "model": model, "arm": arm, "seed": rec.get("seed")},
                {"reward": reward, "passed": passed, "phi_final": prog.get("phi_final"), "stop": meta.get("stop"), "masked": masked,
                 "infra": rec.get("failure") if masked else None, "gates": gates["all"], "final_message": final_v, "patch": patch_v, "patch_truncated": patch_cut,
                 "full_text": full_name},
                {"crucible": s.value(json.loads(json.dumps(attrs_c, default=str)))}, _iso(t0), _iso(t_end), project, key)
    spans.insert(0, root)
    index = {"kind": "episode", "episode": key, "root": root_id, "project": project, "task": rec.get("task"), "model": model, "arm": arm,
             "run": run or rec.get("run_id"), "reward": reward, "passed": passed, "masked": masked, "infra": rec.get("failure") if masked else None,
             "stop": meta.get("stop"), "phi_final": prog.get("phi_final"), "milestones_lost": prog.get("milestones_lost"), "turns": len(calls),
             "tool_calls": len(order), "usd": rec.get("cost_usd"), "gates_passed": gates["gates_passed"], "full": full_name, "record_dir": str(d),
             "started_at": _iso(t0), **{k: v for k, v in dict(attrs or {}).items() if isinstance(v, str | int | float | bool) or v is None}}
    full = s.value(strip_tokens({"episode": key, "record": rec, "calls": calls, "trajectory": tsteps, "trace": trace.get("events") or [], "progress": rows,
                                 "rollout_turns": [{k: v for k, v in x.items()} for x in rollout.get("turns") or []], "patch": patch}))
    return spans, s.value(index), full


def audit_span(root: Mapping[str, Any], finding: Mapping[str, Any], *, scrubber: Scrubber | None = None) -> Span:
    """The trace audit's verdict as a child of the episode's root call: ``finding`` is a ``failures.FailureFinding.to_json()``
    (or any dict with ``cause``, ``first_wrong_step``, ``quote``, ``mode``)."""
    s = scrubber or default_scrubber()
    f = s.value(json.loads(json.dumps(dict(finding), default=str)))
    now = dt.datetime.now(dt.UTC)
    started = _parse_time(root.get("started_at")) or now
    sid = uuid7(int(started.timestamp() * 1000) + 1, f"{root['project']}|{root['episode']}|audit|{f.get('id') or f.get('signature') or ''}")
    fws = f.get("first_wrong_step") or {}
    out = {"cause": f.get("cause"), "cause_source": f.get("cause_source"), "mode": f.get("mode"), "routed": f.get("routed"),
           "first_wrong_step": fws, "quote": f.get("quote"), "milestones_missed": f.get("milestones_missed"), "milestones_lost": f.get("milestones_lost"),
           "facts": f.get("facts"), "signature": f.get("signature")}
    return Span(sid, root["root"], root["root"], "crucible_trace_audit", f"audit: {f.get('cause') or f.get('mode') or 'unlabelled'}",
                {"episode": root["episode"], "finding": f.get("id")}, out,
                {"crucible": {"episode": root["episode"], "cause": f.get("cause"), "mode": f.get("mode"), "first_wrong_step": fws.get("step"),
                              "first_wrong_kind": fws.get("kind")}}, _iso(now), _iso(now), str(root["project"]), str(root["episode"]))


class TraceStore:
    """``<observe root>/weave/<project>/``: ``spans.jsonl`` (what ``weave_sync`` uploads), ``index.jsonl`` (one row per
    episode and per audit), ``full/`` (the full text per episode)."""

    def __init__(self, root: Path | str, project: str, *, scrubber: Scrubber | None = None):
        self.project = project
        self.dir = Path(root) / "weave" / project
        self.dir.mkdir(parents=True, exist_ok=True)
        self.s = scrubber or default_scrubber()
        self._index: dict[str, dict[str, Any]] | None = None

    @property
    def spans_path(self) -> Path:
        return self.dir / "spans.jsonl"

    def index(self) -> dict[str, dict[str, Any]]:
        """episode key -> its row (the latest audit merged in as ``audit``)."""
        if self._index is None:
            idx: dict[str, dict[str, Any]] = {}
            for r in _jsonl(self.dir / "index.jsonl"):
                if r.get("kind") == "episode":
                    idx[r["episode"]] = {**idx.get(r["episode"], {}), **r}
                elif r.get("kind") == "audit" and r.get("episode") in idx:
                    idx[r["episode"]]["audit"] = r
            self._index = idx
        return self._index

    def _append(self, name: str, rows: Iterable[Mapping[str, Any]]) -> None:
        with (self.dir / name).open("a") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True, default=str) + "\n")

    def export(self, adir: Path | str, *, arm: str = "", run: str = "", model: str | None = None, attrs: Mapping[str, Any] | None = None,
               max_chars: int = 2000, force: bool = False) -> dict[str, Any]:
        """One attempt's spans, index row and full text (an episode already exported is skipped unless ``force``)."""
        d = Path(adir)
        key = episode_key(d, _read(d / "record.json", {}) or {})
        if key in self.index() and not force:
            return self.index()[key]
        spans, row, full = episode_spans(d, project=self.project, arm=arm, run=run, model=model, attrs=attrs, scrubber=self.s, max_chars=max_chars)
        fp = self.dir / row["full"]
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(full, indent=1, sort_keys=True, default=str))
        self._append("spans.jsonl", [x.to_json() for x in spans])
        row["url"] = call_url(self.project, row["root"])
        self._append("index.jsonl", [row])
        self.index()[key] = row
        return row

    def export_all(self, root: Path | str, **kw: Any) -> list[dict[str, Any]]:
        return [self.export(d, **kw) for d in attempt_dirs(root)]

    def add_audit(self, episode: str, finding: Mapping[str, Any]) -> Span | None:
        """Attach a trace-audit finding to an exported episode (None: the episode is not in this store)."""
        root = self.index().get(episode)
        if root is None:
            return None
        sp = audit_span(root, finding, scrubber=self.s)
        self._append("spans.jsonl", [sp.to_json()])
        row = {"kind": "audit", "episode": episode, "span": sp.id, **{k: sp.output.get(k) for k in ("cause", "mode", "quote", "routed")},
               "first_wrong_step": (sp.output.get("first_wrong_step") or {}).get("step"),
               "first_wrong_kind": (sp.output.get("first_wrong_step") or {}).get("kind")}
        self._append("index.jsonl", [row])
        root["audit"] = row
        return sp

    def spans(self, episode: str | None = None) -> list[Span]:
        out = [Span.from_json(r) for r in _jsonl(self.spans_path)]
        return [x for x in out if episode is None or x.episode == episode]

    def full_paths(self, episodes: Sequence[str]) -> dict[str, Path]:
        idx = self.index()
        return {Path(idx[e]["full"]).name: self.dir / idx[e]["full"] for e in episodes if e in idx and (self.dir / idx[e]["full"]).exists()}


EPISODE_COLUMNS = ["episode", "task", "arm", "model", "reward", "passed", "masked", "infra", "stop", "phi_final", "milestones_lost", "gates_passed", "turns",
                   "tool_calls", "usd", "cause", "mode", "first_wrong_step", "quote", "trace"]


def episode_rows(rows: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    """Rows of the W&B episode table (``EPISODE_COLUMNS``): the trace link and the audit verdict beside each episode."""
    out = []
    for r in rows:
        a = r.get("audit") or {}
        out.append([r.get("episode"), r.get("task"), r.get("arm"), r.get("model"), r.get("reward"), r.get("passed"), r.get("masked"), r.get("infra"),
                    r.get("stop"), r.get("phi_final"), r.get("milestones_lost"), r.get("gates_passed"), r.get("turns"), r.get("tool_calls"), r.get("usd"),
                    a.get("cause"), a.get("mode"), a.get("first_wrong_step"), a.get("quote"), r.get("url")])
    return out


# ---- local views ---------------------------------------------------------------------------------------------------------------
def _children(spans: Sequence[Span]) -> dict[str | None, list[Span]]:
    ch: dict[str | None, list[Span]] = {}
    for x in spans:
        ch.setdefault(x.parent_id, []).append(x)
    return ch


def render_text(spans: Sequence[Span], *, width: int = 160) -> str:
    """The episode as an indented tree (``scripts/observe.py trace``)."""
    ch = _children(spans)
    lines: list[str] = []

    def one(x: Span, depth: int) -> None:
        pad = "  " * depth
        o = x.output if isinstance(x.output, dict) else {}
        if x.op == "crucible_episode":
            a = (x.attributes.get("crucible") or {})
            lines.append(f"{x.display_name}  reward={a.get('reward')} phi={a.get('phi_final')} stop={a.get('stop')} masked={a.get('masked')} "
                         f"usd={a.get('cost_usd')} model={a.get('model')}")
        elif x.op == "crucible_turn":
            p = o.get("phi") or {}
            txt = (o.get("text") or "").replace("\n", " ")
            lines.append(f"{pad}{x.display_name}: phi={p.get('after')} dphi={p.get('delta')} +{p.get('gained')} -{p.get('lost')} credit={p.get('credit')}"
                         + (f" | {txt}" if txt else ""))
        elif x.op == "crucible_trace_audit":
            fws = o.get("first_wrong_step") or {}
            lines.append(f"{pad}AUDIT cause={o.get('cause')} mode={o.get('mode')} first_wrong_step={fws.get('step')} ({fws.get('kind')}) "
                         f"quote={json.dumps(o.get('quote'))}")
        else:
            args = json.dumps((x.inputs or {}).get("arguments"), default=str)
            res = str(o.get("content") or "").replace("\n", " | ")
            lines.append(f"{pad}{x.display_name} {args} -> {'ok' if o.get('ok') else 'ERROR ' + str(o.get('error'))}: {res}"
                         + (f" [phi {o.get('phi_after')}]" if o.get("phi_after") is not None else ""))
        lines[-1] = lines[-1][:width]
        for c in ch.get(x.id, []):
            one(c, depth + 1)

    for r in ch.get(None, []):
        one(r, 0)
    return "\n".join(lines)


def render_html(spans: Sequence[Span]) -> str:
    """The episode as one HTML page with collapsible turns and tool calls (offline viewing; also an HTML panel)."""
    e = _html.escape
    ch = _children(spans)

    def pre(v: Any) -> str:
        return f"<pre>{e(v if isinstance(v, str) else json.dumps(v, indent=1, default=str))}</pre>"

    def one(x: Span) -> str:
        kids = "".join(one(c) for c in ch.get(x.id, []))
        o = x.output if isinstance(x.output, dict) else {"output": x.output}
        return (f"<details{' open' if x.parent_id is None else ''}><summary><b>{e(x.display_name)}</b> <small>{e(x.op)}</small></summary>"
                f"<div class='b'><h4>inputs</h4>{pre(x.inputs)}<h4>output</h4>{pre(o)}<h4>attributes</h4>{pre(x.attributes)}{kids}</div></details>")

    body = "".join(one(r) for r in ch.get(None, []))
    css = ("body{font:13px system-ui,sans-serif;margin:16px;background:#fcfcfb;color:#0b0b0b}details{margin:4px 0 4px 12px}"
           "pre{white-space:pre-wrap;background:#f1f0ec;padding:6px;border-radius:4px;max-height:360px;overflow:auto}.b{margin-left:8px}"
           "@media (prefers-color-scheme:dark){body{background:#1a1a19;color:#f0efec}pre{background:#262624}}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Episode trace</title><style>{css}</style></head><body>{body}</body></html>"
