"""Typed hand-offs and the family task repo (P1; docs/research/swarm-pipeline/00 §3.1, §3.3, §3.4).

**Hand-offs** (``EVALS`` L6: prose costs about 8.5 points per hand-off, structured about 2.8; sending the full original
spec loses least). Every hand-off into an agentic session is a ``Handoff``: one value per input port, each checked
against its JSON schema (the producer port's ``schema``, else ``PORT_SCHEMAS[name]``, else the port type's), **plus the
full original spec** of the run (``params.spec``) with its sha256. A hand-off whose spec is missing, or is not the run's
original (a summary substituted for it), is refused (``HandoffError``): never a summary alone. Every hand-off is stored
in the task repo, so a failure downstream can be attributed (``attribute``): **hand-off loss** (a port value differs from
what its producer settled, or fails its schema, or the spec is not the original), **grader error** (an independent
re-grade disagrees with the grader), else **actor error**.

**The task repo is the swarm's shared memory** (GitSwarm, ``task-factory/01`` §3.2; PATTERN only: GitSwarm has no
licence, nothing is copied). One git repo per family; work products are files in **immutable commits** (a path, once
committed, is never rewritten: a new version gets a new path), each commit carrying ``informed_by=<sha>`` trailers that
name the commits it built on. Messages between agents carry questions and answers; work products go here (Prime: "have
children write files and read those files for fan-in", ``LAY/usage.md:32``).

- **In-flight announcements** (GitSwarm: about 1 in 6 workers was redundant without them): before a lease starts, a
  worker announces its scope and abort criteria; an announcement whose scope overlaps a live one is refused
  (``DuplicateWork``), so duplicate work is visible before it starts. An announcement ends ``accepted``, ``abstained``,
  ``aborted`` or ``rejected``, or expires.
- **Abstention is free**: "nothing new to add" is a recorded, valid outcome (``abstain``), never a failure.
- **Herding guard** (GitSwarm: building on one adapter rose from 17% to about 90% in 9 h without it): no single parent
  idea may exceed 40% of a family's accepted tasks (``herding_check``, from 5 accepted tasks up). It is the root curator's
  check at accept (``accept_state``); the S0 lease guard is ``lease_state``.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import re
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from crucible.factory.machine import PARAMS, Settle, StateContext

HERDING_CAP = 0.40  # 00 §3.4
HERDING_FLOOR = 5  # below this many accepted tasks a share is noise (root.SURFACE_SHARE_FLOOR)
OUTCOMES = ("accepted", "abstained", "aborted", "rejected")
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")

TYPE_SCHEMAS: dict[str, dict[str, Any]] = {
    "text": {"type": "string"},
    "json": {"type": ["object", "array", "string", "number", "boolean", "null"]},
    "artifact": {"type": "object", "required": ["path", "sha256"],
                 "properties": {"path": {"type": "string", "minLength": 1}, "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}},
}
# schemas of the factory's well-known ports (a producer port's own ``schema`` wins)
PORT_SCHEMAS: dict[str, dict[str, Any]] = {
    "plan": {"type": "object", "required": ["milestones"],
             "properties": {"milestones": {"type": "array", "minItems": 1,
                                           "items": {"type": "object", "required": ["id", "text"],
                                                     "properties": {"id": {"type": "string"}, "text": {"type": "string"},
                                                                    "requires": {"type": "array", "items": {"type": "string"}}}}}}},
    "verdict": {"type": "object", "anyOf": [{"required": ["approved"]}, {"required": ["ok"]}],
                "properties": {"approved": {"type": "boolean"}, "ok": {"type": "boolean"}, "findings": {"type": "array", "items": {"type": "string"}}}},
}


class HandoffError(ValueError):
    def __init__(self, message: str, producers: list[str] | None = None):
        super().__init__(message)
        self.producers = list(producers or [])  # the states whose settled values broke their port schema


class ImmutableError(ValueError):
    """A commit tried to rewrite a path the task repo already holds with other content."""


class DuplicateWork(RuntimeError):
    def __init__(self, conflicts: list[dict[str, Any]]):
        super().__init__(f"{len(conflicts)} live announcement(s) overlap this scope: " + ", ".join(c["id"] for c in conflicts))
        self.conflicts = conflicts


def sha256_json(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


# ---- hand-offs ----------------------------------------------------------------------------------------------------------
@dataclass
class Handoff:
    run_id: str
    state: str
    entry: int
    attempt: int
    ports: dict[str, Any]  # input name -> value
    sources: dict[str, str]  # input name -> "<state>.<output>" | "params.<name>"
    schemas: dict[str, dict[str, Any]]
    spec: Any  # the run's full original spec, always
    spec_sha256: str
    findings: list[str] = field(default_factory=list)  # carried by the loop-back that entered the state
    recovery: dict[str, Any] | None = None
    id: str = ""

    def body(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("id")
        return d

    def seal(self) -> Handoff:
        self.id = sha256_json(self.body())[:20]
        return self

    def to_json(self) -> str:
        return json.dumps({"id": self.id, **self.body()}, sort_keys=True, ensure_ascii=False, indent=1, default=str)


def _port_schema(machine_states: Mapping[str, Mapping[str, Any]], inp: Mapping[str, Any]) -> dict[str, Any]:
    src, _, out = str(inp["from"]).partition(".")
    if src != PARAMS and src in machine_states:
        port = next((o for o in machine_states[src].get("outputs", []) if o.get("name") == out), None)
        if port and isinstance(port.get("schema"), dict):
            return port["schema"]
    return PORT_SCHEMAS.get(out) or PORT_SCHEMAS.get(inp["name"]) or TYPE_SCHEMAS[inp["type"]]


def schema_errors(value: Any, schema: Mapping[str, Any]) -> list[str]:
    v = Draft202012Validator(dict(schema))
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"[:300] for e in v.iter_errors(value)]


def build_handoff(ctx: StateContext, machine_states: Mapping[str, Mapping[str, Any]], *, spec_param: str = "spec") -> Handoff:
    """The typed hand-off into ``ctx``'s state: every bound input checked against its schema, the full original spec
    (``params[spec_param]``) attached. ``HandoffError`` when a value breaks its schema or there is no spec."""
    if spec_param not in ctx.params or ctx.params[spec_param] in (None, "", {}):
        raise HandoffError(f"the run has no original spec (params.{spec_param}): a hand-off never travels without it")
    spec = ctx.params[spec_param]
    ports, sources, schemas, errors, producers = {}, {}, {}, [], []
    for inp in ctx.state.get("inputs", []):
        name = inp["name"]
        val = ctx.inputs.get(name)
        sch = _port_schema(machine_states, inp)
        if val is None and inp.get("optional"):
            ports[name], sources[name], schemas[name] = None, inp["from"], sch
            continue
        errs = schema_errors(val, sch)
        if errs:
            errors.append(f"input {name!r} from {inp['from']}: " + "; ".join(errs[:3]))
            producers.append(str(inp["from"]).partition(".")[0])
        ports[name], sources[name], schemas[name] = copy.deepcopy(val), inp["from"], sch
    if errors:
        raise HandoffError("; ".join(errors), [p for p in producers if p != PARAMS])
    return Handoff(ctx.run_id, ctx.state_id, ctx.entry, ctx.attempt, ports, sources, schemas, copy.deepcopy(spec), sha256_json(spec),
                   list(ctx.findings), copy.deepcopy(ctx.recovery)).seal()


def check_handoff(h: Mapping[str, Any] | Handoff, *, original_spec: Any) -> list[str]:
    """Everything wrong with a stored hand-off: its id, the spec (present, and the run's original, not a summary), each
    port against its schema."""
    d = json.loads(h.to_json()) if isinstance(h, Handoff) else dict(h)
    out = []
    body = {k: v for k, v in d.items() if k != "id"}
    if d.get("id") != sha256_json(body)[:20]:
        out.append("the hand-off's id does not match its content (edited after sealing)")
    if d.get("spec") in (None, "", {}):
        out.append("the hand-off carries no spec")
    elif sha256_json(d["spec"]) != sha256_json(original_spec) or d.get("spec_sha256") != sha256_json(original_spec):
        out.append("the hand-off's spec is not the run's original spec (a summary or an edit)")
    for name, val in (d.get("ports") or {}).items():
        sch = (d.get("schemas") or {}).get(name)
        if sch is None:
            out.append(f"port {name!r} has no schema")
        elif val is not None:
            out += [f"port {name!r}: {e}" for e in schema_errors(val, sch)[:3]]
    return out


@dataclass
class Attribution:
    label: str  # handoff_loss | grader_error | actor_error
    evidence: list[str]


def attribute(handoff: Mapping[str, Any] | Handoff, *, producer_outputs: Mapping[str, Mapping[str, Any]], original_spec: Any,
              grader_pass: bool | None = None, regrade_pass: bool | None = None) -> Attribution:
    """Who owns a downstream failure. ``producer_outputs``: state -> its settled outputs (the journal's); ``grader_pass``
    / ``regrade_pass``: the grader's verdict on the consumer's work and an independent re-grade (a second verifier, the
    oracle diff), when there is one. Executed facts only: a hash mismatch, a schema error, a verdict disagreement."""
    d = json.loads(handoff.to_json()) if isinstance(handoff, Handoff) else dict(handoff)
    ev = [f"hand-off {d.get('id')}: {p}" for p in check_handoff(d, original_spec=original_spec)]
    for name, src in (d.get("sources") or {}).items():
        st, _, out = str(src).partition(".")
        if st == PARAMS or st not in producer_outputs:
            continue
        settled = producer_outputs[st].get(out)
        got = (d.get("ports") or {}).get(name)
        if isinstance(settled, dict) and isinstance(got, dict):  # the canary rides on settled json values; compare without it
            settled = {k: v for k, v in settled.items() if k != "_canary"}
            got = {k: v for k, v in got.items() if k != "_canary"}
        if sha256_json(settled) != sha256_json(got):
            ev.append(f"port {name!r} differs from what {src} settled (sha {sha256_json(got)[:12]} != {sha256_json(settled)[:12]})")
    if ev:
        return Attribution("handoff_loss", ev)
    if grader_pass is not None and regrade_pass is not None and grader_pass != regrade_pass:
        said = (f"the grader said {'pass' if grader_pass else 'fail'}, an independent re-grade said "
                f"{'pass' if regrade_pass else 'fail'}")
        return Attribution("grader_error", [said])
    return Attribution("actor_error", ["the hand-off is the producer's output with the original spec, and the grade stands"])


# ---- the task repo ------------------------------------------------------------------------------------------------------
def _git(root: Path, *args: str, input: bytes | None = None, env: Mapping[str, str] | None = None, check: bool = True) -> str:
    import os

    r = subprocess.run(["git", "-C", str(root), *args], input=input, capture_output=True, check=False, timeout=300,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env or {})})
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} failed: {r.stderr.decode(errors='replace')[-400:]}")
    return r.stdout.decode(errors="replace")


def _scope_overlaps(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Two scopes overlap when every key they share agrees (lists: share an item). No shared key: no overlap."""
    shared = set(a) & set(b)
    if not shared:
        return False
    for k in shared:
        x, y = a[k], b[k]
        if isinstance(x, list) or isinstance(y, list):
            xs = set(map(str, x if isinstance(x, list) else [x]))
            ys = set(map(str, y if isinstance(y, list) else [y]))
            if not xs & ys:
                return False
        elif x != y:
            return False
    return True


@dataclass
class HerdingVerdict:
    ok: bool
    parent: str
    share: float
    accepted: int
    reason: str = ""


class TaskRepo:
    """One family's shared memory: a git repo of immutable commits (see the module doc). Safe for concurrent writers in
    one host (an ``fcntl`` lock around every commit)."""

    def __init__(self, root: Path | str, family: str, *, clock: Any = time.time):
        self.root = Path(root).resolve()
        self.family = family
        self.clock = clock
        self.root.mkdir(parents=True, exist_ok=True)
        if not (self.root / ".git").exists():
            _git(self.root, "init", "-q", "-b", "main")
            _git(self.root, "config", "user.email", "factory@crucible.local")
            _git(self.root, "config", "user.name", "crucible-factory")
            self.commit({"FAMILY.json": json.dumps({"family": family}, indent=1) + "\n"}, f"family {family}")

    @contextmanager
    def _locked(self) -> Any:
        with (self.root / ".git" / "crucible.lock").open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def head(self) -> str | None:
        out = _git(self.root, "rev-parse", "-q", "--verify", "HEAD", check=False).strip()
        return out or None

    def read(self, rel: str, ref: str = "HEAD") -> bytes | None:
        r = subprocess.run(["git", "-C", str(self.root), "show", f"{ref}:{rel}"], capture_output=True, check=False, timeout=120)
        return r.stdout if r.returncode == 0 else None

    def commit(self, files: Mapping[str, str | bytes | Path], message: str, *, informed_by: Sequence[str] = (),
               trailers: Mapping[str, str] | None = None) -> str:
        """Write ``files`` (repo-relative path -> text, bytes, or a host file/dir to copy) and commit with
        ``informed_by=`` and ``key=value`` trailers. A path already committed with other content is refused
        (``ImmutableError``); one committed with the same content is a no-op. Returns the commit sha (HEAD when
        nothing changed)."""
        import shutil

        with self._locked():
            rows: list[str] = []
            for rel, body in files.items():
                p = (self.root / rel).resolve()
                if self.root not in p.parents:
                    raise ImmutableError(f"{rel!r} is outside the task repo")
                if isinstance(body, Path) and body.is_dir():
                    if p.exists() and self.read(rel + "/" if not rel.endswith("/") else rel) is not None:
                        raise ImmutableError(f"{rel!r} is already committed: write a new version path")
                    shutil.copytree(body, p, symlinks=True, dirs_exist_ok=False)
                    rows.append(rel)
                    continue
                data = body.read_bytes() if isinstance(body, Path) else (body.encode() if isinstance(body, str) else body)
                old = self.read(rel)
                if old is not None:
                    if old == data:
                        continue
                    raise ImmutableError(f"{rel!r} is already committed with other content: write a new version path")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
                rows.append(rel)
            if not rows and self.head() is not None:
                return self.head()  # type: ignore[return-value]
            if rows:
                _git(self.root, "add", "--", *rows)
            lines = [f"informed_by={s}" for s in informed_by if s]
            lines += [f"{k}={v}" for k, v in sorted((trailers or {}).items())]
            msg = message.strip() + ("\n\n" + "\n".join(lines) if lines else "") + "\n"
            _git(self.root, "commit", "-q", "--allow-empty", "-F", "-", input=msg.encode())
            return self.head()  # type: ignore[return-value]

    def trailers(self, sha: str) -> dict[str, list[str]]:
        body = _git(self.root, "show", "-s", "--format=%B", sha)
        out: dict[str, list[str]] = {}
        for ln in body.splitlines():
            m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_-]*)=(\S.*)", ln.strip())
            if m:
                out.setdefault(m.group(1), []).append(m.group(2))
        return out

    def log(self, path: str | None = None) -> list[dict[str, Any]]:
        args = ["log", "--format=%H%x1f%s", *(["--", path] if path else [])]
        rows = []
        for ln in _git(self.root, *args).splitlines():
            sha, _, subject = ln.partition("\x1f")
            rows.append({"sha": sha, "subject": subject, "trailers": self.trailers(sha)})
        return rows

    def lineage(self, sha: str) -> list[str]:
        """Every commit ``sha`` was (transitively) informed by."""
        seen: list[str] = []
        todo = list(self.trailers(sha).get("informed_by", []))
        while todo:
            s = todo.pop()
            if s in seen:
                continue
            seen.append(s)
            todo += self.trailers(s).get("informed_by", [])
        return seen

    def _records(self, folder: str) -> dict[str, dict[str, Any]]:
        d = self.root / folder
        if not d.is_dir():
            return {}
        return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}

    # ---- in-flight announcements --------------------------------------------------------------------------------------
    def inflight(self) -> list[dict[str, Any]]:
        now = self.clock()
        retired = self._records("retired")
        return [a for a in self._records("inflight").values() if a["id"] not in retired and a["expires"] > now]

    def conflicts(self, scope: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [a for a in self.inflight() if _scope_overlaps(scope, a["scope"])]

    def announce(self, holder: str, scope: Mapping[str, Any], abort_criteria: Sequence[str], *, ttl_s: float = 6 * 3600,
                 informed_by: Sequence[str] = ()) -> dict[str, Any]:
        """Announce in-flight work before a lease starts; ``DuplicateWork`` when a live announcement overlaps."""
        if not abort_criteria:
            raise ValueError("an announcement states its abort criteria (when the work stops being worth doing)")
        with self._locked():
            hits = self.conflicts(scope)
        if hits:
            raise DuplicateWork(hits)
        ann = {"id": f"a-{uuid.uuid4().hex[:12]}", "holder": holder, "scope": dict(scope), "abort_criteria": list(abort_criteria),
               "at": self.clock(), "expires": self.clock() + ttl_s}
        sha = self.commit({f"inflight/{ann['id']}.json": json.dumps(ann, sort_keys=True, indent=1) + "\n"},
                          f"announce {ann['id']} by {holder}", informed_by=informed_by, trailers={"announce": ann["id"]})
        return {**ann, "commit": sha}

    def retire(self, ann_id: str, outcome: str, *, reason: str = "", task_id: str | None = None, parent: str | None = None,
               informed_by: Sequence[str] = ()) -> str:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        rec = {"id": ann_id, "outcome": outcome, "reason": reason, "task_id": task_id, "parent": parent, "at": self.clock()}
        files: dict[str, str | bytes | Path] = {f"retired/{ann_id}.json": json.dumps(rec, sort_keys=True, indent=1) + "\n"}
        if outcome == "accepted":
            if not task_id or not parent:
                raise ValueError("an accepted task names its task id and its parent idea (the herding guard counts parents)")
            files[f"accepted/{_SAFE.sub('-', task_id)}.json"] = json.dumps({"task_id": task_id, "parent": parent, "announcement": ann_id},
                                                                            sort_keys=True, indent=1) + "\n"
        return self.commit(files, f"{outcome} {ann_id}" + (f": {reason}" if reason else ""), informed_by=informed_by,
                           trailers={"outcome": outcome})

    def abstain(self, ann_id: str, reason: str = "nothing new to add") -> str:
        """Abstention is free: a recorded, valid outcome (GitSwarm)."""
        return self.retire(ann_id, "abstained", reason=reason)

    # ---- the herding guard ---------------------------------------------------------------------------------------------
    def accepted(self) -> list[dict[str, Any]]:
        return list(self._records("accepted").values())

    def herding_check(self, parent: str, *, cap: float = HERDING_CAP, floor: int = HERDING_FLOOR) -> HerdingVerdict:
        """Would accepting one more task built on ``parent`` push that parent over ``cap`` of the family's accepted tasks?
        Below ``floor`` accepted tasks (after this one) a share is noise and passes."""
        acc = self.accepted()
        n = len(acc) + 1
        mine = sum(1 for a in acc if a.get("parent") == parent) + 1
        share = mine / n
        if n >= floor and share > cap:
            return HerdingVerdict(False, parent, share, n - 1, f"parent idea {parent!r} would hold {mine}/{n} = {share:.0%} of accepted "
                                                                  f"tasks (cap {cap:.0%})")
        return HerdingVerdict(True, parent, share, n - 1)


# ---- machine handlers: S0 lease guard and S17 accept --------------------------------------------------------------------
def lease_state(repo: TaskRepo) -> Any:
    """S0: announce the run's scope (``params.scope``, ``params.abort_criteria``, holder ``params.holder``) before any work;
    a live overlapping announcement settles ``lease.ok == false`` with the conflicts (route it to an abstain terminal)."""

    async def handler(ctx: StateContext) -> Settle:
        scope = ctx.inputs.get("scope") or ctx.params.get("scope") or {}
        try:
            ann = repo.announce(str(ctx.params.get("holder") or ctx.run_id), scope, ctx.params.get("abort_criteria") or ["the cell is filled"])
        except DuplicateWork as e:
            return Settle({"lease": {"ok": False, "conflicts": [c["id"] for c in e.conflicts], "reason": str(e)}})
        return Settle({"lease": {"ok": True, "announcement": ann["id"], "commit": ann["commit"]}})

    return handler


def accept_state(repo: TaskRepo, *, cap: float = HERDING_CAP, floor: int = HERDING_FLOOR) -> Any:
    """S17 (root curator check): the herding guard, then the accepted record. Inputs ``lease`` (json) and ``parent``
    (json or text: the parent idea, e.g. the source record or template the task was built from)."""

    async def handler(ctx: StateContext) -> Settle:
        lease = ctx.inputs.get("lease") or {}
        parent = ctx.inputs.get("parent")
        parent = parent if isinstance(parent, str) else json.dumps(parent, sort_keys=True)
        v = repo.herding_check(parent, cap=cap, floor=floor)
        if not v.ok:
            repo.retire(lease["announcement"], "rejected", reason=v.reason)
            return Settle({"accept": {"ok": False, "reason": v.reason, "share": v.share}})
        task_id = str(ctx.params.get("task_id") or ctx.run_id)
        sha = repo.retire(lease["announcement"], "accepted", task_id=task_id, parent=parent)
        return Settle({"accept": {"ok": True, "task_id": task_id, "commit": sha, "share": round(v.share, 4)}})

    return handler
