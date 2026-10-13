"""EnterpriseOps-Gym tasks as Crucible tasks (DATA + PATTERN; ServiceNow EnterpriseOps-Gym, Apache-2.0).

An EnterpriseOps-Gym task (``BenchmarkConfig``, ``benchmark/models.py``: a local ``data/revised/<domain>/*.json`` file or
a row of the HF dataset ``ServiceNow-AI/EnterpriseOps-Gym``, whose ``gym_servers_config`` / ``verifiers`` are JSON text)
becomes an ``EogTask`` (an M10 ``RunTask``). The mapping:

- **Apps.** One ``common/sqlite_app`` per ``gym_servers_config`` entry (app id from the server name: ``sn-csm-server``
  -> ``csm``), seeded from the task's ``seed_database_file`` (read from ``gym_dbs.zip`` or an unzipped copy). The dumps
  are data only (``INSERT`` statements, no DDL), so the schema is **inferred**: one table per inserted table with the
  union of its inserted columns, plus tables and columns the verifiers query but no seed row fills; a first column named
  ``id`` / ``*_id`` that the seed holds as unique integers becomes ``INTEGER PRIMARY KEY`` (so a new row takes the next
  id, which is what the verifiers hard-code: the CSM dumps end at case 1232 and the checks expect case 1233). SQLite's
  ``now`` is pinned to the World clock's ``t0`` while seeding (``sqlite_app.pin_now``); ``t0`` is the task's own
  timestamp (``task_YYYYMMDD_HHMMSS_...``), else the seed's ``Created:`` header, else the sqlite_app default.
- **Tools.** The 512 domain tools exist only inside the benchmark's Docker MCP images, not in the repository, so the
  local World exposes each record system through ``sql_query`` and ``sql_execute`` (SQLite dialect). The task's
  ``selected_tools`` / ``restricted_tools`` are kept in ``meta`` (the ``CatalogSpec`` oracle / hide lists for a World that
  serves the real MCP server).
- **Identity.** ``user_info`` and ``context`` become the policy actor's ``IdentitySpec`` (``context`` keys are the
  ``x-*`` headers EnterpriseOps sends). The tasks declare no roles (the system prompt grants admin rights in prose), so no
  ``rbac`` is emitted. SQL tools cannot see headers, so the instruction names the signed-in user once.
- **Instruction.** The domain policy (``system_prompt``), the signed-in user, then the request (``user_prompt``); both
  prompts are also kept separately (``system_prompt``, ``user_prompt``) for policies that take a system message.
- **Grader.** Each ``database_state`` verifier becomes a ``SqlAssertion`` (``verify.sql_checks.from_enterpriseops``:
  EnterpriseOps' result reduction and comparisons) in one outcome gate ``eog/state``: reward 1 iff every verifier passes
  (EnterpriseOps' task success). ``meta.components["eog/verifier_pass_rate"]`` is EnterpriseOps'
  ``verifier_level_pass_rate`` (passed / total).
- **Refusals.** A task whose verifiers all expect a zero count (nothing may be created), a task with no verifiers, or one
  imported with ``infeasible=True``, is a refusal task: refusing an infeasible request means leaving the systems
  unchanged, so a hard gate ``unchanged`` requires ``VerifierView.unchanged`` on every App. Doing nothing scores 1.
- **Witness.** ``oracle_actions()`` derives an ``sql_execute`` sequence from the expected state (``interop.eog.oracle``).

Not mapped (an import with ``strict=True`` refuses the task; ``import_eog_dir`` lists it under ``skipped``):
``response_check`` verifiers (an LLM judge over a SQL result) and ``tool_execution`` verifiers (they require calls to named
tools that only the Docker MCP servers have); seeds missing from the archive. Not reproduced by the local World: the
servers' business logic (triggers, validation, cascades, side records such as interactions an MCP tool creates),
column types and constraints beyond the inferred primary keys, and row visibility per user.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sqlite3
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from crucible.core.schema.world import WorldSpec
from crucible.verify.sql_checks import SqlAssertion, from_enterpriseops

SEED_ROOT = "Domain Wise DBs and Task-DB Mappings"
_INSERT = re.compile(r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+[`\"\[]?(\w+)[`\"\]]?\s*\(([^)]*)\)\s*VALUES", re.IGNORECASE)
_TABLE_REF = re.compile(r"\b(?:FROM|JOIN)\s+[`\"]?([A-Za-z_]\w*)[`\"]?(?:\s+(?:AS\s+)?([A-Za-z_]\w*))?", re.IGNORECASE)
_QUAL = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")
_IDENT = re.compile(r"\b([A-Za-z_]\w*)\b")
_SQL_WORDS = {"select", "from", "where", "and", "or", "not", "null", "is", "in", "like", "as", "count", "join", "on", "inner", "left", "right",
              "outer", "true", "false", "between", "min", "max", "sum", "avg", "distinct", "group", "by", "order", "limit", "having", "case",
              "when", "then", "else", "end", "exists", "datetime", "date", "lower", "upper", "coalesce", "asc", "desc"}
_TASK_TS = re.compile(r"task_(\d{8})_(\d{6})")
_CREATED = re.compile(r"^--\s*Created:\s*(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})", re.MULTILINE)
UNSUPPORTED_VERIFIERS = {"response_check": "LLM-judged response_check verifiers are not ported",
                         "tool_execution": "tool_execution verifiers need the named MCP tools, which the local World does not serve"}


def load_task(src: str | Path | dict[str, Any]) -> tuple[dict[str, Any], str]:
    """A task JSON file, or an HF dataset row (JSON-text fields decoded) -> (config dict, task id)."""
    if isinstance(src, dict):
        d = dict(src)
        tid = str(d.get("task_id") or d.get("id") or "task")
    else:
        p = Path(src)
        d = json.loads(p.read_text())
        tid = p.stem
    for key in ("gym_servers_config", "verifiers", "selected_tools", "restricted_tools"):
        if isinstance(d.get(key), str):
            d[key] = json.loads(d[key])
    if not d.get("gym_servers_config") and d.get("mcp_server_url"):  # legacy single-gym config
        d["gym_servers_config"] = [{"mcp_server_name": d.get("mcp_server_name", "gym"), "mcp_server_url": d["mcp_server_url"],
                                    "seed_database_file": d.get("seed_database_file", ""), "context": d.get("context") or {}}]
    return d, tid


def app_id_for(server_name: str) -> str:
    """``sn-csm-server`` -> ``csm``, ``gym-google-drive-mcp`` -> ``google-drive``, ``sn-hr-internal`` -> ``hr``."""
    s = re.sub(r"^(sn|gym)-", "", server_name.lower())
    s = re.sub(r"-(server|mcp|internal)$", "", s)
    return re.sub(r"[^a-z0-9_-]", "_", s) or "gym"


class SeedStore:
    """Seed dumps by their ``seed_database_file`` path, from ``gym_dbs.zip`` or an unzipped copy (extracted on demand to
    ``cache_dir``). A path that does not resolve is looked up by file name (the dumps are named by database id)."""

    def __init__(self, source: str | Path, cache_dir: str | Path):
        self.source = Path(source)
        self.cache = Path(cache_dir)
        self._zip_names: dict[str, str] | None = None

    def _names(self) -> dict[str, str]:
        if self._zip_names is None:
            with zipfile.ZipFile(self.source) as z:
                self._zip_names = {Path(n).name: n for n in z.namelist() if n.endswith(".sql")}
        return self._zip_names

    def get(self, rel: str) -> Path | None:
        if not rel:
            return None
        name = Path(rel).name
        if self.source.is_dir():
            for cand in (self.source / rel, self.source / SEED_ROOT / rel):
                if cand.is_file():
                    return cand
            hits = sorted(self.source.rglob(name))
            return hits[0] if hits else None
        member = self._names().get(name)
        if member is None:
            return None
        out = self.cache / "seeds" / name
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self.source) as z:
                out.write_bytes(z.read(member))
        return out


def referenced_columns(query: str) -> dict[str, list[str]]:
    """Tables a verifier query reads and the columns it names on each (aliases resolved; bare names go to the only table)."""
    text = re.sub(r"'(?:[^']|'')*'", "''", query)
    alias: dict[str, str] = {}
    tables: list[str] = []
    for t, a in _TABLE_REF.findall(text):
        tables.append(t)
        alias[t] = t
        if a and a.lower() not in _SQL_WORDS:
            alias[a] = t
    out: dict[str, list[str]] = {t: [] for t in tables}
    for a, c in _QUAL.findall(text):
        if a in alias and c not in out[alias[a]]:
            out[alias[a]].append(c)
    if len(set(tables)) == 1:
        body = _QUAL.sub(" ", text)
        names = set(alias) | {t.lower() for t in tables}
        for c in _IDENT.findall(body):
            if c.lower() not in _SQL_WORDS and c not in names and c.lower() not in names and not c.isdigit() and c not in out[tables[0]]:
                out[tables[0]].append(c)
    return out


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


_INT_TEXT = re.compile(r"-?(?:0|[1-9][0-9]*)")
_REAL_TEXT = re.compile(r"-?(?:0|[1-9][0-9]*)\.[0-9]+")


def _value_kind(v: Any) -> str:
    if isinstance(v, bool | int):
        return "int"
    if isinstance(v, float):
        return "real"
    if isinstance(v, str):
        return "int" if _INT_TEXT.fullmatch(v) else "real" if _REAL_TEXT.fullmatch(v) else "text"
    return "blob"


def infer_schema(seed_sql: str, *, extra: dict[str, list[str]] | None = None, now: str = "2026-01-05 09:00:00") -> tuple[str, dict[str, str]]:
    """DDL for a data-only dump: (schema SQL, table -> primary-key column). See the module docstring for the rules."""
    from crucible.world.apps.sqlite_app import pin_now

    cols: dict[str, list[str]] = {}
    for table, clist in _INSERT.findall(seed_sql):
        cs = cols.setdefault(table, [])
        for c in clist.split(","):
            c = c.strip().strip('`"[]')
            if c and c not in cs:
                cs.append(c)
    seeded = set(cols)
    for table, cs in (extra or {}).items():
        have = cols.setdefault(table, [])
        have += [c for c in cs if c not in have]
    cols = {t: cs for t, cs in cols.items() if cs}
    con = sqlite3.connect(":memory:")
    try:
        pin_now(con, now)
        con.executescript("".join(f"CREATE TABLE {_q(t)} ({', '.join(map(_q, cs))});\n" for t, cs in cols.items()))
        con.executescript(seed_sql)
        pk: dict[str, str] = {}
        for t, cs in cols.items():
            c0 = cs[0]
            if t not in seeded or not (c0.lower() == "id" or c0.lower().endswith("_id")):
                continue
            n, distinct, ints, nulls = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT {_q(c0)}), SUM(typeof({_q(c0)}) = 'integer'), "
                                                   f"SUM({_q(c0)} IS NULL) FROM {_q(t)}").fetchone()
            if n and distinct == n and not nulls:
                pk[t] = c0 if ints == n else c0 + " TEXT"
        # each column's affinity from its seed values, as the benchmark's typed tables have (M11 audit H3 V12: a typeless
        # column kept '427' as text, so the same write graded differently). The dumps quote numbers ('177'), so a value
        # counts as an integer when it is one or is written as one (no leading zero, which a code like '0123' keeps):
        # all integers -> INTEGER, all numbers -> REAL, all text -> TEXT; mixed or empty columns stay typeless
        aff: dict[tuple[str, str], str] = {}
        for t, cs in cols.items():
            if t not in seeded:
                continue
            for c in cs:
                vals = [v for (v,) in con.execute(f"SELECT DISTINCT {_q(c)} FROM {_q(t)} WHERE {_q(c)} IS NOT NULL")]
                if not vals:
                    continue
                kinds = {_value_kind(v) for v in vals}
                aff[(t, c)] = "INTEGER" if kinds == {"int"} else "REAL" if kinds <= {"int", "real"} else "TEXT" if "blob" not in kinds else ""
                if not aff[(t, c)]:
                    del aff[(t, c)]
    finally:
        con.close()
    ddl = []
    for t, cs in cols.items():
        defs = []
        for c in cs:
            key = pk.get(t, "")
            if key.split(" ")[0] == c:
                defs.append(f"{_q(c)} INTEGER PRIMARY KEY" if " " not in key else f"{_q(c)} PRIMARY KEY")
            else:
                defs.append(f"{_q(c)} {aff[(t, c)]}" if (t, c) in aff else _q(c))
        ddl.append(f"CREATE TABLE IF NOT EXISTS {_q(t)} ({', '.join(defs)});")
    return "\n".join(ddl) + "\n", {t: k.split(" ")[0] for t, k in pk.items()}


def _t0(tid: str, seeds: list[Path]) -> str | None:
    if m := _TASK_TS.search(tid):
        return f"{m[1][:4]}-{m[1][4:6]}-{m[1][6:]}T{m[2][:2]}:{m[2][2:4]}:{m[2][4:]}Z"
    for s in seeds:
        with s.open() as f:
            head = f.read(2000)
        if m := _CREATED.search(head):
            return f"{m[1]}T{m[2]}Z"
    return None


@dataclass
class EogTask:
    """One imported EnterpriseOps-Gym task (an M10 ``RunTask``: ``open`` / ``grade``)."""

    id: str
    world_id: str
    spec_doc: WorldSpec
    instruction: str
    system_prompt: str
    user_prompt: str
    assertions: list[SqlAssertion]
    refusal: bool
    pk: dict[str, dict[str, str]]  # app -> table -> primary-key column
    meta: dict[str, Any] = field(default_factory=dict)

    async def open(self, workdir: Path):
        from crucible.core.store import Store
        from crucible.world.world import World

        workdir = Path(workdir)
        w = World(self.spec_doc, instructions={"agent": self.instruction}, store=Store(workdir / "s"), workdir=workdir / "w")
        await w.start()
        await w.reset()
        return w

    def plan(self):
        from crucible.verify.runtime.plan import Check, Gate, Plan, Verdict

        gates = [Gate("eog/state", kind="outcome", description="every EnterpriseOps-Gym database_state verifier passes")]
        checks = [a.check("eog/state", provenance=f"eog:{a.description}"[:200]) for a in self.assertions]
        if self.refusal:
            apps = tuple(a.id for a in self.spec_doc.apps)

            async def unchanged(ctx) -> Verdict:
                got = await ctx.params["view"].unchanged(*apps)
                return Verdict(all(got.values()), {"unchanged": got})

            gates.append(Gate("unchanged", kind="gate", description="an infeasible request is refused: every App keeps its episode-start state"))
            checks.append(Check("unchanged", "unchanged", unchanged, provenance="eog:refusal = leave apps unchanged"))
        if not checks:
            raise ValueError(f"{self.id}: nothing to grade")
        if not self.assertions:  # a refusal task with no verifiers: the state gate is the unchanged check alone
            gates = [g for g in gates if g.id != "eog/state"]
        return Plan(self.id, gates, checks)

    async def grade(self, world: Any, workdir: Path):
        from crucible.verify.runtime.plan import Context
        from crucible.verify.runtime.trusted import Trusted

        rep = await self.plan().arun(Context(Trusted(Path(workdir)), params={"view": world.verifier_view()}))
        mine = [c for c in rep.checks if c.gate == "eog/state"]
        if mine:
            rep.meta.setdefault("components", {})["eog/verifier_pass_rate"] = round(sum(c.status == "pass" for c in mine) / len(mine), 6)
        return rep

    def oracle_actions(self) -> list:
        """A witness for reward 1 (``interop.eog.oracle``): ``sql_execute`` calls, then ``Done``. Raises ``Underivable``.
        A refusal task's witness is doing nothing."""
        from crucible.core.schema.actions import Done, ToolCall
        from crucible.interop.eog.oracle import derive

        if self.refusal:
            return [Done(reason="I cannot do this: the request is not allowed or not possible, so I made no changes.")]
        clock = self.spec_doc.clock.t0 if self.spec_doc.clock else "2026-01-05T09:00:00Z"
        now = datetime.fromisoformat(clock).strftime("%Y-%m-%d %H:%M:%S")
        acts: list = []
        for app in [a.id for a in self.spec_doc.apps]:
            mine = [(a.query, a.expected, a.compare) for a in self.assertions if a.app == app]
            for sql, params in derive(mine, pk=self.pk.get(app, {}), now=now):
                acts.append(ToolCall(tool_call_id=f"o{len(acts) + 1}", function_name=f"{app}.sql_execute", arguments={"sql": sql, "params": params}))
        return [*acts, Done(reason="done")]


def import_eog_task(src: str | Path | dict[str, Any], *, seeds: str | Path, cache_dir: str | Path, infeasible: bool | None = None,
                    strict: bool = True) -> EogTask:
    """One EnterpriseOps-Gym task -> ``EogTask``. ``seeds``: ``gym_dbs.zip`` or its unzipped directory; ``cache_dir``: where
    extracted seeds and the inferred schemas go. ``infeasible``: force (True) or rule out (False) the refusal mapping."""
    cfg, tid = load_task(src)
    store = SeedStore(seeds, cache_dir)
    cache = Path(cache_dir)
    servers = cfg.get("gym_servers_config") or []
    if not servers:
        raise ValueError(f"{tid}: no gym_servers_config")
    verifiers = list(cfg.get("verifiers") or [])
    problems = sorted({UNSUPPORTED_VERIFIERS.get(v.get("verifier_type"), f"unknown verifier type {v.get('verifier_type')!r}")
                       for v in verifiers if v.get("verifier_type") != "database_state"})
    if problems and strict:
        raise ValueError(f"{tid}: " + "; ".join(problems))
    app_of = {g["mcp_server_name"]: app_id_for(g["mcp_server_name"]) for g in servers}
    if len(set(app_of.values())) != len(app_of):
        raise ValueError(f"{tid}: two gym servers map to one app id ({app_of})")
    only = next(iter(app_of.values())) if len(app_of) == 1 else None
    assertions = []
    for i, v in enumerate(verifiers):
        if v.get("verifier_type") != "database_state":
            continue
        a = from_enterpriseops(v, app_of=app_of, index=i)
        if a.app not in app_of.values():
            if only is None:
                raise ValueError(f"{tid}: verifier {i} names gym {v.get('gym_name')!r}, not one of {sorted(app_of)}")
            a = dataclasses.replace(a, app=only)
        assertions.append(a)
    seed_paths: dict[str, Path] = {}
    for g in servers:
        p = store.get(g.get("seed_database_file", ""))
        if p is None:
            raise FileNotFoundError(f"{tid}: seed {g.get('seed_database_file')!r} is not in {store.source}")
        seed_paths[app_of[g["mcp_server_name"]]] = p
    t0 = _t0(tid, list(seed_paths.values()))
    now = datetime.fromisoformat(t0 or "2026-01-05T09:00:00Z").strftime("%Y-%m-%d %H:%M:%S")
    apps, pks = [], {}
    for g in servers:
        app = app_of[g["mcp_server_name"]]
        extra: dict[str, list[str]] = {}
        for a in assertions:
            if a.app == app:
                for t, cs in referenced_columns(a.query).items():
                    extra.setdefault(t, [])
                    extra[t] += [c for c in cs if c not in extra[t]]
        ddl, pk = infer_schema(seed_paths[app].read_text(), extra=extra, now=now)
        schema = cache / "schemas" / tid / f"{app}.sql"
        schema.parent.mkdir(parents=True, exist_ok=True)
        schema.write_text(f"-- inferred by crucible.interop.eog from {seed_paths[app].name} and the task's verifiers\n" + ddl)
        pks[app] = pk
        apps.append({"id": app, "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": f"eog/{app}.db"}},
                     "params": {"schema_sql": str(schema.resolve()), "seed_sql": [str(seed_paths[app].resolve())], "sql_tools": True, "sql_write": True},
                     "description": f"EnterpriseOps-Gym {g['mcp_server_name']} system of record (seed {seed_paths[app].stem})"})
    first = servers[0]
    info, ctx = dict(first.get("user_info") or {}), {k: str(v) for k, v in (first.get("context") or {}).items()}
    actor: dict[str, Any] = {"id": "agent", "kind": "policy", "tools": [f"{a['id']}.*" for a in apps]}
    if info or ctx:
        actor["identity"] = {"user_id": info.get("user_id"), "name": info.get("name"), "email": info.get("email") or ctx.get("x-user-email"),
                             "headers": {k: v for k, v in ctx.items() if k.lower() != "x-user-email"}}
    domain = str(cfg.get("domain") or next(iter(app_of.values())))
    doc: dict[str, Any] = {"world": f"eog/{domain}/{tid}", "description": f"EnterpriseOps-Gym {domain} task {tid}"[:500], "backend": "local",
                           "apps": apps, "actors": [actor]}
    if t0:
        doc["clock"] = {"t0": t0}
    spec = WorldSpec.model_validate(doc)
    db_checks = [v for v in verifiers if v.get("verifier_type") == "database_state"]
    zero = bool(db_checks) and all((v.get("validation_config") or {}).get("comparison_type", "equals") == "equals"
                                   and (v.get("validation_config") or {}).get("expected_value") == 0
                                   and re.match(r"^\s*SELECT\s+COUNT\s*\(", (v.get("validation_config") or {}).get("query", ""), re.IGNORECASE) for v in db_checks)
    # a refusal task only when every zero-count check already holds on the seeded database: a "delete the record" task has
    # the same shape and must stay winnable (M11 audit H3 V11, G3 D11: it became a refusal no action could pass)
    if zero and not _hold_at_seed(db_checks, apps, now):
        zero = False
    if verifiers and not assertions:  # every verifier unported: nothing to grade, refused now, not at grade time (H3 V20)
        raise ValueError(f"{tid}: no verifier Crucible can grade" + (f" ({'; '.join(problems)})" if problems else ""))
    refusal = infeasible if infeasible is not None else (zero or not verifiers)
    who = ""
    ident = actor.get("identity")
    if ident:
        who = f"You are signed in as {ident.get('name') or ident.get('email')}" + (f" <{ident['email']}>" if ident.get("email") and ident.get("name") else "")
        who += f" (user id {ident['user_id']})." if ident.get("user_id") is not None else "."
    system, user = str(cfg.get("system_prompt") or ""), str(cfg.get("user_prompt") or "")
    instruction = "\n\n".join(x for x in (system.strip(), who, user.strip()) if x)
    meta = {"source": "servicenow-enterpriseops-gym", "task_id": tid, "domain": domain, "servers": {app_of[g["mcp_server_name"]]: g["mcp_server_name"] for g in servers},
            "seeds": {a: p.name for a, p in seed_paths.items()}, "selected_tools": list(cfg.get("selected_tools") or []),
            "restricted_tools": list(cfg.get("restricted_tools") or []), "verifiers": len(verifiers), "unported_verifiers": problems,
            "number_of_runs": cfg.get("number_of_runs"), "refusal": refusal}
    return EogTask(id=f"eog/{domain}/{tid}", world_id=spec.world, spec_doc=spec, instruction=instruction, system_prompt=system, user_prompt=user,
                   assertions=assertions, refusal=refusal, pk=pks, meta=meta)


def _hold_at_seed(checks: list[dict], apps: list[dict], now: str) -> bool:
    """Do these ``COUNT(*) = 0`` checks already pass on the seeded databases (schema + seed, pinned ``now``)?"""
    from crucible.world.apps.sqlite_app import pin_now

    dbs = {}
    try:
        for a in apps:
            con = sqlite3.connect(":memory:")
            pin_now(con, now)
            con.executescript(Path(a["params"]["schema_sql"]).read_text())
            for f in a["params"]["seed_sql"]:
                con.executescript(Path(f).read_text())
            dbs[a["id"]] = con
        only = next(iter(dbs.values())) if len(dbs) == 1 else None
        for v in checks:
            con = dbs.get(app_id_for(v.get("gym_name") or ""), only)
            if con is None:
                return False
            try:
                row = con.execute((v.get("validation_config") or {})["query"]).fetchone()
            except sqlite3.Error:
                return False
            if not row or row[0] != 0:
                return False
        return True
    finally:
        for con in dbs.values():
            con.close()


def import_eog_dir(root: str | Path, *, seeds: str | Path, cache_dir: str | Path, pattern: str = "**/*.json") -> tuple[list[EogTask], dict[str, str]]:
    """Every task JSON under ``root`` -> (imported tasks, {path: why skipped})."""
    tasks, skipped = [], {}
    for p in sorted(Path(root).glob(pattern)):
        try:
            tasks.append(import_eog_task(p, seeds=seeds, cache_dir=cache_dir))
        except (ValueError, FileNotFoundError, KeyError, sqlite3.Error) as e:  # one malformed seed skips its task, not the directory
            skipped[str(p)] = str(e)
    return tasks, skipped


def import_hf_rows(rows: list[dict[str, Any]], *, seeds: str | Path, cache_dir: str | Path) -> tuple[list[EogTask], dict[str, str]]:
    """HF dataset rows (``datasets.load_dataset("ServiceNow-AI/EnterpriseOps-Gym", mode, split=domain)``) -> tasks, skipped."""
    tasks, skipped = [], {}
    for r in rows:
        try:
            tasks.append(import_eog_task(r, seeds=seeds, cache_dir=cache_dir))
        except (ValueError, FileNotFoundError, KeyError, sqlite3.Error) as e:
            skipped[str(r.get("task_id"))] = str(e)
    return tasks, skipped


__all__ = ["EogTask", "SeedStore", "app_id_for", "import_eog_dir", "import_eog_task", "import_hf_rows", "infer_schema", "load_task", "referenced_columns"]
