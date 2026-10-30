"""F1 World synthesis, tool first (ScaleEnv 2602.06820's order), held to our gates (part2-paper-core §15.1; evidence doc
§3.2 F1, §3.3 F2).

    worldgen(keyword):
      design  ← tool_designer(keyword)          tool schemas with pre/postconditions, errors, return shapes
      schema  ← schema_deriver(design)          one table per collection the tools touch; every written one exists
      code    ← app_coder(design, schema)       a PyWorld App module (``build(app)``), statically checked
      conform ← the sandbox lists its tools     names, read_only and writes must equal the design
      tests   ← blind_tester(design, schema)    a DIFFERENT family that never sees the code; FROZEN (sha256) at write time
      for round in 1..R:                        tests run in the sandbox (never in this process)
          if every kept test passes: break
          code, disputes ← debugger(code, failures)   code only; the frozen tests are re-hashed before every run
          each dispute → test_approver (a third family, never the debugger's or tester's): approved = the test is
                         dropped, never edited
      accept only at 100% of kept tests, with at most ``max_drop_share`` dropped
      seed    ← seeder(design, schema)          a start state with distractors, loaded and smoke-tested in the sandbox

Crucible's additions over the published recipe: the debugger cannot touch expectations (the tests file's digest is
checked before every run, and a mismatch aborts), a disputed test needs a different-family approval and is dropped, not
rewritten, and the App code only ever runs in the restricted subprocess of ``sandbox.py``.

Output, under ``out_dir``: ``design.json``, ``schema.json``, ``app.py``, ``tests.json`` (frozen), ``test_report.json``,
``seed.json``, ``world.json`` (the manifest: file digests, the frozen tests digest, rounds, dropped tests, the role
families, the sandbox kind). ``open_world(dir, work)`` is the ``WorldRef`` factory the tasks of the domain use.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.roles import (
    APP_API_DOC,
    APP_CODER,
    BLIND_TESTER,
    DEBUGGER,
    SCHEMA_DERIVER,
    SEEDER,
    TEST_APPROVER,
    TEST_FORMAT_DOC,
    TOOL_DESIGNER,
    AppCodeOut,
    ApproveOut,
    DebugOut,
    RoleClient,
    RoleFailed,
    SchemaOut,
    SeedOut,
    ToolDesignOut,
    ToolTestsOut,
    ask,
)
from crucible.synth.sandbox import SANDBOX_KIND, SandboxError, SandboxWorld, check_code, run_tool_tests, static_info

WORLDGEN_ROLES = ("tool_designer", "schema_deriver", "app_coder", "blind_tester", "debugger", "test_approver", "seeder")
TYPES = {"string", "integer", "number", "boolean", "array", "object"}


class WorldgenReport(CrucibleModel):
    keyword: str
    app: str | None = None
    status: str = "started"  # accepted | rejected:<stage>
    reasons: list[str] = Field(default_factory=list)
    tools: int = 0
    tables: int = 0
    tests_written: int = 0
    tests_dropped: list[str] = Field(default_factory=list)
    disputes_refused: list[str] = Field(default_factory=list)
    debug_rounds: int = 0
    pass_history: list[float] = Field(default_factory=list)  # kept-test pass rate after each run
    tests_digest: str | None = None
    seed_rows: dict[str, int] = Field(default_factory=dict)
    families: dict[str, str] = Field(default_factory=dict)
    sandbox: str = SANDBOX_KIND
    dir: str | None = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jwrite(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------------------------------------- checks

def check_design(d: ToolDesignOut) -> None:
    names = [t.name for t in d.tools]
    if len(set(names)) != len(names):
        raise ValueError("tool names repeat")
    if not 6 <= len(d.tools) <= 16:
        raise ValueError(f"design 6 to 16 tools, not {len(d.tools)}")
    ro = [t for t in d.tools if t.read_only]
    rw = [t for t in d.tools if not t.read_only]
    if len(ro) < 2 or len(rw) < 2:
        raise ValueError("the API needs at least two read-only tools and two write tools")
    for t in d.tools:
        if t.read_only and t.writes:
            raise ValueError(f"tool {t.name} is read-only but declares writes")
        if not t.read_only and not t.writes:
            raise ValueError(f"write tool {t.name} declares no collections it writes")
        pn = [p.name for p in t.params]
        if len(set(pn)) != len(pn) or not all(p.isidentifier() for p in pn):
            raise ValueError(f"tool {t.name}: parameter names must be unique identifiers")
        if any(p.name in ("ctx", "self") for p in t.params):
            raise ValueError(f"tool {t.name}: ctx and self are reserved parameter names")


def check_schema(s: SchemaOut, d: ToolDesignOut) -> None:
    names = {t.name for t in s.tables}
    if len(names) != len(s.tables):
        raise ValueError("table names repeat")
    for t in s.tables:
        if not t.name.isidentifier():
            raise ValueError(f"table name {t.name!r} is not an identifier")
        if t.key not in t.fields:
            raise ValueError(f"table {t.name}: the key {t.key!r} is not one of its fields")
        bad = {f: ty for f, ty in t.fields.items() if ty.rstrip("?") not in TYPES}
        if bad:
            raise ValueError(f"table {t.name}: unknown field types {bad}")
    written = {w for tool in d.tools for w in tool.writes}
    missing = sorted(written - names)
    if missing:
        raise ValueError(f"the tools write collections that have no table: {missing}")


def check_tests(tests: ToolTestsOut, d: ToolDesignOut, s: SchemaOut) -> None:
    tools = {t.name for t in d.tools}
    tables = {t.name: t for t in s.tables}
    ids = [t.id for t in tests.tests]
    if len(set(ids)) != len(ids):
        raise ValueError("test ids repeat")
    if len(tests.tests) < 10:
        raise ValueError(f"write at least 10 tests, not {len(tests.tests)}")
    for t in tests.tests:
        for c in t.calls:
            if c.tool.split(".")[-1] not in tools:
                raise ValueError(f"test {t.id}: unknown tool {c.tool!r}")
        for tn, rows in t.fixture.items():
            if tn not in tables:
                raise ValueError(f"test {t.id}: fixture table {tn!r} is not in the schema")
            for r in rows:
                if tables[tn].key not in r:
                    raise ValueError(f"test {t.id}: a {tn} row has no key field {tables[tn].key!r}")
        for tn in [*t.expect_state, *t.expect_absent]:
            if tn not in tables:
                raise ValueError(f"test {t.id}: table {tn!r} is not in the schema")
    covered = {c.tool.split(".")[-1] for t in tests.tests for c in t.calls}
    missing = sorted(tools - covered)
    if missing:
        raise ValueError(f"no test calls the tools {missing}")


def check_seed(seed: SeedOut, s: SchemaOut) -> None:
    tables = {t.name: t for t in s.tables}
    for tn, rows in seed.tables.items():
        if tn not in tables:
            raise ValueError(f"seed table {tn!r} is not in the schema")
        keys = [r.get(tables[tn].key) for r in rows]
        if any(k is None for k in keys):
            raise ValueError(f"a {tn} row has no key field {tables[tn].key!r}")
        if len({str(k) for k in keys}) != len(keys):
            raise ValueError(f"table {tn}: keys repeat")
    empty = sorted(t for t in tables if not seed.tables.get(t))
    if len(empty) == len(tables):
        raise ValueError("the seed has no rows")


def conformance(info: Mapping[str, Any], d: ToolDesignOut) -> list[str]:
    """The registered tools against the design: names, read_only flags, write declarations."""
    got = {t["name"]: t for t in info.get("tools", [])}
    out = []
    for t in d.tools:
        g = got.get(t.name)
        if g is None:
            out.append(f"tool {t.name} is not registered")
            continue
        if bool(g["read_only"]) != t.read_only:
            out.append(f"tool {t.name}: read_only is {g['read_only']}, the design says {t.read_only}")
        if not t.read_only and sorted(g["writes"]) != sorted(t.writes):
            out.append(f"tool {t.name}: writes {sorted(g['writes'])}, the design says {sorted(t.writes)}")
    extra = sorted(set(got) - {t.name for t in d.tools})
    if extra:
        out.append(f"tools not in the design: {extra}")
    return out


# ---------------------------------------------------------------------------------------------------------- the pipeline

def _design_json(d: ToolDesignOut) -> dict[str, Any]:
    return d.model_dump(mode="json")


async def test_and_debug(out: Path, design: ToolDesignOut, design_doc: Mapping[str, Any], schema_doc: Mapping[str, Any], code: AppCodeOut,
                         tests: ToolTestsOut, frozen: str, clients: Mapping[str, RoleClient], rep: WorldgenReport, *, max_debug_rounds: int = 3,
                         max_drop_share: float = 0.25, exclude_tools: frozenset[str] | set[str] = frozenset(), app_api: str = APP_API_DOC,
                         tag: str = "worldgen") -> list[str]:
    """Run the frozen blind tests (and the conformance check) in the sandbox; while any fails, the debugger edits the code
    only, and a disputed test is dropped only on a third family's approval (module docstring). Returns the dropped test
    ids; on failure ``rep.status`` is ``rejected:tests`` with the reason. ``exclude_tools``: registered tools that are not
    the design's (the framework kit). ``design_doc`` is what the roles are shown (the design plus any platform notes)."""
    tests_path = out / "tests.json"
    work = out / "work"
    by_id = {t.id: t.model_dump(mode="json") for t in tests.tests}
    dropped: list[str] = []

    async def run_kept() -> list[dict[str, Any]]:
        if _sha(tests_path) != frozen:  # nothing may edit the frozen expectations
            raise RuntimeError("the frozen tests file changed")
        kept = [t for tid, t in by_id.items() if tid not in dropped]
        res = await run_tool_tests(out, work / f"t{uuid.uuid4().hex[:8]}", kept)
        try:
            info = await static_info(out, work / f"i{uuid.uuid4().hex[:8]}")
            info = {**info, "tools": [t for t in info.get("tools", []) if t["name"] not in exclude_tools]}
            conf = conformance(info, design)
        except SandboxError as e:
            conf = [f"the module does not load: {e}"]
        res += [{"id": f"conformance:{i}", "passed": False, "detail": c} for i, c in enumerate(conf)]
        return res

    def code_ok(c: AppCodeOut) -> None:
        probs = check_code(c.code)
        if probs:
            raise ValueError("the module fails the sandbox's static check: " + "; ".join(probs[:8]))

    results = await run_kept()
    history: list[dict[str, Any]] = []
    for rnd in range(max_debug_rounds + 1):
        failing = [r for r in results if not r["passed"]]
        tests_kept = [r for r in results if not str(r["id"]).startswith("conformance:")]
        rep.pass_history.append(round(sum(r["passed"] for r in tests_kept) / max(1, len(tests_kept)), 4))
        history.append({"round": rnd, "failing": [r["id"] for r in failing], "dropped": list(dropped)})
        if not failing or rnd == max_debug_rounds:
            break
        rep.debug_rounds = rnd + 1
        fails = [{"test": by_id.get(r["id"]), "id": r["id"], "detail": r["detail"]} for r in failing][:20]
        try:
            fix: DebugOut = await ask(clients["debugger"], DEBUGGER, tag=f"{tag}:debug", check=code_ok, retries=1, design=dict(design_doc),
                                      schema=dict(schema_doc), code=code.code, failures=fails, app_api=app_api)
        except RoleFailed as e:
            history.append({"round": rnd, "debugger_failed": str(e)[:500]})
            break
        code = AppCodeOut(code=fix.code)
        (out / "app.py").write_text(code.code)
        failing_ids = {r["id"] for r in failing}
        for dsp in fix.disputes:
            if dsp.test not in by_id or dsp.test not in failing_ids or dsp.test in dropped:
                continue
            fail_detail = next((r["detail"] for r in failing if r["id"] == dsp.test), "")
            try:
                verdict: ApproveOut = await ask(clients["test_approver"], TEST_APPROVER, tag=f"{tag}:approve", retries=1, design=dict(design_doc),
                                                schema=dict(schema_doc), test=by_id[dsp.test], failure=fail_detail, dispute=dsp.why)
            except RoleFailed:
                rep.disputes_refused.append(dsp.test)
                continue
            if verdict.test_is_wrong:
                dropped.append(dsp.test)
            else:
                rep.disputes_refused.append(dsp.test)
        if len(dropped) > max_drop_share * len(by_id):
            rep.status = "rejected:tests"
            rep.reasons.append(f"{len(dropped)} of {len(by_id)} tests were dropped as wrong (cap {max_drop_share:.0%}): the spec is not testable")
            rep.tests_dropped = list(dropped)
            return dropped
        results = await run_kept()
    rep.tests_dropped = list(dropped)
    _jwrite(out / "test_report.json", {"results": results, "history": history, "dropped": dropped, "frozen_digest": frozen})
    failing = [r for r in results if not r["passed"]]
    if failing:
        rep.status = "rejected:tests"
        rep.reasons.append(f"{len(failing)} tests still fail after {rep.debug_rounds} debug rounds: " + ", ".join(str(r["id"]) for r in failing[:10]))
    return dropped


async def worldgen(keyword: str, out_dir: str | Path, clients: Mapping[str, RoleClient], *, guidance: str = "", max_debug_rounds: int = 3,
                   max_drop_share: float = 0.25, rows: tuple[int, int] | None = None) -> WorldgenReport:
    """Synthesize an App for ``keyword`` into ``out_dir`` (module docstring). ``clients``: one ``RoleClient`` per
    ``WORLDGEN_ROLES`` entry (the wire checks their families). ``rows`` = (lo, hi) rows per table asks the seeder for a
    denser start state (more sibling entities per collection, so MUT's neighbourhood reaches n >= 59; §6)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    rep = WorldgenReport(keyword=keyword, dir=str(out), families={r: getattr(clients[r], "family", "?") for r in WORLDGEN_ROLES if r in clients})

    def fail(stage: str, why: str) -> WorldgenReport:
        rep.status = f"rejected:{stage}"
        rep.reasons.append(why[:1500])
        _jwrite(out / "worldgen_report.json", rep.model_dump(mode="json"))
        return rep

    # 1-2: tools first, then the schema derived from them
    try:
        design: ToolDesignOut = await ask(clients["tool_designer"], TOOL_DESIGNER, tag="worldgen:design", check=check_design, retries=2,
                                          domain=keyword, guidance=guidance or None)
    except RoleFailed as e:
        return fail("design", str(e))
    rep.app, rep.tools = design.app, len(design.tools)
    _jwrite(out / "design.json", _design_json(design))
    try:
        schema: SchemaOut = await ask(clients["schema_deriver"], SCHEMA_DERIVER, tag="worldgen:schema", check=lambda s: check_schema(s, design), retries=2,
                                      design=_design_json(design))
    except RoleFailed as e:
        return fail("schema", str(e))
    rep.tables = len(schema.tables)
    _jwrite(out / "schema.json", schema.model_dump(mode="json"))

    # 3: code, statically checked (the feedback names the problems, nothing else)
    def code_ok(c: AppCodeOut) -> None:
        probs = check_code(c.code)
        if probs:
            raise ValueError("the module fails the sandbox's static check: " + "; ".join(probs[:8]))

    try:
        code: AppCodeOut = await ask(clients["app_coder"], APP_CODER, tag="worldgen:code", check=code_ok, retries=2, design=_design_json(design),
                                     schema=schema.model_dump(mode="json"), app_api=APP_API_DOC)
    except RoleFailed as e:
        return fail("code", str(e))
    (out / "app.py").write_text(code.code)

    # 4: blind tests from the spec, frozen at write time
    try:
        tests: ToolTestsOut = await ask(clients["blind_tester"], BLIND_TESTER, tag="worldgen:tests", check=lambda t: check_tests(t, design, schema),
                                        retries=2, design=_design_json(design), schema=schema.model_dump(mode="json"), test_format=TEST_FORMAT_DOC)
    except RoleFailed as e:
        return fail("tests", str(e))
    tests_path = out / "tests.json"
    _jwrite(tests_path, tests.model_dump(mode="json"))
    frozen = _sha(tests_path)
    rep.tests_digest, rep.tests_written = frozen, len(tests.tests)
    dropped = await test_and_debug(out, design, _design_json(design), schema.model_dump(mode="json"), code, tests, frozen, clients, rep,
                                   max_debug_rounds=max_debug_rounds, max_drop_share=max_drop_share)
    if rep.status.startswith("rejected"):
        _jwrite(out / "worldgen_report.json", rep.model_dump(mode="json"))
        return rep
    by_id = {t.id: t for t in tests.tests}

    # 5: the start state, loaded and smoke-tested in the sandbox
    seed_err = ""
    for attempt in range(3):
        try:
            seed: SeedOut = await ask(clients["seeder"], SEEDER, tag="worldgen:seed", check=lambda x: check_seed(x, schema), retries=1,
                                      design=_design_json(design), schema=schema.model_dump(mode="json"),
                                      guidance=(guidance + ("\n" + seed_err if seed_err else "")) or None,
                                      rows=f"{rows[0]} to {rows[1]} rows per table" if rows else None)
        except RoleFailed as e:
            return fail("seed", str(e))
        _jwrite(out / "seed.json", seed.model_dump(mode="json"))
        try:
            smoke = await smoke_world(out, work / f"s{attempt}")
        except (SandboxError, ValueError) as e:
            seed_err = f"The previous start state did not load: {str(e)[:800]}"
            continue
        if smoke:
            seed_err = "The previous start state broke read tools: " + "; ".join(smoke[:5])
            continue
        rep.seed_rows = {t: len(r) for t, r in seed.tables.items()}
        break
    else:
        return fail("seed", seed_err)

    rep.status = "accepted"
    manifest = {"keyword": keyword, "app": design.app, "files": {f: _sha(out / f) for f in ("design.json", "schema.json", "app.py", "tests.json", "seed.json")},
                "tests_digest": frozen, "tests": len(by_id), "dropped": dropped, "debug_rounds": rep.debug_rounds, "families": rep.families,
                "sandbox": SANDBOX_KIND}
    _jwrite(out / "world.json", manifest)
    _jwrite(out / "worldgen_report.json", rep.model_dump(mode="json"))
    return rep


async def smoke_world(app_dir: Path, work: Path) -> list[str]:
    """Start the World on its seed and call every read-only tool that needs no argument: problems (empty = fine). A
    tool error is fine (a refusal); a crash, an undeclared write or a dead child is not."""
    w = SandboxWorld(app_dir, work)
    probs: list[str] = []
    try:
        await w.reset(0)
        c = await w.canon()
        if not any(rows for colls in c.apps.values() for rows in colls.values()):
            probs.append("the start state is empty")
        for t in await w.tools():
            if not t.read_only or (t.input_schema.get("required") or []):
                continue
            o = await w.call(t.name, {})
            if o.error and ("Error" in o.error.split(":", 1)[0] and not o.error.startswith("ToolError")):
                probs.append(f"{t.name} crashed on the seed: {o.error[:200]}")
    finally:
        await w.close()
    return probs


class ReseedReport(CrucibleModel):
    source: str
    dir: str
    status: str = "started"  # accepted | rejected:<why>
    target: dict[str, int] = Field(default_factory=dict)  # table -> rows wanted in total
    before: dict[str, int] = Field(default_factory=dict)
    after: dict[str, int] = Field(default_factory=dict)
    added: dict[str, int] = Field(default_factory=dict)
    refused: dict[str, str] = Field(default_factory=dict)  # table -> why its new rows were not kept
    reasons: list[str] = Field(default_factory=list)


COPY_FILES = ("design.json", "schema.json", "app.py", "tests.json", "test_report.json")


def check_added(new: SeedOut, table: str, existing: Mapping[str, list[dict[str, Any]]], s: SchemaOut) -> None:
    """New rows for one table: only that table, every row keyed, no key that exists or repeats, every schema field."""
    t = {x.name: x for x in s.tables}[table]
    extra = sorted(set(new.tables) - {table})
    if extra:
        raise ValueError(f"reply with rows for table {table!r} only, not {extra}")
    rows = new.tables.get(table) or []
    if not rows:
        raise ValueError(f"no new rows for table {table!r}")
    have = {str(r.get(t.key)) for r in existing.get(table, [])}
    keys = [r.get(t.key) for r in rows]
    if any(k is None for k in keys):
        raise ValueError(f"a new {table} row has no key field {t.key!r}")
    dup = sorted({str(k) for k in keys} & have)
    if dup:
        raise ValueError(f"keys already used in {table}: {dup[:8]} (new rows need new keys)")
    if len({str(k) for k in keys}) != len(keys):
        raise ValueError(f"new {table} keys repeat")
    missing = sorted({f for r in rows for f in t.fields if f not in r})
    if missing:
        raise ValueError(f"new {table} rows miss schema fields {missing[:8]}")


async def reseed(src_dir: str | Path, out_dir: str | Path, seeder: RoleClient, *, target: Mapping[str, int], guidance: str = "") -> ReseedReport:
    """A denser start state for an accepted World (the seeding knob of §6: more sibling entities per collection, so the
    MUT neighbourhood of a generated task reaches n >= 59). The design, schema, App code and frozen tests are copied
    unchanged; the seeder (the swarm) writes NEW rows per table, one table at a time, up to ``target[table]`` rows in
    total, given every existing row; existing rows are never changed. The merged state is checked against the schema,
    loaded and smoke-tested in the sandbox; a table whose new rows break the load is reverted and recorded."""
    src, out = Path(src_dir), Path(out_dir)
    rep = ReseedReport(source=str(src), dir=str(out), target=dict(target))
    if load_report(src).status != "accepted":
        rep.status = "rejected:source"
        rep.reasons.append("the source World was not accepted")
        return rep
    out.mkdir(parents=True, exist_ok=True)
    for f in COPY_FILES:
        if (src / f).exists():
            (out / f).write_bytes((src / f).read_bytes())
    design = ToolDesignOut.model_validate_json((src / "design.json").read_text())
    schema = SchemaOut.model_validate_json((src / "schema.json").read_text())
    seed = SeedOut.model_validate_json((src / "seed.json").read_text())
    tables: dict[str, list[dict[str, Any]]] = {k: list(v) for k, v in seed.tables.items()}
    rep.before = {k: len(v) for k, v in tables.items()}
    for table, want in target.items():
        if table not in {t.name for t in schema.tables}:
            rep.refused[table] = "not a table of the schema"
            continue
        n_new = want - len(tables.get(table, []))
        if n_new <= 0:
            continue
        try:
            new: SeedOut = await ask(seeder, SEEDER, tag="worldgen:reseed", check=lambda x, t=table, have=tables: check_added(x, t, have, schema), retries=2,
                                     design=_design_json(design), schema=schema.model_dump(mode="json"), existing=tables, guidance=guidance or None,
                                     rows=f"add exactly {n_new} new rows to table {table} only (it has {len(tables.get(table, []))}); about a third of "
                                          "them distractors of existing rows (the same pattern with one field different)")
        except RoleFailed as e:
            rep.refused[table] = f"seeder: {str(e)[:300]}"
            continue
        trial = {**tables, table: [*tables.get(table, []), *new.tables[table]]}
        _jwrite(out / "seed.json", SeedOut(tables=trial, notes=seed.notes).model_dump(mode="json"))
        try:
            probs = await smoke_world(out, out / "work" / f"r{uuid.uuid4().hex[:8]}")
        except (SandboxError, ValueError) as e:
            probs = [f"does not load: {str(e)[:300]}"]
        if probs:
            rep.refused[table] = "; ".join(probs[:3])[:500]
            continue
        tables = trial
        rep.added[table] = len(new.tables[table])
    final = SeedOut(tables=tables, notes=(seed.notes + f" | reseeded from {src.name}: +{sum(rep.added.values())} rows").strip(" |"))
    check_seed(final, schema)
    _jwrite(out / "seed.json", final.model_dump(mode="json"))
    probs = await smoke_world(out, out / "work" / f"f{uuid.uuid4().hex[:8]}")
    if probs:
        rep.status = "rejected:smoke"
        rep.reasons += probs
        return rep
    rep.after = {k: len(v) for k, v in tables.items()}
    base = load_report(src)
    acc = base.model_copy(update={"seed_rows": rep.after, "dir": str(out), "reasons": [f"reseeded from {src}"]})
    _jwrite(out / "worldgen_report.json", acc.model_dump(mode="json"))
    manifest = json.loads((src / "world.json").read_text()) if (src / "world.json").exists() else {}
    manifest.update({"files": {f: _sha(out / f) for f in (*COPY_FILES, "seed.json") if (out / f).exists()}, "parent": str(src),
                     "reseed": rep.model_dump(mode="json", include={"target", "added", "refused"})})
    _jwrite(out / "world.json", manifest)
    rep.status = "accepted"
    _jwrite(out / "reseed_report.json", rep.model_dump(mode="json"))
    return rep


def open_world(dir: str, work: str) -> SandboxWorld:
    """The ``WorldRef`` factory of a synthesized domain: a fresh sandboxed World (its own child) per call."""
    w = Path(work) / f"w{uuid.uuid4().hex[:10]}"
    return SandboxWorld(dir, w)


def load_report(dir: str | Path) -> WorldgenReport:
    return WorldgenReport.model_validate_json((Path(dir) / "worldgen_report.json").read_text())
