"""The gates a World family and each World instance must pass before tasks are built on it (part2-paper-core §6, §11).

Every gate is executed where it can be, and its result (passed, the measured details, the reasons) is recorded per
instance in ``gates.json``. An instance failing any gate is **not used**; the reasons say why.

| gate | what is executed | passes when |
|---|---|---|
| ``build`` | the family's frozen blind tests, re-run in the sandbox against the instance's App directory; the copied files re-hashed against the frozen manifest; the sandbox's static check run on the App and on canary modules that must be refused | every kept test passes, every digest matches, the sandbox kind is not ``none``, the canaries are refused |
| ``determinism`` | the instance regenerated from its config; reset twice in one child and once in a fresh child; every reference program run twice | the seed digest and every canonical fingerprint agree |
| ``integrity`` (realism) | ``datagen.integrity`` (types, references, declared values, dates ordered and not in the future, rules, master copies, aggregates); documents against the policy parameters; tool enums against the data | no problem |
| ``surface`` (realism) | pagination (page 1 and 2 of a list tool), filter parameters, every write tool called with an unknown id and with an unknown enum value, the permission test | a list tool pages, a read tool filters, ≥ 80% of write tools refuse an unknown id with a coded tool error, a passing blind test expects PERMISSION_DENIED |
| ``obscurity`` (realism) | every document read through the kit tools; every colleague asked with the fact's keywords; every read tool without required arguments paged through | every policy parameter's value is stated in a document the agent can read, every colleague fact comes back, ≥ 80% of each target collection's entities are listed by some read tool |
| ``density`` | for each family write tool, a **reference oracle** (the tool called on instance entities with arguments taken from its blind tests, fixture-like rows first, executed until it changes state), its skeleton contract, and the MUT falsifier with its effect deduplication | every target collection holds ≥ K entities and is written by some reference oracle reaching n ≥ 59 distinct mutants; tools below 59 are recorded as ``thin_tools`` |
| ``panel`` (realism) | two rater families (never a builder's) score a summary of the instance against ``RUBRIC``; λ calibrated on anchors whose truth is known (a hand-built pack is realistic; framework degradations of it are not) | both agree it is realistic and not obscure at confidence ≥ λ; disagreement or low confidence → ``human_queue.jsonl`` (not used until decided) |
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import re
import statistics
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from crucible.synth import policy
from crucible.synth.datagen import DataSpec, DataSpecError, Generated, NoiseItem, eval_expr, integrity
from crucible.synth.domain import DomainSpec
from crucible.synth.families import (
    PAGE_PARAMS,
    InstanceConfig,
    SchemaOut,
    _jread,
    _jwrite,
    _materialize,
    _sha,
    frozen_manifest,
    schema_map,
    targets,
)
from crucible.synth.iface import SynthWorld
from crucible.synth.program import Lit, Program, Run, Step, run
from crucible.synth.roles import Role, RoleClient, compact
from crucible.synth.sandbox import SANDBOX_KIND, SandboxWorld, check_code, run_tool_tests
from crucible.synth.worldkit import KIT_NAMES, fmt_value, placeholders
from crucible.verify.calibrate import clopper_pearson_upper

KIT_TABLE_NAMES = ("documents", "policy_params", "colleagues")
N_MIN = int(policy.param("R-density", "n_min"))  # R-density (DESIGN item 24): never a constant of its own
CODES = ("NOT_FOUND", "INVALID_ARGUMENT", "INVALID_STATE", "PERMISSION_DENIED", "CONFLICT")
PAGEISH = frozenset({"page", "page_size", "per_page", "limit", "offset", "cursor"})


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class GateResult(_M):
    gate: str
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    elapsed_s: float = 0.0


class InstanceGates(_M):
    family: str
    instance: str
    dir: str
    gates: dict[str, GateResult] = Field(default_factory=dict)
    usable: bool = False
    why: list[str] = Field(default_factory=list)
    remedies: list[dict[str, Any]] = Field(default_factory=list)  # policy.resolve's remedy per failing gate (DESIGN item 24)

    def decide(self, required: Sequence[str]) -> None:
        self.why = [f"{g}: not run" for g in required if g not in self.gates]
        self.why += [f"{g}: {'; '.join(r.reasons[:3]) or 'failed'}" for g, r in self.gates.items() if not r.passed]
        self.usable = not self.why


REQUIRED = ("build", "determinism", "integrity", "surface", "obscurity", "density", "panel")


def _app_name(d: Path) -> str:
    return str(_jread(d / "design.json")["app"])


async def _closing(w: SandboxWorld) -> None:
    await w.close()


# ---------------------------------------------------------------------------------------------------------- build

CANARIES = {"import_os": "import os\ndef build(app):\n    pass\n", "open_file": "def build(app):\n    open('/etc/hosts')\n",
            "dunder": "def build(app):\n    x = ().__class__\n"}


async def build_gate(fdir: Path, inst: Path) -> GateResult:
    t0 = time.monotonic()
    reasons: list[str] = []
    man = frozen_manifest(fdir)
    for f in ("app.py", "tests.json", "design.json", "schema.json"):
        if _sha(inst / f) != man["files"][f]:
            reasons.append(f"{f} differs from the frozen family")
    probs = check_code((inst / "app.py").read_text())
    if probs:
        reasons.append("the App fails the static check: " + "; ".join(probs[:3]))
    canaries = {k: bool(check_code(v)) for k, v in CANARIES.items()}
    if not all(canaries.values()):
        reasons.append(f"the static check admits canaries {[k for k, v in canaries.items() if not v]}")
    if SANDBOX_KIND == "none":
        reasons.append("no OS sandbox layer on this host")
    tests = [t for t in _jread(inst / "tests.json")["tests"] if t["id"] not in set(man.get("dropped") or [])]
    res = await run_tool_tests(inst, inst / "work" / f"bt{uuid.uuid4().hex[:8]}", tests)
    failed = [r["id"] for r in res if not r["passed"]]
    if failed:
        reasons.append(f"{len(failed)} of {len(res)} kept blind tests fail: {failed[:6]}")
    return GateResult(gate="build", passed=not reasons, reasons=reasons, elapsed_s=round(time.monotonic() - t0, 2),
                      details={"tests_kept": len(res), "tests_failed": failed, "dropped": man.get("dropped") or [], "sandbox": SANDBOX_KIND,
                               "canaries_refused": canaries})


# ---------------------------------------------------------------------------------------------------------- determinism

async def canon_fingerprint(inst: Path, work: Path, *, resets: int = 2) -> list[str]:
    w = SandboxWorld(inst, work)
    try:
        out = []
        for _ in range(resets):
            await w.reset(0)
            out.append((await w.canon()).fingerprint())
        return out
    finally:
        await w.close()


async def determinism_gate(fdir: Path, inst: Path, *, replays: Mapping[str, bool] | None = None) -> GateResult:
    t0 = time.monotonic()
    reasons: list[str] = []
    info = _jread(inst / "instance.json")
    cfg = InstanceConfig.model_validate(info["config"])
    app = fdir / "app"
    man = frozen_manifest(fdir)
    redo = inst / "work" / f"regen-{uuid.uuid4().hex[:8]}"
    _materialize(app, redo, DataSpec.model_validate(_jread(app / "datagen.json")), DomainSpec.model_validate_json((app / "domain.json").read_text()),
                 SchemaOut.model_validate(_jread(app / "schema.json")), cfg, k_min=int(cfg.k_min or man.get("k_min", 40)),
                 docs=_docs(app), actors=_actors(app))
    regen_same = _sha(redo / "seed.json") == _sha(inst / "seed.json")
    if not regen_same:
        reasons.append("regenerating the instance from its config gives a different start state")
    a = await canon_fingerprint(inst, inst / "work" / f"d{uuid.uuid4().hex[:6]}")
    b = await canon_fingerprint(inst, inst / "work" / f"d{uuid.uuid4().hex[:6]}", resets=1)
    if len(set(a + b)) != 1:
        reasons.append("reset gives different canonical states")
    bad_replays = sorted(k for k, ok in (replays or {}).items() if not ok)
    if bad_replays:
        reasons.append(f"reference programs replay to different end states: {bad_replays}")
    return GateResult(gate="determinism", passed=not reasons, reasons=reasons, elapsed_s=round(time.monotonic() - t0, 2),
                      details={"regenerated_identical": regen_same, "fingerprints": sorted(set(a + b)), "replays": dict(replays or {})})


def _docs(app: Path) -> Any:
    from crucible.synth.families import DocsOut

    return DocsOut.model_validate(_jread(app / "docs.json"))


def _actors(app: Path) -> Any:
    from crucible.synth.families import ActorsOut

    return ActorsOut.model_validate(_jread(app / "actors.json"))


# ---------------------------------------------------------------------------------------------------------- integrity

def _hkey(v: Any) -> Any:
    """A hashable stand-in for a cell value (a list or dict as canonical JSON)."""
    return json.dumps(v, sort_keys=True) if isinstance(v, list | dict) else v


def _json_or(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def integrity_gate(fdir: Path, inst: Path) -> GateResult:
    t0 = time.monotonic()
    app = fdir / "app"
    sm = schema_map(_jread(app / "schema.json"))
    data_tables = {k: v for k, v in sm.items() if k not in KIT_TABLE_NAMES}
    seed = _jread(inst / "seed.json")["tables"]
    noise = [NoiseItem.model_validate(n) for n in _jread(inst / "noise.json")]
    spec = DataSpec.model_validate(_jread(app / "datagen.json"))
    g = Generated(tables={k: v for k, v in seed.items() if k in data_tables}, noise=noise)
    probs = integrity(spec, data_tables, g)
    reasons = [f"data: {p}" for p in probs[:12]]
    # documents agree with the policy parameters the App reads
    info = _jread(inst / "instance.json")
    params = info["params"]
    rows = {r["name"]: r["value"] for r in seed.get("policy_params", [])}
    for k, v in params.items():
        if rows.get(k) != v:
            reasons.append(f"policy_params.{k} = {rows.get(k)!r} but the instance's policy variant is {v!r}")
    dspec = DomainSpec.model_validate_json((app / "domain.json").read_text())
    for c in dspec.param_constraints:
        try:
            if not eval_expr(c, params):
                reasons.append(f"the policy variant breaks the constraint {c!r}")
        except DataSpecError as e:
            reasons.append(f"param constraint {c!r} cannot be evaluated: {e}")
    templates = {d["doc_id"]: d for d in _jread(app / "docs.json")["documents"]}
    docs = {d["doc_id"]: d for d in seed.get("documents", [])}
    doc_checks = 0
    for did, t in templates.items():
        body = str(docs.get(did, {}).get("body", ""))
        if not body:
            reasons.append(f"document {did} is missing from the instance")
            continue
        if "{{" in body:
            reasons.append(f"document {did} has an unrendered placeholder")
        for ph in placeholders(t["body"]):
            doc_checks += 1
            if fmt_value(params.get(ph)) not in body:
                reasons.append(f"document {did} does not state {ph} = {fmt_value(params.get(ph))}")
    # enums: tool enums against the data's values of a same-named field (stale markers aside)
    design = _jread(app / "design.json")
    stale = {(t.table, f): v for t in spec.tables if t.stale for f, v in t.stale.set.items()}
    enum_checks, enum_bad, enum_disjoint = 0, [], []
    def norm(x: str) -> str:
        return re.sub(r"[\s_\-]+", "_", x.strip().lower())

    for tool in design.get("tools", []):
        if tool.get("read_only"):
            continue
        for p in tool.get("params", []):
            if not p.get("enum"):
                continue
            allowed = {str(x) for x in p["enum"]}
            for tn in tool.get("writes") or []:  # the table the tool writes that field of (pilot 1: comparing every table's
                rws = g.tables.get(tn) or []     # "status" against one tool's enum was a false alarm)
                if not rws or p["name"] not in rws[0]:
                    continue
                vals = {str(r.get(p["name"])) for r in rws if r.get(p["name"]) is not None and r.get(p["name"]) != stale.get((tn, p["name"]))}
                enum_checks += 1
                clash = sorted(v for v in vals - allowed if norm(v) in {norm(a) for a in allowed})  # "In Progress" vs "in_progress"
                if clash:  # another spelling of the same value: the agent would send what the tool refuses
                    enum_bad.append(f"{tool['name']}.{p['name']} accepts {sorted(allowed)[:8]} but {tn}.{p['name']} holds {clash[:6]}")
                elif vals and not vals & allowed:  # pilot 1: usually two concepts under one name (a release reason vs a hold reason)
                    enum_disjoint.append(f"{tool['name']}.{p['name']} vs {tn}.{p['name']}")
    reasons += enum_bad[:5]
    unused = []
    for t in spec.tables:
        n = len(g.tables.get(t.table, []))
        for f, gen in t.fields.items():
            if gen.get("kind") == "choice" and n >= 30:
                # an array or object field's values are lists or dicts drawn from JSON text (worlds-2b: PR's line items;
                # a set of them was a TypeError, so the gate could not run on any PR instance)
                seen = {_hkey(r.get(f)) for r in g.tables.get(t.table, [])}
                missing = [v for v in gen["values"] if _hkey(v) not in seen and _hkey(_json_or(v)) not in seen]
                if missing:
                    unused.append(f"{t.table}.{f}: {missing}")
    return GateResult(gate="integrity", passed=not reasons, reasons=reasons, elapsed_s=round(time.monotonic() - t0, 2),
                      details={"data_problems": len(probs), "doc_param_checks": doc_checks, "enum_checks": enum_checks, "enum_disjoint": enum_disjoint,
                               "declared_values_never_used": unused[:10], "noise": {k: sum(1 for n in noise if n.kind == k) for k in ("near_duplicate", "stale", "conflict")}})


# ---------------------------------------------------------------------------------------------------------- tool surface

def _items(v: Any) -> list[Any] | None:
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        for k in ("items", "results", "records", "data"):
            if isinstance(v.get(k), list):
                return v[k]
        lists = [x for x in v.values() if isinstance(x, list)]
        if len(lists) == 1:
            return lists[0]
    return None


def _ok_refusal(err: str | None) -> bool:
    return bool(err) and str(err).startswith("ToolError")


def _coded(err: str | None) -> bool:
    return bool(err) and any(c in str(err) for c in CODES)


def _passing_calls(inst: Path) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    """tool -> [(test, call)] for calls expected to succeed in blind tests that passed."""
    tests = _jread(inst / "tests.json")["tests"]
    rep = _jread(inst / "test_report.json") if (inst / "test_report.json").exists() else {"results": []}
    passed = {r["id"] for r in rep.get("results", []) if r.get("passed")} or {t["id"] for t in tests}
    out: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for t in tests:
        if t["id"] not in passed:
            continue
        for c in t.get("calls", []):
            if c.get("expect", "ok") == "ok":
                out.setdefault(c["tool"].split(".")[-1], []).append((t, c))
    return out


async def _replay_errors(inst: Path, test: Mapping[str, Any]) -> list[str]:
    """The error messages of a blind test's expected refusals, replayed in the sandbox on the test's own fixture."""
    import shutil

    d = inst / "work" / f"rp{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    for f in ("app.py", "design.json", "schema.json", "kit.json"):
        if (inst / f).exists():
            shutil.copyfile(inst / f, d / f)
    _jwrite(d / "seed.json", {"tables": test.get("fixture") or {}})
    app = _app_name(inst)
    w = SandboxWorld(d, d / "work")
    out = []
    try:
        await w.reset(0)
        for c in test.get("calls", []):
            tool = c["tool"] if "." in c["tool"] else f"{app}.{c['tool']}"
            o = await w.call(tool, c.get("args") or {})
            if c.get("expect") == "error" and o.error:
                out.append(o.error)
    finally:
        await w.close()
    return out


async def surface_gate(inst: Path) -> GateResult:
    t0 = time.monotonic()
    design = _jread(inst / "design.json")
    app = design["app"]
    tools = {t["name"]: t for t in design["tools"]}
    reasons: list[str] = []
    det: dict[str, Any] = {}
    w = SandboxWorld(inst, inst / "work" / f"sf{uuid.uuid4().hex[:6]}")
    try:
        await w.reset(0)
        # pagination
        pages = []
        for n, t in tools.items():
            names = {p["name"] for p in t.get("params", [])}
            req = [p["name"] for p in t.get("params", []) if p.get("required")]
            if not t.get("read_only") or set(req) - PAGEISH:  # pilot 1: page and page_size may be required
                continue
            for s in PAGE_PARAMS:
                if s <= names:
                    size_k = "page_size" if "page_size" in s else ("per_page" if "per_page" in s else "limit")
                    a1 = {size_k: 2, **({"page": 1} if "page" in s else {"offset": 0} if "offset" in s else {})}
                    a2 = {size_k: 2, **({"page": 2} if "page" in s else {"offset": 2} if "offset" in s else {})}
                    o1, o2 = await w.call(f"{app}.{n}", a1), await w.call(f"{app}.{n}", a2)
                    i1, i2 = _items(o1.value), _items(o2.value)
                    works = o1.ok and o2.ok and i1 is not None and i2 is not None and 0 < len(i1) <= 2 and i1 != i2
                    pages.append({"tool": n, "works": works, "page1": len(i1 or []), "page2": len(i2 or [])})
                    break
        det["pagination"] = pages
        if not any(p["works"] for p in pages):
            reasons.append("no list tool pages (page 1 and page 2 of size 2 differ)")
        filters = [n for n, t in tools.items() if t.get("read_only") and sum(1 for p in t.get("params", []) if not p.get("required")) >= 2]
        det["filter_tools"] = filters
        if not filters:
            reasons.append("no read tool has two or more optional filters")
        # refusals on bad input (the blind test's arguments, its fixture keys replaced by instance keys of the same table)
        calls = _passing_calls(inst)
        sm = schema_map(_jread(inst / "schema.json"))
        seed_tables = _jread(inst / "seed.json")["tables"]
        first = {tn: rows[0][sm[tn]["key"]] for tn, rows in seed_tables.items() if tn in sm and rows}
        refusals = []
        for n, t in tools.items():
            if t.get("read_only"):
                continue
            ex = calls.get(n)
            if not ex:
                refusals.append({"tool": n, "probe": "none", "note": "no passing blind-test call to start from"})
                continue
            test, call = ex[0]
            kt = _key_tables(test, sm)
            base = {a: (first.get(kt[str(v)], v) if isinstance(v, str | int) and not isinstance(v, bool) and str(v) in kt else v)
                    for a, v in (call.get("args") or {}).items()}
            key_args = [p["name"] for p in t.get("params", []) if p.get("required") and p.get("type", "string") == "string" and p["name"] in base
                        and (p["name"].endswith("_id") or p["name"] in ("id", "key"))]
            for k in key_args[:1]:
                o = await w.call(f"{app}.{n}", {**base, k: "ZZ-NOT-A-KEY-0"})
                refusals.append({"tool": n, "probe": f"unknown {k}", "refused": _ok_refusal(o.error), "coded": _coded(o.error), "error": (o.error or "")[:120]})
            for p in t.get("params", []):
                if p.get("enum") and p["name"] in base:
                    o = await w.call(f"{app}.{n}", {**base, p["name"]: "NOT_A_VALUE"})
                    refusals.append({"tool": n, "probe": f"bad enum {p['name']}", "refused": _ok_refusal(o.error), "coded": _coded(o.error),
                                     "error": (o.error or "")[:120]})
                    break
        det["refusals"] = refusals
        probed = [r for r in refusals if "refused" in r]
        unknown = [r for r in probed if r["probe"].startswith("unknown")]
        if unknown and sum(r["refused"] for r in unknown) / len(unknown) < 0.8:
            reasons.append(f"write tools accept or crash on unknown ids: {[r['tool'] for r in unknown if not r['refused']]}")
        crashes = [r for r in probed if r.get("error") and not r["refused"]]
        if crashes:
            reasons.append(f"write tools crash instead of refusing: {[(r['tool'], r['probe']) for r in crashes][:4]}")
        det["coded_share"] = round(sum(r["coded"] for r in probed) / max(1, len(probed)), 3)
        # impersonation: a write tool must not accept the acting user (or another "who" argument) the design does not have
        acting = "acting_user_id"
        if (inst / "domain.json").exists():
            acting = DomainSpec.model_validate_json((inst / "domain.json").read_text()).permissions.acting_param
        user_key = next(iter(first.values()), "U-1")
        imp = []
        for n, t in tools.items():
            ex = calls.get(n)
            if t.get("read_only") or not ex:
                continue
            test, call = ex[0]
            kt = _key_tables(test, sm)
            base = {a: (first.get(kt[str(v)], v) if isinstance(v, str | int) and not isinstance(v, bool) and str(v) in kt else v)
                    for a, v in (call.get("args") or {}).items()}
            o = await w.call(f"{app}.{n}", {**base, acting: user_key})
            if not (o.error or "").startswith("bad arguments"):
                imp.append(n)
        det["impersonation_accepted"] = imp
        if imp:
            reasons.append(f"write tools accept the acting user as an argument (impersonation): {imp[:6]}")
    finally:
        await w.close()
    # permission: a passing blind test expects a refusal from a tool that declares PERMISSION_DENIED
    perm_tools = {n for n, t in tools.items() if "PERMISSION_DENIED" in " ".join(t.get("errors") or []).upper()}
    tests = _jread(inst / "tests.json")["tests"]
    rep = _jread(inst / "test_report.json") if (inst / "test_report.json").exists() else {"results": []}
    passed = {r["id"] for r in rep.get("results", []) if r.get("passed")}
    cands = [t for t in tests if t["id"] in passed and any(c.get("expect") == "error" and c["tool"].split(".")[-1] in perm_tools for c in t.get("calls", []))]
    perm_tests = []
    # tests that name the code first (pilot 1: the first 8 candidates were other refusals and the permission tests came later)
    cands.sort(key=lambda t: not any("PERMISSION_DENIED" in str(c.get("error") or "") for c in t.get("calls", [])))
    for t in cands[:8]:  # replayed on the test's own fixture: the refusal must say PERMISSION_DENIED
        errs = await _replay_errors(inst, t)
        if any("PERMISSION_DENIED" in e for e in errs):
            perm_tests.append(t["id"])
    det["permission_tests"] = perm_tests
    if "PERMISSION_DENIED" not in (inst / "app.py").read_text():
        reasons.append("the App never refuses with PERMISSION_DENIED")
    elif not perm_tests:
        reasons.append("no passing blind test exercises a permission refusal")
    return GateResult(gate="surface", passed=not reasons, reasons=reasons, details=det, elapsed_s=round(time.monotonic() - t0, 2))


# ---------------------------------------------------------------------------------------------------------- obscurity

def _strings(v: Any, out: set[str]) -> None:
    if isinstance(v, dict):
        for x in v.values():
            _strings(x, out)
    elif isinstance(v, list):
        for x in v:
            _strings(x, out)
    elif v is not None and not isinstance(v, bool):
        out.add(str(v))


async def _page_all(w: SandboxWorld, tool: str, params: set[str], max_pages: int = 30) -> list[Any]:
    outs = []
    if "page" in params:
        size = "page_size" if "page_size" in params else ("per_page" if "per_page" in params else None)
        for pg in range(1, max_pages + 1):
            o = await w.call(tool, {"page": pg, **({size: 50} if size else {})})
            if not o.ok:
                break
            outs.append(o.value)
            if not (isinstance(o.value, dict) and o.value.get("next_page")):
                break
    elif "offset" in params and "limit" in params:
        for i in range(max_pages):
            o = await w.call(tool, {"offset": i * 50, "limit": 50})
            if not o.ok or not _items(o.value):
                break
            outs.append(o.value)
    else:
        args = {"limit": 1000} if "limit" in params else {}
        o = await w.call(tool, args)
        if o.ok:
            outs.append(o.value)
    return outs


async def obscurity_gate(inst: Path) -> GateResult:
    t0 = time.monotonic()
    design = _jread(inst / "design.json")
    app = design["app"]
    seed = _jread(inst / "seed.json")["tables"]
    info = _jread(inst / "instance.json")
    sm = schema_map(_jread(inst / "schema.json"))
    reasons: list[str] = []
    det: dict[str, Any] = {}
    w = SandboxWorld(inst, inst / "work" / f"ob{uuid.uuid4().hex[:6]}")
    try:
        await w.reset(0)
        bodies = []
        for page in await _page_all(w, f"{app}.list_documents", {"page", "page_size"}):
            for d in _items(page) or []:
                o = await w.call(f"{app}.read_document", {"doc_id": d.get("doc_id")})
                if o.ok:
                    bodies.append(str(o.value.get("body", "")))
        text = "\n".join(bodies)
        det["documents_read"] = len(bodies)
        spec = DomainSpec.model_validate_json((inst / "domain.json").read_text())
        unreachable = [p.name for p in spec.policy_params if p.type != "ref" and fmt_value(info["params"].get(p.name)) not in text]
        det["params_unreachable"] = unreachable
        if unreachable:
            reasons.append(f"policy parameters no readable document states: {unreachable}")
        lost = []
        nfacts = 0
        for c in seed.get("colleagues", []):
            for f in c.get("facts", []):
                nfacts += 1
                o = await w.call(f"{app}.ask_colleague", {"colleague_id": c["colleague_id"], "question": " ".join(f.get("keywords") or [])})
                if not (o.ok and f["text"] in str(o.value.get("reply", ""))):
                    lost.append(f"{c['colleague_id']}:{f['topic']}")
        det["facts"], det["facts_unreachable"] = nfacts, lost
        if lost:
            reasons.append(f"colleague facts their keywords do not reach: {lost[:4]}")
        seen: set[str] = set()
        rtools = {t["name"]: t for t in design["tools"] if t.get("read_only")}
        for n, t in rtools.items():
            if {p["name"] for p in t.get("params", []) if p.get("required")} - PAGEISH:
                continue
            for out in await _page_all(w, f"{app}.{n}", {p["name"] for p in t.get("params", [])}):
                _strings(out, seen)
        # then follow links, as a person would: a get tool keyed by a table's id, called on the ids listed so far (two hops)
        key_of = {tn: v["key"] for tn, v in sm.items()}
        getters = []
        for n, t in rtools.items():
            req = [p["name"] for p in t.get("params", []) if p.get("required")]
            if len(req) == 1:
                tn = next((x for x, k in key_of.items() if req[0] in (k, f"{x}_id", f"{x.rstrip('s')}_id", f"{x.lower()}_id")), None)
                if tn is not None:
                    getters.append((n, req[0], tn))
        asked: set[tuple[str, str]] = set()
        for _hop in range(2):
            for n, arg, tn in getters:
                ids = [str(r.get(key_of[tn])) for r in seed.get(tn, []) if str(r.get(key_of[tn])) in seen][:200]
                for k in ids:
                    if (n, k) in asked:
                        continue
                    asked.add((n, k))
                    o = await w.call(f"{app}.{n}", {arg: k})
                    if o.ok:
                        _strings(o.value, seen)
        det["link_calls"] = len(asked)
        cover = {}
        for tn in targets(design, _jread(inst / "schema.json")):
            if tn not in sm:
                continue
            kf = sm[tn]["key"]
            keys = {str(r.get(kf)) for r in seed.get(tn, [])}
            cover[tn] = round(len(keys & seen) / max(1, len(keys)), 3)
        det["discoverable"] = cover
        low = {k: v for k, v in cover.items() if v < 0.8}
        if low:
            reasons.append(f"target records the read tools do not list (share listed): {low}")
    finally:
        await w.close()
    return GateResult(gate="obscurity", passed=not reasons, reasons=reasons, details=det, elapsed_s=round(time.monotonic() - t0, 2))


# ---------------------------------------------------------------------------------------------------------- the listing rule, executed

SYNTH_ROWS = 6


def synthetic_seed(design: Mapping[str, Any], schema: Mapping[str, Any], spec: DomainSpec | None = None, n: int = SYNTH_ROWS) -> dict[str, Any]:
    """A start state with ``n`` rows per data table, made from the schema alone (no data generator exists at the code
    stage): keys per table; a field named for another table's key holds that table's key of the same row; a field named
    like an enum parameter of the design cycles its values; dates are dates; every other string is unique to its table
    (``<table>:<field>:<i>``), so a listed item can be told apart from a row that only links to it."""
    from crucible.synth.families import words

    tables = [t for t in schema.get("tables", []) if t["name"] not in KIT_TABLE_NAMES]
    keys = {t["name"]: t["key"] for t in tables}
    by_keyname: dict[str, str] = {}
    for t in tables:
        by_keyname.setdefault(t["key"], t["name"])
        ws = words(t["name"])
        if ws:
            by_keyname.setdefault(f"{'_'.join(ws)}_id", t["name"])
    enums: dict[str, list[Any]] = {}
    for tool in design.get("tools", []):
        for p in tool.get("params", []):
            if p.get("enum"):
                enums.setdefault(p["name"], list(p["enum"]))
    roles = [r.role for r in (spec.permissions.roles if spec else [])]
    user_table = spec.permissions.user_table if spec else None

    def key(tn: str, i: int) -> Any:
        ty = str(next(t for t in tables if t["name"] == tn)["fields"].get(keys[tn], "string"))
        return 100 + i if ty.startswith("integer") else f"{''.join(w[0] for w in words(tn)).upper() or 'X'}-{100 + i}"

    out: dict[str, list[dict[str, Any]]] = {}
    for t in tables:
        rows = []
        for i in range(n):
            r: dict[str, Any] = {}
            for f, ty in t["fields"].items():
                base = str(ty).rstrip("?")
                fl = f.lower()
                if f == t["key"]:
                    v: Any = key(t["name"], i)
                elif f in by_keyname and by_keyname[f] != t["name"]:
                    v = key(by_keyname[f], i)
                elif t["name"] == user_table and fl == "role" and roles:
                    v = roles[i % len(roles)]
                elif f in enums:
                    v = enums[f][i % len(enums[f])]
                elif base in ("integer",):
                    v = i + 1
                elif base == "number":
                    v = 100.0 * (i + 1)
                elif base == "boolean":
                    v = False
                elif base == "array":
                    v = []
                elif base == "object":
                    v = {}
                elif "date" in fl or fl.endswith(("_at", "_on")):
                    v = f"2025-12-{10 + i:02d}" + ("T09:00:00Z" if fl.endswith("_at") else "")
                else:
                    v = f"{t['name']}:{f}:{i}"
                r[f] = v
            rows.append(r)
        out[t["name"]] = rows
    if spec is not None:
        params = []
        for p in spec.policy_params:
            v = p.default
            if p.name == spec.permissions.acting_param and user_table in out and out[user_table]:
                v = out[user_table][0][keys[user_table]]
            params.append({"name": p.name, "value": v, "unit": p.unit, "description": p.meaning})
        out["policy_params"] = params
    kit = {t["name"] for t in schema.get("tables", [])} & set(KIT_TABLE_NAMES)
    out = {k: v for k, v in out.items() if k not in KIT_TABLE_NAMES or k in kit}
    for k in kit:
        out.setdefault(k, [])
    return {"tables": out, "notes": "synthetic rows for the listing check"}


_SYNTH = re.compile(r"^[A-Za-z_]\w*:\w+:\d+$")  # synthetic_seed's table-unique strings (a timestamp has colons too: worlds-2)


def _item_rows(items: Sequence[Any], rows: Sequence[Mapping[str, Any]], key: str, table: str) -> set[str]:
    """The rows (by key) a listing returned as items: an item holding the row's key and either one of the row's
    table-unique strings or no other table's (a loss notice item that only links a claim id is not a claim)."""
    got: set[str] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        vals: set[str] = set()
        _strings(it, vals)
        foreign = any(_SYNTH.match(v) and not v.startswith(f"{table}:") for v in vals)
        for r in rows:
            uniq = {v for f, v in r.items() if f != key and isinstance(v, str) and v.startswith(f"{table}:")}
            if str(r[key]) in vals and (uniq & vals or not foreign):
                got.add(str(r[key]))
    return got


async def listing_problems(app_dir: Path, work: Path, design: Mapping[str, Any], schema: Mapping[str, Any], spec: DomainSpec | None = None,
                           *, share: float = 0.0) -> tuple[list[str], dict[str, Any]]:
    """The listing rule, executed (``families.unlisted_targets`` is its design-stage form): the App is loaded on a
    synthetic start state and every paged read tool that needs no other argument is paged through, unfiltered and with
    each value of each of its enum filters; every collection tasks may act on must have more than ``share`` of its rows
    listed as items by some tool (by default: the tool exists and lists that collection; how many of the real rows an
    agent can reach is the obscurity gate's measure, since a list may rightly be scoped to the acting user's
    department). Problems name the collection, the tools tried and any error. worlds-1: insurance claims were reachable
    only through a loss notice's summary."""
    import shutil

    from crucible.synth.families import design_targets, paged

    # the collections the design-stage rule demanded a lister for (``design_targets`` without the schema): the coder
    # cannot add a tool, so a target only the schema's keys reveal is the obscurity gate's (worlds-2: returns' shipments
    # are keyed by a tracking number the receipt tool takes, and three coder attempts could not satisfy a demand the
    # design never made)
    tg = [c for c in design_targets(design) if c in {t["name"] for t in schema.get("tables", [])}]
    if not tg:
        return [], {}
    d = Path(work) / f"lst{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    for f in ("app.py", "design.json", "schema.json", "kit.json"):
        if (Path(app_dir) / f).exists():
            shutil.copyfile(Path(app_dir) / f, d / f)
    seed = synthetic_seed(design, schema, spec)
    _jwrite(d / "seed.json", seed)
    keys = {t["name"]: t["key"] for t in schema.get("tables", [])}
    app = design["app"]
    listers = [t for t in design.get("tools", []) if paged(t)]
    seen: dict[str, set[str]] = {c: set() for c in tg}
    errors: dict[str, str] = {}
    w = SandboxWorld(d, d / "work")
    try:
        await w.reset(0)
        for t in listers:
            pnames = {p["name"] for p in t.get("params", [])}
            variants: list[dict[str, Any]] = [{}]
            for p in t.get("params", []):
                if p.get("enum") and not p.get("required") and p["name"] not in PAGEISH:
                    variants += [{p["name"]: v} for v in p["enum"]]
            for extra in variants[:24]:
                outs = []
                if "page" in pnames:
                    size = "page_size" if "page_size" in pnames else ("per_page" if "per_page" in pnames else None)
                    for pg in range(1, 6):
                        o = await w.call(f"{app}.{t['name']}", {"page": pg, **({size: 50} if size else {}), **extra})
                        if not o.ok:
                            if not extra:
                                errors[t["name"]] = (o.error or "")[:200]
                            break
                        outs.append(o.value)
                        if not (isinstance(o.value, dict) and o.value.get("next_page")):
                            break
                else:
                    o = await w.call(f"{app}.{t['name']}", {"offset": 0, "limit": 50, **extra} if "offset" in pnames else {"limit": 50, **extra})
                    if o.ok:
                        outs.append(o.value)
                    elif not extra:
                        errors[t["name"]] = (o.error or "")[:200]
                for v in outs:
                    items = _items(v) or []
                    for c in tg:
                        seen[c] |= _item_rows(items, seed["tables"].get(c, []), keys[c], c)
    finally:
        await w.close()
    cover = {c: round(len(seen[c]) / max(1, len(seed["tables"].get(c, []))), 3) for c in tg}
    low = sorted(c for c, v in cover.items() if v <= share)
    probs = []
    if low:
        probs.append(f"no paged list or search tool lists the records of {low} (share of synthetic rows listed: {[cover[c] for c in low]}; paged read "
                     f"tools tried: {[t['name'] for t in listers]}{'; errors: ' + json.dumps(errors)[:400] if errors else ''}). Every collection tasks "
                     "act on needs a read-only list_<records> or search_<records> tool with page and page_size that returns its records as items")
    return probs, {"targets": tg, "cover": cover, "listers": [t["name"] for t in listers], "errors": errors}


# ---------------------------------------------------------------------------------------------------------- density

def _key_tables(test: Mapping[str, Any], sm: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """value (as text) -> table, for every key in the test's fixture."""
    out = {}
    for tn, rows in (test.get("fixture") or {}).items():
        if tn in sm and tn not in KIT_TABLE_NAMES:
            for r in rows:
                if sm[tn]["key"] in r:
                    out[str(r[sm[tn]["key"]])] = tn
    return out


def _like_fixture(test: Mapping[str, Any], key: Any, tn: str, seed_tables: Mapping[str, list[dict[str, Any]]], sm: Mapping[str, Mapping[str, Any]],
                  keys: Sequence[Any], rng: random.Random, n: int = 12) -> list[Any]:
    """Instance keys of ``tn`` ordered by how many short scalar fields their row shares with the fixture row ``key``."""
    kf = sm[tn]["key"]
    fx = next((r for r in (test.get("fixture") or {}).get(tn, []) if str(r.get(kf)) == str(key)), None)
    if fx is None:
        return []
    probe = {f: v for f, v in fx.items() if f != kf and isinstance(v, str | bool | int) and (not isinstance(v, str) or len(v) <= 24)}
    allowed = {str(k) for k in keys}
    rows = [r for r in seed_tables.get(tn, []) if str(r.get(kf)) in allowed]
    rng.shuffle(rows)
    rows.sort(key=lambda r: -sum(1 for f, v in probe.items() if r.get(f) == v))
    return [r[kf] for r in rows[:n]]


def _follow_nested(args: Mapping[str, Any], test: Mapping[str, Any], subst: Mapping[str, str], args0: Mapping[str, Any],
                   seed_tables: Mapping[str, list[dict[str, Any]]], sm: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The ids of a substituted row's nested items follow the row. A blind test's call names the fixture row's own line
    items (``lines: [{"line_id": "line-1", ...}]`` for ``po_id: "po-505"``); once ``po_id`` is an instance key, those ids
    become the instance row's ids at the same position. worlds-2b: purchase requisition's record_goods_receipt kept the
    fixture's "line-1" on every instance order and was refused on all of them (density 0 for goods receipts)."""
    mapping: dict[str, Any] = {}
    for a, tn in subst.items():
        kf = sm[tn]["key"]
        fx = next((r for r in (test.get("fixture") or {}).get(tn, []) if str(r.get(kf)) == str(args0.get(a))), None)
        row = next((r for r in seed_tables.get(tn, []) if str(r.get(kf)) == str(args.get(a))), None)
        if fx is None or row is None:
            continue
        for f, fv in fx.items():
            rv = row.get(f)
            if not (isinstance(fv, list) and isinstance(rv, list)):
                continue
            for fi, ri in zip(fv, rv, strict=False):
                if isinstance(fi, dict) and isinstance(ri, dict):
                    for k, v in fi.items():
                        if (k == "id" or k.endswith("_id")) and isinstance(v, str | int) and not isinstance(v, bool) and k in ri:
                            mapping.setdefault(str(v), ri[k])
    if not mapping:
        return dict(args)

    def sub(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: (mapping.get(str(v), v) if (k == "id" or k.endswith("_id")) and isinstance(v, str | int) and not isinstance(v, bool) else sub(v))
                    for k, v in x.items()}
        if isinstance(x, list):
            return [sub(v) for v in x]
        return x

    return {k: (sub(v) if isinstance(v, list | dict) else v) for k, v in args.items()}


async def reference_programs(inst: Path, world: SynthWorld, *, attempts: int = 24, seed: int = 0) -> list[dict[str, Any]]:
    """One reference oracle per family write tool (module docstring): its blind-test arguments with every fixture key
    replaced by an instance key of the same table, tried on seeded entities until the call succeeds and changes state."""
    design = _jread(inst / "design.json")
    app = design["app"]
    sm = schema_map(_jread(inst / "schema.json"))
    seed_tables = _jread(inst / "seed.json")["tables"]
    noise = {(n["table"], str(n["key"])) for n in _jread(inst / "noise.json") if n["kind"] in ("near_duplicate", "stale")}
    keys = {tn: [r[sm[tn]["key"]] for r in rows if (tn, str(r[sm[tn]["key"]])) not in noise] for tn, rows in seed_tables.items() if tn in sm}
    calls = _passing_calls(inst)
    rng = random.Random(f"ref/{inst.name}/{seed}")
    out = []
    for t in design["tools"]:
        if t.get("read_only") or t["name"] in KIT_NAMES:
            continue
        found = None
        tries = 0
        errs: dict[str, int] = {}
        for test, call in calls.get(t["name"], []):
            kt = _key_tables(test, sm)
            args0 = dict(call.get("args") or {})
            subst = {a: kt[str(v)] for a, v in args0.items() if isinstance(v, str | int) and not isinstance(v, bool) and str(v) in kt}
            if not subst or any(not keys.get(tn) for tn in subst.values()):
                continue
            ranked = {a: _like_fixture(test, args0[a], tn, seed_tables, sm, keys[tn], rng) for a, tn in subst.items()}
            for i in range(attempts):
                tries += 1
                # the instance rows most like the fixture's row first (same status and other short fields: pilot 1's random
                # draws found no entity in the state record_promise needs), then seeded random ones
                args = {**args0, **{a: (ranked[a][i] if i < len(ranked[a]) else rng.choice(keys[tn])) for a, tn in subst.items()}}
                args = _follow_nested(args, test, subst, args0, seed_tables, sm)
                prog = Program(steps=[Step(id="s1", tool=f"{app}.{t['name']}", args={k: Lit(v=v) for k, v in args.items()})])
                r = await run(world, prog, f"reference: {t['name']}", seed=0, taint=False)
                if r.error is None and all(c.obs.ok for c in r.calls) and r.start is not None and r.end is not None and r.start.fingerprint() != r.end.fingerprint():
                    found = {"tool": t["name"], "program": prog, "run": r, "subst": subst, "tries": tries}
                    break
                why = r.error or next((c.obs.error for c in r.calls if not c.obs.ok), None) or "no state change"
                errs[_why(why)] = errs.get(_why(why), 0) + 1
            if found:
                break
        out.append(found or {"tool": t["name"], "program": None, "tries": tries, "refusals": errs,
                             "tests": len(calls.get(t["name"], []))})
    return out


def _why(err: str) -> str:
    """A refusal's reason with ids and numbers masked, so equal reasons on different rows count together."""
    e = re.sub(r"^ToolError:\s*", "", str(err)).strip()
    e = re.sub(r"\b[A-Za-z]*[-_]?\d[\w-]*\b", "<id>", e)
    return e[:110]


async def sibling_reach(world: SynthWorld, inst: Path, ref: Mapping[str, Any], *, limit: int = 160) -> dict[str, Any]:
    """How many sibling entities a reference oracle's call succeeds on (``SwapEntity`` and ``ExtraWrite``'s raw
    material): for its key argument, every non-noise instance key of that table in turn, executed. Returns
    ``{"table", "arg", "rows", "ok", "refusals": {reason: count}}``: what the data generator is told when a collection
    is not dense (worlds-1: returns' resolve_return succeeded on a dozen of 80 return requests)."""
    sm = schema_map(_jread(inst / "schema.json"))
    seed_tables = _jread(inst / "seed.json")["tables"]
    noise = {(n["table"], str(n["key"])) for n in _jread(inst / "noise.json") if n["kind"] in ("near_duplicate", "stale")}
    subst = dict(ref.get("subst") or {})
    if not subst or ref.get("program") is None:
        return {}
    arg, tn = next(iter(subst.items()))
    kf = sm[tn]["key"]
    keys = [r[kf] for r in seed_tables.get(tn, []) if (tn, str(r[kf])) not in noise][:limit]
    prog: Program = ref["program"]
    ok, why = 0, {}
    for k in keys:
        st = prog.steps[0].model_copy(deep=True)
        st.args[arg] = Lit(v=k)
        r = await run(world, prog.model_copy(update={"steps": [st]}), f"reach: {ref['tool']}", seed=0, taint=False)
        if r.error is None and all(c.obs.ok for c in r.calls) and r.start is not None and r.end is not None and r.start.fingerprint() != r.end.fingerprint():
            ok += 1
        else:
            w = _why(r.error or next((c.obs.error for c in r.calls if not c.obs.ok), None) or "no state change")
            why[w] = why.get(w, 0) + 1
    return {"table": tn, "arg": arg, "rows": len(keys), "ok": ok, "refusals": dict(sorted(why.items(), key=lambda x: -x[1])[:4])}


async def choose_actor(inst: Path) -> dict[str, Any]:
    """The actor config by execution: for every role of the spec, a user holding it is made the acting user of a copy of
    the instance, and the reference oracles are built (no MUT); the user whose references cover the most target
    collections, then the most write tools, wins. Returns ``{"user", "role", "scores"}`` (the caller re-generates the
    instance with that user in its config, so the choice is reproducible). Pilot 1: ranking roles by their declared
    permissions picked a collections role that could not place credit holds, so a target collection had no operable tool."""
    import shutil

    spec = DomainSpec.model_validate_json((inst / "domain.json").read_text())
    acting = spec.permissions.acting_param
    p = next((x for x in spec.policy_params if x.name == acting), None)
    seed = _jread(inst / "seed.json")
    sm = schema_map(_jread(inst / "schema.json"))
    design = _jread(inst / "design.json")
    if p is None or p.table not in seed["tables"]:
        return {"user": None, "scores": {}}
    kf = sm[p.table]["key"]
    noise = {str(n["key"]) for n in _jread(inst / "noise.json") if n["table"] == p.table and n["kind"] in ("near_duplicate", "stale")}
    users = [r for r in seed["tables"][p.table] if str(r[kf]) not in noise]

    def norm(x: Any) -> str:
        return str(x).strip().lower().replace("_", " ")

    tg = set(targets(design, _jread(inst / "schema.json")))
    writes = {t["name"]: set(t.get("writes") or []) for t in design["tools"] if not t.get("read_only")}
    scores: dict[str, Any] = {}
    best: tuple[tuple[int, int], Any, str] | None = None
    for role in spec.permissions.roles:
        u = next((r for r in users if any(norm(v) == norm(role.role) for v in r.values() if isinstance(v, str))), None)
        if u is None:
            continue
        d = inst / "work" / f"actor-{uuid.uuid4().hex[:6]}"
        d.mkdir(parents=True, exist_ok=True)
        for f in ("app.py", "design.json", "schema.json", "kit.json", "tests.json", "test_report.json", "noise.json", "domain.json"):
            if (inst / f).exists():
                shutil.copyfile(inst / f, d / f)
        s2 = copy.deepcopy(seed)
        for row in s2["tables"].get("policy_params", []):
            if row.get("name") == acting:
                row["value"] = u[kf]
        _jwrite(d / "seed.json", s2)
        w = SandboxWorld(d, d / "work")
        try:
            refs = await reference_programs(d, w, attempts=12)
        finally:
            await w.close()
        ok = [r["tool"] for r in refs if r.get("program") is not None]
        covered = {c for t in ok for c in writes.get(t, set())} & tg
        score = (len(covered), len(ok))
        scores[role.role] = {"user": u[kf], "covered": sorted(covered), "tools": ok}
        if best is None or score > best[0]:
            best = (score, u[kf], role.role)
    return {"user": best[1] if best else None, "role": best[2] if best else None, "scores": scores}


async def density_gate(inst: Path, *, k_min: int | None = None, n_min: int = N_MIN, max_mutants: int = 240, share: float = 0.5, holdout: float | None = None,
                       diagnose: bool = False) -> tuple[GateResult, dict[str, bool]]:
    """The density gate (module docstring) and the replay determinism of each reference program.

    **Shown mutants** (worlds-2; item 22's V5 split, ``falsify.mut.HOLDOUT`` = 0.2 by default): a certificate's n counts
    only the mutants a repair loop is shown, so the bar n >= ``n_min`` is on ``MUT(split="shown")``'s distinct mutants
    and the held-out share is measured beside it (``MUTHeldOut``); about 75 distinct in all reach 59 shown. ``holdout=0``
    is the whole neighbourhood, as before. ``diagnose``: for every reference oracle, the sibling rows its call succeeds
    on (``sibling_reach``), for the data generator's feedback."""
    from crucible.synth.falsify import MUT, FalsifyContext, MUTHeldOut
    from crucible.synth.falsify.base import falsify
    from crucible.synth.falsify.mut import HOLDOUT
    from crucible.synth.infer import skeleton
    from crucible.synth.task import Task, WorldRef

    t0 = time.monotonic()
    k_min = int(policy.param("R-density", "k_min")) if k_min is None else int(k_min)
    h = HOLDOUT if holdout is None else float(holdout)
    design = _jread(inst / "design.json")
    sm = schema_map(_jread(inst / "schema.json"))
    seed_tables = _jread(inst / "seed.json")["tables"]
    noise = {(n["table"], str(n["key"])) for n in _jread(inst / "noise.json") if n["kind"] in ("near_duplicate", "stale")}
    reasons: list[str] = []
    counts = {tn: sum(1 for r in seed_tables.get(tn, []) if (tn, str(r.get(sm[tn]["key"]))) not in noise) for tn in targets(design, _jread(inst / "schema.json")) if tn in sm}
    thin = {k: v for k, v in counts.items() if v < k_min}
    if thin:
        reasons.append(f"target collections with fewer than {k_min} entities: {thin}")
    work = inst / "work" / f"dn{uuid.uuid4().hex[:6]}"
    ref = WorldRef(factory="crucible.synth.worldgen:open_world", params={"dir": str(inst.resolve()), "work": str(work.resolve())})
    w = await ref.build()
    replays: dict[str, bool] = {}
    per_tool: list[dict[str, Any]] = []
    try:
        refs = await reference_programs(inst, w)
        for rf in refs:
            if rf.get("program") is None:
                per_tool.append({"tool": rf["tool"], "reference": False, "tries": rf.get("tries", 0), "refusals": rf.get("refusals") or {},
                                 "tests": rf.get("tests", 0)})
                continue
            r: Run = rf["run"]
            again = await run(w, rf["program"], f"reference: {rf['tool']}", seed=0, taint=False)
            replays[rf["tool"]] = bool(again.end is not None and r.end is not None and again.end.fingerprint() == r.end.fingerprint())
            task = Task(id=f"ref/{inst.parent.parent.name}/{inst.name}/{rf['tool']}", domain=inst.parent.parent.name, world=ref, seed=0,
                        instruction=f"reference: {rf['tool']}", contract=skeleton(r), oracle=rf["program"])
            n_ho: int | None = None
            ctx = FalsifyContext(max_mutants=max_mutants)
            try:
                res = await falsify(task, MUT(split="shown", holdout=h) if h > 0 else MUT(), ctx)
                if h > 0:
                    n_ho = (await falsify(task, MUTHeldOut(holdout=h), ctx)).n
            finally:
                await ctx.close()
            ops: dict[str, int] = {}
            for f in res.findings:
                if f.outcome.valid:
                    ops[f.op] = ops.get(f.op, 0) + 1
            row = {"tool": rf["tool"], "reference": True, "n_distinct": res.n, "n_heldout": n_ho, "holdout": h, "probes": len(res.findings), "by_op": ops,
                   "confirmed": res.c, "args": {k: compact(v.v, 60) for k, v in rf["program"].steps[0].args.items()}}
            if diagnose:
                row["reach"] = await sibling_reach(w, inst, rf)
            per_tool.append(row)
    finally:
        await w.close()
    ns = [p["n_distinct"] for p in per_tool if p.get("reference")]
    reached = [n for n in ns if n >= n_min]
    if not ns:
        reasons.append("no reference oracle could be built from the blind tests on this instance")
    elif not reached:
        reasons.append(f"no reference oracle reaches n >= {n_min} distinct mutants (best {max(ns)})")
    # per target collection (the task spec's rule, made per collection): some reference oracle that writes it reaches n.
    # A tool with no dense reference is "thin": recorded, so a proposer can keep tasks off it (pilot 1 first required half
    # of all write tools, which a realistic App with a few rarely-applicable tools cannot meet)
    writes = {t["name"]: set(t.get("writes") or []) for t in design["tools"] if not t.get("read_only")}
    best_by_coll: dict[str, int] = {}
    for p in per_tool:
        for c in writes.get(p["tool"], set()):
            if c in counts:
                best_by_coll[c] = max(best_by_coll.get(c, 0), int(p.get("n_distinct") or 0))
    short = {c: best_by_coll.get(c, 0) for c in counts if best_by_coll.get(c, 0) < n_min}
    if ns and short:
        reasons.append(f"target collections no reference oracle covers densely (best distinct n per collection): {short}")
    thin = sorted(p["tool"] for p in per_tool if int(p.get("n_distinct") or 0) < n_min)
    ho_by_coll: dict[str, int] = {}
    for p in per_tool:
        for c in writes.get(p["tool"], set()):
            if c in counts and best_by_coll.get(c) == int(p.get("n_distinct") or 0) and p.get("n_heldout") is not None:
                ho_by_coll[c] = max(ho_by_coll.get(c, 0), int(p["n_heldout"]))
    det = {"targets": counts, "k_min": k_min, "n_min": n_min, "holdout": h, "split": "shown" if h > 0 else "all", "heldout_by_collection": ho_by_coll,
           "write_tools": len(per_tool), "references": len(ns), "reached": len(reached), "best": max(ns, default=0),
           "median": statistics.median(ns) if ns else 0, "best_by_collection": best_by_coll, "thin_tools": thin,
           "dense_share": round(len(reached) / max(1, len(per_tool)), 3), "per_tool": per_tool}
    return GateResult(gate="density", passed=not reasons, reasons=reasons, details=det, elapsed_s=round(time.monotonic() - t0, 2)), replays


# ---------------------------------------------------------------------------------------------------------- realism panel

RUBRIC = """REALISM RUBRIC (score each 1 to 5; 3 = acceptable for a real company, 5 = indistinguishable from one)
R1 records: entity names, codes, amounts, dates and statuses look like a real mid-sized company's data (no placeholders, no
   toy values, realistic distributions, references consistent).
R2 rules: the business rules and the policy / SOP documents are coherent, specific and like a real department's.
R3 tools: the tool surface is like a real internal system's (search and filters, pagination, clear error codes, permissions).
R4 noise: near-duplicates, stale records and conflicting sources occur as they do in real systems, and the documents say how
   to resolve them.
R5 transfer: the work is common enterprise practice a new hire would recognise; skills learned here transfer (not obscure,
   not fictional).
realistic = every score >= 3 and the whole would pass as a real company's system. obscure = the domain or its rules are
niche, invented or too unusual to teach transferable skills."""


class RealismOut(_M):
    scores: dict[str, int]
    realistic: bool
    obscure: bool = False
    confidence: float = 0.5
    problems: list[str] = Field(default_factory=list)

    @field_validator("scores")
    @classmethod
    def _s(cls, v: dict[str, int]) -> dict[str, int]:
        want = {"R1", "R2", "R3", "R4", "R5"}
        got = {k.upper().split()[0].strip(":"): int(x) for k, x in v.items()}
        if not want <= set(got):
            raise ValueError(f"score every criterion {sorted(want)}")
        if any(not 1 <= got[k] <= 5 for k in want):
            raise ValueError("scores are 1 to 5")
        return {k: got[k] for k in sorted(want)}

    @field_validator("confidence")
    @classmethod
    def _c(cls, v: float) -> float:
        return min(1.0, max(0.0, float(v)))

    def vote(self) -> bool:
        return self.realistic and not self.obscure and min(self.scores.values()) >= 3


REALISM_RATER = Role(
    name="realism_rater", stage="W5", output=RealismOut, temperature=0.0, max_tokens=3000,
    sees=("summary", "rubric"),
    never=("tests", "code", "contract", "builder"),
    differs_from=("domain_spec_writer", "family_designer", "app_coder", "data_spec_writer", "doc_writer", "actor_writer"),
    independent=True, min_families=2,
    system=("You review an internal enterprise application that will be used to train AI agents for real enterprise work. Judge it "
            "against the rubric strictly: you are the check that stops toy, fake-looking or obscure environments. Read the records, "
            "the documents and the tool list, then score.\n"
            'Reply {"scores": {"R1": n, "R2": n, "R3": n, "R4": n, "R5": n}, "realistic": true|false, "obscure": true|false, '
            '"confidence": <0 to 1>, "problems": ["<concrete problems>"]}.'))


class RealismVote(_M):
    member: str
    family: str
    out: RealismOut | None = None
    error: str | None = None

    @property
    def ok(self) -> bool | None:
        return None if self.out is None else self.out.vote()


class RealismVerdict(_M):
    decision: str  # realistic | unrealistic | abstain
    votes: list[RealismVote]
    lam: float


@dataclass
class RealismPanel:
    members: list[tuple[str, RoleClient]]
    lam: float = 0.7
    queue: Path | None = None
    log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len({getattr(c, "family", n) for n, c in self.members}) < REALISM_RATER.min_families:
            raise ValueError(f"the realism panel needs >= {REALISM_RATER.min_families} families")

    async def judge(self, summary: Mapping[str, Any], *, tag: str = "realism") -> RealismVerdict:
        msgs = REALISM_RATER.render(summary=dict(summary), rubric=RUBRIC)

        async def one(name: str, c: RoleClient) -> RealismVote:
            fam = getattr(c, "family", "?")
            last = ""
            for _ in range(2):
                try:
                    doc = await c.complete_json(msgs, schema=RealismOut, max_tokens=REALISM_RATER.max_tokens, temperature=REALISM_RATER.temperature, tag=tag)
                    return RealismVote(member=name, family=fam, out=RealismOut.model_validate(doc))
                except Exception as e:
                    if type(e).__name__ in ("BudgetExceeded", "TinkerCapReached"):
                        raise
                    last = f"{type(e).__name__}: {str(e)[:200]}"
            return RealismVote(member=name, family=fam, error=last)

        votes = list(await asyncio.gather(*(one(n, c) for n, c in self.members)))
        vals = {v.ok for v in votes}
        if None in vals or len(vals) != 1 or min((v.out.confidence for v in votes if v.out), default=0.0) < self.lam:
            return RealismVerdict(decision="abstain", votes=votes, lam=self.lam)
        return RealismVerdict(decision="realistic" if vals.pop() else "unrealistic", votes=votes, lam=self.lam)

    async def gate(self, summary: Mapping[str, Any], *, ident: str) -> GateResult:
        t0 = time.monotonic()
        v = await self.judge(summary, tag=f"realism:{ident}")
        row = {"id": ident, "decision": v.decision, "lam": v.lam, "votes": [x.model_dump(mode="json") for x in v.votes]}
        self.log.append(row)
        if v.decision == "abstain" and self.queue is not None:
            self.queue.parent.mkdir(parents=True, exist_ok=True)
            with self.queue.open("a") as fh:
                fh.write(json.dumps({**row, "summary": summary}, default=str) + "\n")
        scores = {x.member: x.out.scores for x in v.votes if x.out}
        probs = [p for x in v.votes if x.out for p in x.out.problems][:8]
        reasons = [] if v.decision == "realistic" else [f"panel: {v.decision}" + (" (sent to the human queue)" if v.decision == "abstain" else "")] + probs[:3]
        return GateResult(gate="panel", passed=v.decision == "realistic", reasons=reasons, elapsed_s=round(time.monotonic() - t0, 2),
                          details={"decision": v.decision, "lam": v.lam, "scores": scores, "problems": probs,
                                   "votes": {x.member: (x.ok, x.out.confidence if x.out else None, x.error) for x in v.votes}})


class Anchor(_M):
    kind: str
    truth: bool
    summary: dict[str, Any]


async def calibrate_panel(panel: RealismPanel, anchors: Sequence[Anchor], *, target: float | None = None, grid: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
                          floor: float | None = None, min_decided: int | None = None, require_both: bool | None = None,
                          refuse_uncalibrated: bool | None = None) -> dict[str, Any]:
    """λ from anchors whose truth is known (the rule of ``falsify.agents.calibrate``): the smallest grid value whose
    decided anchors are ≥ ``target`` accurate, with at least ``min_decided`` decided and (``require_both``) both truths
    among them; never below ``floor``. None qualifies: λ above 1 with ``refuse_uncalibrated`` (the panel decides
    nothing), else the largest. Every keyword left None is R-panel's (``policy.panel_calibration()``, DESIGN item 24:
    the strict rule); passing them is a measurement under another rule, recorded in the result."""
    pc = policy.panel_calibration()
    target = pc["target"] if target is None else target
    floor = pc["floor"] if floor is None else floor
    min_decided = pc["min_decided"] if min_decided is None else min_decided
    require_both = pc["require_both"] if require_both is None else require_both
    refuse_uncalibrated = pc["refuse_uncalibrated"] if refuse_uncalibrated is None else refuse_uncalibrated
    rule = {"target": target, "floor": floor, "min_decided": min_decided, "require_both": require_both, "refuse_uncalibrated": refuse_uncalibrated}
    saved = panel.lam
    panel.lam = 0.0
    try:
        verdicts = [await panel.judge(a.summary, tag="realism:calibrate") for a in anchors]
    finally:
        panel.lam = saved
    rows, chosen = [], None
    for lam in grid:
        dec = cor = 0
        truths: set[bool] = set()
        for a, v in zip(anchors, verdicts, strict=True):
            vals = {x.ok for x in v.votes}
            if None in vals or len(vals) != 1 or min(x.out.confidence for x in v.votes if x.out) < lam:
                continue
            dec += 1
            truths.add(a.truth)
            cor += int(vals.pop() == a.truth)
        acc = cor / dec if dec else 0.0
        rows.append({"lam": lam, "decided": dec, "correct": cor, "accuracy": round(acc, 4), "coverage": round(dec / max(1, len(anchors)), 4),
                     "both_truths": len(truths) == 2, "error_upper95": round(clopper_pearson_upper(dec - cor, dec, 0.05), 4) if dec else 1.0})
        if chosen is None and dec >= max(1, min_decided) and acc >= target and (not require_both or len(truths) == 2):
            chosen = lam
    panel.lam = max(floor, chosen if chosen is not None else (1.01 if refuse_uncalibrated else max(grid)))
    votes = [{"kind": a.kind, "truth": a.truth, "votes": [x.model_dump(mode="json") for x in v.votes]} for a, v in zip(anchors, verdicts, strict=True)]
    # per anchor kind at the chosen lambda (worlds-2: the panel's accuracy on instances the executed gates rejected)
    by_kind: dict[str, dict[str, int]] = {}
    for a, v in zip(anchors, verdicts, strict=True):
        k = by_kind.setdefault(a.kind, {"n": 0, "decided": 0, "correct": 0})
        k["n"] += 1
        vals = {x.ok for x in v.votes}
        if None in vals or len(vals) != 1 or min(x.out.confidence for x in v.votes if x.out) < panel.lam:
            continue
        k["decided"] += 1
        k["correct"] += int(vals.pop() == a.truth)
    return {"lam": panel.lam, "calibrated": chosen is not None, "target": target, "rule": rule, "policy_version": policy.tag(), "rows": rows,
            "by_kind": by_kind, "votes": votes}


# ---------------------------------------------------------------------------------------------------------- summaries and anchors

def _short(v: Any, n: int = 120) -> Any:
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + "..."
    if isinstance(v, list):
        return [_short(x, n) for x in v[:6]]
    if isinstance(v, dict):
        return {k: _short(x, n) for k, x in list(v.items())[:20]}
    return v


async def world_summary(world: SynthWorld, *, domain: str, skills: Sequence[str] = (), docs: Mapping[str, str] | None = None, rows: int = 5,
                        noise: Sequence[Mapping[str, Any]] = (), doc_chars: int = 1400, seed: int = 0) -> dict[str, Any]:
    """What the realism panel reads: the tools, a few rows of every collection with its size, the documents (kit
    documents are read from the state; other Worlds pass ``docs``), the colleagues, and examples of the planted noise."""
    await world.reset(seed)
    tools = await world.tools()
    c = await world.canon()
    out: dict[str, Any] = {"domain": domain, "skills": list(skills)[:10],
                           "tools": [{"name": t.name.split(".", 1)[-1], "read_only": t.read_only, "description": t.description[:220],
                                      "params": sorted((t.input_schema.get("properties") or {}).keys())} for t in tools]}
    colls: dict[str, Any] = {}
    kit_docs: dict[str, str] = {}
    colleagues: list[Any] = []
    for app, cs in (c.apps if c is not None else {}).items():
        for coll, ents in cs.items():
            if coll == "documents":
                for e in ents.values():
                    kit_docs[f"{e.get('title')} ({e.get('kind')})"] = str(e.get("body", ""))
                continue
            if coll == "colleagues":
                colleagues = [{"name": e.get("name"), "title": e.get("title"), "facts": [f.get("text") for f in (e.get("facts") or [])][:3]} for e in ents.values()]
                continue
            if coll == "policy_params":
                continue
            items = list(ents.values())
            colls[f"{app}.{coll}"] = {"rows": len(items), "sample": [_short(r) for r in items[:rows]]}
    out["collections"] = colls
    dd = {**kit_docs, **dict(docs or {})}
    out["documents"] = {k: (v[:doc_chars] + ("..." if len(v) > doc_chars else "")) for k, v in list(dd.items())[:6]}
    if colleagues:
        out["colleagues"] = colleagues
    if noise:
        out["noise_examples"] = [dict(n) for n in noise[:6]]
    return out


async def instance_summary(inst: Path) -> dict[str, Any]:
    spec = DomainSpec.model_validate_json((inst / "domain.json").read_text())
    noise = _jread(inst / "noise.json")
    ex: list[dict[str, Any]] = []
    for kind in ("near_duplicate", "stale", "conflict"):
        ex += [{k: v for k, v in n.items() if v is not None} for n in noise if n["kind"] == kind][:2]
    w = SandboxWorld(inst, inst / "work" / f"sm{uuid.uuid4().hex[:6]}")
    try:
        s = await world_summary(w, domain=f"{spec.keyword}: {spec.description}", skills=[f"{x.id}: {x.description}" for x in spec.skills], noise=ex)
    finally:
        await w.close()
    s["business_rules"] = spec.rules[:8]
    return s


_FAKE = ("test", "foo", "bar", "lorem ipsum", "sample", "xyz", "abc")


def degrade(summary: Mapping[str, Any], kind: str, seed: int = 0) -> dict[str, Any]:
    """A summary made unrealistic by construction (the panel's negative anchors): ``placeholder`` (placeholder text and
    round numbers), ``toy`` (two rows, three tools, no documents), ``incoherent`` (dates out of order and in the future,
    negative amounts, broken references, random-letter names)."""
    rng = random.Random(f"degrade/{kind}/{seed}")
    s = copy.deepcopy(dict(summary))

    def walk(v: Any, f: Any) -> Any:
        if isinstance(v, dict):
            return {k: walk(x, f) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x, f) for x in v]
        return f(v)

    if kind == "placeholder":
        def ph(v: Any) -> Any:
            if isinstance(v, str) and not re.match(r"^\d{4}-\d{2}-\d{2}", v):
                return f"{rng.choice(_FAKE)} {rng.randint(1, 9)}"
            if isinstance(v, int | float) and not isinstance(v, bool):
                return 100
            return v
        for c in s.get("collections", {}).values():
            c["sample"] = walk(c["sample"], ph)
        s["documents"] = {"Policy": "TODO: write policy.", "SOP": "lorem ipsum dolor sit amet"}
        for t in s.get("tools", []):
            t["description"] = "does stuff"
    elif kind == "toy":
        for c in s.get("collections", {}).values():
            c["sample"], c["rows"] = c["sample"][:2], 2
        s["tools"] = s.get("tools", [])[:3]
        s["documents"] = {}
        s.pop("noise_examples", None)
        s.pop("colleagues", None)
    elif kind == "incoherent":
        def inc(v: Any) -> Any:
            if isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}", v):
                return f"20{rng.randint(31, 79)}-{rng.randint(13, 19)}-{rng.randint(32, 45)}"
            if isinstance(v, str) and len(v) > 3 and " " in v:
                return " ".join("".join(rng.choice("qxzvkjw") for _ in range(rng.randint(3, 7))).title() for _ in range(2))
            if isinstance(v, int | float) and not isinstance(v, bool):
                return -abs(v) * rng.randint(7, 900)
            return v
        for c in s.get("collections", {}).values():
            c["sample"] = walk(c["sample"], inc)
        s["business_rules"] = ["Records may be changed by anyone at any time for any reason.", "Amounts can be negative or positive as preferred."]
    else:
        raise ValueError(f"unknown degradation {kind!r}")
    return s


def anchors_from(positives: Sequence[Mapping[str, Any]], *, kinds: Sequence[str] = ("placeholder", "toy", "incoherent")) -> list[Anchor]:
    """Positives (summaries of Worlds known to be realistic: a hand-built, reviewed pack) and their degradations."""
    out = [Anchor(kind="reviewed", truth=True, summary=dict(p)) for p in positives]
    for i, p in enumerate(positives):
        for k in kinds:
            out.append(Anchor(kind=k, truth=False, summary=degrade(p, k, seed=i)))
    return out


REALISM_GATES = ("integrity", "surface", "obscurity")


async def rejected_anchors(worlds: str | Path, *, per_family: int = 4) -> list[Anchor]:
    """Negative anchors with known truth from the executed gates (worlds-2): instances a *realism* gate rejected
    (integrity, surface, obscurity: data that breaks its own rules, a tool surface without refusals, records an employee
    cannot find) are not realistic, whatever the panel reads in their summary. A density, build or determinism failure is
    no realism verdict, so those instances are not anchors. worlds-1: the panel rated all 20 instances realistic, the 8
    the gates rejected included; with these anchors its accuracy on such Worlds is measured, not assumed."""
    out: list[Anchor] = []
    for fam in sorted(p for p in Path(worlds).iterdir() if (p / "instances").exists()):
        n = 0
        for g in sorted(fam.glob("instances/*/gates.json")):
            if n >= per_family:
                break
            try:
                rec = json.loads(g.read_text())
            except ValueError:
                continue
            failed = [k for k in REALISM_GATES if k in rec.get("gates", {}) and not rec["gates"][k].get("passed")]
            if not failed:
                continue
            out.append(Anchor(kind=f"gate:{failed[0]}", truth=False, summary=await instance_summary(g.parent)))
            n += 1
    return out


# ---------------------------------------------------------------------------------------------------------- one instance

async def gate_instance(fdir: Path, inst: Path, *, panel: RealismPanel | None = None, k_min: int | None = None, n_min: int = N_MIN,
                        max_mutants: int = 240, only: Sequence[str] | None = None, reseed_tries: int = 0, rebuild_tries: int = 0, can_reseed: bool = True,
                        restartable: bool = False) -> InstanceGates:
    """Every gate on one instance (``only``: a subset), recorded in ``<inst>/gates.json`` (earlier results kept)."""
    fdir, inst = Path(fdir), Path(inst)
    path = inst / "gates.json"
    g = InstanceGates.model_validate_json(path.read_text()) if path.exists() else InstanceGates(family=fdir.name, instance=inst.name, dir=str(inst))
    man = frozen_manifest(fdir)
    icfg = (_jread(inst / "instance.json").get("config") or {}) if (inst / "instance.json").exists() else {}
    k = int(k_min or icfg.get("k_min") or man.get("k_min") or policy.param("R-density", "k_min"))
    want = list(only or REQUIRED)

    def keep(r: GateResult) -> None:
        g.gates[r.gate] = r
        g.decide(REQUIRED)
        _jwrite(path, g.model_dump(mode="json"))

    async def guarded(name: str, coro: Any) -> GateResult:
        t0 = time.monotonic()
        try:
            return await coro
        except Exception as e:
            if type(e).__name__ in ("BudgetExceeded", "TinkerCapReached"):
                raise
            return GateResult(gate=name, passed=False, reasons=[f"the gate could not run: {type(e).__name__}: {str(e)[:400]}"],
                              elapsed_s=round(time.monotonic() - t0, 2))

    if "build" in want:
        keep(await guarded("build", build_gate(fdir, inst)))
    replays: dict[str, bool] = {}
    if "density" in want:
        try:
            r, replays = await density_gate(inst, k_min=k, n_min=n_min, max_mutants=max_mutants)
        except Exception as e:
            if type(e).__name__ in ("BudgetExceeded", "TinkerCapReached"):
                raise
            r = GateResult(gate="density", passed=False, reasons=[f"the gate could not run: {type(e).__name__}: {str(e)[:400]}"])
        keep(r)
    if "determinism" in want:
        keep(await guarded("determinism", determinism_gate(fdir, inst, replays=replays)))
    if "integrity" in want:
        try:
            keep(integrity_gate(fdir, inst))
        except Exception as e:  # noqa: BLE001
            keep(GateResult(gate="integrity", passed=False, reasons=[f"the gate could not run: {type(e).__name__}: {str(e)[:400]}"]))
    if "surface" in want:
        keep(await guarded("surface", surface_gate(inst)))
    if "obscurity" in want:
        keep(await guarded("obscurity", obscurity_gate(inst)))
    if "panel" in want and panel is not None:
        keep(await guarded("panel", _panel_gate(panel, inst, g.gates)))
    g.decide(REQUIRED)
    g.remedies = remedies_for(g, reseed_tries=reseed_tries, rebuild_tries=rebuild_tries, can_reseed=can_reseed, restartable=restartable)
    _jwrite(path, g.model_dump(mode="json"))
    return g


# stages a failing gate points at when a family is rebuilt (``families.build_family(restart_from=)``)
GATE_STAGE = {"build": "debug", "determinism": "datagen", "integrity": "datagen", "surface": "code", "obscurity": "design", "panel": "datagen"}


def prior_reseeds(inst: Path) -> int:
    """How many times R-density's reseed_denser already regenerated this instance (``gates.reseed<k>.json`` kept beside
    it). The gate runner starts its count here, so a re-gate never reseeds past ``max_reseed``."""
    return len(list(Path(inst).glob("gates.reseed*.json")))


def remedies_for(g: InstanceGates, *, reseed_tries: int = 0, rebuild_tries: int = 0, can_reseed: bool = True, restartable: bool = False) -> list[dict[str, Any]]:
    """R-density's remedies for an instance's failing gates (DESIGN item 24): a failing density gate raises
    ``density_gate_failing`` (facts: can_reseed, reseed_tries), any other ``world_gate_failing`` (gate, restartable,
    rebuild_tries); ``policy.resolve`` returns the first lawful remedy, recorded in ``gates.json`` with the rule and the
    policy version. The gate runner carries them out (``scripts/synth_worlds.py gate``: reseed_denser regenerates the
    instance denser and gates it again; rebuild_stage names the family stage to rebuild; not_used keeps the reasons)."""
    out = []
    for name, r in g.gates.items():
        if r.passed:
            continue
        if name == "density":
            c = policy.Condition(kind="density_gate_failing", subject=f"{g.family}/{g.instance}", raised_by="worldgates.gate_instance",
                                 facts={"can_reseed": bool(can_reseed), "reseed_tries": int(reseed_tries)})
        else:
            c = policy.Condition(kind="world_gate_failing", subject=f"{g.family}/{g.instance}", raised_by="worldgates.gate_instance",
                                 facts={"gate": name, "restartable": bool(restartable and name in GATE_STAGE), "rebuild_tries": int(rebuild_tries)})
        rem = policy.resolve(c)
        out.append({"gate": name, "condition": c.kind, "action": rem.action, "rule": rem.rule, "policy_version": rem.policy_version,
                    "stage": GATE_STAGE.get(name) if rem.action == "rebuild_stage" else None, "reason": rem.reason})
    return out


async def _panel_gate(panel: RealismPanel, inst: Path, executed: Mapping[str, GateResult] | None = None) -> GateResult:
    """The realism panel. An uncalibrated panel (λ > 1, R-panel's strict rule) decides nothing: the condition
    ``uncalibrated_panel`` goes to ``policy.resolve`` and its remedy is carried out: ``executed_check`` decides on the
    executed realism gates (integrity, surface, obscurity) and says so; any other remedy leaves the instance in the
    human queue, not used."""
    if panel.lam <= 1.0:
        return await panel.gate(await instance_summary(inst), ident=f"{inst.parent.parent.name}/{inst.name}")
    t0 = time.monotonic()
    ran = {g: r for g, r in (executed or {}).items() if g in REALISM_GATES}
    rem = policy.resolve(policy.Condition(kind="uncalibrated_panel", subject=f"{inst.parent.parent.name}/{inst.name}", raised_by="worldgates.panel",
                                          facts={"executed_check_available": len(ran) == len(REALISM_GATES), "panel_calibrated": False,
                                                 "more_anchors_available": False}))
    det = {"decision": "abstain", "lam": panel.lam, "remedy": rem.action, "rule": rem.rule, "policy_version": rem.policy_version}
    if rem.action == "executed_check":
        ok = all(r.passed for r in ran.values())
        return GateResult(gate="panel", passed=ok, details={**det, "decided_by": sorted(ran)}, elapsed_s=round(time.monotonic() - t0, 2),
                          reasons=[] if ok else [f"panel uncalibrated; executed realism gates failed: {sorted(g for g, r in ran.items() if not r.passed)}"])
    return GateResult(gate="panel", passed=False, details=det, elapsed_s=round(time.monotonic() - t0, 2),
                      reasons=[f"panel uncalibrated (lambda {panel.lam}); remedy {rem.action}: not used until decided"])


__all__ = [
    "N_MIN",
    "REALISM_RATER",
    "REQUIRED",
    "RUBRIC",
    "Anchor",
    "GateResult",
    "InstanceGates",
    "RealismOut",
    "RealismPanel",
    "anchors_from",
    "build_gate",
    "calibrate_panel",
    "degrade",
    "density_gate",
    "determinism_gate",
    "gate_instance",
    "instance_summary",
    "integrity_gate",
    "listing_problems",
    "obscurity_gate",
    "prior_reseeds",
    "reference_programs",
    "rejected_anchors",
    "remedies_for",
    "sibling_reach",
    "surface_gate",
    "synthetic_seed",
    "world_summary",
]
