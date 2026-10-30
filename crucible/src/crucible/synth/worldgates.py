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
| ``density`` | for each family write tool, a **reference oracle** (the tool called on instance entities with arguments taken from its blind tests, executed until it changes state), its skeleton contract, and the MUT falsifier with its effect deduplication | every target collection holds ≥ K entities, the best reference reaches n ≥ 59 distinct mutants, and at least half of the references do |
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
N_MIN = 59
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
                 SchemaOut.model_validate(_jread(app / "schema.json")), cfg, k_min=int(man.get("k_min", 40)),
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
                seen = {r.get(f) for r in g.tables.get(t.table, [])}
                missing = [v for v in gen["values"] if v not in seen]
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
                ids = [str(r.get(key_of[tn])) for r in seed.get(tn, []) if str(r.get(key_of[tn])) in seen][:60]
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
                prog = Program(steps=[Step(id="s1", tool=f"{app}.{t['name']}", args={k: Lit(v=v) for k, v in args.items()})])
                r = await run(world, prog, f"reference: {t['name']}", seed=0, taint=False)
                if r.error is None and all(c.obs.ok for c in r.calls) and r.start is not None and r.end is not None and r.start.fingerprint() != r.end.fingerprint():
                    found = {"tool": t["name"], "program": prog, "run": r, "subst": subst, "tries": tries}
                    break
            if found:
                break
        out.append(found or {"tool": t["name"], "program": None, "tries": tries})
    return out


async def density_gate(inst: Path, *, k_min: int = 40, n_min: int = N_MIN, max_mutants: int = 240, share: float = 0.5) -> tuple[GateResult, dict[str, bool]]:
    """The density gate (module docstring) and the replay determinism of each reference program."""
    from crucible.synth.falsify import MUT, FalsifyContext
    from crucible.synth.falsify.base import falsify
    from crucible.synth.infer import skeleton
    from crucible.synth.task import Task, WorldRef

    t0 = time.monotonic()
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
                per_tool.append({"tool": rf["tool"], "reference": False, "tries": rf.get("tries", 0)})
                continue
            r: Run = rf["run"]
            again = await run(w, rf["program"], f"reference: {rf['tool']}", seed=0, taint=False)
            replays[rf["tool"]] = bool(again.end is not None and r.end is not None and again.end.fingerprint() == r.end.fingerprint())
            task = Task(id=f"ref/{inst.parent.parent.name}/{inst.name}/{rf['tool']}", domain=inst.parent.parent.name, world=ref, seed=0,
                        instruction=f"reference: {rf['tool']}", contract=skeleton(r), oracle=rf["program"])
            ctx = FalsifyContext(max_mutants=max_mutants)
            try:
                res = await falsify(task, MUT(), ctx)
            finally:
                await ctx.close()
            ops: dict[str, int] = {}
            for f in res.findings:
                if f.outcome.valid:
                    ops[f.op] = ops.get(f.op, 0) + 1
            per_tool.append({"tool": rf["tool"], "reference": True, "n_distinct": res.n, "probes": len(res.findings), "by_op": ops,
                             "confirmed": res.c, "args": {k: compact(v.v, 60) for k, v in rf["program"].steps[0].args.items()}})
    finally:
        await w.close()
    ns = [p["n_distinct"] for p in per_tool if p.get("reference")]
    reached = [n for n in ns if n >= n_min]
    if not ns:
        reasons.append("no reference oracle could be built from the blind tests on this instance")
    elif not reached:
        reasons.append(f"no reference oracle reaches n >= {n_min} distinct mutants (best {max(ns)})")
    elif len(reached) / max(1, len(per_tool)) < share:  # a write tool with no reference oracle counts as not reaching n (pilot 1)
        reasons.append(f"only {len(reached)} of {len(per_tool)} write tools have a reference oracle reaching n >= {n_min} ({len(ns)} have one)")
    det = {"targets": counts, "k_min": k_min, "n_min": n_min, "write_tools": len(per_tool), "references": len(ns), "reached": len(reached), "best": max(ns, default=0),
           "median": statistics.median(ns) if ns else 0, "per_tool": per_tool}
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


async def calibrate_panel(panel: RealismPanel, anchors: Sequence[Anchor], *, target: float = 0.9, grid: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
                          floor: float = 0.6) -> dict[str, Any]:
    """λ from anchors whose truth is known (the rule of ``falsify.agents.calibrate``): the smallest grid value whose
    decided anchors are ≥ ``target`` accurate; none → the largest, flagged uncalibrated; never below ``floor``."""
    saved = panel.lam
    panel.lam = 0.0
    try:
        verdicts = [await panel.judge(a.summary, tag="realism:calibrate") for a in anchors]
    finally:
        panel.lam = saved
    rows, chosen = [], None
    for lam in grid:
        dec = cor = 0
        for a, v in zip(anchors, verdicts, strict=True):
            vals = {x.ok for x in v.votes}
            if None in vals or len(vals) != 1 or min(x.out.confidence for x in v.votes if x.out) < lam:
                continue
            dec += 1
            cor += int(vals.pop() == a.truth)
        acc = cor / dec if dec else 0.0
        rows.append({"lam": lam, "decided": dec, "correct": cor, "accuracy": round(acc, 4), "coverage": round(dec / max(1, len(anchors)), 4),
                     "error_upper95": round(clopper_pearson_upper(dec - cor, dec, 0.05), 4) if dec else 1.0})
        if chosen is None and dec and acc >= target:
            chosen = lam
    panel.lam = max(floor, chosen if chosen is not None else max(grid))
    votes = [{"kind": a.kind, "truth": a.truth, "votes": [x.model_dump(mode="json") for x in v.votes]} for a, v in zip(anchors, verdicts, strict=True)]
    return {"lam": panel.lam, "calibrated": chosen is not None, "target": target, "rows": rows, "votes": votes}


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


# ---------------------------------------------------------------------------------------------------------- one instance

async def gate_instance(fdir: Path, inst: Path, *, panel: RealismPanel | None = None, k_min: int | None = None, n_min: int = N_MIN,
                        max_mutants: int = 240, only: Sequence[str] | None = None) -> InstanceGates:
    """Every gate on one instance (``only``: a subset), recorded in ``<inst>/gates.json`` (earlier results kept)."""
    fdir, inst = Path(fdir), Path(inst)
    path = inst / "gates.json"
    g = InstanceGates.model_validate_json(path.read_text()) if path.exists() else InstanceGates(family=fdir.name, instance=inst.name, dir=str(inst))
    man = frozen_manifest(fdir)
    k = int(k_min or man.get("k_min", 40))
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
        keep(await guarded("panel", _panel_gate(panel, inst)))
    g.decide(REQUIRED)
    _jwrite(path, g.model_dump(mode="json"))
    return g


async def _panel_gate(panel: RealismPanel, inst: Path) -> GateResult:
    return await panel.gate(await instance_summary(inst), ident=f"{inst.parent.parent.name}/{inst.name}")


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
    "obscurity_gate",
    "reference_programs",
    "surface_gate",
    "world_summary",
]
