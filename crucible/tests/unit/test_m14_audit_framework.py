"""M14 audit: the pack framework (docs/audit/2026-10-13, H1 V1-V22 and sweep G2 D1-D12, picks 1, 3, 4, 5, 6).

Two mutant campaigns, each a parametrized regression test: every broken pack must fail ``pack check`` with a reason
naming the check that caught it, and the honest controls must pass.
- H1: 25 packs scaffolded with ``pack new`` and mutated (graders that always pass, NaN, out-of-range, flaky, raising;
  missing, empty or foreign hooks; placeholders; licences; entry points), plus an honest lower-is-better pack that must
  pass. (The scaffolded greeting task gets ``oracle_actions`` here: the template change the packs batch makes.)
- G2: 25 broken copies of ``finance_ap`` (lenient graders, lying self-graded fixtures, a reason-blind grader, an answer key
  in the knowledge base, pyproject disagreements, raising hooks, ...).
Then one test per framework fix (isolation, exit codes, redaction, ingest confinement, names, bench thresholds, accept)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from crucible.cli.main import cli


def _pk():
    import crucible.studio.pack as pk

    return pk


def run_cli(*args):
    return CliRunner().invoke(cli, list(args), catch_exceptions=False)


def failing(rep) -> set[str]:
    return {r.id for r in rep.results if r.status == "fail"}


# ---- H1 mutants -------------------------------------------------------------------------------------------------------
HEAD = ("\n\n# ---- m14 audit mutant ----\nimport math as _m, random as _rnd\nfrom dataclasses import replace as _rep\n"
        "GreetingTask.oracle_actions = lambda self: self.gold()\n_orig_grade = GreetingTask.grade\n")


def grade_patch(body: str) -> str:
    return HEAD + ("async def _g(self, world, workdir):\n    rep = await _orig_grade(self, world, workdir)\n    ok = rep.reward == 1.0\n"
                   "    _NS = lambda **kw: _rep(rep, **kw)\n" + "".join("    " + x + "\n" for x in body.strip().splitlines()) + "GreetingTask.grade = _g\n")


def expect(mapping: str) -> str:
    return f"\n_orig_fixtures = fixtures\ndef fixtures():\n    cs = _orig_fixtures()\n    for c in cs:\n        c['reward'] = ({mapping})(c)\n    return cs\n"


ONE = "return _NS(reward=1.0, gates={'outcome': 'pass'})"
GOLD_ONLY = "\n_of = fixtures\ndef fixtures():\n    return [c for c in _of() if c['name'] == 'gold']\n"
H1: dict[str, tuple[dict, set[str] | None]] = {  # mutant -> (patch, failing check ids it must show; None = must pass)
    "always1": ({"py": grade_patch(ONE)}, {"fixture.cases"}),
    "always1_nohooks": ({"py": grade_patch(ONE), "yaml": [(r"(?m)^  tasks: ", "  #tasks: "), (r"(?m)^  fixtures: ", "  #fixtures: ")]},
                        {"fixture.hook", "semantic.tasks"}),
    "always1_builtins_list": ({"py": grade_patch(ONE), "yaml": [(r"(?m)^  tasks: \S+", "  tasks: builtins:list"), (r"(?m)^  fixtures: \S+", "  fixtures: builtins:list")]},
                              {"static.hooks"}),
    "always1_fixtures_empty": ({"py": grade_patch(ONE) + "\ndef fixtures():\n    return []\ndef tasks():\n    return []\n"}, {"fixture.cases", "semantic.tasks"}),
    "always1_gold_only_fixture": ({"py": grade_patch(ONE) + expect("lambda c: 1.0")}, {"fixture.cases"}),
    "always1_requires_network": ({"py": grade_patch(ONE) + expect("lambda c: 1.0"), "yaml": [(r"(?m)^determinism: seeded", "determinism: seeded\nrequires: [network]")]},
                                 {"fixture.cases"}),
    "always0": ({"py": grade_patch("return _NS(reward=0.0, gates={'outcome': 'fail'})")}, {"fixture.cases"}),
    "nan_reward": ({"py": grade_patch("return _NS(reward=float('nan'), gates={})")}, {"fixture.cases"}),
    "out_of_range": ({"py": grade_patch("return _NS(reward=1.5 if ok else -3.0, gates={'outcome': 'pass' if ok else 'fail'})")
                      + expect("lambda c: 1.5 if c['name'] == 'gold' else -3.0")}, {"fixture.cases"}),
    "noop_near_top": ({"py": grade_patch("return _NS(reward=1.0 if ok else 0.999, gates={'outcome': 'pass' if ok else 'fail'})")
                       + expect("lambda c: 1.0 if c['name'] == 'gold' else 0.999")}, {"fixture.kinds", "semantic.noop"}),
    "gate_typo_always_pass_gate": ({"py": grade_patch("return _NS(reward=rep.reward, gates={'result': 'pass'})")}, {"fixture.cases"}),
    "raising": ({"py": grade_patch("raise RuntimeError('grader crashed')")}, {"fixture.error"}),
    "nondet_tasks": ({"py": HEAD + "_orig_init = GreetingTask.__init__\ndef _init(self, seed=0):\n    _orig_init(self, seed)\n"
                            "    self.word = _rnd.SystemRandom().choice(WORDS)\n"
                            "    self.instruction = f'Create greeting.txt containing exactly the word {self.word!r} followed by a newline.'\n"
                            "GreetingTask.__init__ = _init\n"}, {"semantic.determinism"}),
    "license_tbd": ({"yaml": [(r"(?m)^license: .*$", "license: TBD")]}, {"static.license"}),
    "license_none": ({"yaml": [(r"(?m)^license: .*$", "license: none")]}, {"static.license"}),
    "todo_in_src": ({"py": "\n# TODO: write a real grader; this one is the template's\n", "readme": "\n(edit this description)\n"}, {"static.placeholders"}),
    "foreign_hooks": ({"py": grade_patch(ONE), "yaml": [(r"(?m)^  tasks: \S+", "  tasks: crucible_pack_toy.studio:tasks"),
                                                        (r"(?m)^  fixtures: \S+", "  fixtures: crucible_pack_toy.studio:fixtures")]}, {"static.hooks"}),
    "entry_point_broken": ({"pyproject": [(r'= "crucible_pack_', '= "no_such_module_')], "init": "\n\ndef register(registry):\n    raise RuntimeError('register is broken')\n"},
                           {"static.entry_point"}),
    "threshold_out_of_range": ({"yaml": [(r"(?m)^  threshold: .*$", "  threshold: 7.0")]}, {"static.manifest"}),
    "always1_gold_only": ({"py": grade_patch(ONE) + GOLD_ONLY}, {"fixture.kinds", "semantic.noop"}),
    "always1_gold_only_req_net": ({"py": grade_patch(ONE) + GOLD_ONLY, "yaml": [(r"(?m)^determinism: seeded", "determinism: seeded\nrequires: [network]")]},
                                  {"fixture.kinds"}),
    "always1_req_gpu_no_fixtures": ({"py": grade_patch(ONE), "yaml": [(r"(?m)^determinism: seeded", "determinism: seeded\nrequires: [gpu]"),
                                                                      (r"(?m)^  fixtures: ", "  #fixtures: ")]}, {"fixture.hook"}),
    "nondet_grader": ({"py": grade_patch("ok = ok if _FLIP.random() > 0.15 else not ok\nreturn _NS(reward=1.0 if ok else 0.0, gates={'outcome': 'pass' if ok else 'fail'})")
                              .replace("_orig_grade = GreetingTask.grade\n", "_orig_grade = GreetingTask.grade\n_FLIP = _rnd.Random(7)\n")},
                      {"fixture.cases", "fixture.determinism"}),
    "flip_second_grade": ({"py": grade_patch("_N[0] += 1\nreturn _NS(reward=0.0, gates={'outcome': 'fail'}) if ok and _N[0] % 2 == 0 else rep")
                                  .replace("_orig_grade = GreetingTask.grade\n", "_orig_grade = GreetingTask.grade\n_N = [0]\n")},
                          {"fixture.cases", "fixture.determinism", "semantic.determinism", "semantic.gold"}),
    # an honest cost-style pack: gold scores the floor, everything else the top; must PASS (V12: it failed semantic)
    "lower_is_better": ({"yaml": [(r"higher_is_better: true", "higher_is_better: false")],
                         "py": grade_patch("return _NS(reward=0.0 if ok else 1.0, gates={'outcome': 'pass' if ok else 'fail'})")
                         + expect("lambda c: 0.0 if c['name'] == 'gold' else 1.0")}, None),
}


def make_mutant(root: Path, name: str, spec: dict) -> Path:
    pk = _pk()
    pkg_name = "mx_" + name
    d = root / name
    pk.new(pkg_name, d)
    pkg = d / "src" / f"crucible_pack_{pkg_name}"
    y = d / "pack.yaml"
    t = y.read_text()
    for a, b in spec.get("yaml", []):
        assert re.search(a, t), (name, a)
        t = re.sub(a, b, t)
    y.write_text(t)
    if "py" in spec:
        (pkg / "studio.py").write_text((pkg / "studio.py").read_text() + spec["py"])
    if "init" in spec:
        (pkg / "__init__.py").write_text((pkg / "__init__.py").read_text() + spec["init"])
    if "readme" in spec:
        (d / "README.md").write_text((d / "README.md").read_text() + spec["readme"])
    for a, b in spec.get("pyproject", []):
        p = d / "pyproject.toml"
        assert re.search(a, p.read_text()), (name, a)
        p.write_text(re.sub(a, b, p.read_text()))
    return d


@pytest.mark.parametrize("name", sorted(H1))
async def test_h1_mutants_fail_pack_check(tmp_path, name):
    spec, must = H1[name]
    d = make_mutant(tmp_path, name, spec)
    rep = await _pk().check(d, level="semantic")
    got = failing(rep)
    if must is None:
        assert rep.ok, rep.to_json()["problems"]
    else:
        assert not rep.ok and got & must, (name, sorted(got), rep.to_json()["problems"])
        assert all(r.detail for r in rep.results if r.status == "fail")  # every failure says why


def test_h1_raw_template_copy_fails_manifest_and_placeholders(tmp_path):
    """V10: an unsubstituted ``_template`` copy (``name: __PACK_NAME__``) passed every level."""
    pk = _pk()
    d = tmp_path / "rawtpl"
    shutil.copytree(pk.TEMPLATE, d, ignore=shutil.ignore_patterns("__pycache__"))
    (d / "pack.yaml").write_text(re.sub(r"(?m)^template_version: .*\n", "", (d / "pack.yaml").read_text()))
    rep = asyncio.run(pk.check(d, level="semantic"))
    assert {"static.manifest", "static.placeholders"} <= failing(rep), rep.to_json()["problems"]


# ---- G2: broken finance_ap copies ---------------------------------------------------------------------------------------
GHEAD = "\n\n# ---- m14 audit variant ----\nimport dataclasses as _dc, json as _json\n_orig_grade = APTask.grade\n"


def g_grade(body: str) -> str:
    return GHEAD + ("async def _g(self, world, workdir):\n    rep = await _orig_grade(self, world, workdir)\n"
                    + "".join("    " + x + "\n" for x in body.strip().splitlines()) + "APTask.grade = _g\n")


LENIENT = g_grade("st = await self.subgoals.state(world)\ndecided = any((st.get(c.invoice_id) or ('pending', None))[0] != 'pending' for c in self.w.cases)\n"
                  "r = 1.0 if decided else 0.0\nreturn _dc.replace(rep, reward=r, outcome=r, gates={'controls': 'pass'})")
REASON_BLIND = GHEAD + """async def _g(self, world, workdir):
    st = await self.subgoals.state(world)
    saved = []
    for c in self.w.cases:
        s = st.get(c.invoice_id)
        if c.expected[0] == "rejected" and s and s[0] == "rejected" and tuple(s) != tuple(c.expected):
            saved.append((c, c.expected))
            object.__setattr__(c, "expected", tuple(s))
    try:
        return await _orig_grade(self, world, workdir)
    finally:
        for c, e in saved:
            object.__setattr__(c, "expected", e)
APTask.grade = _g
"""
LYING = """
def fixtures():
    async def g1():
        return 1.0, {"controls": "pass"}
    async def g0():
        return 0.0, {"controls": "fail"}
    return [{"name": "gold", "grade": g1, "reward": 1.0}, {"name": "approves_duplicate", "grade": g0, "reward": 0.0, "fails_gate": "controls"}]
"""
TYPO = g_grade("return _dc.replace(rep, gates={k: 'pass' for k in rep.gates})") + """
_orig_fixtures = fixtures
def fixtures():
    cs = _orig_fixtures()
    for c in cs:
        if c.get("fails_gate") == "controls":
            c["fails_gate"] = "contrls"
    return cs
"""
FLAKY = g_grade("_N[0] += 1\nif _N[0] % 3 == 0:  # the controls gate skipped on every third grade\n    return _dc.replace(rep, gates={'controls': 'pass'}, reward=rep.outcome)\nreturn rep"
                ).replace("_orig_grade = APTask.grade\n", "_orig_grade = APTask.grade\n_N = [0]\n")
LEAK = """
from crucible_pack_finance_ap.world import oracle as _oracle
import json as _json2
_orig_write_pack = write_pack
def write_pack(w, root):
    out = _orig_write_pack(w, root)
    (Path(root) / "data" / "kb" / "answers.md").write_text(_json2.dumps(_oracle(w)))
    return out
"""
G2: dict[str, tuple[dict, str, set[str] | None]] = {  # variant -> (patch, level, failing check ids; None = must pass)
    "control_unchanged": ({}, "semantic", None),
    "requires_undeclared_network_paid": ({}, "semantic", None),  # a control: nothing to catch
    "always_one": ({"py": g_grade("return _dc.replace(rep, reward=1.0)")}, "fixture", {"fixture.cases"}),
    "always_nan": ({"py": g_grade("return _dc.replace(rep, reward=float('nan'))")}, "fixture", {"fixture.cases"}),
    "reward_above_declared_range": ({"yaml": [(r"(?m)^  range: .*$", "  range: [0.0, 0.5]"), (r"(?m)^  threshold: .*$", "  threshold: 0.5")]}, "fixture",
                                    {"fixture.cases"}),
    "grader_ignores_reject_reason": ({"py": REASON_BLIND}, "semantic", {"fixture.cases", "semantic.adversaries"}),
    "lenient_grader_no_fixture_hook": ({"py": LENIENT, "yaml": [(r"(?m)^  fixtures: .*$", "#  fixtures: removed")]}, "semantic", {"fixture.hook"}),
    "lenient_grader_lying_fixtures": ({"py": LENIENT + LYING}, "semantic", {"fixture.cases"}),
    "fixtures_empty": ({"py": "\ndef fixtures():\n    return []\n"}, "fixture", {"fixture.cases"}),
    "fails_gate_typo_and_gate_never_fails": ({"py": TYPO}, "fixture", {"fixture.cases"}),
    "flaky_grader": ({"py": FLAKY}, "fixture", {"fixture.cases", "fixture.determinism"}),
    "tasks_empty": ({"py": "\ndef tasks():\n    return []\n"}, "semantic", {"semantic.tasks"}),
    "license_TBD": ({"yaml": [(r"(?m)^license: .*$", "license: TBD")]}, "static", {"static.license"}),
    "license_none": ({"yaml": [(r"(?m)^license: .*$", "license: none")]}, "static", {"static.license"}),
    "license_missing": ({"yaml": [(r"(?m)^license: .*\n", "")]}, "static", {"static.manifest"}),
    "license_mismatch_pyproject_GPL": ({"pyproject": [(r"(?m)^(description = .*)$", '\\1\nlicense = "GPL-3.0-only"')]}, "static", {"static.pyproject"}),
    "placeholder_FIXME_and_template_text": ({"yaml": [(r"(?m)^description: ", "description: FIXME A Crucible domain pack with one workspace task (edit this description). ")],
                                             "readme": [(r"(?m)^# .*$", "# <describe your pack here> XXX")]}, "static", {"static.placeholders"}),
    "placeholder_TODO_in_src_only": ({"init": "# TODO: implement the pack.\n"}, "static", {"static.placeholders"}),
    "bad_name_version": ({"yaml": [(r"(?m)^name: .*$", "name: 'Finance AP!! v2'"), (r"(?m)^version: .*$", "version: banana")]}, "static", {"static.manifest"}),
    "pyproject_entry_point_wrong": ({"pyproject": [(r"(?m)^finance_ap = .*$", 'finance_ap = "does_not_exist:nope"')]}, "static", {"static.entry_point"}),
    "hook_points_outside_pack": ({"yaml": [(r"(?m)^  stages: \S+", "  stages: os:getcwd")]}, "static", {"static.hooks"}),
    "fixtures_hook_raises": ({"py": "\ndef fixtures():\n    raise RuntimeError('fixture hook boom')\n"}, "fixture", {"fixture.error"}),
    "tasks_hook_raises": ({"py": "\ndef tasks():\n    raise RuntimeError('tasks hook boom')\n"}, "semantic", {"semantic.error"}),
    "grade_raises": ({"py": g_grade("raise RuntimeError('grade boom')")}, "fixture", {"fixture.error"}),
    "readme_missing": ({"rm": ["README.md"]}, "publish", {"static.readme"}),  # a warning in dev, a failure for --publish
}


def make_variant(root: Path, name: str, spec: dict) -> Path:
    pk = _pk()
    d = root / name
    shutil.copytree(pk.PACKS_ROOT / "finance_ap", d, ignore=shutil.ignore_patterns("__pycache__"))
    pkg = d / "src" / "crucible_pack_finance_ap"
    for key, f in (("yaml", d / "pack.yaml"), ("pyproject", d / "pyproject.toml"), ("readme", d / "README.md")):
        t = f.read_text()
        for a, b in spec.get(key, []):
            assert re.search(a, t), (name, a)
            t = re.sub(a, b, t, count=1)
        f.write_text(t)
    if "py" in spec:
        (pkg / "studio.py").write_text((pkg / "studio.py").read_text() + spec["py"])
    if "init" in spec:
        (pkg / "__init__.py").write_text(spec["init"] + (pkg / "__init__.py").read_text())
    for r in spec.get("rm", []):
        (d / r).unlink()
    return d


@pytest.mark.parametrize("name", sorted(G2))
async def test_g2_broken_finance_ap_copies_fail(tmp_path, name):
    spec, level, must = G2[name]
    d = make_variant(tmp_path, name, spec)
    pk = _pk()
    rep = await (pk.check(d, level="static", publish=True) if level == "publish" else pk.check(d, level=level))
    got = failing(rep)
    if must is None:
        assert rep.ok, rep.to_json()["problems"]
    else:
        assert not rep.ok and got & must, (name, sorted(got), rep.to_json()["problems"])


async def test_g2_answer_key_in_the_kb_fails_the_accept_level(tmp_path):
    """G2 D3 / pick 3: no level looked for leaks and finance_ap's tasks never reached M9 A5. An answer key written into the
    knowledge base is now an ``accept.leak`` failure (the clean pack's A5 passes)."""
    pk = _pk()
    d = make_variant(tmp_path, "answer_key_leaked_to_kb", {"py": LEAK})
    rep = await pk.check(d, level="accept")
    st = {r.id: r for r in rep.results}
    assert st["accept.leak"].status == "fail" and "A5" in st["accept.leak"].detail, rep.to_json()["problems"]
    clean = await pk.check(make_variant(tmp_path, "clean", {}), level="accept")
    assert {r.id: r.status for r in clean.results}["accept.leak"] == "pass"


# ---- isolation, crashes, exit codes (V7, V14, V15; G2 D5, pick 5) ------------------------------------------------------
def test_v7_a_copy_with_the_same_package_is_checked_with_its_own_code(tmp_path):
    """V7 / G2 D5: after the real pack was checked, a broken copy with the same package name passed (its studio module was
    the original's). Each check imports the pack's own src/ and restores sys.modules afterwards."""
    pk = _pk()
    real = asyncio.run(pk.check("finance_ap", level="fixture"))
    assert real.ok, real.to_json()["problems"]
    mod_before = sys.modules.get("crucible_pack_finance_ap.studio")
    d = make_variant(tmp_path, "always_one_same_pkg", {"py": g_grade("return _dc.replace(rep, reward=1.0)")})
    broken = asyncio.run(pk.check(d, level="fixture"))
    assert not broken.ok and "fixture.cases" in failing(broken)
    assert sys.modules.get("crucible_pack_finance_ap.studio") is mod_before  # the original module is back


def test_v15_check_restores_sys_path_and_refuses_stray_top_level_modules(tmp_path):
    pk = _pk()
    d = tmp_path / "shadow"
    pk.new("shadowpk", d)
    (d / "src" / "shadowmod_m14.py").write_text("SHADOW = 'pack-supplied'\n")
    before = list(sys.path)
    rep = asyncio.run(pk.check(d, level="static"))
    assert "static.layout" in failing(rep)
    assert sys.path == before


def test_v14_crashes_are_check_results_and_exit_codes_are_documented(tmp_path, monkeypatch):
    pk = _pk()
    d = make_mutant(tmp_path, "raising", H1["raising"][0])
    r = run_cli("pack", "check", str(d), "--level", "fixture", "--json")
    j = json.loads(r.output)
    assert r.exit_code == 1 and j["exit"] == 1 and any(p.startswith("fixture.error: RuntimeError: grader crashed") for p in j["problems"])
    # --all: one process per pack, every report kept whatever another pack does
    rep = pk.check_in_subprocess(d, level="fixture")
    assert not rep.ok and "fixture.error" in failing(rep)
    ok = pk.check_in_subprocess(pk.PACKS_ROOT / "_toy", level="static")
    assert ok.ok and ok.pack == "toy"
    # an error outside the pack's code is exit 3
    monkeypatch.setattr(pk, "check_static", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crucible bug")))
    r = run_cli("pack", "check", str(d))
    assert r.exit_code == 3


def test_v6_requires_is_a_loud_skip(tmp_path):
    """V6: declaring an unmet requirement turned semantic into a silent skip with ok=true and exit 0."""
    pk = _pk()
    d = make_mutant(tmp_path, "net", {"py": HEAD, "yaml": [(r"(?m)^determinism: seeded", "determinism: seeded\nrequires: [network]")]})
    rep = asyncio.run(pk.check(d, level="semantic"))
    j = rep.to_json()
    assert not rep.ok and not j["complete"] and j["skipped"][0][0] == "semantic" and rep.exit_code == 2
    assert {"semantic.gold", "semantic.noop"} <= {r.id for r in rep.results if r.status == "skip"}
    assert run_cli("pack", "check", str(d), "--level", "semantic").exit_code == 2
    assert run_cli("pack", "check", str(d), "--level", "semantic", "--allow-skip").exit_code == 0
    assert run_cli("pack", "check", str(d), "--level", "semantic", "--allow", "network").exit_code == 0


def test_v22_docker_probe_has_a_timeout_and_cleanup_runs_when_an_episode_raises(monkeypatch):
    import subprocess

    pk = _pk()
    from crucible.core.schema.pack import PackManifest

    m = PackManifest.model_validate({"name": "dk", "version": "0.1.0", "description": "d", "license": "MIT", "package": "p", "entry_point": "p:register",
                                     "requires": ["docker"]})
    seen = {}

    def hang(cmd, **kw):
        seen["timeout"] = kw.get("timeout")
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout") or 0)

    monkeypatch.setattr(pk.shutil, "which", lambda c: "/usr/bin/" + c)
    monkeypatch.setattr(pk.subprocess, "run", hang)
    assert pk.missing_requirements(m) == ["docker"] and seen["timeout"]

    class Boom:
        id, instruction = "boom@1", "x"
        cleaned = 0

        async def open(self, wd):
            raise RuntimeError("open failed")

        def cleanup(self):
            Boom.cleaned += 1

    class FakeHooks:
        pass

    async def go(tmp: Path):
        m2 = PackManifest.model_validate({"name": "bm", "version": "0.1.0", "description": "d", "license": "MIT", "package": "p", "entry_point": "p:register",
                                          "hooks": {"tasks": "p:tasks"}})
        monkeypatch.setattr(pk, "call_hook", lambda *a, **k: _aw([(Boom(), [])]))
        with pytest.raises(RuntimeError, match="open failed"):
            await pk.check_semantic(tmp, m2)

    async def _aw(v):
        return v

    asyncio.run(go(Path("/nonexistent")))
    assert Boom.cleaned == 1


# ---- static: manifest, licence, pyproject, hooks, entry point (V10, V11, V18; G2 D10, D12, picks 4, 6) ------------------
def test_manifest_identity_threshold_and_ingest_fields():
    from crucible.core.schema.pack import PackManifest

    base = {"name": "xy", "version": "0.1.0", "description": "d", "license": "MIT", "package": "p", "entry_point": "p:register"}
    PackManifest.model_validate(base)
    for bad, match in (({"name": "Finance AP!! v2"}, "name"), ({"version": "banana"}, "version"), ({"label": {"threshold": 7.0}}, "threshold"),
                       ({"entry_point": "other:register"}, "entry_point"), ({"reward": {"range": [0, float("nan")]}}, "finite"),
                       ({"ingest": [{"family": "f", "glob": "/etc/hosts"}]}, "absolute"),
                       ({"ingest": [{"family": "f", "glob": "*.jsonl", "root": "/etc"}]}, "absolute"),
                       ({"ingest": [{"family": "f", "glob": "*.jsonl", "init": {"root": "x"}}]}, "does not use"),
                       ({"ingest": [{"family": "f", "connector": "agent_runs", "glob": "*.jsonl"}]}, "does not use")):
        with pytest.raises(ValueError, match=match):
            PackManifest.model_validate(base | bad)


def test_g2_pick6_core_version_bound_and_wheel_contents(tmp_path):
    pk = _pk()
    d = tmp_path / "dist"
    pk.new("distpk", d)
    pp = d / "pyproject.toml"
    import re

    pp.write_text(re.sub(r"(?m)^dependencies = \[.*\]$", 'dependencies = ["crucible>=99"]', pp.read_text()))  # the template declares its own bound
    rep = asyncio.run(pk.check(d))
    assert "static.pyproject" in failing(rep) and "requires crucible>=99" in {r.id: r.detail for r in rep.results}["static.pyproject"]
    pp.write_text(pp.read_text().replace('"crucible>=99"', '"crucible>=0.1,<1"'))
    if "force-include" not in pp.read_text():
        pp.write_text(pp.read_text() + '\n[tool.hatch.build.targets.wheel.force-include]\n"pack.yaml" = "x/pack.yaml"\n')
    rep = asyncio.run(pk.check(d))
    assert {r.id: r.status for r in rep.results}["static.distribution"] == "pass", rep.to_json()
    assert asyncio.run(pk.check(d, publish=True)).exit_code == 1  # template text and no authors are publication failures


def test_v11_register_must_register_this_pack_here(tmp_path):
    pk = _pk()
    d = tmp_path / "reg"
    pk.new("regpk", d)
    init = d / "src" / "crucible_pack_regpk" / "__init__.py"
    init.write_text(init.read_text().replace('registry.register("pack", "regpk"', 'registry.register("pack", "someone_else"'))
    rep = asyncio.run(pk.check(d))
    assert "static.entry_point" in failing(rep) and "registers no pack 'regpk'" in {r.id: r.detail for r in rep.results}["static.entry_point"]


# ---- fixture self-graded cases (G2 D2b) ----------------------------------------------------------------------------------
def test_self_graded_case_must_be_the_packs_own_code(tmp_path):
    """A ``grade`` callable from outside the pack is not evidence about the pack's grader (packs without live tasks)."""
    pk = _pk()
    d = make_mutant(tmp_path, "selfgraded", {"py": HEAD + "\nfrom crucible.studio.pack import _num as _foreign\n"
                                             "async def _ext():\n    return 1.0, {}\n_ext.__module__ = 'somewhere_else'\n"
                                             "def fixtures():\n    return [{'name': 'gold', 'grade': _ext, 'reward': 1.0}]\n",
                                             "yaml": [(r"(?m)^  tasks: ", "  #tasks: "), (r"(?m)^  generator: ", "  #generator: "), (r"(?m)^  policies: ", "  #policies: ")]})
    rep = asyncio.run(pk.check(d, level="fixture"))
    assert "outside the pack" in {r.id: r.detail for r in rep.results}["fixture.cases"]


# ---- adversaries (G2 pick 2 consumption) ---------------------------------------------------------------------------------
def test_adversaries_from_fixtures_and_policies_are_bounded(tmp_path):
    pk = _pk()
    patch = HEAD + ("\nfrom crucible.core.schema.actions import ToolCall as _TC\n"
                    "def _writes_word(task):\n    return [_TC(tool_call_id='x', function_name='ws.write_file', arguments={'path': 'greeting.txt', 'content': task.word + '\\n'})]\n"
                    "_of = fixtures\ndef fixtures():\n    return _of() + [{'name': 'cheater', 'policy': _writes_word, 'expect_max': 0.0}]\n")
    d = make_mutant(tmp_path, "adv", {"py": patch})
    rep = asyncio.run(pk.check(d, level="semantic"))
    det = {r.id: r.detail for r in rep.results}
    assert "semantic.adversaries" in failing(rep) and "cheater on" in det["semantic.adversaries"]
    d2 = make_mutant(tmp_path, "adv2", {"py": patch.replace("'expect_max': 0.0", "'expect_max': 1.0")
                                        + "\n_op = policies\ndef policies():\n    return [*_op(), {'policy': 'scripted:noop', 'expect_max': 0.0}]\n"})
    rep2 = asyncio.run(pk.check(d2, level="semantic"))
    assert rep2.ok, rep2.to_json()["problems"]
    n_adv = int(re.match(r"(\d+) adversar", {r.id: r.detail for r in rep2.results}["semantic.adversaries"]).group(1))
    base = asyncio.run(pk.check(make_mutant(tmp_path, "adv0", {"py": HEAD}), level="semantic"))
    n0 = int((re.match(r"(\d+) adversar", {r.id: r.detail for r in base.results}["semantic.adversaries"]) or re.match("()", "0")).group(1) or 0)
    assert n_adv == n0 + 2  # the fixtures entry and the bounded policies entry (by id) both ran
    refs = pk.policy_refs([{"policy": pk.policy_refs([lambda t: []])[0], "expect_max": 0.0}, lambda t: []])
    assert all(hasattr(r, "make") for r in refs)


# ---- generator oracle (V13) --------------------------------------------------------------------------------------------
def test_v13_generated_items_without_an_oracle_fail_semantic(tmp_path):
    pk = _pk()
    d = make_mutant(tmp_path, "nooracle", {"py": "\n# ---- m14 audit ----\nGreetingTask.oracle_actions = None\n"})
    rep = asyncio.run(pk.check(d, level="semantic"))
    assert "semantic.generator" in failing(rep) and "no oracle" in {r.id: r.detail for r in rep.results}["semantic.generator"]


# ---- redaction_env (V5) ------------------------------------------------------------------------------------------------
FAKE = "zq9Xk2pp81QQ7mVtFAKECANARY"  # a fake canary, never a real credential


def _ingest_pack(tmp: Path, env_name: str, *, optional: bool = False) -> Path:
    pk = _pk()
    d = tmp / "ing"
    pk.new("ingpk", d)
    (d / ".env.local").write_text(f"DB_PASSWORD={FAKE}\n")
    (d / "data" / "logs").mkdir(parents=True)
    (d / "data" / "logs" / "a.jsonl").write_text(json.dumps({"case": "c1", "id": "c1-1", "time": "2026-01-01T00:00:00Z", "actor": "agent", "kind": "message", "type": "note",
                                                            "text": f"connect with {FAKE}"}) + "\n")
    y = d / "pack.yaml"
    y.write_text(y.read_text().replace("label:\n", f"ingest:\n  - {{connector: file_drop, family: ingpk/case, glob: data/logs/*.jsonl, redaction_env: {env_name}"
                                                  + (", redaction_env_optional: true" if optional else "") + "}\nlabel:\n", 1))
    return d


def test_v5_a_misspelled_redaction_env_fails_check_and_label_refuses(tmp_path):
    from crucible.connectors.redact import RedactionPolicy
    from crucible.studio.label import ingest
    from crucible.studio.pack import load_manifest

    pk = _pk()
    with pytest.raises(FileNotFoundError):
        RedactionPolicy.with_env_file(tmp_path / "missing.env")
    assert RedactionPolicy.with_env_file(tmp_path / "missing.env", missing_ok=True).secret_values == []
    d = _ingest_pack(tmp_path, "./.env.lcoal")
    rep = asyncio.run(pk.check(d))
    assert "static.ingest" in failing(rep) and ".env.lcoal does not exist" in {r.id: r.detail for r in rep.results}["static.ingest"]
    with pytest.raises(ValueError, match="does not exist"):
        ingest(d, load_manifest(d), tmp_path / "store")
    assert not any((tmp_path / "store").rglob("*")) if (tmp_path / "store").exists() else True
    y = d / "pack.yaml"
    y.write_text(y.read_text().replace(".env.lcoal", ".env.local"))
    red: dict = {}
    traces = ingest(d, load_manifest(d), tmp_path / "store2", redaction=red)
    assert traces and FAKE not in json.dumps([t.model_dump(mode="json") for t in traces]) and red["ingpk/case"]["values"] == 1


# ---- ingest confinement (V17, V20) -------------------------------------------------------------------------------------
def test_v17_ingest_paths_are_confined(tmp_path, monkeypatch):
    from crucible.studio.label import ingest
    from crucible.studio.pack import load_manifest

    pk = _pk()
    monkeypatch.delenv("CRUCIBLE_DATA_ROOTS", raising=False)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.jsonl").write_text(json.dumps({"case": "c1", "id": "c1-1", "time": "2026-01-01T00:00:00Z", "actor": "agent", "kind": "message", "type": "note", "text": "hi"}) + "\n")
    d = tmp_path / "packs" / "conf"
    pk.new("confpk", d)
    (d / "data").mkdir()
    os.symlink(outside, d / "data" / "link")
    y = d / "pack.yaml"
    base = y.read_text()
    y.write_text(base.replace("label:\n", "ingest:\n  - {connector: file_drop, family: c/x, glob: data/link/*.jsonl}\nlabel:\n", 1))
    rep = asyncio.run(pk.check(d))
    assert "static.ingest" in failing(rep) and "outside the allowed roots" in {r.id: r.detail for r in rep.results}["static.ingest"]
    with pytest.raises(ValueError, match="outside the allowed roots"):
        ingest(d, load_manifest(d), tmp_path / "store")
    y.write_text(base.replace("label:\n", "ingest:\n  - {connector: file_drop, family: c/x, glob: ../../outside/*.jsonl}\nlabel:\n", 1))
    assert "static.ingest" in failing(asyncio.run(pk.check(d)))
    monkeypatch.setenv("CRUCIBLE_DATA_ROOTS", str(outside))  # a root named in configuration is allowed
    assert "static.ingest" not in failing(asyncio.run(pk.check(d)))


def test_v20_one_resolver_relative_init_and_nested_scope():
    pk = _pk()
    root = Path("/p")
    v = pk.ingest_value({"root": "data/runs", "now": "2026", "nested": {"cache_dir": "c"}, "glob": "*/x.json"}, root, init=True)
    assert v["root"] == root / "data/runs" and v["nested"]["cache_dir"] == root / "c" and v["now"] == "2026" and v["glob"] == "*/x.json"
    s = pk.ingest_value({"paths": ["gpu", "./local"], "repo": "o/r"}, root)
    assert s["paths"] == ["gpu", root / "./local"] and s["repo"] == "o/r"


# ---- names and the registry (V19; G2 D9) -------------------------------------------------------------------------------
def test_v19_names_are_unique_and_resolve_through_the_index(tmp_path, monkeypatch):
    pk = _pk()
    with pytest.raises(pk.ScaffoldConflict, match="already exists"):
        pk.new("toy", tmp_path / "toy")
    monkeypatch.chdir(tmp_path)
    shutil.copytree(pk.PACKS_ROOT / "_toy", tmp_path / "toy", ignore=shutil.ignore_patterns("__pycache__", "data"))
    assert pk.find_pack("toy") == (pk.PACKS_ROOT / "_toy").resolve()  # a ./toy directory does not shadow the toy pack
    fake = tmp_path / "packsroot"
    for n in ("a", "b"):
        (fake / n).mkdir(parents=True)
        (fake / n / "pack.yaml").write_text("name: twin\n")
    monkeypatch.setattr(pk, "PACKS_ROOT", fake)
    with pytest.raises(ValueError, match="ambiguous"):
        pk.find_pack("twin")


def test_g2_d9_installed_packs_are_found_through_the_registry(tmp_path):
    from crucible.core.registry import get_registry

    pk = _pk()
    root = tmp_path / "site" / "zz_installed_pack"
    root.mkdir(parents=True)
    (root / "pack.yaml").write_text("name: zz_installed\n")
    get_registry().register("pack", "zz_installed", root, source="test", replace=True)
    try:
        assert pk.find_pack("zz_installed") == root.resolve()
    finally:
        get_registry()._kinds["pack"].pop("zz_installed", None)


# ---- bench honours the manifest (V12) ----------------------------------------------------------------------------------
def test_v12_bench_run_uses_the_packs_range_and_threshold(tmp_path):
    pk = _pk()
    d = make_mutant(tmp_path, "ten", {"py": grade_patch("return _NS(reward=10.0 if ok else 0.0, gates={'outcome': 'pass' if ok else 'fail'})")
                                      + expect("lambda c: 10.0 if c['name'] == 'gold' else 0.0"),
                                      "yaml": [(r"(?m)^  range: .*$", "  range: [0.0, 10.0]"), (r"(?m)^  threshold: .*$", "  threshold: 10.0")]})
    assert asyncio.run(pk.check(d, level="semantic")).ok
    r = run_cli("bench", "run", str(d), "--out", str(tmp_path / "runs"), "--fresh", "1", "--k", "1", "--json")
    assert r.exit_code == 0, r.output
    s = json.loads(r.output)["summary"]
    assert s["config"]["threshold"] == 10.0
    lib = make_mutant(tmp_path, "lib", H1["lower_is_better"][0])
    r = CliRunner().invoke(cli, ["bench", "run", str(lib), "--out", str(tmp_path / "runs2"), "--fresh", "0", "--k", "1"])
    assert r.exit_code != 0 and "higher_is_better: false" in r.output


def test_v12_gold_and_noop_check_honours_best_and_threshold():
    from crucible.bench.run import gold_and_noop_check
    from crucible.core.schema.actions import ToolCall

    class T:
        id = "t@1"

        def __init__(self, gold, noop):
            self.r = {"gold": gold, "noop": noop}
            self.oracle = [{"function_name": "x.y", "arguments": {}}]

        async def open(self, d):
            from types import SimpleNamespace

            async def close():
                return None

            w = SimpleNamespace(close=close, acts=[])
            return w

        async def grade(self, w, d):
            from types import SimpleNamespace

            return SimpleNamespace(reward=self.r["gold" if w.acts else "noop"])

    import crucible.actors.policy as pol

    async def fake_episode(w, policies, **kw):
        w.acts = list(policies["agent"].script)

    _, chk = gold_and_noop_check(Path("/tmp/x"), best=10.0, threshold=10.0)
    orig = pol.run_episode
    pol.run_episode = fake_episode
    try:
        assert asyncio.run(chk(T(10.0, 0.0)))[0] is True
        assert asyncio.run(chk(T(1.0, 0.0)))[0] is False  # 1.0 is not the top of [0, 10]
        assert asyncio.run(chk(T(float("nan"), 0.0)))[0] is False
        assert asyncio.run(chk(T(10.0, 10.0)))[0] is False  # the untouched task counts as solved
    finally:
        pol.run_episode = orig
    assert ToolCall  # imported for the action shape


# ---- crucible accept without a forge hook (G2 D3) ----------------------------------------------------------------------
def test_g2_d3_accept_works_without_a_forge_hook(tmp_path):
    d = make_mutant(tmp_path, "acc", {"py": HEAD})
    r = run_cli("accept", str(d), "--out", str(tmp_path / "acc_out"), "--json")
    j = json.loads(r.output)
    assert j["total"] >= 2 and all(v["state"] in ("accepted", "provisional", "rejected") for v in j["tasks"].values())
    assert all("A5" in v["checks"] for v in j["tasks"].values())


# ---- exports are self-contained and cover every task (G2 D11, pick 8 in part) ------------------------------------------
def test_g2_d11_openenv_and_nemo_exports_are_self_contained(tmp_path):
    pk = _pk()
    r = run_cli("export", "--to", "openenv", "--pack", "finance_ap", "--out", str(tmp_path / "oe"), "--json")
    assert r.exit_code == 0, r.output
    mod = (tmp_path / "oe" / "openenv_env.py").read_text()
    assert str(pk.PACKS_ROOT) not in mod and str(pk.REPO_ROOT) not in mod  # no checkout paths (it exported task 0 from absolute paths)
    assert (tmp_path / "oe" / "openenv.yaml").exists() and (tmp_path / "oe" / "pack" / "pack.yaml").exists()
    ns = {"__file__": str(tmp_path / "oe" / "openenv_env.py")}
    exec(mod.split("from crucible.interop.openenv")[0], ns)  # noqa: S102 — the exported factory part (OpenEnv itself is not installed here)
    ids = sorted(ns["tasks"]())
    assert len(ids) == 2 and ns["make_task"](1).id == ids[1] and ns["make_task"](item_seed=5).id not in ids
    r = run_cli("export", "--to", "nemo-gym", "--pack", "toy", "--out", str(tmp_path / "ng"), "--json")
    assert r.exit_code == 0 and str(pk.PACKS_ROOT) not in (tmp_path / "ng" / "nemo_tasks.py").read_text()
