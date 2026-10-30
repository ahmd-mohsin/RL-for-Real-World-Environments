"""Part 1 V1 (2026-10-30, docs/research/env-synthesis/part1-framework-improvements.md §2(c)): side-effect negatives, the
work done plus an unrequested change. The debug pool's falsifiers (``synth.falsify.patch.PatchMUT``) had the inert forms
(an edit to a test or config must fail, a comment in untouched source must pass) but no behaviour change, so a fix that
also broke a neighbouring function kept the full reward wherever the hidden tests did not reach it, and nothing measured
that. ``SideEffect`` is B plus one flipped comparison or boolean literal away from B's changes; a survivor is a suspect D2
(LAX-FRAME: unrequested effects pass)."""

from __future__ import annotations

import asyncio

from crucible.synth.falsify.base import falsify
from crucible.synth.falsify.patch import PatchMUT, changed_lines, flip, flip_sites
from crucible.synth.patch import PatchEnv
from tests.unit.synth_patch_fixtures import fixture_runner, make_repo, spec_for

# add and mul are the two bugs (lines 2 and 6); the neighbour sits well clear of both (the fixture's hidden tests call
# add, mul and ident)
TEMPLATE = "def add(a, b):\n    return a {add}\n\n\ndef mul(a, b):\n    return a {mul}\n\n\n\n\n\n\n{neighbour}\n"
COVERED = "def ident(x):\n    return x if True else None"  # test_ident runs it: a flip fails the regression check
BLIND = "def ident(x):\n    return x\n\n\ndef is_zero(x):\n    return x == 0"  # no hidden test calls is_zero


def _side_effects(tmp_path, neighbour: str):
    base = make_repo(tmp_path)
    (base / "calc.py").write_text(TEMPLATE.format(add="- b", mul="+ b", neighbour=neighbour))
    fixed = TEMPLATE.format(add="+ b", mul="* b", neighbour=neighbour)
    env = PatchEnv(spec_for(base, oracle_files={"calc.py": fixed}, mutants=[]))
    res = asyncio.run(falsify(env.task(), PatchMUT(env), env.context(fixture_runner(env))))
    return [f for f in res.findings if f.op == "SideEffect"], res


def test_a_flip_the_hidden_tests_reach_is_rejected(tmp_path) -> None:
    se, res = _side_effects(tmp_path, COVERED)
    assert [(f.site, f.label) for f in se] == [("calc.py:14", "clean")]  # before: no SideEffect probe at all
    assert not [f for f in res.suspects if f.op == "SideEffect"]


def test_a_flip_no_hidden_test_reaches_is_a_suspect_lax_frame(tmp_path) -> None:
    se, res = _side_effects(tmp_path, BLIND)
    assert [(f.site, f.label, f.defect) for f in se] == [("calc.py:18", "suspect", "D2")]  # the fix plus a broken is_zero scores 1
    assert [f.op for f in res.suspects] == ["SideEffect"]  # an open suspect: the certificate waits for triage


def test_sites_skip_comments_strings_and_the_fixs_own_lines() -> None:
    py = 'x = a == b  # c == d\ns = "e == f"\nok = True\n'
    assert [(ln, t) for ln, _, t in flip_sites(py, "python")] == [(0, "=="), (2, "True")]
    go = 'if a == b { // c != d\n\tx := "e == f"\n\treturn false\n}\n'
    assert [(ln, t) for ln, _, t in flip_sites(go, "go")] == [(0, "=="), (2, "false")]
    js = "const same = a === b;\n/* a == b */\n * a != b\n"
    assert [(ln, t) for ln, _, t in flip_sites(js, "node")] == [(0, "===")]
    assert flip(js, 0, 15, "===") == "const same = a !== b;\n/* a == b */\n * a != b\n"
    assert flip(js, 0, 3, "===") is None  # a site that does not hold the token is skipped, never a wrong edit
    old, new = "a\nb\nc\nd\ne\nf\ng\nh\ni\n", "a\nb\nc\nd\nE\nf\ng\nh\ni\n"
    assert changed_lines(old, new) == {1, 2, 3, 4, 5, 6, 7}  # line 4 changed, 3 lines either side
    assert flip_sites("x == 1\ny == 2\n", "python", skip={0}) == [(1, 2, "==")]
