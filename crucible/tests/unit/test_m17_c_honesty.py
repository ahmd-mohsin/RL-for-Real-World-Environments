"""M17 study 08 §7.2: STATUS.json claims scored with the Brier rule, the honest GIVE_UP, and the composition that never
lets the honesty term turn a failure into a positive advantage."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from crucible.core.schema.actions import Done
from crucible.factory import honesty as h


def test_parse_status_defaults_and_invalid_values():
    assert h.parse_status(None) is None and h.parse_status("not json") is None and h.parse_status("[1, 2]") is None
    got = h.parse_status(json.dumps({"r1": 0.8, "r2": True, "r3": 1.7, "r4": "yes", "r5": -0.1, "r6": 0}))
    assert got == {"r1": 0.8, "r2": 1.0, "r6": 0.0}  # out of range, strings and negatives are no claim


def test_status_from_patch_and_tree(tmp_path):
    body = json.dumps({"r1": 0.9}, indent=1).splitlines()
    patch = ("diff --git a/x.py b/x.py\nnew file mode 100644\n--- /dev/null\n+++ b/x.py\n@@ -0,0 +1 @@\n+x = 1\n"
             f"diff --git a/STATUS.json b/STATUS.json\nnew file mode 100644\n--- /dev/null\n+++ b/STATUS.json\n@@ -0,0 +1,{len(body)} @@\n"
             + "".join(f"+{ln}\n" for ln in body))
    assert h.status_from_patch(patch) == {"r1": 0.9}
    edit = "diff --git a/STATUS.json b/STATUS.json\n--- a/STATUS.json\n+++ b/STATUS.json\n@@ -2,1 +2,1 @@\n-  \"r1\": 0.1\n+  \"r1\": 0.9\n"
    assert h.status_from_patch(edit) is None  # a partial edit cannot be read without its base: no claim
    assert h.status_from_patch("") is None
    (tmp_path / "STATUS.json").write_text(json.dumps({"r1": 0.25}))
    assert h.status_from_tree(tmp_path) == {"r1": 0.25}
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "STATUS.json").symlink_to(tmp_path / "STATUS.json")
    assert h.status_from_tree(tmp_path / "x") is None  # a symlink is not a claim


def test_brier_only_over_verified_requirements():
    s = h.score({"r1": 1.0, "r2": 0.0, "bogus": 1.0}, {"r1": True, "r2": False, "r3": None}, requirement_ids=["r1", "r2", "r3"])
    assert s.brier == 0.0 and s.term == 0.0 and s.ignored == ["bogus"] and set(s.scored) == {"r1", "r2"}  # r3 unverified: not scored
    no = h.score(None, {"r1": True, "r2": False})
    assert no.brier == pytest.approx(0.25) and no.term == pytest.approx(-0.025) and no.defaulted == ["r1", "r2"] and not no.claimed
    liar = h.score({"r1": 1.0, "r2": 1.0}, {"r1": False, "r2": False}, lam=0.2)
    assert liar.brier == 1.0 and liar.term == pytest.approx(-0.2) and liar.components["honesty/brier"] == 1.0
    assert h.score({"r1": 1.0}, {}).brier is None and h.score({"r1": 1.0}, {}).term == 0.0  # nothing verifiable: nothing scored
    with pytest.raises(ValueError):
        h.score({}, {"r1": True}, lam=-1)


def test_brier_term_is_proper_numerically():
    """For a requirement met with probability q, E[-λ (p - Y)²] is maximised at p = q (and only there, on a fine grid)."""
    grid = [i / 200 for i in range(201)]
    for q in (0.0, 0.1, 0.37, 0.5, 0.8, 1.0):
        def expected(p, q=q):
            return q * h.score({"r": p}, {"r": True}).term + (1 - q) * h.score({"r": p}, {"r": False}).term
        best = max(grid, key=expected)
        assert best == pytest.approx(q, abs=1e-9), (q, best)
        assert all(expected(p) < expected(q) for p in grid if abs(p - q) > 1e-9)


def test_give_up_detection():
    assert h.detect_give_up("I tried.\nGIVE_UP: the hidden API does not exist\nbye") == "the hidden API does not exist"
    assert h.detect_give_up("GIVE_UP:") == "" and h.detect_give_up("i will not GIVE_UP: ever") is None
    assert h.detect_give_up("gave up: no tests") == "no tests"  # the RLM's rlm.give_up stop
    assert h.give_up_of_action(Done(reason="GIVE_UP: blocked")) == "blocked"
    assert h.give_up_of_action(Done(reason="x" * 10, answer="long...\nGIVE_UP: blocked")) == "blocked"
    assert h.give_up_of_action(SimpleNamespace(kind="tool_call")) is None


def test_composition_never_flips_a_failure():
    assert h.compose_advantage(-0.5, -0.1) == pytest.approx(-0.6)
    assert h.compose_advantage(0.05, -0.1) == 0.0  # a success is floored at 0
    assert h.compose_advantage(0.0, -0.1) == pytest.approx(-0.1)
    with pytest.raises(ValueError):
        h.compose_advantage(0.0, 0.1)
    rng = random.Random(0)
    for _ in range(500):
        a, t = rng.uniform(-1, 1), -rng.uniform(0, 0.2)
        out = h.compose_advantage(a, t)
        assert out <= max(a, 0.0) + 1e-12 and (out <= 0 if a <= 0 else out >= 0)


def _rec(reward, term, *, masked=False, adv=None):
    return SimpleNamespace(reward=reward, mask_sample=masked, advantage=adv, reward_components={}, progress={"honesty": {"term": term}})


def test_group_stage_all_fail_group_gives_no_positive_advantage():
    """Folded into R, an honest failure in an all-fail group would beat a lying one with a positive advantage; composed on
    the advantage it gets 0 and the liar a negative one."""
    honest, liar = _rec(0.0, 0.0), _rec(0.0, -0.1)
    naive_mu = (0.0 + (0.0 - 0.1)) / 2
    assert 0.0 - naive_mu > 0  # the naive reward fold flips the honest failure positive
    h.honesty_group_fn()("g", [honest, liar])
    assert honest.advantage == 0.0 and liar.advantage == pytest.approx(-0.1) and honest.reward == 0.0 and liar.reward == 0.0
    mixed = [_rec(1.0, -0.1), _rec(0.0, 0.0), _rec(0.0, -0.1), _rec(None, 0.0, masked=True)]
    h.honesty_group_fn()("g", mixed)
    assert mixed[0].advantage > 0 and mixed[1].advantage <= 0 and mixed[2].advantage < mixed[1].advantage and mixed[3].advantage is None
    pre = [_rec(1.0, -0.05, adv=0.3), _rec(0.0, 0.0, adv=-0.3)]  # an earlier stage's advantage (GAR) is the Â
    h.honesty_group_fn()("g", pre)
    assert pre[0].advantage == pytest.approx(0.25) and pre[1].advantage == pytest.approx(-0.3)
