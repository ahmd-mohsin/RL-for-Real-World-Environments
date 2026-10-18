"""P5 stitched windows (swarm-pipeline/00 §2): requirement bumps, the host release window, the consumer window, the
two-repository workspace as one synthetic history, the per-side sandbox and the World recipe that keeps B out of the image."""

from __future__ import annotations

import asyncio
from pathlib import Path

from crucible.factory.generators.base import Funnel, git, out, tree_of
from crucible.factory.generators.stitch import (
    _SIDE,
    PYTHONPATH,
    Stitch,
    StitchSandbox,
    bounds,
    consumer_window,
    discover_pairs,
    host_roots,
    host_window,
    imports_host,
    requirement_bumps,
    spec_of,
    stitch_dockerfile,
)
from crucible.factory.sources.lease import EvalLease


def _commit(repo: Path, msg: str, files: dict[str, str], date: str, tag: str | None = None) -> str:
    for p, t in files.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(t)
    env = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg, env=env)
    if tag:
        git(repo, "tag", "-a", tag, "-m", tag, env=env)
    return out(git(repo, "rev-parse", "HEAD"))


def make_pair(tmp: Path) -> tuple[Path, Path, dict[str, str]]:
    h, c = tmp / "clones" / "hostlib", tmp / "clones" / "consumerapp"
    for r in (h, c):
        r.mkdir(parents=True)
        git(r, "init", "-q", "-b", "main")
    s: dict[str, str] = {}
    s["h10"] = _commit(h, "init", {"pyproject.toml": "[project]\nname = 'hostlib'\n", "hostlib/__init__.py": "def greet(n):\n    return 'hi ' + n\n",
                                   "tests/test_h.py": "from hostlib import greet\n\ndef test_greet():\n    assert greet('a') == 'hi a'\n"},
                       "2023-01-01T00:00:00+00:00", "1.0")
    s["h11"] = _commit(h, "docs", {"README.md": "hostlib\n"}, "2023-02-01T00:00:00+00:00", "1.1")
    s["h_loud"] = _commit(h, "Add loud greetings", {"hostlib/__init__.py": "def greet(n, loud=False):\n    s = 'hi ' + n\n    return s.upper() if loud else s\n",
                                                    "tests/test_h.py": "from hostlib import greet\n\ndef test_greet():\n    assert greet('a') == 'hi a'\n\n"
                                                                       "def test_loud():\n    assert greet('a', loud=True) == 'HI A'\n"},
                          "2023-03-01T00:00:00+00:00", "2.0")
    s["c0"] = _commit(c, "init", {"pyproject.toml": "[project]\nname = 'consumerapp'\ndependencies = ['hostlib>=1.0']\n",
                                  "consumerapp/__init__.py": "from hostlib import greet\n\ndef hello(n):\n    return greet(n)\n",
                                  "tests/test_c.py": "from consumerapp import hello\n\ndef test_hello():\n    assert hello('b') == 'hi b'\n"},
                      "2023-01-05T00:00:00+00:00")
    s["c_dev"] = _commit(c, "pin test tools", {"requirements-dev.txt": "pytest==8.0\n"}, "2023-02-05T00:00:00+00:00")
    s["c_bump"] = _commit(c, "Shout hellos with hostlib 2", {"pyproject.toml": "[project]\nname = 'consumerapp'\ndependencies = ['hostlib>=2.0']\n",
                                                             "consumerapp/__init__.py": "from hostlib import greet\n\ndef hello(n, loud=False):\n    return greet(n, loud=loud)\n",
                                                             "tests/test_c.py": "from consumerapp import hello\n\ndef test_hello():\n    assert hello('b') == 'hi b'\n\n"
                                                                                "def test_shout():\n    assert hello('b', loud=True) == 'HI B'\n"},
                          "2023-03-05T00:00:00+00:00")
    return h, c, s


def test_requirement_specs_and_bumps(tmp_path: Path) -> None:
    assert spec_of('dependencies = ["Host-Lib >= 1.2, <3"]', "host_lib") == ">=1.2,<3"
    assert spec_of("install_requires=['hostlib~=2.1']", "hostlib") == "~=2.1" and spec_of("hostlibx>=1", "hostlib") is None
    lo, hi = bounds(">=1.2,<3")
    assert str(lo) == "1.2" and str(hi) == "3"
    h, c, s = make_pair(tmp_path)
    bumps = requirement_bumps(c, "hostlib", since="2020-01-01")
    assert [(b["sha"], b["old"], b["new"], b["target"]) for b in bumps] == [(s["c_bump"], ">=1.0", ">=2.0", "2.0")]
    hw = host_window(h, "2.0")
    assert hw is not None and (hw["a_ref"], hw["b_ref"], hw["qualifying"]) == ("1.1", "2.0", 1)
    assert host_window(h, "9.9") is None
    assert consumer_window(c, bumps[0]) == {"c0": s["c_dev"], "c1": s["c_bump"], "bump": s["c_bump"]}
    assert host_roots(h) == ["hostlib"] and imports_host(c, s["c_bump"], ["hostlib"]) == ["consumerapp/__init__.py"]
    assert imports_host(c, s["c_bump"], ["other"]) == []
    assert discover_pairs({"hostlib": h, "consumerapp": c}, since="2020-01-01") == [("consumerapp", "hostlib")]


def test_the_workspace_is_one_history_of_both_repositories(tmp_path: Path) -> None:
    h, c, s = make_pair(tmp_path)
    gen = Stitch(tmp_path / "st", tmp_path / "clones", funnel=Funnel(tmp_path / "f.jsonl"), lease=EvalLease.default(), since="2020-01-01")
    cands = asyncio.run(gen.candidates([("consumerapp", "hostlib")]))
    assert len(cands) == 1 and cands[0].params["host_window"]["b_ref"] == "2.0" and cands[0].sources[0]["type"] == "consumer"
    win, info = gen.synthesize(cands[0])
    mono = gen.monos[cands[0].repo]
    assert [info["unit_side"][u.sha] for u in win.units] == ["host", "consumer"]
    assert out(git(mono, "show", f"{win.a}:host/hostlib/__init__.py")).startswith("def greet(n):")
    assert "loud" not in out(git(mono, "show", f"{win.a}:consumer/consumerapp/__init__.py"))
    assert out(git(mono, "rev-parse", f"{win.b}:host")) == tree_of(h, s["h_loud"])
    assert out(git(mono, "rev-parse", f"{win.b}:consumer")) == tree_of(c, s["c_bump"])
    assert {p for u in win.units for p in u.files} >= {"host/hostlib/__init__.py", "consumer/consumerapp/__init__.py", "consumer/tests/test_c.py"}
    assert all(u.qualifies for u in win.units)
    assert gen.images.host_versions[cands[0].repo] == "1.1" and (tmp_path / "st" / "pairs.json").exists()
    assert out(git(h, "for-each-ref", "refs/crucible")) == "" and out(git(c, "for-each-ref", "refs/crucible")) == ""  # sources never written


def test_the_sandbox_runs_each_side_from_its_own_root() -> None:
    sb = StitchSandbox("img", guard=True, name_prefix="crucible-p5-")
    tok = _SIDE.set("consumer")
    try:
        cmd = sb.command(Path("/tmp/t"), "crucible-m17-x", "abc", ["tests/test_c.py"])
    finally:
        _SIDE.reset(tok)
    assert cmd[cmd.index("-w") + 1] == "/w/consumer" and f"PYTHONPATH={PYTHONPATH}:/opt/m17" in cmd and "crucible-p5-abc" in cmd
    plain = sb.command(Path("/tmp/t"), "crucible-m17-x", "abc", ["x"])
    assert plain[plain.index("-w") + 1] == "/w"


def test_the_world_recipe_pins_the_host_at_a() -> None:
    needs = {"host": {"apt": ["make"], "pip_build": ["setuptools"], "extras": ["yaml"], "import_deps": []},
             "consumer": {"apt": ["patch"], "pip_build": ["wheel"], "extras": [], "import_deps": ["PyYAML"]}}
    df = stitch_dockerfile("hostlib", "1.1", needs, {"host": [], "consumer": ["hostlib", "rich"]}, {"host": [], "consumer": []},
                           {"host": True, "consumer": True})
    assert "SETUPTOOLS_SCM_PRETEND_VERSION=1.1" in df and "ENV PIP_CONSTRAINT=/opt/lh/cons.txt" in df
    assert df.index("COPY a/host") < df.index("pip install --no-cache-dir \".[test,tests,testing,yaml]\"") < df.index("WORKDIR /opt/lh/c")
    assert "install --no-cache-dir hostlib " not in df and "rich" in df  # the consumer's own requirement on the host is never installed from PyPI
    assert "assert v == open(" in df and "/opt/lh/cons.txt" in df and "make patch" in df
