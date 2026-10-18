"""P4 World fixes (swarm-pipeline/00 §4 S3, §8.3 ``world_bug``): recipe 3's World needs (build tools, the repository's
own build and test commands, declared extras, known optional imports), the executed version search, the World of a task
version, and S3 on a real image (Docker)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from test_m17_w_task import make_task

from crucible.factory.lanes.windows_s3 import implicated_modules, version_ladder, world_of_version, world_s3
from crucible.factory.lanes.windows_world import (
    DEV_TOOLS_APT,
    IMAGE_RECIPE,
    ImageBook,
    declared_extras,
    requirement_pins,
    shell_commands,
    world_dockerfile,
    world_needs,
)
from crucible.factory.traceaudit import Proposal, TaskContext, version_accepted
from crucible.factory.versioning import VersionStore


def _tree(tmp: Path) -> Path:
    t = tmp / "a"
    (t / "pkg").mkdir(parents=True)
    (t / "pkg" / "__init__.py").write_text("")
    (t / "pkg" / "cli.py").write_text("def load(text):\n    import yaml\n    return yaml.safe_load(text)\n")
    (t / "tests").mkdir()
    (t / "tests" / "helpers.py").write_text("X = 1\n")
    (t / "tests" / "test_cli.py").write_text("import helpers\nimport frobnicate_xyz\nimport subprocess\n\n"
                                             "def test_x():\n    subprocess.check_output(['msgfmt', '-o', 'a.mo', 'a.po'])\n")
    (t / "Makefile").write_text("test:\n\t$(MAKE) build && pytest -q\nbuild:\n\tgcc -o x x.c\n")
    (t / "tox.ini").write_text("[tox]\nenvlist = py38\n\n[testenv]\ndeps = pytest\ncommands =\n    pytest {posargs}\n    sqlite3 --version\n")
    (t / "setup.py").write_text("from setuptools import setup\nsetup(name='pkg', extras_require={'yaml': ['PyYAML'], 'docs': ['sphinx']})\n")
    (t / "requirements.txt").write_text("PyYAML==3.13\nattrs>=17\n")
    return t


def test_world_needs_reads_the_repository_not_guesses(tmp_path: Path) -> None:
    n = world_needs(_tree(tmp_path))
    assert set(DEV_TOOLS_APT) <= set(n["apt"]) and {"gcc", "gettext", "sqlite3"} <= set(n["apt"])
    assert n["why"]["apt"]["gcc"] == ["gcc in Makefile"] and n["why"]["apt"]["gettext"] == ["msgfmt in tests/test_cli.py"]
    assert n["pip_build"] == ["setuptools", "wheel"] and n["extras"] == ["yaml"]  # docs extras are not installed
    assert n["import_deps"] == ["PyYAML"]  # a known provider only: the local helper and an unknown name are never installed
    assert "frobnicate_xyz" in n["unresolved"]["imports"] and "helpers" not in n["unresolved"]["imports"]
    assert n["pins"] == {"PyYAML": "3.13"} and requirement_pins("a==1.0\nb>=2\npytest==8\nc[x]==3 ; python_version<'4'\n") == {"a": "1.0", "c": "3"}
    assert declared_extras(tmp_path / "a") == ["yaml"]
    assert shell_commands("run: |\n  make test\n  ./x.sh\nother: zip\n", ".github/workflows/t.yml") == {"make"}
    assert shell_commands("\tFOO=1 sudo env bc -l && cd x; hexdump -C f | head\n", "Makefile") == {"bc", "cd", "hexdump", "head"}
    df = world_dockerfile("pkg", ["PyYAML"], [], needs=n)
    assert "apt-get install" in df and " gcc " in df and "pip install --no-cache-dir setuptools wheel" in df and ".[test,tests,testing,yaml]" in df
    assert df.index("apt-get") < df.index("COPY a") and df.index("pytest==8.3.3") > df.index("PyYAML")


def test_recipe_3_tags_change_with_what_the_build_installs(tmp_path: Path) -> None:
    t = _tree(tmp_path)
    tag, info = ImageBook().tag_for("pkg", t)
    assert IMAGE_RECIPE == "3" and info["recipe"] == "3" and info["needs"]["apt"]
    (t / "scripts").mkdir()
    (t / "scripts" / "x.sh").write_text("jq . a.json\n")
    tag2, info2 = ImageBook().tag_for("pkg", t)
    assert tag2 != tag and "jq" in info2["needs"]["apt"]


def test_version_ladder_and_implicated_modules() -> None:
    vs = ["6.0.3", "6.0.2", "6.0", "5.4.1", "5.4", "5.3.1", "5.1.2", "5.1", "4.2b4", "3.13", "3.12"]
    assert version_ladder(vs, "3.13", current="6.0.3") == ["3.13", "5.1.2", "5.3.1", "5.4.1"]  # newest patch per minor, from the pin up
    assert version_ladder(vs, None, current="6.0.3")[:2] == ["5.4.1", "5.3.1"]
    tail = ("            try:\n>                   target = yaml.load(target_text)\nE                   TypeError: load() missing 1 required positional argument\n"
            "/usr/local/lib/python3.12/site-packages/face/command.py:478: in run\n>       raise CommandLineError(msg)\n"
            "E       face.errors.CommandLineError: could not load\n")
    assert implicated_modules(tail) == ["yaml", "face"]


def test_world_of_version_picks_the_head_world_fix(tmp_path: Path) -> None:
    rec = {"id": "w1", "world": {"image": "crucible-lh-w:old", "digest": "sha256:a"}}
    vs = VersionStore(tmp_path / "v.jsonl")
    assert world_of_version(rec, vs) == rec
    p = Proposal(kind="world_bug", task="w1", traces=[], artifact={"type": "repro"}, change={"world_fix": {"type": "world_image", "image": "crucible-lh-w:new",
                                                                                                      "digest": "sha256:b", "recipe": "3"}},
                 source="fact", why="setuptools missing")
    p.status, p.evidence = "accepted", {"before_fix_on_oracle": {"failed": True}, "after_fix_on_oracle": {"failed": False}, "s3": {"ok": True}}
    v = version_accepted(p, vs, TaskContext("w1", prompt=""))
    got = world_of_version(rec, vs)
    assert got["world"]["image"] == "crucible-lh-w:new" and got["world"]["version"] == v.id and rec["world"]["image"] == "crucible-lh-w:old"
    assert vs.verify() == [] and v.change["type"] == "world_fix"


# ---- Docker ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.docker
def test_s3_on_the_toy_world_image(tmp_path: Path) -> None:
    task, rec, _ = make_task(tmp_path)
    card = asyncio.run(ImageBook().ensure("toyrepo", task.base_dir))
    assert card["ok"], card.get("build_log")
    s3 = asyncio.run(world_s3({**rec, "b_version": "0.2"}, task.base_dir, card["image"], reruns=4))
    assert s3["checks"]["builds"] and s3["checks"]["existing_tests_run"] and s3["checks"]["leak_probes_hold"] and s3["checks"]["grading_unchanged"]
    assert s3["reruns"]["n"] == 4 and s3["reruns"]["clean"] == 4 and not s3["checks"]["reruns_clean"]  # 4 clean runs cannot bound 5%
    assert s3["grading"]["oracle_milestones"] == {"m01": True, "m02": True, "m03": True}
