"""The Python profile (swarm-pipeline/07 §3.0): an adapter over the window lane's own behaviour, unchanged. Every member
delegates to the function the lane always used (``executable`` path rules, ``windows_git._defs``, ``rebreak.strip_tests``,
``windows.changed_defs`` / ``suite_targets``, ``windows_world.ImageBook`` / ``leak_probes``, ``windows.sandbox_factory``).
``base.active()`` treats this profile as "no profile", so the lane's predicates never dispatch back here: Python runs
take exactly the code path they took before the language layer existed."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class PythonProfile:
    name = "python"
    lang = "python"
    extensions = (".py",)
    timeout_s = 600.0
    tmpfs = "256m"
    plumbing_words = "conftest.py, sitecustomize.py, usercustomize.py or .pth files"
    install_words = "install packages"

    def detect(self, tree: Path) -> bool:
        return any((Path(tree) / f).is_file() for f in ("pyproject.toml", "setup.py", "setup.cfg"))

    def is_source(self, path: str) -> bool:
        from crucible.factory.lanes.windows_git import is_source

        return is_source(path)

    def is_test_file(self, path: str) -> bool:
        from crucible.factory.lanes.windows_git import is_test_file

        return is_test_file(path)

    def is_test_path(self, path: str) -> bool:
        from crucible.factory.lanes.executable import is_test_path

        return is_test_path(path)

    def is_protected(self, path: str) -> bool:
        from crucible.factory.lanes.executable import is_protected

        return is_protected(path)

    def is_plumbing(self, path: str) -> bool:
        from crucible.factory.pathcheck import PLUMBING

        return bool(PLUMBING.search(path))

    def is_prose(self, path: str) -> bool:
        from crucible.factory.generators.rebreak import PROSE

        return bool(PROSE.search(path))

    def test_defs(self, text: str | None, path: str = "") -> dict[str, str]:
        from crucible.factory.lanes.windows_git import _defs

        return _defs(text)

    def strip_tests(self, text: str, keys: set[str], path: str = "") -> str | None:
        from crucible.factory.generators.rebreak import strip_tests

        return strip_tests(text, keys)

    def changed_defs(self, before: str | None, after: str | None, path: str = "") -> list[str]:
        from crucible.factory.lanes.windows import changed_defs

        return changed_defs(before, after)

    def key_of_test(self, test_id: str) -> tuple[str, str]:
        from crucible.factory.lanes.windows_git import key_of_test

        return key_of_test(test_id)

    def suite_targets(self, tree: Path) -> list[str]:
        from crucible.factory.lanes.windows import suite_targets

        return suite_targets(Path(tree), {})

    def leaf(self, key: str) -> str:
        return key.rsplit("::", 1)[-1]

    def new_test_names(self, tests_b: Mapping[str, str], a_tests: Mapping[str, str | None], tree: Path) -> set[str]:
        names: set[str] = set()
        for p, txt in tests_b.items():
            before = set(self.test_defs(a_tests.get(p)))
            names |= {q.rsplit("::", 1)[-1] for q in set(self.test_defs(txt)) - before}
        a_text = "\n".join(f.read_text(errors="replace") for f in Path(tree).rglob("*.py") if f.is_file())
        return {n for n in names if n.startswith("test") and n not in a_text}

    def images(self, **kw: Any) -> Any:
        from crucible.factory.lanes.windows_world import ImageBook

        return ImageBook(**kw)

    async def leak_probes(self, image: str, base_dir: Path, **kw: Any) -> dict[str, Any]:
        from crucible.factory.lanes.windows_world import leak_probes

        return await leak_probes(image, base_dir, **kw)

    def world_setup_cmd(self, ws: str) -> str:
        return ""


PROFILE = PythonProfile()
