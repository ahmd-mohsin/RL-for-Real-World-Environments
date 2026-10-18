"""The window lane, World half (study 08 §1, §3; 09): the repository at A in a Docker CPU sandbox, its leak probes, and
the policy's World.

**Image per (repo, A).** ``crucible-lh-base:py312`` carries Python 3.12, git, the test runner and the grading guard
(``executable.GUARD_PLUGIN``). A World image adds A's own dependencies, installed at build time from A's packaging
(``pip install <A>[test,tests,testing]``, PEP 735 test groups, and the names in A's test requirement files). The
project stays installed for its metadata (A's version); ``PYTHONPATH`` puts the workspace first, so every import is the
agent's code. Images are keyed by the recipe and the packaging files' content with version strings blanked, so releases
with the same dependencies share one image. Every image and container
this module makes is named ``crucible-lh-*``.

**Leak probes** (executable, in the image, ``--network none``): ``pip download <pkg>==<B>`` fails; no file named like
B's wheel or sdist exists (the dist-info of A's own install is A's); no file in the image or the seeded workspace defines a new test of the window (``def <name>(``); the seeded workspace's
``git log --all`` shows one commit, tagged ``base``.

**The policy's World.** A ``common/git_repo`` App seeded with A (fresh ``git init``, one commit tagged ``base``) on the
host; its shell, file and git tools run through ``docker exec`` in a ``--network none`` container of the World image
that bind-mounts the workspace at the same absolute path, so ``WorkspaceHarness`` collects the patch from the host tree.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
import warnings
from pathlib import Path
from typing import Any

from crucible.factory.lanes.executable import GUARD_PLUGIN
from crucible.factory.lanes.windows_git import packaging_hash

PREFIX = "crucible-lh-"
BASE_IMAGE = "crucible-lh-base:py312"
BASE_PINS = ("pytest==8.3.3", "pytest-timeout==2.3.1")
BASE_DOCKERFILE = f"""FROM python:3.12-slim
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir {' '.join(BASE_PINS)} hypothesis==6.112.1 freezegun==1.5.1 pytest-mock==3.14.0 pytz simplejson typing_extensions
COPY m17guard.py /opt/m17/m17guard.py
RUN git config --system safe.directory '*' && git config --system user.email agent@sandbox && git config --system user.name agent
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /w
"""
PACKAGING = ("pyproject.toml", "setup.py", "setup.cfg")
# part of every World image tag: a changed recipe never reuses an image built by the old one. Recipe 3 (P4) adds the
# World needs (``world_needs``): the build tools python:3.12-slim lacks, the repository's own build and test tools, its
# declared extras and the optional imports its tests reach (P3 reproduced 7 World bugs on the oracle: setuptools x3,
# make x2, hexdump, PyYAML)
IMAGE_RECIPE = "3"
REQ_FILE = re.compile(r"(^|/)(requirements[-_/]?)?(test|tests|testing|dev)[^/]*\.(txt|in)$|(^|/)requirements[-_](test|tests|testing|dev)[^/]*\.(txt|in)$"
                      r"|^requirements\.txt$", re.IGNORECASE)  # a root requirements.txt: python-dotenv lists its test deps (click, sh) there
EXTRAS = ("test", "tests", "testing")


def sha256_hex(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def docker(*args: str, timeout: float = 1800, input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, check=False, input=input)


def image_id(tag: str) -> str | None:
    r = docker("image", "inspect", "--format", "{{.Id}}", tag, timeout=60)
    return r.stdout.strip() if r.returncode == 0 else None


def ensure_base() -> str:
    d = image_id(BASE_IMAGE)
    if d:
        return d
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "Dockerfile").write_text(BASE_DOCKERFILE)
        (Path(td) / "m17guard.py").write_text(GUARD_PLUGIN)
        r = docker("build", "-q", "-t", BASE_IMAGE, td)
        if r.returncode != 0:
            raise RuntimeError(f"base image build failed: {r.stderr[-1500:]}")
    return image_id(BASE_IMAGE) or ""


# ---- packaging ---------------------------------------------------------------------------------------------------------
def project_name(tree: Path) -> str | None:
    """The distribution name from ``pyproject.toml``, ``setup.cfg`` or ``setup.py``."""
    pp = tree / "pyproject.toml"
    if pp.exists():
        with contextlib.suppress(tomllib.TOMLDecodeError, OSError):
            d = tomllib.loads(pp.read_text())
            n = (d.get("project") or {}).get("name") or ((d.get("tool") or {}).get("poetry") or {}).get("name")
            if n:
                return str(n)
    cfg = tree / "setup.cfg"
    if cfg.exists():
        m = re.search(r"(?m)^\s*name\s*=\s*([\w.-]+)\s*$", cfg.read_text(errors="replace"))
        if m:
            return m.group(1)
    sp = tree / "setup.py"
    if sp.exists():
        m = re.search(r"""\bname\s*=\s*["']([\w.-]+)["']""", sp.read_text(errors="replace"))
        if m:
            return m.group(1)
    return None


def requirement_sources(tree: Path) -> tuple[list[str], list[str]]:
    """``(requirement files, PEP 735 group requirements)`` that install A's test dependencies."""
    files = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file() and p.stat().st_size < 50_000
                   and REQ_FILE.search(p.relative_to(tree).as_posix()) and ".git/" not in p.as_posix() and "/docs/" not in "/" + p.relative_to(tree).as_posix())
    groups: list[str] = []
    pp = tree / "pyproject.toml"
    if pp.exists():
        with contextlib.suppress(tomllib.TOMLDecodeError, OSError):
            dg = tomllib.loads(pp.read_text()).get("dependency-groups") or {}
            for k in ("test", "tests", "testing", "dev"):
                groups += [x for x in dg.get(k) or [] if isinstance(x, str)]
    return files, groups


def requirement_names(text: str) -> list[str]:
    """Requirement names with their pins dropped (old pins rarely install on 3.12); options, paths and URLs skipped."""
    out = []
    for ln in text.splitlines():
        s = ln.split("#", 1)[0].strip()
        if not s or s.startswith(("-", ".", "/", "git+", "http")) or "://" in s:
            continue
        m = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?", s)
        if m and m.group(1).lower() not in ("pytest", "pytest-timeout"):
            out.append(m.group(1) + (m.group(2) or ""))
    return out


# ---- World needs (recipe 3, P4) -------------------------------------------------------------------------------------------
# A developer's workstation baseline the slim image lacks: ``make``, ``hexdump``/``column`` (bsdextrautils), ``xxd``,
# ``patch``, ``file``. Python >= 3.12 environments no longer carry setuptools or wheel, so ``python setup.py ...`` and
# ``import setuptools`` (a repository's own setup.py) fail without them.
DEV_TOOLS_APT = ("make", "bsdextrautils", "xxd", "patch", "file")
PY_BUILD = ("setuptools", "wheel")
# a command a repository's build or test configuration runs -> the Debian package that provides it (only these are ever
# installed: an unknown command is recorded as unresolved, never guessed)
TOOL_APT = {"make": "make", "gmake": "make", "hexdump": "bsdextrautils", "column": "bsdextrautils", "xxd": "xxd", "patch": "patch",
            "file": "file", "gcc": "gcc", "cc": "gcc", "g++": "g++", "c++": "g++", "pkg-config": "pkg-config", "zip": "zip", "unzip": "unzip",
            "bc": "bc", "msgfmt": "gettext", "xgettext": "gettext", "sqlite3": "sqlite3", "jq": "jq", "rsync": "rsync", "tree": "tree",
            "dot": "graphviz", "locale-gen": "locales", "tzdata": "tzdata", "less": "less", "diff": "diffutils", "cmp": "diffutils",
            "xz": "xz-utils", "bzip2": "bzip2", "gzip": "gzip", "tar": "tar", "perl": "perl", "curl": "curl", "wget": "wget"}
# commands never installed for the World: network clients (the World is offline) and what the image already has
NO_INSTALL = {"curl", "wget", "tar", "gzip", "diff", "cmp", "perl", "tzdata"}
# an import a repository's tests or modules make -> the distribution that provides it (only these are ever pip-installed:
# an unknown import name could be a local helper or a squatted package, so it is recorded, never installed)
MODULE_DIST = {"yaml": "PyYAML", "toml": "toml", "tomli": "tomli", "tomli_w": "tomli-w", "dateutil": "python-dateutil", "attr": "attrs",
               "attrs": "attrs", "six": "six", "pytz": "pytz", "simplejson": "simplejson", "ujson": "ujson", "orjson": "orjson",
               "docutils": "docutils", "pygments": "Pygments", "markdown": "Markdown", "babel": "Babel", "mock": "mock", "freezegun": "freezegun",
               "hypothesis": "hypothesis", "typing_extensions": "typing_extensions", "dataclasses_json": "dataclasses-json",
               "msgpack": "msgpack", "regex": "regex", "colorama": "colorama", "termcolor": "termcolor", "wcwidth": "wcwidth",
               "packaging": "packaging", "jinja2": "Jinja2", "markupsafe": "MarkupSafe", "click": "click", "tabulate": "tabulate",
               "face": "face", "boltons": "boltons", "attrs_strict": "attrs-strict", "pytest_mock": "pytest-mock", "tzdata": "tzdata",
               "zoneinfo_backports": "backports.zoneinfo", "importlib_metadata": "importlib-metadata", "jsonschema": "jsonschema",
               "atomicwrites": "atomicwrites", "sortedcontainers": "sortedcontainers", "decorator": "decorator", "wrapt": "wrapt"}
DOC_EXTRAS = {"doc", "docs", "documentation", "rtd", "sphinx"}
_CONFIG = re.compile(r"(^|/)(GNU)?[Mm]akefile$|(^|/)tox\.ini$|(^|/)noxfile\.py$|(^|/)setup\.cfg$|^\.github/workflows/[^/]+\.ya?ml$|^\.travis\.yml$"
                     r"|^\.gitlab-ci\.yml$|^(azure-pipelines|appveyor)\.ya?ml$|^(scripts?|tools|bin|ci)/[^/]+\.sh$|^[^/]+\.sh$")
_SUBPROC = re.compile(r"""(?:subprocess\.\w+|os\.system|os\.popen|check_output|check_call)\(\s*\[?\s*(?:r|b)?['"]([\w.+-]+)""")


def _command_words(line: str) -> set[str]:
    """The first word of every simple command on one shell line (after ``VAR=x`` assignments and ``sudo``/``env``/``exec``)."""
    out: set[str] = set()
    ln = line.split(" #", 1)[0].strip().lstrip("-@+ \t")
    if "$(MAKE)" in ln or "${MAKE}" in ln:
        out.add("make")
    for part in re.split(r"&&|\|\||;|\||\$\(|`", ln):
        words = part.strip().split()
        while words and (re.match(r"^\w+=", words[0]) or words[0] in ("sudo", "env", "exec", "time", "then", "do", "else", "!", "{envpython}", "-")):
            words = words[1:]
        if words and re.fullmatch(r"[a-z][\w.+-]*", words[0]):
            out.add(words[0])
    return out


def shell_commands(text: str, rel: str) -> set[str]:
    """The programs a build or test configuration runs: Makefile recipe lines, tox ``commands``, CI ``run:``/``script:``
    steps, nox ``session.run``, every line of a shell script. Keys and prose are never read as commands."""
    lines: list[str] = []
    name = rel.rsplit("/", 1)[-1]
    if re.fullmatch(r"(GNU)?[Mm]akefile", name):
        lines = [ln for ln in text.splitlines() if ln.startswith("\t")]
    elif name == "tox.ini":
        on = False
        for ln in text.splitlines():
            m = re.match(r"\s*commands(_pre|_post)?\s*=\s*(.*)$", ln)
            if m:
                on = True
                lines.append(m.group(2))
            elif on and ln[:1] in (" ", "\t") and ln.strip():
                lines.append(ln)
            else:
                on = False
    elif name == "noxfile.py":
        return set(re.findall(r"""session\.run\(\s*['"]([\w.+-]+)""", text))
    elif name.endswith((".yml", ".yaml")):
        block = None
        for ln in text.splitlines():
            ind = len(ln) - len(ln.lstrip())
            if block is not None and (ind > block or not ln.strip()):
                lines.append(ln)
                continue
            block = None
            m = re.match(r"\s*-?\s*(run|script|install|before_install|before_script)\s*:\s*(.*)$", ln)
            if m:
                rest = m.group(2).strip()
                if rest in ("|", ">", "|-", ">-", ""):
                    block = ind
                else:
                    lines.append(rest)
    elif name.endswith(".sh"):
        lines = text.splitlines()
    out: set[str] = set()
    for ln in lines:
        out |= _command_words(ln)
    return out


def _imports(text: str) -> set[str]:
    try:
        with warnings.catch_warnings():  # a repository's own invalid escapes are not ours to report
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set()
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            out.add(n.module.split(".")[0])
    return out


def declared_extras(tree: Path) -> list[str]:
    """The optional-dependency group names A declares (pyproject, setup.cfg, a literal ``extras_require`` in setup.py)."""
    names: set[str] = set()
    pp = tree / "pyproject.toml"
    if pp.exists():
        with contextlib.suppress(tomllib.TOMLDecodeError, OSError):
            d = tomllib.loads(pp.read_text())
            names |= set((d.get("project") or {}).get("optional-dependencies") or {})
            names |= set(((d.get("tool") or {}).get("poetry") or {}).get("extras") or {})
    cfg = tree / "setup.cfg"
    if cfg.exists():
        m = re.search(r"(?ms)^\[options\.extras_require\]\s*$(.*?)(?=^\[|\Z)", cfg.read_text(errors="replace"))
        if m:
            names |= set(re.findall(r"(?m)^([A-Za-z0-9][\w.-]*)\s*=", m.group(1)))
    sp = tree / "setup.py"
    if sp.exists():
        with contextlib.suppress(SyntaxError, ValueError, OSError):
            for n in ast.walk(ast.parse(sp.read_text(errors="replace"))):
                if isinstance(n, ast.keyword) and n.arg == "extras_require" and isinstance(n.value, ast.Dict):
                    names |= {k.value for k in n.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return sorted(x for x in names if re.fullmatch(r"[A-Za-z0-9][\w.-]*", x) and x.lower() not in DOC_EXTRAS)


def world_needs(tree: Path) -> dict[str, Any]:
    """What the World must carry beyond A's packaging (recipe 3), each with the file that asked for it:
    ``apt`` (the developer baseline + the commands A's build/test configuration runs, mapped by ``TOOL_APT``), ``pip_build``
    (setuptools, wheel), ``extras`` (every declared extra but docs), ``import_deps`` (third-party imports of A's tests and
    modules with a known distribution, ``MODULE_DIST``) and what stayed ``unresolved`` (commands and imports with no
    known provider: recorded, never installed)."""
    tree = Path(tree)
    apt = {p: ["baseline"] for p in DEV_TOOLS_APT}
    cmds: dict[str, set[str]] = {}
    imports: dict[str, set[str]] = {}
    local: set[str] = set()
    for f in sorted(tree.rglob("*")):
        rel = f.relative_to(tree).as_posix()
        hidden = any(part.startswith(".") for part in rel.split("/")[:-1]) and not rel.startswith(".github/")
        if not f.is_file() or hidden or f.stat().st_size > 400_000:
            continue
        text = f.read_text(errors="replace") if (f.suffix == ".py" or _CONFIG.search(rel)) else ""
        if _CONFIG.search(rel) and (f.suffix != ".py" or f.name == "noxfile.py"):
            for c in shell_commands(text, rel):
                cmds.setdefault(c, set()).add(rel)
            if re.search(r"(^|/)(GNU)?[Mm]akefile$", rel):
                cmds.setdefault("make", set()).add(rel)
        if f.suffix == ".py":
            local.add(f.stem)
            for c in _SUBPROC.findall(text):
                cmds.setdefault(c, set()).add(rel)
            for m in _imports(text):
                imports.setdefault(m, set()).add(rel)
    for d in tree.rglob("*"):
        if d.is_dir() and (d / "__init__.py").exists():
            local.add(d.name)
    unresolved_cmds = []
    for c, files in sorted(cmds.items()):
        pkg = TOOL_APT.get(c)
        if pkg and c not in NO_INSTALL:
            apt.setdefault(pkg, []).append(f"{c} in {min(files)}")
        elif not pkg and c not in _SHELL_BUILTINS and not _is_python_cmd(c):
            unresolved_cmds.append(c)
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
    deps, unresolved_imports = {}, []
    for m, files in sorted(imports.items()):
        if m in stdlib or m in local or m.startswith("_"):
            continue
        if m in MODULE_DIST:
            deps[MODULE_DIST[m]] = f"import {m} in {min(files)}"
        elif m not in ("pytest", "py", "_pytest", "setuptools", "pkg_resources", "distutils", "wheel"):
            unresolved_imports.append(m)
    files, _groups = requirement_sources(tree)
    pins: dict[str, str] = {}
    for f in files:
        pins.update(requirement_pins((tree / f).read_text(errors="replace")))
    return {"apt": sorted(apt), "pip_build": list(PY_BUILD), "extras": declared_extras(tree), "import_deps": sorted(deps), "pins": dict(sorted(pins.items())),
            "why": {"apt": {k: v[:3] for k, v in sorted(apt.items())}, "import_deps": deps}, "unresolved": {"commands": unresolved_cmds[:40], "imports": unresolved_imports[:40]}}


_SHELL_BUILTINS = {"cd", "echo", "export", "set", "if", "fi", "for", "done", "while", "test", "true", "false", "exit", "rm", "cp", "mv", "mkdir",
                   "ls", "cat", "find", "grep", "sed", "awk", "touch", "chmod", "printf", "read", "source", "shift", "case", "esac", "return",
                   "local", "git", "python", "python3", "pip", "pip3", "pytest", "py.test", "tox", "nox", "coverage", "uses", "name", "with",
                   "sh", "bash", "head", "tail", "sort", "uniq", "wc", "xargs", "basename", "dirname", "pwd", "which", "command", "eval",
                   "trap", "wait", "sleep", "kill", "ln", "readlink", "realpath", "mktemp", "tee", "tr", "cut", "date", "env", "unset"}


def _is_python_cmd(c: str) -> bool:
    return c.startswith(("python", "pip", "py", "flake8", "mypy", "black", "isort", "ruff", "pylint", "sphinx", "twine", "pre-commit", "check-",
                         "codecov", "coveralls", "mkdocs", "bump", "towncrier"))


def requirement_pins(text: str) -> dict[str, str]:
    """Exact pins (``name==version``) of a requirement file. The build installs the newest release (old pins rarely work
    on 3.12: glom's PyYAML 3.13 imports but calls ``collections.Hashable``); ``windows_s3.repair_versions`` starts its
    executed version search at the pin when A's own tests fail in the World."""
    out: dict[str, str] = {}
    for ln in text.splitlines():
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*==\s*([A-Za-z0-9.+!-]+)\s*(?:[;#].*)?$", ln)
        if m and m.group(1).lower() not in ("pytest", "pytest-timeout"):
            out[m.group(1)] = m.group(2)
    return out


def world_dockerfile(name: str | None, req_names: list[str], groups: list[str], *, packaged: bool = True, needs: dict[str, Any] | None = None) -> str:
    """``packaged``: A has a ``pyproject.toml`` / ``setup.py`` / ``setup.cfg`` (a bare tree installs nothing of its own).
    ``needs``: ``world_needs`` (recipe 3): system tools first, then setuptools/wheel, A with every non-doc extra, the
    known optional imports."""
    extras = ",".join(dict.fromkeys([*EXTRAS, *((needs or {}).get("extras") or [])]))
    lines = ["FROM " + BASE_IMAGE]
    if needs:
        apt = " ".join(shlex.quote(p) for p in needs.get("apt") or [])
        if apt:
            lines.append(f"RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends {apt} >/dev/null && rm -rf /var/lib/apt/lists/*")
        if needs.get("pip_build"):
            lines.append("RUN pip install --no-cache-dir " + " ".join(shlex.quote(p) for p in needs["pip_build"]))
    lines += ["COPY a /opt/lh/a", "WORKDIR /opt/lh/a"]
    if packaged:
        lines.append(f'RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install --no-cache-dir ".[{extras}]" || SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install --no-cache-dir .')
    if groups:
        lines.append("RUN pip install --no-cache-dir " + " ".join(shlex.quote(g) for g in groups) + " || true")
    for n in req_names:  # one by one: a name that no longer installs never blocks the others
        lines.append(f"RUN pip install --no-cache-dir {shlex.quote(n)} || true")
    for n in (needs or {}).get("import_deps") or []:
        lines.append(f"RUN pip install --no-cache-dir {shlex.quote(n)} || true")
    lines.append("RUN pip install --no-cache-dir " + " ".join(BASE_PINS))
    # the project stays installed: packages read their own metadata (humanize: importlib.metadata.version at import);
    # PYTHONPATH puts the workspace (src/ and root) ahead of site-packages, so every import is the agent's code
    lines += ["WORKDIR /w", "RUN rm -rf /opt/lh /root/.cache"]
    return "\n".join(lines) + "\n"


class ImageBook:
    """World images per (repo, A's packaging), built once, shared by every window with the same dependencies."""

    def __init__(self, *, parallel: int = 3):
        self._locks: dict[str, asyncio.Lock] = {}
        self._sem = asyncio.Semaphore(parallel)
        self.built: dict[str, dict[str, Any]] = {}

    def tag_for(self, repo: str, base_dir: Path) -> tuple[str, dict[str, Any]]:
        files, groups = requirement_sources(base_dir)
        names = sorted({n for f in files for n in requirement_names((base_dir / f).read_text(errors="replace"))})
        name = project_name(base_dir)
        h = packaging_hash(base_dir, [*PACKAGING, *files])
        slug = re.sub(r"[^a-z0-9_.-]+", "-", repo.lower())
        needs = world_needs(base_dir)
        key = {k: needs[k] for k in ("apt", "pip_build", "extras", "import_deps")}  # what the build installs (the reasons are not part of it)
        h = sha256_hex(IMAGE_RECIPE + h + json.dumps(key, sort_keys=True))
        return f"{PREFIX}w:{slug}-{h[:12]}", {"project": name, "requirement_files": files, "requirement_names": names, "groups": groups,
                                               "packaging_hash": h, "recipe": IMAGE_RECIPE, "needs": needs}

    async def ensure(self, repo: str, base_dir: Path) -> dict[str, Any]:
        """The World image card: tag, digest, how it was built, ``ok``."""
        tag, info = self.tag_for(repo, base_dir)
        lock = self._locks.setdefault(tag, asyncio.Lock())
        async with lock:
            if tag in self.built:
                return self.built[tag]
            digest = image_id(tag)
            card: dict[str, Any] = {"image": tag, **info}
            if digest is None:
                async with self._sem:
                    t0 = time.monotonic()
                    ok, log = await asyncio.to_thread(self._build, tag, base_dir, info)
                    card.update({"build_seconds": round(time.monotonic() - t0, 1), "build_log": log[-1500:]})
                digest = image_id(tag) if ok else None
            card.update({"digest": digest or f"unresolved:{tag}", "ok": digest is not None})
            self.built[tag] = card
            return card

    @staticmethod
    def _build(tag: str, base_dir: Path, info: dict[str, Any]) -> tuple[bool, str]:
        ensure_base()
        with tempfile.TemporaryDirectory(prefix="lhimg-") as td:
            ctx = Path(td)
            shutil.copytree(base_dir, ctx / "a", symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
            packaged = any((base_dir / f).exists() for f in PACKAGING)
            (ctx / "Dockerfile").write_text(world_dockerfile(info["project"], info["requirement_names"], info["groups"], packaged=packaged,
                                                             needs=info.get("needs")))
            r = docker("build", "-q", "-t", tag, str(ctx), timeout=1800)
            return r.returncode == 0, (r.stdout + r.stderr)


# ---- leak probes -------------------------------------------------------------------------------------------------------
LEAK_SCRIPT = r"""set -u
pip download --no-deps --no-cache-dir --retries 0 --timeout 3 -d /tmp/dl "$LH_PKG==$LH_VER" >/tmp/pipdl.log 2>&1
echo "PIP_RC=$?"
if [ "$LH_VER" = "0" ]; then echo "DIST_FILES="; else
echo "DIST_FILES=$(find / -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o \( -iname "$LH_DIST-$LH_VER*.whl" -o -iname "$LH_DIST-$LH_VER*.tar.gz" -o -iname "$LH_DIST-$LH_VER*.zip" \) -print 2>/dev/null | head -5 | tr '\n' ' ')"
fi
if [ -s /probe/names ]; then
  grep -rIlF -f /probe/names --include='*.py' / --exclude-dir=proc --exclude-dir=sys --exclude-dir=dev 2>/dev/null > /tmp/hits || true
  echo "NAME_HITS=$(python3 -c "$LH_COUNT" /probe/names /tmp/hits | head -5 | tr '\n' ' ')"
else
  echo "NAME_HITS="
fi
cd /probe/ws && echo "GIT_COMMITS=$(git log --all --format=%H | wc -l)" && echo "GIT_TAGS=$(git tag | tr '\n' ' ')"
"""


# a file leaks B's tests when it defines several of them (one shared name such as test_other is a coincidence: pyflakes'
# own tests matched a new more-itertools test name); a window with a single new test needs that one
LEAK_COUNT = r"""import sys
names = [ln.rstrip("\n") for ln in open(sys.argv[1]) if ln.strip()]
need = min(2, len(names))
for path in open(sys.argv[2]).read().split("\n"):
    if not path:
        continue
    try:
        text = open(path, errors="replace").read()
    except OSError:
        continue
    if sum(1 for n in names if n in text) >= need:
        print(path)
"""


def seed_workspace(base_dir: Path, ws: Path) -> None:
    """What ``GitRepoApp`` provisions: the tree at A, a fresh repository, one commit tagged ``base``."""
    shutil.copytree(base_dir, ws, symlinks=True)
    cmd = ("git init -q -b main . && git add -A && git -c user.email=crucible@local -c user.name=crucible commit -qm base --allow-empty "
           "&& git tag -f base")
    subprocess.run(["bash", "-c", cmd], cwd=ws, check=True, capture_output=True)


def parse_leak(out: str) -> dict[str, str]:
    return {k: v.strip() for k, v in re.findall(r"(?m)^([A-Z_]+)=(.*)$", out)}


def judge_leak(fields: dict[str, str], *, version: str | None) -> dict[str, Any]:
    """Each probe passes, fails, or is ``n/a`` (no B version to download)."""
    pip_rc = fields.get("PIP_RC")
    probes = {
        "pip_download_b_fails": "n/a" if not version else ("pass" if pip_rc not in (None, "", "0") else "fail"),
        "no_b_distribution_file": "n/a" if not version else ("pass" if not fields.get("DIST_FILES") else "fail"),
        "no_new_test_names_in_image": "pass" if not fields.get("NAME_HITS") else "fail",
        "git_log_only_base": "pass" if fields.get("GIT_COMMITS") == "1" and fields.get("GIT_TAGS", "").split() == ["base"] else "fail",
    }
    return {"probes": probes, "ok": all(v in ("pass", "n/a") for v in probes.values()), "evidence": {k: fields.get(k, "")[:300] for k in fields}}


async def leak_probes(image: str, base_dir: Path, *, package: str | None, version: str | None, new_test_names: set[str],
                      name_prefix: str = f"{PREFIX}leak-") -> dict[str, Any]:
    if not name_prefix.startswith(PREFIX):
        raise ValueError(f"World containers are named {PREFIX}*, got prefix {name_prefix!r}")
    with tempfile.TemporaryDirectory(prefix="lhleak-") as td:
        probe = Path(td)
        seed_workspace(base_dir, probe / "ws")
        # a definition of a new test, not the bare word (pip's vendored urllib3 says "timeout_value" too)
        (probe / "names").write_text("".join(f"def {n}(\n" for n in sorted(new_test_names)))
        dist = re.sub(r"[-_.]+", "?", (package or "none-such-package"))
        name = f"{name_prefix}{uuid.uuid4().hex[:10]}"
        args = ["run", "--rm", "--name", name, "--network", "none", "-v", f"{probe.resolve()}:/probe:ro", "-e", f"LH_PKG={package or 'none-such-package'}",
                "-e", f"LH_VER={version or '0'}", "-e", f"LH_DIST={dist}", "-e", f"LH_COUNT={LEAK_COUNT}", image, "bash", "-c", LEAK_SCRIPT]
        r = await asyncio.to_thread(docker, *args, timeout=600)
    if r.returncode not in (0, 1) and "PIP_RC" not in r.stdout:
        return {"probes": {}, "ok": False, "evidence": {"error": (r.stderr or r.stdout)[-500:]}}
    return judge_leak(parse_leak(r.stdout), version=version)


# ---- the policy's World ------------------------------------------------------------------------------------------------
def world_spec(ws: Path, base_dir: Path) -> Any:
    from crucible.core.schema.world import WorldSpec

    return WorldSpec.model_validate({
        "world": "factory/window", "network": "no-network",
        "apps": [{"id": "repo", "template": "common/git_repo", "state": {"adapter": "fs", "params": {"root": str(ws)}},
                  "params": {"shell": True, "init_git": True, "seed_dir": str(base_dir), "max_output": 30000}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["repo.*"]}]})


async def open_world(*, image: str, base_dir: Path, workdir: Path, instruction: str, memory: str = "2g", cpus: str = "2",
                     name_prefix: str = f"{PREFIX}world-", workspace_mode: str = "rw", pythonpath: str | None = None) -> Any:
    """A started, reset World whose tools run in a ``--network none`` container of ``image`` (closing the World removes
    the container). ``world._crucible_ws`` is the host workspace. ``name_prefix`` (always under ``crucible-lh-``) lets a
    caller find exactly the containers it made (a test run asserting its own cleanup, report 11 F7). ``workspace_mode``
    ``"ro"`` mounts the workspace read-only for the World's tools (a seeded tooling fault, P4); the host tree, which the
    harness collects the patch from, is unchanged. ``pythonpath`` (``{ws}`` = the workspace) replaces the default
    ``{ws}/src:{ws}`` (a stitched window's workspace holds two repositories, P5)."""
    if workspace_mode not in ("rw", "ro"):
        raise ValueError(f"workspace_mode is rw or ro, got {workspace_mode!r}")
    if not name_prefix.startswith(PREFIX):
        raise ValueError(f"World containers are named {PREFIX}*, got prefix {name_prefix!r}")
    from crucible.core.store import Store
    from crucible.world.exec import DockerExec
    from crucible.world.world import World

    workdir = Path(workdir).resolve()
    ws = workdir / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    name = f"{name_prefix}{uuid.uuid4().hex[:12]}"
    r = await asyncio.to_thread(docker, "run", "-d", "--name", name, "--network", "none", "--memory", memory, "--cpus", cpus, "--pids-limit", "512",
                                "-v", f"{ws}:{ws}:{workspace_mode}", "-w", str(ws), "-e", f"PYTHONPATH={(pythonpath or '{ws}/src:{ws}').format(ws=ws)}", "-e", "HOME=/tmp",
                                image, "sleep", "infinity", timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"world container did not start: {r.stderr[-500:]}")
    w = World(world_spec(ws, base_dir), instructions={"agent": instruction}, store=Store(workdir / "s"), workdir=workdir / "w")
    try:
        await w.start()
        w.apps["repo"].ctx.extra["policy_exec"] = DockerExec(name)
        await w.reset()
    except BaseException:
        await asyncio.to_thread(docker, "rm", "-f", name, timeout=120)
        raise
    w._crucible_ws = ws
    w._crucible_container = name
    inner = w.close

    async def close() -> None:
        try:
            await inner()
        finally:
            await asyncio.to_thread(docker, "rm", "-f", name, timeout=120)

    w.close = close
    return w


def own_containers() -> list[str]:
    r = docker("ps", "-a", "--filter", f"name={PREFIX}", "--format", "{{.Names}}", timeout=60)
    return [x for x in r.stdout.split() if x.startswith(PREFIX)]


def card_json(card: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in card.items() if k != "build_log"}, sort_keys=True)
