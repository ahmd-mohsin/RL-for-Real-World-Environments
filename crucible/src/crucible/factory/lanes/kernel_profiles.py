"""Kernel debugging lane, the harness profiles (swarm-pipeline/10 §4.1). Standard library only: the kit plan
(``kernel_debug_plan``) imports it inside the verifier image, next to ``kernel_harness``.

A repository plugs into the lane through one ``KernelHarness`` profile, as a language plugs into the debug lane through
a ``LangProfile``. The profile owns everything that depends on the repository's own GPU test driver; the lane (mining,
the CPU gates, the bundle runner, the grader) only calls it:

| piece | what the profile defines |
|---|---|
| the build | ``prepare`` (the grader-owned harness written into the assembled tree), ``cmake_cuda`` / ``cmake_cpu`` / ``host_flags``, ``target``, ``exe`` (the target's path in the build directory), ``device_files`` (what the binary gate reads) |
| case enumeration | ``list_argv`` and ``listed`` (every case of a mode, none run) |
| per-case run and dump | ``argv``, ``env`` (seed, case selection, dump, repetitions), ``cases_file``, ``status``, ``crashed``, ``read_dump`` |
| the reference and tolerance | ``compare``: ``dump`` profiles compare the candidate's outputs with the reference build's reference column inside each case's own tolerance; ``status`` profiles take the repository's own self-checking verdict per case (its host reference and tolerance live in the test), the reference build passing the same case first |
| sanitizer cases | ``sanitizer_argv`` (the run the compute-sanitizer wraps; the cases come from the calibration) |
| perf timing | ``perf`` (whether a perf mode exists) and ``argv(mode="perf")`` with ``env(reps=)`` |
| fixed-seed input streams | ``seeded``: True when a grading seed reaches every input (fresh seeds are then drawn at grading time); False when the repository's tests draw from seeds fixed in their own source, so only those inputs are graded |

``GgmlHarness`` is the op test of ggml repositories (llama.cpp, ik_llama.cpp): a thin layer over ``kernel_harness``,
which stays the one implementation (its file digest is still the reference's identity), so verdicts are byte for byte
what they were before profiles existed. ``Catch2Harness`` is a CMake project whose GPU tests are self-checking Catch2
executables (CCCL's CUB tests): one executable per test source, cases are Catch2 test cases, the verdict is Catch2's own
per-case result in its XML report.

Which profile a repository uses is ``REPOS`` (training repositories, public) plus ``crucible/data/kernel_debug/
harness_repos.json`` (held-out repositories: their names never reach public git; ``HELD_OUT_PLACEHOLDER`` stands for
them in code and docs).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from crucible.factory.lanes import kernel_harness as kh

HELD_OUT_PLACEHOLDER = "<held-out kernel repo>"
LOCAL_REPOS = Path(__file__).resolve().parents[4] / "data" / "kernel_debug" / "harness_repos.json"


class KernelHarness:
    """The interface. Subclasses set the attributes and override the methods they need."""

    name = "base"
    compare_mode = "dump"  # "dump" | "status"
    seeded = True
    perf = False
    harness = ""  # the grader-owned harness path at B (file or directory)
    target = ""
    exe = ""  # the target's path relative to the build directory
    device_glob = ""  # files the binary gate lists SASS from (empty: the executable itself)

    # -- the build
    def prepare(self, tree: Path, harness_text: str) -> None:
        raise NotImplementedError

    def patch(self, text: str, *, need_perf: bool = False, header: str | None = None) -> str:
        return text

    def header(self, tree: Path) -> str | None:
        return None

    def cmake_cuda(self, arches: Sequence[str]) -> list[str]:
        raise NotImplementedError

    def cmake_cpu(self) -> tuple[str, ...]:
        raise NotImplementedError

    def host_flags(self, dispatch_present: Mapping[str, bool]) -> tuple[str, ...]:
        return ()

    # -- runs
    def argv(self, exe: Path | str, mode: str = "test", backend: str = "CUDA0", cases_file: Path | str | None = None) -> list:
        raise NotImplementedError

    def list_argv(self, exe: Path | str, mode: str = "test", backend: str = "CUDA0") -> tuple[list[str], dict[str, str]]:
        raise NotImplementedError

    def env(self, seed: int, cases_file: Path | str, dump: Path | str | None, *, reps: int = 0) -> dict[str, str]:
        raise NotImplementedError

    def sanitizer_argv(self, exe: Path | str, backend: str = "CUDA0", cases_file: Path | str | None = None) -> list:
        return self.argv(exe, "test", backend, cases_file)

    def cases_file(self, cases: Sequence[str]) -> str:
        return "".join(f"{c}\n" for c in cases)

    # -- reading what a run left
    def status(self, stdout: str) -> dict[str, str]:
        raise NotImplementedError

    def crashed(self, stdout: str) -> str | None:
        return None

    def listed(self, stdout: str) -> list[str]:
        raise NotImplementedError

    def read_dump(self, data: bytes) -> list[dict]:
        return []

    def compare(self, cand: Mapping, ref: Mapping, cases: Sequence[str]) -> dict[str, dict]:
        """Per case ``{"pass": bool, ...}``. ``cand`` / ``ref`` are run results: ``recs`` (dump records) and ``status``."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "compare": self.compare_mode, "seeded": self.seeded, "perf": self.perf, "harness": self.harness,
                "target": self.target, "exe": self.exe}


class GgmlHarness(KernelHarness):
    """ggml's ``tests/test-backend-ops.cpp`` (both dialects), every call delegated to ``kernel_harness``."""

    name = "ggml"
    compare_mode = "dump"
    seeded = True
    perf = True
    harness = kh.HARNESS
    target = kh.TARGET
    exe = f"bin/{kh.TARGET}"
    device_glob = "libggml-cuda*"
    FA_REF_ENV = kh.FA_REF_ENV
    EXACT_LINE = kh.EXACT_LINE

    def prepare(self, tree, harness_text):
        kh.prepare_tree(tree, harness_text)

    def patch(self, text, *, need_perf=False, header=None):
        return kh.patch_harness(text, need_perf=need_perf, header=header)

    def header(self, tree):
        return kh.ggml_header(tree)

    def cmake_cuda(self, arches):
        return kh.cmake_cuda(arches)

    def cmake_cpu(self):
        return kh.CMAKE_CPU

    def host_flags(self, dispatch_present):
        return tuple(f for k, flags in kh.HOST_CPU_FLAGS.items() if dispatch_present.get(k) for f in flags)

    def argv(self, exe, mode="test", backend="CUDA0", cases_file=None):
        return [exe, mode, "-b", backend]

    def list_argv(self, exe, mode="test", backend="CUDA0"):
        return [exe, mode, "-b", backend], {"CRUCIBLE_LIST": "1"}

    def env(self, seed, cases_file, dump, *, reps=0):
        env = {"CRUCIBLE_SEED": str(seed), "CRUCIBLE_CASES": str(cases_file), "GGML_CUDA_DISABLE_GRAPHS": "1", "CUDA_DISABLE_PTX_JIT": "1"}
        if dump is not None:
            env["CRUCIBLE_DUMP"] = str(dump)
        if reps:
            env["CRUCIBLE_REPS"] = str(reps)
        return env

    def cases_file(self, cases):
        return kh.cases_file(cases)

    def status(self, stdout):
        return kh.case_status(stdout)

    def crashed(self, stdout):
        return kh.crashed_case(stdout)

    def listed(self, stdout):
        return kh.listed_cases(stdout)

    def read_dump(self, data):
        return kh.read_dump(data)

    def compare(self, cand, ref, cases):
        return kh.compare_dumps(cand["recs"], ref["recs"], cases=cases)


# Catch2 test specs: a name is matched literally once these are escaped (``[`` opens a tag, ``,`` is "or", ``*`` a
# wildcard, ``\`` the escape itself); a leading ``~`` or ``exclude:`` would negate it
_SPEC_SPECIAL = re.compile(r"([\\\[\],*\"])")
_TESTCASE = re.compile(r"<TestCase\b[^>]*\bname=\"(?P<name>[^\"]*)\"[^>]*>(?P<body>.*?)</TestCase>", re.DOTALL)
_RESULT = re.compile(r"<OverallResult\b[^>]*\bsuccess=\"(?P<ok>true|false)\"")
_XML_ENT = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'"}


def _unxml(s: str) -> str:
    return re.sub(r"&(amp|lt|gt|quot|apos);", lambda m: _XML_ENT[m.group(0)], s)


def catch2_spec(name: str) -> str:
    """One Catch2 test spec that matches exactly ``name``."""
    s = _SPEC_SPECIAL.sub(r"\\\1", name)
    return "\\" + s if s.startswith(("~", "exclude:")) else s


class Catch2Harness(KernelHarness):
    """A CMake project whose GPU tests are self-checking Catch2 executables: the test sources at B are the grader's
    (written into the assembled tree, the candidate's copies never used), each case is a Catch2 test case, and its
    verdict is Catch2's own result (the test computes its host reference and applies its tolerance). The reference
    build must pass a case for it to be graded. Inputs come from seeds fixed in the tests' own source (``seeded`` False).

    ``targets`` maps a test source (relative to the repository) to its CMake target and executable; ``cmake`` is the
    project's configuration for the GPU build."""

    compare_mode = "status"
    seeded = False
    perf = False

    def __init__(self, name: str, *, cmake: Sequence[str], targets: Mapping[str, tuple[str, str]], harness_dir: str, cmake_cpu: Sequence[str] = ()):
        self.name = name
        self._cmake = tuple(cmake)
        self._cpu = tuple(cmake_cpu)
        self.targets = dict(targets)
        self.harness = harness_dir
        first = next(iter(self.targets.values()), ("", ""))
        self.target, self.exe = first

    def for_test(self, test_source: str) -> Catch2Harness:
        """The profile narrowed to one test source (its target and executable)."""
        t, e = self.targets[test_source]
        h = Catch2Harness(self.name, cmake=self._cmake, targets={test_source: (t, e)}, harness_dir=self.harness, cmake_cpu=self._cpu)
        return h

    def prepare(self, tree, harness_text):
        """``harness_text`` is a JSON map of test source -> B's text (the grader owns every test source it runs)."""
        for rel, text in json.loads(harness_text).items():
            p = Path(tree) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

    def cmake_cuda(self, arches):
        return [*self._cmake, f"-DCMAKE_CUDA_ARCHITECTURES={kh.cuda_arch_flag(arches)}"]

    def cmake_cpu(self):
        return self._cpu

    def argv(self, exe, mode="test", backend="CUDA0", cases_file=None):
        """Catch2's own selection file (``--input-file``, one test spec per line) and its XML report on stdout."""
        if cases_file is None:
            raise ValueError("a Catch2 run needs its case selection file")
        return [exe, "-r", "xml", "--input-file", str(cases_file)]

    def list_argv(self, exe, mode="test", backend="CUDA0"):
        return [str(exe), "--list-tests", "--verbosity", "quiet"], {}

    def env(self, seed, cases_file, dump, *, reps=0):
        return {"CRUCIBLE_CASES": str(cases_file), "CUDA_DISABLE_PTX_JIT": "1"}

    def cases_file(self, cases):
        return "".join(f"{catch2_spec(c)}\n" for c in cases)

    def status(self, stdout):
        out: dict[str, str] = {}
        for m in _TESTCASE.finditer(stdout):
            r = _RESULT.search(m.group("body"))
            out[_unxml(m.group("name"))] = "OK" if r and r.group("ok") == "true" else "FAIL"
        return out

    def crashed(self, stdout):
        """A case Catch2 opened and never closed (the process died inside it)."""
        opened = re.findall(r"<TestCase\b[^>]*\bname=\"([^\"]*)\"", stdout)
        closed = {m.group("name") for m in _TESTCASE.finditer(stdout)}
        left = [n for n in opened if n not in closed]
        return _unxml(left[-1]) if left else None

    def listed(self, stdout):
        return list(dict.fromkeys(ln.strip() for ln in stdout.splitlines() if ln.strip() and not ln.startswith((" ", "\t", "All available", "Matching"))))

    def compare(self, cand, ref, cases):
        out: dict[str, dict] = {}
        cs, rs = cand.get("status") or {}, ref.get("status") or {}
        for c in cases:
            if rs.get(c) != "OK":
                out[c] = {"pass": False, "infra": "the reference build does not pass this case", "ref": rs.get(c)}
                continue
            st = cs.get(c)
            out[c] = {"pass": st == "OK", "status": st or "missing", "nmse_vs_cpu": 0.0 if st == "OK" else float("inf"), "max_err": 1.0}
        return out


GGML = GgmlHarness()

# CCCL's CUB tests (NVIDIA/cccl): CUB only, testing on, one C++ dialect, Release. CCCL developer builds require the CUDA
# host compiler to be the C++ compiler, and its test configuration needs CMake >= 3.30.4 and fetches Catch2 through CPM
# (``CPM_SOURCE_CACHE``): the grader's image carries that CMake and a pre-filled cache, so candidate builds stay offline.
# Per-test executables are ``bin/cub.test.<category>.<name>.lid_<l>.types_<t>`` (``%PARAM%`` variants of one source).
CCCL_CMAKE = ("-DCMAKE_CXX_COMPILER=/usr/bin/g++", "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++", "-DCCCL_ENABLE_CUB=ON", "-DCCCL_ENABLE_THRUST=OFF",
              "-DCCCL_ENABLE_LIBCUDACXX=OFF", "-DCCCL_ENABLE_CUDAX=OFF", "-DCCCL_ENABLE_TESTING=ON", "-DCUB_ENABLE_TESTING=ON",
              "-DCUB_ENABLE_EXAMPLES=OFF", "-DCCCL_ENABLE_EXAMPLES=OFF", "-DCUB_ENABLE_DIALECT_CPP17=ON", "-DCMAKE_BUILD_TYPE=Release")
CCCL_CMAKE_MIN = (3, 30, 4)


def cccl_target(test_source: str, *, lid: int = 0, types: int | None = 0) -> tuple[str, str]:
    """``cub/test/catch2_test_device_reduce_by_key.cu`` -> (``cub.test.device.reduce_by_key.lid_0.types_0``, ``bin/...``):
    the category is the first word of the stem; a source with no ``%PARAM% TEST_TYPES`` line has no ``types_`` part
    (``types=None``)."""
    stem = re.sub(r"^catch2_test_", "", re.sub(r"\.(cu|cpp)$", "", Path(test_source).name))
    cat, _, rest = stem.partition("_")
    head = f"cub.test.{cat}.{rest}" if rest else f"cub.test.{cat}"
    t = f"{head}.lid_{lid}" + (f".types_{types}" if types is not None else "")
    return t, f"bin/{t}"


def ninja_targets(text: str, test_source: str) -> list[str]:
    """The executables ``ninja -t targets all`` lists for one CUB test source (every launch and type variant)."""
    base = cccl_target(test_source, lid=0, types=None)[0].rsplit(".lid_", 1)[0]
    return sorted({m.group(1) for m in re.finditer(rf"^({re.escape(base)}\.lid_\d+(?:\.types_\d+)?):", text, re.MULTILINE)})


def cccl(test_sources: Sequence[str], *, targets: Mapping[str, Sequence[str]] | None = None) -> Catch2Harness:
    """The CUB profile for ``test_sources``; ``targets`` (source -> [target, exe]) as a task's spec recorded it from the
    configured build, else the naming rule's launch-0 / types-0 variant."""
    tmap = {s: tuple((targets or {}).get(s) or cccl_target(s)) for s in test_sources}
    return Catch2Harness("catch2-cccl", cmake=CCCL_CMAKE, targets=tmap, harness_dir="cub/test")


PROFILES: dict[str, KernelHarness] = {"ggml": GGML}
# training repositories (registry name -> profile kind); held-out ones live in LOCAL_REPOS only
REPOS: dict[str, str] = {"llamacpp": "ggml", "ikllamacpp": "ggml", "cccl": "catch2-cccl"}


def local_repos(path: Path | None = None) -> dict[str, str]:
    p = Path(path or os.environ.get("CRUCIBLE_HARNESS_REPOS") or LOCAL_REPOS)
    try:
        return {str(k): str(v) for k, v in json.loads(p.read_text()).items()}
    except (OSError, ValueError):
        return {}


def kind_for(repo: str, *, local: Mapping[str, str] | None = None) -> str:
    """The profile kind of a registry repository name; a repository no table names is refused."""
    table = {**REPOS, **(local_repos() if local is None else local)}
    if repo not in table:
        raise KeyError(f"no kernel harness profile for repository {repo!r}")
    return table[repo]


def profile(name: str | None = None, *, tests: Sequence[str] = (), targets: Mapping[str, Sequence[str]] | None = None) -> KernelHarness:
    """The profile a spec names (``spec.harness``; absent -> ``ggml``, every task built before profiles existed)."""
    name = name or "ggml"
    if name in PROFILES:
        return PROFILES[name]
    if name == "catch2-cccl":
        return cccl(tests, targets=targets)
    raise KeyError(f"unknown kernel harness profile {name!r}")


def of_spec(spec: Mapping) -> KernelHarness:
    tests = spec.get("harness_tests") or {}
    return profile(spec.get("harness"), tests=list(tests), targets=tests if isinstance(tests, Mapping) else None)
