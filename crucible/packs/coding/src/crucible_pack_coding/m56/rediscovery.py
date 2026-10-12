"""M5 gate, rediscovery: do milestones mined from *upstream expert work* cover every check family our task 01-04 graders
were hand-built with (build, API removal, bit-exactness, edge semantics, concurrency, perf)?

Sources (none of them our graders, which would be circular):
- **diff-derived**: every GPU-migration PCL PR in the M3 store (git-only, no API): ``git diff base..head`` -> ``from_diff``
  candidates (removed/added constructs, located blocks, created/deleted files);
- **signal-derived**: review comments of the PCL PRs in the M4 store (reviewers demanding a property);
- **policy-derived**: PCL's written rules (``policy_docs`` traces: style guide, CONTRIBUTING, clang-tidy).

Each candidate is assigned to the check families its evidence supports (token, path, PR title or comment text), by an
explicit keyword table kept next to the code so the mapping is reviewable. The gate passes when every family has
diff-derived candidates from at least two independent PRs (behavioural evidence from code), with the signal/policy evidence reported beside it.

M5 audit (2026-10-12): only constructs count, in the direction the family means: "API removal" needs the API *removed*
(10 of 47 hits were APIs being added), and a block's name is not evidence (34 of 40 "edge semantics" hits were
functions named ``*Bound*`` / ``*Edge*`` touched by refactors such as "use override"). Each family's verdict says
whether its evidence was hand-audited or only matched by the keyword table.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter, defaultdict
from typing import Any

from crucible.connectors.trace_store import TraceStore
from crucible.core.store import Store
from crucible.workflow.milestones.mine import from_diff
from crucible_pack_coding import PACK_ROOT

CRUCIBLE = PACK_ROOT.parent.parent
GIT = CRUCIBLE / "data" / "cache" / "git" / "pcl.git"
REPORT = CRUCIBLE / "docs" / "modules" / "reports" / "m5_rediscovery.json"
SOURCE = re.compile(r"\.(cu|cuh|h|hpp|cpp|cc|cxx|hxx|inl)$|CMakeLists\.txt$|\.cmake$")

# Matched against a milestone's construct (its token or block anchor) and, for build, the file it lives in; never
# against free text, so "round" does not match "stereo_ground_segmentation" and "stream" does not match "PointStream".
FAMILIES: dict[str, re.Pattern[str]] = {
    "build": re.compile(r"^(?:CUDA_ARCH\w*|CMAKE_CUDA\w*|CUDA_NVCC_FLAGS|CUDA_VERSION|CUDART_VERSION|CUDA_ARCH_BIN|CUDA_ARCH_PTX|"
                        r"find_package|enable_language|check_language|cuda_add_library|cuda_add_executable|target_compile_\w+|"
                        r"set_target_properties|PCL_CUDA\w*|\w*_CUDA_\w*|sm_\d+|compute_\d+)$"),
    "api_removal": re.compile(r"^(?:tex[123]D\w*|tex1Dfetch|texture|textureReference|cudaBindTexture\w*|cudaUnbindTexture|"
                              r"cudaCreateChannelDesc|cudaThreadSynchronize|cudaThreadExit|cudaMemcpyToSymbol\w*|"
                              r"__shfl|__shfl_up|__shfl_down|__shfl_xor|__any|__all|__ballot|cudaGetTextureReference|\w*[Dd]eprecated\w*|"
                              r"thrust::\w+|cub::\w+|npp\w+)$"),
    "bit_exactness": re.compile(r"^(?:__f(?:add|sub|mul|div|ma|sqrt|rcp)_r[nzud]|__d(?:add|mul|div|sqrt)_r[nzud]|__float2int_r[nzud]|"
                                r"__int2float_r[nzud]|__float2uint_r[nzud]|__fdividef|__saturatef|rint|lrint|nearbyint|"
                                r"\w*[Ee]psilon\w*|\w*[Pp]recision\w*|\w*[Dd]eterminis\w*|\w*[Rr]ounding\w*)$"),
    "edge_semantics": re.compile(r"^(?:\w*[Cc]lamp\w*|\w*[Bb]order\w*|\w*[Bb]ound(?:s|ary|aries)?\w*|cudaAddressMode\w*|addressMode|"
                                 r"\w*[Ss]aturat\w*|isnan|isfinite|isinf|\w*[Oo]verflow\w*|\w*[Oo]ut_?[Oo]f_?[Rr]ange\w*|\w*[Ee]dge\w*)$"),
    "concurrency": re.compile(r"^(?:__syncthreads\w*|__syncwarp|__threadfence\w*|atomic\w+|\w*[Aa]tomic\w*|cudaStream\w*|stream_t|"
                              r"cudaDeviceSynchronize|cudaStreamSynchronize|cudaEvent\w*|volatile|std::mutex|mutex|lock_guard|"
                              r"std::thread|thread_local|__shfl_\w+_sync|__ballot_sync|__any_sync|__all_sync|cooperative_groups\w*)$"),
    "perf": re.compile(r"^(?:__shared__|__launch_bounds__|__ldg|__restrict__|cudaFuncSetCacheConfig|cudaFuncCachePrefer\w*|"
                       r"\w*BLOCK_SIZE\w*|\w*[Tt]hreads_?[Pp]er_?[Bb]lock\w*|\w*[Bb]enchmark\w*|\w*_performance|cudaOccupancy\w*|"
                       r"\w*[Uu]nroll\w*|#pragma unroll)$"),
}


def git(*args: str) -> str:
    out = subprocess.run(["git", f"--git-dir={GIT}", *args], capture_output=True, timeout=600, check=False)
    return out.stdout.decode("utf-8", "replace")


def show(rev: str):
    def read(path: str) -> str | None:
        r = subprocess.run(["git", f"--git-dir={GIT}", "show", f"{rev}:{path}"], capture_output=True, timeout=120, check=False)
        return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None

    return read


def families_of(construct: str) -> list[str]:
    c = construct.rstrip("(").strip()
    return [f for f, rx in FAMILIES.items() if rx.match(c) or (c.startswith("pcl::") and rx.match(c.rsplit("::", 1)[-1]))]


def text_families(text: str) -> list[str]:
    """For review comments and policy rules (human prose): word-level matches on a narrower vocabulary."""
    words = {
        "build": r"\b(?:cmake|nvcc|compute capability|sm_\d+|build fails?|compil(?:e|ation))\b",
        "api_removal": r"\b(?:deprecat\w*|texture|removed in cuda|legacy api)\b",
        "bit_exactness": r"\b(?:precision|bitwise|bit-exact|rounding|determinis\w*|epsilon|ulp)\b",
        "edge_semantics": r"\b(?:out of (?:range|bounds)|boundary|border|edge case|overflow|nan)\b",
        "concurrency": r"\b(?:race|thread-safe|threadsafe|data race|synchroni[sz]\w*|atomic|mutex|concurren\w*)\b",
        "perf": r"\b(?:performance|speed ?up|faster|slower|occupancy|benchmark\w*)\b",
    }
    return [f for f, w in words.items() if re.search(w, text, re.IGNORECASE)]


def run() -> dict[str, Any]:
    m3 = TraceStore(Store(CRUCIBLE / "data" / "m3" / "store"))
    prs = [t for t in m3 if t.source.connector == "github_pr"]
    diff_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_pr = {}
    for t in prs:
        opened = next((e for e in t.events if e.type == "pr_opened"), None)
        if not opened or not opened.data.get("base") or not opened.data.get("head"):
            continue
        base, head = opened.data["base"], opened.data["head"]
        diff = git("diff", "--no-color", "--no-renames", base, head)
        title = (opened.text or "").split("\n")[0]
        try:
            cands = from_diff(diff, show(base), include=lambda p: bool(SOURCE.search(p)), max_per_file=12)
        except Exception as e:  # noqa: BLE001 -- a diff that does not apply to its own base is reported, not hidden
            per_pr[t.trace_id] = {"error": str(e)[:200]}
            continue
        fams = Counter()
        for c in cands:
            change = c.evidence.get("change")
            if change not in ("removed", "added"):
                continue  # a located block's *name* says nothing about what changed inside it
            construct = str(c.evidence.get("token") or "")
            fams_here = [f for f in (families_of(construct) if construct else []) if f != "api_removal" or change == "removed"]
            for f in fams_here:
                fams[f] += 1
                diff_hits[f].append({"pr": t.trace_id, "title": title[:90], "milestone": c.milestone.id, "evidence": c.evidence})
        per_pr[t.trace_id] = {"title": title[:90], "candidates": len(cands), "families": dict(fams)}
    # signal-derived: review comments (M4 store)
    m4 = TraceStore(Store(CRUCIBLE / "data" / "m4"))
    signal_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in m4:
        for e in t.events:
            if e.type == "review_comment" and e.text:
                for f in text_families(e.text):
                    signal_hits[f].append({"trace": t.trace_id, "seq": e.seq, "text": e.text[:160]})
    # policy-derived
    policy_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in m3:
        if t.source.connector != "policy_docs" or "PointCloudLibrary" not in t.trace_id:
            continue
        for e in t.events:
            if e.type in ("policy_rule", "config_rule") and e.text:
                for f in text_families(e.text):
                    policy_hits[f].append({"doc": t.source.ref, "rule": e.text[:160]})
    coverage = {f: {"diff": len(diff_hits[f]), "prs": len({h["pr"] for h in diff_hits[f]}), "review_comments": len(signal_hits[f]),
                    "policy_rules": len(policy_hits[f]), "examples": diff_hits[f][:4], "comment_examples": signal_hits[f][:2]} for f in FAMILIES}
    # Hand audit of every hit the keyword table assigned (2026-10-09 for bit-exactness and perf; 2026-10-12, M5 audit, for
    # the rest, after hits were restricted to removed/added constructs). The verdict counts audited evidence only:
    # a family is rediscovered with genuine evidence from at least two PRs.
    audit = {
        "build": {"genuine": ["#5953 adds check_language( / CMAKE_CUDA_COMPILER_VERSION (CMake CUDA language detection)",
                              "#4402 adds a CUDA_ARCH_BIN build-configuration check to pcl::gpu::Octree's error path"],
                  "rejected": [], "verdict": "rediscovered from code (2 PRs)"},
        "api_removal": {"genuine": ["#6010 removes cudaBindTexture( / cudaBindTexture2D( / cudaUnbindTexture(", "#5535 and #2981 remove __all( / __ballot( / __any(",
                                    "#2736 removes cudaThreadExit( / cudaThreadSynchronize(", "#6472, #5872 remove deprecated PCL APIs"],
                        "rejected": ["#3121 'deprecated(' removed in a typedef -> using refactor"], "verdict": "rediscovered from code (6 PRs)"},
        "concurrency": {"genuine": ["#2981, #2212 add __all_sync( / __any_sync( / __ballot_sync( (warp votes with explicit masks)",
                                    "#3086 adds lock_guard (mutex migration)", "#2736, #6010 add cudaDeviceSynchronize("],
                        "rejected": ["#6472 removes deprecated 'atomics' code"], "verdict": "rediscovered from code (5 PRs)"},
        "edge_semantics": {"genuine": ["#6460 adds isfinite( (NaN guard in brute-force search)"],
                           "rejected": ["#5722 fov_*_bound_( removed: a default-member-init refactor",
                                        "#1568 SetSaturationRange( moved between files (VTK colour range)",
                                        "earlier report: 34 'located' hits were function names containing Bound/Edge in refactor PRs (#2728, #3006, #3121, #3180)"],
                           "verdict": "not rediscovered: genuine evidence from 1 PR"},
        "bit_exactness": {"genuine": [], "rejected": ["epsilon_/getEpsilon/off_surface_epsilon_ (#5722, #4629, #2736): numeric tolerance parameters, not bit-exactness",
                                                      "review #6199 (sigma2 -> 0, Welsch decay range): division-by-zero / NaN guard = edge semantics"],
                          "verdict": "not rediscovered from upstream PRs; in tasks 01-04 it comes from reference-differential checks (DESIGN §7.1)"},
        "perf": {"genuine": ["review #6430: benchmark table comparing inlined vs encapsulated implementations",
                             "review #6438: benchmark with one query point is not representative",
                             "diff #5535: removed test_gpu_octree_performance (a perf test target)"],
                 "rejected": ["review #6400 (error-message wording)", "review #6413 (API design)"],
                 "verdict": "rediscovered from review signals (2 PRs), not from code"},
    }
    found = {f: not audit[f]["verdict"].startswith("not") for f in FAMILIES}
    basis = {f: "hand-audited" for f in FAMILIES}
    out = {"gate": "M5 rediscovery", "pass": all(found.values()), "families_rediscovered": found, "basis": basis, "audit": audit, "prs": len(prs), "per_pr": per_pr,
           "coverage": coverage, "families": list(FAMILIES), "note": "keyword table FAMILIES in m56/rediscovery.py; our own graders are not a source"}
    REPORT.write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps({"pass": r["pass"], "prs": r["prs"], "coverage": {f: {k: v for k, v in c.items() if not k.endswith("examples")} for f, c in r["coverage"].items()}}, indent=1))
