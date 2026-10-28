"""Framework health from traces (swarm-pipeline/11 §4.5, 13 §12): the **framework** side of the training monitor. The
monitor (``factory.train_monitor``) labels the agent's behaviour; this module asks, of every anomaly in the stored episodes,
whether the agent or the framework caused it, which component, how many episodes and how much reward it touched, and what
reproduces it. Read-only over stored episodes; it never grades, rewards or gates, and nothing it proposes is applied (V6, V12).

**Detectors** (deterministic; ``DETECTORS`` names each with its component and default severity):

| kind | component | read from | severity |
|---|---|---|---|
| ``masked_failure`` | by the failure's kind and stage (World, harness, grader, sandbox, sampler) | ``record.json`` ``failure`` | HIGH |
| ``unmeasured_checkpoint`` | grader (checkpoint sidecar) | ``checkpoints.jsonl`` rows not measured, with a reason | HIGH |
| ``world_fault`` | World | ``traceaudit.facts`` ``world_fault`` (a missing command or module, a tool fault) | CRITICAL when systematic |
| ``tool_error`` | World or harness | tool results with ``fault == "tool"``, clustered by their first line | CRITICAL when systematic |
| ``host_resource`` | sandbox and host | OOM, disk, lock, network, Apptainer patterns in failures and tool results | HIGH |
| ``host_unreachable`` | sandbox and host | a remote host that did not come back within the pause budget (detail ``host_unreachable``), one cluster per host; ``meta.host_outages`` of an episode that paused and resumed | HIGH / LOW |
| ``token_truncation`` | sampler | model calls that finished on ``length``; a ``max_output_tokens`` stop | HIGH |
| ``provider_error`` | sampler | model calls that finished on an error | HIGH |
| ``tool_timeout`` | harness | a tool result that reports the harness's own time limit | LOW |
| ``nondeterministic_grade`` | grader | the same patch (sha256) measured twice by the checkpoint sidecar with different milestones | CRITICAL |
| ``credit_grade_mismatch`` | grader | the checkpoint sidecar and the final grader disagree on one patch: Φ and per-turn credit miss what the outcome counts | CRITICAL |
| ``path_false_positive`` | grader | a stored path verdict the current rules flip (``factory.repath``), moving the reward | CRITICAL |
| ``path_verdict_drift`` | grader | a stored path verdict the current rules flip without moving the reward | LOW |
| ``stale_visible_test`` | task spec | a pass-to-pass test failing in most solving episodes, or exactly where a milestone passes, across >= 2 families | CRITICAL |
| ``protected_edit_consensus`` | task spec | the same protected test file edited by >= 2 families | HIGH |
| ``unreachable_milestone`` | task spec | a milestone no episode of >= 2 families ever passes (a suspect, verified by the disclosure check) | HIGH |
| ``spec_defect`` | task spec | ``traceaudit.propose_spec_defects`` grouped by name | HIGH |
| ``codec_suspect`` | sampler | one model's bad-tool-name and invalid-argument rate far above the others' | HIGH |
| ``fairness_false_fail`` | grader | an S9b result (``*/fairness/<task>.json``) with false-fail candidates, or low second-verifier agreement | CRITICAL / HIGH |
| ``trainer_numeric`` | trainer | ``steps.jsonl``: NaN or inf loss, a sampler-learner KL spike, dropped groups, staleness | CRITICAL / HIGH |

**Agent or framework.** Every cluster carries ``cls``: ``framework``; ``agent`` for a tool error the agent's own arguments
caused (a wrong ``old_string``, a bad regex, a path that does not exist) and for a missing command no World is expected to
provide (a made-up name in the agent's test script), both LOW and never a pause; ``ambiguous`` for a harness time limit on
a command the agent chose; ``suspect`` for an unreachable milestone until the disclosure check decides.

Each cluster has a signature (normalized: paths, hashes and numbers folded), the episodes and tasks it touches, its first
occurrence, a minimal reproduction (the stage and the error, or the tool call with its arguments and observed result, or
the check with its old and new verdict), and an **impact**: the reward it moved where the corrected reward is known (a path
flip), the reward at risk where it is not (a stale test, a systematic World fault: an upper bound), and the advantage
distortion over each rollout group (``sum |Δ advantage|``, sign flips), or the samples and groups a masked defect cost.

**Severity.** CRITICAL corrupts reward or advantage (an unmasked wrong grade); HIGH wastes episodes or credit (masked
attempts, unmeasured checkpoints, truncation); LOW is cosmetic. ``pause`` is true when any CRITICAL defect is open: the
operator fixes it between steps, records the fix (``record_fix``), regrades the affected stored episodes offline (the
``regrade_plan``), and the trainer resumes. No reward changes silently: ``versions`` lists, per step, the path-rules and
reward-rule versions the grades used, and a run whose versions changed without a fix record is itself a HIGH defect.

**Metadata** (``metadata``) per lane and step: tool failure rate, seconds per tool call and per model call, tokens per turn
and per episode, context use, compactions and Φ across them, checkpoint grading cost and timing, mask rate, World
minutes, budget stops, time to first edit, and useful gradient samples (an episode in a group with reward spread) per
dollar. ``tuning`` turns them into proposals with the expected effect and the evidence.

**Held-out discipline.** Benchmark episodes count only in aggregate: a defect found on held-out traces keeps its component,
kind, severity and counts; its tasks become placeholders, its episodes, reproduction and content-bearing signature are
withheld (``redact``).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from crucible.factory.traceaudit import BUDGET_STOPS, EpisodeBundle, facts, honest_trace_facts, propose_spec_defects, propose_world_bugs
from crucible.factory.train_monitor import Episode, _edit_steps, attempt_dirs, episode_of, in_band_tasks, placeholder, step_health, step_map
from crucible.world.exec import HOST_UNREACHABLE
from crucible.world.hostreach import INTERRUPTED

OUTAGE_META = "host_outages"  # record.meta of an attempt that paused for a remote host (``runtime.run.OUTAGE_META``)

SEVERITY = {"CRITICAL": 3, "HIGH": 2, "LOW": 1}
COMPONENTS = ("world", "harness", "grader", "sandbox", "sampler", "trainer", "task_spec")
DETECTORS: dict[str, tuple[str, str]] = {  # kind -> (default component, default severity)
    "masked_failure": ("harness", "HIGH"), "unmeasured_checkpoint": ("grader", "HIGH"), "world_fault": ("world", "HIGH"),
    "tool_error": ("harness", "HIGH"), "host_resource": ("sandbox", "HIGH"), "token_truncation": ("sampler", "HIGH"),
    "provider_error": ("sampler", "HIGH"), "tool_timeout": ("harness", "LOW"), "nondeterministic_grade": ("grader", "CRITICAL"),
    "path_false_positive": ("grader", "CRITICAL"), "stale_visible_test": ("task_spec", "CRITICAL"), "protected_edit_consensus": ("task_spec", "HIGH"),
    "unreachable_milestone": ("task_spec", "HIGH"), "spec_defect": ("task_spec", "HIGH"), "codec_suspect": ("sampler", "HIGH"),
    "fairness_false_fail": ("grader", "CRITICAL"), "trainer_numeric": ("trainer", "HIGH"), "version_mix": ("grader", "HIGH"),
    "path_verdict_drift": ("grader", "LOW"), "credit_grade_mismatch": ("grader", "CRITICAL"), "job_log_error": ("harness", "HIGH"),
    "unreadable_attempt": ("harness", "HIGH"), "host_unreachable": ("sandbox", "HIGH"),
}
CONTENT_KINDS = {"stale_visible_test", "protected_edit_consensus", "unreachable_milestone", "spec_defect", "fairness_false_fail"}
HOST = ((re.compile(r"out of memory|OOM|Killed\b|exit (status|code) 137|MemoryError|Cannot allocate memory", re.IGNORECASE), "memory"),
        (re.compile(r"No space left on device|disk quota exceeded", re.IGNORECASE), "disk"),
        (re.compile(r"\bflock\b|lock(file)? (held|timeout)|could not acquire", re.IGNORECASE), "lock"),
        (re.compile(r"Temporary failure in name resolution|Network is unreachable|Could not resolve host|Connection (refused|reset)", re.IGNORECASE), "network"),
        (re.compile(r"apptainer|singularity|FATAL:\s+container", re.IGNORECASE), "apptainer"),
        (re.compile(r"(docker|podman)[^\n]{0,40}(daemon|No such container|is not running)", re.IGNORECASE), "container daemon"))
HARNESS_TIMEOUT = re.compile(r"(timed out after \d+|exceeded the \d+ ?s (time )?limit|TimeoutExpired|command timed out)", re.IGNORECASE)
NOT_FOUND = re.compile(r"command not found|not found in the World|No module named", re.IGNORECASE)
# a tool error the agent's own arguments caused (a wrong old_string, a bad regex, a path that does not exist): the agent's
# mistake, reported as LOW agent behaviour, never a framework defect and never a pause
AGENT_INPUT = re.compile(r"old_string (not found|occurs \d+ times)|re\.error|FileNotFoundError|No such file or directory|IsADirectoryError|"
                         r"not a directory|Not a valid object name|unknown revision|pathspec .* did not match|invalid (json|arguments)", re.IGNORECASE)
# commands a World is expected to provide (toolchains, test runners, the World tools' own interpreter): missing one is the
# World's fault; any other missing command is the agent's (a made-up name in its own test script: 'foo', 'x')
KNOWN_TOOLS = frozenset({"python", "python3", "pip", "pip3", "pytest", "node", "npm", "npx", "pnpm", "yarn", "vitest", "jest", "mocha", "tsc", "oxlint",
                         "eslint", "go", "gofmt", "cargo", "rustc", "rustfmt", "java", "javac", "mvn", "gradle", "make", "cmake", "ninja", "gcc", "g++", "clang",
                         "nvcc", "compute-sanitizer", "git", "rg", "grep", "sed", "awk", "find", "xxd", "diff", "patch", "bash", "sh", "timeout", "tar"})
# the ones a task cannot be done without (its toolchain, its test runner, the World tools' own interpreter): missing in most of
# a task's episodes corrupts its rewards (CRITICAL); a missing convenience tool (xxd, rg, diff) only costs tool calls (HIGH)
CORE_TOOLS = frozenset({"python", "python3", "pip", "pip3", "pytest", "node", "npm", "npx", "pnpm", "yarn", "vitest", "jest", "mocha", "tsc", "go", "cargo",
                        "rustc", "java", "javac", "mvn", "gradle", "make", "cmake", "nvcc", "compute-sanitizer", "git", "bash", "sh"})
TOOL_NOT_FOUND = re.compile(r"(?:^|\n|: )(?:sh: \d+: |bash: (?:line \d+: )?)?([\w.+-]+): (?:command )?not found")


# ---- signatures --------------------------------------------------------------------------------------------------------------
def norm(s: Any, n: int = 160) -> str:
    """Fold a message into a signature: paths, hashes, numbers, quoted values and whitespace."""
    x = str(s or "")
    x = re.sub(r"(/[\w.@+-]+){2,}/?", "<path>", x)
    x = re.sub(r"\b[0-9a-f]{7,64}\b", "<hex>", x)
    x = re.sub(r"(?<![A-Za-z_])\d+(\.\d+)?", "N", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x[:n]


def error_line(reason: str) -> str:
    """The telling line of a traceback or a long error: the last line naming an error, else the first non-empty line."""
    lines = [ln.strip() for ln in str(reason or "").splitlines() if ln.strip()]
    errs = [ln for ln in lines if re.search(r"\b\w*(Error|Exception|error:|FATAL)\b", ln)]
    return (errs[-1] if errs else (lines[0] if lines else ""))[:300]


def sig_id(kind: str, component: str, signature: str) -> str:
    return "fd-" + hashlib.sha256(f"{kind}|{component}|{signature}".encode()).hexdigest()[:10]


# ---- one attempt, read once ----------------------------------------------------------------------------------------------------
@dataclass
class View:
    ep: Episode
    b: EpisodeBundle
    rec: dict[str, Any]
    rep: dict[str, Any]
    cps: list[dict[str, Any]]
    calls: list[dict[str, Any]]
    max_prompt: int | None
    group: str  # the rollout group: the run directory of the attempt

    @property
    def regress_failing(self) -> list[str]:
        for c in self.rep.get("checks") or []:
            if isinstance(c, dict) and c.get("id") == "c_regress" and c.get("status") == "fail":
                return list((c.get("detail") or {}).get("failing") or [])
        return []


def _read(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _jsonl(p: Path) -> list[dict[str, Any]]:
    out = []
    if p.exists():
        for ln in p.read_text(errors="replace").splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    return out


def view_of(adir: Path, *, cohort: str = "", step: int | None = None, split_overrides: Mapping[str, str] | None = None,
            tool_limit: int | None = None) -> View:
    ep, b = episode_of(adir, cohort=cohort or adir.parents[2].name, step=step, split_overrides=split_overrides, tool_limit=tool_limit)
    rec = _read(adir / "record.json") or {}
    rep = _read(adir / "verifier" / "report.json") or {}
    calls_raw = _read(adir / "calls.json") or {}
    calls = [c for v in (calls_raw.values() if isinstance(calls_raw, dict) else []) for c in (v if isinstance(v, list) else []) if isinstance(c, dict)]
    traj = _read(adir / "trajectory.agent.json") or {}
    prompts = [int((s.get("metrics") or {}).get("prompt_tokens") or 0) for s in traj.get("steps") or [] if isinstance(s, dict)]
    return View(ep, b, rec, rep, _jsonl(adir / "checkpoints.jsonl"), calls, max(prompts) if prompts else None, str(adir.parents[2]))


def load_views(roots: Sequence[Path | str], *, labels: Mapping[str, str] | None = None, split_overrides: Mapping[str, str] | None = None,
               tool_limit: int | None = None) -> list[View]:
    out = []
    for root in roots:
        smap = step_map(root)
        for a in attempt_dirs([root]):
            try:
                out.append(view_of(a, cohort=(labels or {}).get(str(root), ""), step=smap.get(a.parents[2].name), split_overrides=split_overrides,
                                   tool_limit=tool_limit))
            except Exception:  # noqa: BLE001, S112 - never a crash: health_report reports the attempts no view was read from
                continue
    return out


def unreadable_hits(roots: Sequence[Path | str], views: Sequence[View]) -> list[Hit]:
    """Attempt directories under the roots that no view could be read from: a Runner that wrote a broken record."""
    read = {v.ep.dir for v in views}
    return [Hit("unreadable_attempt", "harness", "an attempt directory whose record or trace cannot be read", "HIGH", str(a), "-", {"attempt": str(a)})
            for a in attempt_dirs(roots) if str(a) not in read]


# ---- per-attempt detectors --------------------------------------------------------------------------------------------------------
@dataclass
class Hit:
    kind: str
    component: str
    signature: str
    severity: str
    episode: str
    task: str
    repro: dict[str, Any] = field(default_factory=dict)
    at: int | None = None
    reward_new: float | None = None  # the corrected reward, when known
    reward_at_risk: float | None = None  # an upper bound, when the corrected reward is not known
    cls: str = "framework"
    core: bool = True  # a World fault in something the task cannot be done without (escalates to CRITICAL when systematic)


def host_gone(f: Mapping[str, Any], rep: Mapping[str, Any] | None = None) -> str | None:
    """The host a masked failure says stopped answering (``world.hostreach``: detail ``host_unreachable``, or a masked grade
    whose ``failure_kind`` is it), ``"?"`` when the reason does not name it; None for any other failure."""
    infra = (rep or {}).get("infra") or {}
    reason = str(f.get("reason") or infra.get("reason") or "")
    if f.get("detail") != HOST_UNREACHABLE and infra.get("failure_kind") != HOST_UNREACHABLE and not reason.startswith(HOST_UNREACHABLE):
        return None
    m = re.search(r" on ([\w.@:-]+): the host", reason)
    return m.group(1) if m else "?"


def _failure_component(kind: str, stage: str, reason: str) -> str:
    k = kind.lower()
    if "container_lost" in k or "sandbox" in k:
        return "sandbox"
    if "answer_exposed" in k:
        return "world"
    if "provider" in k or "sampler" in k or "rate_limit" in k:
        return "sampler"
    if stage == "verification" or "grade" in k:
        return "grader"
    if stage in ("seed", "reset", "world"):
        return "world"
    return "harness"


def per_attempt(v: View) -> list[Hit]:
    """The anomalies one attempt shows (module doc), each with its component and a minimal reproduction."""
    b, ep, out = v.b, v.ep, []
    f = v.rec.get("failure") or {}
    outages = (v.rec.get("meta") or {}).get(OUTAGE_META) or {}
    gone = host_gone(f, v.rep)
    if gone is not None:
        # a remote host that stopped answering and did not come back within the pause budget (world.hostreach): one cluster
        # per host, whatever operation it cut, never the generic mask or a network host_resource cluster of the same fact
        host = next(iter(outages.get("hosts") or []), None) or gone
        out.append(Hit("host_unreachable", "sandbox", f"{HOST_UNREACHABLE}: {host} (masked)", "HIGH", ep.id, ep.task,
                       {"stage": f.get("stage"), "kind": f.get("kind"), "detail": f.get("detail") or HOST_UNREACHABLE, "error": error_line(str(f.get("reason") or "")),
                        "paused_s": outages.get("paused_s")}))
    elif outages.get("resumed"):  # paused for the host and resumed: the episode stands, the framework lost wall time (and maybe a cut call)
        for host in outages.get("hosts") or ["?"]:
            out.append(Hit("host_unreachable", "sandbox", f"{HOST_UNREACHABLE}: {host} (paused, resumed)", "LOW", ep.id, ep.task,
                           {"paused_s": outages.get("paused_s"), "outages": outages.get("outages"), "rows": (outages.get("rows") or [])[:3]}))
    if (ep.masked or f) and gone is None:
        kind, stage, reason = str(f.get("kind") or "masked"), str(f.get("stage") or ""), str(f.get("reason") or (v.rep.get("infra") or {}).get("reason") or "")
        comp = _failure_component(kind, stage, reason)
        host = next((what for rx, what in HOST if rx.search(reason)), None)
        if host:
            out.append(Hit("host_resource", "sandbox", f"{host}: {norm(error_line(reason))}", "HIGH", ep.id, ep.task,
                           {"stage": stage, "error": error_line(reason)}))
        out.append(Hit("masked_failure", comp, f"{kind}@{stage or '?'}: {norm(error_line(reason))}", "HIGH", ep.id, ep.task,
                       {"stage": stage, "kind": kind, "error": error_line(reason), "retryable": f.get("retryable")}))
    for c in v.cps:
        if not c.get("measured", True) and str(c.get("reason") or "").strip():
            out.append(Hit("unmeasured_checkpoint", "grader", norm(error_line(str(c["reason"]))), "HIGH", ep.id, ep.task,
                           {"checkpoint": c.get("index"), "turn": c.get("turn"), "triggers": c.get("triggers"), "error": error_line(str(c["reason"]))},
                           at=c.get("turn")))
    if not ep.masked:
        fs = facts(b)
        for p in propose_world_bugs(b, fs):
            sig = str(p.artifact.get("signature") or "")
            cmd = re.search(r"command '([^']+)' not found in the World", sig)
            agent = cmd is not None and cmd.group(1) not in KNOWN_TOOLS
            out.append(Hit("world_fault", "world", norm(sig), "LOW" if agent else "HIGH", ep.id, ep.task,
                           {"function": p.artifact.get("function"), "arguments": p.artifact.get("arguments"), "observed": p.artifact.get("observed")},
                           at=p.artifact.get("step"), reward_at_risk=None if agent else max(0.0, 1.0 - float(ep.reward or 0.0)),
                           cls="agent" if agent else "framework"))
    seen: set[str] = set()
    for st in b.steps:
        txt = st.result or ""
        if INTERRUPTED in txt:  # a call an outage cut and the host came back: the host_unreachable cluster above, not a tool defect
            continue
        if st.fault == "tool":
            first = next((ln for ln in txt.splitlines() if ln.strip()), "")
            agent = bool(AGENT_INPUT.search(txt)) and not NOT_FOUND.search(txt)
            comp = "world" if NOT_FOUND.search(txt) else "harness"
            sig = f"{st.function}: {norm(first)}"
            if sig not in seen:
                seen.add(sig)
                out.append(Hit("tool_error", comp, sig, "LOW" if agent else "HIGH", ep.id, ep.task,
                               {"function": st.function, "arguments": st.arguments, "observed": first[:300]}, at=st.i,
                               reward_at_risk=None if agent else max(0.0, 1.0 - float(ep.reward or 0.0)), cls="agent" if agent else "framework"))
        for m in TOOL_NOT_FOUND.finditer(txt):
            tool = m.group(1).rsplit("/", 1)[-1]
            if tool in KNOWN_TOOLS and f"missing:{tool}" not in seen:
                seen.add(f"missing:{tool}")
                out.append(Hit("world_fault", "world", f"command {tool!r} missing in the World", "HIGH", ep.id, ep.task,
                               {"function": st.function, "arguments": st.arguments, "observed": m.group(0).strip()[:200]}, at=st.i,
                               reward_at_risk=max(0.0, 1.0 - float(ep.reward or 0.0)) if tool in CORE_TOOLS else None, core=tool in CORE_TOOLS))
        if HARNESS_TIMEOUT.search(txt):
            sig = f"{st.function}: {norm(HARNESS_TIMEOUT.search(txt).group(0))}"  # type: ignore[union-attr]
            if sig not in seen:
                seen.add(sig)
                out.append(Hit("tool_timeout", "harness", sig, "LOW", ep.id, ep.task, {"function": st.function, "arguments": st.arguments}, at=st.i,
                               cls="ambiguous"))
        if not st.ok or st.fault:
            host = next((what for rx, what in HOST if rx.search(txt)), None)
            if host and f"host:{host}" not in seen:
                seen.add(f"host:{host}")
                out.append(Hit("host_resource", "sandbox", f"{host}: {norm(error_line(txt))}", "HIGH", ep.id, ep.task,
                               {"function": st.function, "observed": error_line(txt)}, at=st.i))
    missing = {m.group(1) for h in out if h.kind == "world_fault" for m in [re.match(r"command '([^']+)' missing in the World", h.signature)] if m}
    if missing:  # one missing command, one defect: drop the tool errors and trace facts that are the same fact seen through a tool
        out = [h for h in out if not ((h.kind == "tool_error" and any(f"{t}: command not found" in str(h.repro.get("observed")) for t in missing))
                                     or (h.kind == "world_fault" and any(f"command '{t}' not found" in h.signature for t in missing)))]
    for c in v.cps:
        if not c.get("measured", True) and not str(c.get("reason") or "").strip() and c.get("state") not in ("start", "skipped", "cap"):
            out.append(Hit("unmeasured_checkpoint", "grader", "a checkpoint left unmeasured with no reason recorded", "HIGH", ep.id, ep.task,
                           {"checkpoint": c.get("index"), "turn": c.get("turn"), "state": c.get("state"), "triggers": c.get("triggers")}, at=c.get("turn")))
            break
    lengths = [c for c in v.calls if c.get("finish") == "length"]
    if lengths or str(ep.stop) == "max_output_tokens":
        out.append(Hit("token_truncation", "sampler", "a model call finished on its token limit" + (" and ended the episode" if str(ep.stop) == "max_output_tokens" else ""),
                       "HIGH" if str(ep.stop) == "max_output_tokens" else "LOW", ep.id, ep.task,
                       {"calls_at_limit": len(lengths), "stop": ep.stop, "usage": (lengths[0].get("usage") if lengths else None)}))
    errs = [c for c in v.calls if str(c.get("finish") or "") in ("error", "content_filter") or c.get("error")]
    if errs:
        out.append(Hit("provider_error", "sampler", norm(f"finish {errs[0].get('finish')}: {errs[0].get('error') or ''}"), "HIGH", ep.id, ep.task,
                       {"calls": len(errs)}))
    by_sha: dict[str, dict[str, bool]] = {}
    for c in v.cps:
        if not c.get("measured") or not c.get("patch_sha256") or not isinstance(c.get("milestones"), dict):
            continue
        ms = {k: bool(x) for k, x in c["milestones"].items()}
        prev = by_sha.setdefault(str(c["patch_sha256"]), ms)
        if prev != ms:
            diff = sorted(k for k in set(prev) | set(ms) if prev.get(k) != ms.get(k))
            out.append(Hit("nondeterministic_grade", "grader", f"one patch graded two ways ({len(diff)} milestone(s) differ)", "CRITICAL", ep.id, ep.task,
                           {"patch_sha256": c["patch_sha256"], "milestones_differ": diff, "checkpoint": c.get("index")}, at=c.get("turn")))
            break
    patch_p = Path(ep.dir) / "artifacts" / "model.patch"
    if patch_p.exists() and b.milestones:
        sha = hashlib.sha256(patch_p.read_bytes()).hexdigest()
        if sha in by_sha and by_sha[sha] != b.milestones and set(by_sha[sha]) == set(b.milestones):
            diff = sorted(k for k in b.milestones if by_sha[sha].get(k) != b.milestones[k])
            out.append(Hit("credit_grade_mismatch", "grader", f"the checkpoint sidecar and the final grader disagree on one patch ({len(diff)} milestone(s))",
                           "CRITICAL", ep.id, ep.task, {"patch_sha256": sha, "milestones_differ": diff, "checkpoint": by_sha[sha], "final": b.milestones}))
    return out


# ---- cross-episode detectors ------------------------------------------------------------------------------------------------------
def stale_tests(views: Sequence[View], *, share: float = 0.6, min_solving: int = 3) -> list[Hit]:
    """A pass-to-pass test that fails in most episodes reaching every milestone, across >= 2 families: the start tree still
    asserts the bug (a stale visible test), not the agent breaking it. Each such episode lost its full pass."""
    out = []
    by_task: dict[str, list[View]] = defaultdict(list)
    for v in views:
        if not v.ep.masked and v.ep.milestones and all(v.ep.milestones.values()):
            by_task[v.ep.task].append(v)
    for task, vs in by_task.items():
        if len(vs) < min_solving:
            continue
        fails = Counter(t for v in vs for t in set(v.regress_failing))
        for test, n in fails.items():
            hit = [v for v in vs if test in v.regress_failing]
            fams = {v.ep.family for v in hit}
            if n >= max(2, math.ceil(share * len(vs))) and len(fams) >= 2:
                for v in hit:
                    out.append(Hit("stale_visible_test", "task_spec", f"pass-to-pass test {test} fails in {n}/{len(vs)} solving episodes ({len(fams)} families)",
                                   "CRITICAL", v.ep.id, task, {"test": test, "solving": len(vs), "failing": n, "families": sorted(fams)},
                                   reward_at_risk=max(0.0, 1.0 - float(v.ep.reward or 0.0))))
    every: dict[str, list[View]] = defaultdict(list)
    for v in views:
        if not v.ep.masked and v.ep.milestones:
            every[v.ep.task].append(v)
    done = {(h.task, h.repro.get("test")) for h in out}
    for task, vs in every.items():  # a pass-to-pass test that fails exactly where a milestone passes: it asserts the bug
        tests = {t for v in vs for t in v.regress_failing}
        for test in tests:
            if (task, test) in done:
                continue
            for m in sorted({k for v in vs for k in v.ep.milestones}):
                solved = [v for v in vs if v.ep.milestones.get(m)]
                if len(solved) < min_solving:
                    continue
                agree = sum((test in v.regress_failing) == bool(v.ep.milestones.get(m)) for v in vs) / len(vs)
                fams = {v.ep.family for v in solved if test in v.regress_failing}
                if agree >= 0.9 and len(fams) >= 2:
                    done.add((task, test))
                    for v in solved:
                        if test in v.regress_failing:
                            out.append(Hit("stale_visible_test", "task_spec", f"pass-to-pass test {test} fails whenever {m} passes ({agree:.0%} of {len(vs)} episodes)",
                                           "CRITICAL", v.ep.id, task, {"test": test, "milestone": m, "agreement": round(agree, 3), "families": sorted(fams)},
                                           reward_at_risk=max(0.0, 1.0 - float(v.ep.reward or 0.0))))
                    break
    protected: dict[tuple[str, str], list[View]] = defaultdict(list)
    for v in views:
        if v.ep.masked:
            continue
        for fct in honest_trace_facts(v.b):
            if fct.kind == "protected_edit":
                protected[(v.ep.task, norm(fct.evidence or fct.detail, 120))].append(v)
    for (task, what), vs in protected.items():
        fams = {v.ep.family for v in vs}
        if len(fams) >= 2:
            for v in vs:
                out.append(Hit("protected_edit_consensus", "task_spec", f"protected test edited by {len(fams)} families: {what}", "HIGH", v.ep.id, task,
                               {"edit": what, "families": sorted(fams)}))
    return out


def unreachable(views: Sequence[View], *, min_episodes: int = 6) -> list[Hit]:
    """A milestone no unmasked episode of >= 2 families ever passes: a suspect for the disclosure check (S7r, spec_defect)."""
    out = []
    by_task: dict[str, list[View]] = defaultdict(list)
    for v in views:
        if not v.ep.masked and v.ep.milestones:
            by_task[v.ep.task].append(v)
    for task, vs in by_task.items():
        fams = {v.ep.family for v in vs}
        if len(vs) < min_episodes or len(fams) < 2:
            continue
        for m in sorted({k for v in vs for k in v.ep.milestones}):
            if not any(v.ep.milestones.get(m) for v in vs):
                for v in vs:
                    out.append(Hit("unreachable_milestone", "task_spec", f"milestone {m} passed by 0/{len(vs)} episodes of {len(fams)} families", "HIGH",
                                   v.ep.id, task, {"milestone": m, "episodes": len(vs), "families": sorted(fams)}, cls="suspect"))
    return out


def spec_defects(views: Sequence[View]) -> list[Hit]:
    out = []
    for v in views:
        if v.ep.failed:
            for p in propose_spec_defects(v.b):
                names = (p.change.get("prompt_edit") or {}).get("names") or []
                out.append(Hit("spec_defect", "task_spec", f"hidden tests need {names[0] if names else '?'}, which the request never names", "HIGH", v.ep.id, v.ep.task,
                               {"names": names, "grader_line": p.artifact.get("grader_line")}))
    return out


def codec(views: Sequence[View], *, ratio: float = 3.0, floor: float = 0.2) -> list[Hit]:
    """One policy model whose malformed tool calls (bad tool name, invalid arguments) run far above every other model's: a
    codec or tool-format bug (the Kimi native-section case) rather than the agent."""
    rate: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for v in views:
        m = v.ep.model or v.ep.policy
        for st in v.b.steps:
            rate[m][1] += 1
            rate[m][0] += st.fault == "policy" and str(st.error) in ("bad_tool_name", "invalid_arguments", "unknown_tool")
    rs = {m: k / n for m, (k, n) in rate.items() if n >= 50}
    out = []
    for m, r in rs.items():
        others = sorted(x for mm, x in rs.items() if mm != m)
        med = others[len(others) // 2] if others else None
        if r >= floor and med is not None and r >= ratio * max(med, 1e-3):
            for v in views:
                if (v.ep.model or v.ep.policy) == m:
                    out.append(Hit("codec_suspect", "sampler", f"{m}: malformed tool calls {r:.2f} of calls against a median {med:.2f}", "HIGH", v.ep.id, v.ep.task,
                                   {"model": m, "rate": round(r, 3), "median_other": round(med, 3)}))
    return out


def repath_flips(views: Sequence[View], store: Path | str) -> list[Hit]:
    """Stored path verdicts the current rules flip (``factory.repath``; no model call, no test run). A moved reward is a
    CRITICAL grader defect: the episode was trained on the wrong reward."""
    from crucible.factory.pathcheck import RULES_VERSION
    from crucible.factory.repath import repath

    out = []
    windows: dict[str, tuple[dict[str, Any], Path] | None] = {}
    for v in views:
        if v.ep.lane != "debug" or v.ep.masked:
            continue
        if v.ep.task not in windows:
            hit = next((d for d in Path(store).glob(f"*/tasks/{v.ep.task}") if (d / "window.json").exists()), None)
            windows[v.ep.task] = (json.loads((hit / "window.json").read_text()), hit / "base") if hit else None
        w = windows[v.ep.task]
        if w is None:
            continue
        try:
            r = repath(Path(v.ep.dir), w[0], w[1])
        except Exception:  # noqa: BLE001, S112 - a window that cannot be re-pathed keeps its stored verdict
            continue
        if r is None or not r.flipped:
            continue
        moved = r.old.get("reward") != r.new.get("reward")
        stored_v = ((v.rep.get("meta") or {}).get("path") or {}).get("rules_version") or "unversioned"
        for chk in r.changed or ["reward"]:
            out.append(Hit("path_false_positive" if moved else "path_verdict_drift", "grader", f"{chk}: {r.old.get('gates', {}).get(chk)} -> {r.new.get('gates', {}).get(chk)} "
                           f"({stored_v} -> {RULES_VERSION})", "CRITICAL" if moved else "LOW", v.ep.id, v.ep.task,
                           {"check": chk, "old": r.old.get("gates", {}).get(chk), "new": r.new.get("gates", {}).get(chk), "old_reward": r.old.get("reward"),
                            "new_reward": r.new.get("reward"), "rules": [stored_v, RULES_VERSION], "evidence": r.new.get("evidence")},
                           reward_new=r.new.get("reward") if moved else None))
    return out


def fairness(roots: Sequence[Path | str], *, kappa_min: float = 0.6) -> list[Hit]:
    """S9b results under the roots: false-fail candidates (the grader rejects a correct alternative) and low agreement
    between the two verifiers."""
    out = []
    for root in roots:
        for p in sorted(Path(root).rglob("fairness/*.json")):
            if p.name in ("summary.json",) or p.name.endswith(".suite.json"):
                continue
            d = _read(p) or {}
            task = str(d.get("task") or p.stem)
            for ff in d.get("false_fail") or []:
                fid = ff.get("id") if isinstance(ff, dict) else str(ff)
                out.append(Hit("fairness_false_fail", "grader", "S9b: the grader rejects an independently confirmed alternative", "CRITICAL", str(fid), task,
                               {"result": str(p), "alternative": fid, "failing_hidden": (ff.get("failing_hidden") if isinstance(ff, dict) else None)}))
            kappa = (d.get("agreement") or {}).get("kappa")
            if kappa is not None and float(kappa) < kappa_min:
                out.append(Hit("fairness_false_fail", "grader", f"S9b: second-verifier agreement kappa {float(kappa):.2f} < {kappa_min}", "HIGH", f"{task}/agreement", task,
                               {"result": str(p), "kappa": kappa, "disagreements": len((d.get("agreement") or {}).get("disagreements") or [])}))
    return out


def log_hits(roots: Sequence[Path | str], *, max_bytes: int = 50_000_000) -> list[Hit]:
    """Framework errors outside any episode: the job logs at a root's top level (``*.log``, ``*.jsonl``), e.g. a GPU runner
    whose every grade died before an episode existed (the flock bug lived only in the plumbing log)."""
    out = []
    for root in roots:
        for p in sorted(Path(root).glob("*")):
            if p.suffix not in (".log", ".jsonl") or not p.is_file() or p.stat().st_size > max_bytes or p.name in ("spend.jsonl", "load.jsonl"):
                continue
            text = p.read_text(errors="replace")
            seen: set[str] = set()
            for m in re.finditer(r"Traceback \(most recent call last\):(?:.|\n){0,4000}?\n(\w+(?:Error|Exception)[^\n]*)", text):
                sig = norm(m.group(1))
                if sig not in seen:
                    seen.add(sig)
                    out.append(Hit("job_log_error", "harness", sig, "HIGH", f"{Path(root).name}/{p.name}", "-", {"log": str(p), "error": m.group(1)[:300]}))
            for rx, what in HOST:
                mm = rx.search(text)
                if mm and what not in seen:
                    seen.add(what)
                    line = text[max(0, text.rfind("\n", 0, mm.start()) + 1): text.find("\n", mm.end()) if text.find("\n", mm.end()) > 0 else mm.end()]
                    out.append(Hit("host_resource", "sandbox", f"{what}: {norm(line, 120)}", "HIGH", f"{Path(root).name}/{p.name}", "-", {"log": str(p), "line": line[:300]}))
    return out


def trainer_hits(run_dirs: Sequence[Path | str], *, kl_max: float = 0.05, stale_max: int | None = None) -> list[Hit]:
    out = []
    for rd in run_dirs:
        for s in step_health(rd):
            step = s.get("step")
            loss = s.get("loss")
            if isinstance(loss, int | float) and (math.isnan(loss) or math.isinf(loss)):
                out.append(Hit("trainer_numeric", "trainer", "loss is NaN or inf", "CRITICAL", f"step-{step}", "-", {"step": step, "loss": str(loss)}, at=step))
            if s.get("kl") is not None and abs(float(s["kl"])) > kl_max:
                out.append(Hit("trainer_numeric", "trainer", f"sampler-learner KL above {kl_max}", "HIGH", f"step-{step}", "-", {"step": step, "kl": s["kl"]}, at=step))
            if s.get("drops"):
                out.append(Hit("trainer_numeric", "trainer", "groups dropped at the step: " + ",".join(sorted(map(str, s["drops"]))), "HIGH", f"step-{step}", "-",
                               {"step": step, "drops": s["drops"]}, at=step))
        p = Path(rd) / "steps.jsonl"
        for row in _jsonl(p):
            st = row.get("staleness") or row.get("max_staleness_seen")
            if stale_max is not None and isinstance(st, int | float) and st > stale_max:
                out.append(Hit("trainer_numeric", "trainer", f"rollout staleness above {stale_max}", "HIGH", f"step-{row.get('step')}", "-",
                               {"step": row.get("step"), "staleness": st}, at=row.get("step")))
    return out


# ---- clusters and impact ----------------------------------------------------------------------------------------------------------
def _adv_distortion(views: Sequence[View], corrected: Mapping[str, float]) -> dict[str, Any]:
    """Each rollout group's centred advantage (r - mean of the group's unmasked rewards) with the stored rewards against the
    corrected ones: the sum of |Δ advantage| and the sign flips."""
    by: dict[str, list[View]] = defaultdict(list)
    for v in views:
        if not v.ep.masked and v.ep.reward is not None:
            by[v.group].append(v)
    total, flips, groups = 0.0, 0, 0
    for vs in by.values():
        if not any(v.ep.id in corrected for v in vs):
            continue
        groups += 1
        old = [float(v.ep.reward) for v in vs]  # type: ignore[arg-type]
        new = [float(corrected.get(v.ep.id, v.ep.reward)) for v in vs]  # type: ignore[arg-type]
        mo, mn = sum(old) / len(old), sum(new) / len(new)
        for o, n in zip(old, new, strict=True):
            ao, an = o - mo, n - mn
            total += abs(an - ao)
            flips += (ao > 1e-9 and an < -1e-9) or (ao < -1e-9 and an > 1e-9)
    return {"groups": groups, "sum_abs_delta_advantage": round(total, 4), "sign_flips": flips}


def _mask_cost(views: Sequence[View], ids: set[str]) -> dict[str, Any]:
    by: dict[str, list[View]] = defaultdict(list)
    for v in views:
        by[v.group].append(v)
    starved = sum(1 for vs in by.values() if any(v.ep.id in ids for v in vs) and sum(not v.ep.masked for v in vs) < 2)
    return {"samples_lost": len(ids), "groups_without_signal": starved}


@dataclass
class Defect:
    id: str
    kind: str
    component: str
    cls: str
    severity: str
    signature: str
    episodes: list[str]
    tasks: list[str]
    first: dict[str, Any]
    repro: dict[str, Any]
    impact: dict[str, Any]
    heldout: bool = False
    status: str = "open"  # open | fixed@<version> | recurring
    fix: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def cluster(hits: Sequence[Hit], views: Sequence[View], *, systematic: float = 0.5) -> list[Defect]:
    """One defect per (kind, component, signature): its episodes, tasks, first occurrence, a reproduction, the impact, and
    the severity (a World or tool fault in >= ``systematic`` of a task's unmasked episodes corrupts its rewards: CRITICAL)."""
    by_id = {v.ep.id: v for v in views}
    per_task = Counter(v.ep.task for v in views if not v.ep.masked)
    groups: dict[tuple[str, str, str], list[Hit]] = defaultdict(list)
    for h in hits:
        groups[(h.kind, h.component, h.signature)].append(h)
    out = []
    for (kind, comp, sig), hs in groups.items():
        eps = list(dict.fromkeys(h.episode for h in hs))
        tasks = sorted({h.task for h in hs})
        sev = max((h.severity for h in hs), key=lambda s: SEVERITY[s])
        if kind in ("world_fault", "tool_error") and hs[0].cls == "framework" and hs[0].core:
            share = max((sum(1 for e in eps if by_id.get(e) is not None and by_id[e].ep.task == t) / per_task[t]) for t in tasks if per_task.get(t)) if any(per_task.get(t) for t in tasks) else 0.0
            sev = "CRITICAL" if share >= systematic and len(eps) >= 2 else ("LOW" if len(eps) < 2 else sev)
        corrected = {h.episode: float(h.reward_new) for h in hs if h.reward_new is not None}
        impact: dict[str, Any] = {"episodes": len(eps), "tasks": len(tasks)}
        if corrected:
            moved = [corrected[e] - float(by_id[e].ep.reward or 0.0) for e in corrected if e in by_id]
            impact.update({"reward_moved": round(sum(moved), 4), "reward_up": sum(1 for x in moved if x > 0), "reward_down": sum(1 for x in moved if x < 0),
                           **_adv_distortion(views, corrected)})
        risk = [h.reward_at_risk for h in hs if h.reward_at_risk is not None]
        if risk and not corrected:
            impact["reward_at_risk"] = round(sum(risk), 4)  # an upper bound
            if sev == "CRITICAL":
                impact.update({"advantage_if_corrected": _adv_distortion(views, {h.episode: min(1.0, float(by_id[h.episode].ep.reward or 0.0) + float(h.reward_at_risk or 0.0))
                                                                                 for h in hs if h.episode in by_id and h.reward_at_risk is not None}),
                               "estimate": "upper bound: each episode as if the defect had not cost it the full pass"})
        if kind in ("masked_failure", "host_resource") or any(by_id.get(e) is not None and by_id[e].ep.masked for e in eps):
            impact.update(_mask_cost(views, {e for e in eps if by_id.get(e) is not None and by_id[e].ep.masked}))
        first = min(hs, key=lambda h: ((by_id[h.episode].rec.get("started_at") if h.episode in by_id else "") or "", h.at or 0))
        out.append(Defect(sig_id(kind, comp, sig), kind, comp, hs[0].cls, sev, sig, eps, tasks,
                          {"episode": first.episode, "at": first.at, "started": by_id[first.episode].rec.get("started_at") if first.episode in by_id else None},
                          first.repro, impact, heldout=any(by_id[e].ep.split == "heldout" for e in eps if e in by_id)))
    out.sort(key=lambda d: (-SEVERITY[d.severity], -d.impact.get("episodes", 0)))
    return out


def redact(d: Defect, views: Sequence[View]) -> Defect:
    """A defect touching held-out episodes keeps its component, kind, severity and counts; its held-out tasks become
    placeholders, and its episodes, reproduction and (for content-bearing kinds) signature are withheld."""
    if not d.heldout:
        return d
    split = {v.ep.task: v.ep.split for v in views}
    ho = {t for t in d.tasks if split.get(t) == "heldout"}
    sig = d.signature if d.kind not in CONTENT_KINDS else f"{d.kind} (held-out: signature withheld)"
    return Defect(d.id, d.kind, d.component, d.cls, d.severity, sig, [e for e in d.episodes if not any(t in e for t in ho)],
                  sorted({placeholder(t) if t in ho else t for t in d.tasks}), {"withheld": True}, {"withheld": "held-out"},
                  {k: v for k, v in d.impact.items()}, heldout=True, status=d.status, fix=d.fix)


# ---- versions and the fix journal -------------------------------------------------------------------------------------------------
def versions(views: Sequence[View]) -> dict[str, Any]:
    """Per step (or cohort) the path-rules and reward-rule versions the grades used. Two versions in one training run with no
    fix record between them is a HIGH defect: rewards from two rule sets trained one policy."""
    rows: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: {"path_rules": Counter(), "reward_rule": Counter()})
    for v in views:
        if v.ep.masked:
            continue
        at = f"step {v.ep.step}" if v.ep.step is not None else v.ep.cohort
        meta = v.rep.get("meta") or {}
        rows[at]["path_rules"][str((meta.get("path") or {}).get("rules_version") or "unversioned")] += 1
        rows[at]["reward_rule"][norm(meta.get("reward_rule") or "unrecorded", 60)] += 1
    return {at: {k: dict(c) for k, c in r.items()} for at, r in sorted(rows.items())}


def fixes_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "framework_fixes.jsonl"


def record_fix(run_dir: Path | str, *, defect: str, component: str, version: str, commit: str, after_step: int | None, note: str = "",
               regraded: Sequence[str] = ()) -> dict[str, Any]:
    """Append a fix to the run's journal: the defect, the new rules or verifier version, the commit, the step after which it
    applies, and the stored episodes regraded offline under it (a regrade is reporting only: ``versioning.OfflineGrade``)."""
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "defect": defect, "component": component, "version": version, "commit": commit,
           "after_step": after_step, "note": note, "regraded": list(regraded)}
    p = fixes_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def apply_fix_journal(defects: Sequence[Defect], views: Sequence[View], journal: Sequence[Mapping[str, Any]]) -> None:
    """Mark each defect a fix record covers: ``fixed@<version>`` when no episode after the fix's step shows it, ``recurring``
    when one does."""
    step_of = {v.ep.id: v.ep.step for v in views}
    for d in defects:
        rec = next((r for r in reversed(list(journal)) if r.get("defect") == d.id), None)
        if rec is None:
            continue
        after = rec.get("after_step")
        later = [e for e in d.episodes if after is not None and (step_of.get(e) or -1) > after]
        d.fix, d.status = dict(rec), ("recurring" if later else f"fixed@{rec.get('version')}")


def regrade_plan(d: Defect, views: Sequence[View]) -> dict[str, Any]:
    """How to re-score the episodes a fix touches, offline: path rules re-path from the stored trail and diff; a grader or
    task fix re-grades in the World; a masked defect has no reward to correct (the attempt was dropped, not mis-scored)."""
    dirs = sorted({v.ep.dir for v in views if v.ep.id in set(d.episodes)})
    roots = sorted({str(Path(x).parents[4]) for x in dirs if len(Path(x).parents) > 4})
    if d.kind in ("path_false_positive", "path_verdict_drift", "version_mix"):
        how = {"tool": "repath", "command": "uv run python scripts/lh_repath.py --roots " + " ".join(roots) + " --store <window store> --out <report>"}
    elif d.kind in ("stale_visible_test", "nondeterministic_grade", "fairness_false_fail", "spec_defect", "unreachable_milestone", "world_fault", "tool_error"):
        how = {"tool": "regrade", "command": "factory.tinker.regrade (or lh_debug_target.py regrade) on the affected tasks after the fix; "
               "rows are versioning.OfflineGrade, never training rewards"}
    else:
        how = {"tool": "none", "command": "masked or wasted attempts: nothing to re-score; re-run the lost samples if the group needs them"}
    return {"defect": d.id, "episodes": len(dirs), "roots": roots, **how}


# ---- metadata and tuning ----------------------------------------------------------------------------------------------------------
def _mean(xs: Iterable[float | None]) -> float | None:
    ys = [float(x) for x in xs if x is not None]
    return sum(ys) / len(ys) if ys else None


def metadata(views: Sequence[View], *, context: int = 262144) -> dict[str, Any]:
    """Per lane, and per step when the views carry steps: the numbers ``tuning`` reads (module doc)."""
    def block(vs: Sequence[View]) -> dict[str, Any]:
        un = [v for v in vs if not v.ep.masked]
        steps = [st for v in un for st in v.b.steps]
        tim = [v.rec.get("timing") or {} for v in un]
        us = [v.rec.get("usage") or {} for v in un]
        calls = sum(int(u.get("calls") or 0) for u in us)
        agent_s = sum(float(t.get("agent") or 0.0) for t in tim)
        model_s = sum(float(t.get("model") or 0.0) for t in tim)
        cp_graded = [c for v in un for c in v.cps if c.get("measured") and c.get("graded_s") is not None]
        cp_s = sum(float(c["graded_s"]) for c in cp_graded)
        comp_d = []
        for v in un:
            meas = [c for c in v.cps if c.get("measured") and c.get("phi") is not None]
            for k, c in enumerate(meas):
                if "compaction" in (c.get("triggers") or []) and k > 0:
                    comp_d.append(float(c["phi"]) - float(meas[k - 1]["phi"]))
        firsts = []
        for v in un:
            ed = _edit_steps(v.b)
            if ed:
                firsts.append(min(ed))
        by_group: dict[str, list[View]] = defaultdict(list)
        for v in un:
            if v.ep.reward is not None:
                by_group[v.group].append(v)
        useful = sum(len(g) for g in by_group.values() if len(g) >= 2 and max(float(x.ep.reward) for x in g) - min(float(x.ep.reward) for x in g) > 1e-9)  # type: ignore[arg-type]
        cost = sum(float(v.rec.get("cost_usd") or 0.0) for v in vs)
        trunc = sum(1 for st in steps if re.search(r"\[\.\.\. ?\d+ (chars|bytes|lines)|truncated", st.result or "", re.IGNORECASE))
        return {
            "episodes": len(vs), "masked": len(vs) - len(un), "mask_rate": (len(vs) - len(un)) / len(vs) if vs else None,
            "tool_calls": len(steps), "tool_fail_rate": (sum(1 for st in steps if not st.ok) / len(steps)) if steps else None,
            "tool_fault_mix": dict(Counter(f"{st.fault}:{st.error}" for st in steps if st.fault)),
            "world_s_per_tool_call": ((agent_s - model_s) / len(steps)) if steps and agent_s else None,
            "model_s_per_call": (model_s / calls) if calls else None, "model_share_of_agent_time": (model_s / agent_s) if agent_s else None,
            "input_tokens_per_turn": (sum(int(u.get("input_tokens") or 0) for u in us) / calls) if calls else None,
            "output_tokens_per_turn": (sum(int(u.get("output_tokens") or 0) for u in us) / calls) if calls else None,
            "tokens_per_episode": _mean(int(u.get("input_tokens") or 0) + int(u.get("output_tokens") or 0) for u in us),
            "cache_share": (sum(int(u.get("cache_read_tokens") or 0) for u in us) / max(1, sum(int(u.get("input_tokens") or 0) for u in us))) if us else None,
            "context_use_max": _mean((v.max_prompt / context) for v in un if v.max_prompt), "compactions_per_episode": _mean(v.ep.compactions for v in un),
            "phi_delta_at_compaction": _mean(comp_d), "compactions_losing_phi": (sum(1 for x in comp_d if x < -1e-9) / len(comp_d)) if comp_d else None,
            "checkpoints_graded": len(cp_graded), "checkpoint_s_mean": (cp_s / len(cp_graded)) if cp_graded else None,
            "checkpoint_share_of_agent_time": (cp_s / agent_s) if agent_s else None,
            "checkpoints_unmeasured": sum(1 for v in un for c in v.cps if not c.get("measured", True)),
            "world_min_per_episode": _mean(float(t.get("agent") or 0.0) / 60 for t in tim),
            "budget_stop_share": (sum(1 for v in un if str(v.ep.stop) in BUDGET_STOPS) / len(un)) if un else None,
            "first_edit_call": _mean(firsts), "no_edit_share": (1 - len(firsts) / len(un)) if un else None,
            "truncated_tool_outputs": (trunc / len(steps)) if steps else None,
            "length_finish_share": (sum(1 for v in un for c in v.calls if c.get("finish") == "length") / max(1, sum(len(v.calls) for v in un))) if un else None,
            "useful_samples": useful, "useful_share": (useful / len(un)) if un else None, "usd": round(cost, 4),
            "usd_per_useful_sample": (cost / useful) if useful else None,
            "episode_wall_s": _mean(float(t.get("agent") or 0.0) for t in tim),
        }
    lanes: dict[str, list[View]] = defaultdict(list)
    steps: dict[int, list[View]] = defaultdict(list)
    for v in views:
        lanes[v.ep.lane].append(v)
        if v.ep.step is not None:
            steps[v.ep.step].append(v)
    return {"lanes": {k: block(vs) for k, vs in sorted(lanes.items())}, "steps": {s: block(vs) for s, vs in sorted(steps.items())}, "all": block(views)}


def spread_probability(p: float, k: int) -> float:
    """P(a group of k rollouts has reward spread) for a task solved with probability p: 1 - p^k - (1-p)^k."""
    return 1.0 - p ** k - (1.0 - p) ** k


def tuning(md: Mapping[str, Any], views: Sequence[View], *, cadence: int = 20, cap: int = 15, k: int = 8, band: tuple[float, float] = (0.1, 0.7),
           monitor_modes: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    """Tuning proposals from the metadata (each with its expected effect and the evidence). Never applied here."""
    out: list[dict[str, Any]] = []

    def prop(lever: str, lane: str, text: str, expected: str, evidence: Mapping[str, Any]) -> None:
        out.append({"id": "tp-" + hashlib.sha256(f"{lever}|{lane}|{text}".encode()).hexdigest()[:10], "lever": lever, "lane": lane, "text": text,
                    "expected": expected, "evidence": dict(evidence), "proposal": True, "applied": False,
                    "acceptance": "an A/B over one step on training tasks: the target number moves and useful samples per dollar does not drop"})
    for lane, m in (md.get("lanes") or {}).items():
        share = m.get("checkpoint_share_of_agent_time")
        if share is not None and share > 0.15:
            new = min(2 * cadence, round(cadence * share / 0.08))
            prop("checkpoint_cadence", lane, f"checkpoint every {new} tool calls instead of {cadence}", f"grading overhead {share:.0%} -> about 8% of World time",
                 {"checkpoint_share_of_agent_time": share, "checkpoint_s_mean": m.get("checkpoint_s_mean")})
        if (m.get("checkpoints_unmeasured") or 0) > 0.1 * max(1, m.get("checkpoints_graded") or 0):
            prop("checkpoint_cadence", lane, "fix the unmeasured checkpoints before tuning cadence (Φ is carried forward, credit is delayed)",
                 "per-turn credit booked on the call that earns it", {"unmeasured": m.get("checkpoints_unmeasured"), "graded": m.get("checkpoints_graded")})
        lost = m.get("compactions_losing_phi")
        lac = (monitor_modes or {}).get("lost_after_compaction")
        if (lost is not None and lost > 0.2) or (lac is not None and lac > 0.1):
            prop("compaction", lane, "carry STATUS.json, the milestone vector and the plan into the compaction hand-off; compact after a commit, not mid-edit",
                 "fewer compactions that lose Φ", {"compactions_losing_phi": lost, "lost_after_compaction": lac, "phi_delta_at_compaction": m.get("phi_delta_at_compaction")})
        cu = m.get("context_use_max")
        if cu is not None and cu > 0.85:
            prop("compaction", lane, "lower the compaction threshold (the context runs near the window)", "no token-limit stops; compaction before the window fills",
                 {"context_use_max": cu})
        tr = m.get("truncated_tool_outputs")
        if tr is not None and tr > 0.1:
            prop("tool_truncation", lane, "raise the per-result limit for file reads, keep it for test logs (show the failing lines and the tail)",
                 "fewer re-reads of truncated files; fewer tool calls per milestone", {"truncated_tool_outputs": tr, "input_tokens_per_turn": m.get("input_tokens_per_turn")})
        elif m.get("input_tokens_per_turn") and m["input_tokens_per_turn"] > 60000 and (tr or 0) < 0.02:
            prop("tool_truncation", lane, "lower the per-result limit for test logs (most of the context is tool output)", "lower tokens per turn at the same Φ",
                 {"input_tokens_per_turn": m["input_tokens_per_turn"]})
        lf = m.get("length_finish_share")
        if lf is not None and lf > 0.002:
            prop("sampler", lane, "raise the per-call output-token limit (calls cut at the limit end in a malformed or empty tool call)",
                 "no calls finishing on length; fewer invalid-argument faults", {"length_finish_share": lf})
        ms = m.get("model_share_of_agent_time")
        if ms is not None and ms > 0.7:
            prop("concurrency", lane, "raise sampler concurrency (the episodes wait on the model most of the time)", "more episodes per wall hour",
                 {"model_share_of_agent_time": ms, "model_s_per_call": m.get("model_s_per_call")})
    tool_calls: Counter[str] = Counter()
    tool_bad: Counter[str] = Counter()
    tool_fams: dict[str, set[str]] = defaultdict(set)
    for v in views:
        for st in v.b.steps:
            tool_calls[st.function] += 1
            if st.fault == "tool" and AGENT_INPUT.search(st.result or "") and not NOT_FOUND.search(st.result or ""):
                tool_bad[st.function] += 1
                tool_fams[st.function].add(v.ep.family)
    for fn, bad in tool_bad.most_common():
        share = bad / max(1, tool_calls[fn])
        if share > 0.05 and len(tool_fams[fn]) >= 2:
            first = next((st.result for v in views for st in v.b.steps if st.function == fn and st.fault == "tool" and AGENT_INPUT.search(st.result or "")), "")
            prop("tool_affordance", "all", f"{fn}: {share:.0%} of calls fail on the agent's own arguments across {len(tool_fams[fn])} families; make the tool forgiving "
                 "(a regex error retried as a literal string; a missed old_string answered with the closest matching lines)", "fewer wasted tool calls; a faster first edit",
                 {"calls": tool_calls[fn], "agent_input_errors": bad, "families": sorted(tool_fams[fn]), "example": norm(error_line(first or ""), 120)})
    rates: dict[str, list[bool]] = defaultdict(list)
    for v in views:
        if not v.ep.masked and v.ep.split == "train":
            rates[v.ep.task].append(v.ep.passed)
    ps = [sum(x) / len(x) for x in rates.values() if len(x) >= 4]
    inb = [p for p in ps if band[0] <= p <= band[1]]
    if inb:
        cur = sum(spread_probability(p, k) for p in inb) / len(inb)
        best = next((kk for kk in (4, 6, 8, 12, 16) if sum(spread_probability(p, kk) for p in inb) / len(inb) >= 0.85), 16)
        if best != k:
            prop("group_size", "all", f"group size k = {best} instead of {k}", f"groups with spread {cur:.0%} -> "
                 f"{sum(spread_probability(p, best) for p in inb) / len(inb):.0%} on in-band tasks", {"in_band_tasks": len(inb), "pass_rates": [round(p, 3) for p in inb]})
    if ps:
        share = len(inb) / len(ps)
        if share < 0.5:
            ib = in_band_tasks([v.ep for v in views], band=band)
            prop("difficulty_band", "all", f"only {len(inb)}/{len(ps)} measured training tasks are in the {band} band: run the difficulty controller (checkpoint starts for "
                 "the bimodal ones) before the next step", "more groups with spread per step", {"measured": len(ps), "in_band": len(inb), "latest_in_band": len(ib["in_band"])})
    return out


# ---- the health report --------------------------------------------------------------------------------------------------------------
def health_report(views: Sequence[View], *, roots: Sequence[Path | str] = (), run_dirs: Sequence[Path | str] = (), store: Path | str | None = None,
                  journal: Sequence[Mapping[str, Any]] = (), monitor_modes: Mapping[str, float] | None = None, k: int = 8) -> dict[str, Any]:
    """The ranked defect list, the metadata, the tuning proposals, the versions per step, and ``pause`` (any open CRITICAL)."""
    hits = [h for v in views for h in per_attempt(v)]
    hits += stale_tests(views) + unreachable(views) + spec_defects(views) + codec(views) + fairness(roots) + trainer_hits(run_dirs) + log_hits(roots)
    hits += unreadable_hits(roots, views)
    if store:
        hits += repath_flips(views, store)
    defects = cluster(hits, views)
    vers = versions(views)
    steps_with = {at: r for at, r in vers.items() if at.startswith("step ")}
    mixed = sorted({ver for r in steps_with.values() for ver in r["path_rules"]})
    if len(mixed) > 1 and not any(r.get("component") == "grader" for r in journal):
        defects.append(Defect(sig_id("version_mix", "grader", "|".join(mixed)), "version_mix", "grader", "framework", "HIGH",
                              f"path rules changed mid-run with no fix record: {mixed}", [], [], {}, {}, {"versions": mixed}))
    apply_fix_journal(defects, views, journal)
    defects = [redact(d, views) for d in defects]
    open_crit = [d for d in defects if d.severity == "CRITICAL" and d.cls != "agent" and not d.status.startswith("fixed")]
    md = metadata(views)
    train = [v for v in views if v.ep.split == "train"]  # tuning proposals read training traces only (no held-out content in their evidence)
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "episodes": len(views), "masked": sum(v.ep.masked for v in views),
            "defects": [d.to_json() for d in defects], "by_component": dict(Counter(d.component for d in defects)),
            "by_severity": dict(Counter(d.severity for d in defects)), "agent_vs_framework": dict(Counter(d.cls for d in defects)),
            "pause": bool(open_crit), "pause_reason": [f"{d.id} {d.kind} ({d.component}): {d.signature}" for d in open_crit[:10]],
            "regrade_plans": [regrade_plan(d, views) for d in defects if d.severity == "CRITICAL" and not d.heldout],
            "metadata": md, "tuning": tuning(metadata(train), train, k=k, monitor_modes=monitor_modes), "versions": vers}


def read_journal(run_dirs: Sequence[Path | str]) -> list[dict[str, Any]]:
    return [r for rd in run_dirs for r in _jsonl(fixes_path(rd))]
