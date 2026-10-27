"""Failure targeting: the task fails for the intended reason (swarm-pipeline/00 §7, §4 S15, §15 P4).

A task is pool-eligible only when the target rung fails it *because of the behaviour the intent names*. S15 checks that
with four executed probes and two measured conditions, all on the same policy:

1. **Band** (§7.2, report 11 F1): the target rung's outcome reward has spread across its group (``band``); otherwise
   GRPO and turn credit get nothing. The pass rate is reported, never gated alone.
2. **Cause mix** (§7.3): >= 70% of the trusted auditor labels on its failures are ``skill_gap`` (``cause_mix``; the
   P3 auditor with seeded controls; masked attempts are not labels).
3. **Counterfactual control** (§7.4, ``counterfactual``): the same policy on ``simplify(kind)`` (the difficulty the task
   is about removed) against the full task. The unit is the *target milestone*: it succeeds when it passes with every
   milestone it ``requires`` passing (Phi semantics), so on the full task the policy must also do the prerequisites.
   - **attributed**: the simplified task succeeds significantly more often (one-sided Fisher exact test on success,
     ``alpha``) or, when Phi over the *target set* is graded (several targets, some attempts in between), the bootstrap
     lower bound of its gain is above 0. With a binary Phi (one target) the bootstrap would only approximate the exact test,
     liberally at n = 4, so Fisher alone decides. The Phi of every other kept milestone is never compared: a prefix simplification makes them
     easier too, so their gain says nothing about the intended axis (P4's first run attributed parse on exactly that);
   - **not_attributed**: the policy also fails the simplified task: its success count is significantly below the rate
     an easy task would give (``p_ref``; one-sided binomial). The failure is not on the intended axis: back to S2;
   - **underpowered**: neither. No gate; flagged, with the power of the Fisher test at the reference effect.
4. **Prerequisite probe** (§7.5, ``prerequisite_probe``): a scripted check of the World (it opens, the existing tests
   run through the World's own shell tool, a file can be written and removed) plus one tiny episode of the policy asked
   to run the tests and report the count. The policy must run a test command and report the count the scripted run
   measured. Failing means the intent card is wrong for this policy (it cannot do the basics the intent assumes).
5. **Paraphrase stability** (§7.6, ``check_paraphrase``, ``stability``): two paraphrases of the request (same requirement
   ids, same milestones, ledger and lint re-checked, every public name the hidden tests import still stated) give a
   full-pass rate inside the original's Wilson CI and a mean Phi inside its bootstrap CI.

**Simplify kinds per generator** (``SIMPLIFY_KINDS``; the ``Simplifiable`` Protocol is what a generator implements):
windows have ``prerequisites`` (P0: the executed prerequisite units pre-applied) and ``locate`` (P4: the prompt names the
files and functions of the target milestone's unit); stitched windows ``host_change``, rebreak ``locate`` (the faulty
function named), open tasks ``baseline_bar`` (the bar at the baseline). Those three generators arrive in P5 and plug in by
implementing ``simplify(kind, milestones=...)``.

**S15** (``s15_target``, ``target_machine``): the machine handler reads a ``TargetEvidence`` from ``ctx.services`` and
settles ``{"verdict": {ok, route, owner, findings, flags, evidence}}``. ``route`` is ``S2`` (wrong difficulty, wrong
intent card, unstable prompt) or ``drop`` (a task the policy cannot even start in a correct World); ``underpowered``
counterfactuals and an unmeasured cause mix are flags, never gates.

The seeded tooling-only failure (``seeded_tooling``): the policy's World has its test command hidden (pytest removed) and
its workspace mounted read-only, the grader untouched. Intent-irrelevant by construction, it must come out
``not_attributed`` (the policy fails the simplified task too) and fail the prerequisite probe.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ---- simplify kinds -------------------------------------------------------------------------------------------------------
SIMPLIFY_KINDS: dict[str, tuple[str, ...]] = {
    "mined_window": ("prerequisites", "locate"),
    "swarm_window": ("prerequisites", "locate"),
    "stitched_window": ("host_change", "locate"),
    "rebreak": ("locate",),
    "open": ("baseline_bar",),
}


@runtime_checkable
class Simplifiable(Protocol):
    """What a generator's task implements for the counterfactual control: a new task with one difficulty removed, the
    same checks for what is left, the same World image and grading mode; ``milestones`` are the targets."""

    def simplify(self, kind: str, *, milestones: Sequence[str] | None = None, root: Path | None = None) -> Any: ...


def generator_of(task: Any) -> str:
    """The generator a task came from: its ``generator`` attribute, else ``mined_window`` for a window task."""
    return str(getattr(task, "generator", None) or ("mined_window" if isinstance(getattr(task, "window", None), Mapping) else "unknown"))


def simplify(task: Any, kind: str, *, targets: Sequence[str] | None = None, root: Path | None = None) -> Any:
    """``task.simplify(kind)`` after checking the kind belongs to the task's generator."""
    gen = generator_of(task)
    allowed = SIMPLIFY_KINDS.get(gen, ())
    if kind not in allowed:
        raise ValueError(f"generator {gen!r} simplifies by {allowed}, not {kind!r}")
    if not isinstance(task, Simplifiable):
        raise TypeError(f"{type(task).__name__} does not implement simplify()")
    kw: dict[str, Any] = {"milestones": list(targets) if targets else None}
    if kind == "prerequisites":
        kw["root"] = root
    return task.simplify(kind, **kw)


# ---- outcomes -------------------------------------------------------------------------------------------------------------
@dataclass
class Outcome:
    """One attempt, read at the target: ``success`` (every target milestone passes with its ``requires``), ``phi_set``
    (the share of the compared milestone set passing with their ``requires``), the task's own ``phi`` and ``reward``.
    ``masked`` attempts (infrastructure, V8) are dropped from every statistic."""

    success: bool
    phi_set: float
    phi: float
    reward: float
    full_pass: bool = False
    masked: bool = False
    stop: str = ""
    usd: float = 0.0
    id: str = ""


def outcome(milestones: Mapping[str, bool] | None, requires: Mapping[str, Sequence[str]], targets: Sequence[str], compare: Sequence[str], *,
            phi: float | None = None, reward: float | None = None, masked: bool = False, stop: str = "", usd: float = 0.0, oid: str = "",
            full_pass: bool | None = None) -> Outcome:
    """An ``Outcome`` from a graded milestone vector. A milestone missing from ``milestones`` (pre-applied in a simplified
    task) counts as passing: the simplified start already holds it."""
    if masked or milestones is None:
        return Outcome(False, 0.0, 0.0, 0.0, masked=True, stop=stop, usd=usd, id=oid)

    def ok(m: str, seen: frozenset[str] = frozenset()) -> bool:
        if m not in milestones:
            return True
        return bool(milestones[m]) and all(ok(r, seen | {m}) for r in requires.get(m) or [] if r not in seen)

    succ = all(ok(t) for t in targets)
    ps = sum(1 for m in compare if ok(m)) / len(compare) if compare else 0.0
    return Outcome(succ, round(ps, 6), float(phi or 0.0), float(reward or 0.0), full_pass=bool(full_pass if full_pass is not None else (reward == 1.0)),
                   stop=stop, usd=usd, id=oid)


def outcomes_of(episodes: Sequence[Mapping[str, Any]], task: Any, targets: Sequence[str], compare: Sequence[str]) -> list[Outcome]:
    """``Outcome`` per episode row (``scripts/lh_run.episodes_of``: ``milestones``, ``phi``, ``reward``, ``masked``)."""
    req = {m["id"]: list(m.get("requires") or []) for m in task.milestones}
    out = []
    for e in episodes:
        masked = bool(e.get("masked")) or e.get("milestones") is None
        out.append(outcome(e.get("milestones"), req, targets, compare, phi=e.get("phi"), reward=e.get("reward"), masked=masked, stop=str(e.get("stop") or ""),
                           usd=float(e.get("usd") or 0.0), oid=str(e.get("sample") or "")))
    return out


# ---- statistics ---------------------------------------------------------------------------------------------------------
def binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Binomial(n, p)."""
    return min(1.0, sum(binom_pmf(i, n, p) for i in range(min(k, n) + 1)))


def fisher_one_sided(k_s: int, n_s: int, k_f: int, n_f: int) -> float:
    """One-sided Fisher exact p-value for "the simplified arm succeeds more often": P(X >= k_s) under the hypergeometric
    with both arms' margins fixed."""
    big_n, big_k = n_s + n_f, k_s + k_f
    if n_s == 0 or n_f == 0:
        return 1.0
    denom = math.comb(big_n, n_s)
    return min(1.0, sum(math.comb(big_k, x) * math.comb(big_n - big_k, n_s - x) for x in range(k_s, min(big_k, n_s) + 1)) / denom)


def fisher_power(p_f: float, p_s: float, n_f: int, n_s: int, *, alpha: float = 0.05) -> float:
    """Exact power of ``fisher_one_sided`` at level ``alpha`` when the arms' true success rates are ``p_f`` and ``p_s``."""
    pw = 0.0
    for x_s in range(n_s + 1):
        ps = binom_pmf(x_s, n_s, p_s)
        for x_f in range(n_f + 1):
            if fisher_one_sided(x_s, n_s, x_f, n_f) <= alpha:
                pw += ps * binom_pmf(x_f, n_f, p_f)
    return round(pw, 6)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(xs: Sequence[float], *, level: float = 0.95, b: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    if not xs:
        return (0.0, 1.0)
    rnd = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rnd.randrange(n)] for _ in range(n)) / n for _ in range(b))
    lo = means[int((1 - level) / 2 * b)]
    hi = means[min(b - 1, int((1 + level) / 2 * b))]
    return (round(lo, 6), round(hi, 6))


def bootstrap_diff_ci(a: Sequence[float], s: Sequence[float], *, level: float = 0.90, b: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of mean(s) - mean(a), each arm resampled on its own. At ``level`` 0.90 the lower bound is a
    one-sided 95% bound."""
    if not a or not s:
        return (-1.0, 1.0)
    rnd = random.Random(seed)
    diffs = sorted(sum(s[rnd.randrange(len(s))] for _ in s) / len(s) - sum(a[rnd.randrange(len(a))] for _ in a) / len(a) for _ in range(b))
    return (round(diffs[int((1 - level) / 2 * b)], 6), round(diffs[min(b - 1, int((1 + level) / 2 * b))], 6))


# ---- 7.4 the counterfactual control --------------------------------------------------------------------------------------
@dataclass
class CounterfactualConfig:
    alpha: float = 0.05
    p_ref: float = 0.8  # the success rate an easy (simplified) task should give the policy
    min_power: float = 0.8
    bootstrap: int = 4000
    seed: int = 0


def counterfactual(full: Sequence[Outcome], simple: Sequence[Outcome], *, config: CounterfactualConfig | None = None) -> dict[str, Any]:
    """The §7.4 verdict (module doc) with every number: counts, Fisher p, the bootstrap CI on the Phi gain over the target
    set, the binomial p of the simplified arm against ``p_ref``, and the power at the reference effect."""
    cfg = config or CounterfactualConfig()
    f = [o for o in full if not o.masked]
    s = [o for o in simple if not o.masked]
    n_f, n_s = len(f), len(s)
    k_f, k_s = sum(o.success for o in f), sum(o.success for o in s)
    row: dict[str, Any] = {"n_full": n_f, "n_simple": n_s, "success_full": k_f, "success_simple": k_s,
                           "masked": {"full": len(full) - n_f, "simple": len(simple) - n_s}}
    if not n_f or not n_s:
        return {**row, "verdict": "underpowered", "why": "an arm has no unmasked attempt", "route": None}
    p = fisher_one_sided(k_s, n_s, k_f, n_f)
    dphi = statistics.fmean(o.phi_set for o in s) - statistics.fmean(o.phi_set for o in f)
    ci = bootstrap_diff_ci([o.phi_set for o in f], [o.phi_set for o in s], level=1 - 2 * cfg.alpha, b=cfg.bootstrap, seed=cfg.seed)
    p_low = binom_cdf(k_s, n_s, cfg.p_ref)
    power = fisher_power(k_f / n_f, cfg.p_ref, n_f, n_s, alpha=cfg.alpha)
    # the smallest simplified success count that would have been significant against this full arm
    need = next((x for x in range(n_s + 1) if fisher_one_sided(x, n_s, k_f, n_f) <= cfg.alpha), None)
    row.update(rate_full=round(k_f / n_f, 4), rate_simple=round(k_s / n_s, 4), fisher_p=round(p, 6), dphi=round(dphi, 6), dphi_ci=list(ci),
               ci_level=round(1 - 2 * cfg.alpha, 4), simple_vs_ref_p=round(p_low, 6), p_ref=cfg.p_ref, power_at_ref=power,
               min_simple_successes_for_significance=need, alpha=cfg.alpha)
    graded = any(0.0 < o.phi_set < 1.0 for o in [*f, *s])  # with a binary Phi (one target) the exact test alone decides
    row["phi_route"] = graded
    if p <= cfg.alpha or (graded and ci[0] > 0):
        why = f"simplified {k_s}/{n_s} vs full {k_f}/{n_f}: Fisher p {p:.4f}" + (f"; Phi gain CI {list(ci)} above 0" if ci[0] > 0 else "")
        return {**row, "verdict": "attributed", "why": why, "route": None}
    if p_low <= cfg.alpha:
        return {**row, "verdict": "not_attributed", "route": "S2",
                "why": f"the policy fails the simplified task too ({k_s}/{n_s}; P(X <= {k_s} | p={cfg.p_ref}) = {p_low:.4f}): not the intended axis"}
    return {**row, "verdict": "underpowered", "route": None,
            "why": f"no significant difference (Fisher p {p:.3f}, Phi gain CI {list(ci)}) and the simplified arm is not clearly failing "
                   f"(p {p_low:.3f}); power at the reference effect {power:.2f}" + (" < " + str(cfg.min_power) if power < cfg.min_power else "")}


# ---- 7.5 the prerequisite probe ----------------------------------------------------------------------------------------------
PROBE_PROMPT = ("Please run this repository's existing test suite and tell me how many tests passed. Do not change any file. "
                "End your final message with a line of the form PASSED=<number>.")
TEST_COMMAND = re.compile(r"\b(pytest|py\.test|unittest|tox|nox|nosetests|runtests\.py|make\s+test|setup\.py\s+test)\b|\btest_\w+\.py\b|\btests?/")
PASSED_RE = re.compile(r"PASSED\s*=\s*(\d+)")
PYTEST_PASSED = re.compile(r"(\d+) passed")
UNITTEST_RAN = re.compile(r"Ran (\d+) tests?")


def suite_command(task: Any) -> str:
    """The scripted probe's test command: A's own test files through pytest, as the hidden verifier runs them."""
    from crucible.factory.lanes.windows import suite_targets

    targets = suite_targets(task.base_dir, {})
    return "python -m pytest -q -p no:cacheprovider -o addopts= " + " ".join(targets)


async def scripted_probe(task: Any, workdir: Path, *, command: str | None = None, timeout_s: int = 600) -> dict[str, Any]:
    """Open the task's World (the policy's own) and drive its tools as a scripted policy: the suite through ``repo.shell``,
    then a scratch file written with ``repo.write_file`` and removed. ``ok`` needs the World to open, the tests to run
    (a pytest summary with passes) and the write to land."""
    from crucible.core.schema.actions import ToolCall

    cmd = command or suite_command(task)
    out: dict[str, Any] = {"command": cmd}
    try:
        world = await task.open(Path(workdir))
    except Exception as e:  # noqa: BLE001 -- a World that does not open fails the probe, with the reason
        return {**out, "ok": False, "world_opened": False, "why": f"the World did not open: {type(e).__name__}: {e}"[:300]}
    try:
        out["world_opened"] = True

        async def call(fn: str, **args: Any) -> Any:
            res = await world.step("agent", ToolCall(tool_call_id=f"probe-{fn}", function_name=fn, arguments=args))
            return res.observation.tool_results[0] if res.observation.tool_results else None

        r = await call("repo.shell", command=cmd, timeout_s=timeout_s)
        text = str(getattr(r, "content", "") or "")
        m = PYTEST_PASSED.findall(text)
        out["test_output_tail"] = text[-600:]
        out["passed"] = int(m[-1]) if m else None
        out["tests_ran"] = bool(m) and int(m[-1]) > 0
        # the repository's own configuration (its addopts: doctests, plugins) counts differently; both are the suite
        r2 = await call("repo.shell", command="python -m pytest -q -p no:cacheprovider", timeout_s=timeout_s)
        m2 = PYTEST_PASSED.findall(str(getattr(r2, "content", "") or ""))
        out["passed_repo_config"] = int(m2[-1]) if m2 else None
        out["truths"] = sorted({x for x in (out["passed"], out["passed_repo_config"]) if x})
        w = await call("repo.write_file", path=".crucible_probe.txt", content="probe\n")
        out["write_ok"] = bool(getattr(w, "ok", False))
        out["write_result"] = str(getattr(w, "content", "") or getattr(w, "error", ""))[:200]
        if out["write_ok"]:
            await call("repo.shell", command="rm -f .crucible_probe.txt", timeout_s=60)
    finally:
        await world.close()
    out["ok"] = bool(out.get("tests_ran") and out.get("write_ok"))
    out["why"] = "" if out["ok"] else ("the existing tests do not run through the World's shell" if not out.get("tests_ran") else "the World's tools cannot write")
    return out


def probe_episode(calls: Sequence[Mapping[str, Any]], final_message: str, truth: int | Sequence[int] | None, *, tolerance: float = 0.1) -> dict[str, Any]:
    """The policy's probe episode judged on what it did: it ran a test command through its shell, and its ``PASSED=<n>``
    is within ``tolerance`` of a scripted count (``truth``: one count or several, e.g. the verifier's invocation and the
    repository's own configuration) or of a count one of its own test runs printed (``calls[i]["result"]``: another valid
    invocation, such as a tox command with doctests). An unmeasured truth fails: the World's suite did not run."""
    truths = [int(t) for t in (truth if isinstance(truth, Sequence) else [truth]) if t]
    shells = [c for c in calls if re.search(r"shell|bash|exec|run", str(c.get("function") or ""))]
    ran_calls = [c for c in shells if TEST_COMMAND.search(str((c.get("arguments") or {}).get("command") or ""))]
    ran = [str((c.get("arguments") or {}).get("command") or "") for c in ran_calls]
    # counts the policy's own test runs printed (another valid invocation, e.g. the repository's tox command with doctests)
    seen = sorted({int(x) for c in ran_calls for x in [*PYTEST_PASSED.findall(str(c.get("result") or "")), *UNITTEST_RAN.findall(str(c.get("result") or ""))]})
    m = PASSED_RE.findall(final_message or "")
    reported = int(m[-1]) if m else None
    count_ok = reported is not None and any(abs(reported - t) <= max(1, tolerance * t) for t in [*truths, *seen]) and bool(truths)
    ok = bool(ran) and count_ok
    why = "" if ok else ("no test command was run" if not ran else ("no PASSED=<n> line" if reported is None else
                                                                     f"reported {reported}, the suite has {truths or 'an unmeasured number of'} passing"))
    return {"ok": ok, "ran_tests": bool(ran), "test_commands": ran[:5], "reported": reported, "truth": truths, "seen_in_own_runs": seen, "why": why,
            "tool_calls": len(calls)}


def prerequisite_probe(scripted: Mapping[str, Any], episode: Mapping[str, Any] | None) -> dict[str, Any]:
    """§7.5: the World passes the scripted probe and the policy's episode passes ``probe_episode``. A World failure is the
    World's (S3); a policy failure means the intent card is wrong for this policy."""
    if not scripted.get("ok"):
        return {"ok": False, "owner": "world", "why": f"scripted probe: {scripted.get('why')}", "scripted": dict(scripted), "episode": episode}
    if episode is None:
        return {"ok": False, "owner": "policy", "why": "no probe episode", "scripted": dict(scripted), "episode": None}
    if not episode.get("ok"):
        return {"ok": False, "owner": "policy", "why": f"policy probe: {episode.get('why')}", "scripted": dict(scripted), "episode": dict(episode)}
    return {"ok": True, "owner": None, "why": "", "scripted": dict(scripted), "episode": dict(episode)}


# ---- 7.6 paraphrase stability ---------------------------------------------------------------------------------------------
PARAPHRASE_SYSTEM = ("You rewrite a software maintainer's release request in different words. Keep every requirement's meaning, "
                     "every public name (function, class, method, argument and option names, exactly as written, in backticks when "
                     "the original has them), every value and every example. Do not add or drop a requirement, a hint or a detail. "
                     "Change the wording and sentence structure only. Reply with JSON only.")


def paraphrase_messages(task: Any, k: int) -> list[dict[str, str]]:
    opening, closing, _fixed = split_request(task)
    reqs = [{"id": str(r["id"]), "text": str(r["text"])} for r in task.requirements if r.get("milestone")]
    style = ("Use plain, direct sentences." if k % 2 else "Use a conversational tone, as in an issue comment.")
    body = {"opening": opening, "requirements": reqs, "closing": closing}
    return [{"role": "system", "content": PARAPHRASE_SYSTEM},
            {"role": "user", "content": f"Paraphrase {k}. {style} Return {{\"opening\": str, \"requirements\": [{{\"id\": str, \"text\": str}}], "
                                         f"\"closing\": str}} with the same ids in the same order.\n\n{json.dumps(body, indent=1)}"}]


FIXED_SENTENCES = (re.compile(r"^The release must also expose these public names: .*$", re.MULTILINE),)


def split_request(task: Any) -> tuple[str, str, str]:
    """(opening, closing, fixed) of a window prompt: the text before the first requirement line, the text between the last
    one and the epilogue, and the lane's fixed sentences of that closing (the names line, ``REGRESSION_TEXT``), which a
    paraphrase keeps verbatim. The epilogue is never paraphrased."""
    from crucible.factory.lanes.executable import REGRESSION_TEXT
    from crucible.factory.lanes.windows import EPILOGUE

    text = task.instruction
    e = text.find(EPILOGUE[:40])
    req_text = text[: e if e >= 0 else len(text)]
    lines = req_text.splitlines(keepends=True)
    idx = [i for i, ln in enumerate(lines) if re.match(r"- \[[^\]]+\] ", ln)]
    if not idx:
        return req_text.strip(), "", ""
    opening = re.sub(r"\n*What the release needs:\s*$", "", "".join(lines[: idx[0]]).strip()).strip()
    closing = "".join(lines[idx[-1] + 1:]).strip()
    fixed = [m.group(0) for rx in FIXED_SENTENCES for m in rx.finditer(closing)]
    for f in fixed:
        closing = closing.replace(f, "")
    if REGRESSION_TEXT in closing:
        closing = closing.replace(REGRESSION_TEXT, "")
        fixed.append(REGRESSION_TEXT)
    return re.sub(r"\n{3,}", "\n\n", opening), re.sub(r"\n{3,}", "\n\n", closing).strip(), "\n\n".join(fixed)


def required_public_names(task: Any) -> list[str]:
    """Public names the hidden tests import from the project that the original prompt states: a paraphrase must keep them."""
    from crucible.factory.lanes.windows import required_names

    rn = required_names(task.window.get("tests_b") or {}, task.base_dir)
    names = sorted({n for k in ("module_level", "in_function", "attribute") for n in rn.get(k) or []})
    return [n for n in names if re.search(rf"\b{re.escape(n)}\b", task.instruction)]


def check_paraphrase(task: Any, para: Mapping[str, Any]) -> dict[str, Any]:
    """The paraphrase as a prompt (``compose_window_prompt``: the paraphrased opening, requirements and closing, then the
    lane's fixed sentences and the original epilogue) and its checks: same requirement ids in order, each keeps its
    milestone, the ledger two-way (``window_ledger``), ``lint_prompt`` clean, every required public name the original stated
    still stated. ``ok`` needs all of them."""
    from crucible.factory.lanes.windows import compose_window_prompt, lint_prompt, locate, window_checkset, window_ledger

    orig = [r for r in task.requirements if r.get("milestone")]
    got = [r for r in para.get("requirements") or [] if isinstance(r, Mapping)]
    if [str(r.get("id")) for r in got] != [str(r["id"]) for r in orig]:
        return {"ok": False, "findings": [f"requirement ids changed: {[r.get('id') for r in got]} != {[r['id'] for r in orig]}"]}
    by_id = {str(r["id"]): r for r in task.requirements}
    reqs = [{**by_id[str(r["id"])], "text": " ".join(str(r.get("text") or "").split())} for r in got]
    reqs += [dict(r) for r in task.requirements if not r.get("milestone")]
    _o, _c, fixed = split_request(task)
    closing = "\n\n".join(x for x in (" ".join(str(para.get("closing") or "").split()), fixed) if x)
    prompt = compose_window_prompt(" ".join(str(para.get("opening") or "").split()), [r for r in reqs if r.get("milestone")], closing)
    spans = {str(r["id"]): list(loc) for r in reqs if (loc := locate(prompt, r["text"]))}
    _led, problems = window_ledger(task.window, reqs, spans, prompt)
    findings = [f"ledger: {p}" for p in problems]
    findings += [f"lint: {x}" for x in lint_prompt("\n".join([str(para.get("opening") or ""), str(para.get("closing") or ""),
                                                               *[r["text"] for r in reqs if r.get("milestone")]]))]
    findings += [f"required name no longer stated: {n}" for n in required_public_names(task) if not re.search(rf"\b{re.escape(n)}\b", prompt)]
    if not str(para.get("opening") or "").strip():
        findings.append("empty opening")
    return {"ok": not findings, "findings": findings, "prompt": prompt, "requirements": reqs, "spans": spans,
            "checkset": window_checkset(task.window, reqs)}


def paraphrased_task(task: Any, built: Mapping[str, Any], tag: str) -> Any:
    """The window task under a checked paraphrase (same window, World, checks; another prompt)."""
    from crucible.factory.lanes.windows import WindowTask

    t = WindowTask(task.lane, task.mined, task.window, task_id=f"{task.task_id}~{tag}", prompt=built["prompt"], requirements=built["requirements"],
                   spans=built["spans"], checkset=built["checkset"], phi_weight=task.phi_weight, import_tolerant=task.import_tolerant)
    t.world_image, t.workspace_mode = task.world_image, task.workspace_mode
    return t


def stability(original: Sequence[Outcome], para: Sequence[Outcome], *, b: int = 4000, seed: int = 0) -> dict[str, Any]:
    """§7.6: the paraphrase's full-pass rate inside the original's Wilson 95% CI and its mean Phi inside the original's
    bootstrap 95% CI. An arm with no unmasked attempt is unmeasured (``stable`` None)."""
    o = [x for x in original if not x.masked]
    p = [x for x in para if not x.masked]
    if not o or not p:
        return {"stable": None, "why": "an arm has no unmasked attempt", "n_original": len(o), "n_paraphrase": len(p)}
    lo, hi = wilson(sum(x.full_pass for x in o), len(o))
    rate = sum(x.full_pass for x in p) / len(p)
    plo, phi_hi = bootstrap_ci([x.phi for x in o], b=b, seed=seed)
    mphi = statistics.fmean(x.phi for x in p)
    eps = 1e-9
    in_rate = lo - eps <= rate <= hi + eps
    in_phi = plo - eps <= mphi <= phi_hi + eps
    return {"stable": in_rate and in_phi, "n_original": len(o), "n_paraphrase": len(p), "pass_rate": round(rate, 4), "pass_ci": [round(lo, 4), round(hi, 4)],
            "mean_phi": round(mphi, 6), "phi_ci": [plo, phi_hi], "original_mean_phi": round(statistics.fmean(x.phi for x in o), 6),
            "why": "" if in_rate and in_phi else ("pass rate outside the original's CI" if not in_rate else "mean Phi outside the original's CI")}


# ---- 7.2 / 7.3 band and cause mix ---------------------------------------------------------------------------------------------
def band(rewards: Sequence[float | None]) -> dict[str, Any]:
    """Report 11 F1: the target rung's group has outcome-reward spread (not all equal); the pass rate is reported."""
    rs = [round(float(r), 6) for r in rewards if r is not None]
    return {"n": len(rs), "rewards": rs, "in_band": len(set(rs)) > 1, "pass_rate": (sum(r == 1.0 for r in rs) / len(rs)) if rs else None,
            "reward_range": [min(rs), max(rs)] if rs else None}


NOT_LABELS = ("abstain", "env_flake")
# the keep rule's on-intent bar: a majority of the labelled failures are the intended skill gap (the lead lowered it
# from 0.7, 2026-10-27: with 8 attempts, 0.7 dropped a task whose failures were 4 skill gaps and 3 budget stops)
SKILL_MIN = 0.5


def cause_mix(labels: Mapping[str, int] | Sequence[str], *, skill_min: float = SKILL_MIN, min_labels: int = 3) -> dict[str, Any]:
    """§7.3: the skill-gap share of the trusted labels (masked and abstained attempts are not labels). Fewer than
    ``min_labels`` labels is unmeasured (``ok`` None): a flag, never a pass."""
    counts = dict(labels) if isinstance(labels, Mapping) else dict(__import__("collections").Counter(labels))
    labelled = {k: int(v) for k, v in counts.items() if k not in NOT_LABELS and v}
    n = sum(labelled.values())
    share = labelled.get("skill_gap", 0) / n if n else None
    ok = None if n < min_labels else bool(share is not None and share >= skill_min)
    return {"labels": labelled, "n": n, "skill_gap_share": None if share is None else round(share, 4), "skill_min": skill_min, "ok": ok}


# ---- the keep rule (the task-factory contract: training band, benchmark ceiling, intent, fairness) ----------------------------
@dataclass(frozen=True)
class KeepRule:
    """The task-factory contract's keep rule (user, 2026-10-20). A **training** task: the target student's pass rate over
    its k attempts in [``train_lo``, ``train_hi``], >= ``skill_min`` of its failures a skill gap on the task's intended
    intent, and a frontier solver passes it. A **benchmark** (held-out) task: pass rate < ``bench_hi`` with the same intent
    and fairness conditions. Flags: 0 and frontier-solvable -> ``simplify(locate|prerequisites)``; above ``train_hi`` ->
    too easy."""

    train_lo: float = 0.1
    train_hi: float = 0.6
    bench_hi: float = 0.3
    skill_min: float = SKILL_MIN
    min_labels: int = 3  # fewer labelled failures: the intent share is unmeasured (never a pass)


def band_power(passes: int, n: int, *, split: str, rule: KeepRule | None = None, n_max: int = 64) -> dict[str, Any]:
    """Is the keep rule's band decision settled by ``n`` attempts (the S15 analogue of a benchmark's MDE gate,
    ``bench.power``)? ``crosses``: the band edges the split's rule compares with (training ``train_lo`` / ``train_hi``,
    held out ``bench_hi``) that lie strictly inside the Wilson 95% interval of the pass rate; ``decided`` when none does.
    ``n_needed``: the attempts at the same pass rate after which the interval clears them (``n`` when decided; None
    beyond ``n_max``, e.g. a pass rate on an edge). Reported beside the verdict, never a gate."""
    r = rule or KeepRule()
    if n <= 0:
        return {"decided": False, "crosses": [], "n": 0, "n_needed": None, "ci": None}
    edges = [r.bench_hi] if split == "heldout" else [r.train_lo, r.train_hi]
    lo, hi = wilson(passes, n)
    crosses = [e for e in edges if lo < e < hi]
    p = passes / n
    need: int | None = n if not crosses else None
    if crosses:
        for m in range(n + 1, n_max + 1):
            lo_m, hi_m = wilson(round(p * m), m)
            if not any(lo_m < e < hi_m for e in crosses):
                need = m
                break
    return {"decided": not crosses, "crosses": crosses, "n": n, "n_needed": need, "ci": [round(lo, 4), round(hi, 4)]}


def keep_verdict(*, split: str, passes: int, n: int, on_intent: int, labelled: int, frontier: Mapping[str, Sequence[bool]] | None,
                 rule: KeepRule | None = None, world_issue: str | None = None, spec_issue: str | None = None) -> dict[str, Any]:
    """The keep rule on one task's measured evidence (``KeepRule``). ``passes`` of ``n`` unmasked student attempts;
    ``on_intent`` of ``labelled`` failures are a trusted skill-gap label whose trace evidence is one of the task's intent
    skills (masked attempts and routed failures are not labels); ``frontier``: solver -> its attempts' full passes
    (None or empty: not run); ``world_issue``: an open World defect found in the traces (the task goes back to the World's
    owner whatever the numbers say: ``world_bug``); ``spec_issue``: an open task-spec defect found by an executed check
    (a hidden milestone needs a fault the report never describes: the judges, who never see the hidden tests, label such
    failures skill gaps), the task goes back to its builder (``spec_defect``). ``verdict`` is ``train`` | ``bench`` |
    ``too_easy`` | ``too_hard`` | ``unfair`` | ``off_intent`` | ``out_of_band`` | ``world_bug`` | ``spec_defect`` | ``unmeasured``; ``keep`` is True only for ``train`` /
    ``bench`` on their own split."""
    r = rule or KeepRule()
    if n <= 0:
        return {"verdict": "unmeasured", "keep": False, "why": "no unmasked student attempt", "flags": [], "pass_rate": None}
    p = passes / n
    lo, hi = wilson(passes, n)
    fr = {k: [bool(x) for x in v] for k, v in (frontier or {}).items()}
    fair = any(any(v) for v in fr.values()) if fr else None
    share = on_intent / labelled if labelled else None
    intent_ok = None if labelled < r.min_labels else bool(share is not None and share >= r.skill_min)
    flags: list[str] = []
    if fair is None:
        flags.append("frontier solvers not run: fairness unmeasured")
    if p > r.train_hi:
        flags.append(f"too easy: pass rate {p:.2f} > {r.train_hi}")
    if p == 0 and fair:
        flags.append("simplify(locate|prerequisites): the student never passes and a frontier solver does")
    if intent_ok is None and n - passes > 0:
        flags.append(f"intent share unmeasured: {labelled} labelled failure(s) < {r.min_labels}")
    in_train = r.train_lo - 1e-9 <= p <= r.train_hi + 1e-9
    in_bench = p < r.bench_hi - 1e-9
    base = {"pass_rate": round(p, 4), "passes": passes, "n": n, "ci": [round(lo, 4), round(hi, 4)], "on_intent": on_intent, "labelled": labelled,
            "intent_share": None if share is None else round(share, 4), "intent_ok": intent_ok, "fair": fair, "frontier": fr, "split": split,
            "rule": asdict(r), "flags": flags, "power": band_power(passes, n, split=split, rule=r)}
    if world_issue:
        return {**base, "verdict": "world_bug", "keep": False, "why": f"open World defect (owner World): {world_issue}"}
    if spec_issue:
        return {**base, "verdict": "spec_defect", "keep": False, "why": f"open task-spec defect (owner the builder, S2): {spec_issue}"}
    if fair is False:
        return {**base, "verdict": "unfair", "keep": False, "why": "no frontier solver passes it: not counted as fair"}
    if p > r.train_hi:
        return {**base, "verdict": "too_easy", "keep": False, "why": f"pass rate {p:.2f} above {r.train_hi}"}
    if split == "heldout":
        if not in_bench:
            return {**base, "verdict": "out_of_band", "keep": False, "why": f"held out, pass rate {p:.2f} not below {r.bench_hi}"}
    elif not in_train:
        return {**base, "verdict": "too_hard" if p == 0 else "out_of_band", "keep": False,
                "why": f"pass rate {p:.2f} outside [{r.train_lo}, {r.train_hi}]"}
    if fair is None:
        return {**base, "verdict": "unmeasured", "keep": False, "why": "fairness unmeasured"}
    if intent_ok is None:
        return {**base, "verdict": "unmeasured", "keep": False, "why": f"intent share unmeasured ({labelled} labelled failures)"}
    if not intent_ok:
        return {**base, "verdict": "off_intent", "keep": False, "why": f"on-intent skill-gap share {share:.2f} < {r.skill_min}"}
    v = "bench" if split == "heldout" else "train"
    return {**base, "verdict": v, "keep": True, "why": f"pass rate {p:.2f}, on-intent share {share:.2f}, a frontier solver passes"}


# ---- S15 ------------------------------------------------------------------------------------------------------------------------
@dataclass
class TargetEvidence:
    """What S15 reads (each part optional; a missing part is unmeasured and flagged)."""

    task: str
    band: dict[str, Any] | None = None
    cause: dict[str, Any] | None = None
    counterfactual: dict[str, Any] | None = None
    probe: dict[str, Any] | None = None
    paraphrase: list[dict[str, Any]] | None = None
    notes: dict[str, Any] = field(default_factory=dict)
    numerics: str = "final"  # "dev": the policy ran on a dev-only backend (factory.student: MLX); no verdict counts


def target_verdict(ev: TargetEvidence) -> dict[str, Any]:
    """The S15 verdict: ``ok`` with ``flags``, or a ``route`` (S2 | drop) with its ``owner`` and findings (module doc).
    Evidence from a dev-only backend (``numerics="dev"``) gets ``ok: false``, no route and ``dev_only: true``: it is
    reported, never a pool decision (swarm-pipeline/08)."""
    findings: list[str] = []
    flags: list[str] = []
    route, owner = None, None
    if ev.numerics == "dev":
        return {"ok": False, "route": None, "owner": None, "findings": [], "dev_only": True, "task": ev.task,
                "flags": ["dev-only numerics (factory.student: an MLX-served student): reported, never a pool decision"],
                "counterfactual": (ev.counterfactual or {}).get("verdict"), "evidence": json.loads(json.dumps(asdict(ev), default=str))}

    def back(r: str, o: str, why: str) -> None:
        nonlocal route, owner
        findings.append(why)
        if route is None or r == "drop":
            route, owner = r, o

    pr = ev.probe
    if pr is None:
        flags.append("prerequisite probe not run")
    elif not pr.get("ok"):
        if pr.get("owner") == "world":
            back("drop", "world", f"prerequisite probe: {pr.get('why')} (the World, S3)")
        else:
            back("S2", "intent", f"prerequisite probe: {pr.get('why')}: the intent card is wrong for this policy")
    cf = ev.counterfactual
    if cf is None:
        flags.append("counterfactual not run")
    elif cf.get("verdict") == "not_attributed":
        back("S2", "planner", f"counterfactual not_attributed: {cf.get('why')}")
    elif cf.get("verdict") == "underpowered":
        flags.append(f"counterfactual underpowered: {cf.get('why')}")
    b = ev.band
    if b is None:
        flags.append("band not measured")
    elif not b.get("in_band"):
        back("S2", "planner", f"out of band: the target rung's rewards {b.get('rewards')} have no spread (pass rate {b.get('pass_rate')})")
    c = ev.cause
    if c is None or c.get("ok") is None:
        flags.append(f"cause mix unmeasured ({(c or {}).get('n', 0)} trusted labels)")
    elif not c["ok"]:
        back("S2", "intent", f"cause mix: skill-gap share {c.get('skill_gap_share')} < {c.get('skill_min')}")
    for i, p in enumerate(ev.paraphrase or []):
        if p.get("stable") is False:
            back("S2", "prompt", f"paraphrase {i + 1} unstable: {p.get('why')}")
        elif p.get("stable") is None:
            flags.append(f"paraphrase {i + 1} unmeasured: {p.get('why')}")
    if ev.paraphrase is None:
        flags.append("paraphrase stability not run")
    ok = not findings
    return {"ok": ok, "route": route, "owner": owner, "findings": findings, "flags": flags, "task": ev.task,
            "counterfactual": (cf or {}).get("verdict"), "evidence": json.loads(json.dumps(asdict(ev), default=str))}


async def s15_target(ctx: Any) -> Any:
    """S15 machine handler: ``ctx.services`` is a ``TargetEvidence`` or has ``.evidence`` (a callable may compute it:
    ``ctx.services.collect(ctx)``)."""
    from crucible.factory.machine import Settle

    svc = ctx.services
    ev = svc if isinstance(svc, TargetEvidence) else (await svc.collect(ctx) if hasattr(svc, "collect") else svc.evidence)
    v = target_verdict(ev)
    if hasattr(svc, "results"):
        svc.results.append(v)
    return Settle({"verdict": v}, list(v["findings"])[:40])


def target_handlers(**extra: Callable[..., Any]) -> dict[str, Any]:
    """``{"s15_target": s15_target}`` plus any extra handler (e.g. the S2 planner a test scripts)."""
    return {"s15_target": s15_target, **extra}


def target_machine(*, back_to: str = "s2_plan", max_entries: int = 2) -> dict[str, Any]:
    """S2 -> S15; S15 ok -> accepted (on to S16); route S2 -> back to ``back_to`` with the findings (bounded: an exhausted
    loop escalates, V7); route drop -> dropped."""
    v = [{"name": "verdict", "type": "json"}]
    states = [{"id": back_to, "fn": back_to, "outputs": [{"name": "plan", "type": "json"}], "entry": True, "max_entries": max_entries},
              {"id": "s15_target", "fn": "s15_target", "outputs": v, "max_entries": max_entries},
              {"id": "accepted", "terminal": "done"}, {"id": "dropped", "terminal": "rejected"}]
    trans = [{"from": back_to, "to": "s15_target"},
             {"from": "s15_target", "to": "accepted", "when": "verdict.ok == true"},
             {"from": "s15_target", "to": "dropped", "when": "verdict.route == 'drop'"},
             {"from": "s15_target", "to": back_to, "when": "verdict.route == 'S2'", "carry": "verdict.findings"}]
    return {"run": {"max_transitions": 4 * max_entries + 2}, "states": states, "transitions": trans}


# ---- the seeded tooling-only failure ---------------------------------------------------------------------------------------
SEED_PREFIX = "crucible-lh-p4-seed"
SEED_DOCKERFILE = """FROM {image}
RUN pip uninstall -y -q pytest pytest-timeout pytest-mock >/dev/null 2>&1; rm -f /usr/local/bin/pytest /usr/local/bin/py.test; \\
    python - <<'EOF'
import importlib.util, sys
assert importlib.util.find_spec("pytest") is None, "pytest still importable"
EOF
"""


def seed_image(image: str) -> str:
    """A copy of the World ``image`` with the test command hidden (pytest and its plugins removed), built once and tagged
    ``crucible-lh-p4-seed:<hash>``. Only the policy's World uses it; the grader keeps ``image``."""
    import hashlib

    tag = f"{SEED_PREFIX}:{hashlib.sha256((image + SEED_DOCKERFILE).encode()).hexdigest()[:12]}"
    have = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, check=False)
    if have.returncode == 0:
        return tag
    with tempfile.TemporaryDirectory(prefix="p4seed-") as td:
        (Path(td) / "Dockerfile").write_text(SEED_DOCKERFILE.format(image=image))
        r = subprocess.run(["docker", "build", "-q", "-t", tag, td], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"seed image build failed: {r.stderr[-800:]}")
    return tag


def seeded_tooling(task: Any, image: str, tag: str = "seed-tooling") -> Any:
    """A copy of ``task`` (same prompt, checks, grader) whose policy World is ``image`` (``seed_image``) with the workspace
    mounted read-only: the agent can neither run the tests nor change a file. Works for a full or simplified task."""
    from crucible.factory.lanes.windows import WindowTask

    t = WindowTask(task.lane, task.mined, task.window, task_id=f"{task.task_id}~{tag}", prompt=task.instruction, requirements=task.requirements,
                   spans=task.spans, checkset=task.checkset, phi_weight=task.phi_weight, import_tolerant=task.import_tolerant)
    t.world_image, t.workspace_mode = image, "ro"
    return t
