"""Single-turn tasks (M1 item 2; family 1 of the M1 taxonomy, the largest by count): a prompt, one completion, a checker.

A ``SingleTurnTask`` is an ordinary ``RunTask``. Its World has no Apps and one policy actor with no tools: the prompt is
the instruction, the completion ends the episode (``Done``; the full reply is kept in the trace), and the grade parses
the answer and scores it (``verify/answers.py``). Running through the World keeps the native harness, traces, ATIF
export, token capture for training and the run records unchanged, so single-turn data mixes with agentic data.

Rows load from the shapes the surveyed frameworks use (``rows_from``):

- NeMo-Gym: ``responses_create_params.input`` (messages), ``expected_answer``, ``options`` ([{letter: text}]),
  ``template_metadata.output_regex`` / ``answer_prefix``;
- verifiers / prime-environments: ``prompt`` (messages) or ``question``, ``answer``, ``info``;
- plain: ``prompt`` (text), ``answer``.

Grade: gate ``format`` (the parser found an answer) and outcome ``answer`` (every rubric criterion at or above its
threshold); the reward is the weighted rubric score. A checker that cannot decide (a judge outage, math-verify
missing) masks the sample (``check_infra``) instead of scoring it 0.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.ids import sha256_text
from crucible.verify.answers import CheckerSpec, CheckerUnavailable, Criterion, Judge, ParserSpec, score_answer
from crucible.verify.runtime.plan import CheckResult, GradeReport

ACTOR = "agent"
NEMO_MCQ_MODES = {"strict_single_letter_boxed": "strict_boxed", "lenient_boxed": "lenient_boxed", "lenient_answer_colon": "answer_colon",
                  "lenient_answer_colon_md": "answer_colon_md"}


@dataclass
class SingleTurnTask:
    id: str
    prompt: str | list[dict[str, Any]]  # text, or chat messages: the last user message is the instruction
    answer: Any
    parser: ParserSpec = field(default_factory=ParserSpec)
    criteria: list[Criterion] = field(default_factory=lambda: [Criterion("correct", CheckerSpec())])
    options: dict[str, str] | None = None  # multiple choice: letter -> option text
    system: str | None = None
    judges: dict[str, Judge] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    world_id: str = "single_turn/default"

    def __post_init__(self) -> None:
        self.parser = ParserSpec.of(self.parser)
        self.criteria = [Criterion.of(c) for c in self.criteria]
        if not self.criteria:
            raise ValueError(f"{self.id}: a single-turn task needs at least one criterion")
        if any(c.weight < 0 for c in self.criteria):
            raise ValueError(f"{self.id}: criterion weights must be >= 0")
        if self.parser.kind == "mcq" and not self.options:
            raise ValueError(f"{self.id}: an mcq parser needs options")
        for c in self.criteria:
            if c.checker.kind == "judge" and c.checker.judge not in self.judges:
                raise ValueError(f"{self.id}: criterion {c.id!r} names judge {c.checker.judge!r}, which the task does not configure")
        self.system, self.history, self.instruction = _split_prompt(self.prompt, self.system)

    # ---- RunTask ---------------------------------------------------------------------------------------------------
    def spec(self):
        from crucible.core.schema.world import WorldSpec

        return WorldSpec.model_validate({"world": self.world_id, "apps": [], "actors": [{"id": ACTOR, "kind": "policy", "tools": []}],
                                         "intercept": [], "isolation": {"quiesce": False, "egress_probe": False}})

    async def open(self, workdir: Path):
        from crucible.core.store import Store
        from crucible.world.world import World

        w = World(self.spec(), instructions={ACTOR: self.instruction}, store=Store(Path(workdir) / "s"), workdir=Path(workdir) / "w")
        await w.start()
        await w.reset()
        return w

    @staticmethod
    def completion(world) -> str | None:
        """The policy's final reply: the full answer kept on its ``done`` event (``reason`` may be a cut of it)."""
        for ev in reversed(world.log):
            if ev.type == "done" and ev.actor == ACTOR and ev.text != "environment_terminal":
                return (ev.data or {}).get("answer") or ev.text
        return None

    async def grade(self, world, workdir: Path) -> GradeReport:
        return await self.grade_text(self.completion(world))

    async def grade_text(self, text: str | None) -> GradeReport:
        t0 = time.monotonic()
        row = {"id": self.id, "question": self.instruction, "options": self.options, **self.meta}
        kinds = {"format": "gate", "answer": "outcome"}
        try:
            s = await score_answer(text, self.answer, parser=self.parser, criteria=self.criteria, options=self.options, row=row, judges=self.judges)
        except CheckerUnavailable as e:
            checks = [CheckResult("answer", "answer", "infra", {"infra": str(e)[:500]})]
            return GradeReport(self.id, "full", checks, {"format": "skip", "answer": "infra"}, 0.0, 0.0,
                               {"status": "masked", "failure_kind": "check_infra", "reason": f"answer: {e}"[:500]}, True,
                               round(time.monotonic() - t0, 3), meta={"kinds": kinds})
        checks = [CheckResult("format", "format", "pass" if s.formatted else "fail", {"parser": self.parser.kind}),
                  *(CheckResult(f"answer/{cid}", "answer", "pass" if s.formatted and d["score"] >= c.checker.threshold else ("skip" if not s.formatted else "fail"), d)
                    for (cid, d), c in zip(s.per_criterion.items(), self.criteria, strict=True))]
        gates = {"format": "pass" if s.formatted else "fail", "answer": ("pass" if s.correct else "fail") if s.formatted else "skip"}
        reward = s.score if s.formatted else 0.0
        parsed = s.parsed if isinstance(s.parsed, str | int | float | bool) or s.parsed is None else json.dumps(s.parsed)[:2000]
        return GradeReport(self.id, "full", checks, gates, round(reward, 6), float(s.formatted and s.correct), {"status": "ok"}, True,
                           round(time.monotonic() - t0, 3), meta={"kinds": kinds, "parsed": parsed, "score": s.score, "correct": s.correct})


def _split_prompt(prompt: str | list[dict[str, Any]], system: str | None) -> tuple[str | None, list[dict[str, Any]], str]:
    if isinstance(prompt, str):
        return system, [], prompt
    msgs = [{"role": m["role"], "content": _content_text(m.get("content"))} for m in prompt]
    sys_msgs = [m["content"] for m in msgs if m["role"] in ("system", "developer")]
    rest = [m for m in msgs if m["role"] not in ("system", "developer")]
    if not rest or rest[-1]["role"] != "user":
        raise ValueError("a chat prompt must end with a user message")
    sys_text = "\n\n".join([*([system] if system else []), *sys_msgs]) or None
    return sys_text, rest[:-1], rest[-1]["content"]


def _content_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # OpenAI / Responses content parts: text only here (images are M1 item 8)
        parts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text")]
        if len(parts) != len(c):
            raise ValueError("non-text prompt parts are not supported by single-turn tasks yet")
        return "".join(parts)
    raise ValueError(f"unsupported message content {type(c).__name__}")


# ------------------------------------------------------------------------------------------------- policies
def single_turn_policy(llm, *, temperature: float = 0.7, max_tokens: int = 4096, name: str | None = None):
    """A ``PolicyRef`` that answers in one completion: the task's system prompt and few-shot turns, no tools, no nudges
    (a blank or cut-off reply is the answer, as in every single-turn framework)."""
    from crucible.actors.policy import LLMToolPolicy
    from crucible.runtime.harness.native import PolicyRef

    def make(attempt):
        t = attempt.task
        p = LLMToolPolicy(ACTOR, llm, system=t.system or "", max_nudges=0, max_tokens=max_tokens, temperature=temperature)
        p.messages = [*([{"role": "system", "content": t.system}] if t.system else []), *t.history]
        return {ACTOR: p}

    return PolicyRef(id=name or f"single_turn:{llm.model}", make=make,
                     spend=lambda: (llm.spend.usd, getattr(llm.spend, "uncertain_usd", 0.0)), estimate_usd=0.001)


def answer_policy(text_of) -> Any:
    """A scripted ``PolicyRef`` whose completion is ``text_of(task)`` (gold and designed negatives in tests and gates)."""
    from crucible.actors.policy import ScriptedPolicy
    from crucible.core.schema.actions import Done
    from crucible.runtime.harness.native import PolicyRef

    def make(attempt):
        text = text_of(attempt.task)
        return {ACTOR: ScriptedPolicy(ACTOR, [Done(reason=text[:2000], answer=text if len(text) > 2000 else None)])}

    return PolicyRef(id="scripted:answer", make=make)


# ------------------------------------------------------------------------------------------------- datasets
def rows_from(source: str | Path | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(source, str | Path):
        return [json.loads(ln) for ln in Path(source).read_text().splitlines() if ln.strip()]
    return list(source)


def task_from_row(row: dict[str, Any], *, dataset: str, index: int, parser: ParserSpec | dict | str | None = None,
                  criteria: list[Criterion | dict] | None = None, system: str | None = None, judges: dict[str, Judge] | None = None) -> SingleTurnTask:
    """One row in any of the supported shapes -> a task. Row-level ``parser`` / ``criteria`` override the defaults."""
    if "responses_create_params" in row:  # NeMo-Gym
        rcp = row["responses_create_params"]
        prompt: Any = rcp.get("input") if isinstance(rcp.get("input"), list) else str(rcp.get("input", ""))
        answer = row.get("expected_answer")
        opts = row.get("options")
        system = system or rcp.get("instructions")
    else:  # verifiers / plain
        prompt = row.get("prompt") if row.get("prompt") is not None else row.get("question")
        answer = row.get("answer", row.get("expected_answer"))
        opts = row.get("options") or (row.get("info") or {}).get("options")
    if prompt is None or answer is None:
        raise ValueError(f"{dataset}[{index}]: a row needs a prompt (or question) and an answer")
    options = _options(opts)
    p = row.get("parser", parser)
    tm = row.get("template_metadata") or {}
    if p is None and tm.get("output_regex"):  # NeMo-Gym: a row's own answer regex wins over the grading mode
        rx = tm["output_regex"]
        p = {"kind": "regex", "pattern": rx if isinstance(rx, str) else "|".join(f"(?:{r})" for r in rx)}
    if p is None and options:
        p = {"kind": "mcq", "mode": NEMO_MCQ_MODES[row.get("grading_mode", "strict_single_letter_boxed")]}
        if tm.get("answer_prefix"):
            p["prefix"] = tm["answer_prefix"]
    crit = row.get("criteria") or criteria or [{"id": "correct", "checker": "mcq" if options else "exact"}]
    rid = str(row.get("id") or row.get("uuid") or f"{index:06d}-{sha256_text(json.dumps(prompt, sort_keys=True, default=str))[:8]}")
    return SingleTurnTask(id=f"{dataset}/{rid}@1", prompt=prompt, answer=answer, parser=ParserSpec.of(p), criteria=crit, options=options,
                          system=system, judges=dict(judges or {}), meta={k: v for k, v in (row.get("info") or row.get("metadata") or {}).items()
                                                                          if isinstance(v, str | int | float | bool)},
                          world_id=f"single_turn/{dataset}")


def _options(opts: Any) -> dict[str, str] | None:
    if not opts:
        return None
    if isinstance(opts, dict):
        return {str(k).upper(): v for k, v in opts.items() if v is not None}
    if isinstance(opts, list) and all(isinstance(o, dict) for o in opts):  # NeMo-Gym: [{"A": "..."}, {"B": "..."}]
        return {str(k).upper(): v for o in opts for k, v in o.items() if v is not None}
    if isinstance(opts, list):  # a plain list: lettered in order
        return {chr(ord("A") + i): str(v) for i, v in enumerate(opts)}
    raise ValueError(f"unsupported options {type(opts).__name__}")


def load_tasks(source: str | Path | Iterable[dict[str, Any]], *, dataset: str, **kw: Any) -> list[SingleTurnTask]:
    tasks = [task_from_row(r, dataset=dataset, index=i, **kw) for i, r in enumerate(rows_from(source))]
    ids = [t.id for t in tasks]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"{dataset}: duplicate task ids {dup[:5]}")
    return tasks
