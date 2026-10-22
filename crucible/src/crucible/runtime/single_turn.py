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
missing) masks the grade (``check_infra``) instead of scoring it 0. M10 lab audit R-3: the harness grades the same
completion again, and a failure that repeats is decided by the task's control (``grade_control``: ``control_answer``, or
the gold answer itself for the ``full`` parser, graded by the same parser, criteria and judges). A judge that fails only
on this completion (it raised on a prompt injection) grades its control cleanly, so the sample is the policy's, scored 0
(``crucible:policy_error`` / ``grader_failed_on_answer``); a judge that fails on the control too masks the sample (task
blame); a task without a control text stays masked with flag ``no_control``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.ids import sha256_text
from crucible.core.schema.actions import MediaRef
from crucible.runtime.identity import code_ref
from crucible.verify.answers import CheckerSpec, CheckerUnavailable, Criterion, Judge, ParserSpec, math_verify_available, score_answer
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
    control_answer: str | None = None  # M10 lab audit R-3: a known-good completion for the control grade (``grade_control``)

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
        # M7 R2: what would otherwise decide rewards at grade time is refused at load, naming the fix
        if any(c.checker.kind == "math" for c in self.criteria) and not math_verify_available():
            raise ValueError(f"{self.id}: a 'math' criterion needs math-verify (pip install 'crucible[answers]'); without it every "
                             "grade but an equal number masks")
        if any(c.checker.kind == "judge" for c in self.criteria) and self.control_text() is None:
            raise ValueError(f"{self.id}: a judge criterion needs a control (control_answer, or a parser that reads the gold as it is): "
                             f"a judge failure is decided by the control grade (M10 lab audit R-3), and a {self.parser.kind!r} parser "
                             "has no completion to grade")
        self.system, self.history, self.instruction, self.media = _split_prompt(self.prompt, self.system)

    # ---- RunTask ---------------------------------------------------------------------------------------------------
    def digest(self) -> str:
        """The task's definition (prompt, answer, parser, criteria, judges, ...) in canonical JSON (R-2)."""
        from crucible.runtime.lock import definition_digest

        return definition_digest(self)

    def identity_config(self) -> dict[str, Any]:
        """M10 lab audit R-4: what decides this task's grade, as the run's identity document: the prompt (a long text is a
        digest), the gold answer, the parser, the rubric, the options, and each judge's model and type (a judge is a callable,
        so it cannot be inspected: its model and its code name are what a resume can compare). The prompt and answer are
        the task, so a dataset edited between two runs is refused by name."""
        from crucible.runtime.lock import definition_digest

        judges = {name: {"model": getattr(getattr(j, "llm", j), "model", None), "type": code_ref(type(j)) if hasattr(j, "llm") else code_ref(j)}
                  for name, j in self.judges.items()}
        # a prompt carrying media is compared by its digest: run.json never holds media bytes (as records and traces do not)
        prompt = definition_digest(self.prompt) if self.media else self.prompt
        return {"id": self.id, "prompt": prompt, "answer": self.answer, "parser": self.parser, "criteria": self.criteria, "options": self.options,
                "system": self.system, "judges": judges, "meta": self.meta, "world_id": self.world_id, "control_answer": self.control_answer}

    def spec(self):
        from crucible.core.schema.world import WorldSpec

        return WorldSpec.model_validate({"world": self.world_id, "apps": [], "actors": [{"id": ACTOR, "kind": "policy", "tools": []}],
                                         "intercept": [], "isolation": {"quiesce": False, "egress_probe": False}})

    async def open(self, workdir: Path):
        from crucible.core.store import Store
        from crucible.world import lifecycle
        from crucible.world.world import World

        w = World(self.spec(), instructions={ACTOR: self.instruction}, store=Store(Path(workdir) / "s"), workdir=Path(workdir) / "w")
        if self.media:  # images / audio with the question: stored once, attached to the first observation (M1 item 8)
            w.instruction_media[ACTOR] = [w.put_media(m["kind"], m["mime"], m["data"]) if "data" in m else MediaRef(kind=m["kind"], mime=m["mime"], url=m["url"])
                                          for m in self.media]
        async with lifecycle.opening(w):  # M10 lab audit R-6: a failure after start closes the World
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

    def control_text(self) -> str | None:
        """The completion the control grade scores: ``control_answer``, else the gold answer as text when the ``full``
        parser reads it as it is (a str or a number, never a bool), else None (no control: a ``boxed`` or ``mcq`` gold is
        not a completion its parser would find)."""
        if self.control_answer is not None:
            return self.control_answer
        if self.parser.kind == "full" and isinstance(self.answer, str | int | float) and not isinstance(self.answer, bool):
            return str(self.answer)
        return None

    async def grade_control(self, workdir: Path) -> GradeReport | None:
        """M10 lab audit R-3: the task's control (``runtime.grading.settle``): ``control_text`` graded by the same parser,
        criteria and judges as the policy's completion; None without a control text."""
        text = self.control_text()
        return None if text is None else await self.grade_text(text)

    async def grade_text(self, text: str | None) -> GradeReport:
        t0 = time.monotonic()
        row = {"id": self.id, "question": self.instruction, "options": self.options, **self.meta}
        kinds = {"format": "gate", "answer": "outcome"}
        try:
            s = await score_answer(text, self.answer, parser=self.parser, criteria=self.criteria, options=self.options, row=row, judges=self.judges)
        except CheckerUnavailable as e:
            checks = [CheckResult("answer", "answer", "infra", {"infra": str(e)[:500]})]
            # M10 lab audit R-3: the reason quotes the checker's or judge's own output (a verdict, an exception message),
            # which the answer steers; grading.grade_failure never sniffs it for a node, image or build cause
            return GradeReport(self.id, "full", checks, {"format": "skip", "answer": "infra"}, 0.0, 0.0,
                               {"status": "masked", "failure_kind": "check_infra", "reason": f"answer: {e}"[:500], "reason_untrusted": True}, True,
                               round(time.monotonic() - t0, 3), meta={"kinds": kinds})
        checks = [CheckResult("format", "format", "pass" if s.formatted else "fail", {"parser": self.parser.kind}),
                  *(CheckResult(f"answer/{cid}", "answer", "pass" if s.formatted and d["score"] >= c.checker.threshold else ("skip" if not s.formatted else "fail"), d)
                    for (cid, d), c in zip(s.per_criterion.items(), self.criteria, strict=True))]
        gates = {"format": "pass" if s.formatted else "fail", "answer": ("pass" if s.correct else "fail") if s.formatted else "skip"}
        reward = s.score if s.formatted else 0.0
        parsed = s.parsed if isinstance(s.parsed, str | int | float | bool) or s.parsed is None else json.dumps(s.parsed)[:2000]
        return GradeReport(self.id, "full", checks, gates, round(reward, 6), float(s.formatted and s.correct), {"status": "ok"}, True,
                           round(time.monotonic() - t0, 3), meta={"kinds": kinds, "parsed": parsed, "score": s.score, "correct": s.correct})


def _split_prompt(prompt: str | list[dict[str, Any]], system: str | None) -> tuple[str | None, list[dict[str, Any]], str, list[dict[str, Any]]]:
    """(system, few-shot history as chat messages, instruction text, instruction media). Media parts (``image_url``,
    ``input_image``, ``input_audio``) in the last user message become the instruction's media; in earlier turns they
    stay as chat content parts."""
    if isinstance(prompt, str):
        return system, [], prompt, []
    sys_msgs = [_content(m.get("content"))[0] for m in prompt if m["role"] in ("system", "developer")]
    rest = [m for m in prompt if m["role"] not in ("system", "developer")]
    if not rest or rest[-1]["role"] != "user":
        raise ValueError("a chat prompt must end with a user message")
    sys_text = "\n\n".join([*([system] if system else []), *sys_msgs]) or None
    history = [{"role": m["role"], "content": _chat_content(m.get("content"))} for m in rest[:-1]]
    text, media = _content(rest[-1].get("content"))
    return sys_text, history, text, media


def _media_of(part: dict[str, Any]) -> dict[str, Any] | None:
    """One OpenAI chat / Responses media part -> {kind, mime, data (bytes) | url}."""
    import base64

    t = part.get("type")
    if t in ("image_url", "input_image"):
        url = part.get("image_url")
        url = url.get("url") if isinstance(url, dict) else url
        if not isinstance(url, str) or not url:
            raise ValueError(f"image part without a url: {part}")
        if url.startswith("data:"):
            head, _, b64 = url.partition(",")
            return {"kind": "image", "mime": head[5:].split(";")[0] or "image/png", "data": base64.b64decode(b64)}
        return {"kind": "image", "mime": "image/*", "url": url}
    if t == "input_audio":
        a = part.get("input_audio") or {}
        fmt = a.get("format", "wav")
        return {"kind": "audio", "mime": f"audio/{'mpeg' if fmt == 'mp3' else fmt}", "data": base64.b64decode(a.get("data", ""))}
    return None


def _content(c: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(c, str):
        return c, []
    if isinstance(c, list):
        texts, media = [], []
        for part in c:
            if not isinstance(part, dict):
                raise TypeError(f"unsupported content part {part!r}")
            if part.get("type") in ("text", "input_text", "output_text"):
                texts.append(part.get("text", ""))
            elif (m := _media_of(part)) is not None:
                media.append(m)
            else:
                raise ValueError(f"unsupported content part type {part.get('type')!r}")
        return "".join(texts), media
    raise ValueError(f"unsupported message content {type(c).__name__}")


def _chat_content(c: Any) -> Any:
    """Few-shot turns keep their media as OpenAI chat parts (``input_image`` normalized to ``image_url``)."""
    if not isinstance(c, list):
        return _content(c)[0]
    out = []
    for part in c:
        t = part.get("type")
        if t in ("text", "input_text", "output_text"):
            out.append({"type": "text", "text": part.get("text", "")})
        elif t in ("image_url", "input_image"):
            url = part.get("image_url")
            out.append({"type": "image_url", "image_url": {"url": url.get("url") if isinstance(url, dict) else url}})
        elif t == "input_audio":
            out.append({"type": "input_audio", "input_audio": dict(part.get("input_audio") or {})})
        else:
            raise ValueError(f"unsupported content part type {t!r}")
    return out


# ------------------------------------------------------------------------------------------------- policies
def single_turn_policy(llm, *, temperature: float = 0.7, max_tokens: int = 4096, name: str | None = None, vision: bool = False, audio: bool = False):
    """A ``PolicyRef`` that answers in one completion: the task's system prompt and few-shot turns, no tools, no nudges
    (a blank or cut-off reply is the answer, as in every single-turn framework)."""
    from crucible.actors.policy import LLMToolPolicy
    from crucible.runtime.harness.native import PolicyRef

    def make(attempt):
        t = attempt.task
        p = LLMToolPolicy(ACTOR, llm, system=t.system or "", max_nudges=0, max_tokens=max_tokens, temperature=temperature, vision=vision, audio=audio)
        p.messages = [*([{"role": "system", "content": t.system}] if t.system else []), *t.history]
        return {ACTOR: p}

    settings = {"kind": "single_turn", "temperature": temperature, "max_tokens": max_tokens, "vision": vision, "audio": audio,
                **({"llm": llm.lock_settings()} if hasattr(llm, "lock_settings") else {"llm": {"model": getattr(llm, "model", None)}})}
    return PolicyRef(id=name or f"single_turn:{llm.model}", make=make,
                     spend=lambda: (llm.spend.usd, getattr(llm.spend, "uncertain_usd", 0.0)), estimate_usd=0.001, settings=settings,
                     config={"model": getattr(llm, "model", None), "temperature": temperature, "max_tokens": max_tokens, "vision": vision, "audio": audio})  # M10 lab audit R-4


def answer_policy(text_of) -> Any:
    """A scripted ``PolicyRef`` whose completion is ``text_of(task)`` (gold and designed negatives in tests and gates)."""
    from crucible.actors.policy import ScriptedPolicy
    from crucible.core.schema.actions import Done
    from crucible.runtime.harness.native import PolicyRef

    def make(attempt):
        text = text_of(attempt.task)
        return {ACTOR: ScriptedPolicy(ACTOR, [Done(reason=text[:2000], answer=text if len(text) > 2000 else None)])}

    return PolicyRef(id="scripted:answer", make=make, config={"script": code_ref(text_of)})  # M10 lab audit R-4: the callable's name; its behaviour is its code


# ------------------------------------------------------------------------------------------------- datasets
def rows_from(source: str | Path | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(source, str | Path):
        return [json.loads(ln) for ln in Path(source).read_text().splitlines() if ln.strip()]
    return list(source)


def task_from_row(row: dict[str, Any], *, dataset: str, index: int, parser: ParserSpec | dict | str | None = None,
                  criteria: list[Criterion | dict] | None = None, system: str | None = None, judges: dict[str, Judge] | None = None,
                  grading_mode: str | None = None) -> SingleTurnTask:
    """One row in any of the supported shapes -> a task. Row-level ``parser`` / ``criteria`` override the defaults.

    M7 R2: a NeMo-Gym multiple-choice benchmark states its grading mode in its config, not in its rows (GPQA's is
    ``lenient_answer_colon_md``): ``grading_mode`` passes it, a row's own ``grading_mode`` overrides it, and a NeMo-Gym
    multiple-choice row with neither is refused (it was graded as strict boxed, so ``Answer: B`` scored 0). Each task gets
    a control completion in its parser's own format (``\\boxed{gold}``, ``Answer: <letter>``), so a checker or judge
    failure is decided by the control instead of a mask."""
    nemo = "responses_create_params" in row
    if nemo:  # NeMo-Gym
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
        p = {"kind": "regex", "pattern": rx if isinstance(rx, str) else list(rx)}  # a list is tried in turn (M7 R2)
    if p is None and options:
        mode = row.get("grading_mode") or grading_mode or (None if nemo else "strict_single_letter_boxed")
        if mode is None:
            raise ValueError(f"{dataset}[{index}]: a NeMo-Gym multiple-choice row needs its benchmark's grading_mode (the benchmark "
                             f"config's mcqa grading_mode, e.g. GPQA's 'lenient_answer_colon_md'): pass load_tasks(..., grading_mode=...)")
        if mode not in NEMO_MCQ_MODES:
            raise ValueError(f"{dataset}[{index}]: grading_mode {mode!r} is not supported (supported: {sorted(NEMO_MCQ_MODES)})")
        p = {"kind": "mcq", "mode": NEMO_MCQ_MODES[mode]}
        if tm.get("answer_prefix"):
            p["prefix"] = tm["answer_prefix"]
    crit = row.get("criteria") or criteria or [{"id": "correct", "checker": "mcq" if options else "exact"}]
    rid = str(row.get("id") or row.get("uuid") or f"{index:06d}-{sha256_text(json.dumps(prompt, sort_keys=True, default=str))[:8]}")
    spec = ParserSpec.of(p)
    return SingleTurnTask(id=f"{dataset}/{rid}@1", prompt=prompt, answer=answer, parser=spec, criteria=crit, options=options,
                          system=system, judges=dict(judges or {}), meta={k: v for k, v in (row.get("info") or row.get("metadata") or {}).items()
                                                                          if isinstance(v, str | int | float | bool)},
                          world_id=f"single_turn/{dataset}", control_answer=row.get("control_answer") or control_completion(spec, answer, options))


def control_completion(parser: ParserSpec, answer: Any, options: dict[str, str] | None) -> str | None:
    """M7 R2: the gold answer as a completion its parser reads: ``\\boxed{gold}`` (``boxed``, and the boxed multiple-choice
    modes), ``Answer: <gold>`` (the answer-colon modes), None for a parser with no such form (``full`` grades the gold as
    it is, ``control_text``; ``regex`` and the rest have no general form)."""
    if isinstance(answer, bool) or not isinstance(answer, str | int | float):
        return None
    gold = str(answer).strip()
    if parser.kind == "boxed":
        return f"\\boxed{{{gold}}}"
    if parser.kind == "answer_colon" and not parser.prefix:
        return f"Answer: {gold}"
    if parser.kind == "mcq" and options and gold.upper() in {k.upper() for k in options}:
        return f"\\boxed{{{gold.upper()}}}" if parser.mode in ("strict_boxed", "lenient_boxed") else f"Answer: {gold.upper()}"
    return None


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
