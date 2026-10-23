"""Kernel debugging lane, hardware reachability: can a mined fault be graded on the GPUs we have (swarm-pipeline/10 §3.1)?

The grading GPUs are a GH200 (``sm_90``, DeltaAI) and an RTX 5090 (``sm_120``, the lab machine). A fault qualifies for a
GPU only when the code it changes is compiled *and* dispatched there, and (for the bug family) when it changes results,
not only speed. Mining used to read this from the commit subject alone (``kernel_debug_mine.platform_only``); seven
tasks reached the GPU gates with faults no GPU of ours runs (AMD-only selection, pre-Ampere paths, integrated GPUs,
speed-only "fixes", ops the op test never builds) and were retired. This module decides it from the change itself:

- **Compiled and dispatched** (``changed_on``): the file before and after the change, seen as one target GPU sees it.
  Preprocessor branches are resolved with that GPU's ``__CUDA_ARCH__`` (device pass) or without it (host pass), HIP,
  MUSA and NCCL undefined, and the backend's own feature macros (``AMPERE_MMA_AVAILABLE`` ...) read from ``common.cuh``
  the same way; runtime compute-capability tests (``cc >= GGML_CUDA_CC_AMPERE``, ``GGML_CUDA_CC_IS_AMD(cc)``,
  ``turing_mma_available(cc)``) are folded with that GPU's ``cc``, code after a ``return`` is dropped, and ``if``
  branches that became identical are merged. The fault reaches the GPU when the two views still differ. Anything the
  evaluator cannot decide (a macro set by the build, a runtime value) counts as reached: the filter only rejects on
  evidence, and the GPU gates (S1 executed) stay the final word.
- **From the message** (``message_reasons``): an architecture we do not have, integrated GPUs or several GPUs named in
  the subject; for the bug family, a change described only by speed (no symptom words), a launch-shape change with no
  limit or symptom named, and a precision-only selection between two kernels that both compute the result.
- **Testable** (``covered_ops``): at least one op the change reaches is built by the repository's op test.

Standard library only; deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

TARGETS: dict[str, int] = {"sm_90": 900, "sm_120": 1200}  # GH200 (DeltaAI), RTX 5090 (lab): the cc the backend reads
UNDEFINED = re.compile(r"^(?:GGML_USE_(?:HIP|MUSA|NCCL|SYCL|VULKAN|METAL)|__HIP\w*|__HIPCC\w*|GGML_HIP_\w+|GGML_MUSA_\w+|__MUSA\w*|"
                       r"RDNA\w*|CDNA\w*|GCN\w*|GGML_CUDA_NO_PEER_COPY|GGML_CUDA_FORCE_\w+)$")
DEFINED = {"GGML_USE_CUDA": 1, "__CUDACC__": 1, "GGML_CUDA": 1}
CC_TOKEN = re.compile(r"\b(?:cc|__CUDA_ARCH__|\w*CC_\w+|\w+_available|ggml_cuda_highest_compiled_arch|compute_capability)\b")


# ---- C expressions, three-valued (None = cannot tell) ----------------------------------------------------------------------
TOKEN = re.compile(r"\s*(0[xX][0-9a-fA-F]+|\d+[uUlL]*|[A-Za-z_]\w*|&&|\|\||==|!=|<=|>=|<<|>>|[-+*/%<>!()&|^~?:,.\[\]])")


def _truth(v: Any) -> bool | None:
    return None if v is None else bool(v)


class _Expr:
    def __init__(self, text: str, value_of: Any):
        self.toks = [t for t in TOKEN.findall(text)]
        self.i, self.value_of = 0, value_of

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> str:
        t = self.toks[self.i]
        self.i += 1
        return t

    def parse(self) -> Any:
        v = self.lor()
        return v if self.peek() is None else None

    def lor(self) -> Any:
        v = self.land()
        while self.peek() == "||":
            self.take()
            a, b = _truth(v), _truth(self.land())
            v = True if True in (a, b) else (None if None in (a, b) else False)
        return v

    def land(self) -> Any:
        v = self.cmp()
        while self.peek() == "&&":
            self.take()
            a, b = _truth(v), _truth(self.cmp())
            v = False if False in (a, b) else (None if None in (a, b) else True)
        return v

    def cmp(self) -> Any:
        v = self.bitor()
        while self.peek() in ("==", "!=", "<", ">", "<=", ">="):
            op, w = self.take(), self.bitor()
            v = None if None in (v, w) else {"==": v == w, "!=": v != w, "<": v < w, ">": v > w, "<=": v <= w, ">=": v >= w}[op]
        return v

    def bitor(self) -> Any:
        v = self.add()
        while self.peek() in ("&", "|", "^", "<<", ">>"):
            op, w = self.take(), self.add()
            v = None if None in (v, w) else {"&": int(v) & int(w), "|": int(v) | int(w), "^": int(v) ^ int(w), "<<": int(v) << int(w),
                                             ">>": int(v) >> int(w)}[op]
        return v

    def add(self) -> Any:
        v = self.mul()
        while self.peek() in ("+", "-"):
            op, w = self.take(), self.mul()
            v = None if None in (v, w) else (v + w if op == "+" else v - w)
        return v

    def mul(self) -> Any:
        v = self.unary()
        while self.peek() in ("*", "/", "%"):
            op, w = self.take(), self.unary()
            v = None if None in (v, w) or (op != "*" and not w) else (v * w if op == "*" else v // w if op == "/" else v % w)
        return v

    def unary(self) -> Any:
        t = self.peek()
        if t == "!":
            self.take()
            v = self.unary()
            return None if v is None else not v
        if t in ("-", "~"):
            self.take()
            v = self.unary()
            return None if v is None else (-v if t == "-" else ~int(v))
        return self.atom()

    def atom(self) -> Any:
        t = self.peek()
        if t is None:
            return None
        if t == "(":
            self.take()
            v = self.lor()
            if self.peek() == ")":
                self.take()
                return v
            return None
        self.take()
        if re.match(r"\d|0[xX]", t):
            return int(t.rstrip("uUlL"), 0)
        if t in ("true", "false"):
            return t == "true"
        if t == "defined":
            paren = self.peek() == "("
            if paren:
                self.take()
            name = self.take() if self.peek() else ""
            if paren and self.peek() == ")":
                self.take()
            return self.value_of(("defined", name))
        args: list[str] = []
        if self.peek() == "(":  # a call: its argument text, for the macros and predicates the backend defines
            depth, start = 0, self.i
            while self.peek() is not None:
                x = self.take()
                depth += (x == "(") - (x == ")")
                if depth == 0:
                    break
            args = [" ".join(self.toks[start + 1:self.i - 1])]
            return self.value_of(("call", t, args[0]))
        while self.peek() in (".", "[", "]") or (self.peek() and self.toks[self.i - 1] in (".", "[") and re.match(r"\w", self.peek() or "")):
            t += self.take()  # ggml_cuda_info().devices[id].cc and the like: one name
        return self.value_of(("name", t))


@dataclass
class Env:
    """What one target GPU sees: the macros in force and the ``cc`` the host code reads."""

    cc: int
    device: bool
    macros: dict[str, Any] = field(default_factory=dict)  # name -> object-like body (str) | (params, body) for function-like
    funcs: dict[str, tuple[str, str]] = field(default_factory=dict)  # predicate(cc) -> (param, return expression)

    def value(self, key: tuple[str, ...], depth: int = 0) -> Any:
        kind, name = key[0], key[1]
        if depth > 12:
            return None
        if kind == "defined":
            if name == "__CUDA_ARCH__":
                return self.device
            if self.macros.get(name, None) in (UNKNOWN, DEVICE):
                return None
            if name in self.macros or name in DEFINED:
                return True
            return False if UNDEFINED.match(name) else None
        if kind == "call":
            arg = key[2]
            m = self.macros.get(name)
            if isinstance(m, tuple):
                params, body = m
                return self.eval(re.sub(rf"\b{re.escape(params[0])}\b", f"({arg})", body) if params else body, depth + 1)
            if name in self.funcs:
                param, expr = self.funcs[name]
                return self.eval(re.sub(rf"\b{re.escape(param)}\b", f"({arg})", expr), depth + 1)
            if name == "ggml_cuda_highest_compiled_arch":
                return self.eval(arg, depth + 1)
            return None
        if name == "__CUDA_ARCH__":
            return self.cc if self.device else None
        if name == "cc" or name.endswith(".cc") or name == "compute_capability":
            return self.cc
        m = self.macros.get(name)
        if m is UNKNOWN or m is DEVICE:
            return None
        if isinstance(m, str):
            return self.eval(m, depth + 1) if m.strip() else 1
        if name in DEFINED:
            return DEFINED[name]
        return None

    def eval(self, text: str, depth: int = 0) -> Any:
        try:
            return _Expr(text, lambda k: self.value(k, depth)).parse()
        except (IndexError, ValueError, TypeError, ZeroDivisionError):
            return None


# ---- one GPU's view of a file ------------------------------------------------------------------------------------------------
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
LITERAL = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|//[^\n]*|/\*.*?\*/', re.DOTALL)
DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif|define|undef)\b(.*)$")
DEFINE = re.compile(r"([A-Za-z_]\w*)(\(([^)]*)\))?\s*(.*)")


def strip_comments(text: str) -> str:
    """Comments removed and string/char literals emptied (a brace in a format string must not end a block)."""
    def blank(m: re.Match[str]) -> str:
        t = m.group(0)
        return t[0] * 2 if t[0] in "\"'" else "\n" * t.count("\n")
    return LITERAL.sub(blank, text.replace("\\\n", " "))


def device_only_branch(env: Env, cond: str) -> bool:
    """In the host pass, a branch on ``__CUDA_ARCH__``'s value selects device code only (a kernel body, NO_DEVICE_CODE):
    none of its branches is host code that runs."""
    if env.device:
        return False
    if any(env.macros.get(n) is DEVICE for n in re.findall(r"[A-Za-z_]\w*", cond)):
        return True
    return bool(re.search(r"\b__CUDA_ARCH__\b(?!\s*\))", cond)) and not re.fullmatch(r"\s*!?\s*defined\s*\(?\s*__CUDA_ARCH__\s*\)?\s*", cond)


DEVICE = object()  # a macro defined only for device code (under a branch on __CUDA_ARCH__): the host pass never tests it for real
UNKNOWN = object()  # a macro defined only under a branch the evaluator could not decide: neither defined nor undefined


def preprocess(text: str, env: Env) -> str:
    """``text`` with the preprocessor branches ``env`` does not take removed (a branch it cannot decide is kept, and so are
    its alternatives); the ``#define`` lines it does take are recorded in ``env.macros`` (as ``UNKNOWN`` when they sit
    under an undecided branch)."""
    out: list[str] = []
    stack: list[list[Any]] = []  # [a branch was taken (True/None=unknown/False), this branch active, this branch certain]
    for ln in strip_comments(text).splitlines():
        m = DIRECTIVE.match(ln)
        active = all(f[1] for f in stack)
        certain = all(f[2] for f in stack)
        device_only = any(len(f) == 4 for f in stack) and all(f[1] for f in stack if len(f) != 4)
        if not m:
            if active:
                out.append(ln)
            continue
        d, rest = m.group(1), m.group(2).strip()
        if d in ("if", "ifdef", "ifndef"):
            if device_only_branch(env, rest if d == "if" else (rest.split()[0] if rest else "")):
                stack.append([True, False, True, "device"])  # every branch is device code: the host pass compiles none of it
                continue
            v = env.eval(rest) if d == "if" else env.value(("defined", rest.split()[0] if rest else ""))
            v = (not v if v is not None else None) if d == "ifndef" else _truth(v)
            stack.append([v, v is not False, v is True])
        elif d in ("elif", "else") and stack and len(stack[-1]) == 4:
            continue  # the alternatives of a device-only branch are device code too
        elif d == "elif" and stack:
            f = stack[-1]
            v = _truth(env.eval(rest))
            f[1] = f[0] is not True and v is not False
            f[2] = f[0] is False and v is True
            f[0] = True if (f[0] is True or v is True) else (None if (f[0] is None or v is None) else False)
        elif d == "else" and stack:
            f = stack[-1]
            f[1], f[2] = f[0] is not True, f[0] is False
        elif d == "endif" and stack:
            stack.pop()
        elif d == "define" and device_only:
            dm = DEFINE.match(rest)
            if dm:
                env.macros[dm.group(1)] = DEVICE
        elif d == "define" and active:
            out.append(ln)  # a macro's value is code too: changing it is a change
            dm = DEFINE.match(rest)
            if dm:
                name, params, body = dm.group(1), dm.group(3), dm.group(4)
                env.macros[name] = UNKNOWN if not certain else (
                    ([p.strip() for p in params.split(",") if p.strip()], body) if dm.group(2) else body)
        elif d == "undef" and active:
            out.append(ln)
            name = rest.split()[0] if rest else ""
            if certain:
                env.macros.pop(name, None)
            else:
                env.macros[name] = UNKNOWN
    return "\n".join(out)


PREDICATE = re.compile(r"\b(?:static\s+)?(?:constexpr\s+)?(?:__host__\s+)?(?:__device__\s+)?(?:inline\s+)?bool\s+(\w+_available)\s*\(\s*(?:const\s+)?int\s+(\w+)\s*\)"
                       r"\s*\{\s*return\s+([^;]+);\s*\}")


def env_for(common: str | None, cc: int, device: bool) -> Env:
    """The environment of one target GPU, with the backend's feature macros and ``*_available(cc)`` predicates read from
    its ``common.cuh`` (resolved for that GPU)."""
    env = Env(cc=cc, device=device)
    if common:
        seen = preprocess(common, env)
        for name, param, expr in PREDICATE.findall(re.sub(r"\s+", " ", seen)):
            env.funcs[name] = (param, expr)
    return env


def _match(text: str, i: int, open_: str, close: str) -> int:
    """Index just past the bracket matching ``text[i]`` (an ``open_``), or -1."""
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def _stmt_end(text: str, i: int) -> int:
    """End of the statement starting at ``text[i]`` (a block, an ``if``/``else`` chain, or up to its ``;``), or -1."""
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        return -1
    if text[i] == "{":
        return _match(text, i, "{", "}")
    m = re.match(r"if\s*(?:constexpr\s*)?\(", text[i:])
    if m:
        c = _match(text, i + m.end() - 1, "(", ")")
        if c < 0:
            return -1
        e = _stmt_end(text, c)
        if e < 0:
            return -1
        el = re.match(r"\s*else\b", text[e:])
        return _stmt_end(text, e + el.end()) if el else e
    depth = 0
    for j in range(i, len(text)):
        if text[j] in "({[":
            depth += 1
        elif text[j] in ")}]":
            depth -= 1
            if depth < 0:
                return -1
        elif text[j] == ";" and depth == 0:
            return j + 1
    return -1


IF = re.compile(r"\bif\s*(constexpr\s*)?\(")


def fold(text: str, env: Env) -> str:
    """Runtime compute-capability tests folded with ``env`` (the branch taken kept, the other dropped), ``if`` branches
    that ended up identical merged, and code after a block's ``return`` dropped."""
    for _ in range(64):
        changed = False
        pos = 0
        while True:
            m = IF.search(text, pos)
            if not m:
                break
            c = _match(text, m.end() - 1, "(", ")")
            body_end = _stmt_end(text, c) if c > 0 else -1
            if c < 0 or body_end < 0:
                pos = m.end()
                continue
            cond = text[m.end():c - 1]
            el = re.match(r"\s*else\b", text[body_end:])
            else_start = body_end + el.end() if el else body_end
            end = _stmt_end(text, else_start) if el else body_end
            if end < 0:
                pos = m.end()
                continue
            body, other = text[c:body_end], (text[else_start:end] if el else "")
            v = env.eval(cond) if CC_TOKEN.search(cond) else None
            if v is None and el and _norm(body) == _norm(other):
                v = True  # both branches do the same thing: the test does not matter
            if v is None:
                pos = m.end()
                continue
            kept = (body if v else other).strip()
            if kept.startswith("{") and kept.endswith("}") and _match(kept, 0, "{", "}") == len(kept):
                kept = kept[1:-1]  # the branch taken runs unconditionally: its statements join the enclosing block
            text = text[:m.start()] + " " + kept + " " + text[end:]
            changed = True
            pos = m.start()
        text, cut = _after_return(text)
        if not (changed or cut):
            break
    return text


RETURN = re.compile(r"\breturn\b")


def _after_return(text: str) -> tuple[str, bool]:
    """Statements after a ``return`` in the same block dropped (not inside a ``switch``, where a later ``case`` is live)."""
    for m in RETURN.finditer(text):
        prev = text[:m.start()].rstrip()
        if prev.endswith(")") or re.search(r"\b(?:else|do)$", prev):  # the unbraced body of an if/else/loop: conditional
            continue
        e = _stmt_end(text, m.start())
        if e < 0:
            continue
        depth, j = 0, e
        while j < len(text):
            if text[j] in "{(":
                depth += 1
            elif text[j] in "})":
                if depth == 0:
                    break
                depth -= 1
            j += 1
        if j >= len(text) or text[j] != "}":
            continue
        tail = text[e:j]
        if tail.strip() and not re.search(r"\b(?:case\b|default\s*:)|^\s*\w+\s*:(?!:)", tail, re.MULTILINE):
            return text[:e] + text[j:], True
    return text, False


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"[{}]", "", text))


DEVICE_FN = re.compile(r"\b(?:__global__|__device__)\b(?![^;{()]*\b__host__\b)[^;{}]*?\)\s*(?:const\s*)?\{")


def _drop_device_bodies(code: str) -> str:
    """The host pass: bodies of ``__global__`` and ``__device__``-only functions emptied (no host code runs there)."""
    out, pos = [], 0
    for m in DEVICE_FN.finditer(code):
        if m.start() < pos:
            continue
        end = _match(code, m.end() - 1, "{", "}")
        if end < 0:
            continue
        out.append(code[pos:m.end()] + "}")
        pos = end
    return "".join(out) + code[pos:]


def view(text: str | None, env: Env) -> str:
    """``text`` as ``env``'s GPU compiles and dispatches it, normalized for comparison."""
    if not text:
        return ""
    local = Env(cc=env.cc, device=env.device, macros=dict(env.macros), funcs=dict(env.funcs))
    lines = preprocess(text, local).splitlines()
    directives = "\n".join(ln for ln in lines if ln.lstrip().startswith("#"))  # macro bodies are compared as written, never folded
    code = "\n".join(ln for ln in lines if not ln.lstrip().startswith("#"))
    if not env.device:
        code = _drop_device_bodies(code)
    return _norm(directives) + "\0" + _norm(fold(code, local))


def changed_on(before: str | None, after: str | None, common_before: str | None, common_after: str | None, cc: int) -> bool:
    """Whether a change to one file reaches a GPU of compute capability ``cc`` (device or host pass)."""
    for device in (True, False):
        if view(before, env_for(common_before, cc, device)) != view(after, env_for(common_after, cc, device)):
            return True
    return False


# ---- the message and the op test -------------------------------------------------------------------------------------------
NOT_OURS = re.compile(r"\b(volta|turing|pascal|maxwell|kepler|p100|v100|t4|jetson|tegra|orin|pre[- ]?ampere|integrated gpus?|igpus?|\buma\b|"
                      r"multi[- ]?gpus?|multiple gpus|several gpus|peer[- ]to[- ]peer|\bp2p\b|ring reduc\w*|split[- ]mode|tensor split|row split)\b|"
                      r"\b(?:arch|cc|compute capability)\s*<\s*(?:ampere|8\.?0|800)\b", re.IGNORECASE)
SPEED = re.compile(r"\b(perf(?:ormance)?|speed[- ]?ups?|faster|throughput|t/s|unroll\w*|occupancy|optimi[sz]\w*|latency)\b", re.IGNORECASE)
SYMPTOM = re.compile(r"\b(wrong|incorrect|garbage|gibberish|corrupt\w*|crash\w*|nans?|infs?|races?|hazards?|out[- ]of[- ]bounds|oob|illegal|"
                     r"asserts?|assertion|overflows?|truncat\w*|mismatch\w*|fail\w*|broken|segfault\w*|hangs?|memory fault|invalid|divergent|"
                     r"misalign\w*|alignment|bug|discrepanc\w*|non-?contiguous|transposed|views?|strides?|padding|negative|large|exceed\w*|"
                     r"limits?|65535|64k|more than|too many|uneven|nondetermin\w*|accumulat\w*|precision)\b", re.IGNORECASE)
LIMIT = re.compile(r"\b(large|exceed\w*|limits?|65535|64k|more than|grid[- ]?(?:dim|y|overflow)|overflow\w*|too many|batch(?:es)?)\b", re.IGNORECASE)
WRONG = re.compile(r"\b(wrong|incorrect|corrupt\w*|garbage|gibberish|nans?|crash\w*|illegal|out[- ]of[- ]bounds|memory fault)\b", re.IGNORECASE)
LAUNCH_LINE = re.compile(r"<<<|\b(?:num_blocks|nblocks|block_nums?|block_dims?|grid_dims?|dim3|nthreads|num_threads|block_size|"
                         r"threads_per_block|CUDA_\w+_BLOCK_SIZE)\b")
SELECTION_LINE = re.compile(r"^\s*(?:\}?\s*else\s+)?if\s*\(|^\s*(?:const\s+)?(?:static\s+)?bool\s+\w+\s*=|^\s*\w+\s*=\s*[^;]*(?:&&|\|\|)")


def _changed_lines(diff: str) -> list[tuple[str, str]]:
    """(sign, code) of each changed line with code on it."""
    out = []
    for ln in diff.splitlines():
        if ln.startswith(("+++", "---")) or not ln.startswith(("+", "-")):
            continue
        code = COMMENT.sub("", ln[1:]).strip()
        if code:
            out.append((ln[0], code))
    return out


def _launch_shape_only(lines: Sequence[tuple[str, str]]) -> bool:
    """Every changed line is launch configuration, and a launch's own arguments (after ``>>>``) are the same before and after."""
    if not lines or not all(LAUNCH_LINE.search(code) for _s, code in lines):
        return False
    args = {sign: sorted(re.sub(r"\s+", "", code.split(">>>", 1)[1]) for s2, code in lines if s2 == sign and ">>>" in code) for sign in "+-"}
    return args["+"] == args["-"]


def message_reasons(subject: str, body: str, kind: str, fault_class: str, diff: str, adds_cases: bool) -> list[str]:
    """Why the commit itself says our GPUs cannot grade the fault (empty when nothing does)."""
    out: list[str] = []
    if NOT_OURS.search(subject):
        out.append(f"names hardware we do not grade on ({NOT_OURS.search(subject).group(0)})")
    if kind != "bug":
        return out
    text = f"{subject}\n{body}"
    symptom = SYMPTOM.search(re.sub(r"\bfix\w*\b", "", text, flags=re.IGNORECASE))
    if SPEED.search(text) and not symptom:
        out.append("a speed change: the message names no wrong result")
    lines = _changed_lines(diff)
    if _launch_shape_only(lines) and not (LIMIT.search(text) or WRONG.search(text)):
        out.append("a launch-shape change: the message names no limit it exceeds and no wrong result")
    if fault_class == "precision" and lines and all(SELECTION_LINE.search(code) for _s, code in lines) and not symptom and not adds_cases:
        out.append("a selection between two kernels that both compute the result (the difference stays inside the case tolerance)")
    return out


def covered_ops(ops: Sequence[str], harness: str | None) -> list[str]:
    """The ops among ``ops`` the repository's op test builds (a ``ggml_<op>(`` call or the op's enum named in it)."""
    if harness is None:
        return list(ops)
    return [o for o in ops if re.search(rf"\bggml_{o.lower()}\s*\(|\bGGML_(?:UNARY_|GLU_)?OP_{o}\b", harness)]


@dataclass
class Reach:
    targets: list[str]
    reasons: list[str]

    @property
    def qualifies(self) -> bool:
        return bool(self.targets) and not self.reasons

    def to_json(self) -> dict[str, Any]:
        return {"targets": self.targets, "reasons": self.reasons, "qualifies": self.qualifies}


def reach(*, subject: str, body: str, kind: str, fault_class: str, diff: str, adds_cases: bool,
          files: Mapping[str, tuple[str | None, str | None]], common: tuple[str | None, str | None], ops: Sequence[str],
          harness: str | None, targets: Mapping[str, int] = TARGETS) -> Reach:
    """Which of our GPUs can grade a change, and why not when none can (module doc). ``files``: each changed source as
    (before, after); ``common``: the backend's ``common.cuh`` before and after; ``harness``: the op test at the change."""
    hit = [name for name, cc in targets.items() if any(changed_on(b, a, common[0], common[1], cc) for b, a in files.values())]
    reasons = message_reasons(subject, body, kind, fault_class, diff, adds_cases)
    if not hit:
        reasons.append(f"not compiled or dispatched on {' or '.join(targets)} (the change sits behind other GPUs' guards)")
    if ops and not covered_ops(ops, harness):
        reasons.append(f"the op test builds none of its ops ({', '.join(ops[:4])})")
    return Reach(targets=hit, reasons=reasons)
