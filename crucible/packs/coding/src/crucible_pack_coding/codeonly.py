"""The code of a source file, as the coding pack's dense reward Φ reads it (M14 audit V4, campaign B2).

Φ checks the oracle's milestones (tokens added, calls added, files created) with text queries. Text is not code: before
this module, the oracle's tokens in **string literals** (``static const char *s = "cudaMallocAsync(";``) or in **dead
code** (``#if 0``, ``if False:``) reached Φ 1.0 on every bench seed with nothing migrated. A token earns credit here only
where it takes part in the code the compiler or the interpreter runs:

- **comments** are removed and the **contents of string and char literals** are blanked (the quotes stay), for C-family
  sources by a lexer (raw strings, char literals, C++14 digit separators), for Python by ``tokenize`` (f-string text too),
  for CMake (quoted and bracket arguments, bracket comments) and TOML (basic, literal and multi-line strings);
- **preprocessor-dead** blocks go: ``#if``/``#elif`` on a constant expression is evaluated, and ``#ifdef``/``defined()``
  of a macro the agent introduced and never defines anywhere (``undefined``) is false;
- **constant-false branches** go: C ``if (0)``/``if (false)``/``while (0)`` (``if constexpr`` too, and the dead ``else`` of a
  constant-true ``if``), Python ``if``/``while`` on a constant, ``x if False else y``, ``False and x``, CMake
  ``if(FALSE)``…``endif()``;
- **unreachable statements** go: Python statements after ``return``/``raise``/``break``/``continue`` in the same block,
  C statements after ``return``/``throw``/``break``/``continue``/``goto`` up to the next label or the end of the block;
- **unused new functions** go (``unused_new``): a function, class or CMake ``function``/``macro`` whose name the base
  never had and that nothing else references (Python ``Name``/``Attribute``/import, C uses inside function bodies or
  initializers, any mention in another language's code), iterated to a fixpoint;
- a Python or TOML file that does not **parse** has no code.

Line structure is kept (blanked text becomes spaces), so anchored blocks are still found by line. ``code_only`` is the
per-file view; ``crucible_pack_coding.studio.code_state`` adds the cross-file passes (undefined macros, unused new
functions) over a whole ``FileState``.
"""

from __future__ import annotations

import ast
import io
import operator
import re
import tokenize
from functools import lru_cache

C_LIKE = re.compile(r"\.(?:c|cc|cpp|cxx|cu|cuh|h|hh|hpp|hxx|inl|ipp|tpp|cl|metal|glsl|hip|m|mm|java|js|ts|rs|go|swift|kt)$", re.IGNORECASE)
CMAKE = re.compile(r"(?:^|/)CMakeLists\.txt$|\.cmake$", re.IGNORECASE)
TOML = re.compile(r"\.toml$", re.IGNORECASE)
PY = re.compile(r"\.pyi?$")
HASH_COMMENT = re.compile(r"(?:^|/)CMakeLists\.txt$|\.(?:cmake|sh|bash|yaml|yml|toml|cfg|ini|mk)$|(?:^|/)(?:Makefile|Dockerfile)$", re.IGNORECASE)
IDENT = re.compile(r"[A-Za-z_]\w*")


def lang(path: str) -> str:
    if C_LIKE.search(path):
        return "c"
    if PY.search(path):
        return "py"
    if CMAKE.search(path):
        return "cmake"
    if TOML.search(path):
        return "toml"
    if HASH_COMMENT.search(path):
        return "hash"
    return "text"


def _blank(chars: list[str], a: int, b: int) -> None:
    for k in range(max(a, 0), min(b, len(chars))):
        if chars[k] != "\n":
            chars[k] = " "


# ================================================================================================ C family

_RAW_PREFIX = re.compile(r"(?:u8|[uUL])?R$")


def c_lex(text: str) -> str:
    """Comments removed and literal contents blanked (quotes, prefixes and every newline kept)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and text.startswith("//", i):
            j = i
            while True:  # a line comment continues over a backslash-newline
                k = text.find("\n", j)
                if k < 0:
                    j = n
                    break
                if text[k - 1] == "\\":
                    out.append("\n")
                    j = k + 1
                    continue
                j = k
                break
            i = j
        elif c == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * text.count("\n", i, j))
            i = j
        elif c == '"':
            pre = re.search(r"[A-Za-z0-9_]*$", "".join(out[-4:])).group(0)
            if _RAW_PREFIX.search(pre):  # R"delim( ... )delim"
                k = text.find("(", i + 1)
                delim = text[i + 1:k] if 0 <= k - i - 1 <= 16 else None
                end = text.find(")" + delim + '"', k) if delim is not None else -1
                if end >= 0:
                    out.append('"' + "\n" * text.count("\n", i, end) + '"')
                    i = end + len(delim) + 2
                    continue
            j = i + 1
            while j < n and text[j] != '"' and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append('""' + ("\n" * text.count("\n", i + 1, min(j, n))))
            i = j + 1 if j < n and text[j] == '"' else j
        elif c == "'":
            run = re.search(r"[A-Za-z0-9_.]*$", "".join(out[-40:])).group(0)
            if run[:1].isdigit():  # a C++14 digit separator (1'000'000)
                out.append(c)
                i += 1
                continue
            j = i + 1
            while j < n and text[j] != "'" and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append("''" + ("\n" * text.count("\n", i + 1, min(j, n))))
            i = j + 1 if j < n and text[j] == "'" else j
        else:
            out.append(c)
            i += 1
    return "".join(out)


_PP = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|elifdef|elifndef|else|endif)\b(.*)$")
_SUFFIX = re.compile(r"\b(0[xX][0-9a-fA-F]+|\d+)(?:[uU](?:ll|LL|l|L)?|(?:ll|LL|l|L)[uU]?)\b")
_C_CONST = {"true": "1", "false": "0", "nullptr": "0", "NULL": "0"}


_VALUES = {"U": (0, 1), "V": (0, 1, -1, 2, 10**9, -(10**9))}  # defined() is 0 / 1; a macro's value is any integer


def _const_eval(expr: str, *, unknown_ok: int = 3) -> bool | None:
    """A constant integer expression of the preprocessor or of a C condition: True / False, or None when it depends on
    something unknown. ``__U<k>__`` (a ``defined()`` of unknown answer) and ``__V<k>__`` (a macro of unknown value) are
    unknowns: the value is decided only if every probe assignment agrees (``0 && X``). Ternaries, casts and bitwise
    and/or/xor (whose precedence differs from Python's) are not decided."""
    import itertools

    e = _SUFFIX.sub(r"\1", expr.strip())
    e = re.sub(r"\b(true|false|nullptr|NULL)\b", lambda m: _C_CONST[m.group(1)], e)
    if not e or re.search(r"[?:\[\]{};\"'.]|(?<![&|])[&|](?![&|])|\^|\bsizeof\b", e):
        return None
    unk = sorted(set(re.findall(r"__[UV]\d+__", e)))
    if len(unk) > unknown_ok or [t for t in IDENT.findall(e) if not re.fullmatch(r"__[UV]\d+__", t)]:
        return None
    py = e.replace("&&", " and ").replace("||", " or ")
    py = re.sub(r"!(?!=)", " not ", py).replace("/", "//")
    py = re.sub(r"\b0+(\d)", r"\1", py)  # octal-looking literals
    vals = set()
    for combo in itertools.product(*(_VALUES[u[2]] for u in unk)):
        try:
            vals.add(bool(eval(py, {"__builtins__": {}}, dict(zip(unk, combo, strict=True)))))  # integers and operators only (checked above)
        except Exception:  # noqa: BLE001 - e.g. a division by a probed 0
            return None
        if len(vals) > 1:
            return None
    return vals.pop() if vals else None


def _pp_cond(d: str, rest: str, undefined: frozenset[str]) -> bool | None:
    rest = rest.strip()
    if d in ("ifdef", "ifndef", "elifdef", "elifndef"):
        name = (IDENT.match(rest) or [None])[0]
        if name is None or name not in undefined:
            return None
        return d in ("ifndef", "elifndef")
    k = [0]

    def unknown(kind: str) -> str:
        k[0] += 1
        return f"__{kind}{k[0]}__"

    e = re.sub(r"\bdefined\s*\(\s*(\w+)\s*\)|\bdefined\s+(\w+)", lambda m: "0" if (m.group(1) or m.group(2)) in undefined else unknown("U"), rest)
    e = re.sub(r"\b(?!__[UV]\d+__)(?!(?:true|false)\b)[A-Za-z_]\w*\b", lambda m: "0" if m.group(0) in undefined else unknown("V"), e)
    return _const_eval(e)


def c_preprocess(text: str, undefined: frozenset[str] = frozenset()) -> str:
    """Lines of preprocessor-dead branches blanked (directives of a live nesting level are kept). A branch whose
    condition is unknown is kept, and so is every other branch of it that is not decided."""
    lines = text.split("\n")
    out = list(lines)
    stack: list[list] = []  # [parent_live, a branch surely taken]
    live = True
    i = 0
    while i < len(lines):
        m = _PP.match(lines[i])
        j = i
        while lines[j].rstrip().endswith("\\") and j + 1 < len(lines):  # a continued directive or line
            j += 1
        if not m:
            if not live:
                for k in range(i, j + 1):
                    out[k] = ""
            i = j + 1
            continue
        d = m.group(1)
        rest = " ".join([m.group(2).rstrip("\\")] + [ln.rstrip().rstrip("\\") for ln in lines[i + 1:j + 1]])
        if d in ("if", "ifdef", "ifndef"):
            v = _pp_cond(d, rest, undefined) if live else False
            stack.append([live, v is True])
            dir_live, live = stack[-1][0], live and v is not False
        elif d in ("elif", "elifdef", "elifndef") and stack:
            parent, taken = stack[-1]
            v = False if taken else _pp_cond(d[2:] if d != "elif" else "if", rest, undefined)
            stack[-1][1] = taken or v is True
            dir_live, live = parent, parent and v is not False and not taken
        elif d == "else" and stack:
            parent, taken = stack[-1]
            dir_live, live = parent, parent and not taken
        elif d == "endif" and stack:
            dir_live = live = stack.pop()[0]
        else:
            dir_live = live
        if not dir_live:
            for k in range(i, j + 1):
                out[k] = ""
        i = j + 1
    return "\n".join(out)


def _match(text: str, i: int, op: str, cl: str) -> int:
    """Index after the bracket closing the one at ``i`` (len(text) when unbalanced)."""
    depth, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == op:
            depth += 1
        elif c == cl:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


_KW = re.compile(r"(if|for|while|switch|do|else|try|catch)\b")


def _stmt_end(text: str, i: int) -> int:
    """Index after the statement starting at ``i`` (C family, on lexed text)."""
    i = _skip_ws(text, i)
    n = len(text)
    if i >= n:
        return n
    if text[i] == "{":
        return _match(text, i, "{", "}")
    m = _KW.match(text, i)
    if m and m.group(1) in ("if", "for", "while", "switch", "catch"):
        j = _skip_ws(text, m.end())
        if m.group(1) == "if" and text.startswith("constexpr", j):
            j = _skip_ws(text, j + 9)
        if j < n and text[j] == "(":
            j = _stmt_end(text, _match(text, j, "(", ")"))
            if m.group(1) == "if":
                k = _skip_ws(text, j)
                if re.match(r"else\b", text[k:k + 5]):
                    j = _stmt_end(text, k + 4)
            return j
    if m and m.group(1) in ("do", "else", "try"):
        j = _stmt_end(text, m.end())
        k = _skip_ws(text, j)
        if m.group(1) == "do" and text.startswith("while", k):
            k = _skip_ws(text, k + 5)
            if k < n and text[k] == "(":
                k = _skip_ws(text, _match(text, k, "(", ")"))
                return k + 1 if k < n and text[k] == ";" else k
        while m.group(1) == "try" and text.startswith("catch", k):
            j = _stmt_end(text, k)
            k = _skip_ws(text, j)
        return j
    depth = 0
    while i < n:  # an expression or declaration statement: up to its ';' at depth 0
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                return i  # the end of the enclosing block: a statement without ';' (a macro)
            depth -= 1
        elif c == ";" and depth == 0:
            return i + 1
        elif c == "\n" and depth == 0 and text[i + 1:i + 2].lstrip().startswith("#"):
            return i
        i += 1
    return n


_COND = re.compile(r"\b(if|while)\s*(constexpr\s*)?\(")


def _on_directive(text: str, pos: int) -> bool:
    return text[text.rfind("\n", 0, pos) + 1:pos].lstrip().startswith("#")


def c_dead_branches(text: str) -> str:
    """``if (<constant false>) S`` (its ``else`` kept), the ``else`` of ``if (<constant true>)``, ``while (<false>) S``."""
    chars = list(text)
    pos = 0
    while (m := _COND.search(text, pos)) is not None:
        pos = m.end()
        if chars[m.start()] == " " or _on_directive(text, m.start()):  # already blanked, or a directive (``#if (0)``)
            continue
        po = m.end() - 1
        pc = _match(text, po, "(", ")")
        v = _const_eval(text[po + 1:pc - 1], unknown_ok=0)
        if m.group(1) == "while":
            prev = text[:m.start()].rstrip()
            k = _skip_ws(text, pc)
            if v is False and not (k < len(text) and text[k] == ";") and not prev.endswith("}"):  # not a do-while
                _blank(chars, m.start(), _stmt_end(text, pc))
            continue
        if v is None:
            continue
        body_end = _stmt_end(text, pc)
        k = _skip_ws(text, body_end)
        has_else = re.match(r"else\b", text[k:k + 5]) is not None
        if v is False:
            _blank(chars, m.start(), body_end)
            if has_else:
                _blank(chars, k, k + 4)
        elif has_else:
            _blank(chars, k, _stmt_end(text, k + 4))
    return "".join(chars)


_JUMP = re.compile(r"\b(return|throw|break|continue|goto)\b")
_LABEL = re.compile(r"\s*(?:case\b|default\s*:|[A-Za-z_]\w*\s*:(?!:))")


def c_unreachable(text: str) -> str:
    """Statements after a jump, up to the next label (``case``, ``default:``, ``name:``), a directive or the end of the
    block. Only a jump that is a statement of the block itself counts (after ``;``, ``{``, ``}`` or a label): the body of
    an unbraced ``if`` / ``else`` / loop, or a jump in a macro definition, is conditional."""
    chars = list(text)
    pos = 0
    while (m := _JUMP.search(text, pos)) is not None:
        pos = m.end()
        prev = text[:m.start()].rstrip()
        if _on_directive(text, m.start()) or not prev or prev[-1] not in ";{}:" or prev.endswith("::"):
            continue
        end = _stmt_end(text, m.start())
        if end <= m.start() or end > len(text) or text[end - 1:end] != ";":
            continue
        i = end
        while True:
            k = _skip_ws(text, i)
            if k >= len(text) or text[k] in "})" or text[k] == "#" or _LABEL.match(text, k):
                break
            j = _stmt_end(text, k)
            if j <= k:
                break
            _blank(chars, k, j)
            i = j
    return "".join(chars)


_CTRL = frozenset({"if", "for", "while", "switch", "catch", "return", "sizeof", "alignof", "decltype", "typeid", "static_assert", "defined", "new",
                   "delete", "throw", "case", "do", "else", "operator", "__attribute__", "__declspec", "alignas", "noexcept", "requires", "template"})
_NS_OPEN = re.compile(r"(?:\bnamespace(?:\s+[\w:]+)?|\bextern\s*\"\"|\b(?:class|struct|union|enum)(?:\s+class)?\b[^;{()]*|=)\s*$")


def c_functions(text: str) -> tuple[list[tuple[str, int, int, int]], list[tuple[str, int, bool]]]:
    """Function definitions ``(name, decl_start, name_pos, body_end)`` and identifier occurrences ``(name, pos, in_body)``
    of lexed text. A ``{`` after ``) [qualifiers]`` that follows a name opens a function body; namespace, class and
    initializer braces do not."""
    defs: list[tuple[str, int, int, int]] = []
    stack: list[bool] = []  # brace kind: True = a function body (or inside one)
    occ: list[tuple[str, int, bool]] = []
    last_sep = -1
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "#" and (i == 0 or text[i - 1] == "\n" or not text[text.rfind("\n", 0, i) + 1:i].strip()):
            j = i
            while True:
                k = text.find("\n", j)
                if k < 0 or text[k - 1] != "\\":
                    break
                j = k + 1
            k = n if k < 0 else k
            for t in IDENT.finditer(text, i, k):
                occ.append((t.group(0), t.start(), True))  # a macro body may use anything: counted as a use
            last_sep = k if not any(stack) else last_sep
            i = k
            continue
        if c == "{":
            in_fn = any(stack)
            is_fn = False
            if not in_fn:
                head = text[last_sep + 1:i]
                hs = head.rstrip()
                if not _NS_OPEN.search(hs):
                    q = hs
                    q = re.sub(r"(?:\s*(?:const|volatile|noexcept(?:\s*\([^()]*\))?|override|final|mutable|&&?|->\s*[\w:<>,\s*&]+|__\w+__(?:\s*\([^()]*\))?))+$", "", q)
                    q = re.sub(r"\)\s*:\s*[\s\S]*$", ")", q) if ")" in q and re.search(r"\)\s*:[^:]", q) else q  # constructor initializer list
                    if q.endswith(")"):
                        po = _open_paren(q, len(q) - 1)
                        nm = re.search(r"(~?[A-Za-z_]\w*)\s*(?:<[^()]*>)?\s*$", q[:po]) if po >= 0 else None
                        if nm and nm.group(1).lstrip("~") not in _CTRL:
                            is_fn = True
                            defs.append((nm.group(1).lstrip("~"), last_sep + 1, last_sep + 1 + nm.start(1), -1))
            stack.append(in_fn or is_fn)
            if is_fn:
                defs[-1] = (defs[-1][0], defs[-1][1], defs[-1][2], _match(text, i, "{", "}"))
            i += 1
            if not in_fn:
                last_sep = i - 1
            continue
        if c == "}":
            if stack:
                stack.pop()
            i += 1
            if not any(stack):
                last_sep = i - 1
            continue
        if c == ";" and not any(stack):
            last_sep = i
            i += 1
            continue
        m = IDENT.match(text, i) if (c.isalpha() or c == "_") and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")) else None
        if m:
            occ.append((m.group(0), m.start(), any(stack)))
            i = m.end()
            continue
        i += 1
    return defs, occ


def _open_paren(s: str, close: int) -> int:
    depth = 0
    for k in range(close, -1, -1):
        if s[k] == ")":
            depth += 1
        elif s[k] == "(":
            depth -= 1
            if depth == 0:
                return k
    return -1


@lru_cache(maxsize=4096)
def c_code(text: str, undefined: frozenset[str] = frozenset()) -> str:
    return c_unreachable(c_dead_branches(c_preprocess(c_lex(text), undefined)))


# ================================================================================================ Python

def _const(node: ast.AST) -> tuple[bool, object]:
    if isinstance(node, ast.Constant):
        return True, node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        k, v = _const(node.operand)
        return (True, not v) if k else (False, None)
    if isinstance(node, ast.BoolOp):
        for x in node.values:
            k, v = _const(x)
            if not k:
                return False, None
            if isinstance(node.op, ast.And) and not v or isinstance(node.op, ast.Or) and v:
                return True, v
        return True, v
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        (k1, a), (k2, b) = _const(node.left), _const(node.comparators[0])
        ops = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
               ast.Is: operator.is_, ast.IsNot: operator.is_not}
        f = ops.get(type(node.ops[0]))
        if k1 and k2 and f:
            try:
                return True, f(a, b)
            except TypeError:
                return False, None
    return False, None


_TERMINAL = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def py_dead_spans(tree: ast.AST) -> list[ast.AST]:
    """Nodes whose code never runs: constant-false branches, the dead arm of a constant ternary or and/or, statements after
    a jump in the same block."""
    dead: list[ast.AST] = []

    def block(stmts: list[ast.stmt]) -> None:
        for k, s in enumerate(stmts):
            if isinstance(s, _TERMINAL):
                dead.extend(stmts[k + 1:])
                return

    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            v = getattr(node, field, None)
            if isinstance(v, list) and v and isinstance(v[0], ast.stmt):
                block(v)
        if isinstance(node, ast.ExceptHandler | ast.match_case):
            block(node.body)
        if isinstance(node, ast.If | ast.While):
            known, v = _const(node.test)
            if known and not v:
                dead.append(node.test)
                dead.extend(node.body)
            elif known and isinstance(node, ast.If):
                dead.extend(node.orelse)
        elif isinstance(node, ast.IfExp):
            known, v = _const(node.test)
            if known:
                dead.append(node.orelse if v else node.body)
        elif isinstance(node, ast.BoolOp):
            for k, x in enumerate(node.values[:-1]):
                known, v = _const(x)
                if known and ((isinstance(node.op, ast.And) and not v) or (isinstance(node.op, ast.Or) and v)):
                    dead.extend(node.values[k + 1:])
                    break
    return dead


def _offsets(text: str) -> tuple[list[int], list[str]]:
    lines = text.split("\n")
    starts, acc = [], 0
    for ln in lines:
        starts.append(acc)
        acc += len(ln) + 1
    return starts, lines


def _char_col(line: str, byte_col: int) -> int:
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", "ignore"))


def py_code(text: str, *, drop: frozenset[str] | set[str] = frozenset()) -> str | None:
    """Comments, docstrings, string contents and dead code blanked; ``None`` when the file does not parse. ``drop``:
    names of top-level / class-level definitions to blank (unused new functions)."""
    text = text.replace("\r\n", "\n")
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    starts, lines = _offsets(text)
    chars = list(text)

    def span(node: ast.AST) -> None:
        a = starts[node.lineno - 1] + _char_col(lines[node.lineno - 1], node.col_offset)
        el = node.end_lineno or node.lineno
        b = starts[el - 1] + _char_col(lines[el - 1], node.end_col_offset or 0)
        _blank(chars, a, b)

    for node in py_dead_spans(tree):
        span(node)
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            span(node)  # docstrings and bare string statements
        if drop and isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and node.name in drop:
            for d in node.decorator_list:
                span(d)
            span(node)
    try:
        for t in tokenize.generate_tokens(io.StringIO(text).readline):
            if t.type == tokenize.COMMENT:
                _blank(chars, starts[t.start[0] - 1] + t.start[1], starts[t.end[0] - 1] + t.end[1])
            elif t.type == tokenize.STRING:
                q = re.match(r"[A-Za-z]*('''|\"\"\"|'|\")", t.string)
                a = starts[t.start[0] - 1] + t.start[1] + (q.end() if q else 0)
                b = starts[t.end[0] - 1] + t.end[1] - (len(q.group(1)) if q else 0)
                _blank(chars, a, b)
            elif t.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
                _blank(chars, starts[t.start[0] - 1] + t.start[1], starts[t.end[0] - 1] + t.end[1])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    return "".join(chars)


def py_defs_and_uses(text: str, drop: frozenset[str] = frozenset()) -> tuple[dict[str, list[ast.AST]], dict[str, int]] | None:
    """Definitions (functions and classes, at any level) and references of names in one file's AST; dead code and the
    definitions in ``drop`` are skipped. A reference is a ``Name``, an ``Attribute`` name, an imported name or an
    ``__all__`` entry; one inside its own definition does not count (recursion)."""
    try:
        tree = ast.parse(text.replace("\r\n", "\n"))
    except (SyntaxError, ValueError):
        return None
    dead = {id(n) for n in py_dead_spans(tree)}
    defs: dict[str, list[ast.AST]] = {}
    uses: dict[str, int] = {}

    def use(name: str) -> None:
        uses[name] = uses.get(name, 0) + 1

    def walk(node: ast.AST, inside: frozenset[str]) -> None:
        if id(node) in dead:
            return
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name in drop:
                return
            defs.setdefault(node.name, []).append(node)
            inside = inside | {node.name}
        name = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.alias):
            name = node.name.split(".")[-1]
        elif (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
              and isinstance(node.value, ast.List | ast.Tuple)):
            for e in node.value.elts:
                if isinstance(e, ast.Constant) and isinstance(e.value, str):
                    use(e.value)
        if name is not None and name not in inside:
            use(name)
        for ch in ast.iter_child_nodes(node):
            walk(ch, inside)

    walk(tree, frozenset())
    return defs, uses


# ================================================================================================ CMake

def cmake_lex(text: str) -> str:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "#":
            m = re.match(r"#\[(=*)\[", text[i:i + 20])
            if m:
                j = text.find("]" + m.group(1) + "]", i)
                j = n if j < 0 else j + len(m.group(1)) + 2
                out.append("\n" * text.count("\n", i, j))
                i = j
                continue
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append('"' + "\n" * text.count("\n", i + 1, min(j, n)) + '"')
            i = j + 1
        elif c == "[" and (m := re.match(r"\[(=*)\[", text[i:i + 20])):
            j = text.find("]" + m.group(1) + "]", i)
            j = n if j < 0 else j + len(m.group(1)) + 2
            out.append("[[" + "\n" * text.count("\n", i, j) + "]]")
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


_CMD = re.compile(r"(?m)(?<![\w$])([A-Za-z_]\w*)[ \t]*\(")
_CM_FALSE = re.compile(r"^(?:0+|OFF|NO|FALSE|N|IGNORE|NOTFOUND|\"\"|.*-NOTFOUND)$", re.IGNORECASE)
_CM_TRUE = re.compile(r"^(?:[1-9]\d*|ON|YES|TRUE|Y)$", re.IGNORECASE)


def cmake_commands(text: str) -> list[tuple[str, int, int, str]]:
    """``(name lowercased, start, end, args)`` of each command invocation in lexed text."""
    out, pos = [], 0
    while (m := _CMD.search(text, pos)) is not None:
        po = m.end() - 1
        pc = _match(text, po, "(", ")")
        out.append((m.group(1).lower(), m.start(), pc, text[po + 1:pc - 1]))
        pos = pc
    return out


def _cm_cond(args: str) -> bool | None:
    toks = args.split()
    neg = False
    while toks and toks[0].upper() == "NOT":
        neg, toks = not neg, toks[1:]
    if len(toks) != 1:
        return None
    v = True if _CM_TRUE.match(toks[0]) else False if _CM_FALSE.match(toks[0]) else None
    return None if v is None else v != neg


def cmake_code(text: str, *, drop: frozenset[str] = frozenset()) -> str:
    text = cmake_lex(text)
    chars = list(text)
    stack: list[list] = []
    live = True
    fn_stack: list[tuple[str, int]] = []
    for name, a, b, args in cmake_commands(text):
        dir_live = live
        if name == "if":
            v = _cm_cond(args) if live else False
            stack.append([live, v is True])
            live = live and v is not False
        elif name == "elseif" and stack:
            parent, taken = stack[-1]
            v = False if taken else _cm_cond(args)
            stack[-1][1] = taken or v is True
            dir_live, live = parent, parent and v is not False and not taken
        elif name == "else" and stack:
            parent, taken = stack[-1]
            dir_live, live = parent, parent and not taken
        elif name == "endif" and stack:
            dir_live = live = stack.pop()[0]
        if not dir_live or not live and name not in ("if", "elseif", "else", "endif"):
            _blank(chars, a, b)
        if name in ("function", "macro") and not args.split():
            _blank(chars, a, b)  # a function without a name is a configure error: it never runs
        elif name in ("function", "macro"):
            fn_stack.append((args.split()[0].lower(), a))
        elif name in ("endfunction", "endmacro") and fn_stack:
            fname, fa = fn_stack.pop()
            if fname in drop:
                _blank(chars, fa, b)
    return "".join(chars)


# ================================================================================================ TOML

def toml_code(text: str) -> str | None:
    import tomllib

    try:
        tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError):
        return None
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "#":
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif c in "\"'":
            q = c * 3 if text.startswith(c * 3, i) else c
            j = i + len(q)
            while j < n and not text.startswith(q, j) and (len(q) == 3 or text[j] != "\n"):
                j += 2 if (c == '"' and text[j] == "\\") else 1
            while len(q) == 3 and text.startswith(c, j + 3):  # up to two quotes may end a multi-line string's content
                j += 1
            out.append(q + "\n" * text.count("\n", i + len(q), min(j, n)) + q)
            i = j + len(q)
        else:
            out.append(c)
            i += 1
    return "".join(out)


# ================================================================================================ per-file view

@lru_cache(maxsize=8192)
def code_only(path: str, text: str | None, undefined: frozenset[str] = frozenset(), drop: frozenset[str] = frozenset()) -> str | None:
    """The code of one file (see the module docstring). A file with no code reads as absent, so an empty, comment-only or
    unparseable file does not meet a ``created`` milestone."""
    if text is None:
        return None
    kind = lang(path)
    if kind == "c":
        text = c_code(text, undefined)
        if drop:
            text = c_drop(text, drop)
    elif kind == "py":
        text = py_code(text, drop=set(drop))
    elif kind == "cmake":
        text = cmake_code(text, drop=drop)
    elif kind == "toml":
        text = toml_code(text)
    elif kind == "hash":
        text = "\n".join(re.sub(r"(^|\s)#.*$", r"\1", ln) for ln in text.split("\n"))
    return text if text is not None and text.strip() else None


def c_drop(text: str, drop: frozenset[str]) -> str:
    """Definitions of the named functions blanked, from the start of the declaration to the closing brace."""
    defs, _ = c_functions(text)
    chars = list(text)
    for name, a, _p, b in defs:
        if name in drop and b > 0:
            _blank(chars, a, b)
    return "".join(chars)


# ================================================================================================ cross-file passes

_EXTERNAL_MACRO = re.compile(r"CUDA|CUDART|NVCC|THRUST|CUB|HIP|ROCM|TORCH|OMP|OPENMP|OPENACC|CLANG|GNUC|GCC|MSVC|WIN|APPLE|LINUX|NDEBUG|DEBUG|"
                             r"VERSION|ARCH|SM_|PTX|SYCL|INTEL|AMD|NVIDIA|PYBIND|PYTHON|EIGEN|BOOST|PCL_", re.IGNORECASE)


def undefined_macros(files: dict[str, str | None], base_idents: set[str]) -> frozenset[str]:
    """Macros tested with ``#ifdef``/``#ifndef``/``defined()`` in the C-family files that the agent introduced (no base
    file mentions them), that no file defines (``#define``, a ``-D``, ``add_definitions`` / ``*_compile_definitions`` in
    any file's raw text, so a build-system flag keeps it unknown) and that are not toolchain or reserved names."""
    tested: set[str] = set()
    for p, t in files.items():
        if t is None or lang(p) != "c":
            continue
        for ln in c_lex(t).split("\n"):
            m = re.match(r"\s*#\s*(?:el)?if(n?def)?\b(.*)", ln)
            if not m:
                continue
            if m.group(1):
                tested.update(IDENT.findall(m.group(2))[:1])
            else:
                tested.update(x or y for x, y in re.findall(r"\bdefined\s*\(\s*(\w+)\s*\)|\bdefined\s+(\w+)", m.group(2)))
    out = set()
    for name in tested:
        if name in base_idents or name.startswith("__") or re.match(r"_[A-Z]", name) or _EXTERNAL_MACRO.search(name):
            continue
        defined = any(t is not None and re.search(rf"(?m)^\s*#\s*define\s+{name}\b|-D\s*{name}\b|\b{name}\s*=|"
                                                  rf"(?:add_definitions|_compile_definitions)\s*\([^)]*\b{name}\b", t)
                      for t in files.values())
        if not defined:
            out.add(name)
    return frozenset(out)


_KEEP_PY = re.compile(r"^(?:__\w+__|main|test\w*|setUp\w*|tearDown\w*)$")


def unused_new(files: dict[str, str | None], base_idents: set[str], undefined: frozenset[str] = frozenset(), *, rounds: int = 6) -> dict[str, frozenset[str]]:
    """Per file, the names of definitions to drop: functions / classes / CMake functions and macros whose name no base
    file has and that nothing outside their own definition references, in any file (decorated Python definitions,
    ``main`` and test functions are kept: registration or discovery reaches them). A function used only by dropped ones
    is dropped in a later round."""
    base_lower = {b.lower() for b in base_idents}
    drop: dict[str, set[str]] = {p: set() for p in files}
    for _ in range(rounds):
        uses: dict[str, int] = {}
        cands: dict[str, set[str]] = {}  # name -> paths defining it

        def use(name: str, uses: dict[str, int] = uses) -> None:
            uses[name] = uses.get(name, 0) + 1

        for p, t in files.items():
            if t is None:
                continue
            k = lang(p)
            if k == "py":
                r = py_defs_and_uses(t, frozenset(drop[p]))
                if r is None:
                    continue
                defs, u = r
                for name, nodes in defs.items():
                    if not any(getattr(nd, "decorator_list", None) for nd in nodes) and not _KEEP_PY.match(name):
                        cands.setdefault(name, set()).add(p)
                for name, c in u.items():
                    uses[name] = uses.get(name, 0) + c
            elif k == "c":
                code = c_code(t, undefined)
                code = c_drop(code, frozenset(drop[p])) if drop[p] else code
                defs, occ = c_functions(code)
                def_pos = {pos for _n, _a, pos, _b in defs}
                own = [(a, b, name) for name, a, _pos, b in defs if b > 0]
                for name, pos, in_body in occ:
                    if pos in def_pos or any(a <= pos < b and nm == name for a, b, nm in own):
                        continue  # the definition itself, or recursion
                    if in_body or _not_a_declaration(code, pos, name):
                        use(name)
                for name, _a, _pos, b in defs:
                    if b > 0 and name != "main":
                        cands.setdefault(name, set()).add(p)
            elif k == "cmake":
                for name, _a, _b, args in cmake_commands(cmake_code(t, drop=frozenset(drop[p]))):
                    if name in ("function", "macro"):
                        fname = (args.split() or [""])[0].lower()
                        if fname:
                            cands.setdefault(fname, set()).add(p)
                        for tok in IDENT.findall(" ".join(args.split()[1:])):
                            use(tok)
                    else:
                        use(name)
                        for tok in IDENT.findall(args):
                            use(tok)
            else:
                for tok in IDENT.findall(code_only(p, t) or ""):
                    use(tok)
        lower_uses = {}
        for name, c in uses.items():
            lower_uses[name.lower()] = lower_uses.get(name.lower(), 0) + c
        changed = False
        for name, paths in cands.items():
            cm = any(lang(p) == "cmake" for p in paths)
            if (name.lower() in base_lower) if cm else (name in base_idents):
                continue
            if (lower_uses.get(name.lower(), 0) if cm else uses.get(name, 0)):
                continue
            for p in paths:
                if name not in drop[p]:
                    drop[p].add(name)
                    changed = True
        if not changed:
            break
    return {p: frozenset(v) for p, v in drop.items() if v}


def _not_a_declaration(code: str, pos: int, name: str) -> bool:
    """At namespace or class scope, ``T name(...);`` declares; anything else (an initializer, a function pointer in a
    table, an explicit instantiation's argument) uses."""
    j = _skip_ws(code, pos + len(name))
    if j < len(code) and code[j] == "(":
        k = _skip_ws(code, _match(code, j, "(", ")"))
        k2 = re.match(r"(?:\s*(?:const|noexcept|override|final|=\s*(?:0|default|delete)))*\s*;", code[k:])
        before = code[:pos].rstrip()
        if k2 and before and (before[-1].isalnum() or before[-1] in "_*&>"):
            return False
    return True
