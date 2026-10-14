"""Scripted controls for the long-context families: an RLM solver per family that scores 1.0 by writing the program a capable
model would write (read the document in bounded chunks through ``docs.read`` or ``CONTEXT``, compute the answer in the
kernel, submit it through ``answer.submit``), and a no-op that scores 0.

The solver keeps no hidden memory (``crucible.runtime.scripted_rlm``): what it computes is decided from the visible task
message (the question names everything), and whether it is done from its own visible ``SUBMITTED`` output.
"""

from __future__ import annotations

import re

from crucible.runtime.scripted_rlm import CallView, Reply, ScriptedRLM, message_text

READ_WORLD = """import json as _j, re as _re
_parts, _off = [], 0
while True:
    _r = _j.loads((await world.call('docs.read', offset=_off, n=65536))['content'])
    _parts.append(_r['text'])
    _off = _r['next']
    if _off >= _r['total'] or not _r['text']:
        break
DOC = ''.join(_parts)
print('READ', len(DOC))
"""
READ_CONTEXT = """import json as _j, re as _re
_parts, _off = [], 0
while True:
    _x = await CONTEXT.read(_off, 65536)
    _t = _x if isinstance(_x, str) else (_x.get('text') or '')
    if not _t:
        break
    _parts.append(_t)
    _off += len(_t)
DOC = ''.join(_parts)
print('READ', len(DOC))
"""
SUBMIT = "\n_s = await world.call('answer.submit', text=ANS)\nprint('SUBMITTED', _s.get('ok'), ANS[:120])"


def _oolong(q: str) -> str:
    m = re.search(r"How many data points of user (\d+) have the label '(\w+)'", q)
    if m:
        return (f"ANS = str(sum(1 for l in DOC.splitlines() if '|| User: {m.group(1)} ||' in l and l.endswith('|| Label: {m.group(2)}')))")
    m = re.search(r"How many data points have the label '(\w+)'", q)
    if m:
        return f"ANS = str(sum(1 for l in DOC.splitlines() if l.endswith('|| Label: {m.group(1)}')))"
    m = re.search(r"Is the label '(\w+)' more common, less common, or the same frequency as the label '(\w+)'", q)
    if m:
        return ("from collections import Counter as _C\n_k = _C(l.rsplit('|| Label: ', 1)[-1] for l in DOC.splitlines() if '|| Label: ' in l)\n"
                f"_a, _b = _k['{m.group(1)}'], _k['{m.group(2)}']\n"
                "ANS = 'more common' if _a > _b else 'less common' if _a < _b else 'same frequency'")
    return ("from collections import Counter as _C\n"
            "ANS = _C(l.rsplit('|| Label: ', 1)[-1] for l in DOC.splitlines() if '|| Label: ' in l).most_common(1)[0][0]")


def _mrcr(q: str) -> str:
    m = re.search(r"Prepend (\S{12}) to the (\d+)\w* \(1-indexed\) reply of the assistant to a request to (.+?)\. Do not", q)
    if not m:
        return "ANS = ''"
    h, k, ask = m.group(1), int(m.group(2)), m.group(3).lower()
    return ("_turns = _re.findall(r'User: (.*?)\\n\\nAssistant: (.*?)\\n(?=User: |\\Z)', DOC, _re.S)\n"
            f"_hits = [a for u, a in _turns if {ask!r} in u.lower()]\n"
            f"ANS = {h!r} + _hits[{k - 1}].strip()")


def _docqa(q: str) -> str:
    qs = re.findall(r"(\d+)\. \[Document: ([^\]]+)\] How many tonnes of (\w+) did the (.+?) facility process in (\d{4})\?", q)
    spec = [[int(n), d, mat, city, year] for n, d, mat, city, year in qs]
    return (f"_spec = {spec!r}\n_ans = {{}}\n"
            "for _n, _d, _m, _c, _y in _spec:\n"
            "    _i = DOC.find('=== Document: ' + _d + ' ===')\n"
            "    _k = DOC.find('=== Document: ', _i + 1)\n"
            "    _part = DOC[_i: _k if _k != -1 else len(DOC)]\n"
            "    _v = _re.search('The ' + _re.escape(_c) + r' facility processed ([\\d,]+) tonnes of ' + _m + ' in ' + _y, _part)\n"
            "    _ans[str(_n)] = _v.group(1).replace(',', '') if _v else ''\n"
            "ANS = _j.dumps(_ans)")


COMPUTE = {"oolong": _oolong, "mrcr": _mrcr, "docqa": _docqa}


class LongCtxSolver(ScriptedRLM):
    name = "scripted-longctx-solver"

    def __init__(self, family: str, *, model: str | None = None):
        if family not in COMPUTE:
            raise ValueError(f"family: {' | '.join(COMPUTE)}")
        self.family = family
        super().__init__(model=model or f"{self.name}:{family}")

    def decide(self, view: CallView) -> Reply:
        if view.role == "sub":
            return Reply("sub", "ok")
        if view.role == "summary":
            return Reply("summary", "COMPACTION: the document is in DOC; the answer is computed and submitted by one cell.")
        seen = "\n".join(message_text(m) for m in view.messages if m.get("role") in ("tool", "user"))
        if "SUBMITTED True" in seen:
            return Reply("finish", "Submitted.", "rlm.finish('answered')")
        task = next((message_text(m) for m in view.messages if m.get("role") == "user"), "")
        read = READ_CONTEXT if view.has_context_api() and "CONTEXT" in task else READ_WORLD
        return Reply("solve", "Read the document in chunks, compute the answer in the kernel, submit it.", read + COMPUTE[self.family](task) + SUBMIT)
