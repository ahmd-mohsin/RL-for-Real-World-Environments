"""Workspace file operations for policy tools. STANDALONE: stdlib only; runs locally or in containers.

Ops: list {path, depth}, read {path, offset, limit}, write {path, content}, edit {path, old, new, all}.
Paths are absolute and already confined by the caller.
"""

import os


def _list(a):
    base, depth = a["path"], int(a.get("depth", 1))
    if not os.path.isdir(base):
        raise FileNotFoundError(f"not a directory: {base}")
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        rel = os.path.relpath(dirpath, base)
        level = 0 if rel == "." else rel.count(os.sep) + 1
        dirnames[:] = sorted(d for d in dirnames if d != ".git")
        if level >= depth:
            dirnames[:] = []
        prefix = "" if rel == "." else rel + "/"
        for d in dirnames:
            out.append(prefix + d + "/")
        for f in sorted(filenames):
            full = os.path.join(dirpath, f)
            try:
                size = os.lstat(full).st_size
            except OSError:
                size = -1
            out.append(f"{prefix}{f}  ({size} B)")
        if len(out) > 2000:
            out.append("... (truncated)")
            break
    return {"entries": out}


def _read(a):
    path, offset, limit = a["path"], max(1, int(a.get("offset", 1))), int(a.get("limit", 400))
    with open(path, "rb") as f:
        raw = f.read()
    if b"\0" in raw[:8192]:
        return {"binary": True, "size": len(raw), "text": f"<binary file, {len(raw)} bytes>"}
    lines = raw.decode("utf-8", "replace").splitlines()
    chunk = lines[offset - 1 : offset - 1 + limit]
    width = len(str(offset + len(chunk)))
    text = "\n".join(f"{i:>{width}}\t{line[:2000]}" for i, line in enumerate(chunk, start=offset))
    more = offset - 1 + limit < len(lines)
    return {"text": text, "total_lines": len(lines), "more": more}


def _write(a):
    path = a["path"]
    os.makedirs(os.path.dirname(path) or "/", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(a["content"])
    return {"bytes": len(a["content"].encode("utf-8"))}


def _closest(text, old):
    """Where old_string nearly is: the file's lines most like its first non-blank line, numbered, so the policy can copy
    the exact text (whitespace, tabs) and retry; a hint only, the edit is never applied for it."""
    import difflib

    want = next((ln.strip() for ln in old.splitlines() if ln.strip()), "")
    if not want:
        return ""
    lines = text.splitlines()
    stripped = [ln.strip() for ln in lines]
    near = difflib.get_close_matches(want, stripped, n=3, cutoff=0.6)
    hits = []
    for cand in near:
        for i, s in enumerate(stripped):
            if s == cand and (i + 1, lines[i]) not in hits:
                hits.append((i + 1, lines[i]))
                break
    if not hits:
        return "; no line of the file is close to its first line"
    # one line: the World reports the error's last line only
    return "; the closest lines of the file are " + "; ".join(f"line {n}: {ln[:200]!r}" for n, ln in sorted(hits))


def _edit(a):
    path, old, new = a["path"], a["old"], a["new"]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    n = text.count(old) if old else 0
    if n == 0:
        raise ValueError("old_string not found" + _closest(text, old))
    if n > 1 and not a.get("all"):
        raise ValueError(f"old_string occurs {n} times; make it unique or set replace_all")
    text = text.replace(old, new) if a.get("all") else text.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return {"replacements": n if a.get("all") else 1}


OPS = {"list": _list, "read": _read, "write": _write, "edit": _edit}


def main(a):
    return OPS[a["op"]](a)
