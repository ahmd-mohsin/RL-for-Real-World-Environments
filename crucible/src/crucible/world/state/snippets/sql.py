"""SQLite state canonicalization. STANDALONE: stdlib only; shipped verbatim into containers.

Ops (``main({"op": ..., ...})``):
  dump     {path, include, exclude, rows}  -> {tables: {name: {schema, count, rows_hash, rows?}}, fingerprint}
  backup   {path, out, include, exclude}   -> dump of the copy (consistent online backup)
  restore  {path, src, include, exclude}   -> dump after copying src INTO the live db (same inode,
                                              so open connections in App servers see the new state)
  query    {path, sql, params, max_rows}   -> {columns, rows, truncated}; read-only connection

Canonical form: every schema object (table, index, view, trigger; excluding sqlite_ internals)
contributes its whitespace-normalized SQL; every table contributes the sorted list of its rows, each
row encoded as JSON with typed cells (ints as ints, floats via float.hex, blobs as hex, text as
text). Row order and storage layout never affect the fingerprint; values and schema always do.
"""

import hashlib
import json
import os
import re
import sqlite3


def _cell(v):
    if v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return {"f": v.hex()}
    if isinstance(v, (bytes, memoryview)):
        return {"b": bytes(v).hex()}
    return {"x": repr(v)}


def _norm_sql(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def _match(name, pats):
    import fnmatch

    return any(fnmatch.fnmatchcase(name, p) for p in pats)


def _dump(con, include, exclude, rows):
    objs = con.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_autoindex_%' ORDER BY type, name"
    ).fetchall()
    tables = {}
    schema_lines = []
    for typ, name, tbl, sql in objs:
        if name.startswith("sqlite_") and name != "sqlite_sequence":
            continue
        if include and not _match(tbl, include):
            continue
        if exclude and _match(tbl, exclude):
            continue
        schema_lines.append(f"{typ}\0{name}\0{_norm_sql(sql)}")
        if typ != "table":
            continue
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")').fetchall()]
        enc = sorted(json.dumps([_cell(v) for v in r], separators=(",", ":"), ensure_ascii=False) for r in con.execute(f'SELECT * FROM "{name}"'))
        h = hashlib.sha256()
        for line in enc:
            h.update(line.encode("utf-8"))
            h.update(b"\n")
        t = {"schema": _norm_sql(sql), "columns": cols, "count": len(enc), "rows_hash": h.hexdigest()}
        if rows:
            t["rows"] = enc
        tables[name] = t
    outer = hashlib.sha256()
    for line in sorted(schema_lines):
        outer.update(line.encode("utf-8"))
        outer.update(b"\n")
    for name in sorted(tables):
        outer.update(f"{name}\0{tables[name]['rows_hash']}\n".encode("utf-8"))
    return {"tables": tables, "fingerprint": outer.hexdigest()}


def _ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def dump(a):
    if not os.path.exists(a["path"]):
        return {"tables": {}, "fingerprint": hashlib.sha256(b"").hexdigest(), "missing": True}
    con = _ro(a["path"])
    try:
        return _dump(con, a.get("include") or [], a.get("exclude") or [], bool(a.get("rows")))
    finally:
        con.close()


def backup(a):
    src = sqlite3.connect(a["path"])
    dst = sqlite3.connect(a["out"])
    try:
        src.backup(dst)
        return _dump(dst, a.get("include") or [], a.get("exclude") or [], False)
    finally:
        dst.close()
        src.close()


def restore(a):
    src = sqlite3.connect(f"file:{a['src']}?mode=ro", uri=True)
    os.makedirs(os.path.dirname(a["path"]) or ".", exist_ok=True)
    dst = sqlite3.connect(a["path"])
    try:
        src.backup(dst)
        dst.commit()
        return _dump(dst, a.get("include") or [], a.get("exclude") or [], False)
    finally:
        dst.close()
        src.close()


def query(a):
    con = _ro(a["path"])
    try:
        con.execute("PRAGMA query_only=ON")
        cur = con.execute(a["sql"], a.get("params") or [])
        cols = [d[0] for d in (cur.description or [])]
        max_rows = int(a.get("max_rows", 1000))
        rows = cur.fetchmany(max_rows + 1)
        return {"columns": cols, "rows": [[_cell(v) for v in r] for r in rows[:max_rows]], "truncated": len(rows) > max_rows}
    finally:
        con.close()


OPS = {"dump": dump, "backup": backup, "restore": restore, "query": query}


def main(a):
    return OPS[a["op"]](a)
