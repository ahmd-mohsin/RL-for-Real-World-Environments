"""Write the frozen files of every persisted format's current version (M0 audit 2026-10-14 R-18; see tests/unit/_formats.py).

    uv run --frozen python scripts/freeze_schemas.py            # write the files of versions not yet frozen
    uv run --frozen python scripts/freeze_schemas.py --refresh  # rewrite current schemas after a pydantic upgrade only

Existing files are never rewritten without --refresh: a model whose schema changed needs a new version, not a new
snapshot. --refresh exists for one case: a pydantic upgrade that changes how the same models render (check the lock diff).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.unit._formats import CRUCIBLE, FORMATS, render_example, render_schema


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="rewrite current schema snapshots (a pydantic upgrade only)")
    args = ap.parse_args()
    for fmt in FORMATS:
        for path, text in ((fmt.schema_file(), render_schema(fmt.model())), (fmt.example_file(), render_example(fmt))):
            rel = path.relative_to(CRUCIBLE)
            if path.exists() and (not args.refresh or path.name.endswith(".example.json")):
                if path.read_text() != text and path.name.endswith(".schema.json"):
                    print(f"STALE {rel}: the model changed without a version bump (see tests/unit/_formats.py)")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
