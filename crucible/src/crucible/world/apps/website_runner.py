"""Runs a pack's website: ``python -m crucible.world.apps.website_runner --site site.py --factory create_app --db x.db --port N``."""

from __future__ import annotations

import argparse
import importlib.util
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--factory", default="create_app")
    ap.add_argument("--db", required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("crucible_site", a.site)
    if spec is None or spec.loader is None:
        sys.exit(f"cannot load {a.site}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    app = getattr(mod, a.factory)(a.db)
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
