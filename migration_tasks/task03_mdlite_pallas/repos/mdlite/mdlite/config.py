"""Run configuration: ``key = value`` text (see docs/CONFIG.md for every key).

The text is parsed and validated by the engine; this module only converts
between files, dicts and text.
"""
from __future__ import annotations

from pathlib import Path


def config_text(config) -> str:
    """Accepts a dict, a ``key = value`` string, a path to a config file, or None."""
    if config is None:
        return ""
    if isinstance(config, dict):
        lines = []
        for k, v in config.items():
            if isinstance(v, bool):
                v = "yes" if v else "no"
            lines.append(f"{k} = {v}")
        return "\n".join(lines) + "\n"
    if isinstance(config, Path):
        return config.read_text()
    if isinstance(config, str):
        if "=" not in config and config.strip() and Path(config).exists():
            return Path(config).read_text()
        return config
    raise TypeError(f"unsupported config type {type(config).__name__}")


def read_config(path) -> dict:
    """Parse a config file into a dict of strings (comments and blank lines dropped)."""
    out = {}
    for line in Path(path).read_text().splitlines():
        for c in "#;":
            line = line.split(c, 1)[0]
        line = line.strip()
        if not line:
            continue
        k, _, v = line.partition("=")
        out[k.strip().lower().replace("-", "_")] = v.strip()
    return out
