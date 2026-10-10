"""Grader-owned consumer: runs the library on an input file and writes the outputs (a pristine downstream user)."""
import json
import sys

import mylib

inp, out = sys.argv[1], sys.argv[2]
with open(inp) as f:
    cases = json.load(f)
with open(out + "/result.json", "w") as f:
    json.dump({k: mylib.fast_sum(v) for k, v in cases.items()}, f, sort_keys=True)
