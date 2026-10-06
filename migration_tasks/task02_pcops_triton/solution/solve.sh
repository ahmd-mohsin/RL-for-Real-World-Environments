#!/usr/bin/env bash
# Reference solution: the Triton port of pcops.
set -euo pipefail
cd /workspace/pcops
git apply --whitespace=nowarn "$(dirname "$0")/solution.patch"
pip install --no-build-isolation -q -e .
