#!/bin/bash
# Reference solution: apply the JAX + Pallas port to /workspace/mdlite.
set -e
cd /workspace/mdlite
git apply --whitespace=nowarn "$(dirname "$0")/solution.patch"
pip install --no-build-isolation -q -e .
