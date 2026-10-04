#!/usr/bin/env bash
# Reference solution: apply the hand-written port (see MIGRATION_LOG.md).
set -euo pipefail
cd /app
git apply --whitespace=nowarn "$(dirname "$0")/solution.patch"
