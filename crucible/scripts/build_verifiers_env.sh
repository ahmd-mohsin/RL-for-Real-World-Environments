#!/usr/bin/env bash
# Build crucible/.envs/verifiers: the interpreter the crucible-vf plugin (vf-eval, prime-rl) runs in (M11 lab audit R-3).
#
# verifiers is fetched at the pinned commit below and synced from its own uv.lock (Python 3.12, the `harbor` extra:
# harbor==0.24.0 and mcp==2.0.0, verifiers pyproject.toml:54,97), then Crucible is installed into it, editable. It is a
# separate interpreter from Crucible's main env on purpose: verifiers pins mcp 2 and agentenv / NeMo-Gym need mcp<2, and
# the main env's lock carries Harbor 0.23 (crucible/uv.lock). An unpinned editable checkout broke the plugin once already:
# verifiers removed HarborEnv after ae819de, and a refresh would have stopped `vf-eval crucible-vf` from loading.
#
#   scripts/build_verifiers_env.sh           build (or rebuild) the env at the pin, then check it
#   scripts/build_verifiers_env.sh --check   only check: the env's verifiers commit is the pin and harbor is 0.24.0
#
# Environment: VERIFIERS_REPO (default the GitHub repo), CRUCIBLE_VERIFIERS_ENV (default crucible/.envs/verifiers).
# Needs git and uv. The check is what CI runs where the env exists (tests/unit/test_m11_lab_audit_r3_vf.py).
set -euo pipefail

VERIFIERS_COMMIT="dca5562ee6a7891f820787f710c57d796185f39e"
HARBOR_VERSION="0.24.0"
VERIFIERS_REPO="${VERIFIERS_REPO:-https://github.com/PrimeIntellect-ai/verifiers.git}"

crucible="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${CRUCIBLE_VERIFIERS_ENV:-$crucible/.envs/verifiers}"
src="$crucible/.envs/src/verifiers"

check() {
  [ -x "$env_dir/bin/python" ] || { echo "no env at $env_dir: run $0 to build it" >&2; return 1; }
  "$env_dir/bin/python" - "$VERIFIERS_COMMIT" "$HARBOR_VERSION" <<'PY'
import importlib.metadata as md
import sys

from crucible.runtime.identity import dist_source

want_commit, want_harbor = sys.argv[1], sys.argv[2]
got = dist_source("verifiers")
harbor = md.version("harbor")
problems = []
if got["commit"] != want_commit:
    problems.append(f"verifiers commit {got['commit']} != pinned {want_commit}")
if got["dirty"] is not False:
    problems.append(f"the verifiers checkout is modified or unreadable (dirty={got['dirty']})")
if harbor != want_harbor:
    problems.append(f"harbor {harbor} != {want_harbor}")
import crucible_vf  # noqa: E402,F401 -- the plugin module loads against this verifiers

print(f"verifiers {got['version']} @ {got['commit']}, harbor {harbor}")
if problems:
    sys.exit("; ".join(problems))
PY
}

if [ "${1:-}" = "--check" ]; then
  check
  exit
fi

command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 2; }
mkdir -p "$(dirname "$src")"
if [ ! -d "$src/.git" ]; then
  git init -q "$src"
  git -C "$src" remote add origin "$VERIFIERS_REPO"
fi
git -C "$src" fetch -q --depth 1 origin "$VERIFIERS_COMMIT"
git -C "$src" -c advice.detachedHead=false checkout -q --force FETCH_HEAD
[ "$(git -C "$src" rev-parse HEAD)" = "$VERIFIERS_COMMIT" ] || { echo "fetched the wrong verifiers commit" >&2; exit 1; }

# verifiers' own lock, its harbor extra, no dev group; verifiers itself is installed editable from $src
UV_PROJECT_ENVIRONMENT="$env_dir" uv sync --project "$src" --frozen --python 3.12 --extra harbor --no-dev
# Crucible into the same interpreter, constrained to what verifiers' lock installed: its few base dependencies that the
# lock does not hold (dill, tomli-w) resolve freely, and no locked package moves
constraints="$(mktemp)"
trap 'rm -f "$constraints"' EXIT
uv export --project "$src" --frozen --extra harbor --no-dev --no-emit-project --no-hashes -o "$constraints" >/dev/null
uv pip install --python "$env_dir/bin/python" -c "$constraints" -e "$crucible"
check
