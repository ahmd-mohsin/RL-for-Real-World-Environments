#!/bin/bash
# smoke1 stage gpu_container: GPU in an Apptainer container (--nv), nvcc + a kernel (gpu_container.py).
set -uo pipefail
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
source "$HERE/env.sh"
exec "$PY" "$HERE/gpu_container.py" "${1:?run dir}"
