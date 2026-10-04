#!/usr/bin/env bash
# Re-creates single_task/upstream/: the open-source projects Tessera was
# assembled from, at the exact commits that were used. Not needed to build or
# use the repository; only to audit where code came from.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p upstream
cd upstream

fetch() { # name url commit [sparse paths...]
  local name=$1 url=$2 commit=$3; shift 3
  if [ ! -d "$name" ]; then
    git clone --quiet --filter=blob:none --no-checkout "$url" "$name"
  fi
  if [ $# -gt 0 ]; then
    git -C "$name" sparse-checkout set --no-cone "$@"
  fi
  git -C "$name" checkout --quiet "$commit"
  echo "$name @ $commit"
}

fetch gsplat       https://github.com/nerfstudio-project/gsplat.git     512d366b67073d77ca099ede742683c165dfc23b
fetch pytorch3d    https://github.com/facebookresearch/pytorch3d.git    88e182f989c80836f4bd744e0d9cb1852762ce01 '/LICENSE' '/pytorch3d/csrc/'
fetch cuda-samples https://github.com/NVIDIA/cuda-samples.git           5443602d89ed99aede2e4b7bf329daddeadb320e '/LICENSE' '/Common/' '/cpp/0_Introduction/matrixMul/' '/cpp/2_Concepts_and_Techniques/'
fetch tiny-cuda-nn https://github.com/NVlabs/tiny-cuda-nn.git           3daa6e5b8b8e0fa17971a1a20acfbb3754494b38
fetch thread-pool  https://github.com/bshoshany/thread-pool.git         bd4533f1f70c2b975cbd5769a60d8eaaea1d2233
