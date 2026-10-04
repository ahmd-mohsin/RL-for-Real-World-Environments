# Agent + verifier image for task01_prims_omp. Build from the task directory:
#   docker build -f environment/Dockerfile -t rlsandbox/task01-prims-omp .
# Run with a GPU: docker run --gpus all ...
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates ripgrep python3 less \
      clang-18 clang-tools-18 lld-18 llvm-18 libomp-18-dev \
 && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/lib/llvm-18/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/lib/llvm-18/lib \
    CUDA_HOME=/usr/local/cuda \
    OMP_TARGET_OFFLOAD=MANDATORY

# Pristine original for the verifier's CUDA baseline.
COPY repo/ /opt/tsr-original/
# Agent workspace, committed and tagged so the collector can diff against `base`.
COPY repo/ /app/
RUN cd /app \
 && git init -q \
 && git add -A \
 && git -c user.name=tessera -c user.email=tessera@localhost commit -qm "tsr-prims 0.7.0 (CUDA)" \
 && git tag base
WORKDIR /app
