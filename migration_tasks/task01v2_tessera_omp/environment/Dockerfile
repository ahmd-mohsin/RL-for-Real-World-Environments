# Sandbox image for task01v2_tessera_omp (agent workspace + verifier baseline).
#
# Build from the task directory (an x86_64 image, also from Apple silicon):
#   docker buildx build --platform linux/amd64 -f environment/Dockerfile -t <registry>/task01v2-tessera-omp:v1 .
# Run locally:   docker run --gpus all -it <image> bash
# Run on Runpod: use the image as the pod's container image and set PUBLIC_KEY;
#                the entrypoint starts sshd so the harness can drive /app over SSH.
#
# The image contains only the original CUDA repository. Hidden tests, grader and
# reference solution are never copied in; the verifier mounts them at /tests.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates python3 python3-venv \
      clang-18 clang-tools-18 lld-18 llvm-18 libomp-18-dev \
      gdb ripgrep less vim-tiny nano file patch diffutils procps psmisc time bc jq tree \
      openssh-server \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/vim.tiny /usr/local/bin/vim \
 && mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh \
 && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/; s/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

ENV PATH=/usr/lib/llvm-18/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/lib/llvm-18/lib \
    CUDA_HOME=/usr/local/cuda \
    OMP_TARGET_OFFLOAD=MANDATORY \
    PAGER=cat GIT_PAGER=cat

# Pristine original for the verifier's CUDA baseline.
COPY repo/ /opt/tsr-original/
# Agent workspace, committed and tagged so the collector can diff against `base`.
COPY repo/ /app/
RUN cd /app \
 && git init -q \
 && git add -A \
 && git -c user.name=tessera -c user.email=tessera@localhost commit -qm "tessera 0.7.0 (CUDA)" \
 && git tag base \
 && git config --global --add safe.directory /app \
 && git config --global user.name agent && git config --global user.email agent@localhost

COPY environment/sandbox-start.sh /usr/local/bin/sandbox-start.sh
RUN chmod +x /usr/local/bin/sandbox-start.sh
WORKDIR /app
EXPOSE 22
CMD ["/usr/local/bin/sandbox-start.sh"]
