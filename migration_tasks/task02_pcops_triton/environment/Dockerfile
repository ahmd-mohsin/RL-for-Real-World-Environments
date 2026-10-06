# Sandbox image for task02_pcops_triton. Build from the task directory:
#   docker build -f environment/Dockerfile.base -t task02-base:v1 environment
#   docker build -f environment/Dockerfile -t <registry>/task02-pcops-triton:v1 .
# Run with a GPU: docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ...  (or --gpus all)
# The image holds only the original repositories; the hidden tests are mounted at /tests by the verifier.
ARG BASE=task02-base:v1
FROM ${BASE}

# pristine copies for the verifier
COPY repos/ /opt/originals/
# agent workspace: one git repo per project, tagged `base`
COPY repos/ /workspace/
RUN set -e; for r in pcops pcnet pcdata; do \
      cd /workspace/$r && git init -q && git add -A \
      && git -c user.name=pcops -c user.email=pcops@localhost commit -qm "$r (original)" && git tag base; \
    done \
 && git config --global user.name agent && git config --global user.email agent@localhost \
 && for r in pcops pcnet pcdata; do git config --global --add safe.directory /workspace/$r; done \
 && cd /workspace/pcops && MAX_JOBS=16 pip install --no-build-isolation -q -e . \
 && pip install --no-build-isolation -q -e /workspace/pcdata -e /workspace/pcnet \
 && rm -rf /workspace/pcops/build
COPY environment/sandbox-start.sh /usr/local/bin/sandbox-start.sh
RUN chmod +x /usr/local/bin/sandbox-start.sh
WORKDIR /workspace
CMD ["/usr/local/bin/sandbox-start.sh"]
