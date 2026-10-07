# Sandbox image for task04_pcl_gpu_cuda12. Build from the task directory:
#   docker build -f environment/Dockerfile.base -t task04-base:v1 environment
#   docker build -f environment/Dockerfile -t <registry>/task04-pcl-gpu-cuda12:v1 .
# Run with a GPU: docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all ...  (or --gpus all)
# Only CUDA 12.8 is installed. The hidden tests and the legacy goldens are mounted at /tests by the verifier.
ARG BASE=task04-base:v1
FROM ${BASE}

# PCL at the pinned upstream commit (fetched at image build time; the agent has no network)
ARG PCL_COMMIT=c47198828379841dbbeae6991e0757011a552cec
RUN git clone -q https://github.com/PointCloudLibrary/pcl.git /opt/originals/pcl \
 && cd /opt/originals/pcl && git checkout -q ${PCL_COMMIT} && rm -rf .git
# pristine copies of the consumers for the verifier
COPY repos/ /opt/originals/
# agent workspace: one git repo per project, tagged `base`
RUN mkdir -p /workspace && cp -r /opt/originals/pcl /opt/originals/kinfu_replay /opt/originals/people_eval /workspace/
RUN set -e; for r in pcl kinfu_replay people_eval; do \
      cd /workspace/$r && git init -q && git add -A \
      && git -c user.name=pcl -c user.email=pcl@localhost commit -qm "$r (original)" && git tag base; \
    done \
 && git config --global user.name agent && git config --global user.email agent@localhost \
 && for r in pcl kinfu_replay people_eval; do git config --global --add safe.directory /workspace/$r; done \
 && cd /workspace/people_eval/data/trees && for f in *.bz2; do bzcat $f > ${f%.bz2}; done \
 && chmod -R a+rwX /workspace
COPY environment/sandbox-start.sh /usr/local/bin/sandbox-start.sh
RUN chmod +x /usr/local/bin/sandbox-start.sh
WORKDIR /workspace
CMD ["/usr/local/bin/sandbox-start.sh"]
