#!/bin/bash
# DeltaAI capability and lockdown probe (doc 14-deltaai §3): the Apptainer docker shim's containers checked from inside,
# the site vLLM container's version and flags, and the student's weights on Hugging Face (sizes and architecture only).
# Submit:  unset SBATCH_ACCOUNT SLURM_ACCOUNT; sbatch --account=bjhg-dtai-gh probe.sbatch
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64g
#SBATCH --time=00:30:00
#SBATCH --job-name=crucible-infer-probe
#SBATCH --output=logs/probe-%j.out
set -u
W=/work/hdd/bjhg/mahmedmohsin/crucible-infer
J=/tmp/crucible-$SLURM_JOB_ID
mkdir -p "$J/ws" "$J/state" "$J/ovl" "$J/cfg" "$W/sif" "$W/out"
export CRUCIBLE_SIF_DIR=$W/sif CRUCIBLE_SHIM_STATE=$J/state CRUCIBLE_SHIM_SCRATCH=$J/ovl CRUCIBLE_SHIM_CONFIGDIR=$J/cfg
export CRUCIBLE_SHIM_ALLOW=$J APPTAINER_CACHEDIR=$J/cache APPTAINER_TMPDIR=$J/atmp TMPDIR=$J
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
export PATH=$W/bin:$PATH
O=$W/out/probe-$SLURM_JOB_ID
mkdir -p "$O"
echo "node=$(hostname) arch=$(uname -m) kernel=$(uname -r)"
echo "host_munge=$(ls /run/munge 2>/dev/null | tr '\n' ,)"
cp "$W/probe.sh" "$J/ws/probe.sh"

[ -f "$W/sif/python_3.12-slim.sif" ] || apptainer -q pull "$W/sif/python_3.12-slim.sif" docker://python:3.12-slim
echo "== A: foreground run"
docker run --rm --network none -v "$J/ws:$J/ws" python:3.12-slim bash "$J/ws/probe.sh" "$HOME" 0 > "$O/A.txt" 2> "$O/A.err"; echo "A rc=$?"
echo "== B: detached instance + exec"
docker run -d --name crt-probe python:3.12-slim sleep infinity > /dev/null 2> "$O/B.err"; echo "B start rc=$?"
docker exec crt-probe bash -c "mkdir -p $J/ws" 2>>"$O/B.err"
docker exec crt-probe bash -c 'id; dd if=/dev/zero of=/opt/big bs=1M count=1024 2>&1 | tail -1; ls -la /opt/big; rm -f /opt/big' > "$O/B_overlay.txt" 2>&1; echo "B overlay rc=$?"
docker ps --filter name=crt-probe --format '{{.Names}}' > "$O/B_ps.txt"
docker inspect -f '{{.State.Running}}' crt-probe >> "$O/B_ps.txt"
docker rm -f crt-probe; echo "B rm rc=$?"; docker ps >> "$O/B_ps.txt"
echo "== C: GPU run"
docker run --rm --gpus all -v "$J/ws:$J/ws" python:3.12-slim bash "$J/ws/probe.sh" "$HOME" 1 > "$O/C.txt" 2> "$O/C.err"; echo "C rc=$?"
echo "== D: limits"
CRUCIBLE_SHIM_LIMITS=1 docker run --rm --memory 2g --cpus 2 --pids-limit 256 python:3.12-slim true > "$O/D.txt" 2>&1; echo "D limits rc=$?"
echo "== E: refusals"
docker run --rm --network host python:3.12-slim true > "$O/E1.txt" 2>&1; echo "E1 network host rc=$? (expect 125)"
docker run --rm -v "$HOME:/h" python:3.12-slim true > "$O/E2.txt" 2>&1; echo "E2 home bind rc=$? (expect 125)"
docker run --rm -v "$W:/w" python:3.12-slim true > "$O/E3.txt" 2>&1; echo "E3 work bind rc=$? (expect 125)"

echo "== V: vLLM container"
VS=/sw/llmhub/vllm-v0.28.0-slingshot-deltaai.sif
echo "sha256=$(sha256sum $VS | cut -c1-64)"
apptainer exec --nv --cleanenv $VS vllm --version 2>&1 | tail -1
apptainer exec --nv --cleanenv $VS bash -c 'vllm serve --help=all 2>/dev/null | grep -oE -- "--(enable-prompt-tokens-details|disable-log-requests|enable-log-requests|generation-config|max-lora-rank|enable-lora|max-loras|return-tokens-as-token-ids|enable-prefix-caching|max-num-seqs)" | sort -u' > "$O/V_flags.txt"
apptainer exec --nv --cleanenv $VS python3 -c "
from vllm.model_executor.models import ModelRegistry
print([a for a in ModelRegistry.get_supported_archs() if 'Qwen3' in a])" > "$O/V_archs.txt" 2>&1
echo "== H: Hugging Face (sizes and architecture only)"
printf 'HF_TOKEN=%s\n' "$(cat ~/.hf_token)" > "$J/hf.env"; chmod 600 "$J/hf.env"
apptainer exec --cleanenv --env-file "$J/hf.env" $VS python3 -c "
import json
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
for rid in ('Qwen/Qwen3.6-35B-A3B',):
    try:
        info = api.model_info(rid, files_metadata=True)
        files = [(s.rfilename, s.size or 0) for s in info.siblings]
        tot = sum(sz for f, sz in files if f.endswith('.safetensors'))
        cfg = json.load(open(hf_hub_download(rid, 'config.json')))
        print(json.dumps({'repo': rid, 'safetensors_gb': round(tot/1e9, 1), 'files': len(files), 'arch': cfg.get('architectures'),
                          'max_pos': cfg.get('max_position_embeddings') or (cfg.get('text_config') or {}).get('max_position_embeddings'),
                          'model_type': cfg.get('model_type'), 'sha': info.sha}))
    except Exception as e:
        print(json.dumps({'repo': rid, 'error': type(e).__name__ + ': ' + str(e)[:300]}))
" > "$O/H.txt" 2>&1
rm -f "$J/hf.env"
echo "== results"
for f in "$O"/*.txt; do echo "--- $(basename $f)"; head -60 "$f"; done
for f in "$O"/*.err; do [ -s "$f" ] && { echo "--- $(basename $f)"; tail -20 "$f"; }; done
rm -rf "$J"
echo PROBE_JOB_DONE
