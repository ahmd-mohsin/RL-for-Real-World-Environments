#!/bin/bash
# The local dev student (Qwen/Qwen3.5-4B, bf16, text only) behind an OpenAI-compatible API on 127.0.0.1, with native
# tool calls, for harness development and the pipeline smoke. The target student (Qwen/Qwen3.6-35B-A3B) is sampled on
# Tinker only: a 35B MoE does not fit a 24 GB Mac at bf16.
#
#   crucible/scripts/student_serve.sh [--backend auto|mlx|vllm] [--port 8100] [--model ID] [--thinking off|on]
#                                     [--max-model-len N] [--wait-s S]
#   crucible/scripts/student_serve.sh --stop [--port 8100]
#   crucible/scripts/student_serve.sh --status [--port 8100]
#
# Backends (docs/research/swarm-pipeline/08-student-and-builder.md):
# - mlx  (Apple silicon; DEV ONLY: MLX numerics differ from the CUDA engines and Tinker, so its episodes never count for
#        targeting or benchmark numbers): mlx-lm's server from its own venv ($CRU_MLX_VENV, default ~/.venvs/mlx; never
#        the project .venv). Tool calls: the model's own chat template renders `tools`; mlx-lm's parser (inferred from
#        the template) returns OpenAI `tool_calls`. Qwen3.5's vision tower is dropped on load (text only).
# - vllm (a CUDA box): `vllm serve --dtype bfloat16 --enable-auto-tool-choice --tool-call-parser <from the template>`,
#        images and video off (`--limit-mm-per-prompt`), flashinfer's sampler off (VLLM_USE_FLASHINFER_SAMPLER=0,
#        scripts/marlowe/serve_and_rollout.sh); `--reasoning-parser qwen3` only with --thinking on. $CRU_VLLM (default
#        `vllm`), $CRU_VLLM_EXTRA appended.
# Thinking (--thinking, default off) is the server default; crucible.toml's student provider also sends it with every
# request, so the recorded run setting is what the model saw.
# The descriptor $CRU_STUDENT_DIR/student-<port>.json (default ~/.cache/crucible/student) holds backend, model, dtype,
# engine version, base_url, pid, dev_only, tool parser and thinking. `factory.student.probe` reads it and the server's
# /v1/models; `scripts/lh_run.py` stamps the backend into the policy id (`...@mlx`) and the run's student.json.
set -uo pipefail
BACKEND=auto PORT=${CRU_STUDENT_PORT:-8100} MODEL=${CRU_STUDENT_MODEL:-Qwen/Qwen3.5-4B} THINKING=off MAXLEN=32768 WAIT_S=900 ACTION=start
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --thinking) THINKING=$2; shift 2 ;;
    --max-model-len) MAXLEN=$2; shift 2 ;;
    --wait-s) WAIT_S=$2; shift 2 ;;
    --stop) ACTION=stop; shift ;;
    --status) ACTION=status; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
DIR=${CRU_STUDENT_DIR:-$HOME/.cache/crucible/student}
mkdir -p "$DIR"
DESC=$DIR/student-$PORT.json
LOG=$DIR/student-$PORT.log
BASE=http://127.0.0.1:$PORT/v1

pid_of() { [ -f "$DESC" ] && sed -n 's/.*"pid": *\([0-9][0-9]*\).*/\1/p' "$DESC" | head -1; }
up() { curl -fsS -m 5 "$BASE/models" >/dev/null 2>&1; }

if [ "$ACTION" = stop ]; then
  P=$(pid_of)
  if [ -n "$P" ] && kill -0 "$P" 2>/dev/null; then kill -TERM "$P"; echo "stopped student pid $P (port $PORT)"; else echo "no student on port $PORT"; fi
  rm -f "$DESC"
  exit 0
fi
if [ "$ACTION" = status ]; then
  if up; then cat "$DESC" 2>/dev/null; curl -fsS -m 5 "$BASE/models"; echo; exit 0; fi
  echo "no student answering on $BASE"; exit 1
fi
if up; then echo "a server already answers on $BASE"; cat "$DESC" 2>/dev/null; exit 0; fi

case "$THINKING" in on) TJSON=true ;; off) TJSON=false ;; *) echo "--thinking on|off" >&2; exit 2 ;; esac
if [ "$BACKEND" = auto ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then BACKEND=vllm
  elif [ "$(uname -s)-$(uname -m)" = Darwin-arm64 ]; then BACKEND=mlx
  else echo "no backend: neither a CUDA GPU nor Apple silicon" >&2; exit 2; fi
fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
# the tool-call format the model's own chat template writes (from the HF cache's tokenizer files)
SNAP=$(ls -d "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${MODEL//\//--}/snapshots/"*/ 2>/dev/null | head -1)
if grep -qs '<function=' "${SNAP}chat_template.jinja" "${SNAP}tokenizer_config.json"; then PARSER=qwen3_coder; else PARSER=hermes; fi
if [ "$BACKEND" = mlx ] && [ "$PARSER" = hermes ]; then PARSER=json_tools; fi

case "$BACKEND" in
  mlx)
    PY=${CRU_MLX_VENV:-$HOME/.venvs/mlx}/bin/python
    [ -x "$PY" ] || { echo "no mlx venv at $PY: uv venv ~/.venvs/mlx --python 3.12 && uv pip install --python ~/.venvs/mlx/bin/python mlx-lm" >&2; exit 2; }
    VERSION=$("$PY" -c 'import mlx_lm; print(mlx_lm.__version__)')
    # bf16 as published (no quantization). Memory on a 24 GB Mac (Docker holds 8 GB): the first smoke died in a Metal
    # out-of-memory during a long prefill with a 4 GB prompt cache, so the cache is 2 entries / 1.5 GB and prefill runs in
    # 512-token steps (CRU_MLX_CACHE_BYTES, CRU_MLX_PREFILL_STEP override); one request at a time.
    nohup "$PY" -m mlx_lm.server --model "$MODEL" --host 127.0.0.1 --port "$PORT" --max-tokens 8192 --temp 0.7 \
      --prompt-cache-size 2 --prompt-cache-bytes "${CRU_MLX_CACHE_BYTES:-1610612736}" --prefill-step-size "${CRU_MLX_PREFILL_STEP:-512}" \
      --decode-concurrency 1 --prompt-concurrency 1 --chat-template-args "{\"enable_thinking\": $TJSON}" --log-level INFO > "$LOG" 2>&1 &
    PID=$!
    DEV=true ;;
  vllm)
    VLLM=${CRU_VLLM:-vllm}
    export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0} VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
    VERSION=$(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)
    REASON=(); [ "$THINKING" = on ] && REASON=(--reasoning-parser qwen3)
    # shellcheck disable=SC2086
    nohup "$VLLM" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --max-model-len "$MAXLEN" \
      --gpu-memory-utilization "${CRU_VLLM_GPU_UTIL:-0.85}" --enable-auto-tool-choice --tool-call-parser "$PARSER" ${REASON[@]+"${REASON[@]}"} \
      --limit-mm-per-prompt '{"image": 0, "video": 0}' --served-model-name "$MODEL" ${CRU_VLLM_EXTRA:-} > "$LOG" 2>&1 &
    PID=$!
    DEV=false ;;
  *) echo "backend must be auto, mlx or vllm" >&2; exit 2 ;;
esac
cat > "$DESC" <<EOF
{"backend": "$BACKEND", "model": "$MODEL", "dtype": "bfloat16", "engine_version": "$VERSION", "base_url": "$BASE", "port": $PORT,
 "pid": $PID, "dev_only": $DEV, "tool_parser": "$PARSER", "thinking": $TJSON, "started_at": $(date +%s), "log": "$LOG"}
EOF
t0=$(date +%s)
until up; do
  if ! kill -0 "$PID" 2>/dev/null; then echo "the $BACKEND server exited:"; tail -40 "$LOG"; rm -f "$DESC"; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt "$WAIT_S" ]; then echo "not healthy after $WAIT_S s"; tail -40 "$LOG"; exit 1; fi
  sleep 3
done
echo "student $MODEL on $BASE ($BACKEND $VERSION, pid $PID, parser $PARSER, thinking $THINKING, healthy after $(( $(date +%s) - t0 )) s; dev_only=$DEV)"
