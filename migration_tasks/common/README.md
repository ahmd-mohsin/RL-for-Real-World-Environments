# Common tooling for the migration tasks

- `Dockerfile.base`: CPU-only toolchain image (clang 18 + host-offload plugin).
  Each task's `environment/Dockerfile` builds the GPU sandbox on `nvidia/cuda:12.8.1-devel`.
- `mswea/`: [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (v2.4.6) harness.
  - `migration.yaml`: agent config. Bash tool only, yolo mode, 250 steps, $60 cost cap, tool calling via OpenRouter, high reasoning effort.
  - `ext.py`:
    - `SSHEnvironment` runs each action inside the GPU sandbox container over SSH. The agent loop and API key stay local.
    - `LongOpenRouterModel` replaces upstream's 60 s request timeout.
- `run_mini.sh`: one single-attempt run. Collects `model.patch` exactly like the verifier.

## Running a task against a model

1. Build and push the task image:

   ```
   docker buildx build --platform linux/amd64 -f environment/Dockerfile -t <registry>/<task>:v1 --push .
   ```

2. Start a GPU sandbox from the image. On Runpod: a pod with an RTX 4090, container image `<registry>/<task>:v1` and TCP port 22 exposed; `PUBLIC_KEY` comes from the account settings. Locally: `docker run --gpus all -d -p 2222:22 -e PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)" <image>`.

3. Run the agent. `OPENROUTER_API_KEY` is read from the project `.env`:

   ```
   pip install mini-swe-agent==2.4.6
   migration_tasks/common/run_mini.sh migration_tasks/task01_prims_omp anthropic/claude-fable-5.1 \
       "ssh -p <port> -i ~/.ssh/id_ed25519 root@<host>" runs/task01/fable-5.1
   ```

4. Grade the collected patch. Copy `tests/` into the sandbox, then run
   `MODEL_PATCH=/path/model.patch bash /tests/test.sh` inside it (the image has `/opt/tsr-original`).

The sandbox has internet access (Runpod cannot cut it off), so `SSHEnvironment` refuses network commands, and the trajectory should be audited after a run.
