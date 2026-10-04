"""Pull open-sourced RL environments from the Hugging Face Hub into RL-Environments/.

Usage:
    HF_TOKEN=... .venv/bin/python scripts/pull_envs.py [--only <substring>]

The token is read from the environment and never written to disk.
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent / "RL-Environments"

# (category folder, local name, HF dataset id)
DATASETS = [
    # Frontier/lab releases: environments shipped alongside open-weight models
    ("01-lab-releases", "xiaomi-mimo-v2.6-rl-oss", "XiaomiMiMo/MiMo-V2.6-RL-oss"),
    ("01-lab-releases", "nvidia-nemotron-rl-conversational-tool-use", "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1"),
    ("01-lab-releases", "nvidia-nemotron-rl-function-calling", "nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1"),
    ("01-lab-releases", "nvidia-nemo-gym-conversational-tool-use-assets", "nvidia/NeMo-Gym-Conversational-Tool-Use-Assets"),
    # Enterprise / horizontal workflows: closest to our thesis
    ("02-enterprise-workflows", "servicenow-enterpriseops-gym", "ServiceNow-AI/EnterpriseOps-Gym"),
    ("02-enterprise-workflows", "nvidia-nemo-gym-enterpriseops-assets", "nvidia/NeMo-Gym-EnterpriseOps-Assets"),
    ("02-enterprise-workflows", "nvidia-nemotron-rl-workplace-assistant", "nvidia/Nemotron-RL-agent-workplace_assistant"),
    ("02-enterprise-workflows", "upstage-automationbench", "UpstageShareSpace/automationBench"),
    ("02-enterprise-workflows", "snorkel-finqa", "snorkelai/finqa-data"),
    ("02-enterprise-workflows", "npci-nemo-gym-indian-banking", "NPCI/nemo-gym-indian-banking"),
    ("02-enterprise-workflows", "jetbrains-excelforum-nemo-gym", "JetBrains-Research/excelforum-nemo-gym"),
    # Harbor-format task directories (instruction, env image, tests, grader)
    ("03-harbor-format", "fineenvs-smoldataenvs", "FineEnvs/SmolDataEnvs"),
    ("03-harbor-format", "fineenvs-smoldataenvs-harbor-train", "FineEnvs/SmolDataEnvs-harbor-train"),
    ("03-harbor-format", "fineenvs-smoldataenvs-harbor-eval", "FineEnvs/SmolDataEnvs-harbor-eval"),
    ("03-harbor-format", "terminal-bench-2.1", "harborframework/terminal-bench-2.1"),
    # Coding: execution-verified environments
    ("04-coding-swe", "swe-smith", "SWE-bench/SWE-smith"),
    # Automatic environment synthesis: closest to our "black box"
    ("05-environment-synthesis", "envscaler-rl-scenario", "XXHStudyHard/EnvScaler-RL-Scenario"),
    ("05-environment-synthesis", "envfactory-rl", "LARK-Lab/EnvFactory-RL"),
    ("05-environment-synthesis", "spade-environments-tooluse", "spade-rl/SPADE-Environments-ToolUse"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="pull only datasets whose id contains this substring")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    for category, name, repo_id in DATASETS:
        if args.only and args.only.lower() not in repo_id.lower():
            continue
        dest = ROOT / category / name
        print(f"==> {repo_id} -> {dest.relative_to(ROOT.parent)}", flush=True)
        try:
            snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dest, token=token, max_workers=16)
        except Exception as e:
            print(f"!! failed {repo_id}: {e}", flush=True)


if __name__ == "__main__":
    main()
