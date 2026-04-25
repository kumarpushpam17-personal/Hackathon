# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.13",
#     "unsloth",
#     "openenv-core[core]>=0.2.2",
#     "wandb",
#     "matplotlib",
#     "datasets",
#     "openai",
#     "huggingface_hub",
#     "python-dotenv",
#     "websockets",
# ]
# ///
"""
Self-bootstrapping launcher for HF Jobs.

`hf jobs uv run` only uploads the single script you point it at. Our actual
training code lives in ``api_contract_validator/training/train.py`` and
imports from sibling modules (``inference``, ``client``, ``models``,
``server.*``). This launcher clones our public GitHub repo inside the job,
puts the package on ``sys.path``, and then calls ``training.train.main()``.

Configuration is fully via environment variables — see
``training/train.py`` and ``training/README.md`` for the full list. The
launcher itself only needs ``GIT_REPO_URL`` (defaults to the GitHub fork
that hosts this code) and forwards everything else through to
``train.main()``.

Usage examples:

    # Smoke test (5 min, ~$0.30)
    hf jobs uv run \
        --flavor t4-small \
        -s HF_TOKEN \
        -e BASE_MODEL=unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \
        -e ENV_URL=https://pushpam14-api-contract-validator.hf.space \
        -e MAX_STEPS=10 \
        -e WANDB_RUN=smoke-test \
        api_contract_validator/training/run_in_hf_jobs.py

    # Main run (Qwen-7B on L4, ~2 hr, ~$2.40)
    hf jobs uv run \
        --flavor l4x1 \
        -s HF_TOKEN -s WANDB_API_KEY \
        -e BASE_MODEL=unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
        -e ENV_URL=https://pushpam14-api-contract-validator.hf.space \
        -e MAX_STEPS=300 \
        -e PUSH_TO_HUB=pushpam14/api-contract-validator-grpo-7b \
        -e WANDB_RUN=grpo-7b-l4-300steps \
        api_contract_validator/training/run_in_hf_jobs.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


GIT_REPO_URL = os.getenv(
    "GIT_REPO_URL",
    "https://github.com/kumarpushpam17-personal/Hackathon.git",
)
GIT_REF = os.getenv("GIT_REF", "main")
REPO_DIR = Path(os.getenv("REPO_DIR", "/tmp/eg-repo"))


def _clone_repo() -> Path:
    """Clone the project so we can import its modules."""
    if (REPO_DIR / "api_contract_validator").exists():
        print(f"[launcher] repo already at {REPO_DIR} — skipping clone")
        return REPO_DIR

    print(f"[launcher] cloning {GIT_REPO_URL} @ {GIT_REF} -> {REPO_DIR}")
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", GIT_REF,
         GIT_REPO_URL, str(REPO_DIR)],
        check=True,
    )
    return REPO_DIR


def main() -> None:
    repo = _clone_repo()
    pkg = repo / "api_contract_validator"
    if not pkg.exists():
        sys.exit(f"[launcher] expected {pkg} to exist after clone — aborting")

    # Make the package importable for both top-level (inference, client, models)
    # and sub-package (training.train, server.*) imports.
    sys.path.insert(0, str(pkg))
    print(f"[launcher] sys.path[0] = {pkg}")

    # Print what we're about to run for the job log
    print("[launcher] env summary:")
    for key in (
        "BASE_MODEL", "ENV_URL", "MAX_STEPS", "NUM_GENERATIONS",
        "LORA_R", "LORA_ALPHA", "PUSH_TO_HUB",
        "WANDB_PROJECT", "WANDB_RUN",
    ):
        print(f"  {key} = {os.getenv(key, '(unset)')}")

    # Import after sys.path tweak; train.py reads env vars on instantiation
    from training.train import main as train_main  # type: ignore  # noqa: WPS433

    train_main()


if __name__ == "__main__":
    main()
