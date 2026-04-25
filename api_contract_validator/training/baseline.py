"""
Baseline runner — measure the untrained model's score on every task.

Writes two artefacts that the finale judges look at:

    baseline_scores.json   — top-level scores per task (committed to repo)
    api_contract_validator/results/baseline_table.md   — markdown table for README

Usage:

    # Make sure the env server is up first
    docker run -d -p 7860:7860 --name baseline-env api-contract-validator

    # Then run from the api_contract_validator/ directory
    export HF_TOKEN="hf_xxxxx"
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    python training/baseline.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make api_contract_validator importable when this script is run directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env from api_contract_validator/ before reading os.getenv values
try:
    from dotenv import load_dotenv
    _ENV_FILE = ROOT / ".env"
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)
except ImportError:
    pass

from openai import OpenAI  # noqa: E402

from client import ValidatorEnv  # noqa: E402
from inference import (  # noqa: E402
    BENCHMARK,
    TASKS,
    MODEL_NAME,
    HF_TOKEN,
    API_BASE_URL,
    LOCAL_IMAGE_NAME,
    run_single_task,
)


OUT_PATH = Path(os.getenv("BASELINE_OUT", ROOT.parent / "baseline_scores.json"))
TABLE_PATH = ROOT / "results" / "baseline_table.md"


async def main() -> None:
    if not HF_TOKEN:
        sys.exit("HF_TOKEN not set. Export it before running.")

    openai_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    if LOCAL_IMAGE_NAME:
        env = await ValidatorEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")
        env = ValidatorEnv(base_url=env_url)

    results = []
    try:
        for task in TASKS:
            res = await run_single_task(openai_client, env, task)
            results.append(res)
    finally:
        try:
            await env.close()
        except Exception:
            pass

    out = {
        "model": MODEL_NAME,
        "benchmark": BENCHMARK,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "scores": {r["task"]: r["score"] for r in results},
        "details": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"[INFO] wrote {OUT_PATH}", flush=True)

    # Markdown table for README embedding
    lines = [
        "| Task | Score | Steps | Success |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r['task']}` | {r['score']:.2f} | {r['steps']} | "
            f"{'✅' if r['success'] else '⛔'} |"
        )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n".join(lines) + "\n")
    print(f"[INFO] wrote {TABLE_PATH}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
