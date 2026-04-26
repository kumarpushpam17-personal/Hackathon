# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.13",
#     "unsloth",
#     "openenv-core[core]>=0.2.2",
#     "matplotlib",
#     "datasets",
#     "openai",
#     "huggingface_hub",
#     "python-dotenv",
#     "websockets",
# ]
# ///
"""
Post-training inference job — runs the trained LoRA adapter against
every task in TASKS, captures per-task scores, and uploads
``trained_scores.json`` to the same HF Hub repo as the adapter.

Run via HF Jobs after the GRPO main run:

    hf jobs uv run \
        --flavor t4-small \
        -s HF_TOKEN \
        -e ADAPTER_REPO=pushpam14/api-contract-validator-grpo-7b \
        -e ENV_URL=https://pushpam14-api-contract-validator.hf.space \
        api_contract_validator/training/run_trained_inference.py

Cost: ~$0.30 on t4-small (~30 min for 9 tasks, ~150 total LLM calls).
"""

from __future__ import annotations

import asyncio
import json
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
ADAPTER = os.environ.get(
    "ADAPTER_REPO", "pushpam14/api-contract-validator-grpo-7b"
)
ENV_URL = os.environ.get(
    "ENV_URL", "https://pushpam14-api-contract-validator.hf.space"
)


def _clone_repo() -> Path:
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
    sys.path.insert(0, str(pkg))

    print(f"[INFO] adapter:  {ADAPTER}")
    print(f"[INFO] env_url:  {ENV_URL}")

    from unsloth import FastLanguageModel  # type: ignore
    import torch  # type: ignore

    print(f"[INFO] loading base + adapter: {ADAPTER}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=torch.float16,
    )
    FastLanguageModel.for_inference(model)
    print("[INFO] model ready for inference")

    from inference import (  # type: ignore  # noqa: WPS433
        BENCHMARK,
        MAX_STEPS_PER_TASK,
        TASKS,
        _build_action,
        build_user_prompt,
        _system_prompt_for_phase,
        parse_llm_response,
        log_start,
        log_step,
        log_end,
    )
    from client import ValidatorEnv  # type: ignore  # noqa: WPS433

    def query_local(observation: dict, step: int, history: list) -> dict:
        """Run the trained model on the current observation."""
        phase = observation.get("phase", "detection")
        task_name = observation.get("task_name", "")
        user = build_user_prompt(observation, step, history)
        system = _system_prompt_for_phase(phase, task_name)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        # Temperature is env-configurable so we can tune sampling diversity.
        # 0.2 was too deterministic — the trained model kept reporting the
        # same violation across steps. 0.7 introduces enough variance for
        # the agent to find new violations after the first few.
        temperature = float(os.environ.get("TEMPERATURE", "0.7"))
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=384,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        return parse_llm_response(text)

    async def run_task(env: ValidatorEnv, task_name: str) -> dict:
        max_steps = MAX_STEPS_PER_TASK.get(task_name, 15)
        rewards: list = []
        history: list = []
        score = 0.01
        success = False
        steps_taken = 0
        log_start(task=task_name, env=BENCHMARK, model=ADAPTER)
        try:
            result = await env.reset(task_name=task_name)
            obs = (
                result.observation.model_dump()
                if hasattr(result.observation, "model_dump")
                else result.observation.__dict__
            )
            for step in range(1, max_steps + 1):
                if result.done:
                    break
                action_data = query_local(obs, step, history)
                action = _build_action(action_data)
                result = await env.step(action)
                obs = (
                    result.observation.model_dump()
                    if hasattr(result.observation, "model_dump")
                    else result.observation.__dict__
                )
                reward = float(result.reward or 0.0)
                rewards.append(reward)
                steps_taken = step
                action_str = (
                    f"{action_data.get('action_type','?')}:"
                    f"{action_data.get('field_path', action_data.get('fix_strategy','?'))}"
                )
                log_step(
                    step=step,
                    action=action_str,
                    reward=reward,
                    done=result.done,
                    error=None,
                )
                history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")
                if result.done:
                    break

            try:
                state = await env.state()
                score = float(getattr(state, "score", 0.01)) or 0.01
            except Exception:  # noqa: BLE001
                if rewards:
                    correct = sum(1 for r in rewards if r >= 1.0)
                    total = obs.get("violations_remaining", 0) + len(
                        obs.get("violations_found", [])
                    )
                    score = correct / total if total > 0 else 0.5
            score = min(max(score, 0.01), 0.99)
            success = score >= 0.3
        finally:
            log_end(
                success=success,
                steps=steps_taken,
                score=score,
                rewards=rewards,
            )

        return {
            "task": task_name,
            "score": round(score, 4),
            "steps": steps_taken,
            "success": success,
            "rewards": [round(r, 4) for r in rewards],
        }

    async def main_async() -> None:
        env = ValidatorEnv(base_url=ENV_URL)
        results = []
        try:
            for task in TASKS:
                results.append(await run_task(env, task))
        finally:
            try:
                await env.close()
            except Exception:  # noqa: BLE001
                pass

        out = {
            "model": ADAPTER,
            "benchmark": BENCHMARK,
            "scores": {r["task"]: r["score"] for r in results},
            "details": results,
        }
        out_path = Path("/tmp/trained_scores.json")
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[INFO] wrote {out_path}")

        # Upload to HF Hub adapter repo
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo="trained_scores.json",
            repo_id=ADAPTER,
            repo_type="model",
            commit_message="Add post-training trained_scores.json",
        )
        print(f"[INFO] uploaded trained_scores.json -> {ADAPTER}/trained_scores.json")

    asyncio.run(main_async())
    print("[INFO] done.")


if __name__ == "__main__":
    main()
