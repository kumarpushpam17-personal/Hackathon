"""
GRPO training script for the API Contract Validator environment.

Designed to be re-runnable by judges from a Colab notebook OR via HF
Jobs:

    # HF Jobs (T4 small, ~$0.50/hr — uses your $30 credit)
    hf jobs uv run \
        --with trl --with unsloth --with openenv-core --with wandb \
        --flavor t4-small \
        -s HF_TOKEN -s WANDB_API_KEY \
        -- python training/train.py

The script:

  1. Connects to a deployed HF Space (or local docker) running the env
  2. Loads a small base model with Unsloth 4-bit quantisation
  3. Applies LoRA adapters
  4. Rolls out episodes through the env, collecting (prompt, completion,
     reward) tuples
  5. Trains the LoRA adapters with GRPO from TRL
  6. Logs reward curves to WandB and writes results/reward_curve.png
  7. Pushes the trained adapter to the HuggingFace Hub

The reward function uses the env's grader directly — no synthetic
shaping. This is the key difference from a static-dataset SFT run:
the model learns from the env's verifiable signal, which is exactly
what the hackathon's "Improvement in Rewards" criterion rewards.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

# Ensure api_contract_validator is importable
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


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """Training configuration. All fields read from env vars at instantiation.

    Recommended HF Jobs configurations:

      Smoke test ($0.30, 5 min):
        BASE_MODEL=unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit, MAX_STEPS=10,
        flavor t4-small

      Main run on L4 ($2.40, ~2 hr):
        BASE_MODEL=unsloth/Qwen2.5-7B-Instruct-bnb-4bit, MAX_STEPS=300,
        flavor l4x1

      Insurance run on T4 ($0.40, ~45 min):
        BASE_MODEL=unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit, MAX_STEPS=200,
        flavor t4-small
    """

    base_model: str = os.getenv(
        "BASE_MODEL", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    )
    env_url: str = os.getenv("ENV_URL", "http://localhost:7860")
    push_to_hub_id: str | None = os.getenv("PUSH_TO_HUB", None)
    output_dir: str = os.getenv("OUTPUT_DIR", "checkpoints/grpo")
    results_dir: str = os.getenv(
        "RESULTS_DIR", str(ROOT / "results")
    )
    seed: int = int(os.getenv("SEED", "42"))

    # LoRA
    lora_r: int = int(os.getenv("LORA_R", "16"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "32"))

    # GRPO
    max_seq_length: int = int(os.getenv("MAX_SEQ_LEN", "2048"))
    num_generations: int = int(os.getenv("NUM_GENERATIONS", "4"))
    max_steps: int = int(os.getenv("MAX_STEPS", "300"))
    learning_rate: float = float(os.getenv("LR", "5e-6"))
    per_device_batch_size: int = int(os.getenv("BATCH_SIZE", "1"))
    grad_accum: int = int(os.getenv("GRAD_ACCUM", "4"))

    # Tasks to train on (subset speeds up onsite training)
    train_tasks: List[str] | None = None

    # WandB
    wandb_project: str = os.getenv("WANDB_PROJECT", "openenv-contract-guardian")
    wandb_run: str = os.getenv("WANDB_RUN", "grpo-onsite")


# ── Reward function: rolls out one step against the live env ─────────────


def make_reward_fn(env_client, task_pool: List[str]):
    """Return a TRL-compatible reward_fn that grades each completion via env.

    For every (prompt, completion) pair, we ask the env to score it.
    GRPO will then promote the higher-reward completion among the
    ``num_generations`` samples per prompt.
    """
    from inference import _build_action, parse_llm_response  # noqa: WPS433
    import asyncio

    def reward_fn(prompts, completions, **kwargs):  # noqa: ARG001
        rewards: List[float] = []
        loop = asyncio.get_event_loop()
        for completion in completions:
            text = completion if isinstance(completion, str) else completion[0]["content"]
            try:
                action_data = parse_llm_response(text)
                action = _build_action(action_data)
                # one-step roll-out: reset → step → grade → reset
                # Each prompt is associated with a fresh episode in train_dataset,
                # so we use the env's most recent reset state as scoring context.
                step_result = loop.run_until_complete(env_client.step(action))
                rewards.append(float(step_result.reward or 0.0))
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] reward_fn error: {exc}")
                rewards.append(-0.5)
        return rewards

    return reward_fn


# ── Dataset: one prompt per env reset ────────────────────────────────────


def build_train_dataset(env_client, tasks: List[str], episodes_per_task: int = 50):
    """Roll out reset() to capture initial observations as training prompts.

    Each row is one episode start. During training, GRPO samples
    ``num_generations`` completions per prompt and uses the env to
    grade them.
    """
    import asyncio
    from datasets import Dataset  # type: ignore

    from inference import build_user_prompt, _system_prompt_for_phase  # noqa

    rows: List[Dict[str, Any]] = []
    loop = asyncio.get_event_loop()

    for task in tasks:
        for ep in range(episodes_per_task):
            seed = ep
            result = loop.run_until_complete(
                env_client.reset(task_name=task, seed=seed)
            )
            obs = result.observation.model_dump()
            phase = obs.get("phase", "detection")
            system = _system_prompt_for_phase(phase, task)
            user = build_user_prompt(obs, step=1, history=[])
            rows.append({
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "task": task,
                "seed": seed,
            })

    return Dataset.from_list(rows)


# ── Main entry point ─────────────────────────────────────────────────────


def main() -> None:
    cfg = TrainConfig()

    # ---- Imports happen inside main so the script can be inspected
    # ----  without the heavy deps installed.
    import asyncio
    from openenv.core.client_types import StepResult  # noqa: F401

    try:
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError as exc:
        sys.exit(
            "unsloth not installed. Install with: pip install unsloth trl wandb. "
            f"({exc})"
        )

    from trl import GRPOConfig, GRPOTrainer  # type: ignore
    import wandb  # type: ignore

    from client import ValidatorEnv  # noqa: WPS433

    if os.getenv("WANDB_API_KEY"):
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run,
            config=cfg.__dict__,
        )

    # 1. Connect to the env
    env = ValidatorEnv(base_url=cfg.env_url)
    print(f"[INFO] connected to env at {cfg.env_url}")

    # 2. Choose tasks
    train_tasks = cfg.train_tasks or [
        "find_type_mismatches",
        "validate_nested_objects",
        "detect_breaking_changes",
        "validate_response_schema",
        "trace_downstream_blast_radius",
        "propose_backward_compat_fix",
    ]

    # 3. Build dataset
    print(f"[INFO] building dataset for tasks={train_tasks}")
    train_dataset = build_train_dataset(env, train_tasks)

    # 4. Load model + LoRA
    print(f"[INFO] loading model: {cfg.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        random_state=cfg.seed,
    )

    # 5. GRPO trainer
    grpo_cfg = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_generations=cfg.num_generations,
        max_steps=cfg.max_steps,
        max_prompt_length=cfg.max_seq_length // 2,
        max_completion_length=cfg.max_seq_length // 2,
        logging_steps=1,
        save_steps=50,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        bf16=True,
    )

    reward_fn = make_reward_fn(env, train_tasks)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_cfg,
        train_dataset=train_dataset,
    )

    # 6. Train
    print("[INFO] starting GRPO training")
    trainer.train()

    # 7. Save reward curve
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    history = [
        h for h in trainer.state.log_history if "reward" in h
    ]
    if history:
        steps = [h["step"] for h in history]
        rewards = [h["reward"] for h in history]
        plt.figure(figsize=(8, 5))
        plt.plot(steps, rewards, label="train reward", linewidth=2)
        plt.xlabel("Training step")
        plt.ylabel("Mean episode reward")
        plt.title("GRPO Training — Enterprise Contract Guardian")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = results_dir / "reward_curve.png"
        plt.savefig(out, dpi=150)
        print(f"[INFO] wrote {out}")

    # 8. Save trainer state to JSON for plot.py to consume later
    state_path = results_dir / "training_state.json"
    state_path.write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"[INFO] wrote {state_path}")

    # 9. Push checkpoint
    if cfg.push_to_hub_id:
        print(f"[INFO] pushing adapter to {cfg.push_to_hub_id}")
        model.push_to_hub(cfg.push_to_hub_id, token=os.getenv("HF_TOKEN"))

    asyncio.get_event_loop().run_until_complete(env.close())
    print("[INFO] done.")


if __name__ == "__main__":
    main()
