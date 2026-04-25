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


def _list_value(values: Any, index: int, default: Any) -> Any:
    """Return ``values[index]`` for TRL batch kwargs, with a safe fallback."""
    if isinstance(values, list) and index < len(values):
        return values[index]
    return default


def make_reward_fn(env_url: str, task_pool: List[str]):
    """Return a TRL-compatible reward_fn that grades each completion via env.

    A fresh ``ValidatorEnv`` (WebSocket) is created per ``reward_fn``
    invocation and closed at the end. HF Spaces drops idle WebSockets
    after ~30 s, but GRPO's model-generation and backprop pauses are
    longer than that — sharing one WebSocket across batches caused
    "received 1011 keepalive ping timeout" on every batch after the
    first. A per-call client adds ~50 ms of TCP setup but eliminates
    the keepalive failures entirely.

    Within a single ``reward_fn`` call, all completions are graded
    through one client (calls are rapid so keepalive is not at risk).
    """
    from inference import _build_action, parse_llm_response  # noqa: WPS433
    import asyncio

    from client import ValidatorEnv  # noqa: WPS433

    def reward_fn(prompts, completions, **kwargs):  # noqa: ARG001
        rewards: List[float] = []
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("loop closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        task_names = kwargs.get("task") or []
        seeds = kwargs.get("seed") or []

        env_client = ValidatorEnv(base_url=env_url)
        try:
            for idx, completion in enumerate(completions):
                text = (
                    completion
                    if isinstance(completion, str)
                    else completion[0]["content"]
                )
                task_name = _list_value(task_names, idx, task_pool[0])
                seed = _list_value(seeds, idx, 0)
                try:
                    loop.run_until_complete(
                        env_client.reset(task_name=task_name, seed=int(seed))
                    )
                    action_data = parse_llm_response(text)
                    action = _build_action(action_data)
                    step_result = loop.run_until_complete(
                        env_client.step(action)
                    )
                    rewards.append(float(step_result.reward or 0.0))
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] reward_fn error: {exc}")
                    rewards.append(-0.5)
        finally:
            try:
                loop.run_until_complete(env_client.close())
            except Exception:  # noqa: BLE001
                pass

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

    # 4. Mixed precision setup.
    #
    #    We force fp16 on every GPU rather than auto-selecting bf16 on
    #    Ampere+. Reason: unsloth's fast_lora kernel with bf16 autocast
    #    crashes inside its gradient-checkpointed LoRA forward pass with
    #    "self and mat2 must have the same dtype, but got Half and Float".
    #    fp16 avoids the autocast path that triggers the bug entirely
    #    and works on T4 (smoke test confirmed) and L4 alike.
    #
    #    Tradeoff: slightly less numerical range than bf16. Acceptable
    #    for LoRA training; bf16's main advantage is full-precision FT.
    import torch  # type: ignore
    use_bf16 = False
    torch_dtype = torch.float16
    print(f"[INFO] mixed precision: fp16 (bf16 disabled due to unsloth LoRA issue)")

    # 5. Load model + LoRA. Pass `dtype` explicitly so the model weights
    #    match the dtype the GRPO trainer will use. Without this Unsloth
    #    loads in fp16 by default; with bf16=True in GRPOConfig the LoRA
    #    forward pass crashes with "self and mat2 must have the same
    #    dtype, but got Half and Float".
    print(f"[INFO] loading model: {cfg.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
        dtype=torch_dtype,
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
        bf16=use_bf16,
        fp16=not use_bf16,
    )

    reward_fn = make_reward_fn(cfg.env_url, train_tasks)

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

    # 9. Push checkpoint AND training artefacts (reward_curve, state JSON)
    #    HF Jobs containers are ephemeral — anything written under
    #    results/ is lost when the job exits. To make the reward curve
    #    available after the job finishes, we upload it to the same HF
    #    Hub model repo where the LoRA adapter goes, under a
    #    "training_artifacts/" path.
    if cfg.push_to_hub_id:
        print(f"[INFO] pushing adapter to {cfg.push_to_hub_id}")
        model.push_to_hub(cfg.push_to_hub_id, token=os.getenv("HF_TOKEN"))

        # Upload training artefacts to the same repo
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=os.getenv("HF_TOKEN"))
            for fname in ("reward_curve.png", "training_state.json"):
                local = results_dir / fname
                if local.exists():
                    print(f"[INFO] uploading {fname} -> {cfg.push_to_hub_id}/training_artifacts/{fname}")
                    api.upload_file(
                        path_or_fileobj=str(local),
                        path_in_repo=f"training_artifacts/{fname}",
                        repo_id=cfg.push_to_hub_id,
                        repo_type="model",
                        commit_message=f"Upload {fname} from GRPO run",
                    )
                else:
                    print(f"[WARN] {local} not found — skipping upload")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] artefact upload failed: {exc}")

    # Clean up the dataset-build env client (reward_fn uses its own per-call clients)
    try:
        asyncio.get_event_loop().run_until_complete(env.close())
    except Exception:  # noqa: BLE001
        pass

    print("[INFO] done.")


if __name__ == "__main__":
    main()
